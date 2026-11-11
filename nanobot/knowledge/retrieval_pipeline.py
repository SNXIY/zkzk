"""Enterprise retrieval pipeline: multi-query · weighted RRF · grade · corrective.

Aligned with production patterns (Onyx multi-query + weighted RRF; Corrective RAG
grade→retry with a hard budget). Deterministic by default — no LLM in the hot path —
so chat-model rewrites cannot be the sole retrieval signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from loguru import logger

from nanobot.knowledge.config import KnowledgeRetrievalPolicy
from nanobot.knowledge.store import KnowledgeHit, KnowledgeStore, _lexical_content_tokens

_CLAUSE_SPLIT = re.compile(r"[?？!！;；]+|\band\b|\bor\b|以及|并且|或者", re.IGNORECASE)
_ASCII_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


@dataclass(frozen=True)
class QueryVariant:
    role: str  # anchor | keyword | tool | corrective | decompose
    text: str
    weight: float


@dataclass
class GradeResult:
    passed: bool
    coverage: float
    matched_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    reason: str


@dataclass
class PipelineResult:
    hits: list[KnowledgeHit]
    variants: list[QueryVariant]
    grade: GradeResult
    rounds: int
    confidence: str
    usable_for_answer: bool
    audit: dict[str, Any] = field(default_factory=dict)


def build_query_variants(
    *,
    anchor: str,
    tool_query: str,
    policy: KnowledgeRetrievalPolicy,
    extra_queries: Sequence[tuple[str, str, float]] | None = None,
) -> list[QueryVariant]:
    """Build the system query set for one retrieval round.

    ``extra_queries`` is a list of ``(role, text, weight)`` for corrective expansions.
    """
    variants: list[QueryVariant] = []
    seen: set[str] = set()

    def _add(role: str, text: str, weight: float) -> None:
        q = (text or "").strip()
        if not q:
            return
        key = q.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(QueryVariant(role=role, text=q, weight=float(weight)))

    anchor = (anchor or "").strip()
    tool_query = (tool_query or "").strip()

    if anchor:
        _add("anchor", anchor, policy.anchor_weight)

    if policy.multi_query and policy.keyword_variant:
        tokens = _lexical_content_tokens(anchor or tool_query)
        if tokens:
            _add("keyword", " ".join(tokens), policy.keyword_weight)

    if policy.fuse_tool_query and tool_query:
        _add("tool", tool_query, policy.tool_query_weight)
    elif tool_query and not variants:
        _add("tool", tool_query, policy.tool_query_weight)

    for role, text, weight in extra_queries or ():
        _add(role, text, weight)

    return variants


def grade_hits(
    hits: Sequence[KnowledgeHit],
    *,
    reference_text: str,
    policy: KnowledgeRetrievalPolicy,
) -> GradeResult:
    """Evidence grade: content-token coverage over the top-M fused hits (full text)."""
    tokens = _lexical_content_tokens(reference_text)
    if not hits:
        return GradeResult(
            passed=False,
            coverage=0.0,
            matched_tokens=(),
            missing_tokens=tuple(tokens),
            reason="empty_hits",
        )
    if not tokens:
        # Cannot verify grounding without content tokens — fail closed.
        return GradeResult(
            passed=False,
            coverage=0.0,
            matched_tokens=(),
            missing_tokens=(),
            reason="no_content_tokens",
        )

    top = list(hits)[: max(1, int(policy.grade_top_m))]
    blob = " ".join(
        f"{h.title} {h.filename} {h.breadcrumb} {h.text or h.snippet}" for h in top
    ).lower()
    matched = [t for t in tokens if t.lower() in blob]
    missing = [t for t in tokens if t.lower() not in blob]
    coverage = float(len(matched)) / float(len(tokens))
    # Coverage threshold only — no soft "2-token" bypass.
    passed = coverage >= float(policy.grade_min_coverage)
    reason = "coverage_ok" if passed else f"coverage_low:{coverage:.2f}"
    return GradeResult(
        passed=passed,
        coverage=coverage,
        matched_tokens=tuple(matched),
        missing_tokens=tuple(missing),
        reason=reason,
    )


def _corrective_queries(
    *,
    anchor: str,
    tool_query: str,
    grade: GradeResult,
    round_idx: int,
) -> list[tuple[str, str]]:
    """Deterministic corrective expansions after a failed grade.

    Returns ``(role, query)`` pairs — all are fused in the next round.
    """
    out: list[tuple[str, str]] = []
    missing = [t for t in grade.missing_tokens if t]
    if missing:
        out.append(("corrective", " ".join(missing[:8])))
    base = (anchor or tool_query or "").strip()
    if base and round_idx >= 0:
        parts = [p.strip() for p in _CLAUSE_SPLIT.split(base) if p and p.strip()]
        for part in parts:
            if len(part) >= 8 and part.casefold() != base.casefold():
                out.append(("decompose", part))
    if not out:
        toks = _lexical_content_tokens(base)
        if toks:
            out.append(("corrective", " ".join(toks[:6])))
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for role, q in out:
        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((role, q))
    return uniq


def _confidence_from_grade(grade: GradeResult, hits: Sequence[KnowledgeHit]) -> str:
    if not hits:
        return "none"
    if grade.passed and grade.coverage >= 0.6:
        return "high"
    if grade.passed:
        return "mixed"
    return "low"


def run_retrieval_pipeline(
    store: KnowledgeStore,
    *,
    anchor: str,
    tool_query: str,
    policy: KnowledgeRetrievalPolicy,
    k: int = 5,
) -> PipelineResult:
    """Execute multi-query retrieval with grade-gated corrective rounds."""
    k = max(1, min(int(k), 20))
    max_rounds = max(1, int(policy.corrective_rounds))
    reference = (anchor or tool_query or "").strip()

    all_variants: list[QueryVariant] = []
    hits: list[KnowledgeHit] = []
    grade = GradeResult(False, 0.0, (), (), "not_run")
    round_used = 0
    extras: list[tuple[str, str, float]] = []

    for round_idx in range(max_rounds):
        round_used = round_idx + 1
        variants = build_query_variants(
            anchor=anchor,
            tool_query=tool_query,
            policy=policy,
            extra_queries=extras,
        )
        if not variants:
            grade = GradeResult(False, 0.0, (), (), "no_queries")
            break
        all_variants.extend(variants)
        queries = [v.text for v in variants]
        weights = [v.weight for v in variants]
        hits = store.search_fused(queries, k=k, weights=weights)
        if policy.min_rrf_score > 0:
            hits = [h for h in hits if float(h.score) >= policy.min_rrf_score]
        grade = grade_hits(hits, reference_text=reference, policy=policy)
        logger.info(
            "kb pipeline round={} variants={} hits={} grade_pass={} coverage={:.2f} reason={}",
            round_used,
            [(v.role, v.text[:80]) for v in variants],
            len(hits),
            grade.passed,
            grade.coverage,
            grade.reason,
        )
        if grade.passed or round_idx + 1 >= max_rounds:
            break
        expansions = _corrective_queries(
            anchor=anchor,
            tool_query=tool_query,
            grade=grade,
            round_idx=round_idx,
        )
        if not expansions:
            break
        extras = [(role, text, policy.corrective_weight) for role, text in expansions]

    confidence = _confidence_from_grade(grade, hits)
    usable = bool(hits) and bool(grade.passed)
    if not grade.passed and policy.grade_fail_closed:
        usable = False

    shown: list[QueryVariant] = []
    seen_q: set[str] = set()
    for v in all_variants:
        key = v.text.casefold()
        if key in seen_q:
            continue
        seen_q.add(key)
        shown.append(v)

    answer_hits = hits if usable or not policy.grade_fail_closed else []

    return PipelineResult(
        hits=answer_hits,
        variants=shown,
        grade=grade,
        rounds=round_used,
        confidence=confidence if usable else ("low" if hits else "none"),
        usable_for_answer=usable,
        audit={
            "anchor": anchor,
            "tool_query": tool_query,
            "rounds": round_used,
            "grade_passed": grade.passed,
            "coverage": grade.coverage,
            "missing_tokens": list(grade.missing_tokens),
            "variant_roles": [v.role for v in shown],
            "raw_hit_count": len(hits),
            "usable_for_answer": usable,
            "fail_closed": bool(policy.grade_fail_closed),
        },
    )


def hint_matches_query(query: str, hints: Sequence[str]) -> bool:
    """Match KB hints with word boundaries for ASCII; substring for CJK."""
    q = (query or "").strip().lower()
    if not q:
        return False
    for hint in hints:
        h = str(hint).strip().lower()
        if len(h) < 2:
            continue
        if _ASCII_WORD.match(h):
            if re.search(rf"(?<![a-z0-9_]){re.escape(h)}(?![a-z0-9_])", q):
                return True
        elif h in q:
            return True
    return False
