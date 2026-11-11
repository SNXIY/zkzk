"""Enterprise skill router: hybrid lexical + dense Top-K with optional two-stage category routing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.skills.loader import SkillsLoader
from nanobot.config.skill_router import SkillRouterConfig

_TOKEN = re.compile(r"[a-z0-9_\u4e00-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True)
class SkillRouteHit:
    name: str
    score: float
    category: str
    description: str
    source: str  # lexical | vector | hybrid | always
    requires_kb: bool = False


@dataclass(frozen=True)
class SkillRouteDecision:
    """Enterprise turn gate: which skills + whether the KB channel is required."""

    hits: tuple[SkillRouteHit, ...]
    need_kb: bool
    kb_skills: tuple[str, ...]
    kb_categories: tuple[str, ...]
    reason: str
    gate_mode: str = "soft"


class SkillRouter:
    """Route a natural-language turn onto a small skill subset.

    Design goals (enterprise, not demo):
    - Scale to large catalogs without dumping every SKILL.md into the prompt
    - Reuse DashScope embeddings when configured (same stack as KB RAG)
    - Degrade cleanly to lexical routing if embeddings unavailable
    - Optional two-stage: category → skills (department-aware)
    """

    def __init__(
        self,
        loader: SkillsLoader,
        cfg: SkillRouterConfig | None = None,
        *,
        knowledge_cfg: Any | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.loader = loader
        self.cfg = cfg or SkillRouterConfig()
        self.knowledge_cfg = knowledge_cfg
        self.cache_dir = Path(cache_dir or (loader.workspace / ".skill_router"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._embeddings: dict[str, list[float]] = {}
        self._embedder = None
        self._fingerprint = ""
        self._rebuild_if_needed()

    def _rebuild_if_needed(self) -> None:
        records = self.loader.catalog_records()
        digest = hashlib.sha256(
            json.dumps(
                [(r["name"], r["index_text"], r["path"]) for r in records],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if digest == self._fingerprint and self._records:
            return
        self._records = records
        self._fingerprint = digest
        self._embeddings = {}
        self._maybe_load_embedding_cache()
        if self.cfg.search_mode in ("hybrid", "vector"):
            self._ensure_embeddings()

    def _embedder_client(self):
        if self._embedder is not None:
            return self._embedder
        if self.knowledge_cfg is None:
            return None
        try:
            from nanobot.knowledge.embeddings import resolve_embedder

            self._embedder = resolve_embedder(self.knowledge_cfg)
        except Exception as exc:
            logger.warning("skill router embedder unavailable: {}", exc)
            self._embedder = None
        return self._embedder

    def _embedding_cache_path(self) -> Path:
        return self.cache_dir / "embeddings.json"

    def _maybe_load_embedding_cache(self) -> None:
        path = self._embedding_cache_path()
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._fingerprint:
                return
            raw = payload.get("vectors") or {}
            self._embeddings = {
                str(k): [float(x) for x in v] for k, v in raw.items() if isinstance(v, list)
            }
        except Exception:
            self._embeddings = {}

    def _save_embedding_cache(self) -> None:
        path = self._embedding_cache_path()
        payload = {
            "fingerprint": self._fingerprint,
            "updated_at": time.time(),
            "vectors": self._embeddings,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _ensure_embeddings(self) -> None:
        embedder = self._embedder_client()
        if embedder is None:
            return
        missing = [r for r in self._records if r["name"] not in self._embeddings]
        if not missing:
            return
        texts = [r["index_text"] for r in missing]
        try:
            vectors = embedder.embed(texts)
        except Exception as exc:
            logger.warning("skill router embedding failed: {}", exc)
            return
        for rec, vec in zip(missing, vectors, strict=False):
            if vec:
                self._embeddings[rec["name"]] = list(vec)
        try:
            self._save_embedding_cache()
        except OSError as exc:
            logger.warning("skill router cache write failed: {}", exc)

    def route(self, query: str, *, top_k: int | None = None) -> list[SkillRouteHit]:
        self._rebuild_if_needed()
        q = (query or "").strip()
        if not q or not self._records:
            return []
        k = top_k or self.cfg.top_k
        mode = self.cfg.search_mode

        lexical = self._lexical_rank(q) if mode in ("hybrid", "lexical") else []
        vector = self._vector_rank(q) if mode in ("hybrid", "vector") else []

        if mode == "lexical" or (mode == "hybrid" and not vector):
            fused = lexical
        elif mode == "vector" or (mode == "hybrid" and not lexical):
            fused = vector
        else:
            fused = self._rrf_fuse(vector, lexical)

        if self.cfg.two_stage:
            fused = self._two_stage_filter(fused)

        hits: list[SkillRouteHit] = []
        for name, score, source in fused:
            if score < self.cfg.min_score:
                continue
            rec = self._by_name(name)
            if not rec or rec.get("always"):
                continue
            hits.append(
                SkillRouteHit(
                    name=name,
                    score=score,
                    category=str(rec.get("category") or ""),
                    description=str(rec.get("description") or ""),
                    source=source,
                    requires_kb=bool(rec.get("requires_kb")),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def decide(self, query: str, *, top_k: int | None = None) -> SkillRouteDecision:
        """Route skills and decide whether this turn requires the knowledge channel.

        ``need_kb`` is true when a routed skill declares ``requires_kb``, when its
        category is in the knowledge retrieval policy allowlist, or when the query
        matches configured KB hint terms (fallback for thin catalogs).
        """
        hits = self.route(query, top_k=top_k)
        policy = self._retrieval_policy()
        gate_mode = str(getattr(policy, "gate_mode", "soft") or "soft")
        kb_cats = {
            str(c).strip().lower().replace("-", "_")
            for c in (getattr(policy, "kb_categories", None) or [])
            if str(c).strip()
        }
        kb_skills: list[str] = []
        kb_categories: list[str] = []
        reasons: list[str] = []
        for hit in hits:
            cat = (hit.category or "").strip().lower().replace("-", "_")
            rec = self._by_name(hit.name) or {}
            by_flag = bool(hit.requires_kb or rec.get("requires_kb"))
            by_cat = bool(cat and cat in kb_cats)
            if by_flag or by_cat:
                kb_skills.append(hit.name)
                if cat and cat not in kb_categories:
                    kb_categories.append(cat)
                if by_flag:
                    reasons.append(f"skill:{hit.name}:requires_kb")
                elif by_cat:
                    reasons.append(f"category:{cat}")
        need_kb = bool(kb_skills)
        if not need_kb and self._query_matches_kb_hints(query, policy):
            need_kb = True
            reasons.append("query_hint")
        if not need_kb:
            reasons.append("no_kb_signal")
        return SkillRouteDecision(
            hits=tuple(hits),
            need_kb=need_kb,
            kb_skills=tuple(kb_skills),
            kb_categories=tuple(kb_categories),
            reason=";".join(reasons),
            gate_mode=gate_mode,
        )

    def _retrieval_policy(self) -> Any:
        cfg = self.knowledge_cfg
        if cfg is None:
            from nanobot.knowledge.config import KnowledgeRetrievalPolicy

            return KnowledgeRetrievalPolicy()
        retrieval = getattr(cfg, "retrieval", None)
        if retrieval is not None:
            return retrieval
        from nanobot.knowledge.config import KnowledgeRetrievalPolicy

        return KnowledgeRetrievalPolicy()

    @staticmethod
    def _query_matches_kb_hints(query: str, policy: Any) -> bool:
        from nanobot.knowledge.retrieval_pipeline import hint_matches_query

        hints = getattr(policy, "kb_query_hints", None) or []
        return hint_matches_query(query, hints)

    def _by_name(self, name: str) -> dict | None:
        for rec in self._records:
            if rec["name"] == name:
                return rec
        return None

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]

    def _lexical_rank(self, query: str) -> list[tuple[str, float, str]]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        q_lower = query.lower()
        scored: list[tuple[str, float, str]] = []
        for rec in self._records:
            blob = rec["index_text"].lower()
            doc_tokens = self._tokenize(rec["index_text"])
            if not doc_tokens:
                continue
            d_set = set(doc_tokens)
            score = 0.0
            matched = 0
            for qt in q_set:
                if qt in d_set:
                    score += 1.0
                    matched += 1
                    continue
                # Morphological / substring soft match (escalate↔escalation, tier↔tier1)
                if len(qt) >= 3 and any(
                    dt.startswith(qt) or qt.startswith(dt) for dt in d_set if len(dt) >= 3
                ):
                    score += 0.55
                    matched += 1
                elif len(qt) >= 3 and qt in blob:
                    score += 0.4
                    matched += 1
            if matched == 0:
                continue
            score = score / max(len(q_set), 1)
            if rec["name"].replace("-", " ") in q_lower:
                score += 0.35
            cat = (rec.get("category") or "").replace("_", " ").lower()
            if cat and any(part and part in q_lower for part in cat.split()):
                score += 0.2
            for kw in rec.get("keywords") or []:
                kw_l = str(kw).lower()
                if kw_l and (kw_l in q_lower or any(kw_l.startswith(t) or t.startswith(kw_l) for t in q_set if len(t) >= 4)):
                    score += 0.12
            scored.append((rec["name"], score, "lexical"))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _vector_rank(self, query: str) -> list[tuple[str, float, str]]:
        embedder = self._embedder_client()
        if embedder is None or not self._embeddings:
            return []
        try:
            qvec = embedder.embed([query])[0]
        except Exception as exc:
            logger.warning("skill router query embed failed: {}", exc)
            return []
        scored: list[tuple[str, float, str]] = []
        for rec in self._records:
            vec = self._embeddings.get(rec["name"])
            if not vec:
                continue
            scored.append((rec["name"], _cosine(qvec, vec), "vector"))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rrf_fuse(
        self,
        vector: list[tuple[str, float, str]],
        lexical: list[tuple[str, float, str]],
        *,
        rrf_k: int = 60,
    ) -> list[tuple[str, float, str]]:
        scores: dict[str, float] = {}
        for rank, (name, _, _) in enumerate(vector):
            scores[name] = scores.get(name, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (name, _, _) in enumerate(lexical):
            scores[name] = scores.get(name, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(name, score, "hybrid") for name, score in ranked]

    def _two_stage_filter(
        self, ranked: list[tuple[str, float, str]]
    ) -> list[tuple[str, float, str]]:
        """Keep skills from the top categories, preserving global order inside."""
        if not ranked:
            return ranked
        cat_scores: dict[str, float] = {}
        for name, score, _ in ranked:
            rec = self._by_name(name)
            if not rec:
                continue
            cat = str(rec.get("category") or "general")
            cat_scores[cat] = max(cat_scores.get(cat, 0.0), score)
        top_cats = {
            c
            for c, _ in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[
                : self.cfg.category_top_k
            ]
        }
        primary = [
            row
            for row in ranked
            if str((self._by_name(row[0]) or {}).get("category") or "general") in top_cats
        ]
        # Keep a small global tail so multi-hop / cross-dept queries don't collapse.
        secondary = [row for row in ranked if row not in primary]
        return primary + secondary


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom else 0.0
