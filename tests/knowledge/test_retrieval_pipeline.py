"""Enterprise multi-query + grade + corrective retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.knowledge import KnowledgeSearchTool
from nanobot.knowledge.config import KnowledgeConfig, KnowledgeRetrievalPolicy
from nanobot.knowledge.retrieval_pipeline import (
    build_query_variants,
    grade_hits,
    run_retrieval_pipeline,
)
from nanobot.knowledge.store import KnowledgeHit, KnowledgeStore


def test_build_query_variants_includes_anchor_keyword_tool() -> None:
    policy = KnowledgeRetrievalPolicy(
        multi_query=True,
        keyword_variant=True,
        anchor_user_query=True,
        fuse_tool_query=True,
    )
    variants = build_query_variants(
        anchor="What are AcmeTech's four data classification tiers?",
        tool_query="Data Classification Public Internal Confidential Restricted",
        policy=policy,
    )
    roles = [v.role for v in variants]
    assert "anchor" in roles
    assert "keyword" in roles
    assert "tool" in roles
    # Keyword variant must drop stopwords like what/are
    keyword = next(v for v in variants if v.role == "keyword")
    assert "what" not in keyword.text.lower()
    assert "classification" in keyword.text.lower()


def test_grade_hits_coverage_and_fail_closed_rules() -> None:
    policy = KnowledgeRetrievalPolicy(grade_min_coverage=0.45, grade_top_m=3)
    hits = [
        KnowledgeHit(
            "d1",
            "Security_Policy",
            "Security_Policy.txt",
            0,
            "snippet…",
            0.05,
            source="hybrid",
            text=(
                "AcmeTech classifies data into four tiers: "
                "Public Internal Confidential Restricted."
            ),
        )
    ]
    ok = grade_hits(
        hits,
        reference_text="What are AcmeTech's four data classification tiers?",
        policy=policy,
    )
    assert ok.passed is True
    assert ok.coverage >= 0.45

    bad = grade_hits(
        hits,
        reference_text="employee stock option vesting cliff schedule",
        policy=policy,
    )
    assert bad.passed is False
    assert bad.missing_tokens

    # No soft 2-token bypass: sparse match must fail when below coverage.
    sparse = grade_hits(
        [
            KnowledgeHit(
                "d1",
                "T",
                "a.txt",
                0,
                "x",
                0.1,
                text="classification only",
                source="lexical",
            )
        ],
        reference_text="AcmeTech four data classification tiers Public Internal",
        policy=policy,
    )
    assert sparse.passed is False


def test_grade_no_content_tokens_fails() -> None:
    policy = KnowledgeRetrievalPolicy()
    hits = [KnowledgeHit("d1", "T", "a.txt", 0, "hello", 0.1, text="hello world", source="lexical")]
    grade = grade_hits(hits, reference_text="?? !!", policy=policy)
    assert grade.passed is False
    assert grade.reason == "no_content_tokens"


def test_pipeline_corrective_improves_sparse_first_hit(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "ws", search_mode="lexical")
    store.add_document(
        "Security_Policy.txt",
        (
            "AcmeTech data classification tiers: Public, Internal, Confidential, Restricted.\n"
            "Handling requirements differ by tier.\n"
        ).encode("utf-8"),
    )
    policy = KnowledgeRetrievalPolicy(
        multi_query=True,
        keyword_variant=True,
        corrective_rounds=2,
        grade_min_coverage=0.45,
        grade_fail_closed=True,
        anchor_weight=1.25,
        keyword_weight=1.0,
    )
    # Slightly noisy NL question — pipeline should still grade pass via keyword/corrective.
    result = run_retrieval_pipeline(
        store,
        anchor="What are AcmeTech's four data classification tiers?",
        tool_query="unrelated xyzzy",
        policy=policy,
        k=5,
    )
    assert result.usable_for_answer is True
    assert result.hits
    assert any(
        "classification" in (h.text or h.snippet).lower() or "Public" in (h.text or h.snippet)
        for h in result.hits
    )
    assert result.confidence in {"high", "mixed"}
    assert any(v.role in {"anchor", "keyword"} for v in result.variants)


def test_weighted_rrf_prefers_anchor_overlap(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "ws", search_mode="lexical")
    store.add_document(
        "A.txt",
        b"alpha document about classification tiers Public Internal\n",
    )
    store.add_document(
        "B.txt",
        b"beta document mentions completely unrelated widgets\n",
    )
    # Anchor-focused query should surface classification doc.
    hits = store.search_fused(
        ["classification tiers Public", "widgets"],
        k=2,
        weights=[1.5, 0.5],
    )
    assert hits
    assert "classification" in hits[0].snippet.lower() or hits[0].filename == "A.txt"


@pytest.mark.asyncio
async def test_kb_search_tool_reports_grade_and_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    store = KnowledgeStore(workspace, search_mode="lexical")
    store.add_document(
        "Security_Policy.txt",
        b"AcmeTech data classification tiers: Public Internal Confidential Restricted.\n",
    )
    cfg = KnowledgeConfig(
        enable=True,
        embeddings=False,
        retrieval=KnowledgeRetrievalPolicy(multi_query=True, corrective_rounds=2),
    )
    tool = KnowledgeSearchTool(workspace, cfg)
    monkeypatch.setattr(tool, "_store", lambda: store)
    tool.set_context(
        RequestContext(
            channel="websocket",
            chat_id="c1",
            user_text="What are AcmeTech's four data classification tiers?",
            kb_gate={"need_kb": True, "reason": "test", "gate_mode": "soft"},
        )
    )
    out = await tool.execute(query="classification tiers", k=5)
    assert "grade_passed=" in out
    assert "variants:" in out
    assert "cite=`Security_Policy.txt#chunk" in out
