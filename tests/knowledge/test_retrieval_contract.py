"""Enterprise KB retrieval contract: multi-query fuse + tool anchoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.knowledge import KnowledgeSearchTool
from nanobot.knowledge.config import KnowledgeConfig, KnowledgeRetrievalPolicy
from nanobot.knowledge.store import KnowledgeHit, KnowledgeStore, _rrf_fuse_many


def test_rrf_fuse_many_prefers_overlap() -> None:
    a = [
        KnowledgeHit("d1", "T", "a.pdf", 0, "alpha", 1.0, source="vector"),
        KnowledgeHit("d2", "T", "b.pdf", 0, "beta", 0.9, source="vector"),
    ]
    b = [
        KnowledgeHit("d2", "T", "b.pdf", 0, "beta", 1.0, source="lexical"),
        KnowledgeHit("d3", "T", "c.pdf", 0, "gamma", 0.8, source="lexical"),
    ]
    fused = _rrf_fuse_many([a, b], k=3)
    assert fused[0].doc_id == "d2"
    assert fused[0].source == "hybrid"


def test_search_fused_dedupes_and_anchors(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "ws", search_mode="lexical")
    payload = (
        "AcmeTech data classification tiers: Public Internal Confidential Restricted.\n"
        "Handling requirements for Confidential data require encryption at rest.\n"
    ).encode("utf-8")
    store.add_document("Security_Policy.txt", payload)

    hits_anchor = store.search("What are AcmeTech's four data classification tiers?", k=3)
    assert hits_anchor, "lexical search should hit classification text"

    fused = store.search_fused(
        [
            "What are AcmeTech's four data classification tiers?",
            "Data Classification section 3 Public Internal Confidential Restricted",
            "What are AcmeTech's four data classification tiers?",  # dup
        ],
        k=3,
    )
    assert fused
    assert any("classification" in h.snippet.lower() or "Public" in h.snippet for h in fused)


@pytest.mark.asyncio
async def test_kb_search_tool_anchors_user_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    store = KnowledgeStore(workspace, search_mode="lexical")
    store.add_document(
        "HR_Policy.txt",
        b"Probationary period for new hires is 90 days.\n",
    )

    cfg = KnowledgeConfig(
        enable=True,
        embeddings=False,
        retrieval=KnowledgeRetrievalPolicy(
            anchor_user_query=True,
            fuse_tool_query=True,
            gate_mode="soft",
        ),
    )
    tool = KnowledgeSearchTool(workspace, cfg)
    tool.set_context(
        RequestContext(
            channel="websocket",
            chat_id="c1",
            user_text="How long is the probationary period for new hires?",
            kb_gate={"need_kb": True, "reason": "test"},
        )
    )

    monkeypatch.setattr(tool, "_store", lambda: store)
    # Deliberately weak/divergent tool rewrite — anchor should still recover.
    result = await tool.execute(query="unrelated foo bar bazzzz", k=5)
    assert "probationary" in result.lower() or "90 days" in result.lower()
    assert "anchor_query:" in result
    assert "cite=`HR_Policy.txt#chunk" in result


def test_pipeline_confidence_from_grade_not_score_sign() -> None:
    from nanobot.knowledge.retrieval_pipeline import GradeResult, _confidence_from_grade

    hits = [
        KnowledgeHit("d1", "T", "a.txt", 0, "alpha", -0.2, source="lexical"),
        KnowledgeHit("d1", "T", "a.txt", 1, "beta", -1.5, source="lexical"),
    ]
    high = _confidence_from_grade(
        GradeResult(True, 0.8, ("alpha",), (), "coverage_ok"), hits
    )
    low = _confidence_from_grade(
        GradeResult(False, 0.1, (), ("beta",), "coverage_low"), hits
    )
    assert high == "high"
    assert low == "low"


@pytest.mark.asyncio
async def test_kb_search_prefers_contextvar_over_wiped_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nanobot.agent.tools.context import bind_request_context, reset_request_context

    workspace = tmp_path / "ws"
    store = KnowledgeStore(workspace, search_mode="lexical")
    store.add_document("HR_Policy.txt", b"Probationary period for new hires is 90 days.\n")
    cfg = KnowledgeConfig(enable=True, embeddings=False)
    tool = KnowledgeSearchTool(workspace, cfg)
    monkeypatch.setattr(tool, "_store", lambda: store)

    # Simulate progress-hook wipe: instance context has no user_text.
    tool.set_context(RequestContext(channel="websocket", chat_id="c1"))
    token = bind_request_context(
        RequestContext(
            channel="websocket",
            chat_id="c1",
            user_text="How long is the probationary period for new hires?",
            kb_gate={"need_kb": True, "reason": "test", "gate_mode": "soft"},
        )
    )
    try:
        result = await tool.execute(query="unrelated noise", k=5)
    finally:
        reset_request_context(token)
    assert "90 days" in result.lower() or "probationary" in result.lower()
    assert "anchor_query: How long is the probationary period" in result


@pytest.mark.asyncio
async def test_kb_search_strict_gate_blocks_non_kb_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    store = KnowledgeStore(workspace, search_mode="lexical")
    store.add_document("HR_Policy.txt", b"Probationary period for new hires is 90 days.\n")
    cfg = KnowledgeConfig(
        enable=True,
        embeddings=False,
        retrieval=KnowledgeRetrievalPolicy(gate_mode="strict"),
    )
    tool = KnowledgeSearchTool(workspace, cfg)
    monkeypatch.setattr(tool, "_store", lambda: store)
    tool.set_context(
        RequestContext(
            channel="websocket",
            chat_id="c1",
            user_text="write a fibonacci function",
            kb_gate={
                "need_kb": False,
                "reason": "no_kb_signal",
                "gate_mode": "strict",
                "enforce": True,
            },
        )
    )
    result = await tool.execute(query="probationary period", k=5)
    assert "KB gate blocked" in result


def test_skill_router_decide_need_kb(tmp_path: Path) -> None:
    from nanobot.agent.skills import SkillRouter, SkillsLoader
    from nanobot.config.skill_router import SkillRouterConfig
    from nanobot.knowledge.config import KnowledgeConfig

    workspace = tmp_path / "ws"
    skills = workspace / "skills"
    dest = skills / "security" / "security-policy"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SKILL.md").write_text(
        "---\n"
        "name: security-policy\n"
        'description: "Security policy and data classification"\n'
        "metadata: {\"nanobot\":{\"category\":\"security\",\"requires_kb\":true,"
        "\"keywords\":[\"classification\",\"security\"]}}\n"
        "---\n\n# security-policy\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    router = SkillRouter(
        loader,
        SkillRouterConfig(search_mode="lexical", two_stage=False),
        knowledge_cfg=KnowledgeConfig(),
    )
    decision = router.decide("What are AcmeTech's four data classification tiers?")
    assert decision.need_kb is True
    assert "security-policy" in decision.kb_skills

    chatter = router.decide("write a python fibonacci function")
    assert chatter.need_kb is False
