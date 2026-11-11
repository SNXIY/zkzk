"""Tests for enterprise skill router + hierarchical skill discovery."""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.skills import SkillRouter, SkillRouterConfig, SkillsLoader


def _write_nested_skill(
    root: Path,
    category: str,
    name: str,
    *,
    description: str,
    keywords: list[str] | None = None,
) -> Path:
    dest = root / category / name
    dest.mkdir(parents=True, exist_ok=True)
    import json

    meta = {
        "nanobot": {
            "category": category,
            "keywords": keywords or [],
            "when_to_use": description,
        }
    }
    text = (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description)}\n"
        f"metadata: {json.dumps(meta, separators=(',', ':'))}\n"
        "---\n\n"
        f"# {name}\n"
    )
    path = dest / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_nested_category_discovery(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    skills = workspace / "skills"
    _write_nested_skill(skills, "hr", "leave-policy", description="Leave and PTO policy")
    _write_nested_skill(
        skills, "customer_support", "escalation-guide", description="Escalation tiers and handoff"
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = {e["name"]: e for e in loader.list_skills(filter_unavailable=False)}
    assert entries["leave-policy"]["category"] == "hr"
    assert entries["escalation-guide"]["category"] == "customer_support"
    cats = loader.list_categories()
    assert "hr" in cats and "leave-policy" in cats["hr"]


def test_lexical_router_prefers_matching_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    skills = workspace / "skills"
    _write_nested_skill(
        skills,
        "customer_support",
        "escalation-guide",
        description="Support escalation tiers Tier1 Tier2 handoff",
        keywords=["escalation", "pagerduty"],
    )
    _write_nested_skill(
        skills,
        "finance",
        "expense-policy",
        description="Expense reimbursement and receipts",
        keywords=["expense", "receipt"],
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    router = SkillRouter(
        loader,
        SkillRouterConfig(enable=True, search_mode="lexical", two_stage=True, top_k=3),
    )
    hits = router.route("How do I escalate a P1 ticket to Tier 2?")
    assert hits
    assert hits[0].name == "escalation-guide"
