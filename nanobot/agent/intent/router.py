"""Deterministic multi-channel intent router with tool allowlists.

Production contract (see ``.agent/intent_channels.md``):
- Exclusive channel ownership per turn (OpenAI handoff-style specialist surface)
- Deny-by-default allowlists; empty business MCP ⇒ fail closed to chat
- Align ``need_kb`` with ``channel == knowledge`` (no soft cross-channel KB)
- Lexical + SkillRouter signals only — no LLM secondary classifier in v1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from nanobot.agent.skills.router import SkillRouteDecision, SkillRouter
from nanobot.agent.tools.mcp import _tool_prefix
from nanobot.config.intent_router import IntentChannelName, IntentRouterConfig
from nanobot.knowledge.retrieval_pipeline import hint_matches_query

_TOKEN = re.compile(r"[a-z0-9_\u4e00-\u9fff]+", re.IGNORECASE)

# Named weights (avoid magic scatter). Knowledge outranks equal lexical hits.
_SCORE_KNOWLEDGE_NEED_KB = 3.0
_SCORE_KNOWLEDGE_SKILL_BONUS = 0.2
_SCORE_KNOWLEDGE_CATEGORY_BONUS = 0.2
_SCORE_HINT = 2.0
_SCORE_SERVER_NAME = 1.5
_SCORE_SERVER_TOOLS_PRESENT = 0.5
_SCORE_CHAT_EMPTY = 1.0
_SCORE_CHAT_GREETING = 1.2
_SCORE_CHAT_BASELINE = 0.1

_GREETINGS = (
    "你好",
    "您好",
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "谢谢",
    "嗯",
    "ok",
    "okay",
    "好的",
)


@dataclass(frozen=True)
class ChannelDecision:
    channel: IntentChannelName
    scores: dict[str, float]
    reason: str
    need_kb: bool
    allowed_tools: tuple[str, ...]
    skill_decision: SkillRouteDecision | None = None
    notes: tuple[str, ...] = ()
    confidence: float = 0.0


class IntentChannelRouter:
    """Classify a user turn into one execution channel and prune tools accordingly."""

    def __init__(
        self,
        cfg: IntentRouterConfig | None = None,
        *,
        skill_router: SkillRouter | None = None,
    ) -> None:
        self.cfg = cfg or IntentRouterConfig()
        self.skill_router = skill_router

    def decide(
        self,
        query: str,
        *,
        available_tools: Sequence[str],
    ) -> ChannelDecision:
        q = (query or "").strip()
        available = {str(n) for n in available_tools if str(n).strip()}
        skill_decision: SkillRouteDecision | None = None
        if self.skill_router is not None and q:
            skill_decision = self.skill_router.decide(q)

        if not self.cfg.enable:
            need_kb = bool(skill_decision and skill_decision.need_kb)
            channel: IntentChannelName = "knowledge" if need_kb else "workspace"
            return ChannelDecision(
                channel=channel,
                scores={
                    "knowledge": 1.0 if need_kb else 0.0,
                    "business_api": 0.0,
                    "workspace": 0.0 if need_kb else 1.0,
                    "chat": 0.0,
                },
                reason="intent_router_disabled",
                need_kb=need_kb,
                allowed_tools=tuple(sorted(available)),
                skill_decision=skill_decision,
                notes=("disabled_passthrough",),
                confidence=1.0,
            )

        scores = {
            "knowledge": self._score_knowledge(skill_decision),
            "business_api": self._score_business_api(q, available),
            "workspace": self._score_workspace(q),
            "chat": self._score_chat(q),
        }
        notes: list[str] = []
        channel, confidence = self._pick_channel(scores, notes)

        # Fail closed: business intent without callable MCP tools is not a business turn.
        if channel == "business_api" and self.cfg.fail_closed_empty_business_api:
            business_tools = self._business_mcp_tools(available)
            if not business_tools:
                notes.append("fail_closed_empty_business_api")
                channel = "chat"
                confidence = 0.0

        need_kb = channel == "knowledge"
        allowed = self._allowlist_for(channel, available, notes)
        reason_parts = [
            f"channel={channel}",
            f"confidence={confidence:.2f}",
            f"scores={{{', '.join(f'{k}:{v:.2f}' for k, v in sorted(scores.items()))}}}",
        ]
        if skill_decision is not None:
            reason_parts.append(f"skill={skill_decision.reason}")
        if notes:
            reason_parts.append("notes=" + ",".join(notes))

        return ChannelDecision(
            channel=channel,
            scores=scores,
            reason=";".join(reason_parts),
            need_kb=need_kb,
            allowed_tools=tuple(sorted(allowed)),
            skill_decision=skill_decision,
            notes=tuple(notes),
            confidence=confidence,
        )

    def _pick_channel(
        self,
        scores: dict[str, float],
        notes: list[str],
    ) -> tuple[IntentChannelName, float]:
        priority = list(self.cfg.channel_priority)
        ranked = sorted(
            ((name, float(scores.get(name, 0.0))) for name in priority),
            key=lambda item: (-item[1], priority.index(item[0])),
        )
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        if best_name != "chat" and best_score < float(self.cfg.min_channel_score):
            notes.append("below_min_channel_score")
            return "chat", 0.0

        if (
            best_name != "chat"
            and float(self.cfg.score_margin) > 0
            and margin < float(self.cfg.score_margin)
        ):
            notes.append("below_score_margin")
            return "chat", 0.0

        # Normalize a simple confidence in [0, 1] for audit (not a probability).
        confidence = min(1.0, max(0.0, best_score / max(_SCORE_KNOWLEDGE_NEED_KB, 1.0)))
        if best_name == "chat":
            confidence = min(1.0, max(0.0, best_score))
        return best_name, confidence  # type: ignore[return-value]

    def _score_knowledge(self, skill_decision: SkillRouteDecision | None) -> float:
        if skill_decision is None or not skill_decision.need_kb:
            return 0.0
        score = _SCORE_KNOWLEDGE_NEED_KB
        if skill_decision.kb_skills:
            score += min(1.0, _SCORE_KNOWLEDGE_SKILL_BONUS * len(skill_decision.kb_skills))
        if skill_decision.kb_categories:
            score += _SCORE_KNOWLEDGE_CATEGORY_BONUS
        return score

    def _score_business_api(self, query: str, available: set[str]) -> float:
        score = 0.0
        if hint_matches_query(query, self.cfg.business_api_hints):
            score += _SCORE_HINT
        q_lower = (query or "").lower()
        q_tokens = set(_TOKEN.findall(q_lower))
        servers = [s.strip() for s in self.cfg.business_api_mcp_servers if str(s).strip()]
        matched_tools = False
        for server in servers:
            key = server.lower().replace("-", "_")
            if key and (
                key in q_lower
                or any(key in t or t in key for t in q_tokens if len(t) >= 3)
            ):
                score += _SCORE_SERVER_NAME
            prefix = _tool_prefix(server)
            if any(name.startswith(prefix) for name in available):
                matched_tools = True
        # Presence of configured tools only reinforces an already-fired signal.
        if score > 0 and matched_tools:
            score += _SCORE_SERVER_TOOLS_PRESENT
        return score

    def _score_workspace(self, query: str) -> float:
        if hint_matches_query(query, self.cfg.workspace_hints):
            return _SCORE_HINT
        return 0.0

    def _score_chat(self, query: str) -> float:
        q = (query or "").strip()
        if not q:
            return _SCORE_CHAT_EMPTY
        if len(q) <= 12:
            low = q.lower()
            if any(g in low for g in _GREETINGS):
                return _SCORE_CHAT_GREETING
        return _SCORE_CHAT_BASELINE

    def _allowlist_for(
        self,
        channel: str,
        available: set[str],
        notes: list[str],
    ) -> set[str]:
        always = {n for n in self.cfg.always_tools if n in available}
        if channel == "chat":
            return always or ({"message"} & available)

        if channel == "knowledge":
            allowed = set(always)
            if "kb_search" in available:
                allowed.add("kb_search")
            else:
                notes.append("kb_search_missing")
            for name in self.cfg.knowledge_extra_tools:
                if name in available:
                    allowed.add(name)
            return allowed

        if channel == "business_api":
            allowed = set(always)
            matched = self._business_mcp_tools(available)
            if not matched:
                # Should have been downgraded; keep note if policy disabled.
                notes.append("no_business_tools_configured")
            allowed |= matched
            return allowed

        # workspace — never include kb_search or mcp_* unless explicitly always_tools
        allowed = set(always)
        for name in self.cfg.workspace_tools:
            if name in available and not name.startswith("mcp_") and name != "kb_search":
                allowed.add(name)
        return allowed

    def _business_mcp_tools(self, available: set[str]) -> set[str]:
        servers = [s.strip() for s in self.cfg.business_api_mcp_servers if str(s).strip()]
        if not servers:
            return set()
        out: set[str] = set()
        for server in servers:
            prefix = _tool_prefix(server)
            for name in available:
                if name.startswith(prefix):
                    out.add(name)
        return out


def channel_kb_gate_payload(
    decision: ChannelDecision,
    *,
    gate_mode: str = "strict",
) -> dict[str, Any]:
    """Unify RequestContext.kb_gate with channel routing (audit + execute gate)."""
    skill = decision.skill_decision
    return {
        "need_kb": bool(decision.need_kb),
        "reason": decision.reason,
        "gate_mode": gate_mode,
        "enforce": True,
        "intent_channel": decision.channel,
        "kb_skills": list(skill.kb_skills) if skill else [],
        "kb_categories": list(skill.kb_categories) if skill else [],
        "routed_skills": [h.name for h in skill.hits] if skill else [],
        "allowed_tools": list(decision.allowed_tools),
        "scores": dict(decision.scores),
        "notes": list(decision.notes),
        "confidence": float(decision.confidence),
    }
