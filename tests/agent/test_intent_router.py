"""Multi-channel intent router + production tool pruning / execute gate."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nanobot.agent.intent.router import (
    ChannelDecision,
    IntentChannelRouter,
    channel_kb_gate_payload,
)
from nanobot.agent.skills.router import SkillRouteDecision
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import (
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.intent_router import IntentRouterConfig


@dataclass
class _FakeSkillRouter:
    need_kb: bool = False
    reason: str = "fake"

    def decide(self, query: str) -> SkillRouteDecision:
        return SkillRouteDecision(
            hits=(),
            need_kb=self.need_kb,
            kb_skills=("security-policy",) if self.need_kb else (),
            kb_categories=("security",) if self.need_kb else (),
            reason=self.reason,
            gate_mode="strict",
        )


class _StubTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


_TOOLS = [
    "message",
    "kb_search",
    "read_file",
    "exec",
    "grep",
    # MCP names preserve hyphens in server names (see mcp._sanitize_name).
    "mcp_order-service_create_order",
    "mcp_aws-docs_search",
]


def _router(**kwargs) -> IntentChannelRouter:
    cfg = IntentRouterConfig(
        business_api_mcp_servers=["order-service"],
        **kwargs,
    )
    return IntentChannelRouter(cfg, skill_router=_FakeSkillRouter(need_kb=False))


def test_classify_policy_question_to_knowledge() -> None:
    router = IntentChannelRouter(
        IntentRouterConfig(business_api_mcp_servers=["order-service"]),
        skill_router=_FakeSkillRouter(need_kb=True, reason="query_hint"),
    )
    decision = router.decide(
        "试用期员工请假流程是什么？",
        available_tools=_TOOLS,
    )
    assert decision.channel == "knowledge"
    assert decision.need_kb is True
    assert "kb_search" in decision.allowed_tools
    assert "message" in decision.allowed_tools
    # Strict default: no filesystem helpers on knowledge unless opted in.
    assert "read_file" not in decision.allowed_tools
    assert "mcp_order-service_create_order" not in decision.allowed_tools
    assert "exec" not in decision.allowed_tools


def test_knowledge_extra_tools_opt_in() -> None:
    router = IntentChannelRouter(
        IntentRouterConfig(
            business_api_mcp_servers=["order-service"],
            knowledge_extra_tools=["read_file", "grep"],
        ),
        skill_router=_FakeSkillRouter(need_kb=True),
    )
    decision = router.decide("数据分类政策", available_tools=_TOOLS)
    assert decision.channel == "knowledge"
    assert "read_file" in decision.allowed_tools
    assert "grep" in decision.allowed_tools
    assert "exec" not in decision.allowed_tools


def test_classify_create_order_to_business_api() -> None:
    router = _router()
    decision = router.decide("帮我创建订单", available_tools=_TOOLS)
    assert decision.channel == "business_api"
    assert decision.need_kb is False
    assert "kb_search" not in decision.allowed_tools
    assert "mcp_order-service_create_order" in decision.allowed_tools
    assert "mcp_aws-docs_search" not in decision.allowed_tools
    assert "message" in decision.allowed_tools


def test_business_api_fail_closed_when_no_servers() -> None:
    """Production: do not pretend business_api when nothing is callable."""
    router = IntentChannelRouter(
        IntentRouterConfig(business_api_mcp_servers=[]),
        skill_router=_FakeSkillRouter(need_kb=False),
    )
    decision = router.decide("调用接口下单", available_tools=_TOOLS)
    assert decision.channel == "chat"
    assert "fail_closed_empty_business_api" in decision.notes
    assert decision.allowed_tools == ("message",)
    assert "kb_search" not in decision.allowed_tools


def test_business_api_fail_closed_when_servers_but_no_tools() -> None:
    router = IntentChannelRouter(
        IntentRouterConfig(business_api_mcp_servers=["order-service"]),
        skill_router=_FakeSkillRouter(need_kb=False),
    )
    decision = router.decide(
        "创建订单",
        available_tools=["message", "kb_search", "exec"],
    )
    assert decision.channel == "chat"
    assert "fail_closed_empty_business_api" in decision.notes


def test_classify_pytest_fix_to_workspace() -> None:
    router = _router()
    decision = router.decide("修这个 pytest 报错", available_tools=_TOOLS)
    assert decision.channel == "workspace"
    assert decision.need_kb is False
    assert "kb_search" not in decision.allowed_tools
    assert "exec" in decision.allowed_tools
    assert "grep" in decision.allowed_tools
    assert "mcp_order-service_create_order" not in decision.allowed_tools


def test_classify_greeting_to_chat() -> None:
    router = _router()
    decision = router.decide("你好", available_tools=_TOOLS)
    assert decision.channel == "chat"
    assert decision.need_kb is False
    assert decision.allowed_tools == ("message",)


def test_tie_break_prefers_knowledge_over_business() -> None:
    router = IntentChannelRouter(
        IntentRouterConfig(
            business_api_mcp_servers=["order-service"],
            business_api_hints=["政策"],
            workspace_hints=[],
        ),
        skill_router=_FakeSkillRouter(need_kb=True),
    )
    decision = router.decide("公司政策与下单接口", available_tools=_TOOLS)
    assert decision.channel == "knowledge"


def test_channel_kb_gate_payload_unifies_need_kb() -> None:
    decision = ChannelDecision(
        channel="workspace",
        scores={"knowledge": 0, "business_api": 0, "workspace": 2, "chat": 0.1},
        reason="test",
        need_kb=False,
        allowed_tools=("message", "exec"),
        skill_decision=SkillRouteDecision(
            hits=(),
            need_kb=True,
            kb_skills=(),
            kb_categories=(),
            reason="skill",
            gate_mode="strict",
        ),
        confidence=0.66,
    )
    payload = channel_kb_gate_payload(decision, gate_mode="strict")
    assert payload["need_kb"] is False
    assert payload["intent_channel"] == "workspace"
    assert payload["allowed_tools"] == ["message", "exec"]
    assert payload["enforce"] is True
    assert payload["confidence"] == pytest.approx(0.66)


def test_disabled_router_passthrough_all_tools() -> None:
    router = IntentChannelRouter(
        IntentRouterConfig(enable=False, business_api_mcp_servers=["order-service"]),
        skill_router=_FakeSkillRouter(need_kb=False),
    )
    decision = router.decide("任意问题", available_tools=_TOOLS)
    assert set(decision.allowed_tools) == set(_TOOLS)
    assert "disabled_passthrough" in decision.notes


def test_tool_registry_view_exposes_only_allowed() -> None:
    registry = ToolRegistry()
    for name in _TOOLS:
        registry.register(_StubTool(name))

    view = registry.view({"message", "kb_search"})
    names = set(view.tool_names)
    assert names == {"message", "kb_search"}
    defs = view.get_definitions()
    def_names = {d["function"]["name"] for d in defs}
    assert def_names == {"message", "kb_search"}
    assert "exec" in registry.tool_names
    assert "exec" not in view.tool_names


def test_knowledge_turn_definitions_exclude_mcp() -> None:
    registry = ToolRegistry()
    for name in _TOOLS:
        registry.register(_StubTool(name))
    router = IntentChannelRouter(
        IntentRouterConfig(business_api_mcp_servers=["order-service"]),
        skill_router=_FakeSkillRouter(need_kb=True),
    )
    decision = router.decide("数据分类等级有哪些？", available_tools=registry.tool_names)
    view = registry.view(decision.allowed_tools)
    names = set(view.tool_names)
    assert "kb_search" in names
    assert not any(n.startswith("mcp_") for n in names)


def test_business_api_turn_definitions_exclude_kb_search() -> None:
    registry = ToolRegistry()
    for name in _TOOLS:
        registry.register(_StubTool(name))
    router = _router()
    decision = router.decide("查询订单状态 api", available_tools=registry.tool_names)
    view = registry.view(decision.allowed_tools)
    assert "kb_search" not in view.tool_names
    assert any(n.startswith("mcp_order-service_") for n in view.tool_names)


def test_prepare_call_rechecks_allowlist_even_on_full_registry() -> None:
    """Microsoft-style: tools/call re-auth even if the full registry was passed."""
    registry = ToolRegistry()
    for name in _TOOLS:
        registry.register(_StubTool(name))

    token = bind_request_context(
        RequestContext(
            channel="websocket",
            chat_id="c1",
            kb_gate={
                "enforce": True,
                "intent_channel": "chat",
                "allowed_tools": ["message"],
            },
        )
    )
    try:
        tool, _, err = registry.prepare_call("exec", {})
        assert tool is None
        assert err is not None
        assert "not found" in err
        assert "exec" not in err.split("Available:")[-1] or "Available: message" in err
        ok_tool, _, ok_err = registry.prepare_call("message", {})
        assert ok_tool is not None
        assert ok_err is None
    finally:
        reset_request_context(token)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("你好", "chat"),
        ("修这个 git 报错", "workspace"),
        ("调用 rpc 接口", "business_api"),
    ],
)
def test_parametrized_channels(query: str, expected: str) -> None:
    router = _router()
    assert router.decide(query, available_tools=_TOOLS).channel == expected
