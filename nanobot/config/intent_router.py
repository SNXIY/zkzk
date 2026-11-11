"""Multi-channel intent router configuration (production contract).

See ``.agent/intent_channels.md`` for the deny-by-default / fail-closed rules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, field_validator

from nanobot.config_base import Base

IntentChannelName = Literal["knowledge", "business_api", "workspace", "chat"]


class IntentRouterConfig(Base):
    """Per-turn channel classification + tool allowlist pruning.

    Defaults are strict: empty business MCP list, no knowledge extras, fail closed
    when a business channel cannot actually invoke tools.
    """

    enable: bool = True
    business_api_mcp_servers: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "business_api_mcp_servers",
            "businessApiMcpServers",
        ),
        description="MCP server names whose tools belong to the business_api channel",
    )
    business_api_hints: list[str] = Field(
        default_factory=lambda: [
            "下单",
            "订单",
            "接口",
            "api",
            "rpc",
            "调用服务",
            "调用接口",
            "java",
            "微服务",
            "创建订单",
            "查询订单",
            "业务接口",
            "openapi",
        ],
        validation_alias=AliasChoices("business_api_hints", "businessApiHints"),
        description="Lexical signals for business_api (not an LLM classifier)",
    )
    workspace_hints: list[str] = Field(
        default_factory=lambda: [
            "代码",
            "文件",
            "git",
            "pytest",
            "报错",
            "refactor",
            "bug",
            "修复",
            "编译",
            "test",
            "unittest",
            "shell",
            "终端",
            "仓库",
            "pull request",
            "pr",
            "commit",
            "traceback",
            "stack trace",
        ],
        validation_alias=AliasChoices("workspace_hints", "workspaceHints"),
        description="Lexical signals for workspace (not an LLM classifier)",
    )
    always_tools: list[str] = Field(
        default_factory=lambda: ["message"],
        validation_alias=AliasChoices("always_tools", "alwaysTools"),
    )
    knowledge_extra_tools: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("knowledge_extra_tools", "knowledgeExtraTools"),
        description=(
            "Opt-in read-only helpers on knowledge turns (default empty — "
            "kb_search + always_tools only)"
        ),
    )
    workspace_tools: list[str] = Field(
        default_factory=lambda: [
            "apply_patch",
            "edit_file",
            "write_file",
            "read_file",
            "list_dir",
            "find_files",
            "grep",
            "exec",
            "write_stdin",
            "list_exec_sessions",
            "run_cli_app",
            "spawn",
            "web_search",
            "web_fetch",
            "cron",
            "long_task",
            "complete_goal",
            "my",
        ],
        validation_alias=AliasChoices("workspace_tools", "workspaceTools"),
    )
    # Tie-break when scores equal: earlier = higher priority
    channel_priority: list[str] = Field(
        default_factory=lambda: ["knowledge", "business_api", "workspace", "chat"],
        validation_alias=AliasChoices("channel_priority", "channelPriority"),
    )
    min_channel_score: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices("min_channel_score", "minChannelScore"),
        description="Non-chat channels must reach this absolute score or fall back to chat",
    )
    score_margin: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("score_margin", "scoreMargin"),
        description=(
            "Winner must beat the next channel by this margin; otherwise fall back to chat "
            "(0 = priority-list tie-break only)"
        ),
    )
    fail_closed_empty_business_api: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "fail_closed_empty_business_api",
            "failClosedEmptyBusinessApi",
        ),
        description=(
            "If business_api wins but no allowlisted MCP tools are registered, "
            "downgrade to chat instead of a fake business turn"
        ),
    )

    @field_validator("channel_priority")
    @classmethod
    def _priority_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        if not cleaned:
            return ["knowledge", "business_api", "workspace", "chat"]
        return cleaned
