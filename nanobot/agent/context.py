"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanobot.agent.intent.router import ChannelDecision
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillRouteDecision, SkillRouter, SkillsLoader
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import InboundMessage
from nanobot.config.skill_router import SkillRouterConfig
from nanobot.session.goal_state import goal_state_runtime_lines
from nanobot.utils.helpers import (
    current_time_str,
    detect_image_mime,
    load_bundled_template,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    return cli_app_utils.session_extra(metadata) | mcp_tools.session_extra(metadata)


def runtime_lines(state: Any, msg: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """Return model-visible runtime annotations for turn-attached capabilities."""
    return [
        *cli_app_utils.runtime_lines(msg, workspace, skip=skip),
        *mcp_tools.runtime_lines(
            msg,
            configured_server_names=set(state._mcp_servers),
            connected_server_names=set(state._mcp_stacks),
            skip=skip,
        ),
    ]


async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    await mcp_tools.connect_missing_servers(state, tools)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    return await mcp_tools.handle_runtime_control(state, msg, tools)


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_CHARS = 32_000  # hard cap on recent history section size
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        skill_router: SkillRouterConfig | None = None,
        knowledge_cfg: Any | None = None,
    ):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)
        self.skill_router_cfg = skill_router or SkillRouterConfig()
        self._knowledge_cfg = knowledge_cfg
        self._skill_router: SkillRouter | None = None

    def _router(self) -> SkillRouter:
        if self._skill_router is None:
            self._skill_router = SkillRouter(
                self.skills,
                self.skill_router_cfg,
                knowledge_cfg=self._knowledge_cfg,
            )
        return self._skill_router

    def route_turn(self, route_query: str | None) -> SkillRouteDecision | None:
        """Enterprise skill + KB gate decision for the current user turn."""
        q = (route_query or "").strip()
        if not q or not self.skill_router_cfg.enable:
            return None
        return self._router().decide(q)

    def kb_gate_payload(
        self,
        decision: SkillRouteDecision | None = None,
        *,
        channel_decision: ChannelDecision | None = None,
    ) -> dict[str, Any]:
        """Serialize KB / channel gate for ``RequestContext.kb_gate`` / tool audit.

        Prefer ``channel_decision`` when the intent router ran so need_kb and
        tool allowlists stay aligned with the pruned tool view.
        """
        from nanobot.agent.intent.router import channel_kb_gate_payload
        from nanobot.knowledge.config import KnowledgeRetrievalPolicy

        retrieval = getattr(self._knowledge_cfg, "retrieval", None)
        if not isinstance(retrieval, KnowledgeRetrievalPolicy):
            retrieval = KnowledgeRetrievalPolicy()
        default_mode = str(retrieval.gate_mode or "strict")
        if channel_decision is not None:
            skill = channel_decision.skill_decision
            mode = (
                (skill.gate_mode if skill and skill.gate_mode else None) or default_mode
            )
            return channel_kb_gate_payload(channel_decision, gate_mode=mode)
        if decision is None:
            # Without a router decision we cannot classify the turn — do not
            # hard-block kb_search, but also do not pretend need_kb=true.
            return {
                "need_kb": False,
                "reason": "no_decision",
                "gate_mode": default_mode,
                "enforce": False,
                "kb_skills": [],
                "kb_categories": [],
                "routed_skills": [],
            }
        return {
            "need_kb": bool(decision.need_kb),
            "reason": decision.reason,
            "gate_mode": decision.gate_mode or default_mode,
            "enforce": True,
            "kb_skills": list(decision.kb_skills),
            "kb_categories": list(decision.kb_categories),
            "routed_skills": [h.name for h in decision.hits],
        }

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
        route_query: str | None = None,
        route_decision: SkillRouteDecision | None = None,
        channel_decision: ChannelDecision | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        root = workspace or self.workspace
        parts = [self._get_identity(channel=channel, workspace=root)]

        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)

        parts.append(render_template("agent/tool_contract.md"))

        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        routed_names: list[str] = list(skill_names or [])
        route_hits = []
        cfg = self.skill_router_cfg
        decision = route_decision
        if decision is None and channel_decision is not None:
            decision = channel_decision.skill_decision
        if decision is None and cfg.enable and route_query:
            decision = self.route_turn(route_query)
        if decision is not None:
            route_hits = list(decision.hits)
            for hit in route_hits:
                if hit.name not in always_skills and hit.name not in routed_names:
                    routed_names.append(hit.name)
        elif cfg.enable and route_query:
            route_hits = self._router().route(route_query)
            for hit in route_hits:
                if hit.name not in always_skills and hit.name not in routed_names:
                    routed_names.append(hit.name)

        # Inject full bodies for top routed skills (progressive disclosure depth-2).
        full_names = [
            name
            for name in routed_names[: max(0, cfg.inject_full_top_n)]
            if name not in always_skills
        ]
        if full_names:
            full_content = self.skills.load_skills_for_context(full_names)
            if full_content:
                parts.append(f"# Routed Skills (full)\n\n{full_content}")

        exclude = set(always_skills) | set(full_names)
        if cfg.enable and route_query and not cfg.include_unrouted_names:
            only = set(routed_names) - exclude
            skills_summary = self.skills.build_skills_summary(
                exclude=exclude,
                only=only if only else set(),
                group_by_category=True,
            )
            if cfg.include_category_directory:
                cat_dir = self.skills.build_category_directory()
                if cat_dir:
                    skills_summary = f"{cat_dir}\n\n{skills_summary}" if skills_summary else cat_dir
            if route_hits:
                route_lines = "\n".join(
                    f"- {h.name} (score={h.score:.3f}, cat={h.category or '-'}, via={h.source})"
                    for h in route_hits
                )
                skills_summary = (
                    f"### Router selection for this turn\n{route_lines}\n\n{skills_summary}"
                    if skills_summary
                    else f"### Router selection for this turn\n{route_lines}"
                )
        else:
            skills_summary = self.skills.build_skills_summary(
                exclude=exclude,
                group_by_category=True,
            )
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        channel_policy = self._channel_policy_section(channel_decision)
        if channel_policy:
            parts.append(channel_policy)

        kb_policy = self._knowledge_policy_section(
            decision,
            channel_decision=channel_decision,
        )
        if kb_policy:
            parts.append(kb_policy)

        if include_memory_recent_history:
            entries = self.memory.read_recent_history_for_prompt(
                since_cursor=self.memory.get_last_dream_cursor(),
                session_key=session_key,
                unified_session=unified_session,
            )
            if entries:
                capped = entries[-self._MAX_RECENT_HISTORY:]
                history_text = "\n".join(
                    f"- [{e['timestamp']}] {e['content']}" for e in capped
                )
                history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
                parts.append("# Recent History\n\n" + history_text)

        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        return "\n\n---\n\n".join(parts)

    def _channel_policy_section(self, channel_decision: ChannelDecision | None) -> str:
        """Inject per-turn execution-channel policy (hard tool prune companion)."""
        if channel_decision is None:
            return ""
        return render_template(
            "agent/channel_policy.md",
            channel=channel_decision.channel,
            need_kb=channel_decision.need_kb,
            allowed_tools=", ".join(channel_decision.allowed_tools) or "(none)",
            notes=", ".join(channel_decision.notes),
            reason=channel_decision.reason,
        )

    def _knowledge_policy_section(
        self,
        decision: SkillRouteDecision | None,
        *,
        channel_decision: ChannelDecision | None = None,
    ) -> str:
        """Inject enterprise KB gate instructions for this turn."""
        from nanobot.knowledge.config import KnowledgeRetrievalPolicy

        retrieval = getattr(self._knowledge_cfg, "retrieval", None)
        if not isinstance(retrieval, KnowledgeRetrievalPolicy):
            retrieval = KnowledgeRetrievalPolicy()
        if retrieval.gate_mode == "off":
            return ""
        if channel_decision is not None:
            need_kb = bool(channel_decision.need_kb)
            reason = channel_decision.reason
            skill = channel_decision.skill_decision
            kb_skills = ", ".join(skill.kb_skills) if skill and skill.kb_skills else ""
            kb_categories = (
                ", ".join(skill.kb_categories) if skill and skill.kb_categories else ""
            )
        else:
            need_kb = bool(decision.need_kb) if decision is not None else False
            reason = decision.reason if decision is not None else "router_disabled_or_empty"
            kb_skills = ", ".join(decision.kb_skills) if decision and decision.kb_skills else ""
            kb_categories = (
                ", ".join(decision.kb_categories) if decision and decision.kb_categories else ""
            )
        return render_template(
            "agent/knowledge_policy.md",
            gate_mode=retrieval.gate_mode,
            need_kb=need_kb,
            reason=reason,
            kb_skills=kb_skills,
            kb_categories=kb_categories,
            max_search_rounds=retrieval.max_search_rounds_hint,
            require_citations=retrieval.require_citations,
        )

    def _get_identity(self, channel: str | None = None, workspace: Path | None = None) -> str:
        """Get the core identity section."""
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        sender_id: str | None = None,
        supplemental_lines: Sequence[str] | None = None,
    ) -> str:
        """Build untrusted runtime metadata block appended after user content."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if supplemental_lines:
            lines.extend(supplemental_lines)
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        root = workspace or self.workspace

        for filename in self.BOOTSTRAP_FILES:
            file_path = root / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        tpl = load_bundled_template(template_path)
        if tpl is not None:
            return content.strip() == tpl.strip()
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        sender_id: str | None = None,
        session_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        current_runtime_lines: Sequence[str] | None = None,
        workspace: Path | None = None,
        runtime_state: Any | None = None,
        inbound_message: Any | None = None,
        skip_runtime_lines: bool = False,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
        channel_decision: ChannelDecision | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        root = workspace or self.workspace
        extra = [
            *goal_state_runtime_lines(session_metadata),
        ]
        if runtime_state is not None and inbound_message is not None:
            extra.extend(runtime_lines(runtime_state, inbound_message, root, skip=skip_runtime_lines))
        if current_runtime_lines:
            extra.extend(line for line in current_runtime_lines if line)
        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            sender_id=sender_id,
            supplemental_lines=extra or None,
        )
        user_content = self._build_user_content(current_message, media)
        route_query = current_message if isinstance(current_message, str) else None
        if channel_decision is not None:
            route_decision = channel_decision.skill_decision
        else:
            route_decision = self.route_turn(route_query)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        # Runtime context is appended to keep the user-content prefix stable
        # for prompt-cache hits (the context changes every turn due to time).
        if isinstance(user_content, str):
            merged = f"{user_content}\n\n{runtime_ctx}"
        else:
            merged = user_content + [{"type": "text", "text": runtime_ctx}]
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    include_memory_recent_history=include_memory_recent_history,
                    session_key=session_key,
                    unified_session=unified_session,
                    route_query=route_query,
                    route_decision=route_decision,
                    channel_decision=channel_decision,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]
