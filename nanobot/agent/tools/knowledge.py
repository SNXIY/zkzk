"""Enterprise knowledge-base search tool (anchored hybrid RAG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ContextAware, RequestContext, current_request_context
from nanobot.knowledge.config import KnowledgeConfig, KnowledgeRetrievalPolicy
from nanobot.knowledge.retrieval_pipeline import run_retrieval_pipeline
from nanobot.knowledge.store import KnowledgeHit, KnowledgeStore
from nanobot.webui.knowledge_api import open_knowledge_store


class KnowledgeSearchTool(Tool, ContextAware):
    """Search enterprise documents via DashVector + local FTS hybrid retrieval.

    Contract:
    - System multi-query (anchor + keyword + optional tool) with weighted RRF
    - Evidence grade on full chunk text; fail-closed when grade fails
    - Bounded corrective re-retrieve
    - Cached store per workspace; ContextVar-bound user turn for concurrency safety
    """

    config_key = "knowledge"
    _scopes = {"core", "subagent"}

    def __init__(self, workspace: Path, cfg: KnowledgeConfig):
        self._workspace = Path(workspace)
        self._cfg = cfg
        self._request_ctx: RequestContext | None = None
        self._store_cache: dict[str, KnowledgeStore] = {}

    @classmethod
    def config_cls(cls):
        return KnowledgeConfig

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        cfg = getattr(ctx.config, "knowledge", None)
        if cfg is None:
            return True
        return bool(getattr(cfg, "enable", True))

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        cfg = getattr(ctx.config, "knowledge", None)
        if not isinstance(cfg, KnowledgeConfig):
            cfg = KnowledgeConfig()
        return cls(workspace=Path(ctx.workspace), cfg=cfg)

    def set_context(self, ctx: RequestContext) -> None:
        self._request_ctx = ctx

    def _workspace_root(self) -> Path:
        try:
            from nanobot.security.workspace_access import current_workspace_scope

            scope = current_workspace_scope()
            if scope is not None and scope.project_path:
                return Path(scope.project_path)
        except Exception:
            pass
        return self._workspace

    def _store(self) -> KnowledgeStore:
        root = self._workspace_root().expanduser().resolve(strict=False)
        key = str(root)
        cached = self._store_cache.get(key)
        if cached is not None:
            return cached
        store = open_knowledge_store(root, self._cfg)
        self._store_cache[key] = store
        return store

    def _policy(self) -> KnowledgeRetrievalPolicy:
        retrieval = getattr(self._cfg, "retrieval", None)
        if isinstance(retrieval, KnowledgeRetrievalPolicy):
            return retrieval
        return KnowledgeRetrievalPolicy()

    def _request(self) -> RequestContext | None:
        ctx_var = current_request_context()
        inst = self._request_ctx
        if ctx_var is not None and (ctx_var.user_text or ctx_var.kb_gate):
            return ctx_var
        if inst is not None and (inst.user_text or inst.kb_gate):
            return inst
        return ctx_var or inst

    @property
    def name(self) -> str:
        return "kb_search"

    @property
    def description(self) -> str:
        return (
            "Search the enterprise knowledge base (DashScope + DashVector hybrid with FTS). "
            "Use only for internal manuals, policies, SLAs, and uploaded docs — not chit-chat or coding. "
            "Runtime runs multi-query retrieval (user-turn anchor + keyword variant + optional query), "
            "weighted RRF, evidence grading, and bounded corrective search. "
            "Keep `query` close to the user wording; cite `filename#chunkN`. "
            "If grade fails or usable_for_answer=false, refuse — do not invent policy."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Optional secondary retrieval string. The system always builds "
                        "anchor + keyword variants from the user turn and fuses them; "
                        "this field is an additional path, not a replacement."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Max snippets to return (1-20)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> Any:
        tool_query = str(kwargs.get("query") or "").strip()
        k = kwargs.get("k", 5)
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = 5

        policy = self._policy()
        req = self._request()
        user_text = ""
        if req is not None and req.user_text:
            user_text = str(req.user_text).strip()
        kb_gate = dict(req.kb_gate) if req and req.kb_gate else {}
        need_kb = bool(kb_gate.get("need_kb"))
        gate_mode = str(kb_gate.get("gate_mode") or policy.gate_mode or "strict")

        # Fail-closed gate: strict + enforceable decision + need_kb=false → block.
        enforce = bool(kb_gate.get("enforce", True)) if kb_gate else False
        if gate_mode == "strict" and enforce and not need_kb:
            logger.info(
                "kb_search blocked by strict gate reason={} tool_query={!r}",
                kb_gate.get("reason") or "-",
                tool_query[:160],
            )
            return (
                "KB gate blocked (strict): this turn is not classified as an enterprise "
                "knowledge question. Do not call kb_search; answer without the knowledge base."
            )

        if not user_text and not tool_query:
            return "Error: query is required (and no user turn text was available to anchor)"

        store = self._store()
        result = run_retrieval_pipeline(
            store,
            anchor=user_text,
            tool_query=tool_query,
            policy=policy,
            k=k_int,
        )
        configured = store.search_mode
        effective = "hybrid" if store.vector_enabled else "lexical"

        logger.info(
            "kb_search audit need_kb={} anchor={!r} tool_query={!r} rounds={} "
            "hits={} usable={} grade={} coverage={:.2f} confidence={} "
            "vector_enabled={} gate={}",
            kb_gate.get("need_kb"),
            user_text[:160],
            tool_query[:160],
            result.rounds,
            len(result.hits),
            result.usable_for_answer,
            result.grade.passed,
            result.grade.coverage,
            result.confidence,
            store.vector_enabled,
            kb_gate.get("reason") or "-",
        )

        if not result.usable_for_answer or not result.hits:
            raw_n = result.audit.get("raw_hit_count", 0)
            return (
                "Knowledge retrieval failed closed "
                f"(configured={configured}, effective={effective}, "
                f"vector_enabled={store.vector_enabled}, usable_for_answer=false, "
                f"grade={result.grade.reason}, coverage={result.grade.coverage:.2f}, "
                f"raw_hits={raw_n}, rounds={result.rounds})\n"
                f"anchor_query: {user_text or '-'}\n"
                f"tool_query: {tool_query or '-'}\n"
                f"variants: {_format_variants(result.variants)}\n"
                "Policy: say the knowledge base does not cover this with sufficient evidence; "
                "do not invent facts."
            )

        sources = sorted({h.source for h in result.hits if h.source})
        lines = [
            "## Retrieval audit",
            f"- configured={configured} effective={effective} sources={','.join(sources) or '-'}",
            f"- usable_for_answer={result.usable_for_answer}",
            f"- confidence={result.confidence}",
            f"- grade_passed={result.grade.passed} coverage={result.grade.coverage:.2f} "
            f"reason={result.grade.reason}",
            f"- rounds={result.rounds}",
            f"- anchor_query: {user_text or '(unavailable)'}",
            f"- tool_query: {tool_query or '(none)'}",
            f"- variants: {_format_variants(result.variants)}",
            f"- need_kb_gate: {kb_gate.get('need_kb', 'unknown')}",
            f"- gate_mode: {gate_mode}",
            "",
            "## Hits",
        ]
        for i, hit in enumerate(result.hits, start=1):
            lines.append(_format_hit(i, hit))
        if result.grade.missing_tokens:
            lines.extend(
                [
                    "",
                    "## Missing evidence tokens",
                    ", ".join(result.grade.missing_tokens[:12]),
                ]
            )
        lines.extend(
            [
                "",
                "## Answer policy",
                "- Ground claims only in the hits above; cite as `filename#chunkN`.",
                "- usable_for_answer=true is required before stating policy facts.",
            ]
        )
        return "\n".join(lines)


def _format_variants(variants: list[Any]) -> str:
    if not variants:
        return "-"
    return "; ".join(f"{v.role}={v.text[:60]!r}" for v in variants)


def _format_hit(index: int, hit: KnowledgeHit) -> str:
    cite = f"{hit.filename}#chunk{hit.chunk_index}"
    crumb = f" | {hit.breadcrumb}" if hit.breadcrumb else ""
    return (
        f"[{index}] cite=`{cite}` title={hit.title}{crumb} "
        f"score={float(hit.score):.6f} source={hit.source or '-'}\n"
        f"{hit.snippet}"
    )
