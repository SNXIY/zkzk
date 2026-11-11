"""DashScope (OpenAI-compatible) embedding client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class OpenAICompatibleEmbedder:
    """``POST {api_base}/embeddings`` — DashScope compatible-mode works here."""

    api_key: str
    api_base: str
    model: str
    dimensions: int
    extra_headers: dict[str, str] | None = None
    # DashScope text-embedding-v3/v4 reject batches larger than 10.
    batch_size: int = 10

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base.rstrip("/"),
            default_headers=self.extra_headers or None,
        )
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": batch,
            }
            # DashScope text-embedding-v3/v4 accept dimensions.
            if self.dimensions > 0:
                kwargs["dimensions"] = self.dimensions
            try:
                resp = client.embeddings.create(**kwargs)
            except Exception:
                if "dimensions" in kwargs:
                    kwargs.pop("dimensions", None)
                    resp = client.embeddings.create(**kwargs)
                else:
                    raise
            items = sorted(resp.data, key=lambda row: row.index)
            for item in items:
                out.append(list(item.embedding))
        return out


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def resolve_embedder(cfg: Any) -> OpenAICompatibleEmbedder | None:
    """Build DashScope embedder from KnowledgeConfig + env / providers."""
    from nanobot.knowledge.config import KnowledgeConfig
    from nanobot.knowledge.env import load_knowledge_env

    load_knowledge_env()

    if not isinstance(cfg, KnowledgeConfig):
        return None
    if not cfg.embeddings:
        return None

    api_key = (cfg.embedding_api_key or "").strip() or _env(
        "NANOBOT_EMBEDDING_API_KEY",
        "GREENBOOK_EMBEDDING_API_KEY",
        "DASHSCOPE_API_KEY",
    )
    api_base = (cfg.embedding_api_base or "").strip() or _env(
        "NANOBOT_EMBEDDING_BASE_URL",
        "GREENBOOK_EMBEDDING_BASE_URL",
    )
    model = (cfg.embedding_model or "").strip() or _env(
        "NANOBOT_EMBEDDING_MODEL",
        "GREENBOOK_EMBEDDING_MODEL",
    ) or "text-embedding-v4"
    dims_raw = _env("NANOBOT_EMBEDDING_DIMENSIONS", "GREENBOOK_EMBEDDING_DIMENSIONS")
    dimensions = cfg.embedding_dimensions
    if dims_raw:
        try:
            dimensions = int(dims_raw)
        except ValueError:
            pass

    extra_headers: dict[str, str] | None = None
    if not api_key:
        try:
            from nanobot.config.loader import load_config

            root = load_config()
            for name in ("dashscope", "openai", "custom", "openrouter"):
                provider = getattr(root.providers, name, None)
                if provider and provider.api_key:
                    api_key = provider.api_key
                    api_base = api_base or (provider.api_base or "")
                    extra_headers = provider.extra_headers
                    break
        except Exception as exc:
            logger.warning("knowledge embedder provider resolve failed: {}", exc)

    if not api_key:
        logger.warning(
            "DashScope embedding key missing; vector index disabled (lexical FTS only)"
        )
        return None
    if not api_base:
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    return OpenAICompatibleEmbedder(
        api_key=api_key,
        api_base=api_base,
        model=model,
        dimensions=dimensions,
        extra_headers=extra_headers,
    )
