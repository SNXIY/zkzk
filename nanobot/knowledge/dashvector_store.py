"""DashVector client wrapper for enterprise document chunks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float
    fields: dict[str, Any]


class DashVectorIndex:
    """Thin wrapper around Alibaba DashVector collection upsert/query/delete."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        collection: str,
        dimension: int,
        metric: str = "cosine",
        auto_create: bool = True,
    ) -> None:
        import dashvector

        self.collection_name = collection
        self.dimension = dimension
        self.client = dashvector.Client(api_key=api_key, endpoint=endpoint)
        col = self.client.get(name=collection)
        if not col:
            if not auto_create:
                raise RuntimeError(
                    f"DashVector collection not found: {collection} "
                    f"(code={getattr(col, 'code', '?')} message={getattr(col, 'message', '')})"
                )
            logger.info(
                "Creating DashVector collection name={} dim={} metric={}",
                collection,
                dimension,
                metric,
            )
            created = self.client.create(
                name=collection,
                dimension=dimension,
                metric=metric,
            )
            if not created:
                raise RuntimeError(
                    f"failed to create DashVector collection {collection}: "
                    f"code={getattr(created, 'code', '?')} message={getattr(created, 'message', created)}"
                )
            col = self.client.get(name=collection)
            if not col:
                raise RuntimeError(f"unable to open DashVector collection after create: {collection}")
        self.collection = col

    def upsert_chunks(
        self,
        items: list[tuple[str, list[float], dict[str, Any]]],
        *,
        batch_size: int = 32,
    ) -> None:
        from dashvector import Doc

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            docs = [
                Doc(id=chunk_id, vector=vector, fields=fields)
                for chunk_id, vector, fields in batch
            ]
            ret = self.collection.upsert(docs)
            if not ret:
                raise RuntimeError(
                    f"DashVector upsert failed: code={getattr(ret, 'code', '?')} "
                    f"message={getattr(ret, 'message', ret)}"
                )

    def delete_by_doc_id(
        self,
        doc_id: str,
        *,
        known_chunk_ids: list[str] | None = None,
        page_size: int = 1024,
        max_pages: int = 256,
    ) -> None:
        """Delete all vectors for a document.

        DashVector filter queries are capped by ``topk``, so a single page can
        leave orphans on large docs. We:
        1) delete any known deterministic ids (``{doc_id}_{idx}``)
        2) paginate filter-query → delete until empty
        """
        pending: list[str] = []
        seen: set[str] = set()
        for chunk_id in known_chunk_ids or []:
            cid = str(chunk_id or "").strip()
            if cid and cid not in seen:
                seen.add(cid)
                pending.append(cid)
        self._delete_ids(pending)
        pending.clear()

        filt = f"doc_id = '{doc_id}'"
        for _ in range(max(1, max_pages)):
            try:
                ret = self.collection.query(
                    topk=max(1, min(int(page_size), 1024)),
                    filter=filt,
                    include_vector=False,
                    output_fields=["doc_id"],
                )
            except Exception as exc:
                logger.warning("DashVector filter query for delete failed: {}", exc)
                return
            if not ret:
                return
            output = getattr(ret, "output", None) or []
            batch = [
                str(getattr(doc, "id", ""))
                for doc in output
                if getattr(doc, "id", None)
            ]
            batch = [i for i in batch if i and i not in seen]
            if not batch:
                return
            seen.update(batch)
            self._delete_ids(batch)

    def _delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        for i in range(0, len(ids), 64):
            chunk = ids[i : i + 64]
            try:
                ret = self.collection.delete(chunk)
            except Exception as exc:
                logger.warning("DashVector delete failed for {} ids: {}", len(chunk), exc)
                continue
            if not ret:
                logger.warning(
                    "DashVector delete returned error: code={} message={}",
                    getattr(ret, "code", "?"),
                    getattr(ret, "message", ret),
                )

    def query(self, vector: list[float], *, topk: int = 10) -> list[VectorHit]:
        ret = self.collection.query(
            vector,
            topk=topk,
            include_vector=False,
            output_fields=["doc_id", "title", "filename", "breadcrumb", "chunk_index", "text"],
        )
        if not ret:
            return []
        output = getattr(ret, "output", None) or []
        hits: list[VectorHit] = []
        for doc in output:
            fields = dict(getattr(doc, "fields", None) or {})
            hits.append(
                VectorHit(
                    chunk_id=str(getattr(doc, "id", "")),
                    score=float(getattr(doc, "score", 0.0) or 0.0),
                    fields=fields,
                )
            )
        return hits


_DASHVECTOR_DISABLED_REASON: str | None = None


def resolve_dashvector(cfg: Any) -> DashVectorIndex | None:
    """Open DashVector from KnowledgeConfig + env."""
    global _DASHVECTOR_DISABLED_REASON
    from nanobot.knowledge.config import KnowledgeConfig
    from nanobot.knowledge.env import load_knowledge_env

    if _DASHVECTOR_DISABLED_REASON == "missing_sdk":
        return None

    load_knowledge_env()

    if not isinstance(cfg, KnowledgeConfig):
        return None

    api_key = (cfg.dashvector_api_key or "").strip() or _env(
        "NANOBOT_DASHVECTOR_API_KEY",
        "GREENBOOK_DASHVECTOR_API_KEY",
        "DASHVECTOR_API_KEY",
    )
    endpoint = (cfg.dashvector_endpoint or "").strip() or _env(
        "NANOBOT_DASHVECTOR_ENDPOINT",
        "GREENBOOK_DASHVECTOR_ENDPOINT",
        "DASHVECTOR_ENDPOINT",
    )
    collection = (cfg.dashvector_collection or "").strip() or _env(
        "NANOBOT_DASHVECTOR_COLLECTION",
        "GREENBOOK_DASHVECTOR_COLLECTION",
        "GREENBOOK_CHROMA_COLLECTION",
    ) or "nanobot_enterprise_docs"

    if not api_key or not endpoint:
        logger.warning(
            "DashVector api_key/endpoint missing; vector retrieval disabled "
            "(lexical FTS still available)"
        )
        return None

    try:
        return DashVectorIndex(
            api_key=api_key,
            endpoint=endpoint,
            collection=collection,
            dimension=cfg.embedding_dimensions,
            metric=cfg.dashvector_metric,
            auto_create=cfg.dashvector_auto_create,
        )
    except ImportError as exc:
        _DASHVECTOR_DISABLED_REASON = "missing_sdk"
        logger.warning(
            "DashVector SDK missing ({}). Install with: pip install 'nanobot[knowledge]' "
            "or: pip install dashvector. Falling back to lexical FTS.",
            exc,
        )
        return None
    except Exception as exc:
        msg = str(exc)
        if "No module named 'dashvector'" in msg:
            _DASHVECTOR_DISABLED_REASON = "missing_sdk"
        logger.warning("DashVector init failed: {}; lexical FTS still available", exc)
        return None


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None
