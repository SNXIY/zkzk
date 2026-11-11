"""WebUI knowledge-base API helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.knowledge.config import KnowledgeConfig
from nanobot.knowledge.dashvector_store import resolve_dashvector
from nanobot.knowledge.embeddings import resolve_embedder
from nanobot.knowledge.store import KnowledgeStore


def knowledge_config_from_tools(tools_cfg: Any | None) -> KnowledgeConfig:
    if tools_cfg is None:
        return KnowledgeConfig()
    if isinstance(tools_cfg, KnowledgeConfig):
        return tools_cfg
    raw = getattr(tools_cfg, "knowledge", None)
    if isinstance(raw, KnowledgeConfig):
        return raw
    if raw is None:
        return KnowledgeConfig()
    try:
        return KnowledgeConfig.model_validate(raw)
    except Exception:
        return KnowledgeConfig()


def open_knowledge_store(workspace: Path, tools_cfg: Any | None = None) -> KnowledgeStore:
    from nanobot.knowledge.minio_storage import resolve_object_storage

    cfg = knowledge_config_from_tools(tools_cfg)
    embedder = resolve_embedder(cfg)
    vector_index = resolve_dashvector(cfg) if embedder is not None else None
    object_storage = resolve_object_storage(cfg)
    return KnowledgeStore(
        workspace,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        max_upload_mb=cfg.max_upload_mb,
        embedder=embedder,
        vector_index=vector_index,
        search_mode=cfg.search_mode,
        vector_top_k=cfg.vector_top_k,
        lexical_top_k=cfg.lexical_top_k,
        knowledge_cfg=cfg,
        object_storage=object_storage,
    )


def knowledge_list_payload(workspace: Path, tools_cfg: Any | None = None) -> dict[str, Any]:
    cfg = knowledge_config_from_tools(tools_cfg)
    store = open_knowledge_store(workspace, tools_cfg)
    docs = store.list_documents()
    vector_enabled = bool(store.vector_enabled)
    effective_mode = "hybrid" if vector_enabled and cfg.search_mode == "hybrid" else (
        "vector" if vector_enabled and cfg.search_mode == "vector" else "lexical"
    )
    if vector_enabled:
        health = "ok"
        status_message = "Hybrid RAG ready (DashScope + DashVector + FTS)."
    elif store.embedder is None:
        health = "degraded_lexical_only"
        status_message = (
            "Embedding unavailable — running lexical FTS only. "
            "Check DashScope / embedding API key."
        )
    else:
        health = "degraded_lexical_only"
        status_message = (
            "DashVector unavailable — running lexical FTS only. "
            "Check API key, endpoint, IP whitelist, or `pip install dashvector`."
        )
    if cfg.async_index:
        status_message = (
            f"{status_message} Async pipeline on "
            f"(≥{cfg.async_threshold_mb}MB → MinIO/outbox"
            f"{'+Kafka' if cfg.kafka.enabled else ', local drain'})."
        )
    return {
        "documents": docs,
        "enabled": cfg.enable,
        "max_upload_mb": store.max_upload_bytes // (1024 * 1024),
        "backend": {
            "chunking": "structure-aware",
            "embedding": {
                "enabled": bool(store.embedder),
                "model": cfg.embedding_model,
                "dimensions": cfg.embedding_dimensions,
                "api_base": cfg.embedding_api_base,
            },
            "vector": {
                "provider": "dashvector" if vector_enabled else "none",
                "collection": cfg.dashvector_collection,
                "enabled": vector_enabled,
            },
            "search_mode": cfg.search_mode,
            "effective_mode": effective_mode,
            "vector_enabled": vector_enabled,
            "health": health,
            "status_message": status_message,
            "async_index": bool(cfg.async_index),
            "async_threshold_mb": cfg.async_threshold_mb,
            "storage": {
                "backend": cfg.storage.backend,
                "bucket": cfg.storage.bucket,
                "endpoint": cfg.storage.endpoint,
            },
            "kafka": {
                "enabled": bool(cfg.kafka.enabled),
                "bootstrap_servers": cfg.kafka.bootstrap_servers,
                "topic_index": cfg.kafka.topic_index,
            },
            "tenant_id": cfg.tenant_id,
            "direct_upload": knowledge_direct_upload_available(tools_cfg),
        },
    }


def knowledge_upload(
    workspace: Path,
    *,
    filename: str,
    data: bytes,
    tools_cfg: Any | None = None,
) -> dict[str, Any]:
    store = open_knowledge_store(workspace, tools_cfg)
    doc = store.add_document(filename, data)
    return {"document": doc}


def knowledge_direct_upload_available(tools_cfg: Any | None = None) -> bool:
    cfg = knowledge_config_from_tools(tools_cfg)
    if cfg.storage.backend != "minio":
        return False
    from nanobot.knowledge.minio_storage import resolve_object_storage

    return resolve_object_storage(cfg) is not None


def knowledge_upload_init(
    workspace: Path,
    *,
    filename: str,
    size: int,
    content_type: str = "application/octet-stream",
    tools_cfg: Any | None = None,
) -> dict[str, Any]:
    """Start browser→MinIO direct upload (single PUT or multipart)."""
    import uuid
    from pathlib import Path as _Path

    from nanobot.knowledge.minio_storage import resolve_object_storage
    from nanobot.knowledge.store import _safe_filename

    cfg = knowledge_config_from_tools(tools_cfg)
    store = open_knowledge_store(workspace, tools_cfg)
    if size <= 0:
        raise ValueError("empty file")
    if size > store.max_upload_bytes:
        raise ValueError(f"file exceeds {store.max_upload_bytes // (1024 * 1024)}MB limit")

    storage = resolve_object_storage(cfg)
    if storage is None:
        return {"mode": "legacy_ws", "reason": "minio_unavailable"}

    safe = _safe_filename(filename)
    if not safe:
        raise ValueError("filename is required")
    doc_id = uuid.uuid4().hex
    tenant_id = cfg.tenant_id or "default"
    object_key = storage.object_key(tenant_id=tenant_id, doc_id=doc_id, filename=safe)
    expires = int(cfg.storage.presign_expires_seconds)
    part_size = max(5, int(cfg.storage.part_size_mb)) * 1024 * 1024
    ctype = (content_type or "application/octet-stream").strip() or "application/octet-stream"

    # Single PUT for smaller objects; multipart otherwise (S3-style).
    if size <= part_size:
        put_url = storage.presign_put(object_key, expires_seconds=expires, content_type=ctype)
        return {
            "mode": "presign_put",
            "doc_id": doc_id,
            "filename": safe,
            "title": _Path(safe).stem,
            "object_key": object_key,
            "bytes": size,
            "content_type": ctype,
            "put_url": put_url,
            "expires_seconds": expires,
        }

    upload_id = storage.create_multipart_upload(object_key, content_type=ctype)
    part_count = storage.plan_parts(size, part_size=part_size)
    parts = []
    for n in range(1, part_count + 1):
        parts.append(
            {
                "part_number": n,
                "url": storage.presign_upload_part(
                    object_key,
                    upload_id=upload_id,
                    part_number=n,
                    expires_seconds=expires,
                ),
            }
        )
    return {
        "mode": "presign_multipart",
        "doc_id": doc_id,
        "filename": safe,
        "title": _Path(safe).stem,
        "object_key": object_key,
        "bytes": size,
        "content_type": ctype,
        "upload_id": upload_id,
        "part_size": part_size,
        "part_count": part_count,
        "parts": parts,
        "expires_seconds": expires,
    }


def knowledge_upload_complete(
    workspace: Path,
    *,
    filename: str,
    doc_id: str,
    object_key: str,
    size: int,
    mode: str,
    upload_id: str | None = None,
    parts: list[dict[str, Any]] | None = None,
    sha256: str = "",
    tools_cfg: Any | None = None,
) -> dict[str, Any]:
    """Finish direct upload: complete multipart if needed, then queue indexing."""
    from nanobot.knowledge.minio_storage import resolve_object_storage

    cfg = knowledge_config_from_tools(tools_cfg)
    storage = resolve_object_storage(cfg)
    if storage is None:
        raise RuntimeError("MinIO storage is not configured")

    if mode == "presign_multipart":
        if not upload_id:
            raise ValueError("upload_id is required")
        if not parts:
            raise ValueError("parts are required")
        stored = storage.complete_multipart_upload(
            object_key, upload_id=upload_id, parts=parts
        )
        size = stored.size or size
    else:
        # Verify object exists after single PUT.
        try:
            size = storage.head_size(object_key) or size
        except Exception as exc:
            raise ValueError(f"uploaded object missing: {exc}") from exc

    store = open_knowledge_store(workspace, tools_cfg)
    doc = store.register_minio_document(
        filename,
        doc_id=doc_id,
        object_key=object_key,
        size=size,
        sha256=sha256,
    )
    return {"document": doc}


def knowledge_delete(workspace: Path, doc_id: str, tools_cfg: Any | None = None) -> dict[str, Any]:
    store = open_knowledge_store(workspace, tools_cfg)
    deleted = store.delete_document(doc_id)
    if not deleted:
        raise LookupError("document not found")
    return {"deleted": True, "id": doc_id}


def knowledge_reindex(workspace: Path, tools_cfg: Any | None = None) -> dict[str, Any]:
    store = open_knowledge_store(workspace, tools_cfg)
    return store.reindex()
