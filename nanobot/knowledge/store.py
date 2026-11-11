"""Enterprise knowledge store: local FTS metadata + DashVector dense retrieval."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from nanobot.knowledge.dashvector_store import DashVectorIndex
from nanobot.knowledge.embeddings import Embedder
from nanobot.knowledge.ingest import chunk_document, extract_text

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\-]+")
_FTS_TOKEN = re.compile(r'["]')


@dataclass(frozen=True)
class KnowledgeHit:
    doc_id: str
    title: str
    filename: str
    chunk_index: int
    snippet: str
    score: float
    breadcrumb: str = ""
    source: str = ""  # vector | lexical | hybrid
    text: str = ""  # full chunk text for grading (snippet may be truncated)


class KnowledgeStore:
    """Single-tenant KB: SQLite FTS for lexical + DashVector for dense vectors."""

    def __init__(
        self,
        workspace: Path,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_upload_mb: int = 32,
        embedder: Embedder | None = None,
        vector_index: DashVectorIndex | None = None,
        search_mode: str = "hybrid",
        vector_top_k: int = 20,
        lexical_top_k: int = 20,
        knowledge_cfg: Any | None = None,
        object_storage: Any | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.root = self.workspace / "knowledge"
        self.docs_dir = self.root / "docs"
        self.db_path = self.root / "index.sqlite"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_bytes = max(1, int(max_upload_mb)) * 1024 * 1024
        self.embedder = embedder
        self.vector_index = vector_index
        self.search_mode = search_mode
        self.vector_top_k = vector_top_k
        self.lexical_top_k = lexical_top_k
        self.knowledge_cfg = knowledge_cfg
        self.object_storage = object_storage
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def vector_enabled(self) -> bool:
        return self.embedder is not None and self.vector_index is not None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.id, d.filename, d.title, d.bytes, d.status, d.error,
                       d.created_at, d.updated_at, d.object_key, d.storage_backend, d.tenant_id,
                       (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunk_count
                FROM documents d
                ORDER BY d.updated_at DESC, d.filename ASC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "filename": row["filename"],
                "title": row["title"],
                "bytes": row["bytes"],
                "status": row["status"],
                "error": row["error"] or "",
                "chunk_count": row["chunk_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "object_key": row["object_key"] or "",
                "storage_backend": row["storage_backend"] or "local",
                "tenant_id": row["tenant_id"] or "default",
                "vector_backend": "dashvector" if self.vector_enabled else "none",
            }
            for row in rows
        ]

    def _should_async(self, size: int) -> bool:
        cfg = self.knowledge_cfg
        if cfg is None:
            return False
        if not getattr(cfg, "async_index", False):
            return False
        threshold = int(getattr(cfg, "async_threshold_mb", 5) or 0) * 1024 * 1024
        # Always async when MinIO is active and async_index is on for files >= threshold.
        return size >= max(0, threshold)

    def add_document(self, filename: str, data: bytes) -> dict[str, Any]:
        if not data:
            raise ValueError("empty file")
        if len(data) > self.max_upload_bytes:
            raise ValueError(f"file exceeds {self.max_upload_bytes // (1024 * 1024)}MB limit")

        safe = _safe_filename(filename)
        doc_id = uuid.uuid4().hex
        digest = hashlib.sha256(data).hexdigest()
        stored_name = f"{doc_id}_{safe}"
        now = time.time()
        title = Path(safe).stem
        tenant_id = str(getattr(self.knowledge_cfg, "tenant_id", None) or "default")
        use_async = self._should_async(len(data))
        object_key = ""
        storage_backend = "local"
        dest = self.docs_dir / stored_name

        if self.object_storage is not None:
            storage_backend = "minio"
            object_key = self.object_storage.object_key(
                tenant_id=tenant_id, doc_id=doc_id, filename=safe
            )
            self.object_storage.put_bytes(object_key, data)
            # Keep a small local cache copy for sync path / offline fallback.
            if not use_async:
                dest.write_bytes(data)
        else:
            dest.write_bytes(data)

        return self._insert_document_row(
            doc_id=doc_id,
            safe=safe,
            stored_name=stored_name,
            title=title,
            size=len(data),
            digest=digest,
            now=now,
            tenant_id=tenant_id,
            object_key=object_key,
            storage_backend=storage_backend,
            use_async=use_async,
            dest=dest,
        )

    def register_minio_document(
        self,
        filename: str,
        *,
        doc_id: str,
        object_key: str,
        size: int,
        sha256: str = "",
    ) -> dict[str, Any]:
        """Register a doc whose bytes already live in MinIO (presigned direct upload)."""
        if size <= 0:
            raise ValueError("empty file")
        if size > self.max_upload_bytes:
            raise ValueError(f"file exceeds {self.max_upload_bytes // (1024 * 1024)}MB limit")
        if self.object_storage is None:
            raise RuntimeError("MinIO storage is not configured")

        safe = _safe_filename(filename)
        doc_id = (doc_id or "").strip() or uuid.uuid4().hex
        stored_name = f"{doc_id}_{safe}"
        now = time.time()
        title = Path(safe).stem
        tenant_id = str(getattr(self.knowledge_cfg, "tenant_id", None) or "default")
        digest = (sha256 or "").strip() or f"pending:{doc_id}"
        # Direct MinIO uploads always enter the async index path.
        return self._insert_document_row(
            doc_id=doc_id,
            safe=safe,
            stored_name=stored_name,
            title=title,
            size=size,
            digest=digest,
            now=now,
            tenant_id=tenant_id,
            object_key=object_key,
            storage_backend="minio",
            use_async=True,
            dest=self.docs_dir / stored_name,
        )

    def _insert_document_row(
        self,
        *,
        doc_id: str,
        safe: str,
        stored_name: str,
        title: str,
        size: int,
        digest: str,
        now: float,
        tenant_id: str,
        object_key: str,
        storage_backend: str,
        use_async: bool,
        dest: Path,
    ) -> dict[str, Any]:
        status = "queued" if use_async else "indexing"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    id, filename, stored_name, title, bytes, sha256, status, error,
                    created_at, updated_at, object_key, storage_backend, tenant_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    safe,
                    stored_name,
                    title,
                    size,
                    digest,
                    status,
                    "",
                    now,
                    now,
                    object_key,
                    storage_backend,
                    tenant_id,
                ),
            )
            if use_async:
                payload = {
                    "event_type": "knowledge.index.requested",
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "filename": safe,
                    "title": title,
                    "sha256": digest,
                    "bytes": size,
                    "object_key": object_key,
                    "storage_backend": storage_backend,
                    "stored_name": stored_name,
                }
                conn.execute(
                    """
                    INSERT INTO outbox(
                        tenant_id, aggregate_type, aggregate_id, event_type,
                        payload, created_at, published_at, publish_error, attempts
                    )
                    VALUES (?, 'document', ?, 'knowledge.index.requested', ?, ?, NULL, '', 0)
                    """,
                    (
                        tenant_id,
                        doc_id,
                        __import__("json").dumps(payload, ensure_ascii=False),
                        now,
                    ),
                )
            conn.commit()

        if use_async:
            return self.get_document(doc_id) or {
                "id": doc_id,
                "filename": safe,
                "title": title,
                "bytes": size,
                "status": "queued",
                "error": "",
                "chunk_count": 0,
                "created_at": now,
                "updated_at": now,
                "object_key": object_key,
                "storage_backend": storage_backend,
                "tenant_id": tenant_id,
                "vector_backend": "dashvector" if self.vector_enabled else "none",
            }

        try:
            if storage_backend == "minio" and object_key and not dest.exists():
                dest.write_bytes(self.object_storage.get_bytes(object_key))
            self._index_document(doc_id, dest, title)
        except Exception as exc:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    ("error", str(exc)[:500], time.time(), doc_id),
                )
                conn.commit()
            raise

        return self.get_document(doc_id) or {
            "id": doc_id,
            "filename": safe,
            "title": title,
            "bytes": size,
            "status": "ready",
            "error": "",
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
            "object_key": object_key,
            "storage_backend": storage_backend,
            "tenant_id": tenant_id,
            "vector_backend": "dashvector" if self.vector_enabled else "none",
        }

    def claim_outbox_batch(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch unpublished outbox rows (oldest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, tenant_id, aggregate_type, aggregate_id, event_type, payload,
                       created_at, attempts
                FROM outbox
                WHERE published_at IS NULL
                ORDER BY id ASC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_outbox_published(self, outbox_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE outbox
                SET published_at = ?, publish_error = '', attempts = attempts + 1
                WHERE id = ?
                """,
                (time.time(), outbox_id),
            )
            conn.commit()

    def mark_outbox_failed(self, outbox_id: int, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE outbox
                SET publish_error = ?, attempts = attempts + 1
                WHERE id = ?
                """,
                (error[:500], outbox_id),
            )
            conn.commit()

    def accept_index_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Fast Kafka ack: validate doc; keep/restore ``queued`` if not running/ready.

        Does not run embedding — mirrors Java "message accepted, work elsewhere".
        """
        doc_id = str(payload.get("doc_id") or "")
        if not doc_id:
            raise ValueError("doc_id required")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, status FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"document not found: {doc_id}")
            if row["status"] in ("ready", "indexing"):
                return {"id": doc_id, "status": row["status"], "accepted": True}
            conn.execute(
                """
                UPDATE documents
                SET status = 'queued', error = '', updated_at = ?
                WHERE id = ?
                """,
                (time.time(), doc_id),
            )
            conn.commit()
        return {"id": doc_id, "status": "queued", "accepted": True}

    def claim_index_batch(
        self,
        *,
        stuck_seconds: int = 1800,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Claim next jobs by status machine (no lease renewal).

        Eligible: ``queued``, ``error``, or ``indexing`` stuck longer than ``stuck_seconds``
        (process crash mid-job). Single-writer gateway is enough; multi-instance can still
        rely on compare-and-set on status + updated_at.
        """
        now = time.time()
        stuck_before = now - max(60, int(stuck_seconds))
        claimed: list[dict[str, Any]] = []
        with self._connect() as conn:
            for _ in range(max(1, limit)):
                row = conn.execute(
                    """
                    SELECT id, filename, stored_name, title, status, object_key, storage_backend,
                           index_attempts, updated_at
                    FROM documents
                    WHERE status IN ('queued', 'error')
                       OR (status = 'indexing' AND updated_at < ?)
                    ORDER BY updated_at ASC
                    LIMIT 1
                    """,
                    (stuck_before,),
                ).fetchone()
                if row is None:
                    break
                cur = conn.execute(
                    """
                    UPDATE documents
                    SET status = 'indexing',
                        error = '',
                        index_attempts = index_attempts + 1,
                        updated_at = ?
                    WHERE id = ?
                      AND (
                        status IN ('queued', 'error')
                        OR (status = 'indexing' AND updated_at < ?)
                      )
                    """,
                    (now, row["id"], stuck_before),
                )
                if cur.rowcount != 1:
                    continue
                claimed.append(dict(row))
            conn.commit()
        return claimed

    def mark_index_error(self, doc_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'error', error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error[:500], time.time(), doc_id),
            )
            conn.commit()

    def index_document_job(
        self,
        payload: dict[str, Any] | None = None,
        *,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Idempotent indexer for the async worker thread (not Kafka poll loop)."""
        doc_id = str(doc_id or (payload or {}).get("doc_id") or "")
        if not doc_id:
            raise ValueError("doc_id required")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, filename, stored_name, title, status, object_key, storage_backend
                FROM documents WHERE id = ?
                """,
                (doc_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"document not found: {doc_id}")
        if row["status"] == "ready":
            return self.get_document(doc_id) or {"id": doc_id, "status": "ready"}

        if row["status"] != "indexing":
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE documents
                    SET status = 'indexing', error = '', updated_at = ?
                    WHERE id = ?
                    """,
                    (time.time(), doc_id),
                )
                conn.commit()

        dest = self.docs_dir / row["stored_name"]
        try:
            if not dest.exists():
                object_key = row["object_key"] or (payload or {}).get("object_key") or ""
                if (
                    row["storage_backend"] == "minio"
                    and object_key
                    and self.object_storage is not None
                ):
                    dest.write_bytes(self.object_storage.get_bytes(str(object_key)))
                elif not dest.exists():
                    raise FileNotFoundError(f"missing source file for {doc_id}")
            self._index_document(doc_id, dest, row["title"])
        except Exception as exc:
            self.mark_index_error(doc_id, str(exc))
            raise
        return self.get_document(doc_id) or {"id": doc_id, "status": "ready"}

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        for item in self.list_documents():
            if item["id"] == doc_id:
                return item
        return None

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT stored_name FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return False
            chunk_rows = conn.execute(
                "SELECT chunk_index FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
            known_chunk_ids = [f"{doc_id}_{int(r['chunk_index'])}" for r in chunk_rows]
            object_key_row = conn.execute(
                "SELECT object_key, storage_backend FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            # Delete remote vectors before local rows so retries can still discover ids.
            if self.vector_index is not None:
                try:
                    self.vector_index.delete_by_doc_id(
                        doc_id, known_chunk_ids=known_chunk_ids
                    )
                except Exception as exc:
                    logger.warning("DashVector delete failed for {}: {}", doc_id, exc)
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM outbox WHERE aggregate_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
        if (
            object_key_row
            and object_key_row["storage_backend"] == "minio"
            and object_key_row["object_key"]
            and self.object_storage is not None
        ):
            try:
                self.object_storage.delete(object_key_row["object_key"])
            except Exception as exc:
                logger.warning("MinIO delete failed for {}: {}", doc_id, exc)
        path = self.docs_dir / row["stored_name"]
        if path.exists():
            path.unlink()
        return True

    def reindex(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, stored_name, title FROM documents").fetchall()
        ok = 0
        errors: list[dict[str, str]] = []
        for row in rows:
            path = self.docs_dir / row["stored_name"]
            try:
                if not path.exists():
                    raise FileNotFoundError(f"missing file: {row['stored_name']}")
                self._index_document(row["id"], path, row["title"])
                ok += 1
            except Exception as exc:
                errors.append({"id": row["id"], "error": str(exc)[:300]})
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                        ("error", str(exc)[:500], time.time(), row["id"]),
                    )
                    conn.commit()
        return {
            "reindexed": ok,
            "errors": errors,
            "total": len(rows),
            "vector_backend": "dashvector" if self.vector_enabled else "none",
        }

    def search(self, query: str, *, k: int = 5) -> list[KnowledgeHit]:
        q = (query or "").strip()
        if not q:
            return []
        k = max(1, min(int(k), 20))
        mode = self.search_mode

        lexical_hits: list[KnowledgeHit] = []
        vector_hits: list[KnowledgeHit] = []

        if mode in ("hybrid", "lexical"):
            lexical_hits = self._search_lexical(q, k=self.lexical_top_k)
        if mode in ("hybrid", "vector") and self.vector_enabled:
            try:
                vector_hits = self._search_vector(q, k=self.vector_top_k)
            except Exception as exc:
                logger.warning("DashVector search failed, lexical only: {}", exc)

        if mode == "lexical" or (mode == "hybrid" and not vector_hits):
            return lexical_hits[:k]
        if mode == "vector" or (mode == "hybrid" and not lexical_hits):
            return vector_hits[:k]
        return _rrf_fuse_many([vector_hits, lexical_hits], k=k)

    def search_fused(
        self,
        queries: Sequence[str],
        *,
        k: int = 5,
        per_query_k: int | None = None,
        weights: Sequence[float] | None = None,
    ) -> list[KnowledgeHit]:
        """Run hybrid search for each distinct query and RRF-fuse the ranked lists.

        Used by enterprise ``kb_search`` to anchor on the user turn while still
        allowing an optional model rewrite as a second recall path. When ``weights``
        is provided (same length as deduped queries), applies weighted RRF.
        """
        k = max(1, min(int(k), 20))
        cleaned: list[str] = []
        cleaned_weights: list[float] = []
        seen: set[str] = set()
        raw_weights = list(weights) if weights is not None else None
        for idx, raw in enumerate(queries):
            q = (raw or "").strip()
            if not q:
                continue
            key = q.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(q)
            if raw_weights is not None and idx < len(raw_weights):
                cleaned_weights.append(float(raw_weights[idx]))
            else:
                cleaned_weights.append(1.0)
        if not cleaned:
            return []
        if len(cleaned) == 1:
            return self.search(cleaned[0], k=k)
        candidate_k = per_query_k or max(k * 2, min(20, max(self.vector_top_k, self.lexical_top_k)))
        lists = [self.search(q, k=candidate_k) for q in cleaned]
        return _rrf_fuse_many(lists, k=k, weights=cleaned_weights)

    def _search_lexical(self, query: str, *, k: int) -> list[KnowledgeHit]:
        # FTS5 trigram ignores tokens shorter than 3 characters — use LIKE then.
        use_like = len(query.strip()) < 3
        with self._connect() as conn:
            rows: list[Any] = []
            if not use_like:
                match = _fts_query(query)
                if match and match != '""':
                    try:
                        rows = conn.execute(
                            """
                            SELECT c.doc_id, d.title, d.filename, c.chunk_index, c.text, c.breadcrumb,
                                   bm25(chunks_fts) AS score
                            FROM chunks_fts
                            JOIN chunks c ON c.id = chunks_fts.rowid
                            JOIN documents d ON d.id = c.doc_id
                            WHERE chunks_fts MATCH ? AND d.status = 'ready'
                            ORDER BY score
                            LIMIT ?
                            """,
                            (match, k),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
            if not rows:
                rows = self._lexical_like_fallback(conn, query, k=k)
        return [
            KnowledgeHit(
                doc_id=row["doc_id"],
                title=row["title"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                snippet=_snippet(row["text"], query),
                score=float(row["score"] or 0.0),
                breadcrumb=row["breadcrumb"] or "",
                source="lexical",
                text=str(row["text"] or ""),
            )
            for row in rows
        ]

    def _lexical_like_fallback(
        self, conn: sqlite3.Connection, query: str, *, k: int
    ) -> list[Any]:
        """Token OR LIKE fallback when FTS AND-match is empty (common for NL questions)."""
        tokens = _lexical_content_tokens(query)
        if not tokens:
            like = f"%{query.strip()}%"
            return list(
                conn.execute(
                    """
                    SELECT c.doc_id, d.title, d.filename, c.chunk_index, c.text, c.breadcrumb,
                           0.0 AS score
                    FROM chunks c
                    JOIN documents d ON d.id = c.doc_id
                    WHERE d.status = 'ready' AND (c.text LIKE ? OR c.breadcrumb LIKE ?)
                    ORDER BY c.chunk_index ASC
                    LIMIT ?
                    """,
                    (like, like, k),
                ).fetchall()
            )
        clauses = " OR ".join(["(c.text LIKE ? OR c.breadcrumb LIKE ?)"] * len(tokens))
        params: list[Any] = []
        for tok in tokens:
            params.extend([f"%{tok}%", f"%{tok}%"])
        params.append(max(k * 5, 20))
        candidates = conn.execute(
            f"""
            SELECT c.doc_id, d.title, d.filename, c.chunk_index, c.text, c.breadcrumb
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE d.status = 'ready' AND ({clauses})
            LIMIT ?
            """,
            params,
        ).fetchall()
        scored: list[tuple[float, Any]] = []
        for row in candidates:
            blob = f"{row['text']} {row['breadcrumb']}".lower()
            hits = sum(1 for tok in tokens if tok.lower() in blob)
            if hits <= 0:
                continue
            scored.append((float(hits) / float(len(tokens)), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, row in scored[:k]:
            out.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "filename": row["filename"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "breadcrumb": row["breadcrumb"],
                    "score": -score,
                }
            )
        return out

    def _search_vector(self, query: str, *, k: int) -> list[KnowledgeHit]:
        assert self.embedder is not None and self.vector_index is not None
        qvec = self.embedder.embed([query])[0]
        raw = self.vector_index.query(qvec, topk=k)
        hits: list[KnowledgeHit] = []
        for item in raw:
            fields = item.fields
            text = str(fields.get("text") or "")
            hits.append(
                KnowledgeHit(
                    doc_id=str(fields.get("doc_id") or ""),
                    title=str(fields.get("title") or ""),
                    filename=str(fields.get("filename") or ""),
                    chunk_index=int(fields.get("chunk_index") or 0),
                    snippet=_snippet(text, query),
                    score=float(item.score),
                    breadcrumb=str(fields.get("breadcrumb") or ""),
                    source="vector",
                    text=text,
                )
            )
        return hits

    def _index_document(self, doc_id: str, path: Path, title: str) -> None:
        text = extract_text(path)
        chunks = chunk_document(
            text,
            title=title,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        now = time.time()

        # Replace local FTS rows
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            chunk_rows: list[tuple[int, str, str, int | None]] = []
            for idx, piece in enumerate(chunks):
                cur = conn.execute(
                    """
                    INSERT INTO chunks(doc_id, chunk_index, text, breadcrumb, page)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, idx, piece.text, piece.breadcrumb, piece.page),
                )
                rowid = int(cur.lastrowid)
                conn.execute(
                    """
                    INSERT INTO chunks_fts(rowid, doc_id, title, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rowid, doc_id, title, piece.text),
                )
                chunk_rows.append((idx, piece.text, piece.breadcrumb, piece.page))
            conn.execute(
                """
                UPDATE documents
                SET status = ?, error = '', updated_at = ?, title = ?
                WHERE id = ?
                """,
                ("ready", now, title, doc_id),
            )
            conn.commit()

        # Dense vectors → DashVector
        if self.vector_enabled and chunks:
            assert self.embedder is not None and self.vector_index is not None
            self.vector_index.delete_by_doc_id(doc_id)
            vectors = self.embedder.embed([c.text for c in chunks])
            filename = path.name
            # Prefer original filename from DB
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT filename FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                if row:
                    filename = row["filename"]
            payload = []
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                chunk_id = f"{doc_id}_{idx}"
                payload.append(
                    (
                        chunk_id,
                        vector,
                        {
                            "doc_id": doc_id,
                            "title": title,
                            "filename": filename,
                            "breadcrumb": chunk.breadcrumb,
                            "chunk_index": idx,
                            "text": chunk.text[:4000],
                            "page": chunk.page if chunk.page is not None else -1,
                        },
                    )
                )
            self.vector_index.upsert_chunks(payload)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    breadcrumb TEXT NOT NULL DEFAULT '',
                    page INTEGER,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE TABLE IF NOT EXISTS kb_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Migrate older schemas missing breadcrumb/page
            cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "breadcrumb" not in cols:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN breadcrumb TEXT NOT NULL DEFAULT ''"
                )
            if "page" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN page INTEGER")

            doc_cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
            if "object_key" not in doc_cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN object_key TEXT NOT NULL DEFAULT ''"
                )
            if "storage_backend" not in doc_cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN storage_backend TEXT NOT NULL DEFAULT 'local'"
                )
            if "tenant_id" not in doc_cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "index_attempts" not in doc_cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN index_attempts INTEGER NOT NULL DEFAULT 0"
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    published_at REAL,
                    publish_error TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_unpublished ON outbox(published_at, id)"
            )

            # FTS5 trigram is required for CJK substring search.
            fts_tok = conn.execute(
                "SELECT value FROM kb_meta WHERE key = 'fts_tokenizer'"
            ).fetchone()
            need_fts = fts_tok is None or fts_tok["value"] != "trigram"
            if need_fts:
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        doc_id UNINDEXED,
                        title,
                        text,
                        tokenize = 'trigram'
                    )
                    """
                )
                # Rebuild FTS from existing chunks if any
                rows = conn.execute(
                    """
                    SELECT c.id, c.doc_id, c.text, d.title
                    FROM chunks c
                    JOIN documents d ON d.id = c.doc_id
                    """
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO chunks_fts(rowid, doc_id, title, text)
                        VALUES (?, ?, ?, ?)
                        """,
                        (row["id"], row["doc_id"], row["title"], row["text"]),
                    )
                conn.execute(
                    """
                    INSERT INTO kb_meta(key, value) VALUES('fts_tokenizer', 'trigram')
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """
                )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _rrf_fuse_many(
    hit_lists: Sequence[list[KnowledgeHit]],
    *,
    k: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[KnowledgeHit]:
    """Reciprocal-rank fusion across one or more ranked hit lists.

    Optional per-list ``weights`` implement Onyx-style weighted RRF (anchor boost).
    """
    scores: dict[str, float] = {}
    keep: dict[str, KnowledgeHit] = {}
    sources: dict[str, set[str]] = {}

    def key(hit: KnowledgeHit) -> str:
        return f"{hit.doc_id}:{hit.chunk_index}"

    for list_i, hits in enumerate(hit_lists):
        weight = 1.0
        if weights is not None and list_i < len(weights):
            weight = max(0.0, float(weights[list_i]))
        for rank, hit in enumerate(hits):
            kid = key(hit)
            scores[kid] = scores.get(kid, 0.0) + weight / (rrf_k + rank + 1)
            keep.setdefault(kid, hit)
            sources.setdefault(kid, set()).add(hit.source or "unknown")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out: list[KnowledgeHit] = []
    for kid, score in ranked:
        hit = keep[kid]
        src_set = sources.get(kid) or {hit.source or "hybrid"}
        if len(src_set) > 1 or (src_set & {"vector", "lexical"} and len(hit_lists) > 1):
            source = "hybrid"
        else:
            source = next(iter(src_set)) if len(src_set) == 1 else "hybrid"
        out.append(
            KnowledgeHit(
                doc_id=hit.doc_id,
                title=hit.title,
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                snippet=hit.snippet,
                score=score,
                breadcrumb=hit.breadcrumb,
                source=source,
                text=hit.text or hit.snippet,
            )
        )
    return out


def _rrf_fuse(
    vector_hits: list[KnowledgeHit],
    lexical_hits: list[KnowledgeHit],
    *,
    k: int,
    rrf_k: int = 60,
) -> list[KnowledgeHit]:
    """Backward-compatible wrapper around :func:`_rrf_fuse_many`."""
    return _rrf_fuse_many([vector_hits, lexical_hits], k=k, rrf_k=rrf_k)


def _safe_filename(name: str) -> str:
    base = Path(name or "upload.bin").name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "upload.bin"
    return cleaned[:180]


_FTS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "will",
        "shall",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "this",
        "that",
        "these",
        "those",
        "my",
        "our",
        "your",
        "their",
        "please",
        "tell",
        "me",
        "about",
    }
)


def _lexical_content_tokens(query: str) -> list[str]:
    """Content tokens for FTS/LIKE — drops punctuation and English stopwords."""
    raw = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", query or "")
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        tl = token.lower()
        if len(tl) < 3 and not re.search(r"[\u4e00-\u9fff]", token):
            continue
        if tl in _FTS_STOPWORDS:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(token)
    return out


def _fts_query(query: str) -> str:
    """Build an FTS5 query.

    Uses trigram tokenizer, so quoted phrases work for CJK substring match.
    Multi-word English queries are AND-combined on content tokens only
    (stopwords / punctuation stripped) so natural questions still match.
    """
    tokens = _lexical_content_tokens(query)
    if not tokens:
        return '""'
    parts: list[str] = []
    for token in tokens:
        escaped = _FTS_TOKEN.sub('""', token)
        # Quote so trigram matches the substring / phrase as-is.
        parts.append(f'"{escaped}"')
    return " AND ".join(parts) if len(parts) > 1 else parts[0]


def _snippet(text: str, query: str, *, radius: int = 180) -> str:
    # Strip breadcrumb prefix for display
    display = text
    if display.startswith("[") and "]\n" in display[:200]:
        display = display.split("]\n", 1)[1]
    lower = display.lower()
    needle = (query.split() or [query])[0].lower()
    idx = lower.find(needle) if needle else -1
    if idx < 0:
        return display[: radius * 2].strip()
    start = max(0, idx - radius)
    end = min(len(display), idx + len(needle) + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(display) else ""
    return f"{prefix}{display[start:end].strip()}{suffix}"
