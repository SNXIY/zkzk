"""Async knowledge indexing aligned with internal file-dispatch style.

Java reference (internal):
  URL stream → MinIO staging → Salt push; Kafka carries async result callbacks;
  @Async / CompletableFuture updates task status — work is NOT held inside Kafka poll.

nanobot mapping:
  upload → MinIO + documents(status) + outbox
  outbox → Kafka ``index.requested`` (notify)
  Kafka consumer → fast accept/ack only
  thread-pool indexer → chunk/embed/DashVector, update status queued→indexing→ready|error
  (optional later) Kafka ``index.completed`` callback like Salt result topic

Crash recovery: reclaim ``indexing`` rows older than ``index_stuck_seconds``.
No lease heartbeat / renewal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.knowledge.config import KnowledgeConfig
from nanobot.knowledge.kafka_bus import KnowledgeKafkaBus
from nanobot.webui.knowledge_api import knowledge_config_from_tools, open_knowledge_store


class KnowledgeAsyncPipeline:
    """Outbox publisher + Kafka ack consumer + status-driven indexer worker."""

    def __init__(self, workspace: Path, tools_cfg: Any) -> None:
        self.workspace = Path(workspace)
        self.tools_cfg = tools_cfg
        self.cfg: KnowledgeConfig = knowledge_config_from_tools(tools_cfg)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enable and self.cfg.async_index)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if not self.enabled:
            logger.info("Knowledge async pipeline disabled (async_index=false)")
            return
        loop = loop or asyncio.get_running_loop()
        self._stop.clear()
        self._tasks.append(loop.create_task(self._relay_loop(), name="kb-outbox-relay"))
        if self.cfg.worker_enabled:
            self._tasks.append(
                loop.create_task(self._indexer_loop(), name="kb-index-worker")
            )
            if self.cfg.kafka.enabled:
                self._tasks.append(
                    loop.create_task(self._kafka_ack_loop(), name="kb-kafka-ack")
                )
            else:
                logger.info(
                    "Knowledge Kafka disabled — DB status worker drains queued docs"
                )
        logger.info(
            "Knowledge async pipeline started tenant={} storage={} kafka={}",
            self.cfg.tenant_id,
            self.cfg.storage.backend,
            self.cfg.kafka.enabled,
        )

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _relay_loop(self) -> None:
        """Publish outbox → Kafka notify. Never runs embedding here."""
        bus = KnowledgeKafkaBus(self.cfg.kafka) if self.cfg.kafka.enabled else None
        try:
            while not self._stop.is_set():
                try:
                    store = open_knowledge_store(self.workspace, self.tools_cfg)
                    rows = store.claim_outbox_batch(limit=20)
                    for row in rows:
                        payload = json.loads(row["payload"])
                        try:
                            if bus is not None and bus.enabled:
                                await bus.publish_index_requested(payload)
                            store.mark_outbox_published(int(row["id"]))
                            logger.info(
                                "Outbox published doc_id={} outbox_id={} kafka={}",
                                payload.get("doc_id"),
                                row["id"],
                                bool(bus and bus.enabled),
                            )
                        except Exception as exc:
                            store.mark_outbox_failed(int(row["id"]), str(exc))
                            logger.warning(
                                "Outbox publish failed id={}: {}", row["id"], exc
                            )
                except Exception as exc:
                    logger.warning("Outbox relay loop error: {}", exc)
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=float(self.cfg.outbox_poll_seconds),
                    )
                except TimeoutError:
                    continue
        finally:
            if bus is not None:
                await bus.close()

    async def _kafka_ack_loop(self) -> None:
        """Kafka notify bus: accept job, commit offset immediately (like dispatch accepted)."""
        bus = KnowledgeKafkaBus(self.cfg.kafka)

        async def _handle(payload: dict[str, Any]) -> None:
            def _accept() -> dict[str, Any]:
                store = open_knowledge_store(self.workspace, self.tools_cfg)
                return store.accept_index_request(payload)

            try:
                result = await asyncio.to_thread(_accept)
            except LookupError as exc:
                logger.warning("Kafka ack skip (missing doc): {}", exc)
                return
            logger.info(
                "Kafka ack accepted doc_id={} status={}",
                payload.get("doc_id"),
                result.get("status"),
            )

        try:
            await bus.ensure_topics()
            await bus.consume_index_loop(_handle)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Kafka ack consumer crashed: {}", exc)
        finally:
            await bus.close()

    async def _indexer_loop(self) -> None:
        """Status-driven worker ≈ Java @Async pool: claim task → run → update status."""
        stuck_s = int(self.cfg.index_stuck_seconds)
        poll_s = float(self.cfg.indexer_poll_seconds)
        while not self._stop.is_set():
            try:
                store = open_knowledge_store(self.workspace, self.tools_cfg)
                claimed = store.claim_index_batch(stuck_seconds=stuck_s, limit=1)
                if not claimed:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=poll_s)
                    except TimeoutError:
                        continue
                    continue

                row = claimed[0]
                doc_id = str(row["id"])
                logger.info(
                    "Index worker claimed doc_id={} attempts={}",
                    doc_id,
                    row.get("index_attempts"),
                )
                try:
                    doc = await asyncio.to_thread(
                        store.index_document_job,
                        {
                            "doc_id": doc_id,
                            "object_key": row.get("object_key") or "",
                            "storage_backend": row.get("storage_backend") or "local",
                        },
                        doc_id=doc_id,
                    )
                    logger.info(
                        "Index worker finished doc_id={} status={}",
                        doc_id,
                        (doc or {}).get("status"),
                    )
                except Exception as exc:
                    logger.warning("Index worker failed doc_id={}: {}", doc_id, exc)
                    await asyncio.to_thread(store.mark_index_error, doc_id, str(exc))
            except Exception as exc:
                logger.warning("Index worker loop error: {}", exc)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll_s)
                except TimeoutError:
                    continue
