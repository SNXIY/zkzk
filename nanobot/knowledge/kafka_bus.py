"""Kafka producer/consumer helpers for knowledge index *notifications*."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from nanobot.knowledge.config import KnowledgeKafkaConfig


class KnowledgeKafkaBus:
    def __init__(self, cfg: KnowledgeKafkaConfig) -> None:
        self.cfg = cfg
        self._producer = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def _ensure_aiokafka(self):
        try:
            import aiokafka  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "aiokafka is required for Kafka delivery. "
                "Install with: pip install 'nanobot[knowledge]'"
            ) from exc

    async def publish_index_requested(self, payload: dict[str, Any]) -> None:
        self._ensure_aiokafka()
        from aiokafka import AIOKafkaProducer

        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.cfg.bootstrap_servers,
                client_id=f"{self.cfg.client_id}-producer",
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
            )
            await self._producer.start()
        key = str(payload.get("doc_id") or payload.get("aggregate_id") or "")
        await self._producer.send_and_wait(self.cfg.topic_index, payload, key=key or None)

    async def publish_dlq(self, payload: dict[str, Any], *, error: str) -> None:
        self._ensure_aiokafka()
        from aiokafka import AIOKafkaProducer

        body = dict(payload)
        body["error"] = error[:1000]
        producer = AIOKafkaProducer(
            bootstrap_servers=self.cfg.bootstrap_servers,
            client_id=f"{self.cfg.client_id}-dlq",
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )
        await producer.start()
        try:
            await producer.send_and_wait(self.cfg.topic_dlq, body)
        finally:
            await producer.stop()

    async def close(self) -> None:
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception as exc:
                logger.warning("Kafka producer stop failed: {}", exc)
            self._producer = None

    async def ensure_topics(self) -> None:
        """Create index + DLQ topics if missing (idempotent)."""
        self._ensure_aiokafka()
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic

        admin = AIOKafkaAdminClient(
            bootstrap_servers=self.cfg.bootstrap_servers,
            client_id=f"{self.cfg.client_id}-admin",
        )
        await admin.start()
        try:
            topics = [
                NewTopic(self.cfg.topic_index, num_partitions=1, replication_factor=1),
                NewTopic(self.cfg.topic_dlq, num_partitions=1, replication_factor=1),
            ]
            try:
                await admin.create_topics(topics)
                logger.info(
                    "Kafka topics ready: {}, {}",
                    self.cfg.topic_index,
                    self.cfg.topic_dlq,
                )
            except Exception as exc:
                logger.debug("Kafka ensure_topics: {}", exc)
        finally:
            await admin.close()

    async def consume_index_loop(self, handler) -> None:
        """Consume notifications. Handler must stay fast (ack/enqueue only)."""
        self._ensure_aiokafka()
        from aiokafka import AIOKafkaConsumer

        # Handlers are expected to finish in seconds (DB accept), not minutes.
        consumer = AIOKafkaConsumer(
            self.cfg.topic_index,
            bootstrap_servers=self.cfg.bootstrap_servers,
            group_id=self.cfg.group_id,
            client_id=f"{self.cfg.client_id}-consumer",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_records=10,
            max_poll_interval_ms=300_000,  # 5m safety; work itself is not in this loop
            session_timeout_ms=45_000,
            heartbeat_interval_ms=3_000,
        )
        await consumer.start()
        logger.info(
            "Knowledge Kafka ack consumer started topic={} group={}",
            self.cfg.topic_index,
            self.cfg.group_id,
        )
        try:
            async for msg in consumer:
                try:
                    payload = json.loads(msg.value.decode("utf-8"))
                except Exception as exc:
                    logger.warning("Knowledge Kafka bad message, sending to DLQ: {}", exc)
                    raw = {"raw": msg.value.decode("utf-8", errors="replace")}
                    try:
                        await self.publish_dlq(raw, error=str(exc))
                        await consumer.commit()
                    except Exception:
                        logger.exception("DLQ publish failed")
                    continue

                try:
                    await handler(payload if isinstance(payload, dict) else {"payload": payload})
                except Exception as exc:
                    logger.warning("Knowledge Kafka ack handler failed: {}", exc)
                    body = payload if isinstance(payload, dict) else {"payload": payload}
                    try:
                        await self.publish_dlq(body, error=str(exc))
                    except Exception:
                        logger.exception("DLQ publish failed")
                    try:
                        await consumer.commit()
                    except Exception as commit_exc:
                        logger.warning("Knowledge Kafka commit after failure: {}", commit_exc)
                    continue

                try:
                    await consumer.commit()
                except Exception as commit_exc:
                    logger.warning(
                        "Knowledge Kafka commit failed (may redeliver; accept is idempotent): {}",
                        commit_exc,
                    )
        finally:
            await consumer.stop()
