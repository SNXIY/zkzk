"""Smoke: MinIO put/get + Kafka produce/consume + outbox enqueue."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from nanobot.knowledge.config import (
    KnowledgeConfig,
    KnowledgeKafkaConfig,
    KnowledgeObjectStorageConfig,
)
from nanobot.knowledge.kafka_bus import KnowledgeKafkaBus
from nanobot.knowledge.minio_storage import MinioObjectStorage
from nanobot.knowledge.store import KnowledgeStore


def main() -> None:
    storage = MinioObjectStorage(
        KnowledgeObjectStorageConfig(
            backend="minio",
            endpoint="127.0.0.1:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="nanobot-knowledge",
            secure=False,
        )
    )
    key = storage.object_key(tenant_id="default", doc_id="smoke", filename="smoke.txt")
    payload = b"nanobot middleware smoke " + str(time.time()).encode()
    storage.put_bytes(key, payload)
    got = storage.get_bytes(key)
    assert got == payload, "minio get mismatch"
    print("OK minio put/get", key, len(got))

    async def kafka_roundtrip() -> None:
        # Use a dedicated smoke topic so gateway workers never see these messages.
        cfg = KnowledgeKafkaConfig(
            enabled=True,
            bootstrap_servers="127.0.0.1:29092",
            topic_index="nanobot.knowledge.smoke",
            group_id=f"nanobot-smoke-{int(time.time())}",
            client_id="nanobot-smoke",
        )
        bus = KnowledgeKafkaBus(cfg)
        msg = {
            "event_type": "knowledge.index.requested",
            "doc_id": "smoke",
            "ts": time.time(),
        }
        await bus.publish_index_requested(msg)
        print("OK kafka publish")

        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            cfg.topic_index,
            bootstrap_servers=cfg.bootstrap_servers,
            group_id=cfg.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        await consumer.start()
        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                batch = await consumer.getmany(timeout_ms=2000, max_records=10)
                for _tp, records in batch.items():
                    for rec in records:
                        body = json.loads(rec.value.decode("utf-8"))
                        if body.get("doc_id") == "smoke":
                            print("OK kafka consume", body.get("event_type"))
                            return
            raise TimeoutError("kafka consume timeout")
        finally:
            await consumer.stop()
            await bus.close()

    asyncio.run(kafka_roundtrip())

    ws = Path(tempfile.mkdtemp(prefix="nb-kb-smoke-"))
    cfg = KnowledgeConfig(
        enable=True,
        async_index=True,
        async_threshold_mb=0,
        embeddings=False,
        storage=KnowledgeObjectStorageConfig(
            backend="minio",
            endpoint="127.0.0.1:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="nanobot-knowledge",
            secure=False,
        ),
        kafka=KnowledgeKafkaConfig(enabled=True, bootstrap_servers="127.0.0.1:29092"),
    )
    store = KnowledgeStore(
        ws,
        knowledge_cfg=cfg,
        object_storage=storage,
        embedder=None,
        vector_index=None,
    )
    doc = store.add_document("async-smoke.md", b"# smoke\n\nhello async outbox\n" * 20)
    assert doc["status"] == "queued", doc
    rows = store.claim_outbox_batch()
    assert rows, "expected outbox rows"
    print(
        "OK outbox queued",
        doc["id"],
        "outbox=",
        len(rows),
        "object_key=",
        doc.get("object_key"),
    )
    print("SMOKE PASS", ws)


if __name__ == "__main__":
    main()
