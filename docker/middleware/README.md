# Knowledge middleware (MinIO + Kafka)

Enterprise async indexing for nanobot knowledge uploads.

## Design reference (internal Java file-dispatch)

Internal pattern observed:

> 外部 URL 下载 → MinIO 中转 → SaltStack 推送；HttpClient 流式 + MinIO 分片避免大文件 OOM；  
> Kafka 异步解耦 Salt 执行结果回调；`@Async` / `CompletableFuture` 更新任务状态并回调上游。

Mapped to nanobot knowledge:

| Java dispatch | nanobot knowledge |
|---|---|
| URL stream download | Browser **presigned PUT / multipart** direct to MinIO |
| MinIO staging | MinIO originals (`storage.backend=minio`) |
| Salt push (heavy work) | chunk + embed + DashVector index (thread pool) |
| Kafka result callback | Kafka `index.requested` notify (+ optional future `completed` topic) |
| Task status + `@Async` | SQLite `queued → indexing → ready\|error` + indexer loop |

**Do not** run long indexing inside Kafka `poll` (causes rebalance).  
**Do not** use lease heartbeat/renewal for single-gateway — use status + stuck timeout reclaim.

### Flow

1. Upload txn: MinIO object + `documents(status=queued)` + outbox  
2. Outbox relay → Kafka notify  
3. Kafka consumer: **ack only** (accept/re-queue), commit immediately  
4. Index worker (`asyncio.to_thread`): claim `queued` / stuck `indexing` → index → `ready`/`error`  
5. Crash: `indexing` older than `indexStuckSeconds` can be reclaimed  

MinIO originals stay after index (reindex/audit); deleted only when the document is deleted.

## Start

```bash
docker compose -f docker-compose.middleware.yml up -d
```

Images: `coollabsio/minio` (official Hub images removed) + `apache/kafka`.

## Endpoints

| Service | Address |
|---|---|
| MinIO S3 | `http://127.0.0.1:9000` (console `:9001`, `minioadmin`/`minioadmin`) |
| Kafka | `127.0.0.1:29092` |
| Bucket | `nanobot-knowledge` |

## Config (`tools.knowledge`)

```json
{
  "asyncIndex": true,
  "asyncThresholdMb": 5,
  "indexStuckSeconds": 1800,
  "indexerPollSeconds": 1,
  "storage": {
    "backend": "minio",
    "endpoint": "127.0.0.1:9000",
    "accessKey": "minioadmin",
    "secretKey": "minioadmin",
    "bucket": "nanobot-knowledge",
    "secure": false
  },
  "kafka": {
    "enabled": true,
    "bootstrapServers": "127.0.0.1:29092",
    "topicIndex": "nanobot.knowledge.index.requested",
    "topicDlq": "nanobot.knowledge.index.dlq",
    "groupId": "nanobot-knowledge-indexer"
  }
}
```

```bash
pip install "nanobot[knowledge]"
```

Restart `nanobot gateway` after config changes.
