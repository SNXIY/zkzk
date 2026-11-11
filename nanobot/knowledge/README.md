# Enterprise Knowledge (DashScope + DashVector)

杭州/阿里云企业向知识库：

| 层 | 组件 |
|---|---|
| 切块 | 结构感知（Markdown 标题 / PDF 页 / 段落） |
| Embedding | DashScope `text-embedding-v4`（compatible-mode） |
| 向量库 | 阿里云 **DashVector** |
| 关键词 | 本地 SQLite FTS5 |
| 融合 | Reciprocal Rank Fusion（hybrid） |

## 配置（环境变量）

与 Greenbook 风格对齐，也可用 `NANOBOT_*`：

```bash
# DashScope embedding
GREENBOOK_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
GREENBOOK_EMBEDDING_API_KEY=sk-xxx
GREENBOOK_EMBEDDING_MODEL=text-embedding-v4
GREENBOOK_EMBEDDING_DIMENSIONS=1024

# DashVector
DASHVECTOR_API_KEY=xxx
DASHVECTOR_ENDPOINT=vrs-cn-xxxx.dashvector.cn-hangzhou.aliyuncs.com
NANOBOT_DASHVECTOR_COLLECTION=nanobot_enterprise_docs
```

或在 nanobot config 的 `tools.knowledge` 下填写同名字段。

## 降级

- 未配 Embedding Key → 仅 FTS
- 未配 DashVector → 仅 FTS
- 两者都配齐 → hybrid（向量 + FTS）

## 大文件异步索引（MinIO + Outbox + Kafka + DB Lease）

```bash
docker compose -f docker-compose.middleware.yml up -d
```

详见 [`docker/middleware/README.md`](../../docker/middleware/README.md)。

企业推荐链路（对齐内部文件分发）：**MinIO 中转 + 任务状态机 + Kafka 通知 + 线程池做重活**（不做租约续约；Kafka 不堵长任务）。

在 `tools.knowledge` 中开启：

- `asyncIndex: true`
- `storage.backend: "minio"`
- `kafka.enabled: true`
- `indexStuckSeconds` / `indexerPollSeconds`（可选）

≥ `asyncThresholdMb` 的上传 → MinIO + `queued` + outbox；relay 推 Kafka；consumer 快速 ack；worker 切块/embedding/入库 → `ready`。

## 依赖

```bash
uv pip install -e ".[knowledge]"
# 或: pip install dashvector aiokafka
```
