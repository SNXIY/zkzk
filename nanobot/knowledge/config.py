"""Knowledge base configuration (DashScope + DashVector + MinIO/Kafka async)."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field

from nanobot.config_base import Base


class KnowledgeObjectStorageConfig(Base):
    """Object storage for original documents (local disk or MinIO)."""

    backend: Literal["local", "minio"] = "local"
    endpoint: str = Field(
        default="127.0.0.1:9000",
        validation_alias=AliasChoices("endpoint", "MINIO_ENDPOINT"),
    )
    access_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("access_key", "accessKey", "MINIO_ACCESS_KEY"),
    )
    secret_key: str = Field(
        default="minioadmin",
        repr=False,
        validation_alias=AliasChoices("secret_key", "secretKey", "MINIO_SECRET_KEY"),
    )
    bucket: str = Field(
        default="nanobot-knowledge",
        validation_alias=AliasChoices("bucket", "MINIO_BUCKET"),
    )
    secure: bool = False
    region: str = "us-east-1"
    prefix: str = "docs/"
    # Browser direct upload (presigned PUT / multipart).
    part_size_mb: int = Field(
        default=8,
        ge=5,
        le=64,
        validation_alias=AliasChoices("part_size_mb", "partSizeMb"),
    )
    presign_expires_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        validation_alias=AliasChoices(
            "presign_expires_seconds",
            "presignExpiresSeconds",
        ),
    )


class KnowledgeKafkaConfig(Base):
    """Kafka delivery for outbox-relayed index jobs."""

    enabled: bool = False
    bootstrap_servers: str = Field(
        default="127.0.0.1:29092",
        validation_alias=AliasChoices(
            "bootstrap_servers",
            "bootstrapServers",
            "KAFKA_BOOTSTRAP_SERVERS",
        ),
    )
    topic_index: str = "nanobot.knowledge.index.requested"
    topic_dlq: str = "nanobot.knowledge.index.dlq"
    group_id: str = "nanobot-knowledge-indexer"
    client_id: str = "nanobot-knowledge"


class KnowledgeRetrievalPolicy(Base):
    """Enterprise agentic-RAG retrieval contract (gate + anchored search).

    Separates *whether* to touch the KB (router gate) from *how* to search
    (always anchor on the user turn; optional tool rewrite is fused, never sole).
    """

    gate_mode: Literal["off", "soft", "strict"] = Field(
        default="strict",
        validation_alias=AliasChoices("gate_mode", "gateMode"),
        description=(
            "strict=block kb_search when need_kb=false and require KB turns to retrieve; "
            "soft=prompt-only guidance; off=disable gate section"
        ),
    )
    anchor_user_query: bool = Field(
        default=True,
        validation_alias=AliasChoices("anchor_user_query", "anchorUserQuery"),
        description="Always run retrieval with the raw user turn text",
    )
    fuse_tool_query: bool = Field(
        default=True,
        validation_alias=AliasChoices("fuse_tool_query", "fuseToolQuery"),
        description="Also search the model-supplied query and RRF-fuse with the anchor",
    )
    min_rrf_score: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("min_rrf_score", "minRrfScore"),
        description="Drop fused hits below this absolute RRF score (0 disables)",
    )
    low_confidence_relative: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("low_confidence_relative", "lowConfidenceRelative"),
        description="Hits scoring below top_score * this are marked low-confidence",
    )
    max_search_rounds_hint: int = Field(
        default=3,
        ge=1,
        le=8,
        validation_alias=AliasChoices("max_search_rounds_hint", "maxSearchRoundsHint"),
    )
    require_citations: bool = Field(
        default=True,
        validation_alias=AliasChoices("require_citations", "requireCitations"),
    )
    kb_categories: list[str] = Field(
        default_factory=lambda: [
            "hr",
            "security",
            "legal",
            "finance",
            "engineering",
            "product",
            "customer_support",
            "compliance",
            "policy",
        ],
        validation_alias=AliasChoices("kb_categories", "kbCategories"),
        description="Skill categories that imply need_kb when routed",
    )
    kb_query_hints: list[str] = Field(
        default_factory=lambda: [
            "policy",
            "handbook",
            "sla",
            "sop",
            "leave",
            "pto",
            "expense",
            "nda",
            "classification",
            "escalation",
            "on-call",
            "oncall",
            "合规",
            "制度",
            "手册",
            "政策",
            "请假",
            "报销",
            "升级",
        ],
        validation_alias=AliasChoices("kb_query_hints", "kbQueryHints"),
        description="Substring hints that mark a turn as KB-oriented when no skill hits",
    )

    # --- Onyx-style multi-query + grade + corrective (system-side, not model whim) ---
    multi_query: bool = Field(
        default=True,
        validation_alias=AliasChoices("multi_query", "multiQuery"),
        description="Emit system query variants (anchor/keyword/tool) and weighted-RRF fuse",
    )
    keyword_variant: bool = Field(
        default=True,
        validation_alias=AliasChoices("keyword_variant", "keywordVariant"),
        description="Add a stopword-stripped keyword-emphasized variant (Onyx-style)",
    )
    anchor_weight: float = Field(
        default=1.25,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("anchor_weight", "anchorWeight"),
    )
    keyword_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("keyword_weight", "keywordWeight"),
    )
    tool_query_weight: float = Field(
        default=0.85,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("tool_query_weight", "toolQueryWeight"),
    )
    corrective_weight: float = Field(
        default=1.1,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("corrective_weight", "correctiveWeight"),
    )
    grade_min_coverage: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("grade_min_coverage", "gradeMinCoverage"),
        description="Min fraction of content tokens that must appear in top hits to pass grade",
    )
    grade_top_m: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias=AliasChoices("grade_top_m", "gradeTopM"),
    )
    grade_fail_closed: bool = Field(
        default=True,
        validation_alias=AliasChoices("grade_fail_closed", "gradeFailClosed"),
        description="If final grade fails, withhold hits from answer use (fail-closed)",
    )
    corrective_rounds: int = Field(
        default=2,
        ge=1,
        le=4,
        validation_alias=AliasChoices("corrective_rounds", "correctiveRounds"),
        description="Total retrieval rounds including the first (grade-fail triggers next)",
    )


class KnowledgeConfig(Base):
    """Enterprise knowledge base: structure-aware chunking + DashScope + DashVector.

    Optional MinIO object storage + transactional outbox + Kafka async indexing.
    """

    enable: bool = True
    max_upload_mb: int = Field(default=32, ge=1, le=512)
    chunk_size: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    retrieval: KnowledgeRetrievalPolicy = Field(
        default_factory=KnowledgeRetrievalPolicy,
        validation_alias=AliasChoices("retrieval", "retrievalPolicy"),
    )

    # --- Async / multi-tenant-ready pipeline ---
    tenant_id: str = Field(
        default="default",
        validation_alias=AliasChoices("tenant_id", "tenantId"),
    )
    async_index: bool = Field(
        default=False,
        validation_alias=AliasChoices("async_index", "asyncIndex"),
    )
    async_threshold_mb: int = Field(
        default=5,
        ge=0,
        le=512,
        validation_alias=AliasChoices("async_threshold_mb", "asyncThresholdMb"),
    )
    outbox_poll_seconds: float = Field(default=1.0, ge=0.2, le=30.0)
    indexer_poll_seconds: float = Field(
        default=1.0,
        ge=0.2,
        le=30.0,
        validation_alias=AliasChoices("indexer_poll_seconds", "indexerPollSeconds"),
    )
    # Reclaim orphaned ``indexing`` rows (e.g. process killed mid-job). No lease renewal.
    index_stuck_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        validation_alias=AliasChoices("index_stuck_seconds", "indexStuckSeconds"),
    )
    worker_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("worker_enabled", "workerEnabled"),
    )
    storage: KnowledgeObjectStorageConfig = Field(default_factory=KnowledgeObjectStorageConfig)
    kafka: KnowledgeKafkaConfig = Field(default_factory=KnowledgeKafkaConfig)

    # --- DashScope embedding (OpenAI-compatible) ---
    embeddings: bool = True
    embedding_model: str = Field(
        default="text-embedding-v4",
        validation_alias=AliasChoices(
            "embedding_model",
            "embeddingModel",
            "GREENBOOK_EMBEDDING_MODEL",
        ),
    )
    embedding_dimensions: int = Field(
        default=1024,
        ge=32,
        le=4096,
        validation_alias=AliasChoices(
            "embedding_dimensions", "embeddingDimensions", "GREENBOOK_EMBEDDING_DIMENSIONS"
        ),
    )
    embedding_api_base: str | None = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices(
            "embedding_api_base",
            "embeddingApiBase",
            "GREENBOOK_EMBEDDING_BASE_URL",
        ),
    )
    embedding_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "embedding_api_key",
            "embeddingApiKey",
            "GREENBOOK_EMBEDDING_API_KEY",
        ),
    )
    embedding_provider: str | None = "dashscope"

    # --- DashVector ---
    dashvector_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "dashvector_api_key",
            "dashvectorApiKey",
            "GREENBOOK_DASHVECTOR_API_KEY",
            "DASHVECTOR_API_KEY",
        ),
    )
    dashvector_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "dashvector_endpoint",
            "dashvectorEndpoint",
            "GREENBOOK_DASHVECTOR_ENDPOINT",
            "DASHVECTOR_ENDPOINT",
        ),
    )
    dashvector_collection: str = Field(
        default="nanobot_enterprise_docs",
        validation_alias=AliasChoices(
            "dashvector_collection",
            "dashvectorCollection",
            "GREENBOOK_DASHVECTOR_COLLECTION",
            "GREENBOOK_CHROMA_COLLECTION",
        ),
    )
    dashvector_metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    dashvector_auto_create: bool = True

    search_mode: Literal["hybrid", "vector", "lexical"] = "hybrid"
    vector_top_k: int = Field(default=20, ge=1, le=50)
    lexical_top_k: int = Field(default=20, ge=1, le=50)
