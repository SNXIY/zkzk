"""MinIO / S3-compatible object storage for knowledge originals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from nanobot.knowledge.config import KnowledgeObjectStorageConfig

# S3 multipart requires parts (except the last) to be at least 5 MiB.
_MIN_PART_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class StoredObject:
    key: str
    bucket: str
    etag: str = ""
    size: int = 0


class MinioObjectStorage:
    """Thin boto3 wrapper (MinIO is S3-compatible)."""

    def __init__(self, cfg: KnowledgeObjectStorageConfig) -> None:
        import boto3
        from botocore.client import Config

        self.cfg = cfg
        endpoint = cfg.endpoint.strip()
        if endpoint and not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = f"{'https' if cfg.secure else 'http'}://{endpoint}"
        self.endpoint = endpoint.rstrip("/")
        self.bucket = cfg.bucket
        self.prefix = (cfg.prefix or "").lstrip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            region_name=cfg.region or "us-east-1",
            config=Config(signature_version="s3v4"),
        )
        self.ensure_bucket()
        self.ensure_cors()

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self.bucket)
                logger.info("Created MinIO bucket {}", self.bucket)
            except Exception as exc:
                logger.warning("MinIO ensure_bucket failed: {}", exc)

    def ensure_cors(self) -> None:
        """Allow browser PUT/POST from the WebUI origin (presigned direct upload)."""
        try:
            self._client.put_bucket_cors(
                Bucket=self.bucket,
                CORSConfiguration={
                    "CORSRules": [
                        {
                            "AllowedOrigins": ["*"],
                            "AllowedMethods": ["GET", "PUT", "POST", "HEAD", "DELETE"],
                            "AllowedHeaders": ["*"],
                            "ExposeHeaders": ["ETag", "etag", "x-amz-request-id"],
                            "MaxAgeSeconds": 3600,
                        }
                    ]
                },
            )
        except Exception as exc:
            logger.debug("MinIO CORS setup skipped/failed: {}", exc)

    def object_key(self, *, tenant_id: str, doc_id: str, filename: str) -> str:
        safe = filename.replace("\\", "/").split("/")[-1]
        base = f"{tenant_id}/{doc_id}_{safe}"
        return f"{self.prefix}{base}" if self.prefix else base

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> StoredObject:
        resp = self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return StoredObject(
            key=key,
            bucket=self.bucket,
            etag=str(resp.get("ETag") or "").strip('"'),
            size=len(data),
        )

    def get_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        body = resp["Body"].read()
        return bytes(body)

    def head_size(self, key: str) -> int:
        resp = self._client.head_object(Bucket=self.bucket, Key=key)
        return int(resp.get("ContentLength") or 0)

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            logger.warning("MinIO delete failed key={}: {}", key, exc)

    def presign_put(
        self,
        key: str,
        *,
        expires_seconds: int = 3600,
        content_type: str | None = None,
    ) -> str:
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return self._client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expires_seconds,
        )

    def create_multipart_upload(
        self,
        key: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        resp = self._client.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            ContentType=content_type,
        )
        return str(resp["UploadId"])

    def presign_upload_part(
        self,
        key: str,
        *,
        upload_id: str,
        part_number: int,
        expires_seconds: int = 3600,
    ) -> str:
        return self._client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": int(part_number),
            },
            ExpiresIn=expires_seconds,
        )

    def complete_multipart_upload(
        self,
        key: str,
        *,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> StoredObject:
        normalized = []
        for part in parts:
            num = int(part.get("part_number") or part.get("PartNumber") or 0)
            etag = str(part.get("etag") or part.get("ETag") or "").strip().strip('"')
            if num < 1 or not etag:
                raise ValueError(f"invalid multipart part: {part}")
            normalized.append({"PartNumber": num, "ETag": etag})
        normalized.sort(key=lambda p: p["PartNumber"])
        resp = self._client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": normalized},
        )
        size = 0
        try:
            size = self.head_size(key)
        except Exception:
            pass
        return StoredObject(
            key=key,
            bucket=self.bucket,
            etag=str(resp.get("ETag") or "").strip('"'),
            size=size,
        )

    def abort_multipart_upload(self, key: str, *, upload_id: str) -> None:
        try:
            self._client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
            )
        except Exception as exc:
            logger.warning("MinIO abort multipart failed key={}: {}", key, exc)

    @staticmethod
    def plan_parts(total_bytes: int, *, part_size: int) -> int:
        part_size = max(_MIN_PART_BYTES, int(part_size))
        if total_bytes <= 0:
            raise ValueError("empty file")
        if total_bytes <= part_size:
            return 1
        return (total_bytes + part_size - 1) // part_size


def resolve_object_storage(cfg: Any) -> MinioObjectStorage | None:
    from nanobot.knowledge.config import KnowledgeConfig

    if not isinstance(cfg, KnowledgeConfig):
        return None
    if cfg.storage.backend != "minio":
        return None
    try:
        return MinioObjectStorage(cfg.storage)
    except Exception as exc:
        logger.warning("MinIO storage init failed: {}", exc)
        return None
