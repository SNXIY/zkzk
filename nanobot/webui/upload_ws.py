"""WebUI knowledge / skill uploads over the gateway WebSocket.

Large knowledge files use browser→MinIO presigned PUT/multipart (control plane
on the socket). Legacy base64 data-URL upload remains as a fallback.
"""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from nanobot.config.loader import load_config
from nanobot.webui.knowledge_api import (
    knowledge_upload,
    knowledge_upload_complete,
    knowledge_upload_init,
)
from nanobot.webui.skills_install import install_workspace_skill

_MAX_REQUEST_ID_LENGTH = 80
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;]*);base64,(?P<data>.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _request_id_ok(request_id: Any) -> bool:
    return isinstance(request_id, str) and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH


async def webui_knowledge_upload_event(
    envelope: dict[str, Any],
    *,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    request_id = envelope.get("request_id")
    valid = _request_id_ok(request_id)

    def error(detail: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"detail": detail, **extra}
        if valid:
            payload["request_id"] = request_id
        return "knowledge_upload_error", payload

    if not valid:
        return error("invalid_request")

    try:
        filename, raw = _decode_upload(envelope)
        config = load_config()
        doc = knowledge_upload(
            workspace,
            filename=filename,
            data=raw,
            tools_cfg=config.tools,
        )
    except ValueError as exc:
        return error(str(exc))
    except Exception as exc:
        return error(f"upload_failed: {exc}")
    return "knowledge_uploaded", {"request_id": request_id, **doc}


async def webui_knowledge_upload_init_event(
    envelope: dict[str, Any],
    *,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    request_id = envelope.get("request_id")
    valid = _request_id_ok(request_id)

    def error(detail: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"detail": detail, **extra}
        if valid:
            payload["request_id"] = request_id
        return "knowledge_upload_init_error", payload

    if not valid:
        return error("invalid_request")

    filename = envelope.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        return error("filename is required")
    try:
        size = int(envelope.get("bytes") or envelope.get("size") or 0)
    except (TypeError, ValueError):
        return error("invalid bytes")
    content_type = envelope.get("content_type") or "application/octet-stream"
    if not isinstance(content_type, str):
        content_type = "application/octet-stream"

    try:
        config = load_config()
        payload = knowledge_upload_init(
            workspace,
            filename=filename.strip(),
            size=size,
            content_type=content_type,
            tools_cfg=config.tools,
        )
    except ValueError as exc:
        return error(str(exc))
    except Exception as exc:
        return error(f"upload_init_failed: {exc}")
    return "knowledge_upload_init_ok", {"request_id": request_id, **payload}


async def webui_knowledge_upload_complete_event(
    envelope: dict[str, Any],
    *,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    request_id = envelope.get("request_id")
    valid = _request_id_ok(request_id)

    def error(detail: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"detail": detail, **extra}
        if valid:
            payload["request_id"] = request_id
        return "knowledge_upload_error", payload

    if not valid:
        return error("invalid_request")

    filename = envelope.get("filename")
    doc_id = envelope.get("doc_id")
    object_key = envelope.get("object_key")
    mode = envelope.get("mode") or "presign_put"
    if not isinstance(filename, str) or not filename.strip():
        return error("filename is required")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return error("doc_id is required")
    if not isinstance(object_key, str) or not object_key.strip():
        return error("object_key is required")
    if not isinstance(mode, str):
        return error("invalid mode")
    try:
        size = int(envelope.get("bytes") or envelope.get("size") or 0)
    except (TypeError, ValueError):
        return error("invalid bytes")

    upload_id = envelope.get("upload_id")
    if upload_id is not None and not isinstance(upload_id, str):
        return error("invalid upload_id")
    parts = envelope.get("parts")
    if parts is not None and not isinstance(parts, list):
        return error("invalid parts")
    sha256 = envelope.get("sha256") if isinstance(envelope.get("sha256"), str) else ""

    try:
        config = load_config()
        doc = knowledge_upload_complete(
            workspace,
            filename=filename.strip(),
            doc_id=doc_id.strip(),
            object_key=object_key.strip(),
            size=size,
            mode=mode,
            upload_id=upload_id,
            parts=parts,
            sha256=sha256 or "",
            tools_cfg=config.tools,
        )
    except ValueError as exc:
        return error(str(exc))
    except Exception as exc:
        return error(f"upload_complete_failed: {exc}")
    return "knowledge_uploaded", {"request_id": request_id, **doc}


async def webui_skill_upload_event(
    envelope: dict[str, Any],
    *,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    request_id = envelope.get("request_id")
    valid = isinstance(request_id, str) and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH

    def error(detail: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"detail": detail, **extra}
        if valid:
            payload["request_id"] = request_id
        return "skill_upload_error", payload

    if not valid:
        return error("invalid_request")

    try:
        filename, raw = _decode_upload(envelope)
        result = install_workspace_skill(workspace, filename=filename, data=raw)
    except Exception as exc:
        return error(str(exc))
    return "skill_uploaded", {"request_id": request_id, **result}


def _decode_upload(envelope: dict[str, Any]) -> tuple[str, bytes]:
    filename = envelope.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename is required")
    filename = Path(unquote(filename.strip())).name
    if not filename:
        raise ValueError("filename is required")

    data_url = envelope.get("data_url")
    raw_b64 = envelope.get("content_base64")
    if isinstance(data_url, str) and data_url.strip():
        match = _DATA_URL_RE.match(data_url.strip())
        if not match:
            raise ValueError("invalid data_url")
        raw_b64 = match.group("data")
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        raise ValueError("content_base64 or data_url is required")
    try:
        raw = base64.b64decode(raw_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 content") from exc
    return filename, raw
