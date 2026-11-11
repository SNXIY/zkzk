"""Read HTTP request bodies during websockets ``process_request``.

The gateway HTTP surface rides on the WebSocket handshake path, which only
parses headers. Upload routes drain ``Content-Length`` bytes from the
protocol stream reader while the event loop continues to feed data.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import unquote

from nanobot.webui.http_utils import case_insensitive_header

_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', re.I)


async def read_request_body(
    connection: Any,
    request: Any,
    *,
    max_bytes: int,
    timeout_s: float = 60.0,
) -> bytes:
    """Read exactly ``Content-Length`` bytes from the connection reader."""
    raw = case_insensitive_header(request.headers, "Content-Length")
    if not raw:
        return b""
    try:
        length = int(raw)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length < 0:
        raise ValueError("invalid Content-Length")
    if length > max_bytes:
        raise ValueError(f"body exceeds {max_bytes} bytes")
    if length == 0:
        return b""

    reader = getattr(getattr(connection, "protocol", None), "reader", None)
    if reader is None:
        raise RuntimeError("connection does not expose an HTTP body reader")

    deadline = time.monotonic() + timeout_s
    while len(reader.buffer) < length:
        if getattr(reader, "eof", False):
            raise EOFError(
                f"stream ended after {len(reader.buffer)} bytes, expected {length}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError("timed out reading request body")
        await asyncio.sleep(0.01)

    data = bytes(reader.buffer[:length])
    del reader.buffer[:length]
    return data


def parse_multipart_file(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Extract the first ``file`` part from a multipart body."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "", re.I)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = (match.group(1) or match.group(2) or "").strip()
    if not boundary:
        raise ValueError("empty multipart boundary")

    delimiter = b"--" + boundary.encode("utf-8")
    for part in body.split(delimiter):
        if not part or part in (b"--\r\n", b"--", b"--\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if part == b"--":
            continue
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        if 'name="file"' not in headers and "name=file" not in headers.lower():
            # Also accept a lone file part without a name.
            if "filename=" not in headers.lower() and "filename*=" not in headers.lower():
                continue
        filename = "upload.bin"
        for line in headers.split("\r\n"):
            if "filename" in line.lower():
                found = _FILENAME_RE.search(line)
                if found:
                    filename = unquote(found.group(1).strip().strip('"')) or filename
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return filename, payload
    raise ValueError("file field not found in multipart body")


def parse_upload_payload(
    body: bytes,
    *,
    content_type: str,
    filename_header: str = "",
) -> tuple[str, bytes]:
    """Parse multipart or raw octet-stream uploads."""
    ctype = (content_type or "").lower()
    if "multipart/form-data" in ctype:
        return parse_multipart_file(body, content_type)
    name = (filename_header or "").strip() or "upload.bin"
    return name, body
