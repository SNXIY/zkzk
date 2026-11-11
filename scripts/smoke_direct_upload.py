"""Smoke: MinIO presigned multipart → register queued doc."""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.webui.knowledge_api import knowledge_upload_complete, knowledge_upload_init


def main() -> None:
    cfg = load_config()
    ws = Path(tempfile.mkdtemp(prefix="kb-up-"))
    size = 20 * 1024 * 1024
    data = b"A" * size
    init = knowledge_upload_init(
        ws,
        filename="big.bin",
        size=size,
        content_type="application/octet-stream",
        tools_cfg=cfg.tools,
    )
    print("init", init["mode"], "parts", init.get("part_count"))
    assert init["mode"] == "presign_multipart", init

    parts_out: list[dict[str, object]] = []
    part_size = int(init["part_size"])
    for part in init["parts"]:
        start = (int(part["part_number"]) - 1) * part_size
        end = min(size, start + part_size)
        chunk = data[start:end]
        req = urllib.request.Request(part["url"], data=chunk, method="PUT")
        with urllib.request.urlopen(req, timeout=120) as resp:
            etag = (resp.headers.get("ETag") or resp.headers.get("etag") or "").strip('"')
            print("part", part["part_number"], resp.status, etag[:20], len(chunk))
            assert etag
            parts_out.append({"part_number": int(part["part_number"]), "etag": etag})

    result = knowledge_upload_complete(
        ws,
        filename="big.bin",
        doc_id=str(init["doc_id"]),
        object_key=str(init["object_key"]),
        size=size,
        mode="presign_multipart",
        upload_id=str(init["upload_id"]),
        parts=parts_out,
        tools_cfg=cfg.tools,
    )
    doc = result["document"]
    print("doc", doc["status"], doc["id"], doc["bytes"])
    assert doc["status"] == "queued"
    print("DIRECT MULTIPART OK")


if __name__ == "__main__":
    main()
