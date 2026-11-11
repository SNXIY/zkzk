"""Load local knowledge env files into ``os.environ`` (non-destructive)."""

from __future__ import annotations

import os
from pathlib import Path


def load_knowledge_env() -> list[Path]:
    """Load ``~/.nanobot/knowledge.env`` and repo ``nanobot/knowledge/.env`` if present.

    Existing process env wins; file values only fill missing keys.
    """
    candidates = [
        Path.home() / ".nanobot" / "knowledge.env",
        Path(__file__).resolve().parent / ".env",
    ]
    loaded: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or text.startswith("#") or "=" not in text:
                    continue
                key, _, value = text.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
            loaded.append(path)
        except OSError:
            continue
    return loaded
