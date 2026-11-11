"""Document text extraction and structure-aware chunking."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+\n")
_MULTI_NL = re.compile(r"\n{3,}")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_PAGE_MARK = re.compile(r"^---\s*Page\s+(\d+)\s*---\s*$", re.MULTILINE | re.IGNORECASE)


@dataclass(frozen=True)
class TextChunk:
    """One indexed chunk with a hierarchical breadcrumb for retrieval context."""

    text: str
    breadcrumb: str
    page: int | None = None


def extract_text(path: Path) -> str:
    """Extract plain text from a supported document path."""
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in {".txt", ".md", ".markdown", ".csv", ".log"}:
        return _decode_text(raw)
    if suffix in {".html", ".htm"}:
        return _html_to_text(_decode_text(raw))
    if suffix == ".pdf":
        return _extract_pdf(path, raw)
    return _decode_text(raw)


def chunk_text(
    text: str,
    *,
    title: str = "Document",
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Backward-compatible wrapper returning only chunk texts."""
    return [c.text for c in chunk_document(text, title=title, chunk_size=chunk_size, overlap=overlap)]


def chunk_document(
    text: str,
    *,
    title: str = "Document",
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Structure-aware chunking: headings / PDF pages / paragraphs, with breadcrumbs.

    Strategy (enterprise RAG style, without pulling in LangChain):
    1. Prefer markdown ATX headings as section boundaries.
    2. Else prefer PDF page markers produced by :func:`extract_text`.
    3. Pack section body into ~chunk_size windows with overlap, never crossing
       the section boundary except via overlap at pack time inside the section.
    """
    cleaned = _MULTI_NL.sub("\n\n", text.replace("\r\n", "\n").strip())
    if not cleaned:
        return []

    sections = _split_sections(cleaned, title=title)
    chunks: list[TextChunk] = []
    for section in sections:
        packed = _pack_section(
            section["body"],
            breadcrumb=section["breadcrumb"],
            page=section.get("page"),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(packed)
    return chunks


def _split_sections(text: str, *, title: str) -> list[dict]:
    headings = list(_MD_HEADING.finditer(text))
    if headings:
        sections: list[dict] = []
        stack: list[tuple[int, str]] = []
        # Leading content before first heading
        if headings[0].start() > 0:
            lead = text[: headings[0].start()].strip()
            if lead:
                sections.append({"breadcrumb": title, "body": lead, "page": None})
        for i, match in enumerate(headings):
            level = len(match.group(1))
            heading = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            crumb = " > ".join([title, *[h for _, h in stack]])
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[match.end() : end].strip()
            # Keep heading text inside the chunk for better lexical recall.
            body_full = f"{heading}\n\n{body}".strip() if body else heading
            sections.append({"breadcrumb": crumb, "body": body_full, "page": None})
        return sections

    pages = list(_PAGE_MARK.finditer(text))
    if pages:
        sections = []
        if pages[0].start() > 0:
            lead = text[: pages[0].start()].strip()
            if lead:
                sections.append({"breadcrumb": title, "body": lead, "page": None})
        for i, match in enumerate(pages):
            page_no = int(match.group(1))
            end = pages[i + 1].start() if i + 1 < len(pages) else len(text)
            body = text[match.end() : end].strip()
            if not body:
                continue
            sections.append(
                {
                    "breadcrumb": f"{title} > Page {page_no}",
                    "body": body,
                    "page": page_no,
                }
            )
        return sections

    return [{"breadcrumb": title, "body": text, "page": None}]


def _pack_section(
    body: str,
    *,
    breadcrumb: str,
    page: int | None,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    body = body.strip()
    if not body:
        return []
    if len(body) <= chunk_size:
        return [TextChunk(text=body, breadcrumb=breadcrumb, page=page)]

    parts: list[TextChunk] = []
    start = 0
    length = len(body)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = body[start:end]
            break_at = window.rfind("\n\n")
            if break_at < chunk_size // 3:
                break_at = window.rfind("\n")
            if break_at < chunk_size // 3:
                break_at = window.rfind(". ")
                if break_at >= 0:
                    break_at += 1
            if break_at < chunk_size // 3:
                break_at = window.rfind(" ")
            if break_at >= chunk_size // 3:
                end = start + break_at + 1
        piece = body[start:end].strip()
        if piece:
            # Prefix breadcrumb lightly so embeddings see section context.
            embed_text = f"[{breadcrumb}]\n{piece}" if breadcrumb else piece
            parts.append(TextChunk(text=embed_text, breadcrumb=breadcrumb, page=page))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return parts


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _html_to_text(content: str) -> str:
    content = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", content)
    content = _TAG_RE.sub(" ", content)
    content = html.unescape(content)
    content = _WS_RE.sub("\n", content)
    return _MULTI_NL.sub("\n\n", content).strip()


def _extract_pdf(path: Path, raw: bytes) -> str:
    """Extract PDF text with page markers for structure-aware chunking."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(stream=raw, filetype="pdf")
        try:
            parts: list[str] = []
            for i, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                if text:
                    parts.append(f"--- Page {i} ---\n{text}")
        finally:
            doc.close()
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass

    try:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw))
        parts = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"--- Page {i} ---\n{text}")
        return "\n\n".join(parts)
    except Exception as exc:
        raise ValueError(f"failed to extract PDF text from {path.name}: {exc}") from exc
