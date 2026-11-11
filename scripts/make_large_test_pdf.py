"""Generate a >5MB extractable PDF for async knowledge upload testing."""

from __future__ import annotations

import os
from pathlib import Path

from pypdf import PageObject, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
    TextStringObject,
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "tmp" / "large_async_test.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    page_text = (
        "AcmeTech Escalation Guide: L1 triage, L2 specialist, L3 engineering. "
        "Sev1 SLA is 15 minutes. Sev2 SLA is 4 hours. Sev3 is next business day. "
        "Always update the ticket timeline and page the on-call for Sev1. "
    ) * 40

    writer = PdfWriter()
    pages = 40
    for p in range(pages):
        text = f"PAGE {p + 1} OF {pages}. {page_text} Unique marker {p}-{p * p}."
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines = ["BT /F1 11 Tf 50 760 Td 13 TL"]
        chunk = 140
        for i in range(0, min(len(escaped), chunk * 35), chunk):
            lines.append(f"({escaped[i:i + chunk]}) Tj T*")
        lines.append("ET")
        stream = StreamObject()
        stream.set_data("\n".join(lines).encode("latin-1", errors="replace"))

        page = PageObject.create_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        page[NameObject("/Contents")] = stream
        writer.add_page(page)

    # Pad with incompressible payload so file exceeds async threshold (5MB).
    pad_size = 6 * 1024 * 1024
    pad = StreamObject()
    pad.update(
        {
            NameObject("/Type"): NameObject("/EmbeddedFile"),
            NameObject("/Length"): NumberObject(pad_size),
        }
    )
    pad.set_data(os.urandom(pad_size))
    writer._root_object[NameObject("/Names")] = DictionaryObject(
        {
            NameObject("/EmbeddedFiles"): DictionaryObject(
                {
                    NameObject("/Names"): ArrayObject(
                        [
                            TextStringObject("payload.bin"),
                            DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/F"),
                                    NameObject("/F"): TextStringObject("payload.bin"),
                                    NameObject("/EF"): DictionaryObject(
                                        {NameObject("/F"): pad}
                                    ),
                                }
                            ),
                        ]
                    )
                }
            )
        }
    )

    with out.open("wb") as f:
        writer.write(f)

    size = out.stat().st_size
    print(f"{out}")
    print(f"{size} bytes ({size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
