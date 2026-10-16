"""PDF parsing built on PyMuPDF.

One job: turn a PDF file into an ordered list of (page_number, text).
We keep page numbers so every citation SciBot produces can point at an
exact page in the source document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class ParsedPaper:
    title: str
    path: Path
    pages: list[tuple[int, str]]  # (1-based page number, text)


def parse_pdf(path: Path | str) -> ParsedPaper:
    """Extract text page by page from a PDF.

    Whitespace is normalised (PDFs are full of hard line breaks) but
    paragraph breaks are preserved so chunking can respect them.
    """
    path = Path(path)
    with pymupdf.open(path) as doc:
        meta_title = (doc.metadata or {}).get("title") or ""
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(doc, start=1):
            text = _normalise(page.get_text("text"))
            if text:
                pages.append((i, text))

    title = meta_title.strip() or path.stem
    if not pages:
        raise ValueError(f"No extractable text in {path.name} (scanned PDF?)")
    return ParsedPaper(title=title, path=path, pages=pages)


def _normalise(text: str) -> str:
    """Collapse runs of spaces, drop trailing spaces, keep line structure."""
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines).strip()
