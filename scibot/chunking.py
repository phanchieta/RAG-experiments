"""Chunking: split parsed pages into retrievable passages.

Design choice: we chunk *within* a page and never across pages. A chunk
that spans a page break would have an ambiguous citation, and exact
page-level citations are the whole point of a research assistant. The
cost is that the occasional paragraph crossing a page break gets split
- acceptable for precise sourcing.

Each chunk carries a stable id (hash of title + page + offset) so its
embedding can be cached across re-indexes: change the chunk size and
only genuinely new chunks get embedded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .pdf import ParsedPaper

MIN_CHUNK_CHARS = 80  # discard runt fragments at the end of a page


@dataclass
class Chunk:
    id: str
    paper: str   # paper title
    page: int    # 1-based page number
    text: str

    @property
    def embed_text(self) -> str:
        """Text actually embedded / indexed: title prepended for context.

        "p. 12" alone retrieves poorly for the query "testing effect";
        "The Testing Effect\n\np. 12 ..." retrieves well.
        """
        return f"{self.paper}\n\n{self.text}"


def chunk_paper(paper: ParsedPaper, size: int, overlap: int) -> list[Chunk]:
    """Sliding window over each page of a paper."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    step = size - overlap

    chunks: list[Chunk] = []
    for page_no, text in paper.pages:
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(paper.title, page_no, start),
                    paper=paper.title,
                    page=page_no,
                    text=piece,
                )
            )
    return chunks


def _chunk_id(title: str, page: int, start: int) -> str:
    raw = f"{title}|{page}|{start}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
