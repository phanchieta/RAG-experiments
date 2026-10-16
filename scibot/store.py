"""The Library: papers parsed, chunked, embedded, and searchable.

Persistence model (all under .scibot_cache/):
  index.json      chunk metadata + the index signature it was built with
  embeddings.npz  content-addressed vector cache shared across indexes

The signature hashes the paper files (name, size, mtime) plus the
chunking config and embedding model. If the signature matches, loading
is instant; if anything changed, we rebuild - and thanks to the
embedding cache, only genuinely new chunk text gets embedded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import Chunk, chunk_paper
from .config import CACHE_DIR, DEFAULT_EMBED_MODEL, PAPERS_DIR, RetrievalConfig
from .embed import EmbeddingCache, embed_key, embed_texts, get_client
from .pdf import parse_pdf

INDEX_FILE = CACHE_DIR / "index.json"


@dataclass
class Library:
    chunks: list[Chunk]
    vectors: np.ndarray            # (n_chunks, dim), L2-normalised
    papers: dict[str, dict]        # title -> {path, pages}
    signature: str
    bm25: BM25 = field(init=False)

    def __post_init__(self) -> None:
        corpus = [tokenize(c.embed_text) for c in self.chunks]
        self.bm25 = BM25(corpus)

    # ------------------------------------------------------------ search
    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.chunks) == 0:
            return []
        sims = self.vectors @ query_vec
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(tokenize(query), k)

    # -------------------------------------------------------- persistence
    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        INDEX_FILE.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "papers": self.papers,
                    "chunks": [
                        {"id": c.id, "paper": c.paper, "page": c.page, "text": c.text}
                        for c in self.chunks
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> Library | None:
        """Load a persisted index, or None if absent/corrupt."""
        if not INDEX_FILE.exists():
            return None
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in data["chunks"]]
            cache = EmbeddingCache()
            vecs = np.stack(
                [cache.get(embed_key(DEFAULT_EMBED_MODEL, c.embed_text)) for c in chunks]
            )
            return cls(
                chunks=chunks,
                vectors=vecs,
                papers=data["papers"],
                signature=data["signature"],
            )
        except Exception as exc:
            # a stale/corrupt index falls back to a rebuild, but stay loud
            import sys

            print(f"[scibot] index load failed, rebuilding: {exc!r}", file=sys.stderr)
            return None

    # ------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        cfg: RetrievalConfig,
        client=None,
        cache: EmbeddingCache | None = None,
        papers_dir: Path = PAPERS_DIR,
    ) -> Library:
        """Parse every PDF in papers_dir, chunk, embed, index."""
        client = client or get_client()
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if not pdfs:
            return cls(chunks=[], vectors=np.zeros((0, 0)), papers={}, signature="")

        papers: dict[str, dict] = {}
        chunks: list[Chunk] = []
        for pdf in pdfs:
            parsed = parse_pdf(pdf)
            papers[parsed.title] = {"path": str(pdf), "pages": len(parsed.pages)}
            chunks.extend(chunk_paper(parsed, cfg.chunk_size, cfg.chunk_overlap))

        vectors = embed_texts(
            [c.embed_text for c in chunks],
            model=DEFAULT_EMBED_MODEL,
            cache=cache,
            client=client,
        )
        lib = cls(
            chunks=chunks,
            vectors=vectors,
            papers=papers,
            signature=compute_signature(papers_dir, cfg),
        )
        lib.save()
        return lib


def compute_signature(papers_dir: Path, cfg: RetrievalConfig) -> str:
    """Hash paper files + chunking config; detects when a rebuild is due."""
    parts = [cfg.signature()]
    for pdf in sorted(papers_dir.glob("*.pdf")):
        stat = pdf.stat()
        parts.append(f"{pdf.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def load_or_build(
    cfg: RetrievalConfig, client=None, cache: EmbeddingCache | None = None
) -> tuple[Library, bool]:
    """Return (library, rebuilt). Rebuilds only when the signature changed."""
    sig = compute_signature(PAPERS_DIR, cfg)
    lib = Library.load() if PAPERS_DIR.glob("*.pdf") else None
    if lib is not None and lib.signature == sig and len(lib.chunks):
        return lib, False
    lib = Library.build(cfg, client=client, cache=cache)
    return lib, True
