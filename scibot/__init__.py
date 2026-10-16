"""SciBot: a transparent, local-first RAG playground for papers."""

from .bm25 import BM25, tokenize
from .chunking import Chunk, chunk_paper
from .config import RetrievalConfig
from .embed import EmbeddingCache, embed_texts
from .generation import (
    check_groundedness,
    condense_question,
    stream_answer,
)
from .pdf import ParsedPaper, parse_pdf
from .retrieval import Retrieved, expand_query, retrieve, rrf_fuse
from .store import Library, load_or_build

__all__ = [
    "BM25",
    "Chunk",
    "EmbeddingCache",
    "Library",
    "ParsedPaper",
    "RetrievalConfig",
    "Retrieved",
    "check_groundedness",
    "chunk_paper",
    "condense_question",
    "embed_texts",
    "expand_query",
    "load_or_build",
    "parse_pdf",
    "retrieve",
    "rrf_fuse",
    "stream_answer",
    "tokenize",
]
