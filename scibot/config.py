"""Central configuration for SciBot.

Everything tunable lives here so the app, the tests, and the docs all
agree on defaults. Paths are resolved relative to the repository root
(the directory containing this package).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPERS_DIR = REPO_ROOT / "papers"
CACHE_DIR = REPO_ROOT / ".scibot_cache"

DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class RetrievalConfig:
    """All knobs that affect indexing or retrieval.

    Changing anything in here invalidates the index signature (see
    store.compute_signature), which triggers a rebuild on the next run.
    """

    # --- chunking -----------------------------------------------------
    chunk_size: int = 900       # characters per chunk
    chunk_overlap: int = 150    # characters shared between neighbours

    # --- retrieval ------------------------------------------------------
    strategy: str = "hybrid"    # "dense" | "bm25" | "hybrid"
    top_k: int = 6              # chunks fed to the LLM
    rrf_k: int = 60             # Reciprocal Rank Fusion constant

    # --- query expansion -----------------------------------------------
    multi_query: bool = False   # LLM rewrites the question into N queries
    multi_query_n: int = 3

    def signature(self) -> str:
        """Stable string describing everything that changes an index."""
        return (
            f"chunk={self.chunk_size}:{self.chunk_overlap}|"
            f"embed={DEFAULT_EMBED_MODEL}"
        )


def ensure_dirs() -> None:
    PAPERS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
