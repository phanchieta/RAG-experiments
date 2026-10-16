"""Retrieval strategies and fusion.

One idea powers everything here: *collect rankings, then fuse them*.

  dense   ranking: cosine similarity between query and chunk embeddings
  bm25    ranking: lexical scoring (exact terms, acronyms, names)
  hybrid  both of the above

Multi-query expansion slots into the same idea: an LLM rewrites the
question into N alternative phrasings, each produces its own ranking,
and everything is fused together. It helps when the user's phrasing
doesn't match the paper's terminology.

Fusion is Reciprocal Rank Fusion (RRF): a chunk's score is the sum,
over every ranking it appears in, of 1/(rrf_k + rank). RRF only looks
at ranks, so it needs no score normalisation between incomparable
scales (cosine in [0,1] vs BM25 in [0, ~20]). Scores are meaningful
*within* one query, never across queries or strategies.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._compat import field as _field
from .chunking import Chunk
from .embed import get_client
from .store import Library

VALID_STRATEGIES = ("dense", "bm25", "hybrid")

# how deep each ranking goes before fusion (wider than top_k so fusion
# can pull up chunks that only one signal found)
CANDIDATE_MULTIPLIER = 3


@dataclass
class Retrieved:
    chunk: Chunk
    score: float  # RRF score (comparable within one query only)
    ranks: dict[str, int] = field(default_factory=dict)  # e.g. {"dense": 1, "bm25": 3}


def rrf_fuse(rankings: list[list[int]], rrf_k: int = 60) -> dict[int, float]:
    """Fuse ranked lists of chunk indices into {chunk_index: rrf_score}."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (rrf_k + rank + 1)
    return dict(fused)


def retrieve(
    lib: Library,
    question: str,
    embed_query: Callable[[str], np.ndarray],
    strategy: str = "hybrid",
    top_k: int = 6,
    rrf_k: int = 60,
    extra_queries: list[str] | None = None,
) -> list[Retrieved]:
    """Retrieve the top_k chunks for a question using the chosen strategy.

    `embed_query` turns a query string into a normalised vector (only
    called when the strategy uses dense retrieval). `extra_queries` are
    the output of expand_query(); their rankings join the fusion pool.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}")

    queries = [question] + [q for q in (extra_queries or []) if q.strip()]
    k = top_k * CANDIDATE_MULTIPLIER

    rankings: list[list[int]] = []
    first_dense: list[int] = []
    first_bm25: list[int] = []
    for i, q in enumerate(queries):
        if strategy in ("dense", "hybrid"):
            r = [idx for idx, _ in lib.dense_search(embed_query(q), k=k)]
            rankings.append(r)
            if i == 0:
                first_dense = r
        if strategy in ("bm25", "hybrid"):
            r = [idx for idx, _ in lib.bm25_search(q, k=k)]
            rankings.append(r)
            if i == 0:
                first_bm25 = r

    fused = rrf_fuse(rankings, rrf_k=rrf_k)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    dense_rank = {idx: r + 1 for r, idx in enumerate(first_dense)}
    bm25_rank = {idx: r + 1 for r, idx in enumerate(first_bm25)}

    out = []
    for idx, score in top:
        ranks: dict[str, int] = {}
        if idx in dense_rank:
            ranks["dense"] = dense_rank[idx]
        if idx in bm25_rank:
            ranks["bm25"] = bm25_rank[idx]
        out.append(Retrieved(chunk=lib.chunks[idx], score=score, ranks=ranks))
    return out


def expand_query(
    question: str,
    model: str,
    n: int = 3,
    client=None,
) -> list[str]:
    """Ask a small LLM for N alternative search phrasings of the question.

    Falls back to no expansion on any failure - retrieval must never
    break because a rewrite didn't parse.
    """
    client = client or get_client()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite questions for a scholarly search engine. "
                        "Given a question, write alternative search queries using "
                        "terminology an academic paper would use. Respond with ONLY "
                        "a JSON array of strings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Question: "{question}"\n\n'
                        f"Write {n} alternative queries as a JSON array."
                    ),
                },
            ],
            options={"temperature": 0.3},
        )
        text = _field(_field(resp, "message"), "content", "")
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        queries = json.loads(match.group(0)) if match else []
        return [str(q).strip() for q in queries if str(q).strip()][:n]
    except Exception:
        return []
