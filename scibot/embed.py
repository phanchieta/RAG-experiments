"""Embeddings via Ollama, with a content-addressed disk cache.

Cache key = sha1(model + text). Because chunk ids are derived from
(paper, page, offset), an unchanged paper re-indexes instantly; only
new or re-chunked text hits the model. The cache is a single .npz file
- easy to inspect, easy to delete.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import ollama

from .config import CACHE_DIR, OLLAMA_HOST
from ._compat import field as _field

_BATCH = 32


def get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


class EmbeddingCache:
    """Rows of L2-unnormalised vectors ordered by insertion.

    Kept unnormalised here (normalisation is applied at query time) so
    a cached vector is never mutated.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (CACHE_DIR / "embeddings.npz")
        self.order: list[str] = []
        self.index: dict[str, int] = {}
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with np.load(self.path) as data:
            keys = data["keys"].tolist()
            vecs = data["vecs"].astype(np.float32)
        self.order = list(keys)
        self.index = {k: i for i, k in enumerate(self.order)}
        self.vecs = vecs

    def get(self, key: str) -> np.ndarray | None:
        i = self.index.get(key)
        return None if i is None else self.vecs[i]

    def put(self, key: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self.vecs.shape[1] != vec.shape[1]:
            # embedding dimension changed (different model): reset cache
            self.order, self.index = [], {}
            self.vecs = np.zeros((0, vec.shape[1]), dtype=np.float32)
        self.index[key] = len(self.order)
        self.order.append(key)
        self.vecs = np.vstack([self.vecs, vec])

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            self.path,
            keys=np.array(self.order, dtype="<U64"),
            vecs=self.vecs,
        )


def embed_key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def embed_texts(
    texts: list[str],
    model: str,
    cache: EmbeddingCache | None = None,
    client: ollama.Client | None = None,
) -> np.ndarray:
    """Embed texts, returning an (n, d) float32 matrix, L2-normalised.

    Rows come from the cache when present; only misses hit Ollama.
    """
    client = client or get_client()
    cache = cache or EmbeddingCache()

    dim = _probe_dim(client, model)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    missing: list[int] = []

    for i, text in enumerate(texts):
        cached = cache.get(embed_key(model, text))
        if cached is None:
            missing.append(i)
        else:
            out[i] = cached

    for start in range(0, len(missing), _BATCH):
        idx_batch = missing[start : start + _BATCH]
        batch = [texts[i] for i in idx_batch]
        resp = client.embed(model=model, input=batch)
        for j, emb in enumerate(_field(resp, "embeddings", [])):
            vec = np.asarray(emb, dtype=np.float32)
            i = idx_batch[j]
            out[i] = vec
            cache.put(embed_key(model, batch[j]), vec)
    cache.save()

    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _probe_dim(client: ollama.Client, model: str) -> int:
    """Dimension of the model's vectors (one tiny embed call)."""
    resp = client.embed(model=model, input=["dim"])
    return len(_field(resp, "embeddings")[0])
