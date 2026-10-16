"""Unit tests for the SciBot pipeline (no Ollama server required).

The one exception is Library.build, which uses a FakeClient that mimics
Ollama's embedding API with deterministic hashed bag-of-words vectors -
so the full ingest->index->search loop runs offline and repeatably.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scibot.bm25 import BM25, tokenize
from scibot.chunking import chunk_paper
from scibot.config import DEFAULT_EMBED_MODEL, RetrievalConfig
from scibot.embed import EmbeddingCache, embed_key
from scibot.pdf import ParsedPaper, parse_pdf
from scibot.retrieval import retrieve, rrf_fuse
from scibot.samples import generate_sample_papers
from scibot.store import Library


# --------------------------------------------------------------------- bm25
def test_tokenize_strips_stopwords_and_punctuation():
    toks = tokenize("The Testing-Effect, robust and well-documented!")
    assert "the" not in toks and "and" not in toks
    assert "testing" in toks and "effect" in toks and "documented" in toks


def test_bm25_ranks_matching_doc_first():
    corpus = [tokenize(t) for t in ["the cat sat", "quantum flux lattice", "cat and dog"]]
    bm25 = BM25(corpus)
    hits = bm25.search(tokenize("cat"), k=2)
    assert hits[0][0] == 0          # doc 0 mentions cat twice
    assert hits[1][0] == 2          # doc 2 mentions cat once


def test_bm25_returns_nothing_for_unknown_terms():
    bm25 = BM25([tokenize("alpha beta"), tokenize("gamma delta")])
    assert bm25.search(tokenize("omega"), k=3) == []


# ---------------------------------------------------------------------- rrf
def test_rrf_fusion_prefers_items_in_multiple_lists():
    fused = rrf_fuse([[1, 2, 3], [2, 5, 6]], rrf_k=60)
    assert fused[2] > fused[1]      # chunk 2 appears in both lists
    assert fused[2] > fused[5]
    assert fused[1] == pytest.approx(1 / 61)


def test_rrf_rank_ordering_respected():
    fused = rrf_fuse([[7, 8]], rrf_k=60)
    assert fused[7] > fused[8]


# ----------------------------------------------------------------- chunking
def _paper(texts: list[str]) -> ParsedPaper:
    return ParsedPaper(
        title="T", path=Path("t.pdf"), pages=list(enumerate(texts, start=1))
    )


def test_chunks_respect_size_and_page():
    page_text = " ".join(f"word{i}" for i in range(400))  # ~2800 chars
    chunks = chunk_paper(_paper([page_text, "tiny"]), size=900, overlap=150)
    assert all(len(c.text) <= 900 for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert all(c.paper == "T" for c in chunks)


def test_consecutive_chunks_overlap():
    page_text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_paper(_paper([page_text]), size=900, overlap=150)
    for a, b in zip(chunks, chunks[1:]):
        # the last 8 tokens of a must reappear in b (overlap=150 chars is
        # far more than 8 tokens; token-level check avoids mid-token slice
        # artifacts at chunk boundaries)
        tail = " ".join(tokenize(a.text)[-8:])
        assert tail in " ".join(tokenize(b.text))


def test_runt_fragments_dropped():
    # a 900-char page plus a 30-char remainder: the remainder is dropped
    chunks = chunk_paper(_paper([f"{'x' * 900} {'y' * 30}"]), size=900, overlap=0)
    assert len(chunks) == 1 and "y" not in chunks[0].text


def test_chunk_ids_stable():
    a = chunk_paper(_paper(["hello world this is a chunk"]), 900, 0)
    b = chunk_paper(_paper(["hello world this is a chunk"]), 900, 0)
    assert [c.id for c in a] == [c.id for c in b]


# ------------------------------------------------------------ embed cache
def test_embedding_cache_roundtrip(tmp_path):
    path = tmp_path / "emb.npz"
    cache = EmbeddingCache(path)
    cache.put("k1", np.array([1.0, 0.0]))
    cache.put("k2", np.array([0.0, 1.0]))
    cache.save()

    reloaded = EmbeddingCache(path)
    assert np.allclose(reloaded.get("k1"), [1.0, 0.0])
    assert np.allclose(reloaded.get("k2"), [0.0, 1.0])
    assert reloaded.get("nope") is None


def test_embedding_cache_resets_on_dimension_change(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.npz")
    cache.put("a", np.ones(4))
    cache.put("b", np.ones(8))          # different model dimension
    assert "a" not in cache.index       # old vectors discarded
    assert cache.vecs.shape == (1, 8)


# ------------------------------------------------------------------ library
class FakeClient:
    """Deterministic hashed bag-of-words embeddings, mimicking ollama's API."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    def embed(self, model: str, input: list[str]):
        self.calls += 1
        vecs = []
        for text in input:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in tokenize(text):
                v[hash(tok) % self.dim] += 1.0
            vecs.append(v.tolist())
        return {"embeddings": vecs}


@pytest.fixture(scope="module")
def indexed_library(tmp_path_factory):
    papers = tmp_path_factory.mktemp("papers")
    generate_sample_papers(papers)
    cfg = RetrievalConfig(chunk_size=700, chunk_overlap=100)
    lib = Library.build(cfg, client=FakeClient(), cache=EmbeddingCache(None), papers_dir=papers)
    return lib, cfg, papers


def test_library_built_from_sample_pdfs(indexed_library):
    lib, _, _ = indexed_library
    assert len(lib.papers) == 2
    assert len(lib.chunks) > 8
    assert lib.vectors.shape[0] == len(lib.chunks)


def test_pdf_parse_keeps_pages(indexed_library):
    _, _, papers = indexed_library
    pdf = next(papers.glob("*.pdf"))
    parsed = parse_pdf(pdf)
    assert parsed.pages and parsed.pages[0][0] == 1
    assert "retrieval practice" in " ".join(t for _, t in parsed.pages).lower()


def test_dense_and_bm25_search_shapes(indexed_library):
    lib, _, _ = indexed_library
    q = np.zeros(lib.vectors.shape[1], dtype=np.float32)
    q[0] = 1.0  # deterministic nonzero query
    dense = lib.dense_search(q, k=3)
    assert len(dense) == 3
    lex = lib.bm25_search("spindle density sleep", k=3)
    assert len(lex) == 3
    assert "sleep" in lib.chunks[lex[0][0]].text.lower()


def test_retrieve_strategies_aggregate(indexed_library):
    lib, cfg, _ = indexed_library
    fake = FakeClient()

    def embed_query(q: str) -> np.ndarray:
        return np.asarray(fake.embed("m", [q])["embeddings"][0], dtype=np.float32)

    for strategy in ("dense", "bm25", "hybrid"):
        hits = retrieve(lib, "sleep spindles consolidation", embed_query,
                        strategy=strategy, top_k=3)
        assert len(hits) == 3
        assert all(isinstance(h.score, float) for h in hits)
    with pytest.raises(ValueError):
        retrieve(lib, "q", embed_query, strategy="bogus")


def test_library_save_load_roundtrip(indexed_library, tmp_path, monkeypatch):
    lib, cfg, papers = indexed_library
    import scibot.embed as embed_mod
    import scibot.store as store_mod

    # Library.load() resolves the cache via scibot.embed.CACHE_DIR at call time
    monkeypatch.setattr(store_mod, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(embed_mod, "CACHE_DIR", tmp_path)
    cache = EmbeddingCache()  # now resolves to tmp_path/"embeddings.npz"
    for i, c in enumerate(lib.chunks):
        cache.put(embed_key(DEFAULT_EMBED_MODEL, c.embed_text), lib.vectors[i])
    cache.save()

    lib.save()
    loaded = Library.load()
    assert loaded is not None
    assert [c.id for c in loaded.chunks] == [c.id for c in lib.chunks]
    assert np.allclose(loaded.vectors, lib.vectors, atol=1e-5)


# ------------------------------------------------------------------ config
def test_signature_changes_with_chunking(tmp_path):
    from scibot.store import compute_signature

    a = compute_signature(tmp_path, RetrievalConfig(chunk_size=900))
    b = compute_signature(tmp_path, RetrievalConfig(chunk_size=500))
    assert a != b
