"""Headless end-to-end smoke test: sample PDFs -> index -> retrieve -> answer."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scibot.config import PAPERS_DIR, RetrievalConfig
from scibot.embed import embed_texts, get_client
from scibot.generation import stream_answer
from scibot.retrieval import expand_query, retrieve
from scibot.samples import generate_sample_papers
from scibot.store import load_or_build


def main() -> None:
    made = generate_sample_papers(PAPERS_DIR)
    print(f"[1] sample papers: {len(made)} created")

    t0 = time.perf_counter()
    cfg = RetrievalConfig()
    lib, rebuilt = load_or_build(cfg)
    dt = time.perf_counter() - t0
    print(
        f"[2] library: {len(lib.chunks)} chunks from {len(lib.papers)} papers "
        f"(rebuilt={rebuilt}) in {dt:.1f}s"
    )
    assert lib.chunks, "no chunks!"
    for title, meta in lib.papers.items():
        print(f"    - {title[:60]} ({meta['pages']} pages)")

    # second load must be a cache hit
    t0 = time.perf_counter()
    lib2, rebuilt2 = load_or_build(cfg)
    print(
        f"[3] reload: rebuilt={rebuilt2} in {time.perf_counter() - t0:.2f}s"
    )
    assert not rebuilt2, "index did not survive a reload!"

    question = "Does testing improve long-term retention compared to restudy?"
    client = get_client()

    def embed_query(q: str) -> np.ndarray:
        return embed_texts([q], model="nomic-embed-text", client=client)[0]

    for strategy in ("dense", "bm25", "hybrid"):
        t0 = time.perf_counter()
        hits = retrieve(lib, question, embed_query, strategy=strategy, top_k=4)
        dt = time.perf_counter() - t0
        top = hits[0]
        print(
            f"[4:{strategy:6}] {dt*1000:5.0f}ms  top-1: p{top.chunk.page} "
            f"'{top.chunk.text[:60]}...' ranks={top.ranks}"
        )
        assert "retrieval practice" in top.chunk.text.lower(), (
            f"{strategy} failed to find the testing-effect passage"
        )

    expanded = expand_query(question, model="llama3.2", client=client)
    print(f"[5] multi-query expansion: {expanded}")

    hits = retrieve(lib, question, embed_query, strategy="hybrid", top_k=4)
    t0 = time.perf_counter()
    answer = "".join(
        stream_answer(client, "llama3.2", question, [h.chunk for h in hits])
    )
    dt = time.perf_counter() - t0
    print(f"[6] answer ({dt:.1f}s):\n{answer}\n")

    cited = [h for i, h in enumerate(hits, 1) if i in _cited(answer)]
    print(f"[7] citations used: {sorted(_cited(answer))} of {len(hits)} sources")


def _cited(text: str) -> set[int]:
    import re

    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}


if __name__ == "__main__":
    main()
