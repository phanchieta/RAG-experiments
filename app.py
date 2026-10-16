"""SciBot - a transparent, local-first RAG playground for papers.

Run:  streamlit run app.py     (requires Ollama: `ollama serve`)

Everything the retrieval does is visible in the UI: which strategy
found each chunk, how it was scored, the raw prompt, and the exact
PDF page behind every citation. See the "How it works" tab for a tour.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pymupdf
import streamlit as st

from scibot._compat import field as _field
from scibot.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    OLLAMA_HOST,
    PAPERS_DIR,
    RetrievalConfig,
)
from scibot.embed import embed_texts
from scibot.generation import (
    check_groundedness,
    condense_question,
    stream_answer,
    tidy_citations,
)
from scibot.retrieval import expand_query, retrieve
from scibot.samples import generate_sample_papers
from scibot.store import compute_signature, load_or_build

# --------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def ollama_client():
    import ollama

    return ollama.Client(host=OLLAMA_HOST)


@st.cache_resource(show_spinner="Talking to Ollama...")
def ollama_models() -> tuple[list[str], bool]:
    """(chat model names, embed model available).

    We hit /api/tags over plain HTTP because the `capabilities` field
    that marks embedding models is sent by the server but dropped by
    some ollama client versions' response models. For old servers that
    don't send capabilities at all, fall back to the model's name.
    """
    try:
        import httpx

        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json()["models"]
    except Exception:
        return [], False

    chat, embed_ok = [], False
    for m in models:
        name = m.get("name", "")
        caps = m.get("capabilities") or []
        is_embed = "embedding" in caps or "embed" in name
        if is_embed:
            if name.startswith(DEFAULT_EMBED_MODEL):
                embed_ok = True
        else:
            chat.append(name)
    return sorted(chat), embed_ok


@st.cache_resource(show_spinner=False)
def get_library(signature: str, chunk_size: int, chunk_overlap: int):
    """Build or load the index; keyed on the chunking signature."""
    cfg = RetrievalConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    lib, rebuilt = load_or_build(cfg, client=ollama_client())
    return lib


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def ensure_demo_papers() -> None:
    """First run with an empty library: drop in the synthetic demos."""
    if not list(PAPERS_DIR.glob("*.pdf")):
        generate_sample_papers(PAPERS_DIR)


def page_image(pdf_path: Path, page_no: int, terms: list[str]) -> np.ndarray | None:
    """Render a PDF page with query terms highlighted, as an RGB array."""
    try:
        with pymupdf.open(pdf_path) as doc:
            page = doc[page_no - 1]
            for term in terms[:6]:
                if term and len(term) > 2:
                    rects = page.search_for(term)
                    if rects:
                        page.add_highlight_annot(rects)
            pix = page.get_pixmap(dpi=130)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            return img[:, :, :3]
    except Exception:
        return None


def query_terms(question: str) -> list[str]:
    from scibot.bm25 import tokenize

    return tokenize(question)


def fmt_ranks(ranks: dict[str, int]) -> str:
    if not ranks:
        return ""
    parts = [f"{name} #{rank}" for name, rank in sorted(ranks.items())]
    return " · ".join(parts)


def init_state() -> None:
    st.session_state.setdefault("messages", [])


# --------------------------------------------------------------------------
# page setup
# --------------------------------------------------------------------------

st.set_page_config(page_title="SciBot", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; }
      [data-testid="stSidebar"] .stSlider > label { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()
ensure_demo_papers()

st.title("🔬 SciBot")
st.caption(
    "Chat with your papers - every retrieval decision visible. "
    "100% local, 100% free."
)

# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

chat_models, embed_ok = ollama_models()

with st.sidebar:
    st.header("📚 Library")
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    if pdfs:
        st.caption(f"{len(pdfs)} paper(s) in `papers/`")
    else:
        st.warning("No PDFs yet - drop some below, or add the demo papers.")

    uploaded = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            (PAPERS_DIR / f.name).write_bytes(f.getbuffer())
        st.cache_resource.clear()
        st.success(f"Added {len(uploaded)} file(s) - indexing on next run...")
        st.rerun()

    if pdfs:
        to_remove = st.selectbox("Remove a paper", [p.name for p in pdfs], index=None,
                                 placeholder="pick a paper...")
        if to_remove and st.button("🗑 Remove selected"):
            (PAPERS_DIR / to_remove).unlink()
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.header("🧠 Models")
    if not chat_models:
        st.error(
            f"Ollama is not reachable at `{OLLAMA_HOST}`.\n\n"
            "Install it from ollama.com, pull a chat model "
            f"(`ollama pull {DEFAULT_CHAT_MODEL}`) and an embedding model "
            f"(`ollama pull {DEFAULT_EMBED_MODEL}`), then start with `ollama serve`."
        )
        st.stop()
    if not embed_ok:
        st.error(
            f"Embedding model `{DEFAULT_EMBED_MODEL}` not found. Run:\n\n"
            f"`ollama pull {DEFAULT_EMBED_MODEL}`"
        )
        st.stop()

    def find_default(names: list[str], wanted: str) -> int:
        """Exact match, or match ignoring the :latest-style tag suffix."""
        for i, n in enumerate(names):
            if n == wanted or n.split(":")[0] == wanted:
                return i
        return 0

    chat_model = st.selectbox(
        "Chat model", chat_models, index=find_default(chat_models, DEFAULT_CHAT_MODEL)
    )

    st.divider()
    st.header("🔎 Retrieval")
    strategy = st.radio(
        "Strategy",
        ["hybrid", "dense", "bm25"],
        index=0,
        help=(
            "hybrid = embeddings + keyword search fused by rank (recommended). "
            "dense = semantic similarity only. bm25 = keyword matching only."
        ),
    )
    top_k = st.slider("Chunks per answer", 2, 12, 6)
    multi_query = st.toggle("Multi-query expansion", value=False,
                            help="The LLM rewrites your question into extra "
                                 "search queries; results are fused.")
    verify = st.toggle("Verify groundedness", value=False,
                       help="A second LLM pass flags claims not supported "
                            "by the retrieved excerpts.")
    show_prompt = st.toggle("Show full prompt", value=False)

    with st.expander("Chunking (re-indexes on change)"):
        chunk_size = st.slider("Chunk size (chars)", 400, 2000, 900, step=50)
        chunk_overlap = st.slider("Overlap (chars)", 0, 400, 150, step=25)

    if st.button("♻️ Force re-index"):
        st.cache_resource.clear()
        st.rerun()

cfg = RetrievalConfig(
    chunk_size=chunk_size, chunk_overlap=min(chunk_overlap, chunk_size - 50)
)
signature = compute_signature(PAPERS_DIR, cfg)

try:
    lib = get_library(signature, cfg.chunk_size, cfg.chunk_overlap)
except Exception as exc:
    st.error(f"Indexing failed: {exc}")
    st.stop()

status = st.sidebar.container()
with status:
    st.caption(
        f"Index: **{len(lib.chunks)}** chunks · {len(lib.papers)} papers · "
        f"`{DEFAULT_EMBED_MODEL}`"
    )

# --------------------------------------------------------------------------
# shared: retrieval + answer pipeline
# --------------------------------------------------------------------------

def embed_query(q: str) -> np.ndarray:
    return embed_texts([q], model=DEFAULT_EMBED_MODEL, client=ollama_client())[0]


def run_retrieval(question: str) -> tuple[list, dict]:
    """Retrieve for a question; returns (hits, meta about the retrieval)."""
    t0 = time.perf_counter()
    extras: list[str] = []
    if multi_query:
        extras = expand_query(question, model=chat_model, n=3, client=ollama_client())
    q_vec = embed_query(question)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = retrieve(
        lib, question, embed_query,
        strategy=strategy, top_k=top_k, extra_queries=extras,
    )
    t_search = time.perf_counter() - t0
    meta = {
        "extras": extras,
        "t_embed": t_embed,
        "t_search": t_search,
        "strategy": strategy,
        "top_k": top_k,
    }
    return hits, meta


def render_sources(hits: list, question: str) -> None:
    terms = query_terms(question)
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        with st.expander(f"[{i}] {c.paper} — p. {c.page}   ({fmt_ranks(hit.ranks)})"):
            st.markdown(c.text)
            img = page_image(Path(lib.papers[c.paper]["path"]), c.page, terms)
            if img is not None:
                st.image(img, caption=f"{c.paper}, page {c.page} "
                                      "(highlights = your query terms)")


def render_meta(meta: dict, question: str, hits: list) -> None:
    bits = [f"strategy `{meta['strategy']}`"]
    if meta["extras"]:
        bits.append(f"expanded to {1 + len(meta['extras'])} queries")
    bits.append(f"retrieve {meta['t_search']*1000:.0f} ms")
    if meta.get("t_generate"):
        bits.append(f"generate {meta['t_generate']:.1f} s")
    st.caption(" · ".join(bits))

    rewrote = meta.get("rewrote")
    if rewrote and rewrote != question:
        st.caption(f"searched as: *{rewrote}*")
    if meta["extras"]:
        with st.expander("Alternative queries the LLM wrote"):
            for q in meta["extras"]:
                st.markdown(f"- {q}")
    if show_prompt:
        from scibot.generation import SYSTEM_PROMPT, build_context

        with st.expander("The exact prompt sent to the LLM"):
            st.text(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n"
                    f"## Excerpts from the library\n\n{build_context([h.chunk for h in hits])}"
                    f"\n\n## Question\n\n{question}")


# --------------------------------------------------------------------------
# tab: chat
# --------------------------------------------------------------------------

tab_chat, tab_lab, tab_docs = st.tabs(["💬 Chat", "🧪 RAG Lab", "📖 How it works"])

with tab_chat:
    for msg in st.session_state.messages:
        avatar = "🧑‍🔬" if msg["role"] == "user" else "🔬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(
                tidy_citations(msg["content"]) if msg["role"] == "assistant" else msg["content"]
            )
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    render_sources(msg["sources"], msg.get("question", ""))
                if msg.get("meta"):
                    render_meta(msg["meta"], msg.get("question", ""), msg["sources"])
                if msg.get("groundedness") and not msg["groundedness"].get("grounded", True):
                    st.warning(
                        "⚠️ Groundedness check flagged possible unsupported "
                        "claims:\n\n- "
                        + "\n- ".join(msg["groundedness"]["issues"])
                    )

    question = st.chat_input("Ask about your papers...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🔬"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            # conversational follow-ups need a standalone question for search
            t0 = time.perf_counter()
            search_question = condense_question(
                ollama_client(), chat_model, question,
                history=history[:-1],
            ) if len(st.session_state.messages) > 1 else question
            t_condense = time.perf_counter() - t0

            hits, meta = run_retrieval(search_question)
            if search_question != question:
                meta["rewrote"] = search_question

            if not hits:
                st.warning(
                    "Nothing in the library matched. Upload more papers or "
                    "rephrase the question."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": "_No relevant passages found._",
                     "question": search_question}
                )
            else:
                placeholder = st.empty()
                chunks = [h.chunk for h in hits]
                answer = ""
                t0 = time.perf_counter()
                for token in stream_answer(
                    ollama_client(), chat_model, search_question, chunks,
                    history=history[:-1],
                ):
                    answer += token
                    placeholder.markdown(tidy_citations(answer) + "▌")
                t_generate = time.perf_counter() - t0
                placeholder.markdown(tidy_citations(answer))

                meta["t_generate"] = t_generate
                meta["t_condense"] = t_condense

                groundedness = None
                if verify:
                    with st.spinner("Checking groundedness..."):
                        groundedness = check_groundedness(
                            ollama_client(), chat_model, search_question, answer, chunks
                        )

                # sources under the streamed answer
                render_sources(hits, search_question)
                render_meta(meta, search_question, hits)
                if groundedness and not groundedness.get("grounded", True):
                    st.warning(
                        "⚠️ Groundedness check flagged possible unsupported claims:\n\n- "
                        + "\n- ".join(groundedness["issues"])
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": hits,
                        "meta": meta,
                        "question": search_question,
                        "groundedness": groundedness,
                    }
                )

# --------------------------------------------------------------------------
# tab: RAG lab
# --------------------------------------------------------------------------

with tab_lab:
    st.markdown(
        "Run **one question through every strategy** and compare. Watch "
        "*which passages* each strategy retrieves - dense finds meaning, "
        "bm25 finds exact words, hybrid fuses both."
    )
    lab_q = st.text_input(
        "Question to compare",
        value="Does testing improve long-term retention compared to restudy?",
    )
    lab_generate = st.checkbox("Also generate answers (slower)", value=True)
    run = st.button("▶ Run comparison", type="primary", disabled=not lib.chunks)

    if run and lab_q.strip():
        results: dict[str, tuple[list, str]] = {}
        cols = st.columns(3)
        for col, strat in zip(cols, ("dense", "bm25", "hybrid")):
            with col:
                st.subheader(strat)
                t0 = time.perf_counter()
                hits = retrieve(lib, lab_q.strip(), embed_query,
                                strategy=strat, top_k=top_k)
                t_search = time.perf_counter() - t0
                st.caption(f"retrieved in {t_search*1000:.0f} ms")
                for i, h in enumerate(hits, start=1):
                    st.markdown(
                        f"**[{i}]** p.{h.chunk.page} — {h.chunk.paper[:38]}"
                        f"{'…' if len(h.chunk.paper) > 38 else ''}"
                    )
                answer_text = ""
                if lab_generate:
                    t0 = time.perf_counter()
                    answer_text = "".join(
                        stream_answer(ollama_client(), chat_model, lab_q.strip(),
                                      [h.chunk for h in hits])
                    )
                    st.markdown(f"*{time.perf_counter() - t0:.0f}s*")
                    with st.expander("Answer"):
                        st.markdown(answer_text)
            results[strat] = (hits, answer_text)

        st.divider()
        st.subheader("Overlap analysis")
        ids = {s: {h.chunk.id for h in hits} for s, (hits, _) in results.items()}
        rows = []
        for a in ids:
            for b in ids:
                if a < b:
                    shared = len(ids[a] & ids[b])
                    rows.append({
                        "pair": f"{a} ∩ {b}",
                        "shared chunks": f"{shared}/{top_k}",
                        "overlap": f"{shared / top_k:.0%}",
                    })
        if rows:
            for r in rows:
                st.markdown(
                    f"- **{r['pair']}**: {r['shared chunks']} "
                    f"({r['overlap']} of top chunks identical)"
                )
        st.caption(
            "Low overlap means the strategies disagree - worth reading both "
            "answer sets. If hybrid's answers cite passages the single-signal "
            "runs missed, that's the fusion doing its job. (On a tiny library "
            "like the demo papers, expect near-total overlap: with 6 of ~14 "
            "chunks retrieved, everything collides. Differences appear as "
            "your library grows.)"
        )

# --------------------------------------------------------------------------
# tab: how it works
# --------------------------------------------------------------------------

with tab_docs:
    st.markdown(
        r"""
## The pipeline

```
 PDF ──► parse (pymupdf) ──► chunk per page ──► embed (nomic-embed-text)
                                                   │
                     ┌─────────────────────────────┘
                     ▼
   your question ──► embed query ──┐
                     │             ▼
                     ├──► cosine similarity  (dense ranking)
                     ├──► Okapi BM25         (keyword ranking)
                     │             │
                     ▼             ▼
                    Reciprocal Rank Fusion (RRF)
                                   │
                                   ▼
                    top-k chunks ──► LLM (llama3.2) ──► answer with [n] citations
```

Every arrow above is a file you can read in [`scibot/`](scibot/). No
framework, no hidden prompt templates - roughly 1,500 lines total.

## The techniques

**Chunking** ([chunking.py](scibot/chunking.py)) — pages become sliding
windows of text with a configurable overlap. Chunks never cross page
boundaries, so every citation points at an exact page. Chunk ids are
content hashes, so embeddings are cached: re-chunking only embeds what
actually changed.

**Dense retrieval** ([embed.py](scibot/embed.py), [store.py](scibot/store.py)) —
each chunk is embedded into a 768-dim vector; a question becomes a
vector too. Cosine similarity finds passages that *mean* the same thing
even when no words match. "does sleep help memory?" finds "spindle
activity correlates with overnight retention".

**BM25** ([bm25.py](scibot/bm25.py)) — the classic lexical ranker:
score documents by query-term frequency, saturation-capped and
length-normalised. It wins where embeddings blur: acronyms ("EEG"),
author names, exact figures ("0.81"), rare terms ("n-back").

**Hybrid + RRF** ([retrieval.py](scibot/retrieval.py)) — both rankings
are fused with Reciprocal Rank Fusion. Each chunk earns
`1/(k + rank)` from every ranking it appears in; ranks only, so no
score normalisation is needed between incomparable scales. A passage
found by *both* signals usually beats one found by either alone.

**Multi-query expansion** — a small LLM rewrites your question into
alternative phrasings ("does testing help memory?" → "does retrieval
practice enhance long-term retention?"), each searched separately and
fused. This closes the vocabulary gap between how you ask and how
papers write.

**Groundedness check** ([generation.py](scibot/generation.py)) — a
second LLM pass reads the answer against the excerpts and flags claims
they don't support. Best-effort on small models, but a useful honesty
alarm.

## Things to try

1. Ask something with **exact terminology** ("What was Cohen's d?") -
   watch bm25 beat dense.
2. Ask something **paraphrased** ("Can napping protect new
   memories?") - watch dense beat bm25.
3. Flip the strategy radio and re-ask; compare in the **RAG Lab**.
4. Turn on **multi-query** for questions you'd phrase casually.
5. Shrink chunks to 400 chars and see whether precision improves or
   context breaks.

## Run it yourself

```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text
streamlit run app.py
```

That's the whole install. No API keys, no accounts, no cloud - your
papers never leave your machine.
""",
        unsafe_allow_html=True,
    )
