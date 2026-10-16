"""Answer generation: prompts, streaming, and self-checks.

The contract with the LLM is strict: answer ONLY from the numbered
excerpts and cite every claim as [n]. The UI turns those markers into
links back to the exact chunk and page.

Also here:
  condense_question   - rewrite a follow-up ("what about the limits?")
                        into a standalone question using chat history
  check_groundedness  - a second LLM pass that flags claims in the
                        answer not supported by the excerpts
"""

from __future__ import annotations

import json
import re
from typing import Iterator

import ollama

from ._compat import field as _field
from .chunking import Chunk

SYSTEM_PROMPT = """\
You are SciBot, a careful research assistant answering questions about \
scientific papers.

Rules:
- Answer ONLY from the numbered excerpts provided.
- Cite the excerpt number after every claim, like [1] or [2][3].
- If the excerpts don't contain enough information, say so plainly and \
name what a relevant paper would need to establish.
- Quote key phrases when they matter; do not invent numbers or findings.
- Be concise and direct."""

CITATION_RE = re.compile(r"\[(\d+)\]")
# small models sometimes emit "[3:3.1]" or "[3, p.4]" - normalise for display
MESSY_CITATION_RE = re.compile(r"\[(\d+)[^\]]*\]")


def tidy_citations(text: str) -> str:
    """Collapse malformed citation markers like [3:3.1] down to [3]."""
    return MESSY_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", text)


def build_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks as numbered excerpts for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({c.paper}, p. {c.page})\n{c.text}")
    return "\n\n".join(blocks)


def history_to_messages(history: list[dict], max_turns: int = 4) -> list[dict]:
    """Trim chat history to the last few turns for the prompt."""
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    return [
        {"role": m["role"], "content": m["content"]} for m in turns[-max_turns:]
    ]


def stream_answer(
    client: ollama.Client,
    model: str,
    question: str,
    chunks: list[Chunk],
    history: list[dict] | None = None,
    temperature: float = 0.2,
    show_prompt: tuple[str, str] | None = None,
) -> Iterator[str]:
    """Stream an answer token by token.

    `show_prompt` optionally receives (system, user) for transparency
    display in the UI.
    """
    user_content = (
        f"## Excerpts from the library\n\n{build_context(chunks)}\n\n"
        f"## Question\n\n{question}"
    )
    if show_prompt is not None:
        show_prompt(SYSTEM_PROMPT, user_content)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_to_messages(history or [])
        + [{"role": "user", "content": user_content}]
    )
    stream = client.chat(
        model=model, messages=messages, stream=True,
        options={"temperature": temperature},
    )
    for part in stream:
        yield _field(_field(part, "message"), "content", "")


def cited_indices(text: str, n_chunks: int) -> set[int]:
    """Which excerpt numbers does an answer cite? (1-based, deduped)"""
    return {m for m in (int(x) for x in CITATION_RE.findall(text)) if 1 <= m <= n_chunks}


def condense_question(
    client: ollama.Client,
    model: str,
    follow_up: str,
    history: list[dict],
    max_turns: int = 4,
) -> str:
    """Turn a conversational follow-up into a standalone search question.

    "and does it hold for kids?" -> "Does the testing effect hold for
    children?" Retrieval needs the nouns; the chat doesn't.
    """
    turns = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in history_to_messages(history, max_turns=max_turns)
    )
    try:
        resp = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's follow-up question as a single "
                        "self-contained search question. Resolve pronouns and "
                        "ellipsis using the chat history. Reply with ONLY the "
                        "rewritten question, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Chat history:\n{turns}\n\nFollow-up: {follow_up}",
                },
            ],
            options={"temperature": 0.0},
        )
        rewritten = _field(_field(resp, "message"), "content", "").strip().strip('"')
        return rewritten or follow_up
    except Exception:
        return follow_up


def check_groundedness(
    client: ollama.Client,
    model: str,
    question: str,
    answer: str,
    chunks: list[Chunk],
) -> dict:
    """Second-pass check: is every claim in the answer excerpt-supported?

    Returns {"grounded": bool, "issues": [str]}. Best-effort on small
    models - treat as advice, not proof.
    """
    try:
        resp = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You audit answers for grounding. Compare the ANSWER "
                        "against the EXCERPTS. List every claim in the answer "
                        "that is NOT stated or directly implied by the excerpts. "
                        'Respond with ONLY JSON: {"grounded": true/false, '
                        '"issues": ["unsupported claim 1", ...]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## EXCERPTS\n\n{build_context(chunks)}\n\n"
                        f"## QUESTION\n{question}\n\n"
                        f"## ANSWER\n{answer}"
                    ),
                },
            ],
            options={"temperature": 0.0},
            format="json",
        )
        data = json.loads(_field(_field(resp, "message"), "content", "{}"))
        issues = [str(i) for i in data.get("issues", []) if str(i).strip()]
        grounded = bool(data.get("grounded", True)) and not issues
        return {"grounded": grounded, "issues": issues}
    except Exception as exc:  # audit is optional; never break the app
        return {"grounded": True, "issues": [], "error": str(exc)}
