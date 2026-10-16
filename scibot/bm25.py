"""A small, readable Okapi BM25.

BM25 is the classic lexical ranking function: score documents by query
term frequency, saturated (so a term appearing 50x isn't 50x as
important) and length-normalised. It catches exact matches - acronyms,
author names, "n-back" - that embedding models blur away.

We implement it here (~40 lines) instead of pulling in a dependency,
because this is a playground: you should be able to read everything
that happens to your query.

    score(D, Q) = sum over unique q in Q of
        IDF(q) * tf(q, D) / (tf(q, D) + k1 * (1 - b + b * |D| / avgdl))

IDF uses the non-negative Lucene variant ln(1 + (N - df + 0.5)/(df + 0.5)).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short: dropping too many words hurts phrase-y academic queries.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that "
    "the to was were what which with this these those".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        df = Counter()
        for doc in self.doc_freqs:
            df.update(doc.keys())
        self.idf = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        """Return the top-k (doc_index, score) pairs, best first."""
        if self.n_docs == 0 or not query:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(query):  # unique, order-preserving
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_lens[doc_idx] / self.avgdl
                )
                scores[doc_idx] += idf * tf / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]
