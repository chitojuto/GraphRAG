from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, docs: list[dict], text_key: str = "text", k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.text_key = text_key
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc[text_key]) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq = Counter()

        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, idx: int) -> float:
        score = 0.0
        q_terms = tokenize(query)
        dl = self.doc_lens[idx] or 1
        tf = self.term_freqs[idx]

        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf(term) * freq * (self.k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        scored = [(self.score(query, i), self.docs[i]) for i in range(len(self.docs))]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

