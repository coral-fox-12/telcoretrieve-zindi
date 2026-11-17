"""Retrieval for local runs: BM25 over the mini index (stdlib only).

The real pipeline uses the fixed OTel-Embedding-0.6B embedder plus the
OTel reranker over the 4.6M-chunk index; that runs on Kaggle GPU
(RUNBOOK.md). Locally we use BM25 so the loop is runnable on CPU with
zero model downloads — it measures the pipeline mechanics and gives a
honest lexical floor. The interface (search -> list[RetrievedChunk])
is the same shape the dense path will use, so swapping in embeddings
later is a one-module change.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .prompt import Passage

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievedChunk:
    id: str
    spec: str
    release: str
    text: str
    score: float
    # Parent 512-token chunk id, set only for sub-512 indexes (e.g. the
    # 256-token index): lets evidence checks cite 512-token chunk ids.
    parent: str = ""

    def as_passage(self) -> Passage:
        return Passage(text=self.text, spec=self.spec, release=self.release)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Classic BM25 (k1=1.5, b=0.75) over a JSONL chunk file."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c["text"]) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n = len(chunks)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
        return cls(chunks)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        qterms = _tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tokens in enumerate(self.doc_tokens):
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            dl = self.doc_len[i]
            for term in set(qterms):
                if term not in tf:
                    continue
                f = tf[term]
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                score += idf * f * (self.k1 + 1) / denom
            scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, i in scored[:top_k]:
            c = self.chunks[i]
            out.append(
                RetrievedChunk(
                    id=c["id"],
                    spec=c.get("spec", ""),
                    release=c.get("release", ""),
                    text=c["text"],
                    score=score,
                    parent=c.get("parent", ""),
                )
            )
        return out
