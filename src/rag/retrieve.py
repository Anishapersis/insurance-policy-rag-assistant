"""Sparse, dense and hybrid retrievers.

* BM25: exact keyword matching (deductible amounts, form names like OCF-1).
* Dense: OpenAI embeddings when OPENAI_API_KEY is set; otherwise a local
  TF-IDF + LSA (truncated SVD) semantic space so the project runs offline.
* Hybrid: Reciprocal Rank Fusion of the two rankings.
"""
from __future__ import annotations

import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .ingest import Chunk

STOP = set("a an the of to and or in on for is are be by with if your you it this that as at from".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$%]+(?:-[a-z0-9]+)?", text.lower()) if t not in STOP]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], backend: str | None = None, dims: int = 64):
        self.chunks = chunks
        self.backend = backend or ("openai" if os.getenv("OPENAI_API_KEY") else "lsa")
        texts = [c.text for c in chunks]
        if self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
            self.matrix = self._embed(texts)
        else:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            X = self.tfidf.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1, len(texts) - 1), random_state=0)
            self.matrix = normalize(self.svd.fit_transform(X))

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return normalize(np.array([d.embedding for d in resp.data]))

    def _query_vec(self, query: str) -> np.ndarray:
        if self.backend == "openai":
            return self._embed([query])[0]
        return normalize(self.svd.transform(self.tfidf.transform([query])))[0]

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        sims = self.matrix @ self._query_vec(query)
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across retrievers."""
    name = "hybrid"

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60, depth: int = 20, **dense_kw):
        self.retrievers = [BM25Retriever(chunks), DenseRetriever(chunks, **dense_kw)]
        self.rrf_k, self.depth = rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        lookup: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (chunk, _) in enumerate(r.search(query, self.depth), start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                lookup[chunk.id] = chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(lookup[i], fused[i]) for i in best]
