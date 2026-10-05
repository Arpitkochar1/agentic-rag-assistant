"""Retrieval strategies. Each implements Retriever; RerankingRetriever is a Decorator."""
from __future__ import annotations

from typing import Sequence

from rag_agent.domain.interfaces import Embedder, KeywordIndex, Reranker, Retriever, VectorStore
from rag_agent.domain.models import RetrievedChunk


class VectorRetriever:
    """Baseline: dense semantic search only."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder, self._store = embedder, store

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._store.search(self._embedder.embed_query(query), k)


class KeywordRetriever:
    def __init__(self, index: KeywordIndex) -> None:
        self._index = index

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._index.search(query, k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)). Rank-based, so no score normalisation needed."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    chunks = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.id
            chunks[cid] = item.chunk
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [RetrievedChunk(chunk=chunks[cid], score=s) for cid, s in ordered]


class HybridRetriever:
    """Semantic + keyword retrieval fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        semantic: Retriever,
        keyword: Retriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
        weights: Sequence[float] = (1.0, 1.0),
    ) -> None:
        self._semantic, self._keyword = semantic, keyword
        self._candidate_k, self._rrf_k, self._weights = candidate_k, rrf_k, weights

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        n = max(k, self._candidate_k)
        fused = reciprocal_rank_fusion(
            [self._semantic.retrieve(query, n), self._keyword.retrieve(query, n)],
            k=self._rrf_k,
            weights=self._weights,
        )
        return fused[:k]


class RerankingRetriever:
    """Decorator: over-fetch candidates from any Retriever, then cross-encoder rerank."""

    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 20) -> None:
        self._base, self._reranker, self._candidate_k = base, reranker, candidate_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        candidates = self._base.retrieve(query, max(k, self._candidate_k))
        if not candidates:
            return []
        return self._reranker.rerank(query, candidates, k)
