"""Strategy factory so new retrieval strategies plug in without touching callers."""
from __future__ import annotations

from typing import Callable

from rag_agent.domain.interfaces import Embedder, KeywordIndex, Reranker, Retriever, VectorStore
from rag_agent.retrieval.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    RerankingRetriever,
    VectorRetriever,
)


def build_retriever(
    strategy: str,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    keyword_index: KeywordIndex,
    reranker_provider: Callable[[], Reranker],
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> Retriever:
    semantic = VectorRetriever(embedder, vector_store)
    if strategy == "vector":
        return semantic
    hybrid = HybridRetriever(semantic, KeywordRetriever(keyword_index), candidate_k, rrf_k)
    if strategy == "hybrid":
        return hybrid
    if strategy == "hybrid_rerank":
        return RerankingRetriever(hybrid, reranker_provider(), candidate_k)
    raise ValueError(f"Unknown retrieval strategy '{strategy}'")
