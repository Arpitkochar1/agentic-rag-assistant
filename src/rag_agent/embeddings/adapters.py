"""Adapt our Embedder port to LangChain's Embeddings interface (needed by RAGAS)."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_agent.domain.interfaces import Embedder


class LangChainEmbeddingsAdapter(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text).tolist()
