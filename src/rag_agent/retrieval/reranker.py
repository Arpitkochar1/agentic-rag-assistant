from __future__ import annotations

from typing import Sequence

from rag_agent.domain.models import RetrievedChunk


class FastEmbedReranker:
    """Cross-encoder reranker (ONNX) - scores (query, passage) pairs jointly."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, items: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        items = list(items)
        if not items:
            return []
        scores = list(self._model.rerank(query, [i.chunk.text for i in items]))
        ranked = sorted(zip(items, scores), key=lambda p: p[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=i.chunk, score=float(s)) for i, s in ranked]
