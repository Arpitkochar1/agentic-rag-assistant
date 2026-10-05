"""RAGAS wrapper (v0.2 / v0.3 API). Imported lazily so the runtime image doesn't need ragas."""
from __future__ import annotations

from typing import Any, Sequence

from rag_agent.domain.interfaces import Embedder
from rag_agent.embeddings.adapters import LangChainEmbeddingsAdapter


class RagasEvaluator:
    def __init__(self, judge_llm: Any, embedder: Embedder) -> None:
        self._judge, self._embedder = judge_llm, embedder

    def evaluate(self, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        """rows: [{user_input, response, retrieved_contexts, reference}, ...]"""
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        result = evaluate(
            EvaluationDataset.from_list(list(rows)),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(self._judge.raw),
            embeddings=LangchainEmbeddingsWrapper(LangChainEmbeddingsAdapter(self._embedder)),
        )
        df = result.to_pandas()
        return {c: round(float(df[c].mean()), 4) for c in df.select_dtypes("number").columns}
