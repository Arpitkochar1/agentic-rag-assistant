from __future__ import annotations

from typing import Sequence

from rag_agent.domain.interfaces import Retriever
from rag_agent.evaluation.dataset import EvalSample


def hit_rate_and_mrr(retriever: Retriever, samples: Sequence[EvalSample], k: int) -> dict[str, float]:
    """Cheap, LLM-free retrieval metrics using the labelled source file."""
    labelled = [s for s in samples if s.source]
    if not labelled:
        return {}
    hits, rr = 0, 0.0
    for s in labelled:
        sources = [r.chunk.source for r in retriever.retrieve(s.question, k)]
        if s.source in sources:
            hits += 1
            rr += 1.0 / (sources.index(s.source) + 1)
    n = len(labelled)
    return {f"hit_rate@{k}": hits / n, f"mrr@{k}": rr / n}
