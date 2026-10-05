from __future__ import annotations

import re
from typing import Sequence

from rag_agent.domain.models import Citation, RetrievedChunk

_CITE = re.compile(r"\[(\d+)\]")


def build_citations(answer: str, contexts: Sequence[RetrievedChunk]) -> list[Citation]:
    """Map [n] markers in the answer back to the passages they refer to."""
    if not contexts:
        return []
    referenced = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(contexts)})
    out: list[Citation] = []
    for n in referenced:
        chunk = contexts[n - 1].chunk
        out.append(
            Citation(
                index=n,
                source=chunk.source,
                snippet=" ".join(chunk.text.split())[:240],
                url=chunk.metadata.get("url") or None,
            )
        )
    return out
