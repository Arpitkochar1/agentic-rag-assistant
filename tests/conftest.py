import hashlib
import re
from typing import Iterator, Sequence

import numpy as np
import pytest

from rag_agent.domain.models import ChatMessage


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    dim = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """Routes by inspecting the system prompt, like a tiny scripted model."""

    def __init__(self, route="documents", answer="Hybrid retrieval merges BM25 and dense search [1]."):
        self.route, self.answer, self.calls = route, answer, []

    def invoke(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(messages)
        system = messages[0].content
        if "query router" in system:
            return self.route
        if "maths question" in system:
            return "15/100*2480"
        return self.answer

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        self.calls.append(messages)
        for tok in self.answer.split(" "):
            yield tok + " "


@pytest.fixture
def embedder():
    return FakeEmbedder()
