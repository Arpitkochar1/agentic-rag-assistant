"""Adapter: LangChain chat model -> our LLMClient port."""
from __future__ import annotations

from typing import Any, Iterator, Sequence

from rag_agent.domain.models import ChatMessage


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class LangChainLLM:
    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def raw(self) -> Any:
        """Underlying LangChain model (needed by RAGAS)."""
        return self._model

    @staticmethod
    def _convert(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        return [(m.role, m.content) for m in messages]

    def invoke(self, messages: Sequence[ChatMessage]) -> str:
        return content_to_text(self._model.invoke(self._convert(messages)).content)

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        for chunk in self._model.stream(self._convert(messages)):
            text = content_to_text(chunk.content)
            if text:
                yield text
