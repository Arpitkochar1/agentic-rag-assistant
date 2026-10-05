"""Answer generation shared by the agent and the evaluation harness (DRY)."""
from __future__ import annotations

from typing import Iterator, Sequence

from rag_agent.agent.prompts import DIRECT_SYSTEM_PROMPT, GROUNDED_SYSTEM_PROMPT
from rag_agent.domain.interfaces import LLMClient
from rag_agent.domain.models import ChatMessage, RetrievedChunk

_MAX_CHARS_PER_PASSAGE = 1500


def format_context(contexts: Sequence[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] (source: {c.chunk.source})\n{c.chunk.text[:_MAX_CHARS_PER_PASSAGE]}"
        for i, c in enumerate(contexts, start=1)
    ]
    return "\n\n".join(blocks)


class AnswerGenerator:
    def __init__(self, llm: LLMClient, max_history: int = 6) -> None:
        self._llm, self._max_history = llm, max_history

    def build_messages(
        self,
        question: str,
        contexts: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
        tool_output: str | None = None,
        grounded: bool = True,
    ) -> list[ChatMessage]:
        system = GROUNDED_SYSTEM_PROMPT if grounded else DIRECT_SYSTEM_PROMPT
        msgs = [ChatMessage(role="system", content=system)]
        msgs += [m for m in (history or [])[-self._max_history :] if m.role in ("user", "assistant")]
        if grounded:
            parts = [f"<context>\n{format_context(contexts) or '(no passages found)'}\n</context>"]
            if tool_output:
                parts.append(f"<tool_result>\n{tool_output}\n</tool_result>")
            parts.append(f"Question: {question}")
            user = "\n\n".join(parts)
        else:
            user = question
        msgs.append(ChatMessage(role="user", content=user))
        return msgs

    def stream(self, question, contexts=(), history=None, tool_output=None, grounded=True) -> Iterator[str]:
        yield from self._llm.stream(self.build_messages(question, contexts, history, tool_output, grounded))

    def generate(self, question, contexts=(), history=None, tool_output=None, grounded=True) -> str:
        return self._llm.invoke(self.build_messages(question, contexts, history, tool_output, grounded)).strip()
