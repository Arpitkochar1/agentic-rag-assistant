"""Application service: the only thing the API / UI / eval talk to."""
from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from rag_agent.domain.models import AgentAnswer, AgentEvent, ChatMessage, Route

log = logging.getLogger(__name__)


class AgentService:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @staticmethod
    def _initial(question: str, history: Sequence[ChatMessage] | None) -> dict:
        return {"question": question, "history": list(history or []), "notes": [], "contexts": []}

    @staticmethod
    def _to_answer(question: str, state: dict) -> AgentAnswer:
        return AgentAnswer(
            question=question,
            answer=state.get("answer", ""),
            route=state.get("route", Route.DIRECT),
            citations=state.get("citations", []),
            blocked=bool(state.get("blocked")),
            guardrail_notes=state.get("notes", []),
            tool_output=state.get("tool_output"),
        )

    def ask(self, question: str, history: Sequence[ChatMessage] | None = None) -> AgentAnswer:
        state = self._graph.invoke(self._initial(question, history))
        return self._to_answer(question, state)

    def stream(self, question: str, history: Sequence[ChatMessage] | None = None) -> Iterator[AgentEvent]:
        """Yields route / token events as they happen, then a single 'final' event.
        The final answer may differ from the streamed text if an output guardrail edited/blocked it,
        so clients should replace what they rendered with `final.data["answer"]`."""
        final_state: dict = {}
        try:
            for mode, payload in self._graph.stream(self._initial(question, history), stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield AgentEvent(type=payload["type"], data=payload["data"])
                else:
                    final_state = payload
        except Exception as exc:
            log.exception("Agent stream failed")
            yield AgentEvent(type="error", data=f"{type(exc).__name__}: {exc}")
            return
        yield AgentEvent(type="final", data=self._to_answer(question, final_state).model_dump(mode="json"))
