"""LangGraph nodes. Every collaborator is injected, so nodes are unit-testable with fakes."""
from __future__ import annotations

import logging
from typing import Any, Mapping

from rag_agent.agent.citations import build_citations
from rag_agent.agent.generation import AnswerGenerator
from rag_agent.agent.state import AgentState
from rag_agent.domain.interfaces import Router, Tool
from rag_agent.domain.models import Route, ToolResult
from rag_agent.guardrails.base import GuardrailChain, GuardrailContext
from rag_agent.guardrails.output_guardrails import ContextSanitizer

log = logging.getLogger(__name__)

INPUT_BLOCKED_MESSAGE = "I can't help with that request. Please rephrase your question."
OUTPUT_BLOCKED_MESSAGE = (
    "I couldn't produce a sufficiently grounded answer from the available sources. "
    "Try rephrasing, or add more documents."
)
GENERATION_ERROR_MESSAGE = "Sorry, something went wrong while generating the answer. Please try again."


def _emit(event: dict[str, Any]) -> None:
    """Push a custom stream event (no-op when not streaming)."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:
        pass


class AgentNodes:
    def __init__(
        self,
        *,
        input_guard: GuardrailChain,
        output_guard: GuardrailChain,
        router: Router,
        tools: Mapping[Route, Tool],
        generator: AnswerGenerator,
        sanitizer: ContextSanitizer | None = None,
        fallbacks: Mapping[Route, Route] | None = None,
    ) -> None:
        self._input_guard, self._output_guard = input_guard, output_guard
        self._router, self._tools, self._generator = router, dict(tools), generator
        self._sanitizer = sanitizer or ContextSanitizer()
        self._fallbacks = dict(fallbacks or {Route.DOCUMENTS: Route.WEB})

    # ---- nodes -------------------------------------------------------
    def guard_input(self, state: AgentState) -> dict:
        question = state["question"]
        result = self._input_guard.run(question, GuardrailContext(question=question))
        if result.blocked:
            return {"blocked": True, "route": Route.DIRECT, "answer": INPUT_BLOCKED_MESSAGE, "notes": result.reasons}
        return {"question": result.text, "notes": result.reasons}

    def route_query(self, state: AgentState) -> dict:
        route = self._router.route(state["question"], state.get("history", []))
        _emit({"type": "route", "data": route.value})
        return {"route": route}

    def use_tool(self, state: AgentState) -> dict:
        route, question = state["route"], state["question"]
        notes: list[str] = []
        result = self._run_tool(route, question)
        fallback = self._fallbacks.get(route)
        if not (result.contexts or result.text) and fallback in self._tools:
            notes.append(f"fallback: {route.value} -> {fallback.value} (nothing found)")
            route = fallback
            _emit({"type": "route", "data": route.value})
            result = self._run_tool(route, question)
        contexts, changed = self._sanitizer.sanitize(result.contexts)
        if changed:
            notes.append("context_sanitizer: neutralised suspected instructions in retrieved content")
        return {"route": route, "contexts": contexts, "tool_output": result.text or None, "notes": notes}

    def generate_answer(self, state: AgentState) -> dict:
        grounded = state["route"] is not Route.DIRECT
        parts: list[str] = []
        try:
            for token in self._generator.stream(
                state["question"],
                state.get("contexts", []),
                state.get("history"),
                state.get("tool_output"),
                grounded=grounded,
            ):
                parts.append(token)
                _emit({"type": "token", "data": token})
        except Exception as exc:
            log.exception("Generation failed")
            return {"answer": GENERATION_ERROR_MESSAGE, "notes": [f"generation_error: {type(exc).__name__}"]}
        return {"answer": "".join(parts).strip()}

    def guard_output(self, state: AgentState) -> dict:
        contexts = state.get("contexts", [])
        ctx = GuardrailContext(
            question=state["question"],
            route=state["route"],
            contexts=tuple(c.chunk.text for c in contexts),
        )
        result = self._output_guard.run(state["answer"], ctx)
        if result.blocked:
            return {"answer": OUTPUT_BLOCKED_MESSAGE, "blocked": True, "notes": result.reasons}
        return {"answer": result.text, "notes": result.reasons}

    def add_citations(self, state: AgentState) -> dict:
        if state.get("blocked"):
            return {"citations": []}
        return {"citations": build_citations(state["answer"], state.get("contexts", []))}

    # ---- edges -------------------------------------------------------
    def after_input(self, state: AgentState) -> str:
        return "end" if state.get("blocked") else "route_query"

    def after_route(self, state: AgentState) -> str:
        return "use_tool" if state["route"] in self._tools else "generate_answer"

    # ---- helpers -----------------------------------------------------
    def _run_tool(self, route: Route, question: str) -> ToolResult:
        try:
            return self._tools[route].run(question)
        except Exception as exc:
            log.exception("Tool %s failed", route)
            return ToolResult(ok=False, text="")
