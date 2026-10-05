from __future__ import annotations

import logging
import re
from typing import Sequence

from rag_agent.agent.prompts import ROUTE_DESCRIPTIONS, ROUTER_SYSTEM_PROMPT
from rag_agent.domain.interfaces import LLMClient
from rag_agent.domain.models import ChatMessage, Route

log = logging.getLogger(__name__)
_ROUTE_RE = re.compile(r"\b(documents|web|calculator|direct)\b")


class LLMRouter:
    """Single-label query classifier. Falls back to document search if the LLM misbehaves."""

    def __init__(self, llm: LLMClient, available: Sequence[Route]) -> None:
        self._llm = llm
        self._available = set(available) | {Route.DIRECT}
        self._default = Route.DOCUMENTS if Route.DOCUMENTS in self._available else Route.DIRECT
        descriptions = "\n".join(ROUTE_DESCRIPTIONS[r.value] for r in Route if r in self._available)
        self._system = ROUTER_SYSTEM_PROMPT.format(route_descriptions=descriptions)

    def route(self, question: str, history: Sequence[ChatMessage] | None = None) -> Route:
        recent = "\n".join(f"{m.role}: {m.content[:200]}" for m in (history or [])[-4:])
        user = (f"Conversation so far:\n{recent}\n\n" if recent else "") + f"Latest question: {question}"
        try:
            out = self._llm.invoke(
                [ChatMessage(role="system", content=self._system), ChatMessage(role="user", content=user)]
            )
        except Exception as exc:
            log.warning("Router LLM failed (%s); using default route", exc)
            return self._default
        m = _ROUTE_RE.search(out.lower())
        if not m:
            return self._default
        route = Route(m.group(1))
        return route if route in self._available else self._default
