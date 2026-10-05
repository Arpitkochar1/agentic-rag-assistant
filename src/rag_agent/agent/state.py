import operator
from typing import Annotated, TypedDict

from rag_agent.domain.models import ChatMessage, Citation, RetrievedChunk, Route


class AgentState(TypedDict, total=False):
    question: str
    history: list[ChatMessage]
    route: Route
    contexts: list[RetrievedChunk]
    tool_output: str | None
    answer: str
    citations: list[Citation]
    blocked: bool
    notes: Annotated[list[str], operator.add]  # guardrail / fallback notes accumulate
