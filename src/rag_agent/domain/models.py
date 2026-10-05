"""Pure domain models. No framework or IO imports allowed here."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    DOCUMENTS = "documents"
    WEB = "web"
    CALCULATOR = "calculator"
    DIRECT = "direct"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Document(BaseModel):
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = 0.0


class ToolResult(BaseModel):
    text: str = ""
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    ok: bool = True


class Citation(BaseModel):
    index: int
    source: str
    snippet: str
    url: str | None = None


class AgentAnswer(BaseModel):
    question: str
    answer: str
    route: Route
    citations: list[Citation] = Field(default_factory=list)
    blocked: bool = False
    guardrail_notes: list[str] = Field(default_factory=list)
    tool_output: str | None = None


class AgentEvent(BaseModel):
    type: Literal["route", "token", "final", "error"]
    data: Any = None
