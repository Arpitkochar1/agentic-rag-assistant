from __future__ import annotations

from pydantic import BaseModel, Field

from rag_agent.domain.models import ChatMessage


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    history: list[ChatMessage] = Field(default_factory=list)
    strategy: str | None = Field(default=None, pattern="^(vector|hybrid|hybrid_rerank)$")


class IngestResponse(BaseModel):
    files: int
    chunks_added: int
    skipped: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    llm_provider: str
    llm_model: str
    retrieval_strategy: str
