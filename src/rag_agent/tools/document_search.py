from __future__ import annotations

from rag_agent.domain.interfaces import Retriever
from rag_agent.domain.models import ToolResult


class DocumentSearchTool:
    name = "document_search"
    description = "Search the user's indexed knowledge base (uploaded/ingested documents)."

    def __init__(self, retriever: Retriever, top_k: int = 5) -> None:
        self._retriever, self._k = retriever, top_k

    def run(self, question: str) -> ToolResult:
        return ToolResult(contexts=self._retriever.retrieve(question, self._k))
