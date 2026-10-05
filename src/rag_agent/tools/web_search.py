from __future__ import annotations

from typing import Any, Callable

from rag_agent.domain.models import Chunk, RetrievedChunk, ToolResult

Searcher = Callable[[str, int], list[dict[str, Any]]]


def _ddg_search(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore
    return list(DDGS().text(query, max_results=max_results))


class WebSearchTool:
    name = "web_search"
    description = "Search the public web for current or external information."

    def __init__(self, max_results: int = 5, searcher: Searcher = _ddg_search) -> None:
        self._max, self._search = max_results, searcher

    def run(self, question: str) -> ToolResult:
        try:
            results = self._search(question, self._max)
        except Exception as exc:
            return ToolResult(ok=False, text=f"Web search failed: {exc}")
        contexts = []
        for i, r in enumerate(results):
            url = r.get("href") or r.get("url") or ""
            body = (r.get("body") or r.get("snippet") or "").strip()
            if not body:
                continue
            contexts.append(
                RetrievedChunk(
                    chunk=Chunk(id=url or f"web-{i}", text=body, source=r.get("title") or url or "web", metadata={"url": url}),
                    score=1.0 / (i + 1),
                )
            )
        return ToolResult(contexts=contexts)
