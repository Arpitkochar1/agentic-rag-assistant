"""Abstractions (ports). High-level code depends on these, never on concrete classes (DIP)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np

from .models import (
    Chunk,
    ChatMessage,
    Document,
    RetrievedChunk,
    Route,
    ToolResult,
)


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int: ...
    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class KeywordIndex(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> int: ...
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, items: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class LLMClient(Protocol):
    def invoke(self, messages: Sequence[ChatMessage]) -> str: ...
    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]: ...


class Tool(Protocol):
    name: str
    description: str

    def run(self, question: str) -> ToolResult: ...


class Router(Protocol):
    def route(self, question: str, history: Sequence[ChatMessage] | None = None) -> Route: ...


class DocumentLoader(Protocol):
    def supports(self, path: Path) -> bool: ...
    def load(self, path: Path) -> list[Document]: ...


class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...
