from __future__ import annotations

import hashlib

from rag_agent.domain.models import Chunk, Document


def make_chunk_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:16]


class RecursiveChunker:
    """Paragraph -> line -> sentence -> word aware splitting with overlap."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, text in enumerate(self._splitter.split_text(doc.text)):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.source, text),
                    text=text,
                    source=doc.source,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
        return chunks
