"""Keyword (BM25) index, persisted as JSON and rebuilt in-memory on load."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from rag_agent.domain.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25KeywordIndex:
    def __init__(self, index_dir: Path) -> None:
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "bm25_chunks.json"
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None
        if self._path.exists():
            for d in json.loads(self._path.read_text(encoding="utf-8")):
                c = Chunk(**d)
                self._chunks[c.id] = c
            self._rebuild()

    def _rebuild(self) -> None:
        self._order = list(self._chunks)
        corpus = [tokenize(self._chunks[i].text) for i in self._order]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([c.model_dump() for c in self._chunks.values()]), encoding="utf-8"
        )

    def add(self, chunks: Sequence[Chunk]) -> int:
        new = [c for c in chunks if c.id not in self._chunks]
        for c in new:
            self._chunks[c.id] = c
        if new:
            self._rebuild()
            self._save()
        return len(new)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self._chunks[self._order[i]], score=float(scores[i]))
            for i in top
            if scores[i] > 0
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks, self._order, self._bm25 = {}, [], None
        self._path.unlink(missing_ok=True)
