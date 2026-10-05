from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from rag_agent.domain.interfaces import Chunker, DocumentLoader, Embedder, KeywordIndex, VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files: int = 0
    chunks_added: int = 0
    skipped: list[str] = field(default_factory=list)


class IngestionPipeline:
    """load -> chunk -> embed -> index (vector + keyword). All collaborators injected."""

    def __init__(
        self,
        loaders: Sequence[DocumentLoader],
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        batch_size: int = 64,
    ) -> None:
        self._loaders, self._chunker, self._embedder = loaders, chunker, embedder
        self._vectors, self._keywords, self._batch = vector_store, keyword_index, batch_size

    def ingest_paths(self, paths: Iterable[Path]) -> IngestReport:
        report = IngestReport()
        for path in paths:
            loader = next((l for l in self._loaders if l.supports(path)), None)
            if loader is None:
                report.skipped.append(path.name)
                continue
            try:
                docs = loader.load(path)
            except Exception as exc:  # corrupt PDF etc.
                log.warning("Failed to load %s: %s", path, exc)
                report.skipped.append(path.name)
                continue
            chunks = [c for d in docs for c in self._chunker.split(d)]
            if not chunks:
                report.skipped.append(path.name)
                continue
            for i in range(0, len(chunks), self._batch):
                batch = chunks[i : i + self._batch]
                emb = self._embedder.embed_documents([c.text for c in batch])
                report.chunks_added += self._vectors.add(batch, emb)
            self._keywords.add(chunks)
            report.files += 1
        return report

    def ingest_directory(self, directory: Path) -> IngestReport:
        directory = Path(directory)
        if not directory.exists():
            return IngestReport()
        return self.ingest_paths(sorted(p for p in directory.rglob("*") if p.is_file()))

    def reset(self) -> None:
        self._vectors.reset()
        self._keywords.reset()
