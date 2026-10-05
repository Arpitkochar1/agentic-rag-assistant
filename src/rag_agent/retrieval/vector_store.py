"""Vector store adapters (FAISS default, Chroma optional). Both implement VectorStore."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rag_agent.domain.models import Chunk, RetrievedChunk
from rag_agent.retrieval.math_utils import l2_normalize


class FaissVectorStore:
    """Exact cosine search (inner product on normalised vectors), persisted to disk."""

    def __init__(self, index_dir: Path) -> None:
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "faiss.index"
        self._meta_path = self._dir / "faiss_chunks.json"
        self._index: Any = None
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            import faiss

            self._index = faiss.read_index(str(self._index_path))
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**d) for d in raw]

    def _save(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps([c.model_dump() for c in self._chunks]), encoding="utf-8"
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        import faiss

        chunks = list(chunks)
        known = {c.id for c in self._chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in known]
        if not keep:
            return 0
        vecs = l2_normalize(np.asarray(embeddings, dtype="float32")[keep])
        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self._chunks.extend(chunks[i] for i in keep)
        self._save()
        return len(keep)

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []
        q = l2_normalize(np.asarray(query_embedding, dtype="float32").reshape(1, -1))
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], idx[0])
            if i >= 0
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._index, self._chunks = None, []
        for p in (self._index_path, self._meta_path):
            p.unlink(missing_ok=True)


def _flat_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {"source": chunk.source}
    for k, v in chunk.metadata.items():
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
    return meta


class ChromaVectorStore:
    def __init__(self, index_dir: Path, collection_name: str = "documents") -> None:
        import chromadb

        path = Path(index_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self._name = collection_name
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._open()

    def _open(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        existing = set(self._collection.get(ids=[c.id for c in chunks])["ids"])
        new = [(c, e) for c, e in zip(chunks, embeddings) if c.id not in existing]
        if not new:
            return 0
        self._collection.add(
            ids=[c.id for c, _ in new],
            documents=[c.text for c, _ in new],
            embeddings=[np.asarray(e, dtype="float32").tolist() for _, e in new],
            metadatas=[_flat_metadata(c) for c, _ in new],
        )
        return len(new)

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        n = self._collection.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype="float32").tolist()],
            n_results=min(k, n),
        )
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta = dict(meta or {})
            source = meta.pop("source", "unknown")
            out.append(
                RetrievedChunk(
                    chunk=Chunk(id=cid, text=doc, source=source, metadata=meta),
                    score=1.0 - float(dist),
                )
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()
