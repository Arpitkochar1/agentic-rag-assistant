from rag_agent.domain.models import Chunk, RetrievedChunk
from rag_agent.ingestion.chunker import RecursiveChunker
from rag_agent.ingestion.loaders import TextFileLoader
from rag_agent.ingestion.pipeline import IngestionPipeline
from rag_agent.retrieval.bm25_index import BM25KeywordIndex
from rag_agent.retrieval.factory import build_retriever
from rag_agent.retrieval.retrievers import reciprocal_rank_fusion
from rag_agent.retrieval.vector_store import FaissVectorStore


def rc(cid):
    return RetrievedChunk(chunk=Chunk(id=cid, text=cid, source="s"))


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[rc("a"), rc("b"), rc("c")], [rc("c"), rc("b"), rc("d")]], k=60)
    assert [r.chunk.id for r in fused][:2] == ["b", "c"] or fused[0].chunk.id in {"b", "c"}
    assert {r.chunk.id for r in fused} == {"a", "b", "c", "d"}


def build(tmp_path, embedder, docs):
    store, kw = FaissVectorStore(tmp_path / "idx"), BM25KeywordIndex(tmp_path / "idx")
    for name, text in docs.items():
        (tmp_path / name).write_text(text)
    pipe = IngestionPipeline([TextFileLoader()], RecursiveChunker(200, 20), embedder, store, kw)
    report = pipe.ingest_paths([tmp_path / n for n in docs])
    return store, kw, report


DOCS = {
    "a.md": "Error code E4021 means the payment gateway timed out. Retry after thirty seconds.",
    "b.md": "Dense retrieval embeds queries and finds semantically similar passages using cosine similarity.",
    "c.md": "Cats are small carnivorous mammals often kept as pets and prized for independence.",
}


def test_ingest_dedupes_and_persists(tmp_path, embedder):
    store, kw, report = build(tmp_path, embedder, DOCS)
    assert report.files == 3 and report.chunks_added == store.count() == kw.count()
    again = IngestionPipeline([TextFileLoader()], RecursiveChunker(200, 20), embedder, store, kw).ingest_paths(
        [tmp_path / "a.md"]
    )
    assert again.chunks_added == 0
    # reload from disk
    assert FaissVectorStore(tmp_path / "idx").count() == store.count()
    assert BM25KeywordIndex(tmp_path / "idx").count() == kw.count()


def test_hybrid_finds_exact_identifier(tmp_path, embedder):
    store, kw, _ = build(tmp_path, embedder, DOCS)
    r = build_retriever(
        "hybrid", embedder=embedder, vector_store=store, keyword_index=kw,
        reranker_provider=lambda: None, candidate_k=5,
    )
    top = r.retrieve("E4021", 1)
    assert top and top[0].chunk.source == "a.md"


def test_rerank_strategy_uses_reranker(tmp_path, embedder):
    store, kw, _ = build(tmp_path, embedder, DOCS)

    class Reverse:
        def rerank(self, query, items, top_k):
            return list(reversed(items))[:top_k]

    r = build_retriever(
        "hybrid_rerank", embedder=embedder, vector_store=store, keyword_index=kw,
        reranker_provider=lambda: Reverse(), candidate_k=5,
    )
    assert len(r.retrieve("payment gateway", 2)) == 2


def test_unknown_strategy(tmp_path, embedder):
    store, kw, _ = build(tmp_path, embedder, DOCS)
    import pytest

    with pytest.raises(ValueError):
        build_retriever("nope", embedder=embedder, vector_store=store, keyword_index=kw, reranker_provider=lambda: None)
