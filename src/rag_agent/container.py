"""Composition root: the ONLY place that knows about concrete classes (DIP)."""
from __future__ import annotations

import logging
from functools import cached_property

from rag_agent.agent.generation import AnswerGenerator
from rag_agent.agent.graph import build_agent_graph
from rag_agent.agent.nodes import AgentNodes
from rag_agent.agent.router import LLMRouter
from rag_agent.agent.service import AgentService
from rag_agent.config import Settings, get_settings
from rag_agent.domain.models import Route
from rag_agent.embeddings.fastembed_embedder import FastEmbedEmbedder
from rag_agent.guardrails.factory import build_input_chain, build_output_chain
from rag_agent.guardrails.output_guardrails import ContextSanitizer
from rag_agent.ingestion.chunker import RecursiveChunker
from rag_agent.ingestion.loaders import PdfLoader, TextFileLoader
from rag_agent.ingestion.pipeline import IngestionPipeline
from rag_agent.llm.factory import create_judge_llm, create_llm
from rag_agent.retrieval.bm25_index import BM25KeywordIndex
from rag_agent.retrieval.factory import build_retriever
from rag_agent.retrieval.reranker import FastEmbedReranker
from rag_agent.retrieval.vector_store import ChromaVectorStore, FaissVectorStore
from rag_agent.tools.calculator import CalculatorTool
from rag_agent.tools.document_search import DocumentSearchTool
from rag_agent.tools.web_search import WebSearchTool

log = logging.getLogger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._agents: dict[str, AgentService] = {}

    # --- infrastructure ---
    @cached_property
    def llm(self):
        return create_llm(self.settings)

    @cached_property
    def judge_llm(self):
        return create_judge_llm(self.settings)

    @cached_property
    def embedder(self):
        return FastEmbedEmbedder(self.settings.embedding_model)

    @cached_property
    def reranker(self):
        return FastEmbedReranker(self.settings.reranker_model)

    @cached_property
    def vector_store(self):
        if self.settings.vector_store == "chroma":
            return ChromaVectorStore(self.settings.index_dir)
        return FaissVectorStore(self.settings.index_dir)

    @cached_property
    def keyword_index(self):
        return BM25KeywordIndex(self.settings.index_dir)

    @cached_property
    def pipeline(self) -> IngestionPipeline:
        s = self.settings
        return IngestionPipeline(
            loaders=[TextFileLoader(), PdfLoader()],
            chunker=RecursiveChunker(s.chunk_size, s.chunk_overlap),
            embedder=self.embedder,
            vector_store=self.vector_store,
            keyword_index=self.keyword_index,
        )

    @cached_property
    def generator(self) -> AnswerGenerator:
        return AnswerGenerator(self.llm)

    # --- retrieval / agent ---
    def retriever(self, strategy: str | None = None):
        s = self.settings
        return build_retriever(
            strategy or s.retrieval_strategy,
            embedder=self.embedder,
            vector_store=self.vector_store,
            keyword_index=self.keyword_index,
            reranker_provider=lambda: self.reranker,
            candidate_k=s.candidate_k,
            rrf_k=s.rrf_k,
        )

    def agent(self, strategy: str | None = None) -> AgentService:
        strategy = strategy or self.settings.retrieval_strategy
        if strategy not in self._agents:
            s = self.settings
            tools = {
                Route.DOCUMENTS: DocumentSearchTool(self.retriever(strategy), s.top_k),
                Route.WEB: WebSearchTool(s.web_search_max_results),
                Route.CALCULATOR: CalculatorTool(self.llm),
            }
            nodes = AgentNodes(
                input_guard=build_input_chain(s, self.judge_llm if s.llm_guardrails_enabled else None),
                output_guard=build_output_chain(s, self.judge_llm if s.llm_guardrails_enabled else None),
                router=LLMRouter(self.llm, list(tools)),
                tools=tools,
                generator=self.generator,
                sanitizer=ContextSanitizer(),
            )
            self._agents[strategy] = AgentService(build_agent_graph(nodes))
        return self._agents[strategy]

    def ensure_index(self) -> int:
        """Index data/documents on first start (e.g. fresh Streamlit Cloud / Docker container)."""
        if self.vector_store.count() > 0:
            return 0
        report = self.pipeline.ingest_directory(self.settings.data_dir)
        log.info("Initial ingest: %s files, %s chunks", report.files, report.chunks_added)
        return report.chunks_added
