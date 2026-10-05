"""Central, typed configuration (12-factor: everything comes from env vars / .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

Strategy = Literal["vector", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "groq"] = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    # Optional separate "judge" model for RAGAS / LLM guardrails (reduces self-grading bias)
    judge_provider: Optional[Literal["anthropic", "openai", "groq"]] = None
    judge_model: Optional[str] = None

    # --- Retrieval -------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    vector_store: Literal["faiss", "chroma"] = "faiss"
    retrieval_strategy: Strategy = "hybrid_rerank"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60

    # --- Storage ---------------------------------------------------------
    data_dir: Path = Path("data/documents")
    index_dir: Path = Path("data/index")

    # --- Tools -----------------------------------------------------------
    web_search_max_results: int = 5

    # --- Guardrails ------------------------------------------------------
    guardrails_enabled: bool = True
    llm_guardrails_enabled: bool = False  # LLM-as-judge moderation + grounding (extra latency/cost)
    max_input_chars: int = 2000

    # --- API -------------------------------------------------------------
    api_key: Optional[str] = None  # if set, API requires header X-API-Key
    max_upload_mb: int = 10

    @property
    def llm_api_key(self) -> Optional[str]:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
