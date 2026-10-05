"""Provider registry. Add a provider = register a builder (Open/Closed)."""
from __future__ import annotations

from typing import Any, Callable

from rag_agent.config import Settings
from rag_agent.llm.langchain_client import LangChainLLM

Builder = Callable[[Settings, str], Any]


def _common(settings: Settings, api_key: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _anthropic(settings: Settings, model: str) -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **_common(settings, settings.anthropic_api_key))


def _openai(settings: Settings, model: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, **_common(settings, settings.openai_api_key))


def _groq(settings: Settings, model: str) -> Any:
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, **_common(settings, settings.groq_api_key))


LLM_PROVIDERS: dict[str, Builder] = {
    "anthropic": _anthropic,
    "openai": _openai,
    "groq": _groq,
}


def register_provider(name: str, builder: Builder) -> None:
    LLM_PROVIDERS[name] = builder


def create_llm(settings: Settings, provider: str | None = None, model: str | None = None) -> LangChainLLM:
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Known: {sorted(LLM_PROVIDERS)}")
    return LangChainLLM(LLM_PROVIDERS[provider](settings, model))


def create_judge_llm(settings: Settings) -> LangChainLLM:
    """LLM used for RAGAS / LLM-judge guardrails. Falls back to the main LLM."""
    if settings.judge_provider:
        if not settings.judge_model:
            raise ValueError("JUDGE_MODEL must be set when JUDGE_PROVIDER is set")
        return create_llm(settings, settings.judge_provider, settings.judge_model)
    return create_llm(settings)
