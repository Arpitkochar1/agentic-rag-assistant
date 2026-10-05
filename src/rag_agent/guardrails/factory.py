from __future__ import annotations

from rag_agent.config import Settings
from rag_agent.domain.interfaces import LLMClient
from rag_agent.guardrails.base import GuardrailChain
from rag_agent.guardrails.input_guardrails import (
    LengthGuardrail,
    LLMModerationGuardrail,
    PIIRedactionGuardrail,
    PromptInjectionGuardrail,
)
from rag_agent.guardrails.output_guardrails import CitationGuardrail, LLMGroundingGuardrail


def build_input_chain(settings: Settings, judge: LLMClient | None = None) -> GuardrailChain:
    if not settings.guardrails_enabled:
        return GuardrailChain([])
    guards = [LengthGuardrail(settings.max_input_chars), PromptInjectionGuardrail(), PIIRedactionGuardrail()]
    if settings.llm_guardrails_enabled and judge is not None:
        guards.append(LLMModerationGuardrail(judge))
    return GuardrailChain(guards)


def build_output_chain(settings: Settings, judge: LLMClient | None = None) -> GuardrailChain:
    if not settings.guardrails_enabled:
        return GuardrailChain([])
    guards = [CitationGuardrail()]
    if settings.llm_guardrails_enabled and judge is not None:
        guards.append(LLMGroundingGuardrail(judge))
    guards.append(PIIRedactionGuardrail())
    return GuardrailChain(guards)
