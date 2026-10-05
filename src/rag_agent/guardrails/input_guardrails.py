from __future__ import annotations

from rag_agent.domain.interfaces import LLMClient
from rag_agent.domain.models import ChatMessage
from rag_agent.guardrails.base import GuardrailContext, GuardrailResult
from rag_agent.guardrails.patterns import find_injection, redact_pii


class LengthGuardrail:
    name = "length"

    def __init__(self, max_chars: int) -> None:
        self._max = max_chars

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        if not text.strip():
            return GuardrailResult.block(text, "empty input")
        if len(text) > self._max:
            return GuardrailResult.block(text, f"input exceeds {self._max} characters")
        return GuardrailResult.allow(text)


class PromptInjectionGuardrail:
    name = "prompt_injection"

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        hit = find_injection(text)
        if hit:
            return GuardrailResult.block(text, f"suspected prompt injection ({hit[:40]!r})")
        return GuardrailResult.allow(text)


class PIIRedactionGuardrail:
    """Used on both input (privacy: PII never reaches the LLM) and output."""

    name = "pii_redaction"

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        cleaned, found = redact_pii(text)
        if found:
            return GuardrailResult.redact(cleaned, f"redacted {', '.join(found)}")
        return GuardrailResult.allow(text)


_MODERATION_PROMPT = (
    "You are a content-safety classifier. Reply with exactly one word: SAFE or UNSAFE.\n"
    "UNSAFE = requests for weapons/explosives/malware creation, self-harm instructions, "
    "sexual content involving minors, targeted harassment or hate, or serious illegal wrongdoing. "
    "Everything else, including ordinary research questions, is SAFE."
)


class LLMModerationGuardrail:
    """Optional LLM-as-judge moderation (enable with LLM_GUARDRAILS_ENABLED=true)."""

    name = "llm_moderation"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        verdict = self._llm.invoke(
            [
                ChatMessage(role="system", content=_MODERATION_PROMPT),
                ChatMessage(role="user", content=f"<message>\n{text}\n</message>"),
            ]
        )
        if verdict.strip().upper().startswith("UNSAFE"):
            return GuardrailResult.block(text, "flagged unsafe by moderation model")
        return GuardrailResult.allow(text)
