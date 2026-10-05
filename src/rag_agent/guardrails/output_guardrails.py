from __future__ import annotations

import re

from rag_agent.domain.interfaces import LLMClient
from rag_agent.domain.models import ChatMessage, Route
from rag_agent.guardrails.base import GuardrailContext, GuardrailResult

_CITE = re.compile(r"\[(\d+)\]")
_REFUSAL_HINTS = ("don't know", "do not know", "not enough information", "couldn't find", "could not find", "no information")


class CitationGuardrail:
    """Strips hallucinated citation numbers and warns when a grounded answer has no citations."""

    name = "citations"

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        if not ctx.contexts or ctx.route in (None, Route.DIRECT, Route.CALCULATOR):
            return GuardrailResult.allow(text)
        n = len(ctx.contexts)
        refs = {int(x) for x in _CITE.findall(text)}
        invalid = {r for r in refs if r < 1 or r > n}
        cleaned = text
        notes: list[str] = []
        if invalid:
            cleaned = _CITE.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), cleaned)
            notes.append(f"removed invalid citation(s) {sorted(invalid)}")
        if not (refs - invalid) and not any(h in text.lower() for h in _REFUSAL_HINTS):
            cleaned += "\n\n_Note: this answer has no inline citations - verify it against the sources._"
            notes.append("answer had no citations")
        if notes:
            return GuardrailResult.redact(cleaned, "; ".join(notes))
        return GuardrailResult.allow(text)


_GROUNDING_PROMPT = (
    "You verify answers against source passages. Reply with exactly one word: SUPPORTED if every "
    "factual claim in the ANSWER is backed by the CONTEXT (or the answer says it does not know), "
    "otherwise UNSUPPORTED."
)


class LLMGroundingGuardrail:
    """Optional LLM-as-judge hallucination check (enable with LLM_GUARDRAILS_ENABLED=true)."""

    name = "grounding"

    def __init__(self, llm: LLMClient, max_context_chars: int = 6000) -> None:
        self._llm, self._max = llm, max_context_chars

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult:
        if not ctx.contexts or ctx.route not in (Route.DOCUMENTS, Route.WEB):
            return GuardrailResult.allow(text)
        context = "\n---\n".join(ctx.contexts)[: self._max]
        verdict = self._llm.invoke(
            [
                ChatMessage(role="system", content=_GROUNDING_PROMPT),
                ChatMessage(role="user", content=f"<context>\n{context}\n</context>\n<answer>\n{text}\n</answer>"),
            ]
        )
        if verdict.strip().upper().startswith("UNSUPPORTED"):
            return GuardrailResult.block(text, "answer not supported by retrieved context")
        return GuardrailResult.allow(text)


class ContextSanitizer:
    """Defends against *indirect* prompt injection hidden in retrieved documents / web pages."""

    def sanitize(self, items):
        from rag_agent.domain.models import Chunk, RetrievedChunk
        from rag_agent.guardrails.patterns import INJECTION_PATTERNS

        changed = False
        out = []
        for it in items:
            text = it.chunk.text
            for pat in INJECTION_PATTERNS:
                text = pat.sub("[removed: suspected instruction]", text)
            if text != it.chunk.text:
                changed = True
                it = RetrievedChunk(chunk=Chunk(**{**it.chunk.model_dump(), "text": text}), score=it.score)
            out.append(it)
        return out, changed
