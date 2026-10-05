"""Guardrail abstractions + a composite chain (Composite + Chain of Responsibility)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

from rag_agent.domain.models import Route

log = logging.getLogger(__name__)


class GuardAction(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"  # text was modified, continue
    BLOCK = "block"  # stop the pipeline


@dataclass(frozen=True)
class GuardrailContext:
    question: str = ""
    route: Route | None = None
    contexts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardrailResult:
    action: GuardAction
    text: str
    reason: str = ""

    @classmethod
    def allow(cls, text: str) -> "GuardrailResult":
        return cls(GuardAction.ALLOW, text)

    @classmethod
    def redact(cls, text: str, reason: str) -> "GuardrailResult":
        return cls(GuardAction.REDACT, text, reason)

    @classmethod
    def block(cls, text: str, reason: str) -> "GuardrailResult":
        return cls(GuardAction.BLOCK, text, reason)


class Guardrail(Protocol):
    name: str

    def check(self, text: str, ctx: GuardrailContext) -> GuardrailResult: ...


@dataclass
class ChainResult:
    text: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)


class GuardrailChain:
    """Runs guardrails in order. A guardrail that crashes is logged and skipped (fail-open
    for availability); deterministic guards are cheap and don't crash in practice."""

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self._guardrails = list(guardrails)

    def run(self, text: str, ctx: GuardrailContext | None = None) -> ChainResult:
        ctx = ctx or GuardrailContext()
        reasons: list[str] = []
        current = text
        for g in self._guardrails:
            try:
                result = g.check(current, ctx)
            except Exception as exc:
                log.warning("Guardrail %s failed: %s", g.name, exc)
                reasons.append(f"{g.name}: error ({type(exc).__name__}) - skipped")
                continue
            if result.action is GuardAction.BLOCK:
                return ChainResult(current, True, reasons + [f"{g.name}: {result.reason}"])
            if result.action is GuardAction.REDACT:
                current = result.text
                reasons.append(f"{g.name}: {result.reason}")
        return ChainResult(current, False, reasons)
