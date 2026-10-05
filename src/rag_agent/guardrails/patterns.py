"""Shared detection patterns (injection + PII)."""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
        r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system)\s+\w+",
        r"forget\s+(?:everything|all|your)\s+(?:instructions?|rules?|training)",
        r"you\s+are\s+now\s+(?:in\s+)?(?:dan|developer\s+mode|jailbroken)",
        r"(?:reveal|print|show|repeat|output|leak)\s+(?:me\s+)?(?:your|the)\s+(?:system|hidden|initial|original)\s+(?:prompt|instructions?)",
        r"act\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken)",
        r"do\s+anything\s+now",
        r"<\s*/?\s*(?:system|assistant)\s*>",
        r"\[\s*(?:system|inst)\s*\]",
        r"override\s+(?:your\s+)?(?:safety|guardrails?|instructions?)",
    )
]


def find_injection(text: str) -> str | None:
    for pat in INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("credit_card")
            return "[REDACTED_CARD]"
        return m.group()

    def simple(label: str, token: str):
        def _sub(m: re.Match) -> str:
            found.append(label)
            return token
        return _sub

    text = _CARD.sub(card, text)
    text = _SSN.sub(simple("ssn", "[REDACTED_SSN]"), text)
    text = _AADHAAR.sub(simple("aadhaar", "[REDACTED_ID]"), text)
    text = _EMAIL.sub(simple("email", "[REDACTED_EMAIL]"), text)
    text = _PHONE.sub(simple("phone", "[REDACTED_PHONE]"), text)
    return text, sorted(set(found))
