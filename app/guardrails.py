"""Guardrails: input sanitization and prompt-injection detection.

This module is a *defense-in-depth* layer. It does not replace the structural
defenses in the RAG pipeline (delimiting retrieved context, a strict system
prompt, refusing on low-relevance retrieval) — it complements them by catching
the most common injection patterns before they reach the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns that strongly indicate an attempt to override instructions or
# exfiltrate the system prompt. Kept deliberately conservative to avoid blocking
# legitimate questions — the model + grounding checks are the primary defense.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(all\s+|the\s+|your\s+|previous\s+|above\s+)+", re.I),
    re.compile(r"\bdisregard\s+(all\s+|the\s+|your\s+|previous\s+|above\s+)", re.I),
    re.compile(r"\bforget\s+(everything|all|your|the|previous)\b", re.I),
    re.compile(r"\b(system|developer)\s+prompt\b", re.I),
    re.compile(r"\bnew\s+(instructions|system\s+prompt|rules)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bact\s+as\s+(if|though|a|an)\b", re.I),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are)\b", re.I),
    re.compile(r"\b(reveal|print|repeat|show|output)\s+(your\s+|the\s+|me\s+the\s+)?"
               r"(system\s+prompt|instructions|initial\s+prompt)", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\boverride\s+(your\s+|the\s+|all\s+|previous\s+)", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),
    re.compile(r"</?(system|instructions?|prompt)>", re.I),
]

# Control characters / zero-width chars sometimes used to smuggle instructions.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮]")

MAX_INPUT_LEN = 4000


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    sanitized: str
    reason: str | None = None


def sanitize(text: str) -> str:
    """Strip control/zero-width characters and normalize whitespace."""
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").strip()
    return cleaned


def detect_injection(text: str) -> str | None:
    """Return a reason string if the text looks like a prompt-injection, else None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return f"Input matched a disallowed instruction-override pattern."
    return None


def check_input(text: str) -> GuardrailResult:
    """Run the full input guardrail chain.

    Returns a GuardrailResult; when `allowed` is False the request should be
    refused without ever reaching the model.
    """
    sanitized = sanitize(text)

    if not sanitized:
        return GuardrailResult(False, sanitized, "Empty input after sanitization.")

    if len(sanitized) > MAX_INPUT_LEN:
        return GuardrailResult(
            False, sanitized, f"Input exceeds maximum length of {MAX_INPUT_LEN} characters."
        )

    reason = detect_injection(sanitized)
    if reason:
        return GuardrailResult(False, sanitized, reason)

    return GuardrailResult(True, sanitized, None)


def neutralize_context(text: str) -> str:
    """Defang retrieved-context text so embedded instructions can't act as commands.

    Retrieved documents are untrusted. We strip control characters and any
    delimiter sequences that could be used to break out of the context block in
    the prompt.
    """
    cleaned = _CONTROL_CHARS.sub("", text)
    # Prevent a document from forging our context delimiters.
    cleaned = re.sub(r"</?(context|document|system|instructions?)>", "", cleaned, flags=re.I)
    return cleaned.strip()
