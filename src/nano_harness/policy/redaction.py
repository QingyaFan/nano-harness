"""Default redaction patterns for tool output (I5)."""

from __future__ import annotations

import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|token)\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9_\-\.]{6,})"
)
BARE_KEY_RE = re.compile(
    r"\b(?:sk|ghp|gho|ghs|github_pat)_?[A-Za-z0-9_\-]{8,}\b"
)


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def email(match: re.Match[str]) -> str:
        counts["email"] = counts.get("email", 0) + 1
        return "[REDACTED:email]"

    def key(match: re.Match[str]) -> str:
        counts["api_key"] = counts.get("api_key", 0) + 1
        return f"{match.group(1)}=[REDACTED:api_key]"

    def bare(match: re.Match[str]) -> str:
        counts["api_key"] = counts.get("api_key", 0) + 1
        return "[REDACTED:api_key]"

    text = EMAIL_RE.sub(email, text)
    text = KEY_RE.sub(key, text)
    text = BARE_KEY_RE.sub(bare, text)
    return text, counts


def redact_value(value: Any) -> tuple[Any, dict[str, int]]:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        counts: dict[str, int] = {}
        items = []
        for item in value:
            redacted, part = redact_value(item)
            items.append(redacted)
            for key, count in part.items():
                counts[key] = counts.get(key, 0) + count
        return items, counts
    if isinstance(value, dict):
        counts = {}
        payload = {}
        for key, item in value.items():
            redacted, part = redact_value(item)
            payload[key] = redacted
            for name, count in part.items():
                counts[name] = counts.get(name, 0) + count
        return payload, counts
    return value, {}