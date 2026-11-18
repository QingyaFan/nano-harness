"""Default redaction patterns for tool output (I5)."""

from __future__ import annotations

import re
from typing import Any

PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
    r"(?:[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----|\S*)"
)
GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]{8,}=*")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|token)\s*[\"']?\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9_\-\.]{6,})"
)
CN_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d(?:[-\s]?\d{4}){2}(?!\d)"
)
BARE_KEY_RE = re.compile(r"\bsk[-_]?[A-Za-z0-9_\-]{16,}\b")


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def replace(name: str) -> Any:
        def apply(match: re.Match[str]) -> str:
            counts[name] = counts.get(name, 0) + 1
            return f"[REDACTED:{name}]"

        return apply

    def key(match: re.Match[str]) -> str:
        counts["api_key"] = counts.get("api_key", 0) + 1
        start = match.start(2) - match.start(0)
        end = match.end(2) - match.start(0)
        return (
            match.group(0)[:start]
            + "[REDACTED:api_key]"
            + match.group(0)[end:]
        )

    text = PRIVATE_KEY_RE.sub(replace("private_key"), text)
    text = GITHUB_TOKEN_RE.sub(replace("github_token"), text)
    text = AWS_ACCESS_KEY_RE.sub(replace("aws_key"), text)
    text = BEARER_TOKEN_RE.sub(replace("bearer_token"), text)
    text = EMAIL_RE.sub(replace("email"), text)
    text = KEY_RE.sub(key, text)
    text = CN_PHONE_RE.sub(replace("cn_phone"), text)
    text = BARE_KEY_RE.sub(replace("api_key"), text)
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