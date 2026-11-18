"""Approval requests and resume-value parsing (I5)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .types import RiskLevel


@dataclass(frozen=True)
class Approval:
    approved: bool
    reason: str = ""
    source: str = "resume"
    by: str = "host"
    call_id: str = ""
    tool: str = ""


@dataclass(frozen=True)
class ApprovalRequest:
    call_id: str
    tool: str
    risk: RiskLevel
    reason: str = ""
    rule: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "risk": self.risk.value,
            "reason": self.reason,
            "rule": self.rule,
            "arguments": dict(self.arguments),
            "arguments_digest": self.arguments_digest,
        }


def digest_arguments(arguments: dict[str, Any] | None) -> str:
    canonical = json.dumps(
        arguments or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _clip_reason(reason: str) -> str:
    return reason if len(reason) <= 500 else reason[:499] + "…"


def parse_resume_value(value: str, *, call_id: str, tool: str) -> Approval:
    text = (value or "").strip()
    head, _, tail = text.partition(":")
    word = head.strip().lower()
    reason = _clip_reason(tail.strip())

    if word in {"approve", "approved", "allow", "allowed"}:
        return Approval(
            True, reason, source="resume", by="host",
            call_id=call_id, tool=tool,
        )
    if word in {"deny", "denied", "reject", "rejected"}:
        return Approval(
            False, reason or "denied by approver", source="resume", by="host",
            call_id=call_id, tool=tool,
        )
    raise ValueError(f"unrecognized resume value: {text!r}")