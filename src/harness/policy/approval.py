"""Approval requests and resume-value parsing (I5)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .types import RiskLevel


@dataclass(frozen=True)
class Approval:
    approved: bool
    reason: str = ""
    source: str = "resume"


@dataclass(frozen=True)
class ApprovalRequest:
    call_id: str
    tool: str
    risk: RiskLevel
    arguments_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "risk": self.risk.value,
            "arguments_digest": self.arguments_digest,
        }


def digest_arguments(arguments: dict[str, Any] | None) -> str:
    canonical = json.dumps(
        arguments or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_resume_value(value: str, *, call_id: str, tool: str) -> Approval:
    text = (value or "").strip()
    lowered = text.lower()
    if lowered.startswith(("approve", "allow", "yes")):
        reason = text.split(":", 1)[1].strip() if ":" in text else ""
        return Approval(True, reason or "approved")
    if lowered.startswith(("deny", "reject", "no")):
        reason = text.split(":", 1)[1].strip() if ":" in text else ""
        return Approval(False, reason or "denied")
    return Approval(False, text or "not approved")