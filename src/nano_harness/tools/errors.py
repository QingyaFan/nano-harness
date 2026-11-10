"""Tool errors, error envelopes, and execution outcomes (I2)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..messages import ToolCall

if TYPE_CHECKING:
    from jsonschema import ValidationError

    from .spec import ToolSpec


class ErrorKind(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_JSON = "invalid_json"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_RESULT = "invalid_result"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    SANDBOX_VIOLATION = "sandbox_violation"
    PERMISSION_DENIED = "permission_denied"
    CREDENTIAL_DENIED = "credential_denied"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ToolError:
    kind: ErrorKind
    message: str
    retryable: bool = False
    hint: str = ""
    attempts: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind.value,
            "message": self.message,
            "retryable": self.retryable,
            "hint": self.hint,
            "attempts": self.attempts,
        }
        payload.update(self.details)
        return payload

    def envelope(self) -> str:
        payload = {"ok": False, "error": self.to_dict()}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    content: str
    error: ToolError | None = None
    attempts: int = 0
    replayed: bool = False
    redactions: dict[str, int] = field(default_factory=dict)


def replay(outcome: ToolOutcome) -> ToolOutcome:
    return replace(outcome, attempts=0, replayed=True)


class ToolFailure(Exception):
    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.TOOL_ERROR,
        retryable: bool = False,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.hint = hint


class RetryableToolError(ToolFailure):
    def __init__(
        self,
        message: str = "transient tool failure",
        *,
        kind: ErrorKind = ErrorKind.TOOL_ERROR,
        hint: str = "",
    ) -> None:
        super().__init__(message, kind=kind, retryable=True, hint=hint)


class ToolSuspended(Exception):
    """Raised by a tool or policy when the run must wait for a human decision."""

    def __init__(
        self,
        reason: str,
        *,
        kind: str = "tool",
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.payload = payload or {}


class ToolViolation(Exception):
    """Raised when a workspace or sandbox boundary rejects an operation."""

    def __init__(
        self,
        violation: str,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.SANDBOX_VIOLATION,
    ) -> None:
        super().__init__(message)
        self.violation = violation
        self.kind = kind


class InvalidToolResult(Exception):
    """Raised internally when a handler does not return a string."""


def unknown_tool_error(call: ToolCall, available: Iterable[str]) -> ToolError:
    names = sorted(available)
    hint = "可用工具：" + "、".join(names) if names else "没有已注册的工具"
    return ToolError(
        ErrorKind.UNKNOWN_TOOL,
        f"unknown tool {call.name!r}",
        retryable=False,
        hint=hint,
    )


def invalid_json_error(call: ToolCall) -> ToolError:
    raw = (call.raw_arguments or "").strip()
    hint = f"收到的参数：{raw[:120]}" if raw else "arguments 不是 JSON 对象"
    return ToolError(
        ErrorKind.INVALID_JSON,
        "tool arguments must be a JSON object",
        retryable=False,
        hint=hint,
    )


def argument_error(spec: ToolSpec, errors: list[ValidationError]) -> ToolError:
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    required = [str(item) for item in spec.parameters.get("required", [])]
    available = [str(item) for item in spec.parameters.get("properties", {})]
    if required:
        hint = f"必填参数：{'、'.join(required)}；可用参数：{'、'.join(available)}"
    else:
        hint = f"可用参数：{'、'.join(available)}" if available else ""
    return ToolError(
        ErrorKind.INVALID_ARGUMENTS,
        f"{path}: {first.message}",
        retryable=False,
        hint=hint,
    )


def invalid_result_error(exc: InvalidToolResult) -> ToolError:
    return ToolError(
        ErrorKind.INVALID_RESULT,
        str(exc),
        retryable=False,
        hint="tool handler 必须返回 str",
    )


def tool_exception_error(exc: Exception) -> ToolError:
    return ToolError(
        ErrorKind.TOOL_ERROR,
        f"{type(exc).__name__}: {exc}",
        retryable=False,
    )


def classify(exc: Exception) -> ToolError:
    if isinstance(exc, ToolFailure):
        return ToolError(exc.kind, str(exc), exc.retryable, exc.hint)
    if isinstance(exc, TimeoutError):
        return ToolError(ErrorKind.TIMEOUT, "tool timed out", retryable=True)
    return tool_exception_error(exc)