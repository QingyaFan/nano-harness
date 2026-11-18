"""Tool execution: validation, attribution, retries, and replay (I2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from ..events import (
    CREDENTIAL_ACCESSED,
    CREDENTIAL_DENIED,
    REDACTION_APPLIED,
    SANDBOX_DENIED,
)
from ..messages import ToolCall
from .errors import (
    ErrorKind,
    InvalidToolResult,
    ToolError,
    ToolFailure,
    ToolOutcome,
    ToolSuspended,
    ToolViolation,
    argument_error,
    classify,
    invalid_json_error,
    invalid_result_error,
    replay,
    tool_exception_error,
    unknown_tool_error,
)
from .spec import ToolSpec


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    turn: int
    attempt: int
    tool: str
    idempotency_key: str | None
    credentials: Any = None


_CURRENT: ContextVar[ToolContext | None] = ContextVar(
    "harness_tool_context", default=None
)


def current_tool_context() -> ToolContext:
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("no active tool context")
    return context


class ToolExecutor:
    def __init__(
        self,
        specs: Iterable[ToolSpec],
        *,
        run_id: str = "run",
        default_max_attempts: int = 1,
        policy: Any = None,
        gate: Any = None,
        credentials: Any = None,
        redactor: Any = None,
        events: Any = None,
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._run_id = run_id
        self._default_max_attempts = default_max_attempts
        self._completed: dict[str, ToolOutcome] = {}
        self._policy = policy
        self._gate = gate
        self._credentials = credentials
        self._redactor = redactor
        self._events = events

    def bind_events(self, events: Any) -> None:
        self._events = events
        if self._gate is not None and hasattr(self._gate, "bind_events"):
            self._gate.bind_events(events)

    def _emit(self, event_type: str, turn: int, **data: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, turn, **data)

    def resolve_approval(
        self, call_id: str, value: str, *, tool: str | None = None
    ) -> Any:
        if self._gate is not None and hasattr(self._gate, "resolve"):
            return self._gate.resolve(call_id, value, tool=tool)
        return None

    def specs(self) -> tuple[ToolSpec, ...]:
        specs = tuple(self._specs.values())
        if self._policy is not None and hasattr(self._policy, "visible"):
            return tuple(self._policy.visible(specs))
        return specs

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def execute(
        self,
        call: ToolCall,
        *,
        turn: int = 0,
        on_retry: Callable[[ToolError, int], None] | None = None,
    ) -> ToolOutcome:
        spec = self._specs.get(call.name)
        if spec is None:
            return self._failed(unknown_tool_error(call, self._specs), 0)
        if call.arguments is None:
            return self._failed(invalid_json_error(call), 0)

        errors = sorted(
            spec.validator.iter_errors(call.arguments),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            return self._failed(argument_error(spec, errors), 0)

        key = self._idempotency_key(spec, call.arguments)
        if key is not None and key in self._completed:
            return replay(self._completed[key])

        if self._gate is not None:
            blocked = self._gate.check(call, spec, call.arguments, turn=turn)
            if blocked is not None:
                return blocked

        max_attempts = self._max_attempts(spec)
        attempts = 0
        while True:
            attempts += 1
            try:
                content, redactions = self._invoke(
                    spec, call.arguments, turn, attempts, key, call_id=call.id
                )
            except InvalidToolResult as exc:
                return self._failed(invalid_result_error(exc), attempts)
            except ToolSuspended:
                raise
            except ToolViolation as exc:
                if exc.kind is ErrorKind.SANDBOX_VIOLATION:
                    self._emit(
                        SANDBOX_DENIED, turn,
                        call_id=call.id, tool=spec.name,
                        violation=exc.violation, detail=str(exc),
                    )
                return self._failed(
                    ToolError(
                        exc.kind,
                        str(exc),
                        retryable=False,
                        details={"violation": exc.violation},
                    ),
                    attempts,
                )
            except (ToolFailure, TimeoutError) as exc:
                error = classify(exc)
                if error.retryable and attempts < max_attempts:
                    if on_retry is not None:
                        on_retry(error, attempts)
                    continue
                return self._failed(error, attempts)
            except Exception as exc:  # noqa: BLE001
                return self._failed(tool_exception_error(exc), attempts)

            outcome = ToolOutcome(
                ok=True, content=content, error=None, attempts=attempts,
                redactions=redactions,
            )
            if key is not None:
                self._completed[key] = outcome
            return outcome

    def _invoke(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        turn: int,
        attempt: int,
        key: str | None,
        *,
        call_id: str,
    ) -> tuple[str, dict[str, int]]:
        credentials = None
        if self._credentials is not None:
            def on_access(name: str) -> None:
                self._emit(
                    CREDENTIAL_ACCESSED, turn,
                    call_id=call_id, tool=spec.name, name=name,
                )

            def on_denied(name: str, reason: str) -> None:
                self._emit(
                    CREDENTIAL_DENIED, turn,
                    call_id=call_id, tool=spec.name, name=name, reason=reason,
                )

            credentials = self._credentials.scope(
                spec.credentials, on_access=on_access, on_denied=on_denied
            )
        context = ToolContext(
            run_id=self._run_id,
            turn=turn,
            attempt=attempt,
            tool=spec.name,
            idempotency_key=key,
            credentials=credentials,
        )
        token = _CURRENT.set(context)
        try:
            result = spec.handler(**arguments)
        finally:
            _CURRENT.reset(token)
        if not isinstance(result, str):
            raise InvalidToolResult(
                f"tool {spec.name!r} returned {type(result).__name__}, expected str"
            )
        redactions: dict[str, int] = {}
        if self._redactor is not None:
            result, redactions = self._redactor(result)
            if redactions:
                self._emit(
                    REDACTION_APPLIED, turn,
                    call_id=call_id, tool=spec.name, counts=dict(redactions),
                )
        return result, redactions

    def _idempotency_key(
        self, spec: ToolSpec, arguments: dict[str, Any]
    ) -> str | None:
        if spec.read_only or not spec.idempotent:
            return None
        canonical = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        material = f"v1\x00{self._run_id}\x00{spec.name}\x00{canonical}"
        return hashlib.sha256(material.encode()).hexdigest()

    def _max_attempts(self, spec: ToolSpec) -> int:
        if spec.max_attempts is not None:
            return spec.max_attempts
        return self._default_max_attempts

    def _failed(self, error: ToolError, attempts: int) -> ToolOutcome:
        message = error.message
        hint = error.hint
        if self._redactor is not None:
            message, message_counts = self._redactor(message)
            hint, hint_counts = self._redactor(hint)
            counts = dict(message_counts)
            for name, count in hint_counts.items():
                counts[name] = counts.get(name, 0) + count
            if counts:
                self._emit(REDACTION_APPLIED, 0, counts=counts)
        error = ToolError(
            error.kind,
            message,
            error.retryable,
            hint,
            attempts,
            error.details,
        )
        return ToolOutcome(
            ok=False,
            content=error.envelope(),
            error=error,
            attempts=attempts,
        )