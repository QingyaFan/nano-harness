"""The minimal agent loop: turn budget, stop gate, and no-progress guard (I1)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .context import ContextManager
from .events import (
    CONTEXT_PREPARED,
    MODEL_FAILED,
    MODEL_RESPONSE,
    RUN_FINISHED,
    RUN_STARTED,
    STOP_GATE,
    TOOL_CALL,
    TOOL_RESULT,
    TOOL_RETRY,
    TURN_STARTED,
    EventLog,
)
from .messages import Message, ToolCall, Usage
from .model import Model, ModelResponse
from .tools.errors import ToolOutcome


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    NO_PROGRESS = "no_progress"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str = ""
    feedback: str | None = None

    @classmethod
    def allow(cls, reason: str = "answer") -> GateDecision:
        return cls(True, reason)

    @classmethod
    def block(cls, reason: str, feedback: str | None = None) -> GateDecision:
        return cls(False, reason, feedback)


@dataclass(frozen=True)
class StopContext:
    messages: tuple[Message, ...]
    turn: int
    response: ModelResponse
    tool_calls: int


StopGate = Callable[[StopContext], GateDecision]


def require_answer(context: StopContext) -> GateDecision:
    if context.response.content and context.response.content.strip():
        return GateDecision.allow()
    return GateDecision.block("empty_answer", "请给出最终答案，或调用工具继续任务。")


@dataclass(frozen=True)
class RunConfig:
    max_turns: int = 8
    no_progress_limit: int = 3
    max_stop_blocks: int = 3

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.no_progress_limit < 1:
            raise ValueError("no_progress_limit must be at least 1")
        if self.max_stop_blocks < 1:
            raise ValueError("max_stop_blocks must be at least 1")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    stop_reason: StopReason
    turns: int
    tool_calls: int
    tool_attempts: int
    usage: Usage
    events: tuple[Any, ...]
    messages: tuple[Message, ...]
    output: str | None


class ToolExecutorLike(Protocol):
    def specs(self) -> Sequence[Any]: ...

    def execute(
        self,
        call: ToolCall,
        *,
        turn: int = 0,
        on_retry: Callable[[Any, int], None] | None = None,
    ) -> ToolOutcome: ...


def _signature(calls: tuple[ToolCall, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            call.name,
            json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str),
        )
        for call in calls
    )


def _is_cancelled(token: Any) -> bool:
    return bool(token is not None and getattr(token, "cancelled", False))


class HarnessLoop:
    def __init__(
        self,
        model: Model,
        tools: ToolExecutorLike,
        *,
        stop_gate: StopGate = require_answer,
        config: RunConfig | None = None,
        run_id: str | None = None,
        events: EventLog | None = None,
        context: ContextManager | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._stop_gate = stop_gate
        self._config = config or RunConfig()
        self._run_id = run_id
        self._events = events
        self._context = context

    def run(self, task: str, *, cancel: Any = None) -> RunResult:
        run_id = self._run_id or uuid.uuid4().hex[:12]
        log = self._events or EventLog()
        messages: list[Message] = [Message.user(task)]
        if self._context is not None:
            self._context.add(messages[0])
        turns = 0
        tool_calls = 0
        tool_attempts = 0
        blocks = 0
        repeats = 0
        last_signature: tuple[tuple[str, str], ...] | None = None
        usage = Usage()

        log.emit(RUN_STARTED, 0, task=task)

        while True:
            if _is_cancelled(cancel):
                return self._finish(
                    run_id, log, messages, StopReason.CANCELLED,
                    "cancelled", turns, tool_calls, tool_attempts, usage, None,
                )
            if turns >= self._config.max_turns:
                return self._finish(
                    run_id, log, messages, StopReason.MAX_TURNS,
                    "turn_budget", turns, tool_calls, tool_attempts, usage, None,
                )

            turns += 1
            log.emit(TURN_STARTED, turns)
            specs = self._tools.specs()
            wire = tuple(messages)
            if self._context is not None:
                prepared = self._context.wire(specs)
                wire = prepared.messages
                log.emit(
                    CONTEXT_PREPARED, turns,
                    used=prepared.used,
                    records=[record.kind for record in prepared.records],
                    exhausted=prepared.exhausted,
                )
            try:
                response = self._model.complete(wire, specs)
            except Exception as exc:  # noqa: BLE001
                log.emit(
                    MODEL_FAILED, turns,
                    error=type(exc).__name__, message=str(exc),
                )
                return self._finish(
                    run_id, log, messages, StopReason.FAILED,
                    "model_error", turns, tool_calls, tool_attempts, usage, None,
                )

            usage = usage + response.usage
            if self._context is not None:
                self._context.anchor(response.usage.prompt_tokens)
            log.emit(
                MODEL_RESPONSE, turns,
                content=response.content,
                tool_calls=[call.name for call in response.tool_calls],
            )
            assistant_message = Message.assistant(
                response.content, response.tool_calls
            )
            messages.append(assistant_message)
            if self._context is not None:
                self._context.add(assistant_message)

            if response.tool_calls:
                signature = _signature(response.tool_calls)
                if signature == last_signature:
                    repeats += 1
                else:
                    last_signature = signature
                    repeats = 1

                for index, call in enumerate(response.tool_calls):
                    if _is_cancelled(cancel):
                        reason = getattr(cancel, "reason", "cancelled")
                        for remaining in response.tool_calls[index:]:
                            messages.append(
                                Message.tool_result(remaining, f"CANCELLED: {reason}")
                            )
                            log.emit(
                                TOOL_RESULT, turns,
                                call_id=remaining.id, ok=False, cancelled=True,
                            )
                        return self._finish(
                            run_id, log, messages, StopReason.CANCELLED,
                            reason, turns, tool_calls, tool_attempts, usage, None,
                        )

                    log.emit(
                        TOOL_CALL, turns,
                        call_id=call.id, name=call.name, arguments=call.arguments,
                    )

                    def on_retry(
                        error: Any,
                        attempt: int,
                        call: ToolCall = call,
                        turn: int = turns,
                    ) -> None:
                        log.emit(
                            TOOL_RETRY, turn,
                            call_id=call.id, name=call.name,
                            attempt=attempt,
                            error_kind=getattr(error.kind, "value", error.kind),
                            retryable=error.retryable,
                        )

                    outcome = self._tools.execute(call, turn=turns, on_retry=on_retry)
                    tool_calls += 1
                    tool_attempts += outcome.attempts
                    result_message = Message.tool_result(call, outcome.content)
                    messages.append(result_message)
                    if self._context is not None:
                        self._context.add(result_message, tool=call.name)
                    log.emit(
                        TOOL_RESULT, turns,
                        call_id=call.id, ok=outcome.ok,
                        attempts=outcome.attempts, replayed=outcome.replayed,
                    )

                if repeats >= self._config.no_progress_limit:
                    return self._finish(
                        run_id, log, messages, StopReason.NO_PROGRESS,
                        "repeated_tool_calls", turns, tool_calls,
                        tool_attempts, usage, None,
                    )
                continue

            try:
                decision = self._stop_gate(
                    StopContext(
                        messages=tuple(messages),
                        turn=turns,
                        response=response,
                        tool_calls=tool_calls,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.emit(
                    STOP_GATE, turns,
                    passed=False, reason="gate_error",
                    error=type(exc).__name__, message=str(exc),
                )
                return self._finish(
                    run_id, log, messages, StopReason.FAILED,
                    "stop_gate_error", turns, tool_calls, tool_attempts, usage, None,
                )

            log.emit(
                STOP_GATE, turns,
                passed=decision.passed, reason=decision.reason,
            )
            if decision.passed:
                return self._finish(
                    run_id, log, messages, StopReason.COMPLETED,
                    decision.reason or "answer", turns, tool_calls,
                    tool_attempts, usage, response.content,
                )

            blocks += 1
            if blocks >= self._config.max_stop_blocks:
                return self._finish(
                    run_id, log, messages, StopReason.NO_PROGRESS,
                    "stop_gate_limit", turns, tool_calls, tool_attempts, usage, None,
                )
            feedback = Message.user(
                decision.feedback or "任务尚未完成，请继续。"
            )
            messages.append(feedback)
            if self._context is not None:
                self._context.add(feedback)

    @staticmethod
    def _finish(
        run_id: str,
        log: EventLog,
        messages: list[Message],
        stop_reason: StopReason,
        reason: str,
        turns: int,
        tool_calls: int,
        tool_attempts: int,
        usage: Usage,
        output: str | None,
    ) -> RunResult:
        log.emit(
            RUN_FINISHED, turns,
            stop_reason=stop_reason.value, reason=reason,
            tool_calls=tool_calls, tool_attempts=tool_attempts,
        )
        return RunResult(
            run_id=run_id,
            stop_reason=stop_reason,
            turns=turns,
            tool_calls=tool_calls,
            tool_attempts=tool_attempts,
            usage=usage,
            events=log.snapshot(),
            messages=tuple(messages),
            output=output,
        )