"""Shared types for the optional orchestration layer (A1)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..lifecycle import Suspension
from ..loop import RunResult, StopReason
from ..messages import Usage

LOOP_STARTED = "loop_started"
LOOP_ITERATION = "loop_iteration"
LOOP_STOP_GATE = "loop_stop_gate"
LOOP_SUSPENDED = "loop_suspended"
LOOP_RESUMED = "loop_resumed"
LOOP_SKIPPED = "loop_skipped"
LOOP_FINISHED = "loop_finished"


class LoopError(Exception):
    pass


class SchedulerStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    HUMAN_REJECTED = "human_rejected"
    SKIPPED = "skipped"


class HumanAction(StrEnum):
    PROCEED = "proceed"
    PAUSE = "pause"
    STOP = "stop"


@dataclass(frozen=True)
class StepResult:
    ok: bool = True
    output: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok_output(
        cls,
        output: str,
        *,
        usage: Usage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        return cls(
            ok=True,
            output=output,
            usage=usage or Usage(),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        output: str = "",
        usage: Usage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        return cls(
            ok=False,
            output=output,
            usage=usage or Usage(),
            error=error,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class StepSummary:
    iteration: int
    ok: bool
    output: str
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "ok": self.ok,
            "output": self.output,
            "usage": self.usage.to_dict(),
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StepSummary:
        return cls(
            iteration=int(payload.get("iteration", 0)),
            ok=bool(payload.get("ok", False)),
            output=str(payload.get("output", "")),
            usage=Usage(**dict(payload.get("usage") or {})),
            error=payload.get("error"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LoopState:
    loop_id: str
    task: str
    status: SchedulerStatus
    iteration: int = 0
    output: str = ""
    reason: str = ""
    usage: Usage = field(default_factory=Usage)
    consecutive_failures: int = 0
    history: tuple[StepSummary, ...] = ()
    suspension: Suspension | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.loop_id.strip():
            raise ValueError("loop_id must not be empty")
        if self.iteration < 0:
            raise ValueError("iteration must not be negative")
        if not isinstance(self.status, SchedulerStatus):
            object.__setattr__(self, "status", SchedulerStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "task": self.task,
            "status": self.status.value,
            "iteration": self.iteration,
            "output": self.output,
            "reason": self.reason,
            "usage": self.usage.to_dict(),
            "consecutive_failures": self.consecutive_failures,
            "history": [summary.to_dict() for summary in self.history],
            "suspension": self.suspension.to_dict() if self.suspension else None,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoopState:
        suspension_payload = payload.get("suspension")
        suspension = None
        if isinstance(suspension_payload, dict):
            suspension = Suspension(
                kind=str(suspension_payload["kind"]),
                reason=str(suspension_payload.get("reason", "")),
                payload=dict(suspension_payload.get("payload") or {}),
            )
        return cls(
            loop_id=str(payload["loop_id"]),
            task=str(payload.get("task", "")),
            status=SchedulerStatus(payload.get("status", SchedulerStatus.RUNNING.value)),
            iteration=int(payload.get("iteration", 0)),
            output=str(payload.get("output", "")),
            reason=str(payload.get("reason", "")),
            usage=Usage(**dict(payload.get("usage") or {})),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            history=tuple(
                StepSummary.from_dict(item)
                for item in payload.get("history") or ()
            ),
            suspension=suspension,
            started_at=float(payload.get("started_at", 0.0)),
            updated_at=float(payload.get("updated_at", 0.0)),
        )


@dataclass(frozen=True)
class LoopSnapshot:
    task: str
    iteration: int
    output: str
    usage: Usage
    consecutive_failures: int
    history: tuple[StepSummary, ...]


@dataclass(frozen=True)
class LoopBudget:
    max_iterations: int = 10
    max_tokens: int | None = None
    max_duration_seconds: float | None = None
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1 when set")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive when set")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str = ""
    output: str | None = None

    @classmethod
    def done(cls, reason: str = "goal", output: str | None = None) -> StopDecision:
        return cls(True, reason, output)

    @classmethod
    def continue_(cls, reason: str = "") -> StopDecision:
        return cls(False, reason)


@dataclass(frozen=True)
class HumanDecision:
    action: HumanAction
    reason: str = ""

    @classmethod
    def proceed(cls) -> HumanDecision:
        return cls(HumanAction.PROCEED)

    @classmethod
    def pause(cls, reason: str = "human_review") -> HumanDecision:
        return cls(HumanAction.PAUSE, reason)

    @classmethod
    def stop(cls, reason: str = "human_stop") -> HumanDecision:
        return cls(HumanAction.STOP, reason)


class LoopWorker(Protocol):
    def __call__(
        self,
        task: str,
        iteration: int,
        state: LoopState,
    ) -> StepResult: ...


class LoopStopGate(Protocol):
    def check(self, snapshot: LoopSnapshot) -> StopDecision: ...


class LoopHumanGate(Protocol):
    def check(self, snapshot: LoopSnapshot) -> HumanDecision: ...


class LoopTrigger(Protocol):
    def should_start(self, now: float, last_started_at: float | None) -> bool: ...


@dataclass(frozen=True)
class StopWhenContains:
    text: str
    reason: str = "goal"

    def check(self, snapshot: LoopSnapshot) -> StopDecision:
        if self.text in snapshot.output:
            return StopDecision.done(self.reason, snapshot.output)
        return StopDecision.continue_()


@dataclass(frozen=True)
class StopWhen:
    predicate: Callable[[LoopSnapshot], bool]
    reason: str = "goal"

    def check(self, snapshot: LoopSnapshot) -> StopDecision:
        if self.predicate(snapshot):
            return StopDecision.done(self.reason, snapshot.output)
        return StopDecision.continue_()


@dataclass(frozen=True)
class NeverStop:
    def check(self, snapshot: LoopSnapshot) -> StopDecision:
        return StopDecision.continue_()


@dataclass(frozen=True)
class PauseWhenContains:
    text: str
    reason: str = "human_review"

    def check(self, snapshot: LoopSnapshot) -> HumanDecision:
        if self.text in snapshot.output:
            return HumanDecision.pause(self.reason)
        return HumanDecision.proceed()


@dataclass(frozen=True)
class PauseWhen:
    predicate: Callable[[LoopSnapshot], bool]
    reason: str = "human_review"

    def check(self, snapshot: LoopSnapshot) -> HumanDecision:
        if self.predicate(snapshot):
            return HumanDecision.pause(self.reason)
        return HumanDecision.proceed()


@dataclass(frozen=True)
class NeverPause:
    def check(self, snapshot: LoopSnapshot) -> HumanDecision:
        return HumanDecision.proceed()


@dataclass(frozen=True)
class ManualTrigger:
    def should_start(self, now: float, last_started_at: float | None) -> bool:
        return True


@dataclass(frozen=True)
class IntervalTrigger:
    seconds: float

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("interval must be positive")

    def should_start(self, now: float, last_started_at: float | None) -> bool:
        return last_started_at is None or now - last_started_at >= self.seconds


@dataclass(frozen=True)
class EventTrigger:
    predicate: Callable[[], bool]

    def should_start(self, now: float, last_started_at: float | None) -> bool:
        return bool(self.predicate())


@dataclass(frozen=True)
class AgentWorker:
    """Adapt a single harness run into one outer-loop iteration."""

    runner: Callable[[str], RunResult]

    def __call__(
        self,
        task: str,
        iteration: int,
        state: LoopState,
    ) -> StepResult:
        result = self.runner(task)
        completed = result.stop_reason is StopReason.COMPLETED
        return StepResult(
            ok=completed,
            output=result.output or "",
            usage=result.usage,
            error=None if completed else result.stop_reason.value,
            metadata={
                "run_id": result.run_id,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "stop_reason": result.stop_reason.value,
            },
        )


@dataclass(frozen=True)
class SchedulerResult:
    loop_id: str
    status: SchedulerStatus
    iterations: int
    usage: Usage
    output: str
    reason: str
    events: tuple[Any, ...]
    state: LoopState | None = None