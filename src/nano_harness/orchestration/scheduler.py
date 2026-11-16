"""A1 outer loop: triggers, stop gates, budgets, human gates, and state."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..events import EventLog
from ..lifecycle import Suspension
from .model import (
    LOOP_FINISHED,
    LOOP_ITERATION,
    LOOP_RESUMED,
    LOOP_SKIPPED,
    LOOP_STARTED,
    LOOP_STOP_GATE,
    LOOP_SUSPENDED,
    HumanAction,
    LoopBudget,
    LoopError,
    LoopHumanGate,
    LoopSnapshot,
    LoopState,
    LoopStopGate,
    LoopTrigger,
    LoopWorker,
    ManualTrigger,
    NeverPause,
    SchedulerResult,
    SchedulerStatus,
    StepResult,
    StepSummary,
)
from .state import LoopStateStore


def _cancelled(token: Any) -> bool:
    return bool(token is not None and getattr(token, "cancelled", False))


class Scheduler:
    """Run an outer loop around a worker until a stop gate or budget wins."""

    def __init__(
        self,
        worker: LoopWorker,
        *,
        stop_gate: LoopStopGate,
        human_gate: LoopHumanGate | None = None,
        budget: LoopBudget | None = None,
        store: LoopStateStore | None = None,
        trigger: LoopTrigger | None = None,
        loop_id: str | None = None,
        events: EventLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._worker = worker
        self._stop_gate = stop_gate
        self._human_gate = human_gate or NeverPause()
        self._budget = budget or LoopBudget()
        self._store = store or LoopStateStore()
        self._trigger = trigger or ManualTrigger()
        self._loop_id = loop_id or uuid.uuid4().hex[:12]
        self._events = events
        self._clock = clock
        if not self._loop_id.strip():
            raise ValueError("loop_id must not be empty")

    @property
    def loop_id(self) -> str:
        return self._loop_id

    def should_start(self, now: float | None = None) -> bool:
        moment = self._clock() if now is None else now
        state = self._store.load(self._loop_id)
        last_started = state.started_at if state is not None else None
        return self._trigger.should_start(moment, last_started)

    def run(
        self,
        task: str | None = None,
        *,
        cancel: Any = None,
        now: float | None = None,
    ) -> SchedulerResult:
        moment = self._clock() if now is None else now
        log = self._events or EventLog()
        state = self._store.load(self._loop_id)

        if state is not None and state.status is SchedulerStatus.SUSPENDED:
            raise LoopError("loop is suspended; call resume() instead of run()")

        if not self._trigger.should_start(
            moment, state.started_at if state is not None else None
        ):
            skipped = LoopState(
                loop_id=self._loop_id,
                task=task or (state.task if state is not None else ""),
                status=SchedulerStatus.SKIPPED,
                reason="trigger",
                started_at=moment,
                updated_at=moment,
            )
            log.emit(LOOP_SKIPPED, 0, loop_id=self._loop_id, reason="trigger")
            return SchedulerResult(
                loop_id=self._loop_id,
                status=skipped.status,
                iterations=0,
                usage=skipped.usage,
                output=skipped.output,
                reason=skipped.reason,
                events=log.snapshot(),
                state=skipped,
            )

        if state is None or state.status is not SchedulerStatus.RUNNING:
            state = LoopState(
                loop_id=self._loop_id,
                task=task or "",
                status=SchedulerStatus.RUNNING,
                started_at=moment,
                updated_at=moment,
            )
        else:
            state = replace(
                state,
                task=task or state.task,
                status=SchedulerStatus.RUNNING,
                updated_at=moment,
            )

        self._store.save(state)
        log.emit(LOOP_STARTED, 0, loop_id=self._loop_id, task=state.task)
        return self._drive(state, log, cancel)

    def resume(self, *, approved: bool, cancel: Any = None) -> SchedulerResult:
        log = self._events or EventLog()
        state = self._store.load(self._loop_id)
        if state is None or state.status is not SchedulerStatus.SUSPENDED:
            raise LoopError("loop is not suspended")

        if not approved:
            return self._finish(
                replace(state, suspension=None),
                SchedulerStatus.HUMAN_REJECTED,
                "human_rejected",
                log,
            )

        state = replace(
            state,
            status=SchedulerStatus.RUNNING,
            suspension=None,
            reason="",
            updated_at=self._clock(),
        )
        self._store.save(state)
        log.emit(
            LOOP_RESUMED,
            state.iteration,
            loop_id=self._loop_id,
            iteration=state.iteration,
        )
        return self._drive(state, log, cancel)

    def _drive(
        self,
        state: LoopState,
        log: EventLog,
        cancel: Any,
    ) -> SchedulerResult:
        while True:
            if _cancelled(cancel):
                return self._finish(
                    state, SchedulerStatus.CANCELLED, "cancelled", log
                )

            moment = self._clock()
            if state.iteration >= self._budget.max_iterations:
                return self._finish(
                    state, SchedulerStatus.MAX_ITERATIONS, "iteration_budget", log
                )
            if self._duration_exceeded(state, moment):
                return self._finish(
                    state, SchedulerStatus.BUDGET_EXHAUSTED, "duration_budget", log
                )

            iteration = state.iteration + 1
            try:
                step = self._worker(state.task, iteration, state)
            except Exception as exc:  # noqa: BLE001
                step = StepResult.failure(
                    f"worker_error:{type(exc).__name__}",
                    metadata={"message": str(exc)},
                )
            if not isinstance(step, StepResult):
                step = StepResult.failure("worker_error:invalid_result")

            summary = StepSummary(
                iteration=iteration,
                ok=step.ok,
                output=step.output,
                usage=step.usage,
                error=step.error,
                metadata=dict(step.metadata),
            )
            failures = 0 if step.ok else state.consecutive_failures + 1
            state = replace(
                state,
                iteration=iteration,
                output=step.output or state.output,
                usage=state.usage + step.usage,
                consecutive_failures=failures,
                history=state.history + (summary,),
                updated_at=moment,
            )
            self._store.save(state)
            log.emit(
                LOOP_ITERATION,
                iteration,
                loop_id=self._loop_id,
                iteration=iteration,
                ok=step.ok,
                output=step.output,
                error=step.error,
                usage=step.usage.to_dict(),
            )

            if failures >= self._budget.max_consecutive_failures:
                return self._finish(
                    state, SchedulerStatus.FAILED, "consecutive_failures", log
                )
            if (
                self._budget.max_tokens is not None
                and state.usage.total_tokens > self._budget.max_tokens
            ):
                return self._finish(
                    state, SchedulerStatus.BUDGET_EXHAUSTED, "token_budget", log
                )
            if self._duration_exceeded(state, self._clock()):
                return self._finish(
                    state, SchedulerStatus.BUDGET_EXHAUSTED, "duration_budget", log
                )

            snapshot = self._snapshot(state)
            try:
                decision = self._stop_gate.check(snapshot)
            except Exception as exc:  # noqa: BLE001
                log.emit(
                    LOOP_STOP_GATE,
                    iteration,
                    loop_id=self._loop_id,
                    passed=False,
                    reason="stop_gate_error",
                    error=type(exc).__name__,
                    message=str(exc),
                )
                return self._finish(
                    state, SchedulerStatus.FAILED, "stop_gate_error", log
                )

            log.emit(
                LOOP_STOP_GATE,
                iteration,
                loop_id=self._loop_id,
                stop=decision.stop,
                reason=decision.reason,
            )
            if decision.stop:
                return self._finish(
                    state,
                    SchedulerStatus.COMPLETED,
                    decision.reason or "goal",
                    log,
                    output=decision.output,
                )

            try:
                human = self._human_gate.check(snapshot)
            except Exception as exc:  # noqa: BLE001
                log.emit(
                    LOOP_SUSPENDED,
                    iteration,
                    loop_id=self._loop_id,
                    reason="human_gate_error",
                    error=type(exc).__name__,
                    message=str(exc),
                )
                return self._finish(
                    state, SchedulerStatus.FAILED, "human_gate_error", log
                )

            if human.action is HumanAction.STOP:
                return self._finish(
                    state,
                    SchedulerStatus.HUMAN_REJECTED,
                    human.reason or "human_stop",
                    log,
                )
            if human.action is HumanAction.PAUSE:
                suspension = Suspension(
                    kind="approval",
                    reason=human.reason or "human_review",
                    payload={
                        "iteration": state.iteration,
                        "output": state.output,
                    },
                )
                state = replace(
                    state,
                    status=SchedulerStatus.SUSPENDED,
                    suspension=suspension,
                    reason=suspension.reason,
                    updated_at=self._clock(),
                )
                self._store.save(state)
                log.emit(
                    LOOP_SUSPENDED,
                    state.iteration,
                    loop_id=self._loop_id,
                    reason=suspension.reason,
                )
                return SchedulerResult(
                    loop_id=self._loop_id,
                    status=state.status,
                    iterations=state.iteration,
                    usage=state.usage,
                    output=state.output,
                    reason=state.reason,
                    events=log.snapshot(),
                    state=state,
                )

    def _finish(
        self,
        state: LoopState,
        status: SchedulerStatus,
        reason: str,
        log: EventLog,
        *,
        output: str | None = None,
    ) -> SchedulerResult:
        state = replace(
            state,
            status=status,
            reason=reason,
            output=state.output if output is None else output,
            updated_at=self._clock(),
        )
        self._store.save(state)
        log.emit(
            LOOP_FINISHED,
            state.iteration,
            loop_id=self._loop_id,
            status=status.value,
            reason=reason,
            usage=state.usage.to_dict(),
        )
        return SchedulerResult(
            loop_id=self._loop_id,
            status=status,
            iterations=state.iteration,
            usage=state.usage,
            output=state.output,
            reason=reason,
            events=log.snapshot(),
            state=state,
        )

    def _duration_exceeded(self, state: LoopState, now: float) -> bool:
        limit = self._budget.max_duration_seconds
        return limit is not None and now - state.started_at >= limit

    @staticmethod
    def _snapshot(state: LoopState) -> LoopSnapshot:
        return LoopSnapshot(
            task=state.task,
            iteration=state.iteration,
            output=state.output,
            usage=state.usage,
            consecutive_failures=state.consecutive_failures,
            history=state.history,
        )