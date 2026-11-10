"""Run lifecycle: status, cancellation, checkpoints, resume, SSE (I4)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .loop import GateDecision, RunConfig, StopContext, StopReason, require_answer
from .messages import Message, ToolCall, Usage
from .model import Model
from .tools.errors import ToolOutcome, ToolSuspended


class RunStatus(str, Enum):
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def resumable(self) -> bool:
        return self in (RunStatus.RUNNING, RunStatus.SUSPENDED)


class CheckpointError(Exception):
    pass


class RunCancelled(Exception):
    def __init__(self, reason: str = "cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


SuspendRun = ToolSuspended


class CancelToken:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._reason = "cancelled"
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def on_cancel(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if not self._cancelled.is_set():
                self._callbacks.append(callback)
                return
        callback()

    def cancel(self, reason: str = "cancelled") -> bool:
        with self._lock:
            if self._cancelled.is_set():
                return False
            self._reason = reason
            self._cancelled.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001, S110
                pass
        return True


@dataclass(frozen=True)
class Suspension:
    kind: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("tool", "gate", "approval"):
            raise ValueError(f"unknown suspension kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "payload": self.payload}


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    status: RunStatus
    messages: tuple[Message, ...]
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    pending_call: ToolCall | None = None
    suspension: Suspension | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is RunStatus.SUSPENDED:
            if self.suspension is None:
                raise CheckpointError("suspended checkpoint requires a suspension")
        else:
            if self.suspension is not None:
                raise CheckpointError("only suspended checkpoints carry a suspension")
            if self.pending_call is not None:
                raise CheckpointError("only suspended checkpoints carry a pending call")
        if self.suspension is not None:
            if self.suspension.kind in ("tool", "approval") and self.pending_call is None:
                raise CheckpointError("tool suspension requires a pending call")
            if self.suspension.kind == "gate" and self.pending_call is not None:
                raise CheckpointError("gate suspension cannot carry a pending call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "messages": [message.to_dict() for message in self.messages],
            "turns": self.turns,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
            "pending_call": (
                {
                    "id": self.pending_call.id,
                    "name": self.pending_call.name,
                    "arguments": self.pending_call.arguments,
                }
                if self.pending_call is not None
                else None
            ),
            "suspension": self.suspension.to_dict() if self.suspension else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Checkpoint:
        pending = payload.get("pending_call")
        suspension = payload.get("suspension")
        return cls(
            run_id=payload["run_id"],
            status=RunStatus(payload["status"]),
            messages=tuple(
                Message.from_dict(item) for item in payload.get("messages", ())
            ),
            turns=int(payload.get("turns", 0)),
            usage=Usage(**payload.get("usage", {})),
            pending_call=ToolCall(**pending) if pending else None,
            suspension=Suspension(**suspension) if suspension else None,
            meta=dict(payload.get("meta", {})),
        )


class CheckpointStore:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"

    def save(self, checkpoint: Checkpoint) -> Path:
        path = self.path(checkpoint.run_id)
        handle, temp_name = tempfile.mkstemp(
            dir=self._directory,
            prefix=f".{checkpoint.run_id}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    checkpoint.to_dict(),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return path

    def load(self, run_id: str) -> Checkpoint:
        with open(self.path(run_id), encoding="utf-8") as stream:
            return Checkpoint.from_dict(json.load(stream))

    def latest(self) -> Checkpoint | None:
        files = sorted(
            self._directory.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        with open(files[0], encoding="utf-8") as stream:
            return Checkpoint.from_dict(json.load(stream))


def missing_calls(messages: Sequence[Message]) -> tuple[ToolCall, ...]:
    start = -1
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            start = index
    if start < 0:
        return ()
    calls = messages[start].tool_calls
    results: list[str] = []
    for message in messages[start + 1 :]:
        if message.role != "tool":
            break
        results.append(message.tool_call_id or "")
    for index, call_id in enumerate(results):
        if index >= len(calls) or call_id != calls[index].id:
            raise CheckpointError(
                "checkpoint tool results do not match the last assistant call batch"
            )
    if len(results) == len(calls):
        return ()
    if start + 1 + len(results) < len(messages):
        raise CheckpointError(
            "checkpoint ends a tool batch before all results are recorded"
        )
    return calls[len(results) :]


def _signature(calls: Sequence[ToolCall]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            call.name,
            json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str),
        )
        for call in calls
    )


@dataclass(frozen=True)
class LifecycleResult:
    run_id: str
    status: RunStatus
    stop_reason: StopReason | None
    turns: int
    tool_calls: int
    tool_attempts: int
    usage: Usage
    messages: tuple[Message, ...]
    output: str | None
    checkpoint: str | None = None


class LifecycleLoop:
    """A loop variant that checkpoints every message and supports suspend/resume."""

    def __init__(
        self,
        model: Model,
        tools: Any,
        *,
        stop_gate: Callable[[StopContext], GateDecision] = require_answer,
        config: RunConfig | None = None,
        store: CheckpointStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._stop_gate = stop_gate
        self._config = config or RunConfig()
        self._store = store
        self._run_id = run_id
        self.cancel_token = CancelToken()

    def cancel(self, reason: str = "cancelled") -> bool:
        return self.cancel_token.cancel(reason)

    def run(self, task: str, *, cancel: CancelToken | None = None) -> LifecycleResult:
        run_id = self._run_id or uuid.uuid4().hex[:12]
        token = cancel or self.cancel_token
        messages = [Message.user(task)]
        self._save(run_id, RunStatus.RUNNING, messages, 0, Usage())
        return self._drive(
            messages,
            token=token,
            run_id=run_id,
            turns=0,
            usage=Usage(),
        )

    def resume(
        self,
        run_id: str,
        value: str | None = None,
        *,
        cancel: CancelToken | None = None,
    ) -> LifecycleResult:
        if self._store is None:
            raise CheckpointError("resume requires a checkpoint store")
        checkpoint = self._store.load(run_id)
        if not checkpoint.status.resumable:
            raise CheckpointError(
                f"run {run_id} is {checkpoint.status.value}, not resumable"
            )
        token = cancel or self.cancel_token
        messages = list(checkpoint.messages)
        turns = checkpoint.turns
        usage = checkpoint.usage

        if checkpoint.suspension is not None:
            suspension = checkpoint.suspension
            if suspension.kind in ("tool", "approval"):
                pending = checkpoint.pending_call
                if pending is None:
                    raise CheckpointError("tool suspension has no pending call")
                outstanding = missing_calls(messages)
                if not outstanding or outstanding[0].id != pending.id:
                    raise CheckpointError(
                        "checkpoint pending call does not match the unfinished batch"
                    )
                if suspension.kind == "approval":
                    resolver = getattr(self._tools, "resolve_approval", None)
                    handled = (
                        resolver(pending.id, value or "") if resolver is not None else False
                    )
                    if handled:
                        try:
                            outcome = self._execute(pending, turns)
                        except ToolSuspended as exc:
                            new = Suspension(exc.kind, exc.reason, exc.payload)
                            path = self._save(
                                run_id, RunStatus.SUSPENDED, messages,
                                turns, usage, pending_call=pending, suspension=new,
                            )
                            return self._result(
                                run_id, RunStatus.SUSPENDED, None, turns,
                                0, 0, usage, messages, None, path,
                            )
                        messages.append(Message.tool_result(pending, outcome.content))
                    else:
                        approved = bool(value) and value.lower().startswith(
                            ("approve", "allow", "yes")
                        )
                        if approved:
                            try:
                                outcome = self._execute(pending, turns)
                            except ToolSuspended as exc:
                                new = Suspension(exc.kind, exc.reason, exc.payload)
                                path = self._save(
                                    run_id, RunStatus.SUSPENDED, messages,
                                    turns, usage, pending_call=pending, suspension=new,
                                )
                                return self._result(
                                    run_id, RunStatus.SUSPENDED, None, turns,
                                    0, 0, usage, messages, None, path,
                                )
                            messages.append(Message.tool_result(pending, outcome.content))
                        else:
                            messages.append(
                                Message.tool_result(
                                    pending, f"DENIED: {value or suspension.reason}"
                                )
                            )
                else:
                    messages.append(Message.tool_result(pending, value or ""))
            elif suspension.kind == "gate":
                messages.append(Message.user(value or ""))
            else:
                raise CheckpointError(f"unknown suspension kind: {suspension.kind}")

        self._save(run_id, RunStatus.RUNNING, messages, turns, usage)
        return self._drive(
            messages,
            token=token,
            run_id=run_id,
            turns=turns,
            usage=usage,
        )

    def _drive(
        self,
        messages: list[Message],
        *,
        token: CancelToken,
        run_id: str,
        turns: int,
        usage: Usage,
    ) -> LifecycleResult:
        tool_calls = 0
        tool_attempts = 0
        blocks = 0
        repeats = 0
        last_signature: tuple[tuple[str, str], ...] | None = None

        while True:
            if token.cancelled:
                for call in missing_calls(messages):
                    messages.append(
                        Message.tool_result(call, f"CANCELLED: {token.reason}")
                    )
                path = self._save(
                    run_id, RunStatus.CANCELLED, messages, turns, usage
                )
                return self._result(
                    run_id, RunStatus.CANCELLED, StopReason.CANCELLED,
                    turns, tool_calls, tool_attempts, usage, messages,
                    None, path,
                )

            outstanding = missing_calls(messages)
            if outstanding:
                signature = _signature(outstanding)
                if signature == last_signature:
                    repeats += 1
                else:
                    last_signature = signature
                    repeats = 1
                for position, call in enumerate(outstanding):
                    if token.cancelled:
                        for remaining in outstanding[position:]:
                            messages.append(
                                Message.tool_result(
                                    remaining, f"CANCELLED: {token.reason}"
                                )
                            )
                        path = self._save(
                            run_id, RunStatus.CANCELLED, messages, turns, usage
                        )
                        return self._result(
                            run_id, RunStatus.CANCELLED, StopReason.CANCELLED,
                            turns, tool_calls, tool_attempts, usage, messages,
                            None, path,
                        )
                    try:
                        outcome = self._execute(call, turns)
                    except ToolSuspended as exc:
                        suspension = Suspension(exc.kind, exc.reason, exc.payload)
                        path = self._save(
                            run_id, RunStatus.SUSPENDED, messages, turns, usage,
                            pending_call=call, suspension=suspension,
                        )
                        return self._result(
                            run_id, RunStatus.SUSPENDED, None, turns,
                            tool_calls, tool_attempts, usage, messages, None, path,
                        )
                    tool_calls += 1
                    tool_attempts += outcome.attempts
                    messages.append(Message.tool_result(call, outcome.content))
                    self._save(
                        run_id, RunStatus.RUNNING, messages, turns, usage
                    )
                if repeats >= self._config.no_progress_limit:
                    path = self._save(
                        run_id, RunStatus.FAILED, messages, turns, usage
                    )
                    return self._result(
                        run_id, RunStatus.FAILED, StopReason.NO_PROGRESS,
                        turns, tool_calls, tool_attempts, usage, messages,
                        None, path,
                    )
                continue

            if turns >= self._config.max_turns:
                path = self._save(run_id, RunStatus.FAILED, messages, turns, usage)
                return self._result(
                    run_id, RunStatus.FAILED, StopReason.MAX_TURNS,
                    turns, tool_calls, tool_attempts, usage, messages, None, path,
                )

            turns += 1
            try:
                response = self._model.complete(
                    tuple(messages), self._tools.specs()
                )
            except Exception:  # noqa: BLE001
                path = self._save(run_id, RunStatus.FAILED, messages, turns, usage)
                return self._result(
                    run_id, RunStatus.FAILED, StopReason.FAILED,
                    turns, tool_calls, tool_attempts, usage, messages, None, path,
                )

            usage = usage + response.usage
            messages.append(
                Message.assistant(response.content, response.tool_calls)
            )
            self._save(
                run_id, RunStatus.RUNNING, messages, turns, usage
            )

            if response.tool_calls:
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
            except Exception:  # noqa: BLE001
                path = self._save(run_id, RunStatus.FAILED, messages, turns, usage)
                return self._result(
                    run_id, RunStatus.FAILED, StopReason.FAILED,
                    turns, tool_calls, tool_attempts, usage, messages, None, path,
                )

            if decision.passed:
                path = self._save(
                    run_id, RunStatus.COMPLETED, messages, turns, usage
                )
                return self._result(
                    run_id, RunStatus.COMPLETED, StopReason.COMPLETED,
                    turns, tool_calls, tool_attempts, usage, messages,
                    response.content, path,
                )

            blocks += 1
            if blocks >= self._config.max_stop_blocks:
                path = self._save(
                    run_id, RunStatus.FAILED, messages, turns, usage
                )
                return self._result(
                    run_id, RunStatus.FAILED, StopReason.NO_PROGRESS,
                    turns, tool_calls, tool_attempts, usage, messages, None, path,
                )
            messages.append(
                Message.user(decision.feedback or "任务尚未完成，请继续。")
            )
            self._save(
                run_id, RunStatus.RUNNING, messages, turns, usage
            )

    def _execute(self, call: ToolCall, turn: int) -> ToolOutcome:
        return self._tools.execute(call, turn=turn)

    def _save(
        self,
        run_id: str,
        status: RunStatus,
        messages: Sequence[Message],
        turns: int,
        usage: Usage,
        *,
        pending_call: ToolCall | None = None,
        suspension: Suspension | None = None,
    ) -> str | None:
        if self._store is None:
            return None
        checkpoint = Checkpoint(
            run_id=run_id,
            status=status,
            messages=tuple(messages),
            turns=turns,
            usage=usage,
            pending_call=pending_call,
            suspension=suspension,
        )
        return str(self._store.save(checkpoint))

    @staticmethod
    def _result(
        run_id: str,
        status: RunStatus,
        stop_reason: StopReason | None,
        turns: int,
        tool_calls: int,
        tool_attempts: int,
        usage: Usage,
        messages: Sequence[Message],
        output: str | None,
        checkpoint: str | None,
    ) -> LifecycleResult:
        return LifecycleResult(
            run_id=run_id,
            status=status,
            stop_reason=stop_reason,
            turns=turns,
            tool_calls=tool_calls,
            tool_attempts=tool_attempts,
            usage=usage,
            messages=tuple(messages),
            output=output,
            checkpoint=checkpoint,
        )


@dataclass(frozen=True)
class SseEvent:
    id: int
    event: str
    data: str


def format_sse(event: SseEvent) -> str:
    return f"id: {event.id}\nevent: {event.event}\ndata: {event.data}\n\n"


def parse_sse(text: str) -> tuple[SseEvent, ...]:
    events: list[SseEvent] = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        if not fields:
            continue
        events.append(
            SseEvent(
                id=int(fields.get("id", "0")),
                event=fields.get("event", "message"),
                data=fields.get("data", ""),
            )
        )
    return tuple(events)


class StreamInterrupted(RuntimeError):
    pass


class EmptyStreamError(StreamInterrupted):
    pass


@dataclass(frozen=True)
class StreamPolicy:
    max_attempts: int = 3
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


def stream_with_retry(
    factory: Callable[[], Iterator[Any]],
    *,
    policy: StreamPolicy,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Iterator[Any]:
    for attempt in range(1, policy.max_attempts + 1):
        yielded = False
        try:
            for event in factory():
                yielded = True
                yield event
            return
        except Exception as exc:
            if yielded:
                raise StreamInterrupted(
                    "stream failed after the first event; partial output is not retried"
                ) from exc
            if not isinstance(exc, policy.retry_on):
                raise
            if attempt >= policy.max_attempts:
                raise
            if on_retry is not None:
                on_retry(exc, attempt)


def validate_stream_end(
    *,
    finish_reason: str | None,
    saw_done: bool,
    content: str | None,
    tool_calls: Sequence[ToolCall],
) -> None:
    if finish_reason is None and not saw_done:
        raise StreamInterrupted("stream ended before finish_reason or [DONE]")
    if finish_reason is None and not content and not tool_calls:
        raise EmptyStreamError("stream produced no content, tool calls or finish_reason")