"""Structured events emitted by the runtime."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

RUN_STARTED = "run_started"
TURN_STARTED = "turn_started"
CONTEXT_PREPARED = "context_prepared"
MODEL_RESPONSE = "model_response"
MODEL_FAILED = "model_failed"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
TOOL_RETRY = "tool_retry"
STOP_GATE = "stop_gate"
RUN_FINISHED = "run_finished"
MEMORY_WRITTEN = "memory_written"
MEMORY_DUPLICATE = "memory_duplicate"
MEMORY_CONFLICT = "memory_conflict"
MEMORY_RECALLED = "memory_recalled"
MEMORY_FORGOTTEN = "memory_forgotten"
MEMORY_EXPIRED = "memory_expired"
PROMPT_ASSEMBLED = "prompt_assembled"
PROMPT_ASSERTION_FAILED = "prompt_assertion_failed"
PROMPT_INJECTION_SANITIZED = "prompt_injection_sanitized"


@dataclass(frozen=True)
class Event:
    type: str
    turn: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "turn": self.turn,
            "data": dict(self.data),
            "timestamp": self.timestamp,
        }


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def emit(self, event_type: str, turn: int = 0, **data: Any) -> Event:
        event = Event(event_type, turn, data)
        self._events.append(event)
        return event

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def snapshot(self) -> tuple[Event, ...]:
        return tuple(self._events)