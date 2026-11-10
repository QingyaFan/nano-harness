"""Observability: spans, metrics, and transcripts from runtime events (I7)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

from .events import (
    CONTEXT_PREPARED,
    MODEL_FAILED,
    MODEL_RESPONSE,
    RUN_FINISHED,
    RUN_STARTED,
    TOOL_CALL,
    TOOL_RESULT,
    TOOL_RETRY,
    TURN_STARTED,
    Event,
)
from .messages import Usage

SPAN_RUN = "invoke_agent harness"
SPAN_TURN = "harness.turn"
SPAN_MODEL = "gen_ai.chat"
SPAN_CONTEXT = "harness.context"
SPAN_TOOL = "execute_tool"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_TOOL = "gen_ai.tool.name"


@dataclass(frozen=True)
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float
    end: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end - self.start) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
        }


def build_spans(events: Sequence[Event], trace_id: str = "trace") -> tuple[Span, ...]:
    items = list(events)
    if not items:
        return ()

    ids = count(1)

    def new_id() -> str:
        return f"span-{next(ids):04d}"

    run_span_id = new_id()
    start = items[0].timestamp
    end = items[-1].timestamp
    spans: list[Span] = []
    turn: dict[str, Any] | None = None

    def close_turn(at: float) -> None:
        nonlocal turn
        if turn is None:
            return
        spans.append(
            Span(
                turn["id"],
                run_span_id,
                SPAN_TURN,
                "turn",
                turn["start"],
                max(turn["start"], at),
                {"turn": turn["number"]},
            )
        )
        turn = None

    for event in items:
        moment = event.timestamp
        if event.type == TURN_STARTED:
            close_turn(moment)
            turn = {
                "id": new_id(),
                "start": moment,
                "number": event.turn,
                "model_start": moment,
                "tools": {},
            }
            continue
        if turn is None:
            continue
        if event.type == CONTEXT_PREPARED:
            spans.append(
                Span(
                    new_id(),
                    turn["id"],
                    SPAN_CONTEXT,
                    "context",
                    turn["start"],
                    moment,
                    {
                        "used": event.data.get("used"),
                        "records": list(event.data.get("records", ())),
                        "exhausted": event.data.get("exhausted"),
                    },
                )
            )
            turn["model_start"] = moment
            continue
        if event.type in (MODEL_RESPONSE, MODEL_FAILED):
            attributes = {
                GEN_AI_OPERATION: "chat",
                "content_length": len(event.data.get("content") or ""),
                "tool_calls": len(event.data.get("tool_calls", ())),
            }
            if "usage" in event.data:
                attributes["usage"] = event.data["usage"]
            if event.type == MODEL_FAILED:
                attributes["error"] = event.data.get("error")
                attributes["message"] = event.data.get("message")
            spans.append(
                Span(
                    new_id(),
                    turn["id"],
                    SPAN_MODEL,
                    "model",
                    turn.get("model_start", turn["start"]),
                    moment,
                    attributes,
                )
            )
            continue
        if event.type == TOOL_CALL:
            turn["tools"][event.data.get("call_id")] = (
                moment,
                event.data.get("name", "tool"),
            )
            continue
        if event.type == TOOL_RESULT:
            call_id = event.data.get("call_id")
            tool_start, tool_name = turn["tools"].pop(call_id, (moment, "tool"))
            spans.append(
                Span(
                    new_id(),
                    turn["id"],
                    f"{SPAN_TOOL} {tool_name}",
                    "tool",
                    tool_start,
                    moment,
                    {
                        GEN_AI_OPERATION: "execute_tool",
                        GEN_AI_TOOL: tool_name,
                        "ok": event.data.get("ok"),
                        "attempts": event.data.get("attempts"),
                        "replayed": event.data.get("replayed"),
                        "error_kind": event.data.get("error_kind"),
                    },
                )
            )
            continue
        if event.type == RUN_FINISHED:
            close_turn(moment)
            end = moment

    close_turn(end)

    attributes: dict[str, Any] = {"trace_id": trace_id}
    if items[0].type == RUN_STARTED:
        attributes["task"] = items[0].data.get("task")
    finish = next(
        (item for item in reversed(items) if item.type == RUN_FINISHED), None
    )
    if finish is not None:
        attributes.update(finish.data)
    spans.append(Span(run_span_id, None, SPAN_RUN, "run", start, end, attributes))
    return tuple(sorted(spans, key=lambda span: (span.start, span.span_id)))


@dataclass(frozen=True)
class RunMetrics:
    turns: int = 0
    tool_calls: int = 0
    tool_attempts: int = 0
    retries: int = 0
    tool_errors: int = 0
    compactions: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_attempts": self.tool_attempts,
            "retries": self.retries,
            "tool_errors": self.tool_errors,
            "compactions": dict(self.compactions),
            "stop_reason": self.stop_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": round(self.duration_ms, 3),
        }


def collect_metrics(
    events: Sequence[Event],
    *,
    usage: Usage | None = None,
) -> RunMetrics:
    items = list(events)
    if not items:
        return RunMetrics()

    compactions: Counter[str] = Counter()
    stop_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    for event in items:
        if event.type == CONTEXT_PREPARED:
            compactions.update(event.data.get("records", ()))
        elif event.type == RUN_FINISHED:
            stop_reason = event.data.get("stop_reason")
        elif event.type == MODEL_RESPONSE:
            event_usage = event.data.get("usage") or {}
            prompt_tokens += int(event_usage.get("prompt_tokens", 0))
            completion_tokens += int(event_usage.get("completion_tokens", 0))

    if usage is not None:
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

    return RunMetrics(
        turns=sum(1 for event in items if event.type == TURN_STARTED),
        tool_calls=sum(1 for event in items if event.type == TOOL_CALL),
        tool_attempts=sum(
            int(event.data.get("attempts", 0))
            for event in items
            if event.type == TOOL_RESULT
        ),
        retries=sum(1 for event in items if event.type == TOOL_RETRY),
        tool_errors=sum(
            1
            for event in items
            if event.type == TOOL_RESULT and event.data.get("ok") is False
        ),
        compactions=dict(compactions),
        stop_reason=stop_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=max(0.0, (items[-1].timestamp - items[0].timestamp) * 1000),
    )


def transcript(events: Iterable[Event]) -> str:
    lines = [
        json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        for event in events
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_transcript(events: Iterable[Event], path: str | Path) -> Path:
    target = Path(path)
    target.write_text(transcript(events), encoding="utf-8")
    return target


def read_transcript(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]