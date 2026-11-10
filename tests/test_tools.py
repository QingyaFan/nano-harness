import json

import pytest

from nano_harness.messages import ToolCall
from nano_harness.policy.types import RiskLevel
from nano_harness.tools import (
    RetryableToolError,
    ToolExecutor,
    ToolSpec,
    current_tool_context,
)

METRIC_SCHEMA = {
    "type": "object",
    "properties": {"service": {"type": "string"}},
    "required": ["service"],
    "additionalProperties": False,
}


def make_executor(spec: ToolSpec) -> ToolExecutor:
    return ToolExecutor([spec], run_id="run-1")


def test_missing_argument_returns_structured_envelope() -> None:
    spec = ToolSpec(
        name="read_metric",
        description="read a metric",
        parameters=METRIC_SCHEMA,
        handler=lambda service: "42",
        read_only=True,
    )
    outcome = make_executor(spec).execute(ToolCall("c1", "read_metric", {}))

    assert outcome.ok is False
    assert outcome.attempts == 0
    payload = json.loads(outcome.content)
    assert payload["error"]["kind"] == "invalid_arguments"
    assert payload["error"]["retryable"] is False
    assert payload["error"]["message"] == "<root>: 'service' is a required property"
    assert "必填参数：service" in payload["error"]["hint"]


def test_retryable_error_is_retried_with_budget() -> None:
    attempts = {"n": 0}

    def flaky(service: str) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableToolError("HTTP 503")
        return "42"

    spec = ToolSpec(
        name="read_metric",
        description="read a metric",
        parameters=METRIC_SCHEMA,
        handler=flaky,
        read_only=True,
        max_attempts=3,
    )
    retries: list[int] = []
    outcome = make_executor(spec).execute(
        ToolCall("c1", "read_metric", {"service": "api"}),
        on_retry=lambda error, attempt: retries.append(attempt),
    )

    assert outcome.ok is True
    assert outcome.content == "42"
    assert outcome.attempts == 3
    assert retries == [1, 2]
    assert attempts["n"] == 3


def test_idempotent_write_replays_without_side_effect() -> None:
    writes: list[str] = []

    def store(value: str) -> str:
        writes.append(value)
        return "ok"

    spec = ToolSpec(
        name="store",
        description="store a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=store,
        idempotent=True,
        max_attempts=3,
        risk=RiskLevel.WRITE,
    )
    executor = make_executor(spec)
    call = ToolCall("c1", "store", {"value": "v1"})

    first = executor.execute(call)
    second = executor.execute(call)

    assert first.ok and second.ok
    assert writes == ["v1"]
    assert second.replayed is True
    assert second.attempts == 0


def test_tool_context_exposes_idempotency_key() -> None:
    seen: list[str | None] = []

    def store(value: str) -> str:
        seen.append(current_tool_context().idempotency_key)
        return "ok"

    spec = ToolSpec(
        name="store",
        description="store a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=store,
        idempotent=True,
        risk=RiskLevel.WRITE,
    )
    outcome = make_executor(spec).execute(ToolCall("c1", "store", {"value": "v1"}))

    assert outcome.ok is True
    assert seen[0] is not None
    assert len(seen[0]) == 64


def test_non_idempotent_write_cannot_retry_at_registration() -> None:
    with pytest.raises(ValueError):
        ToolSpec(
            name="store",
            description="store a value",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",
            max_attempts=2,
            risk=RiskLevel.WRITE,
        )


def test_unknown_tool_returns_envelope() -> None:
    executor = ToolExecutor([], run_id="run-1")
    outcome = executor.execute(ToolCall("c1", "missing", {}))

    assert outcome.ok is False
    payload = json.loads(outcome.content)
    assert payload["error"]["kind"] == "unknown_tool"
    assert outcome.attempts == 0


def test_invalid_json_arguments_are_reported() -> None:
    spec = ToolSpec(
        name="read_metric",
        description="read a metric",
        parameters=METRIC_SCHEMA,
        handler=lambda service: "42",
        read_only=True,
    )
    outcome = make_executor(spec).execute(
        ToolCall("c1", "read_metric", raw_arguments="{not json")
    )

    assert outcome.ok is False
    assert json.loads(outcome.content)["error"]["kind"] == "invalid_json"


def test_non_string_result_is_invalid() -> None:
    spec = ToolSpec(
        name="read_metric",
        description="read a metric",
        parameters=METRIC_SCHEMA,
        handler=lambda service: 42,
        read_only=True,
    )
    outcome = make_executor(spec).execute(
        ToolCall("c1", "read_metric", {"service": "api"})
    )

    assert outcome.ok is False
    assert json.loads(outcome.content)["error"]["kind"] == "invalid_result"
    assert outcome.attempts == 1