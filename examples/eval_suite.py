"""Run an I7 fixture suite and print spans, metrics, and pass^k readings."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nano_harness.eval import (
    EvalCase,
    baseline_from_result,
    compare_to_baseline,
    format_suite,
    no_tool_errors,
    output_contains,
    retries_at_most,
    run_case,
    run_suite,
    save_baseline,
    stop_reason,
    turns_at_most,
)
from nano_harness.loop import RunConfig, StopReason
from nano_harness.messages import ToolCall
from nano_harness.model import ModelResponse
from nano_harness.observability import (
    build_spans,
    collect_metrics,
    transcript,
)
from nano_harness.tools import RetryableToolError, ToolExecutor, ToolSpec

EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def retry_case() -> EvalCase:
    def factory(trial: int) -> ToolExecutor:
        attempts = {"n": 0}

        def read() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RetryableToolError("HTTP 503")
            return "42"

        spec = ToolSpec(
            name="read_metric",
            description="read a metric",
            parameters=EMPTY_SCHEMA,
            handler=read,
            read_only=True,
            max_attempts=3,
        )
        return ToolExecutor([spec], run_id=f"retry-{trial}")

    return EvalCase(
        name="retry-budget",
        task="read the metric",
        responses=(
            ModelResponse(
                tool_calls=(ToolCall("c1", "read_metric", {}),)
            ),
            ModelResponse(content="42"),
        ),
        executor_factory=factory,
        checks=(
            stop_reason(StopReason.COMPLETED),
            retries_at_most(2),
            no_tool_errors(),
            output_contains("42"),
        ),
    )


def no_progress_case() -> EvalCase:
    def factory(trial: int) -> ToolExecutor:
        spec = ToolSpec(
            name="echo",
            description="echo",
            parameters=ECHO_SCHEMA,
            handler=lambda text: text,
            read_only=True,
        )
        return ToolExecutor([spec], run_id=f"no-progress-{trial}")

    responses = tuple(
        ModelResponse(
            tool_calls=(ToolCall("c1", "echo", {"text": "same"}),)
        )
        for _ in range(3)
    )
    return EvalCase(
        name="no-progress",
        task="loop forever",
        responses=responses,
        executor_factory=factory,
        config=RunConfig(no_progress_limit=3),
        checks=(
            stop_reason(StopReason.NO_PROGRESS),
            turns_at_most(3),
        ),
    )


def flaky_case() -> EvalCase:
    def factory(trial: int) -> ToolExecutor:
        def echo(text: str) -> str:
            if trial % 2 == 0:
                raise RetryableToolError("HTTP 503")
            return text

        spec = ToolSpec(
            name="echo",
            description="echo",
            parameters=ECHO_SCHEMA,
            handler=echo,
            read_only=True,
        )
        return ToolExecutor([spec], run_id=f"flaky-{trial}")

    return EvalCase(
        name="flaky-tool",
        task="echo hi",
        responses=(
            ModelResponse(tool_calls=(ToolCall("c1", "echo", {"text": "hi"}),)),
            ModelResponse(content="done"),
        ),
        executor_factory=factory,
        checks=(
            stop_reason(StopReason.COMPLETED),
            no_tool_errors(),
            output_contains("done"),
        ),
    )


def main() -> None:
    cases = [retry_case(), no_progress_case(), flaky_case()]
    suite = run_suite("harness-smoke", cases, repeats=4)
    print(format_suite(suite))

    baseline = baseline_from_result(suite)
    report = compare_to_baseline(suite, baseline)
    print("baseline:", report.to_dict())

    with tempfile.TemporaryDirectory() as directory:
        path = save_baseline(baseline, Path(directory) / "baseline.json")
        print("baseline file:", path.name)

    result = run_case(retry_case(), trial=0)
    events = result.result.events if result.result else ()
    metrics = collect_metrics(events, usage=result.result.usage if result.result else None)
    print("metrics:", metrics.to_dict())
    print("spans:")
    for span in build_spans(events, trace_id="demo"):
        print(
            f"  {span.span_id} {span.name} "
            f"{span.duration_ms:.2f}ms parent={span.parent_id}"
        )
    print("transcript lines:", len(transcript(events).splitlines()))


if __name__ == "__main__":
    main()