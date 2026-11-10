import json

from nano_harness.loop import HarnessLoop, StopReason
from nano_harness.messages import ToolCall, Usage
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.observability import (
    build_spans,
    collect_metrics,
    read_transcript,
    write_transcript,
)
from nano_harness.tools import RetryableToolError, ToolExecutor, ToolSpec

ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def echo_executor(handler=None) -> ToolExecutor:
    spec = ToolSpec(
        name="echo",
        description="echo",
        parameters=ECHO_SCHEMA,
        handler=handler or (lambda text: text),
        read_only=True,
    )
    return ToolExecutor([spec], run_id="obs")


def test_spans_metrics_and_transcript(tmp_path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall("c1", "echo", {"text": "hi"}),),
                usage=Usage(10, 2),
            ),
            ModelResponse(content="done", usage=Usage(5, 1)),
        ]
    )
    result = HarnessLoop(model, echo_executor(), run_id="obs").run("go")

    spans = build_spans(result.events, trace_id="t1")
    kinds = {span.kind for span in spans}
    assert {"run", "turn", "model", "tool"} <= kinds

    run = next(span for span in spans if span.kind == "run")
    turn = next(span for span in spans if span.kind == "turn")
    tool = next(span for span in spans if span.kind == "tool")
    assert run.parent_id is None
    assert turn.parent_id == run.span_id
    assert tool.parent_id == turn.span_id
    assert tool.attributes["gen_ai.tool.name"] == "echo"
    assert any(span.name.startswith("execute_tool ") for span in spans)

    metrics = collect_metrics(result.events, usage=result.usage)
    assert metrics.turns == 2
    assert metrics.tool_calls == 1
    assert metrics.tool_attempts == 1
    assert metrics.retries == 0
    assert metrics.tool_errors == 0
    assert metrics.stop_reason == StopReason.COMPLETED.value
    assert metrics.total_tokens == 18

    path = write_transcript(result.events, tmp_path / "run.jsonl")
    lines = read_transcript(path)
    assert len(lines) == len(result.events)
    assert lines[0]["type"] == "run_started"
    assert all("timestamp" in line for line in lines)


def test_metrics_count_retries_and_errors() -> None:
    calls = {"n": 0}

    def flaky(text: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableToolError("HTTP 503")
        return text

    spec = ToolSpec(
        name="echo",
        description="echo",
        parameters=ECHO_SCHEMA,
        handler=flaky,
        read_only=True,
        max_attempts=3,
    )
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "echo", {"text": "42"}),)),
            ModelResponse(content="done"),
        ]
    )
    result = HarnessLoop(
        model, ToolExecutor([spec], run_id="retry"), run_id="retry"
    ).run("go")
    metrics = collect_metrics(result.events, usage=result.usage)

    assert metrics.tool_attempts == 3
    assert metrics.retries == 2
    assert metrics.tool_errors == 0


def test_transcript_is_jsonl() -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    result = HarnessLoop(model, echo_executor(), run_id="t").run("go")
    from nano_harness.observability import transcript

    text = transcript(result.events)
    assert text.endswith("\n")
    parsed = [json.loads(line) for line in text.splitlines()]
    assert parsed[-1]["type"] == "run_finished"