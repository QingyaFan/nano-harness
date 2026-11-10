import pytest

from nano_harness.loop import (
    GateDecision,
    HarnessLoop,
    RunConfig,
    StopReason,
    require_answer,
)
from nano_harness.messages import Message, ToolCall
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.tools import RetryableToolError, ToolExecutor, ToolSpec

ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def make_executor(handler=None, **kwargs) -> ToolExecutor:
    if handler is None:
        handler = lambda text: text
    spec = ToolSpec(
        name="echo",
        description="echo text",
        parameters=ECHO_SCHEMA,
        handler=handler,
        read_only=True,
        **kwargs,
    )
    return ToolExecutor([spec], run_id="test")


def test_gate_blocks_plan_and_loop_completes() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="计划：先回显 hi"),
            ModelResponse(tool_calls=(ToolCall("c1", "echo", {"text": "hi"}),)),
            ModelResponse(content="已回显 hi"),
        ]
    )

    def gate(context) -> GateDecision:
        if (context.response.content or "").startswith("计划"):
            return GateDecision.block("plan_only", "请执行计划，不要只给计划。")
        return GateDecision.allow()

    loop = HarnessLoop(
        model,
        make_executor(),
        stop_gate=gate,
        config=RunConfig(max_turns=8, no_progress_limit=3, max_stop_blocks=2),
        run_id="r1",
    )
    result = loop.run("回显 hi")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.turns == 3
    assert result.tool_calls == 1
    assert result.output == "已回显 hi"
    assert any(event.type == "stop_gate" and not event.data["passed"] for event in result.events)


def test_repeated_tool_calls_stop_as_no_progress() -> None:
    call = ToolCall("c1", "echo", {"text": "same"})
    model = ScriptedModel(
        [ModelResponse(tool_calls=(call,)) for _ in range(3)]
    )
    loop = HarnessLoop(
        model,
        make_executor(),
        config=RunConfig(no_progress_limit=3),
        run_id="r2",
    )
    result = loop.run("loop forever")

    assert result.stop_reason is StopReason.NO_PROGRESS
    assert result.turns == 3
    assert result.tool_calls == 3


def test_empty_answer_is_blocked_until_limit() -> None:
    model = ScriptedModel([ModelResponse(content=""), ModelResponse(content="  ")])
    loop = HarnessLoop(
        model,
        make_executor(),
        stop_gate=require_answer,
        config=RunConfig(max_stop_blocks=2),
        run_id="r3",
    )
    result = loop.run("give an answer")

    assert result.stop_reason is StopReason.NO_PROGRESS
    assert result.turns == 2


def test_turn_budget_stops_run() -> None:
    call = ToolCall("c1", "echo", {"text": "x"})
    model = ScriptedModel([ModelResponse(tool_calls=(call,)) for _ in range(2)])
    loop = HarnessLoop(
        model,
        make_executor(),
        config=RunConfig(max_turns=2, no_progress_limit=10),
        run_id="r4",
    )
    result = loop.run("budget")

    assert result.stop_reason is StopReason.MAX_TURNS
    assert result.turns == 2
    assert result.tool_calls == 2


class BrokenModel:
    def complete(self, messages, tools):
        raise RuntimeError("boom")


def test_model_failure_marks_run_failed() -> None:
    loop = HarnessLoop(BrokenModel(), make_executor(), run_id="r5")
    result = loop.run("fail")

    assert result.stop_reason is StopReason.FAILED
    assert result.turns == 1


def test_tool_attempts_are_counted_across_retries() -> None:
    calls = {"n": 0}

    def flaky(text: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableToolError("HTTP 503")
        return text

    tool = ToolSpec(
        name="echo",
        description="echo text",
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
    loop = HarnessLoop(
        model,
        ToolExecutor([tool], run_id="r6"),
        run_id="r6",
    )
    result = loop.run("retry")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.tool_calls == 1
    assert result.tool_attempts == 3
    retries = [event for event in result.events if event.type == "tool_retry"]
    assert [event.data["attempt"] for event in retries] == [1, 2]


def test_message_invariants() -> None:
    with pytest.raises(ValueError):
        Message(role="tool", content="x")
    with pytest.raises(ValueError):
        Message(role="tool", content="x", tool_call_id="c1", tool_calls=(ToolCall("c2", "echo"),))
    with pytest.raises(ValueError):
        Message(role="user", content="x", tool_call_id="c1")
    with pytest.raises(ValueError):
        Message(role="system", content="x", tool_calls=(ToolCall("c1", "echo"),))