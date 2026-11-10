import json

import pytest

from nano_harness.lifecycle import (
    CancelToken,
    Checkpoint,
    CheckpointError,
    CheckpointStore,
    EmptyStreamError,
    LifecycleLoop,
    RunStatus,
    SseEvent,
    StreamInterrupted,
    StreamPolicy,
    Suspension,
    format_sse,
    missing_calls,
    parse_sse,
    stream_with_retry,
    validate_stream_end,
)
from nano_harness.messages import Message, ToolCall, Usage
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.tools import ToolExecutor, ToolSpec, ToolSuspended


def empty_schema() -> dict:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def test_checkpoint_roundtrip(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = Checkpoint(
        run_id="run-1",
        status=RunStatus.RUNNING,
        messages=(
            Message.user("go"),
            Message.assistant(None, (ToolCall("c1", "echo", {}),)),
            Message.tool_result(ToolCall("c1", "echo", {}), "ok"),
        ),
        turns=1,
        usage=Usage(10, 5),
    )
    path = store.save(checkpoint)
    loaded = store.load("run-1")

    assert path.exists()
    assert loaded.status is RunStatus.RUNNING
    assert loaded.usage.total_tokens == 15
    assert loaded.messages[2].content == "ok"


def test_checkpoint_requires_suspension_for_suspended_status() -> None:
    with pytest.raises(CheckpointError):
        Checkpoint("r", RunStatus.SUSPENDED, ())
    with pytest.raises(CheckpointError):
        Checkpoint(
            "r", RunStatus.RUNNING, (),
            pending_call=ToolCall("c1", "x"),
        )
    with pytest.raises(CheckpointError):
        Checkpoint(
            "r", RunStatus.SUSPENDED, (),
            pending_call=ToolCall("c1", "x"),
            suspension=Suspension("gate", "wait"),
        )


def test_missing_calls_detects_partial_batches() -> None:
    calls = (ToolCall("c1", "a", {}), ToolCall("c2", "b", {}))
    messages = [
        Message.user("go"),
        Message.assistant(None, calls),
        Message.tool_result(calls[0], "ok"),
    ]
    assert missing_calls(messages) == (calls[1],)

    messages.append(Message.tool_result(calls[1], "ok"))
    assert missing_calls(messages) == ()

    with pytest.raises(CheckpointError):
        missing_calls(
            [
                Message.assistant(None, calls),
                Message.tool_result(calls[0], "ok"),
                Message.user("continue"),
            ]
        )


def test_cancel_during_tool_batch_pairs_results(tmp_path) -> None:
    token = CancelToken()

    def first() -> str:
        token.cancel("client gone")
        return "a"

    specs = [
        ToolSpec("a", "", empty_schema(), first, read_only=True),
        ToolSpec("b", "", empty_schema(), lambda: "b", read_only=True),
    ]
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("c1", "a", {}),
                    ToolCall("c2", "b", {}),
                )
            )
        ]
    )
    loop = LifecycleLoop(
        model,
        ToolExecutor(specs, run_id="run-cancel"),
        store=CheckpointStore(tmp_path),
        run_id="run-cancel",
    )
    result = loop.run("go", cancel=token)

    assert result.status is RunStatus.CANCELLED
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert tool_messages[0].content == "a"
    assert "CANCELLED" in (tool_messages[1].content or "")
    assert missing_calls(result.messages) == ()


def test_suspend_and_resume_approval(tmp_path) -> None:
    state = {"approved": False, "effects": 0}

    def deploy() -> str:
        if not state["approved"]:
            raise ToolSuspended(
                "approval_required",
                kind="approval",
                payload={"tool": "deploy"},
            )
        state["effects"] += 1
        return "deployed"

    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "deploy", {}),)),
            ModelResponse(content="done"),
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = LifecycleLoop(
        model,
        ToolExecutor([ToolSpec("deploy", "", empty_schema(), deploy)], run_id="run-1"),
        store=store,
        run_id="run-1",
    )

    first = loop.run("deploy")
    assert first.status is RunStatus.SUSPENDED
    checkpoint = store.load("run-1")
    assert checkpoint.status is RunStatus.SUSPENDED
    assert checkpoint.pending_call is not None
    assert checkpoint.pending_call.id == "c1"
    assert checkpoint.suspension is not None
    assert checkpoint.suspension.kind == "approval"

    state["approved"] = True
    second = loop.resume("run-1", "approve: looks good")

    assert second.status is RunStatus.COMPLETED
    assert second.output == "done"
    assert state["effects"] == 1
    assert any(m.role == "tool" and m.content == "deployed" for m in second.messages)


def test_resume_denied_does_not_execute(tmp_path) -> None:
    effects = []
    state = {"approved": False}

    def deploy() -> str:
        if not state["approved"]:
            raise ToolSuspended("approval_required", kind="approval")
        effects.append(1)
        return "deployed"

    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "deploy", {}),)),
            ModelResponse(content="done"),
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = LifecycleLoop(
        model,
        ToolExecutor([ToolSpec("deploy", "", empty_schema(), deploy)], run_id="run-d"),
        store=store,
        run_id="run-d",
    )

    first = loop.run("deploy")
    assert first.status is RunStatus.SUSPENDED
    second = loop.resume("run-d", "deny")

    assert second.status is RunStatus.COMPLETED
    assert effects == []
    assert any(
        m.role == "tool" and "DENIED" in (m.content or "")
        for m in second.messages
    )


def test_sse_roundtrip() -> None:
    events = (
        SseEvent(1, "run_started", json.dumps({"turn": 0, "type": "run_started"})),
        SseEvent(2, "model_delta", json.dumps({"turn": 1, "data": {"text": "你"}})),
    )
    text = "".join(format_sse(event) for event in events)
    assert parse_sse(text) == events


def test_stream_retries_only_before_first_event() -> None:
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset")
        yield "ok"

    retries = []
    result = list(
        stream_with_retry(
            flaky,
            policy=StreamPolicy(max_attempts=3),
            on_retry=lambda exc, attempt: retries.append(attempt),
        )
    )
    assert result == ["ok"]
    assert retries == [1]

    def partial():
        yield "x"
        raise ConnectionError("reset")

    with pytest.raises(StreamInterrupted):
        list(stream_with_retry(partial, policy=StreamPolicy(max_attempts=3)))


def test_stream_terminal_states() -> None:
    with pytest.raises(StreamInterrupted):
        validate_stream_end(
            finish_reason=None, saw_done=False, content="x", tool_calls=()
        )
    with pytest.raises(EmptyStreamError):
        validate_stream_end(
            finish_reason=None, saw_done=True, content=None, tool_calls=()
        )
    assert (
        validate_stream_end(
            finish_reason="stop", saw_done=False, content="ok", tool_calls=()
        )
        is None
    )