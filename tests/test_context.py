import pytest

from harness.context import (
    ContextBudget,
    ContextManager,
    TokenMeter,
    WindowInfo,
    estimate_text,
    make_artifact_tool,
    resolve_window,
    truncate_to_tokens,
)
from harness.messages import Message, ToolCall


def budget(size: int = 200) -> ContextBudget:
    return ContextBudget(
        WindowInfo(size, "authoritative"),
        threshold_ratio=0.5,
        target_ratio=0.5,
        reserve_ratio=0.1,
    )


def assert_pairs(messages) -> None:
    pending: list[str] = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            assert not pending, "assistant tool call while a batch is unfinished"
            pending = [call.id for call in message.tool_calls]
        elif message.role == "tool":
            assert pending, "orphan tool result"
            assert message.tool_call_id == pending[0], "tool result out of order"
            pending.pop(0)
        elif message.role == "user":
            assert not pending, "user message inside an unfinished tool batch"
    assert not pending, "tool call without result"


def test_estimate_text_accounts_for_cjk() -> None:
    text = "中" * 140
    assert estimate_text(text) == 210
    assert len(text) // 4 == 35


def test_truncate_to_tokens() -> None:
    assert len(truncate_to_tokens("a" * 100, 10)) == 40
    assert truncate_to_tokens("abc", 0) == ""


def test_token_meter_calibrates() -> None:
    meter = TokenMeter()
    meter.observe("a" * 100, 50)
    assert meter.count_text("a" * 100) == 50


def test_resolve_window_validates_input() -> None:
    assert resolve_window(2000).source == "authoritative"
    assert resolve_window(model="gpt-4o-mini").source == "inferred"
    with pytest.raises(ValueError):
        resolve_window(True)
    with pytest.raises(ValueError):
        resolve_window(0)


def test_budget_rejects_impossible_threshold() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            WindowInfo(100, "authoritative"),
            threshold_ratio=0.95,
            reserve_ratio=0.1,
        )


def test_large_tool_result_is_offloaded_to_artifact() -> None:
    context = ContextManager(budget(100_000), artifact_threshold=100)
    call = ToolCall("c1", "dump", {})
    context.add(Message.user("task"))
    context.add(Message.assistant(None, (call,)))
    original = "BEGIN " + "row;" * 2000
    context.add(Message.tool_result(call, original), tool="dump")

    prepared = context.wire(())
    tool_message = prepared.messages[-1]
    assert tool_message.content.startswith("[artifact artifact-001]")
    assert context.artifacts.get("artifact-001").content == original

    tool = make_artifact_tool(context.artifacts)
    assert tool.handler("artifact-001", 0, 5) == "BEGIN"


def test_compaction_cascade_preserves_constraints_and_pairs() -> None:
    context = ContextManager(
        budget(),
        summarizer=lambda span, limit: "SUMMARY",
        summary_max_tokens=20,
        artifact_threshold=10_000,
    )
    context.add(Message.system("sys"))
    context.add(Message.user("TASK: 约束 C 必须保留"))
    for index in range(6):
        call = ToolCall(f"c{index}", "echo", {"text": "x" * 50})
        context.add(Message.assistant(None, (call,)))
        context.add(Message.tool_result(call, "y" * 300), tool="echo")

    prepared = context.wire(())
    kinds = [record.kind for record in prepared.records]
    assert "micro" in kinds
    assert "full" in kinds
    assert any(
        message.role == "user" and "约束 C" in (message.content or "")
        for message in prepared.messages
    )
    assert_pairs(prepared.messages)


def test_wire_keeps_full_record_separate_from_view() -> None:
    context = ContextManager(
        budget(),
        summarizer=lambda span, limit: "SUMMARY",
        summary_max_tokens=20,
        artifact_threshold=10_000,
    )
    for index in range(4):
        call = ToolCall(f"c{index}", "echo", {"text": "x" * 40})
        context.add(Message.user(f"u{index}"))
        context.add(Message.assistant(None, (call,)))
        context.add(Message.tool_result(call, "z" * 200), tool="echo")

    prepared = context.wire(())
    assert len(context.full_messages) > len(prepared.messages)
    assert len(context.records) == len(prepared.records)