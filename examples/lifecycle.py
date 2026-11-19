"""I4 lifecycle demo: stream terminal states, cancellation, suspension, replay."""

from __future__ import annotations

import json
import tempfile

from nano_harness.lifecycle import (
    CancelToken,
    CheckpointStore,
    EmptyStreamError,
    LifecycleLoop,
    SseEvent,
    StreamInterrupted,
    StreamPolicy,
    format_sse,
    missing_calls,
    parse_sse,
    stream_with_retry,
    validate_stream_end,
)
from nano_harness.messages import ToolCall
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.risk import RiskLevel
from nano_harness.tools import ToolExecutor, ToolSpec, ToolSuspended

TASK_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
CHARGE_SCHEMA = {
    "type": "object",
    "properties": {"amount": {"type": "integer"}},
    "required": ["amount"],
    "additionalProperties": False,
}


def stream_scene() -> None:
    validate_stream_end(
        finish_reason="stop", saw_done=False, content="ok", tool_calls=()
    )
    print("  正常流: ok")

    try:
        validate_stream_end(
            finish_reason=None, saw_done=True, content=None, tool_calls=()
        )
    except EmptyStreamError:
        print("  空流: EmptyStreamError")

    def partial():
        yield "partial"
        raise ConnectionError("reset")

    seen = []
    try:
        for delta in stream_with_retry(
            partial, policy=StreamPolicy(max_attempts=3)
        ):
            seen.append(delta)  # noqa: PERF402 - keep partial output visible
    except StreamInterrupted:
        print(f"  半截流: StreamInterrupted, 已发 deltas={seen}")

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("reset")
        yield "ok"

    retried = list(
        stream_with_retry(flaky, policy=StreamPolicy(max_attempts=3))
    )
    print(f"  首事件前重试: 输出={retried} 尝试={attempts['n']}")


def cancel_scene() -> None:
    executed = []
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("c1", "task", {"name": "a"}),
                    ToolCall("c2", "task", {"name": "b"}),
                )
            ),
        ]
    )

    def task(name: str) -> str:
        executed.append(name)
        loop.cancel("client_gone")
        return name

    spec = ToolSpec("task", "", TASK_SCHEMA, task, risk=RiskLevel.WRITE)
    loop = LifecycleLoop(
        model, ToolExecutor([spec], run_id="cancel"), run_id="cancel"
    )
    result = loop.run("run two tasks")
    paired = missing_calls(result.messages) == ()
    print(
        f"  第一次: {result.status.value} 模型调用=1 工具执行={executed} "
        f"配对完整={paired}"
    )

    idle = CancelToken()
    idle.cancel("client_gone")
    second = LifecycleLoop(
        model, ToolExecutor([spec], run_id="cancel-2"), run_id="cancel-2"
    ).run("never starts", cancel=idle)
    print(f"  首轮前取消: {second.status.value} 模型调用=0")


def suspend_scene(directory: str) -> None:
    state = {"calls": 0}

    def deploy() -> str:
        state["calls"] += 1
        raise ToolSuspended(
            "approval_required", kind="tool", payload={"tool": "deploy"}
        )

    spec = ToolSpec("deploy", "", {"type": "object", "properties": {}}, deploy)
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "deploy", {}),)),
            ModelResponse(content="done"),
        ]
    )
    store = CheckpointStore(directory)
    loop = LifecycleLoop(
        model,
        ToolExecutor([spec], run_id="suspend"),
        store=store,
        run_id="suspend",
    )

    first = loop.run("deploy")
    checkpoint = store.load("suspend")
    print(
        f"  第一次: {first.status.value} pending="
        f"{checkpoint.pending_call.name} checkpoint={first.checkpoint is not None}"
    )

    second = loop.resume("suspend", "approved")
    print(
        f"  恢复后: {second.status.value} 内容={second.output} "
        f"handler 调用={state['calls']}"
    )


def replay_scene() -> None:
    charges = []

    def charge(amount: int) -> str:
        charges.append(amount)
        return f"charged {amount}"

    spec = ToolSpec(
        "charge",
        "",
        CHARGE_SCHEMA,
        charge,
        idempotent=True,
        risk=RiskLevel.WRITE,
    )
    executor = ToolExecutor([spec], run_id="replay")
    first = executor.execute(ToolCall("c1", "charge", {"amount": 10}))
    second = executor.execute(ToolCall("c2", "charge", {"amount": 10}))
    print(
        f"  第一次: attempts={first.attempts} replayed={first.replayed}"
    )
    print(
        f"  重复调用: attempts={second.attempts} replayed={second.replayed} "
        f"实际扣款={charges}"
    )


def sse_scene() -> None:
    events = (
        SseEvent(1, "run_started", json.dumps({"type": "run_started"})),
        SseEvent(2, "model_delta", json.dumps({"text": "你"})),
        SseEvent(3, "run_finished", json.dumps({"status": "completed"})),
    )
    text = "".join(format_sse(event) for event in events)
    parsed = parse_sse(text)
    print(
        f"  首帧: id={parsed[0].id} event={parsed[0].event} "
        f"往返一致={parsed == events} 序号={[event.id for event in parsed]}"
    )


def main() -> None:
    print("0. 流式终态")
    stream_scene()
    print("1. 取消")
    cancel_scene()
    with tempfile.TemporaryDirectory() as directory:
        print("2. 挂起与恢复")
        suspend_scene(directory)
    print("3. 幂等重放")
    replay_scene()
    print("4. SSE 协议")
    sse_scene()


if __name__ == "__main__":
    main()