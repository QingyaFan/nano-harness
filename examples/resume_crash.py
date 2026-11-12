"""Crash/resume fixture: replay a ledger result, execute pending calls, fence stale workers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nano_harness.lease import LeaseConflict, LeaseManager
from nano_harness.lifecycle import Checkpoint, CheckpointStore, RunStatus
from nano_harness.messages import Message, ToolCall
from nano_harness.resume import SchemaMismatch, resume_pending
from nano_harness.risk import RiskLevel
from nano_harness.tools import ToolExecutor, ToolSpec, current_tool_context

CHARGE_SCHEMA = {
    "type": "object",
    "properties": {"amount": {"type": "integer"}},
    "required": ["amount"],
    "additionalProperties": False,
}


class Clock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def charge_tool():
    charges: dict[str, str] = {}

    def charge(amount: int) -> str:
        key = current_tool_context().idempotency_key
        if key is None:
            raise RuntimeError("charge must be idempotent")
        if key not in charges:
            charges[key] = f"charged {amount}"
        return charges[key]

    spec = ToolSpec(
        name="charge",
        description="charge an amount",
        parameters=CHARGE_SCHEMA,
        handler=charge,
        idempotent=True,
        risk=RiskLevel.WRITE,
    )
    return charges, spec


def executor(spec: ToolSpec) -> ToolExecutor:
    return ToolExecutor([spec], run_id="run-1")


def checkpoint(store: CheckpointStore, messages, tool_results=None) -> Checkpoint:
    value = Checkpoint(
        run_id="run-1",
        status=RunStatus.RUNNING,
        messages=tuple(messages),
        tool_results=dict(tool_results or {}),
        attempt=1,
        lease_epoch=1,
    )
    store.save(value)
    return value


def leases(clock: Clock):
    manager = LeaseManager(clock)
    first = manager.acquire("run-1", "worker-a", ttl=10)
    clock.value = 1020.0
    second = manager.acquire("run-1", "worker-b", ttl=10)
    return manager, first, second


def main() -> None:
    c1 = ToolCall("c1", "charge", {"amount": 10})
    c2 = ToolCall("c2", "charge", {"amount": 20})

    with tempfile.TemporaryDirectory() as directory:
        clock = Clock()
        charges, spec = charge_tool()
        first_result = executor(spec).execute(c1)
        store = CheckpointStore(Path(directory) / "replay")
        state = checkpoint(
            store,
            [
                Message.user("charge twice"),
                Message.assistant(None, (c1, c2)),
            ],
            tool_results={c1.id: first_result.content},
        )
        manager, stale, fresh = leases(clock)
        print(
            f"lease: stale_epoch={stale.epoch} stale_valid={manager.validate(stale)} "
            f"fresh_epoch={fresh.epoch}"
        )
        report = resume_pending(state, store, executor(spec), manager, fresh)
        loaded = store.load("run-1")
        print(
            f"resume: replayed={list(report.replayed)} "
            f"executed={list(report.executed)} side_effects={len(charges)} "
            f"attempt={loaded.attempt} lease_epoch={loaded.lease_epoch}"
        )
        print(
            f"checkpoint: tool_results={len(loaded.tool_results)} "
            f"messages={len(loaded.messages)}"
        )

        try:
            resume_pending(state, store, executor(spec), manager, stale)
        except LeaseConflict:
            print("stale-worker: rejected")
        try:
            resume_pending(
                state, store, executor(spec), manager, fresh, expected_schema=2
            )
        except SchemaMismatch:
            print("schema-mismatch: rejected")

    with tempfile.TemporaryDirectory() as directory:
        clock = Clock()
        charges, spec = charge_tool()
        executor(spec).execute(c1)
        store = CheckpointStore(Path(directory) / "crash-window")
        state = checkpoint(
            store,
            [
                Message.user("charge twice"),
                Message.assistant(None, (c1, c2)),
            ],
        )
        manager, _, fresh = leases(clock)
        report = resume_pending(state, store, executor(spec), manager, fresh)
        print(
            f"crash-window: replayed={list(report.replayed)} "
            f"executed={list(report.executed)} side_effects={len(charges)}"
        )


if __name__ == "__main__":
    main()