import pytest

from nano_harness.lease import LeaseConflict, LeaseManager
from nano_harness.lifecycle import Checkpoint, CheckpointStore, RunStatus
from nano_harness.messages import Message, ToolCall
from nano_harness.resume import (
    SchemaMismatch,
    resume_pending,
)
from nano_harness.risk import RiskLevel
from nano_harness.tools import ToolExecutor, ToolSpec, current_tool_context


class Clock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


CHARGE_SCHEMA = {
    "type": "object",
    "properties": {"amount": {"type": "integer"}},
    "required": ["amount"],
    "additionalProperties": False,
}


def charge_tool():
    charges: dict[str, str] = {}

    def charge(amount: int) -> str:
        key = current_tool_context().idempotency_key
        if key is None:
            raise RuntimeError("charge must be declared idempotent")
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


def make_executor(spec: ToolSpec) -> ToolExecutor:
    return ToolExecutor([spec], run_id="run-1")


def make_checkpoint(
    store: CheckpointStore,
    messages,
    *,
    tool_results=None,
    schema_version: int = 1,
) -> Checkpoint:
    checkpoint = Checkpoint(
        run_id="run-1",
        status=RunStatus.RUNNING,
        messages=tuple(messages),
        tool_results=dict(tool_results or {}),
        attempt=1,
        lease_epoch=1,
        schema_version=schema_version,
    )
    store.save(checkpoint)
    return checkpoint


def leases_for_resume(clock: Clock) -> tuple[LeaseManager, object, object]:
    leases = LeaseManager(clock)
    first = leases.acquire("run-1", "worker-a", ttl=10)
    clock.value = 1020.0
    second = leases.acquire("run-1", "worker-b", ttl=10)
    return leases, first, second


def test_lease_acquire_expire_renew_release() -> None:
    clock = Clock()
    leases = LeaseManager(clock)
    first = leases.acquire("run-1", "worker-a", ttl=10)
    assert first.epoch == 1
    assert leases.acquire("run-1", "worker-b", ttl=10) is None
    clock.value = 1020.0
    assert not leases.validate(first)
    second = leases.acquire("run-1", "worker-b", ttl=10)
    assert second.epoch == 2
    assert leases.validate(second)
    renewed = leases.renew(second, ttl=10)
    assert renewed is not None
    assert leases.release(renewed)
    assert leases.acquire("run-1", "worker-c", ttl=10).epoch == 3


def test_resume_replays_checkpointed_results(tmp_path) -> None:
    clock = Clock()
    charges, spec = charge_tool()
    c1 = ToolCall("c1", "charge", {"amount": 10})
    c2 = ToolCall("c2", "charge", {"amount": 20})
    first = make_executor(spec).execute(c1)
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [
            Message.user("charge twice"),
            Message.assistant(None, (c1, c2)),
        ],
        tool_results={c1.id: first.content},
    )
    leases, _, lease = leases_for_resume(clock)
    report = resume_pending(
        checkpoint, store, make_executor(spec), leases, lease
    )

    assert report.replayed == ("c1",)
    assert report.executed == ("c2",)
    assert len(charges) == 2
    loaded = store.load("run-1")
    assert loaded.attempt == 2
    assert loaded.lease_epoch == 2
    assert set(loaded.tool_results) == {"c1", "c2"}


def test_crash_after_side_effect_before_checkpoint(tmp_path) -> None:
    clock = Clock()
    charges, spec = charge_tool()
    c1 = ToolCall("c1", "charge", {"amount": 10})
    c2 = ToolCall("c2", "charge", {"amount": 20})
    make_executor(spec).execute(c1)
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [
            Message.user("charge twice"),
            Message.assistant(None, (c1, c2)),
        ],
    )
    leases, _, lease = leases_for_resume(clock)
    report = resume_pending(
        checkpoint, store, make_executor(spec), leases, lease
    )

    assert report.replayed == ()
    assert report.executed == ("c1", "c2")
    assert len(charges) == 2


def test_stale_lease_is_rejected(tmp_path) -> None:
    clock = Clock()
    _, spec = charge_tool()
    c1 = ToolCall("c1", "charge", {"amount": 10})
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [Message.user("charge"), Message.assistant(None, (c1,))],
    )
    leases, stale, fresh = leases_for_resume(clock)

    with pytest.raises(LeaseConflict):
        resume_pending(
            checkpoint, store, make_executor(spec), leases, stale
        )
    resumes = resume_pending(
        checkpoint, store, make_executor(spec), leases, fresh
    )
    assert resumes.executed == ("c1",)


def test_schema_mismatch_is_rejected(tmp_path) -> None:
    clock = Clock()
    _, spec = charge_tool()
    c1 = ToolCall("c1", "charge", {"amount": 10})
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [Message.user("charge"), Message.assistant(None, (c1,))],
        schema_version=1,
    )
    leases, _, lease = leases_for_resume(clock)

    with pytest.raises(SchemaMismatch):
        resume_pending(
            checkpoint, store, make_executor(spec), leases, lease, expected_schema=2
        )


def test_second_resume_has_nothing_pending(tmp_path) -> None:
    clock = Clock()
    _, spec = charge_tool()
    c1 = ToolCall("c1", "charge", {"amount": 10})
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [Message.user("charge"), Message.assistant(None, (c1,))],
    )
    leases, _, lease = leases_for_resume(clock)
    resume_pending(checkpoint, store, make_executor(spec), leases, lease)
    loaded = store.load("run-1")

    second = resume_pending(
        loaded, store, make_executor(spec), leases, lease
    )
    assert second.replayed == ()
    assert second.executed == ()
    assert second.attempt == 2


def test_checkpoint_resume_fields_roundtrip(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = Checkpoint(
        run_id="run-1",
        status=RunStatus.RUNNING,
        messages=(Message.user("task"),),
        attempt=3,
        lease_epoch=7,
        schema_version=2,
        tool_results={"c1": "done"},
    )
    store.save(checkpoint)
    loaded = store.load("run-1")

    assert loaded.attempt == 3
    assert loaded.lease_epoch == 7
    assert loaded.schema_version == 2
    assert loaded.tool_results == {"c1": "done"}


def test_lease_for_other_run_is_rejected(tmp_path) -> None:
    clock = Clock()
    _, spec = charge_tool()
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(
        store,
        [Message.user("task"), Message.assistant(None, (ToolCall("c1", "charge", {"amount": 1}),))],
    )
    leases = LeaseManager(clock)
    other = leases.acquire("run-2", "worker", ttl=10)

    with pytest.raises(LeaseConflict):
        resume_pending(checkpoint, store, make_executor(spec), leases, other)