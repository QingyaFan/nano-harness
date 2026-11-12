"""Resume a checkpointed run: replay completed results and close pending calls (E1)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .lease import Lease, LeaseConflict, LeaseManager
from .lifecycle import (
    Checkpoint,
    CheckpointError,
    CheckpointStore,
    RunStatus,
    missing_calls,
)
from .messages import Message


class ResumeError(Exception):
    pass


class SchemaMismatch(ResumeError):
    pass


@dataclass(frozen=True)
class ResumeReport:
    run_id: str
    attempt: int
    lease_epoch: int
    replayed: tuple[str, ...]
    executed: tuple[str, ...]
    message_count: int
    checkpoint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "lease_epoch": self.lease_epoch,
            "replayed": list(self.replayed),
            "executed": list(self.executed),
            "message_count": self.message_count,
            "checkpoint": self.checkpoint,
        }


def resume_pending(
    checkpoint: Checkpoint,
    store: CheckpointStore,
    executor: Any,
    leases: LeaseManager,
    lease: Lease,
    *,
    expected_schema: int = 1,
) -> ResumeReport:
    if lease.run_id != checkpoint.run_id:
        raise LeaseConflict("lease belongs to a different run")
    leases.assert_valid(lease)
    if checkpoint.schema_version != expected_schema:
        raise SchemaMismatch(
            f"checkpoint schema {checkpoint.schema_version} is not "
            f"compatible with expected {expected_schema}"
        )
    if checkpoint.status is not RunStatus.RUNNING:
        raise ResumeError(
            f"only running checkpoints can be resumed, got {checkpoint.status.value}"
        )

    try:
        pending = missing_calls(checkpoint.messages)
    except CheckpointError as exc:
        raise ResumeError(str(exc)) from exc
    if not pending:
        path = store.path(checkpoint.run_id)
        return ResumeReport(
            run_id=checkpoint.run_id,
            attempt=checkpoint.attempt,
            lease_epoch=checkpoint.lease_epoch,
            replayed=(),
            executed=(),
            message_count=len(checkpoint.messages),
            checkpoint=str(path) if path.exists() else None,
        )

    messages = list(checkpoint.messages)
    results = dict(checkpoint.tool_results)
    replayed: list[str] = []
    executed: list[str] = []
    for call in pending:
        if call.id in results:
            content = results[call.id]
            replayed.append(call.id)
        else:
            outcome = executor.execute(call)
            content = outcome.content
            results[call.id] = content
            executed.append(call.id)
        messages.append(Message.tool_result(call, content))

    updated = replace(
        checkpoint,
        messages=tuple(messages),
        attempt=checkpoint.attempt + 1,
        lease_epoch=lease.epoch,
        tool_results=results,
    )
    path = store.save(updated)
    return ResumeReport(
        run_id=updated.run_id,
        attempt=updated.attempt,
        lease_epoch=updated.lease_epoch,
        replayed=tuple(replayed),
        executed=tuple(executed),
        message_count=len(updated.messages),
        checkpoint=str(path),
    )