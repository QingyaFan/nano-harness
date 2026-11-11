import json

import pytest

from nano_harness.events import EventLog
from nano_harness.memory import (
    JsonMemoryStore,
    MemoryService,
    MemorySource,
    MemoryWrite,
    WriteStatus,
    memory_tools,
)
from nano_harness.messages import ToolCall
from nano_harness.tools import ToolExecutor


class Clock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def source(clock: Clock, run_id: str = "run-1") -> MemorySource:
    return MemorySource(run_id, "test", clock())


def write(
    service: MemoryService,
    clock: Clock,
    key: str,
    value: str,
    *,
    scope: str = "user:a",
    expected_version: int | None = None,
    ttl_seconds: float | None = None,
):
    return service.write(
        MemoryWrite(
            scope=scope,
            key=key,
            value=value,
            source=source(clock),
            expected_version=expected_version,
            ttl_seconds=ttl_seconds,
        )
    )


def test_persist_and_recall_across_runs(tmp_path) -> None:
    clock = Clock()
    path = tmp_path / "memory.json"
    service = MemoryService(JsonMemoryStore(path), clock=clock)
    write(service, clock, "service.api.timeout", "30s")

    reloaded = MemoryService(JsonMemoryStore(path), clock=clock)
    result = reloaded.recall("user:a", query="timeout")

    assert len(result.records) == 1
    assert result.records[0].value == "30s"
    assert "service.api.timeout" in result.text
    assert result.text.startswith("[memory]")


def test_duplicate_write_does_not_bump_version(tmp_path) -> None:
    clock = Clock()
    store = JsonMemoryStore(tmp_path / "memory.json")
    service = MemoryService(store, clock=clock)

    first = write(service, clock, "service.api.timeout", "30s")
    second = write(service, clock, "service.api.timeout", "30s")

    assert first.status is WriteStatus.CREATED
    assert second.status is WriteStatus.DUPLICATE
    assert len(store.history("user:a", "service.api.timeout")) == 1
    assert second.record.version == 1


def test_conflicting_write_requires_expected_version(tmp_path) -> None:
    clock = Clock()
    store = JsonMemoryStore(tmp_path / "memory.json")
    events = EventLog()
    service = MemoryService(store, events=events, clock=clock)
    write(service, clock, "service.api.timeout", "30s")

    conflict = write(service, clock, "service.api.timeout", "60s")

    assert conflict.status is WriteStatus.CONFLICT
    assert conflict.current.value == "30s"
    assert service.get("user:a", "service.api.timeout").value == "30s"
    assert any(event.type == "memory_conflict" for event in events)


def test_optimistic_update_and_stale_conflict(tmp_path) -> None:
    clock = Clock()
    store = JsonMemoryStore(tmp_path / "memory.json")
    service = MemoryService(store, clock=clock)
    first = write(service, clock, "service.api.timeout", "30s")

    updated = write(
        service,
        clock,
        "service.api.timeout",
        "60s",
        expected_version=first.record.version,
    )
    assert updated.status is WriteStatus.UPDATED
    assert updated.record.version == 2
    assert updated.record.supersedes == first.record.id
    assert service.get("user:a", "service.api.timeout").value == "60s"
    assert len(store.history("user:a", "service.api.timeout")) == 2

    stale = write(
        service,
        clock,
        "service.api.timeout",
        "90s",
        expected_version=1,
    )
    assert stale.status is WriteStatus.CONFLICT
    assert stale.current.value == "60s"


def test_scopes_are_isolated(tmp_path) -> None:
    clock = Clock()
    store = JsonMemoryStore(tmp_path / "memory.json")
    service = MemoryService(store, clock=clock)
    write(service, clock, "service.api.timeout", "30s", scope="user:a")
    write(service, clock, "service.api.timeout", "60s", scope="user:b")

    assert service.recall("user:a").records[0].value == "30s"
    assert service.recall("user:b").records[0].value == "60s"
    assert service.recall("user:c").records == ()
    assert service.manifest("user:a") != service.manifest("user:b")


def test_recall_respects_budget(tmp_path) -> None:
    clock = Clock()
    service = MemoryService(JsonMemoryStore(tmp_path / "memory.json"), clock=clock)
    for index in range(6):
        write(service, clock, f"fact.{index}", "x" * 40)

    result = service.recall("user:a", budget_tokens=50)

    assert result.used_tokens <= 50
    assert result.dropped
    assert len(result.records) < 6
    assert result.text.startswith("[memory]")


def test_expiry_and_purge(tmp_path) -> None:
    clock = Clock(1000.0)
    events = EventLog()
    service = MemoryService(
        JsonMemoryStore(tmp_path / "memory.json"), events=events, clock=clock
    )
    write(service, clock, "service.api.timeout", "30s", ttl_seconds=10)

    clock.value = 1020.0
    assert service.recall("user:a").records == ()
    assert service.purge_expired("user:a") == 1
    assert any(event.type == "memory_expired" for event in events)


def test_forget_removes_current_fact(tmp_path) -> None:
    clock = Clock()
    events = EventLog()
    service = MemoryService(
        JsonMemoryStore(tmp_path / "memory.json"), events=events, clock=clock
    )
    write(service, clock, "service.api.timeout", "30s")

    forgotten = service.forget("user:a", "service.api.timeout")
    assert forgotten.status == "forgotten"
    assert service.recall("user:a").records == ()
    assert service.forget("user:a", "service.api.timeout").status == "not_found"
    assert any(event.type == "memory_forgotten" for event in events)


def test_memory_tools_are_scope_bound(tmp_path) -> None:
    clock = Clock()
    service = MemoryService(JsonMemoryStore(tmp_path / "memory.json"), clock=clock)
    tools = memory_tools(service, scope="user:a", run_id="run-1")
    assert all("scope" not in tool.parameters.get("properties", {}) for tool in tools)
    executor = ToolExecutor(tools, run_id="mem")

    created = executor.execute(
        ToolCall(
            "c1",
            "remember",
            {"key": "service.api.timeout", "value": "30s"},
        )
    )
    assert created.ok is True
    assert json.loads(created.content)["status"] == "created"

    recalled = executor.execute(
        ToolCall("c2", "recall", {"query": "timeout"})
    )
    assert recalled.ok is True
    assert "service.api.timeout" in recalled.content

    conflict = executor.execute(
        ToolCall(
            "c3",
            "remember",
            {"key": "service.api.timeout", "value": "60s"},
        )
    )
    payload = json.loads(conflict.content)
    assert payload["status"] == "conflict"
    assert payload["current"]["value"] == "30s"

    updated = executor.execute(
        ToolCall(
            "c4",
            "remember",
            {
                "key": "service.api.timeout",
                "value": "60s",
                "expected_version": 1,
            },
        )
    )
    assert json.loads(updated.content)["status"] == "updated"
    assert service.recall("user:b").records == ()


def test_manifest_changes_and_checkpoint_meta(tmp_path) -> None:
    clock = Clock()
    service = MemoryService(JsonMemoryStore(tmp_path / "memory.json"), clock=clock)
    before = service.manifest("user:a")
    write(service, clock, "service.api.timeout", "30s")
    after = service.manifest("user:a")

    assert before != after
    meta = service.checkpoint_meta("user:a")
    assert meta["memory_keys"] == 1
    assert meta["memory_manifest"] == after


def test_memory_validation(tmp_path) -> None:
    clock = Clock()
    service = MemoryService(JsonMemoryStore(tmp_path / "memory.json"), clock=clock)

    with pytest.raises(ValueError):
        write(service, clock, "", "value")
    with pytest.raises(ValueError):
        write(service, clock, "bad key", "value")
    with pytest.raises(ValueError):
        write(service, clock, "key", "")
    with pytest.raises(ValueError):
        service.write(
            MemoryWrite(
                scope="",
                key="key",
                value="value",
                source=source(clock),
            )
        )
    with pytest.raises(ValueError):
        service.write(
            MemoryWrite(
                scope="user:a",
                key="key",
                value="value",
                source=source(clock),
                ttl_seconds=0,
            )
        )
    with pytest.raises(ValueError):
        service.write(
            MemoryWrite(
                scope="user:a",
                key="key",
                value="value",
                source=source(clock),
                confidence=2.0,
            )
        )