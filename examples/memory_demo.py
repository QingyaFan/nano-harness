"""Two-run memory fixture: write in run 1, recall in run 2."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from nano_harness.events import EventLog
from nano_harness.memory import JsonMemoryStore, MemoryService, MemorySource, MemoryWrite
from nano_harness.messages import ToolCall
from nano_harness.tools import ToolExecutor


def status(outcome: str) -> dict:
    return json.loads(outcome)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        events = EventLog()
        store = JsonMemoryStore(path)
        service = MemoryService(store, events=events)
        from nano_harness.memory import memory_tools

        tools = memory_tools(service, scope="user:demo", run_id="run-001")
        executor = ToolExecutor(tools, run_id="run-001")

        print("run-001 writes:")
        actions = [
            ("created", {"key": "service.api.timeout", "value": "30s"}),
            ("conflict", {"key": "service.api.timeout", "value": "60s"}),
            (
                "updated",
                {
                    "key": "service.api.timeout",
                    "value": "60s",
                    "expected_version": 1,
                },
            ),
        ]
        for index, (label, arguments) in enumerate(actions):
            outcome = executor.execute(
                ToolCall(f"w{index}", "remember", arguments)
            )
            payload = status(outcome.content)
            record = payload.get("record") or {}
            current = payload.get("current") or {}
            print(
                f"  {label}: status={payload['status']} "
                f"version={record.get('version')} "
                f"current={current.get('value')}"
            )

        for index, (key, value) in enumerate(
            [
                ("service.api.retries", "3"),
                ("service.db.pool", "20"),
                ("deploy.region", "cn-north"),
                ("incident.last", "gateway timeout at 2026-11-09"),
            ]
        ):
            executor.execute(
                ToolCall(f"r{index}", "remember", {"key": key, "value": value})
            )

        # Run 2: a new service and store instance read the same file.
        events2 = EventLog()
        service2 = MemoryService(JsonMemoryStore(path), events=events2)
        tools2 = memory_tools(service2, scope="user:demo", run_id="run-002")
        executor2 = ToolExecutor(tools2, run_id="run-002")
        duplicate = executor2.execute(
            ToolCall(
                "d1",
                "remember",
                {"key": "service.api.timeout", "value": "60s"},
            )
        )
        duplicate_payload = status(duplicate.content)
        print(
            "run-002 duplicate: "
            f"status={duplicate_payload['status']} "
            f"version={duplicate_payload['record']['version']}"
        )

        recall = service2.recall("user:demo", query="service", budget_tokens=200)
        print("\nrun-002 recall (budget=200):")
        print(recall.text.rstrip())
        print(f"  tokens={recall.used_tokens} dropped={len(recall.dropped)}")

        tight = service2.recall("user:demo", query="service", budget_tokens=40)
        print(
            f"run-002 recall (budget=40): tokens={tight.used_tokens} "
            f"included={len(tight.records)} dropped={len(tight.dropped)}"
        )

        print(f"scope user:other: records={len(service2.recall('user:other').records)}")

        forgotten = service2.forget("user:demo", "service.api.retries")
        after_forget = service2.recall("user:demo", query="retries")
        print(
            f"forget service.api.retries: {forgotten.status}, "
            f"recall_contains={bool(after_forget.records)}"
        )

        before = service2.manifest("user:demo")
        service2.write(
            MemoryWrite(
                scope="user:demo",
                key="deploy.region",
                value="cn-south",
                source=MemorySource("run-002", "remember"),
                expected_version=1,
            )
        )
        after = service2.manifest("user:demo")
        print(f"manifest changed: {before != after}")
        print(
            "events:",
            sorted({event.type for event in events}),
        )
        print(
            "run-002 events:",
            sorted({event.type for event in events2}),
        )


if __name__ == "__main__":
    main()