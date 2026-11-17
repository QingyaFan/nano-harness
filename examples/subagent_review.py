"""E3 example: a parent agent delegates a read-only review to a subagent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nano_harness.loop import HarnessLoop, RunConfig, RunResult
from nano_harness.messages import ToolCall, Usage
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.tools import ToolExecutor, ToolSpec

FILES = {
    "a.py": "def parse(value):\n    return int(value)\n",
    "b.py": "def load(path):\n    with open(path) as f:\n        return f.read()\n",
    "c.py": "def save(path, text):\n    open(path, 'w').write(text)\n",
}

READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
    "required": ["path", "text"],
    "additionalProperties": False,
}


def read_file(path: str) -> str:
    return FILES[path]


def write_file(path: str, text: str) -> str:
    return f"wrote {path} ({len(text)} bytes)"


READ_SPEC = ToolSpec(
    "read_file",
    "read a repository file",
    READ_SCHEMA,
    read_file,
    read_only=True,
)
WRITE_SPEC = ToolSpec(
    "write_file",
    "write a repository file",
    WRITE_SCHEMA,
    write_file,
    read_only=False,
)


@dataclass(frozen=True)
class Handoff:
    goal: str
    files: tuple[str, ...]
    authority: str = "read_only"
    budget_tokens: int = 500


def _model(
    checked: tuple[str, ...],
    missing: tuple[str, ...],
    finding: dict[str, object] | None,
) -> ScriptedModel:
    responses = [
        ModelResponse(
            tool_calls=(ToolCall(f"c{index}", "read_file", {"path": path}),),
            usage=Usage(10, 5),
        )
        for index, path in enumerate(checked, start=1)
    ]
    payload = {
        "status": "complete" if not missing else "partial",
        "findings": [finding] if finding else [],
        "coverage": {"checked": list(checked), "missing": list(missing)},
        "evidence": [f"{path}:read" for path in checked],
    }
    responses.append(ModelResponse(content=json.dumps(payload), usage=Usage(20, 10)))
    return ScriptedModel(responses)


def run_subagent(
    handoff: Handoff,
    *,
    run_id: str,
    checked: tuple[str, ...],
    missing: tuple[str, ...],
    finding: dict[str, object] | None,
) -> tuple[RunResult, dict[str, Any], ToolExecutor]:
    if handoff.authority != "read_only":
        raise ValueError("this example only grants read-only authority")
    tools = ToolExecutor([READ_SPEC], run_id=run_id)
    loop = HarnessLoop(
        _model(checked, missing, finding),
        tools,
        config=RunConfig(max_turns=10),
        run_id=run_id,
    )
    task = f"{handoff.goal}\nFILES: {', '.join(handoff.files)}"
    result = loop.run(task)
    payload = json.loads(result.output or "{}")
    return result, payload, tools


def accepted(payload: dict[str, Any]) -> bool:
    coverage = payload.get("coverage") or {}
    findings = payload.get("findings") or []
    missing = coverage.get("missing") or []
    return payload.get("status") == "complete" and not missing and bool(
        all(item.get("evidence") for item in findings)
    )


def main() -> None:
    handoff = Handoff(
        "check error handling in the listed files",
        ("a.py", "b.py", "c.py"),
    )
    parent_tools = ToolExecutor([READ_SPEC, WRITE_SPEC], run_id="parent")
    print("parent tools:", [spec.name for spec in parent_tools.specs()])

    partial_result, partial_payload, subagent_tools = run_subagent(
        handoff,
        run_id="subagent-partial",
        checked=("a.py", "b.py"),
        missing=("c.py",),
        finding=None,
    )
    print("subagent tools:", [spec.name for spec in subagent_tools.specs()])
    print(
        "partial:",
        partial_payload["status"],
        partial_payload["coverage"],
        "accepted=",
        accepted(partial_payload),
    )

    complete_result, complete_payload, _ = run_subagent(
        handoff,
        run_id="subagent-complete",
        checked=("a.py", "b.py", "c.py"),
        missing=(),
        finding={
            "file": "c.py",
            "line": 2,
            "issue": "write result is not checked",
            "evidence": "c.py:2",
        },
    )
    print(
        "complete:",
        complete_payload["status"],
        complete_payload["coverage"],
        "accepted=",
        accepted(complete_payload),
    )
    print("findings:", complete_payload["findings"])
    print(
        "tokens:",
        partial_result.usage.total_tokens + complete_result.usage.total_tokens,
    )


if __name__ == "__main__":
    main()