"""Run the full stack with a scripted model: context, tools, policy, loop."""

from __future__ import annotations

import tempfile

from harness.context import ContextBudget, ContextManager, WindowInfo
from harness.loop import HarnessLoop, RunConfig
from harness.messages import ToolCall
from harness.model import ModelResponse, ScriptedModel
from harness.policy import (
    Decision,
    PermissionGate,
    PermissionPolicy,
    PolicyRule,
    RiskLevel,
    Workspace,
    workspace_tools,
)
from harness.tools import ToolExecutor


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Workspace(directory)
        workspace.write_text("notes.txt", "build passed\n")
        workspace.write_text("secret.txt", "top secret\n")

        policy = PermissionPolicy(
            rules=(
                PolicyRule(
                    "deny-secrets",
                    "read_file",
                    Decision.DENY,
                    reason="secret files are not readable",
                    when=lambda args: "secret" in str(args.get("path", "")),
                ),
            ),
            defaults={RiskLevel.WRITE: Decision.DENY},
        )
        tools = ToolExecutor(
            workspace_tools(workspace),
            run_id="demo",
            gate=PermissionGate(policy),
        )
        model = ScriptedModel(
            [
                ModelResponse(
                    tool_calls=(ToolCall("c1", "read_file", {"path": "notes.txt"}),)
                ),
                ModelResponse(
                    tool_calls=(ToolCall("c2", "read_file", {"path": "secret.txt"}),)
                ),
                ModelResponse(
                    content="notes.txt says build passed; secret.txt was denied."
                ),
            ]
        )
        context = ContextManager(
            ContextBudget(WindowInfo(2000, "authoritative"))
        )
        loop = HarnessLoop(
            model,
            tools,
            config=RunConfig(max_turns=6),
            context=context,
            run_id="demo",
        )
        result = loop.run("summarize notes")

        print("stop_reason:", result.stop_reason.value)
        print("turns:", result.turns, "tool_calls:", result.tool_calls)
        print("output:", result.output)
        print("compactions:", [record.kind for record in context.records])
        print("tool results:")
        for message in result.messages:
            if message.role == "tool":
                print(" ", (message.content or "")[:120])


if __name__ == "__main__":
    main()