import json

import pytest

from harness.lifecycle import (
    CheckpointStore,
    LifecycleLoop,
    RunStatus,
    ToolSuspended,
)
from harness.messages import ToolCall
from harness.model import ModelResponse, ScriptedModel
from harness.policy import (
    CredentialBroker,
    CredentialDenied,
    Decision,
    PermissionGate,
    PermissionPolicy,
    PolicyRule,
    RiskLevel,
    SandboxPolicy,
    SandboxViolation,
    SubprocessSandbox,
    Workspace,
    redact_text,
    redact_value,
    workspace_tools,
)
from harness.tools import ToolExecutor, ToolSpec, current_tool_context


def empty_schema() -> dict:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def test_visibility_hides_destructive_tools() -> None:
    read = ToolSpec("read_file", "", empty_schema(), lambda: "", read_only=True)
    write = ToolSpec(
        "write_file", "", empty_schema(), lambda: "", risk=RiskLevel.WRITE
    )
    delete = ToolSpec(
        "delete_file", "", empty_schema(), lambda: "", risk=RiskLevel.DESTRUCTIVE
    )
    policy = PermissionPolicy(visible_risks=(RiskLevel.READ, RiskLevel.WRITE))

    assert [spec.name for spec in policy.visible((read, write, delete))] == [
        "read_file",
        "write_file",
    ]

    exposed = PermissionPolicy(
        rules=(PolicyRule("show-delete", "delete_file", expose=True),)
    )
    assert "delete_file" in [spec.name for spec in exposed.visible((read, delete))]


def test_deny_rule_prevents_secret_read(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    workspace.write_text("secret.txt", "top secret")
    workspace.write_text("public.txt", "hello")
    policy = PermissionPolicy(
        rules=(
            PolicyRule(
                "deny-secrets",
                "read_file",
                Decision.DENY,
                reason="secret files are not readable",
                when=lambda args: "secret" in str(args.get("path", "")),
            ),
        )
    )
    executor = ToolExecutor(
        workspace_tools(workspace),
        run_id="policy",
        gate=PermissionGate(policy),
    )

    denied = executor.execute(ToolCall("c1", "read_file", {"path": "secret.txt"}))
    allowed = executor.execute(ToolCall("c2", "read_file", {"path": "public.txt"}))

    assert denied.ok is False
    payload = json.loads(denied.content)
    assert payload["error"]["kind"] == "permission_denied"
    assert "top secret" not in denied.content
    assert allowed.ok is True
    assert allowed.content == "hello"


def test_ask_suspends_then_resolves(tmp_path) -> None:
    workspace = Workspace(tmp_path)
    policy = PermissionPolicy(defaults={RiskLevel.WRITE: Decision.ASK})
    gate = PermissionGate(policy, can_suspend=True)
    executor = ToolExecutor(
        workspace_tools(workspace), run_id="ask", gate=gate
    )
    call = ToolCall("c1", "write_file", {"path": "a.txt", "content": "hello"})

    with pytest.raises(ToolSuspended):
        executor.execute(call)

    gate.resolve("c1", "approve: looks good")
    outcome = executor.execute(call)
    assert outcome.ok is True
    assert (tmp_path / "a.txt").read_text() == "hello"

    denied_call = ToolCall(
        "c2", "write_file", {"path": "b.txt", "content": "nope"}
    )
    with pytest.raises(ToolSuspended):
        executor.execute(denied_call)
    gate.resolve("c2", "deny: not allowed")
    denied = executor.execute(denied_call)
    assert denied.ok is False
    assert not (tmp_path / "b.txt").exists()


def test_workspace_escape_is_reported(tmp_path) -> None:
    workspace = Workspace(tmp_path / "ws")
    (tmp_path / "secret.txt").write_text("outside secret")
    executor = ToolExecutor(workspace_tools(workspace), run_id="ws")

    outcome = executor.execute(
        ToolCall("c1", "read_file", {"path": "../secret.txt"})
    )

    assert outcome.ok is False
    payload = json.loads(outcome.content)
    assert payload["error"]["kind"] == "sandbox_violation"
    assert payload["error"]["violation"] == "path_escape"
    assert "outside secret" not in outcome.content


def test_sandbox_rejects_shell_strings_and_disallowed_commands(tmp_path) -> None:
    policy = SandboxPolicy(frozenset({"echo", "sleep"}), timeout=0.2)
    sandbox = SubprocessSandbox(policy, tmp_path)

    result = sandbox.run(["echo", "hi; rm -rf ."])
    assert result.stdout.strip() == "hi; rm -rf ."

    with pytest.raises(SandboxViolation) as shell:
        sandbox.run("echo hi")
    assert shell.value.violation == "shell_string"

    with pytest.raises(SandboxViolation) as denied:
        sandbox.run(["rm", "-rf", "."])
    assert denied.value.violation == "executable_not_allowed"

    with pytest.raises(TimeoutError):
        sandbox.run(["sleep", "2"])


def test_credentials_are_scoped() -> None:
    broker = CredentialBroker({"API_TOKEN": "s3cret", "OTHER": "x"})
    accessed, denied = [], []
    scope = broker.scope(
        ("API_TOKEN",),
        on_access=accessed.append,
        on_denied=lambda name, reason: denied.append(name),
    )

    assert scope.get("API_TOKEN") == "s3cret"
    with pytest.raises(CredentialDenied):
        scope.get("OTHER")
    assert accessed == ["API_TOKEN"]
    assert denied == ["OTHER"]


def test_executor_enforces_declared_credentials() -> None:
    def peek() -> str:
        return current_tool_context().credentials.get("API_TOKEN")

    spec = ToolSpec(
        "peek",
        "",
        empty_schema(),
        peek,
        read_only=True,
        credentials=("API_TOKEN",),
    )
    executor = ToolExecutor(
        [spec],
        run_id="cred",
        credentials=CredentialBroker({"API_TOKEN": "s3cret"}),
    )
    outcome = executor.execute(ToolCall("c1", "peek", {}))
    assert outcome.ok is True
    assert outcome.content == "s3cret"

    undeclared = ToolSpec("peek2", "", empty_schema(), peek, read_only=True)
    executor2 = ToolExecutor(
        [undeclared],
        run_id="cred2",
        credentials=CredentialBroker({"API_TOKEN": "s3cret"}),
    )
    outcome2 = executor2.execute(ToolCall("c2", "peek2", {}))
    assert outcome2.ok is False
    assert json.loads(outcome2.content)["error"]["kind"] == "credential_denied"


def test_redaction_patterns_and_executor() -> None:
    text = "contact a@b.com; token=sk-12345678"
    redacted, counts = redact_text(text)
    assert "a@b.com" not in redacted
    assert "sk-12345678" not in redacted
    assert counts == {"email": 1, "api_key": 1}

    value, nested_counts = redact_value(
        {"body": "contact a@b.com; token=sk-12345678", "value": "demo"}
    )
    assert value["value"] == "demo"
    assert nested_counts == {"email": 1, "api_key": 1}

    spec = ToolSpec(
        "dump",
        "",
        empty_schema(),
        lambda: '{"body": "contact a@b.com; token=sk-12345678", "value": "demo"}',
        read_only=True,
    )
    executor = ToolExecutor([spec], run_id="redact", redactor=redact_text)
    outcome = executor.execute(ToolCall("c1", "dump", {}))
    assert "a@b.com" not in outcome.content
    assert outcome.redactions == {"email": 1, "api_key": 1}


def test_lifecycle_approval_gate(tmp_path) -> None:
    workspace = Workspace(tmp_path / "ws")
    policy = PermissionPolicy(defaults={RiskLevel.WRITE: Decision.ASK})
    gate = PermissionGate(policy, can_suspend=True)
    executor = ToolExecutor(
        workspace_tools(workspace), run_id="life", gate=gate
    )
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "c1",
                        "write_file",
                        {"path": "a.txt", "content": "hello"},
                    ),
                )
            ),
            ModelResponse(content="done"),
        ]
    )
    loop = LifecycleLoop(
        model, executor, run_id="life", store=CheckpointStore(tmp_path)
    )
    first = loop.run("write")
    assert first.status is RunStatus.SUSPENDED

    second = loop.resume("life", "approve: ok")
    assert second.status is RunStatus.COMPLETED
    assert (workspace.root / "a.txt").read_text() == "hello"