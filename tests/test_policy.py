import json

import pytest

from nano_harness.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    CREDENTIAL_ACCESSED,
    POLICY_DECISION,
    REDACTION_APPLIED,
    SANDBOX_DENIED,
    EventLog,
)
from nano_harness.lifecycle import (
    CheckpointStore,
    LifecycleLoop,
    RunStatus,
    ToolSuspended,
)
from nano_harness.messages import ToolCall
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.policy import (
    Approval,
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
    WorkspaceViolation,
    redact_text,
    redact_value,
    workspace_tools,
)
from nano_harness.tools import ToolExecutor, ToolSpec, current_tool_context


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


def test_redaction_default_patterns() -> None:
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n"
        "ghp_" + "a" * 36 + "\n"
        "AKIAIOSFODNN7EXAMPLE\n"
        "Authorization: Bearer abcdefgh12345678\n"
        "call 13800138000 or +86 139-0013-8000\n"
    )
    redacted, counts = redact_text(text)

    assert "MIIEow" not in redacted
    assert "ghp_" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "abcdefgh12345678" not in redacted
    assert "13800138000" not in redacted
    assert "139-0013-8000" not in redacted
    assert counts == {
        "private_key": 1,
        "github_token": 1,
        "aws_key": 1,
        "bearer_token": 1,
        "cn_phone": 2,
    }


def test_idempotent_replay_skips_approval() -> None:
    approvals = []

    def approver(call, spec, arguments):
        approvals.append(1)
        return Approval(True, "ok")

    executor = ToolExecutor(
        [
            ToolSpec(
                "write",
                "",
                empty_schema(),
                lambda: "wrote",
                idempotent=True,
                risk=RiskLevel.WRITE,
            )
        ],
        run_id="replay",
        gate=PermissionGate(PermissionPolicy(), approver=approver),
    )

    first = executor.execute(ToolCall("c1", "write", {}))
    second = executor.execute(ToolCall("c2", "write", {}))

    assert first.ok is True
    assert second.ok is True
    assert second.content == "wrote"
    assert len(approvals) == 1


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


def test_read_only_cannot_declare_write_risk() -> None:
    with pytest.raises(ValueError):
        ToolSpec(
            "bad",
            "",
            empty_schema(),
            lambda: "",
            read_only=True,
            risk=RiskLevel.WRITE,
        )


def test_rule_tool_pattern_matches() -> None:
    spec = ToolSpec("read_file", "", empty_schema(), lambda: "", read_only=True)
    policy = PermissionPolicy(
        rules=(PolicyRule("deny-read", "read_*", Decision.DENY),)
    )

    assert policy.decide(spec, {}).decision is Decision.DENY


def test_approver_failure_is_fail_closed() -> None:
    effects = []

    def handler() -> str:
        effects.append(1)
        return "ok"

    def approver(call, spec, arguments):
        raise RuntimeError("approval service down")

    log = EventLog()
    gate = PermissionGate(
        PermissionPolicy(), approver=approver, events=log
    )
    executor = ToolExecutor(
        [ToolSpec("write", "", empty_schema(), handler, risk=RiskLevel.WRITE)],
        run_id="fail-closed",
        gate=gate,
    )

    outcome = executor.execute(ToolCall("c1", "write", {}), turn=1)

    assert outcome.ok is False
    assert effects == []
    assert any(event.type == POLICY_DECISION for event in log)
    resolved = [event for event in log if event.type == APPROVAL_RESOLVED]
    assert resolved[-1].data["approved"] is False
    assert resolved[-1].data["source"] == "callback"


def test_approver_unexpected_value_is_fail_closed() -> None:
    effects = []

    def handler() -> str:
        effects.append(1)
        return "ok"

    gate = PermissionGate(
        PermissionPolicy(), approver=lambda call, spec, arguments: "yes"
    )
    executor = ToolExecutor(
        [ToolSpec("write", "", empty_schema(), handler, risk=RiskLevel.WRITE)],
        run_id="unexpected",
        gate=gate,
    )

    outcome = executor.execute(ToolCall("c1", "write", {}), turn=1)

    assert outcome.ok is False
    assert effects == []


def test_approval_request_fields_and_binding() -> None:
    schema = {
        "type": "object",
        "properties": {"secret": {"type": "string"}},
        "required": ["secret"],
        "additionalProperties": False,
    }

    log = EventLog()
    gate = PermissionGate(PermissionPolicy(), can_suspend=True, events=log)
    executor = ToolExecutor(
        [
            ToolSpec(
                "write",
                "",
                schema,
                lambda secret: secret,
                risk=RiskLevel.WRITE,
            )
        ],
        run_id="request",
        gate=gate,
    )
    call = ToolCall("c1", "write", {"secret": "s3cret"})

    with pytest.raises(ToolSuspended) as suspended:
        executor.execute(call, turn=2)

    payload = suspended.value.payload
    assert payload["call_id"] == "c1"
    assert payload["tool"] == "write"
    assert payload["risk"] == "write"
    assert payload["reason"]
    assert payload["rule"] is None
    assert payload["arguments"] == {"secret": "s3cret"}
    assert len(payload["arguments_digest"]) == 64
    assert [event.type for event in log].count(APPROVAL_REQUESTED) == 1

    gate.resolve("c1", "approve", tool="other")
    with pytest.raises(ValueError):
        executor.execute(call)


def test_credential_and_redaction_events() -> None:
    def peek() -> str:
        token = current_tool_context().credentials.get("API_TOKEN")
        return f"token={token} contact a@b.com"

    log = EventLog()
    executor = ToolExecutor(
        [
            ToolSpec(
                "peek",
                "",
                empty_schema(),
                peek,
                read_only=True,
                credentials=("API_TOKEN",),
            )
        ],
        run_id="events",
        credentials=CredentialBroker({"API_TOKEN": "s3cret"}),
        redactor=redact_text,
        events=log,
    )

    outcome = executor.execute(ToolCall("c1", "peek", {}))

    assert outcome.ok is True
    assert "s3cret" not in outcome.content
    assert "a@b.com" not in outcome.content
    types = [event.type for event in log]
    assert CREDENTIAL_ACCESSED in types
    assert REDACTION_APPLIED in types


def test_sandbox_denial_event(tmp_path) -> None:
    log = EventLog()
    executor = ToolExecutor(
        workspace_tools(Workspace(tmp_path / "ws")),
        run_id="sandbox",
        events=log,
    )

    outcome = executor.execute(
        ToolCall("c1", "read_file", {"path": "../secret"})
    )

    assert outcome.ok is False
    denied = [event for event in log if event.type == SANDBOX_DENIED]
    assert denied[-1].data["violation"] == "path_escape"


def test_workspace_limits_and_symlinks(tmp_path) -> None:
    workspace = Workspace(
        tmp_path / "ws", max_read_bytes=4, max_write_bytes=4
    )
    workspace.write_text("a.txt", "abcd")
    assert workspace.read_text("a.txt") == "abcd"

    with pytest.raises(WorkspaceViolation):
        workspace.write_text("b.txt", "abcde")
    with pytest.raises(WorkspaceViolation):
        workspace.write_text("missing/b.txt", "x")
    with pytest.raises(WorkspaceViolation):
        workspace.write_text(".", "x")

    (workspace.root / "link").symlink_to(workspace.root / "a.txt")
    with pytest.raises(WorkspaceViolation):
        workspace.read_text("link")
    with pytest.raises(WorkspaceViolation):
        workspace.delete("link")

    workspace.delete("a.txt")
    assert not (workspace.root / "a.txt").exists()


def test_sandbox_limits_env_and_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET", "leak")

    short = SubprocessSandbox(
        SandboxPolicy(
            frozenset({"echo"}), timeout=2, max_output_bytes=10
        ),
        tmp_path,
    )
    truncated = short.run(["echo", "hello world"])
    assert "[truncated]" in truncated.stdout

    sandbox = SubprocessSandbox(
        SandboxPolicy(
            frozenset({"echo", "env", "false"}),
            timeout=2,
            env_allowlist=("LANG",),
        ),
        tmp_path,
    )
    environment = sandbox.run(["env"])
    assert "SECRET=leak" not in environment.stdout
    assert "PATH=" in environment.stdout

    failed = sandbox.run(["false"])
    assert failed.returncode == 1

    with pytest.raises(SandboxViolation):
        sandbox.run(["echo", "x" * 9000])
    with pytest.raises(SandboxViolation):
        sandbox.run(["echo", "x"], stdin="x" * 20_000)