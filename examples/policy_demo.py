"""Seven policy and sandbox scenarios from the I5 article."""

from __future__ import annotations

import json
import tempfile

from nano_harness.events import CREDENTIAL_ACCESSED, EventLog
from nano_harness.messages import ToolCall
from nano_harness.policy import (
    CredentialBroker,
    Decision,
    PermissionGate,
    PermissionPolicy,
    PolicyRule,
    SandboxPolicy,
    SandboxViolation,
    SubprocessSandbox,
    Workspace,
    redact_text,
    workspace_tools,
)
from nano_harness.tools import (
    ToolExecutor,
    ToolSpec,
    ToolSuspended,
    current_tool_context,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Workspace(directory)
        workspace.write_text("public.txt", "hello")
        workspace.write_text("secret.txt", "top secret")

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
        gate = PermissionGate(policy, can_suspend=True)
        tools = ToolExecutor(
            workspace_tools(workspace),
            run_id="demo",
            gate=gate,
            policy=policy,
        )

        print("0. 风险分级与可见性")
        print("  模型可见:", [spec.name for spec in tools.specs()])
        print(
            "  destructive 的 delete_file 可见:",
            any(spec.name == "delete_file" for spec in tools.specs()),
        )

        print("1. deny 改变控制流")
        denied = tools.execute(
            ToolCall("c1", "read_file", {"path": "secret.txt"})
        )
        allowed = tools.execute(
            ToolCall("c2", "read_file", {"path": "public.txt"})
        )
        print("  secret.txt:", denied.error.message if denied.error else "ok")
        print("  泄漏 top secret:", "top secret" in denied.content)
        print("  public.txt:", allowed.content)

        print("2. 审批挂起与恢复")
        first = ToolCall(
            "c3", "write_file", {"path": "approved.txt", "content": "approved"}
        )
        try:
            tools.execute(first)
        except ToolSuspended as exc:
            print(
                "  第一次: suspended kind=" + exc.kind
                + " tool=" + str(exc.payload.get("tool"))
                + " risk=" + str(exc.payload.get("risk"))
            )
        print("  文件已写入:", (workspace.root / "approved.txt").exists())
        tools.resolve_approval(first.id, "approve: looks good")
        tools.execute(first)
        print("  恢复后: completed 内容=approved")
        second = ToolCall(
            "c4", "write_file", {"path": "denied.txt", "content": "no"}
        )
        try:
            tools.execute(second)
        except ToolSuspended:
            pass
        tools.resolve_approval(second.id, "deny: not allowed")
        rejected = tools.execute(second)
        print(
            "  拒绝后:",
            rejected.error.kind.value if rejected.error else "ok",
            "文件已写入=" + str((workspace.root / "denied.txt").exists()),
        )

        print("3. ../ 越界")
        outside = workspace.root.parent / "outside.txt"
        outside.write_text("outside secret", encoding="utf-8")
        escaped = tools.execute(
            ToolCall("c5", "read_file", {"path": "../outside.txt"})
        )
        error = escaped.error.to_dict() if escaped.error else {}
        print(
            "  error=" + str(error.get("kind"))
            + " violation=" + str(error.get("violation"))
        )
        print("  读到 outside secret:", "outside secret" in escaped.content)

        print("4. 命令注入")
        sandbox = SubprocessSandbox(
            SandboxPolicy(frozenset({"echo"}), timeout=2), workspace.root
        )
        literal = sandbox.run(["echo", "hi; rm -rf ."])
        print("  分号被当作普通参数:", repr(literal.stdout.strip()))
        try:
            sandbox.run(["sh", "-c", "echo hi"])
        except SandboxViolation as exc:
            print("  允许列表外: sandbox_violation violation=" + exc.violation)

        print("5. 凭证作用域与脱敏")

        def call_api() -> str:
            token = current_tool_context().credentials.get("API_TOKEN")
            return json.dumps(
                {"body": "contact a@b.com", "token": token, "value": "demo"},
                ensure_ascii=False,
            )

        log = EventLog()
        credential_tools = ToolExecutor(
            [
                ToolSpec(
                    "call_api",
                    "",
                    {"type": "object", "properties": {}},
                    call_api,
                    read_only=True,
                    credentials=("API_TOKEN",),
                ),
                ToolSpec(
                    "peek",
                    "",
                    {"type": "object", "properties": {}},
                    call_api,
                    read_only=True,
                ),
            ],
            run_id="cred",
            credentials=CredentialBroker({"API_TOKEN": "s3cret"}),
            redactor=redact_text,
            events=log,
        )
        result = credential_tools.execute(ToolCall("c6", "call_api", {}))
        accessed = [
            event for event in log if event.type == CREDENTIAL_ACCESSED
        ]
        if accessed:
            print(
                "  credential_accessed="
                + str(accessed[-1].data["name"])
                + "（事件里没有值）"
            )
        print("  输出:", result.content)
        print("  原文泄漏 secret:", "s3cret" in result.content)
        print("  redaction counts=", result.redactions)
        peek = credential_tools.execute(ToolCall("c7", "peek", {}))
        print(
            "  未声明凭证: peek -> API_TOKEN ("
            + (peek.error.kind.value if peek.error else "ok")
            + ")"
        )


if __name__ == "__main__":
    main()