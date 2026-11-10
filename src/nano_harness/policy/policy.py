"""Risk-based permission policy and the approval gate (I5)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..tools.errors import ErrorKind, ToolError, ToolOutcome, ToolSuspended
from ..tools.spec import ToolSpec
from .approval import Approval, ApprovalRequest, digest_arguments, parse_resume_value
from .types import Decision, RiskLevel


@dataclass(frozen=True)
class PolicyRule:
    name: str
    tool: str | tuple[str, ...] | None = None
    decision: Decision = Decision.ASK
    reason: str = ""
    when: Callable[[dict[str, Any]], bool] | None = None
    expose: bool | None = None

    def matches_tool(self, name: str) -> bool:
        if self.tool is None or self.tool == "*":
            return True
        if isinstance(self.tool, str):
            return self.tool == name
        return name in self.tool

    def matches_arguments(self, arguments: dict[str, Any]) -> bool:
        return self.when is None or bool(self.when(arguments))


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    risk: RiskLevel
    reason: str
    rule: str | None = None


class PermissionPolicy:
    def __init__(
        self,
        rules: Iterable[PolicyRule] = (),
        *,
        defaults: dict[RiskLevel, Decision] | None = None,
        visible_risks: Iterable[RiskLevel] = (RiskLevel.READ, RiskLevel.WRITE),
    ) -> None:
        self.rules = tuple(rules)
        self.defaults = {
            RiskLevel.READ: Decision.ALLOW,
            RiskLevel.WRITE: Decision.ASK,
            RiskLevel.DESTRUCTIVE: Decision.DENY,
            RiskLevel.EXEC: Decision.ASK,
        }
        if defaults:
            self.defaults.update(defaults)
        self.visible_risks = frozenset(visible_risks)

    def decide(self, spec: ToolSpec, arguments: dict[str, Any] | None) -> PolicyDecision:
        risk = spec.risk
        arguments = arguments or {}
        for rule in self.rules:
            if not rule.matches_tool(spec.name):
                continue
            try:
                matched = rule.matches_arguments(arguments)
            except Exception as exc:  # noqa: BLE001
                return PolicyDecision(
                    Decision.DENY,
                    risk,
                    f"rule {rule.name} failed: {type(exc).__name__}",
                    rule.name,
                )
            if matched:
                return PolicyDecision(
                    rule.decision,
                    risk,
                    rule.reason or f"matched rule {rule.name}",
                    rule.name,
                )
        return PolicyDecision(
            self.defaults.get(risk, Decision.DENY),
            risk,
            f"default for {risk.value} risk",
            None,
        )

    def visible(self, specs: Iterable[ToolSpec]) -> tuple[ToolSpec, ...]:
        visible = []
        for spec in specs:
            exposed = None
            for rule in self.rules:
                if (
                    rule.when is None
                    and rule.expose is not None
                    and rule.matches_tool(spec.name)
                ):
                    exposed = rule.expose
                    break
            if exposed is None:
                exposed = spec.risk in self.visible_risks
            if exposed:
                visible.append(spec)
        return tuple(visible)


def _denied(reason: str, rule: str | None = None) -> ToolOutcome:
    error = ToolError(
        ErrorKind.PERMISSION_DENIED,
        reason,
        details={"rule": rule or ""},
    )
    return ToolOutcome(ok=False, content=error.envelope(), error=error, attempts=0)


class PermissionGate:
    def __init__(
        self,
        policy: PermissionPolicy,
        *,
        approver: Callable[[Any, ToolSpec, dict[str, Any]], Approval] | None = None,
        can_suspend: bool = False,
    ) -> None:
        self._policy = policy
        self._approver = approver
        self._can_suspend = can_suspend
        self._approvals: dict[str, Approval] = {}

    def resolve(self, call_id: str, value: str) -> None:
        self._approvals[call_id] = parse_resume_value(
            value, call_id=call_id, tool=""
        )

    def check(
        self,
        call: Any,
        spec: ToolSpec,
        arguments: dict[str, Any] | None,
        *,
        turn: int = 0,
    ) -> ToolOutcome | None:
        approved = self._approvals.get(call.id)
        if approved is not None:
            if approved.approved:
                return None
            return _denied(f"approval denied: {approved.reason}")

        arguments = arguments or {}
        result = self._policy.decide(spec, arguments)
        if result.decision is Decision.ALLOW:
            return None
        if result.decision is Decision.DENY:
            return _denied(result.reason, result.rule)

        if self._approver is not None:
            approval = self._approver(call, spec, arguments)
            self._approvals[call.id] = approval
            if approval.approved:
                return None
            return _denied(f"approval denied: {approval.reason}")

        if self._can_suspend:
            request = ApprovalRequest(
                call_id=call.id,
                tool=spec.name,
                risk=spec.risk,
                arguments_digest=digest_arguments(arguments),
            )
            raise ToolSuspended(
                "approval_required", kind="approval", payload=request.to_dict()
            )
        return _denied(
            "approval is required but no approval channel is configured",
            result.rule,
        )