"""Risk-based permission policy and the approval gate (I5)."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    POLICY_DECISION,
    REDACTION_APPLIED,
)
from ..tools.errors import ErrorKind, ToolError, ToolOutcome, ToolSuspended
from ..tools.spec import ToolSpec
from .approval import Approval, ApprovalRequest, digest_arguments, parse_resume_value
from .redaction import redact_value
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
            return fnmatch.fnmatchcase(name, self.tool)
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
        events: Any = None,
        redactor: Any = None,
    ) -> None:
        self._policy = policy
        self._approver = approver
        self._can_suspend = can_suspend
        self._events = events
        self._redactor = redactor
        self._approvals: dict[str, Approval] = {}
        self._pending_tools: dict[str, str] = {}

    def bind_events(self, events: Any) -> None:
        self._events = events

    def _emit(self, event_type: str, turn: int, **data: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, turn, **data)

    def resolve(
        self, call_id: str, value: str, *, tool: str | None = None
    ) -> Approval:
        tool = tool or self._pending_tools.get(call_id, "")
        approval = parse_resume_value(value or "", call_id=call_id, tool=tool)
        if self._redactor is not None and approval.reason:
            reason, counts = redact_value(approval.reason)
            if counts:
                self._emit(
                    REDACTION_APPLIED, 0,
                    call_id=call_id, tool=tool, counts=counts,
                )
            approval = Approval(
                approval.approved, reason,
                source=approval.source, by=approval.by,
                call_id=approval.call_id, tool=approval.tool,
            )
        self._approvals[call_id] = approval
        self._emit(
            APPROVAL_RESOLVED, 0,
            approved=approval.approved, reason=approval.reason,
            by=approval.by, source=approval.source,
        )
        return approval

    def _request(
        self,
        call: Any,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: Any,
        turn: int,
    ) -> ApprovalRequest:
        display = dict(arguments)
        if self._redactor is not None:
            display, counts = redact_value(display)
            if counts:
                self._emit(
                    REDACTION_APPLIED, turn,
                    call_id=call.id, tool=spec.name, counts=counts,
                )
        return ApprovalRequest(
            call_id=call.id,
            tool=spec.name,
            risk=spec.risk,
            reason=result.reason,
            rule=result.rule,
            arguments=display,
            arguments_digest=digest_arguments(arguments),
        )

    def _callback(
        self, call: Any, spec: ToolSpec, arguments: dict[str, Any]
    ) -> Approval:
        try:
            approval = self._approver(call, spec, arguments)
        except Exception as exc:  # noqa: BLE001
            return Approval(
                False,
                f"approver failed: {type(exc).__name__}",
                source="callback",
                by="callback",
                call_id=call.id,
                tool=spec.name,
            )
        if not isinstance(approval, Approval):
            return Approval(
                False,
                "approver returned an unrecognized value",
                source="callback",
                by="callback",
                call_id=call.id,
                tool=spec.name,
            )
        return Approval(
            approval.approved,
            approval.reason,
            source="callback",
            by=approval.by or "reviewer",
            call_id=call.id,
            tool=spec.name,
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
            if approved.call_id and approved.call_id != call.id:
                raise ValueError("approval call_id does not match the pending call")
            if approved.tool and approved.tool != spec.name:
                raise ValueError("approval tool does not match the pending call")
            if approved.approved:
                return None
            return _denied(f"approval denied: {approved.reason}")

        arguments = arguments or {}
        result = self._policy.decide(spec, arguments)
        self._emit(
            POLICY_DECISION, turn,
            tool=spec.name, decision=result.decision.value,
            risk=result.risk.value, rule=result.rule,
            arguments_digest=digest_arguments(arguments),
        )
        if result.decision is Decision.ALLOW:
            return None
        if result.decision is Decision.DENY:
            return _denied(result.reason, result.rule)

        request = self._request(call, spec, arguments, result, turn)
        if self._approver is not None:
            approval = self._callback(call, spec, arguments)
            self._approvals[call.id] = approval
            self._emit(
                APPROVAL_RESOLVED, turn,
                approved=approval.approved, reason=approval.reason,
                by=approval.by, source=approval.source,
            )
            if approval.approved:
                return None
            return _denied(f"approval denied: {approval.reason}")

        if self._can_suspend:
            self._pending_tools[call.id] = spec.name
            self._emit(APPROVAL_REQUESTED, turn, **request.to_dict())
            raise ToolSuspended(
                "approval_required", kind="approval", payload=request.to_dict()
            )
        self._emit(
            APPROVAL_RESOLVED, turn,
            approved=False, reason="no approval channel",
            by="", source="unavailable",
        )
        return _denied(
            "approval is required but no approval channel is configured",
            result.rule,
        )