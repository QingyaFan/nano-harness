"""A3 contracts: delegation grants, handoffs, coverage, and completion."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ContractError(Exception):
    pass


class GrantError(ContractError):
    pass


class CoverageError(ContractError):
    pass


@dataclass(frozen=True)
class Coverage:
    sources: tuple[str, ...]
    missing: tuple[str, ...] = ()
    time_window: str = ""

    def __post_init__(self) -> None:
        overlap = set(self.sources) & set(self.missing)
        if overlap:
            raise CoverageError(
                f"source cannot be both checked and missing: {sorted(overlap)}"
            )

    @property
    def complete(self) -> bool:
        return not self.missing

    @classmethod
    def full(
        cls, sources: tuple[str, ...], *, time_window: str = ""
    ) -> Coverage:
        return cls(sources=sources, time_window=time_window)

    def merge(self, other: Coverage) -> Coverage:
        sources = tuple(dict.fromkeys((*self.sources, *other.sources)))
        missing = tuple(dict.fromkeys((*self.missing, *other.missing)))
        window = self.time_window or other.time_window
        return Coverage(sources, missing, window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "missing": list(self.missing),
            "time_window": self.time_window,
        }


@dataclass(frozen=True)
class Authority:
    read: frozenset[str] = frozenset()
    write: frozenset[str] = frozenset()
    tools: frozenset[str] = frozenset()

    def allows_tool(self, name: str) -> bool:
        return name in self.tools

    def subset_of(self, other: Authority) -> bool:
        return (
            self.read <= other.read
            and self.write <= other.write
            and self.tools <= other.tools
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "read": sorted(self.read),
            "write": sorted(self.write),
            "tools": sorted(self.tools),
        }


@dataclass(frozen=True)
class DelegationGrant:
    grant_id: str
    principal: str
    delegate: str
    goal: str
    scope: frozenset[str]
    authority: Authority = field(default_factory=Authority)
    budget_tokens: int | None = None
    deadline: float | None = None
    can_delegate: bool = False
    parent_grant: str | None = None

    def __post_init__(self) -> None:
        if not self.grant_id.strip():
            raise ContractError("grant_id must not be empty")
        if not self.principal.strip() or not self.delegate.strip():
            raise ContractError("principal and delegate must not be empty")
        if not self.goal.strip():
            raise ContractError("goal must not be empty")
        if self.budget_tokens is not None and self.budget_tokens < 1:
            raise ContractError("budget_tokens must be at least 1 when set")

    def assert_tool(self, name: str) -> None:
        if not self.authority.allows_tool(name):
            raise GrantError(f"tool not granted: {name}")

    def child(
        self,
        *,
        delegate: str,
        goal: str,
        scope: frozenset[str],
        authority: Authority,
        budget_tokens: int | None = None,
        deadline: float | None = None,
        can_delegate: bool = False,
    ) -> DelegationGrant:
        if not self.can_delegate:
            raise GrantError("this grant does not allow further delegation")
        if not scope <= self.scope:
            raise GrantError("child scope must be a subset of the parent scope")
        if not authority.subset_of(self.authority):
            raise GrantError("child authority must be a subset of the parent authority")
        if (
            self.budget_tokens is not None
            and budget_tokens is not None
            and budget_tokens > self.budget_tokens
        ):
            raise GrantError("child budget cannot exceed the parent budget")
        if self.deadline is not None and deadline is not None and deadline > self.deadline:
            raise GrantError("child deadline cannot exceed the parent deadline")
        return DelegationGrant(
            grant_id=uuid.uuid4().hex[:12],
            principal=self.delegate,
            delegate=delegate,
            goal=goal,
            scope=scope,
            authority=authority,
            budget_tokens=budget_tokens,
            deadline=deadline if deadline is not None else self.deadline,
            can_delegate=can_delegate,
            parent_grant=self.grant_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "principal": self.principal,
            "delegate": self.delegate,
            "goal": self.goal,
            "scope": sorted(self.scope),
            "authority": self.authority.to_dict(),
            "budget_tokens": self.budget_tokens,
            "deadline": self.deadline,
            "can_delegate": self.can_delegate,
            "parent_grant": self.parent_grant,
        }


@dataclass(frozen=True)
class Handoff:
    task_id: str
    from_agent: str
    to_agent: str
    goal: str
    scope: frozenset[str]
    evidence: tuple[str, ...] = ()
    coverage: Coverage = field(default_factory=lambda: Coverage(()))
    deadline: float | None = None
    stop_condition: str = ""

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ContractError("task_id must not be empty")
        if not self.from_agent.strip() or not self.to_agent.strip():
            raise ContractError("from_agent and to_agent must not be empty")
        if not self.goal.strip():
            raise ContractError("goal must not be empty")

    def child(
        self,
        *,
        task_id: str,
        to_agent: str,
        goal: str,
        scope: frozenset[str],
        evidence: tuple[str, ...] = (),
        coverage: Coverage | None = None,
        deadline: float | None = None,
        stop_condition: str = "",
    ) -> Handoff:
        if not scope <= self.scope:
            raise ContractError("child scope must be a subset of the parent scope")
        return Handoff(
            task_id=task_id,
            from_agent=self.to_agent,
            to_agent=to_agent,
            goal=goal,
            scope=scope,
            evidence=(*self.evidence, *evidence),
            coverage=coverage or self.coverage,
            deadline=deadline if deadline is not None else self.deadline,
            stop_condition=stop_condition or self.stop_condition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "goal": self.goal,
            "scope": sorted(self.scope),
            "evidence": list(self.evidence),
            "coverage": self.coverage.to_dict(),
            "deadline": self.deadline,
            "stop_condition": self.stop_condition,
        }


class OwnerKind(StrEnum):
    HUMAN = "human"
    AGENT = "agent"


@dataclass(frozen=True)
class TaskOwner:
    owner_id: str
    kind: OwnerKind = OwnerKind.HUMAN
    can_stop: bool = True
    can_merge: bool = True
    can_escalate: bool = True

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ContractError("owner_id must not be empty")
        if not isinstance(self.kind, OwnerKind):
            object.__setattr__(self, "kind", OwnerKind(self.kind))
        if not (self.can_stop or self.can_merge or self.can_escalate):
            raise ContractError("owner must retain at least one authority")


@dataclass(frozen=True)
class CompletionContract:
    goal: str
    owner: TaskOwner
    require_evidence: bool = True
    max_failures: int = 0

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ContractError("completion goal must not be empty")
        if self.max_failures < 0:
            raise ContractError("max_failures must not be negative")
        if not self.owner.can_merge:
            raise ContractError("completion owner must be able to merge")

    def check(
        self,
        *,
        ok: bool,
        output: Any,
        evidence: tuple[str, ...],
        failures: tuple[str, ...],
    ) -> str | None:
        if not ok:
            return "task_failed"
        if not str(output or "").strip():
            return "empty_output"
        if self.require_evidence and not evidence:
            return "missing_evidence"
        if len(failures) > self.max_failures:
            return "child_failures"
        return None


@dataclass(frozen=True)
class ApprovalRecord:
    approver: str
    role: str
    scope: str
    approved: bool
    reason: str = ""
    approved_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.approver.strip() or not self.role.strip():
            raise ContractError("approver and role must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "role": self.role,
            "scope": self.scope,
            "approved": self.approved,
            "reason": self.reason,
            "approved_at": self.approved_at,
        }