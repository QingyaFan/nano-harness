"""A3 multi-agent runtime: agents, task trees, delegation, and aggregation."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from ..events import EventLog
from ..messages import Usage
from .contracts import (
    Authority,
    CompletionContract,
    ContractError,
    Coverage,
    DelegationGrant,
    GrantError,
    Handoff,
    TaskOwner,
)

TASK_STARTED = "task_started"
TASK_NODE_STARTED = "task_node_started"
TASK_NODE_FINISHED = "task_node_finished"
TASK_DELEGATED = "task_delegated"
TASK_FINISHED = "task_finished"


class TaskStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    DEPTH_EXCEEDED = "depth_exceeded"
    FANOUT_EXCEEDED = "fanout_exceeded"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DelegationRequest:
    agent: str
    goal: str
    scope: frozenset[str]
    write: frozenset[str] = frozenset()
    tools: frozenset[str] = frozenset()
    budget_tokens: int | None = None
    deadline: float | None = None
    evidence: tuple[str, ...] = ()
    coverage: Coverage | None = None
    can_delegate: bool = False

    def __post_init__(self) -> None:
        if not self.agent.strip() or not self.goal.strip():
            raise ContractError("delegation agent and goal must not be empty")


@dataclass(frozen=True)
class AgentRunResult:
    ok: bool = True
    output: Any = None
    evidence: tuple[str, ...] = ()
    coverage: Coverage | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    delegations: tuple[DelegationRequest, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok_result(
        cls,
        output: Any = None,
        *,
        evidence: tuple[str, ...] = (),
        coverage: Coverage | None = None,
        usage: Usage | None = None,
        delegations: tuple[DelegationRequest, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        return cls(
            ok=True,
            output=output,
            evidence=evidence,
            coverage=coverage,
            usage=usage or Usage(),
            delegations=delegations,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        output: Any = None,
        usage: Usage | None = None,
        coverage: Coverage | None = None,
    ) -> AgentRunResult:
        return cls(
            ok=False,
            output=output,
            usage=usage or Usage(),
            error=error,
            coverage=coverage,
        )


@dataclass(frozen=True)
class TaskNode:
    task_id: str
    agent: str
    parent_id: str | None
    grant_id: str
    handoff: Handoff
    status: TaskStatus
    output: Any = None
    evidence: tuple[str, ...] = ()
    coverage: Coverage | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    agent: str
    status: TaskStatus
    output: Any = None
    evidence: tuple[str, ...] = ()
    coverage: Coverage | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.PARTIAL)


@dataclass(frozen=True)
class AgentCredit:
    runs: int = 0
    failures: int = 0
    tokens: int = 0
    evidence: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "runs": self.runs,
            "failures": self.failures,
            "tokens": self.tokens,
            "evidence": self.evidence,
        }


class TaskTree:
    def __init__(self) -> None:
        self._nodes: list[TaskNode] = []
        self._by_id: dict[str, TaskNode] = {}

    def add(self, node: TaskNode) -> None:
        if node.task_id in self._by_id:
            raise ContractError(f"duplicate task node: {node.task_id}")
        self._nodes.append(node)
        self._by_id[node.task_id] = node

    def update(self, task_id: str, **changes: Any) -> TaskNode:
        node = self._by_id[task_id]
        updated = replace(node, **changes)
        self._nodes[self._nodes.index(node)] = updated
        self._by_id[task_id] = updated
        return updated

    def get(self, task_id: str) -> TaskNode:
        return self._by_id[task_id]

    def nodes(self) -> tuple[TaskNode, ...]:
        return tuple(self._nodes)

    def children(self, parent_id: str) -> tuple[TaskNode, ...]:
        return tuple(node for node in self._nodes if node.parent_id == parent_id)

    def failures(self) -> tuple[str, ...]:
        failed = {
            TaskStatus.FAILED,
            TaskStatus.DEPTH_EXCEEDED,
            TaskStatus.FANOUT_EXCEEDED,
            TaskStatus.CANCELLED,
        }
        return tuple(
            f"{node.task_id}:{node.error or node.status.value}"
            for node in self._nodes
            if node.status in failed
        )

    def credit(self) -> dict[str, AgentCredit]:
        credits: dict[str, AgentCredit] = {}
        failed = {
            TaskStatus.FAILED,
            TaskStatus.DEPTH_EXCEEDED,
            TaskStatus.FANOUT_EXCEEDED,
            TaskStatus.CANCELLED,
        }
        for node in self._nodes:
            current = credits.get(node.agent, AgentCredit())
            credits[node.agent] = AgentCredit(
                runs=current.runs + 1,
                failures=current.failures + (1 if node.status in failed else 0),
                tokens=current.tokens + node.usage.total_tokens,
                evidence=current.evidence + len(node.evidence),
            )
        return credits

    def total_usage(self) -> Usage:
        total = Usage()
        for node in self._nodes:
            total = total + node.usage
        return total


@dataclass(frozen=True)
class AgentNode:
    name: str
    run: Callable[[Handoff, DelegationGrant], AgentRunResult]
    aggregate: (
        Callable[[Handoff, DelegationGrant, tuple[TaskOutcome, ...]], AgentRunResult]
        | None
    ) = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("agent name must not be empty")


@dataclass(frozen=True)
class TaskResult:
    status: TaskStatus
    output: Any
    evidence: tuple[str, ...]
    coverage: Coverage | None
    tree: TaskTree
    usage: Usage
    failures: tuple[str, ...]
    completion_reason: str
    events: tuple[Any, ...]
    owner_id: str


class MultiAgentRuntime:
    def __init__(
        self,
        agents: dict[str, AgentNode],
        *,
        owner: TaskOwner,
        completion: CompletionContract,
        max_depth: int = 3,
        max_fanout: int = 4,
        events: EventLog | None = None,
    ) -> None:
        if not agents:
            raise ContractError("at least one agent is required")
        if completion.owner.owner_id != owner.owner_id:
            raise ContractError("completion owner must match the task owner")
        if not owner.can_merge:
            raise ContractError("task owner must be able to merge results")
        if max_depth < 0:
            raise ContractError("max_depth must not be negative")
        if max_fanout < 1:
            raise ContractError("max_fanout must be at least 1")
        for name, agent in agents.items():
            if name != agent.name:
                raise ContractError(f"agent key mismatch: {name} != {agent.name}")
        self._agents = dict(agents)
        self._owner = owner
        self._completion = completion
        self._max_depth = max_depth
        self._max_fanout = max_fanout
        self._events = events

    def run(
        self,
        task: str,
        *,
        root_agent: str,
        grant: DelegationGrant,
        cancel: Any = None,
    ) -> TaskResult:
        if not task.strip():
            raise ContractError("task must not be empty")
        if root_agent not in self._agents:
            raise ContractError(f"unknown root agent: {root_agent}")
        if grant.delegate != root_agent:
            raise ContractError("root grant delegate must be the root agent")
        if grant.principal != self._owner.owner_id:
            raise ContractError("root grant principal must be the task owner")

        log = self._events or EventLog()
        tree = TaskTree()
        task_id = uuid.uuid4().hex[:12]
        handoff = Handoff(
            task_id=task_id,
            from_agent=self._owner.owner_id,
            to_agent=root_agent,
            goal=task,
            scope=grant.scope,
            coverage=Coverage(tuple(sorted(grant.scope))),
        )
        log.emit(
            TASK_STARTED,
            0,
            task_id=task_id,
            owner=self._owner.owner_id,
            root_agent=root_agent,
        )
        outcome = self._execute(
            handoff, grant, depth=0, parent_id=None, tree=tree, log=log, cancel=cancel
        )
        failures = tree.failures()
        completion_reason = self._completion.check(
            ok=outcome.ok,
            output=outcome.output,
            evidence=outcome.evidence,
            failures=failures,
        )
        if outcome.status in (
            TaskStatus.CANCELLED,
            TaskStatus.DEPTH_EXCEEDED,
            TaskStatus.FANOUT_EXCEEDED,
        ):
            status = outcome.status
        elif not outcome.ok:
            status = TaskStatus.FAILED
        elif outcome.status is TaskStatus.PARTIAL or completion_reason:
            status = TaskStatus.PARTIAL
        else:
            status = TaskStatus.COMPLETED

        log.emit(
            TASK_FINISHED,
            0,
            task_id=task_id,
            status=status.value,
            reason=completion_reason or "",
        )
        return TaskResult(
            status=status,
            output=outcome.output,
            evidence=outcome.evidence,
            coverage=outcome.coverage,
            tree=tree,
            usage=tree.total_usage(),
            failures=failures,
            completion_reason=completion_reason or "",
            events=log.snapshot(),
            owner_id=self._owner.owner_id,
        )

    def _execute(
        self,
        handoff: Handoff,
        grant: DelegationGrant,
        *,
        depth: int,
        parent_id: str | None,
        tree: TaskTree,
        log: EventLog,
        cancel: Any,
    ) -> TaskOutcome:
        if _cancelled(cancel):
            return self._terminal(
                handoff, grant, tree, log, parent_id, TaskStatus.CANCELLED, "cancelled"
            )
        if depth > self._max_depth:
            return self._terminal(
                handoff,
                grant,
                tree,
                log,
                parent_id,
                TaskStatus.DEPTH_EXCEEDED,
                "depth_exceeded",
            )
        agent = self._agents.get(grant.delegate)
        if agent is None:
            return self._terminal(
                handoff,
                grant,
                tree,
                log,
                parent_id,
                TaskStatus.FAILED,
                f"unknown_agent:{grant.delegate}",
            )

        node = TaskNode(
            task_id=handoff.task_id,
            agent=agent.name,
            parent_id=parent_id,
            grant_id=grant.grant_id,
            handoff=handoff,
            status=TaskStatus.RUNNING,
        )
        tree.add(node)
        log.emit(
            TASK_NODE_STARTED,
            depth,
            task_id=node.task_id,
            agent=agent.name,
            parent_id=parent_id,
            grant_id=grant.grant_id,
        )

        try:
            result = agent.run(handoff, grant)
        except Exception as exc:  # noqa: BLE001
            result = AgentRunResult.failure(f"{type(exc).__name__}: {exc}")
        if not isinstance(result, AgentRunResult):
            result = AgentRunResult.failure("invalid_agent_result")

        children: list[TaskOutcome] = []
        if result.ok and result.delegations:
            if not grant.can_delegate:
                result = AgentRunResult.failure(
                    "delegation_not_allowed", usage=result.usage
                )
            elif len(result.delegations) > self._max_fanout:
                result = AgentRunResult.failure(
                    "fanout_exceeded", usage=result.usage
                )
            else:
                for index, request in enumerate(result.delegations, start=1):
                    try:
                        child_grant = grant.child(
                            delegate=request.agent,
                            goal=request.goal,
                            scope=request.scope,
                            authority=Authority(
                                read=request.scope,
                                write=request.write,
                                tools=request.tools,
                            ),
                            budget_tokens=request.budget_tokens,
                            deadline=request.deadline,
                            can_delegate=request.can_delegate,
                        )
                        child_handoff = handoff.child(
                            task_id=f"{handoff.task_id}.{index}",
                            to_agent=request.agent,
                            goal=request.goal,
                            scope=request.scope,
                            evidence=request.evidence,
                            coverage=request.coverage,
                            deadline=request.deadline,
                        )
                    except (ContractError, GrantError) as exc:
                        result = AgentRunResult.failure(
                            f"grant_error:{exc}", usage=result.usage
                        )
                        break
                    log.emit(
                        TASK_DELEGATED,
                        depth,
                        task_id=node.task_id,
                        to_agent=request.agent,
                        grant_id=child_grant.grant_id,
                    )
                    children.append(
                        self._execute(
                            child_handoff,
                            child_grant,
                            depth=depth + 1,
                            parent_id=node.task_id,
                            tree=tree,
                            log=log,
                            cancel=cancel,
                        )
                    )
                if result.ok and children:
                    if agent.aggregate is None:
                        result = AgentRunResult.failure(
                            "missing_aggregator",
                            output=result.output,
                            usage=result.usage,
                        )
                    else:
                        try:
                            result = agent.aggregate(
                                handoff, grant, tuple(children)
                            )
                        except Exception as exc:  # noqa: BLE001
                            result = AgentRunResult.failure(
                                f"aggregate_error:{type(exc).__name__}: {exc}",
                                usage=result.usage,
                            )

        child_failed = any(not child.ok for child in children)
        if not result.ok:
            status = TaskStatus.FAILED
        elif child_failed:
            status = TaskStatus.PARTIAL
        else:
            status = TaskStatus.COMPLETED

        tree.update(
            node.task_id,
            status=status,
            output=result.output,
            evidence=result.evidence,
            coverage=result.coverage,
            usage=result.usage,
            error=result.error,
            children=tuple(child.task_id for child in children),
        )
        log.emit(
            TASK_NODE_FINISHED,
            depth,
            task_id=node.task_id,
            agent=agent.name,
            status=status.value,
            error=result.error,
        )
        return TaskOutcome(
            task_id=node.task_id,
            agent=agent.name,
            status=status,
            output=result.output,
            evidence=result.evidence,
            coverage=result.coverage,
            usage=result.usage,
            error=result.error,
        )

    def _terminal(
        self,
        handoff: Handoff,
        grant: DelegationGrant,
        tree: TaskTree,
        log: EventLog,
        parent_id: str | None,
        status: TaskStatus,
        error: str,
    ) -> TaskOutcome:
        node = TaskNode(
            task_id=handoff.task_id,
            agent=grant.delegate,
            parent_id=parent_id,
            grant_id=grant.grant_id,
            handoff=handoff,
            status=status,
            error=error,
        )
        tree.add(node)
        log.emit(
            TASK_NODE_FINISHED,
            0,
            task_id=node.task_id,
            agent=node.agent,
            status=status.value,
            error=error,
        )
        return TaskOutcome(
            task_id=node.task_id,
            agent=node.agent,
            status=status,
            error=error,
        )


def _cancelled(token: Any) -> bool:
    return bool(token is not None and getattr(token, "cancelled", False))