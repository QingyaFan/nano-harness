import pytest

from nano_harness.messages import Usage
from nano_harness.orchestration import (
    AgentNode,
    AgentRunResult,
    Authority,
    CompletionContract,
    ContractError,
    Coverage,
    DelegationGrant,
    DelegationRequest,
    MultiAgentRuntime,
    TaskOwner,
    TaskStatus,
)


def owner() -> TaskOwner:
    return TaskOwner("oncall")


def completion(*, max_failures: int = 0) -> CompletionContract:
    return CompletionContract("resolve incident", owner(), max_failures=max_failures)


def root_grant(*, can_delegate: bool = True) -> DelegationGrant:
    scope = frozenset({"logs", "metrics"})
    return DelegationGrant(
        "g-root",
        "oncall",
        "commander",
        "resolve incident",
        scope,
        Authority(read=scope),
        can_delegate=can_delegate,
    )


def request(agent: str, scope: str) -> DelegationRequest:
    return DelegationRequest(
        agent,
        f"check {agent}",
        frozenset({scope}),
        evidence=(f"parent:{agent}",),
        coverage=Coverage((scope,)),
    )


def commander(
    delegations: tuple[DelegationRequest, ...] = (),
    *,
    aggregate: bool = True,
) -> AgentNode:
    def run(handoff, grant) -> AgentRunResult:
        return AgentRunResult.ok_result(
            "delegated",
            usage=Usage(1, 1),
            delegations=delegations,
        )

    def aggregate_children(handoff, grant, children) -> AgentRunResult:
        evidence = tuple(item for child in children for item in child.evidence)
        coverage = None
        for child in children:
            if child.coverage is not None:
                coverage = (
                    child.coverage
                    if coverage is None
                    else coverage.merge(child.coverage)
                )
        return AgentRunResult.ok_result(
            "root cause",
            evidence=evidence,
            coverage=coverage,
            usage=Usage(1, 1),
        )

    return AgentNode(
        "commander",
        run,
        aggregate_children if aggregate else None,
    )


def child(name: str, *, ok: bool = True) -> AgentNode:
    def run(handoff, grant) -> AgentRunResult:
        if not ok:
            return AgentRunResult.failure("boom")
        return AgentRunResult.ok_result(
            f"{name} result",
            evidence=(f"{name}:1",),
            coverage=Coverage((name,)),
            usage=Usage(1, 1),
        )

    return AgentNode(name, run)


def runtime(
    *,
    max_depth: int = 3,
    max_fanout: int = 4,
    commander_node: AgentNode | None = None,
) -> MultiAgentRuntime:
    return MultiAgentRuntime(
        {
            "commander": commander_node
            or commander((request("logs", "logs"), request("metrics", "metrics"))),
            "logs": child("logs"),
            "metrics": child("metrics"),
        },
        owner=owner(),
        completion=completion(),
        max_depth=max_depth,
        max_fanout=max_fanout,
    )


def test_multi_agent_task_completes_and_credits() -> None:
    result = runtime().run("outage", root_agent="commander", grant=root_grant())

    assert result.status is TaskStatus.COMPLETED
    assert len(result.tree.nodes()) == 3
    root = result.tree.nodes()[0]
    assert len(result.tree.children(root.task_id)) == 2
    assert result.evidence == ("logs:1", "metrics:1")
    assert result.coverage is not None
    assert set(result.coverage.sources) == {"logs", "metrics"}
    credit = result.tree.credit()
    assert credit["commander"].runs == 1
    assert credit["logs"].runs == 1
    assert credit["metrics"].runs == 1
    assert result.usage.total_tokens == 6
    event_types = [event.type for event in result.events]
    assert "task_started" in event_types
    assert "task_delegated" in event_types
    assert "task_finished" in event_types


def test_delegation_requires_grant() -> None:
    result = runtime().run(
        "outage", root_agent="commander", grant=root_grant(can_delegate=False)
    )

    assert result.status is TaskStatus.FAILED
    assert result.tree.nodes()[0].error == "delegation_not_allowed"


def test_fanout_limit_fails_before_children_run() -> None:
    result = runtime(max_fanout=1).run(
        "outage", root_agent="commander", grant=root_grant()
    )

    assert result.status is TaskStatus.FAILED
    assert result.tree.nodes()[0].error == "fanout_exceeded"
    assert len(result.tree.nodes()) == 1


def test_child_failure_makes_task_partial() -> None:
    runtime_with_failure = MultiAgentRuntime(
        {
            "commander": commander(
                (request("logs", "logs"), request("metrics", "metrics"))
            ),
            "logs": child("logs", ok=False),
            "metrics": child("metrics"),
        },
        owner=owner(),
        completion=completion(),
    )

    result = runtime_with_failure.run(
        "outage", root_agent="commander", grant=root_grant()
    )

    assert result.status is TaskStatus.PARTIAL
    assert any(":boom" in failure for failure in result.failures)
    assert result.failures


def test_missing_aggregator_fails_parent() -> None:
    node = commander(
        (request("logs", "logs"), request("metrics", "metrics")),
        aggregate=False,
    )
    result = runtime(commander_node=node).run(
        "outage", root_agent="commander", grant=root_grant()
    )

    assert result.status is TaskStatus.FAILED
    assert result.tree.nodes()[0].error == "missing_aggregator"
    assert len(result.tree.nodes()) == 3


def test_child_grant_cannot_expand_scope() -> None:
    bad = commander((request("logs", "config"),), aggregate=True)
    runtime_with_bad = MultiAgentRuntime(
        {"commander": bad, "logs": child("logs")},
        owner=owner(),
        completion=completion(),
    )

    result = runtime_with_bad.run(
        "outage", root_agent="commander", grant=root_grant()
    )

    assert result.status is TaskStatus.FAILED
    assert "grant_error" in (result.tree.nodes()[0].error or "")


def test_depth_limit_is_recorded() -> None:
    result = runtime(max_depth=0).run(
        "outage", root_agent="commander", grant=root_grant()
    )

    assert result.status is TaskStatus.PARTIAL
    children = [
        node for node in result.tree.nodes() if node.status is TaskStatus.DEPTH_EXCEEDED
    ]
    assert len(children) == 2


def test_completion_requires_evidence() -> None:
    node = AgentNode(
        "commander",
        lambda handoff, grant: AgentRunResult.ok_result("done"),
    )
    result = MultiAgentRuntime(
        {"commander": node},
        owner=owner(),
        completion=completion(),
    ).run("outage", root_agent="commander", grant=root_grant(can_delegate=False))

    assert result.status is TaskStatus.PARTIAL
    assert result.completion_reason == "missing_evidence"


def test_cancel_before_root_run() -> None:
    class Cancelled:
        cancelled = True

    result = runtime().run(
        "outage",
        root_agent="commander",
        grant=root_grant(),
        cancel=Cancelled(),
    )

    assert result.status is TaskStatus.CANCELLED
    assert len(result.tree.nodes()) == 1


def test_owner_and_grant_must_agree() -> None:
    with pytest.raises(ContractError):
        MultiAgentRuntime(
            {"commander": commander(), "logs": child("logs"), "metrics": child("metrics")},
            owner=TaskOwner("oncall"),
            completion=CompletionContract("resolve", TaskOwner("other")),
        )

    with pytest.raises(ContractError):
        runtime().run(
            "outage",
            root_agent="commander",
            grant=DelegationGrant(
                "g",
                "other",
                "commander",
                "resolve",
                frozenset({"logs"}),
            ),
        )

    with pytest.raises(ContractError):
        runtime().run(
            "outage",
            root_agent="commander",
            grant=DelegationGrant(
                "g",
                "oncall",
                "other",
                "resolve",
                frozenset({"logs"}),
            ),
        )


def test_unknown_root_agent_is_rejected() -> None:
    with pytest.raises(ContractError):
        runtime().run("outage", root_agent="missing", grant=root_grant())