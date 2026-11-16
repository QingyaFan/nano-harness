"""A3 demo: delegated incident triage with grants, handoffs, and credit."""

from nano_harness.messages import Usage
from nano_harness.orchestration import (
    AgentNode,
    AgentRunResult,
    Authority,
    CompletionContract,
    Coverage,
    DelegationGrant,
    DelegationRequest,
    MultiAgentRuntime,
    TaskOwner,
)


def reader(name: str, source: str, evidence: str) -> AgentNode:
    def run(handoff, grant) -> AgentRunResult:
        return AgentRunResult.ok_result(
            f"{name}: checked",
            evidence=(evidence,),
            coverage=Coverage((source,), time_window="15m"),
            usage=Usage(20, 10),
        )

    return AgentNode(name, run)


def commander() -> AgentNode:
    def run(handoff, grant) -> AgentRunResult:
        return AgentRunResult.ok_result(
            "delegated",
            usage=Usage(30, 10),
            delegations=(
                DelegationRequest(
                    "logs",
                    "find error patterns",
                    frozenset({"logs"}),
                    evidence=("handoff:outage",),
                    coverage=Coverage(("logs",)),
                ),
                DelegationRequest(
                    "metrics",
                    "check resource saturation",
                    frozenset({"metrics"}),
                    evidence=("handoff:outage",),
                    coverage=Coverage(("metrics",)),
                ),
                DelegationRequest(
                    "changes",
                    "check deploys and config",
                    frozenset({"config"}),
                    evidence=("handoff:outage",),
                    coverage=Coverage(("config",)),
                ),
            ),
        )

    def aggregate(handoff, grant, children) -> AgentRunResult:
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
            "root cause: connection pool size changed to 20",
            evidence=evidence,
            coverage=coverage,
            usage=Usage(40, 20),
        )

    return AgentNode("commander", run, aggregate)


def main() -> None:
    owner = TaskOwner("oncall")
    scope = frozenset({"logs", "metrics", "config"})
    grant = DelegationGrant(
        "grant-root",
        "oncall",
        "commander",
        "resolve payment latency incident",
        scope,
        Authority(read=scope),
        can_delegate=True,
    )
    runtime = MultiAgentRuntime(
        {
            "commander": commander(),
            "logs": reader("logs", "logs", "log: pool exhausted"),
            "metrics": reader("metrics", "metrics", "metrics: CPU normal"),
            "changes": reader("changes", "config", "config: pool size 20"),
        },
        owner=owner,
        completion=CompletionContract(
            "resolve payment latency incident",
            owner,
            require_evidence=True,
        ),
        max_depth=2,
        max_fanout=3,
    )

    result = runtime.run(
        "resolve payment latency incident",
        root_agent="commander",
        grant=grant,
    )

    print("status:", result.status.value)
    print("output:", result.output)
    print("evidence:", result.evidence)
    for node in result.tree.nodes():
        print(
            f"node: {node.agent} status={node.status.value} "
            f"children={len(node.children)}"
        )
    print("credit:")
    for agent, credit in result.tree.credit().items():
        print(f"  {agent}: {credit.to_dict()}")
    print("failures:", result.failures)


if __name__ == "__main__":
    main()