import pytest

from nano_harness.orchestration import (
    Authority,
    CompletionContract,
    Coverage,
    CoverageError,
    DelegationGrant,
    GrantError,
    TaskOwner,
)


def grant(*, can_delegate: bool = True) -> DelegationGrant:
    scope = frozenset({"logs", "metrics"})
    return DelegationGrant(
        "g1",
        "oncall",
        "commander",
        "resolve incident",
        scope,
        Authority(read=scope, tools=frozenset({"read_logs"})),
        can_delegate=can_delegate,
    )


def test_child_grant_narrows_scope_and_authority() -> None:
    parent = grant()
    child = parent.child(
        delegate="logs",
        goal="find errors",
        scope=frozenset({"logs"}),
        authority=Authority(read=frozenset({"logs"}), tools=frozenset({"read_logs"})),
    )

    assert child.parent_grant == "g1"
    assert child.principal == "commander"
    assert child.scope == {"logs"}


def test_child_grant_rejects_scope_expansion() -> None:
    with pytest.raises(GrantError):
        grant().child(
            delegate="logs",
            goal="look wider",
            scope=frozenset({"logs", "config"}),
            authority=Authority(read=frozenset({"logs", "config"})),
        )


def test_child_grant_rejects_tool_escalation() -> None:
    with pytest.raises(GrantError):
        grant().child(
            delegate="logs",
            goal="restart",
            scope=frozenset({"logs"}),
            authority=Authority(read=frozenset({"logs"}), tools=frozenset({"restart"})),
        )


def test_grant_must_allow_delegation() -> None:
    with pytest.raises(GrantError):
        grant(can_delegate=False).child(
            delegate="logs",
            goal="find errors",
            scope=frozenset({"logs"}),
            authority=Authority(read=frozenset({"logs"})),
        )


def test_assert_tool() -> None:
    grant().assert_tool("read_logs")
    with pytest.raises(GrantError):
        grant().assert_tool("restart")


def test_coverage_rejects_overlapping_sources() -> None:
    with pytest.raises(CoverageError):
        Coverage(("logs",), ("logs",))


def test_coverage_merge_keeps_missing() -> None:
    merged = Coverage(("logs",)).merge(Coverage(("metrics",), missing=("config",)))

    assert merged.sources == ("logs", "metrics")
    assert merged.missing == ("config",)
    assert not merged.complete


def test_completion_contract_requires_evidence() -> None:
    owner = TaskOwner("oncall")
    contract = CompletionContract("resolve", owner)

    assert contract.check(
        ok=True, output="done", evidence=("log:1",), failures=()
    ) is None
    assert contract.check(
        ok=True, output="done", evidence=(), failures=()
    ) == "missing_evidence"
    assert contract.check(
        ok=False, output=None, evidence=("log:1",), failures=()
    ) == "task_failed"