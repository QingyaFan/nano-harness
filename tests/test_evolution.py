import pytest

from nano_harness.artifact import (
    ArtifactError,
    ArtifactKind,
    ArtifactRegistry,
    ArtifactStatus,
)
from nano_harness.distill import (
    CandidateSource,
    make_candidate,
    memory_candidate,
    policy_candidate,
    revoke_memory_source,
    skill_candidate,
)
from nano_harness.eval import CaseAggregate, EvalBaseline, SuiteResult
from nano_harness.evolution import (
    Approval,
    ApprovalError,
    CanaryController,
    CanaryPolicy,
    CanarySample,
    EvalGate,
    ReleaseManager,
)
from nano_harness.memory import JsonMemoryStore, MemoryService, MemorySource, MemoryWrite
from nano_harness.risk import RiskLevel


def source(run_id: str = "run-001") -> CandidateSource:
    return CandidateSource(run_id, "transcript-1")


def suite(rate: float, cases: int = 1) -> SuiteResult:
    aggregate = CaseAggregate(
        trials=3,
        successes=3 if rate == 1.0 else 0,
        rate=rate,
        pass_power_k=rate == 1.0,
    )
    return SuiteResult(
        suite="skill",
        repeats=3,
        cases={"echo": aggregate},
        pass_power_k=rate,
        confidence_interval=(0.0, 1.0),
        eval_set_hash="hash",
        results=(),
    )


def baseline(rate: float) -> EvalBaseline:
    return EvalBaseline(
        suite="skill",
        eval_set_hash="hash",
        repeats=3,
        pass_power_k=rate,
        case_rates={"echo": rate},
    )


def test_registry_promotes_supersedes_and_rolls_back() -> None:
    registry = ArtifactRegistry()
    first = registry.register(
        "pdf-forms", ArtifactKind.SKILL, "v1", proposer="human"
    )
    second = registry.register(
        "pdf-forms", ArtifactKind.SKILL, "v2", proposer="meta"
    )
    assert first.version == 1
    assert second.version == 2
    assert second.parent_version == 1

    registry.promote("pdf-forms", 1)
    registry.promote("pdf-forms", 2)
    assert registry.stable_version("pdf-forms") == 2
    assert registry.get("pdf-forms", 1).status is ArtifactStatus.SUPERSEDED

    registry.pin("run-1", "pdf-forms", 2)
    rolled = registry.rollback("pdf-forms")
    assert rolled.version == 1
    assert registry.get("pdf-forms", 2).status is ArtifactStatus.ROLLED_BACK
    assert registry.resolve("pdf-forms", "run-1").version == 2
    assert registry.resolve("pdf-forms", "run-2").version == 1


def test_eval_gate_rejects_regression_and_accepts_baseline() -> None:
    gate = EvalGate()
    rejected = gate.evaluate(suite(0.0), baseline(1.0))
    accepted = gate.evaluate(suite(1.0), baseline(1.0))

    assert rejected.passed is False
    assert "regression:echo" in rejected.failures
    assert accepted.passed is True


def test_release_manager_marks_canary_or_rejects() -> None:
    registry = ArtifactRegistry()
    manager = ReleaseManager(registry, gate=EvalGate())
    bad = registry.register("p", ArtifactKind.SKILL, "bad", proposer="meta")
    report = manager.evaluate(bad, suite(0.0), baseline(1.0))
    assert report.passed is False
    assert registry.get("p", 1).status is ArtifactStatus.REJECTED

    good = registry.register("p", ArtifactKind.SKILL, "good", proposer="meta")
    assert manager.evaluate(good, suite(1.0), baseline(1.0)).passed is True
    assert registry.get("p", 2).status is ArtifactStatus.CANARY


def test_canary_aborts_and_keeps_stable() -> None:
    registry = ArtifactRegistry()
    stable = registry.register("p", ArtifactKind.SKILL, "v1", proposer="human")
    registry.promote("p", stable.version)
    candidate = registry.register("p", ArtifactKind.SKILL, "v2", proposer="meta")
    registry.set_status("p", candidate.version, ArtifactStatus.CANARY)

    manager = ReleaseManager(
        registry, canary=CanaryController(CanaryPolicy(min_samples=10))
    )
    decision = manager.canary(candidate, CanarySample(20, 5))
    assert decision.abort is True
    assert decision.reason == "error_rate"
    assert registry.stable_version("p") == 1
    assert registry.get("p", 2).status is ArtifactStatus.ROLLED_BACK


def test_approval_rules() -> None:
    registry = ArtifactRegistry()
    manager = ReleaseManager(registry)
    registry.register("p", ArtifactKind.SKILL, "v1", proposer="human")
    registry.promote("p", 1)
    candidate = registry.register("p", ArtifactKind.SKILL, "v2", proposer="meta")

    with pytest.raises(ApprovalError):
        manager.promote(candidate, approvers=(Approval("meta"),))
    with pytest.raises(ApprovalError):
        manager.promote(candidate)
    promoted = manager.promote(candidate, approvers=(Approval("human-1"),))
    assert promoted.status is ArtifactStatus.STABLE

    registry.register("policy", ArtifactKind.POLICY, "p1", proposer="human")
    registry.promote("policy", 1)
    policy = registry.register("policy", ArtifactKind.POLICY, "p2", proposer="meta")
    with pytest.raises(ApprovalError):
        manager.promote(policy, approvers=(Approval("human-1"),))
    assert (
        manager.promote(
            policy,
            approvers=(Approval("human-1"), Approval("human-2")),
        ).status
        is ArtifactStatus.STABLE
    )


def test_memory_source_revocation() -> None:
    service = MemoryService(JsonMemoryStore())
    for run_id, key, value in (
        ("run-001", "service.api.timeout", "60s"),
        ("run-002", "service.api.retries", "3"),
    ):
        service.write(
            MemoryWrite(
                scope="user:a",
                key=key,
                value=value,
                source=MemorySource(run_id, "remember"),
            )
        )

    assert revoke_memory_source(service, "user:a", "run-001") == 1
    keys = [record.key for record in service.list("user:a")]
    assert keys == ["service.api.retries"]


def test_candidate_risk_mapping() -> None:
    assert (
        memory_candidate(
            "fact", "value", proposer="meta", source=source()
        ).risk
        is RiskLevel.WRITE
    )
    assert (
        skill_candidate(
            "skill", "body", proposer="meta", source=source()
        ).risk
        is RiskLevel.WRITE
    )
    assert (
        policy_candidate(
            "policy", "body", proposer="meta", source=source()
        ).risk
        is RiskLevel.DESTRUCTIVE
    )
    assert (
        make_candidate(
            "tool",
            ArtifactKind.TOOL,
            "body",
            proposer="meta",
            source=source(),
        ).risk
        is RiskLevel.EXEC
    )


def test_unknown_artifact_and_invalid_canary() -> None:
    registry = ArtifactRegistry()
    with pytest.raises(ArtifactError):
        registry.get("missing")
    with pytest.raises(ValueError):
        CanarySample(1, 2)