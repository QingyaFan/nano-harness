"""Release gates for evolved artifacts: eval, canary, approvals, rollback (E2)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .artifact import (
    Artifact,
    ArtifactRegistry,
    ArtifactStatus,
)
from .distill import Candidate, risk_for
from .eval import EvalBaseline, SuiteResult, compare_to_baseline
from .risk import RiskLevel


class ReleaseError(Exception):
    pass


class ApprovalError(ReleaseError):
    pass


@dataclass(frozen=True)
class EvalReport:
    passed: bool
    pass_power_k: float
    baseline_pass_power_k: float
    delta: float
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "pass_power_k": self.pass_power_k,
            "baseline_pass_power_k": self.baseline_pass_power_k,
            "delta": self.delta,
            "failures": list(self.failures),
        }


class EvalGate:
    def __init__(self, *, max_drop: float = 0.0) -> None:
        self._max_drop = max_drop

    def evaluate(
        self, current: SuiteResult, baseline: EvalBaseline
    ) -> EvalReport:
        comparison = compare_to_baseline(
            current, baseline, max_drop=self._max_drop
        )
        failures: list[str] = []
        if comparison.drift:
            failures.append("eval_set_drift")
        failures.extend(f"regression:{name}" for name in comparison.regressions)
        if comparison.aggregate_delta < -self._max_drop:
            failures.append("aggregate_drop")
        return EvalReport(
            passed=comparison.passed,
            pass_power_k=current.pass_power_k,
            baseline_pass_power_k=baseline.pass_power_k,
            delta=comparison.aggregate_delta,
            failures=tuple(failures),
        )


@dataclass(frozen=True)
class CanaryPolicy:
    max_error_rate: float = 0.05
    max_latency_ms: float = 2000.0
    max_duplicate_effects: int = 0
    min_samples: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate must be between 0 and 1")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")


@dataclass(frozen=True)
class CanarySample:
    requests: int
    errors: int
    latency_ms: float = 0.0
    duplicate_effects: int = 0

    def __post_init__(self) -> None:
        if self.requests < 0 or self.errors < 0:
            raise ValueError("requests and errors must be non-negative")
        if self.errors > self.requests:
            raise ValueError("errors cannot exceed requests")


@dataclass(frozen=True)
class CanaryDecision:
    abort: bool
    reason: str
    error_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "abort": self.abort,
            "reason": self.reason,
            "error_rate": self.error_rate,
        }


class CanaryController:
    def __init__(self, policy: CanaryPolicy) -> None:
        self._policy = policy
        self._artifact: Artifact | None = None
        self._fraction = 0.0

    def start(self, artifact: Artifact, fraction: float = 0.1) -> None:
        if not 0 < fraction < 1:
            raise ValueError("canary fraction must be between 0 and 1")
        self._artifact = artifact
        self._fraction = fraction

    def observe(self, sample: CanarySample) -> CanaryDecision:
        if self._artifact is None:
            raise ReleaseError("canary has not started")
        rate = sample.errors / sample.requests if sample.requests else 0.0
        if sample.requests < self._policy.min_samples:
            return CanaryDecision(False, "insufficient_samples", rate)
        if rate > self._policy.max_error_rate:
            return CanaryDecision(True, "error_rate", rate)
        if sample.latency_ms > self._policy.max_latency_ms:
            return CanaryDecision(True, "latency", rate)
        if sample.duplicate_effects > self._policy.max_duplicate_effects:
            return CanaryDecision(True, "duplicate_effects", rate)
        return CanaryDecision(False, "healthy", rate)


@dataclass(frozen=True)
class Approval:
    approver: str
    approved: bool = True
    reason: str = ""


class ReleaseManager:
    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        gate: EvalGate | None = None,
        canary: CanaryController | None = None,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._canary = canary

    def propose(self, candidate: Candidate) -> Artifact:
        return self._registry.register(
            candidate.name,
            candidate.kind,
            candidate.content,
            proposer=candidate.proposer,
            provenance={
                "source": candidate.source.to_dict(),
                "confidence": candidate.confidence,
                "evidence": list(candidate.evidence),
                "risk": candidate.risk.value if candidate.risk else None,
            },
        )

    def evaluate(
        self,
        artifact: Artifact,
        current: SuiteResult,
        baseline: EvalBaseline,
    ) -> EvalReport:
        if self._gate is None:
            report = EvalReport(
                passed=True,
                pass_power_k=current.pass_power_k,
                baseline_pass_power_k=baseline.pass_power_k,
                delta=current.pass_power_k - baseline.pass_power_k,
            )
        else:
            report = self._gate.evaluate(current, baseline)
        if report.passed:
            self._registry.set_status(
                artifact.name, artifact.version, ArtifactStatus.CANARY
            )
        else:
            self._registry.reject(artifact.name, artifact.version)
        return report

    def promote(
        self,
        artifact: Artifact,
        *,
        approvers: Iterable[Approval] = (),
    ) -> Artifact:
        risk = risk_for(artifact.kind)
        required = 2 if risk in (RiskLevel.DESTRUCTIVE, RiskLevel.EXEC) else 1
        approved = {
            approval.approver
            for approval in approvers
            if approval.approved
        }
        if artifact.proposer in approved:
            raise ApprovalError("the proposer cannot approve its own change")
        if len(approved) < required:
            raise ApprovalError(
                f"{artifact.kind.value} change requires {required} approver(s), "
                f"got {len(approved)}"
            )
        return self._registry.promote(artifact.name, artifact.version)

    def canary(
        self, artifact: Artifact, sample: CanarySample
    ) -> CanaryDecision:
        if self._canary is None:
            raise ReleaseError("no canary controller configured")
        self._canary.start(artifact)
        decision = self._canary.observe(sample)
        if decision.abort:
            stable = self._registry.stable(artifact.name)
            if stable is None or stable.version != artifact.version:
                self._registry.set_status(
                    artifact.name,
                    artifact.version,
                    ArtifactStatus.ROLLED_BACK,
                )
            else:
                self._registry.rollback(artifact.name)
        return decision

    def rollback(self, name: str, to_version: int | None = None) -> Artifact:
        return self._registry.rollback(name, to_version)

    def pin(self, run_id: str, name: str, version: int) -> None:
        self._registry.pin(run_id, name, version)

    def resolve(self, name: str, run_id: str | None = None) -> Artifact:
        return self._registry.resolve(name, run_id)