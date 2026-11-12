"""Experience distillation: candidate sources, candidates, and provenance revocation (E2)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .artifact import ArtifactKind
from .memory import MemoryService
from .risk import RiskLevel


@dataclass(frozen=True)
class CandidateSource:
    run_id: str
    transcript_id: str = ""
    observed_at: float = field(default_factory=time.time)
    trust: str = "medium"
    samples: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "transcript_id": self.transcript_id,
            "observed_at": self.observed_at,
            "trust": self.trust,
            "samples": self.samples,
        }


def risk_for(kind: ArtifactKind) -> RiskLevel:
    if kind is ArtifactKind.POLICY:
        return RiskLevel.DESTRUCTIVE
    if kind is ArtifactKind.TOOL:
        return RiskLevel.EXEC
    return RiskLevel.WRITE


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: ArtifactKind
    content: str
    proposer: str
    source: CandidateSource
    confidence: float = 0.5
    evidence: tuple[str, ...] = ()
    risk: RiskLevel | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            object.__setattr__(self, "kind", ArtifactKind(self.kind))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")
        if not self.name.strip() or not self.proposer.strip():
            raise ValueError("candidate name and proposer must not be empty")
        if self.risk is None:
            object.__setattr__(self, "risk", risk_for(self.kind))
        elif not isinstance(self.risk, RiskLevel):
            object.__setattr__(self, "risk", RiskLevel(self.risk))


def make_candidate(
    name: str,
    kind: ArtifactKind,
    content: str,
    *,
    proposer: str,
    source: CandidateSource,
    confidence: float = 0.5,
    evidence: tuple[str, ...] = (),
    risk: RiskLevel | None = None,
    metadata: dict[str, object] | None = None,
) -> Candidate:
    return Candidate(
        name=name,
        kind=kind,
        content=content,
        proposer=proposer,
        source=source,
        confidence=confidence,
        evidence=evidence,
        risk=risk,
        metadata=dict(metadata or {}),
    )


def memory_candidate(
    name: str,
    content: str,
    *,
    proposer: str,
    source: CandidateSource,
    confidence: float = 0.6,
    evidence: tuple[str, ...] = (),
) -> Candidate:
    return make_candidate(
        name,
        ArtifactKind.MEMORY_SCHEMA,
        content,
        proposer=proposer,
        source=source,
        confidence=confidence,
        evidence=evidence,
    )


def skill_candidate(
    name: str,
    content: str,
    *,
    proposer: str,
    source: CandidateSource,
    confidence: float = 0.6,
    evidence: tuple[str, ...] = (),
) -> Candidate:
    return make_candidate(
        name,
        ArtifactKind.SKILL,
        content,
        proposer=proposer,
        source=source,
        confidence=confidence,
        evidence=evidence,
    )


def policy_candidate(
    name: str,
    content: str,
    *,
    proposer: str,
    source: CandidateSource,
    confidence: float = 0.5,
    evidence: tuple[str, ...] = (),
) -> Candidate:
    return make_candidate(
        name,
        ArtifactKind.POLICY,
        content,
        proposer=proposer,
        source=source,
        confidence=confidence,
        evidence=evidence,
    )


class Distiller(Protocol):
    def distill(self, source: CandidateSource) -> tuple[Candidate, ...]: ...


def revoke_memory_source(
    service: MemoryService, scope: str, run_id: str
) -> int:
    revoked = 0
    for record in service.list(scope):
        if record.source.run_id != run_id:
            continue
        service.forget(scope, record.key, expected_version=record.version)
        revoked += 1
    return revoked