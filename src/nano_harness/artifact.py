"""Versioned artifacts, status transitions, stable pointers, and run pins (E2)."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class ArtifactKind(StrEnum):
    PROMPT = "prompt"
    SKILL = "skill"
    POLICY = "policy"
    MEMORY_SCHEMA = "memory_schema"
    TOOL = "tool"


class ArtifactStatus(StrEnum):
    CANDIDATE = "candidate"
    CANARY = "canary"
    STABLE = "stable"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class ArtifactError(Exception):
    pass


@dataclass(frozen=True)
class Artifact:
    name: str
    kind: ArtifactKind
    version: int
    content: str
    content_hash: str
    proposer: str
    status: ArtifactStatus = ArtifactStatus.CANDIDATE
    parent_version: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ArtifactError("artifact name must not be empty")
        if self.version < 1:
            raise ArtifactError("artifact version must be at least 1")
        if not self.proposer.strip():
            raise ArtifactError("artifact proposer must not be empty")
        if not isinstance(self.kind, ArtifactKind):
            object.__setattr__(self, "kind", ArtifactKind(self.kind))
        if not isinstance(self.status, ArtifactStatus):
            object.__setattr__(self, "status", ArtifactStatus(self.status))

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "content": self.content,
            "content_hash": self.content_hash,
            "proposer": self.proposer,
            "status": self.status.value,
            "parent_version": self.parent_version,
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
        }


class ArtifactRegistry:
    """In-memory registry; production swaps in a transactional store."""

    def __init__(self) -> None:
        self._artifacts: dict[str, list[Artifact]] = {}
        self._stable: dict[str, int] = {}
        self._pins: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        kind: ArtifactKind,
        content: str,
        *,
        proposer: str,
        parent_version: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Artifact:
        with self._lock:
            history = self._artifacts.get(name, [])
            version = max((item.version for item in history), default=0) + 1
            if parent_version is None:
                parent_version = self._stable.get(name)
                if parent_version is None and history:
                    parent_version = history[-1].version
            artifact = Artifact(
                name=name,
                kind=kind,
                version=version,
                content=content,
                content_hash=self._hash(content),
                proposer=proposer,
                parent_version=parent_version,
                provenance=dict(provenance or {}),
            )
            self._artifacts.setdefault(name, []).append(artifact)
            return artifact

    def get(self, name: str, version: int | None = None) -> Artifact:
        with self._lock:
            history = self._artifacts.get(name, [])
            if not history:
                raise ArtifactError(f"unknown artifact {name!r}")
            if version is None:
                return history[-1]
            for artifact in history:
                if artifact.version == version:
                    return artifact
            raise ArtifactError(f"unknown version {name}@{version}")

    def history(self, name: str) -> tuple[Artifact, ...]:
        with self._lock:
            return tuple(self._artifacts.get(name, ()))

    def stable_version(self, name: str) -> int | None:
        with self._lock:
            return self._stable.get(name)

    def stable(self, name: str) -> Artifact | None:
        version = self.stable_version(name)
        return self.get(name, version) if version is not None else None

    def set_status(
        self, name: str, version: int, status: ArtifactStatus
    ) -> Artifact:
        with self._lock:
            history = self._artifacts.get(name, [])
            for index, artifact in enumerate(history):
                if artifact.version != version:
                    continue
                updated = replace(artifact, status=ArtifactStatus(status))
                history[index] = updated
                if status is ArtifactStatus.STABLE:
                    self._stable[name] = version
                return updated
            raise ArtifactError(f"unknown version {name}@{version}")

    def reject(self, name: str, version: int) -> Artifact:
        return self.set_status(name, version, ArtifactStatus.REJECTED)

    def promote(self, name: str, version: int) -> Artifact:
        with self._lock:
            history = self._artifacts.get(name, [])
            target = next(
                (item for item in history if item.version == version), None
            )
            if target is None:
                raise ArtifactError(f"unknown version {name}@{version}")
            if target.status is ArtifactStatus.REJECTED:
                raise ArtifactError(f"{target.label} is rejected")
            current = self._stable.get(name)
            if current is not None and current != version:
                for index, artifact in enumerate(history):
                    if artifact.version == current:
                        history[index] = replace(
                            artifact, status=ArtifactStatus.SUPERSEDED
                        )
            for index, artifact in enumerate(history):
                if artifact.version == version:
                    updated = replace(artifact, status=ArtifactStatus.STABLE)
                    history[index] = updated
                    self._stable[name] = version
                    return updated
            raise ArtifactError(f"unknown version {name}@{version}")

    def rollback(self, name: str, to_version: int | None = None) -> Artifact:
        with self._lock:
            history = self._artifacts.get(name, [])
            current_version = self._stable.get(name)
            if current_version is None:
                raise ArtifactError(f"{name!r} has no stable version")
            if to_version is None:
                candidates = [
                    item
                    for item in history
                    if item.status is ArtifactStatus.SUPERSEDED
                ]
                if not candidates:
                    raise ArtifactError(f"{name!r} has no previous stable version")
                to_version = max(item.version for item in candidates)
            target = next(
                (item for item in history if item.version == to_version), None
            )
            if target is None or target.status is ArtifactStatus.REJECTED:
                raise ArtifactError(f"cannot roll back to {name}@{to_version}")
            result = None
            for index, artifact in enumerate(history):
                if artifact.version == current_version:
                    history[index] = replace(
                        artifact, status=ArtifactStatus.ROLLED_BACK
                    )
                elif artifact.version == to_version:
                    result = replace(
                        artifact, status=ArtifactStatus.STABLE
                    )
                    history[index] = result
            self._stable[name] = to_version
            if result is None:
                raise ArtifactError(f"unknown version {name}@{to_version}")
            return result

    def pin(self, run_id: str, name: str, version: int) -> None:
        self.get(name, version)
        with self._lock:
            self._pins.setdefault(run_id, {})[name] = version

    def pins(self, run_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self._pins.get(run_id, {}))

    def resolve(self, name: str, run_id: str | None = None) -> Artifact:
        if run_id is not None:
            pinned = self.pins(run_id).get(name)
            if pinned is not None:
                return self.get(name, pinned)
        artifact = self.stable(name)
        if artifact is None:
            raise ArtifactError(f"{name!r} has no stable version")
        return artifact

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]