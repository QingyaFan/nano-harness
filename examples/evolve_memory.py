"""Memory distillation lane: candidate provenance, conflicts, and source revocation."""

from __future__ import annotations

from nano_harness.distill import (
    CandidateSource,
    memory_candidate,
    revoke_memory_source,
)
from nano_harness.memory import (
    JsonMemoryStore,
    MemoryService,
    MemorySource,
    MemoryWrite,
)


def main() -> None:
    service = MemoryService(JsonMemoryStore())
    candidate = memory_candidate(
        "service.api.timeout",
        "60s",
        proposer="meta-agent",
        source=CandidateSource("run-001", "transcript-1"),
        confidence=0.8,
    )
    print(
        f"candidate: kind={candidate.kind.value} risk={candidate.risk.value} "
        f"confidence={candidate.confidence}"
    )

    def write(key: str, value: str, run_id: str, **kwargs):
        return service.write(
            MemoryWrite(
                scope="user:demo",
                key=key,
                value=value,
                source=MemorySource(run_id, "remember"),
                **kwargs,
            )
        )

    created = write("service.api.timeout", "30s", "run-001")
    duplicate = write("service.api.timeout", "30s", "run-001")
    conflict = write("service.api.timeout", "60s", "run-001")
    updated = write(
        "service.api.timeout", "60s", "run-001", expected_version=1
    )
    write("service.api.retries", "3", "run-002")
    print(
        "writes: "
        f"created=v{created.record.version} "
        f"duplicate=v{duplicate.record.version} "
        f"conflict_current={conflict.current.value} "
        f"updated=v{updated.record.version}"
    )

    revoked = revoke_memory_source(service, "user:demo", "run-001")
    keys = [record.key for record in service.list("user:demo")]
    print(f"revoke: run-001 removed={revoked}")
    print(f"recall: {keys}")


if __name__ == "__main__":
    main()