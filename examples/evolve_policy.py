"""Rule lane: policy changes require approval and cannot self-approve."""

from __future__ import annotations

from nano_harness.artifact import ArtifactKind, ArtifactRegistry
from nano_harness.evolution import Approval, ApprovalError, ReleaseManager


def main() -> None:
    registry = ArtifactRegistry()
    manager = ReleaseManager(registry)
    v1 = registry.register(
        "tool-policy", ArtifactKind.POLICY, "v1", proposer="human"
    )
    manager.promote(
        v1, approvers=(Approval("human-1"), Approval("human-2"))
    )
    v2 = registry.register(
        "tool-policy", ArtifactKind.POLICY, "v2", proposer="meta-agent"
    )

    for label, approvers in (
        ("self", (Approval("meta-agent"),)),
        ("one", (Approval("human-1"),)),
        ("two", (Approval("human-1"), Approval("human-2"))),
    ):
        try:
            result = manager.promote(v2, approvers=approvers)
            print(f"{label}: promoted v{result.version}")
        except ApprovalError as exc:
            print(f"{label}: rejected ({exc})")
    print(f"stable: v{registry.stable_version('tool-policy')}")


if __name__ == "__main__":
    main()