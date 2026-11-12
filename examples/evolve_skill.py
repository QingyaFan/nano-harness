"""Skill candidate lane: eval rejection, canary abort, and version pinning."""

from __future__ import annotations

from dataclasses import replace

from nano_harness.artifact import ArtifactKind, ArtifactRegistry
from nano_harness.eval import (
    EvalCase,
    baseline_from_result,
    no_tool_errors,
    output_contains,
    run_suite,
    stop_reason,
)
from nano_harness.evolution import (
    Approval,
    CanaryController,
    CanaryPolicy,
    CanarySample,
    EvalGate,
    ReleaseManager,
)
from nano_harness.loop import StopReason
from nano_harness.messages import ToolCall
from nano_harness.model import ModelResponse
from nano_harness.tools import RetryableToolError, ToolExecutor, ToolSpec

ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def echo_factory(trial: int) -> ToolExecutor:
    spec = ToolSpec(
        "echo",
        "echo",
        ECHO_SCHEMA,
        lambda text: text,
        read_only=True,
    )
    return ToolExecutor([spec], run_id=f"echo-{trial}")


def bad_factory(trial: int) -> ToolExecutor:
    def echo(text: str) -> str:
        raise RetryableToolError("HTTP 503")

    spec = ToolSpec("echo", "echo", ECHO_SCHEMA, echo, read_only=True)
    return ToolExecutor([spec], run_id=f"bad-{trial}")


def eval_case() -> EvalCase:
    return EvalCase(
        name="echo",
        task="go",
        responses=(
            ModelResponse(tool_calls=(ToolCall("c1", "echo", {"text": "hi"}),)),
            ModelResponse(content="done"),
        ),
        executor_factory=echo_factory,
        checks=(
            stop_reason(StopReason.COMPLETED),
            no_tool_errors(),
            output_contains("done"),
        ),
    )


def main() -> None:
    case = eval_case()
    baseline_suite = run_suite("skill", [case], repeats=3)
    baseline = baseline_from_result(baseline_suite)
    bad_suite = run_suite(
        "skill", [replace(case, executor_factory=bad_factory)], repeats=3
    )

    registry = ArtifactRegistry()
    manager = ReleaseManager(
        registry,
        gate=EvalGate(),
        canary=CanaryController(CanaryPolicy(min_samples=10)),
    )
    v1 = registry.register(
        "pdf-forms", ArtifactKind.SKILL, "v1", proposer="human"
    )
    manager.promote(v1, approvers=(Approval("human-1"),))

    v2 = registry.register(
        "pdf-forms", ArtifactKind.SKILL, "v2-bad", proposer="meta-agent"
    )
    bad_report = manager.evaluate(v2, bad_suite, baseline)
    v3 = registry.register(
        "pdf-forms", ArtifactKind.SKILL, "v3-good", proposer="meta-agent"
    )
    good_report = manager.evaluate(v3, baseline_suite, baseline)
    good_status = registry.get("pdf-forms", v3.version).status.value

    manager.pin("run-1", "pdf-forms", v3.version)
    decision = manager.canary(v3, CanarySample(20, 5))
    print(f"baseline: pass^k={baseline_suite.pass_power_k:.3f}")
    print(
        f"bad: pass^k={bad_report.pass_power_k:.3f} "
        f"status={registry.get('pdf-forms', v2.version).status.value} "
        f"failures={list(bad_report.failures)}"
    )
    print(
        f"good: pass^k={good_report.pass_power_k:.3f} "
        f"status={good_status}"
    )
    print(
        f"canary: abort={decision.abort} reason={decision.reason} "
        f"error_rate={decision.error_rate:.2f}"
    )
    print(
        f"versions: stable=v{registry.stable_version('pdf-forms')} "
        f"in_flight=v{registry.resolve('pdf-forms', 'run-1').version} "
        f"new_run=v{registry.resolve('pdf-forms', 'run-2').version}"
    )


if __name__ == "__main__":
    main()