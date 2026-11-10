from dataclasses import replace

import pytest

from nano_harness.eval import (
    EvalBaseline,
    EvalCase,
    baseline_from_result,
    compare_to_baseline,
    eval_set_hash,
    format_suite,
    load_baseline,
    no_tool_errors,
    output_contains,
    retries_at_most,
    run_suite,
    save_baseline,
    stop_reason,
    turns_at_most,
    wilson_interval,
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
        name="echo",
        description="echo",
        parameters=ECHO_SCHEMA,
        handler=lambda text: text,
        read_only=True,
    )
    return ToolExecutor([spec], run_id=f"echo-{trial}")


def echo_case(name: str = "echo") -> EvalCase:
    return EvalCase(
        name=name,
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


def flaky_factory(trial: int) -> ToolExecutor:
    def echo(text: str) -> str:
        if trial % 2 == 0:
            raise RetryableToolError("HTTP 503")
        return text

    spec = ToolSpec(
        name="echo",
        description="echo",
        parameters=ECHO_SCHEMA,
        handler=echo,
        read_only=True,
    )
    return ToolExecutor([spec], run_id=f"flaky-{trial}")


def flaky_case() -> EvalCase:
    return EvalCase(
        name="echo",
        task="go",
        responses=(
            ModelResponse(tool_calls=(ToolCall("c1", "echo", {"text": "hi"}),)),
            ModelResponse(content="done"),
        ),
        executor_factory=flaky_factory,
        checks=(
            stop_reason(StopReason.COMPLETED),
            no_tool_errors(),
            output_contains("done"),
        ),
    )


def test_deterministic_suite_passes_power_k() -> None:
    result = run_suite("smoke", [echo_case()], repeats=3)

    aggregate = result.cases["echo"]
    assert aggregate.successes == 3
    assert aggregate.rate == 1.0
    assert aggregate.pass_power_k is True
    assert result.pass_power_k == 1.0
    assert result.repeats == 3
    assert len(result.results) == 3
    assert "pass^k=1.000" in format_suite(result)


def test_flaky_case_shows_pass_power_k() -> None:
    result = run_suite("flaky", [flaky_case()], repeats=4)

    aggregate = result.cases["echo"]
    assert aggregate.successes == 2
    assert aggregate.rate == 0.5
    assert aggregate.pass_power_k is False
    assert result.pass_power_k == 0.0
    failed = [item for item in result.results if not item.passed]
    assert failed
    assert "no_tool_errors" in failed[0].failures


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.23 < low < 0.25
    assert 0.75 < high < 0.77
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_check_factories_fail_with_names() -> None:
    bad = replace(
        echo_case(),
        checks=(turns_at_most(0), retries_at_most(0)),
    )
    result = run_suite("checks", [bad], repeats=1)
    case = result.results[0]

    assert case.passed is False
    assert case.failures == ("turns<=0",)


def test_baseline_regression_and_drift() -> None:
    current = run_suite("flaky", [flaky_case()], repeats=4)
    baseline = EvalBaseline(
        suite="flaky",
        eval_set_hash=current.eval_set_hash,
        repeats=4,
        pass_power_k=1.0,
        case_rates={"echo": 1.0},
    )
    report = compare_to_baseline(current, baseline)
    assert report.passed is False
    assert report.drift is False
    assert report.regressions == ("echo",)
    assert report.aggregate_delta == -1.0

    drifted = replace(baseline, eval_set_hash="different")
    drift_report = compare_to_baseline(current, drifted)
    assert drift_report.drift is True
    assert drift_report.passed is False

    same = baseline_from_result(current)
    assert compare_to_baseline(current, same).passed is True


def test_baseline_roundtrip(tmp_path) -> None:
    result = run_suite("smoke", [echo_case()], repeats=3)
    baseline = baseline_from_result(result)
    path = save_baseline(baseline, tmp_path / "baseline.json")
    loaded = load_baseline(path)

    assert loaded == baseline
    assert loaded.case_rates == {"echo": 1.0}


def test_eval_set_hash_changes_with_fixture_definition() -> None:
    first = echo_case()
    second = replace(first, task="different")
    assert eval_set_hash([first]) != eval_set_hash([second])


def test_run_suite_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError):
        run_suite("bad", [echo_case()], repeats=0)