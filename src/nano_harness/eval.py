"""Fixture evaluation, pass^k, confidence intervals, and regression gates (I7)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import TOOL_RESULT, TOOL_RETRY
from .loop import HarnessLoop, RunConfig, RunResult, StopReason
from .model import ModelResponse, ScriptedModel
from .observability import RunMetrics, collect_metrics


@dataclass(frozen=True)
class EvalObservation:
    case: EvalCase
    trial: int
    result: RunResult
    metrics: RunMetrics


@dataclass(frozen=True)
class Check:
    name: str
    predicate: Callable[[EvalObservation], bool]

    def __call__(self, observation: EvalObservation) -> bool:
        return bool(self.predicate(observation))


def stop_reason(expected: StopReason) -> Check:
    return Check(
        f"stop_reason={expected.value}",
        lambda observation: observation.result.stop_reason is expected,
    )


def turns_at_most(limit: int) -> Check:
    return Check(
        f"turns<={limit}",
        lambda observation: observation.result.turns <= limit,
    )


def tool_calls_at_most(limit: int) -> Check:
    return Check(
        f"tool_calls<={limit}",
        lambda observation: observation.result.tool_calls <= limit,
    )


def retries_at_most(limit: int) -> Check:
    def predicate(observation: EvalObservation) -> bool:
        count = sum(
            1 for event in observation.result.events if event.type == TOOL_RETRY
        )
        return count <= limit

    return Check(f"retries<={limit}", predicate)


def no_tool_errors() -> Check:
    def predicate(observation: EvalObservation) -> bool:
        return all(
            event.data.get("ok", True)
            for event in observation.result.events
            if event.type == TOOL_RESULT
        )

    return Check("no_tool_errors", predicate)


def output_contains(text: str) -> Check:
    return Check(
        f"output_contains={text!r}",
        lambda observation: text in (observation.result.output or ""),
    )


def usage_under(tokens: int) -> Check:
    return Check(
        f"usage<={tokens}",
        lambda observation: observation.metrics.total_tokens <= tokens,
    )


@dataclass(frozen=True)
class EvalCase:
    name: str
    task: str
    responses: tuple[ModelResponse, ...]
    executor_factory: Callable[[int], Any]
    checks: tuple[Check, ...]
    config: RunConfig = field(default_factory=RunConfig)
    context_factory: Callable[[], Any] | None = None
    fixture_version: str = "1"


@dataclass(frozen=True)
class CaseResult:
    case: str
    trial: int
    passed: bool
    failures: tuple[str, ...]
    result: RunResult | None
    metrics: RunMetrics | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "trial": self.trial,
            "passed": self.passed,
            "failures": list(self.failures),
            "error": self.error,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


def run_case(case: EvalCase, trial: int = 0) -> CaseResult:
    model = ScriptedModel(list(case.responses))
    executor = case.executor_factory(trial)
    context = case.context_factory() if case.context_factory else None
    loop = HarnessLoop(
        model,
        executor,
        config=case.config,
        context=context,
        run_id=f"{case.name}-{trial}",
    )
    try:
        result = loop.run(case.task)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case=case.name,
            trial=trial,
            passed=False,
            failures=(f"error: {type(exc).__name__}: {exc}",),
            result=None,
            metrics=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    metrics = collect_metrics(result.events, usage=result.usage)
    observation = EvalObservation(case, trial, result, metrics)
    failures = tuple(
        check.name for check in case.checks if not check(observation)
    )
    return CaseResult(
        case=case.name,
        trial=trial,
        passed=not failures,
        failures=failures,
        result=result,
        metrics=metrics,
    )


@dataclass(frozen=True)
class CaseAggregate:
    trials: int
    successes: int
    rate: float
    pass_power_k: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "successes": self.successes,
            "rate": self.rate,
            "pass_power_k": self.pass_power_k,
        }


@dataclass(frozen=True)
class SuiteResult:
    suite: str
    repeats: int
    cases: dict[str, CaseAggregate]
    pass_power_k: float
    confidence_interval: tuple[float, float]
    eval_set_hash: str
    results: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "repeats": self.repeats,
            "pass_power_k": self.pass_power_k,
            "confidence_interval": list(self.confidence_interval),
            "eval_set_hash": self.eval_set_hash,
            "cases": {
                name: aggregate.to_dict()
                for name, aggregate in self.cases.items()
            },
        }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _response_payload(response: ModelResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "finish_reason": response.finish_reason,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
    }


def eval_set_hash(cases: Sequence[EvalCase]) -> str:
    payload = []
    for case in sorted(cases, key=lambda item: item.name):
        payload.append(
            {
                "name": case.name,
                "task": case.task,
                "fixture_version": case.fixture_version,
                "responses": [
                    _response_payload(response) for response in case.responses
                ],
                "checks": sorted(check.name for check in case.checks),
                "config": {
                    "max_turns": case.config.max_turns,
                    "no_progress_limit": case.config.no_progress_limit,
                    "max_stop_blocks": case.config.max_stop_blocks,
                },
            }
        )
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def run_suite(
    suite: str,
    cases: Iterable[EvalCase],
    *,
    repeats: int = 3,
) -> SuiteResult:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    ordered = list(cases)
    results: list[CaseResult] = []
    aggregates: dict[str, CaseAggregate] = {}
    for case in ordered:
        trials = [run_case(case, trial) for trial in range(repeats)]
        successes = sum(1 for trial in trials if trial.passed)
        aggregates[case.name] = CaseAggregate(
            trials=repeats,
            successes=successes,
            rate=successes / repeats,
            pass_power_k=successes == repeats,
        )
        results.extend(trials)

    passed_cases = sum(
        1 for aggregate in aggregates.values() if aggregate.pass_power_k
    )
    total = len(ordered)
    rate = passed_cases / total if total else 0.0
    return SuiteResult(
        suite=suite,
        repeats=repeats,
        cases=aggregates,
        pass_power_k=rate,
        confidence_interval=wilson_interval(passed_cases, total),
        eval_set_hash=eval_set_hash(ordered),
        results=tuple(results),
    )


def format_suite(result: SuiteResult) -> str:
    lines = [f"suite: {result.suite} (repeats={result.repeats})"]
    for name, aggregate in result.cases.items():
        lines.append(
            f"  {name}: rate={aggregate.rate:.2f} "
            f"({aggregate.successes}/{aggregate.trials}) "
            f"pass^k={aggregate.pass_power_k}"
        )
    low, high = result.confidence_interval
    lines.append(
        f"  pass^k={result.pass_power_k:.3f} CI95=[{low:.3f}, {high:.3f}]"
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class EvalBaseline:
    suite: str
    eval_set_hash: str
    repeats: int
    pass_power_k: float
    case_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "eval_set_hash": self.eval_set_hash,
            "repeats": self.repeats,
            "pass_power_k": self.pass_power_k,
            "case_rates": dict(self.case_rates),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalBaseline:
        return cls(
            suite=payload["suite"],
            eval_set_hash=payload["eval_set_hash"],
            repeats=int(payload["repeats"]),
            pass_power_k=float(payload["pass_power_k"]),
            case_rates={
                name: float(rate)
                for name, rate in payload.get("case_rates", {}).items()
            },
        )


def baseline_from_result(result: SuiteResult) -> EvalBaseline:
    return EvalBaseline(
        suite=result.suite,
        eval_set_hash=result.eval_set_hash,
        repeats=result.repeats,
        pass_power_k=result.pass_power_k,
        case_rates={
            name: aggregate.rate
            for name, aggregate in result.cases.items()
        },
    )


def save_baseline(baseline: EvalBaseline, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(
        json.dumps(baseline.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return target


def load_baseline(path: str | Path) -> EvalBaseline:
    return EvalBaseline.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


@dataclass(frozen=True)
class RegressionReport:
    passed: bool
    drift: bool
    aggregate_delta: float
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "drift": self.drift,
            "aggregate_delta": self.aggregate_delta,
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
        }


def compare_to_baseline(
    current: SuiteResult,
    baseline: EvalBaseline,
    *,
    max_drop: float = 0.0,
    allow_drift: bool = False,
) -> RegressionReport:
    drift = (
        current.eval_set_hash != baseline.eval_set_hash
        or current.repeats != baseline.repeats
    )
    regressions: list[str] = []
    improvements: list[str] = []
    for name, aggregate in current.cases.items():
        previous = baseline.case_rates.get(name)
        if previous is None:
            drift = True
            continue
        if aggregate.rate < previous - max_drop:
            regressions.append(name)
        elif aggregate.rate > previous + max_drop:
            improvements.append(name)
    for name in baseline.case_rates:
        if name not in current.cases:
            drift = True
    delta = current.pass_power_k - baseline.pass_power_k
    passed = (
        (allow_drift or not drift)
        and not regressions
        and delta >= -max_drop
    )
    return RegressionReport(
        passed=passed,
        drift=drift,
        aggregate_delta=delta,
        regressions=tuple(sorted(regressions)),
        improvements=tuple(sorted(improvements)),
    )