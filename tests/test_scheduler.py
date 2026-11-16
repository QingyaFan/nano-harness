import json
from pathlib import Path

import pytest

from nano_harness.loop import RunResult, StopReason
from nano_harness.messages import Usage
from nano_harness.orchestration import (
    AgentWorker,
    IntervalTrigger,
    LoopBudget,
    LoopError,
    LoopStateStore,
    NeverStop,
    PauseWhenContains,
    Scheduler,
    SchedulerStatus,
    StepResult,
    StopWhenContains,
)


class SequenceWorker:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls = 0

    def __call__(self, task, iteration, state) -> StepResult:
        self.calls += 1
        output = self._outputs[min(iteration - 1, len(self._outputs) - 1)]
        return StepResult.ok_output(output, usage=Usage(10, 5))


def make_scheduler(worker, **kwargs) -> Scheduler:
    kwargs.setdefault("loop_id", "loop-test")
    kwargs.setdefault("budget", LoopBudget(max_iterations=5))
    return Scheduler(worker, stop_gate=kwargs.pop("stop_gate", NeverStop()), **kwargs)


def test_scheduler_stops_on_goal() -> None:
    worker = SequenceWorker(["working", "done"])
    scheduler = make_scheduler(
        worker, stop_gate=StopWhenContains("done"), loop_id="loop-goal"
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.COMPLETED
    assert result.iterations == 2
    assert result.output == "done"
    assert result.usage.total_tokens == 30
    assert len(result.state.history) == 2
    event_types = [event.type for event in result.events]
    assert "loop_started" in event_types
    assert "loop_iteration" in event_types
    assert "loop_finished" in event_types


def test_scheduler_max_iterations() -> None:
    worker = SequenceWorker(["working"])
    scheduler = make_scheduler(
        worker,
        stop_gate=NeverStop(),
        budget=LoopBudget(max_iterations=2),
        loop_id="loop-max",
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.MAX_ITERATIONS
    assert result.iterations == 2
    assert result.reason == "iteration_budget"


def test_scheduler_token_budget() -> None:
    worker = SequenceWorker(["working"])
    scheduler = make_scheduler(
        worker,
        stop_gate=NeverStop(),
        budget=LoopBudget(max_iterations=10, max_tokens=15),
        loop_id="loop-tokens",
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.BUDGET_EXHAUSTED
    assert result.iterations == 2
    assert result.reason == "token_budget"
    assert result.usage.total_tokens == 30


def test_scheduler_consecutive_failures() -> None:
    class FailingWorker:
        def __call__(self, task, iteration, state) -> StepResult:
            return StepResult.failure("boom", usage=Usage(1, 1))

    scheduler = make_scheduler(
        FailingWorker(),
        stop_gate=NeverStop(),
        budget=LoopBudget(max_iterations=5, max_consecutive_failures=2),
        loop_id="loop-fail",
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.FAILED
    assert result.iterations == 2
    assert result.reason == "consecutive_failures"
    assert result.state.consecutive_failures == 2


def test_scheduler_pauses_and_resumes() -> None:
    worker = SequenceWorker(["needs human review", "done"])
    scheduler = make_scheduler(
        worker,
        stop_gate=StopWhenContains("done"),
        human_gate=PauseWhenContains("review"),
        loop_id="loop-human",
    )

    paused = scheduler.run("triage")
    assert paused.status is SchedulerStatus.SUSPENDED
    assert paused.iterations == 1
    assert paused.state.suspension is not None

    resumed = scheduler.resume(approved=True)
    assert resumed.status is SchedulerStatus.COMPLETED
    assert resumed.iterations == 2
    assert len(resumed.state.history) == 2
    assert any(event.type == "loop_resumed" for event in resumed.events)


def test_scheduler_rejects_human_gate() -> None:
    worker = SequenceWorker(["needs human review", "done"])
    scheduler = make_scheduler(
        worker,
        stop_gate=StopWhenContains("done"),
        human_gate=PauseWhenContains("review"),
        loop_id="loop-reject",
    )

    scheduler.run("triage")
    rejected = scheduler.resume(approved=False)

    assert rejected.status is SchedulerStatus.HUMAN_REJECTED
    assert rejected.iterations == 1
    assert rejected.reason == "human_rejected"


def test_scheduler_state_survives_restart(tmp_path) -> None:
    store = LoopStateStore(tmp_path)
    worker = SequenceWorker(["needs human review", "done"])

    first = Scheduler(
        worker,
        stop_gate=StopWhenContains("done"),
        human_gate=PauseWhenContains("review"),
        store=store,
        loop_id="loop-file",
    )
    paused = first.run("triage")
    assert paused.status is SchedulerStatus.SUSPENDED
    assert (tmp_path / "loop-file.json").exists()

    second = Scheduler(
        worker,
        stop_gate=StopWhenContains("done"),
        human_gate=PauseWhenContains("review"),
        store=store,
        loop_id="loop-file",
    )
    resumed = second.resume(approved=True)

    assert resumed.status is SchedulerStatus.COMPLETED
    assert resumed.iterations == 2
    assert resumed.state.task == "triage"
    payload = json.loads((tmp_path / "loop-file.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert len(payload["history"]) == 2


def test_scheduler_interval_trigger_skips_early_runs() -> None:
    worker = SequenceWorker(["done"])
    scheduler = make_scheduler(
        worker,
        stop_gate=StopWhenContains("done"),
        trigger=IntervalTrigger(60),
        loop_id="loop-cron",
    )

    first = scheduler.run("triage", now=100)
    early = scheduler.run("triage", now=120)
    later = scheduler.run("triage", now=200)

    assert first.status is SchedulerStatus.COMPLETED
    assert early.status is SchedulerStatus.SKIPPED
    assert later.status is SchedulerStatus.COMPLETED


def test_agent_worker_adapts_a_harness_run() -> None:
    run = RunResult(
        run_id="run-1",
        stop_reason=StopReason.COMPLETED,
        turns=1,
        tool_calls=0,
        tool_attempts=0,
        usage=Usage(2, 3),
        events=(),
        messages=(),
        output="ok",
    )
    scheduler = make_scheduler(
        AgentWorker(lambda task: run),
        stop_gate=StopWhenContains("ok"),
        loop_id="loop-agent",
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.COMPLETED
    assert result.output == "ok"
    assert result.usage.total_tokens == 5


def test_scheduler_cancel_before_first_step() -> None:
    class Cancelled:
        cancelled = True

    scheduler = make_scheduler(
        SequenceWorker(["done"]),
        stop_gate=StopWhenContains("done"),
        loop_id="loop-cancel",
    )

    result = scheduler.run("triage", cancel=Cancelled())

    assert result.status is SchedulerStatus.CANCELLED
    assert result.iterations == 0


def test_scheduler_stop_gate_error_fails_closed() -> None:
    class BrokenGate:
        def check(self, snapshot):
            raise RuntimeError("gate down")

    scheduler = make_scheduler(
        SequenceWorker(["working"]),
        stop_gate=BrokenGate(),
        loop_id="loop-gate-error",
    )

    result = scheduler.run("triage")

    assert result.status is SchedulerStatus.FAILED
    assert result.reason == "stop_gate_error"
    assert result.iterations == 1


def test_scheduler_resume_requires_suspension() -> None:
    scheduler = make_scheduler(
        SequenceWorker(["done"]),
        stop_gate=StopWhenContains("done"),
        loop_id="loop-not-paused",
    )
    scheduler.run("triage")

    with pytest.raises(LoopError):
        scheduler.resume(approved=True)


def test_core_does_not_import_orchestration() -> None:
    core = Path(__file__).resolve().parents[1] / "src" / "nano_harness"
    for path in core.rglob("*.py"):
        if "orchestration" in path.parts:
            continue
        assert "orchestration" not in path.read_text(encoding="utf-8").lower(), path