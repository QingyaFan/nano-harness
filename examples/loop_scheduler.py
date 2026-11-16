"""A1 scheduler demo: stop gate, human gate, budget, and resume."""

from nano_harness.messages import Usage
from nano_harness.orchestration import (
    LoopBudget,
    LoopStateStore,
    PauseWhenContains,
    Scheduler,
    SchedulerStatus,
    StepResult,
    StopWhenContains,
)


class TriageWorker:
    def __call__(self, task: str, iteration: int, state) -> StepResult:
        if iteration == 1:
            return StepResult.ok_output("发现 3 个待处理问题", usage=Usage(120, 30))
        if iteration == 2:
            return StepResult.ok_output("需要人工确认是否合并", usage=Usage(140, 40))
        return StepResult.ok_output("已完成", usage=Usage(80, 20))


def main() -> None:
    scheduler = Scheduler(
        TriageWorker(),
        stop_gate=StopWhenContains("已完成"),
        human_gate=PauseWhenContains("人工确认"),
        budget=LoopBudget(max_iterations=5, max_tokens=1_000),
        store=LoopStateStore(),
        loop_id="daily-triage",
    )

    first = scheduler.run("每日分诊")
    print(
        f"first: status={first.status.value} iterations={first.iterations} "
        f"output={first.output!r}"
    )
    print(f"history: {len(first.state.history) if first.state else 0}")

    last = first
    if first.status is SchedulerStatus.SUSPENDED:
        last = scheduler.resume(approved=True)
        print(
            f"resumed: status={last.status.value} iterations={last.iterations} "
            f"output={last.output!r}"
        )
        print(f"tokens: {last.usage.total_tokens}")

    print("events:", [event.type for event in last.events])


if __name__ == "__main__":
    main()