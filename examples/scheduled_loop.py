"""A4 example: occurrences, DST, missed runs, and claims."""

from __future__ import annotations

from datetime import UTC, datetime

from nano_harness.loop import HarnessLoop, RunConfig
from nano_harness.model import ModelResponse, ScriptedModel
from nano_harness.orchestration.schedule import (
    CronSchedule,
    DstPolicy,
    InMemoryOccurrenceStore,
    MissedPolicy,
    OccurrenceStatus,
    OverlapPolicy,
    RunOutcome,
    ScheduledJob,
    ScheduledRunner,
)
from nano_harness.tools import ToolExecutor

NY = "America/New_York"


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


def run_occurrence(occ, cancel) -> RunOutcome:
    model = ScriptedModel([ModelResponse(content=f"ran {occ.id}")])
    tools = ToolExecutor([], run_id=occ.id)
    loop = HarnessLoop(
        model,
        tools,
        config=RunConfig(max_turns=1),
        run_id=occ.id,
    )
    result = loop.run("check", cancel=cancel)
    return RunOutcome(OccurrenceStatus.SUCCEEDED, result.stop_reason.value)


def make_job(
    expression: str,
    *,
    dst: DstPolicy = DstPolicy.SKIP,
    missed: MissedPolicy = MissedPolicy.RUN_ONCE,
    catch_up_window: float = 3600.0,
    overlap: OverlapPolicy = OverlapPolicy.SKIP,
) -> ScheduledJob:
    return ScheduledJob(
        job_id="job-42",
        schedule=CronSchedule(expression, timezone=NY, dst=dst),
        task="check",
        overlap=overlap,
        missed=missed,
        catch_up_window=catch_up_window,
    )


def run_tick(
    title: str,
    job: ScheduledJob,
    cursor: float,
    now: float,
) -> tuple[InMemoryOccurrenceStore, list]:
    store = InMemoryOccurrenceStore(clock=lambda: now)
    store.advance_cursor(job.job_id, cursor)
    runner = ScheduledRunner(
        job,
        store,
        run_occurrence,
        owner="runner-a",
        clock=lambda: now,
    )
    results = runner.tick()
    print(f"\n{title}")
    print("occurrences:", len(results))
    for result in results:
        occ = result.occurrence
        print(
            f"  {occ.scheduled_local}  {result.status.value:9s}"
            f"  {result.reason or '-'}  run_id={occ.id}"
        )
    return store, results


def main() -> None:
    run_tick(
        "spring forward (skip)",
        make_job("30 2 * * *", dst=DstPolicy.SKIP),
        utc(2026, 3, 8, 6, 0),
        utc(2026, 3, 8, 8, 0),
    )
    run_tick(
        "spring forward (advance)",
        make_job("30 2 * * *", dst=DstPolicy.ADVANCE),
        utc(2026, 3, 8, 6, 0),
        utc(2026, 3, 8, 8, 0),
    )
    run_tick(
        "fall back (run once)",
        make_job("30 1 * * *", catch_up_window=7200.0),
        utc(2026, 11, 1, 4, 0),
        utc(2026, 11, 1, 6, 45),
    )
    run_tick(
        "missed run-once",
        make_job("*/30 * * * *"),
        utc(2026, 6, 1, 0, 0),
        utc(2026, 6, 1, 1, 45),
    )

    job = make_job("*/30 * * * *", overlap=OverlapPolicy.ALLOW)
    now = utc(2026, 6, 1, 0, 30)
    store = InMemoryOccurrenceStore(clock=lambda: now)
    store.advance_cursor(job.job_id, utc(2026, 6, 1, 0, 0))
    occ = job.occurrence(now)
    lease = store.claim(occ, "runner-b", 300)
    if lease is not None:
        store.mark_running(occ, lease)
    runner = ScheduledRunner(
        job,
        store,
        run_occurrence,
        owner="runner-a",
        clock=lambda: now,
    )
    results = runner.tick()
    print("\ntwo runners (claim conflict)")
    for result in results:
        print(
            f"  {result.occurrence.scheduled_local}  {result.status.value:9s}"
            f"  {result.reason or '-'}"
        )
    record = store.records[occ.id]
    print("ledger owner:", record.owner, "status:", record.status.value)


if __name__ == "__main__":
    main()