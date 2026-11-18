from datetime import UTC, datetime

import pytest

from nano_harness.lease import LeaseConflict
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
    ScheduleParseError,
)

NY = "America/New_York"


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


def make_job(expression: str = "*/30 * * * *", **overrides) -> ScheduledJob:
    values = {
        "job_id": "job-1",
        "schedule": CronSchedule(expression, timezone="UTC"),
        "task": "check",
        "catch_up_window": 3600.0,
    }
    values.update(overrides)
    return ScheduledJob(**values)


def test_cron_parses_steps_ranges_and_weekdays() -> None:
    schedule = CronSchedule("*/30 9-17 * * 1-5", timezone="UTC")
    start = utc(2026, 6, 1, 8, 0)

    assert schedule.next_after(start) == utc(2026, 6, 1, 9, 0)

    with pytest.raises(ScheduleParseError):
        CronSchedule("* * * *", timezone="UTC")
    with pytest.raises(ScheduleParseError):
        CronSchedule("60 * * * *", timezone="UTC")
    with pytest.raises(ScheduleParseError):
        CronSchedule("*/0 * * * *", timezone="UTC")


def test_next_after_respects_weekday_and_minute() -> None:
    schedule = CronSchedule("0 9 * * 1", timezone="UTC")

    assert schedule.next_after(utc(2026, 6, 1, 9, 0)) == utc(2026, 6, 8, 9, 0)


def test_spring_forward_skips_missing_local_time() -> None:
    schedule = CronSchedule("30 2 * * *", timezone=NY, dst=DstPolicy.SKIP)
    occurrences = schedule.occurrences(utc(2026, 3, 8, 6, 0), utc(2026, 3, 8, 8, 0))

    assert occurrences == []


def test_spring_forward_can_advance_to_next_valid_minute() -> None:
    schedule = CronSchedule("30 2 * * *", timezone=NY, dst=DstPolicy.ADVANCE)
    occurrences = schedule.occurrences(utc(2026, 3, 8, 6, 0), utc(2026, 3, 8, 8, 0))

    assert occurrences == [utc(2026, 3, 8, 7, 0)]
    assert schedule.local_label(occurrences[0]) == "2026-03-08T03:00:00-04:00"


def test_fall_back_runs_once_by_default() -> None:
    schedule = CronSchedule("30 1 * * *", timezone=NY)
    occurrences = schedule.occurrences(utc(2026, 11, 1, 4, 0), utc(2026, 11, 1, 8, 0))

    assert len(occurrences) == 1
    assert schedule.local_label(occurrences[0]) == "2026-11-01T01:30:00-04:00"


def test_fall_back_can_emit_both_instants() -> None:
    schedule = CronSchedule("30 1 * * *", timezone=NY, dst=DstPolicy.BOTH)
    occurrences = schedule.occurrences(utc(2026, 11, 1, 4, 0), utc(2026, 11, 1, 8, 0))

    assert len(occurrences) == 2
    labels = [schedule.local_label(instant) for instant in occurrences]
    assert labels == [
        "2026-11-01T01:30:00-04:00",
        "2026-11-01T01:30:00-05:00",
    ]


def test_missed_run_once_runs_only_latest_occurrence() -> None:
    now = [utc(2026, 6, 1, 1, 45)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(
        make_job(missed=MissedPolicy.RUN_ONCE),
        store,
        run,
        clock=lambda: now[0],
        owner="runner-a",
    )
    results = runner.tick()

    assert [occ.scheduled_at for occ in seen] == [utc(2026, 6, 1, 1, 30)]
    assert [result.status for result in results] == [
        OccurrenceStatus.SKIPPED,
        OccurrenceStatus.SKIPPED,
        OccurrenceStatus.SUCCEEDED,
    ]
    assert [result.reason for result in results[:2]] == ["missed", "missed"]


def test_catch_up_runs_all_occurrences_inside_window() -> None:
    now = [utc(2026, 6, 1, 1, 45)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(
        make_job(
            missed=MissedPolicy.CATCH_UP,
            catch_up_window=3000.0,
        ),
        store,
        run,
        clock=lambda: now[0],
        owner="runner-a",
    )
    runner.tick()

    assert [occ.scheduled_at for occ in seen] == [
        utc(2026, 6, 1, 1, 0),
        utc(2026, 6, 1, 1, 30),
    ]


def test_duplicate_tick_does_not_start_twice() -> None:
    now = [utc(2026, 6, 1, 0, 30)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(
        make_job(),
        store,
        run,
        clock=lambda: now[0],
        owner="runner-a",
    )
    runner.tick()
    runner.tick()

    assert len(seen) == 1


def test_overlap_policy_skips_when_another_run_is_active() -> None:
    now = [utc(2026, 6, 1, 1, 5)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    job = make_job()

    active = job.occurrence(utc(2026, 6, 1, 0, 30))
    lease = store.claim(active, "other", 300)
    assert lease is not None
    store.mark_running(active, lease)
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(job, store, run, clock=lambda: now[0], owner="runner-a")
    results = runner.tick()

    assert seen == []
    assert any(result.reason == "overlap" for result in results)
    assert store.records[active.id].status is OccurrenceStatus.RUNNING


def test_claim_conflict_does_not_run_or_overwrite_active_record() -> None:
    now = [utc(2026, 6, 1, 0, 30)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    job = make_job(overlap=OverlapPolicy.ALLOW)
    occ = job.occurrence(utc(2026, 6, 1, 0, 30))
    lease = store.claim(occ, "other", 300)
    assert lease is not None
    store.mark_running(occ, lease)
    seen = []

    def run(item, cancel):
        seen.append(item)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(job, store, run, clock=lambda: now[0], owner="runner-b")
    results = runner.tick()

    assert seen == []
    assert results[0].reason == "claimed"
    assert store.records[occ.id].status is OccurrenceStatus.RUNNING
    assert store.records[occ.id].owner == "other"


def test_missed_skip_runs_nothing() -> None:
    now = [utc(2026, 6, 1, 1, 45)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(
        make_job(missed=MissedPolicy.SKIP),
        store,
        run,
        clock=lambda: now[0],
        owner="runner-a",
    )
    results = runner.tick()

    assert seen == []
    assert all(result.status is OccurrenceStatus.SKIPPED for result in results)
    assert all(result.reason == "missed" for result in results)


def test_stale_lease_cannot_finish_occurrence() -> None:
    now = [utc(2026, 6, 1, 0, 30)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    occ = make_job().occurrence(now[0])
    lease_a = store.claim(occ, "runner-a", 10)
    assert lease_a is not None
    store.mark_running(occ, lease_a)

    now[0] += 11
    lease_b = store.claim(occ, "runner-b", 10)
    assert lease_b is not None
    store.mark_running(occ, lease_b)

    with pytest.raises(LeaseConflict):
        store.mark_finished(occ, lease_a, OccurrenceStatus.SUCCEEDED)

    assert store.records[occ.id].owner == "runner-b"
    assert store.records[occ.id].status is OccurrenceStatus.RUNNING


def test_failed_run_is_recorded_as_failed() -> None:
    now = [utc(2026, 6, 1, 0, 30)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))

    def run(occ, cancel):
        raise RuntimeError("boom")

    runner = ScheduledRunner(make_job(), store, run, clock=lambda: now[0], owner="runner-a")
    results = runner.tick()

    assert results[0].status is OccurrenceStatus.FAILED
    assert "RuntimeError" in results[0].reason
    assert store.records[results[0].occurrence.id].status is OccurrenceStatus.FAILED


def test_occurrence_id_is_stable_and_used_as_run_id() -> None:
    job = make_job()
    first = job.occurrence(utc(2026, 6, 1, 0, 30))
    second = job.occurrence(utc(2026, 6, 1, 1, 0))

    assert first.id == f"job-1:{int(utc(2026, 6, 1, 0, 30))}"
    assert first.id != second.id


def test_queue_policy_persists_until_active_finishes() -> None:
    now = [utc(2026, 6, 1, 1, 5)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    job = make_job(overlap=OverlapPolicy.QUEUE, queue_limit=1)
    active = job.occurrence(utc(2026, 6, 1, 0, 30))
    lease = store.claim(active, "other", 300)
    assert lease is not None
    store.mark_running(active, lease)
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(job, store, run, clock=lambda: now[0], owner="runner-a")
    results = runner.tick()

    queued = job.occurrence(utc(2026, 6, 1, 1, 0))
    assert seen == []
    assert results[-1].status is OccurrenceStatus.QUEUED
    assert store.records[queued.id].status is OccurrenceStatus.QUEUED

    store.mark_finished(active, lease, OccurrenceStatus.SUCCEEDED)
    now[0] = utc(2026, 6, 1, 1, 6)
    runner.tick()

    assert [occ.scheduled_at for occ in seen] == [utc(2026, 6, 1, 1, 0)]


def test_replace_policy_cancels_active_and_starts_new() -> None:
    now = [utc(2026, 6, 1, 1, 5)]
    store = InMemoryOccurrenceStore(clock=lambda: now[0])
    store.advance_cursor("job-1", utc(2026, 6, 1, 0, 0))
    job = make_job(overlap=OverlapPolicy.REPLACE)
    active = job.occurrence(utc(2026, 6, 1, 0, 30))
    lease = store.claim(active, "other", 300)
    assert lease is not None
    token = store.mark_running(active, lease)
    seen = []

    def run(occ, cancel):
        seen.append(occ)
        return RunOutcome(OccurrenceStatus.SUCCEEDED)

    runner = ScheduledRunner(job, store, run, clock=lambda: now[0], owner="runner-a")
    results = runner.tick()

    assert token.cancelled
    assert store.records[active.id].status is OccurrenceStatus.CANCELLED
    assert [occ.scheduled_at for occ in seen] == [utc(2026, 6, 1, 1, 0)]
    assert any(result.status is OccurrenceStatus.SUCCEEDED for result in results)