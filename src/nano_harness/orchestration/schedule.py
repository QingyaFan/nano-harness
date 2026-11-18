"""Scheduled loops: occurrences, overlap, missed runs, and claims (A4)."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from zoneinfo import ZoneInfo

from ..lease import Lease, LeaseConflict, LeaseManager

MAX_HORIZON_DAYS = 800


class ScheduleError(Exception):
    pass


class ScheduleParseError(ScheduleError):
    pass


class DstPolicy(StrEnum):
    SKIP = "skip"
    ADVANCE = "advance"
    BOTH = "both"


class OverlapPolicy(StrEnum):
    SKIP = "skip"
    QUEUE = "queue"
    REPLACE = "replace"
    ALLOW = "allow"


class MissedPolicy(StrEnum):
    SKIP = "skip"
    RUN_ONCE = "run_once"
    CATCH_UP = "catch_up"


class OccurrenceStatus(StrEnum):
    DUE = "due"
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    OccurrenceStatus.SUCCEEDED,
    OccurrenceStatus.FAILED,
    OccurrenceStatus.SKIPPED,
    OccurrenceStatus.CANCELLED,
}


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


def _parse_field(text: str, low: int, high: int, *, dow: bool = False) -> tuple[set[int], bool]:
    if not text:
        raise ScheduleParseError("empty cron field")
    wildcard = text in {"*", "?"}
    if wildcard:
        values = set(range(low, high + 1))
        return ({value % 7 for value in values} if dow else values), True

    values: set[int] = set()
    for part in text.split(","):
        if not part:
            raise ScheduleParseError(f"empty item in cron field: {text!r}")
        step = 1
        base = part
        if "/" in part:
            base, step_text = part.split("/", 1)
            try:
                step = int(step_text)
            except ValueError as exc:
                raise ScheduleParseError(f"invalid step: {part!r}") from exc
            if step <= 0:
                raise ScheduleParseError(f"step must be positive: {part!r}")
        if base in {"*", "?"}:
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ScheduleParseError(f"invalid range: {part!r}") from exc
        else:
            try:
                start = end = int(base)
            except ValueError as exc:
                raise ScheduleParseError(f"invalid value: {part!r}") from exc
        if start < low or end > high or start > end:
            raise ScheduleParseError(f"value out of range: {part!r}")
        for value in range(start, end + 1, step):
            values.add(value % 7 if dow else value)
    return values, False


class CronSchedule:
    def __init__(
        self,
        expression: str,
        *,
        timezone: str = "UTC",
        dst: DstPolicy = DstPolicy.SKIP,
    ) -> None:
        parts = expression.split()
        if len(parts) != 5:
            raise ScheduleParseError("cron expression must have five fields")
        try:
            self._tz = ZoneInfo(timezone)
        except Exception as exc:
            raise ScheduleError(f"unknown timezone: {timezone!r}") from exc

        self.expression = expression
        self.timezone = timezone
        self.dst = dst
        self._minutes, self._minute_wild = _parse_field(parts[0], 0, 59)
        self._hours, self._hour_wild = _parse_field(parts[1], 0, 23)
        self._days, self._day_wild = _parse_field(parts[2], 1, 31)
        self._months, self._month_wild = _parse_field(parts[3], 1, 12)
        self._weekdays, self._weekday_wild = _parse_field(parts[4], 0, 7, dow=True)

    def next_after(self, instant: float) -> float | None:
        occurrences = self.occurrences(instant, instant + MAX_HORIZON_DAYS * 86400)
        return occurrences[0] if occurrences else None

    def occurrences(self, start: float, end: float) -> list[float]:
        if end <= start:
            return []
        start_local = datetime.fromtimestamp(start, self._tz)
        end_local = datetime.fromtimestamp(end, self._tz)
        first_day = start_local.date() - timedelta(days=1)
        last_day = end_local.date() + timedelta(days=1)
        if (last_day - first_day).days > MAX_HORIZON_DAYS + 2:
            raise ScheduleError("occurrence window is too large")
        found: set[float] = set()
        for day in self._dates(first_day, last_day):
            for hour in sorted(self._hours):
                for minute in sorted(self._minutes):
                    naive = datetime(day.year, day.month, day.day, hour, minute)
                    for instant in self._instants(naive):
                        if start < instant <= end:
                            found.add(instant)
        return sorted(found)

    def local_label(self, instant: float) -> str:
        return datetime.fromtimestamp(instant, self._tz).isoformat()

    def _dates(self, first: date, last: date) -> list[date]:
        days: list[date] = []
        current = first
        while current <= last:
            if self._date_matches(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def _date_matches(self, day: date) -> bool:
        if day.month not in self._months:
            return False
        dom_match = day.day in self._days
        dow_match = (day.isoweekday() % 7) in self._weekdays
        if self._day_wild and self._weekday_wild:
            return True
        if self._day_wild:
            return dow_match
        if self._weekday_wild:
            return dom_match
        return dom_match or dow_match

    def _instants(self, naive: datetime) -> list[float]:
        instants = self._resolve(naive)
        if instants:
            return instants if self.dst is DstPolicy.BOTH else instants[:1]
        if self.dst is not DstPolicy.ADVANCE:
            return []
        candidate = naive
        for _ in range(180):
            candidate += timedelta(minutes=1)
            advanced = self._resolve(candidate)
            if advanced:
                return advanced[:1]
        return []

    def _resolve(self, naive: datetime) -> list[float]:
        first = naive.replace(tzinfo=self._tz, fold=0)
        second = naive.replace(tzinfo=self._tz, fold=1)
        first_instant = first.timestamp()
        second_instant = second.timestamp()
        first_valid = self._roundtrip(first_instant) == naive
        second_valid = self._roundtrip(second_instant) == naive and second_instant != first_instant
        if first_valid and second_valid:
            return [first_instant, second_instant]
        if first_valid:
            return [first_instant]
        if second_valid:
            return [second_instant]
        return []

    def _roundtrip(self, instant: float) -> datetime:
        return datetime.fromtimestamp(instant, self._tz).replace(tzinfo=None)


@dataclass(frozen=True)
class Occurrence:
    job_id: str
    scheduled_at: float
    scheduled_local: str = ""

    @property
    def id(self) -> str:
        return f"{self.job_id}:{int(self.scheduled_at)}"


@dataclass(frozen=True)
class ScheduledJob:
    job_id: str
    schedule: CronSchedule
    task: str
    overlap: OverlapPolicy = OverlapPolicy.SKIP
    missed: MissedPolicy = MissedPolicy.RUN_ONCE
    catch_up_window: float = 3600.0
    queue_limit: int = 1
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ScheduleError("job_id must not be empty")
        if self.catch_up_window < 0:
            raise ScheduleError("catch_up_window must not be negative")
        if self.queue_limit < 0:
            raise ScheduleError("queue_limit must not be negative")

    def occurrence(self, instant: float) -> Occurrence:
        return Occurrence(
            job_id=self.job_id,
            scheduled_at=instant,
            scheduled_local=self.schedule.local_label(instant),
        )


@dataclass(frozen=True)
class RunOutcome:
    status: OccurrenceStatus = OccurrenceStatus.SUCCEEDED
    reason: str = ""


@dataclass(frozen=True)
class TickResult:
    occurrence: Occurrence
    status: OccurrenceStatus
    reason: str = ""


@dataclass
class OccurrenceRecord:
    occurrence: Occurrence
    status: OccurrenceStatus
    owner: str = ""
    epoch: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    reason: str = ""


class OccurrenceStore(Protocol):
    def cursor(self, job_id: str) -> float | None: ...
    def advance_cursor(self, job_id: str, instant: float) -> None: ...
    def claim(self, occ: Occurrence, owner: str, ttl: float) -> Lease | None: ...
    def mark_running(self, occ: Occurrence, lease: Lease) -> CancelToken: ...
    def mark_finished(
        self,
        occ: Occurrence,
        lease: Lease,
        status: OccurrenceStatus,
        reason: str = "",
    ) -> None: ...
    def mark_skipped(self, occ: Occurrence, reason: str) -> None: ...
    def active(self, job_id: str) -> Occurrence | None: ...
    def enqueue(self, job_id: str, occ: Occurrence, limit: int) -> bool: ...
    def dequeue(self, job_id: str) -> Occurrence | None: ...
    def cancel_active(self, job_id: str, reason: str) -> Occurrence | None: ...


class InMemoryOccurrenceStore:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._leases = LeaseManager(clock)
        self._records: dict[str, OccurrenceRecord] = {}
        self._cursors: dict[str, float] = {}
        self._queues: dict[str, deque[Occurrence]] = {}
        self._tokens: dict[str, CancelToken] = {}

    @property
    def records(self) -> dict[str, OccurrenceRecord]:
        return dict(self._records)

    def cursor(self, job_id: str) -> float | None:
        return self._cursors.get(job_id)

    def advance_cursor(self, job_id: str, instant: float) -> None:
        current = self._cursors.get(job_id)
        if current is None or instant > current:
            self._cursors[job_id] = instant

    def claim(self, occ: Occurrence, owner: str, ttl: float) -> Lease | None:
        existing = self._records.get(occ.id)
        if existing is not None and existing.status in TERMINAL_STATUSES:
            return None
        current = self._leases.current(occ.id)
        if current is not None and not current.is_expired(self._clock()):
            return None
        lease = self._leases.acquire(occ.id, owner, ttl)
        if lease is None:
            return None
        self._records[occ.id] = OccurrenceRecord(
            occurrence=occ,
            status=OccurrenceStatus.CLAIMED,
            owner=owner,
            epoch=lease.epoch,
        )
        return lease

    def mark_running(self, occ: Occurrence, lease: Lease) -> CancelToken:
        self._leases.assert_valid(lease)
        record = self._records.get(occ.id)
        if record is None:
            raise ScheduleError(f"occurrence {occ.id!r} was not claimed")
        record.status = OccurrenceStatus.RUNNING
        record.started_at = self._clock()
        token = CancelToken()
        self._tokens[occ.id] = token
        return token

    def mark_finished(
        self,
        occ: Occurrence,
        lease: Lease,
        status: OccurrenceStatus,
        reason: str = "",
    ) -> None:
        record = self._records.get(occ.id)
        if record is None:
            raise ScheduleError(f"occurrence {occ.id!r} was not claimed")
        if record.status in TERMINAL_STATUSES:
            return
        if record.owner != lease.owner or record.epoch != lease.epoch:
            raise LeaseConflict(f"stale lease {lease.epoch} for occurrence {occ.id!r}")
        record.status = status
        record.reason = reason
        record.finished_at = self._clock()
        current = self._leases.current(occ.id)
        if current is not None and current.owner == record.owner and current.epoch == record.epoch:
            self._leases.release(current)
        self._tokens.pop(occ.id, None)

    def mark_skipped(self, occ: Occurrence, reason: str) -> None:
        existing = self._records.get(occ.id)
        if existing is not None and existing.status in {
            OccurrenceStatus.CLAIMED,
            OccurrenceStatus.RUNNING,
            *TERMINAL_STATUSES,
        }:
            return
        self._records[occ.id] = OccurrenceRecord(
            occurrence=occ,
            status=OccurrenceStatus.SKIPPED,
            reason=reason,
            finished_at=self._clock(),
        )

    def enqueue(self, job_id: str, occ: Occurrence, limit: int) -> bool:
        if limit <= 0 or occ.id in self._records:
            return False
        queue = self._queues.setdefault(job_id, deque())
        if len(queue) >= limit:
            return False
        queue.append(occ)
        self._records[occ.id] = OccurrenceRecord(
            occurrence=occ,
            status=OccurrenceStatus.QUEUED,
        )
        return True

    def dequeue(self, job_id: str) -> Occurrence | None:
        queue = self._queues.get(job_id)
        if not queue:
            return None
        return queue.popleft()

    def cancel_active(self, job_id: str, reason: str) -> Occurrence | None:
        active = self.active(job_id)
        if active is None:
            return None
        record = self._records[active.id]
        token = self._tokens.get(active.id)
        if token is not None:
            token.cancel()
        record.status = OccurrenceStatus.CANCELLED
        record.reason = reason
        record.finished_at = self._clock()
        lease = self._leases.current(active.id)
        if lease is not None and lease.owner == record.owner and lease.epoch == record.epoch:
            self._leases.release(lease)
        self._tokens.pop(active.id, None)
        return active

    def active(self, job_id: str) -> Occurrence | None:
        now = self._clock()
        for record in self._records.values():
            if record.occurrence.job_id != job_id:
                continue
            if record.status not in {OccurrenceStatus.CLAIMED, OccurrenceStatus.RUNNING}:
                continue
            lease = self._leases.current(record.occurrence.id)
            if lease is not None and not lease.is_expired(now):
                return record.occurrence
        return None


class ScheduledRunner:
    def __init__(
        self,
        job: ScheduledJob,
        store: OccurrenceStore,
        run_occurrence: Callable[[Occurrence, CancelToken], RunOutcome],
        *,
        owner: str | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl: float = 30.0,
    ) -> None:
        if job.overlap is OverlapPolicy.QUEUE and job.queue_limit <= 0:
            raise ScheduleError("queue_limit must be positive for the queue policy")
        if lease_ttl <= 0:
            raise ScheduleError("lease_ttl must be positive")
        self._job = job
        self._store = store
        self._run_occurrence = run_occurrence
        self._owner = owner or f"runner-{uuid.uuid4().hex[:8]}"
        self._clock = clock
        self._lease_ttl = lease_ttl

    @property
    def owner(self) -> str:
        return self._owner

    def tick(self, now: float | None = None) -> list[TickResult]:
        moment = self._clock() if now is None else now
        cursor = self._store.cursor(self._job.job_id)
        start = moment - 60.0 if cursor is None else cursor
        due = [
            self._job.occurrence(instant)
            for instant in self._job.schedule.occurrences(start, moment)
        ]
        results: list[TickResult] = []

        if due:
            if self._job.expires_at is not None and moment >= self._job.expires_at:
                for occ in due:
                    results.append(self._skip(occ, "expired"))
                self._store.advance_cursor(self._job.job_id, due[-1].scheduled_at)
                return results

            selected, skipped = self._select(due, moment)
            for occ in skipped:
                results.append(self._skip(occ, "missed"))

            if selected and self._store.active(self._job.job_id) is not None:
                if self._job.overlap is OverlapPolicy.SKIP:
                    for occ in selected:
                        results.append(self._skip(occ, "overlap"))
                    selected = []
                elif self._job.overlap is OverlapPolicy.QUEUE:
                    for occ in selected:
                        if self._store.enqueue(
                            self._job.job_id,
                            occ,
                            self._job.queue_limit,
                        ):
                            results.append(
                                TickResult(occ, OccurrenceStatus.QUEUED, "queued")
                            )
                        else:
                            results.append(self._skip(occ, "queue_full"))
                    selected = []
                elif self._job.overlap is OverlapPolicy.REPLACE:
                    self._store.cancel_active(self._job.job_id, "replaced")

            for occ in selected:
                results.extend(self._execute(occ))
            self._store.advance_cursor(self._job.job_id, due[-1].scheduled_at)
        else:
            self._store.advance_cursor(self._job.job_id, moment)

        if self._store.active(self._job.job_id) is None:
            for _ in range(max(1, self._job.queue_limit)):
                queued = self._store.dequeue(self._job.job_id)
                if queued is None:
                    break
                results.extend(self._execute(queued))
                if self._store.active(self._job.job_id) is not None:
                    break
        return results

    def _execute(self, occ: Occurrence) -> list[TickResult]:
        lease = self._store.claim(occ, self._owner, self._lease_ttl)
        if lease is None:
            return [self._skip(occ, "claimed")]
        token = CancelToken()
        try:
            token = self._store.mark_running(occ, lease)
            outcome = self._run_occurrence(occ, token)
        except Exception as exc:  # noqa: BLE001 - a run failure must be recorded
            reason = f"{type(exc).__name__}: {exc}"
            try:
                self._store.mark_finished(occ, lease, OccurrenceStatus.FAILED, reason)
            except LeaseConflict:
                reason = "lease_lost"
            return [TickResult(occ, OccurrenceStatus.FAILED, reason)]
        if token.cancelled and outcome.status is OccurrenceStatus.SUCCEEDED:
            outcome = RunOutcome(OccurrenceStatus.CANCELLED, "cancelled")
        try:
            self._store.mark_finished(occ, lease, outcome.status, outcome.reason)
        except LeaseConflict:
            return [TickResult(occ, OccurrenceStatus.FAILED, "lease_lost")]
        return [TickResult(occ, outcome.status, outcome.reason)]

    def _select(
        self,
        due: list[Occurrence],
        now: float,
    ) -> tuple[list[Occurrence], list[Occurrence]]:
        within = [occ for occ in due if now - occ.scheduled_at <= self._job.catch_up_window]
        if len(due) == 1 and within:
            return list(within), []
        if self._job.missed is MissedPolicy.SKIP:
            return [], list(due)
        if self._job.missed is MissedPolicy.RUN_ONCE:
            selected = [within[-1]] if within else []
            skipped = [occ for occ in due if occ not in selected]
            return selected, skipped
        skipped = [occ for occ in due if occ not in within]
        return list(within), skipped

    def _skip(self, occ: Occurrence, reason: str) -> TickResult:
        self._store.mark_skipped(occ, reason)
        return TickResult(occ, OccurrenceStatus.SKIPPED, reason)