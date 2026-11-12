"""Run leases and fencing tokens for resumable runs (E1)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


class LeaseError(Exception):
    pass


class LeaseConflict(LeaseError):
    pass


@dataclass(frozen=True)
class Lease:
    run_id: str
    owner: str
    epoch: int
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at <= (time.time() if now is None else now)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "owner": self.owner,
            "epoch": self.epoch,
            "expires_at": self.expires_at,
        }


class LeaseManager:
    """In-process reference lease store; production swaps in a CAS-capable store."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._leases: dict[str, Lease] = {}
        self._epochs: dict[str, int] = {}
        self._lock = threading.Lock()

    def current(self, run_id: str) -> Lease | None:
        with self._lock:
            return self._leases.get(run_id)

    def acquire(
        self, run_id: str, owner: str, ttl: float = 30.0
    ) -> Lease | None:
        if not owner or ttl <= 0:
            raise ValueError("owner must be non-empty and ttl must be positive")
        now = self._clock()
        with self._lock:
            current = self._leases.get(run_id)
            if current is not None and not current.is_expired(now):
                return None
            last_epoch = (
                current.epoch
                if current is not None
                else self._epochs.get(run_id, 0)
            )
            epoch = last_epoch + 1
            lease = Lease(run_id, owner, epoch, now + ttl)
            self._leases[run_id] = lease
            return lease

    def renew(self, lease: Lease, ttl: float = 30.0) -> Lease | None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        now = self._clock()
        with self._lock:
            current = self._leases.get(lease.run_id)
            if current != lease or current.is_expired(now):
                return None
            renewed = Lease(
                lease.run_id, lease.owner, lease.epoch, now + ttl
            )
            self._leases[lease.run_id] = renewed
            return renewed

    def release(self, lease: Lease) -> bool:
        with self._lock:
            if self._leases.get(lease.run_id) != lease:
                return False
            del self._leases[lease.run_id]
            self._epochs[lease.run_id] = lease.epoch
            return True

    def validate(self, lease: Lease, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        with self._lock:
            return self._leases.get(lease.run_id) == lease and not lease.is_expired(now)

    def assert_valid(self, lease: Lease, now: float | None = None) -> None:
        if not self.validate(lease, now=now):
            raise LeaseConflict(
                f"lease {lease.epoch} for run {lease.run_id!r} is stale or expired"
            )