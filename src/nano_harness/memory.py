"""Cross-run memory: versioned facts, scoped tools, and recall budgets (I8)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .context import estimate_text
from .events import (
    MEMORY_CONFLICT,
    MEMORY_DUPLICATE,
    MEMORY_EXPIRED,
    MEMORY_FORGOTTEN,
    MEMORY_RECALLED,
    MEMORY_WRITTEN,
    EventLog,
)
from .risk import RiskLevel
from .tools.spec import ToolSpec

KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class WriteStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class MemorySource:
    run_id: str
    tool: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tool": self.tool,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemorySource:
        return cls(
            run_id=payload["run_id"],
            tool=payload.get("tool", ""),
            timestamp=float(payload.get("timestamp", 0.0)),
        )


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: str
    key: str
    value: str
    version: int
    source: MemorySource
    created_at: float
    updated_at: float
    expires_at: float | None = None
    supersedes: str | None = None
    confidence: float = 1.0
    tags: tuple[str, ...] = ()
    forgotten_at: float | None = None

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at <= (time.time() if now is None else now)

    def is_active(self, now: float | None = None) -> bool:
        return self.forgotten_at is None and not self.is_expired(now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "key": self.key,
            "value": self.value,
            "version": self.version,
            "source": self.source.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "supersedes": self.supersedes,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "forgotten_at": self.forgotten_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=payload["id"],
            scope=payload["scope"],
            key=payload["key"],
            value=payload["value"],
            version=int(payload["version"]),
            source=MemorySource.from_dict(payload["source"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            expires_at=payload.get("expires_at"),
            supersedes=payload.get("supersedes"),
            confidence=float(payload.get("confidence", 1.0)),
            tags=tuple(payload.get("tags", ())),
            forgotten_at=payload.get("forgotten_at"),
        )


@dataclass(frozen=True)
class MemoryWrite:
    scope: str
    key: str
    value: str
    source: MemorySource
    expected_version: int | None = None
    ttl_seconds: float | None = None
    confidence: float = 1.0
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WriteResult:
    status: WriteStatus
    record: MemoryRecord | None
    current: MemoryRecord | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "record": self.record.to_dict() if self.record else None,
            "current": self.current.to_dict() if self.current else None,
            "message": self.message,
        }


@dataclass(frozen=True)
class RecallResult:
    records: tuple[MemoryRecord, ...]
    text: str
    used_tokens: int
    dropped: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "text": self.text,
            "used_tokens": self.used_tokens,
            "dropped": list(self.dropped),
        }


@dataclass(frozen=True)
class ForgetResult:
    status: str
    record: MemoryRecord | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "record": self.record.to_dict() if self.record else None,
            "message": self.message,
        }


class JsonMemoryStore:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: dict[str, MemoryRecord] = {}
        if self._path is not None and self._path.exists():
            self._load()

    def _load(self) -> None:
        if self._path is None:
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        for item in payload.get("records", ()):
            record = MemoryRecord.from_dict(item)
            self._records[record.id] = record

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema": 1,
                        "records": [
                            record.to_dict()
                            for record in sorted(
                                self._records.values(), key=lambda item: item.id
                            )
                        ],
                    },
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self._path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def add(self, record: MemoryRecord) -> None:
        self._records[record.id] = record

    def replace(self, record: MemoryRecord) -> None:
        self._records[record.id] = record

    def history(self, scope: str, key: str) -> tuple[MemoryRecord, ...]:
        return tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.scope == scope and record.key == key
                ),
                key=lambda item: (item.version, item.updated_at),
            )
        )

    def latest(self, scope: str, key: str) -> MemoryRecord | None:
        history = self.history(scope, key)
        return history[-1] if history else None

    def latest_records(self, scope: str | None = None) -> tuple[MemoryRecord, ...]:
        latest: dict[tuple[str, str], MemoryRecord] = {}
        for record in self._records.values():
            if scope is not None and record.scope != scope:
                continue
            slot = (record.scope, record.key)
            current = latest.get(slot)
            if current is None or record.version > current.version:
                latest[slot] = record
        return tuple(
            sorted(latest.values(), key=lambda item: (item.scope, item.key))
        )

    def active(self, scope: str, now: float) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in self.latest_records(scope)
            if record.is_active(now)
        )

    def manifest(self, scope: str | None = None) -> str:
        records = [
            record
            for record in self._records.values()
            if scope is None or record.scope == scope
        ]
        material = "\n".join(
            f"{record.id}\t{record.version}\t{record.forgotten_at or 0}\t"
            f"{record.expires_at or 0}"
            for record in sorted(records, key=lambda item: item.id)
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class MemoryService:
    def __init__(
        self,
        store: JsonMemoryStore,
        *,
        events: EventLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._events = events
        self._clock = clock

    @property
    def store(self) -> JsonMemoryStore:
        return self._store

    def write(self, request: MemoryWrite, *, now: float | None = None) -> WriteResult:
        now = self._clock() if now is None else now
        self._validate_write(request)
        latest = self._store.latest(request.scope, request.key)
        active = latest is not None and latest.is_active(now)

        if not active:
            if request.expected_version not in (None, 0):
                return self._conflict(
                    latest, "key has no matching version; recall before writing"
                )
            version = latest.version + 1 if latest is not None else 1
            record = self._make_record(
                request,
                version,
                now,
                supersedes=latest.id if latest is not None else None,
            )
            self._store.add(record)
            self._store.save()
            self._emit(
                MEMORY_WRITTEN,
                scope=request.scope,
                key=request.key,
                version=version,
                status="created",
            )
            return WriteResult(WriteStatus.CREATED, record)

        if request.value == latest.value:
            if (
                request.expected_version is not None
                and request.expected_version != latest.version
            ):
                return self._conflict(latest, "version mismatch")
            self._emit(
                MEMORY_DUPLICATE,
                scope=request.scope,
                key=request.key,
                version=latest.version,
            )
            return WriteResult(
                WriteStatus.DUPLICATE,
                latest,
                latest,
                "same value already stored",
            )

        if request.expected_version != latest.version:
            return self._conflict(
                latest,
                "version mismatch; recall and retry with the current version",
            )

        record = self._make_record(
            request, latest.version + 1, now, supersedes=latest.id
        )
        self._store.add(record)
        self._store.save()
        self._emit(
            MEMORY_WRITTEN,
            scope=request.scope,
            key=request.key,
            version=record.version,
            status="updated",
        )
        return WriteResult(WriteStatus.UPDATED, record)

    def recall(
        self,
        scope: str,
        *,
        query: str | None = None,
        keys: Iterable[str] | None = None,
        limit: int = 10,
        budget_tokens: int = 800,
        now: float | None = None,
    ) -> RecallResult:
        self._validate_scope(scope)
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if budget_tokens < 1:
            raise ValueError("budget_tokens must be at least 1")
        now = self._clock() if now is None else now

        records = list(self._store.active(scope, now))
        if keys is not None:
            allowed = set(keys)
            records = [record for record in records if record.key in allowed]
        if query:
            lowered = query.strip().lower()
            records = [
                record
                for record in records
                if lowered in record.key.lower()
                or lowered in record.value.lower()
            ]
            records.sort(
                key=lambda record: (
                    0 if record.key.lower().startswith(lowered) else 1,
                    -record.confidence,
                    -record.updated_at,
                    record.key,
                )
            )
        else:
            records.sort(key=lambda record: (-record.confidence, -record.updated_at))
        records = records[:limit]

        header = "[memory]\n"
        used = estimate_text(header)
        lines: list[str] = []
        included: list[MemoryRecord] = []
        dropped: list[str] = []
        for position, record in enumerate(records):
            line = (
                f"- {record.key}: {record.value} "
                f"[v{record.version}, {record.source.run_id}, "
                f"{self._date(record.updated_at)}]"
            )
            cost = estimate_text(line + "\n")
            if used + cost > budget_tokens:
                dropped.extend(item.key for item in records[position:])
                break
            used += cost
            lines.append(line)
            included.append(record)

        text = header + "\n".join(lines) if lines else ""
        used = estimate_text(text)
        self._emit(
            MEMORY_RECALLED,
            scope=scope,
            count=len(included),
            used=used,
            dropped=len(dropped),
        )
        return RecallResult(tuple(included), text, used, tuple(dropped))

    def get(self, scope: str, key: str, *, now: float | None = None) -> MemoryRecord | None:
        self._validate_scope(scope)
        now = self._clock() if now is None else now
        record = self._store.latest(scope, key)
        return record if record is not None and record.is_active(now) else None

    def list(self, scope: str, *, now: float | None = None) -> tuple[MemoryRecord, ...]:
        self._validate_scope(scope)
        now = self._clock() if now is None else now
        return self._store.active(scope, now)

    def forget(
        self,
        scope: str,
        key: str,
        *,
        expected_version: int | None = None,
        now: float | None = None,
    ) -> ForgetResult:
        self._validate_scope(scope)
        now = self._clock() if now is None else now
        latest = self._store.latest(scope, key)
        if latest is None or latest.forgotten_at is not None:
            return ForgetResult("not_found", None, "key is not stored")
        if expected_version is not None and expected_version != latest.version:
            return ForgetResult(
                "conflict",
                latest,
                "version mismatch; recall before forgetting",
            )
        updated = replace(latest, forgotten_at=now)
        self._store.replace(updated)
        self._store.save()
        self._emit(
            MEMORY_FORGOTTEN,
            scope=scope,
            key=key,
            version=latest.version,
        )
        return ForgetResult("forgotten", updated)

    def purge_expired(
        self, scope: str | None = None, *, now: float | None = None
    ) -> int:
        now = self._clock() if now is None else now
        expired = 0
        for record in self._store.latest_records(scope):
            if record.forgotten_at is None and record.is_expired(now):
                expired += 1
                self._emit(
                    MEMORY_EXPIRED,
                    scope=record.scope,
                    key=record.key,
                    version=record.version,
                )
        return expired

    def manifest(self, scope: str | None = None) -> str:
        return self._store.manifest(scope)

    def checkpoint_meta(self, scope: str) -> dict[str, Any]:
        now = self._clock()
        return {
            "memory_manifest": self.manifest(scope),
            "memory_keys": len(self._store.active(scope, now)),
        }

    def _make_record(
        self,
        request: MemoryWrite,
        version: int,
        now: float,
        *,
        supersedes: str | None,
    ) -> MemoryRecord:
        material = "\x00".join(
            (
                request.scope,
                request.key,
                str(version),
                request.value,
                request.source.run_id,
                str(now),
            )
        )
        record_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        expires_at = (
            now + request.ttl_seconds
            if request.ttl_seconds is not None
            else None
        )
        return MemoryRecord(
            id=record_id,
            scope=request.scope,
            key=request.key,
            value=request.value,
            version=version,
            source=request.source,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            supersedes=supersedes,
            confidence=request.confidence,
            tags=tuple(request.tags),
        )

    def _conflict(self, latest: MemoryRecord | None, message: str) -> WriteResult:
        self._emit(
            MEMORY_CONFLICT,
            scope=latest.scope if latest else "",
            key=latest.key if latest else "",
            current_version=latest.version if latest else 0,
        )
        return WriteResult(WriteStatus.CONFLICT, None, latest, message)

    def _emit(self, event_type: str, **data: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, **data)

    @staticmethod
    def _date(timestamp: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(timestamp))

    def _validate_write(self, request: MemoryWrite) -> None:
        self._validate_scope(request.scope)
        if not request.key or not KEY_RE.match(request.key):
            raise ValueError(
                "memory key must match [A-Za-z0-9][A-Za-z0-9_.:-]*"
            )
        if not isinstance(request.value, str) or not request.value.strip():
            raise ValueError("memory value must be a non-empty string")
        if "\x00" in request.value:
            raise ValueError("memory value must not contain NUL bytes")
        if not request.source.run_id:
            raise ValueError("memory source requires a run_id")
        if request.expected_version is not None and request.expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        if request.ttl_seconds is not None and request.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not 0.0 <= request.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        for tag in request.tags:
            if not tag or "\x00" in tag:
                raise ValueError("memory tags must be non-empty strings")

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if not scope or not scope.strip():
            raise ValueError("memory scope must not be empty")
        if "\x00" in scope:
            raise ValueError("memory scope must not contain NUL bytes")


def memory_tools(
    service: MemoryService,
    *,
    scope: str,
    run_id: str,
    recall_budget: int = 800,
) -> list[ToolSpec]:
    def remember(
        key: str,
        value: str,
        ttl_seconds: int | None = None,
        expected_version: int | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
    ) -> str:
        result = service.write(
            MemoryWrite(
                scope=scope,
                key=key,
                value=value,
                source=MemorySource(run_id, "remember", time.time()),
                expected_version=expected_version,
                ttl_seconds=ttl_seconds,
                confidence=confidence,
                tags=tuple(tags or ()),
            )
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)

    def recall(
        query: str | None = None,
        keys: list[str] | None = None,
        limit: int = 10,
        budget_tokens: int | None = None,
    ) -> str:
        result = service.recall(
            scope,
            query=query,
            keys=keys,
            limit=limit,
            budget_tokens=budget_tokens or recall_budget,
        )
        return result.text

    def forget(key: str, expected_version: int | None = None) -> str:
        result = service.forget(
            scope, key, expected_version=expected_version
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)

    return [
        ToolSpec(
            name="remember",
            description="Store a fact for this scope; updates require the current version.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "ttl_seconds": {"type": "integer", "minimum": 1},
                    "expected_version": {"type": "integer", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            handler=remember,
            idempotent=True,
            max_attempts=2,
            risk=RiskLevel.WRITE,
        ),
        ToolSpec(
            name="recall",
            description="Recall stored facts within the token budget.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "budget_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4000,
                    },
                },
                "additionalProperties": False,
            },
            handler=recall,
            read_only=True,
            risk=RiskLevel.READ,
        ),
        ToolSpec(
            name="forget",
            description="Forget a stored fact within this scope.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "expected_version": {"type": "integer", "minimum": 0},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
            handler=forget,
            idempotent=True,
            max_attempts=2,
            risk=RiskLevel.WRITE,
        ),
    ]