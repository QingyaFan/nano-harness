"""Context budget, token estimation, compaction cascade, and artifacts (I3)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from .messages import Message
from .tools.spec import ToolSpec

SUMMARY_PREFIX = "[summary of"

CJK_RANGES = (
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)

KNOWN_WINDOWS = (
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("claude", 200_000),
    ("gemini", 1_000_000),
    ("qwen", 131_072),
    ("deepseek", 128_000),
)
FALLBACK_WINDOW = 128_000


@dataclass(frozen=True)
class WindowInfo:
    size: int
    source: str
    model: str | None = None


def resolve_window(authoritative: int | None = None, model: str | None = None) -> WindowInfo:
    if authoritative is not None:
        if isinstance(authoritative, bool) or not isinstance(authoritative, int):
            raise ValueError("window must be a positive integer, not a bool or string")
        if authoritative <= 0:
            raise ValueError("window must be a positive integer")
        return WindowInfo(size=authoritative, source="authoritative", model=model)

    name = (model or "").lower()
    for prefix, size in KNOWN_WINDOWS:
        if prefix in name:
            return WindowInfo(size=size, source="inferred", model=model)
    return WindowInfo(size=FALLBACK_WINDOW, source="fallback", model=model)


def _char_units(char: str) -> int:
    code = ord(char)
    if code < 128:
        if char.isalnum() or char in " \n\t":
            return 1
        return 3
    if code > 0xFFFF and not any(start <= code <= end for start, end in CJK_RANGES):
        return 8
    for start, end in CJK_RANGES:
        if start <= code <= end:
            return 6
    return 4


def estimate_text(text: str | None) -> int:
    if not text:
        return 0
    units = sum(_char_units(char) for char in text)
    return (units + 3) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    budget_units = max_tokens * 4
    units = 0
    for index, char in enumerate(text):
        units += _char_units(char)
        if units > budget_units:
            return text[:index]
    return text


def estimate_messages(messages: Iterable[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_text(message.content)
        for call in message.tool_calls:
            total += estimate_text(call.name)
            total += estimate_text(
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            )
    return total


def estimate_tools(specs: Iterable[ToolSpec]) -> int:
    total = 0
    for spec in specs:
        total += estimate_text(spec.name)
        total += estimate_text(spec.description)
        total += estimate_text(
            json.dumps(spec.parameters, ensure_ascii=False, sort_keys=True)
        )
    return total


class TokenMeter:
    """Estimate tokens and calibrate against the provider's reported usage."""

    def __init__(self, samples: int = 8) -> None:
        self._samples = samples
        self._ratios: list[float] = []
        self._calibration = 0

    def count_text(self, text: str | None) -> int:
        base = estimate_text(text)
        if not base:
            return 0
        if self._ratios:
            scale = sorted(self._ratios)[len(self._ratios) // 2]
            return max(1, int(base * scale + 0.5))
        return base

    def count_messages(self, messages: Iterable[Message]) -> int:
        total = 0
        for message in messages:
            total += self.count_text(message.content)
            for call in message.tool_calls:
                total += self.count_text(call.name)
                total += self.count_text(
                    json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
                )
        return total

    def count_tools(self, specs: Iterable[ToolSpec]) -> int:
        total = 0
        for spec in specs:
            total += self.count_text(spec.name)
            total += self.count_text(spec.description)
            total += self.count_text(
                json.dumps(spec.parameters, ensure_ascii=False, sort_keys=True)
            )
        return total

    def observe(self, text: str | None, actual_tokens: int) -> None:
        base = estimate_text(text)
        if base <= 0 or actual_tokens <= 0:
            return
        self._ratios.append(actual_tokens / base)
        if len(self._ratios) > self._samples:
            del self._ratios[0]

    def anchor(self, delta: int) -> None:
        self._calibration = delta

    @property
    def calibration(self) -> int:
        return self._calibration


@dataclass(frozen=True)
class ContextBudget:
    window: WindowInfo
    threshold_ratio: float = 0.60
    target_ratio: float = 0.50
    reserve_ratio: float = 0.10

    def __post_init__(self) -> None:
        if not (0 < self.threshold_ratio < 1):
            raise ValueError("threshold_ratio must be between 0 and 1")
        if not (0 < self.target_ratio <= 1):
            raise ValueError("target_ratio must be between 0 and 1")
        if not (0 <= self.reserve_ratio < 1):
            raise ValueError("reserve_ratio must be between 0 and 1")
        if self.threshold_ratio + self.reserve_ratio >= 1:
            raise ValueError(
                "threshold_ratio plus reserve_ratio must stay below 1; "
                "otherwise compaction can never trigger before the hard limit"
            )
        if self.threshold >= self.hard_limit:
            raise ValueError("threshold must stay below the hard limit")

    @property
    def threshold(self) -> int:
        return max(1, int(self.window.size * self.threshold_ratio))

    @property
    def target(self) -> int:
        return max(1, int(self.threshold * self.target_ratio))

    @property
    def reserve(self) -> int:
        return int(self.window.size * self.reserve_ratio)

    @property
    def hard_limit(self) -> int:
        return self.window.size - self.reserve


@dataclass(frozen=True)
class Artifact:
    handle: str
    name: str
    content: str
    tokens: int


class ArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, Artifact] = {}
        self._counter = 0

    def add(self, name: str, content: str, tokens: int) -> Artifact:
        self._counter += 1
        artifact = Artifact(
            handle=f"artifact-{self._counter:03d}",
            name=name,
            content=content,
            tokens=tokens,
        )
        self._items[artifact.handle] = artifact
        return artifact

    def get(self, handle: str) -> Artifact:
        return self._items[handle]


def artifact_placeholder(artifact: Artifact, preview_chars: int = 40) -> str:
    preview = artifact.content[:preview_chars].replace("\n", " ")
    if len(artifact.content) > preview_chars:
        preview += "…"
    return (
        f"[artifact {artifact.handle}] {len(artifact.content):,} chars / "
        f"~{artifact.tokens:,} tokens from {artifact.name}; "
        f"preview: {preview!r}. "
        f"Read it with read_artifact(handle={artifact.handle!r})."
    )


def make_artifact_tool(store: ArtifactStore) -> ToolSpec:
    def read(handle: str, offset: int = 0, limit: int = 4000) -> str:
        artifact = store.get(handle)
        return artifact.content[offset : offset + limit]

    return ToolSpec(
        name="read_artifact",
        description="Read a stored tool result by handle.",
        parameters={
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20000},
            },
            "required": ["handle"],
            "additionalProperties": False,
        },
        handler=read,
        read_only=True,
    )


@dataclass(frozen=True)
class CompactionRecord:
    kind: str
    status: str
    before: int
    after: int
    masked: int = 0
    dropped: int = 0
    user: int = 0
    note: str = ""


@dataclass(frozen=True)
class PreparedContext:
    messages: tuple[Message, ...]
    records: tuple[CompactionRecord, ...]
    used: int
    tool_tokens: int
    exhausted: bool


Summarizer = Callable[[tuple[Message, ...], int], str]


class ContextManager:
    def __init__(
        self,
        budget: ContextBudget,
        *,
        meter: TokenMeter | None = None,
        summarizer: Summarizer | None = None,
        micro: bool = True,
        summary_max_tokens: int = 400,
        artifact_threshold: int = 600,
        pin: Callable[[Message], bool] | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._budget = budget
        self._meter = meter or TokenMeter()
        self._summarizer = summarizer
        self._micro_enabled = micro
        self._summary_max_tokens = summary_max_tokens
        self._artifact_threshold = artifact_threshold
        self._pin = pin
        self._artifacts = artifacts or ArtifactStore()
        self._full: list[Message] = []
        self._view: list[Message] = []
        self._records: list[CompactionRecord] = []
        self._exhausted = False
        self._last_tool_tokens = 0

    @property
    def full_messages(self) -> tuple[Message, ...]:
        return tuple(self._full)

    @property
    def records(self) -> tuple[CompactionRecord, ...]:
        return tuple(self._records)

    @property
    def artifacts(self) -> ArtifactStore:
        return self._artifacts

    def add(self, message: Message, *, tool: str | None = None) -> Artifact | None:
        self._full.append(message)
        artifact = None
        content = message.content or ""
        if message.role == "tool" and self._meter.count_text(content) > self._artifact_threshold:
            artifact = self._artifacts.add(
                tool or "tool", content, self._meter.count_text(content)
            )
            view_message = replace(message, content=artifact_placeholder(artifact))
        else:
            view_message = message
        self._view.append(view_message)
        return artifact

    def wire(self, specs: Sequence[ToolSpec] = ()) -> PreparedContext:
        tool_tokens = self._meter.count_tools(specs)
        self._last_tool_tokens = tool_tokens
        records: list[CompactionRecord] = []
        used = self._used(tool_tokens)

        if used > self._budget.threshold:
            if self._micro_enabled:
                record = self._micro(tool_tokens)
                if record is not None:
                    records.append(record)
                    used = self._used(tool_tokens)

            if used > self._budget.target and self._summarizer is not None:
                record = self._summarize(tool_tokens)
                if record is not None:
                    records.append(record)
                    used = self._used(tool_tokens)

            if used > self._budget.target:
                record = self._force_trim(tool_tokens)
                if record is not None:
                    records.append(record)
                    used = self._used(tool_tokens)

            if used > self._budget.hard_limit:
                self._exhausted = True

        self._records.extend(records)
        return PreparedContext(
            messages=tuple(self._view),
            records=tuple(records),
            used=used,
            tool_tokens=tool_tokens,
            exhausted=self._exhausted,
        )

    def anchor(self, actual_prompt_tokens: int) -> None:
        estimated = self._meter.count_messages(self._view) + self._last_tool_tokens
        self._meter.anchor(actual_prompt_tokens - estimated)

    def _used(self, tool_tokens: int) -> int:
        return max(
            0,
            self._meter.count_messages(self._view)
            + tool_tokens
            + self._meter.calibration,
        )

    def _units(self) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        index = 0
        while index < len(self._view):
            start = index
            message = self._view[index]
            index += 1
            if message.role == "assistant" and message.tool_calls:
                while index < len(self._view) and self._view[index].role == "tool":
                    index += 1
            units.append((start, index))
        return units

    def _pinned_units(self, units: list[tuple[int, int]]) -> set[int]:
        unit_of: dict[int, int] = {}
        for unit_index, (start, end) in enumerate(units):
            for index in range(start, end):
                unit_of[index] = unit_index
        pinned: set[int] = set()
        first_user: int | None = None
        for index, message in enumerate(self._view):
            if message.role == "system":
                pinned.add(unit_of[index])
            if message.role == "user" and first_user is None:
                first_user = index
            if (
                message.role == "user"
                and (message.content or "").startswith(SUMMARY_PREFIX)
            ):
                pinned.add(unit_of[index])
            if self._pin is not None and self._pin(message):
                pinned.add(unit_of[index])
        if first_user is not None:
            pinned.add(unit_of[first_user])
        if not pinned and units:
            pinned.add(0)
        return pinned

    def _micro(self, tool_tokens: int) -> CompactionRecord | None:
        before = self._used(tool_tokens)
        units = self._units()
        pinned = self._pinned_units(units)
        unit_of: dict[int, int] = {}
        for unit_index, (start, end) in enumerate(units):
            for index in range(start, end):
                unit_of[index] = unit_index

        tool_indices = [
            index for index, message in enumerate(self._view) if message.role == "tool"
        ]
        keep = set(tool_indices[-2:])
        masked = 0
        for index in tool_indices:
            if index in keep or unit_of[index] in pinned:
                continue
            content = self._view[index].content or ""
            if len(content) <= 160:
                continue
            self._view[index] = replace(
                self._view[index], content=content[:120] + " …[masked]"
            )
            masked += 1

        after = self._used(tool_tokens)
        if not masked:
            return None
        return CompactionRecord("micro", "applied", before, after, masked=masked)

    def _summarize(self, tool_tokens: int) -> CompactionRecord | None:
        if self._summarizer is None:
            return None
        before = self._used(tool_tokens)
        units = self._units()
        pinned = self._pinned_units(units)

        runs: list[list[int]] = []
        current: list[int] = []
        for unit_index in range(len(units)):
            if unit_index in pinned or unit_index == len(units) - 1:
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(unit_index)
        if current:
            runs.append(current)
        if not runs:
            return None

        run = max(runs, key=len)
        take = run[: max(1, len(run) // 2)]
        start = units[take[0]][0]
        end = units[take[-1]][1]
        span = tuple(self._view[start:end])

        try:
            summary = self._summarizer(span, self._summary_max_tokens)
        except Exception as exc:  # noqa: BLE001
            return CompactionRecord(
                "full", "failed", before, before,
                note=f"{type(exc).__name__}: {exc}",
            )
        summary = (summary or "").strip()
        if not summary:
            return CompactionRecord(
                "full", "failed", before, before, note="empty summary"
            )
        summary = truncate_to_tokens(summary, self._summary_max_tokens)
        self._view[start:end] = [
            Message.user(f"{SUMMARY_PREFIX} {len(span)} earlier messages]\n{summary}")
        ]
        after = self._used(tool_tokens)
        return CompactionRecord("full", "applied", before, after, dropped=len(span))

    def _force_trim(self, tool_tokens: int) -> CompactionRecord | None:
        before = self._used(tool_tokens)
        units = self._units()
        pinned = self._pinned_units(units)
        candidates = [
            unit_index
            for unit_index in range(len(units))
            if unit_index not in pinned and unit_index != len(units) - 1
        ]
        without_user: list[int] = []
        with_user: list[int] = []
        for unit_index in candidates:
            start, end = units[unit_index]
            if any(message.role == "user" for message in self._view[start:end]):
                with_user.append(unit_index)
            else:
                without_user.append(unit_index)

        need = before - self._budget.target
        removed: set[int] = set()
        dropped = 0
        user = 0
        saved = 0
        for unit_index in without_user + with_user:
            if saved >= need:
                break
            start, end = units[unit_index]
            span = self._view[start:end]
            saved += self._meter.count_messages(span)
            dropped += len(span)
            user += sum(1 for message in span if message.role == "user")
            removed.add(unit_index)

        if not removed:
            return None

        new_view: list[Message] = []
        for unit_index, (start, end) in enumerate(units):
            if unit_index in removed:
                continue
            new_view.extend(self._view[start:end])
        self._view = new_view

        after = self._used(tool_tokens)
        return CompactionRecord(
            "force_trim", "applied", before, after, dropped=dropped, user=user
        )