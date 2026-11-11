"""Prompt assembly: segments, stable prefix, untrusted data, wire assertions (I9)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .events import (
    PROMPT_ASSEMBLED,
    PROMPT_ASSERTION_FAILED,
    PROMPT_INJECTION_SANITIZED,
    EventLog,
)
from .messages import Message
from .tools.spec import ToolSpec

ROLE_RE = re.compile(r"(?im)^\s*(system|assistant|user|tool)\s*:")
SPECIAL_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<</SYS>>|<<SYS>>|\[/?INST\]"
)
BOUNDARY_RE = re.compile(r"</?untrusted(?:\s[^>]*)?>")
SKILL_RE = re.compile(r"^\[skill loaded:\s*([^\s]+)")


class Trust(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class PromptSegment:
    name: str
    content: str
    priority: int = 100
    stable: bool = False
    trust: Trust = Trust.TRUSTED
    version: str = "1"
    channel: str = "system"


@dataclass(frozen=True)
class WireAssertion:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class AssembledPrompt:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    segments: tuple[PromptSegment, ...]
    stable_prefix: str
    stable_hash: str
    prompt_version: str
    assertions: tuple[WireAssertion, ...]
    sanitized: dict[str, int]
    deduplicated_skills: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in self.tools
            ],
        }

    @property
    def failed_assertions(self) -> tuple[WireAssertion, ...]:
        return tuple(item for item in self.assertions if not item.passed)


class WireAssertionError(Exception):
    def __init__(self, report: AssembledPrompt) -> None:
        failed = ", ".join(item.name for item in report.failed_assertions)
        super().__init__(f"wire assertions failed: {failed}")
        self.report = report


def sanitize_untrusted(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def role(match: re.Match[str]) -> str:
        counts["role_markers"] = counts.get("role_markers", 0) + 1
        return "[sanitized-role]"

    def special(match: re.Match[str]) -> str:
        counts["special_tokens"] = counts.get("special_tokens", 0) + 1
        return "[sanitized-token]"

    def boundary(match: re.Match[str]) -> str:
        counts["boundary_tags"] = counts.get("boundary_tags", 0) + 1
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    clean = ROLE_RE.sub(role, text)
    clean = SPECIAL_RE.sub(special, clean)
    clean = BOUNDARY_RE.sub(boundary, clean)
    return clean, counts


def wrap_untrusted(kind: str, content: str, **attributes: str) -> tuple[str, dict[str, int]]:
    clean, counts = sanitize_untrusted(content)
    attrs = "".join(
        f' {key}="{value}"' for key, value in sorted(attributes.items())
    )
    return (
        f'<untrusted kind="{kind}"{attrs}>\n{clean}\n</untrusted>',
        counts,
    )


class PromptAssembler:
    def __init__(
        self,
        base_segments: Sequence[PromptSegment],
        *,
        version: str = "1",
        strict: bool = True,
        events: EventLog | None = None,
    ) -> None:
        for segment in base_segments:
            if segment.stable and segment.trust is Trust.UNTRUSTED:
                raise ValueError("stable segments must be trusted")
        self._base = tuple(base_segments)
        self._version = version
        self._strict = strict
        self._events = events

    def assemble(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        skill_index: str | None = None,
        memory_text: str | None = None,
        dynamic_segments: Sequence[PromptSegment] = (),
    ) -> AssembledPrompt:
        segments = list(self._base)
        if skill_index:
            segments.append(
                PromptSegment(
                    name="skills.index",
                    content=skill_index,
                    priority=200,
                    stable=False,
                    trust=Trust.TRUSTED,
                )
            )
        segments.extend(dynamic_segments)

        system_segments = [
            segment for segment in segments if segment.channel == "system"
        ]
        stable = sorted(
            (segment for segment in system_segments if segment.stable),
            key=lambda segment: (segment.priority, segment.name),
        )
        dynamic = sorted(
            (segment for segment in system_segments if not segment.stable),
            key=lambda segment: (segment.priority, segment.name),
        )
        stable_prefix = "\n\n".join(self._render_system(item) for item in stable)
        dynamic_text = "\n\n".join(self._render_system(item) for item in dynamic)

        input_system = [
            message.content or ""
            for message in messages
            if message.role == "system" and (message.content or "").strip()
        ]
        system_parts = [part for part in (stable_prefix, dynamic_text) if part]
        system_parts.extend(input_system)
        system_content = "\n\n".join(system_parts)

        wire: list[Message] = []
        if system_content:
            wire.append(Message.system(system_content))

        sanitized: dict[str, int] = {}
        if memory_text:
            wrapped, counts = wrap_untrusted("memory", memory_text)
            self._merge_counts(sanitized, counts)
            wire.append(Message.user(wrapped))

        for segment in segments:
            if segment.channel == "system":
                continue
            wrapped, counts = wrap_untrusted(segment.channel, segment.content)
            self._merge_counts(sanitized, counts)
            wire.append(Message.user(wrapped))

        seen_skills: set[str] = set()
        deduplicated = 0
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                content = message.content or ""
                match = SKILL_RE.match(content)
                if match is not None:
                    name = match.group(1)
                    if name in seen_skills:
                        content = f"[skill body omitted: duplicate {name}]"
                        deduplicated += 1
                    else:
                        seen_skills.add(name)
                wrapped, counts = wrap_untrusted(
                    "tool", content, call=message.tool_call_id or ""
                )
                self._merge_counts(sanitized, counts)
                wire.append(replace(message, content=wrapped))
                continue
            wire.append(message)

        assertions = [
            WireAssertion(
                "system_first",
                bool(wire) and wire[0].role == "system",
                "system message must be first",
            ),
            WireAssertion(
                "single_system",
                sum(1 for message in wire if message.role == "system") == 1,
                "wire must contain exactly one system message",
            ),
            self._tool_pairing(wire),
            self._tool_names(tools),
            WireAssertion(
                "skill_unique",
                True,
                f"deduplicated={deduplicated}",
            ),
            self._untrusted_delimited(wire, bool(memory_text)),
            WireAssertion(
                "stable_prefix",
                bool(stable_prefix),
                f"stable_hash={self._hash(stable_prefix)}",
            ),
        ]

        stable_hash = self._hash(stable_prefix)
        prompt_version = self._prompt_version(segments, tools, stable_prefix)

        if self._events is not None:
            self._events.emit(
                PROMPT_ASSEMBLED,
                prompt_version=prompt_version,
                stable_hash=stable_hash,
                segments=[segment.name for segment in segments],
                failed=[item.name for item in assertions if not item.passed],
                sanitized=dict(sanitized),
                deduplicated_skills=deduplicated,
            )
            if sanitized:
                self._events.emit(
                    PROMPT_INJECTION_SANITIZED, counts=dict(sanitized)
                )
            for assertion in assertions:
                if not assertion.passed:
                    self._events.emit(
                        PROMPT_ASSERTION_FAILED,
                        name=assertion.name,
                        detail=assertion.detail,
                    )

        report = AssembledPrompt(
            messages=tuple(wire),
            tools=tuple(tools),
            segments=tuple(segments),
            stable_prefix=stable_prefix,
            stable_hash=stable_hash,
            prompt_version=prompt_version,
            assertions=tuple(assertions),
            sanitized=dict(sanitized),
            deduplicated_skills=deduplicated,
        )
        if self._strict and report.failed_assertions:
            raise WireAssertionError(report)
        return report

    @staticmethod
    def _render_system(segment: PromptSegment) -> str:
        return f"## {segment.name}\n{segment.content}"

    @staticmethod
    def _merge_counts(target: dict[str, int], counts: dict[str, int]) -> None:
        for key, value in counts.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _prompt_version(
        self,
        segments: Sequence[PromptSegment],
        tools: Sequence[ToolSpec],
        stable_prefix: str,
    ) -> str:
        parts = [self._version, stable_prefix]
        for segment in sorted(segments, key=lambda item: item.name):
            digest = self._hash(segment.content)
            parts.append(f"{segment.name}:{segment.version}:{digest}")
        for spec in sorted(tools, key=lambda item: item.name):
            digest = self._hash(
                json.dumps(spec.parameters, ensure_ascii=False, sort_keys=True)
            )
            parts.append(f"tool:{spec.name}:{digest}")
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _tool_pairing(messages: Sequence[Message]) -> WireAssertion:
        pending: list[str] = []
        for message in messages:
            if message.role == "tool":
                if not pending or message.tool_call_id != pending[0]:
                    return WireAssertion(
                        "tool_pairing", False, "orphan or out-of-order tool result"
                    )
                pending.pop(0)
                continue
            if pending:
                return WireAssertion(
                    "tool_pairing", False, "tool batch is not closed"
                )
            if message.role == "assistant" and message.tool_calls:
                pending = [call.id for call in message.tool_calls]
        passed = not pending
        return WireAssertion(
            "tool_pairing",
            passed,
            "" if passed else "tool call without result",
        )

    @staticmethod
    def _tool_names(tools: Sequence[ToolSpec]) -> WireAssertion:
        names = [spec.name for spec in tools]
        passed = len(names) == len(set(names))
        return WireAssertion(
            "tool_names_unique",
            passed,
            "" if passed else "duplicate tool names",
        )

    @staticmethod
    def _untrusted_delimited(
        messages: Sequence[Message], has_memory: bool
    ) -> WireAssertion:
        for message in messages:
            if message.role == "tool":
                content = message.content or ""
                if not content.startswith("<untrusted") or not content.endswith(
                    "</untrusted>"
                ):
                    return WireAssertion(
                        "untrusted_delimited",
                        False,
                        "tool result is not wrapped",
                    )
        if has_memory:
            memory_messages = [
                message
                for message in messages
                if message.role == "user"
                and (message.content or "").startswith('<untrusted kind="memory"')
            ]
            if not memory_messages:
                return WireAssertion(
                    "untrusted_delimited", False, "memory data is not wrapped"
                )
        return WireAssertion("untrusted_delimited", True, "")