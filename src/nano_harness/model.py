"""Model protocol and a scripted model for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .messages import Message, ToolCall, Usage


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


class Model(Protocol):
    def complete(
        self, messages: Sequence[Message], tools: Sequence[Any]
    ) -> ModelResponse: ...


class ScriptedModel:
    """Replay a fixed list of responses; useful for deterministic tests."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[tuple[Message, ...], tuple[Any, ...]]] = []

    def complete(
        self, messages: Sequence[Message], tools: Sequence[Any]
    ) -> ModelResponse:
        self.calls.append((tuple(messages), tuple(tools)))
        if not self._responses:
            raise RuntimeError("scripted model exhausted")
        return self._responses.pop(0)