"""Message and tool-call types shared by the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool message requires a non-empty tool_call_id")
            if self.tool_calls:
                raise ValueError("tool message cannot carry tool_calls")
            if not isinstance(self.content, str):
                raise ValueError("tool message requires string content")
            return
        if self.tool_call_id is not None:
            raise ValueError("only tool messages carry tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages carry tool_calls")

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(
        cls, content: str | None, tool_calls: tuple[ToolCall, ...] = ()
    ) -> Message:
        return cls("assistant", content, tuple(tool_calls))

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls("tool", content, tool_call_id=call.id)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Message:
        return cls(
            role=payload["role"],
            content=payload.get("content"),
            tool_calls=tuple(
                ToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=call.get("arguments"),
                    raw_arguments=call.get("raw_arguments"),
                )
                for call in payload.get("tool_calls", ())
            ),
            tool_call_id=payload.get("tool_call_id"),
        )