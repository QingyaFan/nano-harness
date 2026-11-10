"""Tool declarations and registration-time validation (I2/I5)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from ..risk import RiskLevel

ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    read_only: bool = False
    idempotent: bool = False
    max_attempts: int | None = None
    risk: RiskLevel = RiskLevel.READ
    credentials: tuple[str, ...] = ()
    _validator: Draft202012Validator = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(self.parameters)
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if (
            self.max_attempts is not None
            and self.max_attempts > 1
            and not (self.read_only or self.idempotent)
        ):
            raise ValueError(
                f"tool {self.name!r} is a non-idempotent write; "
                "automatic retries require read_only=True or idempotent=True"
            )
        if not isinstance(self.risk, RiskLevel):
            object.__setattr__(self, "risk", RiskLevel(self.risk))
        if not isinstance(self.credentials, tuple):
            object.__setattr__(self, "credentials", tuple(self.credentials))
        object.__setattr__(self, "_validator", Draft202012Validator(self.parameters))

    @property
    def validator(self) -> Draft202012Validator:
        return self._validator