"""Scoped credentials and access callbacks (I5)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ..tools.errors import ErrorKind, ToolViolation


class CredentialDenied(ToolViolation):
    def __init__(self, name: str, reason: str) -> None:
        super().__init__(
            "credential_denied",
            f"credential {name} is not available: {reason}",
            kind=ErrorKind.CREDENTIAL_DENIED,
        )


@dataclass(frozen=True)
class ScopedCredentials:
    values: Mapping[str, str]
    allowed: frozenset[str]
    on_access: Callable[[str], None] | None = None
    on_denied: Callable[[str, str], None] | None = None

    def get(self, name: str) -> str:
        if name not in self.allowed or name not in self.values:
            if self.on_denied is not None:
                self.on_denied(name, "not granted")
            raise CredentialDenied(name, "not granted")
        if self.on_access is not None:
            self.on_access(name)
        return self.values[name]


class CredentialBroker:
    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    def scope(
        self,
        granted: Iterable[str],
        *,
        on_access: Callable[[str], None] | None = None,
        on_denied: Callable[[str, str], None] | None = None,
    ) -> ScopedCredentials:
        allowed = frozenset(granted)
        values = {
            name: value
            for name, value in self._values.items()
            if name in allowed
        }
        return ScopedCredentials(values, allowed, on_access, on_denied)