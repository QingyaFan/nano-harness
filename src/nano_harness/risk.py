"""Shared risk and decision enums."""

from enum import StrEnum


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXEC = "exec"


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"