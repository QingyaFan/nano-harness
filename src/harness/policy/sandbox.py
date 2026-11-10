"""Subprocess sandbox: argv only, allowlisted executables, bounded env (I5)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools.errors import ToolViolation
from ..tools.spec import ToolSpec
from .types import RiskLevel


class SandboxViolation(ToolViolation):
    pass


@dataclass(frozen=True)
class SandboxPolicy:
    allowed_executables: frozenset[str]
    max_arguments: int = 64
    max_argument_bytes: int = 4096
    timeout: float = 10.0
    env_allowlist: tuple[str, ...] = ("PATH", "LANG", "LC_ALL")

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_executables, frozenset):
            object.__setattr__(
                self, "allowed_executables", frozenset(self.allowed_executables)
            )


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class SubprocessSandbox:
    def __init__(
        self,
        policy: SandboxPolicy,
        workspace_root: str | os.PathLike[str],
    ) -> None:
        self._policy = policy
        self._root = Path(workspace_root).resolve()

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        stdin: str | bytes | None = None,
    ) -> SandboxResult:
        if isinstance(argv, (str, bytes)):
            raise SandboxViolation(
                "shell_string", "argv must be a list of strings, not a shell string"
            )
        command = list(argv)
        if not command:
            raise SandboxViolation("empty_argv", "argv must contain an executable")
        if len(command) > self._policy.max_arguments:
            raise SandboxViolation(
                "too_many_arguments",
                f"argv has {len(command)} items, limit is {self._policy.max_arguments}",
            )
        for item in command:
            if not isinstance(item, str):
                raise SandboxViolation("non_string_argument", "argv items must be strings")
            if "\x00" in item:
                raise SandboxViolation("nul_byte", "argv must not contain NUL bytes")
            if len(item.encode("utf-8")) > self._policy.max_argument_bytes:
                raise SandboxViolation(
                    "argument_too_large",
                    f"argument exceeds {self._policy.max_argument_bytes} bytes",
                )
        if command[0] not in self._policy.allowed_executables:
            raise SandboxViolation(
                "executable_not_allowed",
                f"executable {command[0]!r} is not in the allowlist",
            )
        executable = shutil.which(command[0])
        if executable is None:
            raise SandboxViolation(
                "executable_not_found", f"executable {command[0]!r} was not found"
            )

        if Path(cwd).is_absolute():
            raise SandboxViolation("absolute_cwd", "cwd must be relative")
        workdir = (self._root / cwd).resolve()
        if not workdir.is_relative_to(self._root):
            raise SandboxViolation("path_escape", "cwd escapes the workspace")
        if not workdir.is_dir():
            raise SandboxViolation("cwd_missing", f"{cwd!r} is not a directory")

        environment: dict[str, str] = {}
        for name in self._policy.env_allowlist:
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value
        environment.setdefault("PATH", os.defpath)

        process = subprocess.Popen(
            [executable, *command[1:]],
            cwd=str(workdir),
            env=environment,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        payload = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
        try:
            stdout, stderr = process.communicate(payload, timeout=self._policy.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError(
                f"command timed out after {self._policy.timeout}s"
            ) from None
        return SandboxResult(
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )


def command_tool(sandbox: SubprocessSandbox) -> ToolSpec:
    def run(
        argv: list[str],
        cwd: str = ".",
        stdin: str | None = None,
    ) -> str:
        result = sandbox.run(argv, cwd=cwd, stdin=stdin)
        return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)

    return ToolSpec(
        name="run_command",
        description="Run an allowlisted executable without a shell.",
        parameters={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cwd": {"type": "string", "default": "."},
                "stdin": {"type": "string"},
            },
            "required": ["argv"],
            "additionalProperties": False,
        },
        handler=run,
        risk=RiskLevel.EXEC,
        idempotent=False,
        max_attempts=1,
    )