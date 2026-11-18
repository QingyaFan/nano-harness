"""Subprocess sandbox: argv only, allowlisted executables, bounded env (I5)."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import threading
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
    max_argument_bytes: int = 8192
    max_stdin_bytes: int = 16_384
    max_output_bytes: int = 65_536
    timeout: float = 10.0
    env_allowlist: tuple[str, ...] = (
        "LANG", "LC_ALL", "LC_CTYPE", "TZ", "NO_COLOR",
    )

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
        timeout: float | None = None,
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
        executable = shutil.which(command[0], path=os.defpath)
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

        payload = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
        if payload is not None and len(payload) > self._policy.max_stdin_bytes:
            raise SandboxViolation(
                "stdin_too_large",
                f"stdin exceeds {self._policy.max_stdin_bytes} bytes",
            )
        limit = self._policy.timeout if timeout is None else timeout

        environment: dict[str, str] = {"PATH": os.defpath}
        for name in self._policy.env_allowlist:
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value

        process = subprocess.Popen(
            [executable, *command[1:]],
            cwd=str(workdir),
            env=environment,
            stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=self._limits(limit),  # noqa: PLW1509
        )
        stdout, stderr = self._communicate(process, payload, limit)
        return SandboxResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _communicate(
        self,
        process: subprocess.Popen[bytes],
        payload: bytes | None,
        limit: float,
    ) -> tuple[str, str]:
        captured: dict[str, tuple[bytes, bool]] = {}

        def reader(name: str, stream: Any) -> None:
            cap = self._policy.max_output_bytes
            chunks: list[bytes] = []
            total = 0
            truncated = False
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                if total < cap:
                    take = chunk[: cap - total]
                    chunks.append(take)
                    total += len(take)
                if total >= cap:
                    truncated = True
            captured[name] = (b"".join(chunks), truncated)

        readers = [
            threading.Thread(target=reader, args=("stdout", process.stdout)),
            threading.Thread(target=reader, args=("stderr", process.stderr)),
        ]
        for thread in readers:
            thread.start()
        if payload is not None and process.stdin is not None:
            process.stdin.write(payload)
            process.stdin.close()
        try:
            process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            for thread in readers:
                thread.join()
            raise TimeoutError(f"command timed out after {limit}s") from None
        for thread in readers:
            thread.join()

        results = []
        for name in ("stdout", "stderr"):
            data, truncated = captured.get(name, (b"", False))
            text = data.decode("utf-8", errors="replace")
            if truncated:
                text += "\n[truncated]"
            results.append(text)
        return results[0], results[1]

    def _limits(self, timeout: float) -> Any:
        def apply() -> None:
            try:
                import resource
            except ImportError:
                return
            limits = {
                resource.RLIMIT_CPU: (max(1, math.ceil(timeout) + 1), None),
                resource.RLIMIT_FSIZE: (10 * 1024 * 1024, None),
                resource.RLIMIT_NOFILE: (64, None),
                resource.RLIMIT_AS: (512 * 1024 * 1024, None),
                resource.RLIMIT_CORE: (0, 0),
            }
            for key, (soft, hard) in limits.items():
                try:
                    _, current_hard = resource.getrlimit(key)
                    upper = current_hard if hard is None else hard
                    if current_hard != resource.RLIM_INFINITY:
                        upper = min(upper, current_hard)
                    lower = (
                        soft if upper == resource.RLIM_INFINITY
                        else min(soft, upper)
                    )
                    resource.setrlimit(key, (lower, upper))
                except (ValueError, OSError):
                    pass

        return apply


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