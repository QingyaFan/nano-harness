"""Workspace path isolation, atomic writes, and filesystem tools (I5)."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from ..tools.errors import ToolViolation
from ..tools.spec import ToolSpec
from .types import RiskLevel


class WorkspaceViolation(ToolViolation):
    pass


class FileMissing(ToolViolation):
    def __init__(self, message: str) -> None:
        super().__init__("missing", message)


class Workspace:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _candidate(self, relative: str) -> Path:
        if not relative or not relative.strip():
            raise WorkspaceViolation("empty_path", "path must not be empty")
        if "\x00" in relative:
            raise WorkspaceViolation("nul_byte", "path must not contain NUL bytes")
        pure = PurePosixPath(relative)
        if pure.is_absolute():
            raise WorkspaceViolation("absolute_path", "path must be relative")
        if ".." in pure.parts:
            raise WorkspaceViolation("path_escape", "path must stay inside the workspace")
        return self.root.joinpath(*pure.parts)

    def _resolve(self, relative: str, *, must_exist: bool) -> Path:
        candidate = self._candidate(relative)
        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise FileMissing(f"{relative!r} does not exist") from exc
        if not resolved.is_relative_to(self.root):
            raise WorkspaceViolation(
                "path_escape", "resolved path escapes the workspace"
            )
        return resolved

    def read_text(self, relative: str) -> str:
        return self._resolve(relative, must_exist=True).read_text(encoding="utf-8")

    def list_dir(self, relative: str = ".") -> list[str]:
        resolved = self._resolve(relative, must_exist=True)
        if not resolved.is_dir():
            raise WorkspaceViolation("not_a_directory", f"{relative!r} is not a directory")
        return sorted(item.name for item in resolved.iterdir())

    def write_text(self, relative: str, content: str) -> Path:
        target = self._resolve(relative, must_exist=False)
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        temp_name = f".{target.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        fd = -1
        try:
            fd = os.open(
                temp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=dir_fd,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_name, dir_fd=dir_fd)
            except OSError:
                pass
            os.close(dir_fd)
        return target


def workspace_tools(workspace: Workspace) -> list[ToolSpec]:
    def write(path: str, content: str) -> str:
        workspace.write_text(path, content)
        return f"wrote {len(content)} chars to {path}"

    return [
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda path: workspace.read_text(path),
            read_only=True,
            risk=RiskLevel.READ,
        ),
        ToolSpec(
            name="list_dir",
            description="List entries in a workspace directory.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "additionalProperties": False,
            },
            handler=lambda path=".": "\n".join(workspace.list_dir(path)),
            read_only=True,
            risk=RiskLevel.READ,
        ),
        ToolSpec(
            name="write_file",
            description="Atomically write a UTF-8 file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=write,
            idempotent=True,
            max_attempts=2,
            risk=RiskLevel.WRITE,
        ),
    ]