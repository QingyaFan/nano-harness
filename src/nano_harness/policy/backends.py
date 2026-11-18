"""Sandbox backends: one run, one boundary (E5)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from .sandbox import SandboxViolation, SubprocessSandbox

NetworkMode = Literal["none", "proxy"]
DEFAULT_ENV_ALLOWLIST = frozenset({"PROXY_TOKEN", "LANG", "LC_ALL", "TZ"})


class SandboxSpecError(SandboxViolation):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_spec", message)


class SandboxClosed(SandboxViolation):
    def __init__(self, name: str) -> None:
        super().__init__("sandbox_closed", f"sandbox {name!r} is closed")


class SandboxStartError(SandboxViolation):
    def __init__(self, message: str) -> None:
        super().__init__("sandbox_start_failed", message)


@dataclass(frozen=True)
class SandboxSpec:
    image: str
    workspace: Path
    workspace_mount: str = "/workspace"
    root_read_only: bool = True
    network: NetworkMode = "none"
    proxy_network: str = "agent-egress"
    user: str = "10001:10001"
    capabilities: tuple[str, ...] = ()
    no_new_privileges: bool = True
    memory: str | None = None
    cpus: float | None = None
    pids: int | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image:
            raise SandboxSpecError("image must not be empty")
        object.__setattr__(self, "workspace", Path(self.workspace))
        if not self.workspace.is_absolute() or ":" in str(self.workspace):
            raise SandboxSpecError("workspace must be an absolute host path")
        mount = PurePosixPath(self.workspace_mount)
        if not mount.is_absolute() or mount == PurePosixPath("/") or ".." in mount.parts:
            raise SandboxSpecError("workspace_mount must be an absolute container path")
        if ":" in self.workspace_mount:
            raise SandboxSpecError("workspace_mount must not contain ':'")
        if self.network not in ("none", "proxy"):
            raise SandboxSpecError("network must be 'none' or 'proxy'")
        if self.network == "proxy" and not self.proxy_network:
            raise SandboxSpecError("proxy_network must not be empty")
        if self.pids is not None and self.pids <= 0:
            raise SandboxSpecError("pids must be positive")
        if self.cpus is not None and self.cpus <= 0:
            raise SandboxSpecError("cpus must be positive")
        if self.memory is not None and not re.fullmatch(r"[0-9]+[bkmgBKMG]?", self.memory):
            raise SandboxSpecError("memory must look like 256m or 1g")


@dataclass
class SandboxHandle:
    name: str
    workspace: Path
    mount: str
    closed: bool = False


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class SnapshotRef:
    name: str
    path: Path


class SandboxBackend(Protocol):
    def create(self, spec: SandboxSpec) -> SandboxHandle: ...
    def exec(
        self,
        handle: SandboxHandle,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float = 10.0,
    ) -> ExecResult: ...
    def put(self, handle: SandboxHandle, path: str, data: bytes) -> None: ...
    def get(self, handle: SandboxHandle, path: str) -> bytes: ...
    def snapshot(self, handle: SandboxHandle, name: str) -> SnapshotRef: ...
    def destroy(self, handle: SandboxHandle) -> None: ...


def _ensure_open(handle: SandboxHandle) -> None:
    if handle.closed:
        raise SandboxClosed(handle.name)


def _validate_argv(argv: Sequence[str]) -> None:
    if isinstance(argv, (str, bytes)):
        raise SandboxSpecError("argv must be a sequence of strings, not a shell string")
    if not argv:
        raise SandboxSpecError("argv must not be empty")
    for item in argv:
        if not isinstance(item, str):
            raise SandboxSpecError("argv items must be strings")
        if "\x00" in item:
            raise SandboxSpecError("argv must not contain NUL bytes")


def _safe_host_path(root: Path, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SandboxSpecError(f"path must stay inside the workspace: {path!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SandboxSpecError(f"path escapes the workspace: {path!r}")
    return resolved


def _container_path(mount: str, path: str) -> str:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SandboxSpecError(f"path must stay inside the workspace: {path!r}")
    return str(PurePosixPath(mount) / relative)


def _safe_snapshot_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", name).strip(".-")
    if not cleaned:
        raise SandboxSpecError("snapshot name must contain a letter or digit")
    return cleaned


class ContainerBackend:
    def __init__(
        self,
        *,
        docker: str = "docker",
        require_digest: bool = True,
        env_allowlist: frozenset[str] = DEFAULT_ENV_ALLOWLIST,
        snapshot_dir: Path | None = None,
    ) -> None:
        self._docker = docker
        self._require_digest = require_digest
        self._env_allowlist = env_allowlist
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "nano-harness-snapshots"

    def build_run_args(self, spec: SandboxSpec, name: str) -> list[str]:
        self._check_spec(spec)
        args = [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "nano-harness.managed=true",
        ]
        for key, value in spec.labels.items():
            args.extend(["--label", f"{key}={value}"])
        if spec.root_read_only:
            args.append("--read-only")
        args.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,nodev"])
        args.extend(["-v", f"{spec.workspace}:{spec.workspace_mount}:rw"])
        network = "none" if spec.network == "none" else spec.proxy_network
        args.extend(["--network", network])
        args.extend(["--user", spec.user])
        args.extend(["--cap-drop", "ALL"])
        for capability in spec.capabilities:
            args.extend(["--cap-add", capability])
        if spec.no_new_privileges:
            args.extend(["--security-opt", "no-new-privileges"])
        if spec.pids is not None:
            args.extend(["--pids-limit", str(spec.pids)])
        if spec.memory is not None:
            args.extend(["--memory", spec.memory])
        if spec.cpus is not None:
            args.extend(["--cpus", str(spec.cpus)])
        for key, value in spec.env.items():
            args.extend(["--env", f"{key}={value}"])
        args.extend([spec.image, "sleep", "infinity"])
        return args

    def create(self, spec: SandboxSpec) -> SandboxHandle:
        if shutil.which(self._docker) is None:
            raise SandboxStartError(f"{self._docker!r} CLI not found")
        name = f"nh-{uuid.uuid4().hex[:12]}"
        args = self.build_run_args(spec, name)
        result = self._run(args, timeout=60.0, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise SandboxStartError(detail or "container failed to start")
        return SandboxHandle(name=name, workspace=spec.workspace, mount=spec.workspace_mount)

    def exec(
        self,
        handle: SandboxHandle,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float = 10.0,
    ) -> ExecResult:
        _ensure_open(handle)
        _validate_argv(argv)
        workdir = handle.mount if cwd == "." else _container_path(handle.mount, cwd)
        result = self._run(
            ["exec", "-w", workdir, handle.name, *argv],
            timeout=timeout,
            check=False,
        )
        return ExecResult(result.returncode, result.stdout, result.stderr)

    def put(self, handle: SandboxHandle, path: str, data: bytes) -> None:
        _ensure_open(handle)
        container_path = _container_path(handle.mount, path)
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(data)
            temp.flush()
            result = self._run(
                ["cp", temp.name, f"{handle.name}:{container_path}"],
                check=False,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise SandboxViolation("docker_copy_failed", detail or "docker cp failed")

    def get(self, handle: SandboxHandle, path: str) -> bytes:
        _ensure_open(handle)
        container_path = _container_path(handle.mount, path)
        result = subprocess.run(
            [self._docker, "cp", f"{handle.name}:{container_path}", "-"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
            raise SandboxViolation("docker_copy_failed", detail or "docker cp failed")
        return result.stdout

    def snapshot(self, handle: SandboxHandle, name: str) -> SnapshotRef:
        _ensure_open(handle)
        safe = _safe_snapshot_name(name)
        result = subprocess.run(
            [self._docker, "cp", f"{handle.name}:{handle.mount}", "-"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
            raise SandboxViolation("docker_snapshot_failed", detail or "docker cp failed")
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_dir / f"{handle.name}-{safe}.tar"
        path.write_bytes(result.stdout)
        return SnapshotRef(safe, path)

    def destroy(self, handle: SandboxHandle) -> None:
        if handle.closed:
            return
        try:
            self._run(["rm", "-f", handle.name], check=False)
        finally:
            handle.closed = True

    def _check_spec(self, spec: SandboxSpec) -> None:
        if self._require_digest and "@sha256:" not in spec.image:
            raise SandboxSpecError("image must be pinned by digest")
        if not spec.workspace.is_dir():
            raise SandboxSpecError(f"workspace is not a directory: {spec.workspace}")
        forbidden = set(spec.env) - self._env_allowlist
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise SandboxSpecError(f"env keys are not allowed in the sandbox: {names}")
        for key in spec.labels:
            if not re.fullmatch(r"[A-Za-z0-9./_-]+", key):
                raise SandboxSpecError(f"invalid label key: {key!r}")

    def _run(
        self,
        args: list[str],
        *,
        timeout: float = 30.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [self._docker, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SandboxStartError(f"{self._docker!r} CLI not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"docker command timed out after {timeout}s") from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise SandboxViolation("docker_failed", detail or "docker command failed")
        return result


class ProcessBackend:
    def __init__(self, sandbox: SubprocessSandbox) -> None:
        self._sandbox = sandbox

    def create(self, spec: SandboxSpec) -> SandboxHandle:
        spec.workspace.mkdir(parents=True, exist_ok=True)
        return SandboxHandle(
            name=f"proc-{uuid.uuid4().hex[:12]}",
            workspace=spec.workspace,
            mount=spec.workspace_mount,
        )

    def exec(
        self,
        handle: SandboxHandle,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float = 10.0,
    ) -> ExecResult:
        _ensure_open(handle)
        _validate_argv(argv)
        result = self._sandbox.run(argv, cwd=cwd, timeout=timeout)
        return ExecResult(result.returncode, result.stdout, result.stderr)

    def put(self, handle: SandboxHandle, path: str, data: bytes) -> None:
        _ensure_open(handle)
        target = _safe_host_path(handle.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get(self, handle: SandboxHandle, path: str) -> bytes:
        _ensure_open(handle)
        return _safe_host_path(handle.workspace, path).read_bytes()

    def snapshot(self, handle: SandboxHandle, name: str) -> SnapshotRef:
        _ensure_open(handle)
        safe = _safe_snapshot_name(name)
        path = handle.workspace.parent / f"{handle.name}-{safe}.tar"
        with tarfile.open(path, "w") as archive:
            archive.add(handle.workspace, arcname=".")
        return SnapshotRef(safe, path)

    def destroy(self, handle: SandboxHandle) -> None:
        handle.closed = True