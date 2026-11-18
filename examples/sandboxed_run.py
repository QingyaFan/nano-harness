"""E5 example: one run, one sandbox boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nano_harness.policy import (
    ContainerBackend,
    ProcessBackend,
    SandboxPolicy,
    SandboxSpec,
    SubprocessSandbox,
)


@dataclass
class RunToken:
    value: str
    revoked: bool = False

    def revoke(self) -> None:
        self.revoked = True


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "info"], capture_output=True, check=False)
    return result.returncode == 0


def python_binary() -> str:
    return "python3" if shutil.which("python3") else "python"


def main() -> None:
    token = RunToken("proxy-token")
    container = docker_available()
    if not container:
        print("docker unavailable; using process backend (degraded)")

    with tempfile.TemporaryDirectory(prefix="nano-harness-e5-") as temp:
        workspace = Path(temp)
        if container:
            workspace.chmod(0o777)
            backend = ContainerBackend(require_digest=False)
            spec = SandboxSpec(
                image=os.environ.get("NANO_HARNESS_IMAGE", "python:3.11-slim"),
                workspace=workspace,
                root_read_only=True,
                network="none",
                user="10001:10001",
                capabilities=(),
                no_new_privileges=True,
                pids=64,
                memory="256m",
                cpus=0.5,
                env={"PROXY_TOKEN": token.value},
                labels={"nano-harness.example": "e5"},
            )
            python = "python"
            workspace_file = "/workspace/e5.txt"
        else:
            backend = ProcessBackend(
                SubprocessSandbox(
                    SandboxPolicy(frozenset({python_binary()}), timeout=20.0),
                    workspace,
                )
            )
            spec = SandboxSpec(image="process://local", workspace=workspace)
            python = python_binary()
            workspace_file = "e5.txt"

        handle = backend.create(spec)
        try:
            write = backend.exec(
                handle,
                [python, "-c", f"open({workspace_file!r}, 'w').write('ok')"],
                timeout=20.0,
            )
            if container:
                root = backend.exec(
                    handle,
                    [python, "-c", "open('/etc/passwd', 'a').write('x')"],
                    timeout=20.0,
                )
                metadata = backend.exec(
                    handle,
                    [
                        python,
                        "-c",
                        "import socket; socket.create_connection(('169.254.169.254', 80), 2)",
                    ],
                    timeout=20.0,
                )
            else:
                root = None
                metadata = None
            env = backend.exec(
                handle,
                [python, "-c", "import os; print('|'.join(sorted(os.environ)))"],
                timeout=20.0,
            )
            backend.put(handle, "put.txt", b"roundtrip")
            roundtrip = backend.get(handle, "put.txt")
            snapshot = backend.snapshot(handle, "checkpoint-1")
        finally:
            backend.destroy(handle)
            token.revoke()

        print("backend:", "container" if container else "process (degraded)")
        print("workspace write:", write.ok, write.stderr.strip()[:120])
        if root is None:
            print("root write blocked: not enforced in process mode")
            print("metadata blocked: not enforced in process mode")
        else:
            print("root write blocked:", not root.ok, root.stderr.strip()[:120])
            print("metadata blocked:", not metadata.ok, metadata.stderr.strip()[:120])
        print("env keys:", env.stdout.strip())
        print("put/get roundtrip:", roundtrip.decode())
        print("snapshot:", snapshot.path.name, snapshot.path.stat().st_size, "bytes")
        print("token revoked:", token.revoked)
        snapshot.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()