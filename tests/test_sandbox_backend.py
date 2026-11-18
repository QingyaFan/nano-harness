import os
import shutil
import subprocess

import pytest

from nano_harness.policy import (
    ContainerBackend,
    ProcessBackend,
    SandboxClosed,
    SandboxHandle,
    SandboxPolicy,
    SandboxSpec,
    SandboxSpecError,
    SubprocessSandbox,
)

DIGEST = "registry.example.com/agent-runtime@sha256:" + "a" * 64


def python_binary() -> str:
    return "python3" if shutil.which("python3") else "python"


def make_spec(tmp_path, **overrides) -> SandboxSpec:
    values = {
        "image": DIGEST,
        "workspace": tmp_path,
        "network": "none",
        "user": "10001:10001",
        "pids": 64,
        "memory": "256m",
        "cpus": 0.5,
        "env": {"PROXY_TOKEN": "token"},
    }
    values.update(overrides)
    return SandboxSpec(**values)


def test_spec_maps_to_container_flags(tmp_path) -> None:
    backend = ContainerBackend()
    args = backend.build_run_args(make_spec(tmp_path), "nh-test")

    assert args[:4] == ["run", "-d", "--name", "nh-test"]
    assert "--read-only" in args
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--user") + 1] == "10001:10001"
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert args[args.index("--pids-limit") + 1] == "64"
    assert args[args.index("--memory") + 1] == "256m"
    assert args[args.index("--cpus") + 1] == "0.5"
    assert args.count("-v") == 1
    assert args[args.index("-v") + 1] == f"{tmp_path}:/workspace:rw"
    assert "docker.sock" not in " ".join(args)
    assert args.count("--env") == 1
    assert "PROXY_TOKEN=token" in args
    assert "OPENAI_API_KEY" not in " ".join(args)
    assert args[-3:] == [DIGEST, "sleep", "infinity"]


def test_workspace_and_mount_are_validated(tmp_path) -> None:
    with pytest.raises(SandboxSpecError, match="absolute"):
        SandboxSpec(image=DIGEST, workspace="relative/workspace")

    with pytest.raises(SandboxSpecError, match="container path"):
        make_spec(tmp_path, workspace_mount="/")


def test_unpinned_image_is_rejected(tmp_path) -> None:
    backend = ContainerBackend(require_digest=True)

    with pytest.raises(SandboxSpecError, match="digest"):
        backend.build_run_args(
            make_spec(tmp_path, image="python:3.11-slim"),
            "nh-test",
        )


def test_env_outside_allowlist_is_rejected(tmp_path) -> None:
    backend = ContainerBackend(env_allowlist=frozenset({"PROXY_TOKEN"}))

    with pytest.raises(SandboxSpecError, match="OPENAI_API_KEY"):
        backend.build_run_args(
            make_spec(tmp_path, env={"OPENAI_API_KEY": "secret"}),
            "nh-test",
        )


def test_exec_rejects_shell_strings(tmp_path) -> None:
    backend = ContainerBackend()
    handle = SandboxHandle("nh-test", tmp_path, "/workspace")

    with pytest.raises(SandboxSpecError, match="shell"):
        backend.exec(handle, "echo hi")


def test_process_backend_roundtrip_and_close(tmp_path) -> None:
    python = python_binary()
    sandbox = SubprocessSandbox(
        SandboxPolicy(frozenset({python}), timeout=20.0),
        tmp_path,
    )
    backend = ProcessBackend(sandbox)
    handle = backend.create(make_spec(tmp_path, image="process://local"))

    try:
        backend.put(handle, "a.txt", b"hello")
        assert backend.get(handle, "a.txt") == b"hello"

        result = backend.exec(
            handle,
            [python, "-c", "print(open('a.txt').read())"],
        )
        assert result.ok
        assert "hello" in result.stdout

        snapshot = backend.snapshot(handle, "checkpoint-1")
        assert snapshot.path.exists()
        assert snapshot.path.stat().st_size > 0

        with pytest.raises(SandboxSpecError):
            backend.put(handle, "../escape.txt", b"x")
    finally:
        backend.destroy(handle)

    with pytest.raises(SandboxClosed):
        backend.exec(handle, [python, "-c", "print(1)"])


def test_process_backend_honors_per_call_timeout(tmp_path) -> None:
    python = python_binary()
    backend = ProcessBackend(
        SubprocessSandbox(
            SandboxPolicy(frozenset({python}), timeout=20.0),
            tmp_path,
        )
    )
    handle = backend.create(make_spec(tmp_path, image="process://local"))

    try:
        with pytest.raises(TimeoutError):
            backend.exec(
                handle,
                [python, "-c", "import time; time.sleep(2)"],
                timeout=0.1,
            )
    finally:
        backend.destroy(handle)


@pytest.mark.skipif(
    os.environ.get("NANO_HARNESS_DOCKER") != "1",
    reason="set NANO_HARNESS_DOCKER=1 to run container integration tests",
)
def test_container_blocks_metadata_and_root_writes(tmp_path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not available")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        pytest.skip("docker daemon is not available")

    tmp_path.chmod(0o777)
    backend = ContainerBackend(require_digest=False)
    handle = backend.create(
        SandboxSpec(
            image=os.environ.get("NANO_HARNESS_IMAGE", "python:3.11-slim"),
            workspace=tmp_path,
            network="none",
            pids=64,
            memory="256m",
            env={"PROXY_TOKEN": "token"},
        )
    )

    try:
        python = "python"
        root = backend.exec(
            handle,
            [python, "-c", "open('/etc/passwd', 'a').write('x')"],
        )
        metadata = backend.exec(
            handle,
            [
                python,
                "-c",
                "import socket; socket.create_connection(('169.254.169.254', 80), 2)",
            ],
        )
        workspace = backend.exec(
            handle,
            [python, "-c", "open('/workspace/ok.txt', 'w').write('ok')"],
        )

        assert not root.ok
        assert not metadata.ok
        assert workspace.ok
    finally:
        backend.destroy(handle)