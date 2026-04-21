from __future__ import annotations

import shutil
import subprocess

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_dockerfile_builds_if_docker_is_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is not available in this environment")

    version = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0:
        pytest.skip("docker daemon is not available in this environment")

    result = subprocess.run(
        ["docker", "build", "-f", "server/Dockerfile", "-t", "biomed-test", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
