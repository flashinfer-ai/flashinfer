import pytest

import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
ARTIFACT_SCRIPT = REPO_ROOT / "scripts" / "build_flashinfer_jit_cache_artifact.sh"


def _write_executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _run_artifact_script(tmp_path, *, inspect_exit=0, pull_failures=0, run_exit=0):
    test_repo = tmp_path / "repo"
    script_dir = test_repo / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / ARTIFACT_SCRIPT.name
    shutil.copyfile(ARTIFACT_SCRIPT, script)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.jsonl"
    pull_count = tmp_path / "pull-count"
    sleep_log = tmp_path / "sleep.log"

    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["DOCKER_LOG"]).open("a") as log:
    log.write(json.dumps(args) + "\\n")

if args[:2] == ["image", "inspect"]:
    raise SystemExit(int(os.environ["DOCKER_INSPECT_EXIT"]))
if args[:1] == ["pull"]:
    state = Path(os.environ["DOCKER_PULL_COUNT"])
    count = int(state.read_text()) + 1 if state.exists() else 1
    state.write_text(str(count))
    raise SystemExit(count <= int(os.environ["DOCKER_PULL_FAILURES"]))
if args[:1] == ["run"]:
    raise SystemExit(int(os.environ["DOCKER_RUN_EXIT"]))
raise SystemExit(2)
""",
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

with Path(os.environ["SLEEP_LOG"]).open("a") as log:
    log.write(sys.argv[1] + "\\n")
""",
    )

    env = {
        **os.environ,
        "ARCH": "x86_64",
        "CUDA_VERSION": "13.0",
        "DOCKER_IMAGE": "example.invalid/jit-cache-builder:cu130",
        "FLASHINFER_CI_CACHE": str(tmp_path / "ci-cache"),
        "FLASHINFER_JIT_CACHE_BUILD_TARGET": "shim",
        "FLASHINFER_LOCAL_VERSION": "cu130",
        "PYTORCH_INDEX": "cu130",
        "DOCKER_LOG": str(docker_log),
        "DOCKER_INSPECT_EXIT": str(inspect_exit),
        "DOCKER_PULL_COUNT": str(pull_count),
        "DOCKER_PULL_FAILURES": str(pull_failures),
        "DOCKER_RUN_EXIT": str(run_exit),
        "SLEEP_LOG": str(sleep_log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    docker_calls = [json.loads(line) for line in docker_log.read_text().splitlines()]
    sleeps = sleep_log.read_text().splitlines() if sleep_log.exists() else []
    return result, docker_calls, sleeps


def _calls_for(docker_calls, command):
    return [args for args in docker_calls if args[0] == command]


@pytest.mark.cpu_only
def test_cached_image_skips_pull_and_disables_implicit_pull(tmp_path):
    result, docker_calls, sleeps = _run_artifact_script(tmp_path)

    assert result.returncode == 0
    assert not _calls_for(docker_calls, "pull")
    run_calls = _calls_for(docker_calls, "run")
    assert len(run_calls) == 1
    assert "--pull=never" in run_calls[0]
    assert sleeps == []


@pytest.mark.cpu_only
def test_image_pull_recovers_with_exponential_backoff(tmp_path):
    result, docker_calls, sleeps = _run_artifact_script(
        tmp_path, inspect_exit=1, pull_failures=3
    )

    assert result.returncode == 0
    assert len(_calls_for(docker_calls, "pull")) == 4
    assert len(_calls_for(docker_calls, "run")) == 1
    assert sleeps == ["5", "10", "20"]


@pytest.mark.cpu_only
def test_image_pull_exhaustion_does_not_start_build(tmp_path):
    result, docker_calls, sleeps = _run_artifact_script(
        tmp_path, inspect_exit=1, pull_failures=4
    )

    assert result.returncode == 1
    assert len(_calls_for(docker_calls, "pull")) == 4
    assert not _calls_for(docker_calls, "run")
    assert sleeps == ["5", "10", "20"]
    assert "Failed to pull" in result.stderr


@pytest.mark.cpu_only
def test_build_failure_is_propagated_without_retry(tmp_path):
    result, docker_calls, sleeps = _run_artifact_script(tmp_path, run_exit=42)

    assert result.returncode == 42
    assert not _calls_for(docker_calls, "pull")
    assert len(_calls_for(docker_calls, "run")) == 1
    assert sleeps == []
