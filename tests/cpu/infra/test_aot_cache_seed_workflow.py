"""Contract checks for .github/workflows/aot-cache-seed.yml.

The seed only helps if it writes the key prefixes the PR AOT lanes read, so it
must run the same build script on the same lanes, with write credentials, and
must never cancel a running seed (cancelled seeds were the failure mode of the
per-merge PR Test runs that #6968 removed).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AOT_SCRIPT = "task_test_jit_cache_package_build_import.sh"


def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )


def _triggers(workflow: dict) -> dict:
    # PyYAML parses the bare `on:` key as boolean True.
    return workflow.get("on", workflow.get(True))


def test_seed_runs_after_merges_to_main_and_never_cancels():
    workflow = _workflow("aot-cache-seed.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert workflow["concurrency"] == {
        "group": "aot-cache-seed",
        "cancel-in-progress": False,
    }


def _github_path_regex(pattern: str) -> "re.Pattern[str]":
    """GitHub path filter syntax: `**` crosses directories, `*` does not."""
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


@pytest.mark.parametrize(
    "path",
    [
        "include/flashinfer/utils.cuh",
        "csrc/page.cu",
        "3rdparty/cutlass",
        "flashinfer/jit/fused_moe.py",
        "flashinfer/aot.py",
        "flashinfer/artifacts.py",
        "flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe/shim/cake_jit.py",
        "ci/docker-tags.yml",
        "ci/cuda-versions.json",
        f"scripts/{AOT_SCRIPT}",
        "scripts/jit_cache_build_common.sh",
    ],
)
def test_seed_triggers_on_compile_inputs(path):
    patterns = _triggers(_workflow("aot-cache-seed.yml"))["push"]["paths"]
    assert any(_github_path_regex(p).match(path) for p in patterns), path


@pytest.mark.parametrize("path", ["README.md", "docs/index.rst", "tests/test_x.py"])
def test_seed_ignores_unrelated_changes(path):
    patterns = _triggers(_workflow("aot-cache-seed.yml"))["push"]["paths"]
    assert not any(_github_path_regex(p).match(path) for p in patterns), path


def test_seed_matches_pr_aot_lanes():
    seed_jobs = _workflow("aot-cache-seed.yml")["jobs"]
    pr_test = (REPO_ROOT / ".github" / "workflows" / "pr-test.yml").read_text(
        encoding="utf-8"
    )

    # Same runner pool and lane set as the PR AOT matrix.
    assert seed_jobs["seed"]["runs-on"] == [
        "self-hosted",
        "linux",
        "${{ matrix.arch }}",
        "cpu",
        "on-demand",
    ]
    assert '["self-hosted", "linux", $arch, "cpu", "on-demand"]' in pr_test
    matrix_step = next(
        s for s in seed_jobs["setup"]["steps"] if s.get("id") == "matrix"
    )
    assert "for arch in x64 arm64" in matrix_step["run"]
    assert ".runtime[].label" in matrix_step["run"]
    assert "ci/docker-tags.yml" in matrix_step["run"]
    # One lane failing must not cancel the others.
    assert seed_jobs["seed"]["strategy"]["fail-fast"] is False


def test_seed_builds_with_write_credentials_and_skips_lint():
    steps = _workflow("aot-cache-seed.yml")["jobs"]["seed"]["steps"]
    seed = next(s for s in steps if s.get("name") == "Seed AOT sccache")

    assert (
        seed["env"]["AWS_ACCESS_KEY_ID"] == "${{ secrets.SCCACHE_AWS_ACCESS_KEY_ID }}"
    )
    assert (
        seed["env"]["AWS_SECRET_ACCESS_KEY"]
        == "${{ secrets.SCCACHE_AWS_SECRET_ACCESS_KEY }}"
    )
    assert f"./scripts/{AOT_SCRIPT}" in seed["run"]
    assert "-e AOT_SKIP_SPEC_LINT 1" in seed["run"]
    # A missing secret must fail loudly instead of seeding nothing.
    assert "credentials are missing" in seed["run"]


def test_seed_only_runs_in_the_upstream_repository():
    setup = _workflow("aot-cache-seed.yml")["jobs"]["setup"]
    assert setup["if"] == "github.repository == 'flashinfer-ai/flashinfer'"


@pytest.mark.skipif(sys.platform == "win32", reason="runs a bash fragment")
@pytest.mark.parametrize("skip, linted", [("", True), ("0", True), ("1", False)])
def test_aot_script_lint_switch(skip, linted):
    """AOT_SKIP_SPEC_LINT=1 skips the lint but still prefetches its headers."""
    text = (REPO_ROOT / "scripts" / AOT_SCRIPT).read_text(encoding="utf-8")
    start = text.index('if [ "${AOT_SKIP_SPEC_LINT:-0}" = "1" ]; then')
    end = text.index("\nfi\n", start) + len("\nfi\n")
    fragment = (
        'run_provider_spec_lint() { echo "LINT $*"; }\n'
        'python3() { echo "PYTHON $*"; }\n'
        "SCRIPT_DIR=/scripts\n"
        "PROVIDER_ARCHITECTURE_LIST=(8.0 9.0a)\n" + text[start:end]
    )
    env = {"PATH": "/usr/bin:/bin"}
    if skip:
        env["AOT_SKIP_SPEC_LINT"] = skip
    result = subprocess.run(
        ["bash", "-c", fragment], capture_output=True, text=True, env=env, check=True
    )
    assert ("LINT 3 8.0 9.0a" in result.stdout) is linted
    # The lint prefetches on its own; the skip path must do it explicitly.
    prefetched = "PYTHON /scripts/prefetch_jit_spec_headers.py" in result.stdout
    assert prefetched is not linted
