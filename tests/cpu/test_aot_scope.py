"""Contract tests for the PR AOT gate (scripts/pr_checks/aot_scope.py)."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def aot_scope():
    path = REPO / "scripts" / "pr_checks" / "aot_scope.py"
    spec = importlib.util.spec_from_file_location("aot_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def workflow():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((REPO / ".github/workflows/pr-test.yml").read_text())


@pytest.fixture(scope="module")
def skip_ci_patterns(workflow):
    return workflow["env"]["SKIP_CI_PATTERNS"]


def _tree(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _run(aot_scope, root, *paths, status="M", patterns=""):
    return aot_scope.decide([(status, p) for p in paths], root, patterns)[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "path",
    [
        "csrc/foo.cu",
        "include/flashinfer/x.cuh",
        "3rdparty/cutlass",
        "flashinfer/aot.py",
        "flashinfer/jit/core.py",
        "flashinfer/__init__.py",
        "flashinfer/data/some.json",
        "build_backend.py",
        "pyproject.toml",
        "version.txt",
        "requirements.txt",
        "ci/docker-tags.yml",
        "flashinfer-jit-cache/setup.py",
        "flashinfer-jit-cache-provider/package_config.py",
        "scripts/task_test_jit_cache_package_build_import.sh",
        "scripts/jit_cache_build_common.sh",
        "scripts/verify_all_modules_compiled.py",
        ".github/workflows/pr-test.yml",
        ".github/actions/setup/action.yml",
        "some_new_top_level_file.txt",
    ],
)
def test_paths_that_can_change_the_providers_run_aot(aot_scope, skip_ci_patterns, path):
    assert _run(aot_scope, REPO, path, patterns=skip_ci_patterns)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "path",
    [
        "tests/attention/test_x.py",
        "docs/api/x.rst",
        "benchmarks/bench_x.py",
        "README.md",
        "flashinfer/anything/README.md",
        "docker/Dockerfile.cu129",
        "licenses/LICENSE.cutlass.txt",
        ".github/CODEOWNERS",
        ".github/workflows/nightly-release.yml",
        "scripts/pr_checks/aot_scope.py",
        "scripts/task_jit_run_tests_part1.sh",
    ],
)
def test_unrelated_paths_skip_aot(aot_scope, skip_ci_patterns, path):
    assert not _run(aot_scope, REPO, path, patterns=skip_ci_patterns)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "path",
    ["docs/x.rst", "README.md", "docker/Dockerfile.cu129", "benchmarks/b.py"],
)
def test_whole_run_skips_are_aot_skips(aot_scope, skip_ci_patterns, path):
    # A PR that skips all of CI (skip_build) must also skip AOT: the gate reads
    # the same SKIP_CI_PATTERNS instead of keeping its own copy.
    assert re.search(skip_ci_patterns, path)
    assert not _run(aot_scope, REPO, path, patterns=skip_ci_patterns)


@pytest.mark.cpu_only
def test_without_shared_patterns_the_gate_only_gets_stricter(aot_scope):
    # Unset SKIP_CI_PATTERNS (a local run) can only make the gate run AOT.
    assert _run(aot_scope, REPO, "docs/x.rst")
    assert not _run(aot_scope, REPO, "tests/x.py")


@pytest.mark.cpu_only
def test_one_relevant_file_among_ignorable_ones_runs_aot(aot_scope, skip_ci_patterns):
    assert _run(
        aot_scope,
        REPO,
        "tests/x.py",
        "docs/x.md",
        "csrc/x.cu",
        patterns=skip_ci_patterns,
    )


@pytest.mark.cpu_only
def test_every_jit_module_and_aot_import_is_in_the_closure(aot_scope):
    closure = aot_scope.aot_closure(REPO)
    assert "flashinfer/aot.py" in closure
    text = (REPO / "flashinfer" / "aot.py").read_text()
    for module in re.findall(r"^from \.(jit\.[\w.]+) import", text, re.M):
        rel = "flashinfer/" + module.replace(".", "/")
        assert rel + ".py" in closure or rel + "/__init__.py" in closure, module


@pytest.mark.cpu_only
def test_closure_follows_lazy_and_relative_imports(aot_scope, tmp_path):
    root = _tree(
        tmp_path,
        {
            "flashinfer/__init__.py": "",
            "flashinfer/aot.py": "from .jit import gen\n",
            "flashinfer/jit/__init__.py": "",
            "flashinfer/jit/gen.py": "def f():\n    from ..helpers import h\n",
            "flashinfer/helpers.py": "from flashinfer import deep\n",
            "flashinfer/deep.py": "",
            "flashinfer/unrelated.py": "",
        },
    )
    closure = aot_scope.aot_closure(root)
    assert {"flashinfer/helpers.py", "flashinfer/deep.py"} <= closure
    assert "flashinfer/unrelated.py" not in closure
    assert _run(aot_scope, root, "flashinfer/deep.py")
    assert not _run(aot_scope, root, "flashinfer/unrelated.py")


@pytest.mark.cpu_only
def test_lazy_package_loader_pulls_in_its_subtree(aot_scope, tmp_path):
    root = _tree(
        tmp_path,
        {
            "flashinfer/__init__.py": "",
            "flashinfer/aot.py": "from . import lazy\n",
            "flashinfer/lazy/__init__.py": (
                "import importlib\n"
                "def __getattr__(n):\n    return importlib.import_module('.' + n, __name__)\n"
            ),
            "flashinfer/lazy/sub/deep.py": "",
        },
    )
    assert _run(aot_scope, root, "flashinfer/lazy/sub/deep.py")


@pytest.mark.cpu_only
def test_deleted_python_module_runs_aot_even_when_unreferenced(aot_scope, tmp_path):
    root = _tree(
        tmp_path,
        {"flashinfer/__init__.py": "", "flashinfer/aot.py": ""},
    )
    assert _run(aot_scope, root, "flashinfer/gone.py", status="D")
    assert not _run(aot_scope, root, "flashinfer/new.py", status="A")


@pytest.mark.cpu_only
def test_unreadable_closure_falls_back_to_running_aot(aot_scope, tmp_path, capsys):
    # No flashinfer/aot.py in the tree: the closure cannot be built.
    rc = aot_scope.main(["--base", "HEAD", "--head", "HEAD", "--root", str(tmp_path)])
    assert rc == 0
    assert "run=true" in capsys.readouterr().out


def _step(workflow, name):
    steps = workflow["jobs"]["setup"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def _run_gate_step(workflow, tmp_path, event, script_output, script_status=0):
    """Run the workflow's gate step with a stub aot_scope.py."""
    stub = tmp_path / "scripts" / "pr_checks" / "aot_scope.py"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        f"import sys\nprint({script_output!r}, end='')\nsys.exit({script_status})\n"
    )
    output = tmp_path / "github_output"
    env = dict(
        os.environ,
        EVENT_NAME=event,
        BASE_SHA="base",
        HEAD_SHA="head",
        GITHUB_OUTPUT=str(output),
    )
    script = _step(workflow, "Detect AOT-affecting changes")["run"]
    subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env, check=True)
    return dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )["run_aot"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("event", "script_output", "script_status", "expected"),
    [
        ("pull_request", "run=false\nreason=docs only\n", 0, "false"),
        ("pull_request", "run=true\nreason=csrc\n", 0, "true"),
        # A crashing or garbled gate must not drop the lanes.
        ("pull_request", "", 1, "true"),
        ("pull_request", "run=maybe\n", 0, "true"),
        # Scheduled and manual runs never consult the gate.
        ("schedule", "run=false\n", 0, "true"),
        ("workflow_dispatch", "run=false\n", 0, "true"),
    ],
)
def test_gate_step_output(
    workflow, tmp_path, event, script_output, script_status, expected
):
    assert (
        _run_gate_step(workflow, tmp_path, event, script_output, script_status)
        == expected
    )


@pytest.mark.cpu_only
def test_matrix_step_consumes_the_gate(workflow):
    step = _step(workflow, "Build Test Matrix")
    assert step["env"]["RUN_AOT_SCOPE"] == "${{ steps.aot-scope.outputs.run_aot }}"
    # Only an explicit "false" drops the AOT entries.
    assert '"$RUN_AOT_SCOPE" != "false"' in step["run"]
