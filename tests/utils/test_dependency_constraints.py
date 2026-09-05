"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# apache-tvm-ffi is a header-only ABI dependency: make_object, Optional<T> and the
# ModuleObj vtable are inlined into our .so files, so the version resolved at build
# time is the ABI they require at run time. It was previously spelled three different
# ways across seven files; this test keeps the one constraint in sync and keeps the
# release pin inside it.

import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - CI runs 3.12
    tomllib = pytest.importorskip("tomli", reason="TOML parsing needs Python 3.11+")

REPO_ROOT = Path(__file__).resolve().parents[2]

CONSTRAINT = "apache-tvm-ffi>=0.1.10,<0.2"
# Release builds narrow the range to one version so a source tree cannot produce
# different ABIs on different days. Consumers keep the range.
RELEASE_PIN_FILE = "scripts/build_constraints.txt"

TOML_SITES = [
    "pyproject.toml",
    "flashinfer-jit-cache/pyproject.toml",
    "flashinfer-cubin/pyproject.toml",
]
SCRIPT_SITES = [
    "docker/Dockerfile.flashinfer-nvep",
    "docker/install/build_flashinfer_ep_pytorch.sh",
    "scripts/build_in_container.sh",
]
# Files naming tvm-ffi without declaring a constraint: prose, or the deliberate
# TVM_FFI_REF escape hatch that installs an arbitrary ref for testing tvm-ffi PRs.
UNCONSTRAINED_SITES = {
    "CLAUDE.md",
    "flashinfer/collect_env.py",
    "scripts/setup_test_env.sh",
    # Applies RELEASE_PIN_FILE rather than declaring a version itself.
    "scripts/build_flashinfer_jit_cache_whl.sh",
    # Reads CONSTRAINT back out of requirements.txt to check a built wheel against it.
    "scripts/verify_jit_cache_abi.py",
    RELEASE_PIN_FILE,
}


def _toml(rel: str) -> dict:
    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.fail(f"{rel} is missing from the checkout")
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("rel", TOML_SITES)
def test_build_requires_declare_constraint(rel):
    requires = _toml(rel).get("build-system", {}).get("requires", [])
    assert CONSTRAINT in requires, (
        f"{rel} [build-system].requires must contain {CONSTRAINT!r}, got {requires!r}"
    )


@pytest.mark.parametrize("rel", SCRIPT_SITES)
def test_no_isolation_envs_declare_constraint(rel):
    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.fail(f"{rel} is missing from the checkout")
    installs = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "apache-tvm-ffi" in line and not line.lstrip().startswith("#")
    ]
    assert installs, f"{rel} no longer installs apache-tvm-ffi; update SCRIPT_SITES"
    assert all(CONSTRAINT in line for line in installs), (
        f"{rel} provisions a --no-build-isolation env, where this line *is* the "
        f"compiled ABI; it must install {CONSTRAINT!r}, found {installs!r}"
    )


def test_runtime_metadata_declares_constraint():
    reqs = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert CONSTRAINT in reqs, "requirements.txt must declare " + CONSTRAINT
    # The wheel that actually ships the ABI-bound .so files must state what it needs;
    # an empty list let pip pair the prebuilt kernels with any apache-tvm-ffi.
    deps = _toml("flashinfer-jit-cache/pyproject.toml")["project"]["dependencies"]
    assert CONSTRAINT in deps, (
        f"flashinfer-jit-cache [project].dependencies must contain {CONSTRAINT!r}, "
        f"got {deps!r}"
    )


def test_release_pin_is_inside_the_declared_constraint():
    lines = [
        line.strip()
        for line in (REPO_ROOT / RELEASE_PIN_FILE).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pins = [line for line in lines if line.startswith("apache-tvm-ffi==")]
    assert len(pins) == 1, f"{RELEASE_PIN_FILE} must pin exactly one version: {lines!r}"
    version = pins[0].split("==", 1)[1]
    spec = SpecifierSet(CONSTRAINT.split("apache-tvm-ffi", 1)[1])
    assert spec.contains(version), (
        f"release builds pin {version}, which the declared constraint excludes — the "
        f"published wheel would be incompatible with its own metadata"
    )


def test_no_unlisted_tvm_ffi_references():
    """A new site naming tvm-ffi must be classified, not silently added."""
    try:
        grep = subprocess.run(
            # Both the PyPI name and the VCS URL form used by overrides.
            ["git", "grep", "-lE", "apache[-/]tvm-ffi", "--", ".", ":!3rdparty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable (not a source checkout)")
    if grep.returncode > 1:  # 0 == matches, 1 == none, higher is a real error
        pytest.fail(f"git grep failed ({grep.returncode}): {grep.stderr.strip()}")

    known = (
        set(TOML_SITES)
        | set(SCRIPT_SITES)
        | UNCONSTRAINED_SITES
        | {"requirements.txt", str(Path(__file__).resolve().relative_to(REPO_ROOT))}
    )
    offenders = sorted(
        rel
        for rel in grep.stdout.splitlines()
        if rel and rel not in known and not rel.startswith("docs/")
    )
    assert not offenders, (
        f"these files reference apache-tvm-ffi but are unclassified: {offenders}. "
        f"Add them to TOML_SITES / SCRIPT_SITES, or UNCONSTRAINED_SITES with a reason."
    )
