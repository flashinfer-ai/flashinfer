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

Vendored copies of the Cake RMSNorm training package with the
non-deterministic weight gradient (``flashinfer/cake_rmsnorm_train_backend``)
and its deterministic sibling (``flashinfer/cake_rmsnorm_det_train_backend``),
imported with flashinfer blocked (issues #6553, #7047, #8066).

``deterministic=True`` (the default) runs the deterministic package, which this
package imports lazily as its sibling (``from .. import``); ``deterministic=False``
needs nothing outside this package.  So a consumer vendors the two packages
under one parent package to keep the default, or this package alone for
``deterministic=False`` only.

* Device-free (``cpu_only``): both forms are copied (without ``aot/`` and
  ``__pycache__/``) and imported in a fresh ``python -I`` interpreter in which
  every ``flashinfer`` import raises: the pair under one parent (the
  deterministic default reaches the vendored sibling), the lone copy (it
  imports; a deterministic call raises ``RuntimeError`` naming the missing
  package; a ``deterministic=False`` call reaches this package's own checks),
  the lone deterministic copy; no ``flashinfer`` module is loaded.
* Device (opt-in, ``FLASHINFER_CAKE_VENDORED_SUITE=1``: it compiles both
  manifests with plain ``nvcc``, minutes): ``scripts/check_cake_build_manifests.py
  --vendored-gate`` builds both copies for the visible GPUs' architecture and
  checks that they load prebuilt with flashinfer blocked; the two copies are then
  put under one parent and the backend suites run against them in a ``python -I``
  child with flashinfer blocked (the ``cake_package`` table pointed at the
  copies); the suites that do not need the deterministic package run against the
  lone copy (with no deterministic package importable at all).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NONDET = "cake_rmsnorm_train_backend"
DET = "cake_rmsnorm_det_train_backend"
PARENT = "vendored_cake_rmsnorm"
LONE = f"vendored_{NONDET}"
LONE_DET = f"vendored_{DET}"
# The backend suites run against the vendored pair, and the ones that do not
# import the deterministic package, run against the lone copy.
PAIR_SUITES = (
    "tests/norm/test_cake_rmsnorm_train.py",
    "tests/norm/test_cake_rmsnorm_train_pairs.py",
    "tests/norm/test_cake_rmsnorm_train_ckpt.py",
    "tests/norm/test_cake_rmsnorm_train_windows.py",
)
LONE_SUITES = (
    "tests/norm/test_cake_rmsnorm_train_pairs.py",
    "tests/norm/test_cake_rmsnorm_train_ckpt.py",
    "tests/norm/test_cake_rmsnorm_train_windows.py",
)
SUITE_TIMEOUT_S = 5400

# The blocker every child installs (the vendored gate's): any flashinfer import raises.
BLOCKER = """\
import importlib.abc
import sys


class _BlockFlashinfer(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError("flashinfer is blocked by the vendored test: " + name)
        return None


sys.meta_path.insert(0, _BlockFlashinfer())
sys.modules["flashinfer"] = None
"""


def _dependency_paths() -> list[str]:
    """The interpreter's dependency directories (site-packages and the parents of
    torch and tvm_ffi): a ``-I`` child does not read ``PYTHONPATH``."""
    paths = [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
    for module in ("torch", "tvm_ffi"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            pytest.skip(f"{module} is not importable here")
        paths.append(str(Path(spec.origin).resolve().parent.parent))
    return list(dict.fromkeys(paths))


def _copy(source: Path, target: Path, *, with_aot: bool = False) -> None:
    ignore = ("__pycache__",) if with_aot else ("aot", "__pycache__")
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns(*ignore), symlinks=False
    )


def _child(
    script: str, paths: list[str], *, cwd: Path, env: dict | None = None, timeout=600
):
    code = (
        "import sys\n"
        f"for _p in reversed({paths!r}):\n"
        "    sys.path.insert(0, _p)\n" + BLOCKER + script
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- device-free copy import

_IMPORT_CHECKS = """
import importlib
import json
import os

import torch

results = {}


def check(name, fn):
    try:
        fn()
        results[name] = "ok"
    except Exception as exc:  # noqa: BLE001
        results[name] = f"FAIL {type(exc).__name__}: {exc}"


def raises(error, call, *words):
    try:
        call()
    except error as exc:
        text = str(exc)
        missing = [w for w in words if w not in text]
        assert not missing, f"{type(exc).__name__} without {missing}: {text[:300]}"
        return
    raise AssertionError(f"no {error.__name__}")


def leaked():
    return sorted(
        k for k in sys.modules
        if (k == "flashinfer" and sys.modules[k] is not None) or k.startswith("flashinfer.")
    )


x = torch.zeros(4, 512, dtype=torch.bfloat16)
w = torch.ones(512, dtype=torch.bfloat16)
"""

_PAIR_CHECKS = (
    _IMPORT_CHECKS
    + f"""
nd = importlib.import_module("{PARENT}.{NONDET}")
check("pair: the copy is imported", lambda: None if os.path.realpath(nd.__file__).startswith(COPY) else 1 / 0)
check(
    "pair: deterministic=True reaches the vendored sibling",
    lambda: None if nd._det_package().__name__ == "{PARENT}.{DET}" else 1 / 0,
)
check("pair: registry", lambda: None if nd.cake_jit.ARCHES == ("sm_100a", "sm_103a", "sm_107a") else 1 / 0)
check("pair: manifest", lambda: None if nd.cake_build.manifest()["schema"] == "cake-build/1" else 1 / 0)
check(
    "pair: program windows",
    lambda: None if nd.cake_jit.program_windows("sm_100a", 512, "bwd", False, "bf16") else 1 / 0,
)
check(
    "pair: no device -> unsupported (both modes)",
    lambda: None
    if not nd.is_cake_rmsnorm_train_supported(torch.device("cpu"), deterministic=False)
    and not nd.is_cake_rmsnorm_train_supported(torch.device("cpu"))
    else 1 / 0,
)
check(
    "pair: the default call runs the vendored det package (its CUDA check)",
    lambda: raises(ValueError, lambda: nd.cake_rmsnorm_train_forward(x, w, 1e-6), "CUDA"),
)
check(
    "pair: deterministic=False runs this package (its CUDA check)",
    lambda: raises(
        ValueError, lambda: nd.cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=False), "CUDA"
    ),
)
check("pair: no flashinfer module", lambda: None if not leaked() else 1 / 0)
results["leaked"] = leaked()
print("RESULTS " + json.dumps(results))
"""
)

_LONE_CHECKS = (
    _IMPORT_CHECKS
    + f"""
lone = importlib.import_module("{LONE}")
check("lone: the copy is imported", lambda: None if os.path.realpath(lone.__file__).startswith(COPY) else 1 / 0)
check(
    "lone: _det_package names the missing package",
    lambda: raises(RuntimeError, lone._det_package, "cake_rmsnorm_det_train_backend", "deterministic=False"),
)
check(
    "lone: the default call raises naming the missing package",
    lambda: raises(
        RuntimeError, lambda: lone.cake_rmsnorm_train_forward(x, w, 1e-6), "cake_rmsnorm_det_train_backend"
    ),
)
check(
    "lone: the default support query raises naming the missing package",
    lambda: raises(
        RuntimeError,
        lambda: lone.is_cake_rmsnorm_train_supported(torch.device("cpu")),
        "cake_rmsnorm_det_train_backend",
    ),
)
check(
    "lone: deterministic=False runs this package (its CUDA check)",
    lambda: raises(
        ValueError, lambda: lone.cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=False), "CUDA"
    ),
)
check(
    "lone: deterministic=False support query",
    lambda: None if not lone.is_cake_rmsnorm_train_supported(torch.device("cpu"), deterministic=False) else 1 / 0,
)
det = importlib.import_module("{LONE_DET}")
check(
    "lone det copy: imports and answers",
    lambda: None if not det.is_cake_rmsnorm_det_train_supported(torch.device("cpu")) else 1 / 0,
)
check("lone: no flashinfer module", lambda: None if not leaked() else 1 / 0)
results["leaked"] = leaked()
print("RESULTS " + json.dumps(results))
"""
)


def _results(proc) -> dict:
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULTS ")]
    assert proc.returncode == 0 and lines, (
        f"child exited {proc.returncode}: {proc.stdout[-2000:]} {proc.stderr[-3000:]}"
    )
    return json.loads(lines[-1][len("RESULTS ") :])


@pytest.mark.cpu_only
def test_vendored_copies_import_with_flashinfer_blocked(tmp_path: Path) -> None:
    paths = _dependency_paths()
    source = REPO_ROOT / "flashinfer"
    pair = tmp_path / "pair"
    _copy(source / NONDET, pair / PARENT / NONDET)
    _copy(source / DET, pair / PARENT / DET)
    (pair / PARENT / "__init__.py").write_text("")
    lone = tmp_path / "lone"
    _copy(source / NONDET, lone / LONE)
    _copy(source / DET, lone / LONE_DET)
    report = {}
    for _label, root, script in (
        ("pair", pair, _PAIR_CHECKS),
        ("lone", lone, _LONE_CHECKS),
    ):
        proc = _child(
            f"COPY = {os.path.realpath(root)!r}\n" + script,
            [str(root), *paths],
            cwd=tmp_path,
        )
        report.update(_results(proc))
    failed = {k: v for k, v in report.items() if k != "leaked" and v != "ok"}
    for name, verdict in sorted(report.items()):
        print(f"VENDORED import {name}: {verdict}")
    assert not failed, failed


# --------------------------------------------------------------------------- device: the backend suites against the copies


def _nvidia_smi_gpus() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return sum(line.startswith("GPU ") for line in proc.stdout.splitlines())


requires_opt_in = pytest.mark.skipif(
    os.environ.get("FLASHINFER_CAKE_VENDORED_SUITE") != "1" or _nvidia_smi_gpus() == 0,
    reason=(
        "set FLASHINFER_CAKE_VENDORED_SUITE=1 on a GPU node with nvcc to build both "
        "manifests with plain nvcc and run the backend suites against the vendored copies"
    ),
)


@pytest.fixture(scope="module")
def vendored_build(tmp_path_factory):
    """Both packages built by the vendored gate (compile, sidecar, load check with
    flashinfer blocked), then arranged as the pair under one parent and the lone copy."""
    work = tmp_path_factory.mktemp("cake_rmsnorm_vendored")
    gate = work / "gate"
    report = work / "gate.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_cake_build_manifests.py"),
            "--vendored-gate",
            "--package",
            f"flashinfer.{DET}",
            "--package",
            f"flashinfer.{NONDET}",
            "--jobs",
            str(min(32, os.cpu_count() or 8)),
            "--work-dir",
            str(gate),
            "--keep",
            "--report-json",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SUITE_TIMEOUT_S,
    )
    summary = [
        line
        for line in proc.stdout.splitlines()
        if "compiled" in line or "finding" in line.lower() or "vendored" in line
    ]
    print("\nVENDORED gate rc=%d\n%s" % (proc.returncode, "\n".join(summary[:20])))
    assert proc.returncode == 0, (
        f"vendored gate exited {proc.returncode}: {proc.stdout[-3000:]} {proc.stderr[-2000:]}"
    )
    data = json.loads(report.read_text())
    assert not data.get("findings"), data.get("findings")
    copies = {
        p.name: p
        for p in gate.iterdir()
        if p.is_dir() and p.name.startswith("vendored_")
    }
    assert {f"vendored_{NONDET}", f"vendored_{DET}"} <= set(copies), sorted(copies)
    pair = work / "pair"
    _copy(copies[f"vendored_{NONDET}"], pair / PARENT / NONDET, with_aot=True)
    _copy(copies[f"vendored_{DET}"], pair / PARENT / DET, with_aot=True)
    (pair / PARENT / "__init__.py").write_text("")
    lone = work / "lone"
    _copy(copies[f"vendored_{NONDET}"], lone / LONE, with_aot=True)
    return {"work": work, "pair": pair, "lone": lone, "arches": data.get("arches")}


_SUITE = """
import importlib
import os

os.environ["PYTHONPATH"] = os.pathsep.join(SYS_PATH)
table = importlib.import_module("tests.test_helpers.cake_package")
table.PACKAGES.update(PACKAGES)
import pytest

sys.exit(int(pytest.main(PYTEST_ARGS)))
"""


def _run_suites(root: Path, packages: dict, suites, work: Path, label: str):
    """The suites in a ``python -I`` child with flashinfer blocked and the
    ``cake_package`` table pointed at the copies (``sitecustomize.py`` repeats both
    for any interpreter a test spawns); ``(junit counts, failing ids)``."""
    paths = [str(root), str(REPO_ROOT), *_dependency_paths()]
    (root / "sitecustomize.py").write_text(
        BLOCKER
        + "import importlib\n"
        + f"importlib.import_module('tests.test_helpers.cake_package').PACKAGES.update({packages!r})\n"
    )
    junit = work / f"{label}.xml"
    args = [
        *suites,
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-q",
        "-rfEs",
        f"--junitxml={junit}",
        f"--basetemp={work / (label + '-tmp')}",
    ]
    script = (
        f"SYS_PATH = {paths!r}\nPACKAGES = {packages!r}\nPYTEST_ARGS = {args!r}\n"
        + _SUITE
    )
    proc = _child(script, paths, cwd=REPO_ROOT, timeout=SUITE_TIMEOUT_S)
    (work / f"{label}.log").write_text(proc.stdout + proc.stderr)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failing, skips, messages = [], {}, {}
    if junit.exists():
        root_el = ET.parse(junit).getroot()
        for suite in (
            [root_el] if root_el.tag == "testsuite" else list(root_el.iter("testsuite"))
        ):
            for key in counts:
                counts[key] += int(suite.get(key, 0))
            for case in suite.iter("testcase"):
                kids = {c.tag: c for c in case}
                if "failure" in kids or "error" in kids:
                    name = f"{case.get('classname')}::{case.get('name')}"
                    failing.append(name)
                    bad = kids.get("failure", kids.get("error"))
                    messages[name] = " ".join(
                        (bad.get("message") or bad.text or "").split()
                    )[:300]
                if "skipped" in kids:
                    msg = kids["skipped"].get("message", "")[:120]
                    skips[msg] = skips.get(msg, 0) + 1
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    print(
        f"VENDORED suite {label} rc={proc.returncode} {counts} log={work / (label + '.log')}"
    )
    for msg, n in sorted(skips.items(), key=lambda kv: -kv[1])[:10]:
        print(f"VENDORED skip {label} {n} x {msg}")
    for name in failing[:40]:
        print(f"VENDORED fail {label} {name} :: {messages.get(name, '')}")
    return proc.returncode, counts, failing, proc


@requires_opt_in
def test_backend_suites_against_the_vendored_pair(vendored_build) -> None:
    root = vendored_build["pair"]
    rc, counts, failing, proc = _run_suites(
        root,
        {
            "cake_rmsnorm_train": f"{PARENT}.{NONDET}",
            "cake_rmsnorm_det_train": f"{PARENT}.{DET}",
        },
        PAIR_SUITES,
        vendored_build["work"],
        "pair",
    )
    assert counts["tests"] > 0, (
        f"no test ran: {proc.stdout[-2000:]} {proc.stderr[-2000:]}"
    )
    assert rc == 0 and not failing, (rc, counts, failing[:10], proc.stdout[-1500:])
    assert counts["passed"] > 0


@requires_opt_in
def test_det_independent_suites_against_the_lone_copy(vendored_build) -> None:
    root = vendored_build["lone"]
    rc, counts, failing, proc = _run_suites(
        root,
        {
            "cake_rmsnorm_train": LONE,
            # no deterministic package anywhere: any use of it raises ImportError
            "cake_rmsnorm_det_train": "vendored_cake_rmsnorm_det_train_backend_absent",
        },
        LONE_SUITES,
        vendored_build["work"],
        "lone",
    )
    assert counts["tests"] > 0, (
        f"no test ran: {proc.stdout[-2000:]} {proc.stderr[-2000:]}"
    )
    assert rc == 0 and not failing, (rc, counts, failing[:10], proc.stdout[-1500:])
    assert counts["passed"] > 0
