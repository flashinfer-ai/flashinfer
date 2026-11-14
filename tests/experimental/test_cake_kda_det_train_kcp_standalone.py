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

The Cake deterministic KCP KDA package vendored without FlashInfer (its README "Vendoring"): the package and its
single-rank sibling ``cake_kda_det_train`` (``common.requires.packages`` of its manifests) are copied in two layouts -- both as top-level packages, and as siblings
under the namespace parent ``vendored_x`` -- and imported in a subprocess where ``flashinfer`` cannot be imported.
The sibling must resolve to the copy.  No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "cake_kda_det_train_kcp"
SINGLE_RANK = "cake_kda_det_train"
MODULES = ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit", "cake_build")
SINGLE_RANK_MODULES = ("cake_backend", "cake_build", "cake_jit")


def _cake_root() -> Path:
    """``flashinfer/experimental/cake`` of the FlashInfer under test, found without importing ``flashinfer``."""
    spec = importlib.util.find_spec("flashinfer")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))) / "experimental" / "cake"


# Runs in the subprocess: blocks ``flashinfer``, imports the copy of the package and its modules, resolves the
# single-rank modules eagerly (``cake_kcp``, ``cake_kcp_jit``) and function-locally (``cake_kcp_plan``), and prints
# where every module came from.
_PROBE = """
import importlib, json, sys
sys.modules["flashinfer"] = None
root, package = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
importlib.import_module(package)
found = {sub: importlib.import_module(f"{package}.{sub}") for sub in MODULES}
single_rank = importlib.import_module(f"{package}._single_rank")
plan = found["cake_kcp_plan"]
plan._qk_l2norm_inputs()
cake_jit = found["cake_kcp_jit"].cake_jit
print(json.dumps({
    "name": single_rank.NAME,
    "files": {sub: module.__file__ for sub, module in found.items()},
    "single_rank": {sub: sys.modules[f"{single_rank.NAME}.{sub}"].__file__ for sub in SINGLE_RANK_MODULES},
    "bound": {
        "cake_kcp.cake_backend": found["cake_kcp"].cake_backend.__file__,
        "cake_kcp.cake_jit": found["cake_kcp"].cake_jit.__file__,
        "cake_kcp_jit.cake_jit": cake_jit.__file__,
        "cake_kcp_plan._QK_L2NORM_MODULES": [m.__file__ for m in plan._QK_L2NORM_MODULES],
    },
    "l2n_twin": plan.qk_l2norm_twin(next(iter(cake_jit.MODULES))),
    "flashinfer": sorted(k for k in sys.modules if k.startswith("flashinfer.") or (k == "flashinfer" and sys.modules[k] is not None)),
}))
"""


def _copy(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "aot"))


@pytest.mark.parametrize("parent", ["", "vendored_x"], ids=["top_level", "siblings"])
def test_vendored_package_imports_without_flashinfer(
    tmp_path: Path, parent: str
) -> None:
    cake = _cake_root()
    # what a consumer vendors: the package and the packages its manifests declare (directory names)
    for manifest in ("build.json", "build.test.json"):
        common = json.loads((cake / PACKAGE / manifest).read_text())["common"]
        assert common["requires"]["packages"] == [SINGLE_RANK], manifest
    where = tmp_path / parent if parent else tmp_path
    # no __init__.py: vendored_x is a namespace package
    where.mkdir(parents=True, exist_ok=True)
    for directory in (PACKAGE, SINGLE_RANK):
        _copy(cake / directory, where / directory)
    package = f"{parent}.{PACKAGE}" if parent else PACKAGE
    probe = f"MODULES = {MODULES!r}\nSINGLE_RANK_MODULES = {SINGLE_RANK_MODULES!r}\n{_PROBE}"
    run = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), package],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["name"] == (f"{parent}.{SINGLE_RANK}" if parent else SINGLE_RANK)
    assert out["flashinfer"] == []
    assert isinstance(out["l2n_twin"], bool)
    ours, theirs = (where / PACKAGE).resolve(), (where / SINGLE_RANK).resolve()
    for sub, file in out["files"].items():
        assert Path(file).resolve().parent == ours, (sub, file)
    for sub, file in out["single_rank"].items():
        assert Path(file).resolve() == theirs / f"{sub}.py", (sub, file)
    bound = out["bound"]
    assert bound["cake_kcp.cake_backend"] == out["single_rank"]["cake_backend"]
    assert bound["cake_kcp.cake_jit"] == out["single_rank"]["cake_jit"]
    assert bound["cake_kcp_jit.cake_jit"] == out["single_rank"]["cake_jit"]
    assert bound["cake_kcp_plan._QK_L2NORM_MODULES"] == [
        out["single_rank"][sub] for sub in SINGLE_RANK_MODULES
    ]


def test_package_has_no_import_above_its_parent() -> None:
    """No module of the package imports with a relative level of two or more (``from ..x import``): the single-rank
    sibling is reached through ``_single_rank`` only, so the package does not depend on where its parent lives."""
    package = _cake_root() / PACKAGE
    files = sorted(package.rglob("*.py"))
    assert package / "_single_rank.py" in files
    found = [
        f"{path.relative_to(package)}:{node.lineno}"
        for path in files
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.level >= 2
    ]
    assert found == []
