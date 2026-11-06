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

Names of the Cake packed KDA chunked training package.

The single-rank entry points live in ``flashinfer.cake_kda_det_train`` and the package in
``flashinfer.experimental.cake_kda_det_train``; every public name carries ``det``.  The names before the det
rename are gone: the module ``flashinfer.kda_train``, the package ``flashinfer.experimental.cake_kda_train``, the
helper ``tests.test_helpers.cake_kda_train_reference`` and the previous single-rank names of the facade, of
``cake_backend`` and of the package.  The manifests name the det package; the 78 single-rank programs (65 of the g
gate, 8 of the raw gate, 5 of the Q/K L2 norm fallback) are ``cake_kda_det_train_<digest>`` libraries.  Importing the facade or the package loads
no KCP host module (checked in a fresh interpreter by module file).
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer.cake_kda_det_train as det_api
from flashinfer.experimental import cake_kda_det_train as package
from flashinfer.experimental.cake_kda_det_train import (
    cake_backend,
    cake_jit,
)
from tests.test_helpers.cake_package import cake_package

pytestmark = pytest.mark.cpu_only

NEW_PACKAGE = "flashinfer.experimental.cake_kda_det_train"
OLD_PACKAGE = "flashinfer.experimental.cake_kda_train"
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
# the det package's modules, read from its directory
SUBMODULES = tuple(
    sorted(p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__")
)

# (removed previous name, det name) of the single-rank entry points
FACADE_PAIRS = (
    ("chunk_kda_fwd", "chunk_kda_det_fwd"),
    ("chunk_kda_bwd", "chunk_kda_det_bwd"),
    ("chunk_kda", "chunk_kda_det"),
    ("kda_train_arena", "kda_det_train_arena"),
    ("kda_train_workspace", "kda_det_train_workspace"),
    ("kda_train_workspace_bytes", "kda_det_train_workspace_bytes"),
    ("kda_train_saved_bytes", "kda_det_train_saved_bytes"),
    ("is_kda_train_supported", "is_kda_det_train_supported"),
)
# (removed previous name, det name) of the backend
BACKEND_PAIRS = (
    ("ChunkKDAFunction", "ChunkKDADetFunction"),
    ("KdaArena", "KdaDetArena"),
    ("KdaChunkSaved", "KdaDetChunkSaved"),
    ("KdaWorkspace", "KdaDetWorkspace"),
    ("chunk_kda_fwd", "chunk_kda_det_fwd"),
    ("chunk_kda_bwd", "chunk_kda_det_bwd"),
    ("chunk_kda", "chunk_kda_det"),
)
# the backend names the package __init__ exports
PACKAGE_PAIRS = tuple(p for p in BACKEND_PAIRS if p[0] != "KdaArena")
# the modules removed with the previous names
REMOVED_MODULES = (
    "flashinfer.kda_train",
    OLD_PACKAGE,
    *(
        f"{OLD_PACKAGE}.{m}"
        for m in ("cake_backend", "cake_build", "cake_jit", "cake_kcp", "cake_kcp_jit", "cake_kcp_plan")
    ),
    "tests.test_helpers.cake_kda_train_reference",
)


@pytest.mark.parametrize("old,new", FACADE_PAIRS)
def test_facade_has_only_the_det_names(old: str, new: str) -> None:
    fn = getattr(det_api, new)
    assert fn.__name__ == new
    assert new in det_api.__all__ and old not in det_api.__all__
    assert not hasattr(det_api, old)


def test_facade_shares_the_scale_and_the_feature() -> None:
    assert det_api.DEFAULT_SCALE == 128**-0.5 == cake_backend.SCALE
    for _, new in FACADE_PAIRS[:3]:
        assert getattr(det_api, new).experimental_feature == det_api._FEATURE


@pytest.mark.parametrize("old,new", BACKEND_PAIRS)
def test_backend_has_only_the_det_names(old: str, new: str) -> None:
    obj = getattr(cake_backend, new)
    assert obj.__name__ == new and obj.__qualname__ == new
    assert obj.__module__ == f"{NEW_PACKAGE}.cake_backend"
    assert new in cake_backend.__all__ and old not in cake_backend.__all__
    assert not hasattr(cake_backend, old)


@pytest.mark.parametrize("old,new", PACKAGE_PAIRS)
def test_package_has_only_the_det_names(old: str, new: str) -> None:
    assert getattr(package, new) is getattr(cake_backend, new)
    assert new in package.__all__ and old not in package.__all__
    assert not hasattr(package, old)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_previous_modules_are_removed(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_det_submodules_are_the_package_modules(submodule: str) -> None:
    new = importlib.import_module(f"{NEW_PACKAGE}.{submodule}")
    assert new.__name__ == f"{NEW_PACKAGE}.{submodule}"
    assert Path(new.__file__).resolve().parent == PACKAGE_DIR
    assert cake_package("cake_kda_det_train", submodule) is new


def test_manifests_name_the_det_package_and_libraries() -> None:
    shipped = json.loads((PACKAGE_DIR / "build.json").read_text())
    assert shipped["package"] == NEW_PACKAGE
    test_only = PACKAGE_DIR / "build.test.json"
    if test_only.is_file():
        assert json.loads(test_only.read_text())["package"] == NEW_PACKAGE
    single = {record["library"] for record in cake_jit.MODULES.values()}
    # 65 g-gate, 8 raw-gate, 5 Q/K L2 norm fallback and 17 fused Q/K L2 norm programs (10 g-gate, fwd_intra v37
    # and glm_intra v30 on sm_103a and the replay prep's saved-rstd twin among them, and 7 raw-gate)
    assert len(single) == 95
    for library in single:
        assert re.fullmatch(r"cake_kda_det_train_[0-9a-f]{20}", library), library
    for name, record in cake_jit.MODULES.items():
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
    entries = {entry["name"] for entry in shipped["libraries"]}
    for name, record in cake_jit.MODULES.items():
        for arch in record["arches"]:
            assert f"{record['library']}_{arch}" in entries, (name, arch)
    sources = {p.name for p in (PACKAGE_DIR / "csrc").glob("cake_kda_*_*.cu")}
    for library in single:
        assert {f"{library}_kernel.cu", f"{library}_binding.cu"} <= sources, library
        old = library.replace("cake_kda_det_train_", "cake_kda_train_", 1)
        assert f"{old}_kernel.cu" not in sources, old
    det_prefixed = {s for s in sources if s.startswith("cake_kda_det_train_")}
    assert det_prefixed == {
        f"{library}_{part}.cu" for library in single for part in ("kernel", "binding")
    }


_FRESH = r"""
import importlib, sys
from pathlib import Path
for name in sys.argv[1:]:
    importlib.import_module(name)
loaded = sorted(
    m.__name__ for m in list(sys.modules.values())
    if getattr(m, "__file__", None) and Path(m.__file__).name in ("cake_kcp.py", "cake_kcp_jit.py")
    and "cake_kda" in str(Path(m.__file__).parent)
)
print("KCP_LOADED", loaded)
"""


@pytest.mark.parametrize(
    "imports",
    [
        ("flashinfer.cake_kda_det_train",),
        (NEW_PACKAGE,),
        (f"{NEW_PACKAGE}.cake_backend", f"{NEW_PACKAGE}.cake_jit"),
    ],
)
def test_imports_load_no_kcp_host_module(imports: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _FRESH, *imports],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "KCP_LOADED []" in result.stdout, (imports, result.stdout, result.stderr)
