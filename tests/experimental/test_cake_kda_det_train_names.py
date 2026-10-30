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
``flashinfer.experimental.cake_kda_det_train``.  Every previous name is the same object as its det name:
the single-rank names of ``flashinfer.kda_train``, the backend names of ``cake_backend`` and of the package, and
every name of the compatibility package ``flashinfer.experimental.cake_kda_train``, whose submodules are the det
package's module objects and whose ``__all__`` (so its star import) is the list before the move.  The previous names
stay out of the det ``__all__``.  The manifests name the det package; the 65 single-rank programs are
``cake_kda_det_train_<digest>`` libraries.  Importing a facade or either package loads no KCP host module (checked
in a fresh interpreter by module file).
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
import flashinfer.kda_train as kda_train
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
OLD_DIR = PACKAGE_DIR.parent / "cake_kda_train"
# the det package's modules, read from its directory
SUBMODULES = tuple(
    sorted(p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__")
)

# (previous name, det name) of the single-rank entry points
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
# (previous name, det name) of the backend
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
# the package's __all__ before the move: the det __all__ with the previous backend names
PREVIOUS_ALL = sorted(
    {dict((new, old) for old, new in PACKAGE_PAIRS).get(n, n) for n in package.__all__}
)


@pytest.mark.parametrize("old,new", FACADE_PAIRS)
def test_facade_previous_names_are_the_det_objects(old: str, new: str) -> None:
    fn = getattr(det_api, new)
    assert getattr(kda_train, old) is fn
    assert fn.__name__ == new
    assert new in det_api.__all__ and old not in det_api.__all__
    assert old in kda_train.__all__


def test_facade_shares_the_scale_and_the_feature() -> None:
    assert kda_train.DEFAULT_SCALE is det_api.DEFAULT_SCALE
    assert det_api.DEFAULT_SCALE == 128**-0.5 == cake_backend.SCALE
    for _, new in FACADE_PAIRS[:3]:
        assert getattr(det_api, new).experimental_feature == det_api._FEATURE


@pytest.mark.parametrize("old,new", BACKEND_PAIRS)
def test_backend_previous_names_are_the_det_objects(old: str, new: str) -> None:
    obj = getattr(cake_backend, new)
    assert getattr(cake_backend, old) is obj
    assert obj.__name__ == new and obj.__qualname__ == new
    assert obj.__module__ == f"{NEW_PACKAGE}.cake_backend"
    assert new in cake_backend.__all__ and old not in cake_backend.__all__


@pytest.mark.parametrize("old,new", PACKAGE_PAIRS)
def test_package_previous_names_are_the_det_objects(old: str, new: str) -> None:
    assert getattr(package, old) is getattr(package, new) is getattr(cake_backend, new)
    assert new in package.__all__ and old not in package.__all__


def test_workspace_isinstance_holds_across_the_spellings() -> None:
    assert issubclass(cake_backend.KdaDetWorkspace, cake_backend.KdaWorkspace)
    arena = cake_backend.KdaArena(lean=True)
    assert isinstance(
        arena, cake_backend.KdaDetArena
    ) and arena == cake_backend.KdaDetArena(lean=True)


def test_compatibility_package_is_the_det_package() -> None:
    old = importlib.import_module(OLD_PACKAGE)
    assert sorted(old.__all__) == PREVIOUS_ALL
    for name in package.__all__:
        assert getattr(old, name) is getattr(package, name), name
    for previous, det in PACKAGE_PAIRS:
        assert getattr(old, previous) is getattr(cake_backend, det), previous
    with pytest.raises(AttributeError):
        _ = old.NoSuchName  # type: ignore[attr-defined]


def test_compatibility_package_star_import_binds_the_previous_names() -> None:
    scope: dict[str, object] = {}
    exec(f"from {OLD_PACKAGE} import *", scope)
    names = sorted(name for name in scope if name != "__builtins__")
    assert names == PREVIOUS_ALL
    for previous, det in PACKAGE_PAIRS:
        assert scope[previous] is getattr(cake_backend, det), previous
    for name in names:
        assert scope[name] is getattr(package, name), name


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_compatibility_submodules_are_the_det_modules(submodule: str) -> None:
    new = importlib.import_module(f"{NEW_PACKAGE}.{submodule}")
    assert importlib.import_module(f"{OLD_PACKAGE}.{submodule}") is new
    assert new.__name__ == f"{NEW_PACKAGE}.{submodule}"
    assert Path(new.__file__).resolve().parent == PACKAGE_DIR
    old_package = importlib.import_module(OLD_PACKAGE)
    assert getattr(old_package, submodule) is new
    assert cake_package("cake_kda_train", submodule) is new
    assert cake_package("cake_kda_det_train", submodule) is new


def test_compatibility_package_holds_no_sources() -> None:
    files = {p.name for p in OLD_DIR.iterdir() if p.name != "__pycache__"}
    assert all(name.endswith(".py") for name in files), sorted(files)
    assert {"__init__.py", *(f"{m}.py" for m in SUBMODULES)} <= files


def test_reference_helper_previous_path_is_the_det_module() -> None:
    new = importlib.import_module("tests.test_helpers.cake_kda_det_train_reference")
    assert importlib.import_module("tests.test_helpers.cake_kda_train_reference") is new


def test_manifests_name_the_det_package_and_libraries() -> None:
    shipped = json.loads((PACKAGE_DIR / "build.json").read_text())
    assert shipped["package"] == NEW_PACKAGE
    test_only = PACKAGE_DIR / "build.test.json"
    if test_only.is_file():
        assert json.loads(test_only.read_text())["package"] == NEW_PACKAGE
    single = {record["library"] for record in cake_jit.MODULES.values()}
    assert len(single) == 65
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
        ("flashinfer.kda_train",),
        (NEW_PACKAGE,),
        (OLD_PACKAGE,),
        (f"{OLD_PACKAGE}.cake_backend", f"{OLD_PACKAGE}.cake_jit"),
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
