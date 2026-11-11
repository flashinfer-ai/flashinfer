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

Names of the Cake packed KDA chunked training package (deterministic, single rank).

The single-rank entry points live in ``flashinfer.cake_kda_det_train`` and the package in
``flashinfer.experimental.cake.cake_kda_det_train``.  The shared check of #8928
(``tests.cake.test_helpers.cake_mode_names.assert_cake_mode_module``) covers the facade: every public name states
the mode (``det``) right after the op, no entry takes ``deterministic=``, the facade, the ``flashinfer`` top
level and the package bind each object under one name only, and the names before this scheme fail at import
with no alias: the package path ``flashinfer.experimental.cake_kda_det_train``, the ``chunk_kda_det*`` /
``ChunkKDADetFunction`` / ``KdaDet*`` / ``kda_det_train_*`` names, and the older names before the det rename
(the module ``flashinfer.kda_train``, the package ``flashinfer.experimental.cake_kda_train``, the helper
``tests.test_helpers.cake_kda_train_reference``).  The same mode-word and keyword checks run on the package.
This file adds what the shared check does not cover: the facade entries are stable (not experimental), the
backend objects are defined under their names in ``cake_backend``, the host-plan helpers stay internal, the
manifests name the package and the single-rank programs are ``cake_kda_det_train_<digest>`` libraries, and
importing the facade or the package loads no KCP host module (checked in a fresh interpreter by module file).
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_kda_det_train as det_api
from flashinfer.experimental.cake import cake_kda_det_train as package
from flashinfer.experimental.cake.cake_kda_det_train import (
    cake_backend,
    cake_jit,
)
from tests.cake.test_helpers.cake_mode_names import (
    assert_cake_mode_module,
    assert_mode_names,
    assert_no_deterministic_keyword,
)
from tests.cake.test_helpers.cake_package import cake_package

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_kda_det_train"
NEW_PACKAGE = "flashinfer.experimental.cake.cake_kda_det_train"
PREVIOUS_PACKAGE = "flashinfer.experimental.cake_kda_det_train"
OLD_PACKAGE = "flashinfer.experimental.cake_kda_train"
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
# the det package's modules, read from its directory
SUBMODULES = tuple(
    sorted(p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__")
)

# (removed names, name) of the single-rank entry points: the det names before this scheme, then the names before
# the det rename
FACADE_NAMES = (
    (("chunk_kda_det_fwd", "chunk_kda_fwd"), "cake_kda_det_train_forward"),
    (("chunk_kda_det_bwd", "chunk_kda_bwd"), "cake_kda_det_train_backward"),
    (("chunk_kda_det", "chunk_kda"), "cake_kda_det"),
    (("kda_det_train_arena", "kda_train_arena"), "cake_kda_det_train_arena"),
    (
        ("kda_det_train_workspace", "kda_train_workspace"),
        "cake_kda_det_train_workspace",
    ),
    (
        ("kda_det_train_workspace_bytes", "kda_train_workspace_bytes"),
        "cake_kda_det_train_workspace_bytes",
    ),
    (
        ("kda_det_train_saved_bytes", "kda_train_saved_bytes"),
        "cake_kda_det_train_saved_bytes",
    ),
    (
        ("is_kda_det_train_supported", "is_kda_train_supported"),
        "is_cake_kda_det_train_supported",
    ),
)
# (removed names, name) of the backend; the package exports the same names
BACKEND_NAMES = (
    (("ChunkKDADetFunction", "ChunkKDAFunction"), "CakeKDADetFunction"),
    (("KdaDetArena", "KdaArena"), "CakeKDADetArena"),
    (("KdaDetChunkSaved", "KdaChunkSaved"), "CakeKDADetChunkSaved"),
    (("KdaDetWorkspace", "KdaWorkspace"), "CakeKDADetWorkspace"),
    (("chunk_kda_det_fwd", "chunk_kda_fwd"), "cake_kda_det_train_forward"),
    (("chunk_kda_det_bwd", "chunk_kda_bwd"), "cake_kda_det_train_backward"),
    (("chunk_kda_det", "chunk_kda"), "cake_kda_det"),
)
# backend host-plan helpers: internal to ``cake_backend``, not exported by the package (the facade exposes the
# mode-named entries for them)
BACKEND_HELPERS = (
    "Plan",
    "is_supported",
    "memory_report",
    "output_bytes",
    "plan",
    "saved_bytes",
    "seg_states_saved",
    "workspace_bytes",
)
# the facade functions that call a backend object (a link no attribute shows); the facade's arena / workspace
# entries build the backend types and are not the types themselves
FORWARDS = (
    ("cake_kda_det_train_forward", cake_backend.cake_kda_det_train_forward),
    ("cake_kda_det_train_backward", cake_backend.cake_kda_det_train_backward),
    ("cake_kda_det", cake_backend.cake_kda_det),
    ("cake_kda_det_train_workspace_bytes", cake_backend.workspace_bytes),
    ("cake_kda_det_train_saved_bytes", cake_backend.saved_bytes),
    ("is_cake_kda_det_train_supported", cake_backend.is_supported),
)
# the modules removed with the previous names
OLD_MODULES = (
    PREVIOUS_PACKAGE,
    *(f"{PREVIOUS_PACKAGE}.{m}" for m in ("cake_backend", "cake_build", "cake_jit")),
    "flashinfer.kda_train",
    OLD_PACKAGE,
    *(
        f"{OLD_PACKAGE}.{m}"
        for m in (
            "cake_backend",
            "cake_build",
            "cake_jit",
            "cake_kcp",
            "cake_kcp_jit",
            "cake_kcp_plan",
        )
    ),
    "tests.test_helpers.cake_kda_train_reference",
)
OLD_NAMES = (
    *((FACADE, old) for olds, _ in FACADE_NAMES for old in olds),
    *(
        (module, old)
        for olds, _ in BACKEND_NAMES
        for old in olds
        for module in (NEW_PACKAGE, f"{NEW_PACKAGE}.cake_backend")
    ),
)


def test_facade_follows_the_cake_mode_names() -> None:
    assert_cake_mode_module(
        det_api,
        "det",
        namespaces=[flashinfer, package],
        forwards=FORWARDS,
        old_modules=OLD_MODULES,
        old_names=OLD_NAMES,
        cwd=REPO_ROOT,
    )


def test_package_states_the_mode_and_takes_no_deterministic_keyword() -> None:
    assert_mode_names(package, "det")
    assert_no_deterministic_keyword(package)


def test_facade_exports_the_entry_points_under_their_names() -> None:
    assert sorted(det_api.__all__) == sorted(
        ["DEFAULT_SCALE", *(new for _, new in FACADE_NAMES)]
    )
    for _, new in FACADE_NAMES:
        assert getattr(det_api, new).__name__ == new


def test_facade_shares_the_scale_and_is_not_experimental() -> None:
    assert det_api.DEFAULT_SCALE == 128**-0.5 == cake_backend.SCALE
    assert not hasattr(det_api, "_FEATURE")
    for _, new in FACADE_NAMES:
        assert not getattr(getattr(det_api, new), "is_experimental", False), new


@pytest.mark.parametrize("new", [new for _, new in BACKEND_NAMES])
def test_backend_defines_the_package_names(new: str) -> None:
    obj = getattr(cake_backend, new)
    assert obj.__name__ == new and obj.__qualname__ == new
    assert obj.__module__ == f"{NEW_PACKAGE}.cake_backend"
    assert new in cake_backend.__all__ and new in package.__all__
    assert getattr(package, new) is obj


@pytest.mark.parametrize("name", BACKEND_HELPERS)
def test_package_does_not_export_the_backend_helpers(name: str) -> None:
    assert hasattr(cake_backend, name)
    assert name not in package.__all__ and not hasattr(package, name)


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_det_submodules_are_the_package_modules(submodule: str) -> None:
    new = importlib.import_module(f"{NEW_PACKAGE}.{submodule}")
    assert new.__name__ == f"{NEW_PACKAGE}.{submodule}"
    assert Path(new.__file__).resolve().parent == PACKAGE_DIR
    assert cake_package("cake.cake_kda_det_train", submodule) is new


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
