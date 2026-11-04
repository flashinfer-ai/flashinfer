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

Compatibility names of the Cake fused BF16 RMSNorm training package.

Within each det namespace (the ``flashinfer.cake_rmsnorm_det_train`` facade and its
backend package) every compatibility name is the same object as its det name and
stays out of ``__all__``.  The names before the det rename at the ``flashinfer`` top
level, the module ``flashinfer.cake_rmsnorm_train`` and the package
``flashinfer.cake_rmsnorm_train_backend`` are the package with the deterministic
default and the opt-in non-deterministic weight gradient: their own objects and
modules, with the pre-rename ``__all__``, whose default calls run the det package
(``test_cake_rmsnorm_train_api.py`` holds the bits).  Both module paths are also
attributes of the ``flashinfer`` package after a plain ``import flashinfer``
(checked in fresh interpreters).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_rmsnorm_det_train as facade
import flashinfer.cake_rmsnorm_det_train_backend as backend
import flashinfer.cake_rmsnorm_train as legacy_facade
import flashinfer.cake_rmsnorm_train_backend as legacy_backend

pytestmark = pytest.mark.cpu_only

OLD_FACADE = "flashinfer.cake_rmsnorm_train"
NEW_FACADE = "flashinfer.cake_rmsnorm_det_train"
OLD_PACKAGE = "flashinfer.cake_rmsnorm_train_backend"
NEW_PACKAGE = "flashinfer.cake_rmsnorm_det_train_backend"
SUBMODULES = ("cake_build", "cake_jit")

# (compatibility name, det name) of the facade and the backend package.
PAIRS = (
    ("CakeRMSNormFunction", "CakeRMSNormDetFunction"),
    ("cake_rmsnorm", "cake_rmsnorm_det"),
    ("cake_rmsnorm_train_backward", "cake_rmsnorm_det_train_backward"),
    (
        "cake_rmsnorm_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace",
    ),
    (
        "cake_rmsnorm_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
    ),
    ("cake_rmsnorm_train_forward", "cake_rmsnorm_det_train_forward"),
    ("is_cake_rmsnorm_train_supported", "is_cake_rmsnorm_det_train_supported"),
)
NAMES = dict(PAIRS)
# The compatibility names the ``flashinfer`` top level exports.
TOP_LEVEL_NAMES = (
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
)
NAMESPACES = {"facade": facade, "backend": backend}


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("old", sorted(NAMES))
def test_compatibility_name_is_the_det_object(namespace: str, old: str) -> None:
    module = NAMESPACES[namespace]
    new = NAMES[old]
    assert getattr(module, old) is getattr(module, new), (namespace, old, new)
    assert new in module.__all__, (namespace, new)
    assert old not in module.__all__, (namespace, old)


@pytest.mark.parametrize("old", TOP_LEVEL_NAMES)
def test_top_level_legacy_name_is_the_legacy_package_object(old: str) -> None:
    new = NAMES[old]
    assert getattr(flashinfer, old) is getattr(legacy_facade, old), old
    assert getattr(flashinfer, old) is not getattr(flashinfer, new), (old, new)
    assert getattr(flashinfer, new) is getattr(facade, new), new


def test_det_entry_points_are_the_decorated_defs() -> None:
    # One ``flashinfer_api`` wrapper per entry point: the compatibility names add none.
    old, new = ("cake_rmsnorm_train_forward", "cake_rmsnorm_det_train_forward")
    assert getattr(facade, old).__name__ == new


# The ``__all__`` of the legacy facade and backend package: the names before the det rename.
LEGACY_ALL = sorted(["ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "route_key", *NAMES])


def test_old_facade_is_its_own_module() -> None:
    assert importlib.import_module(OLD_FACADE) is legacy_facade
    assert sys.modules[OLD_FACADE] is not sys.modules[NEW_FACADE]
    assert Path(legacy_facade.__file__).name == "cake_rmsnorm_train.py"
    assert sorted(legacy_facade.__all__) == LEGACY_ALL
    for name in legacy_facade.__all__:
        assert getattr(legacy_facade, name) is getattr(
            legacy_backend, name
        ) or name in (
            "cake_rmsnorm",
            "cake_rmsnorm_train_backward",
            "cake_rmsnorm_train_forward",
        ), name
    # the det names of the previous module (the det facade itself) resolve to the
    # det facade's objects, outside ``__all__``
    for _old, det_name in PAIRS:
        assert det_name not in legacy_facade.__all__, det_name
        assert getattr(legacy_facade, det_name) is getattr(facade, det_name), det_name


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_old_package_submodule_is_its_own_module(submodule: str) -> None:
    old = importlib.import_module(f"{OLD_PACKAGE}.{submodule}")
    new = importlib.import_module(f"{NEW_PACKAGE}.{submodule}")
    assert old is not new, submodule
    assert sys.modules[f"{OLD_PACKAGE}.{submodule}"] is old, submodule
    assert Path(old.__file__).parent == Path(legacy_backend.__file__).parent, submodule


def test_old_package_is_self_contained() -> None:
    assert sorted(legacy_backend.__all__) == LEGACY_ALL
    # the det names of the previous package (an alias of the det backend) resolve to
    # the det backend's objects, outside ``__all__``
    for _old, det_name in PAIRS:
        assert det_name not in legacy_backend.__all__, det_name
        assert getattr(legacy_backend, det_name) is getattr(backend, det_name), det_name
    # its own registry: the forward programs of the det package renamed, the
    # backward records of its single kernels and of the det pairs it renders
    modules = legacy_backend.cake_jit.MODULES
    assert modules and all(name.startswith("cake_rmsnorm_train_") for name in modules)
    assert all(
        record["workspace_rule"] is None
        or record["workspace_rule"]["kind"] in ("claim_sum_f64", "chunk_partials")
        for record in modules.values()
    )
    old_dir = Path(legacy_backend.__file__).parent
    files = {p.name for p in old_dir.iterdir() if p.name != "__pycache__"}
    assert {
        "__init__.py",
        "cake_build.py",
        "cake_jit.py",
        "cake_windows.py",
        "build.json",
        "csrc",
    } <= files


def test_old_package_star_import_binds_its_own_objects() -> None:
    names: dict = {}
    exec(f"from {OLD_PACKAGE} import *", names)
    del names["__builtins__"]
    assert sorted(names) == LEGACY_ALL
    for name in LEGACY_ALL:
        assert names[name] is getattr(legacy_backend, name), name


def test_old_package_dir_lists_its_own_names() -> None:
    listed = set(dir(legacy_backend))
    assert set(LEGACY_ALL) | set(SUBMODULES) <= listed
    assert not set(NAMES.values()) & listed


# Each previous import form in a fresh interpreter: the submodule path first, then the attributes of flashinfer,
# the package alone (its submodules registered), and a star import of the package.
FRESH = {
    "old_submodule_first": (
        "import importlib, sys\n"
        f"old = importlib.import_module('{OLD_PACKAGE}.cake_jit')\n"
        f"new = importlib.import_module('{NEW_PACKAGE}.cake_jit')\n"
        "assert old is not new and old.MODULES is not new.MODULES\n"
        f"assert importlib.import_module('{OLD_FACADE}') is not importlib.import_module('{NEW_FACADE}')\n"
    ),
    "flashinfer_attributes": (
        "import sys\nimport flashinfer\n"
        f"assert flashinfer.{OLD_FACADE.rpartition('.')[2]} is sys.modules['{OLD_FACADE}']\n"
        f"assert flashinfer.{OLD_PACKAGE.rpartition('.')[2]} is sys.modules['{OLD_PACKAGE}']\n"
        f"assert flashinfer.{PAIRS[0][0]} is sys.modules['{OLD_FACADE}'].{PAIRS[0][0]}\n"
    ),
    "old_package_star_import": (
        "import importlib\n"
        f"from {OLD_PACKAGE} import *\n"
        f"old = importlib.import_module('{OLD_PACKAGE}')\n"
        "assert all(globals()[n] is getattr(old, n) for n in old.__all__)\n"
        "assert 'cake_rmsnorm_det_train_forward' not in globals()\n"
    ),
}


@pytest.mark.parametrize("form", sorted(FRESH))
def test_previous_import_forms_in_a_fresh_interpreter(form: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH[form]], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
