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

Compatibility names of the Cake SiLU-gated RMSNorm training package.

Within each namespace (the ``flashinfer.cake_rmsnorm_silu_gate_det_train`` facade, its backend package and the
``flashinfer`` top level) every compatibility name is the same object as its det
name and stays out of ``__all__``.  The module ``flashinfer.cake_rmsnorm_silu_gate_train`` is the facade's
module object, and the submodules of ``flashinfer.cake_rmsnorm_silu_gate_train_backend`` are the det package's
module objects, registered in ``sys.modules`` when that package is imported; that
package's ``__all__`` is the det package's, so a star import binds the det objects,
and its ``dir`` lists the det names.  Both previous module paths are also attributes
of the ``flashinfer`` package after a plain ``import flashinfer`` (checked in fresh
interpreters).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as facade
import flashinfer.cake_rmsnorm_silu_gate_det_train_backend as backend

pytestmark = pytest.mark.cpu_only

OLD_FACADE = "flashinfer.cake_rmsnorm_silu_gate_train"
NEW_FACADE = "flashinfer.cake_rmsnorm_silu_gate_det_train"
OLD_PACKAGE = "flashinfer.cake_rmsnorm_silu_gate_train_backend"
NEW_PACKAGE = "flashinfer.cake_rmsnorm_silu_gate_det_train_backend"
SUBMODULES = ("cake_build", "cake_jit")

# (compatibility name, det name) of the facade and the backend package.
PAIRS = (
    ("CakeRMSNormSiLUGateFunction", "CakeRMSNormSiLUGateDetFunction"),
    ("cake_rmsnorm_silu_gate", "cake_rmsnorm_silu_gate_det"),
    (
        "cake_rmsnorm_silu_gate_train_backward",
        "cake_rmsnorm_silu_gate_det_train_backward",
    ),
    (
        "cake_rmsnorm_silu_gate_train_backward_workspace",
        "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    ),
    (
        "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
        "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    ),
    (
        "cake_rmsnorm_silu_gate_train_forward",
        "cake_rmsnorm_silu_gate_det_train_forward",
    ),
    (
        "is_cake_rmsnorm_silu_gate_train_supported",
        "is_cake_rmsnorm_silu_gate_det_train_supported",
    ),
)
NAMES = dict(PAIRS)
# The compatibility names the ``flashinfer`` top level exports.
TOP_LEVEL_NAMES = (
    "CakeRMSNormSiLUGateFunction",
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_forward",
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
def test_top_level_compatibility_name_is_the_det_object(old: str) -> None:
    new = NAMES[old]
    assert getattr(flashinfer, old) is getattr(flashinfer, new), (old, new)
    assert getattr(flashinfer, new) is getattr(facade, new), new


def test_det_entry_points_are_the_decorated_defs() -> None:
    # One ``flashinfer_api`` wrapper per entry point: the compatibility names add none.
    old, new = (
        "cake_rmsnorm_silu_gate_train_forward",
        "cake_rmsnorm_silu_gate_det_train_forward",
    )
    assert getattr(facade, old).__name__ == new


def test_old_facade_is_the_det_module() -> None:
    assert importlib.import_module(OLD_FACADE) is facade
    assert sys.modules[OLD_FACADE] is sys.modules[NEW_FACADE]


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_old_package_submodule_is_the_det_module(submodule: str) -> None:
    old = importlib.import_module(f"{OLD_PACKAGE}.{submodule}")
    new = importlib.import_module(f"{NEW_PACKAGE}.{submodule}")
    assert old is new, submodule
    assert sys.modules[f"{OLD_PACKAGE}.{submodule}"] is new, submodule
    assert Path(new.__file__).parent == Path(backend.__file__).parent, submodule


def test_old_package_forwards_to_the_det_package() -> None:
    old = importlib.import_module(OLD_PACKAGE)
    for name in (*NAMES, *NAMES.values(), "ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"):
        assert getattr(old, name) is getattr(backend, name), name
    assert old.__all__ == backend.__all__
    # The compatibility package holds Python modules only: no csrc, manifest or libraries.
    old_dir = Path(old.__file__).parent
    files = sorted(p.name for p in old_dir.iterdir() if p.name != "__pycache__")
    assert files == sorted(["__init__.py", *(f"{m}.py" for m in SUBMODULES)])


def test_old_package_star_import_binds_the_det_objects() -> None:
    old_names: dict = {}
    new_names: dict = {}
    exec(f"from {OLD_PACKAGE} import *", old_names)
    exec(f"from {NEW_PACKAGE} import *", new_names)
    del old_names["__builtins__"], new_names["__builtins__"]
    assert sorted(old_names) == sorted(backend.__all__)
    for name in backend.__all__:
        assert old_names[name] is new_names[name] is getattr(backend, name), name


def test_old_package_dir_lists_the_det_names() -> None:
    old = importlib.import_module(OLD_PACKAGE)
    listed = set(dir(old))
    assert set(dir(backend)) <= listed
    assert {*NAMES, *NAMES.values(), *SUBMODULES} <= listed


# Each previous import form in a fresh interpreter: the submodule path first, then the attributes of flashinfer,
# the package alone (its submodules registered), and a star import of the package.
FRESH = {
    "old_submodule_first": (
        "import importlib, sys\n"
        f"old = importlib.import_module('{OLD_PACKAGE}.cake_jit')\n"
        f"new = importlib.import_module('{NEW_PACKAGE}.cake_jit')\n"
        "assert old is new\n"
        f"assert importlib.import_module('{OLD_PACKAGE}').{PAIRS[0][0]} is importlib.import_module('{NEW_PACKAGE}').{PAIRS[0][1]}\n"
        f"assert importlib.import_module('{OLD_FACADE}') is importlib.import_module('{NEW_FACADE}')\n"
    ),
    "flashinfer_attributes": (
        "import sys\nimport flashinfer\n"
        f"assert flashinfer.{OLD_FACADE.rpartition('.')[2]} is flashinfer.{NEW_FACADE.rpartition('.')[2]}\n"
        f"assert flashinfer.{OLD_FACADE.rpartition('.')[2]} is sys.modules['{NEW_FACADE}']\n"
        f"assert flashinfer.{OLD_PACKAGE.rpartition('.')[2]} is sys.modules['{OLD_PACKAGE}']\n"
        f"assert flashinfer.{OLD_PACKAGE.rpartition('.')[2]}.{PAIRS[0][0]} is flashinfer.{NEW_PACKAGE.rpartition('.')[2]}.{PAIRS[0][1]}\n"
    ),
    "old_package_registers_submodules": (
        "import importlib, sys\n"
        f"importlib.import_module('{OLD_PACKAGE}')\n"
        f"for s in {SUBMODULES!r}:\n"
        f"    assert sys.modules['{OLD_PACKAGE}.' + s] is sys.modules['{NEW_PACKAGE}.' + s], s\n"
    ),
    "old_package_star_import": (
        "import importlib\n"
        f"from {OLD_PACKAGE} import *\n"
        f"new = importlib.import_module('{NEW_PACKAGE}')\n"
        "assert all(globals()[n] is getattr(new, n) for n in new.__all__)\n"
    ),
}


@pytest.mark.parametrize("form", sorted(FRESH))
def test_previous_import_forms_in_a_fresh_interpreter(form: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH[form]], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
