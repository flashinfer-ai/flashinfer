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

Names of the Cake SiLU-gated RMSNorm training packages.

The deterministic namespaces (the ``flashinfer.cake_rmsnorm_silu_gate_det_train`` facade and its backend package)
bind only ``det`` names: the names the package had before it took its ``det`` names are not attributes of either
namespace (no alias, no module ``__getattr__``), and every name of their ``__all__`` but the two architecture
constants carries ``det``.  The names without ``det`` at the ``flashinfer`` top level, the module
``flashinfer.cake_rmsnorm_silu_gate_train`` and the package ``flashinfer.cake_rmsnorm_silu_gate_train_backend`` are the
non-deterministic package: their own objects and modules (``deterministic=False`` only;
``test_cake_rmsnorm_silu_gate_train_api.py`` holds the mode rules).  Both non-deterministic module paths are also
attributes of the ``flashinfer`` package after a plain ``import flashinfer`` (checked in fresh interpreters).
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
import flashinfer.cake_rmsnorm_silu_gate_train as nd_facade
import flashinfer.cake_rmsnorm_silu_gate_train_backend as nd_backend

pytestmark = pytest.mark.cpu_only

ND_FACADE = "flashinfer.cake_rmsnorm_silu_gate_train"
DET_FACADE = "flashinfer.cake_rmsnorm_silu_gate_det_train"
ND_PACKAGE = "flashinfer.cake_rmsnorm_silu_gate_train_backend"
DET_PACKAGE = "flashinfer.cake_rmsnorm_silu_gate_det_train_backend"
SUBMODULES = ("cake_build", "cake_jit")

# (name without det, det name): the non-deterministic package's names and the
# deterministic package's names of the same work.  The names without det were
# the deterministic package's names before it took its det names; they are not
# attributes of the deterministic namespaces.
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
# The names without det the ``flashinfer`` top level exports.
TOP_LEVEL_NAMES = (
    "CakeRMSNormSiLUGateFunction",
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_forward",
)
NAMESPACES = {"facade": facade, "backend": backend}
# The names of a det ``__all__`` that are not entry points.
CONSTANTS = {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"}


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("name", sorted(NAMES))
def test_det_modules_bind_no_pre_det_name(namespace: str, name: str) -> None:
    module = NAMESPACES[namespace]
    assert not hasattr(module, name), (namespace, name)
    assert name not in vars(module), (namespace, name)
    assert name not in module.__all__, (namespace, name)
    assert hasattr(module, NAMES[name]), (namespace, NAMES[name])


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_det_modules_have_no_name_shim(namespace: str) -> None:
    # No module ``__getattr__`` (PEP 562) that could resolve a removed name.
    assert "__getattr__" not in vars(NAMESPACES[namespace]), namespace


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_det_all_lists_only_det_names(namespace: str) -> None:
    module = NAMESPACES[namespace]
    names = set(module.__all__)
    assert names >= CONSTANTS, namespace
    assert set(NAMES.values()) <= names, namespace
    for name in names - CONSTANTS:
        assert "det" in name.lower(), (namespace, name)


def test_det_star_import_binds_only_det_names() -> None:
    for package in (DET_FACADE, DET_PACKAGE):
        names: dict = {}
        exec(f"from {package} import *", names)
        del names["__builtins__"]
        assert not set(NAMES) & set(names), package
        assert sorted(names) == sorted(importlib.import_module(package).__all__)


@pytest.mark.parametrize("name", TOP_LEVEL_NAMES)
def test_top_level_name_without_det_is_the_nondeterministic_object(name: str) -> None:
    det = NAMES[name]
    assert getattr(flashinfer, name) is getattr(nd_facade, name), name
    assert getattr(flashinfer, name) is not getattr(flashinfer, det), (name, det)
    assert getattr(flashinfer, det) is getattr(facade, det), det


def test_det_entry_points_are_the_decorated_defs() -> None:
    # One ``flashinfer_api`` wrapper per entry point, under its det name.
    for name in (
        "cake_rmsnorm_silu_gate_det",
        "cake_rmsnorm_silu_gate_det_train_backward",
        "cake_rmsnorm_silu_gate_det_train_forward",
    ):
        assert getattr(facade, name).__name__ == name


# The ``__all__`` of the non-deterministic facade and backend package: the names
# without det and the workspace reset of the non-deterministic backward.
ND_ALL = sorted(
    [
        "ARCHES",
        "SUPPORTED_COMPUTE_CAPABILITIES",
        "cake_rmsnorm_silu_gate_train_reset_workspace",
        *NAMES,
    ]
)
# The non-deterministic facade's own entry points (``flashinfer_api`` wrappers
# that check ``deterministic``); its other names, the autograd function
# included, are the backend package's objects.
ND_FACADE_DEFS = (
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_forward",
    "cake_rmsnorm_silu_gate_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_train_supported",
)


def test_nondeterministic_facade_is_its_own_module() -> None:
    assert importlib.import_module(ND_FACADE) is nd_facade
    assert sys.modules[ND_FACADE] is not sys.modules[DET_FACADE]
    assert Path(nd_facade.__file__).name == "cake_rmsnorm_silu_gate_train.py"
    assert sorted(nd_facade.__all__) == ND_ALL
    for name in nd_facade.__all__:
        assert (getattr(nd_facade, name) is getattr(nd_backend, name)) != (
            name in ND_FACADE_DEFS
        ), name
    assert (
        nd_facade.CakeRMSNormSiLUGateFunction is nd_backend.CakeRMSNormSiLUGateFunction
    )
    # the det names live in the det facade only
    assert not hasattr(nd_facade, "cake_rmsnorm_silu_gate_det_train_forward")


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_nondeterministic_package_submodule_is_its_own_module(submodule: str) -> None:
    nd = importlib.import_module(f"{ND_PACKAGE}.{submodule}")
    det = importlib.import_module(f"{DET_PACKAGE}.{submodule}")
    assert nd is not det, submodule
    assert sys.modules[f"{ND_PACKAGE}.{submodule}"] is nd, submodule
    assert Path(nd.__file__).parent == Path(nd_backend.__file__).parent, submodule


def test_nondeterministic_package_is_self_contained() -> None:
    assert sorted(nd_backend.__all__) == ND_ALL
    assert not hasattr(nd_backend, "cake_rmsnorm_silu_gate_det_train_forward")
    # its own registry: the det forward program renamed, the one-kernel backward in both dw dtypes
    # and the long-input modules
    loader = nd_backend.cake_jit
    assert loader.MODULES and all(
        name.startswith("cake_rmsnorm_silu_gate_train_") for name in loader.MODULES
    )
    roles = {record["role"] for record in loader.MODULES.values()}
    assert roles == (
        set(loader.ROLES)
        | set(loader.XLONG_ROLES)
        | set(loader.LONG_ROLES)
        | set(loader.MID_ROLES)
        | set(loader.SHORT_ROLES)
    )
    nd_dir = Path(nd_backend.__file__).parent
    files = {p.name for p in nd_dir.iterdir() if p.name != "__pycache__"}
    assert {
        "__init__.py",
        "cake_build.py",
        "cake_jit.py",
        "build.json",
        "csrc",
    } <= files


def test_nondeterministic_package_star_import_binds_its_own_objects() -> None:
    names: dict = {}
    exec(f"from {ND_PACKAGE} import *", names)
    del names["__builtins__"]
    assert sorted(names) == ND_ALL
    for name in ND_ALL:
        assert names[name] is getattr(nd_backend, name), name


def test_nondeterministic_package_dir_lists_its_own_names() -> None:
    listed = set(dir(nd_backend))
    assert set(ND_ALL) | set(SUBMODULES) <= listed
    assert not set(NAMES.values()) & listed


# Each import form of the non-deterministic package in a fresh interpreter: the
# submodule path first, then the attributes of flashinfer and a star import of
# the package.
FRESH = {
    "nd_submodule_first": (
        "import importlib, sys\n"
        f"nd = importlib.import_module('{ND_PACKAGE}.cake_jit')\n"
        f"det = importlib.import_module('{DET_PACKAGE}.cake_jit')\n"
        "assert nd is not det and nd.MODULES is not det.MODULES\n"
        f"assert importlib.import_module('{ND_FACADE}') is not importlib.import_module('{DET_FACADE}')\n"
    ),
    "flashinfer_attributes": (
        "import sys\nimport flashinfer\n"
        f"assert flashinfer.{ND_FACADE.rpartition('.')[2]} is sys.modules['{ND_FACADE}']\n"
        f"assert flashinfer.{ND_PACKAGE.rpartition('.')[2]} is sys.modules['{ND_PACKAGE}']\n"
        f"assert flashinfer.{PAIRS[0][0]} is sys.modules['{ND_FACADE}'].{PAIRS[0][0]}\n"
        f"assert not hasattr(sys.modules['{DET_FACADE}'], '{PAIRS[0][0]}')\n"
    ),
    "nd_package_star_import": (
        "import importlib\n"
        f"from {ND_PACKAGE} import *\n"
        f"nd = importlib.import_module('{ND_PACKAGE}')\n"
        "assert all(globals()[n] is getattr(nd, n) for n in nd.__all__)\n"
        "assert 'cake_rmsnorm_silu_gate_det_train_forward' not in globals()\n"
    ),
}


@pytest.mark.parametrize("form", sorted(FRESH))
def test_import_forms_in_a_fresh_interpreter(form: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH[form]], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
