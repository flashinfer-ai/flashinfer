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

Compatibility names of the Cake causal conv1d + SiLU training package.

Within each namespace (the ``flashinfer.cake_causal_conv1d_det_train`` facade, its
backend package and the ``flashinfer`` top level) every compatibility name is the
same object as its det name and stays out of ``__all__``.  The module
``flashinfer.cake_causal_conv1d_train`` is the facade's module object, and the
submodules of ``flashinfer.cake_causal_conv1d_train_backend`` are the det
package's module objects.  Both old module paths are also attributes of the
``flashinfer`` package after a plain ``import flashinfer``.  Importing either facade
or either package, or reaching it as such an attribute, loads no KCP host module
under either name (checked in fresh interpreters by module file, not by name).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_causal_conv1d_det_train as facade
import flashinfer.cake_causal_conv1d_det_train_backend as backend

pytestmark = pytest.mark.cpu_only

OLD_FACADE = "flashinfer.cake_causal_conv1d_train"
NEW_FACADE = "flashinfer.cake_causal_conv1d_det_train"
OLD_PACKAGE = "flashinfer.cake_causal_conv1d_train_backend"
NEW_PACKAGE = "flashinfer.cake_causal_conv1d_det_train_backend"
SUBMODULES = ("cake_build", "cake_jit", "kcp", "kcp_jit")

# (compatibility name, det name) of the facade and the backend package.
PAIRS = (
    ("CakeCausalConv1dFunction", "CakeCausalConv1dDetFunction"),
    ("cake_causal_conv1d", "cake_causal_conv1d_det"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_det_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_det_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace",
        "cake_causal_conv1d_det_train_forward_workspace",
    ),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_det_train_backward_workspace",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    ),
    (
        "is_cake_causal_conv1d_train_supported",
        "is_cake_causal_conv1d_det_train_supported",
    ),
)
NAMES = dict(PAIRS)
# The names the ``flashinfer`` top level exports.
TOP_LEVEL_NAMES = (
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace_bytes",
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
    for old in (
        "cake_causal_conv1d",
        "cake_causal_conv1d_train_forward",
        "cake_causal_conv1d_train_backward",
    ):
        assert getattr(facade, old).__name__ == NAMES[old], old
    det_class = backend.CakeCausalConv1dDetFunction
    assert det_class.__name__ == "CakeCausalConv1dDetFunction"


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
    for name in (*NAMES, *NAMES.values(), "ARCHES", "_FORWARD_WORKSPACES"):
        assert getattr(old, name) is getattr(backend, name), name
    assert old.__all__ == backend.__all__
    # The compatibility package holds Python shims only: no csrc, manifest or libraries.
    old_dir = Path(old.__file__).parent
    files = sorted(p.name for p in old_dir.iterdir() if p.name != "__pycache__")
    assert files == sorted(["__init__.py", *(f"{m}.py" for m in SUBMODULES)])


# The old module paths reached as attributes of ``flashinfer`` with no explicit import.
_ATTRIBUTE_ACCESS = """
import sys

import flashinfer

old_facade = flashinfer.{old_facade}
assert old_facade is flashinfer.{new_facade} is sys.modules["flashinfer.{new_facade}"]
old_package = flashinfer.{old_package}
assert old_package is sys.modules["flashinfer.{old_package}"]
new_package = flashinfer.{new_package}
assert old_package.{old_name} is new_package.{new_name}
"""

# Every import form of the facade and the package, each in a fresh interpreter.
IMPORTS = {
    "flashinfer_attributes": _ATTRIBUTE_ACCESS.format(
        old_facade=OLD_FACADE.rpartition(".")[2],
        new_facade=NEW_FACADE.rpartition(".")[2],
        old_package=OLD_PACKAGE.rpartition(".")[2],
        new_package=NEW_PACKAGE.rpartition(".")[2],
        old_name=PAIRS[2][0],
        new_name=PAIRS[2][1],
    ),
    "flashinfer": "import flashinfer",
    "old_facade": f"import {OLD_FACADE}",
    "new_facade": f"import {NEW_FACADE}",
    "old_package": f"import {OLD_PACKAGE} as p; p.{PAIRS[2][0]}",
    "new_package": f"import {NEW_PACKAGE}",
    "old_package_names": f"from {OLD_PACKAGE} import {PAIRS[2][1]}, cake_jit",
}

_LAZY_CHECK = """
import sys
from pathlib import Path

import {new_package} as package

kcp_files = {{(Path(package.__file__).parent / f).resolve() for f in ("kcp.py", "kcp_jit.py")}}
kcp_names = [p + "." + s for p in ({old_package!r}, {new_package!r}) for s in ("kcp", "kcp_jit")]
loaded = sorted(
    name
    for name, module in list(sys.modules.items())
    if name in kcp_names
    or (getattr(module, "__file__", None) and Path(module.__file__).resolve() in kcp_files)
)
assert not loaded, loaded
print("KCP_NOT_LOADED")
from {old_facade} import KcpRecomputeHandle as a
from {new_facade} import KcpRecomputeHandle as b
from {old_package}.kcp import KcpRecomputeHandle as c
assert a is b is c
assert sys.modules[{old_package!r} + ".kcp"] is sys.modules[{new_package!r} + ".kcp"]
print("KCP_ALIASED")
"""


@pytest.mark.parametrize("form", sorted(IMPORTS))
def test_imports_do_not_load_kcp(form: str) -> None:
    code = (
        IMPORTS[form]
        + "\n"
        + _LAZY_CHECK.format(
            old_facade=OLD_FACADE,
            new_facade=NEW_FACADE,
            old_package=OLD_PACKAGE,
            new_package=NEW_PACKAGE,
        )
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
    assert "KCP_NOT_LOADED" in r.stdout and "KCP_ALIASED" in r.stdout, (
        form,
        r.stdout,
        r.stderr,
    )
