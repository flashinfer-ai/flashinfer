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

Names of the Cake fused SwiGLU + MXFP8 quantization training family.

The family has one mode, ``det``: the facade
``flashinfer.cake_gated_act_mxfp8_det_train`` (top level) and its backend
package ``flashinfer.experimental.cake.cake_gated_act_mxfp8_det_train``.  Every
public name but the module constants (``ARCHES``,
``SUPPORTED_COMPUTE_CAPABILITIES``, ``ALIGNMENT_BYTES``) carries ``det`` right
after the op: functions are named by the module plus a verb, the support query
is ``is_cake_gated_act_mxfp8_det_train_supported`` and the output tuple type is
``CakeGatedActMXFP8DetOutputs``.  No entry point has a ``deterministic``
parameter.  The modules and names the family had before
(``flashinfer.gated_act_mxfp8``, the package
``flashinfer.cake_gated_act_mxfp8_backend``, ``silu_and_mul_mxfp8_quantize``
and the other names without a mode word) fail at import: no alias, no compat
module.  The facade imports its backend package on first use, so ``import
flashinfer`` loads no module of it; the family's ``flashinfer`` top-level names
are rows of the lazy-export table ``flashinfer._LAZY_EXPORTS``.  The shared
checks of the mode-in-name rule (``tests/cake/test_helpers/cake_mode_names.py``) run
over both namespaces; the checks here add what is particular to this family.
The JIT library names (``cake_gated_act_mxfp8_<hash>``) are build identities,
not Python names, and are unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_gated_act_mxfp8_det_train as facade
import flashinfer.experimental.cake.cake_gated_act_mxfp8_det_train as backend
from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

pytestmark = pytest.mark.cpu_only

MODULE = "cake_gated_act_mxfp8_det_train"
FACADE = f"flashinfer.{MODULE}"
BACKEND = f"flashinfer.experimental.cake.{MODULE}"
OUTPUTS = "CakeGatedActMXFP8DetOutputs"
# The facade's functions (its own ``flashinfer_api`` defs) and the backend
# functions three of them call.
FACADE_FUNCTIONS = {
    f"{MODULE}_forward",
    f"{MODULE}_backward",
    f"is_{MODULE}_supported",
    f"{MODULE}_register_custom_ops",
}
FORWARDED = sorted(FACADE_FUNCTIONS - {f"{MODULE}_register_custom_ops"})
PUBLIC = {
    "facade": FACADE_FUNCTIONS | {OUTPUTS},
    "backend": {
        OUTPUTS,
        f"{MODULE}_forward",
        f"{MODULE}_backward",
        f"is_{MODULE}_supported",
        f"{MODULE}_check_inputs",
        f"{MODULE}_empty_outputs",
        f"{MODULE}_logical_outputs",
    },
}
CONSTANTS = {
    "facade": {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"},
    "backend": {"ALIGNMENT_BYTES", "ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"},
}
NAMESPACES = {"facade": facade, "backend": backend}
# The family's rows of ``flashinfer._LAZY_EXPORTS``: name -> (module, attribute).
ROWS = {
    MODULE: (f".{MODULE}", None),
    **{name: (f".{MODULE}", name) for name in FORWARDED},
}
# The modules and packages the family had before the mode-in-name rule.
OLD_MODULES = ("flashinfer.gated_act_mxfp8", "flashinfer.cake_gated_act_mxfp8_backend")
# The names the facade (and, for the two entry points, the top level) had.
OLD_FACADE_NAMES = (
    "GatedActMxfp8Outputs",
    "get_gated_act_mxfp8_module",
    "silu_and_mul_mxfp8_quantize",
    "silu_and_mul_mxfp8_quantize_backward",
)
# The names the backend package had.
OLD_BACKEND_NAMES = (
    "GatedActMxfp8Outputs",
    "check_inputs",
    "empty_outputs",
    "is_cake_gated_act_mxfp8_supported",
    "logical_outputs",
    "silu_and_mul_mxfp8_quantize",
    "silu_and_mul_mxfp8_quantize_backward",
)
# A name of the family: it names the op or one of its former entry points (not
# the unrelated ``silu_and_mul_mxfp8_experts_quantize``).
FAMILY = re.compile(
    r"gated_act_mxfp8|GatedActMXFP8|GatedActMxfp8|silu_and_mul_mxfp8_quantize"
)


def test_all_is_the_mode_names_and_the_constants() -> None:
    for kind, module in NAMESPACES.items():
        assert set(module.__all__) == PUBLIC[kind] | CONSTANTS[kind], kind
        for name in module.__all__:
            assert hasattr(module, name), (kind, name)


@pytest.mark.parametrize("kind", sorted(NAMESPACES))
def test_shared_mode_name_checks(kind: str) -> None:
    """The shared checks of the mode-in-name rule (``assert_cake_mode_module``):
    the mode word and its place in every public name; no entry point binds a
    ``deterministic`` keyword; the namespace, the ``flashinfer`` top level (its
    lazy-export rows included) and, for the facade, its backend package bind
    each object only under its own name; and in a fresh interpreter the former
    modules and names fail at import."""
    if kind == "facade":
        assert_cake_mode_module(
            facade,
            namespaces=[flashinfer, backend],
            forwards=[(name, getattr(backend, name)) for name in FORWARDED],
            old_modules=OLD_MODULES,
            old_names=[
                (path, name)
                for path in ("flashinfer", FACADE)
                for name in OLD_FACADE_NAMES
            ],
        )
    else:
        assert_cake_mode_module(
            backend,
            namespaces=[flashinfer],
            old_names=[(BACKEND, name) for name in OLD_BACKEND_NAMES],
        )


@pytest.mark.parametrize("kind", sorted(NAMESPACES))
def test_old_names_are_not_attributes(kind: str) -> None:
    module = NAMESPACES[kind]
    for name in (*OLD_FACADE_NAMES, *OLD_BACKEND_NAMES):
        assert not hasattr(module, name), (kind, name)
        assert name not in vars(module), (kind, name)


def test_no_name_shim() -> None:
    """The backend package has no module ``__getattr__``; the facade's resolves
    exactly the two architecture constants of the backend and raises
    ``AttributeError`` for anything else."""
    assert "__getattr__" not in vars(backend)
    assert facade._BACKEND_OBJECTS == ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES")
    for name in facade._BACKEND_OBJECTS:
        assert facade.__getattr__(name) is getattr(backend, name), name
    for name in (*OLD_FACADE_NAMES, *OLD_BACKEND_NAMES, "deterministic", "_run"):
        with pytest.raises(AttributeError):
            facade.__getattr__(name)


def test_facade_objects() -> None:
    """The facade's functions are its own ``flashinfer_api`` defs, named as
    exported; its constants are the backend's objects."""
    for name in FACADE_FUNCTIONS:
        obj = getattr(facade, name)
        assert obj.__name__ == name, name
        if name in FORWARDED:
            assert obj is not getattr(backend, name), name
    for name in CONSTANTS["facade"]:
        assert getattr(facade, name) is getattr(backend, name), name


def test_modules_sit_at_their_places() -> None:
    """The facade sits at the ``flashinfer`` top level, the backend package at
    ``flashinfer/experimental/cake/<module>/`` with its own build files."""
    root = Path(flashinfer.__file__).parent
    assert Path(facade.__file__) == root / f"{MODULE}.py"
    package = root / "experimental" / "cake" / MODULE
    assert Path(backend.__file__) == package / "__init__.py"
    files = {p.name for p in package.iterdir() if p.name != "__pycache__"}
    assert {
        "__init__.py",
        "README.md",
        "build.json",
        "cake_build.py",
        "cake_jit.py",
        "csrc",
    } <= files
    for submodule in ("cake_build", "cake_jit"):
        loaded = importlib.import_module(f"{BACKEND}.{submodule}")
        assert Path(loaded.__file__).parent == package, submodule
    # the JIT library names (build identities) are unchanged
    assert backend.cake_jit.MODULES and all(
        re.fullmatch(r"cake_gated_act_mxfp8_[0-9a-f]{20}", name)
        for name in backend.cake_jit.MODULES
    )


@pytest.mark.parametrize("module", OLD_MODULES)
def test_old_module_fails_at_import(module: str) -> None:
    assert importlib.util.find_spec(module) is None, module
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
    assert not hasattr(flashinfer, module.rpartition(".")[2]), module


def test_top_level_binds_only_mode_names() -> None:
    """The family's top-level names are exactly its lazy-export rows, and each
    resolves to the facade's object."""
    rows = {
        name: tuple(row)
        for name, row in flashinfer._LAZY_EXPORTS.items()
        if FAMILY.search(name)
    }
    assert rows == ROWS, sorted(rows)
    assert {name for name in dir(flashinfer) if FAMILY.search(name)} == set(ROWS)
    assert {name for name in vars(flashinfer) if FAMILY.search(name)} <= set(ROWS)
    for name in FORWARDED:
        assert getattr(flashinfer, name) is getattr(facade, name), name
    assert getattr(flashinfer, MODULE) is facade


# A plain ``import flashinfer`` binds none of the family's names (each binds on
# first access) and loads neither the facade nor the backend package; importing
# the facade still loads no backend module, and its first backend access does.
FRESH = (
    "import re, sys, flashinfer\n"
    "def family(names):\n"
    f"    return sorted(n for n in names if re.search({FAMILY.pattern!r}, n))\n"
    "def loaded(prefix):\n"
    "    return sorted(m for m in sys.modules if m.startswith(prefix))\n"
    f"assert family(vars(flashinfer)) == [], family(vars(flashinfer))\n"
    f"assert loaded({FACADE!r}) == [] and loaded({BACKEND!r}) == []\n"
    f"assert family(dir(flashinfer)) == {sorted(ROWS)!r}, family(dir(flashinfer))\n"
    f"import {FACADE} as facade\n"
    f"assert loaded({BACKEND!r}) == [], loaded({BACKEND!r})\n"
    "facade.ARCHES\n"
    f"assert {BACKEND!r} in loaded({BACKEND!r}), loaded({BACKEND!r})\n"
)


def test_import_forms_in_a_fresh_interpreter() -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
