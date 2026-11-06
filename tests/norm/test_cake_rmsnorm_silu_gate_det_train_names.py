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

Names of the Cake SiLU-gated RMSNorm training family, both modes.

The mode is part of every public name, and only of the name.  For each mode
(``det``, ``nondet``) the facade ``flashinfer.cake_rmsnorm_silu_gate_<mode>_train``
(top level) and its backend package
``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_<mode>_train`` export the
autograd functional ``cake_rmsnorm_silu_gate_<mode>``, the autograd class
``CakeRMSNormSiLUGate<Mode>Function``, the support query
``is_cake_rmsnorm_silu_gate_<mode>_train_supported`` and functions named by the
module plus a verb.  Module constants are exempt by their pattern: an
``^[A-Z][A-Z0-9_]*$`` name bound to a plain value (here ``ARCHES`` and
``SUPPORTED_COMPUTE_CAPABILITIES``); any other public name, an ALL_CAPS type
or alias included, carries the mode word.
No entry point has a ``deterministic`` parameter.  The modules and names the
family had before (``flashinfer.cake_rmsnorm_silu_gate_train``, the packages
``flashinfer.cake_rmsnorm_silu_gate_train_backend`` and
``flashinfer.cake_rmsnorm_silu_gate_det_train_backend``, and the names without
a mode word) fail at import: no alias, no compat module, and no module
``__getattr__`` but the facades' own, which resolves exactly the three
backend-owned names (the autograd class and the architecture constants).  Each
facade imports its backend package on first use, so ``import flashinfer``
loads no module of either backend package.  The ``flashinfer`` top level binds
only names of the family with a mode word (checked in this process and in
fresh interpreters).  The JIT
library names of the backends (``cake_rmsnorm_silu_gate_train_<hash>`` and
``cake_rmsnorm_silu_gate_det_train_<hash>``) are build identities, not Python
names, and are unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as det_facade
import flashinfer.cake_rmsnorm_silu_gate_nondet_train as nd_facade
import flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train as det_backend
import flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train as nd_backend

pytestmark = pytest.mark.cpu_only

OP = "cake_rmsnorm_silu_gate"
MODES = ("det", "nondet")
FACADE = {mode: f"flashinfer.{OP}_{mode}_train" for mode in MODES}
BACKEND = {mode: f"flashinfer.experimental.cake.{OP}_{mode}_train" for mode in MODES}
NAMESPACES = {
    ("det", "facade"): det_facade,
    ("det", "backend"): det_backend,
    ("nondet", "facade"): nd_facade,
    ("nondet", "backend"): nd_backend,
}
NAMESPACE_IDS = [f"{mode}-{kind}" for mode, kind in NAMESPACES]
SUBMODULES = ("cake_build", "cake_jit")
# Module constants carry no mode word (the module path carries the mode): a
# name of this pattern bound to a plain value.  An ALL_CAPS name bound to
# anything else (a type, a type alias, a function) is not a constant.
CONSTANT = re.compile(r"[A-Z][A-Z0-9_]*")
PLAIN = (bool, int, float, str, bytes, tuple, frozenset, list, dict, set, type(None))
CLASS = {
    "det": "CakeRMSNormSiLUGateDetFunction",
    "nondet": "CakeRMSNormSiLUGateNondetFunction",
}


def _public_names(mode: str) -> set:
    module = f"{OP}_{mode}_train"
    names = {
        f"{OP}_{mode}",
        CLASS[mode],
        f"is_{module}_supported",
        f"{module}_forward",
        f"{module}_backward",
        f"{module}_backward_workspace",
        f"{module}_backward_workspace_bytes",
    }
    if mode == "nondet":
        names.add(f"{module}_reset_workspace")
    return names


PUBLIC = {mode: _public_names(mode) for mode in MODES}
# The names of the family the ``flashinfer`` top level binds (the autograd
# classes are backend objects: attributes of the facade modules only).
TOP_LEVEL = {
    mode: {
        f"{OP}_{mode}",
        f"{OP}_{mode}_train_forward",
        f"{OP}_{mode}_train_backward",
        f"{OP}_{mode}_train_backward_workspace_bytes",
    }
    for mode in MODES
}
# The modules and packages the family had before the mode-in-name rule.
OLD_MODULES = (
    "flashinfer.cake_rmsnorm_silu_gate_train",
    "flashinfer.cake_rmsnorm_silu_gate_train_backend",
    "flashinfer.cake_rmsnorm_silu_gate_det_train_backend",
)
# The names the family had without a mode word.
OLD_NAMES = (
    "CakeRMSNormSiLUGateFunction",
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_forward",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_train_supported",
)
# A name of the family: it names the op.
FAMILY = re.compile(r"rmsnorm_silu_gate|RMSNormSiLUGate")


def _mode_words(name: str) -> list:
    """The mode words of a name: its ``_``-separated words for a snake-case
    name, its capitalized words for a CamelCase one."""
    if name[:1].isupper():
        words = [w.lower() for w in re.findall(r"[A-Z][a-z0-9]*", name)]
    else:
        words = name.split("_")
    return [w for w in words if w in MODES]


def _constants(module) -> set:
    """The module constants of ``module.__all__``."""
    return {
        name
        for name in module.__all__
        if CONSTANT.fullmatch(name) and isinstance(getattr(module, name), PLAIN)
    }


def _entries(module) -> dict:
    """The callables of ``module.__all__``: functions, and the autograd class's
    ``forward`` and ``backward``."""
    entries = {}
    for name in set(module.__all__) - _constants(module):
        obj = getattr(module, name)
        if inspect.isclass(obj):
            entries[f"{name}.forward"] = obj.forward
            entries[f"{name}.backward"] = obj.backward
        else:
            entries[name] = obj
    return entries


# --------------------------------------------------------------------------- the four namespaces


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_all_is_the_mode_names_and_the_constants(key) -> None:
    mode, _ = key
    module = NAMESPACES[key]
    assert set(module.__all__) - _constants(module) == PUBLIC[mode], key
    for name in PUBLIC[mode]:
        assert hasattr(module, name), (key, name)


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_every_public_name_carries_its_mode(key) -> None:
    """Every public name but the module constants carries exactly one mode
    word, its package's; the names follow rules 2 to 6 and 10 of the scheme."""
    mode, _ = key
    module = NAMESPACES[key]
    prefix = f"{OP}_{mode}_train"
    for name in set(module.__all__) - _constants(module):
        assert _mode_words(name) == [mode], (key, name)
        obj = getattr(module, name)
        if inspect.isclass(obj):
            assert name == CLASS[mode], (key, name)
        else:
            assert name in (f"{OP}_{mode}", f"is_{prefix}_supported") or (
                name.startswith(f"{prefix}_")
            ), (key, name)
            assert not {"fwd", "bwd"} & set(name.split("_")), (key, name)
    for name in vars(module):
        if FAMILY.search(name) and not name.startswith("_"):
            assert _mode_words(name) == [mode], (key, name)


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_no_entry_has_a_deterministic_parameter(key) -> None:
    entries = _entries(NAMESPACES[key])
    assert len(entries) == len(PUBLIC[key[0]]) + 1, key
    for name, entry in entries.items():
        assert "deterministic" not in inspect.signature(entry).parameters, (key, name)
    cls = getattr(NAMESPACES[key], CLASS[key[0]])
    assert list(inspect.signature(cls.forward).parameters) == [
        "ctx",
        "x",
        "z",
        "weight",
        "eps",
    ], key


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_old_names_are_not_attributes(key) -> None:
    module = NAMESPACES[key]
    for name in OLD_NAMES:
        assert not hasattr(module, name), (key, name)
        assert name not in vars(module), (key, name)
    other = "nondet" if key[0] == "det" else "det"
    for name in PUBLIC[other]:
        assert not hasattr(module, name), (key, name)


# The backend package's objects a facade resolves on first access.
BACKEND_OBJECTS = {
    mode: ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", CLASS[mode]) for mode in MODES
}


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_no_name_shim(key) -> None:
    """No module ``__getattr__`` (PEP 562) that could resolve a removed name: a
    backend package has none; a facade's resolves exactly its three
    backend-owned names and raises ``AttributeError`` for anything else."""
    mode, kind = key
    module = NAMESPACES[key]
    if kind == "backend":
        assert "__getattr__" not in vars(module), key
        return
    assert BACKEND_OBJECTS[mode] == module._BACKEND_OBJECTS, key
    backend = importlib.import_module(BACKEND[mode])
    for name in BACKEND_OBJECTS[mode]:
        assert module.__getattr__(name) is getattr(backend, name), (key, name)
    other = "nondet" if mode == "det" else "det"
    for name in (*OLD_NAMES, CLASS[other], "deterministic", "_forward"):
        with pytest.raises(AttributeError):
            module.__getattr__(name)


@pytest.mark.parametrize("key", list(NAMESPACES), ids=NAMESPACE_IDS)
def test_star_import_binds_only_the_mode_names(key) -> None:
    module = NAMESPACES[key]
    names: dict = {}
    exec(f"from {module.__name__} import *", names)
    del names["__builtins__"]
    assert sorted(names) == sorted(module.__all__), key
    for name in names:
        assert names[name] is getattr(module, name), (key, name)


@pytest.mark.parametrize("mode", MODES)
def test_modules_sit_at_their_places(mode: str) -> None:
    """The facade sits at the ``flashinfer`` top level, the backend package at
    ``flashinfer/experimental/cake/<module>/`` with its own build files."""
    root = Path(flashinfer.__file__).parent
    facade = importlib.import_module(FACADE[mode])
    backend = importlib.import_module(BACKEND[mode])
    assert Path(facade.__file__) == root / f"{OP}_{mode}_train.py"
    package = root / "experimental" / "cake" / f"{OP}_{mode}_train"
    assert Path(backend.__file__) == package / "__init__.py"
    files = {p.name for p in package.iterdir() if p.name != "__pycache__"}
    assert {"__init__.py", "cake_build.py", "cake_jit.py", "build.json", "csrc"} <= (
        files
    )
    for submodule in SUBMODULES:
        loaded = importlib.import_module(f"{BACKEND[mode]}.{submodule}")
        assert Path(loaded.__file__).parent == package, (mode, submodule)


def test_the_two_modes_are_separate_modules() -> None:
    assert det_facade is not nd_facade and det_backend is not nd_backend
    for submodule in SUBMODULES:
        det = importlib.import_module(f"{BACKEND['det']}.{submodule}")
        nd = importlib.import_module(f"{BACKEND['nondet']}.{submodule}")
        assert det is not nd, submodule
    assert nd_backend.cake_jit.MODULES is not det_backend.cake_jit.MODULES
    # the JIT library names (build identities) are unchanged
    assert nd_backend.cake_jit.MODULES and all(
        name.startswith("cake_rmsnorm_silu_gate_train_")
        for name in nd_backend.cake_jit.MODULES
    )
    assert det_backend.cake_jit.MODULES and all(
        name.startswith("cake_rmsnorm_silu_gate_det_train_")
        for name in det_backend.cake_jit.MODULES
    )
    loader = nd_backend.cake_jit
    roles = {record["role"] for record in loader.MODULES.values()}
    assert roles == (
        set(loader.ROLES)
        | set(loader.XLONG_ROLES)
        | set(loader.LONG_ROLES)
        | set(loader.MID_ROLES)
        | set(loader.SHORT_ROLES)
    )


def test_facade_objects() -> None:
    """Each facade's functions are its own ``flashinfer_api`` defs; its other
    names (the autograd class and the architecture constants) are its
    backend's objects."""
    for facade, backend, mode in (
        (det_facade, det_backend, "det"),
        (nd_facade, nd_backend, "nondet"),
    ):
        defs = PUBLIC[mode] - {CLASS[mode]}
        for name in facade.__all__:
            own = getattr(facade, name) is not getattr(backend, name)
            assert own == (name in defs), (facade.__name__, name)
            if name in defs:
                assert getattr(facade, name).__name__ == name, name


# --------------------------------------------------------------------------- old modules and the top level


@pytest.mark.parametrize("module", OLD_MODULES)
def test_old_module_fails_at_import(module: str) -> None:
    assert importlib.util.find_spec(module) is None, module
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{module}.cake_jit")
    assert not hasattr(flashinfer, module.rpartition(".")[2]), module


@pytest.mark.parametrize("name", OLD_NAMES)
def test_old_name_fails_at_import(name: str) -> None:
    assert not hasattr(flashinfer, name), name
    for module in ("flashinfer", FACADE["det"], FACADE["nondet"]):
        with pytest.raises(ImportError):
            exec(f"from {module} import {name}", {})


def test_top_level_binds_only_mode_names() -> None:
    family = {name for name in vars(flashinfer) if FAMILY.search(name)}
    modules = {f"{OP}_{mode}_train" for mode in MODES}
    assert family == TOP_LEVEL["det"] | TOP_LEVEL["nondet"] | modules, sorted(family)
    for mode in MODES:
        facade = importlib.import_module(FACADE[mode])
        for name in TOP_LEVEL[mode]:
            assert getattr(flashinfer, name) is getattr(facade, name), name
        assert getattr(flashinfer, f"{OP}_{mode}_train") is facade


# Each import form in a fresh interpreter: a plain ``import flashinfer`` (the
# family's names at the top level), the old modules and names (each fails), and
# the backend submodules first.
FRESH = {
    "top_level": (
        "import re, flashinfer\n"
        "family = sorted(n for n in vars(flashinfer) if re.search("
        "r'rmsnorm_silu_gate|RMSNormSiLUGate', n))\n"
        f"want = sorted({sorted(TOP_LEVEL['det'] | TOP_LEVEL['nondet'] | {f'{OP}_{m}_train' for m in MODES})!r})\n"
        "assert family == want, family\n"
    ),
    "old_modules": (
        "import importlib\n"
        f"for m in {list(OLD_MODULES)!r}:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except ModuleNotFoundError:\n"
        "        continue\n"
        "    raise AssertionError(m)\n"
    ),
    "old_names": (
        f"for n in {list(OLD_NAMES)!r}:\n"
        f"    for m in {['flashinfer', FACADE['det'], FACADE['nondet']]!r}:\n"
        "        try:\n"
        "            exec(f'from {m} import {n}', {})\n"
        "        except ImportError:\n"
        "            continue\n"
        "        raise AssertionError((m, n))\n"
    ),
    "backends_load_on_first_use": (
        "import sys, flashinfer\n"
        "def loaded():\n"
        "    return sorted(m for m in sys.modules if m.startswith("
        f"('{BACKEND['det']}', '{BACKEND['nondet']}')))\n"
        "assert loaded() == [], loaded()\n"
        f"import {FACADE['nondet']} as nd\n"
        "assert loaded() == [], loaded()\n"
        f"nd.{CLASS['nondet']}\n"
        f"assert '{BACKEND['nondet']}' in loaded() and not any(m.startswith('{BACKEND['det']}') for m in loaded()), loaded()\n"
    ),
    "backend_submodule_first": (
        "import importlib, sys\n"
        f"nd = importlib.import_module('{BACKEND['nondet']}.cake_jit')\n"
        f"det = importlib.import_module('{BACKEND['det']}.cake_jit')\n"
        "assert nd is not det and nd.MODULES is not det.MODULES\n"
        f"assert importlib.import_module('{FACADE['nondet']}') is not "
        f"importlib.import_module('{FACADE['det']}')\n"
    ),
}


@pytest.mark.parametrize("form", sorted(FRESH))
def test_import_forms_in_a_fresh_interpreter(form: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH[form]], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
