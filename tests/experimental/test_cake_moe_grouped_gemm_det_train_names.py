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

Names of the deterministic Cake MoE grouped GEMM.

The module ``flashinfer.cake_moe_grouped_gemm_det_train`` and its backend package
``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train`` name every public object after the
module and its mode: every public name but a constant carries ``det``, no
callable takes a ``deterministic`` argument or a catch-all ``**kwargs``, and the
import paths and names this family had before fail at import (no alias, shim or
redirect).  CPU only: nothing here launches a kernel.
"""

import importlib
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("tvm_ffi")  # the backend package imports it at module level

MODULE = "cake_moe_grouped_gemm_det_train"
FACADE = f"flashinfer.{MODULE}"
PACKAGE = f"flashinfer.experimental.cake.{MODULE}"
# public name -> the backend submodule that defines it ("" = the package root)
PUBLIC = {
    f"{MODULE}_forward": "",
    f"{MODULE}_dgrad": "",
    f"{MODULE}_wgrad": "",
    f"{MODULE}_prepare_forward": "cake_backend",
    f"{MODULE}_prepare_dgrad": "cake_backend",
    f"{MODULE}_prepare_wgrad": "cake_backend",
    "cake_moe_grouped_gemm_det": "cake_autograd",
    "CakeMoEGroupedGemmDetFunction": "cake_autograd",
    "CakeMoEGroupedGemmDetLaunch": "cake_backend",
}
SUPPORT = f"is_{MODULE}_supported"
MODE = "det"  # the mode token of every public name

# The import paths of this family before #8928 and before its deterministic-family
# rename (no alias module).
REMOVED_MODULES = (
    "flashinfer.experimental.cake_moe_grouped_det_gemm",
    "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_autograd",
    "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_backend",
    "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_build",
    "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_jit",
    "flashinfer.grouped_mm.cake_det",
    "flashinfer.experimental.cake_moe_grouped_gemm",
    "flashinfer.experimental.cake_moe_grouped_gemm.cake_autograd",
    "flashinfer.experimental.cake_moe_grouped_gemm.cake_backend",
    "flashinfer.experimental.cake_moe_grouped_gemm.cake_build",
    "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit",
    "flashinfer.grouped_mm.cake",
)
# The names of the deterministic objects before #8928, and the det-free aliases
# they had before the deterministic-family rename.
FORMER_NAMES = (
    "grouped_det_gemm_fwd",
    "grouped_det_gemm_dgrad",
    "grouped_det_gemm_wgrad",
    "prepare_grouped_det_gemm_fwd",
    "prepare_grouped_det_gemm_dgrad",
    "prepare_grouped_det_gemm_wgrad",
    "GroupedDetGemmLaunch",
    "cake_grouped_det_mm",
    "CakeGroupedDetMm",
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "prepare_grouped_gemm_fwd",
    "prepare_grouped_gemm_dgrad",
    "prepare_grouped_gemm_wgrad",
    "GroupedGemmLaunch",
    "cake_grouped_mm",
    "CakeGroupedMm",
)
# module -> the former names that it alone had
EXTRA_FORMER_NAMES = {
    f"{PACKAGE}.cake_backend": ("grouped_mm_bf16_cake",),
}
# The modules that must have none of the former names.
NAME_HOSTS = (
    "flashinfer",
    "flashinfer.grouped_mm",
    FACADE,
    PACKAGE,
    f"{PACKAGE}.cake_autograd",
    f"{PACKAGE}.cake_backend",
)
FORMER_HELPER_KEYS = (
    "cake_moe_grouped_det_gemm",
    "cake_moe_grouped_gemm",
)
# Hand-offs into the package outside the ``__all__`` lists: the stable
# ``grouped_mm_bf16(..., backend="cake")`` and the backend function it calls.
HANDOFFS = (
    ("flashinfer.grouped_mm", "grouped_mm_bf16"),
    (f"{PACKAGE}.cake_backend", "grouped_det_mm_bf16_cake"),
)


def _module(name):
    return importlib.import_module(name)


def _tokens(name):
    """The lower-case tokens of a snake_case or CamelCase name (``nondet`` is one
    token, so it does not count as ``det``)."""
    words = []
    for part in name.split("_"):
        words += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+|[A-Z]+", part)
    return [word.lower() for word in words]


def _public_objects():
    """``(where, name, object)`` of every name of the module's and the package's
    ``__all__``."""
    found = []
    for where in (FACADE, PACKAGE):
        module = _module(where)
        found += [(where, name, getattr(module, name)) for name in module.__all__]
    return found


def _leftover_directory(path):
    """The directory that the former import path ``path`` names in this checkout (it
    exists only as a development leftover)."""
    root = Path(_module("flashinfer").__file__).parent
    return root.joinpath(*path.split(".")[1:])


@pytest.mark.parametrize("path", REMOVED_MODULES)
def test_removed_module_paths_do_not_import(path):
    """Every former import path raises ``ModuleNotFoundError``.  Only where a
    development checkout keeps a directory at a former package path (untracked
    ``__pycache__`` or ignored ``aot``), which Python imports as an empty namespace
    package, the check accepts that leftover if it holds no module (``git clean -fdX``
    of the path removes it)."""
    if not _leftover_directory(path).is_dir():
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(path)
        return
    try:
        spec = importlib.util.find_spec(path)
    except ModuleNotFoundError:
        return
    if spec is None:
        return
    assert spec.origin is None, (path, spec.origin)
    for location in spec.submodule_search_locations or ():
        assert not list(Path(location).glob("*.py")), location


@pytest.mark.parametrize("module_name", NAME_HOSTS)
def test_former_names_are_gone(module_name):
    """No former name resolves on the module, the package and its submodules,
    ``flashinfer.grouped_mm`` or ``flashinfer``: ``getattr`` raises
    ``AttributeError``, ``from ... import`` raises ``ImportError``, and neither
    ``__all__`` nor ``dir()`` lists it."""
    module = _module(module_name)
    names = FORMER_NAMES + EXTRA_FORMER_NAMES.get(module_name, ())
    for name in names:
        with pytest.raises(AttributeError):
            getattr(module, name)
        with pytest.raises(ImportError):
            exec(f"from {module_name} import {name}", {})
        assert name not in getattr(module, "__all__", ()), (module_name, name)
        assert name not in dir(module), (module_name, name)


def test_former_test_helper_keys_are_gone():
    """The test helper knows the package by its module name only."""
    from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

    for key in FORMER_HELPER_KEYS:
        assert key not in PACKAGES, key
        with pytest.raises(KeyError):
            cake_package(key)
    assert cake_package(MODULE) is _module(PACKAGE)


def test_the_public_names():
    """The module lists the nine objects of the package and the support query; the
    package lists the nine objects."""
    assert sorted(_module(FACADE).__all__) == sorted([*PUBLIC, SUPPORT])
    assert sorted(_module(PACKAGE).__all__) == sorted(PUBLIC)


def test_every_public_name_carries_the_mode():
    """Every name of the module's and the package's ``__all__`` but an ALL_CAPS
    constant has the token ``det`` (``Det`` in a class name)."""
    assert _tokens("CakeMoEGroupedGemmNondetFunction")[-2:] == ["nondet", "function"]
    for where, name, _ in _public_objects():
        if name.isupper():
            continue
        assert MODE in _tokens(name), (where, name, _tokens(name))


def _callables():
    """``(label, callable)`` of every public function and of the public methods of
    the public classes (``forward`` and ``backward`` of the autograd function
    included)."""
    found = []
    for where, name, obj in _public_objects():
        if not inspect.isclass(obj):
            found.append((f"{where}.{name}", obj))
            continue
        for attr in vars(obj):
            if attr.startswith("_") and attr != "__call__":
                continue
            member = getattr(obj, attr)
            if inspect.isfunction(member) or inspect.ismethod(member):
                found.append((f"{where}.{name}.{attr}", member))
    return found


def test_no_callable_takes_a_deterministic_argument():
    """No public callable, and neither hand-off of ``grouped_mm_bf16(...,
    backend="cake")``, binds ``deterministic=False``: no ``deterministic`` parameter
    and no catch-all ``**kwargs`` that would swallow one.  The autograd function's
    ``forward`` takes the operands, the internal deferred-launch slot and the fuse
    flag only."""
    found = _callables()
    labels = {label.rpartition(".")[2] for label, _ in found}
    assert {"forward", "backward", "launch"} <= labels, labels
    found += [
        (f"{where}.{name}", getattr(_module(where), name)) for where, name in HANDOFFS
    ]
    for label, fn in found:
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        kinds = {p.kind for p in signature.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds, label
    function = _module(f"{PACKAGE}.cake_autograd").CakeMoEGroupedGemmDetFunction
    assert list(inspect.signature(function.forward).parameters) == [
        "ctx",
        "x",
        "w",
        "offs",
        "_launch",
        "fuse_wgrad_accumulation",
    ]


@pytest.mark.parametrize("name", list(PUBLIC))
def test_the_objects_are_named_after_themselves(name):
    """Each object's ``__name__`` and ``__qualname__`` are its public name, it is
    defined in the backend submodule that the package resolves it from, and the
    module's function of the same name (a thin stable wrapper) carries the same
    names and is defined in the module; the module's types are the package's."""
    submodule = PUBLIC[name]
    where = f"{PACKAGE}.{submodule}" if submodule else PACKAGE
    obj = getattr(_module(PACKAGE), name)
    assert obj is getattr(_module(where), name)
    assert obj.__name__ == obj.__qualname__ == name
    assert obj.__module__ == where
    entry = getattr(_module(FACADE), name)
    if inspect.isclass(obj):
        assert entry is obj
        return
    assert entry is not obj
    assert entry.__name__ == entry.__qualname__ == name
    assert inspect.unwrap(entry).__module__ == FACADE


def test_flashinfer_binds_the_names_on_first_access():
    """``flashinfer`` binds every name of the module, and no other name of it, through
    a row of ``flashinfer._LAZY_EXPORTS`` (on first access, so ``import flashinfer``
    loads no backend); the module itself has no row."""
    import flashinfer

    rows = {
        name: row
        for name, row in flashinfer._LAZY_EXPORTS.items()
        if row[0] == f".{MODULE}"
    }
    facade = _module(FACADE)
    assert rows == {name: (f".{MODULE}", name) for name in facade.__all__}
    for name in facade.__all__:
        assert getattr(flashinfer, name) is getattr(facade, name), name


def test_the_shared_mode_name_checks():
    """The shared Cake mode-name checks (``tests/cake/test_helpers/cake_mode_names.py``):
    the mode word in every public name and in order, no ``deterministic``
    argument, one name per object across ``flashinfer``, the module and its
    backend, and the former paths and names failing at import in a fresh
    interpreter."""
    from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

    import flashinfer

    forwards = [
        (name, getattr(_module(f"{PACKAGE}.{sub}" if sub else PACKAGE), name))
        for name, sub in PUBLIC.items()
    ]
    former = [
        (host, name)
        for host in NAME_HOSTS
        for name in FORMER_NAMES + EXTRA_FORMER_NAMES.get(host, ())
    ]
    assert_cake_mode_module(
        _module(FACADE),
        MODE,
        namespaces=(
            flashinfer,
            _module(PACKAGE),
            _module(f"{PACKAGE}.cake_autograd"),
            _module(f"{PACKAGE}.cake_backend"),
        ),
        forwards=forwards,
        old_modules=REMOVED_MODULES,
        old_names=former,
        cwd=Path(flashinfer.__file__).resolve().parent.parent,
    )
