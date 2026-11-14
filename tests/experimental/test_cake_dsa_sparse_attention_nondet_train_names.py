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

Public names of the Cake non-deterministic DSA sparse-attention training package (FlashInfer issue #8928).

The module names the mode: the entry-point module (a stable API) ``flashinfer.cake_dsa_sparse_attention_nondet_train``
and its backend package ``flashinfer.experimental.cake.cake_dsa_sparse_attention_nondet_train``.  Every public name of
the entry-point module (its ``__all__``) has the mode word ``nondet`` as one of its words (the name split at ``_`` and
at CamelCase boundaries), except the module constants (ALL_CAPS); the backend carries the mode in its path, and the
backend objects the entry points forward to or re-export have the entry-point module's names (its internal helpers,
such as ``forward`` / ``backward`` / ``prepare_dsa_train``, keep theirs).  No public callable of the entry-point
module, no backend object it forwards to and no ``forward`` / ``backward`` of the backend's autograd Functions takes a
``deterministic`` keyword (binding one is a ``TypeError``); the modules of the names before the rename do not import,
no module binds an old name, and importing the entry-point module imports no backend.  CPU-only: nothing here
launches, builds or needs a device.
"""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from types import ModuleType

import pytest

from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_dsa_sparse_attention_nondet_train"
PACKAGE = "flashinfer.experimental.cake.cake_dsa_sparse_attention_nondet_train"
BACKEND = f"{PACKAGE}.cake_backend"
REMOVED_MODULES = (
    "flashinfer.dsa_sparse_attention",
    "flashinfer.experimental.cake_dsa_train",
    "flashinfer.experimental.cake_dsa_train.cake_backend",
)
# Module constants keep their names (``DKV_DIRECT_ENV``, ...); every other public name has the mode word as one of
# its words.
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODE_WORD = "nondet"
# One word of a name part between underscores: an upper-case run not followed by a lower-case letter (an acronym), or
# an optional capital and its lower-case letters and digits (``CakeDSASparseAttentionNondetNopeFunction`` -> Cake,
# DSA, Sparse, Attention, Nondet, Nope, Function).
WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+")
# The methods of a public class that a call reaches with the caller's keywords, where the class defines them.
CLASS_METHODS = ("__init__", "forward", "backward", "apply")
# (name before the rename, name after it): the entry-point module's names and the backend objects of the same name.
RENAMED = (
    ("dsa_sparse_attention", "cake_dsa_sparse_attention_nondet"),
    ("dsa_sparse_attention_varlen", "cake_dsa_sparse_attention_nondet_varlen"),
    ("dsa_sparse_attention_nope", "cake_dsa_sparse_attention_nondet_nope"),
    (
        "dsa_sparse_attention_nope_varlen",
        "cake_dsa_sparse_attention_nondet_nope_varlen",
    ),
    ("DSASparseAttentionFunction", "CakeDSASparseAttentionNondetFunction"),
    ("DSASparseAttentionNopeFunction", "CakeDSASparseAttentionNondetNopeFunction"),
    (
        "DSASparseAttentionNopeVarlenFunction",
        "CakeDSASparseAttentionNondetNopeVarlenFunction",
    ),
)
AUTOGRAD_FUNCTIONS = tuple(new for _, new in RENAMED if new.endswith("Function"))
OLD_NAMES = tuple(old for old, _ in RENAMED)


def _public(module: ModuleType) -> list[tuple[str, object]]:
    """``(name, object)`` of every name of ``module.__all__`` (each must resolve), constants excluded."""

    names = list(module.__all__)
    assert names, module.__name__
    return [(name, getattr(module, name)) for name in names if not CONSTANT.match(name)]


def _words(name: str) -> list[str]:
    """The lower-case words of ``name``: split at ``_`` and at CamelCase boundaries (see :data:`WORD`)."""

    return [word.lower() for part in name.split("_") for word in WORD.findall(part)]


def _members(name: str, obj: object) -> list[tuple[str, object]]:
    """``[(name, obj)]`` for a function; for a class, ``(name.method, method)`` of the :data:`CLASS_METHODS` it
    defines."""

    if inspect.isclass(obj):
        return [
            (f"{name}.{method}", getattr(obj, method))
            for method in CLASS_METHODS
            if method in vars(obj)
        ]
    return [(name, obj)] if callable(obj) else []


def test_words_split_at_underscores_and_camel_case() -> None:
    """The mode word counts only as a whole word of a name: not as a part of a longer word, not inside ``det``."""

    for name, words in (
        (
            "CakeDSASparseAttentionNondetNopeVarlenFunction",
            "cake dsa sparse attention nondet nope varlen function",
        ),
        (
            "cake_dsa_sparse_attention_nondet_nope_varlen",
            "cake dsa sparse attention nondet nope varlen",
        ),
    ):
        assert _words(name) == words.split(), (name, _words(name))
    for name in (
        "cake_dsa_sparse_attention_nondeterministic",
        "cake_dsa_sparse_attention_det",
        "DSASparseAttentionFunction",
    ):
        assert MODE_WORD not in _words(name), name


def test_public_names_carry_the_mode_word() -> None:
    """Every name of the entry-point module's ``__all__`` resolves, and every one but the module constants has
    ``nondet`` (any case) as one of its words."""

    module = importlib.import_module(FACADE)
    for name, _ in _public(module):
        assert MODE_WORD in _words(name), (name, _words(name))
    assert {new for _, new in RENAMED} <= set(module.__all__)


def test_no_public_callable_takes_a_mode_keyword() -> None:
    """No public function of the entry-point module, no backend object of the same name and no ``__init__`` /
    ``forward`` / ``backward`` / ``apply`` of the backend's autograd Functions has a ``deterministic`` parameter, and
    ``deterministic=False`` binds to none of them: passing it is Python's own ``TypeError`` (no keyword catch-all, no
    shim)."""

    facade = importlib.import_module(FACADE)
    backend = importlib.import_module(BACKEND)
    found = [member for name, obj in _public(facade) for member in _members(name, obj)]
    for _, new in RENAMED:
        found += _members(f"{BACKEND}.{new}", getattr(backend, new))
    for name in AUTOGRAD_FUNCTIONS:
        assert {"forward", "backward"} <= set(vars(getattr(backend, name))), name
    assert len(found) >= 4 + 4 + 2 * len(AUTOGRAD_FUNCTIONS), found
    for name, fn in found:
        signature = inspect.signature(fn)
        assert "deterministic" not in signature.parameters, name
        with pytest.raises(TypeError, match="deterministic"):
            signature.bind_partial(deterministic=False)


def test_facade_reexports_the_backend_objects() -> None:
    """The autograd Functions of the entry-point module are the backend's classes (resolved on first access)."""

    facade = importlib.import_module(FACADE)
    backend = importlib.import_module(BACKEND)
    for name in AUTOGRAD_FUNCTIONS:
        assert getattr(facade, name) is getattr(backend, name), name
        assert name in dir(facade), name
    with pytest.raises(AttributeError):
        facade.DSASparseAttentionFunction  # noqa: B018


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_module_does_not_import(module_name: str, monkeypatch) -> None:
    """The modules of the names before the rename are gone (no alias module, no re-export): importing one is a
    ``ModuleNotFoundError`` (a stale ``sys.modules`` entry is dropped first, so it cannot mask the failure), and its
    parent package binds no attribute of that name.  A directory left behind by a stale build (``aot/``,
    ``__pycache__``) imports as an empty namespace package; it counts as absent when it carries no module file and
    none of the old names."""

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass
    else:
        assert getattr(module, "__file__", None) is None, module_name
        for old in OLD_NAMES:
            assert not hasattr(module, old), (module_name, old)
    parent, _, leaf = module_name.rpartition(".")
    try:
        parent_module = importlib.import_module(parent)
    except ModuleNotFoundError:
        return  # the parent is a removed module itself
    bound = getattr(parent_module, leaf, None)
    assert bound is None or (
        isinstance(bound, ModuleType) and getattr(bound, "__file__", None) is None
    ), module_name


@pytest.mark.parametrize("module_name", (FACADE, BACKEND))
def test_old_names_are_gone(module_name: str) -> None:
    """No name before the rename is an attribute of the entry-point module or of the backend; the entry-point module
    does not list one in ``__all__``."""

    module = importlib.import_module(module_name)
    for old in OLD_NAMES:
        assert not hasattr(module, old), (module_name, old)
        assert old not in getattr(module, "__all__", ()), (module_name, old)


def test_entry_point_module_imports_no_backend() -> None:
    """In a fresh interpreter, importing the entry-point module imports nothing of the backend package (the
    containment rules of flashinfer/experimental/README.md); the first access of a re-exported autograd Function
    imports it.  ``import flashinfer`` imports neither."""

    code = f"""
import sys
import flashinfer
assert {FACADE!r} not in sys.modules
assert not any(m.startswith({PACKAGE!r}) for m in sys.modules)
import {FACADE} as facade
assert not any(m.startswith({PACKAGE!r}) for m in sys.modules), sorted(
    m for m in sys.modules if m.startswith({PACKAGE!r}))
facade.CakeDSASparseAttentionNondetFunction
assert {BACKEND!r} in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_shared_mode_name_checks() -> None:
    """The shared #8928 checks (``tests/cake/test_helpers/cake_mode_names.py``) over the entry-point module: the mode word
    and its place in every public name, no ``deterministic`` keyword, each object bound only under its own name in the
    entry-point module, ``flashinfer``, the backend package and its ``cake_backend``, and the modules and names before
    the rename failing at import in a fresh interpreter."""

    import flashinfer

    facade = importlib.import_module(FACADE)
    backend = importlib.import_module(BACKEND)
    assert_cake_mode_module(
        facade,
        namespaces=[flashinfer, importlib.import_module(PACKAGE), backend],
        forwards=[
            (new, getattr(backend, new))
            for _, new in RENAMED
            if new not in AUTOGRAD_FUNCTIONS
        ],
        old_modules=REMOVED_MODULES,
        old_names=[(module, old) for module in (FACADE, BACKEND) for old in OLD_NAMES],
    )
