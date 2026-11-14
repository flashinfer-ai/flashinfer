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

The module names the mode (flashinfer-ai/flashinfer#8928): the entry-point module (a stable API)
``flashinfer.cake_dsa_sparse_attention_det_train`` and its backend package
``flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train``.  Every public name of the entry-point module
(its ``__all__``) has the mode word ``det`` as one of its words (the name split at ``_`` and at CamelCase boundaries)
and none has ``nondet``, except the module constants (ALL_CAPS); the backend objects the entry points forward to or
re-export have the entry-point module's names.  No public callable takes a ``deterministic`` keyword, the entry
points take the keywords of the non-deterministic package's entries, and importing the entry-point module imports no
backend.  CPU-only: nothing here launches, builds or needs a device.
"""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys

import pytest

from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_dsa_sparse_attention_det_train"
PACKAGE = "flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train"
BACKEND = f"{PACKAGE}.cake_backend"
NONDET_FACADE = "flashinfer.cake_dsa_sparse_attention_nondet_train"
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+")
ENTRIES = (
    "cake_dsa_sparse_attention_det",
    "cake_dsa_sparse_attention_det_varlen",
    "cake_dsa_sparse_attention_det_nope",
    "cake_dsa_sparse_attention_det_nope_varlen",
)
AUTOGRAD_FUNCTIONS = (
    "CakeDSASparseAttentionDetFunction",
    "CakeDSASparseAttentionDetNopeFunction",
    "CakeDSASparseAttentionDetNopeVarlenFunction",
)


def _words(name: str) -> list[str]:
    return [word.lower() for part in name.split("_") for word in WORD.findall(part)]


def test_public_names_carry_the_mode_word() -> None:
    """Every name of ``__all__`` resolves; every one but the module constants has ``det`` as a word and no
    ``nondet``."""

    module = importlib.import_module(FACADE)
    assert set(ENTRIES) | set(AUTOGRAD_FUNCTIONS) <= set(module.__all__)
    for name in module.__all__:
        getattr(module, name)
        if CONSTANT.match(name):
            continue
        words = _words(name)
        assert "det" in words and "nondet" not in words, (name, words)


def test_entries_take_the_keywords_of_the_nondeterministic_entries() -> None:
    """Each entry has the parameters of its non-deterministic counterpart, in the same order with the same defaults
    (``key_passes`` and ``dkv_direct`` included), and no ``deterministic`` keyword."""

    det = importlib.import_module(FACADE)
    nondet = importlib.import_module(NONDET_FACADE)
    for name in ENTRIES:
        mine = inspect.signature(getattr(det, name)).parameters
        theirs = inspect.signature(
            getattr(nondet, name.replace("_det", "_nondet", 1))
        ).parameters
        assert list(mine) == list(theirs), name
        for param, other in zip(mine.values(), theirs.values(), strict=True):
            assert param.kind == other.kind and param.default == other.default, (
                name,
                param.name,
            )
        assert "deterministic" not in mine, name
        with pytest.raises(TypeError, match="deterministic"):
            inspect.signature(getattr(det, name)).bind_partial(deterministic=True)


def test_facade_reexports_the_backend_objects() -> None:
    facade = importlib.import_module(FACADE)
    backend = importlib.import_module(BACKEND)
    for name in AUTOGRAD_FUNCTIONS:
        assert getattr(facade, name) is getattr(backend, name), name
        assert name in dir(facade), name


def test_facade_reads_no_dkv_direct_variable() -> None:
    """The facade does not consult the non-deterministic package's ``dkv_direct`` variable: the keyword has no
    effect here."""

    facade = importlib.import_module(FACADE)
    assert not hasattr(facade, "DKV_DIRECT_ENV")
    assert "FLASHINFER_CAKE_DSA_DKV_DIRECT" not in inspect.getsource(facade)


def test_entry_point_module_imports_no_backend() -> None:
    """In a fresh interpreter, importing the entry-point module imports nothing of the backend package; the first
    access of a re-exported autograd Function imports it."""

    code = f"""
import sys
import flashinfer
assert {FACADE!r} not in sys.modules
assert not any(m.startswith({PACKAGE!r}) for m in sys.modules)
import {FACADE} as facade
assert not any(m.startswith({PACKAGE!r}) for m in sys.modules), sorted(
    m for m in sys.modules if m.startswith({PACKAGE!r}))
facade.CakeDSASparseAttentionDetFunction
assert {BACKEND!r} in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_shared_mode_name_checks() -> None:
    """The shared #8928 checks over the entry-point module (mode word, no ``deterministic`` keyword, each object
    bound only under its own name in the facade, ``flashinfer``, the backend package and its ``cake_backend``)."""

    import flashinfer

    facade = importlib.import_module(FACADE)
    backend = importlib.import_module(BACKEND)
    assert_cake_mode_module(
        facade,
        namespaces=[flashinfer, importlib.import_module(PACKAGE), backend],
        forwards=[(name, getattr(backend, name)) for name in ENTRIES],
    )
