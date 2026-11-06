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

Public names of the Cake non-deterministic causal conv1d + SiLU training package (FlashInfer issue #8928).

The module names the mode: the experimental entry points ``flashinfer.cake_causal_conv1d_nondet_train`` and
``flashinfer.cake_causal_conv1d_nondet_train_kcp`` and the package
``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train`` (with its ``kcp`` module).  Every public name of the
two entry-point modules (their ``__all__``) carries the mode word ``nondet``, except the module constants (ALL_CAPS);
the package carries the mode in its path, and its public entry points and types have the entry-point modules' names
(its internal helper ``plan_cache_clear`` keeps its name).  No public callable of the four
modules takes a ``deterministic`` keyword (binding one is a ``TypeError``); the modules of the names before the rename
do not import, and no module binds an old name.  CPU-only: nothing here launches, builds or needs a device.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_causal_conv1d_nondet_train"
KCP_FACADE = "flashinfer.cake_causal_conv1d_nondet_train_kcp"
PACKAGE = "flashinfer.experimental.cake.cake_causal_conv1d_nondet_train"
KCP_PACKAGE = f"{PACKAGE}.kcp"
FACADES = (FACADE, KCP_FACADE)
MODULES = (*FACADES, PACKAGE, KCP_PACKAGE)
REMOVED_MODULES = (
    "flashinfer.experimental.cake.cake_causal_conv1d_train",
    "flashinfer.experimental.cake.cake_causal_conv1d_train_kcp",
    "flashinfer.cake_causal_conv1d_train_kcp",
    "flashinfer.experimental.cake.cake_causal_conv1d_train_backend",
)
# Module constants keep their names (``ARCHES``, ``BWD_GRID_MULTS``, ``SUPPORTED_COMPUTE_CAPABILITIES``, ...); every
# other public name of an entry-point module carries the mode word.
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODE_WORD = "nondet"
# The methods of a public class that a call reaches with the caller's keywords, where the class defines them.
CLASS_METHODS = ("__init__", "forward", "backward", "apply")

# (name before the rename, name after it) of the single-rank entry points: the facade and the package.
SINGLE_RANK = (
    ("cake_causal_conv1d", "cake_causal_conv1d_nondet"),
    ("CakeCausalConv1dFunction", "CakeCausalConv1dNondetFunction"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_nondet_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_nondet_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace",
        "cake_causal_conv1d_nondet_train_forward_workspace",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_nondet_train_backward_workspace",
    ),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
    ),
    (
        "is_cake_causal_conv1d_train_supported",
        "is_cake_causal_conv1d_nondet_train_supported",
    ),
)
# The KCP entry points: the KCP facade and the package's ``kcp`` module.
KCP_FACADE_NAMES = (
    ("cake_causal_conv1d_kcp", "cake_causal_conv1d_nondet_kcp"),
    ("cake_causal_conv1d_multi_kcp", "cake_causal_conv1d_nondet_kcp_multi"),
    ("RecomputeHandleKcp", "CakeCausalConv1dNondetKcpRecomputeHandle"),
)
KCP_PACKAGE_NAMES = (
    *KCP_FACADE_NAMES,
    ("CakeCausalConv1dFunctionKcp", "CakeCausalConv1dNondetKcpFunction"),
    ("CakeCausalConv1dMultiFunctionKcp", "CakeCausalConv1dNondetKcpMultiFunction"),
    ("cake_causal_conv1d_forward_kcp", "cake_causal_conv1d_nondet_train_kcp_forward"),
    ("cake_causal_conv1d_backward_kcp", "cake_causal_conv1d_nondet_train_kcp_backward"),
    (
        "cake_causal_conv1d_multi_forward_kcp",
        "cake_causal_conv1d_nondet_train_kcp_multi_forward",
    ),
    (
        "cake_causal_conv1d_multi_backward_kcp",
        "cake_causal_conv1d_nondet_train_kcp_multi_backward",
    ),
    ("graph_cache_clear", "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear"),
)
RENAMED = {
    FACADE: SINGLE_RANK,
    KCP_FACADE: KCP_FACADE_NAMES,
    PACKAGE: SINGLE_RANK,
    KCP_PACKAGE: KCP_PACKAGE_NAMES,
}
# Renamed names that no module lists in ``__all__`` (the KCP backend's host-plan cache helper, now
# ``cake_causal_conv1d_nondet_train_kcp_host_cache_clear``): only their absence is checked.
UNLISTED_OLD_NAMES = ("host_cache_clear",)
OLD_NAMES = sorted(
    {old for pairs in RENAMED.values() for old, _ in pairs} | set(UNLISTED_OLD_NAMES)
)


def _public(module: ModuleType) -> list[tuple[str, object]]:
    """``(name, object)`` of every name of ``module.__all__`` (each must resolve), constants excluded."""

    names = list(module.__all__)
    assert names, module.__name__
    return [(name, getattr(module, name)) for name in names if not CONSTANT.match(name)]


def _callables(module: ModuleType) -> list[tuple[str, object]]:
    """``(qualified name, callable)`` of the public functions of ``module`` and of the methods
    :data:`CLASS_METHODS` its public classes define."""

    found = []
    for name, obj in _public(module):
        if inspect.isclass(obj):
            found += [
                (f"{name}.{method}", getattr(obj, method))
                for method in CLASS_METHODS
                if method in vars(obj)
            ]
        elif callable(obj):
            found.append((name, obj))
    assert found, module.__name__
    return found


@pytest.mark.parametrize("module_name", FACADES)
def test_public_names_carry_the_mode_word(module_name: str) -> None:
    """Every name of an entry-point module's ``__all__`` resolves, and every one but the module constants contains
    ``nondet`` (any case)."""

    module = importlib.import_module(module_name)
    for name, _ in _public(module):
        assert MODE_WORD in name.lower(), (module_name, name)


@pytest.mark.parametrize("module_name", MODULES)
def test_no_public_callable_takes_a_mode_keyword(module_name: str) -> None:
    """No public function, and no ``__init__`` / ``forward`` / ``backward`` / ``apply`` of a public class, has a
    ``deterministic`` parameter, and ``deterministic=False`` binds to none of them: passing it is Python's own
    ``TypeError`` (no keyword catch-all, no shim)."""

    module = importlib.import_module(module_name)
    for name, fn in _callables(module):
        signature = inspect.signature(fn)
        assert "deterministic" not in signature.parameters, (module_name, name)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_module_does_not_import(module_name: str, monkeypatch) -> None:
    """The modules of the names before the rename are gone (no alias module, no re-export): importing one is a
    ``ModuleNotFoundError`` for that module itself (a stale ``sys.modules`` entry is dropped first, so it cannot mask
    the failure), and its parent package binds no attribute of that name."""

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module(module_name)
    assert info.value.name == module_name, (module_name, info.value.name)
    parent, _, leaf = module_name.rpartition(".")
    assert not hasattr(importlib.import_module(parent), leaf), module_name


@pytest.mark.parametrize("module_name", MODULES)
def test_old_names_are_gone_and_the_new_names_bound(module_name: str) -> None:
    """No name before the rename is an attribute (or in ``__all__``) of any of the four modules; each module lists
    the new name of every name it had."""

    module = importlib.import_module(module_name)
    for old in OLD_NAMES:
        assert not hasattr(module, old), (module_name, old)
        assert old not in module.__all__, (module_name, old)
    for old, new in RENAMED[module_name]:
        assert new in module.__all__, (module_name, old, new)
        assert callable(getattr(module, new)), (module_name, new)
