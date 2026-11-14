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

Names of the Cake context-parallel (KCP) packed KDA training package (non-deterministic).

The KCP entry points live in ``flashinfer.cake_kda_nondet_train_kcp`` and the package in
``flashinfer.experimental.cake.cake_kda_nondet_train_kcp``.  The shared check of #8928
(``tests.cake.test_helpers.cake_mode_names.assert_cake_mode_module``) covers the facade: every public name states the mode
(``nondet``) right after the op, no entry takes a ``deterministic`` keyword, the facade, the ``flashinfer`` top level,
the package and its modules ``cake_kcp`` / ``cake_kcp_plan`` bind each object under one name only (the backend object
each facade entry calls included), and the names before this scheme fail at import with no alias: the module
``flashinfer.cake_kda_train_kcp``, the package ``flashinfer.experimental.cake_kda_train_kcp`` and its modules, the
package's names before #8928 (``chunk_kda_kcp``, ``WorkspaceKcp``, ``partition_kcp``, ...) and the names before the det
rename (``KcpWorkspace``, ``chunk_kda_kcp_fwd``, ``kcp_partition``, ...).  The same mode-word and keyword checks run on
the package.  What the facade does is ``test_cake_kda_nondet_train_kcp_api.py``'s.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_kda_nondet_train_kcp as kcp_api
from tests.cake.test_helpers.cake_mode_names import (
    assert_cake_mode_module,
    assert_mode_names,
    assert_no_deterministic_keyword,
)

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_kda_nondet_train_kcp"
PACKAGE = "flashinfer.experimental.cake.cake_kda_nondet_train_kcp"
# the module and the package before #8928 (rule 1): no alias
OLD_FACADE = "flashinfer.cake_kda_train_kcp"
OLD_PACKAGE = "flashinfer.experimental.cake_kda_train_kcp"
SUBMODULES = ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit", "cake_build")
REPO_ROOT = Path(__file__).resolve().parents[2]

# the backend module each facade entry forwards to (the same name)
ENTRY_BACKEND = {
    "cake_kda_nondet_train_kcp_partition": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_build_context": "cake_kcp_plan",
    "cake_kda_nondet_kcp": "cake_kcp",
    "cake_kda_nondet_train_kcp_forward": "cake_kcp",
    "cake_kda_nondet_train_kcp_backward": "cake_kcp",
    "cake_kda_nondet_train_kcp_workspace_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_saved_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_comm_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_reduce_gate_param_grads": "cake_kcp",
    "cake_kda_nondet_train_kcp_handoff_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_handoff_stats": "cake_kcp",
}
# the package's names before #8928 (the issue's table "KCP entries"): no alias, no shim
NAMES_BEFORE_8928 = (
    "CallKcp",
    "ChunkKDAFunctionKcp",
    "ContextKcp",
    "FragmentKcp",
    "OptionsKcp",
    "PartitionKcp",
    "RecomputeHandleKcp",
    "SavedKcp",
    "WorkspaceKcp",
    "build_context_kcp",
    "chunk_kda_bwd_kcp",
    "chunk_kda_fwd_kcp",
    "chunk_kda_kcp",
    "kcp_handoff_stats",
    "kda_comm_bytes_kcp",
    "kda_handoff_bytes_kcp",
    "kda_saved_bytes_kcp",
    "kda_workspace_bytes_kcp",
    "partition_kcp",
    "reduce_gate_param_grads_kcp",
)
# the names before the det rename: neither the facade nor the package nor its modules bind them
PREVIOUS_NAMES = (
    "ChunkKDAKcpFunction",
    "KcpCall",
    "KcpContext",
    "KcpFragment",
    "KcpOptions",
    "KcpPartition",
    "KcpRecomputeHandle",
    "KcpSaved",
    "KcpWorkspace",
    "build_kcp_context",
    "chunk_kda_kcp_bwd",
    "chunk_kda_kcp_fwd",
    "kcp_comm_bytes",
    "kcp_partition",
    "kcp_reduce_gate_param_grads",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
)
# the module and package paths before #8928 and the names before #8928 and before the det rename: no alias, no shim
OLD_MODULES = (
    OLD_FACADE,
    OLD_PACKAGE,
    *(f"{OLD_PACKAGE}.{m}" for m in SUBMODULES),
)
OLD_NAMES = tuple(
    (module, old)
    for old in NAMES_BEFORE_8928 + PREVIOUS_NAMES
    for module in (
        FACADE,
        PACKAGE,
        *(f"{PACKAGE}.{m}" for m in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit")),
    )
)


def _backend(module: str):
    return importlib.import_module(f"{PACKAGE}.{module}")


def test_facade_follows_the_cake_mode_names() -> None:
    # the shared check: the mode word, no ``deterministic`` keyword, each object bound under its facade name only
    # (the facade, the flashinfer top level, the package and the modules it serves its names from, including the
    # backend object each entry calls) and the earlier names and package path failing at import
    assert_cake_mode_module(
        kcp_api,
        "nondet",
        namespaces=[
            flashinfer,
            importlib.import_module(PACKAGE),
            *(_backend(m) for m in ("cake_kcp", "cake_kcp_plan")),
        ],
        forwards=[
            (name, getattr(_backend(module), name))
            for name, module in ENTRY_BACKEND.items()
        ],
        old_modules=OLD_MODULES,
        old_names=OLD_NAMES,
        cwd=REPO_ROOT,
    )


def test_package_states_the_mode_and_takes_no_deterministic_keyword() -> None:
    package = importlib.import_module(PACKAGE)
    assert_mode_names(package, "nondet")
    assert_no_deterministic_keyword(package)
