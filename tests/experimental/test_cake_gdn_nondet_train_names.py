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

The mode names (FlashInfer #8928) of the two NON-DETERMINISTIC GDN facades, ``flashinfer.cake_gdn_nondet_train``
(single rank) and ``flashinfer.cake_gdn_nondet_train_kcp`` (context-parallel), through the shared check
``tests/cake/test_helpers/cake_mode_names.assert_cake_mode_module``, with no exemption: every public name carries the word
``nondet`` right after the op, no public callable binds ``deterministic=`` at its signature, the facade, the
``flashinfer`` top level and the backend package and modules bind each object under its facade name only, and the
former module paths and names fail at import in a fresh interpreter.  CPU only: nothing launches.
"""

from __future__ import annotations

import importlib
import pathlib

from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

ROOT = pathlib.Path(__file__).resolve().parents[2]

FACADE = "flashinfer.cake_gdn_nondet_train"
PACKAGE = "flashinfer.experimental.cake.cake_gdn_nondet_train"
SUBMODULES = ("cake_backend", "cake_build", "cake_chain", "cake_jit", "cake_plan")
# the facade functions that call the package function of their name
FORWARDS = ("cake_gdn_nondet",)
OLD_MODULES = ("flashinfer.experimental.cake.cake_gdn_train",)
OLD_NAMES = (
    "chunk_gated_delta_rule",
    "ChunkGatedDeltaRuleFunction",
    "gdn_chunk_forward",
    "gdn_chunk_backward",
)

KCP_FACADE = "flashinfer.cake_gdn_nondet_train_kcp"
KCP_PACKAGE = "flashinfer.experimental.cake.cake_gdn_nondet_train_kcp"
KCP_SUBMODULES = (
    "cake_backend",
    "cake_build",
    "cake_bytes",
    "cake_fake_backend",
    "cake_jit",
    "cake_kcp",
    "cake_kcp_plan",
    "cake_l2n",
    "cake_options",
    "cake_tables",
)
KCP_FORWARDS = (
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_partition",
    "cake_gdn_nondet_train_kcp_build_context",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_comm_bytes",
)
KCP_OLD_MODULES = (
    "flashinfer.cake_gdn_train_kcp",
    "flashinfer.experimental.cake_gdn_train_kcp",
)
KCP_OLD_NAMES = (
    "chunk_gdn_kcp",
    "ChunkGDNFunctionKcp",
    "chunk_gdn_fwd_kcp",
    "chunk_gdn_bwd_kcp",
    "partition_kcp",
    "build_context_kcp",
    "packed_qkv_views_kcp",
    "gdn_workspace_bytes_kcp",
    "gdn_saved_bytes_kcp",
    "gdn_comm_bytes_kcp",
    "is_gdn_train_kcp_supported",
    "ContextKcp",
    "OptionsKcp",
    "PartitionKcp",
    "RecomputeHandleKcp",
    "SavedKcp",
    "WorkspaceKcp",
)


def _check(facade: str, package: str, submodules, forwards, old_modules, old_names):
    import flashinfer

    module = importlib.import_module(facade)
    pkg = importlib.import_module(package)
    assert_cake_mode_module(
        module,
        "nondet",
        namespaces=[
            flashinfer,
            pkg,
            *(importlib.import_module(f"{package}.{sub}") for sub in submodules),
        ],
        forwards=[(name, getattr(pkg, name)) for name in forwards],
        old_modules=old_modules,
        old_names=[(m, name) for m in (facade, package) for name in old_names],
        cwd=ROOT,
    )


def test_the_single_rank_facade_follows_the_mode_name_rules() -> None:
    _check(FACADE, PACKAGE, SUBMODULES, FORWARDS, OLD_MODULES, OLD_NAMES)


def test_the_kcp_facade_follows_the_mode_name_rules() -> None:
    _check(
        KCP_FACADE,
        KCP_PACKAGE,
        KCP_SUBMODULES,
        KCP_FORWARDS,
        KCP_OLD_MODULES,
        KCP_OLD_NAMES,
    )
