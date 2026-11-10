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
"""

from __future__ import annotations

import functools
from typing import Any

from . import cake_build

# Registry of the generated Cake DSA indexer programs of the non-deterministic
# package: the BF16 scan programs and the plain split merge of the deterministic
# package, each under this package's prefix ``cake_dsa_indexer_nondet_topk``
# with the same twenty-hex identity -- but the sm_107a l6 and CTA-pair l6 keys,
# whose deterministic programs keep the static unit enumeration, name their
# ticket twins (the same builds drawing every work unit from their
# ``TicketWord`` block, each a program of its own).  No program orders a row: the scan writes
# the selected entries of an unsplit row and its tail padding straight into the
# outputs, the plain merge those of a split row, so the registry carries no
# finalize (ascending-id sort) and no rank-fused merge program.  The kernel and
# launcher sources, this registry and ``build.json`` are exporter output:
# programs that differ only in their constants or lever text share one
# translation unit, and each ``build.json`` entry selects its program through
# its own ``-D`` definitions.
#
# Every program is one device translation unit plus one host binding, shared
# by the architectures listed in its record; a program that differs from a
# sibling only in its constants (a scan lever twin) builds the sibling's kernel
# translation unit, and a program whose binding text equals a sibling's up to
# the kernel symbol, the host-shim namespace and the launch constants builds
# the sibling's binding -- each with the compile definitions of its
# ``build.json`` entry.  ``build.json`` next to this file
# (schema ``cake-build/1``, read by ``cake_build``) is the only record of what
# to compile: one entry ``<library>_<arch>`` per program and architecture with
# its sources, nvcc flags and digests.  ``cake_build.load_library`` serves the
# prebuilt ``aot/<entry>.so`` when a builder compiled it and otherwise
# JIT-builds the entry through FlashInfer.  ``ARG_PLANS`` holds the one
# argument order per stage role (``scan``: a persistent fused scoring /
# selection program; ``merge``: the split-range merge), ``PROGRAMS`` every
# program once with its role, its architecture-free ``library`` stem and its
# architectures, ``PROGRAM_KEYS`` the program of every host dispatch key per
# architecture (see ``cake_policy``), ``POLICY`` the per-architecture host
# dispatch record and ``NUMERICS`` the documented numerics of the programs.
# The launch geometry of a program (block, cluster, dynamic shared memory) is
# fixed in its host binding.  A scan binding takes ``zero_region`` (slot kind
# ``zero``) right before the grid: a byte range of the workspace it zeroes on
# the launch stream right before the launch, or an empty tensor (nothing).
# Every scan program takes ``TicketWord`` (a 16-byte block) right before it; a
# program's record lists in ``ticket_arg_arches`` the architectures on which it
# draws its work units from that block instead of the workspace's ticket tail.
# Every scan program takes ``K_scale`` (the deterministic package's per-key FP32
# scales of its FP8 form) right before ``TicketWord``; the BF16 programs of this
# package never read it.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES: list[str] = ["scan", "merge"]
ABI: str = "dsa_indexer_v5"
NUMERICS: dict[str, str] = {"zero_sign_policy": "positive_accumulator"}
FFI_ENTRY: str = "run"
ARG_PLANS: dict[str, list[list[str]]] = {
    "scan": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "W"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "Indices"],
        ["buffer", "Scores"],
        ["buffer", "Cand"],
        ["parameter", "num_segments"],
        ["parameter", "top_k"],
        ["parameter", "ratio"],
        ["parameter", "has_offsets"],
        ["parameter", "cand_cap"],
        ["parameter", "first_cap"],
        ["parameter", "sample_tiles_max"],
        ["parameter", "sample_shift_permille"],
        ["parameter", "far_sample"],
        ["parameter", "check_period"],
        ["parameter", "grid_ctas"],
        ["parameter", "softmax_scale"],
        ["parameter", "n_split"],
        ["buffer", "K_scale"],
        ["buffer", "TicketWord"],
        ["zero", "zero_region"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "merge": [
        ["buffer", "Staging"],
        ["buffer", "Indices"],
        ["buffer", "Scores"],
        ["parameter", "top_k"],
        ["parameter", "n_split"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
# The architectures whose scan programs draw every work unit from a ticket word
# (``cake_policy.TICKET_TAIL_BYTES``): every scan program on every architecture
# but the physical kinds of the ``static_scan_kinds`` of ``POLICY[arch]``,
# which keep the static enumeration.  A program whose record lists the
# architecture in ``ticket_arg_arches`` draws from its ``TicketWord`` block
# (``cake_backend`` gives a prepared runner one of its own, zeroed once); every
# other drawing program draws from the workspace's ticket tail.  Every completed
# scan launch leaves the ticket word at 0; the host (``cake_backend``) zeroes the
# tail once when it prepares a runner on its own workspace, and on a caller-owned
# ``workspace_buffer`` (others may write that buffer between runs) passes it as
# the scan's ``zero_region``, which the scan binding zeroes before every launch.
TICKET_TAIL_ARCHES: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_dsa_indexer_nondet_topk_04db38446d526901fed3": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_04db38446d526901fed3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_091795a85161d6e55248": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_091795a85161d6e55248",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_0d38735ee8e81ae6f5e8": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_0d38735ee8e81ae6f5e8",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_7182498450255eae7083": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_7182498450255eae7083",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_1b6af9940d486a0832d9": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_1b6af9940d486a0832d9",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_23a4ebeaa3ba23c23c80": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_23a4ebeaa3ba23c23c80",
        "arches": ["sm_103a", "sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_2de9f2d0467668d46463": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_2de9f2d0467668d46463",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_2fffba64f3acc3727903": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_2fffba64f3acc3727903",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_31ea61c2c7a160221f01": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_31ea61c2c7a160221f01",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_47acefcdee217bade960": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_47acefcdee217bade960",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_3bd43d833b5d53e81dbe": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_3bd43d833b5d53e81dbe",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_3c6dc460295ddf3c44de": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_3c6dc460295ddf3c44de",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_d188e52d69dea931ebd1": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_d188e52d69dea931ebd1",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_401fb46fc3a69d791fdd": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_401fb46fc3a69d791fdd",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_48e51726170f87b759f8": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_48e51726170f87b759f8",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_506cb2b2c57f0a514cde": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_506cb2b2c57f0a514cde",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_521027e8563a9664acef": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_521027e8563a9664acef",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_59068059bffe0b4c085d": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_59068059bffe0b4c085d",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_nondet_topk_5c6e55c74de033e5f7af": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_5c6e55c74de033e5f7af",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_607b85385dc2e298c030": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_607b85385dc2e298c030",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_616b9e7f58f356a9e146": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_616b9e7f58f356a9e146",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_nondet_topk_629db978ba455dd1e92b": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_629db978ba455dd1e92b",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_72ce45872a80b0b4efb0": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_72ce45872a80b0b4efb0",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_74f132250e5e565dfcc9": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_74f132250e5e565dfcc9",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_763009af2c0de13ffc29": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_763009af2c0de13ffc29",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_7891f3dffb0e0717f725": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_7891f3dffb0e0717f725",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_7ae0067e70a4ac94a716": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_7ae0067e70a4ac94a716",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_nondet_topk_802e619829994de2fab4": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_802e619829994de2fab4",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_7da1e774f9cae35774ef": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_7da1e774f9cae35774ef",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_808a62a6315ae38034c6": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_808a62a6315ae38034c6",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_81ccb257c7338a468c3b": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_81ccb257c7338a468c3b",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_0633d0c5c9ad00d26f4d": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_0633d0c5c9ad00d26f4d",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_88f7b0c3c31837a87b60": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_88f7b0c3c31837a87b60",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_8ee243151e2cdfa34d1a": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_8ee243151e2cdfa34d1a",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_90c88df0fd82d4908fca": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_90c88df0fd82d4908fca",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_93373e506f0537a0ed67": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_93373e506f0537a0ed67",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_nondet_topk_942797c60bc61df40715": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_942797c60bc61df40715",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_99af5257f5e28006f79e": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_99af5257f5e28006f79e",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_9c823649b35909199596": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_9c823649b35909199596",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_9cc421366d6079aa6ca8": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_9cc421366d6079aa6ca8",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_070d919ecf1cb5a20275": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_070d919ecf1cb5a20275",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_a528b54ef347f796744b": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_a528b54ef347f796744b",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_a52da988b9bb26fb3609": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_a52da988b9bb26fb3609",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_a5b550455fe37bc49e1a": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_a5b550455fe37bc49e1a",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_a64563a936f84ea4d9f5": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_a64563a936f84ea4d9f5",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_af8d290aa9010d180ed0": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_af8d290aa9010d180ed0",
        "arches": ["sm_103a", "sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_afa89b4033e466d158b1": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_afa89b4033e466d158b1",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_b37a3ff7b23dd88b939b": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_b37a3ff7b23dd88b939b",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7": {
        "role": "merge",
        "library": "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_c1e4796dfe29082d8e2a": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_c1e4796dfe29082d8e2a",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_d121d20b4ff827b4b6d2": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_d121d20b4ff827b4b6d2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_d31cb7cabf2871acafa3": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_d31cb7cabf2871acafa3",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_db60f32786cbe4ad5f27": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_db60f32786cbe4ad5f27",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_dcddd514cbc366dca9f3": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_dcddd514cbc366dca9f3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_e13c66d0acb361a94a1b": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_e13c66d0acb361a94a1b",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_e7860bf225417a2a3b51": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_e7860bf225417a2a3b51",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_e983ae66ae1ef4d182e5": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_e983ae66ae1ef4d182e5",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_nondet_topk_ec8a5e9e791c7fee7261": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_ec8a5e9e791c7fee7261",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_f59c7d519b4a998a5cf9": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_f59c7d519b4a998a5cf9",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_fa34d3de9230fb3465d6": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_fa34d3de9230fb3465d6",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_fad73cb8b09709a23920": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_fad73cb8b09709a23920",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_nondet_topk_fbcc24125a5af183d1dc": {
        "role": "scan",
        "library": "cake_dsa_indexer_nondet_topk_fbcc24125a5af183d1dc",
        "arches": ["sm_100a", "sm_103a"],
    },
}
PROGRAM_KEYS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "merge": "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_401fb46fc3a69d791fdd",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_f59c7d519b4a998a5cf9",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_d121d20b4ff827b4b6d2",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_ec8a5e9e791c7fee7261",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_93373e506f0537a0ed67",
        "scan:narrow:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_59068059bffe0b4c085d",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_nondet_topk_616b9e7f58f356a9e146",
        "scan:narrow:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_7ae0067e70a4ac94a716",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_90c88df0fd82d4908fca",
        "scan:pair_l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_31ea61c2c7a160221f01",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_88f7b0c3c31837a87b60",
        "scan:pair_l6:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_04db38446d526901fed3",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_fbcc24125a5af183d1dc",
        "scan:split_l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_fa34d3de9230fb3465d6",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_521027e8563a9664acef",
        "scan:split_narrow:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_a52da988b9bb26fb3609",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_nondet_topk_fad73cb8b09709a23920",
        "scan:wide:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_7da1e774f9cae35774ef",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_nondet_topk_e13c66d0acb361a94a1b",
        "scan:wide:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_607b85385dc2e298c030",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_nondet_topk_3c6dc460295ddf3c44de",
        "scan:wide:u2:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_091795a85161d6e55248",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_nondet_topk_808a62a6315ae38034c6",
        "scan:wide:u2:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_dcddd514cbc366dca9f3",
    },
    "sm_103a": {
        "merge": "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_401fb46fc3a69d791fdd",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_f59c7d519b4a998a5cf9",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_d121d20b4ff827b4b6d2",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_ec8a5e9e791c7fee7261",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_23a4ebeaa3ba23c23c80",
        "scan:narrow:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_7891f3dffb0e0717f725",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_nondet_topk_af8d290aa9010d180ed0",
        "scan:narrow:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_629db978ba455dd1e92b",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_90c88df0fd82d4908fca",
        "scan:pair_l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_31ea61c2c7a160221f01",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_88f7b0c3c31837a87b60",
        "scan:pair_l6:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_04db38446d526901fed3",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_fbcc24125a5af183d1dc",
        "scan:split_l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_fa34d3de9230fb3465d6",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_521027e8563a9664acef",
        "scan:split_narrow:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_a52da988b9bb26fb3609",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_nondet_topk_fad73cb8b09709a23920",
        "scan:wide:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_7da1e774f9cae35774ef",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_nondet_topk_e13c66d0acb361a94a1b",
        "scan:wide:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_607b85385dc2e298c030",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_nondet_topk_3c6dc460295ddf3c44de",
        "scan:wide:u2:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_091795a85161d6e55248",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_nondet_topk_808a62a6315ae38034c6",
        "scan:wide:u2:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_dcddd514cbc366dca9f3",
        "scan:wide_desc:u1:s0:f0": "cake_dsa_indexer_nondet_topk_48e51726170f87b759f8",
        "scan:wide_desc:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_d31cb7cabf2871acafa3",
        "scan:wide_desc:u1:s0:f1": "cake_dsa_indexer_nondet_topk_a528b54ef347f796744b",
        "scan:wide_desc:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_a5b550455fe37bc49e1a",
        "scan:wide_desc:u2:s0:f0": "cake_dsa_indexer_nondet_topk_db60f32786cbe4ad5f27",
        "scan:wide_desc:u2:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_72ce45872a80b0b4efb0",
        "scan:wide_desc:u2:s0:f1": "cake_dsa_indexer_nondet_topk_8ee243151e2cdfa34d1a",
        "scan:wide_desc:u2:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_2fffba64f3acc3727903",
    },
    "sm_107a": {
        "merge": "cake_dsa_indexer_nondet_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_d188e52d69dea931ebd1",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_nondet_topk_b37a3ff7b23dd88b939b",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_7182498450255eae7083",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_nondet_topk_c1e4796dfe29082d8e2a",
        "scan:l6:u1:s1:f1": "cake_dsa_indexer_nondet_topk_47acefcdee217bade960",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_23a4ebeaa3ba23c23c80",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_nondet_topk_af8d290aa9010d180ed0",
        "scan:narrow:u1:s1:f1": "cake_dsa_indexer_nondet_topk_506cb2b2c57f0a514cde",
        "scan:narrow:u2:s0:f0": "cake_dsa_indexer_nondet_topk_2de9f2d0467668d46463",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_nondet_topk_802e619829994de2fab4",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_nondet_topk_070d919ecf1cb5a20275",
        "scan:pair_l6:u1:s1:f1": "cake_dsa_indexer_nondet_topk_0633d0c5c9ad00d26f4d",
        "scan:pair_narrow:u1:s1:f1": "cake_dsa_indexer_nondet_topk_0d38735ee8e81ae6f5e8",
        "scan:pair_narrow:u2:s0:f0": "cake_dsa_indexer_nondet_topk_a64563a936f84ea4d9f5",
        "scan:pair_narrow:u2:s0:f1": "cake_dsa_indexer_nondet_topk_942797c60bc61df40715",
        "scan:pair_wide:u1:s1:f0": "cake_dsa_indexer_nondet_topk_74f132250e5e565dfcc9",
        "scan:pair_wide:u1:s1:f1": "cake_dsa_indexer_nondet_topk_1b6af9940d486a0832d9",
        "scan:pair_wide:u2:s0:f0": "cake_dsa_indexer_nondet_topk_3bd43d833b5d53e81dbe",
        "scan:pair_wide:u2:s0:f1": "cake_dsa_indexer_nondet_topk_9c823649b35909199596",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_nondet_topk_9cc421366d6079aa6ca8",
        "scan:split_wide:u1:s0:f0": "cake_dsa_indexer_nondet_topk_763009af2c0de13ffc29",
        "scan:split_wide:u2:s0:f0": "cake_dsa_indexer_nondet_topk_81ccb257c7338a468c3b",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_nondet_topk_5c6e55c74de033e5f7af",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_nondet_topk_e983ae66ae1ef4d182e5",
        "scan:wide:u1:s1:f0": "cake_dsa_indexer_nondet_topk_99af5257f5e28006f79e",
        "scan:wide:u1:s1:f1": "cake_dsa_indexer_nondet_topk_e7860bf225417a2a3b51",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_nondet_topk_afa89b4033e466d158b1",
    },
}
POLICY: dict[str, dict[str, Any]] = {
    "sm_100a": {
        "tile_keys": 128,
        "block_q_narrow": 4,
        "block_q_wide": 8,
        "block_q_l6": 6,
        "candidate_entry_bytes": 8,
        "candidate_multiplier": 4,
        "candidate_slack": 128,
        "cand_mult_rule": [384.0, 8],
        "cand_cap_floor": 8192,
        "cand_cap_floor_rule": [1024, 16384],
        "candidate_multiplier_unsampled": 4,
        "cand_mult_rule_unsampled": [384.0, 8],
        "cand_cap_floor_unsampled": 8192,
        "cand_cap_ceiling": None,
        "cand_cap_ceiling_unsampled": None,
        "l6_rule": [256.0, 512],
        "wide_rule": [5.5, 300.0],
        "wide_first_rule": None,
        "wide_wave_rule": None,
        "desc_walk_kinds": None,
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": None,
        "split_wave_scope": None,
        "split_fit_rule": [4096],
        "split_resident_rule": [4096, 32768, 49152, 32, 8],
        "split_units": ["l6", "narrow"],
        "pair_rule": [128.0, 256.0, 512],
        "pair_units": ["l6"],
        "l6_solo_rule": None,
        "l6_solo_unsampled_max_queries": 16384,
        "wide_pair_snake_only": False,
        "snake_default": False,
        "snake_rule": None,
        "snake_packed_rule": None,
        "snake_l6_fallback_waves": None,
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [["wide"], 256, False],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_fit_kinds": ["l6", "narrow", "pair_l6", "wide"],
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
        "sample_shift_rule": None,
        "sample_class_rule": [8, 500],
        "sample_long_rule": None,
        "lsf_programs": True,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {},
    },
    "sm_103a": {
        "tile_keys": 128,
        "block_q_narrow": 4,
        "block_q_wide": 8,
        "block_q_l6": 6,
        "candidate_entry_bytes": 8,
        "candidate_multiplier": 16,
        "candidate_slack": 128,
        "cand_mult_rule": [384.0, 8],
        "cand_cap_floor": 16384,
        "cand_cap_floor_rule": None,
        "candidate_multiplier_unsampled": 4,
        "cand_mult_rule_unsampled": [384.0, 8],
        "cand_cap_floor_unsampled": 8192,
        "cand_cap_ceiling": 32768,
        "cand_cap_ceiling_unsampled": None,
        "l6_rule": [1024.0, 512],
        "wide_rule": [6.0, 300.0],
        "wide_first_rule": [512.0],
        "wide_wave_rule": None,
        "desc_walk_kinds": [["wide"], []],
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": None,
        "split_wave_scope": None,
        "split_fit_rule": [4096],
        "split_resident_rule": [4096, 32768, 49152, 32, 8],
        "split_units": ["l6", "narrow"],
        "pair_rule": [64.0, 1024.0, 512],
        "pair_units": ["l6"],
        "l6_solo_rule": None,
        "l6_solo_unsampled_max_queries": 16384,
        "wide_pair_snake_only": False,
        "snake_default": False,
        "snake_rule": None,
        "snake_packed_rule": None,
        "snake_l6_fallback_waves": None,
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [["wide"], 256, False],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_fit_kinds": ["l6", "narrow", "pair_l6", "wide"],
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
        "sample_shift_rule": [4096, 375],
        "sample_class_rule": [8, 500],
        "sample_long_rule": None,
        "lsf_programs": True,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {"wide": 32},
    },
    "sm_107a": {
        "tile_keys": 128,
        "block_q_narrow": 4,
        "block_q_wide": 8,
        "block_q_l6": 6,
        "candidate_entry_bytes": 8,
        "candidate_multiplier": 4,
        "candidate_slack": 128,
        "cand_mult_rule": [384.0, 8],
        "cand_cap_floor": 8192,
        "cand_cap_floor_rule": None,
        "candidate_multiplier_unsampled": 4,
        "cand_mult_rule_unsampled": [384.0, 8],
        "cand_cap_floor_unsampled": 8192,
        "cand_cap_ceiling": None,
        "cand_cap_ceiling_unsampled": None,
        "l6_rule": [64.0, 0],
        "wide_rule": [2.0, 250.0],
        "wide_first_rule": None,
        "wide_wave_rule": None,
        "desc_walk_kinds": None,
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": [1000, 0.12],
        "split_wave_scope": [["narrow"], 1, True],
        "split_fit_rule": [4096],
        "split_resident_rule": None,
        "split_units": ["narrow", "wide"],
        "pair_rule": [0.0, 4096.0, 512],
        "pair_units": ["l6", "narrow", "wide"],
        "l6_solo_rule": [128.0, 2],
        "l6_solo_unsampled_max_queries": None,
        "wide_pair_snake_only": True,
        "snake_default": False,
        "snake_rule": [
            ["l6", "narrow", "pair_l6", "pair_narrow", "pair_wide", "wide"],
            32.0,
            128.0,
            0.5,
        ],
        "snake_packed_rule": [["l6"], 48.0, None, 7],
        "snake_l6_fallback_waves": 310,
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [
            ["narrow", "pair_narrow", "pair_wide", "split_wide", "wide"],
            512,
            True,
        ],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_fit_kinds": [
            "l6",
            "narrow",
            "pair_l6",
            "pair_narrow",
            "pair_wide",
            "wide",
        ],
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
        "sample_shift_rule": None,
        "sample_class_rule": [32, 500],
        "sample_long_rule": [["l6"], 500],
        "lsf_programs": True,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {},
    },
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5676"


def library_name(program: str, arch: str) -> str:
    """The ``build.json`` entry of ``program`` for ``arch``: ``<library>_<arch>``."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not built for {arch!r}")
    return f"{record['library']}_{arch}"


@functools.cache
def load_program(program: str, arch: str):
    """The library of ``program`` for ``arch``: prebuilt ``aot/`` else a JIT build."""
    return cake_build.load_library(library_name(program, arch))
