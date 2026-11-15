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

# Registry of the generated Cake DSA indexer programs.  The kernel and
# launcher sources, this registry and ``build.json`` are exporter output:
# programs that differ only in
# their constants or lever text share one translation unit, and each
# ``build.json`` entry selects its program through its own ``-D``
# definitions.
#
# Every program is one device translation unit plus one host binding, shared
# by the architectures listed in its record; a program that differs from a
# sibling only in its constants (a scan lever twin, a rank-finalize program of
# the same window form, a CUB finalize program) builds the sibling's kernel
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
# selection program; ``merge``: the split-range merge; ``finalize``: the
# ascending-id row sort by CUB block radix sort; ``finalize_rank``: the same
# sort as a prefix-popcount rank scatter), ``PROGRAMS`` every program once
# with its role, its architecture-free ``library`` stem and its
# architectures, ``PROGRAM_KEYS`` the program of every host dispatch key per
# architecture (see ``cake_policy``), ``POLICY`` the per-architecture host
# dispatch record, ``POLICY_FP8`` the fields of that record the FP8 input form
# replaces and ``NUMERICS`` the documented numerics of the programs.
# The launch geometry of a program (block, cluster, dynamic shared memory) is
# fixed in its host binding.  A scan binding takes ``zero_region`` (slot kind
# ``zero``) right before the grid: a byte range of the workspace it zeroes on
# the launch stream right before the launch, or an empty tensor (nothing).
# Every scan program takes ``TicketWord`` (a 16-byte block) right before it; a
# program's record lists in ``ticket_arg_arches`` the architectures on which it
# draws its work units from that block instead of the workspace's ticket tail.
# Every scan program takes ``K_scale`` (per-key FP32 scales) right before
# ``TicketWord``; the FP8 programs (keys tagged ``:fp8``) read it, the BF16
# programs never do.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES: list[str] = ["scan", "merge", "finalize", "finalize_rank"]
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
    "finalize": [
        ["buffer", "Indices"],
        ["buffer", "Scores"],
        ["parameter", "top_k"],
        ["parameter", "key_bits"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "finalize_rank": [
        ["buffer", "Indices"],
        ["buffer", "Scores"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["parameter", "top_k"],
        ["parameter", "num_segments"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
# The architectures whose scan programs draw every work unit from a ticket word
# (``cake_policy.TICKET_TAIL_BYTES``): every scan program on every architecture
# but the physical kinds of the ``static_scan_kinds`` of its input form
# (``POLICY[arch]`` for BF16, ``POLICY_FP8[arch]`` for FP8), which keep the
# static enumeration.  A program whose record lists the architecture in
# ``ticket_arg_arches`` draws from its ``TicketWord`` block (``cake_backend``
# gives a prepared runner one of its own, zeroed once); every other drawing
# program draws from the workspace's ticket tail.  Every completed
# scan launch leaves the ticket word at 0; the host (``cake_backend``) zeroes the
# tail once when it prepares a runner on its own workspace, and on a caller-owned
# ``workspace_buffer`` (others may write that buffer between runs) passes it as
# the scan's ``zero_region``, which the scan binding zeroes before every launch.
TICKET_TAIL_ARCHES: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_dsa_indexer_det_topk_034b49542f7f9a407d0e": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_034b49542f7f9a407d0e",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_04db38446d526901fed3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_04db38446d526901fed3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_0633d0c5c9ad00d26f4d": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_0633d0c5c9ad00d26f4d",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_070d919ecf1cb5a20275": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_070d919ecf1cb5a20275",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_074bab39c7585acbdab3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_074bab39c7585acbdab3",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_091795a85161d6e55248": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_091795a85161d6e55248",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_0d38735ee8e81ae6f5e8": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_0d38735ee8e81ae6f5e8",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_1b6af9940d486a0832d9": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_1b6af9940d486a0832d9",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_21f2b0061d472f9494c4": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_21f2b0061d472f9494c4",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_23a4ebeaa3ba23c23c80": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_23a4ebeaa3ba23c23c80",
        "arches": ["sm_103a", "sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_2581ee71bf5e5ecd2556": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_2581ee71bf5e5ecd2556",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_2d40b7d91582905f31c3": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_2d40b7d91582905f31c3",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_2d639f804c39107023ba": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_2d639f804c39107023ba",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_2de9f2d0467668d46463": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_2de9f2d0467668d46463",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_2fffba64f3acc3727903": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_2fffba64f3acc3727903",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_31ea61c2c7a160221f01": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_31ea61c2c7a160221f01",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_38109fa62ccefe5460c2": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_38109fa62ccefe5460c2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_3ba4bff7756ad9176268": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_3ba4bff7756ad9176268",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_3bd43d833b5d53e81dbe": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_3bd43d833b5d53e81dbe",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_3c6dc460295ddf3c44de": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_3c6dc460295ddf3c44de",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_401fb46fc3a69d791fdd": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_401fb46fc3a69d791fdd",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_4088d4e22e0d6d733fef": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_4088d4e22e0d6d733fef",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_44163f96c12681e411cc": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_44163f96c12681e411cc",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_47acefcdee217bade960": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_47acefcdee217bade960",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_48e51726170f87b759f8": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_48e51726170f87b759f8",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_4c792ffa06c1b93d77fa": {
        "role": "finalize",
        "library": "cake_dsa_indexer_det_topk_4c792ffa06c1b93d77fa",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_4fe9fa13b895edcb24e3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_4fe9fa13b895edcb24e3",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_503938c0119dc4ce1520": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_503938c0119dc4ce1520",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_506cb2b2c57f0a514cde": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_506cb2b2c57f0a514cde",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_515346273bb05c921713": {
        "role": "merge",
        "library": "cake_dsa_indexer_det_topk_515346273bb05c921713",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_521027e8563a9664acef": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_521027e8563a9664acef",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_526027b3b42a3dc9a141": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_526027b3b42a3dc9a141",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_54d6e6c62c7e427bab11": {
        "role": "merge",
        "library": "cake_dsa_indexer_det_topk_54d6e6c62c7e427bab11",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_55767b3e4f028548bd92": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_55767b3e4f028548bd92",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_56d38f01a465c2faa6cf": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_56d38f01a465c2faa6cf",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_578634f8fe7efda0f97c": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_578634f8fe7efda0f97c",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_58f1c8a36e2cf5f302ae": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_58f1c8a36e2cf5f302ae",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_59068059bffe0b4c085d": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_59068059bffe0b4c085d",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_5988fd3c547350964be7": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_5988fd3c547350964be7",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_5aa495dca1b0d310978b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_5aa495dca1b0d310978b",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_5c6e55c74de033e5f7af": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_5c6e55c74de033e5f7af",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_604cbe41f1a2fe5b6974": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_604cbe41f1a2fe5b6974",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_607b85385dc2e298c030": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_607b85385dc2e298c030",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_60b4f87115fbecd243e2": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_60b4f87115fbecd243e2",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_616b9e7f58f356a9e146": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_616b9e7f58f356a9e146",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_629db978ba455dd1e92b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_629db978ba455dd1e92b",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_64eb5bd999f161023bed": {
        "role": "merge",
        "library": "cake_dsa_indexer_det_topk_64eb5bd999f161023bed",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_69d2d7c68ccceb095540": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_69d2d7c68ccceb095540",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_6b45fb65871a9150ff10": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_6b45fb65871a9150ff10",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_7182498450255eae7083": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_7182498450255eae7083",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_7286faf88e17986cff38": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_7286faf88e17986cff38",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_72ce45872a80b0b4efb0": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_72ce45872a80b0b4efb0",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_74f132250e5e565dfcc9": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_74f132250e5e565dfcc9",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_75842ae0f80e05addeaa": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_75842ae0f80e05addeaa",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_763009af2c0de13ffc29": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_763009af2c0de13ffc29",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_7891f3dffb0e0717f725": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_7891f3dffb0e0717f725",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_7ae0067e70a4ac94a716": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_7ae0067e70a4ac94a716",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_7da1e774f9cae35774ef": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_7da1e774f9cae35774ef",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_7f646da645feb8b17d42": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_7f646da645feb8b17d42",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_802e619829994de2fab4": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_802e619829994de2fab4",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_808a62a6315ae38034c6": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_808a62a6315ae38034c6",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_81ccb257c7338a468c3b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_81ccb257c7338a468c3b",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_88f7b0c3c31837a87b60": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_88f7b0c3c31837a87b60",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_8a91a587122e3032ace6": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_8a91a587122e3032ace6",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_8ee243151e2cdfa34d1a": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_8ee243151e2cdfa34d1a",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_8f8c393ea106e74d70e9": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_8f8c393ea106e74d70e9",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_90c88df0fd82d4908fca": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_90c88df0fd82d4908fca",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_9247bed3f608806ad0ae": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_9247bed3f608806ad0ae",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_925f0237326fb52b0f94": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_925f0237326fb52b0f94",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_93373e506f0537a0ed67": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_93373e506f0537a0ed67",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_942797c60bc61df40715": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_942797c60bc61df40715",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_96c30b1f33d2a094d9b9": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_96c30b1f33d2a094d9b9",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_99af5257f5e28006f79e": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_99af5257f5e28006f79e",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_9b89c92489b0e60d8ee7": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_9b89c92489b0e60d8ee7",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_9c022a61cbf93fdcc4d3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_9c022a61cbf93fdcc4d3",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_9c823649b35909199596": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_9c823649b35909199596",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_9cc421366d6079aa6ca8": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_9cc421366d6079aa6ca8",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_a42ccc07b6379bcf9549": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_a42ccc07b6379bcf9549",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_a489d4f851f4d3e1057c": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a489d4f851f4d3e1057c",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_a528b54ef347f796744b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a528b54ef347f796744b",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_a52da988b9bb26fb3609": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a52da988b9bb26fb3609",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_a5b550455fe37bc49e1a": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a5b550455fe37bc49e1a",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_a5bdd83845c599e93bf2": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a5bdd83845c599e93bf2",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_a64563a936f84ea4d9f5": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_a64563a936f84ea4d9f5",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_aa0bedf1a5601a3c55be": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_aa0bedf1a5601a3c55be",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_aacebef79deb11b3aea9": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_aacebef79deb11b3aea9",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_af8d290aa9010d180ed0": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_af8d290aa9010d180ed0",
        "arches": ["sm_103a", "sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_afa89b4033e466d158b1": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_afa89b4033e466d158b1",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_b37a3ff7b23dd88b939b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_b37a3ff7b23dd88b939b",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_b63cdb649a4d5e7fdefd": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_b63cdb649a4d5e7fdefd",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_b8440d1d1d54917a47d7": {
        "role": "merge",
        "library": "cake_dsa_indexer_det_topk_b8440d1d1d54917a47d7",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_bacae64bc8920dd8578b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_bacae64bc8920dd8578b",
        "arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_bbc99b799b2c58f8401f": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_bbc99b799b2c58f8401f",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_c02bcf153cf2ddae3b58": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_c02bcf153cf2ddae3b58",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_c1e4796dfe29082d8e2a": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_c1e4796dfe29082d8e2a",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_c2a7046ddd07c57f04fe": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_c2a7046ddd07c57f04fe",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_c401fe6c01bd4f9f95c8": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_c401fe6c01bd4f9f95c8",
        "arches": ["sm_100a"],
    },
    "cake_dsa_indexer_det_topk_c496f83c2e9979f7830a": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_c496f83c2e9979f7830a",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_cb8c20570699061f544d": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_cb8c20570699061f544d",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_cce9be1b9b9709cd3d05": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_cce9be1b9b9709cd3d05",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_cf92373ea526254d61a0": {
        "role": "finalize",
        "library": "cake_dsa_indexer_det_topk_cf92373ea526254d61a0",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_d0b7ecc9edbdcd71b2d7": {
        "role": "finalize",
        "library": "cake_dsa_indexer_det_topk_d0b7ecc9edbdcd71b2d7",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_d121d20b4ff827b4b6d2": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_d121d20b4ff827b4b6d2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_d188e52d69dea931ebd1": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_d188e52d69dea931ebd1",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_d31cb7cabf2871acafa3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_d31cb7cabf2871acafa3",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_d965d76ede39f71e674b": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_d965d76ede39f71e674b",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_db60f32786cbe4ad5f27": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_db60f32786cbe4ad5f27",
        "arches": ["sm_103a"],
    },
    "cake_dsa_indexer_det_topk_dcddd514cbc366dca9f3": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_dcddd514cbc366dca9f3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_e13c66d0acb361a94a1b": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_e13c66d0acb361a94a1b",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_e1e316281190b2a3203b": {
        "role": "finalize",
        "library": "cake_dsa_indexer_det_topk_e1e316281190b2a3203b",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_e3bb94a1bd63e4b54a08": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_e3bb94a1bd63e4b54a08",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_e7860bf225417a2a3b51": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_e7860bf225417a2a3b51",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_e983ae66ae1ef4d182e5": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_e983ae66ae1ef4d182e5",
        "arches": ["sm_107a"],
        "ticket_arg_arches": ["sm_107a"],
    },
    "cake_dsa_indexer_det_topk_ec8a5e9e791c7fee7261": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_ec8a5e9e791c7fee7261",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_ed1192a18ca915b70af7": {
        "role": "finalize",
        "library": "cake_dsa_indexer_det_topk_ed1192a18ca915b70af7",
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_dsa_indexer_det_topk_f244c85e76fde1284ab2": {
        "role": "finalize_rank",
        "library": "cake_dsa_indexer_det_topk_f244c85e76fde1284ab2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_f59c7d519b4a998a5cf9": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_f59c7d519b4a998a5cf9",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_fa34d3de9230fb3465d6": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_fa34d3de9230fb3465d6",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_fad73cb8b09709a23920": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_fad73cb8b09709a23920",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_dsa_indexer_det_topk_fbcc24125a5af183d1dc": {
        "role": "scan",
        "library": "cake_dsa_indexer_det_topk_fbcc24125a5af183d1dc",
        "arches": ["sm_100a", "sm_103a"],
    },
}
PROGRAM_KEYS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "finalize:t128": "cake_dsa_indexer_det_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_det_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_det_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_det_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_det_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_det_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_det_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_det_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_det_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk": "cake_dsa_indexer_det_topk_f244c85e76fde1284ab2",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_det_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_det_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_det_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_det_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_det_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_det_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_det_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_det_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_det_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk": "cake_dsa_indexer_det_topk_38109fa62ccefe5460c2",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_det_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_det_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_det_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk": "cake_dsa_indexer_det_topk_2581ee71bf5e5ecd2556",
        "merge": "cake_dsa_indexer_det_topk_b8440d1d1d54917a47d7",
        "merge:rank:w16384": "cake_dsa_indexer_det_topk_54d6e6c62c7e427bab11",
        "merge:rank:w65536": "cake_dsa_indexer_det_topk_64eb5bd999f161023bed",
        "merge:rank:w8192": "cake_dsa_indexer_det_topk_515346273bb05c921713",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_det_topk_401fb46fc3a69d791fdd",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_f59c7d519b4a998a5cf9",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_det_topk_d121d20b4ff827b4b6d2",
        "scan:l6:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_925f0237326fb52b0f94",
        "scan:l6:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_c2a7046ddd07c57f04fe",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_ec8a5e9e791c7fee7261",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_93373e506f0537a0ed67",
        "scan:narrow:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_59068059bffe0b4c085d",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_det_topk_616b9e7f58f356a9e146",
        "scan:narrow:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_c401fe6c01bd4f9f95c8",
        "scan:narrow:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_034b49542f7f9a407d0e",
        "scan:narrow:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_7ae0067e70a4ac94a716",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_det_topk_90c88df0fd82d4908fca",
        "scan:pair_l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_31ea61c2c7a160221f01",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_det_topk_88f7b0c3c31837a87b60",
        "scan:pair_l6:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_04db38446d526901fed3",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_det_topk_fbcc24125a5af183d1dc",
        "scan:split_l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_fa34d3de9230fb3465d6",
        "scan:split_l6:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_9c022a61cbf93fdcc4d3",
        "scan:split_l6:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_604cbe41f1a2fe5b6974",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_521027e8563a9664acef",
        "scan:split_narrow:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_a52da988b9bb26fb3609",
        "scan:split_narrow:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_4fe9fa13b895edcb24e3",
        "scan:split_narrow:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_7f646da645feb8b17d42",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_det_topk_fad73cb8b09709a23920",
        "scan:wide:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_7da1e774f9cae35774ef",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_det_topk_e13c66d0acb361a94a1b",
        "scan:wide:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_607b85385dc2e298c030",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_det_topk_3c6dc460295ddf3c44de",
        "scan:wide:u2:s0:f0:lsf": "cake_dsa_indexer_det_topk_091795a85161d6e55248",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_det_topk_808a62a6315ae38034c6",
        "scan:wide:u2:s0:f1:fp8": "cake_dsa_indexer_det_topk_4088d4e22e0d6d733fef",
        "scan:wide:u2:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_a489d4f851f4d3e1057c",
        "scan:wide:u2:s0:f1:lsf": "cake_dsa_indexer_det_topk_dcddd514cbc366dca9f3",
    },
    "sm_103a": {
        "finalize:t128": "cake_dsa_indexer_det_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_det_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_det_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_det_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_det_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_det_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_det_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_det_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_det_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk": "cake_dsa_indexer_det_topk_f244c85e76fde1284ab2",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_det_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_det_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_det_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_det_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_det_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_det_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_det_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_det_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_det_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk": "cake_dsa_indexer_det_topk_38109fa62ccefe5460c2",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_det_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_det_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_det_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk": "cake_dsa_indexer_det_topk_2581ee71bf5e5ecd2556",
        "merge": "cake_dsa_indexer_det_topk_b8440d1d1d54917a47d7",
        "merge:rank:w16384": "cake_dsa_indexer_det_topk_54d6e6c62c7e427bab11",
        "merge:rank:w65536": "cake_dsa_indexer_det_topk_64eb5bd999f161023bed",
        "merge:rank:w8192": "cake_dsa_indexer_det_topk_515346273bb05c921713",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_det_topk_401fb46fc3a69d791fdd",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_f59c7d519b4a998a5cf9",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_det_topk_d121d20b4ff827b4b6d2",
        "scan:l6:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_b63cdb649a4d5e7fdefd",
        "scan:l6:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_3ba4bff7756ad9176268",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_ec8a5e9e791c7fee7261",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_23a4ebeaa3ba23c23c80",
        "scan:narrow:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_7891f3dffb0e0717f725",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_det_topk_af8d290aa9010d180ed0",
        "scan:narrow:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_56d38f01a465c2faa6cf",
        "scan:narrow:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_5aa495dca1b0d310978b",
        "scan:narrow:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_629db978ba455dd1e92b",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_det_topk_90c88df0fd82d4908fca",
        "scan:pair_l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_31ea61c2c7a160221f01",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_det_topk_88f7b0c3c31837a87b60",
        "scan:pair_l6:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_04db38446d526901fed3",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_det_topk_fbcc24125a5af183d1dc",
        "scan:split_l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_fa34d3de9230fb3465d6",
        "scan:split_l6:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_21f2b0061d472f9494c4",
        "scan:split_l6:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_a5bdd83845c599e93bf2",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_521027e8563a9664acef",
        "scan:split_narrow:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_a52da988b9bb26fb3609",
        "scan:split_narrow:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_cce9be1b9b9709cd3d05",
        "scan:split_narrow:u1:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_526027b3b42a3dc9a141",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_det_topk_fad73cb8b09709a23920",
        "scan:wide:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_7da1e774f9cae35774ef",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_det_topk_e13c66d0acb361a94a1b",
        "scan:wide:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_607b85385dc2e298c030",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_det_topk_3c6dc460295ddf3c44de",
        "scan:wide:u2:s0:f0:lsf": "cake_dsa_indexer_det_topk_091795a85161d6e55248",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_det_topk_808a62a6315ae38034c6",
        "scan:wide:u2:s0:f1:fp8": "cake_dsa_indexer_det_topk_074bab39c7585acbdab3",
        "scan:wide:u2:s0:f1:fp8:lsf": "cake_dsa_indexer_det_topk_503938c0119dc4ce1520",
        "scan:wide:u2:s0:f1:lsf": "cake_dsa_indexer_det_topk_dcddd514cbc366dca9f3",
        "scan:wide_desc:u1:s0:f0": "cake_dsa_indexer_det_topk_48e51726170f87b759f8",
        "scan:wide_desc:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_d31cb7cabf2871acafa3",
        "scan:wide_desc:u1:s0:f1": "cake_dsa_indexer_det_topk_a528b54ef347f796744b",
        "scan:wide_desc:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_a5b550455fe37bc49e1a",
        "scan:wide_desc:u2:s0:f0": "cake_dsa_indexer_det_topk_db60f32786cbe4ad5f27",
        "scan:wide_desc:u2:s0:f0:lsf": "cake_dsa_indexer_det_topk_72ce45872a80b0b4efb0",
        "scan:wide_desc:u2:s0:f1": "cake_dsa_indexer_det_topk_8ee243151e2cdfa34d1a",
        "scan:wide_desc:u2:s0:f1:lsf": "cake_dsa_indexer_det_topk_2fffba64f3acc3727903",
    },
    "sm_107a": {
        "finalize:t128": "cake_dsa_indexer_det_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_det_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_det_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_det_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_det_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_det_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_det_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_det_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_det_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk:persist": "cake_dsa_indexer_det_topk_7286faf88e17986cff38",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_det_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_det_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_det_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_det_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_det_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_det_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_det_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_det_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_det_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_det_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk:persist": "cake_dsa_indexer_det_topk_8f8c393ea106e74d70e9",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_det_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_det_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_det_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk:persist": "cake_dsa_indexer_det_topk_aa0bedf1a5601a3c55be",
        "merge": "cake_dsa_indexer_det_topk_b8440d1d1d54917a47d7",
        "merge:rank:w16384": "cake_dsa_indexer_det_topk_54d6e6c62c7e427bab11",
        "merge:rank:w65536": "cake_dsa_indexer_det_topk_64eb5bd999f161023bed",
        "merge:rank:w8192": "cake_dsa_indexer_det_topk_515346273bb05c921713",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_det_topk_d188e52d69dea931ebd1",
        "scan:l6:u1:s0:f0:lsf": "cake_dsa_indexer_det_topk_b37a3ff7b23dd88b939b",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_det_topk_7182498450255eae7083",
        "scan:l6:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_bacae64bc8920dd8578b",
        "scan:l6:u1:s0:f1:lsf": "cake_dsa_indexer_det_topk_c1e4796dfe29082d8e2a",
        "scan:l6:u1:s1:f1": "cake_dsa_indexer_det_topk_47acefcdee217bade960",
        "scan:l6:u1:s1:f1:fp8": "cake_dsa_indexer_det_topk_9247bed3f608806ad0ae",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_23a4ebeaa3ba23c23c80",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_det_topk_af8d290aa9010d180ed0",
        "scan:narrow:u1:s1:f1": "cake_dsa_indexer_det_topk_506cb2b2c57f0a514cde",
        "scan:narrow:u2:s0:f0": "cake_dsa_indexer_det_topk_2de9f2d0467668d46463",
        "scan:narrow:u2:s0:f1:fp8": "cake_dsa_indexer_det_topk_2d639f804c39107023ba",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_det_topk_802e619829994de2fab4",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_det_topk_070d919ecf1cb5a20275",
        "scan:pair_l6:u1:s1:f1": "cake_dsa_indexer_det_topk_0633d0c5c9ad00d26f4d",
        "scan:pair_narrow:u1:s1:f1": "cake_dsa_indexer_det_topk_0d38735ee8e81ae6f5e8",
        "scan:pair_narrow:u2:s0:f0": "cake_dsa_indexer_det_topk_a64563a936f84ea4d9f5",
        "scan:pair_narrow:u2:s0:f1": "cake_dsa_indexer_det_topk_942797c60bc61df40715",
        "scan:pair_wide:u1:s1:f0": "cake_dsa_indexer_det_topk_74f132250e5e565dfcc9",
        "scan:pair_wide:u1:s1:f1": "cake_dsa_indexer_det_topk_1b6af9940d486a0832d9",
        "scan:pair_wide:u2:s0:f0": "cake_dsa_indexer_det_topk_3bd43d833b5d53e81dbe",
        "scan:pair_wide:u2:s0:f1": "cake_dsa_indexer_det_topk_9c823649b35909199596",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_det_topk_9cc421366d6079aa6ca8",
        "scan:split_narrow:u1:s0:f1:fp8": "cake_dsa_indexer_det_topk_5988fd3c547350964be7",
        "scan:split_wide:u1:s0:f0": "cake_dsa_indexer_det_topk_763009af2c0de13ffc29",
        "scan:split_wide:u2:s0:f0": "cake_dsa_indexer_det_topk_81ccb257c7338a468c3b",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_det_topk_5c6e55c74de033e5f7af",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_det_topk_e983ae66ae1ef4d182e5",
        "scan:wide:u1:s1:f0": "cake_dsa_indexer_det_topk_99af5257f5e28006f79e",
        "scan:wide:u1:s1:f1": "cake_dsa_indexer_det_topk_e7860bf225417a2a3b51",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_det_topk_afa89b4033e466d158b1",
        "scan:wide:u2:s0:f1:fp8": "cake_dsa_indexer_det_topk_6b45fb65871a9150ff10",
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
        "snake_cap_rule": None,
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
        "lsf_single_rule": None,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {},
        "finalize_items": 8,
        "finalize_threads_fit": [32, 64, 128],
        "finalize_threads_small": 256,
        "finalize_top_k_small": 2048,
        "finalize_threads": 512,
        "finalize_fit": True,
        "finalize_exact_key_bits": True,
        "rank_finalize": True,
        "rank_window_variants": [8192, 16384, 65536, 131072, 262144, 524288],
        "rank_top_k_min": 1025,
        "rank_rule": [524288, 262144],
        "rank_staged": True,
        "rank_staged_rule": [262144],
        "rank_seg_window": True,
        "rank_window_variants_seg_only": [131072],
        "rank_two_level": True,
        "rank_two_level_window_variants": [131072, 262144, 524288, 1048576],
        "rank_slab_window_max": 65536,
        "rank_two_level_rule": [524288],
        "rank_two_level_staged_rule": [524288],
        "rank_bulk_io": True,
        "rank_bulk_align_bytes": 16,
        "rank_t16": True,
        "rank_t16_slots": [4096],
        "rank_persist_max_k": 0,
        "rank_persist_ctas_per_sm": 4,
        "rank_persist_spare_sms": 0,
        "merge_rank": True,
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
        "snake_cap_rule": None,
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
        "lsf_single_rule": None,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {"wide": 32},
        "finalize_items": 8,
        "finalize_threads_fit": [32, 64, 128],
        "finalize_threads_small": 256,
        "finalize_top_k_small": 2048,
        "finalize_threads": 512,
        "finalize_fit": True,
        "finalize_exact_key_bits": True,
        "rank_finalize": True,
        "rank_window_variants": [8192, 16384, 65536, 131072, 262144, 524288],
        "rank_top_k_min": 1025,
        "rank_rule": [524288, 262144],
        "rank_staged": True,
        "rank_staged_rule": [262144],
        "rank_seg_window": True,
        "rank_window_variants_seg_only": [131072],
        "rank_two_level": True,
        "rank_two_level_window_variants": [131072, 262144, 524288, 1048576],
        "rank_slab_window_max": 65536,
        "rank_two_level_rule": [524288],
        "rank_two_level_staged_rule": [524288],
        "rank_bulk_io": True,
        "rank_bulk_align_bytes": 16,
        "rank_t16": True,
        "rank_t16_slots": [4096],
        "rank_persist_max_k": 0,
        "rank_persist_ctas_per_sm": 4,
        "rank_persist_spare_sms": 0,
        "merge_rank": True,
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
        "snake_cap_rule": None,
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
        "lsf_single_rule": None,
        "static_scan_kinds": [],
        "far_sample": 0,
        "check_period_max": 32,
        "check_period_knob_max": 64,
        "check_period_cap_divisor": 512,
        "check_period_kind_overrides": {},
        "finalize_items": 8,
        "finalize_threads_fit": [32, 64, 128],
        "finalize_threads_small": 256,
        "finalize_top_k_small": 2048,
        "finalize_threads": 512,
        "finalize_fit": True,
        "finalize_exact_key_bits": True,
        "rank_finalize": True,
        "rank_window_variants": [8192, 16384, 65536, 131072, 262144, 524288],
        "rank_top_k_min": 1025,
        "rank_rule": [524288, 262144],
        "rank_staged": True,
        "rank_staged_rule": [262144],
        "rank_seg_window": True,
        "rank_window_variants_seg_only": [131072],
        "rank_two_level": True,
        "rank_two_level_window_variants": [131072, 262144, 524288, 1048576],
        "rank_slab_window_max": 65536,
        "rank_two_level_rule": [524288],
        "rank_two_level_staged_rule": [524288],
        "rank_bulk_io": True,
        "rank_bulk_align_bytes": 16,
        "rank_t16": True,
        "rank_t16_slots": [4096],
        "rank_persist_max_k": 2048,
        "rank_persist_ctas_per_sm": 8,
        "rank_persist_spare_sms": 1,
        "merge_rank": True,
    },
}
# The FP8 input form's dispatch record of an architecture is its ``POLICY``
# record with these fields replaced (``cake_policy.fp8_policy_record``); each
# names its own ``static_scan_kinds``: on sm_107a every FP8 scan program keeps
# the static unit enumeration.  sm_107a keeps no split-fit rule until the FP8
# split programs are measured under one.  On sm_100a and sm_103a the FP8 wide
# program decides by waves of the persistent grid (wide_wave_rule): it takes a
# call from the narrow unit when the narrow units need at least 1.7 times its
# waves, and from the three-warpgroup unit when those need at least 1.3
# (sm_100a) or 1.15 (sm_103a) times its waves over at least 84 mean unit tiles
# (sm_100a).  On sm_103a that minimum is 64 mean unit tiles for one document of
# at least 14 wide waves, 3.5 more per wide wave below that but on one document
# at most 8 / (W - 1) ** 2, from two wide waves on at most the larger of that
# and 55 (60 from 8 wide waves on), while the three-warpgroup unit needs
# W >= 1.26 times its waves (8 on one wide wave, 55 at W = 1.5, 60 at 10 wide
# waves and W = 1.4, 72 at W = 4 / 3), at least 112 for packed
# documents and 40 / (W - 1) while the three-warpgroup unit needs W < 1.26 times
# its waves (160 at W = 1.25, 240 at W = 7 / 6), on packed documents then also 9
# / (W - 1) ** 2 (225 at W = 1.2, 324 at W = 7 / 6), and grows as (top_k / 2048)
# ** 1.1 above top_k 2048 (137 and 240 at top_k 4096); below it the
# three-warpgroup unit measured faster.  On sm_107a a call the snake rule marks
# runs the snake copy of the FP8 l6 program only as one document within 40 waves
# of the persistent grid (snake_cap_rule): packed documents and longer calls are
# faster in the plain order.  On sm_103a a one-document call of the FP8 wide
# unit runs its :lsf copy from 64 mean unit tiles on (lsf_single_rule): the same
# units in the same order, measured faster there and slower on shorter units.
# No FP8 call walks its key tiles in descending order (desc_walk_kinds None):
# the descending-walk programs are BF16 programs.  Each FP8 record leaves the
# BF16 snake rules of packed calls and of the l6 fallback off
# (snake_packed_rule, snake_l6_fallback_waves), so the FP8 plans keep their
# own measured rules.  Each FP8 record names every split field itself, so the
# FP8 split counts stay as measured (no split-resident rule) whatever the BF16
# split rule becomes.  The sm_107a FP8 record keeps no sampling class or
# long-unit rule (sample_class_rule, sample_long_rule): the BF16 record's
# #7549 rules were measured on the BF16 ticket programs.  Nor does it run a
# longest-first copy (lsf_programs False): its FP8 programs keep the static unit
# enumeration, which draws no tickets to order, while the BF16 record runs the
# :lsf copies of its l6 ticket programs.
POLICY_FP8: dict[str, dict[str, Any]] = {
    "sm_100a": {
        "static_scan_kinds": [],
        "wide_wave_rule": [1.7, 1.3, 84.0],
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": None,
        "split_wave_scope": None,
        "split_fit_rule": [4096],
        "split_resident_rule": None,
        "split_units": ["l6", "narrow"],
        "desc_walk_kinds": None,
        "snake_packed_rule": None,
        "snake_l6_fallback_waves": None,
    },
    "sm_103a": {
        "static_scan_kinds": [],
        "wide_wave_rule": [
            1.7,
            1.15,
            64.0,
            2048,
            1.1,
            112.0,
            3.5,
            14,
            1.26,
            40.0,
            9.0,
            8.0,
            55.0,
            8,
            60.0,
        ],
        "lsf_single_rule": [64.0],
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": None,
        "split_wave_scope": None,
        "split_fit_rule": [4096],
        "split_resident_rule": None,
        "split_units": ["l6", "narrow"],
        "desc_walk_kinds": None,
        "snake_packed_rule": None,
        "snake_l6_fallback_waves": None,
    },
    "sm_107a": {
        "static_scan_kinds": ["l6", "narrow", "split_narrow", "wide"],
        "sample_class_rule": None,
        "sample_long_rule": None,
        "lsf_programs": False,
        "split_max": 16,
        "split_min_range_tiles": 32,
        "split_wave_rule": [1000, 0.12],
        "split_wave_scope": [["narrow"], 1, True],
        "split_fit_rule": None,
        "split_resident_rule": None,
        "split_units": ["narrow", "wide"],
        "snake_cap_rule": [1, 40],
        "desc_walk_kinds": None,
        "snake_packed_rule": None,
        "snake_l6_fallback_waves": None,
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
