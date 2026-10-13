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
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Registry of the generated Cake DSA indexer programs (written by the
# generated-program export; do not edit by hand).
#
# Every program is one device translation unit plus its host binding, shared
# by the architectures listed in its record; the loader compiles it with the
# exact flag set of the device it runs on.  ``ARG_PLANS`` holds the one
# argument order per stage role (``scan``: a persistent fused scoring /
# selection program; ``merge``: the split-range merge; ``finalize``: the
# ascending-id row sort by CUB block radix sort; ``finalize_rank``: the same
# sort as a prefix-popcount rank scatter), ``COMPILE_FLAGS`` the extra nvcc
# flags per role,
# ``PROGRAMS`` every program once with its role, sources, architectures and
# launch geometry, ``PROGRAM_KEYS`` the program of every host dispatch key per
# architecture (see ``cake_policy``), ``POLICY`` the per-architecture host
# dispatch record and ``NUMERICS`` the documented numerics of the programs.
# The program-text levers of every scan key are documented for the test suite
# in ``tests/test_helpers/cake_dsa_indexer_program_levers.py``; the loader
# does not read them.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}
STAGES: list[str] = ["scan", "merge", "finalize", "finalize_rank"]
ABI: str = "dsa_indexer_v2"
NUMERICS: dict[str, str] = {"zero_sign_policy": "positive_accumulator"}
COMPILE_FLAGS: dict[str, list[str]] = {
    "scan": ["--ptxas-options=--register-usage-level=10"],
    "merge": [],
    "finalize": [],
    "finalize_rank": [],
}
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
        ["parameter", "check_period"],
        ["parameter", "grid_ctas"],
        ["parameter", "softmax_scale"],
        ["parameter", "n_split"],
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
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_dsa_indexer_topk_066b5b4853e26f050fd7": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_066b5b4853e26f050fd7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_066b5b4853e26f050fd7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_0d38735ee8e81ae6f5e8": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0d38735ee8e81ae6f5e8_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0d38735ee8e81ae6f5e8_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_0ee64417f858c0f6be91": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0ee64417f858c0f6be91_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0ee64417f858c0f6be91_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_110194b650e766125f9f": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_110194b650e766125f9f_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_110194b650e766125f9f_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_11956f69dbd0e992157d": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_11956f69dbd0e992157d_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_11956f69dbd0e992157d_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_13547b4979b8fc8c2cd1": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_13547b4979b8fc8c2cd1_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_13547b4979b8fc8c2cd1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_1b6af9940d486a0832d9": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_1b6af9940d486a0832d9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_1b6af9940d486a0832d9_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_247473d9030bd26d7e8e": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_247473d9030bd26d7e8e_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_247473d9030bd26d7e8e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_2581ee71bf5e5ecd2556": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2581ee71bf5e5ecd2556_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2581ee71bf5e5ecd2556_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_2d40b7d91582905f31c3": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2d40b7d91582905f31c3_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2d40b7d91582905f31c3_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_2de9f2d0467668d46463": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2de9f2d0467668d46463_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2de9f2d0467668d46463_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_38109fa62ccefe5460c2": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_38109fa62ccefe5460c2_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_38109fa62ccefe5460c2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_384649a6eee3c6a35aa4": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_384649a6eee3c6a35aa4_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_384649a6eee3c6a35aa4_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_3bd43d833b5d53e81dbe": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3bd43d833b5d53e81dbe_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3bd43d833b5d53e81dbe_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_3c3c1c5b72de566d11a1": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3c3c1c5b72de566d11a1_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3c3c1c5b72de566d11a1_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_3c9bece1433e9cf03979": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3c9bece1433e9cf03979_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3c9bece1433e9cf03979_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_3d8d60dab4805cd50dce": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3d8d60dab4805cd50dce_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3d8d60dab4805cd50dce_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_44163f96c12681e411cc": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_44163f96c12681e411cc_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_44163f96c12681e411cc_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_4c792ffa06c1b93d77fa": {
        "role": "finalize",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_4c792ffa06c1b93d77fa_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_4c792ffa06c1b93d77fa_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_506cb2b2c57f0a514cde": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_506cb2b2c57f0a514cde_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_506cb2b2c57f0a514cde_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_54b1f642a7f0e9e3bf7d": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_54b1f642a7f0e9e3bf7d_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_54b1f642a7f0e9e3bf7d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_55767b3e4f028548bd92": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_55767b3e4f028548bd92_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_55767b3e4f028548bd92_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_578634f8fe7efda0f97c": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_578634f8fe7efda0f97c_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_578634f8fe7efda0f97c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_5b6682b8b07e81595076": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5b6682b8b07e81595076_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5b6682b8b07e81595076_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_5c6e55c74de033e5f7af": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5c6e55c74de033e5f7af_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5c6e55c74de033e5f7af_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_5d0bc5926a696042bf58": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5d0bc5926a696042bf58_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5d0bc5926a696042bf58_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_60b4f87115fbecd243e2": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_60b4f87115fbecd243e2_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_60b4f87115fbecd243e2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_616b9e7f58f356a9e146": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_616b9e7f58f356a9e146_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_616b9e7f58f356a9e146_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_69d2d7c68ccceb095540": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_69d2d7c68ccceb095540_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_69d2d7c68ccceb095540_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_7286faf88e17986cff38": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7286faf88e17986cff38_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7286faf88e17986cff38_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_74f132250e5e565dfcc9": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_74f132250e5e565dfcc9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_74f132250e5e565dfcc9_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_75842ae0f80e05addeaa": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_75842ae0f80e05addeaa_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_75842ae0f80e05addeaa_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_763009af2c0de13ffc29": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_763009af2c0de13ffc29_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_763009af2c0de13ffc29_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_7d1b556b2aedea7f5249": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7d1b556b2aedea7f5249_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7d1b556b2aedea7f5249_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_81ccb257c7338a468c3b": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_81ccb257c7338a468c3b_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_81ccb257c7338a468c3b_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_88619e22363515835efe": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_88619e22363515835efe_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_88619e22363515835efe_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_8a91a587122e3032ace6": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_8a91a587122e3032ace6_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_8a91a587122e3032ace6_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_8f8c393ea106e74d70e9": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_8f8c393ea106e74d70e9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_8f8c393ea106e74d70e9_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_93373e506f0537a0ed67": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_93373e506f0537a0ed67_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_93373e506f0537a0ed67_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_942797c60bc61df40715": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_942797c60bc61df40715_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_942797c60bc61df40715_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_96c30b1f33d2a094d9b9": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_96c30b1f33d2a094d9b9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_96c30b1f33d2a094d9b9_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_99af5257f5e28006f79e": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_99af5257f5e28006f79e_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_99af5257f5e28006f79e_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_9b89c92489b0e60d8ee7": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9b89c92489b0e60d8ee7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9b89c92489b0e60d8ee7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_9c823649b35909199596": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9c823649b35909199596_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9c823649b35909199596_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_9cc421366d6079aa6ca8": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9cc421366d6079aa6ca8_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9cc421366d6079aa6ca8_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_9f4b76cb713088f1285d": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9f4b76cb713088f1285d_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9f4b76cb713088f1285d_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_a42ccc07b6379bcf9549": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a42ccc07b6379bcf9549_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a42ccc07b6379bcf9549_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_a64563a936f84ea4d9f5": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a64563a936f84ea4d9f5_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a64563a936f84ea4d9f5_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_aa0bedf1a5601a3c55be": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_aa0bedf1a5601a3c55be_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_aa0bedf1a5601a3c55be_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_aacebef79deb11b3aea9": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_aacebef79deb11b3aea9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_aacebef79deb11b3aea9_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_afa89b4033e466d158b1": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_afa89b4033e466d158b1_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_afa89b4033e466d158b1_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_b8440d1d1d54917a47d7": {
        "role": "merge",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b8440d1d1d54917a47d7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b8440d1d1d54917a47d7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_bbc99b799b2c58f8401f": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_bbc99b799b2c58f8401f_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_bbc99b799b2c58f8401f_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_c02bcf153cf2ddae3b58": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c02bcf153cf2ddae3b58_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c02bcf153cf2ddae3b58_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_c496f83c2e9979f7830a": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c496f83c2e9979f7830a_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c496f83c2e9979f7830a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_c50a58392e28e85fa0c6": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c50a58392e28e85fa0c6_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c50a58392e28e85fa0c6_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_cb8c20570699061f544d": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_cb8c20570699061f544d_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_cb8c20570699061f544d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_cf92373ea526254d61a0": {
        "role": "finalize",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_cf92373ea526254d61a0_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_cf92373ea526254d61a0_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [64, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7": {
        "role": "finalize",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_d4f8bd31526e93f18c91": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d4f8bd31526e93f18c91_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d4f8bd31526e93f18c91_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_d965d76ede39f71e674b": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d965d76ede39f71e674b_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d965d76ede39f71e674b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e1e316281190b2a3203b": {
        "role": "finalize",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e1e316281190b2a3203b_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e1e316281190b2a3203b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [32, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e7860bf225417a2a3b51": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e7860bf225417a2a3b51_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e7860bf225417a2a3b51_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e983ae66ae1ef4d182e5": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e983ae66ae1ef4d182e5_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e983ae66ae1ef4d182e5_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_ed1192a18ca915b70af7": {
        "role": "finalize",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_ed1192a18ca915b70af7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_ed1192a18ca915b70af7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_f244c85e76fde1284ab2": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_f244c85e76fde1284ab2_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_f244c85e76fde1284ab2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    },
}
PROGRAM_KEYS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "finalize:t128": "cake_dsa_indexer_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk": "cake_dsa_indexer_topk_f244c85e76fde1284ab2",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk": "cake_dsa_indexer_topk_38109fa62ccefe5460c2",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk": "cake_dsa_indexer_topk_2581ee71bf5e5ecd2556",
        "merge": "cake_dsa_indexer_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_c50a58392e28e85fa0c6",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_247473d9030bd26d7e8e",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_93373e506f0537a0ed67",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_616b9e7f58f356a9e146",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_54b1f642a7f0e9e3bf7d",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_5b6682b8b07e81595076",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_topk_13547b4979b8fc8c2cd1",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_066b5b4853e26f050fd7",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_0ee64417f858c0f6be91",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_d4f8bd31526e93f18c91",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_topk_5d0bc5926a696042bf58",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_topk_3d8d60dab4805cd50dce",
    },
    "sm_103a": {
        "finalize:t128": "cake_dsa_indexer_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk": "cake_dsa_indexer_topk_f244c85e76fde1284ab2",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk": "cake_dsa_indexer_topk_38109fa62ccefe5460c2",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk": "cake_dsa_indexer_topk_2581ee71bf5e5ecd2556",
        "merge": "cake_dsa_indexer_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_c50a58392e28e85fa0c6",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_247473d9030bd26d7e8e",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_11956f69dbd0e992157d",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_3c3c1c5b72de566d11a1",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_54b1f642a7f0e9e3bf7d",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_5b6682b8b07e81595076",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_topk_13547b4979b8fc8c2cd1",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_066b5b4853e26f050fd7",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_0ee64417f858c0f6be91",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_d4f8bd31526e93f18c91",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_topk_5d0bc5926a696042bf58",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_topk_3d8d60dab4805cd50dce",
    },
    "sm_107a": {
        "finalize:t128": "cake_dsa_indexer_topk_ed1192a18ca915b70af7",
        "finalize:t256": "cake_dsa_indexer_topk_d0b7ecc9edbdcd71b2d7",
        "finalize:t32": "cake_dsa_indexer_topk_e1e316281190b2a3203b",
        "finalize:t512": "cake_dsa_indexer_topk_4c792ffa06c1b93d77fa",
        "finalize:t64": "cake_dsa_indexer_topk_cf92373ea526254d61a0",
        "finalize_rank:t256:w131072:two:staged": "cake_dsa_indexer_topk_bbc99b799b2c58f8401f",
        "finalize_rank:t256:w131072:two:staged:bulk": "cake_dsa_indexer_topk_8a91a587122e3032ace6",
        "finalize_rank:t256:w131072:two:staged:i16": "cake_dsa_indexer_topk_578634f8fe7efda0f97c",
        "finalize_rank:t256:w131072:two:staged:i16:bulk": "cake_dsa_indexer_topk_d965d76ede39f71e674b",
        "finalize_rank:t256:w16384:staged": "cake_dsa_indexer_topk_55767b3e4f028548bd92",
        "finalize_rank:t256:w16384:staged:bulk:persist": "cake_dsa_indexer_topk_7286faf88e17986cff38",
        "finalize_rank:t256:w16384:staged:i16": "cake_dsa_indexer_topk_aacebef79deb11b3aea9",
        "finalize_rank:t256:w16384:staged:i16:bulk": "cake_dsa_indexer_topk_2d40b7d91582905f31c3",
        "finalize_rank:t256:w262144:two:staged": "cake_dsa_indexer_topk_a42ccc07b6379bcf9549",
        "finalize_rank:t256:w262144:two:staged:bulk": "cake_dsa_indexer_topk_60b4f87115fbecd243e2",
        "finalize_rank:t256:w262144:two:staged:i16": "cake_dsa_indexer_topk_75842ae0f80e05addeaa",
        "finalize_rank:t256:w262144:two:staged:i16:bulk": "cake_dsa_indexer_topk_69d2d7c68ccceb095540",
        "finalize_rank:t256:w524288:two:staged": "cake_dsa_indexer_topk_cb8c20570699061f544d",
        "finalize_rank:t256:w524288:two:staged:bulk": "cake_dsa_indexer_topk_c496f83c2e9979f7830a",
        "finalize_rank:t256:w524288:two:staged:i16": "cake_dsa_indexer_topk_e3bb94a1bd63e4b54a08",
        "finalize_rank:t256:w524288:two:staged:i16:bulk": "cake_dsa_indexer_topk_44163f96c12681e411cc",
        "finalize_rank:t256:w65536:staged": "cake_dsa_indexer_topk_9b89c92489b0e60d8ee7",
        "finalize_rank:t256:w65536:staged:bulk:persist": "cake_dsa_indexer_topk_8f8c393ea106e74d70e9",
        "finalize_rank:t256:w65536:staged:i16": "cake_dsa_indexer_topk_c02bcf153cf2ddae3b58",
        "finalize_rank:t256:w65536:staged:i16:bulk": "cake_dsa_indexer_topk_96c30b1f33d2a094d9b9",
        "finalize_rank:t256:w8192:staged": "cake_dsa_indexer_topk_58f1c8a36e2cf5f302ae",
        "finalize_rank:t256:w8192:staged:bulk:persist": "cake_dsa_indexer_topk_aa0bedf1a5601a3c55be",
        "merge": "cake_dsa_indexer_topk_b8440d1d1d54917a47d7",
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_3c9bece1433e9cf03979",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_110194b650e766125f9f",
        "scan:l6:u1:s1:f1": "cake_dsa_indexer_topk_384649a6eee3c6a35aa4",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_11956f69dbd0e992157d",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_3c3c1c5b72de566d11a1",
        "scan:narrow:u1:s1:f1": "cake_dsa_indexer_topk_506cb2b2c57f0a514cde",
        "scan:narrow:u2:s0:f0": "cake_dsa_indexer_topk_2de9f2d0467668d46463",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_7d1b556b2aedea7f5249",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_9f4b76cb713088f1285d",
        "scan:pair_l6:u1:s1:f1": "cake_dsa_indexer_topk_88619e22363515835efe",
        "scan:pair_narrow:u1:s1:f1": "cake_dsa_indexer_topk_0d38735ee8e81ae6f5e8",
        "scan:pair_narrow:u2:s0:f0": "cake_dsa_indexer_topk_a64563a936f84ea4d9f5",
        "scan:pair_narrow:u2:s0:f1": "cake_dsa_indexer_topk_942797c60bc61df40715",
        "scan:pair_wide:u1:s1:f0": "cake_dsa_indexer_topk_74f132250e5e565dfcc9",
        "scan:pair_wide:u1:s1:f1": "cake_dsa_indexer_topk_1b6af9940d486a0832d9",
        "scan:pair_wide:u2:s0:f0": "cake_dsa_indexer_topk_3bd43d833b5d53e81dbe",
        "scan:pair_wide:u2:s0:f1": "cake_dsa_indexer_topk_9c823649b35909199596",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_9cc421366d6079aa6ca8",
        "scan:split_wide:u1:s0:f0": "cake_dsa_indexer_topk_763009af2c0de13ffc29",
        "scan:split_wide:u2:s0:f0": "cake_dsa_indexer_topk_81ccb257c7338a468c3b",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_5c6e55c74de033e5f7af",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_e983ae66ae1ef4d182e5",
        "scan:wide:u1:s1:f0": "cake_dsa_indexer_topk_99af5257f5e28006f79e",
        "scan:wide:u1:s1:f1": "cake_dsa_indexer_topk_e7860bf225417a2a3b51",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_topk_afa89b4033e466d158b1",
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
        "l6_rule": [256.0, 512],
        "wide_rule": [6.0, 300.0],
        "split_max": 32,
        "split_min_range_tiles": 64,
        "split_wave_rule": None,
        "pair_rule": [128.0, 256.0, 512],
        "snake_default": False,
        "snake_rule": None,
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [["wide"], 256, False],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
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
    },
    "sm_103a": {
        "tile_keys": 128,
        "block_q_narrow": 4,
        "block_q_wide": 8,
        "block_q_l6": 6,
        "candidate_entry_bytes": 8,
        "candidate_multiplier": 4,
        "candidate_slack": 128,
        "cand_mult_rule": [384.0, 8],
        "cand_cap_floor": 8192,
        "l6_rule": [1024.0, 512],
        "wide_rule": [6.0, 300.0],
        "split_max": 32,
        "split_min_range_tiles": 64,
        "split_wave_rule": None,
        "pair_rule": [64.0, 1024.0, 512],
        "snake_default": False,
        "snake_rule": None,
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [["wide"], 256, False],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
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
        "l6_rule": [64.0, 0],
        "wide_rule": [2.0, 250.0],
        "split_max": 32,
        "split_min_range_tiles": 64,
        "split_wave_rule": [1000, 0.12],
        "pair_rule": [0.0, 4096.0, 512],
        "snake_default": False,
        "snake_rule": [
            ["l6", "narrow", "pair_l6", "pair_narrow", "pair_wide", "wide"],
            32.0,
            128.0,
            0.5,
        ],
        "tile_unroll_default": 1,
        "tile_unroll_factor": 2,
        "tile_unroll_rule": [
            ["narrow", "pair_narrow", "pair_wide", "split_wide", "wide"],
            512,
            True,
        ],
        "sample_fit_max_mean_tiles": 512.0,
        "sample_tiles_max": 32,
        "sample_tiles_short_units": 16,
        "sample_dispatch_mean_tiles_max": 640.0,
        "sample_tiles_tiny_units": 8,
        "sample_dispatch_tiny_tiles_max": 128.0,
        "sample_shift_permille": 250,
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
        "rank_persist_ctas_per_sm": 4,
    },
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5676"


@functools.cache
def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?

    ``compute_107a`` needs a CUDA toolkit that lists it; a toolkit without it
    declines the sm_107a builds up front instead of failing inside the JIT
    build, so a checkout that registers sm_107a programs stays importable and
    testable on a toolkit that only knows 10.0 / 10.3.
    """
    if arch not in ARCH_NVCC_FLAGS:
        return False
    if arch == "sm_107a":
        from ...compilation_context import _nvcc_supports_sm107

        return bool(_nvcc_supports_sm107())
    return True


def registered_archs() -> tuple[str, ...]:
    """Architectures with registered programs, in ``ARCHES`` order."""
    return tuple(arch for arch in ARCHES if arch in PROGRAM_KEYS and arch in POLICY)


def program_available(arch: str) -> bool:
    """True when ``arch`` registers a scan, the merge and a finalize program (CUB or rank)."""
    keys = PROGRAM_KEYS.get(arch, {})
    return (
        arch in POLICY
        and any(key.startswith("scan:") for key in keys)
        and "merge" in keys
        and any(key.startswith(("finalize:", "finalize_rank:")) for key in keys)
    )


def select_program(arch: str, key: str) -> str:
    """Return the registered program of dispatch ``key`` on ``arch``."""
    keys = PROGRAM_KEYS.get(arch)
    if keys is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered "
            f"in this checkout (registered: {sorted(PROGRAM_KEYS)}; see {TRACKING_ISSUE})"
        )
    program = keys.get(key)
    if program is None:
        raise NotImplementedError(
            f"The DSA indexer top-k program {key!r} is not registered for {arch} "
            f"(registered on {arch}: {sorted(keys)}; see {TRACKING_ISSUE})"
        )
    return program


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_program(program: str, arch: str):
    """JIT spec of ``program`` compiled for ``arch`` (one cached library per pair)."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not built for {arch!r}")
    if not toolchain_supports(arch):
        raise RuntimeError(f"this checkout cannot compile {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{program}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *COMPILE_FLAGS[record["role"]]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *{p.parent for p in sources}, *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_program(program: str, arch: str):
    return gen_program(program, arch).build_and_load()
