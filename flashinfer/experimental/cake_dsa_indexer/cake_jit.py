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
    "cake_dsa_indexer_topk_02fe1c65eb89cba835b0": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_02fe1c65eb89cba835b0_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_02fe1c65eb89cba835b0_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_0843306409e1e7fc6092": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0843306409e1e7fc6092_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0843306409e1e7fc6092_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_0941c30bf15fe14a9cc3": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0941c30bf15fe14a9cc3_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0941c30bf15fe14a9cc3_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_0bf75ec341ce571b7413": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0bf75ec341ce571b7413_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_0bf75ec341ce571b7413_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_1e987836f71bb5c3ba52": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_1e987836f71bb5c3ba52_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_1e987836f71bb5c3ba52_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_2041b86ba2e56c0591ec": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2041b86ba2e56c0591ec_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2041b86ba2e56c0591ec_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_2a0be0fd36be3320b3df": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2a0be0fd36be3320b3df_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2a0be0fd36be3320b3df_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_2d3c54670435ca47d9d2": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2d3c54670435ca47d9d2_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_2d3c54670435ca47d9d2_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_3006bd3aa47e016d0e03": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3006bd3aa47e016d0e03_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_3006bd3aa47e016d0e03_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_36899ca69836524c0fc3": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_36899ca69836524c0fc3_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_36899ca69836524c0fc3_binding.cu",
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
    "cake_dsa_indexer_topk_43e6b06f87908c5b1172": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_43e6b06f87908c5b1172_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_43e6b06f87908c5b1172_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
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
    "cake_dsa_indexer_topk_4fb74943e6794f4d0b19": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_4fb74943e6794f4d0b19_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_4fb74943e6794f4d0b19_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_545825470c274d10bc04": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_545825470c274d10bc04_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_545825470c274d10bc04_binding.cu",
        ],
        "arches": ["sm_107a"],
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
    "cake_dsa_indexer_topk_5aa34f7e4d8bb3d96fde": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5aa34f7e4d8bb3d96fde_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_5aa34f7e4d8bb3d96fde_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_6608aa8518faca0527e7": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6608aa8518faca0527e7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6608aa8518faca0527e7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_6b1f075769a73ee1b176": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6b1f075769a73ee1b176_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6b1f075769a73ee1b176_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_6ff54f38a7d43aed441b": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6ff54f38a7d43aed441b_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_6ff54f38a7d43aed441b_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_70fbcd7e3a2abe75132b": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_70fbcd7e3a2abe75132b_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_70fbcd7e3a2abe75132b_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_74d7c381e6a2c1219e6c": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_74d7c381e6a2c1219e6c_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_74d7c381e6a2c1219e6c_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_7a8c782839e3650e42e3": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7a8c782839e3650e42e3_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7a8c782839e3650e42e3_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_7d171eb3a76d45726f38": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7d171eb3a76d45726f38_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_7d171eb3a76d45726f38_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_9463e969b9b47fe794d1": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9463e969b9b47fe794d1_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9463e969b9b47fe794d1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_9848f45ca3390e0daeef": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9848f45ca3390e0daeef_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_9848f45ca3390e0daeef_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
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
    "cake_dsa_indexer_topk_a42ccc07b6379bcf9549": {
        "role": "finalize_rank",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a42ccc07b6379bcf9549_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_a42ccc07b6379bcf9549_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_b136f04b21e35062077d": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b136f04b21e35062077d_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b136f04b21e35062077d_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_b206f52857c238bcf951": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b206f52857c238bcf951_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_b206f52857c238bcf951_binding.cu",
        ],
        "arches": ["sm_100a"],
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
    "cake_dsa_indexer_topk_c2598645bc5c4f2e4d18": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c2598645bc5c4f2e4d18_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c2598645bc5c4f2e4d18_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_c9efd59500c6c16020e6": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c9efd59500c6c16020e6_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_c9efd59500c6c16020e6_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_ced4e89bc4e74d1c752e": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_ced4e89bc4e74d1c752e_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_ced4e89bc4e74d1c752e_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_d8608692031a9a3a49cc": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d8608692031a9a3a49cc_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_d8608692031a9a3a49cc_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_dce69060d489ec86df87": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_dce69060d489ec86df87_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_dce69060d489ec86df87_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_df755465364eecf95460": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_df755465364eecf95460_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_df755465364eecf95460_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_e453d14c4a8923a43703": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e453d14c4a8923a43703_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e453d14c4a8923a43703_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e826f6ad8378b7725739": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e826f6ad8378b7725739_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e826f6ad8378b7725739_binding.cu",
        ],
        "arches": ["sm_100a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
    },
    "cake_dsa_indexer_topk_e840189671a665d54ec9": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e840189671a665d54ec9_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_e840189671a665d54ec9_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [512, 1, 1], "cluster": [2, 1, 1]},
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
    "cake_dsa_indexer_topk_efb859170a00f8c96574": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_efb859170a00f8c96574_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_efb859170a00f8c96574_binding.cu",
        ],
        "arches": ["sm_103a", "sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
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
    "cake_dsa_indexer_topk_f51560d9a49f174c0ee7": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_f51560d9a49f174c0ee7_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_f51560d9a49f174c0ee7_binding.cu",
        ],
        "arches": ["sm_107a"],
        "launch": {"block": [384, 1, 1], "cluster": [2, 1, 1]},
    },
    "cake_dsa_indexer_topk_fa80099a4f529287f3be": {
        "role": "scan",
        "sources": [
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_fa80099a4f529287f3be_kernel.cu",
            "cake_dsa_indexer_topk/cake_dsa_indexer_topk_fa80099a4f529287f3be_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
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
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_fa80099a4f529287f3be",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_1e987836f71bb5c3ba52",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_0843306409e1e7fc6092",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_2a0be0fd36be3320b3df",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_6608aa8518faca0527e7",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_9463e969b9b47fe794d1",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_topk_d8608692031a9a3a49cc",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_b206f52857c238bcf951",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_74d7c381e6a2c1219e6c",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_2041b86ba2e56c0591ec",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_topk_e826f6ad8378b7725739",
        "scan:wide:u2:s0:f1": "cake_dsa_indexer_topk_7d171eb3a76d45726f38",
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
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_fa80099a4f529287f3be",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_1e987836f71bb5c3ba52",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_efb859170a00f8c96574",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_9848f45ca3390e0daeef",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_6608aa8518faca0527e7",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_9463e969b9b47fe794d1",
        "scan:split_l6:u1:s0:f0": "cake_dsa_indexer_topk_d8608692031a9a3a49cc",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_43e6b06f87908c5b1172",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_c9efd59500c6c16020e6",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_0bf75ec341ce571b7413",
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
        "scan:l6:u1:s0:f0": "cake_dsa_indexer_topk_70fbcd7e3a2abe75132b",
        "scan:l6:u1:s0:f1": "cake_dsa_indexer_topk_3006bd3aa47e016d0e03",
        "scan:l6:u1:s1:f1": "cake_dsa_indexer_topk_df755465364eecf95460",
        "scan:narrow:u1:s0:f0": "cake_dsa_indexer_topk_efb859170a00f8c96574",
        "scan:narrow:u1:s0:f1": "cake_dsa_indexer_topk_9848f45ca3390e0daeef",
        "scan:narrow:u1:s1:f1": "cake_dsa_indexer_topk_36899ca69836524c0fc3",
        "scan:narrow:u2:s0:f0": "cake_dsa_indexer_topk_6b1f075769a73ee1b176",
        "scan:pair_l6:u1:s0:f0": "cake_dsa_indexer_topk_b136f04b21e35062077d",
        "scan:pair_l6:u1:s0:f1": "cake_dsa_indexer_topk_e840189671a665d54ec9",
        "scan:pair_l6:u1:s1:f1": "cake_dsa_indexer_topk_545825470c274d10bc04",
        "scan:pair_narrow:u1:s1:f1": "cake_dsa_indexer_topk_0941c30bf15fe14a9cc3",
        "scan:pair_narrow:u2:s0:f0": "cake_dsa_indexer_topk_f51560d9a49f174c0ee7",
        "scan:pair_narrow:u2:s0:f1": "cake_dsa_indexer_topk_6ff54f38a7d43aed441b",
        "scan:pair_wide:u1:s1:f0": "cake_dsa_indexer_topk_2d3c54670435ca47d9d2",
        "scan:pair_wide:u1:s1:f1": "cake_dsa_indexer_topk_02fe1c65eb89cba835b0",
        "scan:pair_wide:u2:s0:f0": "cake_dsa_indexer_topk_ced4e89bc4e74d1c752e",
        "scan:pair_wide:u2:s0:f1": "cake_dsa_indexer_topk_5aa34f7e4d8bb3d96fde",
        "scan:split_narrow:u1:s0:f0": "cake_dsa_indexer_topk_43e6b06f87908c5b1172",
        "scan:split_wide:u1:s0:f0": "cake_dsa_indexer_topk_7a8c782839e3650e42e3",
        "scan:split_wide:u2:s0:f0": "cake_dsa_indexer_topk_e453d14c4a8923a43703",
        "scan:wide:u1:s0:f0": "cake_dsa_indexer_topk_c9efd59500c6c16020e6",
        "scan:wide:u1:s0:f1": "cake_dsa_indexer_topk_0bf75ec341ce571b7413",
        "scan:wide:u1:s1:f0": "cake_dsa_indexer_topk_c2598645bc5c4f2e4d18",
        "scan:wide:u1:s1:f1": "cake_dsa_indexer_topk_4fb74943e6794f4d0b19",
        "scan:wide:u2:s0:f0": "cake_dsa_indexer_topk_dce69060d489ec86df87",
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
        "tile_unroll_factor": 1,
        "tile_unroll_rule": None,
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
