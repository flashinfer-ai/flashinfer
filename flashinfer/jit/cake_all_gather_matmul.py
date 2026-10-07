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

JIT loader of the Cake Blackwell all-gather matmul (SM100 / SM103).

Every program is one generated CUDA device translation unit shared by both
architectures (``PROGRAMS`` records its launch block and dynamic shared memory;
``ROUTES`` maps the stage keys to programs). The target compiles one host
sequence module per (world size, dtype, weight layout, architecture)
(``SEQUENCES``): the route's device units together with the rendered one-call
launcher ``run`` (barrier, bridge event, copy-engine pushes with their
readiness epochs, main kernel with cached tensor maps) and, on the routes with
a shape of at most ``SM_PUSH_MAX_ROWS`` padded rows, ``run_push``. The main kernels are tcgen05 code, so a
sequence is compiled for its exact architecture (``sm_100a`` or ``sm_103a``).
The tables are filled by the exporter from the program bundle; the loader never
hashes sources or interprets argument plans.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_all_gather_matmul_0207fc52f60d8009b335": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_0207fc52f60d8009b335_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_215935905975bd30dda9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_215935905975bd30dda9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_40242e9c64d6e66af6f5": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [128, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_72c819bc06bbde0de32b": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72c819bc06bbde0de32b_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_7c66e75c60a4879c3f59": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_7c66e75c60a4879c3f59_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_847f7684f2820f90b306": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_847f7684f2820f90b306_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_8dea171063652cd775f8": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [32, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_a0eff3774e45c89deb2d": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0eff3774e45c89deb2d_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_a80780ede8a1e951deef": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a80780ede8a1e951deef_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_ad1e36214ed76b888e8a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ad1e36214ed76b888e8a_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [32, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_bede04b43f0bd85c421d": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bede04b43f0bd85c421d_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_d3c3aacffd5b8472985d": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d3c3aacffd5b8472985d_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_e485eef2b19e8858c4ea": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_e485eef2b19e8858c4ea_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_fc5668c3f1fb4c1881e9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_fc5668c3f1fb4c1881e9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
}
ROUTES: dict[str, str] = {
    "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
    "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
    "main_bfloat16_ws2_k_major": "cake_all_gather_matmul_0207fc52f60d8009b335",
    "main_bfloat16_ws2_n_major": "cake_all_gather_matmul_72c819bc06bbde0de32b",
    "main_bfloat16_ws4_k_major": "cake_all_gather_matmul_e485eef2b19e8858c4ea",
    "main_bfloat16_ws4_n_major": "cake_all_gather_matmul_a80780ede8a1e951deef",
    "main_bfloat16_ws8_k_major": "cake_all_gather_matmul_847f7684f2820f90b306",
    "main_bfloat16_ws8_n_major": "cake_all_gather_matmul_ad1e36214ed76b888e8a",
    "main_float16_ws2_k_major": "cake_all_gather_matmul_d3c3aacffd5b8472985d",
    "main_float16_ws2_n_major": "cake_all_gather_matmul_bede04b43f0bd85c421d",
    "main_float16_ws4_k_major": "cake_all_gather_matmul_fc5668c3f1fb4c1881e9",
    "main_float16_ws4_n_major": "cake_all_gather_matmul_7c66e75c60a4879c3f59",
    "main_float16_ws8_k_major": "cake_all_gather_matmul_215935905975bd30dda9",
    "main_float16_ws8_n_major": "cake_all_gather_matmul_a0eff3774e45c89deb2d",
    "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
}
SEQUENCES: dict[str, dict[str, Any]] = {
    "bfloat16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_0207fc52f60d8009b335_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_0207fc52f60d8009b335_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72c819bc06bbde0de32b_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "bfloat16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72c819bc06bbde0de32b_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "bfloat16_ws4_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_e485eef2b19e8858c4ea_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws4_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_e485eef2b19e8858c4ea_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws4_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a80780ede8a1e951deef_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "bfloat16_ws4_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a80780ede8a1e951deef_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "bfloat16_ws8_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_847f7684f2820f90b306_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws8_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_847f7684f2820f90b306_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "k_major",
    },
    "bfloat16_ws8_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ad1e36214ed76b888e8a_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "bfloat16_ws8_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ad1e36214ed76b888e8a_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "n_major",
    },
    "float16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d3c3aacffd5b8472985d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d3c3aacffd5b8472985d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bede04b43f0bd85c421d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "n_major",
    },
    "float16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bede04b43f0bd85c421d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "n_major",
    },
    "float16_ws4_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_fc5668c3f1fb4c1881e9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws4_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_fc5668c3f1fb4c1881e9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws4_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_7c66e75c60a4879c3f59_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "n_major",
    },
    "float16_ws4_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_7c66e75c60a4879c3f59_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "n_major",
    },
    "float16_ws8_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_215935905975bd30dda9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws8_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_215935905975bd30dda9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "k_major",
    },
    "float16_ws8_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0eff3774e45c89deb2d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "n_major",
    },
    "float16_ws8_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0eff3774e45c89deb2d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "n_major",
    },
}
COMPILE_FLAGS: list[str] = ["--use_fast_math"]

SOURCE_PACKAGE = "cake_all_gather_matmul"
# The kernels are tcgen05 / TMEM code; one source serves both capabilities,
# each compiled with its exact architecture flag set.
ARCH_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}
K = 8192
BLOCK_M = 128
BLOCK_N = 256
# Rows pushed per peer chunk; the main kernel consumes chunk ``c`` of a peer
# after the matching readiness word reaches the launch epoch.
CHUNK_ROWS = 19 * BLOCK_M
# Physical layouts of the logical ``[K, N]`` weight, each with its own main
# kernel: ``n_major`` = contiguous ``[K, N]``; ``k_major`` = the ``w.t()`` view
# of a contiguous ``[N, K]`` parameter.
B_LAYOUTS: tuple[str, ...] = ("n_major", "k_major")
MAIN_THREADS = 192
# SM push kernel: CTAs per remote peer along grid x, one peer per grid y; padded
# local rows up to which it replaces the copy-engine pushes (every world size,
# dtype and architecture). Both mirror the Cake export adapter.
PUSH_CTAS_PER_PEER = 32
SM_PUSH_MAX_ROWS = 512
# Width ceiling of the SM push route: N <= SM_PUSH_COLS_INTERCEPT - SM_PUSH_COLS_PER_ROW * padded_rows
# (10240 at 128 padded rows, 4096 at 512); wider GEMMs overlap the copy-engine pushes instead.
SM_PUSH_COLS_INTERCEPT = 12288
SM_PUSH_COLS_PER_ROW = 16
# Barrier flag pad: two phase epochs plus two sender-indexed mailbox banks per
# phase and rank (``2 + 2 * 2 * world_size`` uint32 words).
BARRIER_PHASES = 2
BARRIER_BANKS = 2


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    arch: str | None
    sm_count: int


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability, exported architecture and SM count, queried once per device."""

    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    return DeviceFacts(
        capability=capability,
        arch=CAPABILITY_ARCH.get(capability),
        sm_count=int(properties.multi_processor_count),
    )


def barrier_flag_words(world_size: int) -> int:
    """Number of uint32 words of one rank's barrier flag pad."""

    return BARRIER_PHASES + BARRIER_PHASES * BARRIER_BANKS * int(world_size)


def padded_rows(rows: int) -> int:
    """``rows`` rounded up to the 128-row MMA tile."""

    return (int(rows) + BLOCK_M - 1) // BLOCK_M * BLOCK_M


def chunk_plan(rows: int) -> tuple[int, int, int]:
    """``(padded_rows, chunk_rows, num_chunks)`` of the push protocol for ``rows`` local rows."""

    padded = padded_rows(rows)
    chunk_rows = min(padded, CHUNK_ROWS)
    return padded, chunk_rows, (padded + chunk_rows - 1) // chunk_rows


def main_grid(
    rows: int, n: int, *, world_size: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of the persistent main kernel: one CTA per SM, bounded by the
    total output tile count (``world_size * ceil128(rows) / 128 * N / 256``), each
    CTA striding over the arrival-ordered tile list."""

    if int(world_size) <= 0 or int(sm_count) <= 0:
        raise ValueError(
            "the persistent main grid needs a positive world_size and sm_count"
        )
    total_tiles = int(world_size) * (padded_rows(rows) // BLOCK_M) * (int(n) // BLOCK_N)
    return (max(1, min(total_tiles, int(sm_count))), 1, 1)


def sm_push_max_cols(padded_rows: int) -> int:
    """Widest ``N`` the SM push route serves at ``padded_rows`` padded local rows."""

    return SM_PUSH_COLS_INTERCEPT - SM_PUSH_COLS_PER_ROW * int(padded_rows)


def uses_sm_push(*, rows: int, world_size: int, cols: int) -> bool:
    """Padded local rows up to ``SM_PUSH_MAX_ROWS`` with ``N`` up to ``sm_push_max_cols`` push
    through the SM kernel instead of the copy engines at world sizes 4 and 8 (a single peer
    saturates neither route; a wider GEMM overlaps the copy-engine pushes but would wait behind
    the SM push; the copy engine wins in both cases)."""

    padded = padded_rows(rows)
    return (
        int(world_size) >= 4
        and padded <= SM_PUSH_MAX_ROWS
        and int(cols) <= sm_push_max_cols(padded)
    )


def weight_layout(w: torch.Tensor) -> str:
    """Classify the logical ``[K, N]`` weight view by its strides (no copy is ever made)."""

    if w.ndim != 2 or int(w.shape[0]) != K:
        raise ValueError(f"w must be a [{K}, N] view, got shape {tuple(w.shape)}")
    n = int(w.shape[1])
    stride_k, stride_n = (int(stride) for stride in w.stride())
    if stride_n == 1 and stride_k == n:
        return "n_major"
    if stride_k == 1 and stride_n == K:
        return "k_major"
    raise ValueError(
        f"w must be a contiguous [{K}, {n}] tensor or the transposed view of a "
        f"contiguous [{n}, {K}] tensor, got strides {(stride_k, stride_n)}"
    )


def weight_tma_source(w: torch.Tensor, b_layout: str) -> torch.Tensor:
    """The rank-3 tensor the main kernel's ``B`` tensor map is encoded from."""

    n = int(w.shape[1])
    if b_layout == "n_major":
        return w.view(1, K, n)
    return w.t().view(1, n, K)


def barrier_program(phase: int) -> str:
    """The barrier of one phase; its world size is a runtime argument."""

    return ROUTES[f"barrier_p{int(phase)}"]


def main_program(world_size: int, dtype_name: str, b_layout: str) -> str:
    return ROUTES[f"main_{dtype_name}_ws{int(world_size)}_{b_layout}"]


def peer_push_program() -> str:
    return ROUTES["peer_push"]


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake all-gather matmul CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the all-gather matmul package: {relative!r}"
        )
    return _source_dir() / parts[2]


def sequence_name(world_size: int, dtype_name: str, b_layout: str, arch: str) -> str:
    """The host-sequence module of one route: its device units compiled with the one-call launcher."""

    key = f"{dtype_name}_ws{int(world_size)}_{b_layout}_{arch}"
    if key not in SEQUENCES:
        raise ValueError(f"no all-gather matmul host sequence is delivered for {key!r}")
    return key


def sequence_programs(sequence: str) -> tuple[str, ...]:
    """Programs whose device units the sequence module compiles (route order)."""

    row = SEQUENCES[sequence]
    return (
        ROUTES["barrier_p0"],
        ROUTES["barrier_p1"],
        ROUTES["peer_push"],
        ROUTES[f"main_{row['dtype']}_ws{int(row['world_size'])}_{row['b_layout']}"],
    )


@functools.cache
def spec(sequence: str, arch: str) -> JitSpec:
    """Build specification of one host-sequence module for one exact architecture."""

    row = SEQUENCES[sequence]
    if arch not in row["arches"]:
        raise ValueError(
            f"sequence {sequence!r} is delivered for {row['arches']}, not {arch!r}"
        )
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_sequence_{sequence}",
        sources=[_source_path(relative) for relative in row["sources"]],
        extra_cuda_cflags=[*ARCH_FLAGS[arch], *COMPILE_FLAGS],
        extra_include_paths=[_source_dir().parent],
        # The source build carries every math flag in ``COMPILE_FLAGS``; the
        # default ``-use_fast_math`` would otherwise be added or dropped.
        use_fast_math="--use_fast_math" in COMPILE_FLAGS,
    )


@functools.cache
def load(sequence: str, arch: str):
    return spec(sequence, arch).build_and_load()


def launch_block(program: str) -> tuple[int, int, int]:
    block = PROGRAMS[program]["block"]
    return (int(block[0]), int(block[1]), int(block[2]))


def dynamic_smem_bytes(program: str) -> int:
    return int(PROGRAMS[program]["dynamic_smem_bytes"])


__all__ = [
    "ARCH_FLAGS",
    "B_LAYOUTS",
    "BARRIER_BANKS",
    "BARRIER_PHASES",
    "BLOCK_M",
    "BLOCK_N",
    "CAPABILITY_ARCH",
    "CHUNK_ROWS",
    "COMPILE_FLAGS",
    "PUSH_CTAS_PER_PEER",
    "SM_PUSH_MAX_ROWS",
    "SM_PUSH_COLS_INTERCEPT",
    "SM_PUSH_COLS_PER_ROW",
    "K",
    "MAIN_THREADS",
    "PROGRAMS",
    "ROUTES",
    "SEQUENCES",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "DeviceFacts",
    "barrier_flag_words",
    "barrier_program",
    "chunk_plan",
    "device_facts",
    "dynamic_smem_bytes",
    "launch_block",
    "load",
    "main_grid",
    "main_program",
    "padded_rows",
    "peer_push_program",
    "sequence_name",
    "sequence_programs",
    "sm_push_max_cols",
    "spec",
    "uses_sm_push",
    "weight_layout",
    "weight_tma_source",
]
