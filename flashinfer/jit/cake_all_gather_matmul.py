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
A route whose (world size, dtype, weight layout) class is listed in
``N128_CLASSES`` for the device's architecture also compiles the 128-wide main
twin (``main_n128``) into its sequence module; ``main_block_n`` selects that
twin per call from the frozen wave rule (``SM_COUNT_BY_ARCH`` and the
per-architecture wave cap ``N128_MAX_WAVES_BY_ARCH``) and the table, and the
launcher receives the chosen ``block_n``. ``N`` stays a positive multiple of
``BLOCK_N`` (256) either way.
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
    "cake_all_gather_matmul_21d7e2b21b372ffeab36": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_21d7e2b21b372ffeab36_kernel.cu",
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
    "cake_all_gather_matmul_52f3fc23159650a89719": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_52f3fc23159650a89719_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_5d04e5464f88d3d5ab8c": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_5d04e5464f88d3d5ab8c_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_61c61aabb5967bd8e9d9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_61c61aabb5967bd8e9d9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_6e81f55424d75f150b3b": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_6e81f55424d75f150b3b_kernel.cu",
        ],
        "arches": ["sm_100a"],
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
    "cake_all_gather_matmul_a01c4436c21e679aaed9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a01c4436c21e679aaed9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a55baa3bdb6ce5b39f7e_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_b4f547b76a4b6c14c87b": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_b4f547b76a4b6c14c87b_kernel.cu",
        ],
        "arches": ["sm_100a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_bb046ade8e857469f0c6": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bb046ade8e857469f0c6_kernel.cu",
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
    "cake_all_gather_matmul_c0a6678e73a6e854bfac": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c0a6678e73a6e854bfac_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_c27adeb3c14305afd206": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c27adeb3c14305afd206_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_c37bb42697799791a7c5": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c37bb42697799791a7c5_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_dba24f495a33b1a179e7": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_dba24f495a33b1a179e7_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_ef0a40fd8365adef0679": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ef0a40fd8365adef0679_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_f0cbdce7150397ae0989": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f0cbdce7150397ae0989_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_f2e488e6c8974711b331": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f2e488e6c8974711b331_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [128, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_f85e6d8165a64ac49252": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f85e6d8165a64ac49252_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
}
ROUTES: dict[str, str] = {
    "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
    "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
    "main_bfloat16_ws2_k_major": "cake_all_gather_matmul_bb046ade8e857469f0c6",
    "main_bfloat16_ws2_n_major": "cake_all_gather_matmul_c27adeb3c14305afd206",
    "main_bfloat16_ws4_k_major": "cake_all_gather_matmul_a01c4436c21e679aaed9",
    "main_bfloat16_ws4_k_major_n128": "cake_all_gather_matmul_52f3fc23159650a89719",
    "main_bfloat16_ws4_n_major": "cake_all_gather_matmul_dba24f495a33b1a179e7",
    "main_bfloat16_ws8_k_major": "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e",
    "main_bfloat16_ws8_k_major_n128": "cake_all_gather_matmul_c37bb42697799791a7c5",
    "main_bfloat16_ws8_n_major": "cake_all_gather_matmul_f2e488e6c8974711b331",
    "main_bfloat16_ws8_n_major_n128": "cake_all_gather_matmul_ef0a40fd8365adef0679",
    "main_float16_ws2_k_major": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
    "main_float16_ws2_n_major": "cake_all_gather_matmul_61c61aabb5967bd8e9d9",
    "main_float16_ws4_k_major": "cake_all_gather_matmul_21d7e2b21b372ffeab36",
    "main_float16_ws4_k_major_n128": "cake_all_gather_matmul_b4f547b76a4b6c14c87b",
    "main_float16_ws4_n_major": "cake_all_gather_matmul_f0cbdce7150397ae0989",
    "main_float16_ws8_k_major": "cake_all_gather_matmul_f85e6d8165a64ac49252",
    "main_float16_ws8_k_major_n128": "cake_all_gather_matmul_6e81f55424d75f150b3b",
    "main_float16_ws8_n_major": "cake_all_gather_matmul_5d04e5464f88d3d5ab8c",
    "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
    "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
}
SEQUENCES: dict[str, dict[str, Any]] = {
    "bfloat16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bb046ade8e857469f0c6_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_bb046ade8e857469f0c6",
        },
    },
    "bfloat16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bb046ade8e857469f0c6_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_bb046ade8e857469f0c6",
        },
    },
    "bfloat16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c27adeb3c14305afd206_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_c27adeb3c14305afd206",
        },
    },
    "bfloat16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c27adeb3c14305afd206_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_c27adeb3c14305afd206",
        },
    },
    "bfloat16_ws4_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a01c4436c21e679aaed9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_52f3fc23159650a89719_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_a01c4436c21e679aaed9",
            "main_n128": "cake_all_gather_matmul_52f3fc23159650a89719",
        },
    },
    "bfloat16_ws4_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a01c4436c21e679aaed9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_52f3fc23159650a89719_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_a01c4436c21e679aaed9",
            "main_n128": "cake_all_gather_matmul_52f3fc23159650a89719",
        },
    },
    "bfloat16_ws4_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_dba24f495a33b1a179e7_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_dba24f495a33b1a179e7",
        },
    },
    "bfloat16_ws4_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_dba24f495a33b1a179e7_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_dba24f495a33b1a179e7",
        },
    },
    "bfloat16_ws8_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a55baa3bdb6ce5b39f7e_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c37bb42697799791a7c5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e",
            "main_n128": "cake_all_gather_matmul_c37bb42697799791a7c5",
        },
    },
    "bfloat16_ws8_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a55baa3bdb6ce5b39f7e_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c37bb42697799791a7c5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e",
            "main_n128": "cake_all_gather_matmul_c37bb42697799791a7c5",
        },
    },
    "bfloat16_ws8_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f2e488e6c8974711b331_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ef0a40fd8365adef0679_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f2e488e6c8974711b331",
            "main_n128": "cake_all_gather_matmul_ef0a40fd8365adef0679",
        },
    },
    "bfloat16_ws8_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f2e488e6c8974711b331_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ef0a40fd8365adef0679_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws8_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f2e488e6c8974711b331",
            "main_n128": "cake_all_gather_matmul_ef0a40fd8365adef0679",
        },
    },
    "float16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c0a6678e73a6e854bfac_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
        },
    },
    "float16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c0a6678e73a6e854bfac_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
        },
    },
    "float16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_61c61aabb5967bd8e9d9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_61c61aabb5967bd8e9d9",
        },
    },
    "float16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_61c61aabb5967bd8e9d9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_61c61aabb5967bd8e9d9",
        },
    },
    "float16_ws4_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_21d7e2b21b372ffeab36_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_b4f547b76a4b6c14c87b_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_21d7e2b21b372ffeab36",
            "main_n128": "cake_all_gather_matmul_b4f547b76a4b6c14c87b",
        },
    },
    "float16_ws4_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_21d7e2b21b372ffeab36_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_21d7e2b21b372ffeab36",
        },
    },
    "float16_ws4_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f0cbdce7150397ae0989_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f0cbdce7150397ae0989",
        },
    },
    "float16_ws4_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f0cbdce7150397ae0989_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 4,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f0cbdce7150397ae0989",
        },
    },
    "float16_ws8_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f85e6d8165a64ac49252_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_6e81f55424d75f150b3b_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f85e6d8165a64ac49252",
            "main_n128": "cake_all_gather_matmul_6e81f55424d75f150b3b",
        },
    },
    "float16_ws8_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f85e6d8165a64ac49252_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_k_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_f85e6d8165a64ac49252",
        },
    },
    "float16_ws8_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_5d04e5464f88d3d5ab8c_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_5d04e5464f88d3d5ab8c",
        },
    },
    "float16_ws8_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_5d04e5464f88d3d5ab8c_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws8_n_major_sm_103a.cu",
        ],
        "arches": ["sm_103a"],
        "world_size": 8,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
            "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
            "main": "cake_all_gather_matmul_5d04e5464f88d3d5ab8c",
        },
    },
}
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
# Classes ``[world_size, dtype, b_layout]`` with a delivered ``_n128`` main, per
# architecture; ``main_block_n`` selects the 128-wide twin only inside this table.
N128_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [4, "bfloat16", "k_major"],
        [4, "float16", "k_major"],
        [8, "bfloat16", "k_major"],
        [8, "bfloat16", "n_major"],
        [8, "float16", "k_major"],
    ],
    "sm_103a": [
        [4, "bfloat16", "k_major"],
        [8, "bfloat16", "k_major"],
        [8, "bfloat16", "n_major"],
    ],
}
# Frozen inputs of the N128 selection rule, copied from the kernel module (the
# live SM count only clamps the persistent grid; the N256 wave cap is per
# architecture).
SM_COUNT_BY_ARCH: dict[str, int] = {"sm_100a": 148, "sm_103a": 148}
N128_MAX_WAVES_BY_ARCH: dict[str, int] = {"sm_100a": 2, "sm_103a": 1}

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
# Tile widths of the delivered main programs: the public N256 program and the
# ``_n128`` twin ``main_block_n`` selects per call.
MAIN_BLOCK_NS: tuple[int, ...] = (256, 128)
# Rows pushed per peer chunk; the main kernel consumes chunk ``c`` of a peer
# after the matching readiness word reaches the launch epoch.
CHUNK_ROWS = 19 * BLOCK_M
# Physical layouts of the logical ``[K, N]`` weight, each with its own main
# kernel: ``n_major`` = contiguous ``[K, N]``; ``k_major`` = the ``w.t()`` view
# of a contiguous ``[N, K]`` parameter.
B_LAYOUTS: tuple[str, ...] = ("n_major", "k_major")
MAIN_THREADS = 192
# SM push launch: the per-peer payload of a call, ``padded_rows x K x
# dtype_bytes``, selects it (``push_launch``). Below
# ``PUSH_PAYLOAD_THRESHOLD_BYTES``: ``PUSH_CTAS_PER_PEER`` CTAs per remote peer
# along grid x (one peer per grid y) of the ``peer_push`` program
# (``PUSH_UNROLL`` 16-byte vectors in flight per lane); at or above it:
# ``PUSH_WIDE_CTAS_PER_PEER`` CTAs of the ``peer_push_wide`` program
# (``PUSH_WIDE_UNROLL``). ``PUSH_CARVEOUT`` is both push functions' preferred
# shared-memory carveout (-1 = driver default). ``SM_PUSH_MAX_ROWS``: padded
# local rows up to which the SM push replaces the copy-engine pushes (every
# world size, dtype and architecture). All mirror the Cake export adapter; the
# host sequence compiles the launch constants in and chooses per call itself.
PUSH_CTAS_PER_PEER = 32
PUSH_UNROLL = 1
PUSH_WIDE_CTAS_PER_PEER = 8
PUSH_WIDE_UNROLL = 8
PUSH_PAYLOAD_THRESHOLD_BYTES = 4194304
PUSH_CARVEOUT = -1
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


def _check_block_n(block_n: int) -> int:
    if type(block_n) is not int or block_n not in MAIN_BLOCK_NS:
        raise ValueError(f"block_n must be one of {MAIN_BLOCK_NS}, got {block_n!r}")
    return block_n


def main_tiles(rows: int, n: int, *, world_size: int, block_n: int) -> int:
    """Output tiles of one call: ``world_size * ceil128(rows) / 128 * n / block_n``."""

    _check_block_n(block_n)
    n = int(n)
    if n <= 0 or n % block_n:
        raise ValueError(f"N must be a positive multiple of {block_n}, got {n}")
    return int(world_size) * (padded_rows(rows) // BLOCK_M) * (n // block_n)


def _n128_classes(arch: str) -> frozenset[tuple[int, str, str]]:
    """``(world_size, dtype, b_layout)`` classes with a delivered ``_n128`` main on
    ``arch`` (the emitted table holds them as three-element lists)."""

    if arch not in N128_CLASSES:
        raise ValueError(
            f"N128_CLASSES has no entry for arch {arch!r} "
            f"(known: {sorted(N128_CLASSES)})"
        )
    return frozenset(
        (int(world_size), str(dtype_name), str(b_layout))
        for world_size, dtype_name, b_layout in N128_CLASSES[arch]
    )


def main_block_n(
    rows: int,
    n: int,
    *,
    world_size: int,
    arch: str,
    dtype_name: str,
    b_layout: str,
) -> int:
    """Tile width (256 or 128) of the main program for one call.

    ws2 always runs the N256 program (``arch`` is validated for every world
    size; an unknown arch raises, never defaults). At ws4/ws8 the N128 twin
    is selected when halving the tile width fills the SMs with fewer idle slots
    (``ceil(T256 / S) > ceil(2 * T256 / S) / 2`` for ``T256`` N256 tiles and
    ``S = SM_COUNT_BY_ARCH[arch]``), the call spans at most
    ``N128_MAX_WAVES_BY_ARCH[arch]`` N256 waves, and the ``(world_size,
    dtype_name, b_layout)`` class has a delivered ``_n128`` program on ``arch``
    (``N128_CLASSES``). The rule mirrors the kernel module's ``main_block_n``;
    the table restricts it to delivered programs."""

    if arch not in SM_COUNT_BY_ARCH or arch not in N128_MAX_WAVES_BY_ARCH:
        raise ValueError(
            f"main_block_n has no SM count or N128 wave cap for arch {arch!r} "
            f"(known: {sorted(set(SM_COUNT_BY_ARCH) & set(N128_MAX_WAVES_BY_ARCH))})"
        )
    world_size = int(world_size)
    if world_size not in (4, 8):
        return 256
    sm_count = SM_COUNT_BY_ARCH[arch]
    max_waves = N128_MAX_WAVES_BY_ARCH[arch]
    t256 = main_tiles(rows, n, world_size=world_size, block_n=256)
    w256 = -(-t256 // sm_count)
    w128 = -(-(2 * t256) // sm_count)
    qualify = 2 * w256 > w128
    cap = w256 <= max_waves
    delivered = (world_size, str(dtype_name), str(b_layout)) in _n128_classes(arch)
    return 128 if (qualify and cap and max_waves > 0 and delivered) else 256


def main_grid(
    rows: int,
    n: int,
    *,
    world_size: int,
    sm_count: int,
    block_n: int = 256,
) -> tuple[int, int, int]:
    """Launch grid of the persistent main kernel: one CTA per SM, bounded by the
    total output tile count (``world_size * ceil128(rows) / 128 * N / block_n``),
    each CTA striding over the arrival-ordered tile list."""

    if int(world_size) <= 0 or int(sm_count) <= 0:
        message = "the persistent main grid needs a positive world_size and sm_count"
        raise ValueError(message)
    total_tiles = main_tiles(rows, n, world_size=world_size, block_n=block_n)
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
    if int(world_size) < 4 or padded > SM_PUSH_MAX_ROWS:
        return False
    return int(cols) <= sm_push_max_cols(padded)


def push_payload_bytes(padded_rows: int, dtype_bytes: int) -> int:
    """Per-peer payload of one call: the padded local shard, ``padded_rows x K x dtype_bytes``."""

    return int(padded_rows) * K * int(dtype_bytes)


def push_launch(padded_rows: int, dtype_bytes: int) -> tuple[str, int, int]:
    """``(route key, CTAs per peer, vectors in flight)`` of the SM push launch a call takes.

    The mirror of the rule compiled into ``run_push`` (which makes the choice
    itself): the ``peer_push`` program with the 32-CTA grid below
    ``PUSH_PAYLOAD_THRESHOLD_BYTES`` per peer, the ``peer_push_wide`` program
    with the wide grid at or above it.
    """

    if push_payload_bytes(padded_rows, dtype_bytes) >= PUSH_PAYLOAD_THRESHOLD_BYTES:
        return "peer_push_wide", PUSH_WIDE_CTAS_PER_PEER, PUSH_WIDE_UNROLL
    return "peer_push", PUSH_CTAS_PER_PEER, PUSH_UNROLL


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


def _main_route_key(
    world_size: int, dtype_name: str, b_layout: str, block_n: int
) -> str:
    """``ROUTES`` key of one main program (``_n128`` suffix for the 128-wide twin)."""

    width = "" if _check_block_n(block_n) == 256 else f"_n{block_n}"
    return f"main_{dtype_name}_ws{int(world_size)}_{b_layout}{width}"


def main_program(
    world_size: int, dtype_name: str, b_layout: str, block_n: int = 256
) -> str:
    """The main program of one route at tile width ``block_n`` (256, or the delivered
    ``_n128`` twin); a class without that program raises ``ValueError``."""

    key = _main_route_key(world_size, dtype_name, b_layout, block_n)
    if key not in ROUTES:
        raise ValueError(
            f"no all-gather matmul main program is delivered for {key!r} "
            f"(world_size={int(world_size)}, dtype={dtype_name}, "
            f"b_layout={b_layout}, block_n={block_n})"
        )
    return ROUTES[key]


def peer_push_program() -> str:
    return ROUTES["peer_push"]


def peer_push_wide_program() -> str:
    return ROUTES["peer_push_wide"]


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
    """Programs whose device units the sequence module compiles (route order: both
    barrier phases, both SM push programs, the N256 main, then the ``_n128`` main
    when the row's ``programs`` carry ``main_n128``)."""

    row = SEQUENCES[sequence]
    programs = [
        ROUTES["barrier_p0"],
        ROUTES["barrier_p1"],
        ROUTES["peer_push"],
        ROUTES["peer_push_wide"],
        main_program(row["world_size"], row["dtype"], row["b_layout"]),
    ]
    if "main_n128" in row["programs"]:
        programs.append(row["programs"]["main_n128"])
    return tuple(programs)


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
    "PUSH_CARVEOUT",
    "PUSH_CTAS_PER_PEER",
    "PUSH_PAYLOAD_THRESHOLD_BYTES",
    "PUSH_UNROLL",
    "PUSH_WIDE_CTAS_PER_PEER",
    "PUSH_WIDE_UNROLL",
    "SM_PUSH_MAX_ROWS",
    "SM_PUSH_COLS_INTERCEPT",
    "SM_PUSH_COLS_PER_ROW",
    "K",
    "MAIN_BLOCK_NS",
    "MAIN_THREADS",
    "N128_CLASSES",
    "N128_MAX_WAVES_BY_ARCH",
    "PROGRAMS",
    "ROUTES",
    "SEQUENCES",
    "SM_COUNT_BY_ARCH",
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
    "main_block_n",
    "main_grid",
    "main_program",
    "main_tiles",
    "padded_rows",
    "peer_push_program",
    "peer_push_wide_program",
    "push_launch",
    "push_payload_bytes",
    "sequence_name",
    "sequence_programs",
    "sm_push_max_cols",
    "spec",
    "uses_sm_push",
    "weight_layout",
    "weight_tma_source",
]
