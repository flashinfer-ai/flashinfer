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
readiness epochs, main kernel with cached tensor maps) and, at world sizes 4
and 8, ``run_push`` for the calls ``uses_sm_push`` routes through the SM push
kernel (the base rule plus the architecture's ``SM_PUSH_BANDS_BY_ARCH`` bands;
the push launch of a call is the architecture's ``PUSH_LAUNCH_ROWS_BY_ARCH``
row of its world size and padded rows, else the per-peer payload rule; no
two-rank call takes that route, so a world-size-2 sequence compiles neither
push program nor ``run_push``). The main kernels are
tcgen05 code, so a sequence is compiled for its exact architecture
(``sm_100a`` or ``sm_103a``); a row whose host sequence is another
architecture's apart from its namespace lists that architecture's sequence
source and names its own namespace through ``sequence_flags``, compile
definitions passed to the sequence source alone.
A route whose (world size, dtype, weight layout) class is listed in
``N128_CLASSES`` for the device's architecture also compiles the 128-wide main
twin (``main_n128``) into its sequence module; ``main_block_n`` selects that
twin per call from the frozen wave rule (``SM_COUNT_BY_ARCH`` and the
per-architecture wave cap ``N128_MAX_WAVES_BY_ARCH``) and the table, and the
launcher receives the chosen ``block_n``. ``N`` stays a positive multiple of
``BLOCK_N`` (256) either way. On an architecture listed True in
``CG2_N128_PRECEDENCE_ARCHES`` a call inside a measured ``CG2_REGIONS``
rectangle of a class listed in ``CG2_CLASSES`` keeps ``block_n = 256`` so that
``main_form`` selects the CTA-pair twin (the kernel module's default-path rule:
the pair takes precedence over the wave rule).
A route whose class is listed in ``CG2_CLASSES`` for the device's architecture
also compiles the CTA-pair twin of the N256 main (``main_cg2``:
``tcgen05.mma.cta_group::2`` over a cluster of two CTAs, each loading its own
128 A rows and 128 B columns per stage); ``main_form`` selects it per N256 call
from the frozen region table (``CG2_REGIONS``: per class one or more measured
rectangles of padded rows x N, pairwise disjoint, per architecture) and the
delivered-class table, then refines the selection by the frozen point table
``CG2_SPILL_POINTS`` gated on ``CG2S_CLASSES``: a call whose class, padded
rows and exact ``N`` name a point runs that point's spill-wave N-split pair
program (``cg2s1`` / ``cg2s2``: the CTA-pair program whose last persistent
wave runs each spill tile as four 256 x 64 or two 256 x 128 narrower pair
tiles on otherwise idle clusters). ``main_grid`` sizes the even grid of CTA
pairs of every pair form, and the launcher receives the chosen form as the
positional form code ``main_form_code(form)`` and launches every pair form
with the cluster-dimension attribute ``CG2_CLUSTER_DIMS``.
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
    "cake_all_gather_matmul_06c1f7cb4aa36c5918bd": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_06c1f7cb4aa36c5918bd_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_17f60837feb11a634b34": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_17f60837feb11a634b34_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_18bb325baf7347c5f496": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_18bb325baf7347c5f496_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_21d7e2b21b372ffeab36": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_21d7e2b21b372ffeab36_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_378badfea679e1bc51d7": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_378badfea679e1bc51d7_kernel.cu",
        ],
        "arches": ["sm_103a"],
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
    "cake_all_gather_matmul_60cbe405e11489692148": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_60cbe405e11489692148_kernel.cu",
        ],
        "arches": ["sm_103a"],
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
    "cake_all_gather_matmul_72ae81d01ed72fe75fe8": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72ae81d01ed72fe75fe8_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_8a259a5d2c971ec7f65c": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8a259a5d2c971ec7f65c_kernel.cu",
        ],
        "arches": ["sm_103a"],
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
    "cake_all_gather_matmul_8ec093514b46431df041": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8ec093514b46431df041_kernel.cu",
        ],
        "arches": ["sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_9d5d20282a6a1082a079": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_9d5d20282a6a1082a079_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_a01c4436c21e679aaed9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a01c4436c21e679aaed9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_a0785a42337a41e57f18": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0785a42337a41e57f18_kernel.cu",
        ],
        "arches": ["sm_103a"],
        "block": [128, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a55baa3bdb6ce5b39f7e_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_aa1b0acc28e1e23913d9": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_aa1b0acc28e1e23913d9_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_b9a6204da826be87d972": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_b9a6204da826be87d972_kernel.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
    "cake_all_gather_matmul_c4d3d85b74918e225506": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c4d3d85b74918e225506_kernel.cu",
        ],
        "arches": ["sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_d047b764232eb38ac279": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d047b764232eb38ac279_kernel.cu",
        ],
        "arches": ["sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_d0fc2abec5d93b9a81ec": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d0fc2abec5d93b9a81ec_kernel.cu",
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
    "cake_all_gather_matmul_e25f7c2c0ad05f7fb9cc": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_e25f7c2c0ad05f7fb9cc_kernel.cu",
        ],
        "arches": ["sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_ea2faa64e8c1ea61c8b8": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ea2faa64e8c1ea61c8b8_kernel.cu",
        ],
        "arches": ["sm_103a"],
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
    "main_bfloat16_ws2_k_major_cg2": "cake_all_gather_matmul_d0fc2abec5d93b9a81ec",
    "main_bfloat16_ws2_n_major": "cake_all_gather_matmul_c27adeb3c14305afd206",
    "main_bfloat16_ws2_n_major_cg2": "cake_all_gather_matmul_06c1f7cb4aa36c5918bd",
    "main_bfloat16_ws4_k_major": "cake_all_gather_matmul_a01c4436c21e679aaed9",
    "main_bfloat16_ws4_k_major_cg2": "cake_all_gather_matmul_18bb325baf7347c5f496",
    "main_bfloat16_ws4_k_major_cg2s2": "cake_all_gather_matmul_b9a6204da826be87d972",
    "main_bfloat16_ws4_k_major_n128": "cake_all_gather_matmul_52f3fc23159650a89719",
    "main_bfloat16_ws4_k_major_po": "cake_all_gather_matmul_ea2faa64e8c1ea61c8b8",
    "main_bfloat16_ws4_n_major": "cake_all_gather_matmul_dba24f495a33b1a179e7",
    "main_bfloat16_ws8_k_major": "cake_all_gather_matmul_a55baa3bdb6ce5b39f7e",
    "main_bfloat16_ws8_k_major_cg1s2": "cake_all_gather_matmul_378badfea679e1bc51d7",
    "main_bfloat16_ws8_k_major_cg2": "cake_all_gather_matmul_17f60837feb11a634b34",
    "main_bfloat16_ws8_k_major_cg2s2": "cake_all_gather_matmul_9d5d20282a6a1082a079",
    "main_bfloat16_ws8_k_major_ks4": "cake_all_gather_matmul_c4d3d85b74918e225506",
    "main_bfloat16_ws8_k_major_n128": "cake_all_gather_matmul_c37bb42697799791a7c5",
    "main_bfloat16_ws8_k_major_n128_ks4": "cake_all_gather_matmul_60cbe405e11489692148",
    "main_bfloat16_ws8_n_major": "cake_all_gather_matmul_f2e488e6c8974711b331",
    "main_bfloat16_ws8_n_major_ks4": "cake_all_gather_matmul_d047b764232eb38ac279",
    "main_bfloat16_ws8_n_major_n128": "cake_all_gather_matmul_ef0a40fd8365adef0679",
    "main_bfloat16_ws8_n_major_n128_ks4": "cake_all_gather_matmul_8a259a5d2c971ec7f65c",
    "main_float16_ws2_k_major": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
    "main_float16_ws2_k_major_cg2": "cake_all_gather_matmul_aa1b0acc28e1e23913d9",
    "main_float16_ws2_n_major": "cake_all_gather_matmul_61c61aabb5967bd8e9d9",
    "main_float16_ws4_k_major": "cake_all_gather_matmul_21d7e2b21b372ffeab36",
    "main_float16_ws4_k_major_cg2": "cake_all_gather_matmul_72ae81d01ed72fe75fe8",
    "main_float16_ws4_n_major": "cake_all_gather_matmul_f0cbdce7150397ae0989",
    "main_float16_ws8_k_major": "cake_all_gather_matmul_f85e6d8165a64ac49252",
    "main_float16_ws8_k_major_ks4": "cake_all_gather_matmul_8ec093514b46431df041",
    "main_float16_ws8_k_major_n128": "cake_all_gather_matmul_6e81f55424d75f150b3b",
    "main_float16_ws8_n_major": "cake_all_gather_matmul_5d04e5464f88d3d5ab8c",
    "main_float16_ws8_n_major_ks4": "cake_all_gather_matmul_e25f7c2c0ad05f7fb9cc",
    "peer_push": "cake_all_gather_matmul_40242e9c64d6e66af6f5",
    "peer_push_wide": "cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3",
    "peer_push_wide_ks4": "cake_all_gather_matmul_a0785a42337a41e57f18",
}
SEQUENCES: dict[str, dict[str, Any]] = {
    "bfloat16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bb046ade8e857469f0c6_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d0fc2abec5d93b9a81ec_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_bb046ade8e857469f0c6",
            "main_cg2": "cake_all_gather_matmul_d0fc2abec5d93b9a81ec",
        },
    },
    "bfloat16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bb046ade8e857469f0c6_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d0fc2abec5d93b9a81ec_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_k_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_e68ff7dc1d24a48a"],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_bb046ade8e857469f0c6",
            "main_cg2": "cake_all_gather_matmul_d0fc2abec5d93b9a81ec",
        },
    },
    "bfloat16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c27adeb3c14305afd206_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_06c1f7cb4aa36c5918bd_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_c27adeb3c14305afd206",
            "main_cg2": "cake_all_gather_matmul_06c1f7cb4aa36c5918bd",
        },
    },
    "bfloat16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c27adeb3c14305afd206_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_06c1f7cb4aa36c5918bd_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws2_n_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_3d4c6b672fccc2e9"],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "bfloat16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_c27adeb3c14305afd206",
            "main_cg2": "cake_all_gather_matmul_06c1f7cb4aa36c5918bd",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_18bb325baf7347c5f496_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_b9a6204da826be87d972_kernel.cu",
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
            "main_cg2": "cake_all_gather_matmul_18bb325baf7347c5f496",
            "main_cg2s2": "cake_all_gather_matmul_b9a6204da826be87d972",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_18bb325baf7347c5f496_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_b9a6204da826be87d972_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_ea2faa64e8c1ea61c8b8_kernel.cu",
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
            "main_cg2": "cake_all_gather_matmul_18bb325baf7347c5f496",
            "main_cg2s2": "cake_all_gather_matmul_b9a6204da826be87d972",
            "main_po": "cake_all_gather_matmul_ea2faa64e8c1ea61c8b8",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_n_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_4dd55b8a57fc515a"],
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_17f60837feb11a634b34_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_9d5d20282a6a1082a079_kernel.cu",
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
            "main_cg2": "cake_all_gather_matmul_17f60837feb11a634b34",
            "main_cg2s2": "cake_all_gather_matmul_9d5d20282a6a1082a079",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_17f60837feb11a634b34_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_9d5d20282a6a1082a079_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_378badfea679e1bc51d7_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0785a42337a41e57f18_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c4d3d85b74918e225506_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_60cbe405e11489692148_kernel.cu",
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
            "main_cg2": "cake_all_gather_matmul_17f60837feb11a634b34",
            "main_cg2s2": "cake_all_gather_matmul_9d5d20282a6a1082a079",
            "main_cg1s2": "cake_all_gather_matmul_378badfea679e1bc51d7",
            "peer_push_wide_ks4": "cake_all_gather_matmul_a0785a42337a41e57f18",
            "main_ks4": "cake_all_gather_matmul_c4d3d85b74918e225506",
            "main_n128_ks4": "cake_all_gather_matmul_60cbe405e11489692148",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0785a42337a41e57f18_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_d047b764232eb38ac279_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8a259a5d2c971ec7f65c_kernel.cu",
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
            "peer_push_wide_ks4": "cake_all_gather_matmul_a0785a42337a41e57f18",
            "main_ks4": "cake_all_gather_matmul_d047b764232eb38ac279",
            "main_n128_ks4": "cake_all_gather_matmul_8a259a5d2c971ec7f65c",
        },
    },
    "float16_ws2_k_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c0a6678e73a6e854bfac_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_aa1b0acc28e1e23913d9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_100a.cu",
        ],
        "arches": ["sm_100a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
            "main_cg2": "cake_all_gather_matmul_aa1b0acc28e1e23913d9",
        },
    },
    "float16_ws2_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_c0a6678e73a6e854bfac_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_aa1b0acc28e1e23913d9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_k_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_65a8f6938e94e471"],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "k_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
            "main": "cake_all_gather_matmul_c0a6678e73a6e854bfac",
            "main_cg2": "cake_all_gather_matmul_aa1b0acc28e1e23913d9",
        },
    },
    "float16_ws2_n_major_sm_100a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
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
            "main": "cake_all_gather_matmul_61c61aabb5967bd8e9d9",
        },
    },
    "float16_ws2_n_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_61c61aabb5967bd8e9d9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws2_n_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_cd42fdd423ee95ad"],
        "arches": ["sm_103a"],
        "world_size": 2,
        "dtype": "float16",
        "b_layout": "n_major",
        "programs": {
            "barrier_p0": "cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9",
            "barrier_p1": "cake_all_gather_matmul_8dea171063652cd775f8",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72ae81d01ed72fe75fe8_kernel.cu",
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
            "main_cg2": "cake_all_gather_matmul_72ae81d01ed72fe75fe8",
        },
    },
    "float16_ws4_k_major_sm_103a": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8dea171063652cd775f8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_40242e9c64d6e66af6f5_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_21d7e2b21b372ffeab36_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_72ae81d01ed72fe75fe8_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_k_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_126e6d87bacceb57"],
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
            "main_cg2": "cake_all_gather_matmul_72ae81d01ed72fe75fe8",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_float16_ws4_n_major_sm_100a.cu",
        ],
        "sequence_flags": ["-DCAKE_SEQ_SHIM=cake_host_shim_seq_2bf1da83f19d9287"],
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0785a42337a41e57f18_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_8ec093514b46431df041_kernel.cu",
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
            "peer_push_wide_ks4": "cake_all_gather_matmul_a0785a42337a41e57f18",
            "main_ks4": "cake_all_gather_matmul_8ec093514b46431df041",
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
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_a0785a42337a41e57f18_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_e25f7c2c0ad05f7fb9cc_kernel.cu",
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
            "peer_push_wide_ks4": "cake_all_gather_matmul_a0785a42337a41e57f18",
            "main_ks4": "cake_all_gather_matmul_e25f7c2c0ad05f7fb9cc",
        },
    },
}
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
# Classes ``[world_size, dtype, b_layout]`` with a delivered ``_n128`` main, per
# architecture; ``main_block_n`` selects the 128-wide twin only inside this table.
N128_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [4, "bfloat16", "k_major"],
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
# Per-architecture precedence of the CTA pair over the N128 wave rule (copied
# from the kernel module): where True, ``main_block_n`` keeps a call inside a
# ``CG2_REGIONS`` rectangle of a ``CG2_CLASSES`` class at 256; an arch outside
# the table is an error, never a default.
CG2_N128_PRECEDENCE_ARCHES: dict[str, bool] = {"sm_100a": True, "sm_103a": False}
# Per-architecture SM push bands ``[world_size, padded_rows, max_cols]`` beyond
# the base rule of ``uses_sm_push`` (copied from the kernel module; an arch
# outside the table is an error, never the base rule alone).
SM_PUSH_BANDS_BY_ARCH: dict[str, list[list[int]]] = {
    "sm_100a": [[8, 1024, 1619]],
    "sm_103a": [[8, 1024, 1619]],
}
# Per-architecture SM push launch rows ``[world_size, padded_rows, ctas_per_peer,
# unroll]`` beyond the payload rule of ``push_launch`` (copied from the kernel
# module): an SM-push call of exactly ``padded_rows`` padded rows at
# ``world_size`` launches ``ctas_per_peer`` CTAs per peer of the push program
# with ``unroll`` vectors in flight; an arch outside the table is an error,
# never the payload rule alone.
PUSH_LAUNCH_ROWS_BY_ARCH: dict[str, list[list[int]]] = {
    "sm_100a": [[4, 512, 16, 8]],
    "sm_103a": [[4, 512, 16, 8], [8, 1024, 16, 8]],
}
# Per-architecture SM push readiness slice rows ``[world_size, padded_rows,
# k_slices]`` (copied from the kernel module): an SM-push call of exactly
# ``padded_rows`` padded rows at ``world_size`` publishes ``k_slices`` readiness
# words per (peer, chunk) -- its host sequence launches the per-K-slice push
# twin of the call's launch and the per-K-slice main twin of the call's width
# and form (one is compiled for every width and form a call at those rows can
# resolve to) and needs ``k_slices`` times the readiness words and ticket
# counters; every other call publishes one word; an arch outside the table is
# an error, never one word by default.
PUSH_KSLICES_BY_ARCH: dict[str, list[list[int]]] = {
    "sm_100a": [],
    "sm_103a": [[8, 512, 4]],
}
# Classes ``[world_size, dtype, b_layout]`` with a delivered ``_cg2`` main (the
# CTA-pair twin of the N256 main), per architecture; ``main_form`` selects the
# twin only inside this table.
CG2_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [2, "bfloat16", "k_major"],
        [2, "bfloat16", "n_major"],
        [2, "float16", "k_major"],
        [4, "bfloat16", "k_major"],
        [4, "float16", "k_major"],
        [8, "bfloat16", "k_major"],
    ],
    "sm_103a": [
        [2, "bfloat16", "k_major"],
        [2, "bfloat16", "n_major"],
        [2, "float16", "k_major"],
        [4, "bfloat16", "k_major"],
        [4, "float16", "k_major"],
        [8, "bfloat16", "k_major"],
    ],
}
# Frozen input of the pair selection rule, copied from the kernel module: the
# measured rectangles of the twin per architecture, ``[world_size, dtype,
# b_layout, min_padded_rows, max_padded_rows, min_cols, max_cols]`` entries (a
# class may list several, pairwise disjoint; an architecture without one lists
# nothing); an N256 call runs the twin iff some rectangle of its class satisfies
# ``min_padded_rows <= ceil128(rows) <= max_padded_rows`` and
# ``min_cols <= N <= max_cols``.
CG2_REGIONS: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [2, "bfloat16", "n_major", 2048, 2048, 2048, 2048],
        [2, "bfloat16", "n_major", 4096, 4096, 2048, 2048],
        [4, "bfloat16", "k_major", 1024, 1024, 2560, 2560],
        [4, "bfloat16", "k_major", 1152, 1152, 2560, 2560],
        [4, "bfloat16", "k_major", 2048, 2048, 2560, 2560],
        [4, "bfloat16", "k_major", 512, 512, 14336, 14336],
        [4, "bfloat16", "k_major", 1024, 1024, 14336, 14336],
        [8, "bfloat16", "k_major", 512, 512, 7168, 7168],
        [8, "bfloat16", "k_major", 1024, 1024, 7168, 7168],
        [2, "bfloat16", "n_major", 1024, 1024, 2048, 2048],
        [2, "bfloat16", "k_major", 1024, 1024, 2048, 2048],
        [2, "float16", "k_major", 1024, 1024, 2048, 2048],
        [4, "bfloat16", "k_major", 512, 512, 2560, 2560],
        [4, "float16", "k_major", 512, 512, 2560, 2560],
    ],
    "sm_103a": [
        [2, "bfloat16", "n_major", 2048, 5760, 1, 2048],
        [4, "bfloat16", "k_major", 1024, 1536, 1, 6058],
        [4, "bfloat16", "k_major", 512, 512, 14336, 14336],
        [8, "bfloat16", "k_major", 512, 512, 7168, 7168],
        [2, "bfloat16", "n_major", 1024, 1024, 2048, 2048],
        [2, "bfloat16", "k_major", 1024, 1024, 2048, 2048],
        [2, "float16", "k_major", 1024, 1024, 2048, 2048],
        [4, "bfloat16", "k_major", 512, 512, 2560, 2560],
        [4, "float16", "k_major", 512, 512, 2560, 2560],
    ],
}
# Classes ``[world_size, dtype, b_layout, form]`` with a delivered spill-wave
# N-split pair main (``form`` ``cg2s1`` / ``cg2s2``), per architecture;
# ``main_form`` selects a point's program only inside this table.
CG2S_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [4, "bfloat16", "k_major", "cg2s2"],
        [8, "bfloat16", "k_major", "cg2s2"],
    ],
    "sm_103a": [
        [4, "bfloat16", "k_major", "cg2s2"],
        [8, "bfloat16", "k_major", "cg2s2"],
    ],
}
# Frozen refinement of the pair selection rule, copied from the kernel module:
# ``[world_size, dtype, b_layout, padded_rows, n, form]`` points per
# architecture (an architecture without one lists nothing); an N256 call the
# rectangle rule selects runs the point's spill-wave N-split pair program iff
# its class, ``ceil128(rows)`` and exact ``N`` match a point, else the CTA-pair
# twin.
CG2_SPILL_POINTS: dict[str, list[list[Any]]] = {
    "sm_100a": [
        [4, "bfloat16", "k_major", 512, 14336, "cg2s2"],
        [8, "bfloat16", "k_major", 512, 7168, "cg2s2"],
        [4, "bfloat16", "k_major", 2048, 2560, "cg2s2"],
    ],
    "sm_103a": [
        [4, "bfloat16", "k_major", 512, 14336, "cg2s2"],
        [8, "bfloat16", "k_major", 512, 7168, "cg2s2"],
    ],
}
# Classes ``[world_size, dtype, b_layout, form]`` with a delivered tile-order
# main (``form`` ``po`` / ``pi``: the one-CTA N256 main with another tile ->
# (peer, B panel) map of its persistent walk; the one-CTA launch), per
# architecture; ``main_form`` selects a point's program only inside this table.
ORDER_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [],
    "sm_103a": [[4, "bfloat16", "k_major", "po"]],
}
# Frozen refinement of the one-CTA selection, copied from the kernel module:
# ``[world_size, dtype, b_layout, padded_rows, n, form]`` points per
# architecture (an architecture without one lists nothing; ``padded_rows`` is
# always 128, the one-M-tile geometry the twins are authored for); an N256 call
# the pair rule leaves on the one-CTA program runs the point's tile-order twin
# iff its class, ``ceil128(rows)`` and exact ``N`` match a point.
TILE_ORDER_POINTS: dict[str, list[list[Any]]] = {
    "sm_100a": [],
    "sm_103a": [[4, "bfloat16", "k_major", 128, 14336, "po"]],
}
# Classes ``[world_size, dtype, b_layout, form]`` with a delivered one-CTA
# last-pass N-split main (``form`` ``cg1s2``: the one-CTA N256 main whose last
# persistent pass runs each output tile as two 128-wide half items on otherwise
# idle CTAs; the one-CTA launch), per architecture; ``main_form`` selects a
# point's program only inside this table.
CG1S_CLASSES: dict[str, list[list[Any]]] = {
    "sm_100a": [],
    "sm_103a": [[8, "bfloat16", "k_major", "cg1s2"]],
}
# Frozen refinement of the one-CTA selection, copied from the kernel module:
# ``[world_size, dtype, b_layout, padded_rows, n, form]`` points per
# architecture (an architecture without one lists nothing); an N256 call the
# pair rule leaves on the one-CTA program and no tile-order point names runs
# the point's last-pass N-split twin iff its class, ``ceil128(rows)`` and exact
# ``N`` match a point.
CG1S_POINTS: dict[str, list[list[Any]]] = {
    "sm_100a": [],
    "sm_103a": [[8, "bfloat16", "k_major", 1152, 1280, "cg1s2"]],
}

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
# Forms of the delivered N256 main: the one-CTA program (``cg1``), its CTA-pair
# twin (``cg2``, ``tcgen05.mma.cta_group::2``), the twin's spill-wave N-split
# pair twins (``cg2s1`` / ``cg2s2``), the one-CTA program's tile-order twins
# (``po`` / ``pi``) and its last-pass N-split twin (``cg1s2``) ``main_form``
# selects per call; the index of a form here is the positional form code the
# host sequence takes (``main_form_code``). Every pair form launches clusters of
# ``CG2_CLUSTER_DIMS`` CTAs (the pair); the tile-order and last-pass N-split
# forms take the one-CTA launch and grid.
MAIN_FORMS: tuple[str, ...] = ("cg1", "cg2", "cg2s1", "cg2s2", "po", "pi", "cg1s2")
PAIR_FORMS: tuple[str, ...] = ("cg2", "cg2s1", "cg2s2")
ORDER_FORMS: tuple[str, ...] = ("po", "pi")
ONE_CTA_SPLIT_FORMS: tuple[str, ...] = ("cg1s2",)
CG2_CLUSTER_DIMS: tuple[int, int, int] = (2, 1, 1)
# Rows pushed per peer chunk (19 * BLOCK_M); the main kernel consumes chunk
# ``c`` of a peer after the matching readiness word reaches the launch epoch.
# A literal: the export's route contract runs the loader's literals and pure
# functions in a scratch namespace, and ``chunk_plan`` (the pair tiles) reads it.
CHUNK_ROWS = 2432
# Physical layouts of the logical ``[K, N]`` weight, each with its own main
# kernel: ``n_major`` = contiguous ``[K, N]``; ``k_major`` = the ``w.t()`` view
# of a contiguous ``[N, K]`` parameter.
B_LAYOUTS: tuple[str, ...] = ("n_major", "k_major")
MAIN_THREADS = 192
# SM push launch: the architecture's ``PUSH_LAUNCH_ROWS_BY_ARCH`` row of the
# call's (world size, padded rows) when it lists one, else the per-peer payload
# of the call, ``padded_rows x K x dtype_bytes``, selects it (``push_launch``).
# Below ``PUSH_PAYLOAD_THRESHOLD_BYTES``: ``PUSH_CTAS_PER_PEER`` CTAs per remote
# peer along grid x (one peer per grid y) of the ``peer_push`` program
# (``PUSH_UNROLL`` 16-byte vectors in flight per lane); at or above it:
# ``PUSH_WIDE_CTAS_PER_PEER`` CTAs of the ``peer_push_wide`` program
# (``PUSH_WIDE_UNROLL``). ``PUSH_CARVEOUT`` is both push functions' preferred
# shared-memory carveout (-1 = driver default). ``SM_PUSH_MAX_ROWS``: padded
# local rows up to which the base rule routes a ws4/ws8 call through the SM
# push (``SM_PUSH_BANDS_BY_ARCH`` adds per-architecture bands beyond it). All
# mirror the Cake export adapter; the host sequence compiles the launch
# constants, its launch rows and its bands in and chooses per call itself.
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


def _check_form(form: str) -> str:
    if type(form) is not str or form not in MAIN_FORMS:
        raise ValueError(f"form must be one of {MAIN_FORMS}, got {form!r}")
    return form


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


def _cg2_classes(arch: str) -> frozenset[tuple[int, str, str]]:
    """``(world_size, dtype, b_layout)`` classes with a delivered ``_cg2`` main on
    ``arch`` (the emitted table holds them as three-element lists)."""

    if arch not in CG2_CLASSES:
        raise ValueError(
            f"CG2_CLASSES has no entry for arch {arch!r} (known: {sorted(CG2_CLASSES)})"
        )
    return frozenset(
        (int(world_size), str(dtype_name), str(b_layout))
        for world_size, dtype_name, b_layout in CG2_CLASSES[arch]
    )


def _cg2_regions(
    arch: str, world_size: int, dtype_name: str, b_layout: str
) -> list[tuple[int, int, int, int]]:
    """``(min_padded_rows, max_padded_rows, min_cols, max_cols)`` of every measured
    rectangle of the pair twin for one class on ``arch`` (the emitted
    ``CG2_REGIONS`` table, in table order); empty when the class has none there.
    Several rectangles per class are legal; an unknown arch raises."""

    if arch not in CG2_REGIONS:
        raise ValueError(
            f"CG2_REGIONS has no entry for arch {arch!r} (known: {sorted(CG2_REGIONS)})"
        )
    key = (int(world_size), str(dtype_name), str(b_layout))
    return [
        (int(entry[3]), int(entry[4]), int(entry[5]), int(entry[6]))
        for entry in CG2_REGIONS[arch]
        if (int(entry[0]), str(entry[1]), str(entry[2])) == key
    ]


def _cg2_rectangle_qualifies(
    arch: str, world_size: int, dtype_name: str, b_layout: str, padded: int, cols: int
) -> bool:
    """Whether some measured rectangle of the class on ``arch`` (``CG2_REGIONS``)
    contains the call: ``min_padded_rows <= padded <= max_padded_rows`` and
    ``min_cols <= cols <= max_cols``. Shared by ``main_block_n`` and ``main_form``."""

    rectangles = _cg2_regions(arch, world_size, dtype_name, b_layout)
    return any(
        min_rows <= padded <= max_rows and min_cols <= cols <= max_cols
        for min_rows, max_rows, min_cols, max_cols in rectangles
    )


def _cg2s_classes(arch: str) -> frozenset[tuple[int, str, str, str]]:
    """``(world_size, dtype, b_layout, form)`` classes with a delivered spill-wave
    N-split pair main on ``arch`` (the emitted ``CG2S_CLASSES`` table); an
    unknown arch raises."""

    if arch not in CG2S_CLASSES:
        raise ValueError(
            f"CG2S_CLASSES has no entry for arch {arch!r} "
            f"(known: {sorted(CG2S_CLASSES)})"
        )
    return frozenset(
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in CG2S_CLASSES[arch]
    )


def _order_classes(arch: str) -> frozenset[tuple[int, str, str, str]]:
    """``(world_size, dtype, b_layout, form)`` classes with a delivered tile-order
    main on ``arch`` (the emitted ``ORDER_CLASSES`` table); an unknown arch
    raises."""

    if arch not in ORDER_CLASSES:
        raise ValueError(
            f"ORDER_CLASSES has no entry for arch {arch!r} "
            f"(known: {sorted(ORDER_CLASSES)})"
        )
    return frozenset(
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in ORDER_CLASSES[arch]
    )


def _tile_order_point(
    arch: str,
    world_size: int,
    dtype_name: str,
    b_layout: str,
    padded: int,
    cols: int,
) -> str | None:
    """Form of the tile-order point of ``TILE_ORDER_POINTS`` that names the class,
    padded rows and exact ``N`` on ``arch``, or ``None`` when no point does; an
    unknown arch raises."""

    if arch not in TILE_ORDER_POINTS:
        raise ValueError(
            f"TILE_ORDER_POINTS has no entry for arch {arch!r} "
            f"(known: {sorted(TILE_ORDER_POINTS)})"
        )
    key = (int(world_size), str(dtype_name), str(b_layout), int(padded), int(cols))
    for entry in TILE_ORDER_POINTS[arch]:
        world, dtype, layout, rows, n, form = entry
        if (int(world), str(dtype), str(layout), int(rows), int(n)) == key:
            return str(form)
    return None


def _cg1s_classes(arch: str) -> frozenset[tuple[int, str, str, str]]:
    """``(world_size, dtype, b_layout, form)`` classes with a delivered one-CTA
    last-pass N-split main on ``arch`` (the emitted ``CG1S_CLASSES`` table); an
    unknown arch raises."""

    if arch not in CG1S_CLASSES:
        raise ValueError(
            f"CG1S_CLASSES has no entry for arch {arch!r} "
            f"(known: {sorted(CG1S_CLASSES)})"
        )
    return frozenset(
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in CG1S_CLASSES[arch]
    )


def _cg1s_point(
    arch: str,
    world_size: int,
    dtype_name: str,
    b_layout: str,
    padded: int,
    cols: int,
) -> str | None:
    """Form of the one-CTA last-pass N-split point of ``CG1S_POINTS`` that names
    the class, padded rows and exact ``N`` on ``arch``, or ``None`` when no point
    does; an unknown arch raises."""

    if arch not in CG1S_POINTS:
        raise ValueError(
            f"CG1S_POINTS has no entry for arch {arch!r} (known: {sorted(CG1S_POINTS)})"
        )
    key = (int(world_size), str(dtype_name), str(b_layout), int(padded), int(cols))
    for entry in CG1S_POINTS[arch]:
        world, dtype, layout, rows, n, form = entry
        if (int(world), str(dtype), str(layout), int(rows), int(n)) == key:
            return str(form)
    return None


def _cg2_spill_point(
    arch: str,
    world_size: int,
    dtype_name: str,
    b_layout: str,
    padded: int,
    cols: int,
) -> str | None:
    """Form of the spill-wave N-split point of ``CG2_SPILL_POINTS`` that names the
    class, padded rows and exact ``N`` on ``arch``, or ``None`` when no point
    does; an unknown arch raises."""

    if arch not in CG2_SPILL_POINTS:
        raise ValueError(
            f"CG2_SPILL_POINTS has no entry for arch {arch!r} "
            f"(known: {sorted(CG2_SPILL_POINTS)})"
        )
    key = (int(world_size), str(dtype_name), str(b_layout), int(padded), int(cols))
    for entry in CG2_SPILL_POINTS[arch]:
        world, dtype, layout, rows, n, form = entry
        if (int(world), str(dtype), str(layout), int(rows), int(n)) == key:
            return str(form)
    return None


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
    (``N128_CLASSES``). On an architecture listed True in
    ``CG2_N128_PRECEDENCE_ARCHES`` a call inside a measured ``CG2_REGIONS``
    rectangle of a class listed in ``CG2_CLASSES`` keeps 256 (the pair takes
    precedence over the wave rule). The rule mirrors the kernel module's
    ``main_block_n`` in its default path; the tables restrict it to delivered
    programs."""

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
    pair_key = (world_size, str(dtype_name), str(b_layout))
    if (
        CG2_N128_PRECEDENCE_ARCHES[arch]
        and pair_key in _cg2_classes(arch)
        and _cg2_rectangle_qualifies(
            arch, world_size, dtype_name, b_layout, padded_rows(rows), int(n)
        )
    ):
        return 256
    delivered = (world_size, str(dtype_name), str(b_layout)) in _n128_classes(arch)
    return 128 if (qualify and cap and max_waves > 0 and delivered) else 256


def main_form(
    rows: int,
    n: int,
    *,
    world_size: int,
    arch: str,
    dtype_name: str,
    b_layout: str,
    block_n: int,
) -> str:
    """Form (``"cg1"``, ``"cg2"``, ``"cg2s1"``, ``"cg2s2"``, ``"po"``, ``"pi"`` or
    ``"cg1s2"``) of the main program for one call.

    The CTA-pair twin (``cg2``) is selected for an N256 call whose
    ``(world_size, dtype_name, b_layout)`` class has a measured rectangle on
    ``arch`` (``CG2_REGIONS``) and a delivered ``_cg2`` program there
    (``CG2_CLASSES``), when some rectangle of the class satisfies
    ``min_padded_rows <= ceil128(rows) <= max_padded_rows`` and
    ``min_cols <= n <= max_cols``. Such a call whose class, ``ceil128(rows)``
    and exact ``n`` name a point of ``CG2_SPILL_POINTS`` on ``arch`` runs the
    point's spill-wave N-split pair program instead, when that program is
    delivered for the class there (``CG2S_CLASSES``). A 128-wide call, a class
    without a rectangle and a call outside every rectangle of its class run
    the one-CTA program -- or, when the N256 call's class, ``ceil128(rows)``
    and exact ``n`` name a point of ``TILE_ORDER_POINTS`` on ``arch`` whose
    program is delivered for the class there (``ORDER_CLASSES``), the point's
    tile-order twin of the one-CTA program (the same launch and grid); else,
    when they name a point of ``CG1S_POINTS`` on ``arch`` whose program is
    delivered for the class there (``CG1S_CLASSES``), the one-CTA main's
    last-pass N-split twin (the same launch and grid). The rule mirrors the
    kernel module's ``main_form`` with its frozen region and point tables; an
    unknown arch raises, never defaults."""

    if arch not in CG2_REGIONS:
        raise ValueError(
            f"main_form has no pair-form regions for arch {arch!r} "
            f"(known: {sorted(CG2_REGIONS)})"
        )
    if _check_block_n(block_n) != 256:
        return "cg1"
    main_tiles(rows, n, world_size=world_size, block_n=256)
    delivered = (int(world_size), str(dtype_name), str(b_layout)) in _cg2_classes(arch)
    padded = padded_rows(rows)
    cols = int(n)
    qualify = _cg2_rectangle_qualifies(
        arch, world_size, dtype_name, b_layout, padded, cols
    )
    key = (int(world_size), str(dtype_name), str(b_layout))
    if not (qualify and delivered):
        order = _tile_order_point(arch, world_size, dtype_name, b_layout, padded, cols)
        if order is not None and (*key, order) in _order_classes(arch):
            return order
        split = _cg1s_point(arch, world_size, dtype_name, b_layout, padded, cols)
        if split is not None and (*key, split) in _cg1s_classes(arch):
            return split
        return "cg1"
    point = _cg2_spill_point(arch, world_size, dtype_name, b_layout, padded, cols)
    if point is not None and (*key, point) in _cg2s_classes(arch):
        return point
    return "cg2"


def main_form_code(form: str) -> int:
    """The positional form code the host sequence takes after ``block_n``: 0 the
    one-CTA main, 1 the CTA-pair twin, 2 / 3 its spill-wave N-split pair twins,
    4 / 5 the one-CTA main's tile-order twins, 6 its last-pass N-split twin (the
    index into ``MAIN_FORMS``)."""

    return MAIN_FORMS.index(_check_form(form))


def main_pair_tiles(rows: int, n: int, *, world_size: int) -> int:
    """Pair tiles of one call of the CTA-pair main: per peer and N tile, the
    chunks' ``ceil(M tiles / 2)`` pairs summed over the readiness chunks (the
    odd-rank CTA of the last pair of an odd chunk idles)."""

    main_tiles(rows, n, world_size=world_size, block_n=256)
    padded, chunk_rows, num_chunks = chunk_plan(rows)
    first_tiles_m = chunk_rows // BLOCK_M
    last_tiles_m = (padded - (num_chunks - 1) * chunk_rows) // BLOCK_M
    pairs_per_peer = (num_chunks - 1) * ((first_tiles_m + 1) // 2) + (
        last_tiles_m + 1
    ) // 2
    return int(world_size) * pairs_per_peer * (int(n) // 256)


def main_grid(
    rows: int,
    n: int,
    *,
    world_size: int,
    sm_count: int,
    block_n: int = 256,
    form: str = "cg1",
) -> tuple[int, int, int]:
    """Launch grid of the persistent main kernel: one CTA per SM, bounded by the
    total output tile count (``world_size * ceil128(rows) / 128 * N / block_n``),
    each CTA striding over the arrival-ordered tile list. Every pair form (the
    CTA-pair twin and its spill-wave N-split twins) launches an even grid of
    ``2 * min(pair_tiles, SMs / 2)`` CTAs, each cluster striding over the
    pair-tile list. The tile-order twins and the last-pass N-split twin take the
    one-CTA grid at the public width."""

    if int(world_size) <= 0 or int(sm_count) <= 0:
        message = "the persistent main grid needs a positive world_size and sm_count"
        raise ValueError(message)
    if _check_form(form) in ORDER_FORMS and _check_block_n(block_n) != 256:
        raise ValueError(
            f"the tile-order mains exist at block_n=256 only, got {block_n}"
        )
    if form in ONE_CTA_SPLIT_FORMS and _check_block_n(block_n) != 256:
        raise ValueError(
            f"the one-CTA last-pass N-split mains exist at block_n=256 only, "
            f"got {block_n}"
        )
    if _check_form(form) in PAIR_FORMS:
        if _check_block_n(block_n) != 256:
            raise ValueError(
                f"the CTA-pair main exists at block_n=256 only, got {block_n}"
            )
        pair_tiles = main_pair_tiles(rows, n, world_size=world_size)
        return (2 * max(1, min(pair_tiles, int(sm_count) // 2)), 1, 1)
    total_tiles = main_tiles(rows, n, world_size=world_size, block_n=block_n)
    return (max(1, min(total_tiles, int(sm_count))), 1, 1)


def sm_push_max_cols(padded_rows: int) -> int:
    """Widest ``N`` the SM push route serves at ``padded_rows`` padded local rows."""

    return SM_PUSH_COLS_INTERCEPT - SM_PUSH_COLS_PER_ROW * int(padded_rows)


def sm_push_bands(arch: str) -> tuple[tuple[int, int, int], ...]:
    """``SM_PUSH_BANDS_BY_ARCH[arch]`` as ``(world_size, padded_rows, max_cols)`` triples; an arch
    outside the table is an error, never an empty default."""

    if arch not in SM_PUSH_BANDS_BY_ARCH:
        raise ValueError(
            f"SM_PUSH_BANDS_BY_ARCH has no SM push bands for arch {arch!r} (known: {sorted(SM_PUSH_BANDS_BY_ARCH)})"
        )
    return tuple(
        (int(band[0]), int(band[1]), int(band[2]))
        for band in SM_PUSH_BANDS_BY_ARCH[arch]
    )


def uses_sm_push(*, rows: int, world_size: int, cols: int, arch: str) -> bool:
    """Padded local rows up to ``SM_PUSH_MAX_ROWS`` with ``N`` up to ``sm_push_max_cols`` push
    through the SM kernel instead of the copy engines at world sizes 4 and 8 (a single peer
    saturates neither route; a wider GEMM overlaps the copy-engine pushes but would wait behind
    the SM push; the copy engine wins in both cases), and so does a call in a band of
    ``SM_PUSH_BANDS_BY_ARCH[arch]`` (its world size, exactly its padded rows, ``N <= max_cols``).
    The rule mirrors the kernel module's ``uses_sm_push``; ``arch`` is validated on every call."""

    bands = sm_push_bands(arch)
    world_size = int(world_size)
    padded = padded_rows(rows)
    cols = int(cols)
    if (
        world_size >= 4
        and padded <= SM_PUSH_MAX_ROWS
        and cols <= sm_push_max_cols(padded)
    ):
        return True
    return any(
        world_size == band_world_size and padded == band_rows and cols <= band_max_cols
        for band_world_size, band_rows, band_max_cols in bands
    )


def sm_push_reachable(*, world_size: int, cols: int, arch: str) -> bool:
    """Whether some row count of a ``(world_size, N)`` group takes the SM push route on ``arch``:
    the base rule at its widest ceiling (128 padded rows) or a band of the architecture. The
    backend resolves a group's SM push tables at prepare time by this predicate, so a captured
    call allocates nothing."""

    if uses_sm_push(rows=BLOCK_M, world_size=world_size, cols=cols, arch=arch):
        return True
    return any(
        int(world_size) == band_world_size and int(cols) <= band_max_cols
        for band_world_size, _band_rows, band_max_cols in sm_push_bands(arch)
    )


def push_payload_bytes(padded_rows: int, dtype_bytes: int) -> int:
    """Per-peer payload of one call: the padded local shard, ``padded_rows x K x dtype_bytes``."""

    return int(padded_rows) * K * int(dtype_bytes)


def push_launch_rows(arch: str) -> tuple[tuple[int, int, int, int], ...]:
    """``PUSH_LAUNCH_ROWS_BY_ARCH[arch]`` as ``(world_size, padded_rows, ctas_per_peer,
    unroll)`` 4-tuples; an arch outside the table is an error, never an empty default."""

    if arch not in PUSH_LAUNCH_ROWS_BY_ARCH:
        raise ValueError(
            f"PUSH_LAUNCH_ROWS_BY_ARCH has no SM push launch rows for arch {arch!r} "
            f"(known: {sorted(PUSH_LAUNCH_ROWS_BY_ARCH)})"
        )
    return tuple(
        (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in PUSH_LAUNCH_ROWS_BY_ARCH[arch]
    )


def push_launch(
    padded_rows: int, dtype_bytes: int, *, world_size: int, arch: str
) -> tuple[str, int, int]:
    """``(route key, CTAs per peer, vectors in flight)`` of the SM push launch a call takes.

    The mirror of the rule compiled into ``run_push`` (which makes the choice
    itself): the architecture's launch row of exactly ``(world_size,
    padded_rows)`` when ``PUSH_LAUNCH_ROWS_BY_ARCH`` lists one (its route key is
    the push program of its vectors in flight), else the ``peer_push`` program
    with the 32-CTA grid below ``PUSH_PAYLOAD_THRESHOLD_BYTES`` per peer and the
    ``peer_push_wide`` program with the wide grid at or above it. An unknown
    arch raises, never defaults.
    """

    programs = {PUSH_UNROLL: "peer_push", PUSH_WIDE_UNROLL: "peer_push_wide"}
    rows = push_launch_rows(arch)
    for row_world_size, row_padded_rows, ctas_per_peer, unroll in rows:
        if (row_world_size, row_padded_rows) == (int(world_size), int(padded_rows)):
            return programs[unroll], ctas_per_peer, unroll
    if push_payload_bytes(padded_rows, dtype_bytes) >= PUSH_PAYLOAD_THRESHOLD_BYTES:
        return "peer_push_wide", PUSH_WIDE_CTAS_PER_PEER, PUSH_WIDE_UNROLL
    return "peer_push", PUSH_CTAS_PER_PEER, PUSH_UNROLL


def push_kslice_rows(arch: str) -> tuple[tuple[int, int, int], ...]:
    """``PUSH_KSLICES_BY_ARCH[arch]`` as ``(world_size, padded_rows, k_slices)`` 3-tuples; an arch
    outside the table is an error, never an empty default."""

    if arch not in PUSH_KSLICES_BY_ARCH:
        raise ValueError(
            f"PUSH_KSLICES_BY_ARCH has no SM push readiness slice rows for arch {arch!r} "
            f"(known: {sorted(PUSH_KSLICES_BY_ARCH)})"
        )
    return tuple(
        (int(row[0]), int(row[1]), int(row[2])) for row in PUSH_KSLICES_BY_ARCH[arch]
    )


def push_kslices(padded_rows: int, *, world_size: int, arch: str) -> int:
    """Readiness words per (peer, chunk) the SM push of one call publishes.

    The mirror of the rule compiled into ``run_push`` (which selects the per-K-slice
    twins itself, by the call's padded rows and then its ``block_n`` / ``main_form``;
    every width and form a call at a listed row count can resolve to has a twin):
    the architecture's row of exactly ``(world_size, padded_rows)`` when
    ``PUSH_KSLICES_BY_ARCH`` lists one, else 1 (one word per (peer, chunk)). The
    backend clears ``world_size * num_chunks * k_slices`` readiness words after a
    captured call. An unknown arch raises, never defaults.
    """

    for row_world_size, row_padded_rows, k_slices in push_kslice_rows(arch):
        if (row_world_size, row_padded_rows) == (int(world_size), int(padded_rows)):
            return k_slices
    return 1


def max_push_kslices(world_size: int, arch: str) -> int:
    """Largest readiness slice count the architecture's table lists for ``world_size``
    (1 without one). An unknown arch raises, never defaults."""

    return max(
        [1]
        + [
            int(k_slices)
            for row_world_size, _padded_rows, k_slices in push_kslice_rows(arch)
            if row_world_size == int(world_size)
        ]
    )


def push_signal_slots(pitch: int, *, world_size: int, arch: str) -> int:
    """Readiness words per rank row (and SM push ticket counters per peer) a workspace
    of ``pitch`` rows per peer addresses on ``arch``.

    The largest of the pitch's chunk count (every call up to the pitch publishes one
    word per (peer, chunk)) and, per readiness slice row of ``world_size`` whose padded
    rows the pitch can serve, that row's chunk count times its slice count (its calls
    publish one word per (peer, chunk, slice)). The pitch's chunk count wherever no
    such row lies at or under the pitch, so a device of an architecture without a row
    keeps the one-word sizing and a large workspace never grows past it; the backend
    sizes the readiness pad view and the ticket counters by it, so one workspace of
    the device serves every delivered route of the group there. A pitch that is not
    positive or an unknown arch raises, never defaults."""

    if int(pitch) <= 0:
        raise ValueError(f"push_signal_slots: pitch must be positive, got {pitch!r}")
    slots = chunk_plan(pitch)[2]
    for row_world_size, padded_rows, k_slices in push_kslice_rows(arch):
        if row_world_size == int(world_size) and padded_rows <= int(pitch):
            slots = max(slots, chunk_plan(padded_rows)[2] * int(k_slices))
    return slots


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
    world_size: int,
    dtype_name: str,
    b_layout: str,
    block_n: int,
    form: str = "cg1",
    k_slices: int = 1,
) -> str:
    """``ROUTES`` key of one main program (``_n128`` suffix for the 128-wide twin,
    ``_cg2`` for the CTA-pair twin of the N256 main, ``_cg2s1`` / ``_cg2s2`` for
    its spill-wave N-split twins, ``_po`` / ``_pi`` for the one-CTA main's
    tile-order twins, ``_cg1s2`` for its last-pass N-split twin, ``_ks<S>`` last
    for a per-K-slice readiness twin)."""

    width = "" if _check_block_n(block_n) == 256 else f"_n{block_n}"
    if _check_form(form) in PAIR_FORMS and block_n != 256:
        raise ValueError(f"the CTA-pair main exists at block_n=256 only, got {block_n}")
    if form in ORDER_FORMS and block_n != 256:
        raise ValueError(
            f"the tile-order mains exist at block_n=256 only, got {block_n}"
        )
    if form in ONE_CTA_SPLIT_FORMS and block_n != 256:
        raise ValueError(
            f"the one-CTA last-pass N-split mains exist at block_n=256 only, "
            f"got {block_n}"
        )
    pair = "" if form == "cg1" else f"_{form}"
    slices = "" if int(k_slices) == 1 else f"_ks{int(k_slices)}"
    return f"main_{dtype_name}_ws{int(world_size)}_{b_layout}{width}{pair}{slices}"


def main_program(
    world_size: int,
    dtype_name: str,
    b_layout: str,
    block_n: int = 256,
    form: str = "cg1",
    k_slices: int = 1,
) -> str:
    """The main program of one route at tile width ``block_n`` (256, or the delivered
    ``_n128`` twin), form (``cg1``, the delivered ``_cg2`` CTA-pair twin, a
    delivered spill-wave N-split twin, a delivered tile-order twin or the
    delivered last-pass N-split twin) and readiness slice count (1, or a delivered
    per-K-slice ``_ks<S>`` twin of that program); a class without that program
    raises ``ValueError``."""

    key = _main_route_key(world_size, dtype_name, b_layout, block_n, form, k_slices)
    if key not in ROUTES:
        raise ValueError(
            f"no all-gather matmul main program is delivered for {key!r} "
            f"(world_size={int(world_size)}, dtype={dtype_name}, "
            f"b_layout={b_layout}, block_n={block_n}, form={form}, k_slices={k_slices})"
        )
    return ROUTES[key]


def _is_kslice_stage(stage: str) -> bool:
    """Whether a sequence stage name is a per-K-slice readiness twin (``<base>_ks<S>``)."""

    _base, separator, slices = stage.rpartition("_ks")
    return bool(separator) and slices.isdigit()


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
    barrier phases, both SM push programs when the row's ``programs`` carry them
    (the world sizes ``uses_sm_push`` serves, 4 and 8; a world-size-2 row carries
    neither), the N256 main, then the ``_n128`` main when the row's ``programs``
    carry ``main_n128``, then the ``_cg2`` main when
    they carry ``main_cg2``, then the spill-wave N-split mains when they carry
    ``main_cg2s1`` / ``main_cg2s2``, then the tile-order mains when they carry
    ``main_po`` / ``main_pi`` (``ORDER_FORMS`` order), then the last-pass N-split
    main when they carry ``main_cg1s2``, then every per-K-slice
    readiness twin they carry -- the ``_ks<S>`` push twin of each slice point,
    then the ``_ks<S>`` main twin of every width and form a call at a point can
    resolve to -- in the row's order)."""

    row = SEQUENCES[sequence]
    programs = [ROUTES["barrier_p0"], ROUTES["barrier_p1"]]
    # The SM push programs travel with the rows of the world sizes the SM push
    # route serves (``uses_sm_push``: 4 and 8); a world-size-2 row lists neither.
    for stage in ("peer_push", "peer_push_wide"):
        if stage in row["programs"]:
            programs.append(ROUTES[stage])
    programs.append(main_program(row["world_size"], row["dtype"], row["b_layout"]))
    if "main_n128" in row["programs"]:
        programs.append(row["programs"]["main_n128"])
    if "main_cg2" in row["programs"]:
        programs.append(row["programs"]["main_cg2"])
    for stage in ("main_cg2s1", "main_cg2s2"):
        if stage in row["programs"]:
            programs.append(row["programs"][stage])
    # The tile-order twins of the sequence's (class, form) with a routed point, when it has any: ``main_po`` /
    # ``main_pi`` in ``ORDER_FORMS`` order (none for a class without a point).
    for form in ORDER_FORMS:
        if f"main_{form}" in row["programs"]:
            programs.append(row["programs"][f"main_{form}"])
    # The one-CTA main's last-pass N-split twin of the sequence's (class, form)
    # with a routed point, when it has one (none for a class without a point).
    for form in ONE_CTA_SPLIT_FORMS:
        if f"main_{form}" in row["programs"]:
            programs.append(row["programs"][f"main_{form}"])
    # The per-K-slice readiness twins of the sequence's slice points (the push twin of
    # each point, then the main twins), when it has any: every ``_ks<S>`` stage, in order.
    for stage, program in row["programs"].items():
        if _is_kslice_stage(stage):
            programs.append(program)
    return tuple(programs)


@functools.cache
def spec(sequence: str, arch: str) -> JitSpec:
    """Build specification of one host-sequence module for one exact architecture."""

    row = SEQUENCES[sequence]
    if arch not in row["arches"]:
        raise ValueError(
            f"sequence {sequence!r} is delivered for {row['arches']}, not {arch!r}"
        )
    sources = [_source_path(relative) for relative in row["sources"]]
    extra_cuda_cflags = [*ARCH_FLAGS[arch], *COMPILE_FLAGS]
    # The source build carries every math flag in ``COMPILE_FLAGS``; the
    # default ``-use_fast_math`` would otherwise be added or dropped.
    use_fast_math = "--use_fast_math" in COMPILE_FLAGS
    # A row that builds its host sequence from another architecture's sequence
    # source names its own host namespace through ``sequence_flags``, compile
    # definitions of that one source. A per-source list replaces the shared
    # list and the default fast-math flag for its source, so it is spelled in
    # full; the device units are absent from it and keep the shared list.
    # Keyed as ``gen_jit_spec`` declares it (``Mapping[str | Path, ...]``);
    # ``Mapping`` is invariant in its key type, so a ``dict[Path, ...]`` would not do.
    by_source: dict[str | Path, list[str]] | None = None
    if "sequence_flags" in row:
        sequence_flags = [
            *(["-use_fast_math"] if use_fast_math else []),
            *extra_cuda_cflags,
            *row["sequence_flags"],
        ]
        by_source = {sources[-1]: sequence_flags}
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_sequence_{sequence}",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[_source_dir().parent],
        use_fast_math=use_fast_math,
        extra_cuda_cflags_by_source=by_source,
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
    "CG1S_CLASSES",
    "CG1S_POINTS",
    "CG2S_CLASSES",
    "CG2_CLASSES",
    "CG2_CLUSTER_DIMS",
    "CG2_N128_PRECEDENCE_ARCHES",
    "CG2_REGIONS",
    "CG2_SPILL_POINTS",
    "CHUNK_ROWS",
    "COMPILE_FLAGS",
    "PUSH_CARVEOUT",
    "PUSH_CTAS_PER_PEER",
    "PUSH_KSLICES_BY_ARCH",
    "PUSH_LAUNCH_ROWS_BY_ARCH",
    "PUSH_PAYLOAD_THRESHOLD_BYTES",
    "PUSH_UNROLL",
    "PUSH_WIDE_CTAS_PER_PEER",
    "PUSH_WIDE_UNROLL",
    "SM_PUSH_BANDS_BY_ARCH",
    "SM_PUSH_MAX_ROWS",
    "SM_PUSH_COLS_INTERCEPT",
    "SM_PUSH_COLS_PER_ROW",
    "K",
    "MAIN_BLOCK_NS",
    "MAIN_FORMS",
    "ORDER_CLASSES",
    "ORDER_FORMS",
    "PAIR_FORMS",
    "TILE_ORDER_POINTS",
    "MAIN_THREADS",
    "N128_CLASSES",
    "N128_MAX_WAVES_BY_ARCH",
    "ONE_CTA_SPLIT_FORMS",
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
    "main_form",
    "main_form_code",
    "main_grid",
    "main_pair_tiles",
    "main_program",
    "main_tiles",
    "max_push_kslices",
    "padded_rows",
    "peer_push_program",
    "peer_push_wide_program",
    "push_kslice_rows",
    "push_kslices",
    "push_launch",
    "push_launch_rows",
    "push_payload_bytes",
    "push_signal_slots",
    "sequence_name",
    "sequence_programs",
    "sm_push_bands",
    "sm_push_max_cols",
    "sm_push_reachable",
    "spec",
    "uses_sm_push",
    "weight_layout",
    "weight_tma_source",
]
