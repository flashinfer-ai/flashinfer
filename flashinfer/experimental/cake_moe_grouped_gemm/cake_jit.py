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
from typing import Any, Optional

from ...jit import env as jit_env
from ...jit.core import (
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated ragged BF16 grouped GEMM
# programs.  One record per (operation, output dtype, weight-gradient k tile,
# architecture), named ``cake_moe_grouped_gemm_<op>[_<bf16|f32>_k<tile>]__<arch>``.
# Each record carries its architecture, operation, output dtype, k tile,
# public helper, stage list and host-plan summary, plus one physical entry per
# stage (``main``; weight gradient also ``tail_reduce``) with the translation
# units, compile flags, FFI entry, argument plan, closure identity, launch
# geometry, caller-owned TMA descriptor workspace size and, for pointer-ABI
# stages, the descriptor preparation entry the host calls once per prepared
# launch.  Populated verbatim by the generated-program export; do not edit by
# hand.  Empty until the export is delivered.
MODULES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_dgrad__sm_100a": {
        "arch": "sm_100a",
        "op": "dgrad",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_dgrad",
        "stages": ["main"],
        "host_plan": {
            "alignment": "K % 256 == 0 and N % 64 == 0",
            "cols": "K",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_d2d15c1c3c12612d67e8",
            "kernel": "kernel_cake_moe_grouped_gemm_d2d15c1c3c12612d67e8",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_d2d15c1c3c12612d67e8_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_d2d15c1c3c12612d67e8_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "cc9c46a12804c0d20bb7f6865b54ca114adfdf66f3c325528e7cb95f6d17ef07",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "f9acfb81e329100a6c4c9ca72814cf1536d264e57881f81d2806358af0896730",
    },
    "cake_moe_grouped_gemm_dgrad__sm_103a": {
        "arch": "sm_103a",
        "op": "dgrad",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_dgrad",
        "stages": ["main"],
        "host_plan": {
            "alignment": "K % 256 == 0 and N % 64 == 0",
            "cols": "K",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_8221d2109183ee4a02ab",
            "kernel": "kernel_cake_moe_grouped_gemm_8221d2109183ee4a02ab",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8221d2109183ee4a02ab_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8221d2109183ee4a02ab_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "b405ba8aee2968f3bf24e540c503ce74164890988cad847c34444dc4d1677ac0",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "28973108fdcc749f21e6ec0fe55727b0bd17439843f7ded4e9257876b81793b8",
    },
    "cake_moe_grouped_gemm_dgrad__sm_107a": {
        "arch": "sm_107a",
        "op": "dgrad",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_dgrad",
        "stages": ["main"],
        "host_plan": {
            "alignment": "K % 256 == 0 and N % 64 == 0",
            "cols": "K",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_aa76ca9772836d1d0ed2",
            "kernel": "kernel_cake_moe_grouped_gemm_aa76ca9772836d1d0ed2",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_aa76ca9772836d1d0ed2_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_aa76ca9772836d1d0ed2_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "62f58cf45ff0934f9dc0c3df4ad8ad7a6827d83bc3f39f2953ab369a132a7be9",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "08f3a27a55b54aaa42b64590f3f160f5de7b11cc8b0ae4bb74026122a520f501",
    },
    "cake_moe_grouped_gemm_fwd__sm_100a": {
        "arch": "sm_100a",
        "op": "fwd",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_fwd",
        "stages": ["main"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 64 == 0",
            "cols": "N",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_73ac1d3c7bdae3274f86",
            "kernel": "kernel_cake_moe_grouped_gemm_73ac1d3c7bdae3274f86",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_73ac1d3c7bdae3274f86_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_73ac1d3c7bdae3274f86_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "34053b35626d23a766bd764deacacbc3b7a9d7d850a79b0f089739fd50d95a98",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "5cc59cb933c24e355241fab72afd7c6c9f44c862c3f761421f2f8bf1b9be07ad",
    },
    "cake_moe_grouped_gemm_fwd__sm_103a": {
        "arch": "sm_103a",
        "op": "fwd",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_fwd",
        "stages": ["main"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 64 == 0",
            "cols": "N",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_8ab91783e2ab41eea2c6",
            "kernel": "kernel_cake_moe_grouped_gemm_8ab91783e2ab41eea2c6",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8ab91783e2ab41eea2c6_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8ab91783e2ab41eea2c6_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0a73887ddc2345db25c53ceb2fad961e270616b946ae8bac1a94c266f07dfd07",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "cedd73a132a1aef098bf1645857742b6f0428109a7baf667d58d4d4a3ca7eaae",
    },
    "cake_moe_grouped_gemm_fwd__sm_107a": {
        "arch": "sm_107a",
        "op": "fwd",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_fwd",
        "stages": ["main"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 64 == 0",
            "cols": "N",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "main": {
            "module": "cake_moe_grouped_gemm_d95957f7809384928cce",
            "kernel": "kernel_cake_moe_grouped_gemm_d95957f7809384928cce",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_d95957f7809384928cce_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_d95957f7809384928cce_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "6724bfb88c039abdf238534876e191d4de125c02fc801f229739f0c6d4c00e6b",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "closure_sha256": "f8a238d2e9cb3abaea50215143bc17bdef55adf91ade1cfd1e63e8a0fa2b694b",
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256__sm_100a": {
        "arch": "sm_100a",
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_194459642b2814e4a851",
            "kernel": "kernel_cake_moe_grouped_gemm_194459642b2814e4a851",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_194459642b2814e4a851_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_194459642b2814e4a851_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "7fe55b860d97acb66c9121b863b02656733c262d8b2212645fd4f20cfe585d52",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_bb93ce685f3ff2a8eeea",
            "kernel": "kernel_cake_moe_grouped_gemm_bb93ce685f3ff2a8eeea",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_bb93ce685f3ff2a8eeea_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_bb93ce685f3ff2a8eeea_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f08ddacd6bc6080062c1cd98919c1f0a5a8fbeb802d3ea07519b7185541f5e9d",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "41afb29d64535c34abae20a30c589fcf0e87aa29142ffb6f22958277d42df4d8",
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256__sm_103a": {
        "arch": "sm_103a",
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_8992b1313d908b92ea8f",
            "kernel": "kernel_cake_moe_grouped_gemm_8992b1313d908b92ea8f",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8992b1313d908b92ea8f_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_8992b1313d908b92ea8f_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "c91d9d287680493156fad5f14268321b1334f9668cbdb2bee83de01eb710a535",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_b2d20946be54fd09c880",
            "kernel": "kernel_cake_moe_grouped_gemm_b2d20946be54fd09c880",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_b2d20946be54fd09c880_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_b2d20946be54fd09c880_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "75e590f3a33b864a8f2b2f4c699b3cfa823fabee6fecec392330a6f30af51632",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "b69f0068256abab79c37abdff7d375ba19b5ec055ab2027daa557a2c409c670b",
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256__sm_107a": {
        "arch": "sm_107a",
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_bca4ce736574bed1c298",
            "kernel": "kernel_cake_moe_grouped_gemm_bca4ce736574bed1c298",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_bca4ce736574bed1c298_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_bca4ce736574bed1c298_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ae7256efa0f59448d53b5c1dd5af9acaf36da8c73755c050cd976f50f71df384",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_92738dc56302af160174",
            "kernel": "kernel_cake_moe_grouped_gemm_92738dc56302af160174",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_92738dc56302af160174_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_92738dc56302af160174_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "9cbea3e77b469f11ff8f431b1a081231c340b685fa376247a80184b02c20898d",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "e7fcd61c29d840b20980ada08e7001bc248547d1a23e49c623e28acd38818a66",
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k512__sm_100a": {
        "arch": "sm_100a",
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 512 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_19b3bd1aeb5910fdd9a2",
            "kernel": "kernel_cake_moe_grouped_gemm_19b3bd1aeb5910fdd9a2",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_19b3bd1aeb5910fdd9a2_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_19b3bd1aeb5910fdd9a2_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d26d9fd9d5b70bf2dba7d290aff924e644fddfba787226730633ed5c5bca86cb",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_e227d5245ddecad4cf5a",
            "kernel": "kernel_cake_moe_grouped_gemm_e227d5245ddecad4cf5a",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_e227d5245ddecad4cf5a_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_e227d5245ddecad4cf5a_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ec172f288365add96b7655d83f6e3dd68186c59261e7ac9ab012d05dcbe31e02",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "85c8968c9b06e843f8c9fe4e10a65f17b6d14a8d248a713c99dc7c179c0243ef",
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k512__sm_103a": {
        "arch": "sm_103a",
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 512 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_961f3945be0bad09998f",
            "kernel": "kernel_cake_moe_grouped_gemm_961f3945be0bad09998f",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_961f3945be0bad09998f_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_961f3945be0bad09998f_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "59dbebaa246fc4f585499448fafd013cf50b5f7a4876e6518299863cff01abd1",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_75bcedaaab0053cfc686",
            "kernel": "kernel_cake_moe_grouped_gemm_75bcedaaab0053cfc686",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_75bcedaaab0053cfc686_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_75bcedaaab0053cfc686_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "05f201d3e4f9049fa766e109034adf676e21aa5519711a76e2013f7e512fa527",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "1bcb523b92c1cd93b92f3db1e678675214185233b522a8bcdac53a0635c96367",
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256__sm_100a": {
        "arch": "sm_100a",
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_54b3b2f1bbcccf35ee11",
            "kernel": "kernel_cake_moe_grouped_gemm_54b3b2f1bbcccf35ee11",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_54b3b2f1bbcccf35ee11_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_54b3b2f1bbcccf35ee11_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "495084edfc42151c55c9b6ceffee46d9378a254e68976c2f96edbc31cdcb3826",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_29b5f4b0984f048ade1e",
            "kernel": "kernel_cake_moe_grouped_gemm_29b5f4b0984f048ade1e",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_29b5f4b0984f048ade1e_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_29b5f4b0984f048ade1e_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "72c3636029277332b1ddbdee2086602b3ce12286d1d810e7022bdb377e7fb0c5",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "3d7e45190876a89effd42434cbafaf338a5d135374feed849a0a0e0b5238ba20",
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256__sm_103a": {
        "arch": "sm_103a",
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_9baf8e10da72c2485a41",
            "kernel": "kernel_cake_moe_grouped_gemm_9baf8e10da72c2485a41",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_9baf8e10da72c2485a41_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_9baf8e10da72c2485a41_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "8233b53fd66e2f44f53529a845c1b66dfb29e5b3522bbf54b119caa46f3e56a4",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_4336edbb29cf48e7ea29",
            "kernel": "kernel_cake_moe_grouped_gemm_4336edbb29cf48e7ea29",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_4336edbb29cf48e7ea29_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_4336edbb29cf48e7ea29_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "1eceea8eb0ab4dc0749151c537254ef699136655d1745bc802b674f19d52ff98",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "f4b1388585b9d3288188c6c1040611d325124ffeadfc744b20f1a45691f58ded",
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256__sm_107a": {
        "arch": "sm_107a",
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 256 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_c4c7b6bb9ac20c6ee992",
            "kernel": "kernel_cake_moe_grouped_gemm_c4c7b6bb9ac20c6ee992",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_c4c7b6bb9ac20c6ee992_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_c4c7b6bb9ac20c6ee992_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "30af95a97a1bc00ff00e5e5cc6299209f8d4a71fd33bb8fb13ded36d550b6a99",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_947cda358514fc146aca",
            "kernel": "kernel_cake_moe_grouped_gemm_947cda358514fc146aca",
            "sources": [
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_947cda358514fc146aca_kernel.cu",
                "cake_moe_grouped_gemm/sm_107a/cake_moe_grouped_gemm_947cda358514fc146aca_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "929d9ae7303019d26c3dcbd87e3e507eed2f82c938f75668bcdb6f70087c0bce",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "e4c2c1579aab4e06e97a8a3f9f0e64efb270468d10c0d38335578f440726fab8",
    },
    "cake_moe_grouped_gemm_wgrad_f32_k512__sm_100a": {
        "arch": "sm_100a",
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 512 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_6440b5b59bc23c87e4c9",
            "kernel": "kernel_cake_moe_grouped_gemm_6440b5b59bc23c87e4c9",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_6440b5b59bc23c87e4c9_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_6440b5b59bc23c87e4c9_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "4eeabc95027845a9db89c0c3124bed40f2d8cdd77a518f1be4b5c22d8029e0e0",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_13cdfd8f500e38920719",
            "kernel": "kernel_cake_moe_grouped_gemm_13cdfd8f500e38920719",
            "sources": [
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_13cdfd8f500e38920719_kernel.cu",
                "cake_moe_grouped_gemm/sm_100a/cake_moe_grouped_gemm_13cdfd8f500e38920719_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "79be1e21dc5b1c194fc05fd97284fc3ac153a8fd758c7dd4389420b351302978",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "d1cd81014748952d3a4aedc022e33a712a71dfa08e4f7243b8eb6357262bd7b2",
    },
    "cake_moe_grouped_gemm_wgrad_f32_k512__sm_103a": {
        "arch": "sm_103a",
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
        "stages": ["main", "tail_reduce"],
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 512 == 0",
            "clusters": "grid_x // 2",
            "grid": "persistent_grid(sm_count, num_groups * (N // 256) * (K // tile_k))",
            "launches": "1 + (tail_splits > 1)",
            "num_tail": "(num_groups * (N // 256) * (K // tile_k)) % clusters",
            "partials_bytes": "4 * 256 * tile_k * max(num_tail * tail_splits if tail_splits > 1 else 0, 1)",
            "raster_rows": "wgrad_raster_rows(N // 256, K // tile_k, clusters, tile_k)",
            "tail_reduce_grid": "wgrad_reduce_grid(num_tail, tile_k) when tail_splits > 1, else no tail_reduce launch",
            "tail_splits": "wgrad_tail_plan(num_tail, clusters, sum_m, num_groups, tile_k) if num_tail else 1",
            "tensormap_workspace_bytes": "grid_x * tmap_slots_per_cta * tmap_slot_bytes (uint8, 128-byte aligned)",
        },
        "main": {
            "module": "cake_moe_grouped_gemm_bc942a529e11a39106f3",
            "kernel": "kernel_cake_moe_grouped_gemm_bc942a529e11a39106f3",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_bc942a529e11a39106f3_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_bc942a529e11a39106f3_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["buffer", "tensormap_workspace"],
                ["buffer", "partials"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["parameter", "num_groups"],
                ["parameter", "sum_m"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["workspace", "tma_descriptor_workspace"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "fb71a6479142362e80ea72d522dc19ae298fdc0ea7501dd56e25be66e32386c4",
            "tma_workspace_bytes": 256,
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
            "tma_prepare_entry": "run_prepare_tma",
        },
        "tail_reduce": {
            "module": "cake_moe_grouped_gemm_82d5fc4d07580d80b638",
            "kernel": "kernel_cake_moe_grouped_gemm_82d5fc4d07580d80b638",
            "sources": [
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_82d5fc4d07580d80b638_kernel.cu",
                "cake_moe_grouped_gemm/sm_103a/cake_moe_grouped_gemm_82d5fc4d07580d80b638_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "partials"],
                ["buffer", "C"],
                ["buffer", "offs"],
                ["parameter", "num_groups"],
                ["parameter", "N"],
                ["parameter", "K"],
                ["parameter", "ldc"],
                ["parameter", "stride_e"],
                ["parameter", "num_clusters"],
                ["parameter", "tail_splits"],
                ["parameter", "raster_rows"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f2886d84195a3f25fe573e1aaaab97634edd0a838f0ba03c0d7446da9fad4055",
            "tma_workspace_bytes": 0,
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "8b8afb92b27056e171ae468eaa8ae808ab4bc51b5082a7ae7ef06da4a4fc348b",
    },
}

# Module constants of the production host planner (tile geometry, cluster
# shape, descriptor slot layout, k-tile selection thresholds and the split-K
# tail cost model), filled by the same export.  ``cake_backend`` reproduces the
# production launch plans from these values alone.
HOST_PLAN_CONSTANTS: dict[str, Any] = {
    "block_m": 128,
    "block_n": 256,
    "block_k": 64,
    "cta_group": 2,
    "cluster_m": 256,
    "epi_chunk": 16,
    "tmap_slot_bytes": 128,
    "tmap_slots_per_cta": 2,
    "k512_archs": ["sm_100a", "sm_103a"],
    "k512_min_rows_per_group": 20480,
    "wgrad_tail_split": 1,
    "tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    "int32_elems": 2147483647,
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")
# Exact-architecture payloads (tcgen05 / TMEM): one module per exact target,
# built with FlashInfer's exact flag sets, never a family target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def record_name(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Registry key of the program serving ``op`` on ``arch``.

    Weight-gradient programs are further keyed by output dtype (``"bfloat16"``
    / ``"float32"``) and k tile (256 / 512).
    """
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient programs are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return f"{base}__{arch}"


def registered_arches() -> tuple[str, ...]:
    """Architectures with at least one registered program, in ``ARCHES`` order."""
    present = {record["arch"] for record in MODULES.values()}
    return tuple(arch for arch in ARCHES if arch in present)


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> bool:
    """True when this checkout registers the program for ``op`` on ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.
    """
    if op != "wgrad":
        return record_name(op, arch) in MODULES
    dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
    tiles = (tile_k,) if tile_k else (256, 512)
    return any(
        record_name(op, arch, out_dtype=d, tile_k=t) in MODULES
        for d in dtypes
        for t in tiles
    )


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered record name for ``op`` on ``arch`` or raise."""
    name = record_name(op, arch, out_dtype=out_dtype, tile_k=tile_k)
    record = MODULES.get(name)
    if record is None:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM program {name!r} is not registered "
            "in this checkout yet (the module registry of "
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit is filled by the "
            "generated-program export)"
        )
    if record["arch"] != arch or record["op"] != op:
        raise RuntimeError(
            f"registered module {name!r} is a {record['op']} program for "
            f"{record['arch']}, bound to {op} on {arch}"
        )
    return name


def build_target_arches() -> frozenset[str]:
    """Exact architectures FlashInfer builds for, restricted to ``ARCHES``.

    Follows ``FLASHINFER_CUDA_ARCH_LIST`` when set and the visible devices
    otherwise (``flashinfer.compilation_context.CompilationContext``); the
    loader never probes ``nvcc`` or the device itself.
    """
    context = current_compilation_context
    if not context.TARGET_CUDA_ARCHS:
        context = refresh_current_compilation_context()
    targets = {f"sm_{major}{minor}" for major, minor in context.TARGET_CUDA_ARCHS}
    return frozenset(targets) & frozenset(ARCHES)


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


def _physical(name: str, stage: str) -> dict[str, Any]:
    record = MODULES[name]
    if stage not in record["stages"]:
        raise KeyError(
            f"generated program {name!r} has stages {list(record['stages'])}, not {stage!r}"
        )
    return record[stage]


@functools.cache
def gen_module(name: str, stage: str):
    """JIT spec of one physical stage of a registered program.

    The cache name carries the sealed closure identity of the stage so a
    changed generated closure never reuses a stale extension; the JIT
    workspace directory already encodes the target set.
    """
    record = MODULES[name]
    physical = _physical(name, stage)
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_module(name: str, stage: str):
    """Build (once) and load one physical stage of a registered program."""
    arch = MODULES[name]["arch"]
    targets = build_target_arches()
    if arch not in targets:
        raise RuntimeError(
            f"generated program {name!r} targets {arch}, which is not a FlashInfer "
            f"build target (targets: {sorted(targets) or 'none'}); set "
            "FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return gen_module(name, stage).build_and_load()
