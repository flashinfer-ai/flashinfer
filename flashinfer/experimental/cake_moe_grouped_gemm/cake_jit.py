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

# Registry of the generated ragged grouped GEMM programs (filled verbatim by the
# Cake generated-program export; do not edit the literals by hand).
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by every architecture in ``arches`` (genuine
# per-architecture lowering differences live inside the source under exact
# ``__CUDA_ARCH__`` regions) -- with its compile flags, FFI entry, argument
# plan, launch shape and per-architecture closure identity.  A program is
# JIT-compiled once per (architecture, compile-line specialization).
#
# ``ROUTES``: one record per public route -- ``fwd``, ``dgrad`` and
# ``wgrad_{bf16,f32}_k{256,512}`` -- naming its architectures, host plan and
# ordered stages.  Each stage binds a program and the compile-line
# specializations the loader defines for it (the weight-gradient tail-reduce
# program takes the k width of the partials it sums, ``WGRAD_TILE``, from the
# route's tile instead of carrying one source per tile).
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_05498a17b6c266797a62": {
        "kernel": "kernel_cake_moe_grouped_gemm_05498a17b6c266797a62",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_05498a17b6c266797a62_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_05498a17b6c266797a62_binding.cu",
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
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "abbfa115c214e49b89626e627d72e3afa13014359b48220d92f5fd857a043344",
            "sm_103a": "52d1a8e2658ba012514bc589f462c077ee228b0fbf0a262db4b9d7b6a550a623",
            "sm_107a": "e4c396a0d972a4f789792e789a99cf8471b1252984c162a480c8230aa5e1ed46",
        },
    },
    "cake_moe_grouped_gemm_1edcad995dcc790600cb": {
        "kernel": "kernel_cake_moe_grouped_gemm_1edcad995dcc790600cb",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_1edcad995dcc790600cb_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_1edcad995dcc790600cb_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "fa648c04b22543a51dc0304904b2e26c6a0353c02581e9762493d14dd8343ac0",
            "sm_103a": "ebc610dea0a23724755fd2a7c06bf1b8016c855c1343ad984334f70c31ff39e0",
            "sm_107a": "b510f7c4acaa0c9ab610d4c71ef639637f667bb15c743f64147cc87140b09979",
        },
    },
    "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb": {
        "kernel": "kernel_cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "abf84e571383ff0278d803512b2d92ae1bad287ebb351a0fdf3db5237db27409",
            "sm_103a": "4d7ef5c662fdc78fbda315ab684cd737dfcfd94ce3b2668cd141efcd7b0b865a",
            "sm_107a": "ef3292e9e48676eb3f26bbbd3d20c385f462aec2bf1932a24f9caf11112f7d84",
        },
    },
    "cake_moe_grouped_gemm_611af0ca12365b0e7c3b": {
        "kernel": "kernel_cake_moe_grouped_gemm_611af0ca12365b0e7c3b",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_611af0ca12365b0e7c3b_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_611af0ca12365b0e7c3b_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": {
            "sm_100a": "32976c1b3ec7553b54342b7af147cd38b8afb26451c8de2dbb50a3b4e47e43aa",
            "sm_103a": "1ab5c07494fd3b057f3f83bd838187ff30f0c06bd36e352e4f3344ea425773d1",
        },
    },
    "cake_moe_grouped_gemm_7ac85aef90878e42ea02": {
        "kernel": "kernel_cake_moe_grouped_gemm_7ac85aef90878e42ea02",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_7ac85aef90878e42ea02_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_7ac85aef90878e42ea02_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "e01f627a13dbd064cf6093a4f4170dad8dfd1960b3360fe01c6109328e0ce670",
            "sm_103a": "8c67bddc052f52d4786d02f3ca081c52c65228c06c1ee2c3bf1ef25e24d1724a",
            "sm_107a": "983458d66e864b545e87e4a869d20b65561a2028907c12e5a19b13734f2d5add",
        },
    },
    "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c": {
        "kernel": "kernel_cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_bc774d802e4f0b3ac27c_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_bc774d802e4f0b3ac27c_binding.cu",
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
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "2b4f8b7d1972d2b3d9bd22216efe03d676afc7181d9bda2e03d97259db3956c9",
            "sm_103a": "d8fab5232bb4e5a4552cf1e9c1cf7d65ca7fe19a77152cece9f24eb6d4ebf6a8",
            "sm_107a": "b417bf2fe46c7dbe5da3bf6966f7c0d04ca4685cbcfa7a48c695e236f479b7de",
        },
    },
    "cake_moe_grouped_gemm_d77aff0f30a2661aecc8": {
        "kernel": "kernel_cake_moe_grouped_gemm_d77aff0f30a2661aecc8",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_d77aff0f30a2661aecc8_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_d77aff0f30a2661aecc8_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": {
            "sm_100a": "453cc5721078de16900f4397c98ec54764bbd85f4e06890695235963c64769de",
            "sm_103a": "5a44ba4f92678f880e07068a671444b36bd3fa27a55bb1e7f3d1510aa8531150",
        },
    },
    "cake_moe_grouped_gemm_ed9a74f36ba13a52183d": {
        "kernel": "kernel_cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
        "sources": [
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_ed9a74f36ba13a52183d_kernel.cu",
            "cake_moe_grouped_gemm/cake_moe_grouped_gemm_ed9a74f36ba13a52183d_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "closure_sha256": {
            "sm_100a": "97e5ebea1fdc233d118880f6d3b06859e0fb0f603a586c82a8766c3bfaf29b8d",
            "sm_103a": "a9916dbefd98fa75c03cc5d696841ee122bba9b80b6bb08cd45b413d17fbde82",
            "sm_107a": "67303368dde73148ab1746007f00b77fbf68e9df1edb95ad5a6387f227394e64",
        },
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_dgrad": {
        "op": "dgrad",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_dgrad",
        "host_plan": {
            "alignment": "K % 256 == 0 and N % 64 == 0",
            "cols": "K",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_fwd": {
        "op": "fwd",
        "out_dtype": "bfloat16",
        "tile_k": None,
        "public_helper": "grouped_gemm_fwd",
        "host_plan": {
            "alignment": "N % 256 == 0 and K % 64 == 0",
            "cols": "N",
            "grid": "persistent_grid(sm_count, max_cluster_tiles_upper_bound(sum_m, num_groups, cols))",
            "launches": 1,
        },
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256": {
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
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
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k512": {
        "op": "wgrad",
        "out_dtype": "bfloat16",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
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
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_611af0ca12365b0e7c3b",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256": {
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 256,
        "public_helper": "grouped_gemm_wgrad",
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
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_7ac85aef90878e42ea02",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_05498a17b6c266797a62",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k512": {
        "op": "wgrad",
        "out_dtype": "float32",
        "tile_k": 512,
        "public_helper": "grouped_gemm_wgrad",
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
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_d77aff0f30a2661aecc8",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_05498a17b6c266797a62",
                "specializations": {"WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
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
# Exact-architecture payloads (tcgen05 / TMEM): one build per exact target with
# FlashInfer's exact flag sets, never a family target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def record_name(
    op: str, *, out_dtype: Optional[str] = None, tile_k: Optional[int] = None
) -> str:
    """Route key of ``op``; weight-gradient routes are further keyed by output dtype and k tile."""
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient routes are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return base


def registered_arches() -> tuple[str, ...]:
    """Architectures with at least one registered route, in ``ARCHES`` order."""
    present = {arch for route in ROUTES.values() for arch in route["arches"]}
    return tuple(arch for arch in ARCHES if arch in present)


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> bool:
    """True when this checkout registers a route of ``op`` for ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.
    """
    if arch not in ARCHES:
        return False
    if op != "wgrad":
        route = ROUTES.get(record_name(op))
        return route is not None and arch in route["arches"]
    dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
    tiles = (tile_k,) if tile_k else (256, 512)
    for dtype in dtypes:
        for tile in tiles:
            route = ROUTES.get(record_name(op, out_dtype=dtype, tile_k=tile))
            if route is not None and arch in route["arches"]:
                return True
    return False


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered route name serving ``op`` on ``arch`` or raise."""
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    name = record_name(op, out_dtype=out_dtype, tile_k=tile_k)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM route {name!r} is not registered for "
            f"{arch} in this checkout (the registry of "
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit is filled by the "
            "generated-program export)"
        )
    if route["op"] != op:
        raise RuntimeError(
            f"registered route {name!r} is a {route['op']} route, bound to {op}"
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


def stage_binding(name: str, stage: str) -> dict[str, Any]:
    """The ``{"name", "program", "specializations"}`` record of ``stage`` of route ``name``."""
    route = ROUTES[name]
    for item in route["stages"]:
        if item["name"] == stage:
            return item
    raise KeyError(
        f"route {name!r} has stages {[s['name'] for s in route['stages']]}, not {stage!r}"
    )


def _specialization_items(
    specializations: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(specializations.items()))


@functools.cache
def _gen_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    tag = "".join(f"_{key.lower()}{value}" for key, value in specializations)
    return gen_jit_spec(
        # The spec name carries the exact target, the compile-line specialization
        # values and the sealed closure identity, so a changed closure or value
        # never reuses a stale extension.
        name=f"{program}_{arch}{tag}_" + record["closure_sha256"][arch][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            *[f"-D{key}={value}" for key, value in specializations],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


def gen_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """JIT spec of one physical program for ``arch`` under ``specializations``."""
    return _gen_module(program, arch, _specialization_items(specializations or {}))


@functools.cache
def _load_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    targets = build_target_arches()
    if arch not in targets:
        raise RuntimeError(
            f"generated program {program!r} is requested for {arch}, which is not a "
            f"FlashInfer build target (targets: {sorted(targets) or 'none'}); set "
            "FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return _gen_module(program, arch, specializations).build_and_load()


def load_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """Build (once per target and specialization) and load one physical program."""
    return _load_module(program, arch, _specialization_items(specializations or {}))
