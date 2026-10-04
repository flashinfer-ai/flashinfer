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
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Registry of the generated Cake programs (written by the generated-program
# export; do not edit by hand).
#
# Every program is one device translation unit plus its host binding, shared by
# every architecture in ``ARCHES``; the loader compiles it with the exact flag
# set of the device it runs on.  ``ARG_PLANS`` holds the one argument order per
# stage role (``main``: the persistent decode kernel; ``merge``: the split-KV
# merge kernel), ``ROUTES`` maps ``"<dtype>_p<page_size>_g<item_groups>_pm<partition_mode>"``
# -- the physical main variant the host plan selects from host-known scalars --
# to its program, and ``MERGE_PROGRAM`` names the merge kernel launched
# programmatically behind the main kernel by split-capable ticket plans.
ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
STAGES = ("main", "merge")
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
FFI_ENTRY: str = "run"
ARG_PLANS: dict[str, list[list[str]]] = {
    "main": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_k"],
        ["tma_buffer", "tmap_v"],
        ["tma_buffer", "tmap_o"],
        ["tma_buffer", "tmap_po"],
        ["buffer", "O"],
        ["buffer", "LSE"],
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "page_table"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "causal_global"],
        ["buffer", "sched_counters"],
        ["buffer", "unit_flags"],
        ["buffer", "split_meta"],
        ["buffer", "merge_ctl"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "tiles_max"],
        ["parameter", "num_heads"],
        ["parameter", "max_pages"],
        ["parameter", "cp_world"],
        ["parameter", "cp_rank"],
        ["parameter", "num_items"],
        ["parameter", "unit_min"],
        ["parameter", "static_tiles"],
        ["parameter", "unit_num"],
        ["parameter", "unit_den"],
        ["parameter", "max_units"],
        ["parameter", "static_only"],
        ["buffer", "dbg"],
        ["parameter", "partial_slots"],
        ["parameter", "fd_tiles_max"],
        ["parameter", "fd_num_heads"],
        ["parameter", "fd_cp_world"],
        ["parameter", "fd_num_items"],
        ["parameter", "fd_unit_min"],
        ["parameter", "fd_static_tiles"],
        ["parameter", "fd_unit_den"],
        ["parameter", "fd_clusters"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "merge": [
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "O"],
        ["buffer", "LSE"],
        ["buffer", "unit_flags"],
        ["buffer", "split_meta"],
        ["buffer", "merge_ctl"],
        ["buffer", "cum_seq_lens_q"],
        ["parameter", "num_heads"],
        ["parameter", "tiles_max"],
        ["parameter", "max_records"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_mla_varq_dcp_decode_0a58ba24b3133512b928": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_0a58ba24b3133512b928_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_0a58ba24b3133512b928_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_2694148151c989006fa4": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_2694148151c989006fa4_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_2694148151c989006fa4_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_4ad9a542b809cf44d979": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_4ad9a542b809cf44d979_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_4ad9a542b809cf44d979_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_54bce6b66e691dc646b1": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_54bce6b66e691dc646b1_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_54bce6b66e691dc646b1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_557713ccd2895b2aa3e4": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_557713ccd2895b2aa3e4_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_557713ccd2895b2aa3e4_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_9ccf6c8ca6b368402712": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_9ccf6c8ca6b368402712_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_9ccf6c8ca6b368402712_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_a90f4c737cfcc7bbdd55": {
        "role": "merge",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_a90f4c737cfcc7bbdd55_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_a90f4c737cfcc7bbdd55_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_be8f417457dc53a8e527": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_be8f417457dc53a8e527_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_be8f417457dc53a8e527_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_varq_dcp_decode_ffdb0eb3b15e52f2b9a2": {
        "role": "main",
        "sources": [
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_ffdb0eb3b15e52f2b9a2_kernel.cu",
            "cake_mla_varq_dcp_decode/cake_mla_varq_dcp_decode_ffdb0eb3b15e52f2b9a2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
ROUTES: dict[str, str] = {
    "bf16_p32_g4_pm1": "cake_mla_varq_dcp_decode_4ad9a542b809cf44d979",
    "bf16_p64_g16_pm0": "cake_mla_varq_dcp_decode_557713ccd2895b2aa3e4",
    "bf16_p64_g4_pm0": "cake_mla_varq_dcp_decode_54bce6b66e691dc646b1",
    "bf16_p64_g4_pm1": "cake_mla_varq_dcp_decode_ffdb0eb3b15e52f2b9a2",
    "fp8_p128_g4_pm1": "cake_mla_varq_dcp_decode_9ccf6c8ca6b368402712",
    "fp8_p64_g16_pm0": "cake_mla_varq_dcp_decode_2694148151c989006fa4",
    "fp8_p64_g4_pm0": "cake_mla_varq_dcp_decode_be8f417457dc53a8e527",
    "fp8_p64_g4_pm1": "cake_mla_varq_dcp_decode_0a58ba24b3133512b928",
}
MERGE_PROGRAM: str = "cake_mla_varq_dcp_decode_a90f4c737cfcc7bbdd55"
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5782"


def route_name(
    dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> str:
    return (
        f"{dtype}_p{int(page_size)}_g{int(item_groups)}_pm{int(bool(partition_mode))}"
    )


def main_route_available(
    dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> bool:
    return route_name(dtype, page_size, item_groups, partition_mode) in ROUTES


def select_main_program(
    dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> str:
    """Return the registered main-kernel program of one physical variant."""
    name = route_name(dtype, page_size, item_groups, partition_mode)
    program = ROUTES.get(name)
    if program is None:
        raise NotImplementedError(
            "The generated Cake MLA var-Q DCP decode program "
            f"{name!r} (dtype={dtype}, page_size={page_size}, "
            f"item_groups={item_groups}, partition_mode={int(bool(partition_mode))}) "
            f"is not registered in this checkout; registered: {sorted(ROUTES)} "
            f"(see {TRACKING_ISSUE})"
        )
    return program


def select_merge_program() -> str:
    """Return the registered split-KV merge kernel program."""
    return MERGE_PROGRAM


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
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{program}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *COMPILE_FLAGS],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *{p.parent for p in sources}, *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_program(program: str, arch: str):
    return gen_program(program, arch).build_and_load()
