# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell contiguous grouped FP8 GEMM programs.

``PROGRAMS`` lists every generated program once: its two translation units
under ``csrc/cake_grouped_fp8_gemm`` (one source shared by every architecture
it runs on), the exact compile flags of the source build, the tvm-ffi entry
and the id of its positional argument plan in ``ARG_PLANS``.  ``MODULES`` names one JIT module per
(program, architecture) and ``ROUTES`` maps every host route to its program.
The three tables and ``ROUTE_GEOMETRY`` are written by the generated-program
export; do not edit them by hand.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

ARG_PLANS: dict[str, list[list[str]]] = {
    "plan0": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C_tma"],
        ["buffer", "C"],
        ["buffer", "a_scale"],
        ["buffer", "b_scale"],
        ["buffer", "m_indices"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["parameter", "G"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan1": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C_tma"],
        ["buffer", "C"],
        ["tma_buffer", "a_scale"],
        ["tma_buffer", "b_scale"],
        ["buffer", "m_indices"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["parameter", "G"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_grouped_fp8_gemm_052db8d74e99d306ddd1": {
        "kernel": "kernel_cake_grouped_fp8_gemm_052db8d74e99d306ddd1",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_052db8d74e99d306ddd1_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_052db8d74e99d306ddd1_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_1b8a46c5188189851e7e": {
        "kernel": "kernel_cake_grouped_fp8_gemm_1b8a46c5188189851e7e",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_1b8a46c5188189851e7e_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_1b8a46c5188189851e7e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_3b638be257142f78ec7a": {
        "kernel": "kernel_cake_grouped_fp8_gemm_3b638be257142f78ec7a",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_3b638be257142f78ec7a_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_3b638be257142f78ec7a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_4cb684af9950c96d43f1": {
        "kernel": "kernel_cake_grouped_fp8_gemm_4cb684af9950c96d43f1",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_4cb684af9950c96d43f1_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_4cb684af9950c96d43f1_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_69974ba44e35be8085c8": {
        "kernel": "kernel_cake_grouped_fp8_gemm_69974ba44e35be8085c8",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_69974ba44e35be8085c8_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_69974ba44e35be8085c8_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7": {
        "kernel": "kernel_cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f": {
        "kernel": "kernel_cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_c9d4520e67204f6fab94": {
        "kernel": "kernel_cake_grouped_fp8_gemm_c9d4520e67204f6fab94",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_c9d4520e67204f6fab94_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_c9d4520e67204f6fab94_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_d74a826eaa282b4ae559": {
        "kernel": "kernel_cake_grouped_fp8_gemm_d74a826eaa282b4ae559",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_d74a826eaa282b4ae559_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_d74a826eaa282b4ae559_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_e9a88795c566eb333bb4": {
        "kernel": "kernel_cake_grouped_fp8_gemm_e9a88795c566eb333bb4",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_e9a88795c566eb333bb4_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_e9a88795c566eb333bb4_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_f1969b1c8be8317cb028": {
        "kernel": "kernel_cake_grouped_fp8_gemm_f1969b1c8be8317cb028",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_f1969b1c8be8317cb028_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_f1969b1c8be8317cb028_binding.cu",
        ],
        "compile_flags": ["-Xptxas=-O1"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
}

MODULES: dict[str, dict[str, str]] = {
    "cake_grouped_fp8_gemm_052db8d74e99d306ddd1_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_052db8d74e99d306ddd1",
    },
    "cake_grouped_fp8_gemm_052db8d74e99d306ddd1_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_052db8d74e99d306ddd1",
    },
    "cake_grouped_fp8_gemm_1b8a46c5188189851e7e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_1b8a46c5188189851e7e",
    },
    "cake_grouped_fp8_gemm_1b8a46c5188189851e7e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_1b8a46c5188189851e7e",
    },
    "cake_grouped_fp8_gemm_3b638be257142f78ec7a_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_3b638be257142f78ec7a",
    },
    "cake_grouped_fp8_gemm_3b638be257142f78ec7a_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_3b638be257142f78ec7a",
    },
    "cake_grouped_fp8_gemm_4cb684af9950c96d43f1_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_4cb684af9950c96d43f1",
    },
    "cake_grouped_fp8_gemm_4cb684af9950c96d43f1_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_4cb684af9950c96d43f1",
    },
    "cake_grouped_fp8_gemm_69974ba44e35be8085c8_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_69974ba44e35be8085c8",
    },
    "cake_grouped_fp8_gemm_69974ba44e35be8085c8_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_69974ba44e35be8085c8",
    },
    "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7",
    },
    "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7",
    },
    "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f",
    },
    "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f",
    },
    "cake_grouped_fp8_gemm_c9d4520e67204f6fab94_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_c9d4520e67204f6fab94",
    },
    "cake_grouped_fp8_gemm_c9d4520e67204f6fab94_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_c9d4520e67204f6fab94",
    },
    "cake_grouped_fp8_gemm_d74a826eaa282b4ae559_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_d74a826eaa282b4ae559",
    },
    "cake_grouped_fp8_gemm_d74a826eaa282b4ae559_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_d74a826eaa282b4ae559",
    },
    "cake_grouped_fp8_gemm_e9a88795c566eb333bb4_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_e9a88795c566eb333bb4",
    },
    "cake_grouped_fp8_gemm_e9a88795c566eb333bb4_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_e9a88795c566eb333bb4",
    },
    "cake_grouped_fp8_gemm_f1969b1c8be8317cb028_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_f1969b1c8be8317cb028",
    },
    "cake_grouped_fp8_gemm_f1969b1c8be8317cb028_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_f1969b1c8be8317cb028",
    },
}

ROUTES: dict[str, Any] = {
    "deepk_c2_ab6_scale1_n256_or_m4096": "cake_grouped_fp8_gemm_4cb684af9950c96d43f1",
    "deepk_c2_k_multiple_512": "cake_grouped_fp8_gemm_e9a88795c566eb333bb4",
    "deepk_c2_kg1_non_kg4_k_blocks": "cake_grouped_fp8_gemm_c9d4520e67204f6fab94",
    "deepk_c2_scalar_output": "cake_grouped_fp8_gemm_69974ba44e35be8085c8",
    "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4": "cake_grouped_fp8_gemm_1b8a46c5188189851e7e",
    "deepk_cg2_ab5_x16_pair_wait_kg4": "cake_grouped_fp8_gemm_d74a826eaa282b4ae559",
    "deepk_cg2_ab6_early4_output_alias_kg4": "cake_grouped_fp8_gemm_8d257b8a7a26cde8a91f",
    "deepk_cg2_ab7_bscale_prefetch_kg4": "cake_grouped_fp8_gemm_3b638be257142f78ec7a",
    "deepk_n256_ab4_three_panel_kg4": "cake_grouped_fp8_gemm_052db8d74e99d306ddd1",
    "k128_exact": "cake_grouped_fp8_gemm_f1969b1c8be8317cb028",
    "k384_exact_three_partial": "cake_grouped_fp8_gemm_6aaa2e7a81d7413f5ee7",
}

# Tile geometry of every host route (including routes without an exported
# program), resolved from the source dispatcher: {route: {tile_m, tile_n,
# cluster_ctas}}.
ROUTE_GEOMETRY: dict[str, dict[str, int]] = {
    "deepk_c2_ab6_scale1_n256_or_m4096": {
        "tile_m": 128,
        "tile_n": 128,
        "cluster_ctas": 1,
    },
    "deepk_c2_k_multiple_512": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
    "deepk_c2_kg1_non_kg4_k_blocks": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
    "deepk_c2_scalar_output": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
    "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "deepk_cg2_ab5_x16_four_load_wait_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "deepk_cg2_ab5_x16_pair_wait_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "deepk_cg2_ab6_early4_output_alias_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "deepk_cg2_ab7_bscale_prefetch_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "deepk_n256_ab4_three_panel_kg4": {"tile_m": 128, "tile_n": 256, "cluster_ctas": 1},
    "k128_exact": {"tile_m": 128, "tile_n": 256, "cluster_ctas": 1},
    "k384_exact_three_partial": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
}

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def _source_dirs() -> tuple[Path, Path]:
    checkout = Path(__file__).resolve().parents[3]
    source_root = checkout / "csrc"
    include_root = checkout / "include"
    if not (source_root / "tvm_ffi_utils.h").is_file():
        source_root = jit_env.FLASHINFER_CSRC_DIR
        include_root = jit_env.FLASHINFER_INCLUDE_DIR
    if not (source_root / "tvm_ffi_utils.h").is_file():
        raise FileNotFoundError("FlashInfer binding headers were not found")
    return source_root, include_root


@functools.cache
def device_arch(device_index: int) -> str | None:
    """Generated-program architecture of CUDA device ``device_index`` (``None`` when unsupported)."""
    import torch

    return SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device_index)
    )


def generated_program_available(device) -> bool:
    """True when this checkout registers generated programs for ``device``."""
    import torch

    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    arch = device_arch(index)
    return arch is not None and any(r["arch"] == arch for r in MODULES.values())


def route_program(arch: str, route: str) -> str:
    """Return the generated program serving ``route`` on ``arch``."""
    table = ROUTES.get(arch, ROUTES)  # per-arch tables only when they differ
    program = table.get(route)
    if program is None or arch not in PROGRAMS[program]["arches"]:
        raise NotImplementedError(
            f"no generated contiguous grouped FP8 GEMM program is registered for "
            f"route {route!r} on {arch}"
        )
    return program


def select_module(arch: str, route: str) -> str:
    """Return the registered JIT module name for one ``(arch, route)`` pair."""
    return f"{route_program(arch, route)}_{arch}"


@functools.cache
def gen_cake_grouped_fp8_gemm_module(name: str) -> JitSpec:
    record = MODULES[name]
    program = PROGRAMS[record["program"]]
    source_root, include_root = _source_dirs()
    sources = [source_root / relative for relative in program["sources"]]
    return gen_jit_spec(
        name=name,
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *program["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            source_root,
            *sorted({p.parent for p in sources}),
            include_root,
        ],
        use_fast_math=False,
    )


@functools.cache
def load_cake_grouped_fp8_gemm_module(name: str):
    return gen_cake_grouped_fp8_gemm_module(name).build_and_load()


__all__ = [
    "ARG_PLANS",
    "MODULES",
    "PROGRAMS",
    "ROUTES",
    "ROUTE_GEOMETRY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "device_arch",
    "gen_cake_grouped_fp8_gemm_module",
    "generated_program_available",
    "load_cake_grouped_fp8_gemm_module",
    "route_program",
    "select_module",
]
