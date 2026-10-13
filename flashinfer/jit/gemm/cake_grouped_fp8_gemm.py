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
    "plan2": [
        ["tma_buffer", "A"],
        ["tma_buffer", "A64"],
        ["tma_buffer", "A32"],
        ["tma_buffer", "B"],
        ["buffer", "SFA"],
        ["buffer", "SFB"],
        ["buffer", "m_indices"],
        ["tma_buffer", "C_tma"],
        ["parameter", "shape_m"],
        ["parameter", "shape_n"],
        ["parameter", "grid_n"],
        ["parameter", "k_tiles"],
        ["parameter", "sfa_row_stride"],
        ["parameter", "sfa_col_stride"],
        ["parameter", "sfb_group_stride"],
        ["parameter", "sfb_row_stride"],
        ["parameter", "sfb_col_stride"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_grouped_fp8_gemm_0244c42d588c7898347e": {
        "kernel": "kernel_cake_grouped_fp8_gemm_0244c42d588c7898347e",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_0244c42d588c7898347e_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_0244c42d588c7898347e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_1035a4c29fa0da40de08": {
        "kernel": "kernel_cake_grouped_fp8_gemm_1035a4c29fa0da40de08",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_1035a4c29fa0da40de08_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_1035a4c29fa0da40de08_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_114d583151bfd499cedb": {
        "kernel": "kernel_cake_grouped_fp8_gemm_114d583151bfd499cedb",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_114d583151bfd499cedb_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_114d583151bfd499cedb_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961": {
        "kernel": "kernel_cake_grouped_fp8_gemm_28b94ccb85e6d2d50961",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_28b94ccb85e6d2d50961_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_28b94ccb85e6d2d50961_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62": {
        "kernel": "kernel_cake_grouped_fp8_gemm_38eca7c5f2fb16897c62",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_38eca7c5f2fb16897c62_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_38eca7c5f2fb16897c62_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3": {
        "kernel": "kernel_cake_grouped_fp8_gemm_6da15715d9ab967a1cc3",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_6da15715d9ab967a1cc3_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_6da15715d9ab967a1cc3_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_800df0f72fc29eca3650": {
        "kernel": "kernel_cake_grouped_fp8_gemm_800df0f72fc29eca3650",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_800df0f72fc29eca3650_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_800df0f72fc29eca3650_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_80f755e56ac14f76e72c": {
        "kernel": "kernel_cake_grouped_fp8_gemm_80f755e56ac14f76e72c",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_80f755e56ac14f76e72c_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_80f755e56ac14f76e72c_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_815c907592a28b332939": {
        "kernel": "kernel_cake_grouped_fp8_gemm_815c907592a28b332939",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_815c907592a28b332939_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_815c907592a28b332939_binding.cu",
        ],
        "compile_flags": ["-Xptxas=-O1"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_917859181466ebe2dd3b": {
        "kernel": "kernel_cake_grouped_fp8_gemm_917859181466ebe2dd3b",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_917859181466ebe2dd3b_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_917859181466ebe2dd3b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_942f06939975f32fd8d7": {
        "kernel": "kernel_cake_grouped_fp8_gemm_942f06939975f32fd8d7",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_942f06939975f32fd8d7_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_942f06939975f32fd8d7_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_966750bd68250909df74": {
        "kernel": "kernel_cake_grouped_fp8_gemm_966750bd68250909df74",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_966750bd68250909df74_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_966750bd68250909df74_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455": {
        "kernel": "kernel_cake_grouped_fp8_gemm_cd2dca5f68bd9134a455",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_cd2dca5f68bd9134a455_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_cd2dca5f68bd9134a455_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6": {
        "kernel": "kernel_cake_grouped_fp8_gemm_e032da34f6810f0aa5b6",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_e032da34f6810f0aa5b6_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_e032da34f6810f0aa5b6_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297": {
        "kernel": "kernel_cake_grouped_fp8_gemm_f2e4effd9e0455a7e297",
        "sources": [
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_f2e4effd9e0455a7e297_kernel.cu",
            "cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_f2e4effd9e0455a7e297_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
}

MODULES: dict[str, dict[str, str]] = {
    "cake_grouped_fp8_gemm_0244c42d588c7898347e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_0244c42d588c7898347e",
    },
    "cake_grouped_fp8_gemm_0244c42d588c7898347e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_0244c42d588c7898347e",
    },
    "cake_grouped_fp8_gemm_1035a4c29fa0da40de08_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_1035a4c29fa0da40de08",
    },
    "cake_grouped_fp8_gemm_1035a4c29fa0da40de08_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_1035a4c29fa0da40de08",
    },
    "cake_grouped_fp8_gemm_114d583151bfd499cedb_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_114d583151bfd499cedb",
    },
    "cake_grouped_fp8_gemm_114d583151bfd499cedb_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_114d583151bfd499cedb",
    },
    "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961",
    },
    "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961",
    },
    "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62",
    },
    "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62",
    },
    "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3",
    },
    "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3",
    },
    "cake_grouped_fp8_gemm_800df0f72fc29eca3650_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_800df0f72fc29eca3650",
    },
    "cake_grouped_fp8_gemm_800df0f72fc29eca3650_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_800df0f72fc29eca3650",
    },
    "cake_grouped_fp8_gemm_80f755e56ac14f76e72c_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_80f755e56ac14f76e72c",
    },
    "cake_grouped_fp8_gemm_80f755e56ac14f76e72c_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_80f755e56ac14f76e72c",
    },
    "cake_grouped_fp8_gemm_815c907592a28b332939_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_815c907592a28b332939",
    },
    "cake_grouped_fp8_gemm_815c907592a28b332939_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_815c907592a28b332939",
    },
    "cake_grouped_fp8_gemm_917859181466ebe2dd3b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_917859181466ebe2dd3b",
    },
    "cake_grouped_fp8_gemm_917859181466ebe2dd3b_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_917859181466ebe2dd3b",
    },
    "cake_grouped_fp8_gemm_942f06939975f32fd8d7_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_942f06939975f32fd8d7",
    },
    "cake_grouped_fp8_gemm_942f06939975f32fd8d7_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_942f06939975f32fd8d7",
    },
    "cake_grouped_fp8_gemm_966750bd68250909df74_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_966750bd68250909df74",
    },
    "cake_grouped_fp8_gemm_966750bd68250909df74_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_966750bd68250909df74",
    },
    "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455",
    },
    "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455",
    },
    "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6",
    },
    "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6",
    },
    "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297",
    },
    "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297",
    },
}

ROUTES: dict[str, Any] = {
    "bs_ue8m0_n128": "cake_grouped_fp8_gemm_942f06939975f32fd8d7",
    "bs_ue8m0_n128_run32": "cake_grouped_fp8_gemm_38eca7c5f2fb16897c62",
    "bs_ue8m0_n256": "cake_grouped_fp8_gemm_1035a4c29fa0da40de08",
    "bs_ue8m0_n256_run32": "cake_grouped_fp8_gemm_6da15715d9ab967a1cc3",
    "deepk_c2_ab6_scale1_n256_or_m4096": "cake_grouped_fp8_gemm_966750bd68250909df74",
    "deepk_c2_k_multiple_512": "cake_grouped_fp8_gemm_917859181466ebe2dd3b",
    "deepk_c2_kg1_non_kg4_k_blocks": "cake_grouped_fp8_gemm_cd2dca5f68bd9134a455",
    "deepk_c2_scalar_output": "cake_grouped_fp8_gemm_80f755e56ac14f76e72c",
    "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4": "cake_grouped_fp8_gemm_e032da34f6810f0aa5b6",
    "deepk_cg2_ab5_x16_pair_wait_kg4": "cake_grouped_fp8_gemm_28b94ccb85e6d2d50961",
    "deepk_cg2_ab6_early4_output_alias_kg4": "cake_grouped_fp8_gemm_800df0f72fc29eca3650",
    "deepk_cg2_ab7_bscale_prefetch_kg4": "cake_grouped_fp8_gemm_f2e4effd9e0455a7e297",
    "deepk_n256_ab4_three_panel_kg4": "cake_grouped_fp8_gemm_0244c42d588c7898347e",
    "k128_exact": "cake_grouped_fp8_gemm_815c907592a28b332939",
    "k384_exact_three_partial": "cake_grouped_fp8_gemm_114d583151bfd499cedb",
}

# Tile geometry of every host route (including routes without an exported
# program), resolved from the source dispatcher: {route: {tile_m, tile_n,
# cluster_ctas}}.
ROUTE_GEOMETRY: dict[str, dict[str, int]] = {
    "bs_ue8m0_n128": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
    "bs_ue8m0_n128_run32": {"tile_m": 128, "tile_n": 128, "cluster_ctas": 1},
    "bs_ue8m0_n256": {"tile_m": 128, "tile_n": 256, "cluster_ctas": 1},
    "bs_ue8m0_n256_run32": {"tile_m": 128, "tile_n": 256, "cluster_ctas": 1},
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
