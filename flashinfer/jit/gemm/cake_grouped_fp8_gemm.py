# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell contiguous grouped FP8 GEMM programs.

One record per exported physical program (one generated kernel route on one
architecture).  Each record names its generated translation units under
``csrc/cake_grouped_fp8_gemm``, the exact compile flags of the source build,
the tvm-ffi entry, the positional argument plan and the caller-owned TMA
descriptor storage size.  ``MODULES`` and ``ROUTE_GEOMETRY`` are populated
verbatim by the generated-program export; do not edit them by hand.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

MODULES: dict[str, dict[str, Any]] = {
    "cake_grouped_fp8_gemm_35aa26324d5da2412ba8": {
        "arch": "sm_100a",
        "route": "k384_exact_three_partial",
        "kernel": "kernel_cake_grouped_fp8_gemm_35aa26324d5da2412ba8",
        "cache_name": "cake_grouped_fp8_gemm_35aa26324d5da2412ba8_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_35aa26324d5da2412ba8_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_35aa26324d5da2412ba8_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "eafa137f07c9b2ecb6af24bccc27354bb13d59800bcab6eb12661c844ce625a8",
    },
    "cake_grouped_fp8_gemm_50892258a10271c8c818": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab7_bscale_prefetch_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_50892258a10271c8c818",
        "cache_name": "cake_grouped_fp8_gemm_50892258a10271c8c818_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_50892258a10271c8c818_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_50892258a10271c8c818_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "78d2a6bf9085da5a6894bdc348f911a36b328a3f80b0eba8840ee03a8bf880ab",
    },
    "cake_grouped_fp8_gemm_820417434dc231a85102": {
        "arch": "sm_100a",
        "route": "deepk_c2_k_multiple_512",
        "kernel": "kernel_cake_grouped_fp8_gemm_820417434dc231a85102",
        "cache_name": "cake_grouped_fp8_gemm_820417434dc231a85102_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_820417434dc231a85102_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_820417434dc231a85102_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "3a9e950764a0c4a9b4150143349aa7c3c2767b7e22da63b8bec99baddc574b02",
    },
    "cake_grouped_fp8_gemm_840c9c693e988f7d33a6": {
        "arch": "sm_100a",
        "route": "deepk_n256_ab4_three_panel_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_840c9c693e988f7d33a6",
        "cache_name": "cake_grouped_fp8_gemm_840c9c693e988f7d33a6_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_840c9c693e988f7d33a6_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_840c9c693e988f7d33a6_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "1f222bac1349dcc22eafc00564e577292b96aa317752b8663267b1498140054a",
    },
    "cake_grouped_fp8_gemm_92562983a1b78171dfbb": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_92562983a1b78171dfbb",
        "cache_name": "cake_grouped_fp8_gemm_92562983a1b78171dfbb_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_92562983a1b78171dfbb_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_92562983a1b78171dfbb_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "9a09e4b8ca53c498d2664a9c10e2dfa84f7efcd216cad690ebef8e17b0bdbf71",
    },
    "cake_grouped_fp8_gemm_a3d61d047942c047ddf0": {
        "arch": "sm_100a",
        "route": "deepk_c2_ab6_scale1_n256_or_m4096",
        "kernel": "kernel_cake_grouped_fp8_gemm_a3d61d047942c047ddf0",
        "cache_name": "cake_grouped_fp8_gemm_a3d61d047942c047ddf0_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_a3d61d047942c047ddf0_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_a3d61d047942c047ddf0_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "tma_buffer",
                "a_scale",
            ],
            [
                "tma_buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 640,
        "closure_sha256": "248346fa36fd6471d3792141c9aacac2cc517a8197832463882c4145e1f50857",
    },
    "cake_grouped_fp8_gemm_a781d09144e5fc8398b6": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab6_early4_output_alias_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_a781d09144e5fc8398b6",
        "cache_name": "cake_grouped_fp8_gemm_a781d09144e5fc8398b6_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_a781d09144e5fc8398b6_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_a781d09144e5fc8398b6_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "c9e86a1987809f923fc62062ef347ba7b6c60441e780f9a20c17991833c2057b",
    },
    "cake_grouped_fp8_gemm_d1937e95c0a47c99210e": {
        "arch": "sm_100a",
        "route": "deepk_c2_scalar_output",
        "kernel": "kernel_cake_grouped_fp8_gemm_d1937e95c0a47c99210e",
        "cache_name": "cake_grouped_fp8_gemm_d1937e95c0a47c99210e_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d1937e95c0a47c99210e_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d1937e95c0a47c99210e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "b4641736935e1d83c2d4c23ffdab1b8e3bf28bac8d0e0c00cecfa18e7eec32b7",
    },
    "cake_grouped_fp8_gemm_d33456ef73586060263f": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab5_x16_pair_wait_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_d33456ef73586060263f",
        "cache_name": "cake_grouped_fp8_gemm_d33456ef73586060263f_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d33456ef73586060263f_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d33456ef73586060263f_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "538a72cf37e19cbc899ace0bbbc30c9bd74ca9850cfe9e3c1e18893bbd026e71",
    },
    "cake_grouped_fp8_gemm_d9bf310c8d9d2456a515": {
        "arch": "sm_100a",
        "route": "k128_exact",
        "kernel": "kernel_cake_grouped_fp8_gemm_d9bf310c8d9d2456a515",
        "cache_name": "cake_grouped_fp8_gemm_d9bf310c8d9d2456a515_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d9bf310c8d9d2456a515_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d9bf310c8d9d2456a515_binding.cu",
        ],
        "compile_flags": [
            "-Xptxas=-O1",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "1e538963295dd07622ff4ca6543c7808e512b3ed390d76b01c2c6d7d54d643a0",
    },
    "cake_grouped_fp8_gemm_e5ae93fe77385fc29f12": {
        "arch": "sm_100a",
        "route": "deepk_c2_kg1_non_kg4_k_blocks",
        "kernel": "kernel_cake_grouped_fp8_gemm_e5ae93fe77385fc29f12",
        "cache_name": "cake_grouped_fp8_gemm_e5ae93fe77385fc29f12_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_e5ae93fe77385fc29f12_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_e5ae93fe77385fc29f12_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "buffer",
                "C",
            ],
            [
                "buffer",
                "a_scale",
            ],
            [
                "buffer",
                "b_scale",
            ],
            [
                "buffer",
                "m_indices",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "G",
            ],
            [
                "workspace",
                "tma_descriptor_workspace",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "tma_workspace_bytes": 384,
        "closure_sha256": "3ca9e77d73cebd457b1a93cb16f4e0d31648695c3dea18579b93ab0a4c61965f",
    },
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
    "deepk_c2_k_multiple_512": {
        "tile_m": 128,
        "tile_n": 128,
        "cluster_ctas": 1,
    },
    "deepk_c2_kg1_non_kg4_k_blocks": {
        "tile_m": 128,
        "tile_n": 128,
        "cluster_ctas": 1,
    },
    "deepk_c2_scalar_output": {
        "tile_m": 128,
        "tile_n": 128,
        "cluster_ctas": 1,
    },
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
    "deepk_n256_ab4_three_panel_kg4": {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_ctas": 1,
    },
    "k128_exact": {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_ctas": 1,
    },
    "k384_exact_three_partial": {
        "tile_m": 128,
        "tile_n": 128,
        "cluster_ctas": 1,
    },
}

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a"}


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


def generated_program_available(device) -> bool:
    """True when this checkout registers generated programs for ``device``."""
    import torch

    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and any(r["arch"] == arch for r in MODULES.values())


def select_module(arch: str, route: str) -> str:
    """Return the registered module name for one ``(arch, route)`` pair."""
    for name, record in MODULES.items():
        if record["arch"] == arch and record["route"] == route:
            return name
    raise NotImplementedError(
        f"no generated contiguous grouped FP8 GEMM program is registered for "
        f"route {route!r} on {arch}"
    )


@functools.cache
def gen_cake_grouped_fp8_gemm_module(name: str) -> JitSpec:
    record = MODULES[name]
    source_root, include_root = _source_dirs()
    sources = [source_root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=record["cache_name"],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
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
    "MODULES",
    "ROUTE_GEOMETRY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gen_cake_grouped_fp8_gemm_module",
    "generated_program_available",
    "load_cake_grouped_fp8_gemm_module",
    "select_module",
]
