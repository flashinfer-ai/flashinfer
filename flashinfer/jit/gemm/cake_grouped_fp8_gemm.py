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
    "cake_grouped_fp8_gemm_05d9392a254fd1e1a82d": {
        "arch": "sm_100a",
        "route": "deepk_n256_ab4_three_panel_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_05d9392a254fd1e1a82d",
        "cache_name": "cake_grouped_fp8_gemm_05d9392a254fd1e1a82d_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_05d9392a254fd1e1a82d_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_05d9392a254fd1e1a82d_binding.cu",
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
        "closure_sha256": "8a0dd8edaea8629f2ad8348e6086e6008bb611881a91944e62aa3721abc3a436",
    },
    "cake_grouped_fp8_gemm_4349348d47faa3da5e4c": {
        "arch": "sm_100a",
        "route": "deepk_c2_kg1_non_kg4_k_blocks",
        "kernel": "kernel_cake_grouped_fp8_gemm_4349348d47faa3da5e4c",
        "cache_name": "cake_grouped_fp8_gemm_4349348d47faa3da5e4c_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_4349348d47faa3da5e4c_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_4349348d47faa3da5e4c_binding.cu",
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
        "closure_sha256": "349470d35a05458e176bafc780e552a187de85cfa4e0ff480d7c7d0170203134",
    },
    "cake_grouped_fp8_gemm_47c6a3211c2c38184a50": {
        "arch": "sm_100a",
        "route": "deepk_c2_k_multiple_512",
        "kernel": "kernel_cake_grouped_fp8_gemm_47c6a3211c2c38184a50",
        "cache_name": "cake_grouped_fp8_gemm_47c6a3211c2c38184a50_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_47c6a3211c2c38184a50_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_47c6a3211c2c38184a50_binding.cu",
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
        "closure_sha256": "d3a28081806eea610f9752a945385692c2828d4c24b12af2f135b7861646f145",
    },
    "cake_grouped_fp8_gemm_6bf8b4f8ce900bd50808": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab7_bscale_prefetch_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_6bf8b4f8ce900bd50808",
        "cache_name": "cake_grouped_fp8_gemm_6bf8b4f8ce900bd50808_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_6bf8b4f8ce900bd50808_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_6bf8b4f8ce900bd50808_binding.cu",
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
        "closure_sha256": "5ffba4292aa109523bc5b6c64207d4b421dcfea9ebe837603fbf06eb6b02a84c",
    },
    "cake_grouped_fp8_gemm_81102c099db74a613267": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab5_x16_pair_wait_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_81102c099db74a613267",
        "cache_name": "cake_grouped_fp8_gemm_81102c099db74a613267_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_81102c099db74a613267_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_81102c099db74a613267_binding.cu",
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
        "closure_sha256": "1704954b695525e67f98946f980b20bc7ee08ed4c6e680b7c0de45c77a3d5a3a",
    },
    "cake_grouped_fp8_gemm_aacc7ec3034ef4442dcb": {
        "arch": "sm_100a",
        "route": "deepk_c2_ab6_scale1_n256_or_m4096",
        "kernel": "kernel_cake_grouped_fp8_gemm_aacc7ec3034ef4442dcb",
        "cache_name": "cake_grouped_fp8_gemm_aacc7ec3034ef4442dcb_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_aacc7ec3034ef4442dcb_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_aacc7ec3034ef4442dcb_binding.cu",
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
        "closure_sha256": "522cadb13b71f74490530837211b67a97663d268caf70ad3bcfc7275540e632f",
    },
    "cake_grouped_fp8_gemm_cbf40f90c6edef24c9c4": {
        "arch": "sm_100a",
        "route": "k384_exact_three_partial",
        "kernel": "kernel_cake_grouped_fp8_gemm_cbf40f90c6edef24c9c4",
        "cache_name": "cake_grouped_fp8_gemm_cbf40f90c6edef24c9c4_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_cbf40f90c6edef24c9c4_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_cbf40f90c6edef24c9c4_binding.cu",
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
        "closure_sha256": "c2b7aa0703bb7629089b6c262e7cc231157da804af028b339af39df4c499b0f7",
    },
    "cake_grouped_fp8_gemm_cf93e6fa926c6736f686": {
        "arch": "sm_100a",
        "route": "deepk_c2_scalar_output",
        "kernel": "kernel_cake_grouped_fp8_gemm_cf93e6fa926c6736f686",
        "cache_name": "cake_grouped_fp8_gemm_cf93e6fa926c6736f686_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_cf93e6fa926c6736f686_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_cf93e6fa926c6736f686_binding.cu",
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
        "closure_sha256": "291138ab86bc18aa7f8e76a6ad642d80c49ba0c9bd1b2bf39905de6d6071f6c1",
    },
    "cake_grouped_fp8_gemm_d7f8d5132ca62707fae1": {
        "arch": "sm_100a",
        "route": "k128_exact",
        "kernel": "kernel_cake_grouped_fp8_gemm_d7f8d5132ca62707fae1",
        "cache_name": "cake_grouped_fp8_gemm_d7f8d5132ca62707fae1_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d7f8d5132ca62707fae1_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_d7f8d5132ca62707fae1_binding.cu",
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
        "closure_sha256": "2df081aaf694d8ae614e6fb1bb3f195b39e140012d81f9320c15e1b57fec8238",
    },
    "cake_grouped_fp8_gemm_ee7f6d407fcf62bd1dfc": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_ee7f6d407fcf62bd1dfc",
        "cache_name": "cake_grouped_fp8_gemm_ee7f6d407fcf62bd1dfc_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_ee7f6d407fcf62bd1dfc_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_ee7f6d407fcf62bd1dfc_binding.cu",
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
        "closure_sha256": "c49899f983ce085af3de248eaa57166fd9eca8b47dc26ced4ee098118c1cf2c8",
    },
    "cake_grouped_fp8_gemm_fd349a3926d8ae06cece": {
        "arch": "sm_100a",
        "route": "deepk_cg2_ab6_early4_output_alias_kg4",
        "kernel": "kernel_cake_grouped_fp8_gemm_fd349a3926d8ae06cece",
        "cache_name": "cake_grouped_fp8_gemm_fd349a3926d8ae06cece_sm_100a",
        "sources": [
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_fd349a3926d8ae06cece_kernel.cu",
            "cake_grouped_fp8_gemm/sm_100a/cake_grouped_fp8_gemm_fd349a3926d8ae06cece_binding.cu",
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
        "closure_sha256": "c5bc480e7122ee0aee5f91ab4b963c9ca4409129f103a93192d562b45bf66d7a",
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
