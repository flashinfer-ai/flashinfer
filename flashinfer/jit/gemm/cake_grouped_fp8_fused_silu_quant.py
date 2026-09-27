# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization programs.

One record per exported physical program (one generated kernel route on one
architecture).  Each record names its generated translation units under
``csrc/cake_grouped_fp8_fused_silu_quant``, the exact compile flags of the source build,
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
    "cake_grouped_fp8_fused_silu_quant_3976a874cc916684c6b0": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_3976a874cc916684c6b0",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_3976a874cc916684c6b0_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_3976a874cc916684c6b0_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_3976a874cc916684c6b0_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "buffer",
                "y",
            ],
            [
                "buffer",
                "out_q",
            ],
            [
                "buffer",
                "out_s",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "H",
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
        "tma_workspace_bytes": 0,
        "closure_sha256": "5515138571101fd184ae4873481201cad419b970d11eff2cf857781f3de04f1c",
    },
    "cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_kg4",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2_binding.cu",
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
                "buffer",
                "out_q",
            ],
            [
                "buffer",
                "out_s",
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
        "tma_workspace_bytes": 256,
        "closure_sha256": "5edfabacedc2aaf4df2b95aac264bbbb630de181dfebecc379808261db9cd73a",
    },
}

ROUTE_GEOMETRY: dict[str, dict[str, int]] = {
    "fused_cg2_ab7_pairsched_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "gemm_then_silu_mul_group_quant_fp8": {
        "tile_m": 1,
        "tile_n": 256,
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
        f"no generated fused grouped FP8 gate_up+SwiGLU+quant program is registered for "
        f"route {route!r} on {arch}"
    )


@functools.cache
def gen_cake_grouped_fp8_fused_silu_quant_module(name: str) -> JitSpec:
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
def load_cake_grouped_fp8_fused_silu_quant_module(name: str):
    return gen_cake_grouped_fp8_fused_silu_quant_module(name).build_and_load()


__all__ = [
    "MODULES",
    "ROUTE_GEOMETRY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gen_cake_grouped_fp8_fused_silu_quant_module",
    "generated_program_available",
    "load_cake_grouped_fp8_fused_silu_quant_module",
    "select_module",
]
