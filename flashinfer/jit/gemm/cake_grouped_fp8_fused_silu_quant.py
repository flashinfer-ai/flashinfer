# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization programs.

One record per exported physical program (one generated kernel stage of one
route on one architecture; the fused route launches a pair kernel and a tail
kernel, the small-M route one activation kernel).  Each record names its
generated translation units under ``csrc/cake_grouped_fp8_fused_silu_quant``,
the exact compile flags of the source build, the tvm-ffi entry, the positional
argument plan and the caller-owned TMA descriptor storage size.  ``MODULES``
and ``ROUTE_GEOMETRY`` are populated verbatim by the generated-program export;
do not edit them by hand.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

MODULES: dict[str, dict[str, Any]] = {
    "cake_grouped_fp8_fused_silu_quant_0ea2266c93766fea306d": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_0ea2266c93766fea306d",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_0ea2266c93766fea306d_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_0ea2266c93766fea306d_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_0ea2266c93766fea306d_binding.cu",
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
        "pdl": False,
        "closure_sha256": "f2542c4f4f5547e30d4c5c7e54f1bf0a4d5135e9e63d59e99d105cd35a60a32e",
    },
    "cake_grouped_fp8_fused_silu_quant_2cdb7a86771e30a3f7a1": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "pair",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_2cdb7a86771e30a3f7a1",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_2cdb7a86771e30a3f7a1_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_2cdb7a86771e30a3f7a1_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_2cdb7a86771e30a3f7a1_binding.cu",
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
        "pdl": False,
        "closure_sha256": "b89b45268ae600def262487c2126b037d18ec4a96b0cdad69e02b1b7d81b56b9",
    },
    "cake_grouped_fp8_fused_silu_quant_6e486bb3e3c56ba4784d": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8_wide",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_6e486bb3e3c56ba4784d",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_6e486bb3e3c56ba4784d_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_6e486bb3e3c56ba4784d_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_6e486bb3e3c56ba4784d_binding.cu",
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
        "pdl": False,
        "closure_sha256": "85dde64192518f3ff12d1bd1d75dac32b9ce06625af084b437cd64f4830762af",
    },
    "cake_grouped_fp8_fused_silu_quant_c1f6ef1750e980b906a0": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "tail",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_c1f6ef1750e980b906a0",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_c1f6ef1750e980b906a0_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_c1f6ef1750e980b906a0_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_c1f6ef1750e980b906a0_binding.cu",
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
        "pdl": True,
        "closure_sha256": "f6f47274c244633982c4f0e2730a11a49fc1dcad011f78da2f1d241ea8a07ca6",
    },
}

ROUTE_GEOMETRY: dict[str, dict[str, int]] = {
    "fused_cg2_ab7_pairsched_solotail_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
    "fused_cg2_ab7_pairsched_solotail_kg4_tail": {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_ctas": 1,
    },
    "gemm_then_silu_mul_group_quant_fp8": {
        "tile_m": 1,
        "tile_n": 256,
        "cluster_ctas": 1,
    },
    "gemm_then_silu_mul_group_quant_fp8_wide": {
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


def select_stage_module(arch: str, route: str, stage: str) -> str:
    """Return the registered module name of one generated kernel stage of ``route`` on ``arch``."""
    for name, record in MODULES.items():
        if (
            record["arch"] == arch
            and record["route"] == route
            and record.get("stage", "main") == stage
        ):
            return name
    raise NotImplementedError(
        f"no generated fused grouped FP8 gate_up+SwiGLU+quant program is registered for "
        f"route {route!r} stage {stage!r} on {arch}"
    )


def select_module(arch: str, route: str) -> str:
    """Return the registered module name of the first generated kernel of ``route`` on ``arch``."""
    stages = [
        record.get("stage", "main")
        for record in MODULES.values()
        if record["arch"] == arch and record["route"] == route
    ]
    if not stages:
        raise NotImplementedError(
            f"no generated fused grouped FP8 gate_up+SwiGLU+quant program is registered for "
            f"route {route!r} on {arch}"
        )
    first = min(stages, key=lambda stage: STAGE_ORDER.get(stage, len(STAGE_ORDER)))
    return select_stage_module(arch, route, first)


# Launch order of the generated kernel stages (records without a stage are single-kernel routes).
STAGE_ORDER = {"main": 0, "pair": 0, "tail": 1}


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
    "select_stage_module",
    "STAGE_ORDER",
]
