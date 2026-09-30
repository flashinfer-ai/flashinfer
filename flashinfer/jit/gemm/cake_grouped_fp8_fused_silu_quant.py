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
    "cake_grouped_fp8_fused_silu_quant_56971d378e3a91c5133b": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "pair",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_56971d378e3a91c5133b",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_56971d378e3a91c5133b_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_56971d378e3a91c5133b_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_56971d378e3a91c5133b_binding.cu",
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
        "closure_sha256": "d6d1a4cf4fe99dd04c24c04110c0ac0f9fd33864c913f05fd9cbf6b84358dace",
    },
    "cake_grouped_fp8_fused_silu_quant_584484a22d5c9ae40cc0": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "tail",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_584484a22d5c9ae40cc0",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_584484a22d5c9ae40cc0_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_584484a22d5c9ae40cc0_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_584484a22d5c9ae40cc0_binding.cu",
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
        "closure_sha256": "0eb04abe7fac19c00ab7a4b9a16c04d1a507cd8876c739863ff28541b8a16179",
    },
    "cake_grouped_fp8_fused_silu_quant_933b2c891934f1c3b281": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8_wide",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_933b2c891934f1c3b281",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_933b2c891934f1c3b281_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_933b2c891934f1c3b281_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_933b2c891934f1c3b281_binding.cu",
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
        "closure_sha256": "d51996ed0c704912c8211b16635e0e681d1b372a8854637be70278b0f2dec0ce",
    },
    "cake_grouped_fp8_fused_silu_quant_b3d86b40d8a1d300dbed": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_mixedsched_lpt_kg4",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_b3d86b40d8a1d300dbed",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_b3d86b40d8a1d300dbed_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_b3d86b40d8a1d300dbed_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_b3d86b40d8a1d300dbed_binding.cu",
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
                "A64",
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
        "tma_workspace_bytes": 384,
        "pdl": False,
        "closure_sha256": "e8da52a9397a71fe21084dc2f104818eb49d203456394599f79f64c0ee63eb68",
    },
    "cake_grouped_fp8_fused_silu_quant_daa1f1cbe7d2b1ee1482": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_daa1f1cbe7d2b1ee1482",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_daa1f1cbe7d2b1ee1482_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_daa1f1cbe7d2b1ee1482_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_daa1f1cbe7d2b1ee1482_binding.cu",
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
        "closure_sha256": "f039658adb6391ba79a4983473cdcc99df1c7908d329cefd0fc23125785b262f",
    },
}

ROUTE_GEOMETRY: dict[str, dict[str, int]] = {
    "fused_cg2_ab7_mixedsched_lpt_kg4": {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_ctas": 2,
    },
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
