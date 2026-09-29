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
    "cake_grouped_fp8_fused_silu_quant_0979ff47de2f0a213b53": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_mixedsched_lpt_kg4",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_0979ff47de2f0a213b53",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_0979ff47de2f0a213b53_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_0979ff47de2f0a213b53_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_0979ff47de2f0a213b53_binding.cu",
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
        "closure_sha256": "d09234a0e4fa59fdd2448998f18a9415df636169a489adddaded25b2346fa09b",
    },
    "cake_grouped_fp8_fused_silu_quant_83a61feba610b0fc82e6": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_83a61feba610b0fc82e6",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_83a61feba610b0fc82e6_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_83a61feba610b0fc82e6_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_83a61feba610b0fc82e6_binding.cu",
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
        "closure_sha256": "6be1f67696ceecb891b734103e926f02c2c18f49449df0fab7088b19399558fa",
    },
    "cake_grouped_fp8_fused_silu_quant_96b1c66b030aa5102196": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "pair",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_96b1c66b030aa5102196",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_96b1c66b030aa5102196_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_96b1c66b030aa5102196_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_96b1c66b030aa5102196_binding.cu",
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
        "closure_sha256": "f31036da65b828dc0ee8f5e7eaa281054610bbe6eec2665ee22d5e9bb6eec22e",
    },
    "cake_grouped_fp8_fused_silu_quant_e23c4e332758f7f5f405": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8_wide",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_e23c4e332758f7f5f405",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_e23c4e332758f7f5f405_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_e23c4e332758f7f5f405_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_e23c4e332758f7f5f405_binding.cu",
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
        "closure_sha256": "e4abf5b231526af2adb292b7f7fd499aab210063f816b72146b8c9e4b59bddfe",
    },
    "cake_grouped_fp8_fused_silu_quant_ee44e6d5f77cb4105519": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "tail",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_ee44e6d5f77cb4105519",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_ee44e6d5f77cb4105519_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_ee44e6d5f77cb4105519_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_ee44e6d5f77cb4105519_binding.cu",
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
        "closure_sha256": "bc5c0c19f99380b8fa9a75a1a6607a7c1e46b379279f562f328347236c2e1da9",
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
