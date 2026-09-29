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
    "cake_grouped_fp8_fused_silu_quant_17106c4a725cda131dad": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8_wide",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_17106c4a725cda131dad",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_17106c4a725cda131dad_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_17106c4a725cda131dad_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_17106c4a725cda131dad_binding.cu",
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
        "closure_sha256": "12ed4010de50fe7844dde1237eec0be93b85fcf636df2ce9efbcfc15b33a18a2",
    },
    "cake_grouped_fp8_fused_silu_quant_6b90356750732078015e": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_mixedsched_lpt_kg4",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_6b90356750732078015e",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_6b90356750732078015e_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_6b90356750732078015e_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_6b90356750732078015e_binding.cu",
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
        "closure_sha256": "6d6666be08efdf06993484cbddad4cb80c938f4ca94f07de58aa3bf70ec67c1c",
    },
    "cake_grouped_fp8_fused_silu_quant_79f3c6728d9a934fc67e": {
        "arch": "sm_100a",
        "route": "gemm_then_silu_mul_group_quant_fp8",
        "stage": "main",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_79f3c6728d9a934fc67e",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_79f3c6728d9a934fc67e_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_79f3c6728d9a934fc67e_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_79f3c6728d9a934fc67e_binding.cu",
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
        "closure_sha256": "276e323256df56bcf4c2a41111992ad6e46afa5a8dd404206327429f228531d0",
    },
    "cake_grouped_fp8_fused_silu_quant_ab758f32f1a4c65b047c": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "pair",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_ab758f32f1a4c65b047c",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_ab758f32f1a4c65b047c_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_ab758f32f1a4c65b047c_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_ab758f32f1a4c65b047c_binding.cu",
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
        "closure_sha256": "07fc506870ce1fc5b43c9da7d6823fccfe35ae6b918cda094ab007822e5468c0",
    },
    "cake_grouped_fp8_fused_silu_quant_f47a8009af8639c08f11": {
        "arch": "sm_100a",
        "route": "fused_cg2_ab7_pairsched_solotail_kg4",
        "stage": "tail",
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_f47a8009af8639c08f11",
        "cache_name": "cake_grouped_fp8_fused_silu_quant_f47a8009af8639c08f11_sm_100a",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_f47a8009af8639c08f11_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/sm_100a/cake_grouped_fp8_fused_silu_quant_f47a8009af8639c08f11_binding.cu",
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
        "closure_sha256": "ef5d06f8e0d2c20c1adad9b5fea09b3ead0134ca6bfa9c8c54df299143f73d4a",
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
