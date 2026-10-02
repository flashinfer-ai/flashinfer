# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization programs.

``PROGRAMS`` lists every generated program once (one generated kernel stage of
one route: the fused route launches a pair kernel and a tail kernel, the
mixed-schedule route one kernel, the GEMM + act routes one activation kernel):
its two translation units under ``csrc/cake_grouped_fp8_fused_silu_quant``
(one source shared by every architecture it runs on), the exact compile flags
of the source build, the tvm-ffi entry, whether the stage is launched with the
programmatic-dependent-launch attribute, and the id of its positional argument
plan in ``ARG_PLANS``.  ``MODULES`` names one JIT module per (program,
architecture) and ``ROUTES`` maps every ``route:stage`` key to its program.
The tables and ``ROUTE_GEOMETRY`` are written by the generated-program export;
do not edit them by hand.
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
        ["buffer", "out_q"],
        ["buffer", "out_s"],
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
        ["tma_buffer", "A64"],
        ["tma_buffer", "B"],
        ["buffer", "out_q"],
        ["buffer", "out_s"],
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
    "plan2": [
        ["buffer", "y"],
        ["buffer", "out_q"],
        ["buffer", "out_s"],
        ["parameter", "M"],
        ["parameter", "H"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528": {
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "pdl": True,
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682": {
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "pdl": False,
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767": {
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "pdl": False,
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862": {
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "pdl": False,
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac": {
        "kernel": "kernel_cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac",
        "sources": [
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac_kernel.cu",
            "cake_grouped_fp8_fused_silu_quant/cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "pdl": False,
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
}

MODULES: dict[str, dict[str, str]] = {
    "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528",
    },
    "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528",
    },
    "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682",
    },
    "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682",
    },
    "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767",
    },
    "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767",
    },
    "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862",
    },
    "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862",
    },
    "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac",
    },
    "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac",
    },
}

ROUTES: dict[str, Any] = {
    "fused_cg2_ab7_mixedsched_lpt_kg4:main": "cake_grouped_fp8_fused_silu_quant_29d4dcf04d2354078682",
    "fused_cg2_ab7_pairsched_solotail_kg4:pair": "cake_grouped_fp8_fused_silu_quant_e47f76da02f89b7261ac",
    "fused_cg2_ab7_pairsched_solotail_kg4:tail": "cake_grouped_fp8_fused_silu_quant_0f3e441101291f871528",
    "gemm_then_silu_mul_group_quant_fp8:main": "cake_grouped_fp8_fused_silu_quant_a0d6cfec3d9e63feb767",
    "gemm_then_silu_mul_group_quant_fp8_wide:main": "cake_grouped_fp8_fused_silu_quant_a548221c000bdefa7862",
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

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# Launch order of the generated kernel stages (records without a stage are single-kernel routes).
STAGE_ORDER = {"main": 0, "pair": 0, "tail": 1}


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


def _route_table(arch: str) -> dict[str, str]:
    return ROUTES[arch] if arch in ROUTES else ROUTES  # per-arch tables only when they differ


def stage_program(arch: str, route: str, stage: str) -> str:
    """Return the generated program of one kernel stage of ``route`` on ``arch``."""
    program = _route_table(arch).get(f"{route}:{stage}")
    if program is None or arch not in PROGRAMS[program]["arches"]:
        raise NotImplementedError(
            f"no generated fused grouped FP8 gate_up+SwiGLU+quant program is registered for "
            f"route {route!r} stage {stage!r} on {arch}"
        )
    return program


def select_stage_module(arch: str, route: str, stage: str) -> str:
    """Return the registered JIT module name of one generated kernel stage of ``route`` on ``arch``."""
    return f"{stage_program(arch, route, stage)}_{arch}"


def select_module(arch: str, route: str) -> str:
    """Return the registered JIT module name of the first generated kernel of ``route`` on ``arch``."""
    stages = [
        key.split(":", 1)[1]
        for key in _route_table(arch)
        if key.split(":", 1)[0] == route
    ]
    if not stages:
        raise NotImplementedError(
            f"no generated fused grouped FP8 gate_up+SwiGLU+quant program is registered for "
            f"route {route!r} on {arch}"
        )
    first = min(stages, key=lambda stage: STAGE_ORDER.get(stage, len(STAGE_ORDER)))
    return select_stage_module(arch, route, first)


@functools.cache
def gen_cake_grouped_fp8_fused_silu_quant_module(name: str) -> JitSpec:
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
def load_cake_grouped_fp8_fused_silu_quant_module(name: str):
    return gen_cake_grouped_fp8_fused_silu_quant_module(name).build_and_load()


__all__ = [
    "ARG_PLANS",
    "MODULES",
    "PROGRAMS",
    "ROUTES",
    "ROUTE_GEOMETRY",
    "STAGE_ORDER",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "device_arch",
    "gen_cake_grouped_fp8_fused_silu_quant_module",
    "generated_program_available",
    "load_cake_grouped_fp8_fused_silu_quant_module",
    "select_module",
    "select_stage_module",
    "stage_program",
]
