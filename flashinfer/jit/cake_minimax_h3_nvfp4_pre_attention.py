# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the Cake MiniMax-H3 NVFP4 pre-attention stages.

Two generated programs per exact target: the norm + AdaLN + NVFP4 quantize
stage (one source shared by both targets) and the fused NVFP4 QKV GEMM + Q/K
norm + RoPE + destination pack stage (one schedule per target).  The token
count ``M`` and the destination geometry are runtime launch parameters, so
every destination partition count ``P`` routes to the same two programs of its
target.  Tensor maps are passed by value; no descriptor workspace exists.
``MODULES`` and ``ROUTES`` are written by the Cake export; the empty tables are
the source-only placeholder and every route lookup fails until they are filled.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

MiniMaxH3Nvfp4Target = Literal["sm100a", "sm103a"]
MiniMaxH3Nvfp4Stage = Literal[
    "norm_adaln_nvfp4_quantize",
    "qkv_nvfp4_gemm_fused_pack",
]

MINIMAX_H3_NVFP4_STAGES: tuple[MiniMaxH3Nvfp4Stage, ...] = (
    "norm_adaln_nvfp4_quantize",
    "qkv_nvfp4_gemm_fused_pack",
)
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)
_TARGET_FLAGS: dict[str, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# program name -> {"sources": [kernel, binding] (relative to the csrc family
# directory), "stage", "template", "ffi_entry", "compile_flags", "arg_plan",
# "tma_abi", "launch_grid_rule", "launch_block", "cluster",
# "dynamic_smem_bytes", "use_pdl", "closure": {target: sealed closure identity}}.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_nvfp4_pre_attention_10cb85a9569b85023fde": {
        "sources": [
            "cake_minimax_h3_nvfp4_pre_attention_10cb85a9569b85023fde_kernel.cu",
            "cake_minimax_h3_nvfp4_pre_attention_10cb85a9569b85023fde_binding.cu",
        ],
        "stage": "qkv_nvfp4_gemm_fused_pack",
        "template": "minimax_h3_qkv_nvfp4_gemm_fused_pack_k96_v1",
        "ffi_entry": "run",
        "compile_flags": ["--use_fast_math"],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "alpha"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "rope_cos_sin"],
            ["buffer", "out_global_scale"],
            ["buffer", "out_q"],
            ["buffer", "out_sf"],
            ["tma_buffer", "OUTQ"],
            ["buffer", "qkv_words"],
            ["buffer", "debug_q_words"],
            ["buffer", "debug_k_words"],
            ["parameter", "write_debug"],
            ["parameter", "eps"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "HEADS_PER_DESTINATION"],
            ["parameter", "ROWS_PER_DESTINATION"],
            ["parameter", "SCALE_STRIDE"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_abi": "grid_constant",
        "launch_grid_rule": {
            "block_m": 128,
            "cta_group": 2,
            "kind": "gemm_cluster_tiles",
            "n_tiles": 84,
        },
        "launch_block": [640, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 231552,
        "use_pdl": False,
        "closure": {
            "sm103a": "473380c36e60d96b6923b3cda0964ff89aed69ec6f449b58f6752028cccd9c02",
        },
    },
    "cake_minimax_h3_nvfp4_pre_attention_283f669f7590d1c4684f": {
        "sources": [
            "cake_minimax_h3_nvfp4_pre_attention_283f669f7590d1c4684f_kernel.cu",
            "cake_minimax_h3_nvfp4_pre_attention_283f669f7590d1c4684f_binding.cu",
        ],
        "stage": "norm_adaln_nvfp4_quantize",
        "template": "minimax_h3_norm_adaln_nvfp4_quantize_v1",
        "ffi_entry": "run",
        "compile_flags": ["--use_fast_math"],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "x_norm_weight"],
            ["buffer", "adaln_scale"],
            ["buffer", "adaln_shift"],
            ["buffer", "adaln_index"],
            ["buffer", "x_global_scale"],
            ["buffer", "activation_q"],
            ["buffer", "activation_sf"],
            ["buffer", "debug_adaln_bf16"],
            ["parameter", "write_debug"],
            ["parameter", "eps"],
            ["parameter", "M"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_abi": "grid_constant",
        "launch_grid_rule": {
            "kind": "norm_rows",
            "row_alignment": 128,
            "rows_per_cta": 2,
        },
        "launch_block": [128, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 128,
        "use_pdl": False,
        "closure": {
            "sm100a": "a0097d6e9764374258fa1dc4b39d0e51b0383cc7b5b0e576e46d15444be3248d",
            "sm103a": "3e14e9b5bd152167667b3d3e3de48445b0e23d2bd34c8cbb769951c8298dfbb4",
        },
    },
    "cake_minimax_h3_nvfp4_pre_attention_bdfc61bd8a6ada56a764": {
        "sources": [
            "cake_minimax_h3_nvfp4_pre_attention_bdfc61bd8a6ada56a764_kernel.cu",
            "cake_minimax_h3_nvfp4_pre_attention_bdfc61bd8a6ada56a764_binding.cu",
        ],
        "stage": "qkv_nvfp4_gemm_fused_pack",
        "template": "minimax_h3_qkv_nvfp4_gemm_fused_pack_epi8_v1",
        "ffi_entry": "run",
        "compile_flags": ["--use_fast_math"],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "alpha"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "rope_cos_sin"],
            ["buffer", "out_global_scale"],
            ["buffer", "out_q"],
            ["buffer", "out_sf"],
            ["tma_buffer", "OUTQ"],
            ["buffer", "qkv_words"],
            ["buffer", "debug_q_words"],
            ["buffer", "debug_k_words"],
            ["parameter", "write_debug"],
            ["parameter", "eps"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "HEADS_PER_DESTINATION"],
            ["parameter", "ROWS_PER_DESTINATION"],
            ["parameter", "SCALE_STRIDE"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_abi": "grid_constant",
        "launch_grid_rule": {
            "block_m": 128,
            "cta_group": 2,
            "kind": "gemm_cluster_tiles",
            "n_tiles": 84,
        },
        "launch_block": [320, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 228480,
        "use_pdl": False,
        "closure": {
            "sm100a": "87d239c4e7efc432c97f27de9607f3754828b261360feebf7de8dedc53d10625",
        },
    },
}
# "<target>:<stage>" -> {"module": program name}.
ROUTES: dict[str, dict[str, Any]] = {
    "sm100a:norm_adaln_nvfp4_quantize": {
        "module": "cake_minimax_h3_nvfp4_pre_attention_283f669f7590d1c4684f",
    },
    "sm100a:qkv_nvfp4_gemm_fused_pack": {
        "module": "cake_minimax_h3_nvfp4_pre_attention_bdfc61bd8a6ada56a764",
    },
    "sm103a:norm_adaln_nvfp4_quantize": {
        "module": "cake_minimax_h3_nvfp4_pre_attention_283f669f7590d1c4684f",
    },
    "sm103a:qkv_nvfp4_gemm_fused_pack": {
        "module": "cake_minimax_h3_nvfp4_pre_attention_10cb85a9569b85023fde",
    },
}


def minimax_h3_nvfp4_target(device: torch.device) -> MiniMaxH3Nvfp4Target:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 NVFP4 pre-attention requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_minimax_h3_nvfp4_pre_attention"
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_minimax_h3_nvfp4_pre_attention"
    )
    for candidate in (installed, checkout):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MiniMax-H3 NVFP4 pre-attention CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _program(target: str, stage: str) -> tuple[str, dict[str, Any]]:
    key = f"{target}:{stage}"
    try:
        name = str(ROUTES[key]["module"])
    except KeyError as exc:
        raise RuntimeError(f"no exact MiniMax-H3 NVFP4 route for {key}") from exc
    return name, MODULES[name]


def minimax_h3_nvfp4_route_record(device: torch.device, P: int) -> dict[str, Any]:
    """``{"P", "target", "stages": {stage: program record + "name"}}`` for one destination partition count.

    ``P`` is validated here because it is part of the caller-facing contract;
    it selects no program (the destination geometry is a launch parameter).
    """
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    target = minimax_h3_nvfp4_target(device)
    stages = {}
    for stage in MINIMAX_H3_NVFP4_STAGES:
        name, record = _program(target, stage)
        stages[stage] = {**record, "name": name}
    return {"P": P, "target": target, "stages": stages}


@functools.cache
def gen_minimax_h3_nvfp4_stage_module(
    target: MiniMaxH3Nvfp4Target, stage: MiniMaxH3Nvfp4Stage
) -> JitSpec:
    """JIT spec of one stage program for one exact target."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported MiniMax-H3 NVFP4 target: {target}")
    if stage not in MINIMAX_H3_NVFP4_STAGES:
        raise ValueError(f"unsupported MiniMax-H3 NVFP4 stage: {stage}")
    _name, record = _program(target, stage)
    csrc = _get_csrc_dir()
    spec = gen_jit_spec(
        name=f"cake_minimax_h3_nvfp4_pre_attention_{target}_{stage}_{record['closure'][target][:20]}",
        sources=[csrc / source for source in record["sources"]],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info(
        "Generated MiniMax-H3 NVFP4 %s %s JIT spec: %s", target, stage, spec.name
    )
    return spec


@functools.cache
def load_minimax_h3_nvfp4_stage_build(
    target: MiniMaxH3Nvfp4Target, stage: MiniMaxH3Nvfp4Stage
):
    return gen_minimax_h3_nvfp4_stage_module(target, stage).build_and_load()


def load_minimax_h3_nvfp4_route(device: torch.device, P: int):
    """``(norm_module, gemm_module)`` of the device's exact target."""
    route = minimax_h3_nvfp4_route_record(device, P)
    target = route["target"]
    return tuple(
        load_minimax_h3_nvfp4_stage_build(target, stage)
        for stage in MINIMAX_H3_NVFP4_STAGES
    )


__all__ = [
    "MINIMAX_H3_NVFP4_STAGES",
    "MODULES",
    "ROUTES",
    "MiniMaxH3Nvfp4Stage",
    "MiniMaxH3Nvfp4Target",
    "gen_minimax_h3_nvfp4_stage_module",
    "load_minimax_h3_nvfp4_route",
    "load_minimax_h3_nvfp4_stage_build",
    "minimax_h3_nvfp4_route_record",
    "minimax_h3_nvfp4_target",
]
