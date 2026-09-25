# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-architecture JIT loader for MiniMax-H3 NVFP4 CUDA stages."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

MiniMaxH3Nvfp4Stage = Literal[
    "norm_adaln_nvfp4_quantize",
    "qk_rope_destination_nvfp4_pack",
]

_TARGET = "sm103a"
_TARGET_FLAGS = sm103a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm103a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"66b8cf1d85be760ff2d9a11c6bff17f66790e80df57759be38330c19efa29ec7","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_48cb6c9930b9457abf81_kernel.cu","sha256":"96ad0f2eafbacfac3c3c3b49334cb665068bdb5969499dceb15d9a0ba29c58cb"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_48cb6c9930b9457abf81_binding.cu","sha256":"f0417fb6ae5a80e5d1d0cad8ef6cb440dab28ff5a0e4f9f25426b12bcef04ef7"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_48cb6c9930b9457abf81","role":"kernel","source_identity":"48cb6c9930b9457abf814fff6f8dc081be149639c6e478d301258c56f3e35416","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"6ff52ec419fd303ceab99e239474d7211acbb5998282bde0f2f21423efe1fd73","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_81abed0f0e98f0eb7eb2_kernel.cu","sha256":"f5f892b09e0761cb829404b965e52b324cacc06edf2689197fe3ff77b9e3039f"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_81abed0f0e98f0eb7eb2_binding.cu","sha256":"e66082900fec0e3aa23de60d6f8913c10b7cccc2a05caa69f40ea971057b977a"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_81abed0f0e98f0eb7eb2","role":"kernel","source_identity":"81abed0f0e98f0eb7eb28e572790d08f12fccc20f913dbd08f7487bf28c5b8ab","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"3589fbc50e4c8b2d68524aeb4234298ca4b7fe0778394ddca3da91ff1d987ce4","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_04e768b12389064a0d67_kernel.cu","sha256":"957a31c395ae4694406bb0e338f9d72c41b49549360175fe1e244dd43eae3016"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_04e768b12389064a0d67_binding.cu","sha256":"75dac0bd0736378f052606342382e9c0ac918a908bd95312fd8b1108f89ef68c"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_04e768b12389064a0d67","role":"kernel","source_identity":"04e768b12389064a0d67a1db51759628047df0c809e6a33254d29108a3367db1","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8aaa4ea3fab16d76f68c5f52dc80858f0429a3d44128c9a3e506420ccdd86493","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_d4210d4dddb5d2ae6134_kernel.cu","sha256":"e0ccd90ab91960dbb81b42bc3d36c6087be022fbd68eadd8463de492736007e9"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_d4210d4dddb5d2ae6134_binding.cu","sha256":"740faefbc5dd5240db870313ad199ec49e57e6ea1e8ec638d8fb364904ae2932"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_d4210d4dddb5d2ae6134","role":"kernel","source_identity":"d4210d4dddb5d2ae61341357d001049d1dc7b0f0daa1d1961c5283f8723b4756","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"}}"""
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
        "MiniMax-H3 CUDA sources were not found. Checked:\n"
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


def _verified_source(record: dict[str, str]) -> Path:
    relative = PurePosixPath(record["path"])
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("csrc", "cake_minimax_h3_nvfp4_pre_attention")
        or ".." in relative.parts
    ):
        raise RuntimeError(f"invalid MiniMax-H3 source path: {relative}")
    path = _get_csrc_dir().joinpath(*relative.parts[2:])
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 source was not installed: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"MiniMax-H3 source identity mismatch: {path}")
    return path


def minimax_h3_nvfp4_route_record(P: int) -> dict:
    """Route record for one destination partition count ``P`` (1, 2, 4, 8).

    Both generated stages take the token count ``M`` as a runtime parameter,
    so one route per ``P`` serves every ``M``.
    """
    key = f"{_TARGET}:{P}"
    try:
        return _ROUTES[key]
    except KeyError as exc:
        raise RuntimeError(f"no exact MiniMax-H3 NVFP4 route for {key}") from exc


@functools.cache
def gen_minimax_h3_nvfp4_stage_module(
    P: int,
    stage: MiniMaxH3Nvfp4Stage,
) -> JitSpec:
    route = minimax_h3_nvfp4_route_record(P)
    try:
        record = route["stages"][stage]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax-H3 NVFP4 stage: {stage}") from exc
    sources = [_verified_source(item) for item in record["files"]]
    uri = f"cake_minimax_h3_nvfp4_{_TARGET}_{stage}_{record['closure_sha256']}"
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=[*_TARGET_FLAGS, *record["compile_flags"]],
        extra_include_paths=[
            _get_csrc_dir(),
            _get_csrc_dir().parent,
            _get_include_dir(),
        ],
        needs_device_linking=True,
    )
    logger.info("Generated MiniMax-H3 NVFP4 %s JIT spec: %s", stage, spec.name)
    return spec


@functools.cache
def load_minimax_h3_nvfp4_stage_module(
    P: int,
    stage: MiniMaxH3Nvfp4Stage,
):
    return gen_minimax_h3_nvfp4_stage_module(P, stage).build_and_load()


__all__ = [
    "MiniMaxH3Nvfp4Stage",
    "gen_minimax_h3_nvfp4_stage_module",
    "load_minimax_h3_nvfp4_stage_module",
    "minimax_h3_nvfp4_route_record",
]
