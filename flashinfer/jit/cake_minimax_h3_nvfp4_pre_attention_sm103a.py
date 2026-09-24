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
    r"""{"sm103a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4e831be50c1792b62fbc96109db2cfa7ad93b967535e0a10ae46128d6c0cfa1a","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a0bedef3295e5b252d45_kernel.cu","sha256":"f37ac20ac52ae4bfc54b0ce253cb0686a08e2c80e6cfd4f0622ba452b3fbc6aa"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a0bedef3295e5b252d45_binding.cu","sha256":"8bb3379eabf25ac12515284403fc58525962f7786ac6236cdc7df4ef40207b3a"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_a0bedef3295e5b252d45","role":"kernel","source_identity":"a0bedef3295e5b252d450392fff7bd6bfbeaed4fc579c1ee1786763420a60fa7","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"25b610d1a61e28e78b65d8968f1ce0cbe9859b73ca57e163e8f2cb12688a77fa","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_4e8ea93a995c2b3bbef4_kernel.cu","sha256":"b01394d64b5e5e89d66d71ae306bfac5b4404593cedf9e08a54e8dad421ad1a1"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_4e8ea93a995c2b3bbef4_binding.cu","sha256":"961ae963d3031f538c08e6f3300c7ce40d610504825baf70589e474f2ebf4264"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_4e8ea93a995c2b3bbef4","role":"kernel","source_identity":"4e8ea93a995c2b3bbef4955eae4d2da7fc88f7f3cba5e843662452717b2e460d","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"9590017fc0de148c69cc160a035f2905a4c899590171ead572089c9ef3205e2d","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_33cb5bc5017dd2ad9d66_kernel.cu","sha256":"f4a9fd002dca80d9c0ec58b9d9f352f4aa6905dfb502a8fc503070863bc66498"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_33cb5bc5017dd2ad9d66_binding.cu","sha256":"04c2f38e1112d9b55448e4bb4d99208d07b362cbc9af50c58769f81daf1abd6d"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_33cb5bc5017dd2ad9d66","role":"kernel","source_identity":"33cb5bc5017dd2ad9d66a38dc09d74fe45c07672c336face8aa18002d5173d98","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"},"sm103a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"db4d7c02db4865799b68df49ca815e06a87e9a84cab354bc9e7932c178ffbaba","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_kernel.cu","sha256":"0fec7c10bda697f5c1a7da3a90b5d50db6a69943062baa261389a5d3cb86ba05"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139_binding.cu","sha256":"7857feef50ae2eb322405ed596edd841fb3ae5ee5f7d88d9d6d4653031ffd0f0"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_1057bb7ff47bd0786139","role":"kernel","source_identity":"1057bb7ff47bd07861398f5b7721827d182ce49756a99892b6030cc106d03d7d","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"7bd4e62f3f99476ee5b81146803ef3b7823beeeb1da4dee52b90547be2f616dc","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_341330a98b7380153c4b_kernel.cu","sha256":"6f73d0eb7aad51de9fc5f624b0fca4e34b931203d952a8e4c7dfe0fa7b56dd32"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_341330a98b7380153c4b_binding.cu","sha256":"6a1e8f46478acfaa3b2b68bb8aa4a50247d073da85f54940c834c59fa63296fd"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_341330a98b7380153c4b","role":"kernel","source_identity":"341330a98b7380153c4bf44669bec0a5a96c3b1f707fc83af580aebf367385e8","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm103a"}}"""
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
