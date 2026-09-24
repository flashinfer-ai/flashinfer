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
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags

MiniMaxH3Nvfp4Stage = Literal[
    "norm_adaln_nvfp4_quantize",
    "qk_rope_destination_nvfp4_pack",
]

_TARGET = "sm100a"
_TARGET_FLAGS = sm100a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm100a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"d87956d803a7eb5acb5aead0a552228506378ba6df3e96964bb2bdb359f5bc15","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_95873899a8517699a855_kernel.cu","sha256":"409b9b5209a7b09e15b5c34d838c2c1d1d88b8ff04c8a46b974db03cdb1f6711"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_95873899a8517699a855_binding.cu","sha256":"153a1a06ca5cd66275456045bc4e7cc46e67ee3e0cf3c3d053673f30b7193f9d"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_95873899a8517699a855","role":"kernel","source_identity":"95873899a8517699a855e0429fa40d5619cba53347798db7720ca3f0470d4d6b","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"f85d32b5d3776d1cd6487b869ff8035dda4f8e7b63bc92ebbf20727740130b05","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_7610cbc186116fdb121d_kernel.cu","sha256":"9341153e8838ab805b8dffb0b5bdceec069c4e0e191185c9d7b13649350673d9"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_7610cbc186116fdb121d_binding.cu","sha256":"0f0d3212f620d6318061647735342d4502439b73830b621b0d1252be9d31c5c6"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_7610cbc186116fdb121d","role":"kernel","source_identity":"7610cbc186116fdb121d1d513ab9b792ed8aec82c2379cc029a7ee8799f77589","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"7584ca7434ef563062497f343fa5abd57e74d2a36da6abcbffa387cbd8ab7a34","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_e4567fb3fbb9937db1c8_kernel.cu","sha256":"7357fb86a5b0fbd1912e754a208ca6f0ae26bcabd81d40fce9383f1d25a01894"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_e4567fb3fbb9937db1c8_binding.cu","sha256":"0069648f0d84a2c47c587687fa96b03d4c66daf8c7ee530b7d42dfc08aa50bd2"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_e4567fb3fbb9937db1c8","role":"kernel","source_identity":"e4567fb3fbb9937db1c8786d8178836638c9db9a8cdfbf57f25fd0bc9a598183","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"0c8099cb232fd3c504f5da3eec4c2470195071c7e2e62dfc4820b2ebc93d496a","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":28672,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_5589e3d02f8e6ea011a3_kernel.cu","sha256":"ea9a3083cbf9c9a0cc3fdefe868e3f0880647ca80383e9870058ca735888a053"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_5589e3d02f8e6ea011a3_binding.cu","sha256":"24de04c29008eba61dea5e75f2d7d5d46392e7354dd405729f3b45e9e8f85256"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_5589e3d02f8e6ea011a3","role":"kernel","source_identity":"5589e3d02f8e6ea011a3a909883aab651085f77ce158751c691465f742aa40b7","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v3","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"}}"""
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
