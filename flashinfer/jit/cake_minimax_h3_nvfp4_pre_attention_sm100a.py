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
    r"""{"sm100a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"18b886207195049c1ba1c09f6a38db3ace09f3a39f2e01058207b97761a942da","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_2becb19eabe3bd030e00_kernel.cu","sha256":"1f9d58283d34de133064044f50f5468dc85e119e22a3fc71ef64788a2cfe92db"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_2becb19eabe3bd030e00_binding.cu","sha256":"09823e2f16fa165cad3f8bbe84ddb4192ea593f41d0076a9af6e29bb8cca50e4"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_2becb19eabe3bd030e00","role":"kernel","source_identity":"2becb19eabe3bd030e004d10bcc85b5032d5ed7f0444bab1c9c1ec91fca5d2ab","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"dcaaf4239e02684bf6692d86c92f8d4def8b4c8b553d433927c1d546dee32b48","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_40bd7070a5abb11023ed_kernel.cu","sha256":"4e9e7acf61d94b0e0f5939c2239855f2b7f943b9cd2d53f7fd4d7bfcf1c52b01"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_40bd7070a5abb11023ed_binding.cu","sha256":"422835cfc330632c65b2b33e401bbfd169633cf8730c55b5c332e1754ca75ee8"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_40bd7070a5abb11023ed","role":"kernel","source_identity":"40bd7070a5abb11023edcc8f7ef426996838077fb8767656fa87115f847512bc","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"696a4f980282eac90c56df2138ecc6fa7495246ed5b1ce82fe056fa15bda165a","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_43607f08690670a8d7f2_kernel.cu","sha256":"461c379210705ba89ef855c7bcf21d2e365ae363607d39e36935642e05a312d4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_43607f08690670a8d7f2_binding.cu","sha256":"8150746f7009149df48451f92b9b902ddd7f594133efd9c31ee03b6b047fec30"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_43607f08690670a8d7f2","role":"kernel","source_identity":"43607f08690670a8d7f24b374360c566a006dd0f61a7a44c43bbc337ff8fb57d","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"},"sm100a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"8df92615c3a3b759b30bd2555fd78b37466b20ff4480a8d30e38f08eb805c130","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_kernel.cu","sha256":"31e746229b95147b3f95f659ea77372ee7b7dde770c67da6f7753da39af1d6f4"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38_binding.cu","sha256":"3662a9ce555dc68d843b1e119f4a28d2da4151abd4bb9a00102ff74bc00a3922"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_9edde0d17b6d62645e38","role":"kernel","source_identity":"9edde0d17b6d62645e38eb7bdb9be11396cd8df83ef47e9844d3ad720e530b9f","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qk_rope_destination_nvfp4_pack":{"arg_plan":[["buffer","qkv_bf16"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["buffer","debug_q_bf16"],["buffer","debug_k_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4398a6f40caad91b51cc74403999a4b1b8b92cd6710e8441d4c983aa3441e2ad","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_168e22aec086616f7d05_kernel.cu","sha256":"08ed085860056a61abb4afa1993210ade79e980736bdd73e6132067f05182b5a"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_168e22aec086616f7d05_binding.cu","sha256":"3286cb801b97c40ae59bf6924cccf75274e85ea55f6155578ab53a520bcd8261"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"post_warps_2d","rows_per_warp":4,"warps_per_cta":8},"name":"cake_minimax_h3_nvfp4_pre_attention_168e22aec086616f7d05","role":"kernel","source_identity":"168e22aec086616f7d05096ad922129b34ac5e60e8eeaef96163b2c5cc5a990c","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qk_rope_destination_nvfp4_pack_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false}},"target":"sm100a"}}"""
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
