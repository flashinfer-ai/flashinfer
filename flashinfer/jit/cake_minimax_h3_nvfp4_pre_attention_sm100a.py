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
    "qkv_nvfp4_gemm_fused_pack",
]

_TARGET = "sm100a"
_TARGET_FLAGS = sm100a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm100a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"9c870bfd4e0ef241a8800ca86dd181c20684d66d29262799d388cc58deb354c3","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_kernel.cu","sha256":"1d576cbe25a520804e635eb10d7b242cfa8ecc8d7cfb3cfefa4c7cbb48b93b7f"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_binding.cu","sha256":"ef0c1209ffa85994782f1caa2f2c13f0bcd641bec6edaaf78e65e551220a50c3"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5","role":"kernel","source_identity":"4d04d24e494b533867e5ba8cdca529b6c55f7b8f433840d1d7dbb9cecd09d3fc","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"ee2b3ec19ddcc3297bbf5494b8d0ad13acf8de79ff5b50562d3b69c0400ce1f8","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_kernel.cu","sha256":"ffd10adaff2fb172fbe2d55b43cfb95241158ad283dd1a0e0df3c8773e80bb48"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_binding.cu","sha256":"92cd65486ac83e80a58409f23388bf9285531e753ae6b35dd1042528ce59d164"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c","role":"kernel","source_identity":"8df3ceffffb3a9957c1c1c555336cc0c574668d3cf8fee78d257a022800c6c40","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_epi8_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm100a"},"sm100a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"9c870bfd4e0ef241a8800ca86dd181c20684d66d29262799d388cc58deb354c3","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_kernel.cu","sha256":"1d576cbe25a520804e635eb10d7b242cfa8ecc8d7cfb3cfefa4c7cbb48b93b7f"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_binding.cu","sha256":"ef0c1209ffa85994782f1caa2f2c13f0bcd641bec6edaaf78e65e551220a50c3"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5","role":"kernel","source_identity":"4d04d24e494b533867e5ba8cdca529b6c55f7b8f433840d1d7dbb9cecd09d3fc","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"ee2b3ec19ddcc3297bbf5494b8d0ad13acf8de79ff5b50562d3b69c0400ce1f8","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_kernel.cu","sha256":"ffd10adaff2fb172fbe2d55b43cfb95241158ad283dd1a0e0df3c8773e80bb48"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_binding.cu","sha256":"92cd65486ac83e80a58409f23388bf9285531e753ae6b35dd1042528ce59d164"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c","role":"kernel","source_identity":"8df3ceffffb3a9957c1c1c555336cc0c574668d3cf8fee78d257a022800c6c40","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_epi8_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm100a"},"sm100a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"9c870bfd4e0ef241a8800ca86dd181c20684d66d29262799d388cc58deb354c3","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_kernel.cu","sha256":"1d576cbe25a520804e635eb10d7b242cfa8ecc8d7cfb3cfefa4c7cbb48b93b7f"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_binding.cu","sha256":"ef0c1209ffa85994782f1caa2f2c13f0bcd641bec6edaaf78e65e551220a50c3"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5","role":"kernel","source_identity":"4d04d24e494b533867e5ba8cdca529b6c55f7b8f433840d1d7dbb9cecd09d3fc","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"ee2b3ec19ddcc3297bbf5494b8d0ad13acf8de79ff5b50562d3b69c0400ce1f8","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_kernel.cu","sha256":"ffd10adaff2fb172fbe2d55b43cfb95241158ad283dd1a0e0df3c8773e80bb48"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_binding.cu","sha256":"92cd65486ac83e80a58409f23388bf9285531e753ae6b35dd1042528ce59d164"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c","role":"kernel","source_identity":"8df3ceffffb3a9957c1c1c555336cc0c574668d3cf8fee78d257a022800c6c40","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_epi8_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm100a"},"sm100a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"9c870bfd4e0ef241a8800ca86dd181c20684d66d29262799d388cc58deb354c3","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_kernel.cu","sha256":"1d576cbe25a520804e635eb10d7b242cfa8ecc8d7cfb3cfefa4c7cbb48b93b7f"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5_binding.cu","sha256":"ef0c1209ffa85994782f1caa2f2c13f0bcd641bec6edaaf78e65e551220a50c3"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_4d04d24e494b533867e5","role":"kernel","source_identity":"4d04d24e494b533867e5ba8cdca529b6c55f7b8f433840d1d7dbb9cecd09d3fc","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"ee2b3ec19ddcc3297bbf5494b8d0ad13acf8de79ff5b50562d3b69c0400ce1f8","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_kernel.cu","sha256":"ffd10adaff2fb172fbe2d55b43cfb95241158ad283dd1a0e0df3c8773e80bb48"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_100a/cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c_binding.cu","sha256":"92cd65486ac83e80a58409f23388bf9285531e753ae6b35dd1042528ce59d164"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_8df3ceffffb3a9957c1c","role":"kernel","source_identity":"8df3ceffffb3a9957c1c1c555336cc0c574668d3cf8fee78d257a022800c6c40","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_epi8_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm100a"}}"""
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

    Both generated stages take the token count ``M`` as a runtime parameter
    and the fused GEMM takes the destination geometry at launch, so every
    route of one target names the same two programs; the table keeps one
    record per ``P`` because ``P`` is part of the caller-facing contract.
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
