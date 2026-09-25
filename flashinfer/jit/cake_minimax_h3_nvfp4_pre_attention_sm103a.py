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
    "qkv_nvfp4_gemm_fused_pack",
]

_TARGET = "sm103a"
_TARGET_FLAGS = sm103a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm103a:1":{"P":1,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4622c2a5ec122b2a689210138f3935096b486eb96bcca41b780799d600da763e","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_kernel.cu","sha256":"3eaee223dde2a5757b76f6eff74ca8a357dda216d5245d7fc8bac1b219eb8ab0"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_binding.cu","sha256":"8fc9e286501ae71294be4616ccb06d619d4a71bcd324b112bf72d576dd251943"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9","role":"kernel","source_identity":"a6d128025625bf09a7d9e65ea989e68647663487cbde51404dca5f1a56d26ba0","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"70f0b3e36419275dc539d5f75be429a72b7e19a9678b11a71b37d7c62a83f1e7","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_kernel.cu","sha256":"eeb8f497bc088841605df570cee6d55143cb6d4dcfbe2f0e8b52c2baea7d4507"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_binding.cu","sha256":"ff390b2a0675d9c74d3674f5564fb092f355f14bffe68a2359b5b79c4aa343b3"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a","role":"kernel","source_identity":"21545cf935aa585ef34af3130b0ba0bf119229be384294e053fa780bdcd50347","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_k96_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm103a"},"sm103a:2":{"P":2,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4622c2a5ec122b2a689210138f3935096b486eb96bcca41b780799d600da763e","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_kernel.cu","sha256":"3eaee223dde2a5757b76f6eff74ca8a357dda216d5245d7fc8bac1b219eb8ab0"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_binding.cu","sha256":"8fc9e286501ae71294be4616ccb06d619d4a71bcd324b112bf72d576dd251943"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9","role":"kernel","source_identity":"a6d128025625bf09a7d9e65ea989e68647663487cbde51404dca5f1a56d26ba0","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"70f0b3e36419275dc539d5f75be429a72b7e19a9678b11a71b37d7c62a83f1e7","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_kernel.cu","sha256":"eeb8f497bc088841605df570cee6d55143cb6d4dcfbe2f0e8b52c2baea7d4507"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_binding.cu","sha256":"ff390b2a0675d9c74d3674f5564fb092f355f14bffe68a2359b5b79c4aa343b3"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a","role":"kernel","source_identity":"21545cf935aa585ef34af3130b0ba0bf119229be384294e053fa780bdcd50347","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_k96_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm103a"},"sm103a:4":{"P":4,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4622c2a5ec122b2a689210138f3935096b486eb96bcca41b780799d600da763e","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_kernel.cu","sha256":"3eaee223dde2a5757b76f6eff74ca8a357dda216d5245d7fc8bac1b219eb8ab0"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_binding.cu","sha256":"8fc9e286501ae71294be4616ccb06d619d4a71bcd324b112bf72d576dd251943"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9","role":"kernel","source_identity":"a6d128025625bf09a7d9e65ea989e68647663487cbde51404dca5f1a56d26ba0","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"70f0b3e36419275dc539d5f75be429a72b7e19a9678b11a71b37d7c62a83f1e7","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_kernel.cu","sha256":"eeb8f497bc088841605df570cee6d55143cb6d4dcfbe2f0e8b52c2baea7d4507"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_binding.cu","sha256":"ff390b2a0675d9c74d3674f5564fb092f355f14bffe68a2359b5b79c4aa343b3"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a","role":"kernel","source_identity":"21545cf935aa585ef34af3130b0ba0bf119229be384294e053fa780bdcd50347","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_k96_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm103a"},"sm103a:8":{"P":8,"stages":{"norm_adaln_nvfp4_quantize":{"arg_plan":[["buffer","x"],["buffer","x_norm_weight"],["buffer","adaln_scale"],["buffer","adaln_shift"],["buffer","adaln_index"],["buffer","x_global_scale"],["buffer","activation_q"],["buffer","activation_sf"],["buffer","debug_adaln_bf16"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4622c2a5ec122b2a689210138f3935096b486eb96bcca41b780799d600da763e","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":128,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_kernel.cu","sha256":"3eaee223dde2a5757b76f6eff74ca8a357dda216d5245d7fc8bac1b219eb8ab0"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9_binding.cu","sha256":"8fc9e286501ae71294be4616ccb06d619d4a71bcd324b112bf72d576dd251943"}],"launch_block":[128,1,1],"launch_grid_rule":{"kind":"norm_rows","row_alignment":128,"rows_per_cta":2},"name":"cake_minimax_h3_nvfp4_pre_attention_a6d128025625bf09a7d9","role":"kernel","source_identity":"a6d128025625bf09a7d9e65ea989e68647663487cbde51404dca5f1a56d26ba0","specializations":{},"template":"minimax_h3_norm_adaln_nvfp4_quantize_v1","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"qkv_nvfp4_gemm_fused_pack":{"arg_plan":[["tma_buffer","A"],["tma_buffer","B"],["tma_buffer","SFA"],["tma_buffer","SFB"],["buffer","alpha"],["buffer","q_norm_weight"],["buffer","k_norm_weight"],["buffer","rope_cos_sin"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["tma_buffer","OUTQ"],["buffer","qkv_words"],["buffer","debug_q_words"],["buffer","debug_k_words"],["parameter","write_debug"],["parameter","eps"],["parameter","M"],["parameter","m_tiles"],["parameter","HEADS_PER_DESTINATION"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["workspace","tma_descriptor_workspace"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"70f0b3e36419275dc539d5f75be429a72b7e19a9678b11a71b37d7c62a83f1e7","cluster":[2,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":228480,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_kernel.cu","sha256":"eeb8f497bc088841605df570cee6d55143cb6d4dcfbe2f0e8b52c2baea7d4507"},{"path":"csrc/cake_minimax_h3_nvfp4_pre_attention/sm_103a/cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a_binding.cu","sha256":"ff390b2a0675d9c74d3674f5564fb092f355f14bffe68a2359b5b79c4aa343b3"}],"launch_block":[320,1,1],"launch_grid_rule":{"block_m":128,"cta_group":2,"kind":"gemm_cluster_tiles","n_tiles":84},"name":"cake_minimax_h3_nvfp4_pre_attention_21545cf935aa585ef34a","role":"kernel","source_identity":"21545cf935aa585ef34af3130b0ba0bf119229be384294e053fa780bdcd50347","specializations":{},"template":"minimax_h3_qkv_nvfp4_gemm_fused_pack_k96_v1","tma_abi":"pointer","tma_workspace_bytes":640,"use_pdl":false}},"target":"sm103a"}}"""
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
