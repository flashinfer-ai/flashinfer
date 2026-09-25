# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-architecture JIT loader for the MiniMax-H3 QKV quantize-and-pack programs."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags

MiniMaxH3QkvPackFormat = Literal["nvfp4", "mxfp8"]

_TARGET = "sm100a"
_TARGET_FLAGS = sm100a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
# Keys are "<target>:<P>:<format>" -> {"P", "format", "target", "module"}; the
# token count M is a runtime launch parameter, so one route per (P, format)
# serves every M.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm100a:1:mxfp8":{"P":1,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"5f922dbf3b1afcc4ff517e50649d3eff2891557f2bc7adba930e439678dceade","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_6e64aabfb6124266c987_kernel.cu","sha256":"0815a72fe0b4d58f5f72892c8a25bc0e12d8e6726f7c5919db128bea05c77aa7"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_6e64aabfb6124266c987_binding.cu","sha256":"48640ac66bbd6dfc713428fc2015076a11734895d06300fb3eb616617613ba09"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_6e64aabfb6124266c987","role":"kernel","source_identity":"6e64aabfb6124266c9871bff2c8ec000e1de4fac4d00fb977d1a272f855bff3a","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:1:nvfp4":{"P":1,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"30cf08f2a276f3da6222cf801160d5e30383c85fbaae7deae3f2addfbda21e59","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_5393eee7f5070c71f08a_kernel.cu","sha256":"99058e1ba5a475bb5579e18c92d99659a8e5ff1850642456e6e016738c503df9"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_5393eee7f5070c71f08a_binding.cu","sha256":"c8e623a0df313955facc7c09514fd58e314985ac6c42aab24908cf06bb7119fd"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_5393eee7f5070c71f08a","role":"kernel","source_identity":"5393eee7f5070c71f08a45ba94b3f0ec44ff0bb2375c02b23419f73847113157","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:2:mxfp8":{"P":2,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"de2e77f302e8b4a4058c75d9cfdeef4bf03799a6d556ce16772b0cd77c078d0b","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_d914c0bd7c72f17566cf_kernel.cu","sha256":"9ecc978dd95f9dccbcb613da3c10f4e6c93c309ad71e83116d2fee12a26621b8"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_d914c0bd7c72f17566cf_binding.cu","sha256":"30bec17a58244282f4819543881372f3adc0d90a4d3f61440358587de7e1fd54"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_d914c0bd7c72f17566cf","role":"kernel","source_identity":"d914c0bd7c72f17566cf3b1ef13c4606c7c0dc0f8adb49d9f9b54c95b86ffb2e","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:2:nvfp4":{"P":2,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"6105060c8949deab0f84e0419c0afb643af9e1705227e94031e7da9a9c144009","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_cd5888c3e6b8846a31c7_kernel.cu","sha256":"9c6934bd20651bab30131be154822dbfb78d906940904fd4ef816a8d192ce1c2"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_cd5888c3e6b8846a31c7_binding.cu","sha256":"09b6ef3f93575f2953427dda87027da8b7031633876fede7fd0f276fbd5f501a"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_cd5888c3e6b8846a31c7","role":"kernel","source_identity":"cd5888c3e6b8846a31c7dcb464bdd4d0998d784614630e8123f607fa3eae5038","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:4:mxfp8":{"P":4,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"e4f79d862066f9225eb2bc6224eb0da0475ffcdd289f3b76de34fa7e72e382fc","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_7376770cf95753ebc031_kernel.cu","sha256":"e71d5ecbf4aa5136e2b0e8aaf0c3ed1894758cc77e4605557db49ed6b871b1fb"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_7376770cf95753ebc031_binding.cu","sha256":"6f6204eb49219efeef8b3883aacdfe8ebc0ec0ca522a33feccbfd9917f8e2fe6"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_7376770cf95753ebc031","role":"kernel","source_identity":"7376770cf95753ebc0316c6982950490d79249fc22f7b6c8f8a2795a4d09550c","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:4:nvfp4":{"P":4,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"662f9191282f6be9a5b8b9ffb69694fdedb58e541727f108c588c6b3ab87c58b","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_6d6803218a213663eade_kernel.cu","sha256":"01178edf66a6d04cf2301072652417162ff5e729e28d953fe24cf148f6ab34be"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_6d6803218a213663eade_binding.cu","sha256":"5e70afeee2fe7fd04ceb32aed0c6de706a4fe7a5353f9a2bd40f69f605fdb299"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_6d6803218a213663eade","role":"kernel","source_identity":"6d6803218a213663eade55718ffd8c36f4c6c240c7a98b35bd9ae47cc9aac78f","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:8:mxfp8":{"P":8,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"21d2b88a5b8d3e8a0953bcac120480e0a56ed948acd4451e27783b0b9b87e109","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_3e6cf43ecb7b415ff17c_kernel.cu","sha256":"b647dcb1e7f24ef1ebb166b66f99a4a644489f41cc3d7ff219dc4598c28fd5dd"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_3e6cf43ecb7b415ff17c_binding.cu","sha256":"ac4b98a5c1a55346bba1a4431490f103dfd15b02e72b81549104963e2719f497"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_3e6cf43ecb7b415ff17c","role":"kernel","source_identity":"3e6cf43ecb7b415ff17cec4dd611122ceab2ba9b51328ce5384e16faed79d1e1","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"},"sm100a:8:nvfp4":{"P":8,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"af5bc9be4400cd91c7692e7f409fcf1c0663b707a46c2be23efd41e82fb0e864","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_a540a698e46f8c4429ad_kernel.cu","sha256":"30fd5b177b0d75f99c35cae11ca185b1b4c5f0cfdc58b5afe1e9bd8a74afdf15"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_100a/cake_minimax_h3_qkv_quantize_pack_a540a698e46f8c4429ad_binding.cu","sha256":"d84db76171320ceb392ea2d820622c2b428378d1390cd683143e59619741660b"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_a540a698e46f8c4429ad","role":"kernel","source_identity":"a540a698e46f8c4429ad80353849add619debd53abb635845db66446e83df369","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm100a"}}"""
)


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_minimax_h3_qkv_quantize_pack"
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_minimax_h3_qkv_quantize_pack"
    )
    for candidate in (installed, checkout):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MiniMax-H3 QKV quantize-and-pack CUDA sources were not found. Checked:\n"
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
        or relative.parts[:2] != ("csrc", "cake_minimax_h3_qkv_quantize_pack")
        or ".." in relative.parts
    ):
        raise RuntimeError(f"invalid MiniMax-H3 QKV pack source path: {relative}")
    path = _get_csrc_dir().joinpath(*relative.parts[2:])
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 QKV pack source was not installed: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"MiniMax-H3 QKV pack source identity mismatch: {path}")
    return path


def minimax_h3_qkv_pack_route_record(P: int, fmt: MiniMaxH3QkvPackFormat) -> dict:
    """Route record for one destination partition count P and output format.

    The generated program takes the token count M (and the derived
    ROWS_PER_DESTINATION / SCALE_STRIDE) as runtime parameters, so one
    route per (P, format) serves every M.
    """
    key = f"{_TARGET}:{P}:{fmt}"
    try:
        return _ROUTES[key]
    except KeyError as exc:
        raise RuntimeError(
            f"no exact MiniMax-H3 QKV quantize-and-pack route for {key}"
        ) from exc


@functools.cache
def gen_minimax_h3_qkv_pack_module(P: int, fmt: MiniMaxH3QkvPackFormat) -> JitSpec:
    route = minimax_h3_qkv_pack_route_record(P, fmt)
    record = route["module"]
    sources = [_verified_source(item) for item in record["files"]]
    uri = (
        f"cake_minimax_h3_qkv_quantize_pack_{_TARGET}_{fmt}_p{P}_"
        f"{record['closure_sha256']}"
    )
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
    logger.info(
        "Generated MiniMax-H3 QKV quantize-and-pack %s P=%d JIT spec: %s",
        fmt,
        P,
        spec.name,
    )
    return spec


@functools.cache
def load_minimax_h3_qkv_pack_module(P: int, fmt: MiniMaxH3QkvPackFormat):
    return gen_minimax_h3_qkv_pack_module(P, fmt).build_and_load()


__all__ = [
    "MiniMaxH3QkvPackFormat",
    "gen_minimax_h3_qkv_pack_module",
    "load_minimax_h3_qkv_pack_module",
    "minimax_h3_qkv_pack_route_record",
]
