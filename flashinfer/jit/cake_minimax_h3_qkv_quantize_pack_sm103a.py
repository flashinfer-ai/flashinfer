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
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

MiniMaxH3QkvPackFormat = Literal["nvfp4", "mxfp8"]

_TARGET = "sm103a"
_TARGET_FLAGS = sm103a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
# Keys are "<target>:<P>:<format>" -> {"P", "format", "target", "module"}; the
# token count M is a runtime launch parameter, so one route per (P, format)
# serves every M.
_ROUTES: dict[str, dict[str, Any]] = json.loads(
    r"""{"sm103a:1:mxfp8":{"P":1,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"d727a89f9e50968bcf64de5f0c089af21a9b1f6e1638219b163c1a4ae62e5f22","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_34d8570c909ad390e410_kernel.cu","sha256":"42da4bedd8d07ff69d20e3340a84b7df5b50c7aa22154d04977465d787be3273"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_34d8570c909ad390e410_binding.cu","sha256":"5c84716861570c8e755d721e5d9b3be364e7d20866398a3ac43da5392cfd6937"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_34d8570c909ad390e410","role":"kernel","source_identity":"34d8570c909ad390e410e10a1faefb3c0165deaa41526acde0144021a2ad32d0","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:1:nvfp4":{"P":1,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"228eeee71e85cdf8cf76c64e2265499d1f8b1d40021bcccf0bc94b555b9f378c","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_3c509bb0f57d9dffea18_kernel.cu","sha256":"e619ec0228bb85cc3ee19953b4757b9fd9a487d2995a9b3dee6d97de6500408c"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_3c509bb0f57d9dffea18_binding.cu","sha256":"fd2bcf6dfa8f354860f1b7c9c05faeb19d4701c51cab5fdaeef4c993e476aed4"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_3c509bb0f57d9dffea18","role":"kernel","source_identity":"3c509bb0f57d9dffea185c675d3c23dce432c33084f8f5f1db7cc8766eeb02b0","specializations":{"HEADS_PER_DESTINATION":56,"P":1},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:2:mxfp8":{"P":2,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"f819ee092bea7222a98c296947a6c0834aa7697ef5ac18f2b8be579d3f030109","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_684d11eac4ab0bf47b30_kernel.cu","sha256":"fe54de11aef26ddcdcfc101c0a7efefdcc784751d3c27436eaf74583e627fe4e"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_684d11eac4ab0bf47b30_binding.cu","sha256":"7472de6127a1f89d8316fb93da13852e139b43b027667eddef5e09209601c271"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_684d11eac4ab0bf47b30","role":"kernel","source_identity":"684d11eac4ab0bf47b3082fac602ffb88ece41f05355907aa424857d769f48b8","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:2:nvfp4":{"P":2,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"ef326ffd7acbc8c0dade422e8d5de6d999d7d39856997d88e29381ed7c358188","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_dbb7afe5bb6cac82fb51_kernel.cu","sha256":"a5d064763c3a91d3d3fb300999f03ee3e329837cf797e50a7836ed29053c0f90"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_dbb7afe5bb6cac82fb51_binding.cu","sha256":"bc55a97921dc7d2004b4315e6d6c02266f4e22d1227b5ed80fd22015019870cc"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_dbb7afe5bb6cac82fb51","role":"kernel","source_identity":"dbb7afe5bb6cac82fb51477e2fb2007ed79e8537390cee401d09edec30d98029","specializations":{"HEADS_PER_DESTINATION":28,"P":2},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:4:mxfp8":{"P":4,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"14b2ca141ab7fbf443bb366594455b4b5408fa2da208a49deb57916fc50e3f0b","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_02bef338903ea7711eaa_kernel.cu","sha256":"e5a3d17dbc9f81b3ff9844c07e2f1ec5273a7364130341c8b4797917f6c026b8"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_02bef338903ea7711eaa_binding.cu","sha256":"00e4244a3970d2b9ea5637b595d5c457af919bb7bd106c80386205c49dd96a75"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_02bef338903ea7711eaa","role":"kernel","source_identity":"02bef338903ea7711eaa5cfa588e2280d099fe0b8ba4f0fb17a2e4a5ceb931e4","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:4:nvfp4":{"P":4,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"e4a163c9ee3c2a2037137a40f461adb525a51740bd2ea06b3bcd05ec53b8741b","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_d7420bf8258dc1739675_kernel.cu","sha256":"39c31cd42d07f6da98b4f0e65331df9e3fe2fd9e81cd6e7b7ef6f5f980ffc9da"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_d7420bf8258dc1739675_binding.cu","sha256":"de3d32ef72e4d344149b604045cf3384cd3e4b2e34543c0ea4764d5b9df060d1"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_d7420bf8258dc1739675","role":"kernel","source_identity":"d7420bf8258dc17396753c826c91dc3a5c825efddab04e9ded6d9ff095fd9b07","specializations":{"HEADS_PER_DESTINATION":14,"P":4},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:8:mxfp8":{"P":8,"format":"mxfp8","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"a44bb99663eaa72a1b07a2a85889e658499b1665e5a5b612c663ccaf8230bfc9","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_43f6f13ba521ef87c81b_kernel.cu","sha256":"5f015760e58f7ba13dd930c8cda4e1a9b68f8f3781106f80d6c79324b8e7052d"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_43f6f13ba521ef87c81b_binding.cu","sha256":"6641ad2412995934c54a2f0be4955d39f8067a58c5b98eab6eed45e6f1ea7710"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_43f6f13ba521ef87c81b","role":"kernel","source_identity":"43f6f13ba521ef87c81be98db3084ab7ba2f4a26ddccba29de89fff573c36e48","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qkv_pack_mxfp8","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"},"sm103a:8:nvfp4":{"P":8,"format":"nvfp4","module":{"arg_plan":[["buffer","q"],["buffer","k"],["buffer","v"],["buffer","out_global_scale"],["buffer","out_q"],["buffer","out_sf"],["parameter","M"],["parameter","token_stride"],["parameter","head_stride"],["parameter","ROWS_PER_DESTINATION"],["parameter","SCALE_STRIDE"],["grid","grid_x"],["grid","grid_y"],["grid","grid_z"]],"closure_sha256":"4c7e1152ff98bfa7472b7645728e5d55719dc8b1d9f3a30a161daf3094a9eb09","cluster":[1,1,1],"compile_flags":["--use_fast_math"],"dynamic_smem_bytes":0,"ffi_entry":"run","files":[{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_76f1c062fa82bda9c60e_kernel.cu","sha256":"b81e1a0c53dbe19c2bd42b623880fcacb2d93cfb6530def469274c5095c0b750"},{"path":"csrc/cake_minimax_h3_qkv_quantize_pack/sm_103a/cake_minimax_h3_qkv_quantize_pack_76f1c062fa82bda9c60e_binding.cu","sha256":"5c998de780b9f4ef492b562cc059cd0da75a0542b1257329dd6e976c74b4725f"}],"launch_block":[256,1,1],"launch_grid_rule":{"kind":"pack_warps_2d_plus_padding","tokens_per_warp":16,"warps_per_cta":8},"name":"cake_minimax_h3_qkv_quantize_pack_76f1c062fa82bda9c60e","role":"kernel","source_identity":"76f1c062fa82bda9c60e3ceb23a71543c21b2fc5c08a742784289de188513294","specializations":{"HEADS_PER_DESTINATION":7,"P":8},"template":"minimax_h3_qkv_pack_nvfp4","tma_abi":"pointer","tma_workspace_bytes":0,"use_pdl":false},"target":"sm103a"}}"""
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
