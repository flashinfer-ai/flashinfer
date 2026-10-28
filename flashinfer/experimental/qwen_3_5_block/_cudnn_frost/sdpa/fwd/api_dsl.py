# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.

"""cuDNN-frontend adapter over the Frost DSL SDPA prefill kernels."""

from __future__ import annotations

import inspect
import logging
import math
import os
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Hashable, Iterator, Optional

import torch
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import stream_context
from cuda.bindings import driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import APIBase, TensorDesc, TupleDict
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._device import ensure_current_context as _ensure_current_context
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.template_loader import load_template
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_O_MXFP8,
    DTYPE_O_NVFP4,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm90 import (
    BAND_COORDINATE_LIMIT as _SM90_BAND_COORDINATE_LIMIT,
    SCALE_NEGATIVE as _SM90_SCALE_NEGATIVE,
    SCALE_POSITIVE as _SM90_SCALE_POSITIVE,
    SCALE_ZERO as _SM90_SCALE_ZERO,
    TILE_M as _SM90_TILE_M,
    TILE_N as _SM90_TILE_N,
    TemplateParams as Sm90TemplateParams,
    head_dims_mismatch as _sm90_head_dims_mismatch,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES as _SM107_F16_THD_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import supports_native_d128_dense_partial_pack, supports_native_d128_narrow_tile, supports_native_d512_dense_pack_split
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_F16_SMALL_D_SHAPES as _SM107_F16_SMALL_D_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_SMALL_D_CGAS as _SM107_SMALL_D_CGAS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_small_tile_leg as _sm107_small_tile_leg
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_FP8_THD_SHAPES as _SM107_FP8_THD_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_FP8_SPLIT_SHAPES as _SM107_FP8_SPLIT_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import supports_fp8_d256_padded_split
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_FP8_PACK_GQA_SHAPES as _SM107_FP8_PACK_GQA_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_MXFP8_THD_SHAPES as _SM107_MXFP8_THD_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES as _SM107_EPILOGUE_GATE_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_SCORE_STATS_SHAPES as _SM107_SCORE_STATS_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_BIAS_D_SHAPES as _SM107_BIAS_D_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import SM100_BIAS_D_SHAPES as _SM100_BIAS_D_SHAPES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_bias_auto_cga as _sm100_bias_auto_cga
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_bias_mode_pick as _sm100_bias_mode_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_bias_resident_s_kv_max as _sm100_bias_resident_s_kv_max
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_bias_stages_pick as _sm100_bias_stages_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import SM100_TILE_NS_BY_D_SHAPE as _SM100_TILE_NS_BY_D_SHAPE, sm100_tile_n_pick as _sm100_tile_n_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import SM100_FP32_VECTOR_WITH_BIAS_TILE_N as _SM100_FP32_VECTOR_WITH_BIAS_TILE_N
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_padded_tail_form_pick as _sm100_padded_tail_form_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_dead_subtile_quiet_pick as _sm100_dead_subtile_quiet_pick

# The SM100 line's device capabilities that serve the small tiles' bias / kv_mask arms (the sm100 row's sm 100..106 band: cc 10.0 and
# 10.3); cc 10.7 is the Rubin row's.
_SM100_BIAS_CCS = ((10, 0), (10, 3))
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_resident_s_kv_max as _sm107_bias_resident_s_kv_max
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_auto_cga as _sm107_bias_auto_cga
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_TILE_NS_BY_D_SHAPE as _SM107_TILE_NS_BY_D_SHAPE
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SM107_FP32_VECTOR_WITH_BIAS_TILE_N as _SM107_FP32_VECTOR_WITH_BIAS_TILE_N
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import KV_MASK_ADDITIVE_GRANULE_ELEMS as _KV_MASK_ADDITIVE_GRANULE_ELEMS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import KV_MASK_BOOL_WORD as _KV_MASK_BOOL_WORD
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import KV_MASK_ADDITIVE_16 as _KV_MASK_ADDITIVE_16
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import KV_MASK_ADDITIVE_F32 as _KV_MASK_ADDITIVE_F32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_pair_form as _sm107_bias_pair_form
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_n_chunk_pick as _sm107_n_chunk_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import BIAS_MODE_NONE as _BIAS_MODE_NONE, BIAS_MODE_RESIDENT as _BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED as _BIAS_MODE_STREAMED
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import BIAS_MODE_STREAMED_MULTICAST as _BIAS_MODE_STREAMED_MULTICAST, BIAS_STREAMED_MODES as _BIAS_STREAMED_MODES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import BIAS_RING_STAGES as _BIAS_RING_STAGES
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_mode_pick as _sm107_bias_mode_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_stages_pick as _sm107_bias_stages_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_padded_tail_form_pick as _sm107_padded_tail_form_pick, PADDED_TAIL_FORMS as _PADDED_TAIL_FORMS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_dead_subtile_quiet_pick as _sm107_dead_subtile_quiet_pick, DEAD_SUBTILE_QUIET_FORMS as _DEAD_SUBTILE_QUIET_FORMS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import _tile_m_d128 as _sm107_tile_m_d128
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import BIAS_MAX_HW_UPPER_BOUND as _BIAS_MAX_HW_UPPER_BOUND, BIAS_MAX_FORMS as _BIAS_MAX_FORMS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_bias_max_form_pick as _sm107_bias_max_form_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_softmax_wgs_pick as _sm107_softmax_wgs_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import SMALL_D_SOFTMAX_WGS as _SMALL_D_SOFTMAX_WGS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_s_slots_pick as _sm107_s_slots_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_q_in_tmem_pick as _sm107_q_in_tmem_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_pair_item_active as _sm107_pair_item_active
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_tile_n_pick as _sm107_tile_n_pick
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import epilogue_gate_layout_declarable as _epilogue_gate_layout_declarable
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import decode_d256_q_tile as _decode_d256_q_tile_sm107
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import SM100_DENSE_ONLY_D_SHAPES as _SM100_DENSE_ONLY_D_SHAPES, SM100_SMALL_D_CGAS as _SM100_SMALL_D_CGAS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import (
    d256_mxfp8_pair_leg,
    FP8_DUAL_KV_STREAM_MAX_ROWS,
    fp8_dual_kv_stream_routed,
    supports_thd_split,
    supports_scalar_kv_tail_split,
    supports_quantized_split_sink,
    supports_thd_split_sink,
    supports_paged_prefill_cga1,
    supports_paged_d256_pack_gqa,
    supports_nonpaged_d128_thd_prefill,
    supports_paged_d128_partial_pack_gqa,
    supports_nonpaged_d128_thd_pack_gqa,
    _PAGED_KV_FLAVORS as _SM100_PAGED_KV_FLAVORS,
    TemplateParams as Sm100TemplateParams,
    SM100_THD_PACK_GQA_SHAPES,
    SM100_NATIVE_D576_SHAPE,
    SM100_PAGED_FP8_THD_SHAPES,
    SM100_SCORE_STATS_SHAPES,
    CfgD128,
    canonicalize_d192_lowering,
    canonicalize_d256_lowering,
    canonicalize_d512_mxfp8_lowering,
    d256_wide_decode_launch,
    decode_d256_q_tile,
    derive_d192_internal_params,
    derive_d256_internal_params,
    pack_gqa_group_size,
    pack_gqa_supported,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import (
    HEAD_TILE_GRANULE as _SM120_HEAD_TILE_GRANULE,
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    GENERAL_HEAD_TILE_MAX as _SM120_GENERAL_HEAD_TILE_MAX,
    FP8_GENERAL_HEAD_TILE_MAX as _SM120_FP8_GENERAL_HEAD_TILE_MAX,
    FP8_HEAD_TILE_GRANULE as _SM120_FP8_HEAD_TILE_GRANULE,
    TemplateParams as Sm120TemplateParams,
    D256_FLAVOR as _SM120_D256_FLAVOR,
    D512_FLAVOR as _SM120_D512_FLAVOR,
    DECODE_Q_TILES as _SM120_DECODE_Q_TILES,
    decode_kv_stage as _sm120_decode_kv_stage,
    head_tiles as _sm120_head_tiles,
    is_decode_tile as _sm120_is_decode_tile,
    kv_tile_preference as _sm120_kv_tile_preference,
    pick_flavor as _sm120_pick_flavor,
    smem_bytes as _sm120_smem_bytes,
    tile_domain as _sm120_tile_domain,
)


@lru_cache(maxsize=256)
def _parameter_names(entry) -> frozenset:
    """Plan-time names shared across plans of the same adapter/kernel function.

    Key the function itself so a replaced entry is inspected independently.
    Cache only module/class functions and immutable names, never adapter
    instances or compiled launchables.
    """
    return frozenset(inspect.signature(entry).parameters)


def _q_lens_addr(t: Optional[torch.Tensor]) -> int:
    """Kernel-side slot for the per-batch Q lengths: the (B,) int32 tensor's device address, 0 when absent.

    A raw Int64 scalar rather than a tensor parameter: on SM107 prefill_d512_mxfp8 the extra cute.Tensor parameter
    alone pushed ptxas from 159 to 254 registers (+28 percent device time); the same reads through a raw pointer
    cost nothing. The caller keeps the tensor alive across execute (it is the graph's own seq_len_q buffer)."""
    return 0 if t is None else int(t.data_ptr())


def dtype_name(buffer) -> str:
    """The buffer's dtype as a bare name, whoever produced it.

    A caller buffer reaches these checks as whatever the graph normalized it
    into, which is a variant-pack slot rather than a torch tensor. Comparing
    ``buffer.dtype is torch.float32`` therefore rejects a perfectly good fp32
    buffer with "must be float32; got float32". Names are the one spelling
    every producer agrees on -- torch prints ``torch.float32``, numpy and the
    slot print ``float32``.
    """
    return str(buffer.dtype).rsplit(".", 1)[-1]


_SM100_FLAVORS = (
    (64, 64),
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
)  # ordered smallest-first: (max D_QK, max D_V) envelope
# Flavors whose f16/bf16 kernel packs a proper divisor of a GQA group that does
# not divide the 128-row tile (partial PackGQA, Cfg.PACK_G); the others pack the
# whole group only.  Mirror of Capabilities.pack_gqa_partial_d_shapes.
_SM100_PARTIAL_PACK_GQA_FLAVORS = ((128, 128), (256, 256), SM100_NATIVE_D576_SHAPE)
_SM100_KERNEL_FILES = {
    SM100_NATIVE_D576_SHAPE: "sm100/prefill_d512_f16_2x2.py",
    (512, 512): "sm100/prefill_d512_f16.py",
    (256, 256): "sm100/prefill_d256_f16.py",
    (192, 128): "sm100/prefill_d192_d128_f16.py",
    (128, 128): "sm100/prefill_d128_f16.py",
    # Same file as (128, 128): one pipeline, two head-dim geometries, selected
    # by TemplateParams.d_flavor. d<=64 graphs used to ride the d128 envelope
    # and pay a zero-filled 128-wide MMA tile for it.
    (64, 64): "sm100/prefill_d128_f16.py",
    # The 32-wide small tile of the same file (TemplateParams.d_flavor = 32; config_sm100.make_cfg_d32): the SM100 twin of the Rubin
    # small tile.  A kernel-file entry only until the engine row and the flavor walk (_SM100_FLAVORS) serve it.
    (32, 32): "sm100/prefill_d128_f16.py",
}
# The d512 f16/bf16 forward on the 2x2 DATAPATH (TemplateParams.mma_2x2 -> the sibling file below):
# one pipeline per CTA on the cta_group::2 M=128 atom instead of the cga4x1 role split.  Selected by
# the record field, never by a knob; D512_2X2 is the CALL-TIME module-constant switch that
# template_params() reads -- the DQ_SINGLE_LAUNCH precedent: a module constant, not configuration, and
# never per plan.  DEFAULT TRUE since 2026-10-06: the 2x2 kernel won every measured shape on both arch
# lines (B1 H128 S8192 +21 % dense / +21 % causal on B200, +21 / +18 % on cc 10.7; GQA 64x1 Sq16384 ladder
# +22..46 % / +11..24 %; short and small-batch shapes, see the tracker's forward footnote).  False is the
# role-split A/B arm (the `d512_arm` / `two_by_two` test fixtures set both explicitly): the record keeps
# mma_2x2=False for the original plans. Native Rubin dense packed/split plans require 2x2 under either
# value. Other plans the twin does not serve (paged, quantized, SM100 split / G128) retain the role split.
_SM100_D512_2X2_KERNEL_FILE = "sm100/prefill_d512_f16_2x2.py"
_SM107_D512_2X2_KERNEL_FILE = "sm107/prefill_d512_f16_2x2.py"
D512_2X2: bool = True
# The d128 f16/bf16 DECODE tile (TILES_Q=1, cga1, one softmax warpgroup, three
# KV stages -- config_sm100.CfgD128Decode): what a (128, 128) plan with
# TILE_CGA_M=1 lowers to on dense graphs.  cga1 on this flavor IS the decode
# tile; the TILES_Q=2 prefill body at cga1 stays reachable only by loading its
# template directly (its cga1 arm is kept for that).
_SM100_DECODE_KERNEL_FILE = "sm100/decode_d128_f16.py"
_SM100_DECODE_FLAVOR = (128, 128)
# Q rows one d64 decode tile covers (CfgD64Decode: TILES_Q=1 x TILE_M=128).
_D64_DECODE_TILE_ROWS = 128
# Live Q rows one TMEM lane quadrant holds: the d64 decode tile's quadrant softmax (TemplateParams.decode_quad) serves up to this many.
_D64_QUAD_ROWS = 32
# The cc the decode tile's early-Q form (TemplateParams.early_q) was measured on.  Positive gate: cc 10.7 loads the same tile (dense d128 decode, issue
# #1472) and keeps the staged O store until a Rubin run says otherwise.
_EARLY_Q_CC: frozenset[tuple[int, int]] = frozenset({(10, 0), (10, 3)})
# Decode-shaped alternates selected by a TemplateParams field instead of a knob
# (TemplateParams.decode_q_tile != 0, set by SdpaFwdDslSm100._decode_q_tile when
# S_q * pack_g rows fit the tile's N extent): the d256 flavor's swap-AB tile.
# The two decode tiles are disjoint by flavor -- (128, 128) rides the cga1 knob
# above, (256, 256) this record -- so _load_sm100_kernel_module tests both.
_SM100_DECODE_KERNEL_FILES = {
    (256, 256): "sm100/decode_d256_f16.py",
}
# The Rubin (cc 10.7) sibling of the d256 decode tile: the same swap-AB body under
# config_sm107.make_cfg_d256_decode (the Rubin SMEM-carveout and descriptor-window
# validators, the module's DESC_VERSION / SPIN_RING_WAITS constants).  Selected by the
# same record field; _decode_q_tile routes both arch lines.
_SM107_DECODE_KERNEL_FILES = {
    (256, 256): "sm107/decode_d256_f16.py",
}
# DTYPE_* codes: E4M3=0, E5M2=1, BF16=2, FP16=3. FP8 inputs (0/1) route to the
# FP8 kernel families; the output dtype is encoded the same way.
_SM100_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
_SM100_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _torch_fp4():
    """``torch.float4_e2m1fn_x2`` when this torch build has it, else ``None``.

    The packed FP4 dtype arrived in torch 2.8 and the torch dependency group is
    unversioned, so every ORDINARY path resolves it lazily: a missing symbol
    never matches a dtype comparison and never enters a dtype list. Only a
    caller that actually hands over an FP4 O needs it (and cannot without it)."""
    return getattr(torch, "float4_e2m1fn_x2", None)


def _with_fp4(dtypes):
    """``dtypes`` plus the packed FP4 dtype when the torch build exposes it."""
    fp4 = _torch_fp4()
    return [*dtypes, fp4] if fp4 is not None else list(dtypes)


# FP8 kernels use E4M3/E5M2 inputs and BF16/FP16/FP8 outputs. Block-scale
# Per-tensor and block-scale FP8 select independently from their native maps.
_SM100_MXFP8_KERNEL_FILES = {
    (128, 128): "sm100/prefill_d128_mxfp8.py",
    # Native d64: the same file at TILE_K = TILE_O = 64 (TemplateParams.d_flavor),
    # instead of zero-filling a 128-wide tile for gpt-oss-class head dims.
    (64, 64): "sm100/prefill_d128_mxfp8.py",
    (192, 128): "sm100/prefill_d192_d128_mxfp8.py",
    (256, 256): "sm100/prefill_d256_mxfp8.py",
    (512, 512): "sm100/prefill_d512_mxfp8.py",
}
# Rubin (SM107) siblings.  Separate maps rather than entries in the SM100
# ones: the lowerings genuinely diverge (dense K=64 FP8 MMA, 576-column TMEM,
# version-1 tcgen05 SMEM descriptors for operands above 256 KiB), which is the
# same reason engines.py keeps one row per ARCH LINE.
_SM107_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_f16.py",
    (256, 256): "sm107/prefill_d256_f16.py",
    (192, 128): "sm107/prefill_d192_d128_f16.py",
    (128, 128): "sm107/prefill_d128_f16.py",
    # The same file at TILE_K = TILE_O = 64 and 32 (TemplateParams.d_flavor; config_sm107.make_cfg_d64 / make_cfg_d32):
    # a head dim <= 64 used to ride the d128 envelope and pay a zero-filled 128-wide MMA tile (2x / 4x the BMM1 work and
    # the full d128 SMEM) for it.  The loader sets d_flavor from the flavor so no caller can pair the key with the
    # d128 geometry (config_sm107.SM107_F16_SMALL_D_SHAPES is the row's claim).
    (64, 64): "sm107/prefill_d128_f16.py",
    (32, 32): "sm107/prefill_d128_f16.py",
}
# The Rubin f16/bf16 flavor walk (smallest-first envelopes) -- _SM100_FLAVORS plus the 32-wide tile, which has no SM100
# kernel: a graph's head dim lands on the smallest covering width (8 / 16 / 24 -> 32, 40 / 48 / 56 -> 64, 72..120 -> 128,
# the natives on themselves; config_sm107.sm107_f16_tile_width is the same pick as a function).
_SM107_FLAVORS = ((32, 32),) + _SM100_FLAVORS
_SM107_FP8_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_fp8.py",
    (256, 256): "sm107/prefill_d256_fp8.py",
    (192, 128): "sm107/prefill_d192_d128_fp8.py",
    (128, 128): "sm107/prefill_d128_fp8.py",
    (64, 64): "sm107/prefill_d128_fp8.py",
}
_SM107_MXFP8_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_mxfp8.py",
    (256, 256): "sm107/prefill_d256_mxfp8.py",
    (192, 128): "sm107/prefill_d192_d128_mxfp8.py",
    (128, 128): "sm107/prefill_d128_mxfp8.py",
}
_SM100_FP8_KERNEL_FILES = {
    (128, 128): "sm100/prefill_d128_fp8.py",
    # Native d64: the same file at TILE_K = TILE_O = 64 (TemplateParams.d_flavor),
    # instead of zero-filling a 128-wide tile for gpt-oss-class head dims.
    (64, 64): "sm100/prefill_d128_fp8.py",
    (192, 128): "sm100/prefill_d192_d128_fp8.py",
    (256, 256): "sm100/prefill_d256_fp8.py",
    (512, 512): "sm100/prefill_d512_fp8.py",
}


def _sm100_fp8_shapes(pertensor: bool, device_cc: tuple[int, int]) -> frozenset[tuple[int, int]]:
    """NATIVE (exact) FP8 kernel-flavor shapes for the device line."""
    if device_cc == (10, 7):
        return frozenset(_SM107_FP8_KERNEL_FILES if pertensor else _SM107_MXFP8_KERNEL_FILES)
    kernel_files = _SM100_FP8_KERNEL_FILES if pertensor else _SM100_MXFP8_KERNEL_FILES
    return frozenset(kernel_files)


# Per-native-shape ENVELOPE FLOOR (see engines.Capabilities.d_envelope_floors,
# which MUST agree — test_fp8_envelope_floor_matches_engine_row pins it). The
# d512 flavor serves the (256, 512] band on both head dims: the range no
# smaller FP8 flavor reaches, at most 2x zero-padding. A smaller graph is
# declined rather than routed onto a kernel whose cga4x1 role-split geometry is
# tuned for d = 512.
_SM100_FP8_ENVELOPE_FLOORS = {(512, 512): 256}


def _fp8_envelope_covers(d_qk: int, d_v: int, shapes) -> bool:
    """Does some native FP8 shape cover ``(d_qk, d_v)`` as an envelope bound,
    respecting that shape's floor?"""
    return any(d_qk <= sq and d_v <= sv and min(d_qk, d_v) > _SM100_FP8_ENVELOPE_FLOORS.get((sq, sv), 0) for sq, sv in shapes)


# Both flavors tile KV in TILE_N=128 columns; the KV tail is only masked when
# the padded/causal mask paths are active (see check_support).
_SM100_TILE_N = 128

# Keyed by kernel flavor (config_sm120.F16_FLAVORS / FP8_FLAVORS); None = the general template. The fp8
# family has a dedicated d512 flavor above the general template's head range.
_SM120_KERNEL_FILES = {
    _SM120_D256_FLAVOR: "sm120/prefill_d256_f16.py",
    _SM120_D512_FLAVOR: "sm120/prefill_d512_f16.py",
    None: "sm120/prefill_f16.py",
}
_SM120_DECODE_KERNEL_FILE = "sm120/decode_f16.py"
_SM120_FP8_KERNEL_FILES = {_SM120_D512_FLAVOR: "sm120/prefill_d512_fp8.py", None: "sm120/prefill_fp8.py"}
_SM120_MXFP8_KERNEL_FILES = {_SM120_D512_FLAVOR: "sm120/prefill_d512_mxfp8.py", None: "sm120/prefill_mxfp8.py"}


_SM120_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}

# Workspace-carve chunk alignment. The contract minimum is 16 bytes; 128 is
# used so the per-sequence O TMA descriptors carved for the THD path satisfy
# the cuTensorMap GMEM alignment (64 B) with margin. torch storage bases are
# 512 B aligned, so 128 B-multiple offsets stay 128 B aligned absolutely.
_WS_ALIGN = 128


@contextmanager
def _torch_stream_context(
    current_stream: Optional[cuda.CUstream],
    device: torch.device,
    *,
    verify_current: bool = False,
) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch (Rule 5;
    the sentinel mapping and the raw-handle fast path live in ``cudnn._torch_stream``).

    ``verify_current`` bypasses the raw-handle shortcut when a caller supplies
    the stream, making that handle authoritative for the context.
    """
    with stream_context(current_stream, device, verify_current=verify_current):
        yield


# Causal-balancing scheduler choice: use the L2-GROUPED LPT when ONE head's K+V
# working set fits this budget -- that is the condition under which the
# block-cyclic grouping can actually keep that K/V resident; otherwise plain
# reverse-row LPT.
_SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024

# SM count per device for the persistent THD grid. A device property cannot
# change under a live process, and the query costs ~7 us on the execute hot
# path, so resolve it once per device.
_THD_CTAS_CACHE: dict = {}
# Raw multi_processor_count, cached separately: _THD_CTAS_CACHE holds an
# already-scaled CTA count, so the two cannot share a key.
_THD_SMS_CACHE: dict = {}


def _thd_cache_key(device):
    """Cache key for a device. ``torch.device("cuda")`` carries index None and
    means the CURRENT device, so resolve it — keying on None would hand every
    device on a multi-GPU host whichever entry landed first."""
    key = getattr(device, "index", None)
    return torch.cuda.current_device() if key is None else key


def _device_sm_count(device) -> int:
    """``multi_processor_count``, resolved once per device (see above)."""
    key = _thd_cache_key(device)
    n = _THD_SMS_CACHE.get(key)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _THD_SMS_CACHE[key] = n
    return n


def _causal_sched_policy(s_kv: int, d_qk: int, d_v: int, elem_bytes: int) -> int:
    """SCHED_LPT_L2 vs SCHED_LPT for a causal graph (see _SCHED_L2_BUDGET_BYTES)."""
    one_head_bytes = int(s_kv) * (int(d_qk) + int(d_v)) * int(elem_bytes)
    return SCHED_LPT_L2 if _SCHED_L2_BUDGET_BYTES >= one_head_bytes else SCHED_LPT


def ws_align(nbytes: int) -> int:
    """Round a scratch-chunk size up to the carve alignment (128 B)."""
    return -(-int(nbytes) // _WS_ALIGN) * _WS_ALIGN


class WorkspaceCarver:
    """Carves fixed-size, aligned scratch views out of the CALLER's workspace.

    FROST executor contract (see ``engine._FrostSdpaFwdPlan``): an executor that
    records a non-zero ``workspace_bytes`` is handed the caller's workspace
    buffer (``ExecutionContext.workspace``) at execute and
    carves its per-execute scratch from it instead of allocating. Chunks are
    dealt sequentially at 128-byte relative alignment and never reach beyond
    the buffer; an absent, non-torch, or undersized buffer raises immediately
    with the required size in the message (never silent corruption).
    """

    def __init__(self, workspace, required: int, owner: str):
        if workspace is None:
            raise ValueError(
                f"cudnn.sdpa: {owner} requires a {required}-byte workspace but execute() "
                f"received none; allocate graph.get_workspace_size() bytes (uint8, on the "
                f"graph's device) and pass the buffer to execute()"
            )
        if not (hasattr(workspace, "numel") and hasattr(workspace, "element_size") and hasattr(workspace, "view")):
            raise TypeError(f"cudnn.sdpa: {owner} carves its scratch out of the caller's workspace and needs a torch.Tensor; got {type(workspace).__name__}")
        flat = workspace if workspace.dtype == torch.uint8 else workspace.view(torch.uint8)
        flat = flat.reshape(-1)
        if flat.numel() < required:
            raise ValueError(
                f"cudnn.sdpa: {owner} requires a {required}-byte workspace; the provided "
                f"buffer has only {flat.numel()} bytes (size it with graph.get_workspace_size())"
            )
        if flat.data_ptr() % 16 != 0:
            raise ValueError(f"cudnn.sdpa: {owner} workspace must be at least 16-byte aligned; got data_ptr=0x{flat.data_ptr():x}")
        self._flat = flat
        self._off = 0
        self._owner = owner

    def take(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """The next scratch chunk: a 1-D ``numel``-element view of ``dtype``."""
        nbytes = int(numel) * dtype.itemsize
        start, end = self._off, self._off + nbytes
        if end > self._flat.numel():
            raise ValueError(f"cudnn.sdpa: {self._owner} workspace overrun: chunk [{start}, {end}) exceeds the {self._flat.numel()}-byte buffer (sizing bug)")
        self._off = start + ws_align(nbytes)
        try:
            return self._flat[start:end].view(dtype)
        except RuntimeError as exc:
            raise ValueError(f"cudnn.sdpa: {self._owner} workspace is not sufficiently aligned for {dtype} scratch: {exc}") from None

    def remaining(self) -> torch.Tensor:
        """The unconsumed tail (uint8) — handed down to a nested carver."""
        return self._flat[self._off :]


def _flavor_tag(flavor: tuple[int, int]) -> str:
    d_qk, d_v = flavor
    return f"d{d_qk}" if d_qk == d_v else f"d{d_qk}_d{d_v}"


def _pick_flavor(d_qk: int, d_v: int, candidates: Optional[tuple[tuple[int, int], ...]] = None) -> tuple[int, int]:
    """Smallest flavor whose envelope covers ``(d_qk, d_v)``.

    ``candidates`` restricts the walk to the flavors that have a kernel for the
    caller's quantization; ``None`` = the full f16/bf16 list.

    ENVELOPE (zero-padding) semantics: one flavor covers ``d_qk`` and ``d_v``
    with its own max extents — e.g. (192, 128) runs on the d192/d128 kernel. The
    kernel's TMA descriptors are built from the ACTUAL tensor extents while
    the tile box stays the compile-time D, so loads past d_qk / d_v hardware
    zero-fill (adding exact zero terms to every QK^T dot product — S, softmax
    and P·V are bit-identical to the unpadded problem) and O stores past d_v
    are OOB-clipped. Per-tensor FP8 serves the dense envelope of every native
    flavor it ships (d % 16 at 1 byte/elem); MXFP8 stays exact-shape because
    its block-scale tensors are not padded. All of it is gated in check_support
    / engines.mismatch, including the f16 alignment rule (d % 8, the TMA
    16-byte global-stride rule at 2 bytes/elem).
    """
    pool = candidates if candidates is not None else _SM100_FLAVORS
    for flavor in pool:
        fdqk, fdv = flavor
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(f"Frost SM100 DSL SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); available envelopes: {sorted(pool)}.")


# The exp2 MUFU / FMA split of the sm100 softmax (``TemplateParams.exp2_fma_split``, the ``_E2E_*`` block of
# sm100/prefill_d128_mxfp8.py, prefill_d128_fp8.py and prefill_d192_d128_f16.py) is claimed per KERNEL and per
# ARCH.  It trades MUFU.EX2 pipe-time for FMA pipe-time, so its sign follows the part's MUFU rate -- MEASURED
# 16 elements/clk/SM on cc 10.0 (B200) and 32 on cc 10.7 (Rubin: the same split is -9..-10 %, an emulated exp2
# costs 1.99x the MUFU time it frees); cc 10.3 (GB300) DOCUMENTS the same doubled exp2 throughput, so the split
# stays off there until a GB300 A/B says otherwise.  Positive gate on the measured cc, never a negative gate on
# 10.3: the engine rows are cc RANGES and an unmeasured part gets the develop (all-MUFU) kernel.  Precisely: with
# the gate OFF the four kernels trace the all-MUFU exp2 spelling (``_E2E_ENABLED`` False, ``cute.math.exp2`` at
# every softmax site), and the d128 MXFP8 kernel ADDITIONALLY carries its arch-independent Amax_O FMNMX fold
# (``fmax_f32``, not behind this gate -- MEASURED +0.5..+1.8 % on B200, unmeasured on GB300), so a cc 10.3 build of
# that kernel is develop's exp burst plus the fold, not develop's kernel byte for byte.
#
# Per kernel, keyed by (quantization kind, flavor) -- the spelling ``_load_sm100_kernel_module`` selects the
# kernel file by.  ON = MEASURED wins on B200 (A/B/A x3, CUPTI medians, 2026-09-22): ("mxfp8", (128, 128))
# +7.8 % S=16K dense / +9.6 % causal (B=1 H=24/8); ("fp8", (128, 128), E4M3 / E5M2 per-tensor) +4.5 % S=8K
# dense (H=64/8), causal +0.3 % = noise; ("f16", (192, 128), bf16 / fp16) +1.9 % S=8K dense / +1.75 % causal
# (H=128/128); ("fp8", (192, 128), E4M3 / E5M2 per-tensor) -- NOT an _E2E_* block: this entry gates that
# kernel's own pre-existing _exp2_* helper mix (chunk-0 / late-tail / alpha), which used to be unconditional.
# MEASURED 2026-09-28 at the DSv3 layer (B=2 H=128/128 S=2K) vs the all-MUFU spelling: +7.0 % causal / +5.7 %
# dense on B200, and -10 % causal / -6 % dense with the mix left on at cc 10.3 (B300) -- the reason it is now
# gated at all.  ("mxfp8", (192, 128)) is the same kind of entry: it gates that kernel's own pre-existing
# ex2_emulation_2 mix (chunk-0 mask-aware / 6-pair / late-tail / scalar; unconditional before 2026-09-29), not an
# _E2E_* block.  B200 keeps the mix, the spelling it was tuned with; MEASURED at the DSv3 layer (B=2 H=128/128)
# with it left on at cc 10.3 (B300): dense S=2K 1.17x of cuDNN, 1.02x with it off (S=8K 1.14x -> 0.98x; kimi-K3
# 1.16x -> 1.00x).  ("f16", (128, 128)) (bf16 / fp16): dense +3.9 % but causal -1.9 / -2.4 % on B200, so the kernel
# folds its _E2E_* block in on the DENSE band only (module constant _E2E_DENSE_BAND: no causal / sliding-window bit in
# CFG.MASK_FLAGS) -- this entry switches the field on, the band gate inside the kernel keeps the causal builds all-MUFU.
# OFF = MEASURED losses or no measurement: an additional _E2E_* block on ("fp8", (192, 128)) dense +1 % marginal, causal -1.6 %, and one on
# ("mxfp8", (192, 128)) -3.3..-4.0 % dense -- neither block was merged, and neither is what those entries gate;
# BF16 D64 nonpaged causal interiors also use the existing mix; masked tiles, dense bands and other layouts stay on MUFU.
# Every d256 / d512 flavor is unmeasured.  Widening either set is a per-cc, per-kernel measurement -- never a default.
_EXP2_FMA_SPLIT_CC: frozenset[tuple[int, int]] = frozenset({(10, 0)})
_EXP2_FMA_SPLIT_KERNELS: frozenset[tuple[str, tuple[int, int]]] = frozenset(
    {("mxfp8", (128, 128)), ("fp8", (128, 128)), ("fp8", (192, 128)), ("mxfp8", (192, 128)), ("f16", (192, 128)), ("f16", (128, 128)), ("f16", (64, 64))}
)


# The L2-prefetching loader of the half-input single-CTA d192x128 f16 tile (``TemplateParams.cga1_l2_prefetch``,
# sm100/prefill_d192_d128_f16.py at cta_mma=1) is claimed per arch the same way: on the cc it is measured on; off
# elsewhere, where the tile keeps the develop loader.
_CGA1_L2_PREFETCH_CC: frozenset[tuple[int, int]] = frozenset({(10, 3)})


def _cga1_l2_prefetch_for(device_cc: tuple[int, int], *, kind: str, flavor: tuple[int, int]) -> bool:
    """``TemplateParams.cga1_l2_prefetch`` for one build: the f16 d192x128 kernel on a cc in ``_CGA1_L2_PREFETCH_CC``."""
    return bool(kind == "f16" and tuple(flavor) == (192, 128) and tuple(device_cc) in _CGA1_L2_PREFETCH_CC)


def _quant_kind(fp8: bool, pertensor: bool) -> str:
    """The quantization-kind tag of one SM100-family build -- ``"fp8"`` (per-tensor E4M3 / E5M2), ``"mxfp8"``
    (block-scale) or ``"f16"`` (fp16 / bf16) -- the same spelling ``_load_sm100_kernel_module`` keys the kernel
    file by."""
    return ("fp8" if pertensor else "mxfp8") if fp8 else "f16"


# The d64 decode tile (``TemplateParams.decode_tile``: sm100/decode_d128_f16.py at d_flavor 64) shares its (kind, flavor) key with the d64 prefill
# tile but is another kernel, so it claims the split in its own set: the tile's one softmax warpgroup is MUFU-bound at a long KV on cc 10.0, and the
# FMA pipe takes a quarter of every row's exp2 columns.  MEASURED on B200 (same-node A/B/A/B, kernel time, gpt-oss d64 full attention with a sink,
# all four tp shards): -8..-9 % from 8 sequences x 32K keys up (b=128 x 128K keys 919 -> 841 us at tp8), -3..-6 % at 2K keys, 0 at b=1 x 2K; S_q = 4
# -4..-5 %.  The kernel folds the split out of the sliding-window builds (a unit there is one or two KV tiles: +2..+5 % at S_q >= 2).  The d128 decode
# tile keeps the dense path off (HBM-bound where measured). Its THD split form separately uses the existing f16/d128 flag with a kernel-local
# packed-partials gate. D64 prefill claims its own BF16 causal-interior mix in the prefill set.
_EXP2_FMA_SPLIT_DECODE_KERNELS: frozenset[tuple[str, tuple[int, int]]] = frozenset({("f16", (64, 64))})


def _exp2_fma_split_for(device_cc: tuple[int, int], *, kind: str, flavor: tuple[int, int], decode_tile: bool = False) -> bool:
    """``TemplateParams.exp2_fma_split`` for one build: the (quantization kind, flavor) kernels that carry the split
    AND measured a win, on a cc where it was measured (``_EXP2_FMA_SPLIT_KERNELS`` x ``_EXP2_FMA_SPLIT_CC``); False
    everywhere else.  ``kind`` is ``_quant_kind(fp8, pertensor)``; ``decode_tile`` selects the decode tile's own set
    (``_EXP2_FMA_SPLIT_DECODE_KERNELS``) in place of the prefill kernels'."""
    kernels = _EXP2_FMA_SPLIT_DECODE_KERNELS if decode_tile else _EXP2_FMA_SPLIT_KERNELS
    return bool((kind, tuple(flavor)) in kernels and tuple(device_cc) in _EXP2_FMA_SPLIT_CC)


def supported_cgas_for(flavor: tuple[int, int], *, fp8: bool, device_cc: tuple[int, int], pertensor: bool = True) -> tuple[int, ...]:
    """CGA widths the STANDALONE adapter serves for a kernel flavor.

    A module-level function, not an inline expression in ``check_support``, so a
    test can assert it without a live device of the right arch -- the Rubin arm
    below is unreachable from any host that is not cc 10.7, which is exactly the
    kind of branch that rots untested.

    d192x128 accepts both widths on Blackwell.  On the RUBIN QUANTIZED line it
    is cga2 ONLY, and that is a descriptor constraint rather than a tuning
    choice: at cga1 the K/V rings are not halved, which pushes the MXFP8
    scale-factor tiles (and, with a half-precision O, the FP8 kernel's row-sum
    "ones" tile) at or past the 256 KiB version-0 tcgen05 descriptor window.  A
    wrapped descriptor reads Q data as its operand: silently wrong LSE/O, no
    crash (rules/mma-tma-matrix.md S6).

    Both kernels also raise at import if handed cga1, but a kernel-side raise
    alone is not enough -- ``check_support()`` would still return True and the
    failure would escape as a bare ValueError from ``compile()``, i.e. a plan
    that clears eligibility and dies in the lowering (contract rule 8b').  This
    is the wrapper twin of the engine rows leaving (192, 128) on their default
    ``cgas={2}``.  Keep the three in lockstep.
    """
    if device_cc == (10, 7) and fp8 and flavor == (192, 128):
        return (2,)
    if flavor == (192, 128):
        return (1, 2)
    if device_cc == (10, 7) and not fp8 and flavor in _SM107_F16_SMALL_D_SHAPES:
        # The Rubin small head-dim tiles (the d128 file at TILE_K = TILE_O = 32 / 64): ONE constant with the config factory
        # (which refuses any other width) and the engine row's cgas_by_d_shape (rule 8b') -- cga1 for both, cga2 also at d64.
        return tuple(sorted(_SM107_SMALL_D_CGAS[flavor]))
    if device_cc != (10, 7) and not fp8 and flavor in _SM100_DENSE_ONLY_D_SHAPES:
        # The SM100 32-wide small tile: ONE constant with the config factory (which refuses any other width) and the engine row's
        # cgas_by_d_shape (rule 8b') -- cga1 (at TILE_O 32 a cga2 V operand would be a 32-byte MN-major atom no kernel runs).
        return tuple(sorted(_SM100_SMALL_D_CGAS[flavor]))
    # d64: cga1 is the tuned width (halved slabs clear the SMEM cap without the
    # Q/O alias, and a 2-CTA cluster only widens the Q rows a cluster must cover
    # under a narrow band). cga2 still builds, so both are offered.
    if device_cc == (10, 7) and fp8 and pertensor and flavor == (64, 64):
        return (1, 2)  # Native Rubin P.V descriptors include the 32-byte cga2 V swizzle.
    if flavor == (64, 64):
        # The quantized d64 legs run cga1 only: at cga2 the halved V slab would
        # need a 32-byte swizzle the FP8 kernels' P.V descriptors do not model.
        return (1,) if fp8 else (1, 2)
    if fp8 and flavor == (256, 256):
        # cc 10.7 MXFP8 admits both widths: the dense unsplit legs run the 2-CTA pair (config_sm100.d256_mxfp8_pair_leg,
        # the width template_params derives); per-tensor FP8 d256 keeps cga1.
        return (1, 2) if (device_cc == (10, 7) and not pertensor) else (1,)
    if device_cc != (10, 7) and fp8 and pertensor and flavor == (128, 128):
        # Per-tensor FP8 d128 builds at both widths: cga1 is one 256-row CTA
        # (no collective MMA, STAGES_KV=2, Q/O aliased), cga2 the 2-CTA pair.
        return (1, 2)
    if device_cc != (10, 7) and fp8 and not pertensor and flavor == (512, 512):
        return (1,)
    if device_cc != (10, 7) and fp8 and not pertensor and flavor == (128, 128):
        # MXFP8 d128 builds at both widths on the SM100 line: cga1 is one 256-row CTA of the prefill body (no
        # collective MMA, three K/V stages), cga2 the 2-CTA pair; heuristics._auto_sched_cga picks between them.
        return (1, 2)
    if not fp8 and flavor == _SM100_DECODE_FLAVOR:
        # cga1 on the d128 f16/bf16 flavor selects the DECODE tile (sm100/decode_d128_f16.py) on the SM100 line and,
        # since issue #1472, on cc 10.7 (the same body compiled for sm_107a, inside the version-0 descriptor window);
        # dense graphs only -- check_support declines it for THD, mirroring engines.mismatch / effective_cgas.
        return (1, 2)
    return (2,)


# cc 10.7 softmax levers, per selected flavor (the engine rows and config_sm107.SM107_*_FLAVORS carry the same
# claims): HALF = the f16x2 exponent arm of the FP8 / MXFP8 kernels; the pre-folded scale = the MXFP8 and half
# kernels' arm (per-tensor FP8 is declined on contract, see check_support).
_SM107_HALF_SOFTMAX_FLAVORS = frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)})
_SM107_PREFOLDED_FLAVORS = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})


def softmax_arms_of(k_mod, *, has_lse: bool) -> str:
    """The softmax arms a compiled kernel module TRACED, as a short tag: ``f32`` or ``f16`` (the exponent), plus
    ``+fold`` (the pre-folded scale: raw max, plain subtract) and ``+fused`` (the fused shift+convert, which the
    stats-less build traces when both levers are set).  Read from the module constants SOFTMAX_F16 /
    SCALE_PREFOLDED / _FUSED_SHIFT_CVT; a kernel without them traces the f32 chain with the in-kernel scale.  The
    compiled executor exposes the tag so a sweep can assert that the arm it asked for actually compiled."""
    tag = "f16" if getattr(k_mod, "SOFTMAX_F16", 0) else "f32"
    if getattr(k_mod, "SCALE_PREFOLDED", 0):
        tag += "+fold"
    if getattr(k_mod, "_FUSED_SHIFT_CVT", False) and not has_lse:
        tag += "+fused"
    return tag


def _load_kernel_template(filename: str, params: Hashable, tag: str):
    """Load one uniquely named kernel module per template parameter set."""

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    # The sm107 config module's private sweep seams specialize a body WITHOUT touching the record: fold their state into the module
    # cache key and the source digest (None on every plan path = the key every caller has had).
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import sm100_sweep_seam_state
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sweep_seam_state

    seams, seams_sm100 = sweep_seam_state(), sm100_sweep_seam_state()
    # The SM100 small tiles' own seam (config_sm100._STAGES_KV_OVERRIDE_FOR_SWEEPS) joins the key ONLY when it is set, so every key a
    # process without it builds is the key it always had (the sm107 seams' alone, or None).
    extra_key = seams if seams_sm100 is None else (seams, seams_sm100)
    return load_template(path, params, tag=tag, extra_key=extra_key)


def _load_sm100_kernel_module(flavor: tuple[int, int], params: Sm100TemplateParams, fp8: bool = False, pertensor: bool = False, rubin: bool = False):
    """Load one SM100-family module for the selected flavor and quantization
    path. Rubin uses its SM107 siblings, including paged D128 half, except for
    the shared half single-Q split and D256 paged pipelines, whose SMEM fits the version-0 descriptor window,
    and, on dense (128, 128) half graphs, the shared d128 decode tile at cga1
    and native D128 prefill under PackGQA."""

    tag = _flavor_tag(flavor)
    if rubin and not fp8 and getattr(params, "decode_q_tile", 0):
        # Decode-shaped d256 f16/bf16 graphs on cc 10.7: the Rubin twin of the swap-AB
        # tile (the record field routes, exactly as on the SM100 arm below).  It serves
        # the paged contract natively, so it precedes the shared paged-prefill arm.
        return _load_kernel_template(_SM107_DECODE_KERNEL_FILES[flavor], params, f"sdpa_fwd_sm107_decode_{tag}")
    if (
        rubin
        and not fp8
        and params.cta_mma == 1
        and params.thd_varlen
        and ((flavor == (192, 128) and not params.paged_kv) or (flavor == (128, 128) and params.split_kv > 1))
    ):
        # The single-CTA half pipeline stays below the version-0 descriptor
        # window and can compile natively for Rubin without another kernel body.
        params = replace(params, single_q_head_dim=flavor[0])
        return _load_kernel_template(_SM100_DECODE_KERNEL_FILE, params, f"sdpa_fwd_sm107_{tag}_single_q")
    if rubin and not fp8 and flavor == (256, 256) and params.thd_varlen and not params.paged_kv and params.split_kv > 1:
        return _load_kernel_template(_SM100_KERNEL_FILES[flavor], params, f"sdpa_fwd_sm107_{tag}_packed_split")
    if rubin and not fp8 and params.paged_kv and flavor == (256, 256):
        return _load_kernel_template(_SM100_KERNEL_FILES[flavor], params, f"sdpa_fwd_sm107_{tag}_paged")
    if rubin and not fp8 and flavor == _SM100_DECODE_FLAVOR and params.cta_mma == 1 and not params.thd_varlen and not params.paged_kv and not params.ragged_q:
        # Dense d128 half at TILE_CGA_M=1 on cc 10.7: the shared 128-row DECODE tile (CfgD128Decode, 224 KiB, inside the
        # version-0 descriptor window) -- the dense twin of the single-CTA THD leg above (issue #1472).  THD_VARLEN at cga1
        # is the two-slab paged prefill (above) or the single-Q split (above); never this body.
        return _load_kernel_template(_SM100_DECODE_KERNEL_FILE, params, f"sdpa_fwd_sm107_{tag}_decode")
    if rubin:
        # Tag spelling is load-bearing: it keys the template-module cache, and
        # "sdpa_fwd_sm107_fp8_<flavor>" is what the shipped d128 FP8 row has
        # always produced -- keep it byte-identical.
        kind = ("fp8" if pertensor else "mxfp8") if fp8 else "f16"
        files = (_SM107_FP8_KERNEL_FILES if pertensor else _SM107_MXFP8_KERNEL_FILES) if fp8 else _SM107_KERNEL_FILES
        filename = files[flavor]
        tag = f"sdpa_fwd_sm107_{kind}_{tag}"
        if (not fp8 and flavor in _SM107_F16_SMALL_D_SHAPES) or (fp8 and pertensor and flavor == (64, 64)):
            # The small head-dim tiles are the d128 FILE at d_flavor = 32 / 64: bind the record field to the flavor HERE,
            # so a caller that pairs the (32, 32) key with a default record (the SASS probes, a test's bare
            # TemplateParams()) traces the 32-wide geometry and never the d128 one under a d32 tag.
            params = replace(params, d_flavor=flavor[0])
        if not fp8 and flavor == (512, 512) and getattr(params, "mma_2x2", False):
            # The 2x2-datapath d512 sibling on cc 10.7 (the record is what routes): a distinct file and tag, so the
            # role-split and the 2x2 specializations coexist in one process, as on the SM100 arm below.
            filename = _SM107_D512_2X2_KERNEL_FILE
            tag = f"{tag}_2x2"
    elif not fp8 and flavor == SM100_NATIVE_D576_SHAPE:
        params = replace(params, d_flavor=576, mma_2x2=True)
        filename = _SM100_D512_2X2_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{tag}_2x2"
    elif not fp8 and flavor == (512, 512) and getattr(params, "mma_2x2", False):
        # d512 f16/bf16 on the 2x2 datapath: a distinct record -> a distinct module cache key and tag, so
        # the role-split and the 2x2 specializations coexist in one process (the decode_q_tile precedent).
        filename = _SM100_D512_2X2_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{tag}_2x2"
    elif fp8:
        filename = _SM100_FP8_KERNEL_FILES[flavor] if pertensor else _SM100_MXFP8_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{'fp8' if pertensor else 'mxfp8'}_{tag}"
    elif flavor == (192, 128) and params.cta_mma == 1 and params.thd_varlen and not params.paged_kv:
        params = replace(params, single_q_head_dim=192)
        filename = _SM100_DECODE_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{tag}_single_q"
    elif getattr(params, "decode_tile", False) or (flavor == _SM100_DECODE_FLAVOR and params.cta_mma == 1 and (not params.thd_varlen or params.split_kv > 1)):
        # D64's explicit decode tile and D128's one-CTA tile share this body.
        # D128 also owns admitted paged and nonpaged THD split members.
        filename = _SM100_DECODE_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{tag}_decode"
    elif getattr(params, "decode_q_tile", 0):
        # Decode-shaped d256 f16/bf16 graphs: the swap-AB tile (keys on the MMA M
        # axis, the packed Q rows on N) instead of the 256-row prefill tile.
        filename = _SM100_DECODE_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_decode_{tag}"
    else:
        filename = _SM100_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{tag}"
    return _load_kernel_template(filename, params, tag)


def _load_sm120_kernel_module(flavor: Optional[tuple[int, int]], params: Sm120TemplateParams, fp8: bool = False, pertensor: bool = False):
    if not fp8 and _sm120_is_decode_tile(params.q_tile):
        # The decode tile serves every half head tile up to 256 in one body.
        return _load_kernel_template(_SM120_DECODE_KERNEL_FILE, params, tag="sdpa_fwd_sm120_decode")
    files = (_SM120_FP8_KERNEL_FILES if pertensor else _SM120_MXFP8_KERNEL_FILES) if fp8 else _SM120_KERNEL_FILES
    tag = f"sdpa_fwd_sm120_{'fp8' if pertensor else 'mxfp8'}" if fp8 else "sdpa_fwd_sm120"
    if flavor is not None:
        tag = f"{tag}_{_flavor_tag(flavor)}"
    return _load_kernel_template(files[flavor], params, tag=tag)


# SM100/SM107/SM120 fold attn_scale into exp2 after an unscaled, -inf-masked running max: 0 * -inf (#1435).
_ZERO_SCALE_UNSUPPORTED = "attn_scale = 0 is not supported on this kernel (#1435)"


def _thd_binder_serves_the_gate() -> bool:
    """Whether this build's native THD binder (``_pybind_module._SdpaThdBinder``) binds the epilogue gate as a
    per-call slot.  The role arrived with the THD gate descriptor of the f16/bf16 d256 prefill; an older pybind
    extension has no such role, and a gated THD plan is declined typed instead of reaching a launch whose gate
    pointer is the template's None."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    return bool(getattr(_pybind_module._SdpaThdBinder, "supports_epilogue_gate", False))


def _binders_serve_the_paged_gate() -> bool:
    """Whether this build's native binders bind the epilogue gate on a PAGED plan: the shared SM100 d256 half body compiled for
    cc 10.7 carries the gate seams on its paged / THD / PackGQA arms (the serving EXTEND form), and both the THD binder (packed
    queries) and the dense binder (dense queries past the decode tile) gained the role with it.  An older extension's constructors
    refuse a gated paged plan, so the adapter declines typed instead of reaching that refusal at plan build."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    return bool(getattr(_pybind_module._SdpaThdBinder, "supports_paged_epilogue_gate", False)) and bool(
        getattr(_pybind_module._SdpaDenseBinder, "supports_paged_epilogue_gate", False)
    )


class SdpaFwdDsl(APIBase):
    """Implementation-agnostic interface for FROST DSL SDPA-forward kernels."""

    def __init__(
        self,
        sample_q: torch.Tensor | TensorDesc,
        sample_k: torch.Tensor | TensorDesc,
        sample_v: torch.Tensor | TensorDesc,
        sample_o: torch.Tensor | TensorDesc,
        sample_lse: Optional[torch.Tensor | TensorDesc] = None,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        scale_softmax: Optional[float] = None,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        cu_seq_q_lens: bool = False,
        cu_seq_kv_lens: bool = False,
        has_sink: bool = False,
        thd: bool = False,
        max_total_seq_len_q: Optional[int] = None,
        max_total_seq_len_kv: Optional[int] = None,
        dtype_o: Optional[torch.dtype] = None,
        pertensor_fp8: bool = False,
        sched_policy: Optional[int] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        cga: Optional[int] = None,
        split_kv: Optional[int] = None,
        softmax_precision: Optional[int] = None,
        softmax_scale_prefolded: bool = False,
        pack_gqa: Optional[bool] = None,
        paged_page_size: int = 0,
        paged_max_seq_len_kv: Optional[int] = None,
        paged_table_stride: Optional[tuple] = None,
        paged_table_v_stride: Optional[tuple] = None,
        thd_stats_padded: bool = False,
        sample_amax_o: Optional[torch.Tensor | TensorDesc] = None,
        pv_bf16: bool = False,
        stats_log2: bool = False,
        sample_gate: Optional[torch.Tensor | TensorDesc] = None,
        has_amax_o: bool = True,
        sample_sf_o: Optional[torch.Tensor | TensorDesc] = None,
        sample_scale_o: Optional[torch.Tensor | TensorDesc] = None,
        ragged_divisors: Optional[tuple[int, int, int]] = None,
        ragged_offsets_int64: bool = False,
        # APPENDED (2026-10-15): a CAP on the Rubin d256 decode tile's single-unit width table (config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES =
        # (16, 32, 64)) for this plan -- None = the whole table; 32 = the route before the 64-column width landed (value for value).  A
        # PERFORMANCE-ONLY knob (the same function under every width); read by _decode_q_tile_for.  Refused off cc 10.7 at check_support,
        # whatever the graph (the table it caps is the Rubin lowering's); on cc 10.7 it has no effect on a graph that never consults the
        # decode route (a prefill-shaped or THD graph, a quantized one) -- documented, not a decline.
        decode_q_tile_max: Optional[int] = None,
        # APPENDED: programmatic dependent launch -- the plan's kernels launch with the programmatic-stream-serialization attribute and
        # carry the griddepcontrol pair (a wait before their first global read, a release after their prologue).  Performance-only:
        # the same math, bitwise.  Served where the loaded kernel's compile() carries the knob (the cc 10.7 d256 decode tile and its
        # split combine); a kernel without it is a typed decline at compile().
        use_pdl: bool = False,
        sample_score_max: Optional[torch.Tensor | TensorDesc] = None,
        sample_score_sum_exp: Optional[torch.Tensor | TensorDesc] = None,
        sample_bias: Optional[torch.Tensor | TensorDesc] = None,
        sample_kv_mask: Optional[torch.Tensor | TensorDesc] = None,
        n_chunk: Optional[int] = None,
        bias_mode: Optional[int] = None,
        bias_stages: Optional[int] = None,
        padded_tail_form: Optional[int] = None,
        dead_subtile_quiet: Optional[int] = None,
        bias_max_form: Optional[int] = None,
        softmax_wgs: Optional[int] = None,
    ) -> None:
        """Capture the common SDPA operation and tuning contract.

        ``n_chunk`` (APPENDED, optional; the resident-bias arm): the chunk length of the (n, n+1) PAIR work item -- a
        work item is (a 128-row q tile, a head, up to ``n_chunk`` consecutive entries of the flattened batch axis), the
        two softmax sub-tiles are rows n / n + 1 of that q tile and the bias row-block is loaded once per item.  None =
        the heuristics' pick (``config_sm107.sm107_n_chunk_pick``: 0 = the one-batch-entry item unless the bias is
        batch-broadcast on >= 2 rows at cga1); an explicit value >= 1 is honored on a plan that qualifies for the pair
        item and declined typed otherwise.  Performance-only (every value computes the same function).

        ``bias_max_form`` (APPENDED, optional): the ROW-MAX FORM of the bias / kv_mask softmax body -- 0 = the software row max over
        the biased, masked scores, 1 = the hardware upper bound (the fused ``tcgen05.ld.red...max`` over the raw scores plus the thread's
        per-row bias-tile max; any m >= the true row max is valid for the online softmax).  None = the pick
        (``config_sm107.sm107_bias_max_form_pick``: the measured per-form default of ``sm107_bias_max_form_default`` -- 1 on the
        kv_mask-only form, 0 with a bias); 1 is honoured with a bias and / or a boolean kv_mask and declined typed with an additive
        kv_mask vector or without either arm.  Performance-only (the same function, within fp32 rounding of the exp2 arguments).

        ``sample_bias`` / ``sample_kv_mask`` (APPENDED, optional): the additive attention bias
        ``[1|B, 1|H_q, S_q, S_kv]`` in the Q/K/V dtype (added to the scaled scores before the softmax,
        broadcast over the dims declared 1) and the per-key boolean mask ``[1|B, 1, 1, S_kv]``
        (``torch.bool``; a False key is masked for every query, a query whose every key is masked reads
        O = 0 / LSE = -inf).  Plan-time state like ``sample_lse``: their presence is a compile-time arm of
        the kernel (the resident bias row-block of the Rubin f16/bf16 32- / 64-wide tiles).  Served on
        dense, unsplit, unpaged plans without the score side outputs, at an S_kv the block holds
        (``config_sm107.sm107_bias_resident_s_kv_max``); every other selection declines at ``check_support``.
        ``bias_mode`` (APPENDED, optional): the bias FORM -- 1 = the resident row-block (declined typed above its budget), 2 = the
        STREAMED ring (one bias tile per kv tile through a 2- / 3-stage SMEM ring, any S_kv the operands leave room for; the
        fallback the heuristics pick above the resident budget); None = the pick (``config_sm107.sm107_bias_mode_pick``).
        ``bias_stages`` (APPENDED, optional): the streamed ring's depth, 2 or 3 (None = the deepest that fits); a request on a
        plan that runs the resident block is a typed decline.
        ``softmax_wgs`` (APPENDED, optional): the number of softmax warp groups of the Rubin f16/bf16 d128-file body -- 2 (the
        classic 16-warp body) or 3 (the third softmax warp group: a 20-warp body whose pair work item takes THREE rows of the chunk per
        step; an EXPERIMENT, served on the (n, n+1) pair item at tile_n 64 / cga 1 only and declined typed elsewhere).  None = 2 (or
        the sweep seam ``config_sm107._SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS``).  Performance-only (every value computes the same function).

        ``sample_score_max`` / ``sample_score_sum_exp`` (APPENDED, optional): the graph's two score
        side outputs -- fp32 ``(B, H_q, S_q[, 1])`` row statistics in natural units, the softmax
        row max (the sink folded in as one more column) and the fp32 row-sum of ``exp(s - max)`` the
        LSE is built from, so ``Stats == score_max + ln(score_sum_exp)`` to fp32 rounding; a keyless
        row reads ``-inf`` / ``0`` (with a sink ``sink_logit`` / ``1``), like its LSE.  Plan-time state
        exactly like ``sample_lse``: None compiles the store out of the kernel.  Served by the f16/bf16
        d128 prefill kernels (SM100 / SM107) on dense, unsplit, unpaged plans of the 2-CTA tile; every
        other selection declines at ``check_support``.

        ``ragged_divisors`` (THD only): the (Q, O, Stats) elements-per-token
        divisors of the graph's ragged-offset tensors -- ``row_elems /
        ragged_offset_multiplier`` (``engines._thd_decode_leg_divisors``), so the
        kernel reads ``offset[b] // divisor`` as the token row. Read by the SM100
        decode tile's ragged-Q leg, which addresses the packed rows from the
        ragged offsets themselves; the prefill tile's THD leg never reads them.

        ``sample_sf_o`` (per-tensor FP8 only): the block-scaled O scale-factor
        output. With an FP4 (``torch.float4_e2m1fn_x2``) O it holds one E4M3
        scale per 16 d elements; with an E4M3 O it holds one UE8M0 scale per
        32 (MXFP8 output). Rank-4 ``[B, H, rows, cols]`` in the F8_128x4 atom
        order, declared either as per-(b, h) planes (BHRC strides, rows >=
        S_q rounded up to 128) or token-major (BRHC strides: one
        ``[B*rows, H*cols]`` matrix a downstream GEMM consumes).

        ``sample_scale_o`` (MXFP8 input only -- the per-tensor FP8 op's
        ``scale_o`` is a required execute operand): declares that ``execute()``
        binds a 1-element fp32 ``scale_o``, the FP4 global scale folded into the
        row normalization (optional with an E4M3 O). Its presence is a compile
        form of the kernel: without it the ``scale_o`` operand is
        None-specialized and the kernel folds an identity -- no device constant
        is allocated or filled for it (Rule 8), so a first execute under CUDA
        graph capture followed by an eager execute reads no half-initialized
        dummy.

        Optional operands are accepted by every adapter. A concrete
        implementation that cannot lower one raises :class:`NotImplementedError`
        from ``check_support``.

        ``sample_gate``: a logical BHSD ``(B, H_q, S_q, D_v)`` tensor of O's
        shape enables the FUSED EPILOGUE GATE, ``O := O * sigmoid(GATE)`` --
        the kernel TMA-stages the gate tile after its KV loop and multiplies
        the fp32 pre-cast accumulator by ``sigmoid`` in the correction
        epilogue (the ``mul(O_v, sigmoid(G))`` tail of a graph, or a gated
        attention block's gate branch). Compile-time: it selects a distinct
        kernel specialization (``TemplateParams.epilogue_gate``), so an
        ``execute()`` must then pass ``gate=`` and one built without it must
        not. Served by the Rubin (SM107) d256 f16/bf16, per-tensor FP8 and
        block-scale MXFP8 kernels only; every other adapter / flavor declines
        it with :class:`NotImplementedError` from ``check_support``. The gate
        is Q's dtype on the half kernels and ``bfloat16`` on the quantized
        (FP8 / MXFP8) ones. On the quantized kernels the O quantization applies
        to the GATED value, while ``Amax_O`` (when requested) is the amax of
        the UNGATED normalised O -- the sdpa node's own output, which precedes
        the ``sigmoid``/``mul`` tail on the graph -- so it is independent of
        ``gate``; the graph path and this adapter share that one contract (in
        ``scale_o`` units on FP8; in the O's own units on MXFP8, which has no
        per-tensor ``scale_o``).

        ``has_amax_o``: quantized (FP8 / MXFP8) path only. ``True`` (default) keeps the
        legacy contract -- ``amax_o`` at ``execute()`` is optional and an
        unrequested amax lands in a cached dummy slot. ``False`` records that
        the graph has NO ``Amax_O`` output: a kernel that carries the
        ``has_amax`` compile knob folds the atomicMax out entirely (the d256
        Rubin FP8 and MXFP8 kernels), and ``execute(amax_o=...)`` is then a
        :class:`ValueError`. Kernels without the knob keep the legacy dummy
        slot, so the flag is honoured where it can be and harmless elsewhere.
        On SM100 MXFP8 BF16-PV, ``TemplateParams.emit_amax_o`` emits Amax_O
        only when ``sample_amax_o`` declares the buffer AND ``has_amax_o`` is
        true. Both host entry points specialize away the omitted operand.
        The graph path derives both from the same output declaration.

        Paged KV (``paged_page_size > 0``): ``sample_k`` / ``sample_v`` are the
        page POOLS ``[num_pages, H_kv, page_size, D]`` — HND compact, or NHD
        storage declared through the strides; the layout is nothing but those
        strides, bound at execution — execute() takes the ``(B, max_pages)``
        int32 block tables, and ``paged_max_seq_len_kv`` (required) is the logical
        S_kv the masks clamp against (per-batch lengths are mandatory:
        ``seq_kv_lens_present``). ``paged_table_stride`` / ``paged_table_v_stride``
        are the tables' declared ``(batch, page)`` strides, bound explicitly so any
        layout (row-major, batch-innermost, padded) binds as a view; None =
        row-major compact.
        """

        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), 3, "lse")
        # Score side outputs (score_max / score_sum_exp): the Stats geometry without its unit axis; the
        # declared strides a non-contiguous buffer binds with are recorded by check_support per output.
        self.score_max_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_score_max, name="score_max"), 3, "score_max")
        self.score_sum_exp_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_score_sum_exp, name="score_sum_exp"), 3, "score_sum_exp")
        self._score_strides: dict = {}
        # Additive bias / per-key boolean mask (the resident-block arm); None = the arm is compiled out.
        self.bias_desc = self._make_tensor_desc(sample_bias, name="bias")
        self.kv_mask_desc = self._make_tensor_desc(sample_kv_mask, name="kv_mask")
        self.n_chunk = None if n_chunk is None else int(n_chunk)
        # The bias FORM (APPENDED; the resident-bias arm): None = the pick (config_sm107.sm107_bias_mode_pick: the resident block
        # where it holds S_kv, else the streamed ring), 1 / 2 = a request honoured where that form fits and declined typed elsewhere;
        # `bias_stages` (APPENDED) = the streamed ring's depth, 2 / 3 (None = the derived deepest), a request on a streamed plan only.
        self.bias_mode = None if bias_mode is None else int(bias_mode)
        self.bias_stages = None if bias_stages is None else int(bias_stages)
        # `padded_tail_form` (APPENDED) = the PLAIN padded body's tail form on the Rubin small tiles (TemplateParams.padded_tail_form): None =
        # the pick (config_sm107.sm107_padded_tail_form_pick), 0 / 1 = a request honoured on the eligible plan (padded, no bias / kv_mask,
        # no band, the 32- / 64-wide tiles) and declined typed elsewhere.
        self.padded_tail_form = None if padded_tail_form is None else int(padded_tail_form)
        # `dead_subtile_quiet` (APPENDED) = the dead-sub-tile quiet body of the one-batch-entry item on the Rubin small tiles
        # (TemplateParams.dead_subtile_quiet): None = the pick (config_sm107.sm107_dead_subtile_quiet_pick), 0 / 1 = a request honoured on
        # the eligible plan (the 32- / 64-wide f16/bf16 tiles, no bias, no PackGQA / THD / gate) and declined typed elsewhere.
        self.dead_subtile_quiet = None if dead_subtile_quiet is None else int(dead_subtile_quiet)
        # The row-max form (APPENDED): None = the pick, 0 / 1 = a request (check_support validates it against the arms this plan carries).
        self.bias_max_form = None if bias_max_form is None else int(bias_max_form)
        # `softmax_wgs` (APPENDED) = the softmax warp-group count of the d128-file body, 2 / 3 (None = 2 or the sweep seam); 3 is honoured on the
        # TILE_N-64 pair item at cga1 and declined typed elsewhere.
        self.softmax_wgs = None if softmax_wgs is None else int(softmax_wgs)
        self.amax_o_desc = self._make_tensor_desc(sample_amax_o, name="amax_o")
        # Fused epilogue gate (O := O * sigmoid(GATE)); None = ungated specialization.
        self.gate_desc = self._make_tensor_desc(sample_gate, name="gate") if sample_gate is not None else None
        # The direct hybrid specialization already derives EMIT_AMAX_O from
        # sample_amax_o. Keep its prepared reset/argument contract identical.
        self.has_amax_o = bool(has_amax_o and (not pv_bf16 or sample_amax_o is not None))
        # The gate's BSHD strides the artifact is COMPILED at (None = compact),
        # decided by check_support exactly like the Q/K/V/O zero-copy strides.
        self.sf_o_desc = self._make_tensor_desc(sample_sf_o, name="sf_o")
        # MXFP8 input: whether execute() binds scale_o (see the constructor doc);
        # False compiles the operand out and the kernel folds an identity.
        self.has_scale_o = sample_scale_o is not None
        # Block-scaled O: 0 (plain), 16 (FP4 O + E4M3 SF), 32 (E4M3 O + UE8M0
        # SF); set by check_support together with the SF_O layout geometry.
        self.o_block_scale = 0
        self._sfo_geometry: Optional[tuple[int, int, int, int]] = None

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        # window_size_left is an offset W ("keep k in [q-W, q]"); callers pass W = L - 1 for a cuDNN window length L.
        self.window_size_left = window_size_left
        # window_size_right is the diagonal-band right bound R ("keep k in
        # [.., q+R]", cuDNN diagonal_band_right_bound): the causal upper limit
        # widened by R columns. None = no right band; requires is_causal (the
        # band is the causal diagonal, possibly widened).
        self.window_size_right = window_size_right
        # The ONE canonical band every adapter lowers (same model as the
        # analyzer facts / config TemplateParams): per-side offsets from the
        # diagonal, None = unbounded. is_causal means "right bound 0";
        # window_size_right widens it (check_support validates it requires
        # is_causal). The padding mask stays orthogonal (seq_kv_lens_present).
        self.window_left: Optional[int] = window_size_left
        self.window_right: Optional[int] = (window_size_right or 0) if self.is_causal else None
        self.scale_softmax = scale_softmax
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
        # O := 0 / LSE := -inf. The per-batch Q lengths are a SEPARATE
        # (B,)-int32 kernel parameter (like cuDNN's SEQLEN_Q pointer and
        # FA's seqused_q) bound directly at execute — no packing, no
        # per-execute copies. Dense-only.
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # cu_seq_len form (cuDNN 9.24+): the corresponding seq-lens execute
        # argument arrives as a (B+1,)-int32 PREFIX-SUM tensor instead of
        # (B,) per-batch lengths. THD-only today: the setup kernels consume
        # either form on device (issue #552); the dense
        # kernels have no CU read mode yet (check_support rejects).
        self.cu_seq_q_lens = bool(cu_seq_q_lens)
        self.cu_seq_kv_lens = bool(cu_seq_kv_lens)
        self.has_sink = bool(has_sink)
        # Base-2 Stats (sdpa(stats_use_log2=True)): a compile-time epilogue
        # specialization. Under split_kv > 1 the per-split partials stay
        # natural (the combine merges them that way) and only the combine's
        # final LSE converts.
        self.stats_log2 = bool(stats_log2)
        self.thd = bool(thd)
        self.ragged_divisors = (1, 1, 1) if ragged_divisors is None else tuple(int(m) for m in ragged_divisors)
        self.ragged_offsets_int64 = bool(ragged_offsets_int64)
        self.decode_q_tile_max: Optional[int] = None if decode_q_tile_max is None else int(decode_q_tile_max)
        if self.decode_q_tile_max is not None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import D256_DECODE_SINGLE_UNIT_Q_TILES as _widths

            if self.decode_q_tile_max not in _widths:
                raise ValueError(
                    f"decode_q_tile_max must be one of {_widths} (the Rubin d256 decode tile's single-unit widths) or None; got {decode_q_tile_max}"
                )
        self.use_pdl = bool(use_pdl)
        # THD Stats declared WITHOUT ragged offsets: per-batch padded (b, s_max, h)
        # rows (FlashInfer's form). The kernel stores per batch; the adapter fills
        # the rows past each sequence's length with -inf, as the backend does.
        self.thd_stats_padded = bool(thd_stats_padded)
        # Caller-declared packed token totals. These only ever TIGHTEN the
        # execute-time token extents (they are min'd against the capacity the
        # bound buffers can address), so a wrong or stale value cannot make a
        # launch address memory the caller does not own -- it can only make it
        # address less. None = not declared; the extent falls back to the
        # buffer-derived capacity.
        self.max_total_seq_len_q = None if max_total_seq_len_q is None else int(max_total_seq_len_q)
        self.max_total_seq_len_kv = None if max_total_seq_len_kv is None else int(max_total_seq_len_kv)
        # MXFP8: FP8 (E4M3/E5M2) Q/K/V in, half (BF16/FP16) O out. dtype_o overrides
        # the output dtype; None inherits Q's dtype. _fp8 is set in check_support once
        # Q's dtype is known.
        self.dtype_o = dtype_o
        self._fp8 = False
        # Per-tensor FP8 (sdpa_fp8) vs block-scale MXFP8 (sdpa_mxfp8); both use FP8 Q/K/V.
        self._pertensor = bool(pertensor_fp8)
        # Direct-adapter-only experiment. It deliberately has no graph node or
        # engine capability: its purpose is to isolate the QK-MXFP8/BF16-PV
        # kernel tradeoff before exposing an API contract. It writes BF16 O;
        # Amax_O is compiled only when ``sample_amax_o`` declares that output.
        self.pv_bf16 = bool(pv_bf16)
        self._device_cc = None  # (major, minor); set in check_support
        # Tuning-knob choice, already validated against the engine's
        # Capabilities domain by the probe (engines.mismatch). None means the
        # caller stated NO preference: the graph path always arrives with an
        # explicit value (the heuristic emits complete assignments), so a None
        # here is the standalone-wrapper tier, where compile() derives the
        # policy itself. An explicit value — including NATURAL — is honored
        # verbatim, never re-derived.
        self.sched_policy = None if sched_policy is None else int(sched_policy)
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)
        self.cga = None if cga is None else int(cga)
        # Unlike scheduler/CGA defaults, standalone split_kv=None means unsplit;
        # graph heuristics pass an explicit split count when splitting wins.
        self.split_kv = 1 if split_kv is None else int(split_kv)
        # softmax_precision=HALF: the f16x2 exponent arm of the cc 10.7 FP8 / MXFP8 kernels (every flavor);
        # anything else non-None is rejected in check_support.
        self.softmax_precision = softmax_precision
        # Q already carries attn_scale * log2(e): the kernel applies no scale (and fuses the shift with the
        # f32->f16 convert under softmax_precision=HALF).  Compile-time; served by the cc10.7 MXFP8 and half
        # kernels (_SM107_PREFOLDED_FLAVORS), see check_support.  scale_softmax must stay None -- it would be
        # silently ignored.
        self.softmax_scale_prefolded = bool(softmax_scale_prefolded)
        self._scale_softmax_requested = scale_softmax is not None
        self.pack_gqa = bool(pack_gqa) if pack_gqa is not None else False
        self.paged_page_size = int(paged_page_size or 0)
        self.paged_max_seq_len_kv = None if paged_max_seq_len_kv is None else int(paged_max_seq_len_kv)
        self.paged_table_stride = None if paged_table_stride is None else tuple(int(s) for s in paged_table_stride)
        self.paged_table_v_stride = None if paged_table_v_stride is None else tuple(int(s) for s in paged_table_v_stride)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self._initialize_implementation()
        self._logger.debug("__init__ completed")

    @property
    def paged(self) -> bool:
        return self.paged_page_size > 0

    @abstractmethod
    def _initialize_implementation(self) -> None:
        """Initialize state private to specific implementations."""

    # -- THD declared-stride binding ------------------------------------------
    # A THD tensor may DECLARE a wider token stride than the packed h*d — e.g.
    # a K/V view of a kv-interleaved [T, 2, H, D] buffer (token stride 2*h*d),
    # the layout torch.nn.attention.varlen users produce by slicing a fused KV
    # projection. The SM120 kernels (f16 and per-tensor FP8) address declared
    # strides NATIVELY (layout-driven offset math + TMA-encoded strides);
    # declarations the hardware cannot express are REJECTED in check_support
    # — no normalization-copy fallback (AGENTS.md Hard Rule 2) — so the router
    # picks an engine that honors them instead. The SM100 FP8/MXFP8 kernels
    # still serve only the packed contract (_thd_check_strides_packed).

    def _checked_padded_lse(self, lse_tensor):
        """A per-batch padded Stats buffer holds exactly B*H_q*s_max fp32 values;
        the declared strides are then applied over it (the caller may hand any
        view of that storage -- rank-4 graph Stats, (b, s_max, h) -- so the
        element count, not the shape, is the contract). The device is checked
        first: the seed hands ``data_ptr()`` to a raw CUDA fill on the launch
        stream, which would fault on a host or foreign-device pointer."""
        self._value_error_if(
            lse_tensor.device != self.q_desc.device or lse_tensor.device.type != "cuda",
            f"lse_tensor must be on {self.q_desc.device}; got {lse_tensor.device}",
        )
        self._value_error_if(dtype_name(lse_tensor) != "float32", f"lse_tensor must be float32; got {lse_tensor.dtype}")
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse_tensor.numel()}",
        )
        return lse_tensor

    @staticmethod
    def _thd_declared(desc: TensorDesc):
        """(token, head, elem) strides a THD tensor declares, and whether they
        are the packed contract (h*d, d, 1)."""
        h, d = desc.shape[1], desc.shape[3]
        st = (int(desc.stride[2]), int(desc.stride[1]), int(desc.stride[3]))
        return st, st == (h * d, d, 1)

    def _thd_descs(self) -> tuple:
        """The THD-packed operands: Q/K/V/O, or just Q/O when K/V are paged pools."""
        return (self.q_desc, self.o_desc) if self.paged else (self.q_desc, self.k_desc, self.v_desc, self.o_desc)

    def _thd_check_strides_native(self) -> None:
        """Reject THD stride declarations the kernels cannot address
        natively: TMA's 16-byte global-stride rule — the head dim must be
        innermost-contiguous (elem stride 1) and the token/head strides
        multiples of ``16 // itemsize`` elements (which also keeps every
        per-sequence ragged base 16-byte aligned). Whole-token gaps always
        qualify for supported head dims; sub-token gaps only in 16-byte
        multiples. The strides must also COVER the tensor (head >= d,
        token >= h*head): an overlapping declaration would alias distinct O
        rows onto the same storage (a write race) and is outside the
        kernels' addressing contract.  The epilogue gate (when declared) is
        addressed through the same packed map as Q / O and takes the same rule."""
        for desc in self._thd_descs() + ((self.gate_desc,) if self.gate_desc is not None else ()):
            (ts, hs, es), _ = self._thd_declared(desc)
            h, d = desc.shape[1], desc.shape[3]
            # The 16-byte TMA rule in this tensor's OWN element units: 8 at
            # 2 B/elem (f16/bf16), 16 at 1 B/elem (fp8), 4 at 4 B/elem.
            quantum = 16 // desc.dtype.itemsize
            self._not_implemented_error_if(
                es != 1 or ts % quantum != 0 or hs % quantum != 0 or hs < d or ts < h * hs,
                f"{desc.name} THD strides {tuple(desc.stride)} are not TMA-expressible "
                f"(head dim must be innermost-contiguous, token/head strides 16-byte — "
                f"{quantum}-element — multiples, and non-overlapping: head stride >= {d}, "
                f"token stride >= heads * head stride)",
            )

    def _thd_check_strides_packed(self) -> None:
        """The SM100 FP8/MXFP8 THD path serves only the packed contract (those
        kernels are not audited for declared strides) — decline anything else
        rather than adapt (AGENTS.md Hard Rule 2)."""
        for desc in self._thd_descs():
            _, packed = self._thd_declared(desc)
            self._not_implemented_error_if(
                not packed,
                f"{desc.name}: non-packed THD strides {tuple(desc.stride)} are not supported by the SM100 FP8 path",
            )

    def _thd_decl(self, desc: TensorDesc) -> tuple:
        """``(h, d, token_stride, head_stride, elem_stride, row_span)`` of a THD declaration."""
        h, d = desc.shape[1], desc.shape[3]
        (ts, hs, es), _ = self._thd_declared(desc)
        return (h, d, ts, hs, es, (h - 1) * hs + (d - 1) * es + 1)

    def _scratch_base(self, workspace, label: str, required: Optional[int] = None) -> int:
        """The device address of the caller's per-execute scratch, validated
        against ``scratch_workspace_bytes()`` (size and 16-byte alignment).
        FROST executor contract (``engine._FrostSdpaFwdPlan``): scratch is fixed
        offsets into the caller's buffer, never a per-execute allocation."""
        nbytes = workspace.numel() * workspace.element_size()
        if required is None:
            required = self.scratch_workspace_bytes()
        if nbytes < required:
            raise ValueError(
                f"cudnn.sdpa: {label} requires a {required}-byte workspace; the provided buffer has {nbytes} bytes (size it with graph.get_workspace_size())"
            )
        base = workspace.data_ptr()
        if base % 16 != 0:
            raise ValueError(f"cudnn.sdpa: {label} workspace must be at least 16-byte aligned; got data_ptr=0x{base:x}")
        return base

    # -- block-scaled O (sf_o) ------------------------------------------------

    def _sf_o_geometry(self, block: int, d_v: int) -> tuple[int, int, int, int]:
        """Kernel offsets for the declared SF_O tensor: ``(plane_stride, row_off_b, col_off_h, cols)``.

        The kernel writes byte ``(r, c)`` of a ``[rows, cols]`` matrix in the
        F8_128x4 atom order at ``plane + (r//128)*128*cols + (c//4)*512 +
        (r%32)*16 + ((r//32)%4)*4 + c%4`` with ``r = q_row + b*row_off_b``,
        ``c = block_idx + h*col_off_h``, ``plane = (b*H + h)*plane_stride``.
        Per-(b,h) planes (BHRC strides) and the token-major matrix (BRHC
        strides) are the two declared layouts.
        """
        d = self.sf_o_desc
        b, h_q = int(self.q_desc.shape[0]), int(self.q_desc.shape[1])
        s_q = int(self.q_desc.shape[2])
        c_need = d_v // block
        self._value_error_if(len(d.shape) != 4, f"sf_o must be rank-4 [B, H, rows, cols]; got {tuple(d.shape)}")
        B_, H_, R, C = (int(x) for x in d.shape)
        st = tuple(int(x) for x in d.stride)
        self._value_error_if((B_, H_) != (b, h_q), f"sf_o batch/head extents {(B_, H_)} must match Q {(b, h_q)}")
        self._value_error_if(C % 4 != 0 or C < c_need, f"sf_o cols must be a multiple of 4 and >= d_v/{block} = {c_need}; got {C}")
        self._value_error_if(st[3] != 1, f"sf_o innermost stride must be 1; got {st}")
        rows_pad = -(-s_q // 128) * 128
        if st[2] == C and st[1] == R * C and st[0] == H_ * R * C:
            # per-(b, h) planes: each plane is its own 128-row-padded atom matrix
            self._value_error_if(R < rows_pad or R % 128 != 0, f"sf_o plane rows must be S_q rounded up to 128 (>= {rows_pad}, %128); got {R}")
            return (R * C, 0, 0, C)
        if st[1] == C and st[2] == H_ * C and st[0] == R * H_ * C:
            # token-major: one [B*R, H*C] matrix; rows of batch b start at b*R
            self._value_error_if(R < s_q, f"sf_o token-major rows must cover S_q = {s_q}; got {R}")
            self._value_error_if((H_ * C) % 4 != 0, f"sf_o token-major needs H*cols % 4 == 0; got {H_ * C}")
            return (0, R, C, H_ * C)
        raise ValueError(f"sf_o strides {st} are neither per-(b,h) planes (BHRC) nor token-major (BRHC) for dims {(B_, H_, R, C)}")

    def _checked_lse_view(self, lse_tensor: torch.Tensor) -> torch.Tensor:
        """Validate a caller-provided LSE buffer and return the kernel's (B, H_q, S_q) view.

        The logical contract is exactly ``B*H_q*S_q`` fp32 elements. Graph
        Stats commonly arrive as a rank-4 ``(B, H_q, S_q, 1)`` view, which is
        reinterpreted as the declared rank-3 LSE layout without copying. The
        kernel writes through the returned view, so a silent ``reshape`` copy
        of a non-contiguous buffer would leave the caller's Stats unwritten.
        Dense adapters therefore record ``_lse_stride`` and rebuild that
        declared view directly over the caller's storage.
        """
        return self._checked_row_stat_view(lse_tensor, "lse_tensor", getattr(self, "_lse_stride", None))

    def _checked_row_stat_view(self, tensor: torch.Tensor, name: str, stride: Optional[tuple]) -> torch.Tensor:
        """The (B, H_q, S_q) fp32 view of one per-row statistic buffer (the LSE, ``score_max`` or
        ``score_sum_exp``) over the caller's storage: ``stride`` is the declared non-contiguous layout
        recorded at check_support (None = compact).  See :meth:`_checked_lse_view`."""
        shape = (self.batch_size, self.h_q, self.s_q_max)
        if tensor.dtype == torch.float32 and tuple(tensor.shape) == shape:
            if (stride is None and tensor.is_contiguous()) or (stride is not None and tuple(tensor.stride()) == stride):
                return tensor
        self._value_error_if(
            dtype_name(tensor) != "float32",
            f"{name} must be float32; got {tensor.dtype}",
        )
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            tensor.numel() != expected,
            f"{name} must have B*H_q*S_q = {expected} elements; got {tensor.numel()}",
        )
        if stride is None:
            self._value_error_if(
                not tensor.is_contiguous(),
                f"{name} must be contiguous (the kernel writes through this buffer)",
            )
            return tensor.view(shape)
        if tuple(tensor.shape) == shape and tuple(tensor.stride()) == stride:
            return tensor
        try:
            return tensor.as_strided(shape, stride, tensor.storage_offset())
        except RuntimeError as exc:
            raise ValueError(
                f"{name} backing storage is too small for declared shape {shape}, stride {stride}, and storage_offset {tensor.storage_offset()}"
            ) from exc

    def _checked_sinks_1d(self, sinks: torch.Tensor) -> torch.Tensor:
        """Validate caller-provided sink logits and return the kernel's (H_q,) fp32 view.

        Strictly a view: the kernels consume fp32 sinks directly, and an
        implicit ``.to(float32)`` here would allocate and launch a cast kernel
        on the execute hot path (and break CUDA-graph pointer stability).
        """
        self._value_error_if(
            dtype_name(sinks) != "float32",
            f"sinks must be float32; got {sinks.dtype}",
        )
        self._value_error_if(
            sinks.numel() != self.h_q,
            f"sinks must have H_q = {self.h_q} elements; got {sinks.numel()}",
        )
        self._value_error_if(
            not sinks.is_contiguous(),
            "sinks must be contiguous (bound to the kernel as a flat (H_q,) view)",
        )
        return sinks.reshape(-1)

    def _checked_seq_lens(self, seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate caller-provided per-batch lengths and return the kernel's (B,) int32 view.

        Strictly a view: an implicit ``.to(torch.int32)`` here would allocate
        and launch a cast kernel on the execute hot path (and break CUDA-graph
        pointer stability).
        """
        self._value_error_if(
            dtype_name(seq_lens) != "int32",
            f"{name} must be int32; got {seq_lens.dtype}",
        )
        self._value_error_if(
            seq_lens.numel() != self.batch_size,
            f"{name} must have B = {self.batch_size} elements; got {seq_lens.numel()}",
        )
        self._value_error_if(
            not seq_lens.is_contiguous(),
            f"{name} must be contiguous (bound to the kernel as a flat (B,) view)",
        )
        return seq_lens.reshape(-1)

    def _checked_cu_seq_lens(self, cu_seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a caller-provided (B+1,)-int32 prefix-sum tensor (cu_seq_len form).

        Strictly a view, like :meth:`_checked_seq_lens`. The prefix-sum
        INVARIANTS (non-decreasing; any base — the setup kernel normalizes by
        subtracting element 0) are runtime values and caller contract: a
        validation that needs a device read is not a validation (Rule 3).
        """
        self._value_error_if(
            cu_seq_lens.dtype != torch.int32,
            f"{name} must be int32; got {cu_seq_lens.dtype}",
        )
        self._value_error_if(
            cu_seq_lens.numel() != self.batch_size + 1,
            f"{name} must have B + 1 = {self.batch_size + 1} elements (prefix sums); got {cu_seq_lens.numel()}",
        )
        self._value_error_if(
            not cu_seq_lens.is_contiguous(),
            f"{name} must be contiguous (read as a flat (B+1,) view)",
        )
        return cu_seq_lens.reshape(-1)

    def _check_seq_lens_contract(self, seq_q_lens, seq_kv_lens) -> None:
        """Reject seq-length tensors inconsistent with the compiled specialization.

        Like sinks, presence is a compile-time specialization: substituting a
        zeros dummy for a required tensor masks every row (silently wrong
        output), and lengths passed to a specialization compiled without them
        are silently ignored. THD is exempt — it always requires both (they
        source the packed cu_seqlens metadata).
        """
        if self.thd:
            self._value_error_if(
                seq_q_lens is None or seq_kv_lens is None,
                "THD execute requires seq_q_lens and seq_kv_lens",
            )
            return
        self._value_error_if(
            self.seq_kv_lens_present and seq_kv_lens is None,
            "seq_kv_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_kv_lens_present and seq_kv_lens is not None,
            "this specialization was compiled without per-batch KV lengths; construct the API with seq_kv_lens_present=True",
        )
        self._value_error_if(
            self.seq_q_lens_present and seq_q_lens is None,
            "seq_q_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_q_lens_present and seq_q_lens is not None,
            "this specialization was compiled without per-batch Q lengths; construct the API with seq_q_lens_present=True",
        )

    @abstractmethod
    def scratch_workspace_bytes(self) -> int:
        """Return the per-execution scratch requirement for this implementation."""

    # -- KV-split shared helpers (SM100 + SM120 dense split paths) -----------

    def _prepared_operand_layout(self, desc):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import dense_bind_strides

        shape, stride, width = tuple(desc.shape), tuple(desc.stride), desc.dtype.itemsize
        if desc is self.o_desc and self.o_block_scale == 16:
            import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry

            geometry = storage_geometry(shape, stride, cudnn.data_type.FP4_E2M1)
            if geometry is None:
                return None
            shape, stride = geometry
            width = 1
        return dense_bind_strides(shape, stride, width)

    def _can_prepare_block_output(self):
        return bool(self.o_block_scale) and not (
            self.thd or self.paged or self.split_kv > 1 or self.pack_gqa or self.gate_desc is not None or getattr(self, "pv_bf16", False)
        )

    def _execute_fp8_prepared(self, q, k, v, o, lse, sinks, q_lens, kv_lens, scales, scale, workspace, stream, block_table=None, block_table_v=None):
        """Prepared FP8 always uses caller scratch, including omitted scales / amax.

        Standalone callers allocate scratch_workspace_bytes() before execute;
        graph callers use get_workspace_size(). No plan-owned scalar buffers.
        """
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import (
            _NATIVE_DENSE_ROLES,
            _native_quant_roles,
            execute_native_dense_tensors,
            execute_native_thd_tensors,
            facts_of_tensor,
        )

        spec = self._thd_spec if self.thd else self._dense_spec
        if self.thd:
            # Standalone declarations fix both counts, including independent
            # prefix-sum forms. Graph binding may use a smaller effective batch.
            for name, lengths, count in (("seq_q_lens", q_lens, spec.n_q_lens), ("seq_kv_lens", kv_lens, spec.n_kv_lens)):
                if lengths is None or lengths.numel() != count:
                    raise ValueError(f"cudnn.sdpa: {name} must have {count} elements for this specialization")
        required = spec.quant.scratch_offset + ws_align(8)
        if workspace is None:
            raise ValueError(f"cudnn.sdpa prepared FP8 requires a {required}-byte workspace; pass scratch_workspace_bytes() bytes")
        ws = facts_of_tensor(workspace)
        if ws.device != (2, int(q.device.index or 0)) or not ws.contiguous or ws.numel * workspace.element_size() < required:
            raise ValueError(f"cudnn.sdpa: prepared FP8 workspace must cover {required} bytes on the Q device")
        stream = self._get_default_stream(stream)
        stream_int = int(stream)
        _ensure_current_context(stream_int, q.device.index)
        if spec.quant.block_output is None and scales.get("sf_o") is not None:
            raise ValueError("cudnn.sdpa: this specialization does not produce sf_o")
        if self.thd:
            if scales.get("gate") is not None:
                raise ValueError("cudnn.sdpa: this specialization was compiled without an epilogue gate")
            buffers = (q, k, v, o, q_lens, kv_lens, lse, sinks, block_table, block_table_v)
            buffers += tuple(scales.get(role) for role in _native_quant_roles(spec.quant))
            launched = execute_native_thd_tensors(spec, buffers, ws.ptr, stream, scale * math.log2(math.e))
        else:
            buffers = (q, k, v, o, lse, sinks, kv_lens, q_lens, block_table, block_table_v, scales.get("gate"))
            buffers += (None,) * (len(_NATIVE_DENSE_ROLES) - len(buffers))
            buffers += tuple(scales.get(role) for role in _native_quant_roles(spec.quant))
            execute_native_dense_tensors(spec, buffers, stream, scale * math.log2(math.e), ws.ptr)
            launched = True
        # Preserve the retired tensor path's diagnostics at the live entry.
        if self.thd and getattr(self, "_prepared_mxfp8", False):
            if launched:
                self._logger.debug("execute (MXFP8 THD) completed")
            else:
                self._logger.debug("execute (MXFP8 THD): no addressable Q token, nothing to do")
        elif self.thd:
            if launched:
                self._logger.debug("execute (FP8 per-tensor THD) completed")
            else:
                self._logger.debug("execute (FP8 THD): no addressable Q token, nothing to do")
        self._logger.debug("execute (prepared FP8) completed")

    def _o_dtype(self):
        """The torch dtype O is STORED as.

        Read the O DESCRIPTOR, not self.dtype_o: the latter is a cudnn.data_type
        enum on some rows (SM120 fp8) and a torch dtype on others, so comparing
        it against a torch dtype silently misclassifies those rows.  self.dtype
        is the fp8 INPUT type on the quantized rows, so it cannot stand in
        either; it is only the fallback for the non-quantized rows, where O
        follows Q.
        """
        o_dtype = getattr(self.o_desc, "dtype", None) if self._fp8 else self.dtype
        if o_dtype is None:
            o_dtype = self.dtype_o if self.dtype_o is not None else self.dtype
        return o_dtype

    def _quantized_split(self) -> bool:
        """A split whose O is stored quantized: the partials stay wide and the
        cast down to the FP8 O moves from the kernel epilogue to the combine."""
        return self.split_kv > 1 and self._fp8 and self._o_dtype() in _SM100_FP8_DTYPES

    def _split_scale_o(self) -> bool:
        """Whether that cast also applies a scalar scale_o.

        Only the per-tensor rows have one.  Block-scaled (MXFP8) O carries its
        scaling in the SF tensors, not a scalar, so its combine casts unscaled —
        exactly what its single-pass epilogue does.
        """
        return self._quantized_split() and self._pertensor

    def _partial_torch_dtype(self) -> torch.dtype:
        """The type the split kernels WRITE.

        Never narrower than O, so the rounding down to O's dtype happens once,
        on the recombined value, rather than once per split.  fp32 where the
        kernel can store its accumulator registers straight to the workspace
        (see :meth:`_fp32_partial_split`), half elsewhere."""
        if self._fp32_partial_split():
            return torch.float32
        # A quantized O has no half counterpart to inherit, so pick f16 -- its
        # 10-bit mantissa carries the partials more precisely than bf16's 7, and
        # the range that would favour bf16 is what scale_o already handles.
        return torch.bfloat16 if self._o_dtype() == torch.bfloat16 else torch.float16

    def _fp32_partial_split(self) -> bool:
        """Whether this launch's partials are fp32.

        Not a tuning knob: a split-capable SM100 kernel stores fp32 partials
        unconditionally.  The exclusions below are all the same fact -- those
        kernels are compiled WITHOUT the extra partial-tensor slot, so handing
        one the fp32 buffer passes an argument it does not declare:

        * SM120: its ``sO`` aliases ``sKV``, so there is no room to widen the O
          tile, and it keeps half partials.
        * SM107 (Rubin) outside per-tensor FP8 D64/D128/D192, half D128/D192, native
          half D512 2x2, and the d256 DECODE tile: the other siblings do not carry the FP32
          partial-output slot.  The decode tile (sm107/decode_d256_f16.py) is
          the SM100 body and stores fp32 partials unconditionally under a split
          (its O slot and o_partial slot are both Float32 then), so it takes
          this path like its SM100 twin -- sized half, its fp32 stores would
          overrun the partial slab and the combine would read half words from
          fp32 bytes (garbage O, the first Rubin graph-path split showed exactly
          that); native per-tensor D256 prefill also uses FP32 partials.
        * MXFP8 d512: sm100/prefill_d512_mxfp8 wires SplitHelpers but was
          written against the staged epilogue, so it keeps half partials until
          it is ported.

        Tracking the wired FLAVOR rather than the arch is what keeps this true
        to the kernels; test_every_split_capable_sm100_kernel_has_the_slot fails
        if a new kernel arrives split-capable without the slot."""
        if type(self).__name__ != "SdpaFwdDslSm100":
            return False
        if self.split_kv <= 1:
            return False
        if self._device_cc == (10, 7):
            return (
                bool(not self._fp8 and self.flavor in ((128, 128), (192, 128)))
                or bool(self._fp8 and self._pertensor and self.flavor in _SM107_FP8_SPLIT_SHAPES)
                or self._decode_q_tile() > 0
                or self._rubin_native_d512_leg()
            )
        if self._fp8 and not self._pertensor and self.flavor == (512, 512):
            return False  # MXFP8 d512: split-capable, no o_partial_f32 slot
        return True

    def _partial_dtype_tag(self) -> str:
        d = self._partial_torch_dtype()
        if d == torch.float32:
            return "f32"
        return "bf16" if d == torch.bfloat16 else "f16"

    def _o_itemsize(self) -> int:
        return self._partial_torch_dtype().itemsize

    def _combine_dtype_tag(self) -> str:
        # The combine reduces INTO the O dtype.  A quantized O is a legal split
        # target: this pass performs the single cast down to it, from half
        # partials (see _partial_dtype_tag).
        o_dtype = self._o_dtype()
        if o_dtype in _SM100_FP8_DTYPES:
            return "e5m2" if o_dtype == torch.float8_e5m2 else "e4m3"
        return "bf16" if o_dtype == torch.bfloat16 else "f16"

    def _split_workspace_bytes(self):
        rows = self.split_kv * self.batch_size * self.s_q_max * self.h_q
        return ws_align(rows * self.head_dim_v * self._o_itemsize()) + ws_align(rows * 4)

    def _split_partials(self, workspace, device, current_stream=None):
        """The split-major (O, LSE) partial buffers, carved from the caller's
        workspace when there is one and torch-allocated otherwise (standalone
        use, matching what the rest of this adapter does).

        The allocation happens ON the launch stream: the caching allocator tags
        a block with the stream it was allocated on, and the kernels that write
        and read these buffers run on ``current_stream``. Allocating on torch's
        current stream instead would leave a later free/reuse unordered against
        those launches."""
        rows = self.split_kv * self.batch_size
        o_shape = (rows, self.s_q_max, self.h_q, self.head_dim_v)
        lse_shape = (rows, self.h_q, self.s_q_max)
        # NOT o_like.dtype: a quantized O is stored narrower than its partials,
        # which stay wide (fp32 on SM100, half on SM120) so the reduction runs
        # wider than the final cast.
        o_dtype = self._partial_torch_dtype()
        if workspace is None:
            with _torch_stream_context(current_stream, device):
                return (
                    torch.empty(o_shape, dtype=o_dtype, device=device),
                    torch.empty(lse_shape, dtype=torch.float32, device=device),
                )
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), f"{type(self).__name__} (KV split)")
        o_part = carver.take(rows * self.s_q_max * self.h_q * self.head_dim_v, o_dtype).view(o_shape)
        lse_part = carver.take(rows * self.h_q * self.s_q_max, torch.float32).view(lse_shape)
        return o_part, lse_part

    @abstractmethod
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the compiled kernel on the common SDPA operand set.

        This is the keyword contract of the shared engine lowering
        (``lower_dsl_prefill`` in ``fwd/engines.py``), which drives every
        adapter through one ``execute_kwargs`` dict: the arguments above are
        always passed by keyword, and ``workspace`` is included iff
        ``scratch_workspace_bytes()`` is non-zero. Subclasses may extend the
        signature only with additional optional keyword arguments; an adapter
        whose engine capabilities accept FP8/MXFP8 graphs must also accept the
        FP8 operand set the lowering adds for those graphs (``sf_q/sf_k/sf_v``,
        ``descale_q/descale_k/descale_v``, ``scale_o``, ``amax_o`` — see :meth:`SdpaFwdDslSm100.execute`).
        An adapter whose capabilities claim the fused epilogue gate also
        accepts ``gate`` (the ``(B, H_q, S_q, D_v)`` tensor of a specialization
        constructed with ``sample_gate``; the lowering passes it by keyword
        only when the graph carries the ``mul(O_v, sigmoid(G))`` tail).
        """


class SdpaFwdDslSm100(SdpaFwdDsl):
    """SM100 (Blackwell) SDPA forward via the FROST DSL template kernels."""

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[tuple[int, int]] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._k_mod = None
        # The decode tile's ragged-Q leg (resolved in check_support): a THD
        # graph over PAGED K/V at S_q(max) == 1 on the (128, 128) f16 flavor
        # with cga=1 rides sm100/decode_d128_f16.py's RAGGED_Q mode -- the
        # dense prepared launch with the Q rows at the ragged offsets and the
        # split combine placing the final O / Stats rows -- instead of the
        # prefill tile's THD leg.
        self.thd_decode_leg = False
        self.packed_thd_split = False

    @property
    def _quantized_q_lens_abi(self) -> bool:
        """Whether the selected quantized kernel takes the dense Q-length slot
        positionally after amax_o: every fp8 / mxfp8 flavor on every arch line."""
        return self._fp8

    def _resident_arm_pair_eligible(self) -> bool:
        """Whether this BIAS plan qualifies for the (n, n+1) pair work item at cga1: a batch-broadcast bias on >= 2
        flattened rows (``config_sm107.sm107_bias_pair_form``, the engine row's and the heuristics' predicate)."""
        if self.bias_desc is None or self.n_chunk == 0:  # an explicit n_chunk=0 keeps the one-batch-entry item (and ITS budget)
            return False
        if self._device_cc != (10, 7):
            # The (n, n+1) pair item is not on the SM100 file yet: every bias plan there runs the one-batch-entry item (an explicit
            # n_chunk >= 1 is declined typed in check_support, never degraded).
            return False
        b, h_qo, s_qo, _ = (int(x) for x in self.q_desc.shape)
        # the form predicate AND a chunk length >= 1 (the request, else the rule / the sweep seam): the budget of the item that RUNS
        return _sm107_pair_item_active(
            int(self.bias_desc.shape[0]),
            b,
            h_qo,
            s_qo,
            _device_sm_count(self.q_desc.device),
            1,
            self.n_chunk,
            s_kv=int(self.k_desc.shape[2]),
            d_flavor=self.flavor[0],
            tile_n=self._effective_tile_n(),
        )

    def _kv_mask_form(self) -> int:
        """``TemplateParams.kv_mask_form`` of this plan from the kv_mask sample's dtype: 0 without one; the bit word for a
        byte mask (bool / int8 / uint8); the additive vector in the io dtype's 16-bit pairs or in fp32 (check_support
        validated the dtype, the whole-granule extent and the 16-byte strides)."""
        if self.kv_mask_desc is None:
            return 0
        if self.kv_mask_desc.dtype in (torch.bool, torch.int8, torch.uint8):
            return _KV_MASK_BOOL_WORD
        return _KV_MASK_ADDITIVE_F32 if self.kv_mask_desc.dtype == torch.float32 else _KV_MASK_ADDITIVE_16

    def _kv_padded(self) -> bool:
        """Whether this plan's KV axis is PADDED for the small-tile picks: real per-batch lengths (``seq_kv_lens_present``), or a
        dense S_kv off the 128-wide KV tile that no causal diagonal provably covers -- the plan then compiles the padded mask
        against the scalar S_kv (``_kv_tail_mask``, #1425; the lowering no longer synthesizes per-batch lengths for such a tail).
        Derived here rather than read back from ``_kv_tail_mask`` because ``check_support`` evaluates the picks before it sets that
        flag.  The engine row reads the same fact as ``facts.padded or _synth_kv_padding`` (rule 8b'), so the row's admission and
        the adapter's picks stay in lockstep; the kernel renders the SAME padded body either way (``_resolve_seqlen_kv`` returns
        the scalar when no lengths buffer is present)."""
        if self.seq_kv_lens_present:
            return True
        s_q, s_kv = int(self.q_desc.shape[2]), int(self.k_desc.shape[2])
        if s_kv % _SM100_TILE_N != 0:
            _br = int(self.window_size_right or 0)
            causal_covers_tail = bool(self.is_causal) and (
                (bool(self.causal_bottom_right) and _br == 0) or (not self.causal_bottom_right and s_q + _br <= s_kv)
            )
            return not causal_covers_tail
        return False

    def _effective_tile_n(self) -> int:
        """The KV tile width this plan renders: the request, else -- on the Rubin small tiles -- the heuristics' measured pick
        (``config_sm107.sm107_tile_n_pick``: 64 for a bias TOGETHER WITH a kv_mask, for a bias at one or two 128-wide kv tiles and for the
        plain / bias body on the padding leg, 128 everywhere else; the fp32 ADDITIVE vector WITH a bias lands on 64 there too, its only
        served width -- ``config_sm107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N``: the 128-wide rendering spills on the softmax role, and
        ``check_support`` declines an explicit 128 for it typed), else 128."""
        if self.tile_n is not None:
            return int(self.tile_n)
        if self.flavor in dict(_SM107_TILE_NS_BY_D_SHAPE) and self._device_cc == (10, 7) and not self._fp8:
            return _sm107_tile_n_pick(
                self.flavor[0],
                int(self.k_desc.shape[2]),
                has_bias=self.bias_desc is not None,
                has_kv_mask=self.kv_mask_desc is not None,
                padded=self._kv_padded(),
                domain=dict(_SM107_TILE_NS_BY_D_SHAPE)[self.flavor],
            )
        if self.flavor in dict(_SM100_TILE_NS_BY_D_SHAPE) and self._device_cc in _SM100_BIAS_CCS and not self._fp8:
            # The SM100 small tiles: config_sm100.sm100_tile_n_pick (128 unless the fp32 vector WITH a bias, which only the 64-wide tile serves, the
            # bias body on the PADDING leg inside the 64-wide resident budget (measured on GB200), or the BUDGET rule; the row's _bias_arm_decline
            # and the heuristics read the same arithmetic, rule 8b').
            return _sm100_tile_n_pick(
                self.flavor[0],
                int(self.k_desc.shape[2]),
                has_bias=self.bias_desc is not None,
                has_kv_mask=self.kv_mask_desc is not None,
                kv_mask_fp32=self.kv_mask_desc is not None and self.kv_mask_desc.dtype == torch.float32,
                padded=bool(self.seq_kv_lens_present),
                domain=dict(_SM100_TILE_NS_BY_D_SHAPE)[self.flavor],
            )
        return 128

    def _resident_arm_cga(self) -> int:
        """The width a BIAS plan runs at (a kv_mask-only plan stages no block and keeps the flavor's own default): the
        requested cga, or -- none requested -- the NARROWEST width of the flavor's domain whose resident block holds S_kv (``config_sm107.sm107_bias_auto_cga``, the heuristics' pick, so
        a d64 graph at S_kv 256 lands on cga2 whichever path built the plan); 1 when no width holds it, which
        ``check_support`` then declines with the budget.  Read by ``check_support`` and ``template_params`` alike.
        The pair item's 128-row block is evaluated at cga1 when the plan qualifies for it (``_resident_arm_pair_eligible``)."""
        if self.cga is not None:
            return int(self.cga)
        if self._device_cc != (10, 7):
            # The SM100 line: the narrowest width whose RESIDENT block holds S_kv at the SM100 budget (config_sm100.sm100_bias_auto_cga, the
            # heuristics' and the row's arithmetic); 1 when none holds it, which check_support then declines typed.
            if self.flavor not in _SM100_BIAS_D_SHAPES:
                return 1
            return _sm100_bias_auto_cga(self.flavor[0], int(self.k_desc.shape[2]), _SM100_SMALL_D_CGAS[self.flavor], tile_n=self._effective_tile_n()) or 1
        if self.flavor not in _SM107_BIAS_D_SHAPES:
            return 1
        return (
            _sm107_bias_auto_cga(
                self.flavor[0],
                int(self.k_desc.shape[2]),
                _SM107_SMALL_D_CGAS[self.flavor],
                pair=self._resident_arm_pair_eligible(),
                tiles_q=self._resident_arm_softmax_wgs(),
            )
            or 1
        )

    def _resident_arm_n_chunk(self) -> int:
        """The ``TemplateParams.n_chunk`` of this plan: 0 off the pair item; the request when the caller made one (validated by
        ``check_support``); else the heuristics' pick at this device's SM count (``config_sm107.sm107_n_chunk_pick``)."""
        if not self._resident_arm_pair_eligible() or self._resident_arm_cga() != 1:
            return 0
        if self.n_chunk is not None:
            return int(self.n_chunk)
        b, h_qo, s_qo, _ = (int(x) for x in self.q_desc.shape)
        return _sm107_n_chunk_pick(
            int(self.bias_desc.shape[0]),
            b,
            h_qo,
            s_qo,
            _device_sm_count(self.q_desc.device),
            1,
            s_kv=int(self.k_desc.shape[2]),
            d_flavor=self.flavor[0],
            tile_n=self._effective_tile_n(),
            rows_per_step=self._resident_arm_softmax_wgs(),
        )

    def _resident_arm_bias_mode(self) -> int:
        """The ``TemplateParams.bias_mode`` of this plan: 0 without a bias; else the request (``bias_mode=``, validated by
        ``check_support``) or the pick at the plan's width and item form (``config_sm107.sm107_bias_mode_pick``: the resident block
        where it holds S_kv, else the streamed ring -- the pair item's in its cluster-multicast form at the measured widths under
        global lengths (``_kv_padded`` False: real per-batch lengths, or a dense S_kv tail the plan masks against the scalar S_kv,
        keep the cga1 ring) and the NATURAL scheduler; 0 = no form fits --
        ``check_support`` declines that typed)."""
        if self.bias_desc is None:
            return _BIAS_MODE_NONE
        _cga = self._resident_arm_cga()
        if self._device_cc != (10, 7):
            # The SM100 line: the RESIDENT block only (the streamed / multicast ring is not on the SM100 file yet) -- the budget pick at this
            # width (config_sm100.sm100_bias_mode_pick) when it is the resident block, else 0 (check_support declines typed; a ring REQUEST
            # is declined by name there).
            _mode = _sm100_bias_mode_pick(self.flavor[0], _cga, int(self.k_desc.shape[2]), tile_n=self._effective_tile_n(), request=self.bias_mode)
            return _mode if _mode in (_BIAS_MODE_RESIDENT, _BIAS_MODE_STREAMED) else _BIAS_MODE_NONE
        return _sm107_bias_mode_pick(
            self.flavor[0],
            _cga,
            int(self.k_desc.shape[2]),
            pair=self._resident_arm_pair_eligible() and _cga == 1,
            request=self.bias_mode,
            tiles_q=self._resident_arm_softmax_wgs(),
            # the two facts the pair ring's cluster-multicast default needs (config_sm107.SM107_PAIR_RING_MULTICAST_D_FLAVORS): no
            # per-batch lengths, and the NATURAL scheduler (a None policy derives to NATURAL on this device, template_params below)
            global_lengths=not (self._kv_padded() or self.seq_q_lens_present),
            natural=self.sched_policy is None or self.sched_policy == SCHED_NATURAL,
        )

    def _native_d128_narrow_tile(self) -> bool:
        return supports_native_d128_narrow_tile(
            self.flavor,
            device_cc=self._device_cc,
            half_precision=self.q_desc.dtype in (torch.float16, torch.bfloat16),
            thd=self.thd,
            paged=self.paged,
            cga=self.cga,
            split_kv=self.split_kv,
            has_bias=self.bias_desc is not None,
            has_kv_mask=self.kv_mask_desc is not None,
        )

    def _s_slots(self) -> int:
        """The ``TemplateParams.s_slots`` of this plan: Two for the native D128 narrow tile. On Rubin 32- / 64-wide f16/bf16 tiles,
        ``config_sm107.sm107_s_slots_pick`` decides (the MEASURED rule: 2 on the plain body -- no bias, no kv_mask -- at TILE_N 64, 1 on
        every other form; the sweep seam's value verbatim, which the factory validates against TILE_N 64 and the TMEM fit)."""
        # D128's explicit narrow tile uses the double-buffered score pipeline.
        # Its default tile width remains 128; this does not choose a plan.
        if self._native_d128_narrow_tile() and self._effective_tile_n() == 64:
            return 2
        if self.flavor in _SM107_F16_SMALL_D_SHAPES and self._device_cc == (10, 7) and not self._fp8:
            return _sm107_s_slots_pick(self.flavor[0], self._effective_tile_n(), has_bias=self.bias_desc is not None, has_kv_mask=self.kv_mask_desc is not None)
        return 1

    def _q_in_tmem(self) -> int:
        """The ``TemplateParams.q_in_tmem`` of this plan: 0 everywhere but the Rubin 32- / 64-wide f16/bf16 tiles, where
        ``config_sm107.sm107_q_in_tmem_pick`` decides from the record's own geometry (the measured rule: the plain 128-wide body, the bias-only
        64-wide body and the bias + kv_mask 64-wide body of the 64-wide tile without per-row lengths take Q in TMEM; the sweep seam's value
        verbatim, which the factory validates against the arm's domain: cga1, the 128-row tile, two sub-tiles, the TMEM fit)."""
        if self.flavor in _SM107_F16_SMALL_D_SHAPES and self._device_cc == (10, 7) and not self._fp8:
            return _sm107_q_in_tmem_pick(
                self.flavor[0],
                dtype_qkv=_SM100_DTYPE_QKV_CODE[self.dtype],
                tile_n=self._effective_tile_n(),
                has_bias=self.bias_desc is not None,
                has_kv_mask=self._kv_mask_form() != 0,
                softmax_wgs=self._resident_arm_softmax_wgs(),
                # the cta_mma arm of template_params(): a bias plan takes the heuristics' width, a plan without one its request or the tiles' cga1
                cta_mma=self._resident_arm_cga() if self.bias_desc is not None else (int(self.cga) if self.cga is not None else 1),
                # the Q tile height the factory renders (128, or 64 under the TILE_M sweep seam): the same resolver the budgets read
                tile_m=_sm107_tile_m_d128(self.flavor[0]),
                seq_kv_lens_present=bool(self.seq_kv_lens_present),
                dense_unsplit=not (self.thd or int(self.split_kv) > 1 or self.pack_gqa or self.gate_desc is not None),
            )
        return 0

    def _resident_arm_bias_stages(self) -> int:
        """The ``TemplateParams.bias_stages`` of this plan: 0 off the streamed ring; else the request (``bias_stages=``, validated by
        ``check_support``) or the derived deepest ring (``config_sm107.sm107_bias_stages_pick``)."""
        if self._resident_arm_bias_mode() not in _BIAS_STREAMED_MODES:
            return 0
        _cga = self._resident_arm_cga()
        if self._device_cc != (10, 7):
            # The SM100 line: the derived deepest ring at this width (config_sm100.sm100_bias_stages_pick; 2 at the 32-wide tile's
            # one-batch-entry item), the request when it fits.
            return _sm100_bias_stages_pick(self.flavor[0], _cga, tile_n=self._effective_tile_n(), request=self.bias_stages or 0)
        return _sm107_bias_stages_pick(
            self.flavor[0],
            _cga,
            pair=self._resident_arm_pair_eligible() and _cga == 1,
            request=self.bias_stages or 0,
            tiles_q=self._resident_arm_softmax_wgs(),
        )

    def _resident_arm_softmax_wgs(self) -> int:
        """The ``TemplateParams.softmax_wgs`` of this plan: the request (``softmax_wgs=``, validated by ``check_support``), else the sweep seam,
        else the measured rule (``config_sm107.sm107_softmax_wgs_pick``: 3 for a bias + kv_mask graph on the pair item at tile_n 64 / cga 1 with
        dense KV lengths at ``s_kv >= SM107_THREE_WG_S_KV_MIN``, 2 everywhere else).  Threaded as ``tiles_q`` into every bias budget / pick this
        plan evaluates (the 3-WG body's third Q and O slab shrink the SMEM left for the bias block) and as ``rows_per_step`` into the chunk pick,
        so the plan is admitted on the budget of the body that RUNS and its chunk is a multiple of the body's rows per step (rule 8b').  The
        pair predicate it reads does not read the width or the chunk pick back (no cycle); ``_s_slots`` reads the width only (no cycle either)."""
        if self.bias_desc is None:
            return _sm107_softmax_wgs_pick(self.softmax_wgs, s_slots=self._s_slots())
        return _sm107_softmax_wgs_pick(
            self.softmax_wgs,
            d_flavor=self.flavor[0],
            s_kv=int(self.k_desc.shape[2]),
            tile_n=self._effective_tile_n(),
            pair=self._resident_arm_pair_eligible(),
            has_bias=True,
            has_kv_mask=self.kv_mask_desc is not None,
            padded=self._kv_padded(),
            cga=1 if self.cga is None else int(self.cga),
            b_flat=int(self.q_desc.shape[0]),
            s_slots=self._s_slots(),  # a two-slot record is served with two groups only (the pick degrades its default; a request stays typed)
        )

    def _padded_tail_form_eligible(self) -> bool:
        """Whether this plan runs the Rubin small tiles' PLAIN padded body (MASK_FLAGS == MASK_PADDED alone): the only rendering with a
        padded tail form to choose (``TemplateParams.padded_tail_form``)."""
        if self._device_cc in _SM100_BIAS_CCS:
            # The SM100 small tiles' padded body WITHOUT a kv_mask -- the plain body, or the bias body (whose -inf loop renders at MASK_NONE under
            # the pre-masked tail); a d64 plan the DECODE tile would take (S_q <= 128 at cga1) is not eligible: the decode tile has no tail form
            # and the small-tile factory refuses a decode record.  The (64, 64) flavor's other legs -- paged KV, PackGQA, split-KV -- ride the
            # plain d64 record, which the small-tile factory refuses too, so they are not eligible either; they are excluded BEFORE
            # ``_d64_decode_tile``, which reads the GQA group (``h_q // h_kv``) that ``check_support`` resolves after it asks this predicate.
            return (
                not self._fp8
                and self.flavor in _SM100_BIAS_D_SHAPES
                and bool(self.seq_kv_lens_present)
                and not self.thd
                and not self.paged
                and not self.pack_gqa
                and self.split_kv == 1
                and self.kv_mask_desc is None
                and not self.is_causal
                and self.window_size_left is None
                and self.window_size_right is None
                and not self._d64_decode_tile()
            )
        return (
            self._device_cc == (10, 7)
            and not self._fp8
            and self.flavor in _SM107_F16_SMALL_D_SHAPES
            and self._kv_padded()
            and not self.thd
            and self.bias_desc is None
            and self.kv_mask_desc is None
            and not self.is_causal
            and self.window_size_left is None
            and self.window_size_right is None
        )

    def _padded_tail_form(self) -> int:
        """The ``TemplateParams.padded_tail_form`` of this plan: 0 off the eligible rendering; else the request (``padded_tail_form=``,
        validated by ``check_support``) or the pick (``config_sm107.sm107_padded_tail_form_pick``: the default constant or the sweep seam)."""
        if not self._padded_tail_form_eligible():
            return 0
        if self._device_cc != (10, 7):
            return _sm100_padded_tail_form_pick(
                self.flavor[0],
                padded=True,
                has_bias=self.bias_desc is not None,
                has_kv_mask=False,
                request=self.padded_tail_form,
                tile_n=self._effective_tile_n(),
            )
        return _sm107_padded_tail_form_pick(
            self.flavor[0],
            padded=True,
            has_bias=False,
            has_kv_mask=False,
            request=self.padded_tail_form,
            tile_n=self._effective_tile_n(),
        )

    def _dead_subtile_quiet_eligible(self) -> bool:
        """Whether this plan runs a rendering whose dead softmax sub-tile (no live row) can take the quiet body (``TemplateParams.dead_subtile_quiet``):
        on the Rubin small tiles the one-batch-entry item without a bias (the quiet body there replicates the plain row mapping's barrier sequence,
        none of the bias barriers); on the SM100 small tiles the -inf body (a bias or a kv_mask; the quiet body there carries the ring handshake),
        off the d64 decode tile."""
        if self._device_cc in _SM100_BIAS_CCS:
            return (
                not self._fp8
                and self.flavor in _SM100_BIAS_D_SHAPES
                and not self.thd
                and not self.pack_gqa
                and self.gate_desc is None
                and (self.bias_desc is not None or self.kv_mask_desc is not None)
                and not self._d64_decode_tile()
            )
        return (
            self._device_cc == (10, 7)
            and not self._fp8
            and self.flavor in _SM107_F16_SMALL_D_SHAPES
            and not self.thd
            and not self.pack_gqa
            and self.gate_desc is None
            and self.bias_desc is None
        )

    def _dead_subtile_quiet(self) -> int:
        """The ``TemplateParams.dead_subtile_quiet`` of this plan: 0 off the eligible rendering; else the request (``dead_subtile_quiet=``,
        validated by ``check_support``) or the pick (``config_sm107.sm107_dead_subtile_quiet_pick``: the default constant or the sweep seam)."""
        if not self._dead_subtile_quiet_eligible():
            return 0
        if self._device_cc != (10, 7):
            return _sm100_dead_subtile_quiet_pick(
                self.flavor[0],
                has_bias=self.bias_desc is not None,
                has_kv_mask=self.kv_mask_desc is not None,
                request=self.dead_subtile_quiet,
                s_q=int(self.q_desc.shape[2]),
                # the width the plan RUNS at -- the cta_mma arm of template_params(): a bias plan takes the heuristics' width (the narrowest
                # resident block; D 64 at S_kv <= 128 is cga2), a kv_mask-only plan its request or the small tiles' cga1 -- so the dead-sub-tile
                # rows rule (2 x tile_m x cga) sees the cluster tile the kernel really covers
                cga=self._resident_arm_cga() if self.bias_desc is not None else (int(self.cga) if self.cga is not None else 1),
                padded=bool(self.seq_kv_lens_present),
                tile_n=self._effective_tile_n(),
            )
        return _sm107_dead_subtile_quiet_pick(
            self.flavor[0],
            has_bias=False,
            pair_item=False,
            request=self.dead_subtile_quiet,
            has_kv_mask=self.kv_mask_desc is not None,
            padded=self._kv_padded(),
            tile_n=self._effective_tile_n(),
            s_q=int(self.q_desc.shape[2]),
            # the eligible plans (the f16/bf16 small tiles without a bias on cc 10.7) run at cga1 unless the TILE_CGA_M knob asks for 2 --
            # the cta_mma arm of template_params()
            cga=int(self.cga) if self.cga is not None else 1,
            # the Q tile height the factory renders (128, or 64 under the TILE_M sweep seam): the same resolver the budgets read
            tile_m=_sm107_tile_m_d128(self.flavor[0]),
        )

    def _resident_arm_bias_max_form(self) -> int:
        """The ``TemplateParams.bias_max_form`` of this plan: the request (``bias_max_form=``, validated by ``check_support``), else the
        pick (``config_sm107.sm107_bias_max_form_pick``: the sweep seam degraded to the software max where the hardware bound is not
        served -- no arm, an additive vector -- else the measured per-form default: the hardware bound on the kv_mask-only form, the
        software tree with a bias); 0 on a plan without either arm."""
        return _sm107_bias_max_form_pick(self._resident_arm_bias_mode(), self._kv_mask_form(), request=self.bias_max_form)

    def _single_cta_half_leg(self) -> bool:
        """True when the loader would pick the single-CTA half body (``_load_sm100_kernel_module``):
        cc 10.7, half inputs, cga=1, THD, and either the D192 single-Q leg (unpaged) or the D128
        packed split.  Kept in lockstep with that loader rule and with engines.mismatch."""
        return bool(
            self._device_cc == (10, 7)
            and not self._fp8
            and self.cga == 1
            and self.thd
            and ((self.flavor == (192, 128) and not self.paged) or (self.flavor == (128, 128) and self.split_kv > 1))
        )

    def _rubin_native_d512_leg(self) -> bool:
        return supports_native_d512_dense_pack_split(self.flavor, device_cc=self._device_cc, fp8=self._fp8, thd=self.thd, paged=self.paged, cga=self.cga)

    def _rubin_shared_dense_leg(self) -> bool:
        """Historical dense D128 decode/packing admission predicate (issue #1472).

        CGA1 uses the shared decode tile; packed CGA2 now uses native SM107 prefill.
        The existing public pre-folded-scale and split admission gates still use this
        predicate. Keep those gates in sync with engines.rubin_dense_d128_shared_leg.
        """
        return bool(
            self._device_cc == (10, 7)
            and not self._fp8
            and self.flavor == _SM100_DECODE_FLAVOR
            and not self.thd
            and not self.paged
            and (self.cga == 1 or self.pack_gqa)
        )

    def _rubin_native_thd_prefill_leg(self) -> bool:
        """The exact-D128 native THD prefill domain, shared with graph admission."""
        return not self.softmax_scale_prefolded and supports_nonpaged_d128_thd_prefill(
            (int(self.q_desc.shape[-1]), int(self.v_desc.shape[-1])),
            device_cc=self._device_cc,
            fp8=self._fp8,
            thd=self.thd,
            paged=self.paged,
            cga=self.cga,
            split_kv=self.split_kv,
        )

    def _rubin_shared_thd_pack_leg(self) -> bool:
        """Unsplit nonpaged D128 THD packing on the native prefill body."""
        return self.pack_gqa and supports_nonpaged_d128_thd_pack_gqa(
            self.flavor,
            device_cc=self._device_cc,
            fp8=self._fp8,
            thd=self.thd,
            paged=self.paged,
            cga=self.cga,
            split_kv=self.split_kv,
        )

    def check_support(self) -> bool:
        """Validate the call against the kernel contract on metadata alone (dtypes, shapes, strides, masks, knobs, the CGA
        width) and record the outcome in ``_is_supported``; raises the typed decline for an unsupported combination."""
        self._logger.debug("Entering check_support")

        if self.pv_bf16:
            device_cc = torch.cuda.get_device_capability(self.q_desc.device)
            self._not_implemented_error_if(
                device_cc not in ((10, 0), (10, 3)),
                "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
            )

        # Preserve the existing dense conversion domain. Native layouts bind
        # directly; other accepted layouts receive prepared gather/scatter
        # recipes and explicit caller workspace during compile().
        # THD (ragged) keeps the BSHD order over (H, S, D): the varlen path
        # rebuilds packed [1,T,H,D] views from the token, head and element
        # strides, and only that packing is defined. The batch stride is
        # never read -- every sequence base comes from the ragged offsets and
        # the batch axis is bound at extent 1 -- so it is not gated (same rule
        # as graph_analyzer.packed_layout_ok, which the engine gate applies).
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok, packed_layout_ok, thd_stats_packing

        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"] + (["gate_desc"] if self.gate_desc is not None else []):
            d = getattr(self, desc_name)
            self._value_error_if(
                d.ndim != 4,
                f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}",
            )
            _shape, _stride = d.shape, d.stride
            # Paged pools are dense tensors even under THD; Q / O are packed, and so is the
            # epilogue gate G (O-shaped, TMA'd through the same packed [1, T, H, D] map on the
            # f16/bf16 d256 THD prefill -- the gate x THD interaction is settled below).
            if self.thd and not (self.paged and desc_name in ("k_desc", "v_desc")):
                self._value_error_if(
                    not packed_layout_ok(tuple(_shape), tuple(_stride)),
                    f"{d.name} must have d, h, s stride order (head dim innermost, then heads, then tokens) for THD; got stride {tuple(_stride)} shape {tuple(_shape)}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(_shape, _stride),
                    f"{d.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {_stride} shape {_shape}",
                )

        if self.thd:
            if self.q_desc.dtype in _SM100_FP8_DTYPES:
                # FP8/MXFP8 THD serves only the packed contract (their
                # compile() builds compact fakes); the f16 kernels address
                # declared strides natively.
                self._thd_check_strides_packed()
            else:
                self._thd_check_strides_native()

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, _, _, d_v = self.v_desc.shape
        if self.paged:
            # K/V are page pools [num_pages, H_kv, page_size, D]; the logical
            # S_kv is the declared maximum, else what the block table addresses.
            num_pages, h_kv, page_size, _ = self.k_desc.shape
            self._value_error_if(page_size != self.paged_page_size, f"K container page_size {page_size} != declared {self.paged_page_size}")
            self._check_tensor_shape(self.k_desc, (num_pages, h_kv, page_size, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (num_pages, h_kv, page_size, d_v), name="V")
            self._value_error_if(
                self.paged_max_seq_len_kv is None or self.paged_max_seq_len_kv <= 0,
                "paged KV needs paged_max_seq_len_kv (the logical S_kv the block tables address)",
            )
            s_kv = self.paged_max_seq_len_kv
        else:
            _, h_kv, s_kv, _ = self.k_desc.shape
            self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (
            ("B", b),
            ("H_q", h_qo),
            ("H_kv", h_kv),
            ("S_q", s_qo),
            ("S_kv", s_kv),
            ("D_QK", d_qk),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        # The decode tile's ragged-Q leg: ragged Q/O/Stats over PAGED K/V at
        # S_q(max) == 1 on the native (128, 128) half flavor with cga=1 -- the
        # FlashInfer prefill-style graph at one token per sequence.  The
        # kernel reads the batch's Q ragged offset as its row coordinate and
        # the split combine places the final rows, so the leg needs ragged
        # Stats (a per-batch padded Stats has no ragged base), int32 offsets
        # whose multiplier divides the row (checked with the ragged tensors
        # in engines.mismatch), per-batch KV lengths (the dense kernel's
        # SEQ_KV read; the cu form is not plumbed) and no sink (sink + split is
        # declined everywhere).  Twin of engines._thd_decode_leg; keep in lockstep.
        self.thd_decode_leg = bool(
            self.thd
            and self.paged
            and self.cga == 1
            and self.q_desc.dtype not in _SM100_FP8_DTYPES
            and int(s_qo) == 1
            and (int(d_qk), int(d_v)) == _SM100_DECODE_FLAVOR
            and not self.thd_stats_padded
            and not self.cu_seq_kv_lens
        )

        # Q/K/V dtype: half (BF16/FP16, DTYPE_O == input) or FP8 (E4M3/E5M2 → MXFP8,
        # d128 only, DTYPE_O independent — typically BF16/FP16).
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        self._not_implemented_error_if(
            self.amax_o_desc is not None and (not self._fp8 or self._pertensor),
            "sample_amax_o is supported only by the block-scale MXFP8 path",
        )
        self._not_implemented_error_if(
            self.pack_gqa and self._fp8 and not self._pertensor and self.thd,
            "PackGQA on MXFP8 is served by the d128 flavor on dense batches only: the kernel gathers "
            "the packed tile's scale factors out of the group's F8_128x4 atoms per CTA "
            "(see the PackGQA note in sm100/prefill_d128_mxfp8.py)",
        )
        self._check_dtype(self.k_desc, self.dtype, name=self.k_desc.name, extra_error_msg=f"{self.k_desc.name} must match Q dtype")
        if self.pv_bf16:
            self._not_implemented_error_if(not self._fp8 or self._pertensor, "pv_bf16 requires block-scale MXFP8 Q/K")
            self._check_dtype(self.v_desc, torch.bfloat16, name="V", extra_error_msg="V must be BF16 when pv_bf16=True")
        else:
            self._check_dtype(self.v_desc, self.dtype, name=self.v_desc.name, extra_error_msg=f"{self.v_desc.name} must match Q dtype")
        if self._fp8:
            # FP8 input: O may be BF16/FP16 (half), FP8, or (per-tensor FP8 with
            # sample_sf_o) the packed FP4 container -- decoupled from the input dtype.
            self.dtype_o = self._check_dtype(self.o_desc, _with_fp4([torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES]), name="O")
        else:
            self._check_dtype(
                self.o_desc,
                self.dtype,
                name=self.o_desc.name,
                extra_error_msg=f"{self.o_desc.name} must match Q dtype (FP16/BF16 on SM100 DSL)",
            )
            self.dtype_o = self.dtype
        self._not_implemented_error_if(
            self.pv_bf16 and self.dtype_o != torch.bfloat16,
            "pv_bf16 requires BF16 O",
        )
        if self.amax_o_desc is not None:
            self._check_dtype(self.amax_o_desc, torch.float32, name="Amax_O")
            self._value_error_if(math.prod(self.amax_o_desc.shape) != 1, f"Amax_O must contain exactly one float32 element; got {self.amax_o_desc.shape}")
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            self._value_error_if(
                self.thd_stats_padded and not self.thd,
                "thd_stats_padded is THD-only (a padded Stats without ragged offsets); construct the API with thd=True",
            )
            if self.thd and self.thd_stats_padded:
                # Per-batch padded Stats (no ragged offsets): (b, h, s_max) in
                # any non-overlapping layout -- the kernel indexes [batch, head,
                # row] through the declared strides.
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"THD padded LSE must be a non-overlapping (b, h, s_max) layout; got stride {self.lse_desc.stride}",
                )
                self._lse_stride = tuple(int(stride) for stride in self.lse_desc.stride)
            elif self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                packing = thd_stats_packing(stride_h, stride_s, h_qo)
                head_major = packing == "head_major"
                self._value_error_if(
                    packing is None,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)
        for _name, _desc in (("score_max", self.score_max_desc), ("score_sum_exp", self.score_sum_exp_desc)):
            if _desc is not None:
                # The dense Stats contract, per score side output: fp32 (b, h, s) in a non-overlapping layout
                # (the kernel indexes [batch, head, row] through the declared strides).
                self._check_dtype(_desc, torch.float32, name=_name)
                self._check_tensor_shape(_desc, (b, h_qo, s_qo), name=_name)
                self._value_error_if(
                    not dense_layout_ok((*_desc.shape, 1), (*_desc.stride, 1)),
                    f"{_name} must use a dense-compatible B/H/S permutation or padded layout with non-broadcast, non-overlapping-by-span strides; got {_desc.stride}",
                )
                self._score_strides[_name] = None if _desc.is_contiguous() else tuple(int(stride) for stride in _desc.stride)

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM100 DSL SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        # cc10.0 (SM100) and cc10.3 (Blackwell-class) both run these kernels; cc10.3
        # additionally has the fused LDTM.STAT row-max, auto-enabled for MXFP8 in compile().
        self._device_cc = (major, minor)
        # The Rubin route's width cap is validated ONCE, here, before the early returns of _decode_q_tile_for: a cap on another arch line
        # is a typed decline whatever the graph (the table it caps is the cc 10.7 lowering's), while on cc 10.7 a graph that never consults
        # the decode route (prefill-shaped, THD, quantized) simply does not read it.
        self._not_implemented_error_if(
            self.decode_q_tile_max is not None and self._device_cc != (10, 7),
            f"decode_q_tile_max={self.decode_q_tile_max} caps the Rubin (cc 10.7) d256 decode route's width table; this plan lowers for "
            f"cc {major}.{minor}, whose decode tile has one routed width",
        )
        # The split combine's programmatic-dependent-launch form is wired on its sink-free dense half entries only (the sink entries
        # carry no use_pdl): a split plan with an attention sink declares use_pdl=False -- refused HERE, before any entry is compiled,
        # on every arch line (the same knob combination compile_ptr refuses; this is the typed decline a ranked plan must see).
        self._not_implemented_error_if(
            self.use_pdl and self.has_sink and self.split_kv > 1,
            "use_pdl=True with an attention sink and split_kv > 1: the split combine's sink entries carry no programmatic-dependent-launch "
            "form; declare use_pdl=False for this plan",
        )
        paged_prefill_cga1 = supports_paged_prefill_cga1(
            (int(d_qk), int(d_v)),
            device_cc=self._device_cc,
            fp8=self._fp8,
            thd=self.thd,
            paged=self.paged,
            split_kv=self.split_kv,
            max_q=int(s_qo),
        )
        # An unsplit Rubin paged request uses the prefill template even at
        # one query token; the split decode leg belongs to the SM100 family.
        self.thd_decode_leg = self.thd_decode_leg and not paged_prefill_cga1
        split_cga = self.cga if self.cga is not None else (1 if (int(d_qk), int(d_v)) == (64, 64) else 2)
        self.packed_thd_split = bool(
            split_cga == (2 if (int(d_qk), int(d_v)) in ((256, 256), SM100_NATIVE_D576_SHAPE) else 1)
            and self.split_kv > 1
            and supports_thd_split(
                (int(d_qk), int(d_v)),
                device_cc=self._device_cc,
                fp8=self.q_desc.dtype in _SM100_FP8_DTYPES,
                thd=self.thd,
                paged=self.paged,
                max_q=int(s_qo),
                padded_stats=self.thd_stats_padded,
            )
        )
        if self.packed_thd_split:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

            self._not_implemented_error_if(
                not getattr(
                    _pybind_module._SdpaThdBinder,
                    (
                        "supports_native_d576_packed_split"
                        if (int(d_qk), int(d_v)) == SM100_NATIVE_D576_SHAPE
                        else (
                            "supports_paged_d64_packed_split"
                            if int(d_v) == 64
                            else (
                                ("supports_paged_d256_packed_split" if self.paged else "supports_nonpaged_d256_packed_split")
                                if int(d_v) == 256
                                else (
                                    "supports_paged_packed_split"
                                    if self.paged
                                    else ("supports_nonpaged_d128_packed_split" if int(d_qk) == 128 else "supports_nonpaged_packed_split")
                                )
                            )
                        )
                    ),
                    False,
                ),
                "packed split requires the matching native cuDNN Frontend extension",
            )

        arch_error = cutedsl_arch_requirement_error(self._device_cc)
        self._not_implemented_error_if(arch_error is not None, arch_error)
        # The ragged-Q decode leg is an sm100/decode_d128_f16.py mode not wired on cc 10.7 (its dense d128 graphs ride that
        # tile through TILE_CGA_M=1 since issue #1472); the admitted unsplit paged cga1 THD leg uses the two-slab prefill template.
        self._not_implemented_error_if(
            self.thd_decode_leg and self._device_cc == (10, 7),
            "the d128 decode tile's ragged-Q leg is not wired on cc10.7 (Rubin); THD graphs keep the prefill tile there",
        )
        # cc10.7 (Rubin) now runs every dtype family through its own SM107
        # sibling kernels (f16/bf16, per-tensor FP8 and MXFP8 -- the SM107 port).
        # The per-arch-line split lives in the kernel FILES and the engine rows;
        # this adapter serves both lines, so the gate is simply "a cc10.x line
        # we have kernels for".
        _allowed_cc = ((10, 0), (10, 3), (10, 7))
        _allowed_msg = "cc=10.0/10.3 (Blackwell) or 10.7 (Rubin)"
        self._value_error_if(
            self._device_cc not in _allowed_cc,
            f"SdpaFwdDslSm100 requires {_allowed_msg}; found SM{major}{minor} on {device}",
        )

        # FP8 flavor shapes: SM100 per-tensor FP8 serves d128/d128,
        # d192/d128, and d256/d256; MXFP8 serves the shapes in its independent
        # native map. Rubin serves d128/d256/d512 in both quantized families
        # through its SM107 siblings. Per-tensor FP8 serves
        # the dense ENVELOPE of every flavor it has (TMA zero-padding, like
        # the f16 flavors — exact in FP8, and the descales are scalars, so
        # the envelope is arch-independent): head dims componentwise <= a
        # flavor shape and multiples of 16 (TMA 16-byte global-stride rule
        # at 1 byte/elem). THD stays native-shape (the packed THD compile
        # key carries no head-dim entries — engines.thd_d_shapes) and MXFP8
        # stays exact (SF plumbing not audited for zero-padding).
        fp8_shapes = _sm100_fp8_shapes(self._pertensor, self._device_cc)
        _fp8_envelope_ok = (
            self._pertensor and not self.thd and _fp8_envelope_covers(int(d_qk), int(d_v), fp8_shapes) and int(d_qk) % 16 == 0 and int(d_v) % 16 == 0
        )
        self._value_error_if(
            self._fp8 and (int(d_qk), int(d_v)) not in fp8_shapes and not _fp8_envelope_ok,
            f"{'FP8' if self._pertensor else 'MXFP8'} (E4M3/E5M2 inputs) requires a native shape in {sorted(fp8_shapes)}"
            + (
                f" — or, dense only, its envelope (head dims <= a flavor shape, multiples of 16; " f"floors {sorted(_SM100_FP8_ENVELOPE_FLOORS.items())})"
                if self._pertensor
                else " (no envelope padding)"
            )
            + f"; got (D_QK={d_qk}, D_V={d_v})",
        )
        # Envelope alignment gate: the TMA descriptors are built from the
        # actual tensor extents, and cuTensorMapEncodeTiled requires every
        # non-innermost global stride to be a multiple of 16 bytes. For the
        # compact BSHD views the H stride is D * BPE (2 bytes at fp16/bf16),
        # so both head dims must be multiples of 8.
        self._value_error_if(
            int(d_qk) % 8 != 0 or int(d_v) % 8 != 0,
            f"SM100 DSL envelope requires D_QK and D_V to be multiples of 8 "
            f"(TMA 16-byte global-stride constraint at 2 bytes/elem); got "
            f"(D_QK={d_qk}, D_V={d_v})",
        )
        # A graph must only land on a flavor that HAS a kernel for its
        # quantization AND its arch line: the per-tensor and block-scale
        # families have different native maps, and the Rubin f16 pool is
        # exactly _SM107_KERNEL_FILES (today the four SM100 shapes d128,
        # d192xd128, d256, d512; no native d64).  Without the Rubin narrowing a
        # shape the map lacks would be picked and then KeyError in
        # _load_sm100_kernel_module; with it, the graph falls to the next
        # covering envelope exactly as it does for a missing FP8 flavor.
        # The FP8 walk must also agree with check_support: a flavor whose
        # envelope floor excludes (d_qk, d_v) is skipped, so the graph lands on
        # the next covering flavor instead of the one the floor exists to keep
        # it off.  (`fp8_shapes` is already arch-aware -- _sm100_fp8_shapes
        # returns the SM107 maps at cc 10.7 -- so the FP8 arm needs no separate
        # Rubin narrowing.)
        if self._fp8:
            _flavor_pool = tuple(
                f for f in _SM100_FLAVORS if f in fp8_shapes and (f == (int(d_qk), int(d_v)) or min(int(d_qk), int(d_v)) > _SM100_FP8_ENVELOPE_FLOORS.get(f, 0))
            )
        elif self._device_cc == (10, 7):
            # The Rubin f16 pool: a small head-dim tile (config_sm107.SM107_F16_SMALL_D_SHAPES) only for a graph on its
            # DENSE leg (no THD / paged KV / split-KV / PackGQA / score side outputs -- sm107_small_tile_leg) at a cga it
            # is validated at (SM107_SMALL_D_CGAS, or unset); any other graph walks past the small tiles onto the d128
            # flavor it rode before they existed (zero-filled to 128), where that leg is wired.  The twin of the engine
            # row's dense_only_d_shapes walk in engines._selected_d_shape (rule 8b'); the tests assert they agree per leg.
            _leg = _sm107_small_tile_leg(
                prefolded=bool(self.softmax_scale_prefolded),
                thd=self.thd,
                paged=self.paged,
                split_kv=self.split_kv,
                pack_gqa=self.pack_gqa,
                score_stats=self.score_max_desc is not None or self.score_sum_exp_desc is not None,
            )
            _flavor_pool = tuple(
                f
                for f in _SM107_FLAVORS
                if f in _SM107_KERNEL_FILES
                and (f not in _SM107_F16_SMALL_D_SHAPES or (_leg is None and (self.cga is None or self.cga in _SM107_SMALL_D_CGAS[f])))
            )
        else:
            # The SM100 f16/bf16 pool: the 32-wide small tile (the sm100 d128 file at d_flavor 32, config_sm100.SM100_DENSE_ONLY_D_SHAPES) only for a
            # graph on its DENSE leg at a cga in its domain (SM100_SMALL_D_CGAS); every other graph walks past it onto the native d64 / d128 flavors,
            # which carry those legs on this arch line.  The twin of the engine row's dense_only_d_shapes walk in engines._selected_d_shape (rule 8b').
            _leg = _sm107_small_tile_leg(
                thd=self.thd,
                paged=self.paged,
                split_kv=self.split_kv,
                pack_gqa=self.pack_gqa,
                score_stats=self.score_max_desc is not None or self.score_sum_exp_desc is not None,
            )
            _flavor_pool = tuple(
                f
                for f in tuple(sorted(_SM100_DENSE_ONLY_D_SHAPES)) + _SM100_FLAVORS
                if f not in _SM100_DENSE_ONLY_D_SHAPES or (_leg is None and (self.cga is None or self.cga in _SM100_SMALL_D_CGAS[f]))
            )
        # This exact half-precision shape is specific to the SM100 line; do
        # not widen the shared flavor pool inherited by Rubin and FP8.
        if not self._fp8 and self._device_cc != (10, 7) and (int(d_qk), int(d_v)) == SM100_NATIVE_D576_SHAPE:
            _flavor_pool += (SM100_NATIVE_D576_SHAPE,)
        self.flavor = _pick_flavor(d_qk, d_v, _flavor_pool)
        if self.flavor == SM100_NATIVE_D576_SHAPE:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

            self._not_implemented_error_if(
                not getattr(_pybind_module._SdpaDenseBinder, "supports_native_d576", False),
                "native D576x512 requires the matching cuDNN Frontend native extension",
            )
            self._not_implemented_error_if(
                not self._can_prepare_dense_layout(),
                "native D576x512 requires Q/K/V/O layouts that bind without a copy",
            )
        if self.thd and self.flavor == SM100_NATIVE_D576_SHAPE:
            self._not_implemented_error_if(
                not getattr(_pybind_module._SdpaThdBinder, "supports_native_d576", False),
                "native D576x512 THD requires the matching cuDNN Frontend native extension",
            )
            # Even compact standalone THD uses sequence-local maps. Graph lowering
            # replaces these absent offsets with the caller's per-port declarations.
            self._thd_origin_specs = (None,) * 5
        self._not_implemented_error_if(
            self.pack_gqa and self._fp8 and not self._pertensor and self.flavor != (128, 128),
            "PackGQA on MXFP8 is wired in the d128 flavor only (the per-CTA SF_Q gather of sm100/prefill_d128_mxfp8.py)",
        )
        if self.pack_gqa:
            # The d256 DECODE tile (sm107/decode_d256_f16.py) packs the WHOLE group into its 16-row
            # Q tile over a dense OR paged cache (HEADS_PER_TILE = QH_PER_KH), so a decode-shaped
            # half d256 graph packs on cc 10.7 whatever its cache form; every other Rubin half
            # packing rides the paged / packed-split D128 pipelines.  engines.mismatch mirrors this
            # exemption (d256_decode_tile_selected); keep the two in lockstep.
            # Dense d128 half packs on native SM107 prefill at CGA2 (shared decode at CGA1).
            # Its existing pre-folded-scale decline remains below.
            self._not_implemented_error_if(
                self._device_cc == (10, 7)
                and not self._fp8
                and not self.paged
                and not self.packed_thd_split
                and not self._rubin_shared_thd_pack_leg()
                and not self._rubin_native_d512_leg()
                and not (self.flavor == _SM100_DECODE_FLAVOR and not self.thd)
                and not self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv)),
                "Rubin half PackGQA requires paged KV, the D128 packed split, an unsplit dense or ragged D128 GQA graph without the pre-folded scale, dense D512, or the d256 decode tile (a decode-shaped graph: S_q x G packed rows within its routed envelope)",
            )
            self._not_implemented_error_if(
                self.thd
                and not self.thd_decode_leg
                and not (self.packed_thd_split and int(d_qk) in (64, 128, 576))
                and not supports_paged_d256_pack_gqa(
                    (int(d_qk), int(d_v)),
                    device_cc=self._device_cc,
                    fp8=self._fp8,
                    thd=self.thd,
                    paged=self.paged,
                    cga=self.cga,
                    split_kv=self.split_kv,
                )
                and not (
                    (self._device_cc != (10, 7) or self.paged or self._rubin_shared_thd_pack_leg())
                    and not self._fp8
                    and (int(d_qk), int(d_v)) in SM100_THD_PACK_GQA_SHAPES
                    and (self.cga in (None, 2) or paged_prefill_cga1 or self._rubin_native_thd_prefill_leg())
                    and self.split_kv == 1
                ),
                "THD PackGQA requires half D128 cga2 unsplit/cga1 split, native Rubin nonpaged D128 cga1/2 unsplit, "
                "Rubin paged D128 cga1 unsplit, or SM100/103/107 paged D256 cga2 unsplit",
            )
            # Partial PackGQA (the largest divisor of the group that divides the
            # tile) is wired in SM100 d128 / d256 half and the qualified Rubin
            # native and paged D128 routes. Other legs retain whole-group admission.
            _partial = (
                (not self._fp8 and self._device_cc != (10, 7) and self.flavor in _SM100_PARTIAL_PACK_GQA_FLAVORS)
                or self._rubin_native_d512_leg()
                or self._rubin_shared_thd_pack_leg()
                or supports_paged_d128_partial_pack_gqa(
                    (int(d_qk), int(d_v)),
                    device_cc=self._device_cc,
                    fp8=self._fp8,
                    thd=self.thd,
                    paged=self.paged,
                    cga=self.cga,
                    split_kv=self.split_kv,
                )
                or supports_native_d128_dense_partial_pack(self.flavor, device_cc=self._device_cc, fp8=self._fp8, thd=self.thd, paged=self.paged, cga=self.cga)
            )
            # The d256 DECODE tile packs the WHOLE group whenever it fits its Q tile
            # (HEADS_PER_TILE = QH_PER_KH; 24/2 puts its 12 heads in the 16-row tile, four
            # tail rows zero-filled; two tokens x 12 heads in the 32-row tile on cc 10.7), so
            # the prefill tiles' tile_m divisibility rule does not apply to a cc 10.7 graph
            # that lowers onto it (_decode_q_tile_for is non-zero exactly when the line routes
            # the S_q x G rows onto the tile; engines.mismatch's _decode_packs_whole_group is
            # the twin -- keep the two in lockstep).  The SM100 line keeps the rule as is: its
            # partial PackGQA already admits the groups its decode tile takes, and a group the
            # rule declines there (an odd ratio) stays the typed ValueError it was.
            # On cc 10.7 the (256, 256) packing IS the decode tile (the d256 prefill kernel runs unpacked), so a
            # packed d256 request past its route is the typed decline engines.mismatch gives (not the prefill
            # tiles' divisibility rule, which would name tile_m for a 24/2 group the tile serves at S_q <= 4).
            self._not_implemented_error_if(
                self._device_cc == (10, 7)
                and not self._fp8
                and self.flavor == (256, 256)
                and not self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv))
                and not supports_paged_d256_pack_gqa(
                    (int(d_qk), int(d_v)),
                    device_cc=self._device_cc,
                    fp8=self._fp8,
                    thd=self.thd,
                    paged=self.paged,
                    cga=self.cga,
                    split_kv=self.split_kv,
                ),
                "Rubin half PackGQA at D256 is wired on the decode tile (a decode-shaped dense graph: S_q x G packed rows within its routed envelope) and on the paged half THD prefill (CGA2, unsplit) only; the dense d256 prefill kernel runs unpacked",
            )
            self._value_error_if(
                not pack_gqa_supported(int(h_qo), int(h_kv), partial=_partial)
                and not (self._device_cc == (10, 7) and self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv))),
                f"PackGQA requires h_q/h_kv to {'share a factor with' if _partial else 'divide'} the kernel tile_m; got h_q/h_kv = {int(h_qo)}/{int(h_kv)}",
            )
        # Block-scaled O (sf_o): per-tensor FP8, d128 flavor, dense/unsplit/unpacked.
        self._dtype_o_code = _SM100_DTYPE_QKV_CODE.get(self.dtype_o)
        if self.sf_o_desc is not None or self.dtype_o == _torch_fp4():
            self._not_implemented_error_if(not self._fp8, "a block-scaled O (sf_o / FP4 O) is served by the quantized paths (per-tensor FP8, MXFP8) only")
            if self.dtype_o == _torch_fp4():
                self._value_error_if(self.sf_o_desc is None, "an FP4 (float4_e2m1fn_x2) O requires sample_sf_o (E4M3 scale factors, one per 16 d elements)")
                self._check_dtype(self.sf_o_desc, torch.float8_e4m3fn, name="sf_o")
                self.o_block_scale, self._dtype_o_code = 16, DTYPE_O_NVFP4
            else:
                self._value_error_if(
                    self.dtype_o != torch.float8_e4m3fn, "sf_o with a non-FP4 O requires an FP8 E4M3 O (MXFP8 output: one UE8M0 scale per 32 d elements)"
                )
                self._check_dtype(self.sf_o_desc, [torch.uint8, torch.float8_e8m0fnu], name="sf_o")
                self.o_block_scale, self._dtype_o_code = 32, DTYPE_O_MXFP8
            self._not_implemented_error_if(
                self.flavor != (128, 128), f"block-scaled O is served by the d128 flavor only; got head dims {(int(d_qk), int(d_v))}"
            )
            self._not_implemented_error_if(int(d_v) != 128, "block-scaled O needs d_v == 128 (no head-dim envelope)")
            self._not_implemented_error_if(
                self.thd or self.seq_q_lens_present or self.pack_gqa or self.split_kv > 1,
                "block-scaled O serves dense, untrimmed, unsplit, unpacked graphs only",
            )
            self._sfo_geometry = self._sf_o_geometry(self.o_block_scale, int(d_v))
        self._value_error_if(
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM100 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
        )
        self._value_error_if(self.tile_m is not None and self.tile_m != 128, "SM100 DSL SDPA only supports tile_m=128")
        # The KV tile width: 128 on every flavor; the Rubin f16/bf16 small tiles also render 64 (the TILE_N knob --
        # config_sm107.SM107_TILE_NS_BY_D_SHAPE, the engine row's per-shape domain, rule 8b').
        # the KV tile widths this plan may request: the Rubin small tiles' domain on cc 10.7, the SM100 small tiles' on cc 10.0 / 10.3 (both from the

        # row's tile_ns_by_d_shape constants, rule 8b'), 128 everywhere else

        _tile_ns = (
            dict(_SM107_TILE_NS_BY_D_SHAPE).get(self.flavor, frozenset({128}))
            if (self._device_cc == (10, 7) and not self._fp8)
            else (
                dict(_SM100_TILE_NS_BY_D_SHAPE).get(self.flavor, frozenset({128}))
                if (self._device_cc in _SM100_BIAS_CCS and not self._fp8)
                else frozenset({128})
            )
        )
        if self._native_d128_narrow_tile():
            _tile_ns = frozenset({64, 128})
        self._value_error_if(
            self.tile_n is not None and self.tile_n not in _tile_ns,
            f"SM100 DSL SDPA serves tile_n in {sorted(_tile_ns)} on the {self.flavor} flavor; got {self.tile_n}",
        )
        supported_cgas = (
            (1, 2)
            if paged_prefill_cga1
            else (
                ((2,) if self.flavor in ((256, 256), SM100_NATIVE_D576_SHAPE) else (1,))
                if self.packed_thd_split
                else supported_cgas_for(self.flavor, fp8=self._fp8, device_cc=self._device_cc, pertensor=self._pertensor)
            )
        )
        # Only a non-None request is checked: None means "let the lowering pick",
        # which is how every graph that does not pin the knob gets here.  Dropping
        # this check is not cosmetic -- it is precisely the rule-8b' failure the
        # helper exists to prevent, since an explicit cga=1 on the Rubin quantized
        # d192 path would then clear check_support() and die inside compile().
        self._value_error_if(
            self.cga is not None and self.cga not in supported_cgas,
            f"SM100 DSL SDPA only supports cga in {supported_cgas}",
        )
        self._not_implemented_error_if(
            self.cga == 2
            and self._device_cc == (10, 7)
            and self._fp8
            and not self._pertensor
            and self.flavor == (256, 256)
            and not self.packed_thd_split
            and not d256_mxfp8_pair_leg(device_cc=self._device_cc, pertensor=False, thd=self.thd, paged=self.paged, split_kv=self.split_kv),
            "the d256 MXFP8 2-CTA pair serves dense unsplit queries on cc 10.7; THD, split-KV and paged graphs run cga1 (leave cga unset or pass cga=1)",
        )
        self._value_error_if(
            self.flavor == (192, 128) and self.split_kv > 1 and self.cga == 1 and not self.packed_thd_split,
            "D192 split_kv > 1 is validated only with cga=2",
        )
        # D128 half cga1 uses the decode tile for one-query or split THD,
        # and a two-slab prefill body for the admitted paged or native Rubin
        # nonpaged unsplit legs. Keep these gates aligned with effective_cgas
        # and the engine row's mismatch checks.
        self._not_implemented_error_if(
            self.flavor == _SM100_DECODE_FLAVOR
            and self.cga == 1
            and not self._fp8
            and self.thd
            and not (self.thd_decode_leg or self.packed_thd_split or paged_prefill_cga1 or self._rubin_native_thd_prefill_leg()),
            "cga=1 on the d128 half flavor requires an admitted one-query/split THD decode leg, paged prefill, "
            "or native Rubin nonpaged exact D128 unsplit prefill; "
            "other THD (ragged) graphs run the cga2 prefill tile",
        )
        # The per-tensor FP8 d128 flavor admits cga1 too, as its dense unsplit
        # prefill leg (heuristics._auto_sched_cga); its THD leg is validated on
        # the cga2 pair only, which is what engines.mismatch declines for every
        # dtype (the decode tile's ragged-Q leg never admits FP8).
        self._not_implemented_error_if(
            self.flavor == _SM100_DECODE_FLAVOR and self.cga == 1 and self._fp8 and self.thd,
            "cga=1 on the per-tensor FP8 d128 flavor is its dense unsplit prefill leg; THD (ragged) graphs run the cga2 pair",
        )
        self._not_implemented_error_if(
            self.thd_decode_leg and self.split_kv < 2,
            "the d128 decode tile's ragged-Q leg rides the split path (the combine places the ragged O / Stats rows); split_kv must be >= 2",
        )
        self._not_implemented_error_if(
            self.pv_bf16 and (self.flavor not in ((128, 128), (192, 128)) or self.thd or self.split_kv != 1 or self.paged),
            "pv_bf16 is an experimental direct-only MXFP8 D128 or D192xD128 dense specialization (THD, split-KV and paged KV are not wired)",
        )
        if self.score_max_desc is not None or self.score_sum_exp_desc is not None:
            # Score side outputs: the twin of the engine row's gates (engines.mismatch; rule 8b) through
            # the SAME shape constant (rule 8b').  The store lives in the f16/bf16 d128 prefill file's
            # correction epilogue -- dense, unsplit, unpaged, on the 2-CTA tile (cga=1 selects the decode
            # tile, which has no store; the d64 flavor's automatic width is 1, so it needs an explicit cga=2).
            _score_shapes = _SM107_SCORE_STATS_SHAPES if self._device_cc == (10, 7) else SM100_SCORE_STATS_SHAPES
            self._not_implemented_error_if(
                self._fp8, "score_max / score_sum_exp are served by the f16/bf16 engines only (the quantized kernels carry no store)"
            )
            self._not_implemented_error_if(
                self.thd or self.paged or self.split_kv > 1,
                "score_max / score_sum_exp are served on dense, unsplit, unpaged plans only (no THD, no split_kv > 1, no paged KV)",
            )
            self._not_implemented_error_if(
                self.flavor not in _score_shapes,
                f"score_max / score_sum_exp are wired in the {sorted(_score_shapes)} kernel flavors only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            self._not_implemented_error_if(
                self.cga == 1 or (self.cga is None and self.flavor == (64, 64)),
                "score_max / score_sum_exp are stored by the 2-CTA prefill tile; cga=1 selects a tile without the store (pin cga=2)",
            )
        if self.bias_desc is None:
            # n_chunk is the resident-bias arm's work-item form (the (n, n+1) pair item): typed decline on a plan without a bias.
            self._not_implemented_error_if(
                self.n_chunk is not None and self.n_chunk >= 1, "n_chunk is the resident-bias arm's work-item form; this plan has no bias"
            )
            self._not_implemented_error_if(
                self._resident_arm_softmax_wgs() != 2,
                "softmax_wgs=3 (the third softmax warp group) is the resident-bias pair item's arm; this plan has no bias",
            )
        # The PLAIN padded body's tail form (padded_tail_form=): a REQUEST is honoured on the Rubin small tiles' padded, band-less,
        # bias- / kv_mask-less plan and declined typed everywhere else (never degraded); None = the pick.
        self._value_error_if(
            self.padded_tail_form is not None and self.padded_tail_form not in _PADDED_TAIL_FORMS,
            f"padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {self.padded_tail_form}",
        )
        self._not_implemented_error_if(
            self.padded_tail_form is not None and not self._padded_tail_form_eligible(),
            f"padded_tail_form={self.padded_tail_form} selects the tail form of the f16/bf16 32- / 64-wide tiles' padded body (per-batch KV "
            f"lengths or an S_kv off the tile multiple, no kv_mask, no causal / sliding-window band; dense, unpaged, unsplit, non-PackGQA; the PLAIN "
            f"body on cc 10.7, the plain or the bias body on cc 10.0 / 10.3); head dims ({d_qk}, {d_v}) select {self.flavor} on cc {self._device_cc}, "
            f"padded={bool(self.seq_kv_lens_present)}, paged={self.paged}, pack_gqa={self.pack_gqa}, split_kv={self.split_kv}, "
            f"bias={self.bias_desc is not None}, "
            f"kv_mask={self.kv_mask_desc is not None}, causal={self.is_causal}, window=({self.window_size_left}, {self.window_size_right})",
        )
        # The dead-sub-tile quiet body (dead_subtile_quiet=): a REQUEST is honoured on the Rubin small tiles' bias-less one-batch-entry
        # plan and declined typed everywhere else (never degraded); None = the pick.
        self._value_error_if(
            self.dead_subtile_quiet is not None and self.dead_subtile_quiet not in _DEAD_SUBTILE_QUIET_FORMS,
            f"dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {self.dead_subtile_quiet}",
        )
        self._not_implemented_error_if(
            self.dead_subtile_quiet is not None and not self._dead_subtile_quiet_eligible(),
            f"dead_subtile_quiet={self.dead_subtile_quiet} selects the dead-sub-tile body of the f16/bf16 32- / 64-wide tiles' one-batch-entry item "
            f"(on cc 10.7 the plain body: no bias; on cc 10.0 / 10.3 the -inf body: a bias or a kv_mask, off the d64 decode tile; never PackGQA / THD / "
            f"the gated epilogue); head dims ({d_qk}, {d_v}) select {self.flavor} on cc {self._device_cc}, bias={self.bias_desc is not None}, "
            f"kv_mask={self.kv_mask_desc is not None}, pack_gqa={self.pack_gqa}, thd={self.thd}, gate={self.gate_desc is not None}",
        )
        if self.bias_desc is not None or self.kv_mask_desc is not None:
            # The resident-block arms bind through the PREPARED launch only: the staging-copy executor (a layout TMA cannot bind
            # zero-copy) forwards neither operand, so that path is a typed decline here, never an execute-time error (the engine
            # row's _resident_arm_layout_decline is the twin, rule 8b).
            self._value_error_if(
                not self._fp8 and not self._can_prepare_dense_layout(),
                "the additive bias / kv_mask arms bind through the prepared launch only: Q/K/V/O need 16-byte-aligned BSHD-physical strides "
                "that bind without a copy (the staging-copy path does not carry these operands)",
            )
            # Additive bias / per-key mask: the twin of the engine row's gates (engines._bias_arm_decline; rule 8b)
            # through the SAME constants (rule 8b').  The resident row-block lives in the Rubin f16/bf16 d128 file's
            # 32- / 64-wide tiles: dense, unsplit, unpaged, non-PackGQA, ungated, without the score side outputs,
            # and -- with a bias -- at an S_kv whose tiles the carveout holds beside the operands at the plan's cga (the
            # kv_mask alone stages nothing in SMEM: no block, no S_kv budget).
            _what = "additive bias" if self.bias_desc is not None else "per-key kv_mask"
            self._not_implemented_error_if(
                (self._device_cc != (10, 7) and self._device_cc not in _SM100_BIAS_CCS) or self._fp8,
                f"the {_what} is served by the Rubin and SM100 f16/bf16 engines' 32- / 64-wide tiles only",
            )
            if self._device_cc != (10, 7):
                # The SM100 line serves the RESIDENT block of the one-batch-entry item at TILE_N 128 so far (the port's first body seam):
                # the streamed / multicast ring, the (n, n+1) pair item, the 64-wide KV tile and the fp32 additive vector WITH a bias (served
                # at that tile only) are typed declines until they land -- the engine row's _bias_arm_decline is the twin (rule 8b).
                self._not_implemented_error_if(
                    self.bias_mode == _BIAS_MODE_STREAMED_MULTICAST,
                    f"bias_mode={self.bias_mode} (the multicast bias ring) is not on the SM100 file yet; the resident block (1) and the streamed ring (2) are",
                )
                self._not_implemented_error_if(
                    self.n_chunk is not None and self.n_chunk >= 1,
                    f"n_chunk={self.n_chunk} (the (n, n+1) pair work item) is not on the SM100 file yet; every bias plan runs the one-batch-entry item there",
                )
                _sm100_tile_ns = dict(_SM100_TILE_NS_BY_D_SHAPE).get(self.flavor, frozenset({128}))
                self._not_implemented_error_if(
                    self.tile_n is not None and int(self.tile_n) not in _sm100_tile_ns,
                    f"tile_n={self.tile_n}: the small tiles render TILE_N {sorted(_sm100_tile_ns)} on this line",
                )
                self._not_implemented_error_if(
                    self.bias_desc is not None
                    and self.kv_mask_desc is not None
                    and self.kv_mask_desc.dtype == torch.float32
                    and self._effective_tile_n() != _SM100_FP32_VECTOR_WITH_BIAS_TILE_N,
                    f"an fp32 additive kv_mask together with a bias is served at TILE_N {_SM100_FP32_VECTOR_WITH_BIAS_TILE_N} (the heuristics' pick; the TILE_N "
                    f"knob) or as a vector in the Q/K/V dtype: the 128-wide rendering spills on the softmax role; got TILE_N {self._effective_tile_n()}",
                )
            self._not_implemented_error_if(
                self.thd or self.paged or self.split_kv > 1 or bool(self.pack_gqa) or self.gate_desc is not None,
                f"the {_what} is served on dense, unsplit, unpaged, non-PackGQA, ungated plans only",
            )
            self._not_implemented_error_if(
                self.score_max_desc is not None or self.score_sum_exp_desc is not None,
                f"the {_what} and the score side outputs are not wired together (the -inf masked softmax arm publishes no true max)",
            )
            _bias_shapes = _SM107_BIAS_D_SHAPES if self._device_cc == (10, 7) else _SM100_BIAS_D_SHAPES
            self._not_implemented_error_if(
                self.flavor not in _bias_shapes,
                f"the {_what} is wired on the {sorted(_bias_shapes)} kernel flavors only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            if self.bias_desc is not None:
                _bias_cga = self._resident_arm_cga()
                _pair = self._resident_arm_pair_eligible() and _bias_cga == 1  # the item that RUNS (its block's budget)
                _wgs = self._resident_arm_softmax_wgs()
                _s_kv_max = (
                    _sm107_bias_resident_s_kv_max(self.flavor[0], _bias_cga, pair=_pair, tiles_q=_wgs)
                    if self._device_cc == (10, 7)
                    else _sm100_bias_resident_s_kv_max(self.flavor[0], _bias_cga, tile_n=self._effective_tile_n())
                )
                _item = "the (n, n+1) pair item" if _pair else "the one-batch-entry item"
                # The third softmax warp group (softmax_wgs=3): the pair item at tile_n 64 / cga 1 only -- the one form whose roles loop over the
                # sub-tiles (the one-batch-entry paths hand-unroll two) and the one width its 128-register softmax holds; declined typed, never
                # degraded (config_sm107._validate_softmax_wgs is the factory's twin).
                self._value_error_if(_wgs not in _SMALL_D_SOFTMAX_WGS, f"softmax_wgs must be one of {sorted(_SMALL_D_SOFTMAX_WGS)}; got {_wgs}")
                self._not_implemented_error_if(
                    _wgs == 3 and not (_pair and self._effective_tile_n() == 64 and _bias_cga == 1),
                    f"softmax_wgs=3 (the third softmax warp group) is served on the (n, n+1) pair item at tile_n 64 / cga 1 only (this plan: {_item}, "
                    f"tile_n {self._effective_tile_n()}, cga {_bias_cga})",
                )
                self._value_error_if(
                    self.bias_mode is not None and self.bias_mode not in (_BIAS_MODE_RESIDENT, _BIAS_MODE_STREAMED, _BIAS_MODE_STREAMED_MULTICAST),
                    f"bias_mode must be 1 (the resident block), 2 (the streamed ring) or 3 (the streamed ring, cluster multicast); got {self.bias_mode}",
                )
                self._not_implemented_error_if(
                    self.bias_mode == _BIAS_MODE_STREAMED_MULTICAST and not _pair,
                    f"bias_mode=3 (the multicast bias ring) is the (n, n+1) pair item's form: it needs a batch-broadcast bias on >= 2 flattened rows "
                    f"at cga 1 (this plan runs {_item} at cga {_bias_cga})",
                )
                _mode = self._resident_arm_bias_mode()
                self._not_implemented_error_if(
                    self._device_cc != (10, 7) and _mode == _BIAS_MODE_NONE,
                    f"no bias form fits at head-dim tile {self.flavor[0]} / cga {_bias_cga} on this arch line for S_kv {s_kv}: the resident block holds "
                    f"S_kv <= {_s_kv_max} and no 2-stage streamed ring of its tiles fits beside the operands (the pair item is not on the SM100 file yet)",
                )
                if _mode == _BIAS_MODE_STREAMED_MULTICAST:
                    # The multicast ring (a request, or the test seam): the PAIR item's ring at cga1 run as a cluster of two independent CTAs
                    # along the chunk axis that read ONE bias tile sequence -- a form, never a degraded plan: typed declines for what breaks
                    # the lockstep (the config validator pins the same couplings; this is the row's twin, rule 8b).
                    self._not_implemented_error_if(
                        self.seq_kv_lens_present or self.seq_q_lens_present,
                        "bias_mode=3 (the multicast bias ring) streams ONE kv-tile sequence to both CTAs of a cluster: per-batch lengths "
                        "(seq_kv_lens_present / seq_q_lens_present) would make the two chunks' sequences differ -- served with global lengths only",
                    )
                    self._not_implemented_error_if(
                        self.sched_policy is not None and self.sched_policy != SCHED_NATURAL,
                        f"bias_mode=3 (the multicast bias ring) pairs chunks through the NATURAL scheduler's payload; got sched_policy={self.sched_policy}",
                    )
                self._not_implemented_error_if(
                    _mode == _BIAS_MODE_NONE,
                    (
                        f"the requested bias form {self.bias_mode} does not fit at head-dim tile {self.flavor[0]} / cga {_bias_cga} ({_item}) for S_kv {s_kv}: "
                        f"the resident bias block holds S_kv <= {_s_kv_max}, the streamed ring needs two {128 if _pair else 256} x 128 tiles of SMEM beside the operands"
                        if self.bias_mode is not None
                        else f"no bias form fits at head-dim tile {self.flavor[0]} / cga {_bias_cga} ({_item}) for S_kv {s_kv}: the resident bias block holds "
                        f"S_kv <= {_s_kv_max} and no 2-stage streamed ring of its tiles fits beside the operands"
                    ),
                )
                self._value_error_if(
                    self.bias_stages is not None and self.bias_stages not in _BIAS_RING_STAGES,
                    f"bias_stages must be one of {sorted(_BIAS_RING_STAGES)} (the streamed ring's depth); got {self.bias_stages}",
                )
                self._not_implemented_error_if(
                    self.bias_stages is not None and _mode not in _BIAS_STREAMED_MODES,
                    f"bias_stages={self.bias_stages} is the STREAMED ring's depth; this plan runs the resident block (bias_mode 1) -- request bias_mode=2 / 3",
                )
                self._not_implemented_error_if(
                    _mode in _BIAS_STREAMED_MODES and self._resident_arm_bias_stages() == 0,
                    f"a {self.bias_stages}-stage streamed bias ring does not fit at head-dim tile {self.flavor[0]} / cga {_bias_cga} ({_item}) beside the operands",
                )
                # An explicit n_chunk is honored on a plan that runs the pair item and declined typed elsewhere (never degraded).
                self._not_implemented_error_if(
                    self.n_chunk is not None and self.n_chunk >= 1 and not _pair,
                    f"n_chunk={self.n_chunk} is the (n, n+1) pair work item: it needs a batch-broadcast bias ([1, 1|H, S_q, S_kv]; got batch extent "
                    f"{int(self.bias_desc.shape[0])}), >= 2 flattened batch rows (got {int(b)}) and cga 1 (got {_bias_cga})",
                )
                self._value_error_if(self.n_chunk is not None and self.n_chunk < 0, f"n_chunk must be >= 0; got {self.n_chunk}")
                if _pair and self.n_chunk != 0:
                    _n_chunks = -(-int(b) // max(1, self._resident_arm_n_chunk()))
                    self._not_implemented_error_if(
                        _n_chunks > 65535,
                        f"the pair item's chunk index rides the 16-bit batch field of the scheduler payload: ceil(B / n_chunk) = {_n_chunks} > 65535",
                    )
                self._check_dtype(self.bias_desc, self.q_desc.dtype, name="bias")
                _bd, _bs = tuple(int(x) for x in self.bias_desc.shape), tuple(int(x) for x in self.bias_desc.stride)
                self._value_error_if(
                    len(_bd) != 4 or _bd[2:] != (int(s_qo), int(s_kv)) or _bd[0] not in (1, int(b)) or _bd[1] not in (1, int(h_qo)),
                    f"bias must be (1|B, 1|H_q, S_q, S_kv) = (1|{b}, 1|{h_qo}, {s_qo}, {s_kv}); got {_bd}",
                )
                self._value_error_if(_bs[3] != 1, f"bias must have its key dim contiguous (stride 1); got strides {_bs}")
                self._value_error_if(
                    any(n > 1 and (s * 2) % 16 for n, s in zip(_bd[:3], _bs[:3])),
                    f"every live bias stride must be a 16-byte multiple for TMA (S_kv % 8 == 0 for a compact bias); got strides {_bs}",
                )
            if self.kv_mask_desc is not None:
                # A byte mask (the bit word) or the ADDITIVE vector in the io dtype / fp32 (never the other half dtype).
                self._check_dtype(self.kv_mask_desc, [torch.bool, torch.int8, torch.uint8, torch.float32, self.q_desc.dtype], name="kv_mask")
                _md, _ms = tuple(int(x) for x in self.kv_mask_desc.shape), tuple(int(x) for x in self.kv_mask_desc.stride)
                self._value_error_if(
                    len(_md) != 4 or _md[1:3] != (1, 1) or _md[3] != int(s_kv) or _md[0] not in (1, int(b)) or _ms[3] != 1,
                    f"kv_mask must be (1|B, 1, 1, S_kv) = (1|{b}, 1, 1, {s_kv}) with a contiguous key dim; got {_md} / {_ms}",
                )
                if self._kv_mask_form() != _KV_MASK_BOOL_WORD:
                    # The additive vector is read as warp-uniform 16-byte granules: a whole-granule extent (the tail granule is
                    # clamped in-kernel) and a 16-byte batch stride (the engine row's gate and the binder read the same rule).
                    _elem = self.kv_mask_desc.dtype.itemsize
                    self._value_error_if(
                        int(s_kv) % _KV_MASK_ADDITIVE_GRANULE_ELEMS != 0,
                        f"an additive kv_mask vector needs S_kv % {_KV_MASK_ADDITIVE_GRANULE_ELEMS} == 0 (16-byte granule loads); got S_kv={s_kv}",
                    )
                    self._value_error_if(
                        _md[0] > 1 and (_ms[0] * _elem) % 16 != 0,
                        f"an additive kv_mask vector's batch stride must be a 16-byte multiple; got strides {_ms}",
                    )
                    # The fp32 vector WITH a bias is served at tile_n 64 only (the adapter's default for the form, _effective_tile_n):
                    # its 128-wide rendering spills on the softmax role (config_sm107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N carries the
                    # measured counts; the engine row's _bias_arm_decline is the twin, rule 8b) -- an explicit 128 is declined typed.
                    self._value_error_if(
                        self.bias_desc is not None
                        and self._kv_mask_form() == _KV_MASK_ADDITIVE_F32
                        and self._effective_tile_n() != _SM107_FP32_VECTOR_WITH_BIAS_TILE_N,
                        f"an fp32 additive kv_mask together with a bias is served at tile_n={_SM107_FP32_VECTOR_WITH_BIAS_TILE_N} (the default for "
                        f"this form) or as a vector in the Q/K/V dtype ({self.q_desc.dtype}; cast the fp32 mask on the host): the 128-wide rendering "
                        f"spills on the softmax role; got tile_n={self.tile_n}",
                    )
        # The row-max FORM of the bias / kv_mask body (bias_max_form, APPENDED): a request is honoured where the hardware upper bound is
        # served (a bias and / or the boolean kv_mask) and declined typed elsewhere, never degraded (config_sm107.sm107_bias_max_form_pick
        # degrades only the DEFAULT; the config factory pins the same couplings).
        if self.bias_max_form is not None:
            self._value_error_if(
                self.bias_max_form not in _BIAS_MAX_FORMS,
                f"bias_max_form must be 0 (the software row max) or 1 (the hardware upper-bound row max); got {self.bias_max_form}",
            )
            self._not_implemented_error_if(
                self.bias_max_form == _BIAS_MAX_HW_UPPER_BOUND and self.bias_desc is None and self.kv_mask_desc is None,
                "bias_max_form=1 is the row-max form of the bias / kv_mask softmax body; this plan carries neither a bias nor a kv_mask",
            )
            self._not_implemented_error_if(
                self.bias_max_form == _BIAS_MAX_HW_UPPER_BOUND and self.kv_mask_desc is not None and self._kv_mask_form() != _KV_MASK_BOOL_WORD,
                "bias_max_form=1 (the hardware upper-bound row max) is served with a boolean kv_mask or none: an additive kv_mask vector has no "
                "per-tile max to bound -- keep bias_max_form=0 with it",
            )
        # softmax_precision values are cudnn.data_type (the knob vocabulary
        # fixed by #692); imported locally — this file otherwise speaks torch
        # dtypes and frost constants only.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost import data_type as _cudnn_dtype

        self._value_error_if(
            self.softmax_precision is not None and not self._fp8,
            "softmax_precision is served on the quantized (FP8 / MXFP8) paths only (the half families run the f32 pipeline)",
        )
        self._value_error_if(
            self.softmax_precision is not None and self.softmax_precision not in (_cudnn_dtype.FLOAT, _cudnn_dtype.HALF),
            f"softmax_precision must be cudnn.data_type.FLOAT or HALF; got {self.softmax_precision}",
        )
        # The f16x2 exponent arm is numerics-changing and lives in the cc 10.7
        # quantized kernels only -- honored exactly or declined (the sm107 FP8 /
        # MXFP8 engine rows declare HALF; the half row declares FLOAT only).
        self._value_error_if(
            self.softmax_precision == _cudnn_dtype.HALF and (self._device_cc != (10, 7) or self.flavor not in _SM107_HALF_SOFTMAX_FLAVORS),
            "softmax_precision=HALF is served for FP8 and MXFP8 inputs on cc10.7 (d128, d192x128, d256, d512) only (FLOAT is the default everywhere)",
        )
        # The pre-folded scale: per-tensor FP8 folds descale_q * descale_k into the softmax scale
        # in-kernel from device tensors, so skipping the per-score multiply would drop the
        # dequantization -- declined whatever the flavor (a contract fork, not a missing port).
        self._value_error_if(
            self.softmax_scale_prefolded and self._fp8 and self._pertensor,
            "softmax_scale_prefolded is not served on per-tensor FP8 (the kernel folds descale_q * descale_k into the softmax scale); MXFP8 block scales or an in-kernel scale serve that case",
        )
        self._value_error_if(
            self.softmax_scale_prefolded and not (self._device_cc == (10, 7) and self.flavor in _SM107_PREFOLDED_FLAVORS),
            "softmax_scale_prefolded is served on cc10.7 for MXFP8 and f16/bf16 inputs (d128, d192x128, d256, d512) only",
        )
        self._value_error_if(
            self.softmax_scale_prefolded and self._scale_softmax_requested,
            "softmax_scale_prefolded: leave scale_softmax unset -- Q already carries attn_scale * log2(e) and the kernel applies no scale",
        )
        if self.softmax_scale_prefolded and (
            self.paged or self.packed_thd_split or self._single_cta_half_leg() or self._rubin_shared_dense_leg() or self._rubin_shared_thd_pack_leg()
        ):
            # These shared packed-split / single-CTA / paged / dense D128 half bodies
            # (_load_sm100_kernel_module), which apply the scale in-kernel.  A decline, so the plan walk
            # moves to the cga2 prefill body of the same flavor (engines.mismatch / effective_cgas mirror this rule).
            raise NotImplementedError(
                "softmax_scale_prefolded is not wired in the paged-KV / packed-split THD / single-CTA THD / shared dense D128 half kernel bodies (they apply the scale in-kernel)"
            )
        if self.paged:
            # Paged KV rides the PAGED_KV specialization of the f16/bf16 kernels
            # on the flavors config_sm100._PAGED_KV_FLAVORS names (the same set
            # its _validate_params backstops, and engines' paged_d_shapes) and of
            # the d64/d128/d256 per-tensor FP8 and the MXFP8 kernels; every kernel file without it (
            # the unwired SM107 half and per-tensor FP8 siblings and the SM107 d192x128 / d512
            # MXFP8 siblings, the d192x128 / d512 FP8 flavors, the d512
            # 2x2-datapath twin) backstops
            # with a module-scope guard on paged_kv, and these declines keep that
            # guard unreachable from here.  On cc 10.7: half pools serve THD queries
            # (native D128 or the remaining shared SM100 bodies), MXFP8 pools serve dense queries on d128 / d256
            # (the sm107 siblings' own PAGED_KV loader); per-tensor FP8 pools and dense
            # half queries are not wired (engines.mismatch's Rubin clause is the twin).
            # The d256 DECODE tile (sm107/decode_d256_f16.py) walks the block table itself and
            # folds the sink per Q row, so a decode-shaped half d256 graph is served paged on
            # cc 10.7 whatever its THD-ness or sink -- the same exception engines.mismatch makes
            # (d256_decode_tile_selected); the paged PREFILL pipeline keeps its THD-only scope.
            self._not_implemented_error_if(
                self._device_cc == (10, 7)
                and ((self._fp8 and self._pertensor) or (not self._fp8 and not self.thd) or self.flavor not in ((128, 128), (256, 256)))
                and not self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv)),
                "Rubin paged KV serves half THD queries, MXFP8 pools on D128/D256, and decode-shaped half D256 graphs on the decode tile (per-tensor FP8 pools and other dense half queries are not wired)",
            )
            self._not_implemented_error_if(
                self._fp8 and self.thd and not (self._pertensor and self._device_cc in ((10, 0), (10, 3)) and self.flavor in SM100_PAGED_FP8_THD_SHAPES),
                "paged FP8 THD queries require native per-tensor D128/D256 on SM100/SM103",
            )
            self._not_implemented_error_if(
                self._fp8 and self.o_block_scale > 0,
                "paged KV with a block-scaled O (sf_o) is served on dense K/V only (the FP8 kernel's block-scaled epilogue over pools is not validated)",
            )
            self._not_implemented_error_if(
                self._pertensor and self.flavor not in ((128, 128), (64, 64), (256, 256)),
                f"paged KV for per-tensor FP8 is wired on the d64 / d128 / d256 flavors only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            self._not_implemented_error_if(
                self._fp8 and not self._pertensor and self.flavor == (64, 64),
                "paged KV for MXFP8 is not validated on the native d64 flavor (dense / unsplit / unpaged only; the row's paged_d_shapes leaves (64, 64) out)",
            )
            self._not_implemented_error_if(
                not self._fp8 and f"d{self.flavor[0]}" not in _SM100_PAGED_KV_FLAVORS,
                f"paged KV is wired on the {sorted(_SM100_PAGED_KV_FLAVORS)} flavors only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            self._value_error_if(not self.seq_kv_lens_present, "paged KV requires per-batch KV lengths (seq_kv_lens_present)")
            # has_sink composes with paged KV (epilogue fold vs. loader); the
            # split x sink exclusion below is the only sink gate on this path.
            p = self.paged_page_size
            self._value_error_if(
                p % 8 != 0 or (p < _SM100_TILE_N and _SM100_TILE_N % p != 0) or (p > _SM100_TILE_N and p % _SM100_TILE_N != 0),
                f"page_size {p} must be a multiple of 8 that divides {_SM100_TILE_N} or is a multiple of it",
            )
            self._not_implemented_error_if(
                self._fp8 and not self._pertensor and p % _SM100_TILE_N != 0,
                f"paged MXFP8 KV needs page_size to be a multiple of {_SM100_TILE_N} (whole F8_128x4 SF atoms per page); got {p}",
            )
        native_fp8_padded_split = supports_fp8_d256_padded_split(
            self.flavor, device_cc=self._device_cc, pertensor=self._fp8 and self._pertensor, thd=self.thd, paged=self.paged
        )
        if self.split_kv > 1:
            # Split-KV: partials weighted by the per-split LSE, recombined by
            # sm100/split_combine (which also owns the FP8 amax of the
            # recombined O). Structural limits mirror mismatch()'s
            # facts x knobs gate so the standalone API declines identically.
            self._not_implemented_error_if(
                self.thd and not (self.thd_decode_leg or self.packed_thd_split),
                "split_kv > 1 is dense-only, except the decode tile's ragged-Q leg and native D128/D256 or nonpaged D192 packed split",
            )
            # The split partials are sink-free and the combine folds the sink
            # once. Quantized lowering disables the main kernel's sink, and the
            # packed THD split carries a sink only where its native binder folds it
            # (supports_thd_split_sink).
            self._value_error_if(
                self.has_sink
                and (
                    (self._fp8 and not supports_quantized_split_sink(device_cc=self._device_cc, pertensor=self._pertensor, thd=self.thd))
                    or self.gate_desc is not None
                    or (
                        self.packed_thd_split
                        and not supports_thd_split_sink(
                            (int(d_qk), int(d_v)),
                            device_cc=self._device_cc,
                            fp8=self._fp8,
                            thd=self.thd,
                            paged=self.paged,
                            max_q=int(s_qo),
                        )
                    )
                ),
                "split_kv > 1 with an attention sink requires half or SM100/SM103/SM107 per-tensor FP8 sink-free partials without an epilogue gate; packed THD sinks require a supported packed combine",
            )
            if self.has_sink and self._fp8:
                from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

                self._not_implemented_error_if(
                    not getattr(_pybind_module._SdpaDenseBinder, "supports_quantized_split_sink", False),
                    "FP8 split attention with a sink needs the native quantized sink-combine binder; rebuild the extension",
                )
            if self.has_sink and self.packed_thd_split:
                from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

                self._not_implemented_error_if(
                    not getattr(_pybind_module._SdpaThdBinder, "supports_paged_split_sink", False),
                    "packed split sinks require the matching native cuDNN Frontend extension",
                )
            # Paged KV is padded by construction; its split composes with the
            # per-batch lengths (validated in test_sdpa_fwd_paged_sm100).
            self._value_error_if(
                not self.paged
                and not self.packed_thd_split
                and self.flavor != SM100_NATIVE_D576_SHAPE
                and not native_fp8_padded_split
                and (self.seq_kv_lens_present or self.seq_q_lens_present),
                "split_kv > 1 serves unpadded dense graphs only",
            )
            # Keep the standalone contract aligned with the Rubin engine row.  The d256 DECODE tile
            # (sm107/decode_d256_f16.py) writes the SM100 tile's fp32 split partials into the split-major
            # workspace that sm100/split_combine reduces, packed or not, over a dense or paged cache -- so a
            # decode-shaped half d256 graph keeps its split on cc 10.7 (engines.mismatch lifts the same two
            # rules through d256_decode_tile_selected); the d256 PREFILL kernel's split stays the paged THD
            # packed split.
            _rubin_decode_tile = self._device_cc == (10, 7) and self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv)) > 0
            self._not_implemented_error_if(
                self._device_cc == (10, 7)
                and self.pack_gqa
                and not _rubin_decode_tile
                and not self._rubin_native_d512_leg()
                and not (
                    self.packed_thd_split or (self._fp8 and self._pertensor and self.flavor in _SM107_FP8_PACK_GQA_SHAPES) or self._rubin_shared_dense_leg()
                ),
                "split_kv > 1 with PackGQA on cc10.7 requires per-tensor FP8 D64/D128/D192x128/D256, half D128 paged THD, a dense D128 half graph, native dense D512 half, or the d256 decode tile",
            )
            self._not_implemented_error_if(
                self._device_cc == (10, 7)
                and not _rubin_decode_tile
                and not self._rubin_native_d512_leg()
                and not (
                    self.packed_thd_split
                    or (not self._fp8 and self.flavor in ((128, 128), (192, 128)))
                    or (self._fp8 and self._pertensor and self.flavor in _SM107_FP8_SPLIT_SHAPES)
                ),
                "split_kv > 1 on cc10.7 requires per-tensor FP8 D64, half/per-tensor FP8 D128 or D192/V128, per-tensor FP8 D256, half D256 THD, native dense D512 half, or the d256 decode tile",
            )
            # The MXFP8 row serves its native d64 leg dense / unsplit / unpaged
            # (split_d_shapes leaves (64, 64) out); mirror it here.
            self._not_implemented_error_if(
                self._fp8 and not self._pertensor and self.flavor == (64, 64),
                "split_kv > 1 for MXFP8 is not validated on the native d64 flavor (dense / unsplit / unpaged only)",
            )

        swa_left = self.window_size_left
        self._value_error_if(
            swa_left is not None and swa_left < 0,
            f"window_size_left must be >= 0; got {swa_left}",
        )
        band_right = self.window_size_right
        self._value_error_if(
            band_right is not None and band_right < 0,
            f"window_size_right must be >= 0; got {band_right}",
        )
        self._value_error_if(
            band_right is not None and not self.is_causal,
            "SM100 DSL SDPA: window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "SM100 DSL SDPA: causal_bottom_right requires is_causal=True",
        )
        if self.thd:
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
        )
        # Keep direct construction aligned with each quantized family's THD
        # kernels; graph routing enforces the same per-family shape domain.  The
        # native d64 leg of both families is dense-only (the rows' thd_d_shapes
        # leave (64, 64) out: the packed THD lowering is not validated at
        # d_flavor=64), so it is excluded here as well.
        _thd_fp8_shapes = set(_SM100_FP8_KERNEL_FILES if self._pertensor else _SM100_MXFP8_KERNEL_FILES) - {(64, 64)}
        self._not_implemented_error_if(
            self.thd and self._fp8 and (int(d_qk), int(d_v)) not in _thd_fp8_shapes,
            f"THD/varlen on this quantized path supports {sorted(_thd_fp8_shapes)}; " f"got (D_QK={d_qk}, D_V={d_v})",
        )
        # THD on the Rubin line: the per-tensor FP8 kernels at every ported
        # shape (SM107_FP8_THD_SHAPES: d128, d192xd128, d256, d512), the
        # f16/bf16 kernels at the same four (SM107_F16_THD_SHAPES) and the
        # MXFP8 kernel at d256 (SM107_MXFP8_THD_SHAPES) -- all on the FROST THD
        # contract (the 14-arg setup helper, the 4B+4 metadata the shared
        # decode reads); the MXFP8 d256 body additionally reads its PACKED
        # per-sequence-tile-padded scale factors.  The d128 / d192xd128 / d512
        # MXFP8 bodies keep the pre-upstream THD arm and stay declined.
        #
        # This gate is the STANDALONE-wrapper twin of the rows' decline
        # (`thd_d_shapes` on every Rubin row), which the rows
        # cannot cover because the wrapper never consults them.  Without it
        # check_support() returns True and compile() dies with a bare TypeError
        # on the lse_head_major kwarg -- an untyped escape, not a decline.
        # KEEP THE TWO IN LOCKSTEP: widening one without the other either
        # admits a graph that then dies untyped (row wider), or declines a graph
        # the row advertises (wrapper wider).  Contract rule 8b'.
        self._not_implemented_error_if(
            self.thd
            and self._device_cc == (10, 7)
            and not (
                (self._fp8 and self._pertensor and (int(d_qk), int(d_v)) in _SM107_FP8_THD_SHAPES)
                or (self._fp8 and not self._pertensor and (int(d_qk), int(d_v)) in _SM107_MXFP8_THD_SHAPES)
                or (not self._fp8 and (int(d_qk), int(d_v)) in _SM107_F16_THD_SHAPES)
            ),
            f"THD/varlen on the Rubin (SM107) line is per-tensor FP8 {sorted(_SM107_FP8_THD_SHAPES)}, "
            f"MXFP8 {sorted(_SM107_MXFP8_THD_SHAPES)} or f16/bf16 {sorted(_SM107_F16_THD_SHAPES)} only; "
            f"got (D_QK={d_qk}, D_V={d_v}) on the "
            f"{'MXFP8' if (self._fp8 and not self._pertensor) else 'FP8' if self._fp8 else 'f16/bf16'} path",
        )
        if self.gate_desc is not None:
            # STANDALONE twin of the rows' `epilogue_gate` / `epilogue_gate_d_shapes`
            # -- KEEP IN LOCKSTEP (rule 8b / 8b').  The rows and this block read
            # ONE constant (config_sm107.SM107_EPILOGUE_GATE_SHAPES), so the shape
            # claim cannot drift; the feature interactions below mirror mismatch()
            # clause for clause.  EXACT dims, not the envelope flavor: the gate
            # tile is TMA'd at the kernel's TILE_O and a zero-padded G has never
            # been validated, so d=200 (which the envelope pads to (256, 256))
            # is declined here as it is in the rows.
            self._not_implemented_error_if(
                self._device_cc != (10, 7),
                "epilogue gate fusion (O * sigmoid(G)) is served by the Rubin (SM107) d256 kernels only",
            )
            self._not_implemented_error_if(
                (int(d_qk), int(d_v)) not in _SM107_EPILOGUE_GATE_SHAPES,
                f"epilogue gate fusion is wired at head dims {sorted(_SM107_EPILOGUE_GATE_SHAPES)} only; got (D_QK={d_qk}, D_V={d_v})",
            )
            # MXFP8 (PR-B): the Rubin d256 block-scale kernel carries the same gate
            # seams as the f16 / per-tensor FP8 bodies, so the block-scale path is
            # admitted on the SAME terms as FP8 above -- cc 10.7, exactly (256, 256),
            # a bf16 G (checked below) -- and nothing narrower: the two clauses
            # above already pin the arch and the shape, so no MXFP8-specific decline
            # remains here (its row: engines._sm107_mxfp8_spec, epilogue_gate=True).
            # What differs is the OUTPUT: the MXFP8 kernel has no per-tensor
            # scale_o (block scales dequantize in-MMA), so a gated e4m3 O is written
            # UNSCALED -- the same unscaled e4m3 O the ungated MXFP8 path writes; a
            # `scale_o` handed to execute() is not applied on the MXFP8 path (as
            # today, gate or no gate).  Callers that need a scaled quantized O
            # keep a bf16 O and quantize downstream (gated-attention block D8).
            # Amax_O (when requested) follows the FP8 path's contract: the amax of
            # the UNGATED normalised O -- the sdpa node's output, independent of G
            # -- in the O's own units here (there is no scale_o to divide by).
            #
            # THD (packed sequences) x gate: SERVED on the f16/bf16 d256 prefill, the one THD body with
            # the gate seams -- its host maps G in the packed [1, T, H, D] form Q / O take (the kernel
            # body is the dense gated body) and the prepared THD binder binds G per call.  The quantized
            # THD bodies (per-tensor FP8 / MXFP8) build the dense gate map and their prepared launch
            # carries no gate: declined by name.  The GRAPH path keeps its "dense-only" decline on the
            # rows (engines.mismatch: the graph's THD lowering binds no gate operand) -- the standalone
            # adapter is wider than the row here, by design, never the reverse (rule 8b: a row must not
            # admit what the adapter declines).
            self._not_implemented_error_if(
                self.thd and self._fp8,
                "epilogue gate fusion under THD (packed sequences) is served by the f16/bf16 d256 prefill only: "
                "the quantized (FP8 / MXFP8) THD bodies have no THD gate descriptor",
            )
            self._not_implemented_error_if(
                self.thd and not _thd_binder_serves_the_gate(),
                "epilogue gate fusion under THD needs the prepared THD binder's gate role; this build's pybind extension "
                "predates it (rebuild the extension, or run the gate as its own stage)",
            )
            # The three kernel-side interactions below vanish when the gate rides the split
            # COMBINE (the d256 decode tile, split_kv > 1: the tile writes un-gated fp32
            # partials and sm100/split_combine applies sigmoid(G) to the merged value, so
            # the kernel's gate TMA box, its paged flavor and its packed rows are all
            # irrelevant); engines.mismatch lifts the same three through
            # d256_decode_tile_selected(..., split_kv) -- keep them in lockstep.
            _gate_in_combine = self._gate_in_combine_for(int(s_qo), int(h_qo), int(h_kv))
            # The gate on the PAGED d256 half PREFILL body: a Rubin paged half (256, 256) plan that is not a decode-shaped one
            # loads the SHARED SM100 body (_load_sm100_kernel_module's paged arm: sm100/prefill_d256_f16.py compiled for sm_107a),
            # whose paged / THD / PackGQA arms carry the gate seams -- the serving EXTEND form's SDPA (packed queries over the
            # pools, PackGQA on or off) and dense paged queries past the decode tile alike; unsplit only (the split's partials
            # bypass the gated epilogue).  The GRAPH rows keep their narrower claim (the graph's THD lowering binds no gate):
            # a standalone adapter wider than the row is the admitted direction (rule 8b), never the reverse.
            _gate_on_paged_prefill = (
                self._device_cc == (10, 7)
                and not self._fp8
                and self.paged
                and (int(d_qk), int(d_v)) == (256, 256)
                and self.split_kv <= 1
                and not self._decode_q_tile_for(int(s_qo), int(h_qo), int(h_kv))
            )
            self._not_implemented_error_if(
                _gate_on_paged_prefill and not _binders_serve_the_paged_gate(),
                "epilogue gate fusion over paged KV needs the prepared binders' paged-gate role; this build's pybind extension "
                "predates it (rebuild the extension, or run the gate as its own stage)",
            )
            # The split FIRST, so a split plan hears the feature it asked for (a paged split's decline would otherwise read "paged").
            self._not_implemented_error_if(
                self.split_kv > 1 and not _gate_in_combine,
                "epilogue gate fusion with split_kv > 1 is not supported: the combine would write the un-gated O "
                "(the d256 decode tile's split applies the gate in its combine)",
            )
            self._not_implemented_error_if(
                self.paged and not _gate_in_combine and not _gate_on_paged_prefill,
                "epilogue gate fusion with paged KV is served on the Rubin d256 half prefill body (unsplit, packed or dense queries "
                "past the decode tile) and by the d256 decode tile's split combine; this plan is neither",
            )
            self._not_implemented_error_if(
                self.pack_gqa and not _gate_in_combine and not _gate_on_paged_prefill,
                "epilogue gate fusion with PackGQA is served on the Rubin d256 half PAGED prefill body (unsplit, packed queries) and by "
                "the d256 decode tile's split combine; the dense d256 prefill kernel runs unpacked",
            )
            if _gate_in_combine:
                # The gated combine is reached through the prepared dense launch only
                # (the staged path binds the split through the same native binder, which
                # knows no gate on a split): the operands must bind zero-copy.
                self._not_implemented_error_if(
                    not self._can_prepare_dense_layout(),
                    "the gate-in-combine split (the d256 decode tile) needs Q/K/V layouts the prepared dense launch binds zero-copy (BSHD-physical)",
                )
            # Only now the request's OWN well-formedness (ValueError), so a
            # caller on another arch line / flavor first hears "not served
            # here", not "wrong gate dtype".  The gate multiplies O
            # element-wise: exactly O's logical shape (a broadcast gate is a
            # legal GRAPH that FROST declines, not a layout).
            self._check_tensor_shape(self.gate_desc, (b, h_qo, s_qo, d_v), name="GATE")
            # The gate rides the half kernels in Q's dtype (one STORAGE_DTYPE
            # for Q, O and the gate tile) and the quantized kernel as bf16 (its
            # GATE_STORAGE_DTYPE; an e4m3 gate would quantize the sigmoid's
            # input, and the fp8 kernel's O geometry is not the gate's).
            self._check_dtype(
                self.gate_desc,
                torch.bfloat16 if self._fp8 else self.dtype,
                name="GATE",
                extra_error_msg=(
                    "the epilogue gate is bfloat16 on the quantized (FP8) path" if self._fp8 else "the epilogue gate must match Q dtype on the f16/bf16 path"
                ),
            )
            # The gate binds zero-copy or not at all: compact, or a TMA-expressible
            # token-major declaration the artifact is compiled at (a G sliced out
            # of a fused projection slab).  ONE rule decides -- the same
            # `epilogue_gate_layout_declarable` the engine rows apply to G in
            # mismatch() -- so a graph that entered the ranked list cannot die
            # here (rule 8b lockstep).  It is STRICTER than dense_layout_ok:
            # a head-major (torch-contiguous [B, H, S, D]) or 16-byte-unaligned
            # G is a legal Q/K/V/O layout (those take the normalisation copy)
            # but not a gate layout, because no hidden copy is made for the gate.
            self._value_error_if(
                not _epilogue_gate_layout_declarable(tuple(self.gate_desc.shape), tuple(self.gate_desc.stride), self.gate_desc.dtype.itemsize),
                f"GATE layout is not TMA-expressible for a zero-copy binding: the gate must be BSHD-compact or a token-major "
                f"declaration (D innermost-contiguous, then heads, then tokens; seq and head strides 16-byte multiples; covering) "
                f"-- a head-major or unaligned G is neither, and no hidden copy is made for the gate; got BHSD strides {tuple(self.gate_desc.stride)}",
            )
        self._value_error_if(
            not self.has_amax_o and not self._fp8,
            "has_amax_o=False is meaningful on the quantized (FP8 / MXFP8) path only (the half kernels produce no Amax_O)",
        )
        # Dense padded-Q trim backstops (engines.lower_dsl_prefill never sets
        # these combinations; a direct caller could).
        self._value_error_if(
            self.seq_q_lens_present and self.thd,
            "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )
        self._value_error_if(
            self.seq_q_lens_present and self._fp8 and not self._quantized_q_lens_abi,
            "seq_q_lens_present (dense padded-Q LSE trim) is not supported by the selected quantized flavor",
        )
        # KV-tail correctness: the kernel zero-fills the last KV tile via TMA
        # OOB but only *masks* those columns on the padded / causal paths. A
        # ragged S_kv is safe when a padding mask carries the real lengths, or
        # when the causal diagonal provably covers the tail (kv >= S_kv implies
        # kv > q for every query row). Otherwise the plan compiles the padded
        # mask against the scalar S_kv (kv_tail_mask, #1425).
        self._kv_tail_mask = False
        if int(s_kv) % _SM100_TILE_N != 0:
            # A right-widened band pushes the last unmasked column to
            # (S_q - 1) + R (top-left) or (S_kv - 1) + R (bottom-right), so the
            # KV tail is only provably masked when it stays below S_kv.
            _br = int(self.window_size_right or 0)
            causal_covers_tail = self.is_causal and ((self.causal_bottom_right and _br == 0) or (not self.causal_bottom_right and int(s_qo) + _br <= int(s_kv)))
            self._kv_tail_mask = not (self.seq_kv_lens_present or causal_covers_tail)
            self._value_error_if(
                self._kv_tail_mask
                and (self.split_kv or 1) > 1
                and not native_fp8_padded_split
                and not supports_scalar_kv_tail_split((int(d_qk), int(d_v)), device_cc=self._device_cc, fp8=self._fp8, pertensor=self._pertensor),
                f"S_kv ({s_kv}) is not a multiple of {_SM100_TILE_N}: split-KV needs a padding mask (seq_len_kv) or a causal mask covering the KV tail",
            )

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)
        self._not_implemented_error_if(self.scale_softmax == 0, _ZERO_SCALE_UNSUPPORTED)
        if self.softmax_scale_prefolded:
            # The pre-folded contract: Q already carries attn_scale * log2(e) and the kernel traced no per-score
            # scale, so every derived scale_softmax_log2 (= scale_softmax * log2 e) is pinned to exactly 1.0 here,
            # at the one place the adapter's scale is resolved.
            self.scale_softmax = 1.0 / math.log2(math.e)
        # A negative scale is served by negating S in BMM1 and running at |scale| (#1435).
        self._score_negated = self.scale_softmax < 0

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _decode_q_tile_for(self, s_q: int, h_q: int, h_kv: int, b: Optional[int] = None, s_kv: Optional[int] = None) -> int:
        """:meth:`_decode_q_tile` on explicit shapes -- for ``check_support``, which
        reads the sample shapes before it records ``s_q_max`` / ``h_q`` / ``h_kv``
        (its Rubin paged gate admits the decode-shaped graph the decode tile serves).
        ``b`` / ``s_kv`` are the launch size the Blackwell line's wide-tile route reads
        (config_sm100.d256_wide_decode_launch); without them only the 16-column tile is
        routed there."""
        if self._fp8 or self.thd or self.flavor != (256, 256):
            return 0
        if self.softmax_scale_prefolded:
            # Not wired on the decode tile (the scale is applied in-kernel): the d256
            # prefill kernel serves it (config_sm107's make_cfg_d256_decode declines
            # such a record as the backstop).
            return 0
        if self.gate_desc is not None and self.split_kv <= 1:
            # The tile has no gate seams either, so a GATED graph rides it only through a
            # split: the combine applies ``O *= sigmoid(G)`` to the fp32 merged value
            # (sm100/split_combine, gate=True -- :meth:`_gate_in_combine`); unsplit, the
            # d256 prefill kernel's fused epilogue serves the graph as before.
            return 0
        pack_g = (int(h_q) // int(h_kv)) if self.pack_gqa else 1
        if self._device_cc == (10, 7):
            # The Rubin route (config_sm107.decode_d256_q_tile, table-driven): the 32- and 64-column tiles are routed in
            # one token unit (the 64-column width for the forms its predicate admits -- a packed group over a dense cache,
            # or over pages under the causal / window mask: the record's MASK_CAUSAL / MASK_SWA bits, read here off the
            # declared band exactly as config_sm100._mask_flags_from derives them), then up to two TOKEN UNITS of the
            # widest admitted width for a packed MTP step; decode_q_tile_max caps the table for this plan.
            # engines.decode_d256_q_tile_for_row is the facts-level twin of this dispatch.
            return _decode_d256_q_tile_sm107(
                int(s_q),
                pack_g,
                paged=bool(self.paged),
                masked=self.window_right is not None or self.window_left is not None,
                max_q_tile=self.decode_q_tile_max,
            )
        # (a cap off cc 10.7 was refused once, in check_support, before this dispatch is reached)
        wide_launch = b is not None and s_kv is not None and d256_wide_decode_launch(int(b), int(h_q) // pack_g, int(s_kv), self.window_left is not None)
        return decode_d256_q_tile(int(s_q), pack_g, wide_launch=wide_launch)

    def _gate_in_combine_for(self, s_q: int, h_q: int, h_kv: int) -> bool:
        """:meth:`_gate_in_combine` on explicit shapes (for ``check_support``, like
        :meth:`_decode_q_tile_for`)."""
        return self.gate_desc is not None and self.split_kv > 1 and self._device_cc == (10, 7) and self._decode_q_tile_for(s_q, h_q, h_kv) > 0

    def _gate_in_combine(self) -> bool:
        """Whether this plan's fused epilogue gate rides the split COMBINE instead of the
        kernel epilogue: a gated f16/bf16 d256 graph lowered onto the Rubin decode tile
        with ``split_kv > 1``.  The tile is compiled WITHOUT the gate (it has no gate
        seams; ``TemplateParams.epilogue_gate`` stays False) and writes its fp32
        partials as any split does; ``sm100/split_combine.compile_ptr(gate=True)``
        multiplies the merged fp32 value by ``sigmoid(G)`` before the single cast --
        FROST's one-rounding convention, the same arithmetic as the fused kernels'
        epilogue (module docstring there).  ``engines.d256_decode_tile_selected``
        (its ``split_kv`` argument) is the facts-level twin; keep the two in lockstep."""
        return self._gate_in_combine_for(self.s_q_max, self.h_q, self.h_kv)

    def _decode_q_tile(self) -> int:
        """N extent of the decode tile (config_sm100.decode_d256_q_tile) when this
        plan lowers onto the d256 decode tile -- sm100/decode_d256_f16.py on the
        Blackwell line, sm107/decode_d256_f16.py on cc 10.7 -- else 0 (the prefill tile).

        A LOWERING choice, like the flavor pick: the decode tile serves the same
        graph contract as prefill_d256_f16 (paged / dense, padding, causal
        bottom-right, SWA, right band, sink, Stats natural or base-2, dense
        padded-Q trim, split-KV partials) for f16/bf16 d256-flavor graphs whose
        S_q x packed-heads rows fit the line's routed envelope -- the 16-wide N
        tile on SM100 / SM103 (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS; the
        32-wide tile is issue-bound per CTA and is routed there only for a launch
        big enough to be HBM-bound, config_sm100.d256_wide_decode_launch), on
        cc 10.7 also the 32-wide tile in up to two TOKEN UNITS of the
        kernel for a packed MTP step (config_sm107.decode_d256_q_tile; the value
        is the rows of ONE unit); everything else (THD, quantized, larger S_q,
        the pre-folded scale, the fused epilogue gate at split_kv == 1) stays on
        the prefill tile.  A GATED graph rides the tile
        when it splits: the gate moves into the split combine
        (:meth:`_gate_in_combine`).  The TILE_CGA_M / SCHED_POLICY knobs describe
        the prefill pipeline and are no-ops here (one cta_group::1 CTA per unit,
        nothing to schedule).  engines.d256_decode_tile_selected is the
        facts-level twin; keep the two in lockstep.
        """
        return self._decode_q_tile_for(self.s_q_max, self.h_q, self.h_kv, self.batch_size, self.s_k_max)

    def _fp8_dual_kv_stream(self) -> bool:
        """Whether this per-tensor FP8 d128 plan runs its two Q sub-tiles as two KV
        streams over one 128-row Q tile (TemplateParams.dual_kv_stream).

        The same LOWERING choice as the f16 decode tiles: the decode-shaped step
        of sm100/prefill_d128_fp8.py is bound by its per-KV-tile chain, not by
        the MMA, so a 128-row graph gains nothing from the second Q sub-tile's
        rows and everything from a second KV tile per step.  cga1 only (the pair
        shares no K/V ring), dense or paged K/V, not THD; off while
        config_sm100.FP8_DUAL_KV_STREAM_ROUTED is False.
        """
        if not fp8_dual_kv_stream_routed() or not (self._fp8 and self._pertensor) or self.thd or self.flavor != (128, 128) or self._device_cc == (10, 7):
            return False
        if self.cga != 1:
            # The width must be NAMED: an unset knob resolves to this flavor's
            # cga2 default below (template_params), and the pair shares no ring.
            # The heuristics name cga1 for every dense / paged fp8 d128 plan.
            return False
        pack_g = (self.h_q // self.h_kv) if self.pack_gqa else 1
        return int(self.s_q_max) * pack_g <= FP8_DUAL_KV_STREAM_MAX_ROWS

    def _d64_decode_tile(self) -> bool:
        """Whether this d64 plan lowers onto the 128-row decode tile.

        The twin of :meth:`_decode_q_tile`, and the same LOWERING choice: the
        decode tile serves the d64 flavor's whole graph contract (paged / dense,
        padding, causal bottom-right, SWA, right band, sink, Stats natural or
        base-2, split-KV partials) for graphs whose S_q x packed heads fit one
        128-row tile -- S_q = 1 decode and MTP. Explicit paged THD split
        plans use the same tile across multiple Q blocks. Other THD, larger
        dense S_q and an explicit cga2 stay on the prefill tile.  d128 keys the same
        tile off cga1; d64 cannot, because cga1 IS its prefill width.
        """
        if self._fp8 or self.flavor != (64, 64) or self._device_cc == (10, 7):
            return False
        if self.bias_desc is not None or self.kv_mask_desc is not None:
            # The decode tile carries no additive-bias / kv_mask arm (compile() refuses the operands): a d64 plan with either
            # stays on the prefill tile at its width -- the kv_mask alone at cga1 (nothing staged in SMEM), the bias at the
            # width whose resident block holds S_kv (cga2 on this line).  Found by the cc 10.0 tier: a kv_mask-only d64 graph
            # at S_q 128 lowered onto the decode tile and failed at build instead of running the prefill tile.
            return False
        if self.cga not in (None, 1):
            # An explicit cga2 is the prefill pipeline (the f16 row admits both
            # widths at d64): the decode tile is cga1-only (make_cfg_d64_decode),
            # so a pinned or autotuned cga2 must not reach it through this flag.
            return False
        pack_g = (self.h_q // self.h_kv) if self.pack_gqa else 1
        # the Q extent: s_q_max once check_support resolved it, the declared Q shape before (the padded tail form's eligibility asks earlier)
        s_q = int(self.s_q_max) if self.s_q_max is not None else int(self.q_desc.shape[2])
        if self.thd:
            # Paged THD: the packed split (the combine places the rows) or, unsplit, a declared S_q that fits
            # one 128-row tile -- FlashInfer's paged extend at up to 128 new tokens per request, where the
            # 256-row prefill unit would run a dead second sub-tile for every unit. A taller declared S_q
            # stays on the prefill unit: multi-tile decode units only break even under a sliding window and
            # lose without one. The ragged (non-paged) K/V leg stays on the prefill tile, whose packed-total
            # clamped descriptors it needs.
            fits = self.paged and self.split_kv == 1 and not self.pack_gqa and s_q <= _D64_DECODE_TILE_ROWS
            return self.packed_thd_split or fits
        return s_q * pack_g <= _D64_DECODE_TILE_ROWS

    def _d64_decode_quad(self) -> bool:
        """Whether the d64 decode tile runs its quadrant-cooperative softmax (``TemplateParams.decode_quad``).

        The live Q rows -- S_q x the packed group -- must fit one 32-row quadrant: the tile loads them into all four TMEM lane quadrants and each
        softmax warp owns a quarter of the key columns, which is what lifts the one-warp-owns-all-live-rows bound of a decode step.  A larger S_q
        keeps the 128-row tile's single-quadrant softmax.  Dense / paged, split or not, with or without a sink (not THD: the tile's THD legs keep the staged
        O store the merge replaces).
        """
        if not self._d64_decode_tile() or self.thd or tuple(self._device_cc) not in _EARLY_Q_CC:
            return False
        pack_g = (self.h_q // self.h_kv) if self.pack_gqa else 1
        return int(self.s_q_max) * pack_g <= _D64_QUAD_ROWS

    def _decode_early_q(self, params) -> bool:
        """Whether the decode tile (sm100/decode_d128_f16.py: the d64 decode tile, the d128 cga1 tile) takes its early-Q form (``TemplateParams.early_q``)
        on this build. Unsplit THD keeps its staged O store; split THD writes FP32 partials directly and leaves the Q slab free."""
        if tuple(self._device_cc) not in _EARLY_Q_CC or self._fp8:
            return False
        if self.thd and not (tuple(self._device_cc) == (10, 0) and tuple(self.flavor) == _SM100_DECODE_FLAVOR and params.cta_mma == 1 and params.split_kv > 1):
            return False
        return bool(params.decode_tile) or (tuple(self.flavor) == _SM100_DECODE_FLAVOR and params.cta_mma == 1)

    def template_params(self) -> Sm100TemplateParams:
        """The compile-time record ``compile()`` loads the kernel module with.

        Exposed (rather than inlined in ``compile``) so a caller -- a probe, a
        test, a block that wants to know WHICH specialization it is driving --
        reads the record the adapter really loads instead of re-typing it,
        which is how two copies would drift (engine contract S6:
        ``TemplateParams`` is the OUTPUT of the match).  ``epilogue_gate`` is
        keyed here on ``sample_gate``, so the gated and ungated forms of one
        flavor are two coexisting kernel modules.  Pure extraction of the
        former ``compile()`` prologue; requires ``check_support()``.
        """
        self._ensure_support_checked()
        # cc10.3+ fuses the S_acc row-max into the LDTM (tcgen05.ld.red.f32.max).
        # Wired in the quantized kernels (MXFP8, per-tensor FP8 d128 / d192x128) and
        # the f16 d128 kernel's unmasked softmax arm; the other f16 kernels do not
        # read this flag, and the SM107 siblings carry the instruction unconditionally.
        # Paged half D128 on SM107 uses the shared SM100 body, so it needs the
        # same explicit flag as SM103 to avoid a software row-max reduction.
        fused_ldtm_stat = (self._fp8 or tuple(self.flavor) == (128, 128)) and (
            self._device_cc == (10, 3) or (self._device_cc == (10, 7) and self.paged and not self._fp8)
        )
        # The exp2 MUFU / FMA split is on ONLY where it was measured (cc 10.0 x the d128 MXFP8 / d128 FP8 /
        # d192x128 f16 / d192x128 FP8 kernels) -- see _exp2_fma_split_for; the kernels not listed there never
        # read the field.
        exp2_fma_split = _exp2_fma_split_for(
            self._device_cc, kind=_quant_kind(self._fp8, self._pertensor), flavor=self.flavor, decode_tile=self._d64_decode_tile()
        )
        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path never hits this branch —
        # the heuristic emits an explicit policy (the same primary this
        # derivation picks) and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            # THD is excluded: the LPT decodes assume a dense rectangular
            # tile space, while a ragged batch carries its own scheduler,
            # which walks the live units through batch_remap.
            #
            # Rubin is excluded for a different reason.  `_causal_sched_policy`
            # can pick SCHED_LPT_L2, which only the d128 / d192x128 FP8 and
            # MXFP8 kernels honour (the f16, d256 and d512 call sites do not
            # pass its qh_per_kh / seqlen_kv inputs), and every LPT variant is
            # claimed PER FLAVOR on the SM107 rows (`sched_policies_by_d_shape`:
            # f16 (128, 128) / (256, 256) LPT; FP8 (256, 256) LPT, (128, 128) and (192, 128)
            # LPT + LPT_L2;
            # MXFP8 (128, 128) and (192, 128) LPT + LPT_L2 -- each validated
            # bit-identical to NATURAL), not row-wide: the d512 role-split
            # kernels still lack the
            # `lpt_q_tiles_in_cga_units` argument (#1001) and write nothing
            # under LPT.  The wrapper never consults a row, so this derivation
            # stays NATURAL on Rubin and a standalone caller REQUESTS
            # `sched_policy=SCHED_LPT` for a validated flavor (the gated
            # attention block does).  Folding the rows' per-flavor domain into
            # this derivation is the follow-up.  (The 2026-09-08 "causal d512
            # FP8 -> NaN" this comment used to cite was that missing argument.)
            _rubin = self._device_cc == (10, 7)
            if self.window_right is not None and not self.thd and not _rubin:
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost import data_type as _cudnn_dtype

        params = Sm100TemplateParams(
            dtype_qkv=_SM100_DTYPE_QKV_CODE[self.dtype],
            # A quantized-O split compiles the kernel to write HALF partials;
            # the combine performs the single cast to the real O dtype.
            dtype_o=(_SM100_DTYPE_QKV_CODE[torch.float16] if self._quantized_split() else self._dtype_o_code),
            window_left=self.window_left,
            window_right=self.window_right,
            bottom_right=self.causal_bottom_right,
            # Quantized and packed partial normalizers exclude the sink. The
            # prepared plan retains the logical sink and folds it once in combine.
            has_sink=self.has_sink and not self.packed_thd_split and not (self._fp8 and self.split_kv > 1),
            # Packed split owns the final combine in this template. Its main
            # kernel keeps partial Stats natural-log; retain the requested
            # final base for the combine instead of discarding it here.
            stats_log2=self.stats_log2 and (self.split_kv == 1 or self.packed_thd_split),
            seq_kv_lens_present=self.seq_kv_lens_present,
            seq_q_lens_present=self.seq_q_lens_present,
            kv_tail_mask=getattr(self, "_kv_tail_mask", False),
            sched_policy=sched_policy,
            # The ragged-Q decode leg is a mode of the dense decode tile, not
            # the prefill tile's THD_VARLEN leg (mutually exclusive params).
            thd_varlen=self.thd and not self.thd_decode_leg,
            ragged_q=self.thd_decode_leg,
            pack_gqa=self.pack_gqa,
            qh_per_kh=int(self.q_desc.shape[1]) // int(self.k_desc.shape[1]),
            split_kv=self.split_kv,
            # d64 defaults to cga1, the width cuDNN's own native kernel picks
            # for this geometry: the collective cga2 MMA exists to halve
            # per-CTA K/V, which the halved d64 slabs no longer need, and a
            # 2-CTA cluster doubles the Q rows a cluster must cover -- wasted
            # work under a narrow diagonal band.
            # d64 prefill runs cga1: the narrow slabs need no collective MMA to
            # halve K/V, and a 512-row cga2 cluster wastes most of a narrow
            # diagonal band. Its decode leg is picked by decode_tile, not cga.
            # The Rubin small tiles (d32 / d64 on the sm107 d128 file) likewise default to cga1 -- their validated width
            # (config_sm107.SM107_SMALL_D_CGAS; at TILE_O = 32 a cga2 V operand would be a 32-byte MN-major atom).
            # A BIAS plan without a requested width takes the narrowest one whose resident block holds S_kv
            # (_resident_arm_cga: the heuristics' pick), not the small tiles' cga1 default; a kv_mask-only plan stages no
            # block and keeps the tiles' own rule below.
            cta_mma=(
                self._resident_arm_cga()
                if self.bias_desc is not None
                else (
                    (
                        1
                        if (self._fp8 and self.flavor == (256, 256))
                        or self.flavor in ((32, 32), (64, 64))
                        or (self._device_cc == (10, 7) and self.flavor in _SM107_F16_SMALL_D_SHAPES)
                        else 2
                    )
                    if self.cga is None
                    else self.cga
                )
            ),
            d_flavor=self.flavor[0] if self.flavor in ((32, 32), (64, 64), SM100_NATIVE_D576_SHAPE) else 128,
            # The bias arms (resident block / streamed ring): a compile-time axis of the d128 file (check_support admitted the operands and the form).
            bias_mode=self._resident_arm_bias_mode(),
            kv_mask_form=self._kv_mask_form(),
            # The KV tile width (the TILE_N knob): 128 on every record so far, 64 on the Rubin small tiles (check_support validated it).
            tile_n=self._effective_tile_n(),
            # The (n, n+1) pair item's chunk length (0 = the one-batch-entry item): the request, else the heuristics' pick.
            n_chunk=self._resident_arm_n_chunk(),
            # The streamed ring's depth (0 off the ring): the request, else the derived deepest (2 / 3).
            bias_stages=self._resident_arm_bias_stages(),
            # The PLAIN padded body's tail form on the Rubin small tiles (0 = the legacy second masked body, 1 = the pre-masked tail): the
            # request, else the pick; 0 on every other rendering.
            padded_tail_form=self._padded_tail_form(),
            # The dead-sub-tile quiet body of the one-batch-entry item (0 = the full body on a dead sub-tile, 1 = the quiet body): the
            # request, else the pick; 0 on every other rendering.
            dead_subtile_quiet=self._dead_subtile_quiet(),
            # The row-max form of the bias / kv_mask body (0 = the software tree, 1 = the hardware upper bound): the request, else the pick.
            bias_max_form=self._resident_arm_bias_max_form(),
            # The softmax warp-group count (2 = every record so far; 3 = the third softmax warp group of the TILE_N-64 pair item, an experiment).
            softmax_wgs=self._resident_arm_softmax_wgs(),
            # The S / P TMEM slot count per sub-tile (1 on every record but the Rubin small tiles' double-buffered S at TILE_N 64):
            # config_sm107.sm107_s_slots_pick -- the measured rule (2 on the plain TILE_N 64 body, else 1), or the sweep seam's value verbatim
            # (a typed factory raise off its domain).
            s_slots=self._s_slots(),
            # Q resident in TMEM (1 = the MMA warp's once-per-step SMEM -> TMEM copy of Q and the A-from-TMEM BMM1 on the Rubin small tiles):
            # config_sm107.sm107_q_in_tmem_pick -- the measured rule on the record's geometry (the plain 128-wide body, the bias-only 64-wide body,
            # the bias + kv_mask 64-wide body at D 64 without per-row lengths), or the sweep seam's value verbatim (a typed factory raise off its domain).
            q_in_tmem=self._q_in_tmem(),
            fused_ldtm_stat=fused_ldtm_stat,
            exp2_fma_split=exp2_fma_split,
            cga1_l2_prefetch=_cga1_l2_prefetch_for(self._device_cc, kind=_quant_kind(self._fp8, self._pertensor), flavor=self.flavor),
            softmax_f16=self.softmax_precision == _cudnn_dtype.HALF,
            softmax_scale_prefolded=self.softmax_scale_prefolded,
            negate_scores=self._score_negated,
            paged_kv=self.paged,
            page_size=self.paged_page_size,
            pv_bf16=self.pv_bf16,
            # Preserve the existing MXFP8 kernel contract. The hybrid path
            # compiles the atomic reduction only when its plan declares the
            # output, so a runtime execute() argument cannot silently change
            # the launched kernel.
            emit_amax_o=(not self.pv_bf16) or self.has_amax_o,
            # A gate that rides the split COMBINE (the decode tile, _gate_in_combine) is
            # not the kernel's: the tile has no gate seams and compiles ungated.
            epilogue_gate=self.gate_desc is not None and not self._gate_in_combine(),
            thd_batch_one=(
                self.packed_thd_split
                or (
                    self.thd
                    and not self.thd_decode_leg
                    and not self._fp8
                    and self.flavor == (128, 128)
                    # Qualified prefill variants of the shared template.
                    # Other variants keep their existing compiled record.
                    and ((self._device_cc == (10, 0) and self.dtype == torch.bfloat16) or (self._device_cc == (10, 7) and self.paged))
                )
            )
            and self.batch_size == 1,
        )
        if self.flavor == (192, 128):
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import select_d192_auto_knobs

            auto_sched, auto_cga = select_d192_auto_knobs(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
                paged=bool(self.paged),
                has_sink=bool(self.has_sink),
                gqa_ratio=int(params.qh_per_kh),
                device_cc=self._device_cc,
            )
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d192_lowering(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
            params = derive_d192_internal_params(
                params,
                pertensor=self._pertensor,
                batch_size=self.batch_size,
                h_q=self.h_q,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
        elif self.flavor == (256, 256):
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import select_d256_auto_knobs

            auto_sched, auto_cga = select_d256_auto_knobs(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
                device_cc=self._device_cc,
                paged=self.paged,
            )
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d256_lowering(params, s_q=self.s_q_max, s_kv=self.s_k_max)
            params = derive_d256_internal_params(
                params,
                pertensor=self._pertensor,
                batch_size=self.batch_size,
                h_q=self.h_q,
                s_q=self.s_q_max,
            )
            decode_q_tile = self._decode_q_tile()
            if decode_q_tile:
                params = replace(params, decode_q_tile=decode_q_tile)
        elif self.flavor == (128, 128) and not self._fp8 and self._device_cc != (10, 7) and self.cga is None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import select_d128_auto_cga

            # The standalone tier's default width is the graph heuristics' own
            # (select_d128_auto_cga): cga1 -- the DECODE tile, which
            # _load_sm100_kernel_module selects for TILE_CGA_M=1 -- when one
            # 128-row tile covers every live row of a (batch, packed head)
            # unit, cga2 -- the prefill pipeline -- otherwise.  The f16 row's
            # cgas_by_d_shape, supported_cgas_for above and this default stay
            # in lockstep; a requested cga was honored verbatim in the params.
            # (thd_decode_leg is False here: the ragged-Q leg needs cga == 1
            # requested, so a ragged graph with no width keeps the prefill tile.)
            # cc 10.7 keeps cga2 as the STANDALONE default width on this flavor (the graph heuristics, which always pass
            # an explicit cga, pick the shared decode tile there -- issue #1472); an explicit cga=1 is honoured through
            # supported_cgas_for.
            params = replace(
                params,
                cta_mma=select_d128_auto_cga(
                    s_q=self.s_q_max,
                    # The kernel's PACK_G: the whole group when it divides the tile,
                    # its largest divisor that does (partial PackGQA), 1 unpacked.
                    pack_g=pack_gqa_group_size(int(params.qh_per_kh), CfgD128.TILE_M, partial=True) if self.pack_gqa else 1,
                    thd=self.thd,
                    thd_decode_leg=self.thd_decode_leg,
                ),
            )
        if self._d64_decode_tile():
            params = replace(params, decode_tile=True)
        if self._decode_early_q(params):
            params = replace(params, early_q=True)
        if self._d64_decode_tile():
            params = replace(params, decode_quad=self._d64_decode_quad())
        if self._fp8_dual_kv_stream():
            params = replace(params, dual_kv_stream=True)
        elif self._device_cc != (10, 7) and self.flavor == (512, 512) and self._fp8 and not self._pertensor:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import select_d512_auto_knobs

            auto_sched, auto_cga = select_d512_auto_knobs(params)
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d512_mxfp8_lowering(params, s_q=self.s_q_max, s_kv=self.s_k_max)
        if self.flavor == SM100_NATIVE_D576_SHAPE and not self._fp8:
            params = replace(params, mma_2x2=True)
        if self.flavor == (512, 512) and not self._fp8:
            # The legacy D512_2X2 switch selects unsplit, unpaged half plans
            # whose entire packed group divides the 64-row tile. Native SM107
            # dense packed/split plans below always use their qualified 2x2 body.
            two_by_two = D512_2X2 and self.split_kv == 1 and not self.paged and (not self.pack_gqa or 64 % max(1, int(params.qh_per_kh)) == 0)
            # New native packed/split plans require this body even if the
            # legacy unpacked A/B seam selects the role-split implementation.
            two_by_two = two_by_two or (self._rubin_native_d512_leg() and (self.pack_gqa or self.split_kv > 1))
            if two_by_two:
                params = replace(params, mma_2x2=True)
        return params

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        self._staged_spec = None
        if (self._fp8 and not (self._can_prepare_fp8() or self._can_prepare_mxfp8())) or (not self._fp8 and not self._can_prepare_dense_layout()):
            from .prepared_staged_forward import compile_plan

            self._dense_spec = self._thd_spec = None
            self._prepared_fp8 = self._prepared_mxfp8 = False
            self._staged_spec = compile_plan(self)
            self._compiled_kernel = self._staged_spec.core.owner
            self._logger.debug("compile completed")
            return
        params = self.template_params()
        self._k_mod = _load_sm100_kernel_module(self.flavor, params, fp8=self._fp8, pertensor=self._pertensor, rubin=(self._device_cc == (10, 7)))
        # Which template serves this plan (its file stem, e.g. "prefill_d256_f16" /
        # "decode_d256_f16"): a lowering choice the engine's executor exposes so
        # a test can assert the route without inferring it from the source.
        self.kernel_template = os.path.splitext(os.path.basename(self._k_mod.__file__))[0]
        # ... and the softmax arms it traced (softmax_arms_of): the executor exposes both.
        self.softmax_arms = softmax_arms_of(self._k_mod, has_lse=self.lse_desc is not None)
        # The kernel compile() keyword surface, read ONCE per plan (like
        # prepared host): the optional knobs below are passed only to a
        # kernel that carries them, so a kernel without the knob keeps its
        # legacy behaviour and never sees an unknown kwarg.  The THD compile
        # key (_explicit_compile_kwargs) reads the SAME set, so every key of this
        # plan agrees on `has_amax`.
        self._kernel_accepts = None  # a (re)loaded module: re-read its signature
        _kc = self._kernel_compile_accepts()
        if self.use_pdl and "use_pdl" not in _kc:
            # Rule 1: a plan declared with the programmatic-dependent-launch form never runs a kernel that silently lacks it.
            raise NotImplementedError(
                f"use_pdl=True: the {self.kernel_template} kernel carries no programmatic-dependent-launch sites (served by the cc 10.7 "
                "d256 decode tile and its split combine); declare use_pdl=False for this plan"
            )
        # Backstop of check_support's score-shape gate (rule 8b twin): reaching this means the shape
        # constant names a flavor whose file carries no store.
        self._not_implemented_error_if(
            (self.score_max_desc is not None or self.score_sum_exp_desc is not None) and "has_score_max" not in _kc,
            f"{self.kernel_template}: this kernel template carries no score_max / score_sum_exp store",
        )
        self._not_implemented_error_if(
            (self.bias_desc is not None or self.kv_mask_desc is not None) and "has_bias" not in _kc,
            f"{self.kernel_template}: this kernel template carries no additive-bias / kv_mask arm",
        )
        # Explicit pointer/int host entry (the f16/bf16 SM100 / SM107 prefill templates): every
        # extent and stride is a runtime argument, so the compile key is layout-only and the
        # dense / THD launches bind pointers through cudnn.sdpa.fwd.prepared.
        self._prepared_fp8 = self._can_prepare_fp8()
        self._prepared_mxfp8 = self._can_prepare_mxfp8()
        _explicit = bool(getattr(self._k_mod, "EXPLICIT_ABI", False)) or self._prepared_fp8 or self._prepared_mxfp8
        self._thd_spec = self._dense_spec = None
        # Backstop only: check_support already declined every flavor whose
        # kernel lacks the gate (rule 8b twin); reaching this means the twin and
        # the kernel disagree.
        if self.gate_desc is not None and not _explicit and "gate_stride" not in _kc:
            raise NotImplementedError(f"{self.flavor} kernel does not carry the epilogue gate")
        # Whether the amax atomicMax is compiled OUT (kernel carries the knob and
        # the caller asked): execute then binds None in the amax slot -- on the
        # dense AND the THD arm, whose compile key carries the same `has_amax`
        # (_explicit_compile_kwargs); a kernel without the knob keeps the legacy
        # dummy slot even at has_amax_o=False.
        self._amax_folded_out = ("has_amax" in _kc) and not self.has_amax_o
        if _explicit:
            # One artifact per layout kind: THD, dense and paged all compile HERE, at plan
            # time, and execute() only binds pointers. has_lse=False compiles the LSE store
            # out; a split requires the in-kernel LSE (the per-split LSE is the combine weight).
            compile_fn = self._k_mod.compile_prepared if (self._prepared_fp8 or self._prepared_mxfp8) else self._k_mod.compile
            if self.packed_thd_split:
                compile_fn = self._k_mod.compile_thd_split
            self._compiled_kernel = compile_fn(**self._explicit_compile_kwargs())
            self._build_prepared_specs()
        elif self.thd:
            raise NotImplementedError("THD forward requires a prepared pointer host")
        self._combine_kernel = None
        self._logger.debug("compile completed")

    def _can_prepare_dense_layout(self):
        operands = (self.q_desc,) + (() if self.paged else (self.k_desc, self.v_desc)) + (() if self.split_kv > 1 else (self.o_desc,))
        return self.thd or all(self._prepared_operand_layout(desc) is not None for desc in operands)

    def _can_prepare_fp8(self):
        if not (
            self._fp8
            and self._pertensor
            and self._device_cc[0] == 10
            and (self._o_dtype() in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2) or self._can_prepare_block_output())
            and (not self.o_block_scale or self._can_prepare_block_output())
        ):
            return False
        if self._device_cc == (10, 7) and self.split_kv > 1 and self.flavor not in _SM107_FP8_SPLIT_SHAPES and not self.packed_thd_split:
            return False
        return self.thd or all(
            self._prepared_operand_layout(desc) is not None
            for desc in (self.q_desc,)
            + (() if self.paged else (self.k_desc, self.v_desc))
            + (() if self.split_kv > 1 else (self.o_desc,))
            + (() if self.gate_desc is None else (self.gate_desc,))
        )

    def _can_prepare_mxfp8(self):
        if not (
            self._fp8
            and not self._pertensor
            and self._device_cc[0] == 10
            and (self._o_dtype() in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2) or self._can_prepare_block_output())
            and (not self.o_block_scale or self._can_prepare_block_output())
        ):
            return False
        if self._device_cc == (10, 7) and (self.split_kv > 1 or self.pack_gqa):
            return False
        return self.thd or all(
            self._prepared_operand_layout(desc) is not None
            for desc in (self.q_desc,)
            + (() if self.paged else (self.k_desc, self.v_desc))
            + (() if self.split_kv > 1 else (self.o_desc,))
            + (() if self.gate_desc is None else (self.gate_desc,))
        )

    def _prepared_quant_offset(self):
        if self.split_kv > 1:
            return self._split_workspace_bytes()
        if self.thd:
            return self._thd_workspace_layout()[1]
        return 0

    def _explicit_compile_kwargs(self) -> dict:
        """The compile key of an explicit-ABI template: only what specializes the
        traced code (every extent and stride is a runtime argument)."""
        if self.packed_thd_split:
            ps = self._paged_pool_stride(self.k_desc) if self.paged else None
            return dict(
                has_lse=self.lse_desc is not None,
                lse_kind="head" if self.thd_stats_head_major else "token",
                paged_hnd=ps is not None and ps[1] < ps[2],
                **(
                    {"paged_v_hnd": self.paged and self._paged_pool_stride(self.v_desc)[1] < self._paged_pool_stride(self.v_desc)[2]}
                    if self.flavor == SM100_NATIVE_D576_SHAPE
                    else {}
                ),
                **({"has_sink": True} if self.has_sink else {}),
                **({"thd_origin_specs": self._thd_origin_specs} if getattr(self, "_thd_origin_specs", None) is not None else {}),
            )
        if not self.thd or self.thd_decode_leg:
            # The ragged-Q decode leg's in-kernel LSE is the dense split-major
            # partial slab; the combine writes the ragged Stats rows.
            kind = "dense"
        elif self.thd_stats_padded:
            kind = "padded"
        elif self.thd_stats_head_major:
            kind = "head"
        else:
            kind = "token"
        km = self._k_mod
        compile_fn = km.compile_prepared if (getattr(self, "_prepared_fp8", False) or getattr(self, "_prepared_mxfp8", False)) else km.compile
        accepted = _parameter_names(compile_fn)
        kw = dict(has_lse=(self.lse_desc is not None) or self.split_kv > 1, lse_kind=kind)
        if getattr(self, "_thd_origin_specs", None) is not None:
            if "thd_origin_specs" not in accepted:
                raise NotImplementedError("this THD body does not consume independent physical origins")
            kw["thd_origin_specs"] = self._thd_origin_specs
        if "has_score_max" in accepted:
            # Score side outputs: each store is its own compile-time arm (None folds it out).
            kw["has_score_max"] = self.score_max_desc is not None
            kw["has_score_sum_exp"] = self.score_sum_exp_desc is not None
        if "has_bias" in accepted:
            # The bias / kv_mask operand slots of the host entry (the arm itself is the TemplateParams').
            kw["has_bias"] = self.bias_desc is not None
            kw["has_kv_mask"] = self.kv_mask_desc is not None
        if "static_lse_strides" in accepted and self.lse_desc is not None and not self.thd and self.split_kv == 1:
            # Plan metadata only. The compiled host retains a generic branch
            # for other legal runtime Stats strides. SM100 D512 half outputs
            # generate faster code with dynamic Stats addressing; FP8 outputs
            # benefit from the declared-layout specialization.
            dynamic_d512_half = self._device_cc != (10, 7) and self.flavor == (512, 512) and self._o_dtype() in (torch.float16, torch.bfloat16)
            if not dynamic_d512_half:
                kw["static_lse_strides"] = tuple(int(x) for x in self.lse_desc.stride)
        if "sfo_geometry" in accepted and self.o_block_scale:
            kw["sfo_geometry"] = self._sfo_geometry
        if "has_scale_o" in accepted:
            kw["has_scale_o"] = bool(self.o_block_scale and self.has_scale_o)
        if "has_amax" in accepted:
            kw["has_amax"] = self.has_amax_o
        if "scale_o_in_combine" in accepted:
            kw["scale_o_in_combine"] = self._split_scale_o()
        if "d_qk" in accepted:  # head-dim envelope
            kw.update(d_qk=self.head_dim_qk, d_v=self.head_dim_v)
        if "paged_hnd" in accepted and self.paged:
            ps = self._paged_pool_stride(self.k_desc)  # kernel order: page, row, head, d
            kw["paged_hnd"] = ps[1] < ps[2]
        if "paged_v_hnd" in accepted and self.paged:
            ps = self._paged_pool_stride(self.v_desc)
            kw["paged_v_hnd"] = ps[1] < ps[2]
        if "ragged_i64" in accepted and self.thd_decode_leg:
            # The ragged-Q leg's offset read width (int32 or int64 offsets).
            kw["ragged_i64"] = self.ragged_offsets_int64
        if "use_pdl" in accepted and self.use_pdl:
            # The programmatic-dependent-launch form (the launch attribute + the griddepcontrol pair); compile() declined a kernel without the knob.
            kw["use_pdl"] = True
        return kw

    def _build_prepared_specs(self) -> None:
        """The prepared launch of this plan (``cudnn.sdpa.fwd.prepared``): the THD spec for a ragged
        plan, the dense spec otherwise. The graph plan and ``execute()`` both bind through it; a host
        slot outside the launch's vocabulary declines the plan here, not at first launch."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import build_dense_spec, build_thd_spec

        self._thd_spec = self._dense_spec = None
        if self.thd and not self.thd_decode_leg:
            if self.split_kv == 1 or self.packed_thd_split:
                self._thd_spec = build_thd_spec(self, scale_softmax=None)
        else:
            # Dense plans and the ragged-Q decode leg (a dense split launch whose
            # Q rows sit at the ragged offsets and whose combine places the rows).
            self._dense_spec = build_dense_spec(self, scale_softmax=None)

    def _execute_dense_prepared(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        scale_softmax_log2,
        workspace,
        current_stream,
        block_table,
        block_table_v,
        gate=None,
        ragged=None,
        score_max=None,
        score_sum_exp=None,
        bias=None,
        kv_mask=None,
    ) -> None:
        """Dense (padded) launch through the prepared native binder from this call's
        torch tensors. Conversion plans use the shared staged entry instead; a split writes the
        partial slabs and recombines through the plan-time-compiled combine pass.
        ``ragged`` (the decode tile's ragged-Q
        leg): the (Q, O, Stats) ragged-offset tensors; Q / O / Stats then bind AS PASSED (packed, no
        repack) and the binder addresses their rows from the offsets."""
        spec = self._dense_spec
        device = q_tensor.device
        stream_int = int(current_stream) if current_stream is not None else torch.cuda.current_stream(device).cuda_stream
        _ensure_current_context(stream_int, device.index)  # every CUDA call below runs on the CALLER's thread, which reads its own context stack
        with _torch_stream_context(current_stream, device):  # legacy split scratch, launch and combine use the launch stream
            self._execute_dense_prepared_on_stream(
                spec,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                scale_softmax_log2,
                workspace,
                current_stream,
                stream_int,
                block_table,
                block_table_v,
                gate,
                ragged,
                score_max,
                score_sum_exp,
                bias,
                kv_mask,
            )

    def _execute_dense_prepared_on_stream(
        self,
        spec,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        scale_softmax_log2,
        workspace,
        current_stream,
        stream_int,
        block_table,
        block_table_v,
        gate,
        ragged=None,
        score_max=None,
        score_sum_exp=None,
        bias=None,
        kv_mask=None,
    ) -> None:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import bind_dense, bind_dense_split, execute_native_dense_tensors, facts_of_tensor

        workspace_ptr = 0
        if self.split_kv > 1:
            required = self.scratch_workspace_bytes()
            if workspace is None:
                raise ValueError(f"cudnn.sdpa: split prepared execution requires a {required}-byte workspace")
            ws = facts_of_tensor(workspace)
            if ws.device != (2, int(q_tensor.device.index or 0)) or not ws.contiguous:
                raise ValueError("cudnn.sdpa: split workspace must be contiguous and on the Q tensor's CUDA device")
            if ws.numel * workspace.element_size() < required:
                raise ValueError(f"cudnn.sdpa: split workspace requires {required} bytes")
            workspace_ptr = ws.ptr

        if spec.native is not None:
            # Standalone lengths require exactly the declared batch, whereas
            # graph binding allows a larger carrier for an effective batch.
            kv_lens = self._checked_seq_lens(seq_kv_lens, "seq_kv_lens") if seq_kv_lens is not None else None
            q_lens = self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None
            if spec.combine is not None and spec.combine.gate is not None:
                # The gate rides the combine (the decode tile's split): bound to the gated
                # combine entry for this launch, never handed to the native binder (which
                # knows no gate on a split) -- execute_native_dense_tensors does both.
                self._logger.debug("gate bound to the split combine (the decode tile has no gate seams)")
            launched = execute_native_dense_tensors(
                spec,
                (
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    sinks,
                    kv_lens,
                    q_lens,
                    block_table,
                    block_table_v,
                    gate,
                    *(ragged or (None, None, None)),
                ),
                current_stream,
                scale_softmax_log2,
                workspace_ptr,
            )
            if launched is False:
                self._logger.debug("execute skipped: ragged-Q leg with no addressable token / empty producer")
            else:
                self._logger.debug("execute completed")
            return

        # The plans the native binder cannot bind -- the score side outputs and the resident-block arms (bias / kv_mask),
        # whose host slots postdate its vocabulary (build_dense_spec leaves ``native`` None for exactly those) -- bind
        # through the python frame: the caller's BHSD facts, validated per call by ``prepared.bind_dense``.
        facts = dict(
            q=facts_of_tensor(q_tensor),
            k=facts_of_tensor(k_tensor),
            v=facts_of_tensor(v_tensor),
            o=facts_of_tensor(o_tensor),
            lse=facts_of_tensor(lse_tensor),
            sinks=facts_of_tensor(self._checked_sinks_1d(sinks) if sinks is not None else None),
            seq_kv_lens=facts_of_tensor(self._checked_seq_lens(seq_kv_lens, "seq_kv_lens") if seq_kv_lens is not None else None),
            seq_q_lens=facts_of_tensor(self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None),
            block_table=facts_of_tensor(block_table),
            block_table_v=facts_of_tensor(block_table_v),
            gate=facts_of_tensor(gate),
            # Score side outputs: a spec that requests them has no native binder (build_dense_spec), so
            # this python frame is the only path that binds them.
            score_max=facts_of_tensor(score_max),
            score_sum_exp=facts_of_tensor(score_sum_exp),
            # The resident-block arms (bias / kv_mask) likewise bind through this python frame only.
            bias=facts_of_tensor(bias),
            kv_mask=facts_of_tensor(kv_mask),
        )
        if ragged is not None:
            facts.update(ragged_q=facts_of_tensor(ragged[0]), ragged_o=facts_of_tensor(ragged[1]), ragged_lse=facts_of_tensor(ragged[2]))
        if self.split_kv > 1:
            bound = bind_dense_split(spec, facts, workspace_ptr, current_stream, stream_int)
            if bound is None:
                self._logger.debug("execute skipped: ragged-Q leg with no addressable token / empty producer")
                return
            frame, combine_args = bound
        else:
            frame = bind_dense(spec, facts, current_stream, stream_int)
        if scale_softmax_log2 != spec.template[spec.index["scale_softmax_log2"]]:
            frame[spec.index["scale_softmax_log2"]] = scale_softmax_log2
        if "inv_scale_softmax" in spec.index and scale_softmax_log2 != 0.0:
            # 1 / softmax_scale for the pre-scale bias add (the bias arm reads it; the slot exists on every host that
            # carries the arm): derived from this call's scale like scale_softmax_log2, so a per-call scale stays consistent.
            inv_scale = math.log2(math.e) / scale_softmax_log2
            if inv_scale != spec.template[spec.index["inv_scale_softmax"]]:
                frame[spec.index["inv_scale_softmax"]] = inv_scale
        spec.fn(*frame)
        if self.split_kv > 1:
            spec.combine.fn(*combine_args)
        self._logger.debug("execute completed")

    @staticmethod
    def _paged_pool_stride(desc) -> tuple:
        """Container strides ``[num_pages, H_kv, page_size, D]`` re-expressed in
        the kernel's ``[num_pages, page_size, H_kv, D]`` order (a permutation,
        so HND and NHD storage both bind as views)."""
        s = tuple(int(x) for x in desc.stride)
        return (s[0], s[2], s[1], s[3])

    def _paged_table_expected_stride(self, which_v: bool, n_pages: int) -> tuple:
        declared = self.paged_table_v_stride if which_v else self.paged_table_stride
        return declared if declared is not None else (n_pages, 1)

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch ``execute()`` carves from its ``workspace``.

        Fixed by the compiled geometry (call after ``check_support()``); 0 when
        the path allocates nothing per execute. This is the api-level share of
        a FROST executor's ``workspace_bytes`` (see
        ``engines.lower_dsl_prefill``). When ``execute()`` is called WITHOUT a
        workspace, legacy standalone paths allocate their scratch internally.
        Prepared FP8 requires this workspace for standalone calls too: it
        holds unused amax and identity-scale words as well as THD metadata.
        """
        self._ensure_support_checked()
        staged = getattr(self, "_staged_spec", None)
        if staged is not None:
            return staged.workspace_bytes
        b, qh = self.batch_size, self.h_q
        compiled = self._compiled_kernel is not None
        if not compiled and (
            (self._fp8 and not (self._can_prepare_fp8() or self._can_prepare_mxfp8())) or (not self._fp8 and not self._can_prepare_dense_layout())
        ):
            from .prepared_staged_forward import workspace_bytes

            return workspace_bytes(self)
        if (self._prepared_fp8 or self._prepared_mxfp8) if compiled else (self._can_prepare_fp8() or self._can_prepare_mxfp8()):
            return self._prepared_quant_offset() + ws_align(8)
        if self.thd and not self.thd_decode_leg:
            # [meta(seq_kv, cu_q, cu_k) | o_desc | sinks dummy]
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552).
            # o_desc: 16 int64 per sequence + the dead-unit pad slot + two
            # slots for the packed-total-clamped K/V runtime descriptors the
            # setup kernel writes (see the kernels' THD closures). Every THD
            # flavor carries those two now, not just FP8/MXFP8 (issue #624).
            return self._thd_workspace_layout()[1]
        if self._fp8 and self.split_kv == 1:
            return 0  # dense FP8/MXFP8: no per-execute scratch (dummies are cached one-time)
        if self.split_kv > 1:
            # Split-major partial slabs the main kernel writes and the combine
            # pass reduces: O_s [splits*B, S_q, H, d_v] in the PARTIAL dtype
            # (wider than O -- fp32 on SM100, half on SM120; the combine owns
            # the cast down) and lse_s [splits*B, H, S_q] fp32. Carved from the caller's
            # workspace — zero per-execute allocations (Hard Rule 1).
            return self._split_workspace_bytes()
        # Dense padded-Q lens bind directly as their own kernel parameter
        # (no combine buffer since the seq_len_q-as-parameter change) — no scratch.
        return 0

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        block_table_v: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        sf_o: Optional[torch.Tensor] = None,
        ragged_q: Optional[torch.Tensor] = None,
        ragged_o: Optional[torch.Tensor] = None,
        ragged_lse: Optional[torch.Tensor] = None,
        score_max_tensor: Optional[torch.Tensor] = None,
        score_sum_exp_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        kv_mask_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the compiled kernel.

        ``bias_tensor`` / ``kv_mask_tensor`` (APPENDED): the additive bias ``[1|B, 1|H_q, S_q, S_kv]`` and the per-key
        boolean mask ``[1|B, 1, 1, S_kv]`` of a specialization constructed with ``sample_bias`` / ``sample_kv_mask``,
        bound through (never copied), required iff the matching sample was given (the LSE's presence contract).

        ``score_max_tensor`` / ``score_sum_exp_tensor`` (APPENDED): the score side outputs of a
        specialization constructed with ``sample_score_max`` / ``sample_score_sum_exp`` -- fp32
        ``(B, H_q, S_q[, 1])`` buffers bound as the declared (B, H_q, S_q) view, written through (never
        copied), required iff the matching sample was given (the strict presence contract of the LSE).

        ``ragged_q`` / ``ragged_o`` / ``ragged_lse`` (the decode tile's ragged-Q
        leg only, ``thd_decode_leg``): the graph's (B+1,) int32 or int64 ragged-offset
        tensors of Q, O and Stats; the kernel reads them on device (Rule 3).

        ``gate``: the fused epilogue gate ``G`` of a specialization built with
        ``sample_gate`` -- logical BHSD, O's shape, bound as a zero-copy BSHD
        view (compact, or the declared strides check_support pinned; never a
        hidden copy). Required iff ``sample_gate`` was given.

        ``sf_o``: the block-scaled O scale-factor buffer (per-tensor FP8 with
        ``sample_sf_o``); its bytes are laid out per the declared geometry.

        ``workspace``: optional caller-provided scratch buffer (uint8, at
        least ``scratch_workspace_bytes()`` bytes). When given, every
        per-execute scratch buffer (the THD metadata / O-descriptor buffers)
        is carved from it — zero per-execute allocations. When None,
        non-split THD paths allocate those buffers as before.
        Dense and packed split plans require caller workspace; allocate
        ``scratch_workspace_bytes()`` bytes before calling ``execute()``.
        The public Torch wrapper allocates this scratch on the caller's behalf.
        Prepared D128 FP8-to-half requires caller workspace even without Stats;
        it also holds unused amax and identity-scale words.

        ``block_table`` / ``block_table_v``: paged KV only — ``(B, max_pages)``
        int32 device tensors (``block_table_v`` defaults to ``block_table``);
        ``k_tensor`` / ``v_tensor`` are then the page pools in their declared
        layout, bound as views.
        """
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm100 is not compiled")
        if self.paged:
            if block_table is None:
                raise ValueError("paged KV: block_table is required")
            block_table_v = block_table if block_table_v is None else block_table_v
            min_pages = -(-self.s_k_max // self.paged_page_size)
            for name, bt, is_v in (("block_table", block_table, False), ("block_table_v", block_table_v, True)):
                if bt.ndim != 2 or bt.shape[0] != self.batch_size or bt.shape[1] < min_pages or bt.dtype != torch.int32:
                    raise ValueError(
                        f"paged KV: {name} must be an int32 ({self.batch_size}, >= {min_pages}) tensor covering S_kv={self.s_k_max}; got {tuple(bt.shape)} {bt.dtype}"
                    )
                # Strides are compiled in (a view of the declared layout, never a
                # copy); size-1 axes are stride-agnostic.
                want = self._paged_table_expected_stride(is_v, bt.shape[1])
                got = tuple(bt.stride())
                if any(g != w for g, w, n in zip(got, want, bt.shape) if n != 1):
                    raise ValueError(f"paged KV: {name} strides {got} do not match the declared table strides {want}")
            # The kernel compiles both tables on ONE dynamic page-axis extent and
            # reads its KV maximum from block_table: decline a mismatch here, by
            # name, rather than let the compiled callable's argument check raise
            # (the graph path declines it in graph_analyzer).
            if block_table_v.shape[1] != block_table.shape[1]:
                raise ValueError(
                    f"paged KV: block_table and block_table_v must have the same page-axis extent; got {block_table.shape[1]} vs {block_table_v.shape[1]}"
                )
            for name, t, d in (("k_tensor", k_tensor, self.k_desc), ("v_tensor", v_tensor, self.v_desc)):
                if tuple(t.shape) != tuple(d.shape) or tuple(t.stride()) != tuple(d.stride):
                    raise ValueError(
                        f"paged KV: {name} must match the declared pool {tuple(d.shape)} / {tuple(d.stride)}; got {tuple(t.shape)} / {tuple(t.stride())}"
                    )
        elif block_table is not None or block_table_v is not None:
            raise ValueError("block_table given but the adapter was not built for paged KV")
        # Run on the caller's stream (ExecutionContext.stream, resolved from the
        # execute-time handle); None -> the default stream. Threaded to every
        # kernel launch below (dense / fp8 / mxfp8 / THD).
        explicit_stream = current_stream is not None
        if getattr(self, "_staged_spec", None) is None:
            current_stream = self._get_default_stream(current_stream)

        self._value_error_if(
            self.softmax_scale_prefolded and scale_softmax is not None,
            "softmax_scale_prefolded: an execute-time scale_softmax is not applied -- Q already carries attn_scale * log2(e)",
        )
        scale_val = self.scale_softmax if scale_softmax is None else float(scale_softmax)
        self._value_error_if(scale_val == 0, _ZERO_SCALE_UNSUPPORTED)
        self._value_error_if((scale_val < 0) != self._score_negated, "attn_scale sign must match the compiled plan's (#1435)")
        scale_val = abs(scale_val)
        scale_softmax_log2 = scale_val * math.log2(math.e)

        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        if self.seq_q_lens_present:
            # Dense Q lengths are passed as a raw address, so validate the
            # storage before bypassing the compiled launcher's tensor binding.
            self._value_error_if(
                seq_q_lens.device.type != "cuda" or seq_q_lens.device.index != q_tensor.device.index,
                f"seq_q_lens must be a CUDA tensor on the same device as q_tensor ({q_tensor.device}); got {seq_q_lens.device}",
            )
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        # Strict presence contract, both directions: every SM100 kernel
        # (f16/bf16, FP8, MXFP8) is compiled with has_lse keyed on sample_lse
        # (no Stats output -> the LSE store is compiled out and there is no
        # LSE slot to bind), and a THD lse_tensor is bound in its DECLARED
        # packed layout (recorded at check_support) — so an lse_tensor without
        # a sample_lse cannot be honored and is rejected rather than silently
        # dropped.
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        self._value_error_if(
            self.has_amax_o and self.amax_o_desc is not None and amax_o is None,
            "amax_o is required by this compiled specialization",
        )
        self._value_error_if(
            self.amax_o_desc is None and self.pv_bf16 and amax_o is not None,
            "this hybrid specialization was compiled without Amax_O; construct the API with sample_amax_o",
        )
        self._value_error_if(self.pv_bf16 and sf_v is not None, "this PV-BF16 specialization does not consume sf_v")
        # The same strict presence contract for the epilogue gate: the gated
        # module reads a gate tile every tile (no gate -> stale SMEM), and the
        # ungated module has no gate slot at all.
        self._value_error_if(
            self.gate_desc is not None and gate is None,
            "gate is required by this compiled specialization (constructed with sample_gate)",
        )
        self._value_error_if(
            self.gate_desc is None and gate is not None,
            "this specialization was compiled without an epilogue gate; construct the API with sample_gate",
        )
        self._value_error_if(
            not self.has_amax_o and amax_o is not None,
            "this specialization was compiled with has_amax_o=False; it produces no Amax_O",
        )
        self._value_error_if(
            self.o_block_scale > 0 and sf_o is None,
            "sf_o is required by this compiled specialization (block-scaled O)",
        )
        self._value_error_if(
            self.o_block_scale == 0 and sf_o is not None,
            "this specialization was compiled without a block-scaled O; construct the API with sample_sf_o",
        )
        if self.thd:
            pass  # bound in _execute_thd (declared packed layout)
        elif lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)
        # Score side outputs: the LSE's strict presence contract in both directions (a store compiled out
        # has no slot to bind; a store compiled in reads a stale slot without its buffer), then the
        # declared (B, H_q, S_q) view over the caller's storage.
        for _name, _desc, _t in (("score_max", self.score_max_desc, score_max_tensor), ("score_sum_exp", self.score_sum_exp_desc, score_sum_exp_tensor)):
            self._value_error_if(
                _desc is not None and _t is None, f"{_name}_tensor is required by this compiled specialization (constructed with sample_{_name})"
            )
            self._value_error_if(
                _desc is None and _t is not None, f"this specialization was compiled without a {_name} output; construct the API with sample_{_name}"
            )
        if score_max_tensor is not None:
            score_max_tensor = self._checked_row_stat_view(score_max_tensor, "score_max_tensor", self._score_strides.get("score_max"))
        if score_sum_exp_tensor is not None:
            score_sum_exp_tensor = self._checked_row_stat_view(score_sum_exp_tensor, "score_sum_exp_tensor", self._score_strides.get("score_sum_exp"))
        for _name, _desc, _t in (("bias", getattr(self, "bias_desc", None), bias_tensor), ("kv_mask", getattr(self, "kv_mask_desc", None), kv_mask_tensor)):
            self._value_error_if(
                _desc is not None and _t is None, f"{_name}_tensor is required by this compiled specialization (constructed with sample_{_name})"
            )
            self._value_error_if(_desc is None and _t is not None, f"this specialization was compiled without a {_name}; construct the API with sample_{_name}")

        if getattr(self, "_staged_spec", None) is not None:
            from .prepared_staged_forward import execute as execute_staged

            self._value_error_if(self._pertensor and any(t is not None for t in (sf_q, sf_k, sf_v)), "per-tensor FP8 does not accept MXFP8 input scales")
            scales = dict(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v) if self._pertensor else dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v)
            tensors = dict(
                q=q_tensor,
                k=k_tensor,
                v=v_tensor,
                o=o_tensor,
                lse=lse_tensor,
                sinks=sinks,
                seq_q_lens=seq_q_lens,
                seq_kv_lens=seq_kv_lens,
                scale_o=scale_o,
                amax_o=amax_o,
                sf_o=sf_o,
                gate=gate,
                block_table=block_table,
                block_table_v=block_table_v,
                **scales,
            )
            execute_staged(self, tensors, workspace, current_stream if explicit_stream else None, scale_val)
            return
        if getattr(self, "_prepared_fp8", False):
            self._execute_fp8_prepared(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_q_lens,
                seq_kv_lens,
                dict(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o, amax_o=amax_o, sf_o=sf_o, gate=gate),
                scale_val,
                workspace,
                current_stream,
                block_table=block_table,
                block_table_v=block_table_v,
            )
            return
        if getattr(self, "_prepared_mxfp8", False):
            self._execute_fp8_prepared(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_q_lens,
                seq_kv_lens,
                dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v, amax_o=amax_o, sf_o=sf_o, scale_o=scale_o, gate=gate),
                scale_val,
                workspace,
                current_stream,
                block_table=block_table,
                block_table_v=block_table_v,
            )
            return
        if self.thd and not self.thd_decode_leg:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
                block_table=block_table,
                block_table_v=block_table_v,
                gate=gate,
            )
            return

        ragged = None
        if self.thd_decode_leg:
            self._value_error_if(
                ragged_q is None or ragged_o is None or (self.lse_desc is not None and ragged_lse is None),
                "the decode tile's ragged-Q leg needs the Q, O (and Stats, when declared) ragged-offset tensors (ragged_q= / ragged_o= / ragged_lse=)",
            )
            ragged = (ragged_q, ragged_o, ragged_lse if self.lse_desc is not None else None)
        else:
            self._value_error_if(
                ragged_q is not None or ragged_o is not None or ragged_lse is not None,
                "ragged offsets are read only by the decode tile's ragged-Q leg (thd_decode_leg); this specialization does not take them",
            )

        self._execute_dense_prepared(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            sinks,
            seq_kv_lens,
            seq_q_lens,
            scale_softmax_log2,
            workspace,
            current_stream,
            block_table,
            block_table_v,
            gate,
            ragged=ragged,
            score_max=score_max_tensor,
            score_sum_exp=score_sum_exp_tensor,
            bias=bias_tensor,
            kv_mask=kv_mask_tensor,
        )

    def _kernel_compile_accepts(self) -> frozenset:
        """The loaded kernel module's ``compile()`` parameter names, memoized per plan.

        Every optional compile knob (``gate_stride``, ``has_amax``, the declared
        q/k/v/o strides) is passed only to a kernel that carries it -- and by
        EVERY compile key of the plan (the dense key in ``compile()`` and the
        THD key in ``_explicit_compile_kwargs``), or execute binds an argument the
        artifact was not specialised for.  ``compile()`` resets the memo when it
        (re)loads the module."""
        kc = getattr(self, "_kernel_accepts", None)
        if kc is None:
            entry = self._k_mod.compile_prepared if self._fp8 else self._k_mod.compile
            kc = _parameter_names(entry)
            self._kernel_accepts = kc
        return kc

    def _thd_unit_envelope(self) -> int:
        """PLAN-TIME upper bound on live THD units:
        ``B * ceil(S_q_declared / CGA_TILE_M) * QH``.

        Every sequence's length is bounded by the declared S_q (the padding
        contract), so this covers ``Σ_b ceil(s_b / tile) * QH``. Units past
        the live total are DEAD by kernel contract (the decode's
        ``batch == n_batch`` sentinel): no loads, no O/LSE writes, one
        empty-mainloop barrier dance each. The grid is host-known at PLAN
        time — execute reads nothing from the lengths — which removes the
        last THD D2H sync and unblocks CUDA-graph capture (issue #552). The
        dead-tile tax mirrors the C++ backend's THD grid strategy; callers
        declaring S_q far above their live totals pay it. The over-launch
        test pads this to pin the dead-unit contract."""
        cga_tile_m = int(self._k_mod.CGA_TILE_M)
        s_q_decl = int(self.q_desc.shape[2])
        return self.batch_size * ((s_q_decl + cga_tile_m - 1) // cga_tile_m) * self.h_q

    def _thd_meta_words(self):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_WORK_META_WORDS

        # Workspace queries are legal before compile(). Reserve the work-prefix
        # tail ([4B+4, 5B+5): the remap-ordered unit prefix the f16 d128-family
        # prefill / decode kernels and the packed split decode through) for
        # every THD plan; kernels that still walk the sequences leave these B+1
        # words unused. No compiled module is needed.
        return THD_WORK_META_WORDS(self.batch_size)

    def _thd_workspace_layout(self):
        b, qh = self.batch_size, self.h_q
        map_bytes = (b + 3) * 16 * 8
        if getattr(self, "_thd_origin_specs", None) is not None:
            from .thd_origins import descriptor_bytes

            map_bytes = descriptor_bytes(b)
        base = ws_align(self._thd_meta_words() * 4) + ws_align(map_bytes) + (0 if self.has_sink else ws_align(qh * 4))
        if self.packed_thd_split:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import thd_split_workspace

            capacity = b * self.s_q_max
            if self.max_total_seq_len_q is not None:
                capacity = min(capacity, self.max_total_seq_len_q)
            return thd_split_workspace(base, self.split_kv, capacity, qh, self.head_dim_v)
        return None, base

    def _thd_plan(self):
        """The THD launch's per-plan constants — the operands' declared strides
        and row spans, the scratch layout, the unit count, the length form —
        resolved once (``prepared.build_thd_spec`` reads them)."""
        plan = getattr(self, "_thd_plan_cached", None)
        if plan is not None:
            return plan
        b = self.batch_size
        # Persistent THD kernels pull work from a device-bounded counter. A body
        # may assign a small envelope directly across its declared initial waves;
        # larger envelopes use one resident wave. The packed family below also
        # tightens the safety envelope to its known token-tile count.
        wave_policy = None
        wave_geometry = None
        env = units = self._thd_unit_envelope()
        if self.packed_thd_split:
            env = units = env * self.split_kv
        if getattr(self._k_mod, "THD_PERSISTENT", False):
            cluster_ctas = int(getattr(self._k_mod, "CGA_SIZE", 0) or getattr(self._k_mod, "CTA_MMA", 1))
            resident = max(1, _device_sm_count(self.q_desc.device) // max(1, cluster_ctas))
            initial_waves = max(1, int(getattr(self._k_mod, "THD_INITIAL_WAVES", 1)))
            initial_limit = initial_waves * resident
            cfg = self._k_mod.CFG
            ratio = getattr(self._k_mod, "THD_INITIAL_WORK_RATIO", None)
            geometry = None
            if ratio is not None:
                numerator, denominator = ratio
                if denominator <= 0 or numerator < denominator or numerator > initial_waves * denominator:
                    raise ValueError("invalid THD initial-work ratio")
                initial_limit = resident * numerator // denominator
                pack_g = int(cfg.PACK_G) if cfg.PACK_GQA else 1
                tile = int(self._k_mod.CGA_TILE_M)
                if pack_g <= 0 or tile % pack_g or self.h_q % pack_g:
                    raise ValueError("invalid THD packed initial-wave geometry")
                geometry = (tile // pack_g, self.h_q // pack_g, initial_limit)
            units = env if env <= initial_limit else resident
            wave_envelope = env
            if (
                ((self._device_cc == (10, 0) and self.paged) or (self._device_cc == (10, 7) and not self.paged))
                and self.dtype == torch.bfloat16
                and self.head_dim_qk == self.head_dim_v == 128
                and self.batch_size == 1
                and self.is_causal
                and self.window_left is None
                and self.window_right == 0
                and not self.has_sink
                and self.gate_desc is None
                and cfg.CTA_MMA == 2
                and cfg.SPLIT_KV == 1
                and cfg.SCHEDULER_POLICY == SCHED_LPT
                and cfg.PACK_G in (4, 8)
                and self.h_q == self.h_kv * cfg.PACK_G
            ):
                # Count packed TOKEN tiles, not the looser safety envelope above:
                # for H32/GQA4/Q1025 those bounds are 72 and 96, respectively.
                # With B=1 the declared Q also bounds the single live sequence;
                # a ragged batch's maximum would overestimate its actual work.
                tile = int(self._k_mod.CGA_TILE_M)
                packed_units = ((int(self.q_desc.shape[2]) * cfg.PACK_G + tile - 1) // tile) * (self.h_q // cfg.PACK_G)
                if resident < packed_units <= (initial_limit if geometry is not None else 2 * resident):
                    units = min(env, packed_units)
                    wave_envelope = units
            dbg = int(os.environ.get("FROST_THD_CLUSTERS", "0"))  # debug override
            if dbg > 0:
                units = min(env, dbg)
            elif initial_waves > 1:
                # Re-evaluate the wave choice against each invocation's host-known
                # capacity. Broad graph declarations must not freeze a one-wave
                # cap when current storage fits the body's initial waves.
                wave_policy = (wave_envelope, resident, initial_waves)
                wave_geometry = geometry
        n_meta = self._thd_meta_words()
        plan = SimpleNamespace(
            q=self._thd_decl(self.q_desc),
            k=self._thd_decl(self.k_desc),
            v=self._thd_decl(self.v_desc),
            o=self._thd_decl(self.o_desc),
            units=units,
            wave_policy=wave_policy,
            wave_geometry=wave_geometry,
            cga_tile_m=int(self._k_mod.CGA_TILE_M),
            n_q_lens=b + 1 if self.cu_seq_q_lens else b,
            n_kv_lens=b + 1 if self.cu_seq_kv_lens else b,
            lens_form=(1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0),
            # scratch layout: [meta(seq_kv, cu_q, cu_k) | o_desc | ...] (see scratch_workspace_bytes)
            n_meta=n_meta,
            n_o_desc=(b + 3) * 16,
            off_o_desc=ws_align(n_meta * 4),
            scratch_bytes=self.scratch_workspace_bytes(),
            split_workspace=self._thd_workspace_layout()[0],
            total_q=None if self.max_total_seq_len_q is None else max(int(self.max_total_seq_len_q), 0),
            total_kv=None if self.max_total_seq_len_kv is None else max(int(self.max_total_seq_len_kv), 0),
        )
        self._thd_plan_cached = plan
        return plan

    def _execute_thd(
        self,
        q_buf,
        k_buf,
        v_buf,
        o_buf,
        scale_softmax_log2,
        sinks,
        seq_len_kv,
        seq_q_lens,
        lse_tensor=None,
        workspace=None,
        current_stream=None,
        block_table=None,
        block_table_v=None,
        gate=None,
    ):
        """THD / varlen execute (f16 kernels): shared packing + launch.

        ``lse_tensor``, when given, is the caller's ragged Stats buffer,
        written by the kernel directly in its declared layout (token-major
        packed ``(T, H)`` or head-major ``(H, head_stride)``); when ``None``
        the kernel compiles the LSE store out (has_lse=False) and no scratch
        exists. Host round-trips (issue #552): NONE — the lengths never reach
        the host (the setup kernel builds the metadata buffer device-side),
        every ragged view binds its buffer's capacity, and the launch grid is
        the plan-time envelope (dead units exit by kernel contract) — the
        execute is fully async and CUDA-graph capturable. No compile is keyed
        on runtime data: the kernels compile with DYNAMIC token extents, so a
        new packed total re-binds the same artifact.  ``gate`` (appended): the
        epilogue gate G of a gated plan -- a packed ``(T, H_q, D_v)`` buffer (or the
        graph's ``(B, H_q, S, D_v)`` view) the binder validates like Q / O."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import execute_native_thd_tensors

        spec = self._thd_spec
        if spec is None:
            raise NotImplementedError("SdpaFwdDslSm100 (THD): this plan has no prepared launch (see _build_thd_spec)")
        if spec.split_workspace is not None and workspace is None:
            raise ValueError("prepared packed split requires caller-owned workspace")
        stream_int = int(current_stream) if current_stream is not None else torch.cuda.current_stream(q_buf.device).cuda_stream
        _ensure_current_context(stream_int, q_buf.device.index)  # Stats initialization is a driver call on the CALLER's thread.
        if workspace is not None:
            ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm100 (THD)", spec.scratch_bytes)
        else:
            # standalone use without a workspace: GPU-written scratch is per INVOCATION (never shared
            # through the spec), allocated on the launch stream
            with _torch_stream_context(current_stream, q_buf.device):
                scratch = torch.empty(spec.scratch_bytes, dtype=torch.uint8, device=q_buf.device)
            ws_ptr = scratch.data_ptr()
        launched = execute_native_thd_tensors(
            spec,
            (q_buf, k_buf, v_buf, o_buf, seq_q_lens, seq_len_kv, lse_tensor, sinks, block_table, block_table_v),
            ws_ptr,
            current_stream,
            scale_softmax_log2,
            gate=gate,
        )
        if not launched:
            self._logger.debug("execute (THD): no addressable Q token, nothing to do")
            return
        self._logger.debug("execute (THD) completed")
        return

    @staticmethod
    def _ceil_div(x: int, a: int) -> int:
        return (x + a - 1) // a


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TensorSignature:
    """Tensor metadata that changes support or compilation."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _SdpaFwdCacheKey:
    """Architecture-tagged cache key shared by direct FROST SDPA wrappers."""

    api_type: type[SdpaFwdDsl]
    q: _TensorSignature
    k: _TensorSignature
    v: _TensorSignature
    o: _TensorSignature
    lse: Optional[_TensorSignature]
    is_causal: bool
    causal_bottom_right: bool
    window_size_left: Optional[int]
    window_size_right: Optional[int]
    scale_softmax: Optional[float]
    seq_q_lens_present: bool
    seq_kv_lens_present: bool
    has_sink: bool
    thd: bool
    sched_policy: Optional[int]
    tile_m: Optional[int]
    tile_n: Optional[int]
    cga: Optional[int]
    softmax_precision: Optional[int] = None
    softmax_scale_prefolded: bool = False


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return _TensorSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _optional_tensor_signature(tensor: Optional[torch.Tensor]) -> Optional[_TensorSignature]:
    return None if tensor is None else _tensor_signature(tensor)


def _make_cache_key(
    api_type: type[SdpaFwdDsl],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    lse: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens_present: bool = False,
    seq_kv_lens_present: bool = False,
    has_sink: bool = False,
    thd: bool = False,
    sched_policy: Optional[int] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cga: Optional[int] = None,
    softmax_precision: Optional[int] = None,
    softmax_scale_prefolded: bool = False,
) -> _SdpaFwdCacheKey:
    return _SdpaFwdCacheKey(
        api_type=api_type,
        q=_tensor_signature(q),
        k=_tensor_signature(k),
        v=_tensor_signature(v),
        o=_tensor_signature(o),
        lse=_optional_tensor_signature(lse),
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        scale_softmax=scale_softmax,
        has_sink=has_sink,
        seq_q_lens_present=seq_q_lens_present,
        seq_kv_lens_present=seq_kv_lens_present,
        thd=thd,
        sched_policy=sched_policy,
        tile_m=tile_m,
        tile_n=tile_n,
        cga=cga,
        softmax_precision=softmax_precision,
        softmax_scale_prefolded=softmax_scale_prefolded,
    )


_cache_of_objects: dict[_SdpaFwdCacheKey, SdpaFwdDsl] = {}


def _get_or_create_api(
    cache_key: _SdpaFwdCacheKey,
    **api_kwargs,
) -> SdpaFwdDsl:
    api = _cache_of_objects.get(cache_key)
    if api is None:
        _logger.debug("Building new %s", cache_key.api_type.__name__)
        api = cache_key.api_type(**api_kwargs)
        api.check_support()
        api.compile()
        _cache_of_objects[cache_key] = api
    return api


def _allocate_lse_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim != 4:
        raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
    b, h, s_q, _ = q_tensor.shape
    return torch.empty((b, h, s_q), dtype=torch.float32, device=q_tensor.device)


def sdpa_fwd_wrapper_dsl_sm100(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size_left: Optional[int] = None,
    causal_bottom_right: bool = False,
    scale_softmax: Optional[float] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM100 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""
    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm100: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    cache_key = _make_cache_key(
        SdpaFwdDslSm100,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=False,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        has_sink=sinks is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
    )

    required = sdpa_fwd.scratch_workspace_bytes()
    workspace = torch.empty(required, dtype=torch.uint8, device=q_tensor.device) if required else None

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        scale_softmax=scale_softmax,
        sinks=sinks,
        seq_kv_lens=seq_kv_lens,
        current_stream=current_stream,
        workspace=workspace,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# ---------------------------------------------------------------------------
# SM90: adapter over the Hopper D512 prefill kernel template
# ---------------------------------------------------------------------------


class SdpaFwdDslSm90(SdpaFwdDsl):
    """Compile and execute native dense or packed SM90 D512 forward attention.

    Q, K, V and O keep their declared strides: the kernel binds native TMA
    views over caller storage, with no BSHD gather. Dtype, shape, band,
    lengths, sink, Stats and the sign of ``scale_softmax`` are compile-time
    specializations; an explicit zero scale stays zero. Contradictory
    declarations raise ValueError, unserved ones NotImplementedError.
    Non-overlapping operands are caller contract, as on the sibling adapters.
    The compiled launcher refuses dtype, device, layout and alignment
    mismatches, a THD workspace off the 128-byte tensor-map boundary included.
    """

    def _initialize_implementation(self) -> None:
        """Initialize prepared-launch and packed-Stats plan metadata."""
        self._dense_spec = self._thd_spec = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0

    def check_support(self) -> bool:
        """Validate the Hopper tile geometry and native operand layouts."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import buffers
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok, thd_stats_packing

        # Rule 7: reject a too-old DSL before compile() loads the SM90 template.
        too_old = buffers.cutedsl_requirement_error("SM90 SDPA")
        if too_old is not None:
            raise ImportError(too_old)
        self._logger.debug("Entering check_support")
        self._not_implemented_error_if(buffers.current_sm() != 90, "SM90 D512 SDPA requires a Hopper SM90 device")
        self._device_cc = (9, 0)

        # Contradictory declarations first.
        descs = (self.q_desc, self.k_desc, self.v_desc, self.o_desc)
        device = buffers.current_device_id()
        for desc in (*descs, self.lse_desc):
            if desc is not None:
                self._value_error_if(
                    desc.device.type != "cuda" or desc.device.index not in (None, device),
                    f"SM90 SDPA: {desc.name} must name CUDA storage on the plan's device {device}",
                )
        for desc in descs:
            self._value_error_if(desc.ndim != 4, f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}")
        b, h_q, s_q, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        d_v = self.v_desc.shape[3]
        for label, value in (("B", b), ("H_q", h_q), ("H_kv", h_kv), ("S_q", s_q), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(value <= 0, f"{label} must be > 0; got {value}")
        self._value_error_if(h_q % h_kv != 0, f"H_q ({h_q}) must be divisible by H_kv ({h_kv}) for GQA / MQA")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_v), name="O")
        self._check_tensor_shape(self.lse_desc, (b, h_q, s_q), name="Stats")
        for desc in descs[1:]:
            self._check_dtype(desc, self.q_desc.dtype, name=desc.name, extra_error_msg=f"{desc.name} must match Q")
        self._value_error_if(self.thd_stats_padded and not self.thd, "SM90 SDPA: thd_stats_padded requires thd=True")
        self._value_error_if(self.thd and self.seq_q_lens_present, "SM90 SDPA: seq_q_lens_present is dense-only; THD always requires Q lengths")
        if self.thd:
            self.seq_kv_lens_present = True
        self._value_error_if(self.seq_q_lens_present and not self.seq_kv_lens_present, "SM90 SDPA: dense Q and KV lengths must be provided together")
        for name, value in (("window_size_left", self.window_size_left), ("window_size_right", self.window_size_right)):
            self._value_error_if(value is not None and value < 0, f"SM90 SDPA: {name} must be >= 0; got {value}")
        self._value_error_if(self.window_size_right is not None and not self.is_causal, "SM90 SDPA: window_size_right requires is_causal=True")
        self._value_error_if(self.causal_bottom_right and not self.is_causal, "SM90 SDPA: causal_bottom_right requires is_causal=True")

        # Well-formed declarations this engine does not serve.
        refusals = (
            (self.paged, "paged KV"),
            (self.gate_desc is not None, "epilogue gate fusion (served by the SM107 d256 SDPA engines only)"),
            (self._pertensor or self.pv_bf16, "FP8 or PV BF16 inputs"),
            # has_amax_o is a quantized-path flag; inert here, so only the declared output is refused.
            (self.amax_o_desc is not None, "an Amax_O output"),
            (self.dtype_o is not None and self.dtype_o != self.q_desc.dtype, "an output dtype different from Q"),
            (
                self.softmax_precision is not None and getattr(self.softmax_precision, "name", str(self.softmax_precision)).lower() not in ("float", "float32"),
                "non-FP32 softmax precision",
            ),
            (self.softmax_scale_prefolded, "pre-folded softmax scale (the cc10.7 kernels only)"),
            (self.split_kv != 1, "split-KV"),
            (self.tile_m not in (None, _SM90_TILE_M) or self.tile_n not in (None, _SM90_TILE_N), "tiles other than 64/64"),
            (self.cga not in (None, 1), "CGA other than 1"),
            (self.q_desc.dtype not in (torch.float16, torch.bfloat16), "Q/K/V/O other than FP16/BF16"),
            (self.lse_desc is not None and self.lse_desc.dtype != torch.float32, "Stats other than FP32"),
            (any(value >= 2**31 for value in (b, h_q, h_kv, s_q, s_kv)), "graph dimensions outside positive Int32"),
        )
        for refused, feature in refusals:
            self._not_implemented_error_if(refused, f"SM90 SDPA does not support {feature}")
        reason = _sm90_head_dims_mismatch(d_qk, d_v)
        self._not_implemented_error_if(reason is not None, reason)
        self._not_implemented_error_if((self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd, "SM90 SDPA: cumulative lengths are THD-only")
        self._not_implemented_error_if(self.thd and self.thd_stats_padded, "SM90 SDPA: THD Stats must stay packed (SDPA Rule S1)")

        # Native TMA binding (Rule 2): the declared strides are the kernel's, with no BSHD gather.
        if self.thd:
            self._thd_check_strides_native()
            strides = tuple((0, *desc.stride[1:]) for desc in descs)
        else:
            quantum = 16 // self.q_desc.dtype.itemsize
            strides = []
            for desc in descs:
                self._not_implemented_error_if(
                    not dense_layout_ok(desc.shape, desc.stride),
                    f"{desc.name}: dense strides {desc.stride} are outside dense_flex (D innermost-contiguous, non-broadcast, each covering the full pitch below it)",
                )
                # dense_layout_ok wildcards a unit axis, whose stride the kernel's maps still read: pin it to D.
                pinned = tuple(desc.shape[3] if size == 1 else stride for size, stride in zip(desc.shape[:3], desc.stride))
                self._not_implemented_error_if(
                    any(stride % quantum for stride in pinned), f"{desc.name}: TMA strides must be 16-byte multiples; got {desc.stride}"
                )
                strides.append((*pinned, 1))
            strides = tuple(strides)
        lse_stride = None
        if self.lse_desc is not None:
            hs, ss = self.lse_desc.stride[1:]
            if self.thd:
                # Rule S1: the shared packing classifier; a head stride covering the packed total is caller contract.
                self._not_implemented_error_if(
                    thd_stats_packing(hs, ss, h_q) is None,
                    f"Stats: packed Stats must be token-major or head-major (SDPA Rule S1); got {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = thd_stats_packing(hs, ss, h_q) == "head_major"
                self.thd_stats_head_stride = hs if self.thd_stats_head_major else 0
                lse_stride = (0, hs, ss)
            else:
                lse_stride = tuple(stride if size > 1 else 0 for stride, size in zip(self.lse_desc.stride, (b, h_q, s_q)))
                self._not_implemented_error_if(
                    not dense_layout_ok((b, h_q, s_q, 1), (*lse_stride, 1)),
                    f"Stats: dense Stats strides {self.lse_desc.stride} are outside dense_flex (non-broadcast, non-overlapping, each covering the full pitch below it)",
                )

        # The base's canonical band; a bound at or beyond S_q + S_kv masks nothing.
        unpadded = not (self.thd or self.seq_q_lens_present or self.seq_kv_lens_present)
        envelope = s_q + s_kv
        left = None if self.window_left is None or self.window_left >= envelope else int(self.window_left)
        right = None if self.window_right is None or self.window_right >= envelope else int(self.window_right)
        if left is None and right is None:
            band = dict(causal=False)
        else:
            self._not_implemented_error_if(
                envelope + 2 * _SM90_TILE_M >= _SM90_BAND_COORDINATE_LIMIT,
                "SM90 SDPA: a diagonal band needs S_q + S_kv below 2**30 (Int32 band coordinates)",
            )
            causal = right == 0
            bottom_right = self.causal_bottom_right or (unpadded and s_q == s_kv)
            band = dict(causal=causal, window_left=left, window_right=right or None, bottom_right=None if bottom_right == causal else bottom_right)
        # Pending cases inspect the declared left bound, vacuous bounds included.
        pending = (
            (self.causal_bottom_right and unpadded and s_q > s_kv, "unpadded bottom-right S_q > S_kv"),
            (self.window_size_left is not None and unpadded and s_q > s_kv, "unpadded sliding-window S_q > S_kv"),
            (self.pack_gqa and self.thd, "THD PackGQA"),
        )
        for refused, feature in pending:
            self._not_implemented_error_if(refused, f"{feature} is not yet supported on SM90")

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)
        # The kernel specializes on the scale's sign; a literal zero stays zero.
        scale = self.scale_softmax
        scale_mode = _SM90_SCALE_ZERO if scale == 0 else _SM90_SCALE_NEGATIVE if scale < 0 else _SM90_SCALE_POSITIVE
        sched = self.sched_policy
        if sched is None:
            # The shared heuristic's first candidate and the sibling adapters' rule.
            sched = SCHED_NATURAL if self.thd or not self.is_causal else _causal_sched_policy(s_kv, d_qk, d_v, elem_bytes=2)
        self._not_implemented_error_if(
            sched not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2), "SM90 SDPA supports only NATURAL/LPT/LPT_L2 single-tile scheduling"
        )
        self._not_implemented_error_if(
            self.thd and sched != SCHED_NATURAL,
            "SM90 SDPA: THD walks its live units with the natural single-tile work decoder; SCHED_LPT and SCHED_LPT_L2 are dense-only",
        )
        self._not_implemented_error_if(
            self.pack_gqa and not pack_gqa_supported(h_q, h_kv, _SM90_TILE_M), "SM90 D512 PackGQA requires a head ratio dividing tile_m=64"
        )

        self.params = Sm90TemplateParams(
            dtype_qkv=DTYPE_BF16 if self.q_desc.dtype == torch.bfloat16 else DTYPE_FP16,
            **band,
            thd_varlen=self.thd,
            has_lse=self.lse_desc is not None,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2,
            pack_gqa=self.pack_gqa,
            qh_per_kh=h_q // h_kv,
            sched_policy=sched,
            scale_mode=scale_mode,
            seq_q_lens_present=self.seq_q_lens_present,
            seq_kv_lens_present=self.seq_kv_lens_present,
        )
        self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max = b, h_q, h_kv, s_q, s_kv
        self.head_dim_qk, self.head_dim_v = d_qk, d_v
        self.dtype = self.q_desc.dtype
        self._strides, self._lse_stride = strides, lse_stride
        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Build one pointer artifact and its immutable dense or THD launch spec."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        self._k_mod = template = _load_kernel_template("sm90/prefill_d512_f16.py", self.params, tag="sdpa_fwd_sm90_d512")
        self._compiled_kernel = template.compile(
            self.batch_size,
            self.h_q,
            self.h_kv,
            self.s_q_max,
            self.s_k_max,
            *self._strides,
            self._lse_stride,
            target="sm_90a",
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
        )
        from .prepared import build_dense_spec, build_thd_spec

        if self.thd:
            self._thd_spec = build_thd_spec(self, scale_softmax=None)
        else:
            self._dense_spec = build_dense_spec(self, scale_softmax=None)
        self._logger.debug("compile completed")

    def scratch_workspace_bytes(self) -> int:
        """THD: the metadata, then the ``B + 3`` tensor maps, bound as ``seq_kv_lens``; a dense plan needs none."""
        self._ensure_support_checked()
        if not self.thd:
            return 0
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_MAPS_META_WORDS

        return THD_MAPS_META_WORDS(self.batch_size) * 4

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization, on the plan's device: the launcher only checks that operands agree."""
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SM90 SDPA: execute requires a compiled plan; runtime JIT is forbidden")
        self._value_error_if(self.has_sink and sinks is None, "sinks is required by this compiled specialization")
        self._value_error_if(
            not self.has_sink and sinks is not None, "this specialization was compiled without sink support; construct the API with has_sink=True"
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        self._value_error_if(self.lse_desc is not None and lse_tensor is None, "lse_tensor is required by this compiled specialization")
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None, "this specialization was compiled without an LSE output; construct the API with sample_lse"
        )
        scale = self.scale_softmax if scale_softmax is None else float(scale_softmax)
        self._value_error_if(
            (scale == 0) != (self.scale_softmax == 0) or (scale < 0) != (self.scale_softmax < 0), "SM90 SDPA scale does not match its compile-time sign mode"
        )
        from .prepared import execute_native_dense_tensors, execute_native_thd_tensors

        stream = self._get_default_stream(current_stream)
        _ensure_current_context(int(stream), q_tensor.device.index)
        if not self.thd:
            # Preserve the direct adapter's declared-layout contract. Graph runtime
            # shape/stride overrides remain unsupported for this fixed tile.
            for tensor, desc, strides in zip((q_tensor, k_tensor, v_tensor, o_tensor), (self.q_desc, self.k_desc, self.v_desc, self.o_desc), self._strides):
                self._check_tensor_shape(tensor, desc.shape, name=desc.name)
                self._value_error_if(
                    any(size != 1 and actual != expected for size, actual, expected in zip(desc.shape, tensor.stride(), strides)),
                    f"{desc.name} tensor stride mismatch: expected {strides} (size-1 axes ignored), got {tensor.stride()}",
                )
        spec = self._thd_spec if self.thd else self._dense_spec
        if self.thd:
            if workspace is None:
                raise ValueError(f"SdpaFwdDslSm90 requires a {spec.scratch_bytes}-byte workspace; pass scratch_workspace_bytes() bytes")
            if workspace.device != q_tensor.device or not workspace.is_contiguous():
                raise ValueError("cudnn.sdpa: THD workspace must be contiguous and on the Q tensor's CUDA device")
            ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm90", spec.scratch_bytes)
            buffers = (q_tensor, k_tensor, v_tensor, o_tensor, seq_q_lens, seq_kv_lens, lse_tensor, sinks)
            lse_geometry = None if self.lse_desc is None else (tuple(self.lse_desc.shape), tuple(self.lse_desc.stride))
            # SM90's host consumes natural scale units, including literal zero.
            execute_native_thd_tensors(spec, buffers, ws_ptr, stream, scale, lse_bhs_geometry=lse_geometry)
        else:
            buffers = (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, sinks, seq_kv_lens, seq_q_lens, None, None, None)
            execute_native_dense_tensors(spec, buffers, stream, scale)
        self._logger.debug("execute completed")

    def _thd_plan(self):
        """Describe fixed-batch metadata and tensor maps in caller workspace."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_MAPS_OFF

        b = self.batch_size
        return SimpleNamespace(
            q=self._thd_decl(self.q_desc),
            k=self._thd_decl(self.k_desc),
            v=self._thd_decl(self.v_desc),
            o=self._thd_decl(self.o_desc),
            units=b * self.h_q * ((self.s_q_max + _SM90_TILE_M - 1) // _SM90_TILE_M),
            cga_tile_m=_SM90_TILE_M,
            n_q_lens=b + int(self.cu_seq_q_lens),
            n_kv_lens=b + int(self.cu_seq_kv_lens),
            lens_form=int(self.cu_seq_q_lens) | (int(self.cu_seq_kv_lens) << 1),
            off_o_desc=THD_MAPS_OFF(b) * 4,
            scratch_bytes=self.scratch_workspace_bytes(),
            total_q=self.max_total_seq_len_q,
            total_kv=self.max_total_seq_len_kv,
        )


def sdpa_fwd_wrapper_dsl_sm90(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens: Optional[torch.Tensor] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM90 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""

    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm90: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 BHSD; got Q={q_tensor.ndim}D K={k_tensor.ndim}D V={v_tensor.ndim}D")
    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)
    cache_key = _make_cache_key(
        SdpaFwdDslSm90,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
    )
    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# ---------------------------------------------------------------------------
# SM120: adapter over the SM120 / SM121 prefill kernel template
# ---------------------------------------------------------------------------


class SdpaFwdDslSm120(SdpaFwdDsl):
    """Compile and execute SM120/SM121 SDPA forward.

    Q, K, V, and O use logical ``(B, H, S, D)`` shapes over any dense layout
    with the head dim innermost (``dense_flex``, same envelope as SM100):
    Native layouts bind directly; conversion layouts use the plan's prepared
    gather/scatter recipes and caller workspace. The kernels support
    FP16/BF16 MHA, GQA, and MQA; head dimensions in multiples of 8 through
    256 (ENVELOPE: the general template compiles at tiles rounded up to 16 and
    TMA zero-fills the pad columns; head dims inside the d256 flavor's envelope
    run the d256 template instead, ``config_sm120.pick_flavor``) plus independently
    sized Q/K and V/O head dimensions in (256, 512], in multiples of 8
    (the d512 template: two warps per Q slab split the head dim,
    CTA tile (64, 32) only); top-left or
    bottom-right causal masks; left sliding
    windows; optional per-batch query and key/value lengths; optional
    per-Q-head attention-sink logits; and THD (ragged / fully packed
    variable-length) batches, compiled at plan time with runtime token totals.

    Half and per-tensor FP8 THD and zero-copy dense plans use a
    prepared pointer entry: pointers, sequence lengths and Int64 strides bind
    at execute without tensor reconstruction. Head geometry and scheduler
    knobs remain compile-time constants. Half dense split-KV also uses this
    entry, retaining batch/Q size; FP8 split and block-scaled O remain tensor entries.

    FP8 Q/K/V constructed without ``pertensor_fp8`` select block-scaled MXFP8
    (the ``sdpa_mxfp8`` op): ``execute()`` takes ``sf_q`` / ``sf_k`` / ``sf_v``
    E8M0 scale factors in the F8_128x4 order instead of per-tensor descales,
    on the MXFP8 template ``sm120/prefill_mxfp8.py`` (both head dims up to
    256, multiples of 16) or its d512 flavor ``sm120/prefill_d512_mxfp8.py``
    (both head dims in (256, 512], multiples of 16).
    Native layouts bind through the prepared MXFP8 entry and conversion layouts
    through the staged forward (the scale tensors bind unchanged). A
    block-scaled MXFP8 O (E4M3 + UE8M0 per 32, ``sample_sf_o``, with an
    optional global ``sample_scale_o``) is served as on the per-tensor path
    at d128; the FP4 (NVFP4) O is per-tensor only.

    ``scale_softmax`` is a runtime parameter. Dtype, head geometry, tile sizes,
    masks and length-tensor / sink / THD presence specialize every path. Legacy
    tensor entries additionally specialize dense batch and sequence extents.
    ``tile_m`` / ``tile_n`` are honored on every template within its kernel
    table (``config_sm120.tile_domain``); left unset, each runs the first KV
    tile of ``config_sm120.kv_tile_preference`` in that table that fits SMEM.
    ``tile_m`` 16 / 32 runs the decode tile (``sm120/decode_f16.py``), whose
    KV tile is its stage width (``config_sm120.decode_kv_stage``).
    """

    def _initialize_implementation(self) -> None:
        self.q_tile = _SM120_Q_TILES[0] if self.tile_m is None else self.tile_m
        self.kv_tile = _SM120_KV_TILES[0] if self.tile_n is None else self.tile_n
        self.flavor: Optional[tuple[int, int]] = None
        self._tile_domain: frozenset[tuple[int, int]] = frozenset()
        self.compute_capability: Optional[tuple[int, int]] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._k_mod = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._not_implemented_error_if(
            self.pv_bf16,
            "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
        )
        self._not_implemented_error_if(self.paged, "paged KV is served by the SM100 f16/bf16 engine only")
        self._not_implemented_error_if(self.gate_desc is not None, "epilogue gate fusion is served by the SM107 d256 SDPA engines only")
        if self.thd:
            self._value_error_if(self.seq_q_lens_present, "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
        )
        self._value_error_if(
            self.cga is not None and self.cga != 1,
            "SM120 DSL SDPA only supports cga=1",
        )

        # Layout gate (same envelope as SM100): THD keeps the strict BSHD
        # stride order (the varlen path binds (1, T, H, D) views with the
        # DECLARED token/head strides; _thd_check_strides_native declines
        # what TMA cannot express); dense graphs get the dense_flex
        # relaxation — execute() normalizes to the kernel's compact-BSHD
        # storage via the prepared staging plan (zero-copy when already
        # BSHD-compact, one gather / scatter copy otherwise), so only what
        # normalization needs is required: head dim innermost-contiguous
        # (stride 1), non-broadcast, non-overlapping strides, any B/H/S
        # order, padded strides allowed.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok, packed_layout_ok, thd_stats_packing

        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            if self.thd:
                # Same rule as the sm100 adapter and the engine gate: the packed
                # path reads token, head and element strides; the batch stride
                # is never stepped under ragged offsets.
                self._value_error_if(
                    not packed_layout_ok(tuple(desc.shape), tuple(desc.stride)),
                    f"THD (ragged) {desc.name} must have d, h, s stride order (head dim innermost, then heads, then tokens); got stride {tuple(desc.stride)}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(desc.shape, desc.stride),
                    f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {desc.stride} shape {desc.shape}",
                )
        if self.thd:
            self._thd_check_strides_native()

        b, h_q, s_q, d_q = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        d_v = self.v_desc.shape[3]
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_q), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_v), name="O")
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_q, s_q), name="LSE")
            self._value_error_if(
                self.thd_stats_padded and not self.thd,
                "thd_stats_padded is THD-only (a padded Stats without ragged offsets); construct the API with thd=True",
            )
            if self.thd and self.thd_stats_padded:
                # per-batch padded Stats (no ragged offsets): (b, h, s_max) in any
                # non-overlapping layout; the kernel indexes [batch, head, row]
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"THD padded LSE must be a non-overlapping (b, h, s_max) layout; got stride {self.lse_desc.stride}",
                )
                self._lse_stride = tuple(int(stride) for stride in self.lse_desc.stride)
            elif self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                packing = thd_stats_packing(stride_h, stride_s, h_q)
                head_major = packing == "head_major"
                self._value_error_if(
                    packing is None,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        for label, val in (
            ("B", b),
            ("H_q", h_q),
            ("H_kv", h_kv),
            ("S_q", s_q),
            ("S_kv", s_kv),
            ("D_QK", d_q),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._value_error_if(
            h_q % h_kv != 0,
            f"H_q ({h_q}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        # Kernel flavor (config_sm120.F16_FLAVORS / FP8_FLAVORS); None = the general template.
        self.flavor = _sm120_pick_flavor(int(d_q), int(d_v), self._fp8)
        # Block-scaled O (sf_o): FP8 inputs, d_v = 128, dense/unsplit/unpacked; the
        # FP4 (NVFP4) O is per-tensor FP8 only.
        self._dtype_o_code = _SM120_DTYPE_QKV_CODE.get(self.o_desc.dtype)
        if self.sf_o_desc is not None or self.o_desc.dtype == _torch_fp4():
            self._not_implemented_error_if(not self._fp8, "a block-scaled O (sf_o / FP4 O) is served by the quantized paths (per-tensor FP8, MXFP8) only")
            if self.o_desc.dtype == _torch_fp4():
                self._not_implemented_error_if(not self._pertensor, "an FP4 (NVFP4) O is served by the per-tensor FP8 path only")
                self._value_error_if(self.sf_o_desc is None, "an FP4 (float4_e2m1fn_x2) O requires sample_sf_o (E4M3 scale factors, one per 16 d elements)")
                self._check_dtype(self.sf_o_desc, torch.float8_e4m3fn, name="sf_o")
                self.o_block_scale, self._dtype_o_code = 16, DTYPE_O_NVFP4
            else:
                self._value_error_if(
                    self.o_desc.dtype != torch.float8_e4m3fn, "sf_o with a non-FP4 O requires an FP8 E4M3 O (MXFP8 output: one UE8M0 scale per 32 d elements)"
                )
                self._check_dtype(self.sf_o_desc, [torch.uint8, torch.float8_e8m0fnu], name="sf_o")
                self.o_block_scale, self._dtype_o_code = 32, DTYPE_O_MXFP8
            self._not_implemented_error_if(int(d_v) != 128 or int(d_q) != 128, f"block-scaled O needs d_qk = d_v = 128; got {(int(d_q), int(d_v))}")
            self._not_implemented_error_if(
                self.thd or self.seq_q_lens_present or self.pack_gqa or self.split_kv > 1,
                "block-scaled O serves dense, untrimmed, unsplit, unpacked graphs only",
            )
            self._sfo_geometry = self._sf_o_geometry(self.o_block_scale, int(d_v))
        # Head dims above the general template's cap exist only where a flavor's
        # tiles reach (d512: both dims in (256, 512]), so the flavor sets the cap.
        head_tile_max = _SM120_FP8_GENERAL_HEAD_TILE_MAX if self._fp8 else _SM120_GENERAL_HEAD_TILE_MAX
        if self.flavor is not None:
            head_tile_max = max(head_tile_max, *self.flavor)
        above_cap = (
            f" (above {_SM120_GENERAL_HEAD_TILE_MAX}, both head dimensions must be in ({_SM120_GENERAL_HEAD_TILE_MAX}, {_SM120_D512_FLAVOR[0]}])"
            if max(int(d_q), int(d_v)) > _SM120_GENERAL_HEAD_TILE_MAX
            else ""
        )
        self._value_error_if(
            d_q % 8 != 0 or not 0 < d_q <= head_tile_max,
            f"D_QK ({d_q}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {head_tile_max}{above_cap}",
        )
        self._value_error_if(
            d_v % 8 != 0 or not 0 < d_v <= head_tile_max,
            f"D_V ({d_v}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {head_tile_max}{above_cap}",
        )
        # The kernel table bounds the CTA tiles (d512: one tile, D512_TILE in
        # f16/bf16 and FP8_D512_TILE in FP8/MXFP8). An unset
        # tile_m takes the flavor's Q tile here so the checks below see it; the
        # KV tile is picked by SMEM fit further down.
        self._tile_domain = _sm120_tile_domain(int(d_q), int(d_v), self._fp8)
        if self.tile_m is None and not any(m == self.q_tile for m, _ in self._tile_domain):
            self.q_tile = max(m for m, _ in self._tile_domain)
        decode_tile = _sm120_is_decode_tile(int(self.q_tile))
        if decode_tile:
            self._not_implemented_error_if(self._fp8, "the SM120 decode tile (tile_m 16 / 32) serves FP16/BF16 only")
            self._not_implemented_error_if(self.thd, "the SM120 decode tile (tile_m 16 / 32) is dense-only (THD keeps the prefill templates)")
            self._not_implemented_error_if(
                self.flavor == _SM120_D512_FLAVOR, f"the SM120 decode tile serves head dims up to {_SM120_GENERAL_HEAD_TILE_MAX}; got {(int(d_q), int(d_v))}"
            )
            self._value_error_if(
                self.sched_policy not in (None, SCHED_NATURAL), f"the SM120 decode tile runs SCHED_NATURAL only; got sched_policy={self.sched_policy}"
            )
        if self.pack_gqa:
            self._not_implemented_error_if(
                self.thd,
                "PackGQA is dense-only (THD/ragged runs unpacked)",
            )
            # Every SM120 template packs any GQA ratio up to the tile, tile_q % G != 0 included.
            self._value_error_if(
                not pack_gqa_supported(int(h_q), int(h_kv), int(self.q_tile), requires_divisible=False),
                f"PackGQA needs h_q/h_kv <= q_tile ({self.q_tile}); got h_q/h_kv = {int(h_q)}/{int(h_kv)}",
            )
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            if self._fp8 and desc is self.o_desc:
                # SDPA_FP8's O dtype is independent of QKV: fp16/bf16 ride the
                # staging epilogue, fp8 the direct quantizing store.
                self._check_dtype(desc, _with_fp4([torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES]), name="O")
            else:
                self._check_dtype(
                    desc,
                    self.dtype,
                    name=desc.name,
                    extra_error_msg=f"{desc.name} must match Q",
                )
            self._value_error_if(
                desc.device != self.q_desc.device,
                f"{desc.name} must be on device {self.q_desc.device}, got {desc.device}",
            )
        if self._fp8:
            self._value_error_if(
                d_q % 16 != 0 or d_v % 16 != 0,
                f"SM120 fp8 requires D_QK/D_V multiples of 16, got ({d_q}, {d_v})",
            )
        if self._fp8 and not self._pertensor:
            self._value_error_if(
                self.has_scale_o and not self.o_block_scale,
                "the MXFP8 scale_o is accepted only together with sf_o (block-scaled O)",
            )

        self._value_error_if(
            self.q_desc.device.type != "cuda",
            f"Q must be a CUDA tensor, got device {self.q_desc.device}",
        )
        self._value_error_if(
            self.q_tile not in _SM120_Q_TILES + _SM120_DECODE_Q_TILES,
            f"q_tile must be one of {_SM120_Q_TILES + _SM120_DECODE_Q_TILES}",
        )
        if decode_tile:
            # The decode tile's KV tile is its stage width, fixed by the head tiles.
            stage = _sm120_decode_kv_stage(*_sm120_head_tiles(d_q, d_v, _SM120_HEAD_TILE_GRANULE))
            self._value_error_if(
                self.tile_n is not None and self.tile_n != stage,
                f"the SM120 decode tile streams {stage}-key K/V stages at D=({d_q}, {d_v}); kv_tile must be {stage} or unset, got {self.tile_n}",
            )
            self.kv_tile = stage
        else:
            self._value_error_if(
                self.kv_tile not in _SM120_KV_TILES,
                f"kv_tile must be one of {_SM120_KV_TILES}",
            )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "causal_bottom_right requires is_causal=True (a band graph arrives as is_causal with its right bound)",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right is not None and self.window_size_right < 0,
            f"window_size_right must be >= 0; got {self.window_size_right}",
        )
        self._value_error_if(
            self.window_size_right is not None and not self.is_causal,
            "window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        self.compute_capability = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            self.compute_capability not in {(12, 0), (12, 1)},
            f"SdpaFwdDslSm120 requires SM120 or SM121, found SM{self.compute_capability[0]}{self.compute_capability[1]}",
        )

        import cutlass

        arch = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import get_smem_capacity_in_bytes

        smem_capacity_bytes = get_smem_capacity_in_bytes(arch)

        # General head dims round to the dtype's granule. The SMEM model also
        # promotes envelope-served dimensions to their flavor's fixed tiles;
        # TMA zero-fills the pad columns in those full-sized allocations.
        d_qp, d_vp = _sm120_head_tiles(d_q, d_v, _SM120_FP8_HEAD_TILE_GRANULE if self._fp8 else _SM120_HEAD_TILE_GRANULE)

        def _smem_bytes(kv_tile: int) -> int:
            # FP8 stages a byte per KV element but still writes O in half; MXFP8
            # adds the K/V scale tiles.
            return _sm120_smem_bytes(
                d_qp, d_vp, self.q_tile, kv_tile, self.dtype.itemsize, 2 if self._fp8 else self.dtype.itemsize, mxfp8=self._fp8 and not self._pertensor
            )

        if self.tile_n is None and not decode_tile:
            # Pick the fastest KV tile in the kernel table that fits this device.
            kv_tiles = _sm120_kv_tile_preference(d_q, mxfp8=self._fp8 and not self._pertensor)
            self.kv_tile = next((t for t in kv_tiles if (self.q_tile, t) in self._tile_domain and _smem_bytes(t) <= smem_capacity_bytes), self.kv_tile)
        self._not_implemented_error_if(
            _smem_bytes(self.kv_tile) > smem_capacity_bytes,
            (
                f"SM120 SDPA requires {_smem_bytes(self.kv_tile)} bytes of shared memory for D={d_q}, "
                f"q_tile={self.q_tile}, and kv_tile={self.kv_tile}, but {arch} provides {smem_capacity_bytes} bytes"
            ),
        )
        self._not_implemented_error_if(
            (self.q_tile, self.kv_tile) not in self._tile_domain,
            f"SM120 SDPA has no kernel for q_tile={self.q_tile}, kv_tile={self.kv_tile} at D=({d_q}, {d_v}); supported: {sorted(self._tile_domain)}",
        )

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_q)
        self._not_implemented_error_if(self.scale_softmax == 0, _ZERO_SCALE_UNSUPPORTED)
        # A negative scale is served by negating Q in the kernel and running at |scale| (#1435).
        self._score_negated = self.scale_softmax < 0

        self._value_error_if(
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM120 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
        )
        if self.split_kv > 1:
            # The SM120 kernel's inline split chunking + the shared (arch-
            # agnostic, one block per row) split_combine pass. The config
            # backstop additionally bars a split under the LPT remaps —
            # validated at compile via make_cfg, and the heuristic's split
            # sets ride SCHED_NATURAL.
            #
            # Partials stay half here (see _fp32_partial_split): sO aliases sKV
            # on this arch, so there is no room to widen the O tile the way the
            # SM100 kernels do.
            #
            # The half templates write sink-free partials and the combine folds
            # the sink once; the FP8 epilogues still fold it per split.
            self._not_implemented_error_if(self.thd, "split_kv > 1 is dense-only (THD packs its own flat grid)")
            self._value_error_if(self.has_sink and self._fp8, "split_kv > 1 with an attention sink is served by the FP16/BF16 templates only")
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                "split_kv > 1 serves unpadded dense graphs only",
            )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM120 DSL SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )
        self._value_error_if(
            self.softmax_scale_prefolded,
            "softmax_scale_prefolded is served by the cc10.7 kernels only (SdpaFwdDslSm100)",
        )

        self.batch_size = int(b)
        self.s_q_max = int(s_q)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_q)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_q)
        self.head_dim_v = int(d_v)
        self._is_supported = True

        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the shape-specialized SM120 FROST template."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        self._staged_spec = None
        self._dense_spec = self._thd_spec = None
        if not self._can_prepare_layout():
            from .prepared_staged_forward import compile_plan

            self._prepared_fp8 = self._prepared_mxfp8 = False
            self._staged_spec = compile_plan(self)
            self._compiled_kernel = self._staged_spec.core.owner
            self._logger.debug("compile completed")
            return

        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path arrives with an explicit
        # policy from the heuristic and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            # THD is excluded: the LPT decodes assume a dense rectangular
            # tile space, while a ragged batch carries its own scheduler,
            # which walks the live units through batch_remap. So is the decode
            # tile, whose few units carry no triangular load to balance.
            if self.window_right is not None and not self.thd and not _sm120_is_decode_tile(int(self.q_tile)):
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        params = Sm120TemplateParams(
            dtype_qkv=_SM120_DTYPE_QKV_CODE[self.dtype],
            # A quantized-O split writes HALF partials; the combine performs
            # the single cast to the real O dtype.
            dtype_o=(_SM120_DTYPE_QKV_CODE[torch.float16] if self._quantized_split() else self._dtype_o_code),
            sched_policy=sched_policy,
            window_left=self.window_left,
            window_right=self.window_right,
            bottom_right=self.causal_bottom_right,
            seq_q_lens_present=self.seq_q_lens_present,
            seq_kv_lens_present=self.seq_kv_lens_present,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2 and self.split_kv == 1,
            thd_varlen=self.thd,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
            pack_gqa=self.pack_gqa,
            split_kv=self.split_kv,
            negate_scores=self._score_negated,
        )
        self._k_mod = _load_sm120_kernel_module(self.flavor, params, fp8=self._fp8, pertensor=self._pertensor)
        self._dense_spec = self._thd_spec = None
        direct_layout = self._can_prepare_layout()
        self._prepared_fp8 = self._can_prepare_fp8()
        self._prepared_mxfp8 = self._can_prepare_mxfp8()
        if not ((not self._fp8 or self._prepared_fp8 or self._prepared_mxfp8) and direct_layout):
            raise NotImplementedError("SM120 forward requires a supported prepared pointer specialization")
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import build_dense_spec, build_thd_spec

        self._compiled_kernel = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=self.batch_size if self.thd else 1,
            qh=self.h_q,
            kh=self.h_kv,
            sq=self.s_q_max if self.thd else 1,
            skv=1,
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            has_lse=(self.lse_desc is not None) or self.split_kv > 1,
            lse_head_major=self.thd_stats_head_major,
            lse_head_stride=self.thd_stats_head_stride,
            lse_padded_rows=self.s_q_max if self.thd_stats_padded else 0,
            lse_stride=self._lse_stride if self.thd_stats_padded else None,
            prepared=True,
            **(
                {
                    "has_amax": self.has_amax_o,
                    "scale_o_in_combine": self._split_scale_o(),
                    **({"sfo_geometry": self._sfo_geometry} if self.o_block_scale else {}),
                    # MXFP8 input: the block-scaled O's optional global scale.
                    **({"has_scale_o": bool(self.o_block_scale and self.has_scale_o)} if self._prepared_mxfp8 else {}),
                }
                if (self._prepared_fp8 or self._prepared_mxfp8)
                else {}
            ),
            persistent_ctas=self._persistent_ctas(self.q_desc.device) if self.flavor == _SM120_D512_FLAVOR else 0,
        )
        if self.thd:
            self._thd_spec = build_thd_spec(self, scale_softmax=None)
            self._logger.debug("compile completed (prepared THD)")
            self._logger.debug("compile completed (THD, dynamic token extents)")
        else:
            self._dense_spec = build_dense_spec(self, scale_softmax=None)
            self._logger.debug("compile completed (prepared dense)")
        self._logger.debug("compile completed")
        return

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        sf_o: Optional[torch.Tensor] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization.

        ``sf_o``: the block-scaled O scale-factor buffer (FP8 or MXFP8 with
        ``sample_sf_o``); its bytes are laid out per the declared geometry.
        On MXFP8, ``scale_o`` is that O's optional global scale (bound exactly
        when constructed with ``sample_scale_o``).
        Prepared FP8 / MXFP8, half THD, split-KV and dense layout conversions require caller-owned ``workspace``
        of at least ``scratch_workspace_bytes()`` bytes, contiguous, 16-byte
        aligned and on the Q device. Graph callers use ``get_workspace_size()``.
        """

        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm120 kernel is not compiled")
        self._value_error_if(
            self.o_block_scale > 0 and sf_o is None,
            "sf_o is required by this compiled specialization (block-scaled O)",
        )
        self._value_error_if(
            self.o_block_scale == 0 and sf_o is not None,
            "this specialization was compiled without a block-scaled O; construct the API with sample_sf_o",
        )
        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        scale_val = self.scale_softmax if scale_softmax is None else float(scale_softmax)
        self._value_error_if(scale_val == 0, _ZERO_SCALE_UNSUPPORTED)
        self._value_error_if((scale_val < 0) != self._score_negated, "attn_scale sign must match the compiled plan's (#1435)")
        scale_val = abs(scale_val)
        mxfp8 = self._fp8 and not self._pertensor
        self._value_error_if(
            mxfp8 and any(t is not None for t in (descale_q, descale_k, descale_v)),
            "SM120 MXFP8 takes block scale factors (sf_q / sf_k / sf_v), not per-tensor descales",
        )
        self._value_error_if(
            mxfp8 and (scale_o is not None) != bool(self.o_block_scale and self.has_scale_o),
            "the SM120 MXFP8 scale_o is the block-scaled O's global scale: pass it exactly when constructed with sample_sf_o and sample_scale_o",
        )
        if mxfp8:
            quant = dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v, amax_o=amax_o, sf_o=sf_o, scale_o=scale_o)
        elif self._fp8:
            quant = dict(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o, amax_o=amax_o, sf_o=sf_o)
        else:
            quant = {}
        if getattr(self, "_staged_spec", None) is not None:
            from .prepared_staged_forward import execute as execute_staged

            self._value_error_if(
                not mxfp8 and any(t is not None for t in (sf_q, sf_k, sf_v)),
                "SM120 staged forward takes MXFP8 input scales on MXFP8 plans only",
            )
            tensors = dict(q=q_tensor, k=k_tensor, v=v_tensor, o=o_tensor, lse=lse_tensor, sinks=sinks, seq_q_lens=seq_q_lens, seq_kv_lens=seq_kv_lens)
            execute_staged(self, {**tensors, **quant}, workspace, current_stream, scale_val)
            return
        if self._fp8:
            self._value_error_if(
                not mxfp8 and any(t is not None for t in (sf_q, sf_k, sf_v)),
                "SM120 fp8 is per-tensor (scalar descales); block-scale SF tensors are MXFP8-only",
            )
            self._execute_fp8_prepared(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_q_lens,
                seq_kv_lens,
                quant,
                scale_val,
                workspace,
                current_stream,
            )
            self._logger.debug("execute (SM120 MXFP8) completed" if mxfp8 else "execute (SM120 FP8 per-tensor) completed")
            return
        scale_softmax_log2 = scale_val * math.log2(math.e)
        if self._dense_spec is not None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import execute_native_dense_tensors, facts_of_tensor

            current_stream = self._get_default_stream(current_stream)
            stream_int = int(current_stream)
            _ensure_current_context(stream_int, q_tensor.device.index)
            ws_ptr = 0
            if self.split_kv > 1:
                if workspace is None:
                    raise ValueError(f"SdpaFwdDslSm120 requires a {self.scratch_workspace_bytes()}-byte workspace; pass scratch_workspace_bytes() bytes")
                if workspace.device != q_tensor.device or not workspace.is_contiguous():
                    raise ValueError("cudnn.sdpa: split workspace must be contiguous and on the Q tensor's CUDA device")
                ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm120 (split)")
            execute_native_dense_tensors(
                self._dense_spec,
                (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, sinks, seq_kv_lens, seq_q_lens, None, None, None),
                current_stream,
                scale_softmax_log2,
                ws_ptr,
            )
            self._logger.debug("execute completed (prepared dense)")
            return
        if self.thd:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
            )
            return
        raise RuntimeError("SM120 forward has no compiled prepared launch")

    def _thd_plan(self):
        """SM120 uses the common operand contract and its own persistent metadata recipe."""
        b = self.batch_size
        return SimpleNamespace(
            q=self._thd_decl(self.q_desc),
            k=self._thd_decl(self.k_desc),
            v=self._thd_decl(self.v_desc),
            o=self._thd_decl(self.o_desc),
            units=self._persistent_ctas(self.q_desc.device),
            cga_tile_m=int(self.q_tile),
            n_q_lens=b + int(self.cu_seq_q_lens),
            n_kv_lens=b + int(self.cu_seq_kv_lens),
            lens_form=int(self.cu_seq_q_lens) | (int(self.cu_seq_kv_lens) << 1),
            # SM120 has no per-sequence O descriptor. This common ABI slot is dead.
            off_o_desc=0,
            scratch_bytes=self.scratch_workspace_bytes(),
            total_q=self.max_total_seq_len_q,
            total_kv=self.max_total_seq_len_kv,
        )

    def _execute_thd(
        self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_kv_lens, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None
    ):
        """Execute every half THD plan through its prepared pointer binding."""
        if self._thd_spec is None:
            raise RuntimeError("SM120 half THD requires a compiled prepared launch")
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import execute_native_thd_tensors

        spec = self._thd_spec
        current_stream = self._get_default_stream(current_stream)
        stream_int = int(current_stream)
        _ensure_current_context(stream_int, q_buf.device.index)
        if workspace is None:
            raise ValueError(f"SdpaFwdDslSm120 requires a {spec.scratch_bytes}-byte workspace; pass scratch_workspace_bytes() bytes")
        if workspace.device != q_buf.device or not workspace.is_contiguous():
            raise ValueError("cudnn.sdpa: THD workspace must be contiguous and on the Q tensor's CUDA device")
        ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm120 (THD)", spec.scratch_bytes)
        buffers = (q_buf, k_buf, v_buf, o_buf, seq_q_lens, seq_kv_lens, lse_tensor, sinks)
        execute_native_thd_tensors(spec, buffers, ws_ptr, current_stream, scale_softmax_log2)
        self._logger.debug("execute completed (prepared THD)")

    def _persistent_ctas(self, device) -> int:
        """CTA count for a persistent grid (THD on every template, dense on d512).

        Sized to the MACHINE, not to the work: the live unit total is a
        device-side quantity (issue #552), a CTA with nothing left to claim just
        retires, and one with more work loops. So over-launching is harmless and
        under-launching only costs parallelism. One CTA per SM matches the
        kernel's ``min_blocks_per_mp``.
        """
        key = _thd_cache_key(device)
        n = _THD_CTAS_CACHE.get(key)
        if n is None:
            forced = int(os.environ.get("FROST_THD_CTAS", "0"))
            if forced > 0:
                n = forced
            else:
                sms = torch.cuda.get_device_properties(device).multi_processor_count
                per_sm = int(os.environ.get("FROST_THD_CTAS_PER_SM", "1"))
                n = max(1, sms * max(1, per_sm))
            _THD_CTAS_CACHE[key] = n
        return n

    def _can_prepare_layout(self):
        operands = (self.q_desc, self.k_desc, self.v_desc) + (() if self.split_kv > 1 else (self.o_desc,))
        return self.thd or all(self._prepared_operand_layout(desc) is not None for desc in operands)

    def _can_prepare_fp8(self):
        return self._fp8 and self._pertensor and (not self.o_block_scale or self._can_prepare_block_output()) and self._can_prepare_layout()

    def _can_prepare_mxfp8(self):
        return self._fp8 and not self._pertensor and (not self.o_block_scale or self._can_prepare_block_output()) and self._can_prepare_layout()

    def _prepared_quant_offset(self):
        if self.split_kv > 1:
            return self._split_workspace_bytes()
        # SM120 has metadata (the work prefix included) but no per-sequence TMA O descriptor array.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_WORK_META_WORDS

        return ws_align(THD_WORK_META_WORDS(self.batch_size) * 4) if self.thd else 0

    def scratch_workspace_bytes(self) -> int:
        self._ensure_support_checked()
        compiled = self._compiled_kernel is not None
        staged = getattr(self, "_staged_spec", None)
        if staged is not None:
            return staged.workspace_bytes
        if not compiled and not self._can_prepare_layout():
            from .prepared_staged_forward import workspace_bytes

            return workspace_bytes(self)
        if (self._prepared_fp8 or self._prepared_mxfp8) if compiled else (self._can_prepare_fp8() or self._can_prepare_mxfp8()):
            return self._prepared_quant_offset() + ws_align(8)
        if self.thd:
            # [meta(seq_kv, cu_q, cu_k)].
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552). No sinks-dummy chunk: the kernel None-specializes
            # on sinks. No O-descriptor chunk: SM120 stores O with plain
            # guarded GMEM stores, so THD needs no per-sequence tensor maps.
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_WORK_META_WORDS

            # [kv | cu_q | cu_k | remap | live | ctr | unit prefix]: the setup
            # launch publishes the remap-ordered unit prefix the kernel's unit
            # decode binary-searches (thd_decode_unit_prefix).
            return ws_align(THD_WORK_META_WORDS(self.batch_size) * 4)
        if self.split_kv > 1:
            # Split-major partial slabs (see the SM100 sibling): O_s in the O
            # dtype (half) + lse_s fp32, carved from the caller's workspace.
            return self._split_workspace_bytes()
        return 0


def sdpa_fwd_wrapper_dsl_sm120(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens: Optional[torch.Tensor] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    q_tile: Optional[int] = None,
    kv_tile: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM120 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""

    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm120: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 BHSD; got Q={q_tensor.ndim}D K={k_tensor.ndim}D V={v_tensor.ndim}D")
    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)
    cache_key = _make_cache_key(
        SdpaFwdDslSm120,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        tile_m=q_tile,
        tile_n=kv_tile,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        tile_m=q_tile,
        tile_n=kv_tile,
    )
    required = sdpa_fwd.scratch_workspace_bytes()
    workspace = torch.empty(required, dtype=torch.uint8, device=q_tensor.device) if required else None
    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
        workspace=workspace,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# =============================================================================
# SM80 (A100) adapter — SdpaFwdDslSm80 + the sdpa_fwd_wrapper_sm80 entry point.
#
# SM80 templates specialize through TemplateParams/load_template. Native dense
# and packed launches use prepared pointer hosts; dense conversion routes
# share prepared copies, including the standalone RoPE specialization.
# =============================================================================

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import config_sm80 as _sm80_config

_SM80_FLAVOR_CFGS = {
    "gptoss": _sm80_config.GPTOSS_CFG,
    "llama": _sm80_config.LLAMA_CFG,
    "dsv3": _sm80_config.DSV3_CFG,
    "qwen": _sm80_config.QWEN_CFG,
}

# (D_QK, D_V) envelope per flavor.
_SM80_FLAVOR_DIMS = {name: (cfg.D_QK, cfg.D_V) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# (tile_m, num_warps, tile_n) per flavor — frozen from the A100 perf sweep.
_SM80_FLAVOR_KNOBS = {name: (cfg.TILE_M, cfg.NUM_WARPS, cfg.TILE_N) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# Causal L2 budget (MiB) for ``sched=lpt_l2`` per flavor.  Larger d_qk
# inflates the per-(B, H) resident set so dsv3 needs a smaller group.
_SM80_FLAVOR_CAUSAL_L2_MIB = {
    "llama": 16,
    "gptoss": 16,
    "dsv3": 8,
    "qwen": 8,
}

# Ascending (D_QK, D_V) order so the flavor pick walks closest-from-above.
_SM80_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")

# Flavors that route to the dedicated d=256 kernel (symmetric K+V prefetch);
# all others use the shared generic kernel.
_SM80_D256_FLAVORS = ("qwen",)


def _sm80_pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest kernel flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)``.  Exact-match wins when both axes match; otherwise walk
    gptoss → llama → dsv3 → qwen and pick the first that fits.  Raises if
    nothing fits (heads bigger than the qwen envelope are not supported on
    SM80 yet)."""
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(
        f"SM80 SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}).  "
        f"Supported envelopes: {_SM80_FLAVOR_DIMS}.  Heads larger than qwen "
        "(256/256) are not yet ported to SM80."
    )


def _sm80_resolve_scheduler(
    *,
    scheduler: str,
    flavor: str,
    is_causal: bool,
    swa_window: int,
    skv: int,
) -> tuple[str, int]:
    """Return ``(sched_token, sched_l2_mib)`` to pass to the kernel."""
    l2_mib = _SM80_FLAVOR_CAUSAL_L2_MIB[flavor]
    if scheduler == "auto":
        if is_causal:
            return "lpt_l2", l2_mib
        if swa_window > 0:
            # SWA heuristic — LPT wins for 1K ≤ SKV ≤ 16K.
            return ("lpt" if 1024 <= skv <= 16384 else "default"), l2_mib
        return "default", l2_mib
    if scheduler in ("natural", "default"):
        return "default", l2_mib
    if scheduler == "lpt":
        return "lpt", l2_mib
    if scheduler == "lpt_l2":
        return "lpt_l2", l2_mib
    raise ValueError(f"SM80 SDPA: scheduler must be 'auto' / 'default' / 'natural' / 'lpt' / 'lpt_l2', got {scheduler!r}")


# --- SM80 template loading ---------------------------------------------------

_LOG2E = math.log2(math.e)

_SM80_KERNEL_FILES = {
    "d256": "sm80/prefill_d256_f16.py",
    "f16": "sm80/prefill_f16.py",
}


def _sm80_load_kernel_module(flavor: str, params):
    """One uniquely-named module per (kernel file, TemplateParams) — the same
    ``frost.template_loader`` mechanism the SM100/SM120 templates use. qwen
    (d=256) routes to the symmetric-K+V-prefetch file; the rest share the
    generic kernel."""
    filename = _SM80_KERNEL_FILES["d256" if flavor in _SM80_D256_FLAVORS else "f16"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    # Tag from the BASENAME: `filename` carries the arch subdirectory, and a
    # slash would land in the generated template module name.
    stem = os.path.splitext(os.path.basename(filename))[0]
    return load_template(path, params, tag=f"sm80_{stem}")


def _sm80_sched_policy_int(token: str) -> int:
    return {"default": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[token]


class SdpaFwdDslSm80(SdpaFwdDsl):
    """SM80 (A100) SDPA forward via the FROST template kernels.

    Since the TemplateParams conversion this adapter has the same shape as
    its SM100/SM120 siblings end to end: ``check_support`` resolves the
    flavor/mask/scheduler into a plan-time :class:`config_sm80.TemplateParams`,
    ``compile()`` loads the specialized template module and compiles the
    artifact once (packed token capacities bind as runtime Int64 values), and ``execute()`` re-binds
    caller buffers to the cached artifact (a compile-cache miss at execute is
    a bug by contract).

    SM80-only compile axes that have no home in the shared constructor arrive
    as extra keyword-only arguments (``bias_present`` / ``bias_fp32`` /
    ``rope_max_s``); the engine lowering forwards them only when the graph
    declares the operands. ALiBi, block_mask and the score-stat side outputs
    are deliberately NOT served: the capability row declines such graphs and
    the backend takes them.

    Dense vector-aligned declarations with a native V width use one prepared
    pointer host, including native GQA and strided Q/K/V/O/Stats. Other layouts,
    and off-flavor widths use prepared bitwise copies and native GQA. RoPE
    shares that plan, retaining only its standalone angle-table preprocessing.
    """

    def __init__(self, *args, scheduler: Optional[str] = None, bias_present: bool = False, bias_fp32: bool = False, rope_max_s: int = 0, **kwargs) -> None:
        # SM80-only plan-time axes (see class docstring). ``scheduler`` is the
        # token override for standalone callers; the graph path leaves it None
        # and carries the heuristic's explicit sched_policy knob instead
        # (None = derive via "auto", explicit ints map to their tokens).
        self._scheduler_token = scheduler
        self._bias_present = bool(bias_present)
        self._bias_fp32 = bool(bias_fp32)
        self._rope_max_s = int(rope_max_s)
        super().__init__(*args, **kwargs)

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.kernel_tile_m: Optional[int] = None
        self.kernel_num_warps: Optional[int] = None
        self.kernel_tile_n: Optional[int] = None
        self.sched_token: Optional[str] = None
        self.sched_l2_mib: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound_runtime: int = 0
        self._k_mod = None
        self._params = None
        self._sm80_spec = None
        self._sm80_copy_spec = None
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._thd_meta_fn = None
        self._thd_lse_head_stride = 0
        self._thd_lse_token_major = False
        self._t_q_cap = 0
        self._t_kv_cap = 0
        self._thd_launch = None

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._not_implemented_error_if(
            self.pv_bf16,
            "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
        )
        self._not_implemented_error_if(
            self.sf_o_desc is not None or self.o_desc.dtype == _torch_fp4(),
            "block-scaled O (sf_o / FP4 O) is served by the SM100-family per-tensor FP8 engines only",
        )

        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok, packed_layout_ok, thd_stats_packing

        self._not_implemented_error_if(self.paged, "paged KV is served by the SM100 f16/bf16 engine only")
        self._not_implemented_error_if(self.gate_desc is not None, "epilogue gate fusion is served by the SM107 d256 SDPA engines only")
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            _shape, _stride = tuple(desc.shape), tuple(desc.stride)
            if self.thd:
                self._not_implemented_error_if(
                    not packed_layout_ok(_shape, _stride) or not self._thd_declared(desc)[1],
                    f"SM80 THD {desc.name} requires compact packed BSHD strides; got {_stride} for shape {_shape}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(_shape, _stride),
                    f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {_stride} shape {_shape}",
                )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        max_d_qk = max(fdqk for fdqk, _ in _SM80_FLAVOR_DIMS.values())
        max_d_v = max(fdv for _, fdv in _SM80_FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_d_qk or d_v > max_d_v,
            f"SM80 SDPA: head dim (D_QK={d_qk}, D_V={d_v}) exceeds "
            f"supported envelope (D_QK<={max_d_qk}, D_V<={max_d_v}).  "
            f"Larger heads are not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16 on SM80)",
            )
        self._not_implemented_error_if(
            self._pertensor or self.dtype_o is not None,
            "SM80 SDPA serves f16/bf16 only (no FP8/MXFP8 and no dtype_o override)",
        )
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            if self.thd:
                packing = thd_stats_packing(int(self.lse_desc.stride[1]), int(self.lse_desc.stride[2]), h_qo)
                if self.max_total_seq_len_q == 0 and tuple(self.lse_desc.stride[1:3]) == (0, 1):
                    # A zero-token packed Stats tensor has no head span, so
                    # PyTorch naturally reports a zero head stride.
                    packing = "head_major"
                self._not_implemented_error_if(
                    self.thd_stats_padded or packing is None,
                    f"SM80 THD Stats must be packed token-major or head-major; got stride {self.lse_desc.stride}",
                )
                self._thd_lse_token_major = packing == "token_major"
                self._thd_lse_head_stride = int(self.lse_desc.stride[1])
                if self._thd_lse_token_major:
                    # The batch axis has extent one in the packed kernel ABI;
                    # its stride is never stepped. Both remaining strides are
                    # static, independent of the dynamic packed token total.
                    self._lse_stride = (0, 1, h_qo)
            else:
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        self._not_implemented_error_if(
            not self.thd and (self.cu_seq_q_lens or self.cu_seq_kv_lens),
            "SdpaFwdDslSm80 does not serve dense cu_seq_len graphs",
        )
        if self.thd:
            self._not_implemented_error_if(
                self.cu_seq_q_lens and self.seq_q_lens_present,
                "SM80 THD Q lengths cannot be both per-batch and cumulative",
            )
            self._not_implemented_error_if(
                (d_qk, d_v) not in _SM80_FLAVOR_DIMS.values(),
                f"SM80 THD graph path supports native head dims {sorted(_SM80_FLAVOR_DIMS.values())} only",
            )
            self._not_implemented_error_if(
                self._bias_present or self.has_sink or self._rope_max_s,
                "SM80 THD graph path does not support bias, sinks, or RoPE fusion",
            )
            self._not_implemented_error_if(
                not self.seq_kv_lens_present and not self.cu_seq_kv_lens,
                "SM80 THD graph path requires Q and KV lengths",
            )
            self._t_q_cap = min(b * s_qo, self.max_total_seq_len_q) if self.max_total_seq_len_q is not None else b * s_qo
            self._t_kv_cap = min(b * s_kv, self.max_total_seq_len_kv) if self.max_total_seq_len_kv is not None else b * s_kv
            if self.lse_desc is not None and not self._thd_lse_token_major:
                if self.max_total_seq_len_q is None:
                    # A compact packed Stats head stride also declares a
                    # conservative Q capacity when no explicit total exists.
                    # Runtime Q/O buffers must still fit this planned bound.
                    self._t_q_cap = min(self._t_q_cap, self._thd_lse_head_stride)
                self._not_implemented_error_if(
                    self._thd_lse_head_stride < self._t_q_cap,
                    f"SM80 THD head-major Stats stride must cover packed Q capacity {self._t_q_cap}; got {self._thd_lse_head_stride}",
                )
                # A static head stride is part of the declared layout. Keeping
                # it explicit also lets a later execute rebind a smaller Q
                # token extent without changing the compiled LSE addressing.
                self._lse_stride = (0, self._thd_lse_head_stride, 1)
        self._not_implemented_error_if(
            self.window_size_right is not None and not self.is_causal,
            "SM80 SDPA: window_size_right without is_causal=True has no diagonal to anchor to",
        )
        self._not_implemented_error_if(
            self._rope_max_s and (self.seq_kv_lens_present or self.seq_q_lens_present),
            "SM80 SDPA: RoPE fusion is dense-unpadded-only",
        )
        self._not_implemented_error_if(
            self.split_kv > 1,
            "SM80 SDPA has no split-KV path (no partial slabs, no combine kernel)",
        )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM80 SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )
        self._value_error_if(
            self.softmax_scale_prefolded,
            "softmax_scale_prefolded is served by the cc10.7 kernels only (SdpaFwdDslSm100)",
        )

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._device_cc = (major, minor)
        self._value_error_if(
            (major, minor) != (8, 0),
            f"SdpaFwdDslSm80 requires SM80 (A100); found SM{major}{minor} on {device}",
        )

        self.flavor = _sm80_pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _SM80_FLAVOR_DIMS[self.flavor]
        tile_m_default, num_warps_default, tile_n_default = _SM80_FLAVOR_KNOBS[self.flavor]
        self.kernel_tile_m = tile_m_default if self.tile_m is None else int(self.tile_m)
        self.kernel_num_warps = num_warps_default
        self.kernel_tile_n = tile_n_default if self.tile_n is None else int(self.tile_n)
        self._value_error_if(
            self.cga not in (None, 1),
            f"SM80 SDPA has no CGA clustering; cga must be 1 (or unset), got {self.cga}",
        )

        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or (self.window_size_left is not None and self.window_size_left >= 0)),
            "SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).",
        )

        swa_left = -1 if self.window_size_left is None else int(self.window_size_left)
        swa_right = 0 if self.window_size_right is None else int(self.window_size_right)
        self.right_bound_runtime = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound_runtime = max(0, swa_right)
        elif swa_left >= 0:
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            self.mask_token = "none"
            self.swa_window_runtime = 0

        token = self._scheduler_token
        if token is None:
            # None = no preference anywhere -> "auto" (the adapter derives, a
            # standalone-wrapper convenience). An explicit knob is honored
            # verbatim — NATURAL included — never re-derived.
            if self.sched_policy is None:
                token = "auto"
            else:
                token = {SCHED_NATURAL: "default", SCHED_LPT: "lpt", SCHED_LPT_L2: "lpt_l2"}.get(self.sched_policy)
                self._value_error_if(
                    token is None,
                    f"SM80 SDPA: unsupported sched_policy {self.sched_policy}",
                )
        else:
            _VALID = ("auto", "natural", "default", "lpt", "lpt_l2")
            self._value_error_if(token not in _VALID, f"scheduler must be one of {_VALID}; got {token!r}")
        if self.thd:
            self._not_implemented_error_if(
                token not in ("auto", "natural", "default"),
                "SM80 THD forward requires the natural scheduler (the varlen grid is 3-D)",
            )
            token = "default"
        self.sched_token, self.sched_l2_mib = _sm80_resolve_scheduler(
            scheduler=token,
            flavor=self.flavor,
            is_causal=self.is_causal,
            swa_window=self.swa_window_runtime,
            skv=int(s_kv),
        )

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Load the TemplateParams-specialized module and compile the artifact.

        Plan-time only (Hard Rule 4): every key component here is graph
        declaration or capability data. execute() binds current buffers and
        launches the compiled host without looking up a compiler cache.
        """
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import config_sm80 as _sm80_cfg

        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_sm80 import build_spec, native_layouts

        prepared = not self.thd and native_layouts(self)
        self._params = _sm80_cfg.TemplateParams(
            io_bf16=(self.dtype == torch.bfloat16),
            d_qk=self.flavor_d_qk,
            d_v=self.flavor_d_v,
            tile_m=self.kernel_tile_m,
            num_warps=self.kernel_num_warps,
            tile_n=self.kernel_tile_n,
            is_causal=self.mask_token in ("causal", "causal_swa"),
            has_swa=self.mask_token in ("swa", "causal_swa"),
            causal_bottom_right=self.causal_bottom_right,
            has_seq_kv_lens=self.seq_kv_lens_present and not self.thd,
            has_seq_q_lens=self.seq_q_lens_present and not self.thd,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2,
            has_bias=self._bias_present,
            bias_is_fp32=self._bias_fp32,
            has_rope=self._rope_max_s > 0,
            thd_varlen=self.thd,
            sched_policy=_sm80_sched_policy_int(self.sched_token),
            sched_l2_mib=self.sched_l2_mib,
            has_lse=self.lse_desc is not None,
            sink_natural=True,
            score_sign=_sm80_cfg.score_sign(self.scale_softmax),
        )
        self._k_mod = _sm80_load_kernel_module(self.flavor, self._params)
        if self.thd:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm80.prepared_host import compile_thd_host

            self._compiled_kernel, self._thd_launch = compile_thd_host(
                self._k_mod,
                self.h_q,
                self.h_kv,
                self.batch_size,
                int(self.swa_window_runtime),
                lse_stride=self._lse_stride,
                initialize_outputs=False,
            )
            self._compile_thd_meta()
            self._logger.debug("compile completed")
            return
        if prepared:
            self._sm80_spec = build_spec(self)
            self._compiled_kernel = self._sm80_spec.artifact
            self._logger.debug("compile completed")
            return
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_staged_sm80 import compile_plan

        self._sm80_copy_spec = compile_plan(self)
        self._compiled_kernel = self._sm80_copy_spec.core.artifact
        self._logger.debug("compile completed")

    def _compile_thd_meta(self) -> None:
        """Build the device-side lengths-to-cumulative-offsets launch at plan time."""
        import cutlass
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_META_WORDS
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.bwd.kernels.thd_helpers import thd_meta_host

        b = self.batch_size
        i32 = lambda n: make_fake_compact_tensor(cutlass.Int32, (n,), stride_order=(0,), assumed_align=4)  # noqa: E731
        self._thd_meta_fn = cutlass.cute.compile(
            thd_meta_host,
            i32(THD_META_WORDS(b)),
            i32(b + 1 if self.cu_seq_q_lens else b),
            i32(b + 1 if self.cu_seq_kv_lens else b),
            cutlass.Int32(0),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi --gpu-arch=sm_80",
        )

    def scratch_workspace_bytes(self) -> int:
        """Plan-owned gather/scatter storage; native operands need no scratch."""
        if self._sm80_spec is not None:
            return 0
        if self._sm80_copy_spec is not None:
            return self._sm80_copy_spec.workspace_bytes
        self._ensure_support_checked()
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_sm80 import native_layouts
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_staged_sm80 import workspace_bytes

        if self.thd:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_META_WORDS

            return ws_align(THD_META_WORDS(self.batch_size) * 4)
        return 0 if native_layouts(self) else workspace_bytes(self)

    def _execute_thd(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor],
        seq_q_lens: Optional[torch.Tensor],
        seq_kv_lens: Optional[torch.Tensor],
        scale_softmax: Optional[float],
        workspace: Optional[torch.Tensor],
        current_stream: Optional[cuda.CUstream],
    ) -> None:
        """Bind native packed buffers and launch the already-compiled THD artifact."""
        import cutlass
        from cutlass.cute.runtime import from_dlpack as from_dlpack_raw

        device = self.q_desc.device
        self._value_error_if(seq_q_lens is None or seq_kv_lens is None, "SM80 THD requires Q and KV lengths")
        q_lens = self._checked_cu_seq_lens(seq_q_lens, "cu_seq_len_q") if self.cu_seq_q_lens else self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        kv_lens = self._checked_cu_seq_lens(seq_kv_lens, "cu_seq_len_kv") if self.cu_seq_kv_lens else self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        self._value_error_if(q_lens.device != device or kv_lens.device != device, f"SM80 THD lengths must be on {device}")
        self._value_error_if((lse_tensor is None) != (self.lse_desc is None), "SM80 THD Stats presence must match the compiled plan")

        def capacity(buf, desc):
            self._value_error_if(buf.dtype != desc.dtype or buf.device != device, f"{desc.name}: runtime dtype/device must match its declaration")
            self._value_error_if(buf.data_ptr() % 16 != 0, f"{desc.name}: runtime buffer base address must be 16-byte aligned")
            if buf.numel() == 0:
                return 0
            span = 1 + sum((n - 1) * st for n, st in zip(buf.shape, buf.stride()))
            return span // (int(desc.shape[1]) * int(desc.shape[3]))

        t_q = min(capacity(q_tensor, self.q_desc), capacity(o_tensor, self.o_desc))
        if self.max_total_seq_len_q is not None:
            t_q = min(t_q, self.max_total_seq_len_q)
        if self.max_total_seq_len_q is None and self.lse_desc is not None and not self._thd_lse_token_major:
            # The packed head-major Stats stride is an exact Q-token bound
            # when the caller did not supply max_total_seq_len_q. Q/O may
            # legitimately carry extra allocation capacity beyond it.
            t_q = min(t_q, self._thd_lse_head_stride)
        t_kv = min(capacity(k_tensor, self.k_desc), capacity(v_tensor, self.v_desc))
        if self.max_total_seq_len_kv is not None:
            t_kv = min(t_kv, self.max_total_seq_len_kv)
        # Planned capacities bound live totals (caller contract); extra
        # allocation beyond them is never addressed.
        t_q = min(t_q, self._t_q_cap)
        t_kv = min(t_kv, self._t_kv_cap)

        lse = None
        if lse_tensor is not None:
            self._value_error_if(
                lse_tensor.dtype != torch.float32 or lse_tensor.device != device,
                f"SM80 THD Stats must be fp32 on {device}",
            )
            if self._thd_lse_token_major:
                self._value_error_if(
                    lse_tensor.ndim != 2 or lse_tensor.shape[0] < t_q or lse_tensor.shape[1] != self.h_q or tuple(lse_tensor.stride()) != (self.h_q, 1),
                    f"SM80 THD token-major Stats must have shape (>= {t_q}, {self.h_q}) and compact strides",
                )
                lse = lse_tensor.as_strided((1, self.h_q, t_q), (0, 1, self.h_q), lse_tensor.storage_offset())
            else:
                if t_q == 0:
                    self._value_error_if(lse_tensor.numel() != 0, "SM80 THD zero-token Stats must be empty")
                else:
                    self._value_error_if(
                        lse_tensor.ndim != 3
                        or lse_tensor.shape[0] != 1
                        or lse_tensor.shape[1] != self.h_q
                        or lse_tensor.shape[2] < t_q
                        or tuple(lse_tensor.stride()[1:]) != (self._thd_lse_head_stride, 1),
                        f"SM80 THD head-major Stats must have shape (1, {self.h_q}, >= {t_q}) and strides (head={self._thd_lse_head_stride}, token=1)",
                    )
                lse = lse_tensor.as_strided(
                    (1, self.h_q, t_q),
                    (0, self._thd_lse_head_stride, 1),
                    lse_tensor.storage_offset(),
                )

        self._value_error_if(workspace is not None and workspace.device != device, f"SM80 THD workspace must be on {device}")
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "SdpaFwdDslSm80 THD")
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_META_WORDS

        meta = carver.take(THD_META_WORDS(self.batch_size), torch.int32)
        b = self.batch_size
        cu_q = meta[b : 2 * b + 1]
        cu_kv = meta[2 * b + 1 : 3 * b + 2]
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import config_sm80 as _sm80_cfg

        logical_scale = self.scale_softmax if scale_softmax is None else float(scale_softmax)
        self._value_error_if(
            _sm80_cfg.score_sign(logical_scale) != self._params.score_sign,
            "SM80 THD launch scale sign must match the compiled specialization",
        )
        scale = _sm80_cfg.kernel_scale(logical_scale)
        launch_stream = self._get_default_stream(current_stream)
        tvm = lambda tensor: from_dlpack_raw(tensor, enable_tvm_ffi=True)  # noqa: E731

        with _torch_stream_context(current_stream, device):
            lens_form = (1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)
            self._thd_meta_fn(tvm(meta), tvm(q_lens), tvm(kv_lens), cutlass.Int32(lens_form), cutlass.Int32(b), launch_stream)
            self._thd_launch(
                q_tensor.data_ptr(),
                k_tensor.data_ptr(),
                v_tensor.data_ptr(),
                o_tensor.data_ptr(),
                lse.data_ptr() if lse is not None else None,
                cu_q.data_ptr(),
                cu_kv.data_ptr(),
                None,
                int(t_q),
                int(t_kv),
                int(self.s_q_max),
                self.h_q * self.head_dim_qk,
                self.head_dim_qk,
                self.h_kv * self.head_dim_qk,
                self.head_dim_qk,
                self.h_kv * self.head_dim_v,
                self.head_dim_v,
                scale * _LOG2E,
                1.0 / scale,
                int(self.right_bound_runtime),
                (1, 1, 1),
                int(launch_stream),
            )

    # ------------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm80 is not compiled")
        if self.thd:
            self._value_error_if(
                sinks is not None or bias_tensor is not None or rope_freqs is not None, "SM80 THD does not accept sinks, bias, or RoPE tensors"
            )
            self._execute_thd(q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, seq_q_lens, seq_kv_lens, scale_softmax, workspace, current_stream)
            return
        if self._sm80_copy_spec is not None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_staged_sm80 import execute

            self._value_error_if(self._params.has_rope != (rope_freqs is not None), "rope_freqs presence must match the compiled specialization")
            buffers = (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, seq_kv_lens, seq_q_lens, sinks, bias_tensor)
            execute(self, buffers, workspace, current_stream, scale_softmax, rope_freqs=rope_freqs)
            self._logger.debug("execute completed")
            return
        if self._sm80_spec is not None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_sm80 import execute_tensors

            self._value_error_if(rope_freqs is not None, "rope_freqs was not compiled into this specialization")
            buffers = (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, seq_kv_lens, seq_q_lens, sinks, bias_tensor)
            device = q_tensor.device
            context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
            with context:
                stream = self._get_default_stream(current_stream)
                _ensure_current_context(int(stream), device.index)
                execute_tensors(self._sm80_spec, buffers, int(stream), scale=scale_softmax)
            self._logger.debug("execute completed")
            return

        raise RuntimeError("SM80 forward is missing its prepared launch plan")


def _sm80_thd_forward(q, k, v, *, cu_q, cu_k, max_s_q, scale_softmax, is_causal, window_size, causal_bottom_right, bias_tensor, sinks, current_stream=None):
    """THD / varlen forward: q/k/v are PACKED ``[1, T, H, D]`` (already BSHD —
    no transpose), cu_q/cu_k are ``[B+1]`` cumulative seqlens.  Rides the same
    TemplateParams-specialized module as the dense path; the packed token
    extents and strides are Int64 runtime arguments, so new packed capacities
    re-bind the same pointer artifact without a tensor wrapper or new compile.
    Returns packed
    ``[1, T_q, H, D_v]`` O + packed ``[1, H, T_q]`` LSE."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import config_sm80 as _sm80_cfg

    if bias_tensor is not None:
        raise NotImplementedError("SM80 SDPA THD does not support bias (varlen has no single [1,H,SQ,SKV] bias shape)")
    if q.dtype not in (torch.float16, torch.bfloat16) or q.device.type != "cuda":
        raise ValueError("SM80 THD requires CUDA FP16/BF16 inputs")
    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.dtype != q.dtype or tensor.device != q.device:
            raise ValueError(f"SM80 THD {name} must be [1,T,H,D] on Q's device and dtype")
    if k.shape[2] < 1 or q.shape[2] < 1 or q.shape[2] % k.shape[2] or k.shape[1:3] != v.shape[1:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("SM80 THD requires matching K/V token and head counts, Q/K widths, and integral GQA")
    if int(max_s_q) < 1:
        raise ValueError("SM80 THD max_s_q must be positive")
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    h_kv = k.shape[2]
    device = q.device
    flavor = _sm80_pick_flavor(d_qk, d_v)
    fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
    tile_m, num_warps, tile_n = _SM80_FLAVOR_KNOBS[flavor]
    if scale_softmax is None:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    pad_v = d_v < fdv
    if d_qk < fdqk or pad_v:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.packed_copy_sm80 import copy_packed_half

        q, k, v = copy_packed_half((q, k, v), (fdqk, fdqk, fdv), (True, False, False))
    wl, wr = window_size
    right_bound = wr if (is_causal and wr is not None and wr > 0) else 0

    n_seqs = int(cu_q.numel()) - 1
    if n_seqs < 1:
        raise ValueError("cu_seqlens_q must have >= 2 entries")
    if cu_k is None or cu_k.numel() != n_seqs + 1:
        raise ValueError("cu_seqlens_q / cu_seqlens_k length mismatch")

    def prefix(tensor):
        if tensor.device == device and tensor.ndim == 1 and tensor.dtype in (torch.int32, torch.int64):
            return tensor
        return tensor.to(dtype=torch.int32, device=device).contiguous().view(-1)

    cu_q_t, cu_k_t = prefix(cu_q), prefix(cu_k)
    sinks_b = sinks
    if sinks is not None:
        if sinks.numel() != h_q:
            raise ValueError("SM80 THD sinks must contain H_q elements")
        if sinks.device != device or sinks.ndim != 1 or sinks.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            sinks_b = sinks.to(dtype=torch.float32, device=device).reshape(h_q).contiguous()

    params = _sm80_cfg.TemplateParams(
        io_bf16=(q.dtype == torch.bfloat16),
        d_qk=fdqk,
        d_v=fdv,
        tile_m=tile_m,
        num_warps=num_warps,
        tile_n=tile_n,
        is_causal=bool(is_causal),
        has_swa=wl is not None and wl >= 0,
        causal_bottom_right=bool(causal_bottom_right),
        has_sink=sinks is not None,
        thd_varlen=True,
        has_lse=True,
        sink_natural=True,
        score_sign=_sm80_cfg.score_sign(scale_softmax),
    )
    mod = _sm80_load_kernel_module(flavor, params)
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_sm80_thd import build_launch, execute

    launch = build_launch(
        mod,
        int(device.index or 0),
        h_q,
        h_kv,
        n_seqs,
        int(max(0, wl)) if wl is not None else 0,
        (str(cu_q_t.dtype), str(cu_k_t.dtype)),
        str(sinks_b.dtype) if sinks_b is not None else "torch.float32",
    )

    t_q = q.shape[1]
    # The prepared host initializes both complete capacities before attention.
    o_buf = torch.empty(1, t_q, h_q, fdv, dtype=q.dtype, device=device)
    lse_buf = torch.empty(1, h_q, t_q, dtype=torch.float32, device=device)
    stream = current_stream if current_stream is not None else torch.cuda.current_stream(device).cuda_stream
    execute(
        launch,
        (q, k, v, o_buf, lse_buf, cu_q_t, cu_k_t, sinks_b),
        int(max_s_q),
        _sm80_cfg.kernel_scale(float(scale_softmax)),
        int(right_bound),
        int(stream),
    )
    if pad_v:
        (o_buf,) = copy_packed_half((o_buf,), (d_v,), (True,))
    return TupleDict(o_tensor=o_buf, lse_tensor=lse_buf)


_sm80_wrapper_cache: dict = {}


def sdpa_fwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    scheduler: str = "auto",
    current_stream: Optional[cuda.CUstream] = None,
    causal_bottom_right: bool = False,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    rope_freqs: Optional[torch.Tensor] = None,
) -> TupleDict:
    """SM80 (A100) SDPA forward.

    Returns ``TupleDict(o_tensor=..., lse_tensor=...)`` matching the DSL
    wrappers' contract.  Dense calls route through :class:`SdpaFwdDslSm80`;
    packed THD calls (``cum_seqlen_*``) ride the same template with dynamic
    token extents.  ALiBi, block_mask and the score-stat side outputs are not
    supported (use the graph API, which routes them to the cuDNN backend).

    THD cumulative tensors contain raw packed-row offsets, including a possible
    nonzero first offset. Callers must provide nonnegative, nondecreasing values
    bounded by the corresponding Q or K/V token capacity, representable as int32.
    ``max_s_q`` must bound every adjacent Q-offset difference. These device-value
    preconditions apply to each eager call and each graph replay; this wrapper
    validates tensor metadata without reading cumulative values back to the host.
    """
    # Rule 7 (python/cudnn/AGENTS.md): this entry reaches the kernel module on its
    # own, so decline by DSL version here instead of surfacing the DSL's own
    # TypeError/ModuleNotFoundError from the template load.
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

    _too_old = cutedsl_requirement_error("sdpa_fwd_wrapper_sm80")
    if _too_old is not None:
        raise NotImplementedError(_too_old)
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")
    if scale_output not in (None, 1.0):
        raise NotImplementedError(f"SM80 SDPA: scale_output != 1.0 is not supported yet (got {scale_output})")

    if cum_seqlen_q_tensor is not None:
        if max_s_q is None:
            raise ValueError("THD path requires max_s_q (host int) for the grid")
        if causal_bottom_right and not (is_causal or window_size[0] >= 0):
            raise ValueError("SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).")
        for label, present in (
            ("rope_freqs", rope_freqs is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
            ('scheduler != "auto"', scheduler not in (None, "auto")),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) path does not support {label}; the dense path serves it")
        # A missing/current stream handle does not switch CUDA devices.
        # Compile and launch on the operand device, then restore the caller.
        device_context = (
            nullcontext() if q_tensor.device.type != "cuda" or torch.cuda.current_device() == q_tensor.device.index else torch.cuda.device(q_tensor.device)
        )
        with device_context, _torch_stream_context(current_stream, q_tensor.device):
            return _sm80_thd_forward(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                max_s_q=max_s_q,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                bias_tensor=bias_tensor,
                sinks=sinks,
                current_stream=current_stream,
            )

    # Tag per-call outputs and scratch with the stream that consumes them so
    # the allocator cannot recycle a block while an explicit stream is pending.
    with stream_context(current_stream, q_tensor.device) if current_stream is not None else nullcontext():
        b, h_q, s_q, _ = q_tensor.shape
        d_v = v_tensor.shape[-1]
        o_tensor = torch.empty(
            (b, s_q, h_q, d_v),
            dtype=q_tensor.dtype,
            device=q_tensor.device,
        ).transpose(1, 2)
        lse_tensor = _allocate_lse_tensor(q_tensor)

        wl, wr = window_size
        if not is_causal and wr >= 0:
            raise NotImplementedError("SM80 SDPA: window_size_right without is_causal=True has no effect; pass is_causal=True or a left window")
        rope_max_s = int(rope_freqs.shape[0]) if rope_freqs is not None else 0
        cache_key = (
            q_tensor.shape,
            k_tensor.shape,
            v_tensor.shape,
            q_tensor.stride(),
            k_tensor.stride(),
            v_tensor.stride(),
            q_tensor.dtype,
            bool(is_causal),
            (wl, wr),
            scale_softmax,
            scheduler,
            bool(causal_bottom_right),
            seq_kv_lens is not None,
            seq_len_q is not None,
            sinks is not None,
            bias_tensor is not None,
            (bias_tensor.dtype if bias_tensor is not None else None),
            rope_max_s,
            q_tensor.device,
        )
        api = _sm80_wrapper_cache.get(cache_key)
        if api is None:
            api = SdpaFwdDslSm80(
                sample_q=q_tensor,
                sample_k=k_tensor,
                sample_v=v_tensor,
                sample_o=o_tensor,
                sample_lse=lse_tensor,
                is_causal=is_causal,
                causal_bottom_right=causal_bottom_right,
                window_size_left=(wl if wl >= 0 else None),
                window_size_right=(wr if (is_causal and wr >= 0) else None),
                scale_softmax=scale_softmax,
                seq_kv_lens_present=seq_kv_lens is not None,
                seq_q_lens_present=seq_len_q is not None,
                has_sink=sinks is not None,
                scheduler=scheduler,
                bias_present=bias_tensor is not None,
                bias_fp32=(bias_tensor is not None and bias_tensor.dtype == torch.float32),
                rope_max_s=rope_max_s,
            )
            api.check_support()
            api.compile()
            _sm80_wrapper_cache[cache_key] = api
        required = api.scratch_workspace_bytes()
        workspace = torch.empty(required, dtype=torch.uint8, device=q_tensor.device) if required else None
        api.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            sinks=sinks,
            seq_q_lens=seq_len_q,
            seq_kv_lens=seq_kv_lens,
            scale_softmax=scale_softmax,
            current_stream=current_stream,
            bias_tensor=bias_tensor,
            rope_freqs=rope_freqs,
            workspace=workspace,
        )
        return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
