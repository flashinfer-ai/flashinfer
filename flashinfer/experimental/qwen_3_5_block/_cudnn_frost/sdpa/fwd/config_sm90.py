# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-facing configuration for the FROST SM90 D512 SDPA prefill template.

The frozen ``TemplateParams`` record contains only graph declarations and
selected tuning knobs. Runtime lengths, pointers, and packed token totals stay
out of compile keys. Graph-adapter declaration logic lives with the adapter;
this module keeps the small interface consumed by the kernel.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

SCALE_POSITIVE = 1
SCALE_ZERO = 0
SCALE_NEGATIVE = -1


# The D envelope. Every WGMMA descriptor and SMEM slab is 512 columns wide, so any
# head dim up to that tile is computed on it: the GMEM descriptors carry the actual
# extent, loads past it zero-fill and O stores past it clip. 8 is the TMA 16-byte
# global-stride rule at 2 bytes per element. A small D pays the full D512 tile cost;
# this is envelope coverage, not a native small-D flavor.
D_TILE = 512
D_ALIGN = 8
# Fixed SM90 D512 instruction geometry, matching the FFPA donor. QK is
# m64n64k512 and each PV half is m64n256k64; these are not tuning axes.
TILE_M = 64
TILE_N = 64


def head_dims_mismatch(d_qk, d_v) -> str | None:
    """Why ``(d_qk, d_v)`` is outside the D envelope, or None; D_QK != D_V is served."""
    for value in (d_qk, d_v):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 < value <= D_TILE or value % D_ALIGN:
            return (
                f"SM90 SDPA serves positive D_QK/D_V multiples of {D_ALIGN} up to its {D_TILE}-column tile "
                f"(TMA 16-byte global-stride rule); graph has D_QK={d_qk}/D_V={d_v}"
            )
    return None


def dense_bind_strides(shape: tuple, strides: tuple, elem_bytes: int) -> tuple | None:
    """SM90 binds every covering dense permutation with TMA-aligned live strides."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok

    if not dense_layout_ok(shape, strides):
        return None
    bs, hs, ss = (int(shape[3]) if size == 1 else int(stride) for size, stride in zip(shape[:3], strides))
    if any(stride * elem_bytes % 16 for stride in (bs, hs, ss)):
        return None
    return bs, ss, hs


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph choices that change the traced SM90 kernel specialization."""

    dtype_qkv: int = DTYPE_FP16
    causal: bool = False  # Top-left on dense unpadded S_q == S_kv is spelled bottom-right by the adapter.
    thd_varlen: bool = False
    has_lse: bool = False
    pack_gqa: bool = False
    qh_per_kh: int = 1
    sched_policy: int = SCHED_NATURAL
    scale_mode: int = SCALE_POSITIVE
    # Dense (B,) length reads; THD sets seq_kv_lens_present, its metadata operand.
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    # Diagonal band, appended so every earlier specialization keeps its defaults.
    # `window_left` is the sliding-window offset W (cuDNN left bound - 1): key k is visible
    # from row q when k >= q + delta - W. `window_right` is a widened right bound R > 0
    # (k <= q + delta + R); plain causal stays `causal=True`. `bottom_right=None` is the
    # original alignment, bottom-right exactly when causal; SdpaFwdDslSm90.check_support canonicalizes it.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool | None = None
    # Per-Q-head FP32 sink logits: an extra bound operand in these specializations only.
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the Stats store
    # scales the natural-log LSE by log2(e).
    stats_log2: bool = False


# A band bound at or beyond S_q + S_kv never masks a key inside the declared envelopes, and
# every band bound is added to Int32 token coordinates in the kernel.
BAND_COORDINATE_LIMIT = 2**30


def validate_params(params: TemplateParams) -> None:
    """Validate the immutable specialization before constructing a kernel."""
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError("SM90 D512 requires uniform FP16 or BF16 Q/K/V/O")
    if params.qh_per_kh < 1 or (params.pack_gqa and TILE_M % params.qh_per_kh):
        raise ValueError("SM90 D512 PackGQA requires a positive head ratio dividing tile_m=64")
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError("SM90 D512 supports only NATURAL/LPT/LPT_L2 single-tile scheduling")
    if params.thd_varlen and params.sched_policy != SCHED_NATURAL:
        raise ValueError("SM90 D512 THD uses the natural single-tile work decoder")
    if params.scale_mode not in (SCALE_POSITIVE, SCALE_ZERO, SCALE_NEGATIVE):
        raise ValueError("SM90 D512 scale mode must be positive, zero, or negative")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError("SM90 D512 thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError("SM90 D512 seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
    if params.causal and params.window_right is not None:
        raise ValueError("SM90 D512 plain causal is causal=True; window_right holds only a widened right bound")
    for name, bound, low in (("window_left", params.window_left, 0), ("window_right", params.window_right, 1)):
        if bound is not None and (type(bound) is not int or not low <= bound < BAND_COORDINATE_LIMIT):
            raise ValueError(f"SM90 D512 {name} must be an integer in [{low}, 2**30)")
    if params.bottom_right is not None:
        if params.window_left is None and params.window_right is None and not params.causal:
            raise ValueError("SM90 D512 diagonal alignment needs a band bound")
        if params.bottom_right == params.causal:
            raise ValueError("SM90 D512 bottom_right repeats the alignment causal implies; the canonical spelling is None")
