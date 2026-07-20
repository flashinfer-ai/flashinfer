# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/kernel.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Launch / dispatch entrypoints for the active SM120 sparse-MLA backend.

The ``run_unified_decode`` entrypoint builds views, plans split-K chunk ranges,
launches the warp-specialized decode grid, and runs the shared base-2 split merge
to produce final output. DSV4 compressed MLA and GLM_NSA use the same promoted
SM120 runtime; the retired pre-unification kernels have been removed.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack

from flashinfer.experimental.sm12x.attention._shared.cute.ops import LOG2_E
from flashinfer.experimental.sm12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    key_field,
    tensor_key,
)
from flashinfer.experimental.sm12x._lib.compiler import (
    launch as sm12x_launch,
)
from flashinfer.experimental.sm12x._lib.intrinsics import shared_ptr_to_u32

from .decode_math import (
    s0_load_q_bf16_to_smem,
    s0_quantize_q_to_smem,
    s1_qk_nope_block_scaled,
    s1_qk_nope_block_scaled_dsv4_h8_swap_ab,
    s1_qk_nope_block_scaled_glm_h8_swap_ab,
    s2_qk_rope_bf16,
    s2_qk_rope_bf16_glm_h8_swap_ab,
    s3_mask_and_scale,
    s3_mask_and_scale_glm_h8_swap_ab,
    s4_online_softmax,
    s4_online_softmax_glm_h8_swap_ab,
    s5_fill_sm_p_full,
    s6_xv_nope,
    s6_xv_nope_dsv4_h8_swap_ab,
    s6_xv_nope_glm_h8_swap_ab,
    s6b_xv_rope,
    s6b_xv_rope_h8_swap_ab,
    s7_epilogue,
)
from .io import io_issue_gather
from .smem import get_unified_shared_storage_cls, make_smem_layout
from .traits import (
    ModelType,
    ScaleFormat,
    infer_model_type,
    make_unified_traits,
)

# natural-log of 2 (base2 <-> natural LSE conversion).
_LN2 = math.log(2.0)

# BI=64 candidates per chunk (one full/empty KV buffer window). The split-K
# planner cuts the topk into chunk-aligned ranges so split partials are disjoint
# and the merge is exact (split boundary == chunk boundary).
_CAND_WINDOW = 64

# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512
# GLM_NSA uncompressed contract head dim (q_nope 512 + q_rope 64).
_GLM_HEAD_DIM = 576
# GLM per-token packed cache record (reference.pack_mla_kv_cache_reference).
_GLM_KV_GMEM_STRIDE = 656
# DSV4 H8 packs the contiguous 576-byte data record into a 592-byte smem row.
# The 16-byte pad preserves KV_SMEM_STRIDE/4 % 32 == 20, matching the generic
# 464-byte row's bank rotation while allowing one bulk copy per candidate.
_DSV4_PACKED_SMEM_STRIDE = 592
_DSV4_PACKED_ROPE_OFFSET = 448
# Largest DSV4 trace regime validated for the H8 specialization: 2 SWA chunks
# plus 128 indexed chunks.  Keep this shape-only so graph capture and replay
# select identical launch geometry without inspecting runtime tensor values.
_DSV4_H8_MAX_CHUNKS = 130


@dsl_user_op
def _exit_thread(
    *,
    loc=None,
    ip=None,
):
    """Terminate a thread after a CTA-uniform empty-split decision."""
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# Optional decode num_splits override (P9b AutoTuner sweep hook). ``<= 0`` (or an
# unparseable value, or unset) means "use the FlashInfer-ported wave-balanced
# heuristic"; ``>= 1`` pins num_splits to that value (still clamped to
# [1, num_chunks] and to the workspace split capacity). Read PER CALL (not cached
# at import) so a sweep / AutoTuner can flip it between launches.
_MLA_SM120_NUM_SPLITS_ENV = "FLASHINFER_EXP_SM12X_MLA_SM120_NUM_SPLITS"

# Exact GLM TP8 decode specialization: four math warps, swapped QK A/B, and
# packed HIGH/LOW PV.  Default-on; the per-call escape hatch exists for numeric
# and performance A/B diagnostics without changing serving code.
_MLA_SM120_GLM_H8_NATIVE_ENV = "FLASHINFER_EXP_SM12X_MLA_SM120_GLM_H8_NATIVE"

# Native DSV4 H16 two-group decode (two 8-head H8 groups sharing one KV
# pipeline per CTA): halves h_blocks (CTA count + KV gather traffic) and runs
# 8 math warps (2/scheduler) so the two groups' serial chunk chains interleave.
# Opt-in while being validated: unset/0 -> off; 1/true/on/yes -> on.
_MLA_SM120_DSV4_H16_NATIVE_ENV = "FLASHINFER_EXP_SM12X_MLA_SM120_DSV4_H16_NATIVE"

# FlashInfer's decode-dsv4 chunks_per_block wave-balance cap
# (csrc/sparse_mla_sm120_decode_dsv4.cu:85). cpb candidates whose last-wave tail
# gap looks small but require more than this many integer waves are rejected.
_CEIL_WAVES_MAX = 3

# Observability: the most recent decode split plan (read by benchmarks / the
# P9c AutoTuner sweep to report num_splits_used). Pure side-channel -- does NOT
# affect numerics or the launch. Keys are informational.
LAST_DECODE_PLAN: dict = {}


def _env_num_splits_override() -> int:
    """Read ``FLASHINFER_EXP_SM12X_MLA_SM120_NUM_SPLITS`` per call. ``<= 0`` / unset / bad -> 0
    (heuristic). Mirrors FlashInfer's ``chunks_per_block_override <= 0 -> C++
    heuristic`` convention, but for OUR num_splits (== FlashInfer's active block
    count num_splits_eff)."""
    raw = os.environ.get(_MLA_SM120_NUM_SPLITS_ENV)
    if raw is None:
        return 0
    try:
        v = int(raw.strip())
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


def _env_glm_h8_native_enabled() -> bool:
    raw = os.environ.get(_MLA_SM120_GLM_H8_NATIVE_ENV)
    return raw is None or raw.strip().lower() not in {"0", "false", "off", "no"}


def _env_dsv4_h16_native_mode() -> bool | None:
    """Tri-state H16 override: None = auto policy, False = never, True = always."""
    raw = os.environ.get(_MLA_SM120_DSV4_H16_NATIVE_ENV)
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in {"1", "true", "on", "yes"}:
        return True
    if text in {"0", "false", "off", "no"}:
        return False
    return None


# Auto H8/H16 policy constants (measured on RTX PRO 6000 Blackwell, 188 SMs).
# H16 wins when the shape is KV-gather-bound (many chunks -> halving h_blocks
# halves gather traffic) or when the H8 grid would exceed ~1.4 waves of
# 1-CTA/SM blocks; the H8 CTA keeps a ~4us-lower fixed latency, so it wins the
# small-rows sub-wave regime.
_DSV4_H16_MIN_BW_CHUNKS = 24
_DSV4_H16_MIN_BW_ROWS = 2
_DSV4_H16_H8_CTA_LIMIT = 256


def _dsv4_h16_auto(
    *,
    rows: int,
    heads: int,
    num_chunks: int,
    h8_num_splits: int,
    sm_count: int | None = None,
) -> bool:
    h8_ctas = rows * (heads // 8) * max(1, int(h8_num_splits))
    # Spark reaches a full H8 launch wave much earlier than the 188-SM tuning
    # target. C128's 130-chunk gather is enough to amortize H16 even at B=1;
    # shorter gathers usually need B=8 before sharing each KV stage across 16
    # heads wins. Two short-gather wave cliffs are exceptions: C4/B7's H8 plan
    # launches more than two waves, while C1/B13-15 halves its launch grid with
    # H16. Both regimes are measured wins on the 48-SM Spark.
    if sm_count is not None and sm_count <= 64:
        if num_chunks >= 128:
            return True
        if rows >= 8 and num_chunks > 2:
            return True
        if heads == 32 and sm_count == 48:
            if num_chunks == 2 and 13 <= rows <= 15:
                return True
            if num_chunks == 10 and h8_ctas > 2 * sm_count:
                return True
    if num_chunks >= _DSV4_H16_MIN_BW_CHUNKS and rows >= _DSV4_H16_MIN_BW_ROWS:
        return True
    return h8_ctas > _DSV4_H16_H8_CTA_LIMIT


def _dsv4_spark_short_gather_num_splits(
    *,
    rows: int,
    heads: int,
    num_chunks: int,
    sm_count: int | None,
    native_dsv4_h16: bool,
) -> int | None:
    """Return measured Spark split overrides for DSV4 TP2 short gathers."""
    if not native_dsv4_h16 or heads != 32 or sm_count is None or sm_count != 48:
        return None
    if num_chunks == 2 and 13 <= rows <= 15:
        return 1
    if num_chunks == 10:
        return {7: 2, 13: 4, 14: 3}.get(rows)
    return None


def _wave_balanced_num_splits(
    *, num_chunks: int, per_token_head: int, sm_count: int
) -> int:
    """Replicate FlashInfer's decode-dsv4 occupancy decision in OUR chunk-range
    parameterization, returning OUR ``num_splits`` (== FlashInfer's active block
    count ``num_splits_eff``).

    FlashInfer splits MAXIMALLY: its ``num_splits`` == ``num_chunks`` (one block
    per KV chunk; sparse_mla_sm120.py:259-260,286), then a C++ heuristic
    (csrc/sparse_mla_sm120_decode_dsv4.cu:69-102) picks ``chunks_per_block`` to
    wave-balance the launch and computes ``num_splits_eff = ceil(num_splits /
    cpb)`` ACTIVE blocks. Our UnifiedDecodeKernel processes a contiguous
    chunk-RANGE per CTA, so OUR ``num_splits`` directly IS that active count: we
    port the cpb tail-gap search VERBATIM over ``num_chunks`` chunks, then return
    ``ceil(num_chunks / cpb*)``.

    Tail-gap formula (VERBATIM from the .cu):
        per_token_head = num_tokens * H_BLOCKS
        for cpb in 1..num_chunks:                  # FlashInfer: 1..num_splits
            eff    = ceil(num_chunks / cpb)
            active = per_token_head * eff
            ceil_w = ceil(active / sm_count)
            if ceil_w > CEIL_WAVES_MAX(=3): continue
            waves  = active / sm_count
            gap    = ceil_w - waves
            pick cpb minimizing gap (tie -> larger cpb, fewer launched blocks)
        num_splits = ceil(num_chunks / cpb*)       # == FlashInfer num_splits_eff
    """
    num_chunks = max(int(num_chunks), 1)
    per_token_head = max(int(per_token_head), 1)
    sm_count = max(int(sm_count), 1)

    chunks_per_block = 1
    best_gap = 2.0
    for cpb in range(1, num_chunks + 1):
        eff = (num_chunks + cpb - 1) // cpb
        active = per_token_head * eff
        ceil_w = (active + sm_count - 1) // sm_count
        if ceil_w > _CEIL_WAVES_MAX:
            continue
        waves = active / sm_count
        gap = ceil_w - waves
        if gap < best_gap - 1e-6 or (gap < best_gap + 1e-6 and cpb > chunks_per_block):
            best_gap = gap
            chunks_per_block = cpb
    # OUR num_splits == FlashInfer's num_splits_eff = ceil(num_splits / cpb*),
    # where FlashInfer's num_splits == num_chunks (maximal split).
    return (num_chunks + chunks_per_block - 1) // chunks_per_block


# ---------------------------------------------------------------------------
# Split-K planning. Reuse the compressed planner's chunk-count idiom but pin the
# per-split chunk granularity to the kernel's BI=64 window so split boundaries
# land on chunk boundaries (a candidate is processed by exactly one split ->
# multi-split is numerically identical to single-split).
# ---------------------------------------------------------------------------
def plan_unified_decode_splits(
    *,
    topk: int,
    max_chunks: int,
    forced_num_splits: int | None = None,
    num_tokens: int = 1,
    h_blocks: int = 1,
    sm_count: int | None = None,
    extra_topk: int = 0,
    preferred_num_splits: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(num_chunks, num_splits, chunks_per_split)``.

    ``num_chunks = ceil(topk / BI) + ceil(extra_topk / BI)`` is the number of
    BI=64 candidate windows spanning BOTH the main and the EXTRA cache sections
    (DSV4 dual-cache; ``extra_topk=0`` reduces to the single-cache main count).
    ``num_splits`` is chosen by replicating FlashInfer's decode launch tuning:
    FlashInfer splits MAXIMALLY (one block per KV chunk) then wave-balances via a
    chunks_per_block heuristic to an ACTIVE block count
    ``num_splits_eff = ceil(num_chunks / cpb*)``. Our CTA owns a chunk-RANGE, so
    OUR ``num_splits`` directly IS that active count -- we port the same
    CEIL_WAVES_MAX=3 tail-gap search (see ``_wave_balanced_num_splits``).

    Override precedence (highest first):
      1. ``forced_num_splits`` (explicit caller arg -- multi-split numeric checks).
      2. ``FLASHINFER_EXP_SM12X_MLA_SM120_NUM_SPLITS`` env (>=1 pins; <=0/unset -> heuristic).
      3. ``preferred_num_splits`` (shape-only hardware policy, when provided).
      4. The FlashInfer-ported wave-balanced heuristic (needs ``sm_count``;
         falls back to 1 if ``sm_count`` is unavailable).

    ``num_splits`` is clamped to ``[1, num_chunks]`` and to ``max_chunks`` (the
    workspace mid_out/mid_lse split capacity).
    """
    # ZERO-WIDTH MAIN (DSV4 dual-cache, all KV in the EXTRA cache): topk==0 means
    # NO main chunks -- the main section is elided and the decode attends ONLY the
    # extra chunks. Keep ceil(0/BI)==0 here (do NOT floor topk to 1, which would
    # fabricate a phantom main chunk that the num_main_chunks==0 dispatch would
    # then mis-route as an over-allocated extra chunk). The single-cache /
    # no-extra empty decode is still protected by the num_chunks>=1 floor below.
    topk = max(int(topk), 0)
    extra_topk = max(int(extra_topk), 0)
    # num_chunks spans main + extra sections (FlashInfer num_splits = ceil(topk/BI)
    # + ceil(extra_topk/BI)); the wave-balance heuristic then picks the active
    # block count over the COMBINED chunk count. Floored to >=1 so a fully-empty
    # decode (topk==0 and extra_topk==0) still launches one (masked) chunk.
    num_chunks = max(
        1,
        (topk + _CAND_WINDOW - 1) // _CAND_WINDOW
        + ((extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW),
    )

    if forced_num_splits is not None:
        num_splits = max(1, int(forced_num_splits))
    else:
        env_override = _env_num_splits_override()
        if env_override > 0:
            num_splits = env_override
        elif preferred_num_splits is not None:
            num_splits = max(1, int(preferred_num_splits))
        elif sm_count and sm_count > 0:
            num_splits = _wave_balanced_num_splits(
                num_chunks=num_chunks,
                per_token_head=max(1, int(num_tokens)) * max(1, int(h_blocks)),
                sm_count=int(sm_count),
            )
        else:
            num_splits = 1

    num_splits = min(num_splits, num_chunks, max(1, int(max_chunks)))
    chunks_per_split = (num_chunks + num_splits - 1) // num_splits
    return num_chunks, num_splits, chunks_per_split


class UnifiedDecodeKernel:
    """Warp-specialized sparse-MLA decode with split-K partial writeback.

    Grid = (num_tokens, H_BLOCKS, num_splits). Each CTA owns one query token, one
    HPB=16-head block, and one chunk-range slice (split). The generic path uses
    8 math warps plus one IO warp; native DSV4 H16 uses two IO warps, while the
    native GLM and DSV4 H8 paths use 4 math warps plus one IO warp. Math consumes
    the double-buffered KV gathered with
    cp.async.bulk and mbarriers (io.py), then runs S0-S6b over the split's chunks
    and S7 writes this split's NORMALIZED partial O + base-2 LSE into mid_out /
    mid_lse in the exact split.py merge convention.
    """

    def __init__(
        self,
        traits,
        layout,
        page_block_size,
        chunks_per_split,
        h_blocks,
        num_splits,
        num_heads,
        q_head_dim,
        topk,
        extra_topk,
        q_stride,
        swa_indices_stride0,
        extra_indices_stride0,
        mid_out_stride,
        mid_lse_stride,
        has_extra=False,
        pbs_extra=1,
        valid_hpb=None,
        head_block_offset=0,
        per_token_len=False,
        native_glm_h8=False,
        native_dsv4_h8=False,
        native_dsv4_h16=False,
    ):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.chunks_per_split = int(chunks_per_split)
        self.h_blocks = int(h_blocks)
        self.num_splits = int(num_splits)
        self.num_heads = int(num_heads)
        self.q_head_dim = int(q_head_dim)
        self.topk = int(topk)
        self.extra_topk = int(extra_topk)
        self.q_stride_row = int(q_stride[0])
        self.q_stride_head = int(q_stride[1])
        self.q_stride_dim = int(q_stride[2])
        self.swa_indices_stride_row = int(swa_indices_stride0)
        self.extra_indices_stride_row = int(extra_indices_stride0)
        self.mid_out_stride_row = int(mid_out_stride[0])
        self.mid_out_stride_head = int(mid_out_stride[1])
        self.mid_out_stride_split = int(mid_out_stride[2])
        self.mid_out_stride_dim = int(mid_out_stride[3])
        self.mid_lse_stride_row = int(mid_lse_stride[0])
        self.mid_lse_stride_head = int(mid_lse_stride[1])
        self.mid_lse_stride_split = int(mid_lse_stride[2])
        # DSV4 dual-cache (P7c). When False the extra-section code is const_expr-
        # elided -> no-extra DSV4 / GLM PTX byte-identical.
        self.has_extra = bool(has_extra)
        self.pbs_extra = int(pbs_extra)
        # P10b multi-token per-token topk_length. When False the section_len /
        # extra_section_len are uniform Int32 scalars (the byte-identical base /
        # uniform-length path; the per-token length tensors never enter the device
        # entry). When True a per-token int32 length tensor is read in-kernel at
        # t=blockIdx.x and clamped to [0, topk] -> the per-CTA section bound. The
        # launcher routes here ONLY for a genuinely-mixed-length multi-token batch
        # (a uniform batch collapses to the scalar path -> PTX byte-identical).
        self.per_token_len = bool(per_token_len)
        self.native_glm_h8 = bool(native_glm_h8)
        self.native_dsv4_h8 = bool(native_dsv4_h8)
        # Native H16: two independent 8-head H8 groups (4 math warps each)
        # sharing the CTA's packed KV stage. Grid keeps HPB=16 head blocks.
        self.native_dsv4_h16 = bool(native_dsv4_h16)
        self.native_h8 = self.native_glm_h8 or self.native_dsv4_h8
        if self.native_dsv4_h8 or self.native_dsv4_h16:
            packed_span = int(layout.kv_bufs) * int(
                traits.bi
            ) * _DSV4_PACKED_SMEM_STRIDE + int(layout.kv_bufs) * int(
                layout.kv_sc_buf_bytes
            )
            available_span = int(layout.mbar_off) - int(layout.kv_fp8_off)
            if packed_span > available_span:
                raise ValueError(
                    "DSV4 H8 packed KV stage exceeds its shared-memory alias: "
                    f"need {packed_span} bytes, have {available_span}"
                )
        # VALID_HPB (small-TP / non-multiple-of-16 head shards). Upstream
        # VALID_HPB=min(NUM_HEADS,HPB) (decode_dsv4_kernel.cuh:152): the kernel
        # computes a FULL HPB=16 tile with zero-Q padding then gates output/LSE
        # writes to valid_hpb rows. ``valid_hpb`` is a const_expr (s0/s4/s7 gate on
        # it). When valid_hpb == t.hpb (the FULL-block default) the s0/s4/s7 calls
        # pass the IDENTICAL constexpr value as the pre-P10 kernel, so the
        # full-block trace + PTX stay byte-identical. A REMAINDER block (heads not
        # a multiple of 16) is launched as a SEPARATE 1-block grid with
        # valid_hpb=remainder and head_block_offset shifting head_base to the
        # tail head range.
        self.valid_hpb = int(valid_hpb) if valid_hpb is not None else int(traits.hpb)
        # head_block_offset shifts head_base = (head_block + offset) * hpb so a
        # remainder-only 1-block grid writes the correct (tail) head range. When 0
        # (the full-block / base path) the const_expr branch is elided -> the
        # head_base computation is byte-identical to the pre-P10 kernel.
        self.head_block_offset = int(head_block_offset)
        self.math_threads = int(traits.math_threads)
        self.block_threads = 320 if self.native_dsv4_h16 else int(traits.block_threads)
        self.io_threads = self.block_threads - self.math_threads

    @cute.jit
    def __call__(
        self,
        q_all: cute.Tensor,  # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,  # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,  # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,  # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,  # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        latent_scale: Float32,
        section_len: Int32,  # MAIN per-row valid topk length
        stride_kv_block: Int64,  # MAIN per-block byte stride
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # SINGLE-CACHE entry: the deployed eight data args plus the runtime
        # latent_scale (and stream). The dispatcher selects this (func=kernel ->
        # __call__) when has_extra=False; extra-section args never enter it.
        self.kernel(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            section_len,
            stride_kv_block,
        ).launch(
            grid=(num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            # The unified DSV4/GLM decode layouts consume 93--101 KiB of
            # shared memory, so only one CTA can reside on an SM120 SM.  Make
            # that existing resource contract visible to ptxas so it need not
            # target artificial multi-CTA register occupancy.
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def call_extra(
        self,
        q_all: cute.Tensor,  # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,  # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,  # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,  # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,  # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        latent_scale: Float32,
        section_len: Int32,  # MAIN per-row valid topk length
        stride_kv_block: Int64,  # MAIN per-block byte stride
        extra_kv_cache_u8: cute.Tensor,  # flat u8 EXTRA cache (DSV4 dual-cache)
        extra_indices: cute.Tensor,  # (rows, extra_topk) int32
        extra_section_len: Int32,  # EXTRA per-row valid length
        num_main_chunks: Int32,  # ceil(main_len/BI); chunks >= this read the EXTRA cache
        stride_extra_kv_block: Int64,  # EXTRA per-block byte stride
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # DUAL-CACHE entry (DSV4 P7c): the dispatcher selects this (func=
        # kernel.call_extra) only when has_extra=True, so its DISTINCT mangled name
        # never collides with the single-cache __call__. It launches
        # the 14-param @cute.kernel (self.kernel_extra), which shares the body via
        # _kernel_body(has_extra=True).
        self.kernel_extra(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            section_len,
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            extra_section_len,
            num_main_chunks,
            stride_extra_kv_block,
        ).launch(
            grid=(num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def call_pertok(
        self,
        q_all: cute.Tensor,  # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,  # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,  # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,  # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,  # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        latent_scale: Float32,
        topk_length: cute.Tensor,  # (rows,) int32 per-token MAIN valid length
        stride_kv_block: Int64,  # MAIN per-block byte stride
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # SINGLE-CACHE PER-TOKEN entry (P10b multi-token mixed-length): a DISTINCT
        # mangled name (the topk_length tensor replaces the section_len scalar) so
        # it never collides with the byte-identical uniform-length __call__. Each
        # CTA reads section_len = clamp(topk_length[blockIdx.x], 0, topk) -> the
        # per-token section bound; over-allocated chunks for short tokens are fully
        # masked (idx<0 + section bound) so their partials are -inf and the merge
        # ignores them.
        self.kernel_pertok(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            topk_length,
            stride_kv_block,
        ).launch(
            grid=(num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def call_extra_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        topk_length: cute.Tensor,  # (rows,) int32 per-token MAIN valid length
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,  # (rows,) int32 per-token EXTRA valid length
        num_main_chunks: Int32,  # ceil(MAX main_len/BI); chunks >= this read EXTRA
        stride_extra_kv_block: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # DUAL-CACHE PER-TOKEN entry (P10b): both the MAIN and EXTRA section lengths
        # are read per-token (topk_length / extra_topk_length at t=blockIdx.x). The
        # main/extra chunk split (num_main_chunks) stays UNIFORM (workspace geometry
        # over the MAX topk); per-token clamping zeroes the over-allocated chunks.
        self.kernel_extra_pertok(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            topk_length,
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            extra_topk_length,
            num_main_chunks,
            stride_extra_kv_block,
        ).launch(
            grid=(num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
    ):
        t = self.traits
        L = self.layout
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)

        token_idx, head_block, split_idx = cute.arch.block_idx()
        token_idx = Int32(token_idx)
        head_block = Int32(head_block)
        split_idx = Int32(split_idx)
        # REMAINDER-block grids shift head_base to the tail head range; the offset
        # const_expr is elided (==0) for the full-block / base path -> byte-identical.
        if cutlass.const_expr(self.head_block_offset != 0):
            head_block = head_block + Int32(self.head_block_offset)
        head_base = head_block * Int32(8 if self.native_h8 else t.hpb)

        # The launch grid is capacity-based and therefore fixed across CUDA-graph
        # replay.  Compact this split to the chunks that are valid for the current
        # row, and retire a wholly empty CTA before it allocates/initializes the KV
        # pipeline.  The merge treats LSE=-inf as a neutral partial and does not
        # read the corresponding (potentially stale) mid_out row.
        cps = Int32(self.chunks_per_split)
        split_first_chunk = split_idx * cps
        split_last_chunk = split_first_chunk + cps
        main_valid_chunks = (section_len + Int32(_CAND_WINDOW - 1)) // Int32(
            _CAND_WINDOW
        )
        if main_valid_chunks < Int32(0):
            main_valid_chunks = Int32(0)
        max_main_chunks = Int32((self.topk + _CAND_WINDOW - 1) // _CAND_WINDOW)
        if main_valid_chunks > max_main_chunks:
            main_valid_chunks = max_main_chunks
        main_chunk_end = split_last_chunk
        if main_chunk_end > main_valid_chunks:
            main_chunk_end = main_valid_chunks
        active_chunks = main_chunk_end - split_first_chunk
        if active_chunks < Int32(0):
            active_chunks = Int32(0)
        if active_chunks == Int32(0):
            if tid < Int32(self.valid_hpb):
                mid_lse[token_idx, head_base + tid, split_idx] = Float32(-Float32.inf)
            _exit_thread()

        smem = cutlass_utils.SmemAllocator()
        SharedStorage = get_unified_shared_storage_cls(t)
        st = smem.allocate(SharedStorage)

        q_fp8_addr = shared_ptr_to_u32(st.q_fp8.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        kv_fp8_addr = shared_ptr_to_u32(st.kv_fp8.data_ptr())
        if cutlass.const_expr(t.has_extra_cache):
            kv_sc_addr = shared_ptr_to_u32(st.kv_sc.data_ptr())
        else:
            kv_sc_addr = Int32(0)
        kv_rope_addr = shared_ptr_to_u32(st.kv_rope.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        reduce_max_addr = reduce_addr + Int32(L.reduce_warp_max_off - L.reduce_off)
        reduce_sum_addr = reduce_addr + Int32(L.reduce_warp_sum_off - L.reduce_off)
        w_fp8_addr = shared_ptr_to_u32(st.w_fp8.data_ptr())
        sm_p_full_addr = shared_ptr_to_u32(st.sm_p_full.data_ptr())

        q_sc_view = st.q_sc.get_tensor(cute.make_layout(int(L.q_sc_bytes // 4)))
        amax_view = st.reduce.get_tensor(cute.make_layout(int(L.reduce_bytes // 4)))
        token_idx_view = st.token_idx.get_tensor(
            cute.make_layout(int(L.token_idx_buf_bytes * L.token_idx_bufs // 4))
        )
        w_head_sc_view = st.w_head_sc.get_tensor(
            cute.make_layout(int(L.w_head_sc_bytes // 4))
        )

        # Math warps are consumers; the final warp is the IO producer.
        is_io = warp_id >= Int32(self.math_threads // 32)

        # Native H8 stages one contiguous data record per candidate. GLM keeps
        # its 656-byte source layout; DSV4 pads its 576-byte source record to a
        # bank-friendly 592-byte shared row and puts the two 512-byte scale
        # footers immediately after the data buffers. Both alias the existing
        # kv_fp8/kv_sc/kv_rope allocation, so graph-time workspace is unchanged.
        staged_kv_stride = t.kv_smem_stride
        if cutlass.const_expr(self.native_glm_h8):
            staged_kv_stride = _GLM_KV_GMEM_STRIDE
        if cutlass.const_expr(self.native_dsv4_h8):
            staged_kv_stride = _DSV4_PACKED_SMEM_STRIDE
        kv_fp8_buf = Int32(t.bi * staged_kv_stride)
        kv_rope_buf = Int32(L.kv_rope_buf_bytes)
        kv_sc_buf = Int32(L.kv_sc_buf_bytes)
        if cutlass.const_expr(self.native_dsv4_h8):
            kv_rope_addr = kv_fp8_addr + Int32(_DSV4_PACKED_ROPE_OFFSET)
            kv_rope_buf = kv_fp8_buf
            kv_sc_addr = kv_fp8_addr + Int32(2) * kv_fp8_buf
        tok_buf_elems = Int32(L.token_idx_buf_bytes // 4)

        # mbarrier array: full[0], full[1], empty[0], empty[1] (u64 each).
        mbar_base = st.mbar.data_ptr()
        n_buf = int(L.kv_bufs)

        full_arrivals = 2 if self.native_dsv4_h16 else 1
        if tid == Int32(0):
            for s in cutlass.range_constexpr(n_buf):
                cute.arch.mbarrier_init(mbar_base + s, Int32(full_arrivals))
                cute.arch.mbarrier_init(mbar_base + n_buf + s, Int32(1))  # empty[s]
        cute.arch.barrier()  # Full-CTA structural fence.

        # swa_indices for THIS token row: a 1-D (topk,) slice.
        topk_row = cute.make_tensor(
            swa_indices.iterator
            + token_idx.to(Int64) * Int64(self.swa_indices_stride_row),
            cute.make_layout(self.topk),
        )
        # q for THIS token row: a 2-D (heads, D_QK) view (s0 indexes [head_base+h, d]).
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout(
                (self.num_heads, self.q_head_dim),
                stride=(self.q_stride_head, self.q_stride_dim),
            ),
        )
        warp_first_cand = warp_id * Int32(8)
        if cutlass.const_expr(self.native_h8):
            warp_first_cand = warp_id * Int32(16)

        # ════════════════════════════════════════════════════════════════════
        # IO WARP (PRODUCER) vs MATH WARPS (CONSUMER).
        # ════════════════════════════════════════════════════════════════════
        if is_io:
            io_lane = tid - Int32(self.math_threads)
            prod_phase = Int32(1)
            prod_idx = Int32(0)
            for lc in cutlass.range(active_chunks, unroll=1):
                ci = split_first_chunk + Int32(lc)
                buf = Int32(lc) & Int32(1)
                g_start = ci * Int32(_CAND_WINDOW)
                g_end = g_start + Int32(_CAND_WINDOW)
                if g_end > section_len:
                    g_end = section_len

                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)

                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                io_issue_gather(
                    kv_cache_u8,
                    topk_row,
                    kv_fp8_addr + buf * kv_fp8_buf,
                    kv_rope_addr + buf * kv_rope_buf,
                    kv_sc_addr + buf * kv_sc_buf,
                    tok_buf_view,
                    mbar_base + buf,  # full[buf]
                    g_start,
                    g_end,
                    Int32(self.page_block_size),
                    stride_kv_block,
                    io_lane,
                    bi=t.bi,
                    kv_smem_stride=staged_kv_stride,
                    rope_smem_stride=t.d_rope,
                    scale_bytes_per_token=8,
                    bulk_tx_bytes=t.bulk_tx_bytes,
                    scale_format=t.scale_format,
                    io_threads=self.io_threads,
                    split_mbar_arrival=self.native_dsv4_h16,
                    fp8_rope=t.fp8_rope,
                    packed_glm=self.native_glm_h8,
                    packed_dsv4=self.native_dsv4_h8,
                )
                prod_idx += Int32(1)
                if prod_idx == Int32(n_buf):
                    prod_idx = Int32(0)
                    prod_phase ^= Int32(1)

        else:
            # MATH WARPS (CONSUMER, warps 0-7 = 256 threads).
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)
            if cutlass.const_expr(t.scale_format == ScaleFormat.NVFP4_E4M3):
                s0_load_q_bf16_to_smem(
                    q_token,
                    q_fp8_addr,
                    q_rope_addr,
                    head_base,
                    Int32(self.valid_hpb),
                    tid,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    hpb=t.hpb,
                    q_nope_bf16_stride=t.q_nope_stride,
                    q_rope_stride=L.q_rope_stride,
                    num_threads=self.math_threads,
                    barrier_id=2,
                )
            else:
                s0_quantize_q_to_smem(
                    q_token,
                    q_fp8_addr,
                    q_sc_view,
                    q_rope_addr,
                    amax_view,
                    head_base,
                    Int32(self.valid_hpb),
                    tid,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    d_qk=t.d_nope + t.d_rope,
                    quant_tile=t.quant_tile,
                    num_scales=t.num_scales,
                    hpb=(8 if self.native_h8 else t.hpb),
                    q_nope_stride=t.q_nope_stride,
                    q_rope_stride=L.q_rope_stride,
                    num_threads=self.math_threads,
                    barrier_id=2,
                    fused_subgroup_quant=self.native_dsv4_h8,
                    subgroup_amax=self.native_dsv4_h8,
                    vectorized_rope_copy=self.native_dsv4_h8,
                    packed_q_scale_words=self.native_dsv4_h16,
                )

            accn_frag = cute.make_rmem_tensor(n_acc_tiles * 4, Float32)
            rope_acc_elems = 8 if self.native_dsv4_h8 else 4
            accr_frag = cute.make_rmem_tensor(rope_acc_elems, Float32)
            gmax_frag = cute.make_rmem_tensor(2, Float32)
            gsum_frag = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(n_acc_tiles * 4):
                accn_frag[k] = Float32(0.0)
            for k in cutlass.range_constexpr(rope_acc_elems):
                accr_frag[k] = Float32(0.0)
            gmax_frag[0] = Float32(-1e30)
            gmax_frag[1] = Float32(-1e30)
            gsum_frag[0] = Float32(0.0)
            gsum_frag[1] = Float32(0.0)

            cons_phase = Int32(0)
            cons_idx = Int32(0)

            for lc in cutlass.range(active_chunks, unroll=1):
                ci = split_first_chunk + Int32(lc)
                split_cand_start = ci * Int32(_CAND_WINDOW)
                buf = Int32(lc) & Int32(1)

                kv_fp8_b = kv_fp8_addr + buf * kv_fp8_buf
                kv_rope_b = kv_rope_addr + buf * kv_rope_buf
                kv_sc_b = kv_sc_addr + buf * kv_sc_buf
                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )

                acc_nope = [
                    [
                        accn_frag[at * 4 + 0],
                        accn_frag[at * 4 + 1],
                        accn_frag[at * 4 + 2],
                        accn_frag[at * 4 + 3],
                    ]
                    for at in range(n_acc_tiles)
                ]
                acc_rope = [accr_frag[k] for k in range(rope_acc_elems)]
                global_max = [gmax_frag[0], gmax_frag[1]]
                global_sum = [gsum_frag[0], gsum_frag[1]]

                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)

                # P10f: GLM keeps RAW e4m3 K/V (no S0b dequant+requant -- that
                # discarded the per-group e4m3 mantissa headroom and cost ~0.003
                # cos). The arbitrary fp32 group scale is applied POST-MMA in S1
                # (QK) / inline in S6 (V). DSV4 (UE8M0_BYTE) is unaffected (it never
                # ran S0b: scale_format==0), so its trace/PTX stay byte-identical.

                qk = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                split_cand_end = split_cand_start + Int32(_CAND_WINDOW)
                if split_cand_end > section_len:
                    split_cand_end = section_len
                if cutlass.const_expr(self.native_h8):
                    if cutlass.const_expr(self.native_glm_h8):
                        qk = s1_qk_nope_block_scaled_glm_h8_swap_ab(
                            qk,
                            q_fp8_addr,
                            kv_fp8_b,
                            q_sc_view,
                            warp_first_cand,
                            lane,
                            num_scales=t.num_scales,
                            quant_tile=t.quant_tile,
                            q_nope_stride=t.q_nope_stride,
                            kv_smem_stride=staged_kv_stride,
                        )
                        h8_rope_addr = kv_fp8_b + Int32(t.kv_smem_stride)
                        h8_rope_stride = staged_kv_stride
                    else:
                        qk = s1_qk_nope_block_scaled_dsv4_h8_swap_ab(
                            qk,
                            q_fp8_addr,
                            kv_fp8_b,
                            q_sc_view,
                            kv_sc_b,
                            warp_first_cand,
                            lane,
                            num_scales=t.num_scales,
                            quant_tile=t.quant_tile,
                            q_nope_stride=t.q_nope_stride,
                            kv_smem_stride=staged_kv_stride,
                            scale_bytes_per_token=8,
                            packed_footer_words=self.native_dsv4_h16,
                            packed_q_scale_words=self.native_dsv4_h16,
                        )
                        h8_rope_addr = kv_rope_b
                        h8_rope_stride = staged_kv_stride
                    qk = s2_qk_rope_bf16_glm_h8_swap_ab(
                        qk,
                        q_rope_addr,
                        h8_rope_addr,
                        warp_first_cand,
                        lane,
                        d_rope=t.d_rope,
                        q_rope_stride=L.q_rope_stride,
                        kv_rope_stride_bytes=h8_rope_stride,
                    )
                    qk = s3_mask_and_scale_glm_h8_swap_ab(
                        qk,
                        tok_buf_view,
                        warp_first_cand,
                        split_cand_start,
                        split_cand_end,
                        section_len,
                        sm_scale_log2,
                        lane,
                    )
                    p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                    p, wr0, wr1 = s4_online_softmax_glm_h8_swap_ab(
                        qk,
                        p,
                        acc_nope,
                        acc_rope,
                        global_max,
                        global_sum,
                        reduce_max_addr,
                        reduce_sum_addr,
                        warp_id,
                        lane,
                        tid,
                        n_acc_tiles=n_acc_tiles,
                        hpb=t.hpb,
                        n_warps=4,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        rope_tiles_per_warp=(2 if self.native_dsv4_h8 else 0),
                    )
                    w_pre = [
                        p[0] * wr0,
                        p[1] * wr1,
                        p[2] * wr0,
                        p[3] * wr1,
                    ]
                    if cutlass.const_expr(self.native_glm_h8):
                        acc_nope = s6_xv_nope_glm_h8_swap_ab(
                            w_pre,
                            acc_nope,
                            kv_fp8_b,
                            w_head_sc_view,
                            w_fp8_addr,
                            warp_id,
                            lane,
                            tid,
                            n_v_chunks=t.n_v_chunks,
                            v_chunk=t.quant_tile,
                            hpb=t.hpb,
                            bi=t.bi,
                            kv_smem_stride=staged_kv_stride,
                            w_fp8_stride=t.bi + 16,
                            n_warps=4,
                            nt_per_warp_xv=t.nt_per_warp_xv,
                            num_threads=self.math_threads,
                            barrier_id=3,
                        )
                    else:
                        acc_nope = s6_xv_nope_dsv4_h8_swap_ab(
                            w_pre,
                            acc_nope,
                            kv_fp8_b,
                            kv_sc_b,
                            w_head_sc_view,
                            w_fp8_addr,
                            sm_p_full_addr,
                            warp_id,
                            lane,
                            tid,
                            n_v_chunks=t.n_v_chunks,
                            v_chunk=t.quant_tile,
                            hpb=t.hpb,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            kv_smem_stride=staged_kv_stride,
                            w_fp8_stride=t.bi + 16,
                            n_warps=4,
                            nt_per_warp_xv=t.nt_per_warp_xv,
                            scale_bytes_per_token=8,
                            num_threads=self.math_threads,
                            barrier_id=3,
                            packed_footer_words=self.native_dsv4_h16,
                        )
                else:
                    qk = s1_qk_nope_block_scaled(
                        qk,
                        q_fp8_addr,
                        kv_fp8_b,
                        q_sc_view,
                        kv_sc_b,
                        warp_first_cand,
                        lane,
                        latent_scale,
                        num_scales=t.num_scales,
                        quant_tile=t.quant_tile,
                        q_nope_stride=t.q_nope_stride,
                        kv_smem_stride=t.kv_smem_stride,
                        scale_bytes_per_token=8,
                        scale_format=t.scale_format,
                    )
                    qk = s2_qk_rope_bf16(
                        qk,
                        q_rope_addr,
                        kv_rope_b,
                        warp_first_cand,
                        lane,
                        d_rope=t.d_rope,
                        q_rope_stride=L.q_rope_stride,
                        fp8_rope=t.fp8_rope,
                    )
                    qk = s3_mask_and_scale(
                        qk,
                        tok_buf_view,
                        warp_first_cand,
                        split_cand_start,
                        split_cand_end,
                        section_len,
                        sm_scale_log2,
                        lane,
                    )
                    p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                    p, wr0, wr1 = s4_online_softmax(
                        qk,
                        p,
                        acc_nope,
                        acc_rope,
                        global_max,
                        global_sum,
                        reduce_max_addr,
                        reduce_sum_addr,
                        False,
                        warp_id,
                        lane,
                        tid,
                        n_v_chunks=t.n_v_chunks,
                        hpb=t.hpb,
                        n_warps=8,
                        valid_hpb=self.valid_hpb,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        n_acc_tiles=n_acc_tiles,
                    )
                    w_pre = [
                        p[0] * wr0,
                        p[1] * wr0,
                        p[2] * wr1,
                        p[3] * wr1,
                    ]
                    s5_fill_sm_p_full(
                        w_pre,
                        sm_p_full_addr,
                        w_head_sc_view,
                        warp_id,
                        lane,
                        tid,
                        bi=t.bi,
                        sm_p_stride=L.sm_p_full_stride,
                        n_v_chunks=t.n_v_chunks,
                        hpb=t.hpb,
                        num_threads=self.math_threads,
                        barrier_id=3,
                    )
                    cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)
                    acc_nope = s6_xv_nope(
                        w_pre,
                        acc_nope,
                        kv_fp8_b,
                        kv_sc_b,
                        w_head_sc_view,
                        w_fp8_addr,
                        warp_id,
                        lane,
                        tid,
                        latent_scale,
                        n_v_chunks=t.n_v_chunks,
                        v_chunk=t.quant_tile,
                        hpb=t.hpb,
                        bi=t.bi,
                        kv_smem_stride=t.kv_smem_stride,
                        w_fp8_stride=t.bi + 16,
                        n_warps=8,
                        scale_bytes_per_token=8,
                        nt_per_warp_xv=t.nt_per_warp_xv,
                        scale_format=t.scale_format,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        sm_p_full_addr=sm_p_full_addr,
                        sm_p_stride=L.sm_p_full_stride,
                    )

                # S6b (XV-RoPE) is DSV4-only (V_HAS_ROPE). const_expr-elided for GLM.
                if cutlass.const_expr(t.v_has_rope):
                    if cutlass.const_expr(self.native_dsv4_h8):
                        acc_rope = s6b_xv_rope_h8_swap_ab(
                            acc_rope,
                            sm_p_full_addr,
                            kv_rope_b,
                            warp_id,
                            lane,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            d_rope=t.d_rope,
                            n_warps=4,
                            tiles_per_warp=2,
                            kv_rope_stride_bytes=staged_kv_stride,
                        )
                    else:
                        acc_rope = s6b_xv_rope(
                            acc_rope,
                            sm_p_full_addr,
                            kv_rope_b,
                            warp_id,
                            lane,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            d_rope=t.d_rope,
                            n_warps=8,
                        )

                for at in cutlass.range_constexpr(n_acc_tiles):
                    accn_frag[at * 4 + 0] = acc_nope[at][0]
                    accn_frag[at * 4 + 1] = acc_nope[at][1]
                    accn_frag[at * 4 + 2] = acc_nope[at][2]
                    accn_frag[at * 4 + 3] = acc_nope[at][3]
                for k in cutlass.range_constexpr(rope_acc_elems):
                    accr_frag[k] = acc_rope[k]
                gmax_frag[0] = global_max[0]
                gmax_frag[1] = global_max[1]
                gsum_frag[0] = global_sum[0]
                gsum_frag[1] = global_sum[1]

                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + n_buf + cons_idx)
                cons_idx += Int32(1)
                if cons_idx == Int32(n_buf):
                    cons_idx = Int32(0)
                    cons_phase ^= Int32(1)

            # ── S7: write this split's NORMALIZED partial + base-2 LSE into
            #    mid_out[token, :, split, :] / mid_lse[token, :, split]. ──
            fin_acc_nope = [
                [
                    accn_frag[at * 4 + 0],
                    accn_frag[at * 4 + 1],
                    accn_frag[at * 4 + 2],
                    accn_frag[at * 4 + 3],
                ]
                for at in range(n_acc_tiles)
            ]
            fin_acc_rope = [accr_frag[k] for k in range(rope_acc_elems)]
            fin_gmax = [gmax_frag[0], gmax_frag[1]]
            fin_gsum = [gsum_frag[0], gsum_frag[1]]
            if cutlass.const_expr(self.native_h8):
                gid = lane >> Int32(2)
                pair_lane = gid >> Int32(1)
                row_gmax0 = cute.arch.shuffle_sync(gmax_frag[0], pair_lane)
                row_gmax1 = cute.arch.shuffle_sync(gmax_frag[1], pair_lane)
                row_gsum0 = cute.arch.shuffle_sync(gsum_frag[0], pair_lane)
                row_gsum1 = cute.arch.shuffle_sync(gsum_frag[1], pair_lane)
                fin_gmax[0] = row_gmax0
                fin_gsum[0] = row_gsum0
                if (gid & Int32(1)) != Int32(0):
                    fin_gmax[0] = row_gmax1
                    fin_gsum[0] = row_gsum1

            # mid_out[token, head_base + h, split, dim]: (HPB, D_V) view for this
            # (token, head_block, split). mid_out stride = (h*S*Dv, S*Dv, Dv, 1).
            out_o = cute.make_tensor(
                mid_out.iterator
                + token_idx.to(Int64) * Int64(self.mid_out_stride_row)
                + head_base.to(Int64) * Int64(self.mid_out_stride_head)
                + split_idx.to(Int64) * Int64(self.mid_out_stride_split),
                cute.make_layout(
                    (t.hpb, t.d_v),
                    stride=(self.mid_out_stride_head, self.mid_out_stride_dim),
                ),
            )
            # mid_lse[token, head_base + h, split]: (HPB,) view.
            out_lse = cute.make_tensor(
                mid_lse.iterator
                + token_idx.to(Int64) * Int64(self.mid_lse_stride_row)
                + head_base.to(Int64) * Int64(self.mid_lse_stride_head)
                + split_idx.to(Int64) * Int64(self.mid_lse_stride_split),
                cute.make_layout((t.hpb,), stride=(self.mid_lse_stride_head,)),
            )
            s7_epilogue(
                fin_acc_nope,
                fin_acc_rope,
                fin_gmax,
                fin_gsum,
                out_o,
                out_lse,
                warp_id,
                lane,
                n_v_chunks=t.n_v_chunks,
                v_chunk=t.quant_tile,
                d_nope=t.d_nope,
                d_rope=t.d_rope,
                n_warps=(4 if self.native_h8 else 8),
                valid_hpb=self.valid_hpb,
                nt_per_warp_xv=t.nt_per_warp_xv,
                v_has_rope=t.v_has_rope,
                rope_tiles_per_warp=(2 if self.native_dsv4_h8 else 1),
            )

    @cute.kernel
    def kernel_extra(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_section_len: Int32,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
    ):
        # DUAL-CACHE @cute.kernel: 14 device params; threads the real extra-section
        # args into the shared body (has_extra=True).
        self._kernel_body(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            section_len,
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            extra_section_len,
            num_main_chunks,
            stride_extra_kv_block,
            swa_indices,
            extra_indices,  # length tensors unused (per_token_len=False)
            has_extra=True,
            per_token_len=False,
        )

    @cute.kernel
    def kernel_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        topk_length: cute.Tensor,
        stride_kv_block: Int64,
    ):
        # SINGLE-CACHE PER-TOKEN @cute.kernel (P10b): the per-token MAIN length
        # tensor replaces the scalar section_len; the body reads
        # section_len = clamp(topk_length[blockIdx.x], 0, topk) per CTA. The scalar
        # section_len/extra_section_len/num_main_chunks args are dummies (elided by
        # per_token_len/has_extra const_expr); the extra tensor slots alias
        # swa_indices (never read when has_extra=False).
        self._kernel_body(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            Int32(0),
            stride_kv_block,
            kv_cache_u8,
            swa_indices,
            Int32(0),
            Int32(0),
            stride_kv_block,
            topk_length,
            swa_indices,
            has_extra=False,
            per_token_len=True,
        )

    @cute.kernel
    def kernel_extra_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        topk_length: cute.Tensor,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
    ):
        # DUAL-CACHE PER-TOKEN @cute.kernel (P10b): both MAIN and EXTRA section
        # lengths are read per-token (topk_length / extra_topk_length at
        # t=blockIdx.x). num_main_chunks (the uniform main/extra chunk split) stays
        # a scalar; per-token clamping masks the over-allocated chunks.
        self._kernel_body(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            latent_scale,
            Int32(0),
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            Int32(0),
            num_main_chunks,
            stride_extra_kv_block,
            topk_length,
            extra_topk_length,
            has_extra=True,
            per_token_len=True,
        )

    @cute.jit
    def _kernel_body(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        latent_scale: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_section_len: Int32,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
        topk_length: cute.Tensor,
        extra_topk_length: cute.Tensor,
        *,
        has_extra: cutlass.Constexpr,
        per_token_len: cutlass.Constexpr,
    ):
        t = self.traits
        L = self.layout
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)

        token_idx, head_block, split_idx = cute.arch.block_idx()
        token_idx = Int32(token_idx)
        head_block = Int32(head_block)
        split_idx = Int32(split_idx)
        # REMAINDER-block grids shift head_base to the tail head range; the offset
        # const_expr is elided (==0) for the full-block / base path -> byte-identical.
        if cutlass.const_expr(self.head_block_offset != 0):
            head_block = head_block + Int32(self.head_block_offset)
        head_base = head_block * Int32(8 if self.native_h8 else t.hpb)

        # Load and clamp replay-time lengths before touching the shared KV
        # pipeline. Capacity planning, launch geometry, and workspace addresses
        # remain fixed across CUDA-graph capture and replay.
        if cutlass.const_expr(per_token_len):
            topk_total = Int32(self.topk)
            section_len = Int32(topk_length[token_idx])
            if section_len < Int32(0):
                section_len = Int32(0)
            if section_len > topk_total:
                section_len = topk_total
            if cutlass.const_expr(has_extra):
                extra_total = Int32(self.extra_topk)
                extra_section_len = Int32(extra_topk_length[token_idx])
                if extra_section_len < Int32(0):
                    extra_section_len = Int32(0)
                if extra_section_len > extra_total:
                    extra_section_len = extra_total

        # Compact this split's intersection with the independently valid main
        # and extra chunk prefixes. This also handles a short-main gap before the
        # fixed extra-section boundary. Producer and consumer use the same compact
        # order, so their mbarrier phases remain matched.
        cps = Int32(self.chunks_per_split)
        split_first_chunk = split_idx * cps
        split_last_chunk = split_first_chunk + cps

        main_valid_chunks = (section_len + Int32(_CAND_WINDOW - 1)) // Int32(
            _CAND_WINDOW
        )
        if main_valid_chunks < Int32(0):
            main_valid_chunks = Int32(0)
        max_main_chunks = Int32((self.topk + _CAND_WINDOW - 1) // _CAND_WINDOW)
        if main_valid_chunks > max_main_chunks:
            main_valid_chunks = max_main_chunks
        main_chunk_end = split_last_chunk
        if main_chunk_end > main_valid_chunks:
            main_chunk_end = main_valid_chunks
        main_chunk_count = main_chunk_end - split_first_chunk
        if main_chunk_count < Int32(0):
            main_chunk_count = Int32(0)

        extra_first_chunk = split_first_chunk
        extra_chunk_count = Int32(0)
        if cutlass.const_expr(has_extra):
            extra_valid_chunks = (extra_section_len + Int32(_CAND_WINDOW - 1)) // Int32(
                _CAND_WINDOW
            )
            if extra_valid_chunks < Int32(0):
                extra_valid_chunks = Int32(0)
            max_extra_chunks = Int32(
                (self.extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW
            )
            if extra_valid_chunks > max_extra_chunks:
                extra_valid_chunks = max_extra_chunks
            if extra_first_chunk < num_main_chunks:
                extra_first_chunk = num_main_chunks
            extra_chunk_end = split_last_chunk
            extra_valid_end = num_main_chunks + extra_valid_chunks
            if extra_chunk_end > extra_valid_end:
                extra_chunk_end = extra_valid_end
            extra_chunk_count = extra_chunk_end - extra_first_chunk
            if extra_chunk_count < Int32(0):
                extra_chunk_count = Int32(0)

        active_chunks = main_chunk_count + extra_chunk_count
        if active_chunks == Int32(0):
            if tid < Int32(self.valid_hpb):
                mid_lse[token_idx, head_base + tid, split_idx] = Float32(-Float32.inf)
            _exit_thread()

        smem = cutlass_utils.SmemAllocator()
        SharedStorage = get_unified_shared_storage_cls(t)
        st = smem.allocate(SharedStorage)

        q_fp8_addr = shared_ptr_to_u32(st.q_fp8.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        kv_fp8_addr = shared_ptr_to_u32(st.kv_fp8.data_ptr())
        if cutlass.const_expr(t.has_extra_cache):
            kv_sc_addr = shared_ptr_to_u32(st.kv_sc.data_ptr())
        else:
            kv_sc_addr = Int32(0)
        kv_rope_addr = shared_ptr_to_u32(st.kv_rope.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        reduce_max_addr = reduce_addr + Int32(L.reduce_warp_max_off - L.reduce_off)
        reduce_sum_addr = reduce_addr + Int32(L.reduce_warp_sum_off - L.reduce_off)
        w_fp8_addr = shared_ptr_to_u32(st.w_fp8.data_ptr())
        sm_p_full_addr = shared_ptr_to_u32(st.sm_p_full.data_ptr())

        q_sc_view = st.q_sc.get_tensor(cute.make_layout(int(L.q_sc_bytes // 4)))
        amax_view = st.reduce.get_tensor(cute.make_layout(int(L.reduce_bytes // 4)))
        token_idx_view = st.token_idx.get_tensor(
            cute.make_layout(int(L.token_idx_buf_bytes * L.token_idx_bufs // 4))
        )
        w_head_sc_view = st.w_head_sc.get_tensor(
            cute.make_layout(int(L.w_head_sc_bytes // 4))
        )

        # Math warps are consumers; the final warp is the IO producer.
        is_io = warp_id >= Int32(self.math_threads // 32)

        # Match the single-cache body's allocation-preserving packed H8 layout.
        staged_kv_stride = t.kv_smem_stride
        if cutlass.const_expr(self.native_glm_h8):
            staged_kv_stride = _GLM_KV_GMEM_STRIDE
        if cutlass.const_expr(self.native_dsv4_h8 or self.native_dsv4_h16):
            staged_kv_stride = _DSV4_PACKED_SMEM_STRIDE
        kv_fp8_buf = Int32(t.bi * staged_kv_stride)
        kv_rope_buf = Int32(L.kv_rope_buf_bytes)
        kv_sc_buf = Int32(L.kv_sc_buf_bytes)
        if cutlass.const_expr(self.native_dsv4_h8 or self.native_dsv4_h16):
            kv_rope_addr = kv_fp8_addr + Int32(_DSV4_PACKED_ROPE_OFFSET)
            kv_rope_buf = kv_fp8_buf
            kv_sc_addr = kv_fp8_addr + Int32(2) * kv_fp8_buf
        tok_buf_elems = Int32(L.token_idx_buf_bytes // 4)

        # mbarrier array: full[0], full[1], empty[0], empty[1] (u64 each).
        mbar_base = st.mbar.data_ptr()
        n_buf = int(L.kv_bufs)

        full_arrivals = 2 if self.native_dsv4_h16 else 1
        if tid == Int32(0):
            for s in cutlass.range_constexpr(n_buf):
                cute.arch.mbarrier_init(mbar_base + s, Int32(full_arrivals))
                cute.arch.mbarrier_init(mbar_base + n_buf + s, Int32(1))  # empty[s]
        cute.arch.barrier()  # Full-CTA structural fence.

        # swa_indices for THIS token row: a 1-D (topk,) slice.
        # ZERO-WIDTH MAIN (DSV4 dual-cache, all KV in the EXTRA cache):
        # self.topk == 0 means there are NO main chunks (num_main_chunks
        # ==0) and the main section is never gathered/scanned. cute.make_layout(0)
        # is illegal (size must be strictly positive), so build a degenerate
        # 1-extent main view instead -- it is never referenced because every
        # ``ci < num_main_chunks(==0)`` branch is unreachable, so the kernel
        # attends ONLY the extra section. This is a const_expr (concrete-shape
        # trace), so the non-zero-main DSV4/GLM traces are byte-identical.
        if cutlass.const_expr(self.topk == 0):
            topk_row = cute.make_tensor(
                swa_indices.iterator
                + token_idx.to(Int64) * Int64(self.swa_indices_stride_row),
                cute.make_layout(1),
            )
        else:
            topk_row = cute.make_tensor(
                swa_indices.iterator
                + token_idx.to(Int64) * Int64(self.swa_indices_stride_row),
                cute.make_layout(self.topk),
            )
        # extra_indices for THIS token row (DSV4 dual-cache). Built ONLY when
        # has_extra (extra_indices is None otherwise); const_expr-elided so the
        # no-extra trace never references the extra tensor.
        if cutlass.const_expr(has_extra):
            extra_row = cute.make_tensor(
                extra_indices.iterator
                + token_idx.to(Int64) * Int64(self.extra_indices_stride_row),
                cute.make_layout(self.extra_topk),
            )
        else:
            extra_row = topk_row
        # q for THIS token row: a 2-D (heads, D_QK) view (s0 indexes [head_base+h, d]).
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout(
                (self.num_heads, self.q_head_dim),
                stride=(self.q_stride_head, self.q_stride_dim),
            ),
        )
        warp_first_cand = warp_id * Int32(8)
        if cutlass.const_expr(self.native_h8):
            warp_first_cand = warp_id * Int32(16)

        # Native H16 two-group mapping: math warps 0-3 are group 0 (heads
        # [head_base, head_base+8)), warps 4-7 group 1 (+8). Each group runs
        # the H8 math on its own staging half; the KV stage/mbarriers/token
        # buffer are shared. group/warp_sel/tid_sel collapse to the identity
        # for every other specialization (const_expr elided).
        group = Int32(0)
        warp_sel = warp_id
        tid_sel = tid
        if cutlass.const_expr(self.native_dsv4_h16):
            group = warp_id >> Int32(2)
            warp_sel = warp_id & Int32(3)
            tid_sel = tid - group * Int32(128)
            warp_first_cand = warp_sel * Int32(16)

        # ════════════════════════════════════════════════════════════════════
        # IO WARP (PRODUCER) vs MATH WARPS (CONSUMER).
        # ════════════════════════════════════════════════════════════════════
        if is_io:
            io_lane = tid - Int32(self.math_threads)
            prod_phase = Int32(1)
            prod_idx = Int32(0)
            for lc in cutlass.range(active_chunks, unroll=1):
                active_idx = Int32(lc)
                ci = split_first_chunk + active_idx
                if cutlass.const_expr(has_extra):
                    if active_idx >= main_chunk_count:
                        ci = extra_first_chunk + active_idx - main_chunk_count
                buf = Int32(lc) & Int32(1)

                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)

                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                io_kw = dict(
                    bi=t.bi,
                    kv_smem_stride=staged_kv_stride,
                    rope_smem_stride=t.d_rope,
                    scale_bytes_per_token=8,
                    bulk_tx_bytes=t.bulk_tx_bytes,
                    scale_format=t.scale_format,
                    io_threads=self.io_threads,
                    split_mbar_arrival=self.native_dsv4_h16,
                    fp8_rope=t.fp8_rope,
                    packed_glm=self.native_glm_h8,
                    packed_dsv4=self.native_dsv4_h8 or self.native_dsv4_h16,
                    overlap_footer_gather=self.native_dsv4_h16,
                )
                # Per-chunk section dispatch (DSV4 dual-cache; FlashInfer
                # decode_dsv4 :243-322). chunks [0, num_main_chunks) gather from the
                # MAIN cache; chunks >= num_main_chunks gather from the EXTRA cache
                # (different base ptr / page size / indices / stride). is_extra is
                # uniform across the IO warp (derived from the chunk index) so the
                # runtime branch is divergence-free. When has_extra=False this is
                # const_expr-pinned to the main gather -> byte-identical PTX.
                if cutlass.const_expr(has_extra):
                    if ci >= num_main_chunks:
                        cis = ci - num_main_chunks
                        g_start = cis * Int32(_CAND_WINDOW)
                        g_end = g_start + Int32(_CAND_WINDOW)
                        if g_end > extra_section_len:
                            g_end = extra_section_len
                        io_issue_gather(
                            extra_kv_cache_u8,
                            extra_row,
                            kv_fp8_addr + buf * kv_fp8_buf,
                            kv_rope_addr + buf * kv_rope_buf,
                            kv_sc_addr + buf * kv_sc_buf,
                            tok_buf_view,
                            mbar_base + buf,
                            g_start,
                            g_end,
                            Int32(self.pbs_extra),
                            stride_extra_kv_block,
                            io_lane,
                            **io_kw,
                        )
                    else:
                        g_start = ci * Int32(_CAND_WINDOW)
                        g_end = g_start + Int32(_CAND_WINDOW)
                        if g_end > section_len:
                            g_end = section_len
                        io_issue_gather(
                            kv_cache_u8,
                            topk_row,
                            kv_fp8_addr + buf * kv_fp8_buf,
                            kv_rope_addr + buf * kv_rope_buf,
                            kv_sc_addr + buf * kv_sc_buf,
                            tok_buf_view,
                            mbar_base + buf,
                            g_start,
                            g_end,
                            Int32(self.page_block_size),
                            stride_kv_block,
                            io_lane,
                            **io_kw,
                        )
                else:
                    g_start = ci * Int32(_CAND_WINDOW)
                    g_end = g_start + Int32(_CAND_WINDOW)
                    if g_end > section_len:
                        g_end = section_len
                    io_issue_gather(
                        kv_cache_u8,
                        topk_row,
                        kv_fp8_addr + buf * kv_fp8_buf,
                        kv_rope_addr + buf * kv_rope_buf,
                        kv_sc_addr + buf * kv_sc_buf,
                        tok_buf_view,
                        mbar_base + buf,
                        g_start,
                        g_end,
                        Int32(self.page_block_size),
                        stride_kv_block,
                        io_lane,
                        **io_kw,
                    )
                prod_idx += Int32(1)
                if prod_idx == Int32(n_buf):
                    prod_idx = Int32(0)
                    prod_phase ^= Int32(1)

        else:
            # MATH WARPS (CONSUMER, warps 0-7 = 256 threads).
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)

            # ── Native H16 per-group staging bases. Identity aliases when the
            #    two-group mode is off (const_expr; zero IR change). Each group
            #    strides its cooperative loops by nt_stage=128 threads while
            #    every named barrier spans the full bt_stage=256-thread math
            #    domain (barrier_threads override), so the groups share one
            #    barrier schedule but touch disjoint staging halves. ──
            q_fp8_stage = q_fp8_addr
            q_rope_stage = q_rope_addr
            q_sc_stage_view = q_sc_view
            amax_stage_view = amax_view
            reduce_max_stage = reduce_max_addr
            reduce_sum_stage = reduce_sum_addr
            w_fp8_stage = w_fp8_addr
            sm_p_stage = sm_p_full_addr
            w_head_sc_stage_view = w_head_sc_view
            head_base_stage = head_base
            nt_stage = 128 if self.native_dsv4_h16 else self.math_threads
            bt_stage = self.math_threads if self.native_dsv4_h16 else 0
            if cutlass.const_expr(self.native_dsv4_h16):
                head_base_stage = head_base + group * Int32(8)
                q_fp8_stage = q_fp8_addr + group * Int32(8 * t.q_nope_stride)
                q_rope_stage = q_rope_addr + group * Int32(8 * L.q_rope_stride * 2)
                q_sc_stage_view = cute.make_tensor(
                    q_sc_view.iterator + group * Int32(8 * t.num_scales),
                    cute.make_layout(int(8 * t.num_scales)),
                )
                amax_stage_view = cute.make_tensor(
                    amax_view.iterator + group * Int32(64),
                    cute.make_layout(64),
                )
                reduce_max_stage = reduce_max_addr + group * Int32(4 * t.hpb * 4)
                reduce_sum_stage = reduce_sum_addr + group * Int32(4 * t.hpb * 4)
                w_fp8_stage = w_fp8_addr + group * Int32(8 * (t.bi + 16))
                sm_p_stage = sm_p_full_addr + group * Int32(8 * L.sm_p_full_stride * 2)
                # Group 1 gets the dedicated tail region (w_head_sc packs
                # scale+reciprocal across its full 16-wide row, so the groups
                # cannot split one row).
                w_head_sc_stage_view = cute.make_tensor(
                    w_head_sc_view.iterator
                    + group * Int32((L.w_head_sc2_off - L.w_head_sc_off) // 4),
                    cute.make_layout(int(L.w_head_sc_bytes // 4)),
                )

            if cutlass.const_expr(t.scale_format == ScaleFormat.NVFP4_E4M3):
                s0_load_q_bf16_to_smem(
                    q_token,
                    q_fp8_stage,
                    q_rope_stage,
                    head_base_stage,
                    Int32(self.valid_hpb),
                    tid_sel,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    hpb=t.hpb,
                    q_nope_bf16_stride=t.q_nope_stride,
                    q_rope_stride=L.q_rope_stride,
                    num_threads=nt_stage,
                    barrier_id=2,
                    barrier_threads=bt_stage,
                )
            else:
                s0_quantize_q_to_smem(
                    q_token,
                    q_fp8_stage,
                    q_sc_stage_view,
                    q_rope_stage,
                    amax_stage_view,
                    head_base_stage,
                    Int32(8 if self.native_dsv4_h16 else self.valid_hpb),
                    tid_sel,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    d_qk=t.d_nope + t.d_rope,
                    quant_tile=t.quant_tile,
                    num_scales=t.num_scales,
                    hpb=(8 if (self.native_h8 or self.native_dsv4_h16) else t.hpb),
                    q_nope_stride=t.q_nope_stride,
                    q_rope_stride=L.q_rope_stride,
                    num_threads=nt_stage,
                    barrier_id=2,
                    barrier_threads=bt_stage,
                    fused_subgroup_quant=self.native_dsv4_h8,
                    subgroup_amax=(self.native_dsv4_h8 or self.native_dsv4_h16),
                    vectorized_rope_copy=(self.native_dsv4_h8 or self.native_dsv4_h16),
                    packed_q_scale_words=self.native_dsv4_h16,
                )

            accn_frag = cute.make_rmem_tensor(n_acc_tiles * 4, Float32)
            rope_acc_elems = 8 if (self.native_dsv4_h8 or self.native_dsv4_h16) else 4
            accr_frag = cute.make_rmem_tensor(rope_acc_elems, Float32)
            gmax_frag = cute.make_rmem_tensor(2, Float32)
            gsum_frag = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(n_acc_tiles * 4):
                accn_frag[k] = Float32(0.0)
            for k in cutlass.range_constexpr(rope_acc_elems):
                accr_frag[k] = Float32(0.0)
            gmax_frag[0] = Float32(-1e30)
            gmax_frag[1] = Float32(-1e30)
            gsum_frag[0] = Float32(0.0)
            gsum_frag[1] = Float32(0.0)

            cons_phase = Int32(0)
            cons_idx = Int32(0)

            for lc in cutlass.range(active_chunks, unroll=1):
                active_idx = Int32(lc)
                ci = split_first_chunk + active_idx
                if cutlass.const_expr(has_extra):
                    if active_idx >= main_chunk_count:
                        ci = extra_first_chunk + active_idx - main_chunk_count
                split_cand_start = ci * Int32(_CAND_WINDOW)
                buf = Int32(lc) & Int32(1)

                kv_fp8_b = kv_fp8_addr + buf * kv_fp8_buf
                kv_rope_b = kv_rope_addr + buf * kv_rope_buf
                kv_sc_b = kv_sc_addr + buf * kv_sc_buf
                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )

                acc_nope = [
                    [
                        accn_frag[at * 4 + 0],
                        accn_frag[at * 4 + 1],
                        accn_frag[at * 4 + 2],
                        accn_frag[at * 4 + 3],
                    ]
                    for at in range(n_acc_tiles)
                ]
                acc_rope = [accr_frag[k] for k in range(rope_acc_elems)]
                global_max = [gmax_frag[0], gmax_frag[1]]
                global_sum = [gsum_frag[0], gsum_frag[1]]

                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)

                # P10f: GLM keeps RAW e4m3 K/V (no S0b dequant+requant -- that
                # discarded the per-group e4m3 mantissa headroom and cost ~0.003
                # cos). The arbitrary fp32 group scale is applied POST-MMA in S1
                # (QK) / inline in S6 (V). DSV4 (UE8M0_BYTE) is unaffected (it never
                # ran S0b: scale_format==0), so its trace/PTX stay byte-identical.

                qk = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                if cutlass.const_expr(self.native_h8 or self.native_dsv4_h16):
                    if cutlass.const_expr(self.native_glm_h8):
                        qk = s1_qk_nope_block_scaled_glm_h8_swap_ab(
                            qk,
                            q_fp8_stage,
                            kv_fp8_b,
                            q_sc_stage_view,
                            warp_first_cand,
                            lane,
                            num_scales=t.num_scales,
                            quant_tile=t.quant_tile,
                            q_nope_stride=t.q_nope_stride,
                            kv_smem_stride=staged_kv_stride,
                        )
                        h8_rope_addr = kv_fp8_b + Int32(t.kv_smem_stride)
                        h8_rope_stride = staged_kv_stride
                    else:
                        qk = s1_qk_nope_block_scaled_dsv4_h8_swap_ab(
                            qk,
                            q_fp8_stage,
                            kv_fp8_b,
                            q_sc_stage_view,
                            kv_sc_b,
                            warp_first_cand,
                            lane,
                            num_scales=t.num_scales,
                            quant_tile=t.quant_tile,
                            q_nope_stride=t.q_nope_stride,
                            kv_smem_stride=staged_kv_stride,
                            scale_bytes_per_token=8,
                            packed_footer_words=self.native_dsv4_h16,
                            packed_q_scale_words=self.native_dsv4_h16,
                        )
                        h8_rope_addr = kv_rope_b
                        h8_rope_stride = staged_kv_stride
                    qk = s2_qk_rope_bf16_glm_h8_swap_ab(
                        qk,
                        q_rope_stage,
                        h8_rope_addr,
                        warp_first_cand,
                        lane,
                        d_rope=t.d_rope,
                        q_rope_stride=L.q_rope_stride,
                        kv_rope_stride_bytes=h8_rope_stride,
                    )
                    sc_start = split_cand_start
                    sec_len = section_len
                    if cutlass.const_expr(has_extra):
                        if ci >= num_main_chunks:
                            sc_start = (ci - num_main_chunks) * Int32(_CAND_WINDOW)
                            sec_len = extra_section_len
                    sc_end = sc_start + Int32(_CAND_WINDOW)
                    if sc_end > sec_len:
                        sc_end = sec_len
                    qk = s3_mask_and_scale_glm_h8_swap_ab(
                        qk,
                        tok_buf_view,
                        warp_first_cand,
                        sc_start,
                        sc_end,
                        sec_len,
                        sm_scale_log2,
                        lane,
                    )
                    p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                    p, wr0, wr1 = s4_online_softmax_glm_h8_swap_ab(
                        qk,
                        p,
                        acc_nope,
                        acc_rope,
                        global_max,
                        global_sum,
                        reduce_max_stage,
                        reduce_sum_stage,
                        warp_sel,
                        lane,
                        tid_sel,
                        n_acc_tiles=n_acc_tiles,
                        hpb=t.hpb,
                        n_warps=4,
                        num_threads=nt_stage,
                        barrier_id=3,
                        rope_tiles_per_warp=(
                            2 if (self.native_dsv4_h8 or self.native_dsv4_h16) else 0
                        ),
                        barrier_threads=bt_stage,
                    )
                    w_pre = [
                        p[0] * wr0,
                        p[1] * wr1,
                        p[2] * wr0,
                        p[3] * wr1,
                    ]
                    if cutlass.const_expr(self.native_glm_h8):
                        acc_nope = s6_xv_nope_glm_h8_swap_ab(
                            w_pre,
                            acc_nope,
                            kv_fp8_b,
                            w_head_sc_view,
                            w_fp8_addr,
                            warp_id,
                            lane,
                            tid,
                            n_v_chunks=t.n_v_chunks,
                            v_chunk=t.quant_tile,
                            hpb=t.hpb,
                            bi=t.bi,
                            kv_smem_stride=staged_kv_stride,
                            w_fp8_stride=t.bi + 16,
                            n_warps=4,
                            nt_per_warp_xv=t.nt_per_warp_xv,
                            num_threads=self.math_threads,
                            barrier_id=3,
                        )
                    else:
                        acc_nope = s6_xv_nope_dsv4_h8_swap_ab(
                            w_pre,
                            acc_nope,
                            kv_fp8_b,
                            kv_sc_b,
                            w_head_sc_stage_view,
                            w_fp8_stage,
                            sm_p_stage,
                            warp_sel,
                            lane,
                            tid_sel,
                            n_v_chunks=t.n_v_chunks,
                            v_chunk=t.quant_tile,
                            hpb=t.hpb,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            kv_smem_stride=staged_kv_stride,
                            w_fp8_stride=t.bi + 16,
                            n_warps=4,
                            nt_per_warp_xv=t.nt_per_warp_xv,
                            scale_bytes_per_token=8,
                            num_threads=nt_stage,
                            barrier_id=3,
                            barrier_threads=bt_stage,
                            packed_footer_words=self.native_dsv4_h16,
                        )
                else:
                    qk = s1_qk_nope_block_scaled(
                        qk,
                        q_fp8_addr,
                        kv_fp8_b,
                        q_sc_view,
                        kv_sc_b,
                        warp_first_cand,
                        lane,
                        latent_scale,
                        num_scales=t.num_scales,
                        quant_tile=t.quant_tile,
                        q_nope_stride=t.q_nope_stride,
                        kv_smem_stride=t.kv_smem_stride,
                        scale_bytes_per_token=8,
                        scale_format=t.scale_format,
                    )
                    qk = s2_qk_rope_bf16(
                        qk,
                        q_rope_addr,
                        kv_rope_b,
                        warp_first_cand,
                        lane,
                        d_rope=t.d_rope,
                        q_rope_stride=L.q_rope_stride,
                        fp8_rope=t.fp8_rope,
                    )

                    # Per-chunk section dispatch for the S3 mask: compare the
                    # candidate's offset within its main/extra section.
                    if cutlass.const_expr(has_extra):
                        sc_start = split_cand_start
                        sec_len = section_len
                        if ci >= num_main_chunks:
                            sc_start = (ci - num_main_chunks) * Int32(_CAND_WINDOW)
                            sec_len = extra_section_len
                        sc_end = sc_start + Int32(_CAND_WINDOW)
                        if sc_end > sec_len:
                            sc_end = sec_len
                        qk = s3_mask_and_scale(
                            qk,
                            tok_buf_view,
                            warp_first_cand,
                            sc_start,
                            sc_end,
                            sec_len,
                            sm_scale_log2,
                            lane,
                        )
                    else:
                        split_cand_end = split_cand_start + Int32(_CAND_WINDOW)
                        if split_cand_end > section_len:
                            split_cand_end = section_len
                        qk = s3_mask_and_scale(
                            qk,
                            tok_buf_view,
                            warp_first_cand,
                            split_cand_start,
                            split_cand_end,
                            section_len,
                            sm_scale_log2,
                            lane,
                        )

                    p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                    p, wr0, wr1 = s4_online_softmax(
                        qk,
                        p,
                        acc_nope,
                        acc_rope,
                        global_max,
                        global_sum,
                        reduce_max_addr,
                        reduce_sum_addr,
                        False,
                        warp_id,
                        lane,
                        tid,
                        n_v_chunks=t.n_v_chunks,
                        hpb=t.hpb,
                        n_warps=8,
                        valid_hpb=self.valid_hpb,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        n_acc_tiles=n_acc_tiles,
                    )
                    w_pre = [
                        p[0] * wr0,
                        p[1] * wr0,
                        p[2] * wr1,
                        p[3] * wr1,
                    ]
                    s5_fill_sm_p_full(
                        w_pre,
                        sm_p_full_addr,
                        w_head_sc_view,
                        warp_id,
                        lane,
                        tid,
                        bi=t.bi,
                        sm_p_stride=L.sm_p_full_stride,
                        n_v_chunks=t.n_v_chunks,
                        hpb=t.hpb,
                        num_threads=self.math_threads,
                        barrier_id=3,
                    )
                    cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)
                    acc_nope = s6_xv_nope(
                        w_pre,
                        acc_nope,
                        kv_fp8_b,
                        kv_sc_b,
                        w_head_sc_view,
                        w_fp8_addr,
                        warp_id,
                        lane,
                        tid,
                        latent_scale,
                        n_v_chunks=t.n_v_chunks,
                        v_chunk=t.quant_tile,
                        hpb=t.hpb,
                        bi=t.bi,
                        kv_smem_stride=t.kv_smem_stride,
                        w_fp8_stride=t.bi + 16,
                        n_warps=8,
                        scale_bytes_per_token=8,
                        nt_per_warp_xv=t.nt_per_warp_xv,
                        scale_format=t.scale_format,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        sm_p_full_addr=sm_p_full_addr,
                        sm_p_stride=L.sm_p_full_stride,
                    )

                # S6b (XV-RoPE) is DSV4-only (V_HAS_ROPE). const_expr-elided for GLM.
                if cutlass.const_expr(t.v_has_rope):
                    if cutlass.const_expr(self.native_dsv4_h8 or self.native_dsv4_h16):
                        acc_rope = s6b_xv_rope_h8_swap_ab(
                            acc_rope,
                            sm_p_stage,
                            kv_rope_b,
                            warp_sel,
                            lane,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            d_rope=t.d_rope,
                            n_warps=4,
                            tiles_per_warp=2,
                            kv_rope_stride_bytes=staged_kv_stride,
                        )
                    else:
                        acc_rope = s6b_xv_rope(
                            acc_rope,
                            sm_p_full_addr,
                            kv_rope_b,
                            warp_id,
                            lane,
                            bi=t.bi,
                            sm_p_stride=L.sm_p_full_stride,
                            d_rope=t.d_rope,
                            n_warps=8,
                        )

                for at in cutlass.range_constexpr(n_acc_tiles):
                    accn_frag[at * 4 + 0] = acc_nope[at][0]
                    accn_frag[at * 4 + 1] = acc_nope[at][1]
                    accn_frag[at * 4 + 2] = acc_nope[at][2]
                    accn_frag[at * 4 + 3] = acc_nope[at][3]
                for k in cutlass.range_constexpr(rope_acc_elems):
                    accr_frag[k] = acc_rope[k]
                gmax_frag[0] = global_max[0]
                gmax_frag[1] = global_max[1]
                gsum_frag[0] = global_sum[0]
                gsum_frag[1] = global_sum[1]

                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + n_buf + cons_idx)
                cons_idx += Int32(1)
                if cons_idx == Int32(n_buf):
                    cons_idx = Int32(0)
                    cons_phase ^= Int32(1)

            # ── S7: write this split's NORMALIZED partial + base-2 LSE into
            #    mid_out[token, :, split, :] / mid_lse[token, :, split]. ──
            fin_acc_nope = [
                [
                    accn_frag[at * 4 + 0],
                    accn_frag[at * 4 + 1],
                    accn_frag[at * 4 + 2],
                    accn_frag[at * 4 + 3],
                ]
                for at in range(n_acc_tiles)
            ]
            fin_acc_rope = [accr_frag[k] for k in range(rope_acc_elems)]
            fin_gmax = [gmax_frag[0], gmax_frag[1]]
            fin_gsum = [gsum_frag[0], gsum_frag[1]]
            if cutlass.const_expr(self.native_h8 or self.native_dsv4_h16):
                gid = lane >> Int32(2)
                pair_lane = gid >> Int32(1)
                row_gmax0 = cute.arch.shuffle_sync(gmax_frag[0], pair_lane)
                row_gmax1 = cute.arch.shuffle_sync(gmax_frag[1], pair_lane)
                row_gsum0 = cute.arch.shuffle_sync(gsum_frag[0], pair_lane)
                row_gsum1 = cute.arch.shuffle_sync(gsum_frag[1], pair_lane)
                fin_gmax[0] = row_gmax0
                fin_gsum[0] = row_gsum0
                if (gid & Int32(1)) != Int32(0):
                    fin_gmax[0] = row_gmax1
                    fin_gsum[0] = row_gsum1

            # mid_out[token, head_base + h, split, dim]: (HPB, D_V) view for this
            # (token, head_block, split). mid_out stride = (h*S*Dv, S*Dv, Dv, 1).
            out_o = cute.make_tensor(
                mid_out.iterator
                + token_idx.to(Int64) * Int64(self.mid_out_stride_row)
                + head_base_stage.to(Int64) * Int64(self.mid_out_stride_head)
                + split_idx.to(Int64) * Int64(self.mid_out_stride_split),
                cute.make_layout(
                    (t.hpb, t.d_v),
                    stride=(self.mid_out_stride_head, self.mid_out_stride_dim),
                ),
            )
            # mid_lse[token, head_base + h, split]: (HPB,) view.
            out_lse = cute.make_tensor(
                mid_lse.iterator
                + token_idx.to(Int64) * Int64(self.mid_lse_stride_row)
                + head_base_stage.to(Int64) * Int64(self.mid_lse_stride_head)
                + split_idx.to(Int64) * Int64(self.mid_lse_stride_split),
                cute.make_layout((t.hpb,), stride=(self.mid_lse_stride_head,)),
            )
            s7_epilogue(
                fin_acc_nope,
                fin_acc_rope,
                fin_gmax,
                fin_gsum,
                out_o,
                out_lse,
                warp_sel,
                lane,
                n_v_chunks=t.n_v_chunks,
                v_chunk=t.quant_tile,
                d_nope=t.d_nope,
                d_rope=t.d_rope,
                n_warps=(4 if (self.native_h8 or self.native_dsv4_h16) else 8),
                valid_hpb=(8 if self.native_dsv4_h16 else self.valid_hpb),
                nt_per_warp_xv=t.nt_per_warp_xv,
                v_has_rope=t.v_has_rope,
                rope_tiles_per_warp=(
                    2 if (self.native_dsv4_h8 or self.native_dsv4_h16) else 1
                ),
            )


def _to_cute(x, dtype, align=16, dynamic_layout=False):
    c = from_dlpack(x, assumed_align=align)
    c.element_type = dtype
    if dynamic_layout and x.ndim >= 1:
        leading_dim = next(
            (idx for idx, stride in enumerate(x.stride()) if stride == 1), None
        )
        if leading_dim is not None:
            c = c.mark_layout_dynamic(leading_dim=leading_dim)
    return c


def _cache_base_tensor(cache: torch.Tensor) -> torch.Tensor:
    # Contiguous cache can use the historical flat view. Packed vLLM cache views
    # are non-contiguous tensors whose storage_offset points at this layer's
    # payload inside a larger packed block. Do not reshape those: reshape would
    # materialize a contiguous copy and lose the packed layout.
    if cache.is_contiguous():
        return cache.reshape(-1)
    if cache.ndim < 2:
        return cache

    # Kernels compute raw byte offsets as block * stride_kv_block + in-page byte.
    # Give Cute a 1-D view over the physical byte span for this layer so
    # pointer+offset arithmetic stays raw-address based while preserving the
    # original storage_offset. The explicit stride_kv_block argument still
    # carries the packed block stride; this view only defines the base pointer.
    span = 1
    for size, stride in zip(cache.shape, cache.stride(), strict=True):
        span += (int(size) - 1) * int(stride)
    return torch.as_strided(cache, size=(span,), stride=(1,))


def _cache_block_stride_bytes(
    cache: torch.Tensor,
    *,
    page_size: int,
    model_type: int,
    record_bytes: int | None = None,
) -> int:
    from flashinfer.experimental.sm12x.attention._shared.mla.compressed_reference import (
        compressed_mla_page_nbytes,
    )

    if int(model_type) == int(ModelType.GLM_NSA):
        # GLM-family per-token contiguous record: 656B (ARBITRARY_FP32) or
        # 432B (NVFP4_E4M3). ``record_bytes`` comes from traits.kv_gmem_stride.
        rec = int(record_bytes) if record_bytes is not None else _GLM_KV_GMEM_STRIDE
        expected = int(page_size) * rec
    else:
        expected = int(compressed_mla_page_nbytes(int(page_size)))
    # Contiguous inputs are flattened before launch, so their original rank is
    # not a physical-layout contract and the standard page stride applies.
    # Packed vLLM page views are non-contiguous and carry the physical
    # per-block stride in dimension 0.
    if not cache.is_contiguous() and cache.ndim >= 2:
        stride = int(cache.stride(0)) * int(cache.element_size())
        if stride < expected:
            raise ValueError(
                f"SM120 sparse MLA cache block stride {stride} is smaller than "
                f"page payload {expected}"
            )
        return stride
    return expected


def _topk_bucket(topk: int) -> int:
    """Coarse topk bucket for the compile key (chunks_per_split is the real
    specialization driver; the bucket just keeps the key compact)."""
    return 1 << (max(int(topk), 1) - 1).bit_length()


def _sparse_mla_decode_grid_flat_launch(
    q_all: torch.Tensor,
    kv_flat: torch.Tensor,
    swa_indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    swa_len_t: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    latent_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    fp8_rope: bool,
    swa_page_size: int,
    topk: int,
    extra_topk: int,
    num_main_chunks: int,
    num_splits: int,
    chunks_per_split: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    grid_h_blocks: int,
    valid_hpb: int,
    head_block_offset: int,
    has_extra: bool,
    per_token_len: bool,
) -> None:
    q_head_dim = int(q_all.shape[-1])
    rows = int(q_all.shape[0])
    heads = int(q_all.shape[1])
    native_glm_h8 = bool(
        int(model_type) == int(ModelType.GLM_NSA)
        and heads == 8
        and int(valid_hpb) == 8
        and int(grid_h_blocks) == 1
        and int(head_block_offset) == 0
        and not bool(has_extra)
        # NVFP4 records are validated on the generic HPB=16 arm only.
        and int(scale_format) != int(ScaleFormat.NVFP4_E4M3)
        and _env_glm_h8_native_enabled()
    )
    native_dsv4_h8 = bool(
        int(model_type) == int(ModelType.DSV4) and int(valid_hpb) == 8
    )
    native_dsv4_h16 = bool(
        int(model_type) == int(ModelType.DSV4)
        and int(valid_hpb) == 16
        and int(head_block_offset) == 0
        and (int(topk) + _CAND_WINDOW - 1) // _CAND_WINDOW
        + (int(extra_topk) + _CAND_WINDOW - 1) // _CAND_WINDOW
        <= _DSV4_H8_MAX_CHUNKS
        and bool(per_token_len)
        and _env_dsv4_h16_native_mode() is not False
    )
    native_h8 = native_glm_h8 or native_dsv4_h8
    traits = make_unified_traits(
        int(model_type),
        int(compute_mode),
        int(scale_format),
        fp8_rope=bool(fp8_rope),
    )
    if native_h8:
        # Four warps cover 4*16 candidates in swapped QK.  PV keeps the same
        # output coverage with twice the H16 N-tiles per warp.
        traits = replace(
            traits,
            nt_per_warp_xv=int(traits.nt_per_warp_xv) * 2,
            math_threads=128,
            block_threads=160,
        )
    elif native_dsv4_h16:
        # Two H8 groups of four warps each keep the H8 per-warp tile shape:
        # 16 swapped QK candidates + doubled PV N-tiles, 256 math threads.
        traits = replace(
            traits,
            nt_per_warp_xv=int(traits.nt_per_warp_xv) * 2,
        )
    layout = make_smem_layout(traits)
    hpb = int(traits.hpb)
    d_v = int(traits.d_v)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if per_token_len:
        pertok_base = (
            _to_cute(q_all, cutlass.BFloat16, dynamic_layout=True),
            _to_cute(kv_flat, cutlass.Uint8, align=16),
            _to_cute(swa_indices, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(mid_out, cutlass.BFloat16, align=16, dynamic_layout=True),
            _to_cute(mid_lse, cutlass.Float32, align=4, dynamic_layout=True),
            Float32(float(sm_scale) * LOG2_E),
            Float32(float(latent_scale)),
            _to_cute(swa_len_t, cutlass.Int32, align=4, dynamic_layout=True),
            Int64(stride_kv_block),
        )
        if has_extra:
            args = pertok_base + (
                _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
                _to_cute(extra_indices_t, cutlass.Int32, align=4, dynamic_layout=True),
                _to_cute(extra_len_t, cutlass.Int32, align=4, dynamic_layout=True),
                Int32(num_main_chunks),
                Int64(stride_extra_kv_block),
                Int32(rows),
                stream,
            )
        else:
            args = pertok_base + (Int32(rows), stream)
    else:
        base_args = (
            _to_cute(q_all, cutlass.BFloat16, dynamic_layout=True),
            _to_cute(kv_flat, cutlass.Uint8, align=16),
            _to_cute(swa_indices, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(mid_out, cutlass.BFloat16, align=16, dynamic_layout=True),
            _to_cute(mid_lse, cutlass.Float32, align=4, dynamic_layout=True),
            Float32(float(sm_scale) * LOG2_E),
            Float32(float(latent_scale)),
            Int32(topk),
            Int64(stride_kv_block),
        )
        if has_extra:
            args = base_args + (
                _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
                _to_cute(extra_indices_t, cutlass.Int32, align=4, dynamic_layout=True),
                Int32(extra_topk),
                Int32(num_main_chunks),
                Int64(stride_extra_kv_block),
                Int32(rows),
                stream,
            )
        else:
            args = base_args + (Int32(rows), stream)

    kernel = UnifiedDecodeKernel(
        traits,
        layout,
        int(swa_page_size),
        int(chunks_per_split),
        h_blocks=int(grid_h_blocks),
        num_splits=int(num_splits),
        num_heads=heads,
        q_head_dim=q_head_dim,
        topk=topk,
        extra_topk=extra_topk,
        q_stride=tuple(q_all.stride()),
        swa_indices_stride0=int(swa_indices.stride(0)),
        extra_indices_stride0=int(extra_indices_t.stride(0)),
        mid_out_stride=tuple(mid_out.stride()),
        mid_lse_stride=tuple(mid_lse.stride()),
        has_extra=bool(has_extra),
        pbs_extra=int(pbs_extra),
        valid_hpb=int(valid_hpb),
        head_block_offset=int(head_block_offset),
        per_token_len=bool(per_token_len),
        native_glm_h8=native_glm_h8,
        native_dsv4_h8=native_dsv4_h8,
        native_dsv4_h16=native_dsv4_h16,
    )
    spec_fields = [
        key_field("model_type", traits.model_type),
        key_field("compute_mode", traits.compute_mode),
        key_field("scale_format", traits.scale_format),
        key_field("fp8_rope", int(traits.fp8_rope)),
        key_field("num_heads", heads),
        key_field("hpb", hpb),
        key_field("valid_hpb", int(valid_hpb)),
        key_field("head_block_offset", int(head_block_offset)),
        key_field("grid_h_blocks", int(grid_h_blocks)),
        key_field("num_splits", int(num_splits)),
        key_field("chunks_per_split", int(chunks_per_split)),
        key_field("page_block_size", int(swa_page_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("has_extra", int(has_extra)),
        key_field("pbs_extra", int(pbs_extra)),
        key_field("extra_topk_bucket", _topk_bucket(extra_topk) if has_extra else 0),
        key_field("per_token_len", int(per_token_len)),
        key_field("native_glm_h8", int(native_glm_h8)),
        key_field("native_dsv4_h8", int(native_dsv4_h8)),
        key_field("native_dsv4_h16", int(native_dsv4_h16)),
        tensor_key(
            "q_all",
            q_all,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(heads),
                DimKey.exact(q_head_dim),
            ),
        ),
        tensor_key(
            "swa_indices",
            swa_indices,
            dims=(DimKey.dynamic(), DimKey.bucket(topk)),
        ),
        tensor_key(
            "mid_out",
            mid_out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(heads),
                DimKey.bucket(num_splits),
                DimKey.exact(d_v),
            ),
        ),
        tensor_key(
            "mid_lse",
            mid_lse,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(heads),
                DimKey.bucket(num_splits),
            ),
        ),
    ]
    if has_extra:
        spec_fields.append(
            tensor_key(
                "extra_indices",
                extra_indices_t,
                dims=(DimKey.dynamic(), DimKey.bucket(max(extra_topk, 1))),
            )
        )
    if per_token_len:
        spec_fields.append(
            tensor_key("topk_length", swa_len_t, dims=(DimKey.dynamic(),))
        )
        if has_extra:
            spec_fields.append(
                tensor_key(
                    "extra_topk_length",
                    extra_len_t,
                    dims=(DimKey.dynamic(),),
                )
            )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.sm120.decode",
        # v17 adds the runtime FP8-RoPE record format specialization.  The
        # explicit cache key does not hash source, so this ABI bump rejects
        # cubins compiled for the 432-byte BF16-RoPE record.
        17,
        *spec_fields,
    )
    if per_token_len:
        entry = kernel.call_extra_pertok if has_extra else kernel.call_pertok
    else:
        entry = kernel.call_extra if has_extra else kernel
    sm12x_launch(
        entry,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::sparse_mla_sm120_decode_grid",
    mutates_args=("mid_out", "mid_lse"),
)
def _sparse_mla_decode_grid_op(
    q_all: torch.Tensor,
    kv_flat: torch.Tensor,
    swa_indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    swa_len_t: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    latent_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    fp8_rope: bool,
    swa_page_size: int,
    topk: int,
    extra_topk: int,
    num_main_chunks: int,
    num_splits: int,
    chunks_per_split: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    grid_h_blocks: int,
    valid_hpb: int,
    head_block_offset: int,
    has_extra: bool,
    per_token_len: bool,
) -> None:
    _sparse_mla_decode_grid_flat_launch(
        q_all,
        kv_flat,
        swa_indices,
        mid_out,
        mid_lse,
        swa_len_t,
        extra_kv_flat,
        extra_indices_t,
        extra_len_t,
        sm_scale,
        latent_scale,
        model_type,
        compute_mode,
        scale_format,
        fp8_rope,
        swa_page_size,
        topk,
        extra_topk,
        num_main_chunks,
        num_splits,
        chunks_per_split,
        stride_kv_block,
        pbs_extra,
        stride_extra_kv_block,
        grid_h_blocks,
        valid_hpb,
        head_block_offset,
        has_extra,
        per_token_len,
    )


@_sparse_mla_decode_grid_op.register_fake
def _sparse_mla_decode_grid_fake(
    q_all: torch.Tensor,
    kv_flat: torch.Tensor,
    swa_indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    swa_len_t: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    latent_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    fp8_rope: bool,
    swa_page_size: int,
    topk: int,
    extra_topk: int,
    num_main_chunks: int,
    num_splits: int,
    chunks_per_split: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    grid_h_blocks: int,
    valid_hpb: int,
    head_block_offset: int,
    has_extra: bool,
    per_token_len: bool,
) -> None:
    return None


def run_unified_decode(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    workspace,
    sm_scale: float,
    latent_scale: float = 1.0,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    indexed_page_table: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    return_lse: bool = False,
    lse_scale: str = "base2",
    forced_num_splits: int | None = None,
    out: torch.Tensor | None = None,
    scale_format_override: int | None = None,
    fp8_rope_override: bool | None = None,
):
    """Active SM120 sparse-MLA decode: kernel (split-K partials) + merge.

    Routes DSV4 (q_head_dim==512, UE8M0 footer) AND GLM_NSA (q_head_dim==576,
    ARBITRARY_FP32 inline scales) to the SAME warp-specialized kernel via the
    cute.constexpr traits branches (model_type/scale_format/v_has_rope). The
    dispatch gate guarantees SM120; this entrypoint rejects unsupported features
    so an accidental route never silently mis-computes.

    DSV4 DUAL-CACHE (P7c): when ``indexed_k_cache`` / ``indexed_indices`` /
    ``indexed_topk_lengths`` are supplied (the "extra"-tokens second KV pool), the
    decode attends over the UNION of the MAIN paged topk cache and the EXTRA cache
    in ONE online softmax. The chunk loop processes ``num_main_chunks =
    ceil(topk/BI)`` main chunks (gathering from the main cache) then
    ``num_extra_chunks = ceil(extra_topk/BI)`` extra chunks (gathering from
    ``indexed_k_cache`` with ``indexed_page_size`` / its own per-block stride);
    ``num_splits`` spans both. The extra cache is DSV4-only (GLM has no extra
    section). With no extra cache the kernel is compiled with ``has_extra=False``
    so the extra-section code is const_expr-elided (PTX byte-identical).
    """
    has_extra = (
        indexed_k_cache is not None
        or indexed_indices is not None
        or indexed_topk_lengths is not None
    )
    # Mapped extra page table is GENUINELY-UPSTREAM-UNSUPPORTED (upstream is
    # raw-slot-id only; no page-table indirection). RAISE, not fallback. Checked
    # BEFORE the has_extra branch so a mapped page table passed WITHOUT the extra
    # trio is still a hard error (never silently ignored / silently routed).
    if indexed_page_table is not None:
        raise ValueError(
            "SM120 sparse MLA decode: indexed_page_table (mapped extra pages) is "
            "unsupported on SM120 sparse-MLA; upstream addresses the extra cache "
            "by raw slot id only"
        )
    if has_extra:
        # Partial dual-cache trio is a HARD ERROR (upstream ICHECKs extra_indices
        # requires extra_kv_cache; sparse_mla_sm120.cu:171-174). NOT a fallback.
        if (
            indexed_k_cache is None
            or indexed_indices is None
            or indexed_page_size is None
        ):
            raise ValueError(
                "SM120 sparse MLA decode dual-cache requires indexed_k_cache, "
                "indexed_indices, and indexed_page_size together (partial extra "
                "trio is unsupported, matching upstream sparse_mla_sm120.cu:171-174)"
            )
        if int(q_all.shape[-1]) != _DSV4_HEAD_DIM:
            raise ValueError(
                "SM120 sparse MLA decode dual-cache (extra tokens) is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )

    q_head_dim = int(q_all.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        raise NotImplementedError(
            f"SM120 sparse MLA decode supports q_head_dim 512 (DSV4) or 576 (GLM); "
            f"got {q_head_dim}"
        )

    rows, heads, _ = q_all.shape
    hpb = 16
    if heads <= 0:
        raise ValueError(f"SM120 sparse MLA decode requires heads > 0, got {heads}")

    # attn_sink [num_heads] f32: upstream applies it in the DECODE MERGE
    # (sparse_mla_sm120_decode_dsv4.cu:128-129). Validate shape/dtype/device; the
    # fold itself is wired into the split.py sink-merge below (no kernel change).
    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if attn_sink.shape != (heads,):
            raise ValueError(
                f"SM120 sparse MLA decode attn_sink must have shape [{heads}], "
                f"got {tuple(attn_sink.shape)}"
            )
        if attn_sink.dtype != torch.float32:
            raise TypeError(
                f"SM120 sparse MLA decode attn_sink must be float32, got {attn_sink.dtype}"
            )
        if attn_sink.device != q_all.device:
            raise ValueError(
                "SM120 sparse MLA decode attn_sink must be on the same device as q_all"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("SM120 sparse MLA decode attn_sink must be contiguous")
    if lse_scale not in ("base2", "natural"):
        raise ValueError(
            f"SM120 sparse MLA decode lse_scale must be 'base2' or 'natural', got {lse_scale!r}"
        )
    # VALID_HPB<16 / non-multiple-of-16 heads (small-TP shards): upstream
    # VALID_HPB=min(NUM_HEADS,HPB) (decode_dsv4_kernel.cuh:152) computes a full
    # HPB=16 tile with zero-Q padding and gates writes to valid_hpb rows. We
    # realise this with up to TWO grid launches: ``h_blocks_full`` full blocks
    # (valid_hpb=16) plus one REMAINDER block (valid_hpb=heads%16) when heads is
    # not a multiple of 16. heads in {8} -> 0 full blocks + 1 remainder block of
    # valid_hpb=8. The base case (heads multiple of 16, e.g. 128) is a single
    # full-block grid -> byte-identical to the pre-P10 kernel.
    h_blocks_full = heads // hpb
    rem_heads = heads % hpb
    h_blocks = h_blocks_full + (1 if rem_heads else 0)

    model_type, compute_mode, scale_format = infer_model_type(
        q_head_dim, swa_k_cache.dtype
    )
    if scale_format_override is not None:
        scale_format = int(scale_format_override)
    if scale_format == ScaleFormat.NVFP4_E4M3 and fp8_rope_override is None:
        record_bytes = int(swa_k_cache.shape[-1])
        if record_bytes not in (368, 432):
            raise ValueError(
                f"NVFP4 cache record must be 368 or 432 bytes, got {record_bytes}"
            )
        fp8_rope_override = record_bytes == 368
    traits = make_unified_traits(
        model_type,
        compute_mode,
        scale_format,
        fp8_rope=fp8_rope_override,
    )
    if scale_format == ScaleFormat.NVFP4_E4M3 and int(swa_k_cache.shape[-1]) != int(
        traits.kv_gmem_stride
    ):
        raise ValueError(
            "NVFP4 cache record width disagrees with fp8_rope_override: "
            f"got {int(swa_k_cache.shape[-1])} bytes, expected "
            f"{int(traits.kv_gmem_stride)}"
        )
    d_v = int(traits.d_v)  # output O dim (512 for both; V == nope for GLM)

    topk = int(swa_indices.shape[1])
    extra_topk = int(indexed_indices.shape[1]) if has_extra else 0
    num_main_chunks = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    num_extra_chunks = (extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    # The swapped H8 DSV4 kernel is validated across the traced C1/C4/C128
    # regimes, including C128's three chunks per split. This is a shape-only
    # policy decision, so capture and replay use the same kernel and workspace.
    max_chunks = int(workspace.max_chunks_per_row)
    # SM count read early: both the H8/H16 policy and the split plan need it.
    sm_count = None
    if q_all.is_cuda:
        sm_count = int(
            torch.cuda.get_device_properties(q_all.device).multi_processor_count
        )

    h16_allowed = bool(
        int(model_type) == int(ModelType.DSV4)
        and heads % 16 == 0
        and num_main_chunks + num_extra_chunks <= _DSV4_H8_MAX_CHUNKS
    )
    h16_mode = _env_dsv4_h16_native_mode()
    if h16_allowed and h16_mode is None:
        # AUTO policy: pre-plan the H8 grid, then choose by regime (see
        # _dsv4_h16_auto): gather-bound many-chunk shapes and >1.4-wave H8
        # grids go H16; the sub-wave latency regime keeps H8.
        _nc8, _ns8, _ = plan_unified_decode_splits(
            topk=topk,
            max_chunks=max_chunks,
            forced_num_splits=forced_num_splits,
            num_tokens=rows,
            h_blocks=heads // 8,
            sm_count=sm_count,
            extra_topk=extra_topk,
        )
        native_dsv4_h16 = _dsv4_h16_auto(
            rows=rows, heads=heads, num_chunks=_nc8, h8_num_splits=_ns8
        )
    else:
        native_dsv4_h16 = bool(h16_allowed and h16_mode)
    native_dsv4_h8 = bool(
        int(model_type) == int(ModelType.DSV4)
        and heads % 8 == 0
        and num_main_chunks + num_extra_chunks <= _DSV4_H8_MAX_CHUNKS
        and not native_dsv4_h16
    )
    if native_dsv4_h8:
        hpb = 8
        h_blocks_full = heads // hpb
        rem_heads = heads % hpb
        h_blocks = h_blocks_full + (1 if rem_heads else 0)

    # ── P10b PER-TOKEN topk_length threading ──────────────────────────────────
    # Decide whether to route to the per-token kernel (section_len read per CTA
    # from a (rows,) int32 length tensor) or the byte-identical UNIFORM scalar
    # path. The scalar path is taken when every token's length equals the full
    # topk (the common decode contract: swa_topk_lengths[t] == topk for all t, OR
    # the caller -1-pads indices past the length so the uniform full-topk section
    # bound + the S3 idx<0 mask already realise the per-token length). For a
    # GENUINELY-mixed-length batch (some topk_length[t] < topk) the per-token
    # kernel reads each token's clamped length, so over-allocated chunks for short
    # tokens are fully masked (-> mid_lse=-inf -> merge ignores). A batch is uniform
    # ONLY when EVERY row's length >= the full section width (so the scalar bound
    # already equals each clamped length); a single SHORT row (lt[0] < cap) is NOT
    # uniform and must take the per-token clamp path.
    def _length_tensor(lengths, name, cap):
        if lengths is None:
            return None, True
        if not isinstance(lengths, torch.Tensor):
            raise TypeError(f"SM120 sparse MLA decode {name} must be a torch.Tensor")
        if lengths.shape != (rows,):
            raise ValueError(
                f"SM120 sparse MLA decode {name} must have shape [{rows}], "
                f"got {tuple(lengths.shape)}"
            )
        lt = lengths.to(device=q_all.device, dtype=torch.int32).contiguous()
        # CUDA serving must not read length values back to the host to choose a
        # kernel. The per-token path reads each token's CLAMPED length in-kernel,
        # so it is correct for uniform full-width rows and genuinely-short rows;
        # it only skips the scalar fast path. NOTE: rows==1 is NOT automatically
        # uniform here -- a single row whose length is SHORTER than the section
        # width (lt[0] < cap) must still be clamped, so it is per-token, not
        # scalar.
        if q_all.is_cuda:
            return lt, False
        # Uniform iff every token's CLAMPED length is the full section width: then
        # the scalar section bound (Int32(cap)) already equals every token's length
        # -> byte-identical scalar path. A length >= cap clamps to cap in-kernel, so
        # >= (not ==) is the correct full-section test (seqlen can exceed topk).
        is_uniform = bool(torch.all(lt >= int(cap)).item())
        return lt, is_uniform

    swa_len_t, main_uniform = _length_tensor(swa_topk_lengths, "swa_topk_lengths", topk)
    if has_extra:
        extra_len_t, extra_uniform = _length_tensor(
            indexed_topk_lengths, "indexed_topk_lengths", extra_topk
        )
    else:
        extra_len_t, extra_uniform = None, True
    # Per-token kernel only when there IS a length tensor that is not uniform.
    per_token_len = (swa_len_t is not None and not main_uniform) or (
        has_extra and extra_len_t is not None and not extra_uniform
    )
    if per_token_len:
        # Both length tensors must exist for the per-token entries (the kernel reads
        # topk_length[t] / extra_topk_length[t]). Synthesize a full-length tensor
        # for whichever section is uniform / unset so the read collapses to topk.
        if swa_len_t is None:
            swa_len_t = torch.full(
                (rows,), topk, dtype=torch.int32, device=q_all.device
            )
        if has_extra and extra_len_t is None:
            extra_len_t = torch.full(
                (rows,), extra_topk, dtype=torch.int32, device=q_all.device
            )
    preferred_num_splits = None
    if int(model_type) == int(ModelType.DSV4):
        preferred_num_splits = _dsv4_spark_short_gather_num_splits(
            rows=rows,
            heads=heads,
            num_chunks=num_main_chunks + num_extra_chunks,
            sm_count=sm_count,
            native_dsv4_h16=native_dsv4_h16,
        )
    num_chunks, num_splits, chunks_per_split = plan_unified_decode_splits(
        topk=topk,
        max_chunks=max_chunks,
        forced_num_splits=forced_num_splits,
        num_tokens=rows,
        h_blocks=h_blocks,
        sm_count=sm_count,
        extra_topk=extra_topk,
        preferred_num_splits=preferred_num_splits,
    )
    # Side-channel record of the chosen split plan (benchmarks / AutoTuner read
    # LAST_DECODE_PLAN["num_splits"]). Informational only.
    native_glm_h8 = bool(
        int(model_type) == int(ModelType.GLM_NSA)
        and int(heads) == 8
        and not has_extra
        # NVFP4 records (432B, packed E2M1+E4M3) are validated on the generic
        # HPB=16 arm only; the packed-656B H8 staging would misread them.
        and int(traits.scale_format) != int(ScaleFormat.NVFP4_E4M3)
        and _env_glm_h8_native_enabled()
    )
    native_h8 = native_glm_h8 or native_dsv4_h8
    LAST_DECODE_PLAN.clear()
    LAST_DECODE_PLAN.update(
        model_type=str(model_type),
        native_glm_h8=native_glm_h8,
        native_dsv4_h8=native_dsv4_h8,
        native_dsv4_h16=native_dsv4_h16,
        heads_per_block=(8 if native_h8 else 16),
        math_warps=(4 if native_h8 else 8),
        block_threads=(
            160
            if native_h8
            else (320 if native_dsv4_h16 else int(traits.block_threads))
        ),
        io_warps=(2 if native_dsv4_h16 else 1),
        kv_stage_packed=native_h8 or native_dsv4_h16,
        kv_smem_stride=(
            _GLM_KV_GMEM_STRIDE
            if native_glm_h8
            else (
                _DSV4_PACKED_SMEM_STRIDE
                if (native_dsv4_h8 or native_dsv4_h16)
                else int(traits.kv_smem_stride)
            )
        ),
        kv_gmem_stride=int(traits.kv_gmem_stride),
        fp8_rope=bool(traits.fp8_rope),
        qk_candidates_per_warp=(16 if (native_h8 or native_dsv4_h16) else 8),
        qk_swap_ab=native_h8 or native_dsv4_h16,
        topk=int(topk),
        extra_topk=int(extra_topk),
        has_extra=bool(has_extra),
        num_main_chunks=int(num_main_chunks),
        num_chunks=int(num_chunks),
        num_splits=int(num_splits),
        chunks_per_split=int(chunks_per_split),
        num_tokens=int(rows),
        h_blocks=int(h_blocks),
        sm_count=(int(sm_count) if sm_count else None),
        per_token_len=bool(per_token_len),
    )
    # Workspace mid_out/mid_lse must hold num_splits partials per (token, head).
    if num_splits > max_chunks:
        raise ValueError(
            f"SM120 sparse MLA decode num_splits {num_splits} exceeds workspace "
            f"max_chunks_per_row {max_chunks}"
        )

    if workspace.tmp_output is None or workspace.tmp_lse is None:
        raise RuntimeError(
            "SM120 sparse MLA decode workspace is missing mid_out/mid_lse"
        )

    # mid_out / mid_lse views over the workspace split buffers (the merge's
    # exact tmp_output[rows,heads,chunks,dim] / tmp_lse[rows,heads,chunks]). The
    # partial O dim is d_v (512) for both models.
    mid_out = workspace.tmp_output[:rows, :heads, :num_splits, :d_v]
    mid_lse = workspace.tmp_lse[:rows, :heads, :num_splits]

    stride_kv_block = _cache_block_stride_bytes(
        swa_k_cache,
        page_size=int(swa_page_size),
        model_type=int(model_type),
        record_bytes=int(traits.kv_gmem_stride),
    )

    # ── EXTRA (indexed) cache views. When there is no extra cache they alias the
    #    main cache / main indices and the kernel's has_extra=False const_expr
    #    elides the extra-section reads -> PTX byte-identical. ──
    if has_extra:
        pbs_extra = int(indexed_page_size)
        stride_extra_kv_block = _cache_block_stride_bytes(
            indexed_k_cache,
            page_size=pbs_extra,
            model_type=int(model_type),
        )
        extra_kv_flat = _cache_base_tensor(indexed_k_cache)
        extra_indices_t = indexed_indices.contiguous()
    else:
        pbs_extra = 1
        stride_extra_kv_block = 0
        extra_kv_flat = _cache_base_tensor(
            swa_k_cache
        )  # alias (never read when has_extra=False)
        extra_indices_t = swa_indices  # alias (never read)

    if out is not None:
        if tuple(out.shape) != (rows, heads, d_v):
            raise ValueError(
                f"SM120 sparse MLA decode out must have shape "
                f"{(rows, heads, d_v)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(
                f"SM120 sparse MLA decode out must be bfloat16, got {out.dtype}"
            )
        if out.device != q_all.device:
            raise ValueError(
                "SM120 sparse MLA decode out must be on the same device as q_all"
            )
        if not out.is_contiguous():
            raise ValueError("SM120 sparse MLA decode out must be contiguous")
        output = out
    else:
        output = workspace.output_buffer[:rows, :heads, :d_v]

    kv_flat = _cache_base_tensor(swa_k_cache)
    swa_len_for_op = swa_len_t if swa_len_t is not None else swa_indices
    extra_len_for_op = extra_len_t if extra_len_t is not None else swa_indices

    def _launch_grid(grid_h_blocks: int, valid_hpb: int, head_block_offset: int):
        torch.ops.flashinfer_sm12x.sparse_mla_sm120_decode_grid(
            q_all,
            kv_flat,
            swa_indices,
            mid_out,
            mid_lse,
            swa_len_for_op,
            extra_kv_flat,
            extra_indices_t,
            extra_len_for_op,
            float(sm_scale),
            float(latent_scale),
            int(model_type),
            int(traits.compute_mode),
            int(traits.scale_format),
            bool(traits.fp8_rope),
            int(swa_page_size),
            int(topk),
            int(extra_topk),
            int(num_main_chunks),
            int(num_splits),
            int(chunks_per_split),
            int(stride_kv_block),
            int(pbs_extra),
            int(stride_extra_kv_block),
            int(grid_h_blocks),
            int(valid_hpb),
            int(head_block_offset),
            bool(has_extra),
            bool(per_token_len),
        )

    if h_blocks_full > 0:
        # FULL HPB=16 head-blocks (the base path when heads is a multiple of 16).
        _launch_grid(h_blocks_full, hpb, 0)
    if rem_heads:
        # REMAINDER tail head-block: a 1-block grid with valid_hpb=rem_heads at
        # head_block offset h_blocks_full (so head_base = h_blocks_full*16).
        _launch_grid(1, rem_heads, h_blocks_full)

    # ── REUSED base-2 merge over the split axis -> final O. num_splits=1 is the
    #    trivial 1-split merge (partial == final O). ──
    from .merge import (
        build_sparse_mla_split_decode_merge_binding,
        run_sparse_mla_split_decode_merge,
    )

    # When attn_sink is supplied, the merge SELECTS the sink-folding merge kernel
    # (merge.py SparseMLASplitDecodeSinkMergeKernel) which applies the FlashMLA V4
    # fold output *= sigmoid(lse_e - sink) directly into O (exactly upstream's
    # sink-in-merge design, sparse_mla_sm120_decode_dsv4.cu:128-129). With no sink
    # it is the plain base-2 merge -> PTX/numerics byte-identical to the base path.
    merge_binding = build_sparse_mla_split_decode_merge_binding(
        tmp_output=mid_out,
        tmp_lse=mid_lse,
        num_chunks_ptr=workspace.num_chunks_ptr,
        output=output,
        num_chunks=num_splits,
        attn_sink=attn_sink,
        scratch=workspace,
    )
    run_sparse_mla_split_decode_merge(binding=merge_binding)
    if not return_lse:
        return output

    # return_lse: reconstruct the FINAL LSE from the per-split base-2 mid_lse
    # (logsumexp over the split axis, base2->natural). mid_lse aliases
    # workspace.tmp_lse[:rows,:heads,:num_splits], so reuse the shared helper.
    from flashinfer.experimental.sm12x.attention._shared.mla.api import (
        _final_lse_from_split_workspace,
    )

    lse_natural = _final_lse_from_split_workspace(
        workspace=workspace,
        q_rows=rows,
        num_heads=heads,
        launch_num_chunks=num_splits,
        scale="natural",
    )
    if attn_sink is not None:
        # Fold the per-head sink into the LSE in the natural-log domain (the merge
        # already folded it into O): lse' = log(exp(lse) + exp(sink)).
        sink = attn_sink.float().view(1, heads)
        lse_natural = torch.logaddexp(lse_natural.float(), sink)
    if lse_scale == "base2":
        return output, (lse_natural / _LN2)
    return output, lse_natural


def run_unified_prefill(*args, **kwargs):
    """Active SM120 sparse-MLA DSV4 prefill (P8).

    Thin re-export of ``prefill.run_unified_prefill`` (the correctness-first
    single-pass DSV4 prefill that REUSES the proven 288-thread / 1-IO-warp decode
    pipeline with a FINAL_BF16 epilogue). Imported lazily so launch.py has no hard
    dependency on the prefill module's CuTe symbols at import time."""
    from .prefill import run_unified_prefill as _impl

    return _impl(*args, **kwargs)
