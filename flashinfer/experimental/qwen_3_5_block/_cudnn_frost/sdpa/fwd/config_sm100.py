# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the Frost SM100 DSL SDPA flavors.

The per-flavor tile geometry is fixed here (``d512``, ``d256``);
the per-graph compile-time parameters (dtype, mask, sink, ...) arrive as a
:class:`TemplateParams` instance, built by the adapter in
:mod:`cudnn.sdpa.fwd.api_dsl`. These are graph-derived semantics plus the
chosen tuning-knob values — see the Facts / Capabilities / Knobs section of
python/cudnn/frost/README.md. Nothing in this
module reads environment variables; a kernel template receives its ``TemplateParams``
via the loader (see ``cudnn.frost.template_loader.load_template``) and calls :func:`make_cfg`.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier, by ``graph_analyzer.probe`` /
``api_dsl.SdpaFwdDslSm100.check_support`` — a ``ValueError`` from here means
those support checks have a gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional, Tuple

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_O_MXFP8,
    DTYPE_O_NVFP4,
    O_BLOCK_SCALE_BY_DTYPE,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# The arch-NEUTRAL half of the d128-file small-tile configuration (the field block, the TMEM map formula, the budgets and the
# validators, each taking the arch BUDGET as a parameter) -- shared with config_sm107; this module adds the SM100 budget
# (SM100_ARCH, below the d64 flavor) and the small-tile factory the f16/bf16 d128 file builds its 32- / 64-wide forms from.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_d128_family import (
    BIAS_MAX_FORMS,
    BIAS_MAX_HW_UPPER_BOUND,
    BIAS_MCAST_CTAS,
    BIAS_MODES,
    BIAS_MODE_NONE,
    BIAS_MODE_RESIDENT,
    BIAS_MODE_STREAMED,
    BIAS_MODE_STREAMED_MULTICAST,
    BIAS_PRELOAD_CHUNKS_DOMAIN,
    BIAS_RING_STAGES,
    BIAS_STREAMED_MODES,
    BIAS_SWZ_BYTES,
    D128FamilyArch,
    D128FamilySmallTileFields,
    D128_CORRECTION_WARPS,
    D128_OTHER_WARPS,
    D128_REG_SPLIT,
    D128_SOFTMAX_WG_WARPS,
    DEAD_SUBTILE_QUIET_FORMS,
    KV_MASK_ADDITIVE_16,
    KV_MASK_ADDITIVE_F32,
    KV_MASK_FORMS,
    PADDED_TAIL_FORMS,
    SMALL_D_Q_IN_TMEM,
    SMALL_D_SOFTMAX_WGS,
    SMALL_D_STAGES_KV,
    bias_block_rows,
    bias_resident_block_bytes,
    bias_resident_tiles,
    bias_ring_stages,
    d128_read_tile_arrivers,
    d128_total_warps,
    validate_bias,
    validate_head_dim_tile,
    validate_q_in_tmem,
    validate_regs,
    validate_small_tile_masks,
    validate_smem,
    validate_softmax_wgs,
    validate_swizzles,
    validate_tile_n,
    validate_warp_layout,
)

# Native half THD flavors whose worklist and Stats stores use packed heads.
SM100_THD_PACK_GQA_SHAPES = frozenset({(128, 128), (576, 512)})

# Native per-tensor FP8 bodies that compose page pools with THD Q/O maps.
SM100_PAGED_FP8_THD_SHAPES = frozenset({(128, 128), (256, 256)})

# Native half flavors whose PREFILL kernel stores the score side outputs -- the graph's optional
# ``score_max`` / ``score_sum_exp`` fp32 (B, H_q, S_q, 1) ports: per Q row the softmax max in
# natural units and the fp32 row-sum of exp(s - max), the two operands the LSE is built from
# (Stats == score_max + ln(score_sum_exp); a keyless row reads -inf / 0, with a sink sink / 1).
# The d128 file carries the store from its correction epilogue; the decode tile (cga1 on d128),
# the native d64 flavor (the same file at d_flavor=64, not yet validated with the outputs) and the
# d192x128 / d256 / d512 files do not.  The engine row's ``score_stats_d_shapes`` and the adapter's
# check_support read THIS set (rule 8b').  Dense, unsplit, unpaged plans only (engines.mismatch).
SM100_SCORE_STATS_SHAPES = frozenset({(128, 128)})


# Whether the adapter routes per-tensor FP8 d128 decode shapes onto the two-KV-
# stream specialization (TemplateParams.dual_kv_stream); heuristics mirror the
# predicate through _sm100_fp8_dual_kv_stream.  Measured on B300 (llama 64/8,
# 128k keys, same node, A/B/A/B): dense b=1 75 -> 63 us, b=8 469 -> 333, b=128
# 6.85 -> 5.43 ms; paged (page 16) b=1 87 -> 73 us, b=128 5.40 -> 4.19 ms --
# the latter at the HBM ceiling.  A module attribute so a test can switch it.
FP8_DUAL_KV_STREAM_ROUTED = True
# Q rows one 128-row tile holds: the two-stream decode serves S_q x packed heads
# up to this many rows (one Q tile at cga1).
FP8_DUAL_KV_STREAM_MAX_ROWS = 128


def fp8_dual_kv_stream_routed() -> bool:
    """Late-bound read of FP8_DUAL_KV_STREAM_ROUTED, so the adapter and the
    heuristics agree with a test that monkeypatches the module attribute."""
    return bool(FP8_DUAL_KV_STREAM_ROUTED)


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph compile-time parameters threaded into a kernel template.

    Design notes (see also python/cudnn/frost/README.md):

    - Contents: graph-derived semantics (dtype, masks, sink/padded/THD) plus
      the chosen values of true tuning knobs (sched_policy). Shapes are
      deliberately NOT here — they are handled by the template's per-shape
      ``compile()`` cache; this record holds only what changes the *traced
      code* of the kernel.
    - Direction: this is the OUTPUT of a successful eligibility match, never
      an input to it. The probe compares facts and requested knobs against
      an engine's Capabilities; only after that passes does the engine's
      ``lower`` hook assemble this record.
    - Frozen + hashable on purpose: it doubles as the kernel-module cache key
      in frost.template_loader — one distinct TemplateParams == one distinct
      compiled specialization, and identical params reuse the compiled one.
    - Validation here (``_validate_params`` and the ``make_cfg_*`` checks)
      raises ValueError but is a BACKSTOP: every reachable violation must be
      rejected earlier by a Capabilities row; tripping it means that row is
      dishonest, not that a user erred.
    """

    dtype_qkv: int = DTYPE_FP16  # E4M3/E5M2 (0/1, the quantized kernels) or BF16/FP16 (2/3)
    dtype_o: int = -1  # output dtype (0..3); -1 = inherit dtype_qkv. MXFP8 writes BF16/FP16.
    # The mask is ONE diagonal band (the model FlashAttention / CUTLASS FMHA /
    # the analyzer facts all share): per-side OFFSETS from the diagonal, None =
    # unbounded on that side. Row q attends kv in [q - window_left, q +
    # window_right] (shifted by S_kv - S_q when bottom_right):
    #   window_right = None -> no upper bound;  0 -> plain causal;  R > 0 ->
    #   causal widened by R future tokens (diagonal_band_right_bound).
    #   window_left  = None -> no lower bound;  W -> sliding window of W past
    #   tokens (cuDNN diagonal_band_left_bound - 1).
    #   bottom_right anchors the band's diagonal at the bottom-right corner.
    # make_cfg_* derives the kernels' MASK_FLAGS bits from these.
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    bottom_right: bool = False
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the
    # epilogue scales the natural-log LSE by log2(e) right before the store.
    stats_log2: bool = False
    seq_kv_lens_present: bool = False
    # Dense padded-Q trim: per-batch seq_len_q is a SEPARATE (B,)-int32
    # kernel parameter (seq_q_lens_addr — mirrors cuDNN's distinct SEQLEN_Q
    # pointer / FA's seqused_q; its reads compile out unless this flag is
    # set); q rows >= seq_len_q[b] write O := 0 / LSE := -inf (cuDNN >= 9.14
    # convention). Dense-only — THD carries per-sequence Q lengths via
    # cu_seqlens instead.
    seq_q_lens_present: bool = False
    sched_policy: int = SCHED_NATURAL
    # Compile-time LPT head/batch grouping. Keep 1 unless the selected kernel
    # and concrete graph shape opt into a divisor of B*Hq.
    lpt_head_group: int = 1
    # Dense FP8 kernels may specialize scheduler selection/decoding to the
    # graph's compile-time number of query tiles. Zero keeps runtime derivation.
    lpt_q_tiles: int = 0
    # Optional L2 working-set budget for SCHED_LPT_L2. Zero keeps the flavor's
    # default budget.
    lpt_l2_size_mib: int = 0
    thd_varlen: bool = False
    # Ragged Q/O/Stats over PAGED K/V on the d128 DECODE tile (S_q(max) == 1):
    # the Q row coordinate is the batch's ragged offset over the packed Q view
    # and the split combine places the final O / Stats rows at their ragged
    # offsets; the dense grid, PackGQA and the split path are all kept.  Not
    # the prefill tile's THD leg (thd_varlen), which is mutually exclusive.
    ragged_q: bool = False
    # PackGQA: pack Q rows from the G query heads sharing one KV head into a
    # single TILE_M tile, token-major (row r ↔ token r // G, head r % G), so
    # tiles stay full for GQA/MQA.  When G does not divide TILE_M the d128 and
    # d256 f16 kernels pack its largest divisor that does (Cfg.PACK_G, from
    # pack_gqa_group_size); ``qh_per_kh`` is always the graph's GQA ratio.
    pack_gqa: bool = False
    qh_per_kh: int = 1
    # KV split: each Q tile's KV loop range is cut into ``split_kv`` contiguous
    # chunks, each run as its own persistent tile writing a partial (O, LSE)
    # that kernels/sm100/split_combine.py reduces.  1 = off (byte-identical
    # codegen to the single-pass kernel).
    #
    # A split writes its partials as fp32, straight from the epilogue's
    # accumulator registers and bypassing the SMEM O tile and its TMA store, so
    # the combine reduces an unrounded input and performs the only rounding to
    # O's dtype.  Partial-only: the caller-visible O is unchanged.
    split_kv: int = 1
    # MMA cluster width: 2 = cga2 collective tcgen05.mma.cta_group::2 (a CTA
    # pair share one MMA, each holding half of every K/V tile); 1 = cga1, one
    # independent CTA per tile.
    #
    # cga1 has no collective MMA to halve per-CTA K/V, so at a fixed STAGES_KV
    # it doubles that footprint.  d128 buys the 64 KiB back by aliasing Q and O
    # into one slab (make_cfg_d128 turns QO_ALIAS on for cga1; see
    # _validate_cfg_d128's SMEM check); the fp8 family instead scales the stage
    # count with the width so stages x per-CTA-buffer stays constant.
    cta_mma: int = 2
    # Native head-dim geometry of the flavor this template compiles to (TILE_K =
    # TILE_O). The SM100 f16 prefill file serves both the llama d128 geometry and
    # the narrow d64 one (gpt-oss class, d_qk = d_v = 64): at 64 the MMA runs a
    # K=64 tile instead of zero-filling a 128-wide one, which is half the BMM1
    # work and half the Q/K/V/O SMEM. Shapes are deliberately absent from this
    # record, but the flavor changes the TRACED code (tile extents, SMEM slabs,
    # TMA boxes), so it is a template-key field rather than a compile() arg.
    d_flavor: int = 128
    # cc10.3+ fuses the S_acc row-max into the LDTM (tcgen05.ld.red.f32.max); cc10.0
    # lacks it and uses the manual load + software reduction. Auto-set from the device
    # capability at compile time (MXFP8 + the per-tensor FP8 d192x128 kernel; the f16
    # kernels do not read it, and the SM107 siblings carry the instruction unconditionally).
    fused_ldtm_stat: bool = False
    # exp2 MUFU / FMA split of the sm100 softmax (the _E2E_* block of sm100/prefill_d128_mxfp8.py,
    # prefill_d128_fp8.py and prefill_d192_d128_f16.py; the _exp2_* helper mix of
    # prefill_d192_d128_fp8.py): a slice of the exp2 per row on the FMA pipe instead of MUFU.EX2.
    # Claimed per KERNEL and per ARCH, because its sign follows the part's MUFU.EX2 rate: MEASURED
    # 16 elements/clk/SM on cc 10.0 (B200, where the split is +7.8 % on d128 MXFP8, +4.5 % on d128
    # FP8, +1.9 % on d192x128 bf16 at the chart layers, +7.0 % causal / +5.7 % dense on d192x128 FP8
    # at the DSv3 layer) and 32 on cc 10.7 (Rubin, where the same split is -9..-10 %: an emulated
    # exp2 costs 1.99x the MUFU time it frees); cc 10.3 (B300) has the doubled rate too -- MEASURED
    # -10 % causal / -6 % dense on d192x128 FP8 with the split on.  Auto-set by the adapter from the
    # BUILD device (api_dsl._exp2_fma_split_for: cc == (10, 0) x the four kernels above, and the d64
    # decode tile, sm100/decode_d128_f16.py, through its own set) -- widen only after an A/B on the new
    # cc / kernel.  Off, the kernel traces the plain MUFU exp2 (the develop spelling).  Only those
    # kernels read it.
    exp2_fma_split: bool = False
    # The L2-prefetching loader of the half-input single-CTA d192x128 tile (sm100/prefill_d192_d128_f16.py at cta_mma=1,
    # half precision): the TMA-LDG warp prefetches the K/V tiles one kv step ahead and the next tile's Q / first K/V
    # into L2, runs its kv loop unrolled 1x, and the dense-causal arm keeps the plain MUFU exp2.  Claimed per arch like
    # exp2_fma_split: set by the adapter (api_dsl._cga1_l2_prefetch_for) on the cc it is measured on; off, the tile
    # keeps the develop loader.  Only that kernel reads it.
    cga1_l2_prefetch: bool = False
    # sdpa(softmax_precision=cudnn.data_type.HALF) op attribute: exponent + P-cast run as
    # f16x2 pairs (MUFU EX2.F16x2 + cvt.rn.satfinite.*x2.f16x2) instead of
    # scalar f32 ex2. The cc 10.7 FP8 / MXFP8 kernels (every flavor; see
    # config_sm107.SM107_SOFTMAX_F16_FLAVORS) — the exp arguments are bounded
    # (<= RESCALE_THRESHOLD [+ P_CAST_LOG2_SCALE on per-tensor FP8]), so f16
    # range is exact where it matters and P quantizes to FP8 either way.
    softmax_f16: bool = False
    # The caller has already multiplied Q by attn_scale * log2(e): the kernel runs exp2(S - m) on the
    # raw QK^T (no per-score FFMA2 by the scale) and, together with softmax_f16, fuses the shift and the
    # f32->f16 convert into one instruction per pair.  The published Stats are unchanged -- the running
    # max and the scores are in the same log2 domain as when the kernel applies the scale itself.
    # Served by the cc 10.7 MXFP8 and f16/bf16 kernels (config_sm107.SM107_SCALE_PREFOLDED_FLAVORS); per-tensor
    # FP8 declines it (the kernel folds descale_q * descale_k into the softmax scale), as do the cc 10.0 / 10.3
    # line at config time and the adapters of the other architectures at check_support.
    softmax_scale_prefolded: bool = False
    # Paged KV cache (FlashInfer / vLLM decode contract): K/V are page pools
    # indexed through a per-batch ``block_table`` [B, max_pages] int32, and
    # the per-batch KV length is the (B,) ``seq_kv_lens`` device tensor
    # (``seq_kv_lens_present`` is therefore mandatory). ``page_size`` tokens
    # per page. The in-page layout (HND ``[num_pages, H_kv, page_size, D]`` vs
    # NHD ``[num_pages, page_size, H_kv, D]``) is NOT a parameter: it is the
    # pool's strides, which the per-shape compile() already pins. Dense-only.
    paged_kv: bool = False
    page_size: int = 0
    # Experimental D128 MXFP8 specialization: Q/K remain block-scaled FP8,
    # while the complete softmax(P) @ V leg uses BF16 operands. This is a
    # template axis because it changes the TMA maps, shared-memory layout and
    # BMM2 instruction kind. It is intentionally not wired into graph routing.
    pv_bf16: bool = False
    # Output Amax is an independently optional graph output. It is normally
    # enabled for the existing MXFP8 kernels; the direct QK-MXFP8/BF16-PV
    # experiment enables it only when its plan declares an Amax_O buffer.
    # This is plan-time state: it must never be inferred from execute() args.
    emit_amax_o: bool = True
    # Epilogue gate: O := O * sigmoid(GATE). GATE is TMA-staged by the load warp after the KV loop into a
    # 64 KiB SMEM tile and consumed in the correction epilogue. Compile-time: sizes the tile + 2 mbarriers.
    # Served by the SM107 d256 f16/bf16 and per-tensor FP8 kernels only (config_sm107._EPILOGUE_GATE_FLAVORS).
    # Being a TemplateParams field it is part of the module-cache key: gate on/off are two coexisting
    # specializations of one template, and an ungated module traces byte-identically to before the field existed.
    epilogue_gate: bool = False
    # Decode-shaped d256 f16/bf16 graphs (S_q * pack_g within the arch line's routed
    # rows: D256_DECODE_ROUTED_MAX_Q_ROWS here, config_sm107's table (16, 32, 64) on
    # cc 10.7, where the 32- and 64-column tiles serve the MTP step in token units) lower onto
    # sm100/decode_d256_f16.py, the swap-AB tile: the KV tokens ride the MMA M axis
    # and the packed Q rows ride N, so the MMA and exp work scale with the LIVE
    # rows instead of a 128-row Q tile.  The value is the N extent the kernel is
    # compiled for -- the rows of ONE token unit (16; 32 at the template level, 32
    # and 64 on the Rubin route, from decode_d256_q_tile); 0 = the prefill tile.
    # Plan-time only (S_q is a declared shape).
    decode_q_tile: int = 0
    # The 128-row DECODE tile (sm100/decode_d128_f16.py) for flavors that do not
    # key it off cta_mma.  d128 routes that tile by cga1, because its prefill
    # pipeline wants cga2 anyway; d64's prefill is FASTEST at cga1 (a 512-row
    # cga2 cluster wastes most of a narrow diagonal band), so the two legs would
    # be indistinguishable by cta_mma alone -- and TemplateParams is the template
    # cache key, so one record must never name two kernels.  Set by
    # SdpaFwdDslSm100._d64_decode_tile, the twin of _decode_q_tile.
    decode_tile: bool = False
    # The quadrant-cooperative softmax of the d64 decode tile (sm100/decode_d128_f16.py QUAD): the live Q rows (S_q x packed heads <= 32) are loaded into
    # all four TMEM lane quadrants and each softmax warp owns a quarter of the key columns, the correction warps merging the four partials per unit.
    # Needs decode_tile; set by SdpaFwdDslSm100._d64_decode_quad.
    decode_quad: bool = False
    # The decode tile's early-Q form (sm100/decode_d128_f16.py EARLY_Q): the Q slab is released by the last BMM1 of a unit instead of the O store's
    # return edge, and the correction warps store the half output straight from TMEM.  Claimed per arch, like exp2_fma_split: set by
    # SdpaFwdDslSm100._decode_early_q on the cc it was measured on (10.0, 10.3); an unmeasured part (cc 10.7 loads the same tile) keeps the staged
    # TMA store.  The THD / ragged legs keep it whatever the field says.
    early_q: bool = False
    # Per-tensor FP8 d128 decode: the two Q sub-tiles of sm100/prefill_d128_fp8.py
    # become two KV STREAMS over one 128-row Q tile (even KV tiles on sub-tile 0,
    # odd on sub-tile 1, each with its own running max / sum, merged once in
    # the epilogue).  Doubles the K/V bytes per pipeline step, which is what the
    # decode-shaped step -- chain-latency-bound at ~1.1 us a 128-key tile -- is
    # short of.  cga1, dense or paged, S_q x packed heads <= 128 rows; set by
    # SdpaFwdDslSm100._fp8_dual_kv_stream.  Consumed by that template only.
    dual_kv_stream: bool = False
    # Internal specialization of the shared one-Q-tile half pipeline.
    single_q_head_dim: int = 128
    # The graph declares capacity for exactly one sequence. Runtime lengths
    # remain device data; the prepared binder rejects a larger batch.
    thd_batch_one: bool = False
    # d512 f16/bf16 forward on the "2x2 datapath": ONE pipeline per CTA on the
    # cta_group::2 M=128 atom (64 Q rows per CTA, 12 warps), a (4,1,1) cluster of
    # two pairs sharing K/V by TMA multicast -- sm100/prefill_d512_f16_2x2.py
    # (make_cfg_d512_2x2 / CfgD512X2) instead of the cga4x1 role-split kernel.
    # APPEND-ONLY and default False: every existing record renders byte-identically
    # (make_cfg_d512 and the loader read it through getattr).  Plan-time only;
    # api_dsl.D512_2X2 is the call-time switch that sets it (default True since 2026-10-06; False = the
    # role-split A/B arm).
    mma_2x2: bool = False
    # attn_scale < 0: BMM1 negates Q (tcgen05 a_negate), so the kernel's raw-score max, masks and exp2 run on -S at
    # |attn_scale| (#1435). APPEND-ONLY, default False.
    negate_scores: bool = False
    # A dense S_kv off the KV tile with no mask covering the tail: compile the padded mask against the scalar
    # S_kv, with no per-batch lengths buffer (MASK_PADDED without SEQ_KV_LENS_PRESENT; #1425).
    kv_tail_mask: bool = False
    # Index-list (block-sparse) attention on the cc 10.7 d256 f16/bf16 line (sm107/sparse_d256_f16.py,
    # config_sm107.make_cfg_d256_sparse): the query attends the keys of its caller-selected 4-token blocks plus the open
    # tail block of its visible range, under the causal mask.  These four are DECLARATION attributes (the selection IS the
    # function), never knobs, and they are the template's compile-time geometry:
    #   qsa_block_topk       0 = not an index-list record (every dense template declines a non-zero value: it would read
    #                        K/V densely and silently ignore the list).  Otherwise the list width: the block kernel's
    #                        CFG.BLOCK_TOPK, 4 <= top_k <= 512, top_k % 4 == 0 -- every ids byte count in the kernel is
    #                        top_k x 4 (the staging slot, the bulk-copy length, the expect_tx), one compiled module per value.
    #   qsa_block_size       tokens per selectable block; 4 = one tma_gather4 (four rows) per block per 128-B column box.
    #   qsa_include_open_block  True = the kernel appends the open tail block from the query's position (the model's
    #                        semantics).  False is a KERNEL-ONLY test arm (a pure caller list, for A/B against references
    #                        that materialise the tail); it is not a public field of the block's spec.
    #   qsa_list_per_sequence  the decode / MTP form: block_ids is [B, top_k], one list per sequence, and the tail runs
    #                        from the first decode token's position.  False = one list per query token ([T_q, top_k]).
    # APPEND-ONLY, defaults inert: every existing record renders byte-identically (the make_cfg_* factories read these
    # through the record, and 0 / 0 / True / False is "no index list").
    qsa_block_topk: int = 0
    qsa_block_size: int = 0
    qsa_include_open_block: bool = True
    qsa_list_per_sequence: bool = False
    # The sparse kernel's softmax SPLIT -- a PERFORMANCE knob of its softmax role, never numerics (the same function, bitwise, at
    # either value): 0 = the flavor default (TWO 4-warp column groups of 8 columns each, 20 warps, 96 registers flat), 1 = ONE 4-warp
    # column group over the 16 columns (16 warps, the pre-split body).  A template-cache axis like mma_2x2 (two coexisting renderings of one file);
    # APPEND-ONLY, default inert; every dense template declines a non-zero value together with the other qsa_* fields.
    qsa_softmax_groups: int = 0
    # Additive attention bias (the graph's ``bias`` operand, ``[1|B, 1|H_q, S_q, S_kv]`` in the io dtype) on the Rubin
    # f16/bf16 d128-file tiles: 0 = none (every existing record renders byte-identically), 2 = the STREAMED ring (one bias
    # tile per kv tile through a ``bias_stages``-deep SMEM ring, any S_kv -- the fallback above the resident budget), 1 = the RESIDENT row-block --
    # the TMA-LDG warp loads the CTA's ``TILES_Q * TILE_M`` bias rows for its whole KV range into SMEM once per work
    # item and the softmax warps read their row slice with swizzled lane loads, adding it pre-scale as one FFMA per
    # score (``s + bias * inv_scale``) before the band mask and the software row max.  A compile-time axis (the SMEM
    # block, two mbarriers and the masked-path softmax arm fold on it) and therefore part of the module-cache key.
    # Served by ``sm107/prefill_d128_f16.py`` at the widths whose SMEM holds at least one 128-column block
    # (``config_sm107.sm107_bias_resident_s_kv_max``); the SM100 line and every other flavor decline it.
    bias_mode: int = 0
    # Per-key mask (the python-only ``kv_mask`` input of ``sdpa``, ``[1|B, 1, 1, S_kv]``): 0 = none; 1 = the BOOLEAN
    # mask in the bit-word form -- each softmax warp assembles one 32-bit keep word per 32 kv columns of the tile by a
    # warp ballot over byte loads and folds it into the bit-word mask (``tile_dsl.mask.apply_mask_chunk``); a key whose
    # byte is 0 is masked like a padded one, a row whose every key is masked reads O = 0 / Stats = -inf; 2 / 3 = the
    # ADDITIVE per-key vector in the io dtype (bf16 / fp16 pairs) / in fp32, loaded as warp-uniform 16-byte granules and
    # folded into the raw scores pre-scale exactly like the bias (``s + v * inv_scale``, one packed FFMA per pair) --
    # the models' finite -1e9 fold preserved as a VALUE (a fully -1e9 row is their uniform average, not the empty-row
    # contract); nothing staged in SMEM, no barrier row.  Same kernel and same serving set as ``bias_mode``; composes
    # with ``seq_kv_lens_present`` and with the bias.
    kv_mask_form: int = 0
    # KV tile width of the d128-file softmax (the S-accumulator columns per sub-tile, the K/V ring tile, the P packing
    # and the softmax chunk count all derive from it).  128 is the value every rendering has run at; 64 is the
    # one-chunk arm of the f16/bf16 small tiles (trace-compiled and validated on cc 10.7; served through the TILE_N
    # knob, ``config_sm107.SM107_TILE_NS_BY_D_SHAPE``) and 256 is refused by the TMEM predicate at two sub-tiles -- the
    # config factory pins ``N_BMM2_CHUNKS x BMM2_CHUNK_SIZE == TILE_N``.
    tile_n: int = 128
    # The (n, n+1) PAIR work item of the resident-bias arm (``bias_mode = 1`` on the Rubin f16/bf16 d128-file tiles): 0 = the
    # item is a 256-row Q super-tile of ONE batch entry (every existing record renders byte-identically); n >= 1 = a work item
    # is (a 128-row q tile, a head, a CHUNK of up to n consecutive entries of the flattened batch axis), the two softmax
    # sub-tiles are rows (nb, nb + 1) of the SAME q tile, and the bias row-block (now 128 rows) is loaded ONCE per item and
    # read by every step of the chunk -- the per-item fixed cost and the bias feed divided by n.  Needs a batch-broadcast
    # bias ([1, 1|H, S_q, S_kv]: two rows share one block), cta_mma = 1 and a top-left band (config_sm107._validate_bias).
    # A compile-time axis (the block rows, the grid decode, the pair's ring protocol fold on it), hence a cache-key field.
    n_chunk: int = 0
    # The STREAMED bias ring's depth (``bias_mode = 2`` on the Rubin f16/bf16 d128-file tiles: the bias row-block is no longer
    # resident for the item's whole KV range -- the TMA-LDG warp streams ONE ``BIAS_BLOCK_ROWS x TILE_N`` bias tile per kv tile
    # through a ``bias_stages``-deep SMEM ring the softmax warps release per kv tile, so S_kv is bounded by nothing but the
    # payload / stride rules).  0 = the factory's derived default (the deepest of 3 / 2 the carveout holds beside the operands;
    # ``config_sm107.sm107_bias_stages_pick``); 2 / 3 = a request.  Meaningless (and must be 0) on ``bias_mode`` 0 / 1.  A
    # compile-time axis (the ring's SMEM extent and its two mbarrier rings fold on it), hence a cache-key field; APPENDED.
    bias_stages: int = 0
    # The PADDED tail form of the PLAIN padded kv loop on the Rubin f16/bf16 32- / 64-wide tiles (MASK_FLAGS == MASK_PADDED alone: per-batch
    # KV lengths or an S_kv that is not a tile multiple, no bias / kv_mask): 0 = the tail kv tile runs the SOFTWARE-max masked body as a
    # second loop beside the HW-max body (a second full softmax body per warp group -- the 128-wide rendering outgrows the per-SM
    # instruction cache, the padding leg's measured pathology); 1 = the PRE-MASKED tail: one loop, the HW-max body on every tile, and on
    # the tail tile the softmax warp first rewrites its sub-tile's out-of-range S columns as -inf in TMEM
    # (``config_sm107.sm107_padded_tail_form_pick``).  A compile-time axis of the kv loop, hence a cache-key field; 0 on every other
    # rendering (the factory refuses 1 off the eligible set).  APPENDED.
    padded_tail_form: int = 0
    # The DEAD-SUB-TILE QUIET body of the one-batch-entry item on the Rubin f16/bf16 32- / 64-wide tiles (no bias): a softmax sub-tile
    # whose FIRST row is at or past its batch entry's S_q has no live row -- sub-tile 1 of the last 256-row super-tile when S_q % 256 is
    # in 1..128 (S 384: a quarter of the item's softmax work), and under cga2 the non-leader CTA's sub-tiles past S_q.  0 = that warp
    # group runs the FULL softmax body on rows nobody stores (the correction trims them, the TMA store clips them); 1 = it runs the QUIET
    # body: every barrier interaction of the live body (the bmm1_done wait, alpha = 1 and the stat_full arrive, the bmm2_ready arrives,
    # the warp-group ping-pong, the stat_empty wait), no TMEM load and no exp2 / max / sum, so the live sibling warp group runs the
    # ping-pong alone (``config_sm107.sm107_dead_subtile_quiet_pick``).  A compile-time axis of the kv loop, hence a cache-key field;
    # 0 on every other rendering (the factory refuses 1 with a bias, on the pair item, on PackGQA / THD / the gated epilogue).  APPENDED.
    dead_subtile_quiet: int = 0
    # The ROW-MAX FORM of the bias / kv_mask softmax body (the -inf masked body of the Rubin f16/bf16 d128-file tiles): 0 = the
    # SOFTWARE row max over the biased, masked scores (every existing record renders byte-identically); 1 = the HARDWARE UPPER
    # BOUND -- the fused ``tcgen05.ld.red...max`` over the RAW scores (the plain body's own load) plus the thread's row max of the
    # bias tile in the packed 16-bit domain: ``max_k S + max_k bias >= max_k (S + bias)``, so the online softmax runs on an upper
    # bound of the true row max (valid for stability: P <= 2^RESCALE_THRESHOLD as before; the LSE and O are shift-invariant), the
    # exps shrink by at most the row's RAW score range inside the tile (tens of log2 units against the fp32 flush at -126), and the
    # software max tree leaves the post-load critical path.  Served with ``bias_mode`` / ``kv_mask_form`` 1 only (an additive
    # kv_mask vector has no cheap per-tile max); the score side outputs stay on the software arm (the bias arms decline them).  The record
    # field defaults to 0; the ADAPTER picks the measured per-form default for a plan that requests none (``config_sm107.
    # sm107_bias_max_form_default``: 1 on the kv_mask-only form, 0 with a bias).  A compile-time axis (the body folds on it), hence a
    # cache-key field; APPENDED.
    bias_max_form: int = 0
    # The number of SOFTMAX WARP GROUPS of the Rubin f16/bf16 d128-file body = its Q sub-tiles per work item (``TILES_Q``): 2 = the
    # classic 16-warp body (every existing record renders byte-identically); 3 = the 20-warp body of the (n, n+1) pair item at TILE_N 64
    # (three rows of the chunk per step, the split 12 x 128 + 4 x 56 + 4 x 40 = the 20-warp entry pool at the 32-wide tile) -- the
    # MEASURED default of ONE form, a bias together with a kv_mask on that item at cga1 with dense KV lengths from S_kv 640 up to 3072
    # flattened batch rows (``config_sm107.sm107_softmax_wgs_pick``), selectable through the adapter's ``softmax_wgs=`` / the sweep seam
    # on the rest of its serving set.  A compile-time axis (the warp table, the per-sub-tile ring depths and the TMEM map fold on it),
    # hence a cache-key field -- the kernel-module cache is keyed by this record, so the 3-WG build is its own module; APPENDED.
    # ``config_sm107._validate_softmax_wgs`` pins its serving set (TILE_N 64, the pair item, cga1, the f16/bf16 32- / 64-wide tiles).
    softmax_wgs: int = 2
    # The S / P TMEM slot count per softmax sub-tile of the Rubin 32- / 64-wide f16/bf16 tiles at TILE_N 64 (APPENDED after ``softmax_wgs``; the sm107
    # d128 file reads it through getattr, every other template ignores it): 1 = one slot per sub-tile (every record so far renders
    # byte-identically), 2 = the double-buffered S -- the MMA warp issues BMM1(k + 1) into the free slot ahead of BMM2(k) so the
    # tensor core runs it while the softmax is still on S(k).  A template-key field (not a compile() arg, not a config seam): the
    # slot count changes the TRACED code (the TMEM map, two barrier rings' extents, three roles' cursors) and the template-module
    # cache keys on this record alone, so a process that builds both forms must see two records.  Picked by
    # config_sm107.sm107_s_slots_pick; validated by the factory (TILE_N 64, the small f16/bf16 tiles, the TMEM fit).
    s_slots: int = 1
    # Q resident in TMEM (the Rubin f16/bf16 32- / 64-wide d128-file tiles; every other template ignores it): 0 = BMM1 reads Q from
    # SMEM through its descriptor on every kv tile (every record so far renders byte-identically), 1 = the MMA warp copies each
    # sub-tile's Q tile SMEM -> TMEM once per n-step (``tcgen05.cp 128x128b`` through the same descriptor, one elect block) and issues
    # every BMM1 of the step as an A-from-TMEM ``tcgen05.mma``; the UTCCP's commit releases the Q SMEM slab (``mb_q_empty``) at the START of
    # the step instead of after its last BMM1, so the TMA-LDG warp prefetches the next step's Q / K / V during the current kv loop.  The
    # same MMAs over the same operands in the same k-step order: O and Stats are BITWISE the SMEM-Q kernel's.  A template-key field (the
    # TMEM map gains the Q slots, the MMA form and the commit site move): the template-module cache keys on this record alone.  Picked by
    # config_sm107.sm107_q_in_tmem_pick (0 until measured; the sweep seam pins 1); validated by the factory (config_sm107._validate_q_in_tmem).
    q_in_tmem: int = 0


# Paged KV is wired through the K/V TMA-LDG sites of these flavors only; any
# other flavor must reject it rather than silently reading K/V as dense.
# Flavor tags as make_cfg_* / _validate_params spell them ("d192" is the
# d192x128 kernel, whose K and V pools differ in row width). engines'
# ``paged_d_shapes`` and the adapter's check_support name the same set.
SM100_NATIVE_D576_SHAPE = (576, 512)
_PAGED_KV_FLAVORS = frozenset({"d64", "d128", "d192", "d256", "d512", "d576"})

# The fused epilogue gate (TemplateParams.epilogue_gate) is a RUBIN feature.  ONE body of this line reads
# CFG.EPILOGUE_GATE: the SHARED f16/bf16 d256 prefill (sm100/prefill_d256_f16.py), whose paged / THD / PackGQA
# arms the Rubin paged half (256, 256) route loads for sm_107a -- its gate seams (sGate, mb_gate_full / mb_gate_empty,
# the TMA-LDG issue after the KV loop, the packed tanh epilogue) are the sm107 d256 body's, through the shared hook in
# _common_blackwell; the gated rendering needs ~258 KiB of SMEM, so it exists on the 327 KiB Rubin carveout only (the
# adapter admits the gate at cc 10.7 and nowhere else).  The quantized d256 bodies and every other flavor of this line
# do NOT read the flag, so a module loaded with it there would silently produce the UN-gated O: _validate_params
# refuses them by name.  The Rubin siblings that serve the gate are config_sm107._EPILOGUE_GATE_FLAVORS.
_EPILOGUE_GATE_FLAVORS = frozenset({"d256"})

# split_kv / cta_mma live on the TemplateParams shared by every SM100 flavor, but
# a flavor only honours them once its make_cfg_* threads them into a Cfg AND its
# kernel reads them.  Accepting them elsewhere would silently ignore them — and
# for split_kv that is not merely surprising but WRONG: the caller sizes an
# (S*B)-batch partial workspace and runs the combine, while the kernel keeps
# writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
# than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
# corrupt the result instead of dropping out.  Grow these sets as flavors land.
_SPLIT_KV_FLAVORS = frozenset({"d64", "d128", "d192", "d256", "d512"})
_CTA_MMA_FLAVORS = frozenset({"d32", "d64", "d128", "d192"})


def supports_thd_split(d_shape, *, device_cc, fp8, thd, paged, max_q, padded_stats):
    """Packed half partials for the explicitly wired native head geometries."""
    return (
        device_cc in ((10, 0), (10, 3), (10, 7))
        and not fp8
        and thd
        and not padded_stats
        and (
            (device_cc in ((10, 0), (10, 3)) and paged and d_shape == (64, 64) and max_q > 0)
            or (d_shape == (128, 128) and max_q > (1 if paged else 0))
            or (not paged and d_shape == (192, 128) and max_q > 0)
            or (d_shape == (256, 256) and max_q > 0)
            or (device_cc in ((10, 0), (10, 3)) and d_shape == SM100_NATIVE_D576_SHAPE and max_q > 0)
        )
    )


def supports_scalar_kv_tail_split(d_shape, *, device_cc, fp8, pertensor):
    """Scalar-tail masking composes with D128 and native Rubin half D512 split."""
    return (device_cc in ((10, 0), (10, 3), (10, 7)) and d_shape == (128, 128) and (not fp8 or pertensor)) or (
        device_cc == (10, 7) and d_shape == (512, 512) and not fp8
    )


def supports_quantized_split_sink(*, device_cc, pertensor, thd):
    """Per-tensor SM100/SM103/SM107 dense-Q partials fold the sink in combine.

    Existing split, packing and paged head-dimension domains still apply.
    MXFP8, packed queries and native SM120 need separate qualification.
    """
    return device_cc in ((10, 0), (10, 3), (10, 7)) and pertensor and not thd


def supports_paged_split_sink(d_shape, *, device_cc, fp8, thd, paged, max_q):
    """Sink-aware packed combine, qualified on Rubin D128 paged half Q>1."""
    return device_cc == (10, 7) and d_shape == (128, 128) and not fp8 and thd and paged and max_q > 1


def supports_thd_split_sink(d_shape, *, device_cc, fp8, thd, paged, max_q):
    """Half partials fold the sink once in the final packed combine."""
    return supports_paged_split_sink(d_shape, device_cc=device_cc, fp8=fp8, thd=thd, paged=paged, max_q=max_q) or (
        device_cc in ((10, 0), (10, 3)) and d_shape == SM100_NATIVE_D576_SHAPE and not fp8 and thd and max_q > 0
    )


def supports_paged_prefill_cga1(d_shape, *, device_cc, fp8, thd, paged, split_kv, max_q):
    """The two-slab D128 prefill body, distinct from its split/decode tile."""
    # SM100 Q=1 keeps the existing ragged-Q decode/combine contract.
    return (device_cc == (10, 7) or (device_cc == (10, 0) and max_q > 1)) and d_shape == (128, 128) and not fp8 and thd and paged and split_kv == 1


def supports_paged_d256_pack_gqa(d_shape, *, device_cc, fp8, thd, paged, cga, split_kv):
    """The shared D256 packed-head path serves paged half THD on SM100/103/107."""
    return device_cc in ((10, 0), (10, 3), (10, 7)) and d_shape == (256, 256) and not fp8 and thd and paged and cga in (None, 2) and split_kv == 1


def supports_nonpaged_d128_thd_prefill(d_shape, *, device_cc, fp8, thd, paged, cga, split_kv):
    """Native Rubin D128 unsplit THD prefill, with either CTA width."""
    return device_cc == (10, 7) and d_shape == (128, 128) and not fp8 and thd and not paged and cga in (None, 1, 2) and split_kv == 1


def supports_paged_d128_partial_pack_gqa(d_shape, *, device_cc, fp8, thd, paged, cga, split_kv):
    """Rubin's two-CTA D128 paged half body packs a divisor of the GQA group."""
    return device_cc == (10, 7) and d_shape == (128, 128) and not fp8 and thd and paged and cga in (None, 2) and split_kv == 1


def supports_nonpaged_d128_thd_pack_gqa(d_shape, *, device_cc, fp8, thd, paged, cga, split_kv):
    """Native Rubin ragged D128 packing shares the prefill CTA domain."""
    return supports_nonpaged_d128_thd_prefill(d_shape, device_cc=device_cc, fp8=fp8, thd=thd, paged=paged, cga=cga, split_kv=split_kv)


def d256_mxfp8_pair_leg(*, device_cc, pertensor, thd, paged, split_kv) -> bool:
    """The d256 MXFP8 legs that run the 2-CTA pair (cta_group::2, K/V halved per CTA) on cc 10.7: dense queries,
    unsplit, not THD, not paged.  Measured 2026-10-12 on cc 10.7 (B1 H32/8 GQA, CUDA-graph replay interleaved with the
    one-CTA build, O bitwise identical between the widths): dense S=8k / 16k / 32k -8.5 / -10.1 / -13.3 %, causal (LPT)
    -5.9 / -6.6 / -7.4 %.  The kernel's THD arm at the pair is untested, the split leg keeps one-CTA partials and the
    paged loader was not measured at the pair, so those legs stay at cga1; per-tensor FP8 d256 keeps cga1 as well."""
    return tuple(device_cc or ()) == (10, 7) and not pertensor and not thd and not paged and (split_kv or 1) == 1


def _validate_params(flavor: str, k: TemplateParams, *, scale_prefolded_wired: bool = False, d512_partial_pack_wired: bool = False) -> None:
    if k.qsa_block_topk or k.qsa_block_size or k.qsa_softmax_groups:
        # The loader routes an index-list record (qsa_block_topk != 0) to the cc 10.7 sparse kernel
        # (config_sm107.make_cfg_d256_sparse); a dense template must never consume one -- it would read K/V densely and
        # silently ignore the block list.  The selection is the FUNCTION, so this is a decline, not a knob fallback (the
        # sparse kernel's own softmax-split knob qsa_softmax_groups has no meaning on a dense template either).
        raise ValueError(
            f"{flavor}: qsa_block_topk={k.qsa_block_topk} / qsa_block_size={k.qsa_block_size} / qsa_softmax_groups={k.qsa_softmax_groups} select the "
            f"index-list sparse kernel; the dense templates do not consume them"
        )
    if k.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_QKV must be E4M3/E5M2/BF16/FP16 (0..3); got {k.dtype_qkv}")
    fp8 = k.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if fp8 and flavor not in ("d64", "d128", "d192", "d256", "d512"):
        raise ValueError(f"{flavor}: FP8/MXFP8 inputs (DTYPE_QKV 0/1) are only supported on d64, d128, d192, d256, and d512")
    if k.softmax_f16 and not fp8:
        raise ValueError(f"{flavor}: softmax_f16 is a quantized-kernel (FP8 / MXFP8) specialization (f16/bf16 softmax already runs the f32 pipeline)")
    if k.softmax_scale_prefolded and not scale_prefolded_wired:
        # The cc 10.7 kernels carry the arm (config_sm107.SM107_SCALE_PREFOLDED_FLAVORS; the d512 2x2 twin
        # validates here with the flag); this line's bodies apply the scale in-kernel.
        raise ValueError(f"{flavor}: softmax_scale_prefolded is served by the cc 10.7 kernels only (this line applies the scale in-kernel)")
    if k.pv_bf16 and (not fp8 or flavor not in ("d128", "d192")):
        raise ValueError(f"{flavor}: pv_bf16 is an experimental MXFP8 D128/D192 specialization")
    dtype_o = k.dtype_qkv if k.dtype_o < 0 else k.dtype_o
    if dtype_o not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16, DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        raise ValueError(f"{flavor}: DTYPE_O must be 0..5; got {dtype_o}")
    if dtype_o in (DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        if not fp8:
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) requires FP8 inputs")
        if flavor != "d128":
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) is only supported on d128")
        if k.thd_varlen or k.seq_q_lens_present or k.split_kv > 1 or k.pack_gqa or k.paged_kv:
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) serves dense (unpaged), unsplit, unpacked graphs only")
    if not fp8 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires DTYPE_O == DTYPE_QKV; got dtype_o={dtype_o}")
    if k.window_left is not None and k.window_left < 0:
        raise ValueError(f"{flavor}: window_left must be >= 0 (or None for unbounded); got {k.window_left}")
    if k.window_right is not None and k.window_right < 0:
        raise ValueError(f"{flavor}: window_right must be >= 0 (or None for unbounded); got {k.window_right}")
    if k.bottom_right:
        if k.window_right is None:
            raise ValueError(f"{flavor}: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if k.thd_varlen and not k.seq_kv_lens_present:
        raise ValueError(f"{flavor}: THD/varlen requires SEQ_KV_LENS_PRESENT (per-sequence padded masking)")
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    if k.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: only SCHED_NATURAL (0) / SCHED_LPT (1) / SCHED_LPT_L2 (2) are wired up; got {k.sched_policy}")
    if k.cta_mma not in (1, 2):
        raise ValueError(f"{flavor}: cta_mma must be 1 (cga1) or 2 (cga2); got {k.cta_mma}")
    if k.decode_tile and flavor != "d64":
        # The loader routes a decode_tile record to sm100/decode_d128_f16.py at
        # d_flavor=64 (make_cfg_d64_decode); no other template consumes it.
        raise ValueError(f"{flavor}: decode_tile selects the d64 decode tile; this template does not consume it")
    if k.decode_quad and not (k.decode_tile and k.early_q):
        raise ValueError(f"{flavor}: decode_quad is the d64 decode tile's quadrant softmax; it needs decode_tile and early_q")
    if k.dual_kv_stream and (flavor != "d128" or k.dtype_qkv > 1 or k.cta_mma != 1 or k.thd_varlen or k.pv_bf16):
        # Two KV streams over one Q tile: the per-tensor FP8 d128 pipeline at cga1
        # on a dense or paged graph (sm100/prefill_d128_fp8.py); nothing else
        # consumes the record (the sibling templates raise on it at import).
        raise ValueError(
            f"{flavor}: dual_kv_stream is the per-tensor FP8 d128 cga1 dense/paged decode specialization; got dtype_qkv={k.dtype_qkv} cta_mma={k.cta_mma} thd={k.thd_varlen}"
        )
    if k.decode_q_tile:
        # The loader routes a decode_q_tile record to sm100/decode_d256_f16.py
        # (make_cfg_d256_decode); a prefill template must never consume one.
        raise ValueError(f"{flavor}: decode_q_tile={k.decode_q_tile} selects the d256 decode tile; the prefill templates do not consume it")
    # A flavor must explicitly consume split_kv / cta_mma. Accepting either
    # elsewhere would silently ignore it; for split_kv that is also WRONG: the
    # caller sizes an (S*B)-batch partial workspace and runs the combine, while the kernel keeps
    # writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
    # than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
    # corrupt the result instead of dropping out.  Reject at the door.
    if k.split_kv != 1 and flavor not in _SPLIT_KV_FLAVORS:
        raise ValueError(f"{flavor}: split_kv is not implemented on this flavor (got {k.split_kv}); supported: {sorted(_SPLIT_KV_FLAVORS)}")
    # D256 selects its topology from the input family rather than exposing a
    # free CTA-MMA knob: half inputs stay CTA2, while FP8/MXFP8 use CTA1.
    d256_quantized_cta1 = flavor == "d256" and fp8 and k.cta_mma == 1
    if k.cta_mma != 2 and flavor not in _CTA_MMA_FLAVORS and not d256_quantized_cta1:
        raise ValueError(f"{flavor}: cta_mma is not selectable on this flavor (got {k.cta_mma}); supported: {sorted(_CTA_MMA_FLAVORS)}")
    if flavor == "d192" and k.split_kv > 1 and k.cta_mma != 2:
        raise ValueError("d192: split_kv > 1 is validated only with cta_mma=2")
    if k.split_kv < 1:
        raise ValueError(f"{flavor}: split_kv must be >= 1 (1 = KV-split off); got {k.split_kv}")
    if k.split_kv > 1:
        # Each of these would need extra machinery in the combine pass, so the
        # backstop rejects them rather than silently producing a wrong answer.
        if k.thd_varlen and not (
            (flavor == "d64" and k.decode_tile and k.cta_mma == 1 and k.paged_kv and not fp8)
            or (flavor == "d256" and k.cta_mma == 2 and not fp8 and not k.pack_gqa)
            or (flavor == "d512" and k.d_flavor == 576 and k.cta_mma == 2 and not fp8)
            or (
                flavor == "d128"
                and k.cta_mma == 1
                and not fp8
                and (k.single_q_head_dim == 128 or (k.single_q_head_dim == 192 and not k.paged_kv and not k.pack_gqa))
            )
        ):
            raise ValueError(f"{flavor}: split_kv > 1 is dense-only (THD packs its own flat grid)")
        if k.has_sink and fp8:
            # The half kernels write sink-free partials under a split and the
            # combine folds the sink once; the FP8 epilogues still fold it per
            # tile, so a split would count it once per split.
            raise ValueError(f"{flavor}: split_kv > 1 with attention sink is not supported for FP8 (the sink would be counted once per split)")
    lpt_head_groups = (1, 8, 16, 32) if flavor == "d256" else (1, 8, 16)
    if k.lpt_head_group not in lpt_head_groups:
        raise ValueError(f"{flavor}: LPT_HEAD_GROUP must be one of {lpt_head_groups}; got {k.lpt_head_group}")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.pack_gqa:
        if k.thd_varlen and not (
            not fp8
            and (
                (flavor == "d64" and k.decode_tile and k.cta_mma == 1 and k.paged_kv and k.split_kv > 1)
                or (
                    flavor == "d128"
                    and (
                        ((k.cta_mma == 2 or (k.cta_mma == 1 and k.paged_kv)) and k.split_kv == 1)
                        or (k.cta_mma == 1 and k.split_kv > 1 and k.single_q_head_dim == 128)
                    )
                )
                or (flavor == "d256" and k.paged_kv and k.cta_mma == 2 and k.split_kv == 1)
                or (flavor == "d512" and k.d_flavor == 576 and k.cta_mma == 2)
            )
        ):
            raise ValueError(
                f"{flavor}: THD PackGQA requires half d128 cga2 unsplit/cga1 split, paged half d64 cga1 split, "
                "paged half d256 cga2 unsplit, or native half d576 cga2"
            )
    if k.ragged_q:
        # The decode tile's ragged-Q leg (sm100/decode_d128_f16.py): dense grid
        # over the declared batch, Q rows at the ragged offsets, final O / Stats
        # placed by the split combine -- so the split path is mandatory and the
        # K/V side must be the page pools (a ragged K/V needs the THD leg).
        if flavor != "d128" or k.cta_mma != 1:
            raise ValueError(f"{flavor}: ragged_q is wired on the d128 decode tile only (cta_mma=1)")
        if k.thd_varlen:
            raise ValueError("d128 decode: ragged_q and thd_varlen are mutually exclusive")
        if k.split_kv < 2:
            raise ValueError("d128 decode: ragged_q rides the split path (the combine places the ragged O / Stats rows); split_kv must be >= 2")
        if not k.paged_kv:
            raise ValueError("d128 decode: ragged_q serves ragged Q/O/Stats over PAGED K/V only")
        if k.seq_q_lens_present:
            raise ValueError("d128 decode: ragged_q derives per-sequence Q lengths from the ragged offsets; seq_q_lens_present is dense-only")
    if k.paged_kv:
        if flavor not in _PAGED_KV_FLAVORS:
            raise ValueError(f"{flavor}: paged_kv is not implemented on this flavor; supported: {sorted(_PAGED_KV_FLAVORS)}")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: paged_kv requires seq_kv_lens_present (the per-batch KV length bounds the block-table walk)")
        # dtype_qkv alone cannot tell per-tensor FP8 (d128 wired) from MXFP8
        # (wired on every native flavor), so
        # the dtype family is NOT gated here: every kernel file WITHOUT the
        # PAGED_KV specialization raises at module scope on paged_kv=True
        # (next to its softmax_f16 guard), which is the backstop that cannot
        # silently read a page pool as dense K/V.
        # A K/V tile is loaded as a stack of page-sized row boxes (or one box
        # inside a page when the page is taller than the tile). Either way a
        # box must never straddle a page, and the 128 B swizzle atom is 8 rows.
        p = k.page_size
        if p < 8 or p % 8 != 0:
            raise ValueError(f"{flavor}: page_size must be a positive multiple of 8; got {p}")
        if (p < 128 and 128 % p != 0) or (p > 128 and p % 128 != 0):
            raise ValueError(f"{flavor}: page_size must divide the 128-row KV tile or be a multiple of it; got {p}")
    elif k.page_size:
        raise ValueError(f"{flavor}: page_size requires paged_kv=True")
    if k.epilogue_gate:
        # Only the HALF d256 body of this line carries the gate seams (_EPILOGUE_GATE_FLAVORS above); the quantized d256
        # bodies and every other flavor would trace an ungated epilogue and silently return O instead of O * sigmoid(G).
        if flavor not in _EPILOGUE_GATE_FLAVORS or fp8:
            raise ValueError(
                f"{flavor}: epilogue_gate is wired on the f16/bf16 d256 body of this line only (the paged / THD / PackGQA prefill compiled "
                f"for cc 10.7); the quantized d256 bodies and the other flavors are served by the SM107 d256 kernels (config_sm107)"
            )
        if k.split_kv > 1:
            # A split's fp32 partials bypass the gated epilogue and the combine would write the un-gated O (the d256 decode
            # tile's split applies the gate in ITS combine; this prefill body has no such arm).
            raise ValueError(f"{flavor}: epilogue_gate with split_kv > 1 is not served on this body (the split partials bypass the gated epilogue)")
    if getattr(k, "mma_2x2", False):
        # The 2x2-datapath kernel exists for the half d512 flavor only (sm100/prefill_d512_f16_2x2.py).
        if flavor != "d512":
            raise ValueError(f"{flavor}: mma_2x2 selects the d512 2x2-datapath kernel; the other flavors do not consume it")
        if fp8:
            raise ValueError("d512: mma_2x2 is wired for BF16/FP16 inputs only (the fp8 / mxfp8 d512 kernels stay role-split)")
        if k.paged_kv and k.d_flavor != 576:
            raise ValueError("d512: mma_2x2 does not serve paged KV; the three-chunk d576 specialization is required")
        if k.pack_gqa and not d512_partial_pack_wired and (k.qh_per_kh <= 0 or _D512_2X2_TILE_M % k.qh_per_kh != 0):
            raise ValueError(f"d512: mma_2x2 packs whole GQA groups into its {_D512_2X2_TILE_M}-row tile; qh_per_kh ({k.qh_per_kh}) must divide it")


# Q rows per CTA of the d512 2x2-datapath kernel (CfgD512X2.TILE_M); named here so
# _validate_params can state the PackGQA domain before the dataclass is defined.
_D512_2X2_TILE_M = 64


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding: the kernels
    fold trace-time branches on these bits; the band VALUES ride the CFG's
    WINDOW_LEFT / WINDOW_RIGHT fields (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present or params.kv_tail_mask:
        flags |= MASK_PADDED
    return flags


# ---------------------------------------------------------------------------
# Shared derivation helpers
# ---------------------------------------------------------------------------


def bpe(dtype: int) -> int:
    """Bytes per STORAGE element: the block-scaled O codes are byte containers
    (E2M1 packs two per byte -- see ``o_pack_div``)."""
    if dtype in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        return 1
    return 2


def o_pack_div(dtype_o: int) -> int:
    """Logical O elements per storage byte-container: 2 for E2M1, else 1."""
    return 2 if dtype_o == DTYPE_O_NVFP4 else 1


def o_row_bytes(tile_o: int, dtype_o: int) -> int:
    return tile_o * bpe(dtype_o) // o_pack_div(dtype_o)


def tile_k_hw(dtype_qkv: int) -> int:
    if dtype_qkv <= 1:
        return 64
    return 16


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} not multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int, pack_div: int = 1) -> int:
    return 128 if (tile_o * bpe_o // pack_div) % 128 == 0 else 64


def bshd_compact(shape_bhsd: tuple, stride_bhsd: tuple) -> bool:
    """True when a logical BHSD ``(b, h, s, d)`` operand already IS the kernels'
    canonical BSHD-compact layout (batch, then tokens, then heads, then a
    contiguous head dim): nothing to declare, the artifact binds it as a view."""
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    return (bs, ss, hs, es) == (s * h * d, h * d, d, 1)


def bshd_zero_copy_stride(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int) -> Optional[tuple]:
    """The BSHD ``(b, s, h, d)`` stride tuple a dense prefill kernel is COMPILED
    at for a logical BHSD operand, or None.

    None means EITHER "compact, nothing to declare" (``bshd_compact``) OR "a
    layout the kernels cannot bind zero-copy"; a caller that must tell the two
    apart tests ``bshd_compact`` first (the epilogue gate does -- it has no
    copy fallback). The rules describe the kernels' TMA layout requirements,
    checked before compilation so the engine rows
    (``engines.mismatch`` via ``config_sm107.epilogue_gate_layout_declarable``)
    and the standalone adapter's admission and prepared binder
    judge a layout with ONE function (rule 8b lockstep):

      * the head dim is innermost-contiguous (stride 1);
      * the batch, seq and head strides are 16-byte multiples (TMA global-stride rule;
        the DSL floors a misaligned stride to TMA units, so a violation would mis-address);
      * the declaration is TOKEN-MAJOR and COVERING: head >= d, seq >= h*head,
        batch >= s*seq.  A head-major nest (a torch-contiguous ``[B, H, S, D]``:
        seq stride d < h*head) returns None because the kernels' TMA
        descriptors are built in BSHD order and no other nesting has been
        validated zero-copy; an overlapping declaration returns None because
        it would alias distinct rows onto one address (a write race on O).

    ``graph_analyzer.dense_layout_ok`` is deliberately WIDER (any B/H/S stride
    order, any alignment): that is the contract of the copy-normalising Q/K/V/O
    path, not of a zero-copy binding.
    """
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    if (bs, ss, hs, es) == (s * h * d, h * d, d, 1):
        return None  # compact: nothing to declare
    if es != 1:
        return None
    per16 = 16 // elem_bytes
    if ss % per16 or hs % per16 or (b > 1 and bs % per16):  # every non-innermost TMA global stride, the batch one included
        return None
    if hs < d or ss < h * hs or bs < s * ss:
        return None
    return (bs, ss, hs, es)


def canonical_bhsd_strides(shape_bhsd: tuple, stride_bhsd: tuple) -> tuple:
    """``stride_bhsd`` with every extent-1 axis given the covering canonical stride.

    A stride on an axis of extent 1 is never stepped, so a buffer's own value there is
    unobservable (torch's ``is_contiguous`` wildcards such axes and a one-KV-head K keeps
    whatever stride its allocation had); the kernels' layout rules below are stated for
    stepped axes, so the singleton ones are spelled the way a compact BSHD buffer would
    spell them before the rules are applied."""
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    if h == 1:
        hs = d
    if s == 1:
        ss = h * hs
    if b == 1:
        bs = s * ss
    return (bs, hs, ss, es)


@lru_cache(maxsize=256)
def dense_bind_strides(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int) -> Optional[tuple]:
    """The ``(batch, seq, head)`` element strides a dense prefill kernel binds a logical BHSD
    operand at zero-copy, or None when the layout needs a repack: singleton axes canonicalized,
    then BSHD-compact or ``bshd_zero_copy_stride``. ONE predicate for the lowering's decision to
    attach the prepared launch and for the binder's per-call admission (rule 8b lockstep).

    Only this pure layout result is cached; the binder still checks each call's dtype,
    device, address alignment, observed span and compiled shape envelope. The bounded
    cache retains no buffers, pointers or per-call frames."""
    st = canonical_bhsd_strides(shape_bhsd, stride_bhsd)
    if not (bshd_compact(shape_bhsd, st) or bshd_zero_copy_stride(shape_bhsd, st, elem_bytes) is not None):
        return None
    bs, hs, ss, _es = st
    return (bs, ss, hs)


def rescale_threshold(dtype_qkv: int) -> float:
    return 4.0 if dtype_qkv <= 1 else 8.0


def pack_gqa_group_size(qh_per_kh: int, tile_m: int = 128, *, partial: bool = False, requires_divisible: bool = True) -> int:
    """Heads packed into one Q tile row-group for the GQA ratio ``G = qh_per_kh``.

    Full contract (``partial=False``): ``G`` when it divides ``tile_m``, else 0
    (nothing can be packed).  Partial contract (``partial=True``, the d128 and
    d256 f16 kernels): the largest divisor of ``G`` that divides ``tile_m`` --
    ``gcd(G, tile_m)`` -- so a 12-head group packs 4 heads per token row-group
    (three packed heads per KV head), a 6-head group packs 2, and a group with
    no factor in common with the tile (3, 5, ...) yields 1 = unpacked.
    Without the divisibility requirement (``requires_divisible=False``, the
    SM120 kernels): ``G`` whenever it fits the tile, ``tile_m % G != 0`` included --
    ``tile_m // G`` tokens per tile, the rows past ``(tile_m // G) * G`` dead
    (G = 6 at 64 rows: ten tokens in 60 live rows) -- else 0.  MHA (``G == 1``)
    is the identity, 1.
    """
    if qh_per_kh <= 0:
        return 0
    if tile_m % qh_per_kh == 0 or (not requires_divisible and qh_per_kh <= tile_m):
        return qh_per_kh
    return math.gcd(qh_per_kh, tile_m) if partial else 0


def pack_gqa_supported(h_q: int, h_kv: int, tile_m: int = 128, *, partial: bool = False, requires_divisible: bool = True) -> bool:
    """Whether a PACK_GQA=1 plan is HONORABLE for this head count: the group
    divides the tile (full packing), -- ``partial`` -- shares a factor > 1
    with it, or -- without ``requires_divisible`` -- fits it.  MHA (G == 1) is the bit-exact
    unpacked fold, so it is honorable too; a group that cannot pack at all
    (G=3 at tile_m=128) is declined rather than silently run unpacked
    (frost/README.md rule 5)."""
    if h_q <= 0 or h_kv <= 0 or h_q % h_kv != 0:
        return False
    g = h_q // h_kv
    p = pack_gqa_group_size(g, tile_m, partial=partial, requires_divisible=requires_divisible)
    return p == g or p > 1


def _pack_g(params: TemplateParams, tile_m: int, *, partial: bool) -> int:
    """``Cfg.PACK_G`` for a flavor: the packed group size, validated the way
    ``engines.mismatch`` admits it (full contract, or partial where the kernel
    carries it), so a knob that passed the gate can never reach a kernel that
    would silently run it unpacked."""
    if not params.pack_gqa:
        return 1
    g = int(params.qh_per_kh)
    p = pack_gqa_group_size(g, tile_m, partial=partial)
    if p == 0 or (p == 1 and g != 1):
        how = "share a factor with" if partial else "divide"
        raise ValueError(f"qh_per_kh ({g}) must {how} TILE_M ({tile_m}) when PACK_GQA is enabled")
    return p


def cga_tile_m(d_qk: int, cta_mma: Optional[int] = None) -> int:
    """Q rows one cluster covers for a flavor: TILES_Q * TILE_M * CTA_MMA.

    ``cta_mma`` overrides the flavor default when the selected kernel exposes a
    CGA-width knob (D192). D128 CGA1 here models the 128-row decode/split
    tile; the paged prefill leg has two slabs and its caller accounts for them.
    """
    cls = _GEOMETRY_BY_D_QK[d_qk]
    if d_qk == 128 and cta_mma == 1:
        # Default D128 half CGA1 geometry is the decode/split tile
        # (sm100/decode_d128_f16.py, TILES_Q=1), not the two-slab paged prefill.
        cls = CfgD128Decode
    return cls.TILES_Q * cls.TILE_M * (cls.CTA_MMA if cta_mma is None else cta_mma)


def cga_ctas(d_qk: int, cta_mma: Optional[int] = None) -> int:
    """Physical CTAs per cluster, including the D512 non-MMA role CTAs.

    Public CGA selects the MMA width. It is not a physical launch count for
    the role-split D512 flavor, whose CGA_M/CTA_MMA ratio is two.
    """
    cls = _GEOMETRY_BY_D_QK[d_qk]
    return cls.CGA_M // cls.CTA_MMA * (cls.CTA_MMA if cta_mma is None else cta_mma)


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"inner {inner_bytes} not multiple of swz {swz_b}")
    return inner_bytes // swz_b


@dataclass(frozen=True)
class TmaIters:
    """TMA iteration granularity derived from a flavor's tile geometry."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, getattr(cfg, "BPE_V", cfg.BPE), cfg.V_SWZ_BYTES)
    return TmaIters(
        QK_ITERS=qk,
        VO_ITERS=vo,
        QK_GRANU_ELEMS=cfg.TILE_K // qk,
        VO_GRANU_ELEMS=cfg.TILE_O // vo,
    )


# ---------------------------------------------------------------------------
# d256 flavor — d_qk = d_v = 256, SM100 (Blackwell), Qwen-class models
# Half inputs use CTA2 collective MMA; FP8/MXFP8 use an independent CTA1 tile.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD256:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    SPLIT_PIPELINE: int = 0

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4
    FUSED_CORR_SPLIT_P: int = 0

    SOFTMAX_REGS: int = 240
    # The DSL traces the unreachable WG1 dispatch for one-WG specializations,
    # so its placeholder must still be a legal setmaxnreg operand.
    SOFTMAX_WG1_REGS: int = 40
    CORRECTION_REGS: int = 96
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
    STATS_LOG2: int = 0  # LSE stored in base 2 (stats_use_log2)
    BOTTOM_RIGHT: int = 0  # band diagonal anchored bottom-right

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    READ_TILE_ARRIVERS: int = ((1 * 4) + 4 + 2) * (2 * 1) + (2 // 2)

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  See TemplateParams.split_kv.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0

    # The graph's GQA ratio H_q / H_kv -- the bottom-right diagonal and the
    # LPT_L2 head grouping are derived from it.
    QH_PER_KH: int = 1
    # Heads packed into one Q tile row-group under PACK_GQA (the kernel's
    # HEADS_PER_TILE): QH_PER_KH when the group divides TILE_M, else -- the
    # d128 / d256 f16 kernels only -- its largest divisor that does
    # (pack_gqa_group_size); QH_PER_KH // PACK_G packed heads then share one KV
    # head.  1 when unpacked.  Distinct from QH_PER_KH on purpose: conflating
    # the pack size with the GQA ratio would mis-mask MTP rows.
    PACK_G: int = 1

    # Paged KV cache; see TemplateParams.paged_kv.  PAGE_SIZE tokens per page.
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0

    # Fused epilogue gate (O := O * sigmoid(G), TemplateParams.epilogue_gate), read by the SHARED f16/bf16 d256 body only
    # (sm100/prefill_d256_f16.py; every seam folds on cutlass.const_expr(CFG.EPILOGUE_GATE), so 0 traces the pre-gate
    # instruction stream byte for byte -- the render gate of the paged gate).  Appended: every existing record
    # keeps its value.  GATE_BPE: bytes per gate element -- the gate is Q-dtype on the half body (2 B); every gate
    # constant in the kernel (TMA box, swizzle, subtile walk) derives from THIS, never from BPE_O (config_sm107 has the
    # e4m3-O lesson).
    EPILOGUE_GATE: int = 0
    GATE_BPE: int = 2

    # Half causal schedule: the dedicated correction WG also generates P1.
    # Unlike MXFP8 FUSED_CORR_SPLIT_P, this retains one softmax WG and four
    # correction warps. Append-only internal topology, not an autotune knob.
    HALF_FUSED_CORR_SPLIT_P: int = 0


def d256_gate_smem_bytes(cfg: CfgD256) -> int:
    """The SMEM the GATED d256 half rendering allocates, from the SAME constants its arrays are declared with:
    sQO (Q u O alias) + the K ring + the V ring + the 64 KiB gate staging tile + the fixed overhead (barriers, the
    scheduler ring, the TMEM pointer, the 1024-B alignment pad of the first array; 2 KiB, config_sm107's figure)."""
    operands = (
        cfg.TILE_M * cfg.TILE_K + cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) + cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA)
    ) * cfg.BPE
    gate = cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * cfg.GATE_BPE if cfg.EPILOGUE_GATE else 0
    return operands + gate + 2 * 1024


# The usable Rubin carveout the gated d256 rendering is checked against (config_sm107.SMEM_USABLE_BYTES: 320 KiB of the
# 327 KiB capacity, never the capacity -- the d192 f16 50 %-zeros lesson).  The gated rendering (~258 KiB at STAGES_KV=2)
# is past the 227 KiB SM100 cap, so it is a cc 10.7 rendering by construction; the adapter routes it there only.
_D256_GATE_SMEM_USABLE_BYTES = 320 * 1024


def _validate_cfg_d256(cfg: CfgD256) -> None:
    """Consistency checks on the (mostly hardcoded) d256 geometry."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    split_p = cfg.SOFTMAX_WARPGROUPS == 2
    fused_corr_split_p = cfg.FUSED_CORR_SPLIT_P == 1
    split_p_supported = fp8 and (cfg.MASK_FLAGS == MASK_NONE or ((cfg.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and cfg.BOTTOM_RIGHT == 0))
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d256: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (
            cfg.MMA_REGS + (0 if fused_corr_split_p else cfg.CORRECTION_REGS) + cfg.SOFTMAX_REGS + (cfg.SOFTMAX_WG1_REGS if split_p else 0) <= 512,
            "d256: register budget over 512",
        ),
        (
            cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0 and cfg.SOFTMAX_WG1_REGS % 8 == 0,
            "d256: per-role regs must be multiples of 8",
        ),
        (
            not cfg.HALF_FUSED_CORR_SPLIT_P
            or (
                cfg.DTYPE_QKV == DTYPE_BF16
                and cfg.MASK_FLAGS == MASK_CAUSAL
                and not cfg.BOTTOM_RIGHT
                and not cfg.WINDOW_LEFT
                and not cfg.WINDOW_RIGHT
                and not cfg.THD_VARLEN
                and not cfg.PAGED_KV
                and not cfg.EPILOGUE_GATE
                and cfg.SPLIT_KV == 1
                and not split_p
                and not fused_corr_split_p
                and cfg.CORRECTION_WARPS == 4
            ),
            "d256: half fused correction requires dense unsplit BF16 top-left causal",
        ),
        (cfg.CGA_M == cfg.CTA_MMA, "d256 flavor pairs CGA_M with CTA_MMA"),
        (cfg.CTA_MMA == (1 if fp8 else 2), "d256 SM100: FP8 requires CTA1; BF16/FP16 requires CTA2"),
        (cfg.STAGES_KV == 2, "d256 SM100 uses two full/half-width KV stages"),
        (
            not cfg.EPILOGUE_GATE or (not fp8 and cfg.SPLIT_KV == 1 and cfg.GATE_BPE == cfg.BPE == 2 and cfg.TILES_Q == 1),
            "d256: the epilogue gate is the half body's unsplit arm (GATE_BPE == BPE == 2, TILES_Q == 1: one gate tile per Q tile)",
        ),
        (
            not cfg.EPILOGUE_GATE or d256_gate_smem_bytes(cfg) <= _D256_GATE_SMEM_USABLE_BYTES,
            f"d256: the gated rendering's SMEM ({d256_gate_smem_bytes(cfg) // 1024} KiB: Q/K/V/O + the 64 KiB gate tile + 2 KiB fixed) exceeds the "
            f"{_D256_GATE_SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d256: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d256: V/O swizzle out of range"),
        (cfg.TILES_Q == 1, "d256 pipeline mandates TILES_Q == 1"),
        (
            not fused_corr_split_p or not (cfg.MASK_FLAGS & MASK_SWA),
            "d256/d512: SWA P reuse and split-P correction require different alias phases",
        ),
        (not split_p or split_p_supported, "d256: unsupported split-P specialization"),
        (cfg.SOFTMAX_WARPGROUPS == 2 if cfg.MASK_FLAGS == MASK_NONE and fp8 else True, "d256: dense FP8 must split P generation"),
        (cfg.TOTAL_WARPS == (12 if fused_corr_split_p else 16 if split_p else 12), "d256: role layout and warp count disagree"),
        (not fused_corr_split_p or (split_p and cfg.CORRECTION_WARPS == 0), "d256: fused split-P must replace the correction warp group"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if fp8 else 16),
            "d256: TILE_K_HW must be 32 for FP8 and 16 for BF16/FP16",
        ),
        (fp8 or cfg.DTYPE_O == cfg.DTYPE_QKV, "d256: half input requires DTYPE_O == DTYPE_QKV"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def d256_square_br_as_tl(params: TemplateParams, *, s_q: int, s_kv: int) -> bool:
    """Whether a D256 bottom-right mask is exactly top-left causal."""

    return (
        params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
        and not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
    )


def canonicalize_d256_lowering(params: TemplateParams, *, s_q: int, s_kv: int) -> TemplateParams:
    """Apply strictly equivalent D256 lowering canonicalizations."""

    return replace(params, bottom_right=False) if d256_square_br_as_tl(params, s_q=s_q, s_kv=s_kv) else params


def canonicalize_d512_mxfp8_lowering(params: TemplateParams, *, s_q: int, s_kv: int) -> TemplateParams:
    """Canonicalize an exactly square D512 MXFP8 causal diagonal."""

    square_bottom_right = (
        not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
    )
    return replace(params, bottom_right=False) if square_bottom_right else params


def derive_d256_internal_params(
    params: TemplateParams,
    *,
    pertensor: bool,
    batch_size: int,
    h_q: int,
    s_q: int,
) -> TemplateParams:
    """Derive D256-private codegen fields after public knobs are fixed."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if not fp8 or params.thd_varlen:
        return params

    pack_gqa_ratio = params.qh_per_kh if params.pack_gqa else 1
    groups = batch_size * h_q // pack_gqa_ratio
    lpt_head_group = 32 if pertensor and groups % 32 == 0 else 8 if not pertensor and groups % 8 == 0 else 1
    q_tiles = (s_q + 255) // 256 if pertensor else 0
    mask_flags = _mask_flags_from(params)
    pt_lpt_l2 = pertensor and params.sched_policy == SCHED_LPT_L2 and mask_flags == MASK_CAUSAL and not params.bottom_right and q_tiles >= 16
    return replace(
        params,
        lpt_head_group=lpt_head_group,
        lpt_l2_size_mib=32 if pt_lpt_l2 else 0,
    )


def _make_cfg_d256(params: TemplateParams, *, mxfp8: bool) -> Tuple[CfgD256, TmaIters]:
    _validate_params("d256", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    cga = 1 if fp8 else 2
    if params.cta_mma != cga:
        raise ValueError(f"d256: {'FP8/MXFP8' if fp8 else 'BF16/FP16'} requires cta_mma={cga}; got {params.cta_mma}")
    mask_flags = _mask_flags_from(params)
    pt_plain_top_left_causal = (
        not mxfp8 and (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_left and not params.window_right
    )
    # The fused correction/split-P schedule is the strict top-left causal fast
    # path. Right-band widening uses the generic masked schedule; forcing the
    # widened specialization through this path makes CUTLASS DSL 4.7 lowering
    # grow pathologically without changing the supported mask semantics.
    strict_top_left_causal = (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_right
    split_p = fp8 and (mask_flags == MASK_NONE or pt_plain_top_left_causal)
    pt_thd_split_p = pt_plain_top_left_causal and bool(mask_flags & MASK_PADDED)
    fused_corr_split_p = mxfp8 and strict_top_left_causal
    half_fused_corr_split_p = (
        not mxfp8
        and params.dtype_qkv == DTYPE_BF16
        and mask_flags == MASK_CAUSAL
        and strict_top_left_causal
        and not params.window_left
        and not params.thd_varlen
        and not params.paged_kv
        and not params.epilogue_gate
        and params.split_kv == 1
    )
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)

    # These register profiles are coupled to the kernel's role topologies;
    # they are not independent public knobs.
    pt_e4_causal_regs = not mxfp8 and params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_CAUSAL and not params.bottom_right
    pt_e5_causal_regs = not mxfp8 and params.dtype_qkv == DTYPE_E5M2 and mask_flags == MASK_CAUSAL and not params.bottom_right
    softmax_regs, softmax_wg1_regs, correction_regs = 240, 136 if split_p else 40, 96
    if half_fused_corr_split_p:
        # 4 warps each: 240 softmax + 216 correction/P1 + 48 other = 504.
        # WG1's legal tracing placeholder does not represent a live warpgroup.
        correction_regs = 216
    elif fused_corr_split_p:
        softmax_regs, softmax_wg1_regs, correction_regs = 248, 216, 64
    elif mxfp8 and split_p:
        softmax_regs, softmax_wg1_regs, correction_regs = 256, 144, 72
    elif pt_thd_split_p:
        softmax_wg1_regs, correction_regs = 144, 88
    elif pt_e4_causal_regs:
        correction_regs = 64
        if split_p:
            softmax_regs = 216
            softmax_wg1_regs = 168
    elif pt_e5_causal_regs:
        correction_regs = 96 if split_p else 112
    elif mxfp8 and mask_flags != MASK_NONE:
        correction_regs = 64
        if params.dtype_qkv == DTYPE_E5M2 or (params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_CAUSAL):
            softmax_regs = 248
    elif not mxfp8 and fp8 and mask_flags != MASK_NONE:
        correction_regs = 104
    elif not mxfp8 and params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_NONE:
        correction_regs = 88

    if fused_corr_split_p:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 12, 4, 64, 8, 11
    elif split_p:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 16, 4, 8, 12, 15
    else:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id = 12, 64, 4, 8
        read_tile_arrivers = 11 if fp8 else 21

    cfg = CfgD256(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        # FP8 uses one M128 CTA per work unit. With a full K/V slice per CTA,
        # two KV stages consume the same SMEM payload as CTA2's four half-slices.
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        Q_SWZ_BYTES=q_swz_bytes(256, b),
        K_SWZ_BYTES=q_swz_bytes(256, b),
        V_SWZ_BYTES=v_swz_bytes(256, 1 if fp8 else 2, b),
        O_SWZ_BYTES=o_swz_bytes(256, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32 if fp8 else tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=32 if fp8 else tile_k_hw(params.dtype_qkv),
        STAGES_KV=2,
        SOFTMAX_WARPGROUPS=2 if split_p or fused_corr_split_p else 1,
        CORRECTION_WARPS=0 if fused_corr_split_p else 4,
        FUSED_CORR_SPLIT_P=1 if fused_corr_split_p else 0,
        HALF_FUSED_CORR_SPLIT_P=int(half_fused_corr_split_p),
        # Forty registers spill the producer/scheduler's live metadata under
        # the fused role split; 48 stays within the 64K register file.
        MMA_REGS=48 if half_fused_corr_split_p else 40,
        TMALDG_REGS=48 if half_fused_corr_split_p else 40,
        TMASTG_REGS=48 if half_fused_corr_split_p else 40,
        SCHEDULER_REGS=48 if half_fused_corr_split_p else 40,
        OTHER_REGS=48 if half_fused_corr_split_p else 40,
        SOFTMAX_REGS=softmax_regs,
        SOFTMAX_WG1_REGS=softmax_wg1_regs,
        CORRECTION_REGS=correction_regs,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG1_BASE=softmax_wg1_base,
        CORR_WARP_BASE=correction_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=read_tile_arrivers,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        L2_SIZE_MIB=params.lpt_l2_size_mib or 60,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA is wired in the f16/bf16 d256 kernel; the fp8 / mxfp8
        # siblings pack HEADS_PER_TILE = QH_PER_KH and keep the full-ratio contract.
        PACK_G=_pack_g(params, CfgD256.TILE_M, partial=not fp8),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        # The fused epilogue gate (the half body reads it; _validate_params refused it on the quantized bodies above).
        EPILOGUE_GATE=int(params.epilogue_gate),
        GATE_BPE=2,
    )
    _validate_cfg_d256(cfg)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256(params, mxfp8=False)


def make_cfg_d256_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256(params, mxfp8=True)


# ---------------------------------------------------------------------------
# d256 DECODE tile — d_qk, d_v <= 256, SM100 (Blackwell), f16/bf16, swap-AB
# ---------------------------------------------------------------------------
#
# The prefill tile pays for 256 collective Q rows per KV tile whatever the
# number of live rows; a decode step has S_q * G of them (16 for Qwen3.5's
# 32/2 heads at S_q = 1).  The decode tile transposes the problem: the KV
# tokens are the MMA M axis (128 per tile, the standard TMEM lane = key
# layout) and the packed Q rows are the N axis (16 or 32), so BMM1 is
# S^T = K Q^T and BMM2 is O^T = V^T P^T, and the softmax reduces over lanes.
# One CTA per (KV-head group, batch, split) unit, cta_group::1, no cluster.

# Largest packed Q-row count the decode tile COMPILES for (N = 32 keeps the
# 3-slot 64 KiB K/V ring, the Q tile and the two P^T buffers inside 227 KiB).
D256_DECODE_MAX_Q_ROWS = 32
_D256_DECODE_Q_TILES = (16, 32)
# Largest packed Q-row count the adapter ROUTES onto the decode tile ON THE
# BLACKWELL LINE (SM100 / SM103) for every launch.  The 32-column tile (two
# softmax column groups over the same 128 TMEM lanes) is issue-bound per CTA:
# a lone CTA streams a KV tile in 2.8 us against the 16-column tile's 1.75 us
# (b=32 x 2 KV heads x 4096 keys on B200: 90 us unsplit where the prefill tile
# takes 66), so a launch too small to put ~130 of those streams in flight is
# served better by the prefill tile.  A launch that is big enough saturates HBM
# whatever the per-CTA rate, and there the wide tile is the faster one -- the
# 256-row prefill tile reaches ~78 % of the HBM roofline on these shapes, the wide
# tile the roofline itself (B200, d256 S_q x G = 32: -20 % at b=128 x 128K keys,
# -12 % at b=4 x 128K).  D256_WIDE_DECODE_MIN_UNITS / D256_WIDE_DECODE_MIN_KV_TILES mark
# that boundary (d256_wide_decode_launch); the Blackwell line routes S_q x G in
# (16, 32] onto the wide tile for launches past it.  The Rubin line routes the
# wide tile for every launch and its own 64-column tile, table-driven
# (config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES = (16, 32, 64),
# D256_DECODE_ROUTED_MAX_Q_ROWS = 64), in up to two TOKEN UNITS of the widest
# admitted width for a packed MTP step past the table
# (D256_DECODE_ROUTED_MAX_TOKEN_UNITS = 2, config_sm107.decode_d256_q_tile); the
# 32-column two-unit form was measured on cc 10.7 against the prefill body at the
# 24/2 geometry, the 64-column form on the Rubin performance class (its
# per-width stream rates: heuristics._DECODE_TILE_COST_BY_Q_TILE_SM107).
D256_DECODE_ROUTED_MAX_Q_ROWS = 16
# The 16-column tile: one unit covers S_q x pack_g <= 16 rows (today's decode step).
_D256_DECODE_NARROW_Q_TILE = 16


# What a launch must stream for the wide tile to pay on the Blackwell line: batch x
# KV-head groups (the decode tile's units: one CTA each before the split) and the
# 128-key KV tiles of all of them.  Measured on B200 (S_q x G = 32, qwen3.5-like
# d256, 2 heads x 16-wide groups and 1 head x 8-wide): wins from 8 units x 512
# tiles (b=4 x 64K keys) up, loses at one or two units (+1..+10 % at b=1, the
# serial combine of its 64-128 splits) and at 64 units x 16-32 tiles (+2..+7 %).
D256_WIDE_DECODE_MIN_UNITS = 8
D256_WIDE_DECODE_MIN_KV_TILES = 4096


def d256_wide_decode_launch(batch: int, groups: int, s_kv: int, windowed: bool = False) -> bool:
    """Whether a launch of ``batch`` x ``groups`` (KV-head groups, or heads when the
    graph does not pack) units over ``s_kv`` keys is one the 32-column decode tile
    serves faster than the prefill tile on the Blackwell line: at least
    ``D256_WIDE_DECODE_MIN_UNITS`` units streaming at least
    ``D256_WIDE_DECODE_MIN_KV_TILES`` 128-key tiles between them.  A windowed
    (SWA) graph never qualifies: its declared ``s_kv`` says nothing about the
    window's tiles."""
    if windowed or min(int(batch), int(groups), int(s_kv)) <= 0:
        return False
    units = int(batch) * int(groups)
    return units >= D256_WIDE_DECODE_MIN_UNITS and units * -(-int(s_kv) // 128) >= D256_WIDE_DECODE_MIN_KV_TILES


def decode_d256_q_tile(s_q: int, pack_g: int, routed_max_rows: int = D256_DECODE_ROUTED_MAX_Q_ROWS, max_token_units: int = 1, wide_launch: bool = False) -> int:
    """The decode tile's N extent -- the Q rows of ONE TOKEN UNIT -- for ``s_q``
    tokens packed ``pack_g`` heads per token (1 = unpacked), or 0 when the
    prefill tile serves the graph.

    ``routed_max_rows`` is the widest tile the caller's arch line routes for every
    launch (16 on SM100 / SM103; 32 on cc 10.7); ``wide_launch`` says the launch is
    one the 32-column tile pays for even where the line does not route it
    unconditionally (:func:`d256_wide_decode_launch`: the Blackwell line's
    boundary); ``max_token_units`` how many
    token units of ``N // pack_g`` consecutive tokens a (head group, batch,
    split) may be cut into (the kernel's ``Q_TOKEN_UNITS`` axis: each unit
    streams the KV range once more, so a line admits it only where it measured
    the trade -- 1 everywhere but cc 10.7, which measured the 24/2 MTP form and
    admits two units for every packed group the tile holds).  The rule: rows that fit
    the 16-column tile ride it in one unit (the KV streamed once, the cheapest
    per-CTA stream); past 16 rows the 32-column tile serves
    ``ceil(s_q / (32 // pack_g))`` units when that count is within the cap, the
    line routes it (or the launch qualifies) and the group fits the tile, else 0.
    With the defaults this is the SM100 rule for a launch too small for the wide
    tile (16 for S_q x G <= 16 rows, 0 above)."""
    s_q, pack_g = int(s_q), int(pack_g)
    if s_q <= 0 or pack_g <= 0:
        return 0
    rows = s_q * pack_g
    if rows <= _D256_DECODE_NARROW_Q_TILE:
        return _D256_DECODE_NARROW_Q_TILE
    wide = D256_DECODE_MAX_Q_ROWS
    if (wide > int(routed_max_rows) and not wide_launch) or pack_g > wide:
        return 0
    if decode_d256_q_units(s_q, pack_g, wide) > int(max_token_units):
        return 0
    return wide


def decode_d256_q_units(s_q: int, pack_g: int, n_q: int) -> int:
    """How many TOKEN UNITS (CTAs per (head group, batch, split)) the decode tile of N
    extent ``n_q`` launches for ``s_q`` tokens packed ``pack_g`` heads per token:
    ``ceil(s_q / (n_q // pack_g))`` -- the kernel's own derivation (``Q_BOX_TOKENS =
    N_Q // HEADS_PER_TILE``); 0 when ``n_q`` is 0 (the prefill tile) or the group does
    not fit the tile.  The decode split model's unit count is ``batch x head groups x
    this``."""
    s_q, pack_g, n_q = int(s_q), int(pack_g), int(n_q)
    if n_q <= 0 or pack_g <= 0 or s_q <= 0 or pack_g > n_q:
        return 0
    tokens_per_unit = n_q // pack_g
    return -(-s_q // tokens_per_unit)


@dataclass(frozen=True)
class CfgD256Decode:
    # KV tokens per MMA (the M axis); TILE_K / TILE_O are the head-dim envelopes.
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256
    # Packed Q rows per unit (the MMA N axis).
    N_Q: int = 16

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    # P^T rows are N_Q half-precision values: 32 B (N_Q = 16) or 64 B (N_Q = 32).
    P_SWZ_BYTES: int = 32
    TILE_K_HW: int = 16

    # K and V tiles share one ring of 64 KiB slots (K(t), V(t), K(t+1), ...).
    STAGES_KV: int = 3

    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1

    PAGED_KV: int = 0
    PAGE_SIZE: int = 0

    # 4 softmax warps per 16 S^T columns (one group at N_Q = 16, two at 32),
    # then the MMA warp and the TMA warp.
    SOFTMAX_WARPS: int = 4
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TOTAL_WARPS: int = 6
    THREADS_PER_CTA: int = 6 * 32


def _validate_cfg_d256_decode(cfg: CfgD256Decode, q_tiles: Tuple[int, ...] = _D256_DECODE_Q_TILES) -> None:
    """``q_tiles`` (appended): the tile widths the CALLING arch line compiles -- the Blackwell set by default; the
    cc 10.7 factory passes its own (config_sm107._D256_DECODE_Q_TILES_SM107, which adds the 64-column tile the 320 KiB
    Rubin carveout holds).  Every other predicate is arch-neutral."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    checks = (
        (not fp8, "d256 decode: f16/bf16 inputs only"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d256 decode: half input requires DTYPE_O == DTYPE_QKV"),
        (cfg.N_Q in q_tiles, f"d256 decode: N_Q must be one of {q_tiles}"),
        (cfg.SOFTMAX_WARPS == 4 * (cfg.N_Q // 16), "d256 decode: 4 softmax warps per 16 Q columns"),
        (
            cfg.MMA_WARP_ID == cfg.SOFTMAX_WARPS and cfg.TMALDG_WARP_ID == cfg.SOFTMAX_WARPS + 1 and cfg.TOTAL_WARPS == cfg.SOFTMAX_WARPS + 2,
            "d256 decode: role layout is softmax warps, then the MMA warp, then the TMA warp",
        ),
        (cfg.THREADS_PER_CTA == 32 * cfg.TOTAL_WARPS, "d256 decode: THREADS_PER_CTA must be 32 * TOTAL_WARPS"),
        (cfg.TILE_N == 128, "d256 decode: the KV tile is the 128-lane TMEM layout"),
        (cfg.STAGES_KV >= 2, "d256 decode: the K/V ring needs at least K(t) and V(t) resident"),
        (
            cfg.P_SWZ_BYTES == cfg.N_Q * cfg.BPE and cfg.P_SWZ_BYTES in (32, 64, 128),
            "d256 decode: P^T row bytes must be the 32 B, 64 B or 128 B swizzle atom (N_Q x BPE)",
        ),
        (not cfg.PACK_GQA or cfg.QH_PER_KH <= cfg.N_Q, "d256 decode: a packed head group must fit the Q tile"),
        (cfg.QH_PER_KH >= 1, "d256 decode: qh_per_kh must be >= 1"),
        (cfg.SPLIT_KV >= 1, "d256 decode: split_kv must be >= 1"),
        (not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1, "d256 decode: paged KV requires per-batch KV lengths"),
        (
            not cfg.PAGED_KV or (cfg.PAGE_SIZE >= 8 and cfg.PAGE_SIZE % 8 == 0 and (128 % cfg.PAGE_SIZE == 0 or cfg.PAGE_SIZE % 128 == 0)),
            "d256 decode: page_size must be a positive multiple of 8 that divides the 128-row tile or is a multiple of it",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d256_decode(params: TemplateParams, q_tiles: Tuple[int, ...] = _D256_DECODE_Q_TILES) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm100/decode_d256_f16.py from the adapter's TemplateParams.

    Backstop only (see TemplateParams): the adapter selects this tile for
    decode-shaped f16/bf16 d256 graphs and keeps every other graph on the
    prefill tile, so a ValueError here is an adapter gap, not a user error.
    ``cta_mma`` / ``sched_policy`` are accepted and unused — the decode tile
    is one cta_group::1 CTA per unit with nothing to schedule.

    ``q_tiles`` (appended): the tile widths the calling arch line compiles.
    The default is this line's ``_D256_DECODE_Q_TILES`` (16, 32: what 227 KiB
    holds); ``config_sm107.make_cfg_d256_decode`` passes the Rubin set, which
    adds the 64-column tile (four softmax groups, 258 KiB of SMEM under the
    320 KiB Rubin carveout) and re-validates the record against its own facts.
    """
    if params.qsa_block_topk or params.qsa_block_size or params.qsa_softmax_groups:
        # Same decline as the prefill templates' _validate_params: an index-list record belongs to the cc 10.7 sparse
        # kernel (its own swap-AB body over gathered K/V), never to this dense ring.
        raise ValueError(
            f"d256 decode: qsa_block_topk={params.qsa_block_topk} / qsa_block_size={params.qsa_block_size} / qsa_softmax_groups={params.qsa_softmax_groups} "
            f"select the index-list sparse kernel; the decode tile does not consume them"
        )
    if params.decode_q_tile not in q_tiles:
        raise ValueError(f"d256 decode: decode_q_tile must be one of {q_tiles}; got {params.decode_q_tile}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d256 decode: f16/bf16 inputs only")
    if params.thd_varlen:
        raise ValueError("d256 decode: THD/varlen queries ride the prefill tile")
    if params.pv_bf16 or params.softmax_f16:
        raise ValueError("d256 decode: pv_bf16 / softmax_f16 are quantized-kernel specializations")
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"d256 decode: window_left must be >= 0 (or None); got {params.window_left}")
    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"d256 decode: window_right must be >= 0 (or None); got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("d256 decode: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("d256 decode: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b = bpe(params.dtype_qkv)
    softmax_warps = 4 * (int(params.decode_q_tile) // 16)
    cfg = CfgD256Decode(
        N_Q=int(params.decode_q_tile),
        SOFTMAX_WARPS=softmax_warps,
        MMA_WARP_ID=softmax_warps,
        TMALDG_WARP_ID=softmax_warps + 1,
        TOTAL_WARPS=softmax_warps + 2,
        THREADS_PER_CTA=(softmax_warps + 2) * 32,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        P_SWZ_BYTES=int(params.decode_q_tile) * b,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d256_decode(cfg, q_tiles)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d512 flavor — d_qk = d_v = 512, SM100 (Blackwell), cga4x1 role-split (DSv4-class models)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD512:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 512
    TILE_O: int = 512

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2

    SPLIT_PIPELINE: int = 1

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
    XFER_STAGES: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 0

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
    STATS_LOG2: int = 0  # LSE stored in base 2 (stats_use_log2)
    BOTTOM_RIGHT: int = 0  # band diagonal anchored bottom-right

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 512 // 256
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 256

    TOTAL_WARPS: int = 8
    THREADS_PER_CTA: int = 8 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 0
    CORR_WARP_BASE: int = 0
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TMASTG_WARP_ID: int = 6
    SCHED_WARP_ID: int = 7

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    READ_TILE_ARRIVERS: int = ((1 * 4) + 1) * (4 * 1) + (4 * 1) // (2 * 2) + (4 * 1) // 2 + 2 * 1

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  See TemplateParams.split_kv.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0

    # The graph's GQA ratio H_q / H_kv -- the bottom-right diagonal and the
    # LPT_L2 head grouping are derived from it.
    QH_PER_KH: int = 1
    # Heads packed into one Q tile row-group under PACK_GQA (the kernel's
    # HEADS_PER_TILE): QH_PER_KH when the group divides TILE_M, else -- the
    # d128 / d256 f16 kernels only -- its largest divisor that does
    # (pack_gqa_group_size); QH_PER_KH // PACK_G packed heads then share one KV
    # head.  1 when unpacked.  Distinct from QH_PER_KH on purpose: conflating
    # the pack size with the GQA ratio would mis-mask MTP rows.
    PACK_G: int = 1

    # Paged KV cache; see TemplateParams.paged_kv.  PAGE_SIZE tokens per page.
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0


def _validate_cfg_d512(cfg: CfgD512) -> None:
    """Consistency checks on the (mostly hardcoded) d512 geometry."""
    _fp8 = cfg.DTYPE_QKV <= DTYPE_E5M2
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d512: MMA/TMALDG/TMASTG/Scheduler regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d512: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d512: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 4 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, "d512 flavor: cga4x1 / CTA_MMA=2 only"),
        (cfg.CGA_M // cfg.CTA_MMA == 2, "d512 flavor: exactly two sub-groups (CGA_M / CTA_MMA == 2)"),
        (cfg.TILE_K == 512 and cfg.TILE_O == 512, "d512: d_qk = d_v = 512"),
        (cfg.TILES_Q == 1, "d512 (role-split): TILES_Q must be 1"),
        (cfg.SOFTMAX_WARPGROUPS == 1, "d512 (role-split): SOFTMAX_WARPGROUPS must be 1"),
        (cfg.CORRECTION_WARPS == 0, "d512 (role-split): CORRECTION_WARPS must be 0"),
        (cfg.READ_TILE_ARRIVERS == 25, f"d512 cga4x1: expected READ_TILE_ARRIVERS=25, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128, "d512: Q/K/V swizzle must all be 128B"),
        (cfg.O_SWZ_BYTES == 128, "d512: O swizzle must be 128B"),
        # Both dtype families run the SM10x 1-chunk MMA step: f16 at K=16,
        # FP8 at the K=32 QMMA (k_dim=0).  The Rubin K=64 2-chunk fast path is
        # silently WRONG on Blackwell — see rules/mma-tma-matrix.md § 1.
        (
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if _fp8 else 16),
            "d512: TILE_K_HW must be 32 (fp8 K=32 QMMA) / 16 (f16, 1-chunk on SM10x — 2-chunk silently wrong)",
        ),
        # SMEM-cap driven: the K/V ring costs STAGES_KV * 64 KiB at f16 and
        # STAGES_KV * 32 KiB at FP8, so FP8 buys a third stage under the same
        # 227 KiB cap.
        (cfg.STAGES_KV == (3 if _fp8 else 2), "d512 (SM100): STAGES_KV must be 3 (fp8) / 2 (f16) — SMEM-cap driven"),
        # TMEM-cap driven (512 cols on Blackwell): the sg0 carve is
        # XFER_STAGES * TILE_N (S_acc parities) + TILE_K / (4 // BPE) (Q, moved
        # to TMEM by UTCCP).  f16: 2*128 + 256 = 512.  FP8 packs 4 elems per
        # 4-byte column, so Q costs only 128 cols and a third parity fits:
        # 3*128 + 128 = 512.
        (cfg.XFER_STAGES == (3 if _fp8 else 2), "d512 (SM100): XFER_STAGES must be 3 (fp8) / 2 (f16) — TMEM-cap driven"),
        (
            cfg.XFER_STAGES * cfg.TILE_N + cfg.TILE_K // (4 // cfg.BPE) == 512,
            f"d512 (SM100): sg0 TMEM carve (S_acc {cfg.XFER_STAGES * cfg.TILE_N} + Q {cfg.TILE_K // (4 // cfg.BPE)}) must be exactly 512 cols",
        ),
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16) if _fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d512: DTYPE_O must equal DTYPE_QKV for half input; fp8 allows an independent output dtype",
        ),
        (not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1, "d512: paged KV requires per-batch KV lengths"),
        (
            not cfg.PAGED_KV or (cfg.PAGE_SIZE >= 8 and cfg.PAGE_SIZE % 8 == 0 and (128 % cfg.PAGE_SIZE == 0 or cfg.PAGE_SIZE % 128 == 0)),
            f"d512: page_size must be a multiple of 8 that divides the 128-row KV tile or is a multiple of it; got {cfg.PAGE_SIZE}",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    # The 2x2-datapath record (TemplateParams.mma_2x2, appended, default False) is read through
    # getattr so a record built before the field existed takes the role-split arm unchanged.
    if getattr(params, "mma_2x2", False):
        return make_cfg_d512_2x2(params, cga_m=2 if params.d_flavor == 576 else 4)
    _validate_params("d512", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= DTYPE_E5M2  # E4M3/E5M2 inputs → the fp8 kernel file
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    # FP8 pins the Blackwell K=32 QMMA path.  NOT tile_k_hw(), which returns the
    # Rubin K=64 answer (see mma-tma-matrix.md § 1 "latent trap in the shared
    # helper") — k_dim=0 with TILE_K_HW=64 is the silently-wrong combination.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD512(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        Q_SWZ_BYTES=q_swz_bytes(512, b),
        K_SWZ_BYTES=q_swz_bytes(512, b),
        V_SWZ_BYTES=v_swz_bytes(512, 2, b),
        O_SWZ_BYTES=o_swz_bytes(512, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=tile_k_hw_fp8,
        # FP8 halves the K/V ring and the TMEM-resident Q, buying a third KV
        # stage and a third S_acc parity under the same SMEM / TMEM caps.
        STAGES_KV=3 if fp8 else 2,
        XFER_STAGES=3 if fp8 else 2,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d512(cfg)
    return cfg, _tma_iters(cfg)


def make_cfg_d512_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    """Build the SM100 D512 block-scale CTA1 configuration."""

    cfg, _ = _make_cfg_d256(params, mxfp8=True)
    # A 512-column MXFP8 accumulator leaves no TMEM columns for block scales.
    # Keep the proven CTA1 M128 pipeline and emit two 256-column O slices.
    cfg = replace(cfg, TILE_K=512, STAGES_KV=1)
    if cfg.CTA_MMA != 1 or cfg.CGA_M != 1:
        raise ValueError("d512 MXFP8 requires one-CTA M128 MMA")
    if cfg.TILE_M != 128 or cfg.TILE_N != 128 or cfg.TILE_K != 512 or cfg.TILE_O != 256:
        raise ValueError("d512 MXFP8 requires M128xN128, K512, and a 256-column output slice")
    if cfg.PACK_GQA:
        raise ValueError("d512 MXFP8 does not support PackGQA")
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d512 flavor on the 2x2 DATAPATH -- d_qk = d_v = 512, SM100, sm100/prefill_d512_f16_2x2.py
#
# ONE pipeline per CTA on the tcgen05.mma.cta_group::2 M=128 atom (64 Q rows per
# CTA; fp32 D row m lands on TMEM lane (m % 64) + 64 * (n // (N/2)), column
# n % (N/2)), 12 warps per CTA (4 softmax + 4 correction/epilogue + MMA +
# TMA-LDG + TMA-STG + scheduler), cluster (CGA_M, 1, 1) of CGA_M // CTA_MMA
# cta_group::2 pairs that share every K/V sub-chunk by TMA multicast (KV_SHARE
# pairs: CTA c and its twin c ^ 2 each issue half of each 32 KiB sub-chunk with
# mask (1 << c) | (1 << (c ^ 2)); the bring-up arm CGA_M=2 is the same body
# with KV_SHARE=1 and own-bit loads).  Every CTA runs TMA(Q,K,V) + BMM1 +
# softmax + BMM2 + correction + epilogue + TMA-STG for its own 64 rows -- the
# role split, the DSMEM P/alpha/stats ships, the UTCCP of Q and the six
# xfer-barrier families of CfgD512 do not exist here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD512X2:
    TILE_M: int = 64
    TILE_N: int = 128
    TILE_K: int = 512
    TILE_O: int = 512

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    # Cluster (CGA_M, CGA_N, 1) = CGA_M // CTA_MMA cta_group::2 pairs sharing K/V.
    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2
    # Pairs that share each K/V sub-chunk by TMA multicast: 2 = twin multicast
    # (each CTA issues half of every sub-chunk to itself and its twin c ^ 2),
    # 1 = own-bit loads (the CGA_M=2 bring-up arm).  Always CGA_M // CTA_MMA.
    KV_SHARE: int = 2
    # Q super-tiles (64-row CTA tiles) per cluster: every CTA of the cluster owns
    # its own 64 rows, so the scheduler / bounds helpers decode in CGA_M units
    # (make_sdpa_helpers(kv_shared_cluster=True)); the role-split d512 decodes in
    # CTA_MMA units because its two pairs own the SAME rows.
    Q_SUPERS_PER_CLUSTER: int = 4
    ROWS_PER_CLUSTER: int = 4 * 64

    SPLIT_PIPELINE: int = 1

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    # f16 TILE_K_HW = 16 on SM10x (the K=64 2-chunk form is FP8-only; the f16
    # 2-chunk K=32 path is silently wrong on SM10x -- see _validate_cfg_d512).
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    # K ring: 32 KiB d-half sub-chunks (this CTA's 64 kv rows x 256 d cols); two
    # per KV iteration.  V ring: 32 KiB 128-col sub-chunks (128 kv rows x this
    # CTA's d_v [256*cta_in_pair + 128c, +128)); two per KV iteration.  Each
    # ring holds STAGES_*_SUB sub-chunks = one KV tile of buffering at 2.
    STAGES_K_SUB: int = 2
    STAGES_V_SUB: int = 2
    # S parities (TMEM) = P ring slots (SMEM).  BMM1 runs BMM1_LOOKAHEAD
    # iterations ahead of BMM2; the S/P slot reuse is ordered by the MMA
    # thread's own issue order (p_full(i) -> BMM2(i) -> BMM1(i + XFER_STAGES)),
    # which needs XFER_STAGES >= BMM1_LOOKAHEAD + 1 (checked by the validator).
    XFER_STAGES: int = 2
    BMM1_LOOKAHEAD: int = 1
    # STAGES_KV: the role-split name, kept for the shared helpers that read it
    # (max of the two sub-chunk rings, in sub-chunks).
    STAGES_KV: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4

    # setmaxnreg split: 4 softmax warps INCREASE to 192 (64 fp32 S + 32 packed P
    # + mask words), 4 correction warps INCREASE to 208 (two live 64-fp32 epilogue
    # batches), the 4 single warps DECREASE to 40.  Validator: 40 + 208 + 192 <= 512
    # and 32 * (4*192 + 4*208 + 4*40) <= 65536.
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 208
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    # BMM2 is issued per 256-wide (collective) N-block = one 32 KiB V sub-chunk;
    # each N-block has its own bmm2_ready credit.
    BMM2_N_PER_CALL: int = 256
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 256

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 0
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    # --- mbarrier arrival ledger (exact per-phase sums; the kernel's init counts
    # are THESE constants and a host-only test re-derives them) ---
    # read_tile_id_arrive lands ONE arrive per calling warp on EVERY CTA of the
    # cluster: 4 softmax + 4 correction + TMA-LDG + TMA-STG = 10 warps per CTA,
    # plus the MMA warp of each pair LEADER (the quiet non-leader never credits):
    # 10 * CGA_M + CGA_M // CTA_MMA = 42 (CGA_M=4) / 21 (CGA_M=2, = CfgD256's).
    READ_TILE_ARRIVERS: int = 10 * 4 + 4 // 2
    # k/v_empty: one tcgen05.commit per pair LEADER landing on every CTA of the
    # share group (mask 0xF under KV_SHARE=2, the pair mask under KV_SHARE=1).
    KV_EMPTY_ARRIVERS: int = 4 // 2
    # mb_o_full[s]: the 64 lanes of ONE column half (d_v half s // 4) arrive per
    # 8 KiB O subtile -- NOT the 128 lanes of the role-split epilogue.
    O_CHUNK_ARRIVERS: int = 64
    # mb_o_empty: the 32 TMA-STG lanes of this CTA AND of every twin that
    # multicasts V into the aliased sVO (arrive + arrive_on_peer(cta ^ 2)):
    # ONE_WARP * KV_SHARE = 64 (CGA_M=4) / 32 (CGA_M=2).
    O_EMPTY_ARRIVERS: int = 32 * 2
    # mb_p_full / mb_bmm2_ready / mb_empty_mainloop / mb_tmem_dealloc: every lane
    # of the producing warpgroup on BOTH CTAs of the pair.
    PAIR_LANES: int = 128 * 2

    # TMEM: O 64x512 fp32 in the 2x2 atom = 256 cols at [0,256), S parity p =
    # 64 cols at [256 + 64p, +64), alpha[s] at 384+s, tile stats of ring slot s
    # at 386+2s / 387+2s (slot-indexed: a fixed stats pair raced the alpha ring).
    TMEM_COLS: int = 512
    O_TMEM_COLS: int = 256
    S_TMEM_COLS: int = 64
    # tcgen05 SMEM-descriptor version (0 = SM100 format, every operand below 256 KiB).
    DESC_VERSION: int = 0
    # O staging aliases the V ring (today's sg1 O u V protocol: the first V load
    # of a tile waits mb_o_empty).
    OV_ALIAS: int = 1
    # Per-CTA dynamic SMEM cap the budget is checked against (opt-in max).
    SMEM_CAP_BYTES: int = 227 * 1024
    # Worst-case dynamic-SMEM base alignment pad for the 1024-aligned first array.
    SMEM_ALIGN_PAD: int = 1008

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  The 2x2 body carries the half-aware fp32-partials arm;
    # the adapter twin keeps split_kv > 1 on the role-split kernel for now.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    # SM100 packs whole groups; the native SM107 body also serves divisors of G.
    PACK_G: int = 1

    # D512 has two 256-column K chunks. Native D576 uses three 192-column
    # chunks so resident Q and both K/V rings fit under the SM100 SMEM cap.
    N_BMM1_CHUNKS: int = 2
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0


def d512_2x2_smem_bytes(cfg: CfgD512X2, n_o_chunks: int = 8) -> dict:
    """The kernel's SMEM allocation from the SAME constants its arrays are declared with.

    data     = sQ (resident) + sK ring + (sV ring u sO staging) + sP ring
    scratch  = sXchgMax [XFER_STAGES parities][2 halves][TILE_M] fp32 + sXchgSum [2 halves][TILE_M] fp32
    barriers = mbarrier words (see the ledger in make_d512_2x2_bars) + scheduler bars + tile ids + tmem_ptr
    total    = data + scratch + barriers + SMEM_ALIGN_PAD (worst-case base pad for the 1024-aligned first array)
    """
    bpe = cfg.BPE
    q = cfg.TILE_M * cfg.TILE_K * bpe
    k_sub = (cfg.TILE_N // cfg.CTA_MMA) * (cfg.TILE_K // cfg.N_BMM1_CHUNKS) * bpe
    v_sub = cfg.TILE_N * (cfg.TILE_O // cfg.CTA_MMA // cfg.N_BMM2_CHUNKS) * bpe  # 128 kv rows x 128 d_v
    o = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O
    p = cfg.TILE_M * cfg.TILE_N * bpe
    v_ring = cfg.STAGES_V_SUB * v_sub
    vo = max(v_ring, o) if cfg.OV_ALIAS else v_ring + o
    data = q + cfg.STAGES_K_SUB * k_sub + vo + cfg.XFER_STAGES * p
    scratch = cfg.XFER_STAGES * 2 * cfg.TILE_M * 4 + 2 * cfg.TILE_M * 4
    n_bars = (
        2  # q_full, q_empty
        + 2 * cfg.STAGES_K_SUB  # k_full, k_empty
        + 2 * cfg.STAGES_V_SUB  # v_full, v_empty
        + 2 * cfg.XFER_STAGES  # bmm1_done, bmm2_done
        + cfg.XFER_STAGES * cfg.N_BMM2_CHUNKS  # bmm2_ready
        + cfg.XFER_STAGES  # p_full
        + 2 * 2  # stat_full, stat_empty (2-stage ring)
        + n_o_chunks  # o_full
        + 1  # o_empty
        + 1  # empty_mainloop
        + 1  # tmem_dealloc
    )
    barriers = n_bars * 8 + 2 * cfg.SCHEDULER_STAGES * 8 + cfg.SCHEDULER_STAGES * 8 * 4 + 16
    return dict(
        q=q,
        k_sub=k_sub,
        v_sub=v_sub,
        o=o,
        p=p,
        data=data,
        scratch=scratch,
        n_bars=n_bars,
        barriers=barriers,
        total=data + scratch + barriers + cfg.SMEM_ALIGN_PAD,
    )


def d512_2x2_p_ring_start_bytes(cfg: CfgD512X2) -> int:
    """SMEM byte offset at which the P ring begins = sQ + the K ring + the (V ring u O staging) slab -- the first
    MMA-operand tile that can cross the 256 KiB version-0 tcgen05 descriptor window (config_sm107.TCGEN05_V0_ADDR_LIMIT):
    192 KiB at 2/2 sub-chunk stages (SM100), exactly 262144 at 3/3 (SM107)."""
    smem = d512_2x2_smem_bytes(cfg)
    vo = max(cfg.STAGES_V_SUB * smem["v_sub"], smem["o"]) if cfg.OV_ALIAS else cfg.STAGES_V_SUB * smem["v_sub"] + smem["o"]
    return smem["q"] + cfg.STAGES_K_SUB * smem["k_sub"] + vo


def d512_2x2_geometry_checks(cfg: CfgD512X2) -> tuple:
    """The ARCH-NEUTRAL consistency checks on the 2x2-datapath d512 geometry, as ``(ok, message)`` pairs: every count
    the kernel's mbarrier inits and setmaxnreg take is re-derived here.  config_sm107 appends the Rubin arch checks
    (descriptor version from the layout, the 320 KiB usable budget) to this list; the SM100 ones follow below."""
    smem = d512_2x2_smem_bytes(cfg)
    n_pairs = cfg.CGA_M // cfg.CTA_MMA
    return (
        (cfg.TILE_M == 64 and cfg.TILE_N == 128, "d512 2x2: TILE_M=64 (cta_group::2 M=128 atom) / TILE_N=128"),
        (
            (cfg.TILE_K == 512 and cfg.N_BMM1_CHUNKS == 2) or (cfg.TILE_K == 576 and cfg.N_BMM1_CHUNKS == 3 and cfg.CGA_M == 2),
            "2x2 QK geometry requires D512 with two K chunks or D576 with three K chunks and one CTA pair",
        ),
        (cfg.TILE_O == 512, "2x2 wide-head geometry requires D_V=512"),
        (cfg.CTA_MMA == 2 and cfg.CGA_N == 1, "d512 2x2: cta_group::2 pairs, CGA_N=1"),
        (cfg.CGA_M in (2, 4) and cfg.CGA_M % cfg.CTA_MMA == 0, f"d512 2x2: CGA_M must be 2 (bring-up arm) or 4; got {cfg.CGA_M}"),
        (cfg.KV_SHARE == n_pairs, f"d512 2x2: KV_SHARE ({cfg.KV_SHARE}) must equal CGA_M // CTA_MMA ({n_pairs})"),
        (cfg.Q_SUPERS_PER_CLUSTER == cfg.CGA_M * cfg.CGA_N, "d512 2x2: one 64-row Q super-tile per CTA of the cluster"),
        (cfg.ROWS_PER_CLUSTER == cfg.TILES_Q * cfg.TILE_M * cfg.CGA_M * cfg.CGA_N, "d512 2x2: ROWS_PER_CLUSTER = TILES_Q * TILE_M * CGA_M * CGA_N"),
        (cfg.TILES_Q == 1 and cfg.SPLIT_PIPELINE == 1, "d512 2x2: TILES_Q=1, blocked NATURAL decode"),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16) and cfg.DTYPE_O == cfg.DTYPE_QKV, "d512 2x2: half inputs with DTYPE_O == DTYPE_QKV"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d512 2x2: f16 TILE_K_HW must be 16 on SM10x (the 2-chunk form is silently wrong)"),
        (cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == cfg.V_SWZ_BYTES == cfg.O_SWZ_BYTES == 128, "d512 2x2: Q/K/V/O swizzle must all be 128B"),
        (
            cfg.XFER_STAGES >= cfg.BMM1_LOOKAHEAD + 1,
            f"d512 2x2: XFER_STAGES ({cfg.XFER_STAGES}) must be >= BMM1_LOOKAHEAD + 1 ({cfg.BMM1_LOOKAHEAD + 1}): S/P slot reuse rides the MMA issue order",
        ),
        (cfg.STAGES_K_SUB >= 2 and cfg.STAGES_V_SUB >= 2, "d512 2x2: both sub-chunk rings need >= 2 slots (two sub-chunks per iteration)"),
        (cfg.STAGES_KV == max(cfg.STAGES_K_SUB, cfg.STAGES_V_SUB), "d512 2x2: STAGES_KV mirrors the deeper sub-chunk ring"),
        (cfg.N_BMM2_CHUNKS == 2 and cfg.BMM2_N_PER_CALL == 256, "d512 2x2: BMM2 = two collective N=256 blocks (one 32 KiB V sub-chunk each)"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d512 2x2: 12 warps"),
        (cfg.SOFTMAX_WG_WARPS == 4 and cfg.CORRECTION_WARPS == 4 and cfg.SOFTMAX_WARPGROUPS == 1, "d512 2x2: 4 softmax + 4 correction warps"),
        (
            (cfg.SOFTMAX_WG0_BASE, cfg.CORR_WARP_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (0, 4, 8, 9, 10, 11),
            "d512 2x2: warp roles 0-3 softmax, 4-7 correction, 8 MMA, 9 TMA-LDG, 10 TMA-STG, 11 scheduler",
        ),
        (cfg.SOFTMAX_LANES == 128 and cfg.CORR_LANES == 128 and cfg.PAIR_LANES == 256, "d512 2x2: 128 lanes per compute warpgroup, 256 across the pair"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS == cfg.OTHER_REGS, "d512 2x2: single-warp roles share OTHER_REGS"),
        (cfg.OTHER_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_REGS <= 512, "d512 2x2: register budget over 512"),
        (32 * (4 * cfg.SOFTMAX_REGS + 4 * cfg.CORRECTION_REGS + 4 * cfg.OTHER_REGS) <= 65536, "d512 2x2: per-SM register file over 64K"),
        (cfg.SOFTMAX_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.OTHER_REGS % 8 == 0, "d512 2x2: per-role regs must be multiples of 8"),
        (
            cfg.READ_TILE_ARRIVERS == 10 * cfg.CGA_M + n_pairs,
            f"d512 2x2: READ_TILE_ARRIVERS must be 10 * CGA_M + CGA_M // CTA_MMA = {10 * cfg.CGA_M + n_pairs}; got {cfg.READ_TILE_ARRIVERS}",
        ),
        (cfg.KV_EMPTY_ARRIVERS == n_pairs, f"d512 2x2: KV_EMPTY_ARRIVERS must be CGA_M // CTA_MMA = {n_pairs}; got {cfg.KV_EMPTY_ARRIVERS}"),
        (cfg.O_CHUNK_ARRIVERS == cfg.CORR_LANES // 2, "d512 2x2: mb_o_full is arrived by the 64 lanes of ONE column half"),
        (
            cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE,
            f"d512 2x2: O_EMPTY_ARRIVERS must be ONE_WARP * KV_SHARE = {cfg.ONE_WARP * cfg.KV_SHARE} (own + every twin's TMA-STG warp); got {cfg.O_EMPTY_ARRIVERS}",
        ),
        (
            cfg.O_TMEM_COLS == cfg.TILE_M * cfg.TILE_O // 128 and cfg.S_TMEM_COLS == cfg.TILE_M * cfg.TILE_N // 128,
            "d512 2x2: 2x2 atom TMEM footprints (N/2 cols)",
        ),
        (
            cfg.O_TMEM_COLS + cfg.XFER_STAGES * cfg.S_TMEM_COLS + 2 + 2 * 2 <= cfg.TMEM_COLS,
            f"d512 2x2: TMEM carve O {cfg.O_TMEM_COLS} + S {cfg.XFER_STAGES * cfg.S_TMEM_COLS} + alpha ring 2 + per-slot stats 4 > {cfg.TMEM_COLS}",
        ),
        (
            smem["total"] <= cfg.SMEM_CAP_BYTES,
            f"d512 2x2: SMEM {smem['total']} B (incl. {cfg.SMEM_ALIGN_PAD} B pad) over the {cfg.SMEM_CAP_BYTES} B cap: {smem}",
        ),
        (not cfg.PACK_GQA or cfg.TILE_M % cfg.PACK_G == 0, "d512 2x2: PACK_G must divide TILE_M"),
    )


def _validate_cfg_d512_2x2(cfg: CfgD512X2) -> None:
    """The SM100 2x2-datapath d512 configuration: the arch-neutral geometry checks plus the SM100 arch facts
    (227 KiB opt-in cap, every operand below the 256 KiB version-0 descriptor window)."""
    checks = d512_2x2_geometry_checks(cfg) + (
        (cfg.SMEM_CAP_BYTES == 227 * 1024, f"d512 2x2 (SM100): the budget is checked against the 227 KiB opt-in cap, got {cfg.SMEM_CAP_BYTES}"),
        (
            cfg.DESC_VERSION == 0 and d512_2x2_p_ring_start_bytes(cfg) + cfg.XFER_STAGES * cfg.TILE_M * cfg.TILE_N * cfg.BPE <= 256 * 1024,
            "d512 2x2 (SM100): every operand sits below 256 KiB -> tcgen05 SMEM descriptor version 0",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512_2x2(params: TemplateParams, *, cga_m: int = 4) -> Tuple[CfgD512X2, TmaIters]:
    """The 2x2-datapath d512 configuration (``TemplateParams.mma_2x2``).

    ``cga_m`` = 4 (two twin pairs sharing K/V by multicast, the shipped arm) or 2
    (the bring-up / bitwise-twin arm: one pair, own-bit loads).  Not a knob:
    tests load the template with the arm they want through the module constant
    the kernel file reads."""
    if not getattr(params, "mma_2x2", False):
        raise ValueError("d512 2x2: make_cfg_d512_2x2 needs a TemplateParams record with mma_2x2=True (make_cfg_d512 dispatches on it)")
    native_d576 = params.d_flavor == 576
    _validate_params("d512", params, d512_partial_pack_wired=native_d576)
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d512 2x2: BF16/FP16 inputs only")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    n_pairs = cga_m // CfgD512X2.CTA_MMA
    cfg = CfgD512X2(
        TILE_K=576 if native_d576 else 512,
        N_BMM1_CHUNKS=3 if native_d576 else 2,
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        CGA_M=cga_m,
        KV_SHARE=n_pairs,
        Q_SUPERS_PER_CLUSTER=cga_m,
        ROWS_PER_CLUSTER=cga_m * CfgD512X2.TILE_M,
        READ_TILE_ARRIVERS=10 * cga_m + n_pairs,
        KV_EMPTY_ARRIVERS=n_pairs,
        O_EMPTY_ARRIVERS=CfgD512X2.ONE_WARP * n_pairs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=native_d576),
    )
    _validate_cfg_d512_2x2(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d128 flavor — d_qk = d_v = 128, SM100 (Blackwell), cga2 (Llama-class models)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD128(D128FamilySmallTileFields):
    # The small-tile seams' field block (bias / kv_mask / pair / ring / max form / tail / quiet / S_SLOTS / BIAS_MCAST_CTAS / Q_IN_TMEM) is
    # inherited from config_d128_family at its OFF defaults: the d128 / plain d64 bodies read none of them, so their renderings are
    # byte-identical; the small-tile factory (_make_cfg_small_d_sm100) is the only path that sets one.
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128
    TILE_O: int = 128

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    # V may be BF16 in the experimental MXFP8-QK/BF16-PV path while Q/K
    # retain their one-byte MXFP8 storage.
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    # Block-scaled O (per-tensor FP8 fprop only): scale-factor block along d
    # (16 = E2M1 + E4M3 SF, 32 = E4M3 + UE8M0 SF, 0 = plain O) and logical
    # O elements per storage byte (2 for E2M1).
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1

    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    # Q∪O SMEM alias — unused on llama (d_qk=d_v=128 fits without it at cga2).
    QO_ALIAS: int = 0

    SPLIT_PIPELINE: int = 0

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 2
    # Q sub-tiles that hold LIVE Q rows: TILES_Q, or 1 under DUAL_KV_STREAM, where
    # sub-tile 1 re-uses sub-tile 0's rows as a second KV stream.  Sizes the
    # per-CTA Q span (row bases, mask bounds, grid), never the TMEM / SMEM /
    # barrier structure, which stays TILES_Q wide.
    Q_TILES_LIVE: int = 2
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2

    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4

    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
    STATS_LOG2: int = 0  # LSE stored in base 2 (stats_use_log2)
    BOTTOM_RIGHT: int = 0  # band diagonal anchored bottom-right

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64

    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    # Two softmax warpgroups (WG0 @ warp 0, WG1 @ warp 4); corr @ 8; single
    # roles @ 12..15.
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    # 8 softmax + 4 corr + 1 MMA + 1 TMALDG + 1 TMASTG = 15 arrivers.
    READ_TILE_ARRIVERS: int = 15

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  See TemplateParams.split_kv.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0

    # The graph's GQA ratio H_q / H_kv -- the bottom-right diagonal and the
    # LPT_L2 head grouping are derived from it.
    QH_PER_KH: int = 1
    # Heads packed into one Q tile row-group under PACK_GQA (the kernel's
    # HEADS_PER_TILE): QH_PER_KH when the group divides TILE_M, else -- the
    # d128 / d256 f16 kernels only -- its largest divisor that does
    # (pack_gqa_group_size); QH_PER_KH // PACK_G packed heads then share one KV
    # head.  1 when unpacked.  Distinct from QH_PER_KH on purpose: conflating
    # the pack size with the GQA ratio would mis-mask MTP rows.
    PACK_G: int = 1

    # Paged KV cache; see TemplateParams.paged_kv.  PAGE_SIZE tokens per page.
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0


# Blackwell SM100 per-CTA dynamic SMEM cap (228 KiB physical, 227 KiB usable).
_SM100_MAX_DYN_SMEM = 227 * 1024


def _d128_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d128 pipeline (barriers/TMEM ptr are noise).

    Q and O are TILES_Q slabs each; under QO_ALIAS they share one slab sized to
    the larger.  K/V are STAGES_KV buffers each, and their PER-CTA size is
    divided by CTA_MMA because the cga2 collective MMA lets a CTA pair hold half
    of every tile.  That divisor is exactly what cga1 gives up, which is why
    cga1 needs the Q/O alias to break even:

        cga2, no alias : 64(Q) + 64(O) + 32(K) + 32(V) = 192 KiB
        cga1, no alias : 64(Q) + 64(O) + 64(K) + 64(V) = 256 KiB  (over cap)
        cga1, alias    : 64(Q u O)     + 64(K) + 64(V) = 192 KiB
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV
    # Two KV streams over one Q tile: one Q slab and one O slab (the fp8 kernel
    # never aliases them), while P stays one tile per stream.
    q_tiles = getattr(cfg, "Q_TILES_LIVE", cfg.TILES_Q) if getattr(cfg, "DUAL_KV_STREAM", 0) else cfg.TILES_Q
    qo = q_tiles * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
    # The per-tensor FP8 kernel stages P (TILE_M x TILE_N fp8 per Q sub-tile) in SMEM so the next
    # step's QK^T can overlap the softmax; counted for every quantized row (pessimistic for MXFP8).
    p = cfg.TILES_Q * cfg.TILE_M * cfg.TILE_N * cfg.BPE if cfg.DTYPE_QKV <= 1 else 0
    return qo + k + v + p


def _validate_cfg_d128(cfg: CfgD128) -> None:
    """Consistency checks on the (mostly hardcoded) d128 (llama) geometry."""
    _fp8 = cfg.DTYPE_QKV <= 1  # E4M3/E5M2 inputs (MXFP8): STAGES_KV=4, TILE_K_HW=32, independent DTYPE_O
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d128 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            cfg.QO_ALIAS == 1 if cfg.CTA_MMA == 1 and not cfg.DUAL_KV_STREAM else True,
            "d128 cga1: QO_ALIAS is mandatory — cga1 doubles per-CTA K/V (no collective MMA to halve it), "
            "so Q and O must share one slab to stay inside the SMEM cap (the two-stream fp8 decode keeps "
            "one unaliased Q and O slab and fits through its 1-byte K/V stages)",
        ),
        (
            cfg.Q_TILES_LIVE == (1 if cfg.DUAL_KV_STREAM else cfg.TILES_Q),
            "d128: Q_TILES_LIVE is TILES_Q, or 1 under DUAL_KV_STREAM (sub-tile 1 is a KV stream, not Q rows)",
        ),
        (
            not cfg.DUAL_KV_STREAM or (_fp8 and cfg.CTA_MMA == 1 and not cfg.THD_VARLEN and not cfg.PV_BF16),
            "d128: DUAL_KV_STREAM is the per-tensor fp8 cga1 dense/paged decode specialization",
        ),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILE_K == 128 and cfg.TILE_O == 128, "d128: d_qk = d_v = 128"),
        (cfg.TILES_Q == 2, "d128 (llama): TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d128 (llama): SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d128 (llama): CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d128 (llama): 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d128 llama: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            (cfg.STAGES_KV == 4 if (cfg.CTA_MMA == 2 or cfg.DUAL_KV_STREAM) else cfg.STAGES_KV in (2, 3)) if _fp8 else cfg.STAGES_KV == 2,
            "d128 SM100: STAGES_KV must be 2 (f16/bf16) or, for fp8/mxfp8, 4 at cga2 and 2 or 3 at cga1 — "
            "the stage depth scales with the cluster width so stages x per-CTA-buffer stays constant; "
            "the two-stream fp8 decode holds a K/V pair per step and runs 4 at cga1",
        ),
        (
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if _fp8 else 16)),
            "d128: BMM1 uses K=32 for MXFP8; experimental BF16 PV uses K=16",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d128: Q/K swizzle must be 64/128B"),
        (cfg.PV_BF16 in (0, 1) and (not cfg.PV_BF16 or _fp8), "d128: pv_bf16 requires MXFP8 Q/K"),
        (cfg.BPE_V == (2 if cfg.PV_BF16 else cfg.BPE), "d128: V bytes/element must match the PV specialization"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d128: V/O swizzle out of range"),
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16, DTYPE_O_NVFP4, DTYPE_O_MXFP8) if _fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d128: DTYPE_O must equal DTYPE_QKV for half input; fp8/mxfp8 allows an independent output dtype",
        ),
        (
            cfg.O_BLOCK_SCALE == O_BLOCK_SCALE_BY_DTYPE[cfg.DTYPE_O] and cfg.O_PACK_DIV == o_pack_div(cfg.DTYPE_O),
            "d128: O_BLOCK_SCALE / O_PACK_DIV must follow DTYPE_O",
        ),
        (
            cfg.O_BLOCK_SCALE == 0 or not (cfg.THD_VARLEN or cfg.SEQ_Q_LENS_PRESENT or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or cfg.PAGED_KV),
            "d128: block-scaled O serves dense (unpaged), unsplit, unpacked graphs only",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128(params: TemplateParams, *, cga1_kv_stages: int = 2) -> Tuple[CfgD128, TmaIters]:
    """``cga1_kv_stages``: K/V ring depth of the fp8 family's single-CTA (cga1) configuration -- 2 by default (the
    per-tensor fp8 kernel's verified geometry); the MXFP8 kernel asks for 3 (see STAGES_KV below)."""
    _validate_params("d128", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= 1  # E4M3/E5M2 inputs → MXFP8 kernel
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    b_v = 2 if params.pv_bf16 else b
    # FP8/MXFP8 pins the Blackwell K=32 QMMA path (TILE_K_HW=32) and STAGES_KV=4
    # (BPE=1 → 8 KiB/stage, fits 4); f16/bf16 keep 16 / 2.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD128(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=O_BLOCK_SCALE_BY_DTYPE[dtype_o],
        O_PACK_DIV=o_div,
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # cga1 has no collective MMA to halve per-CTA K/V, so Q and O must share
        # one slab to stay under the SMEM cap (_validate_cfg_d128 enforces it).
        QO_ALIAS=1 if params.cta_mma == 1 and not params.dual_kv_stream else 0,
        Q_TILES_LIVE=1 if params.dual_kv_stream else 2,
        DUAL_KV_STREAM=int(params.dual_kv_stream),
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(128, b_o, o_div),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=16 if params.pv_bf16 else tile_k_hw_fp8,
        # KV stage depth scales with the cluster width, as in cuDNN's own
        # kernels (stages_kv = N * CTA_MMA): cga1 has no collective MMA to halve
        # per-CTA K/V, so the stage count halves instead to keep the product --
        # and hence the SMEM -- constant.  Only the fp8 family needs this here:
        # f16/bf16 already fit at cga1 by aliasing Q and O, and their verified
        # cga1 configuration keeps STAGES_KV=2.  mxfp8 additionally stages E8M0
        # scale factors that the SMEM model cannot see, and at STAGES_KV=4 that
        # pushed a cga1 CTA to 237024 B against the 232448 B cap.
        # The MXFP8 kernel takes a THIRD full-width stage at cga1 (cga1_kv_stages=3): Q and O already share a slab
        # there, so 3 x (16 K + 16 V) KiB plus the scale factors still fit, and the deeper prefetch covers the
        # tile boundary of the short packed GQA legs that run cga1.  The two-stream per-tensor fp8 decode holds a
        # K/V pair per step and runs 4 at cga1.
        STAGES_KV=(4 if (params.cta_mma == 2 or params.dual_kv_stream) else cga1_kv_stages) if fp8 else 2,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA is wired in the f16/bf16 d128 kernel; the fp8 / mxfp8
        # siblings pack HEADS_PER_TILE = QH_PER_KH and keep the full-ratio contract.
        PACK_G=_pack_g(params, CfgD128.TILE_M, partial=not fp8),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        # f16 / bf16 register split: 8 more per lane to the MMA / TMA / scheduler warpgroup, taken from the two
        # softmax warpgroups (2 x 184 + 88 + 56 = 512).  At 40 the f16 kernel's MMA warp held the 16 per-k-step Q
        # descriptors of both sub-tiles through local memory (7 STL / 14 LDL pairs per KV step on the path from the
        # K-full / P-ready waits to the tcgen05.mma issue, 112 B stack); at 56 it is spill-free and the softmax still
        # is at 184.  MEASURED (same-node A/B, cuDNN 9.28 control): B300 llama bf16 causal -2 % time (0.92x -> 0.90x
        # of cuDNN @2K, 0.95x -> 0.93x @8K), B200 and dense within noise.  The fp8 / mxfp8 siblings keep the defaults
        # (their MMA warp already fits 40; not re-measured).
        SOFTMAX_REGS=CfgD128.SOFTMAX_REGS if fp8 else 184,
        MMA_REGS=CfgD128.MMA_REGS if fp8 else 56,
        TMALDG_REGS=CfgD128.TMALDG_REGS if fp8 else 56,
        TMASTG_REGS=CfgD128.TMASTG_REGS if fp8 else 56,
        SCHEDULER_REGS=CfgD128.SCHEDULER_REGS if fp8 else 56,
        OTHER_REGS=CfgD128.OTHER_REGS if fp8 else 56,
    )
    _validate_cfg_d128(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d128 decode tile — d_qk = d_v = 128, SM100, cga1, TILES_Q=1 (decode / MTP shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD128Decode(CfgD128):
    """The d128 f16/bf16 DECODE tile (``sm100/decode_d128_f16.py``).

    Same head geometry, masks, paged loader and split/epilogue contract as
    :class:`CfgD128` (incl. partial PackGQA: ``PACK_G`` heads per token
    row-group), but shaped for graphs whose Q rows (times the packed group)
    fit ONE 128-row tile per packed head -- S_q = 1 decode and MTP S_q in
    [2, 8]. The prefill pipeline computes 512 Q rows per cga2 cluster
    (TILES_Q=2 x TILE_M=128 x CTA_MMA=2) and, with 1..G of them live, spends
    its time on dead-row MMA and softmax; this tile computes 128 rows per
    independent CTA:

    - ``TILES_Q=1`` / ``CTA_MMA=1``: one Q slab, one S/P slot, one O
      accumulator; BMM1 and BMM2 are M=128 (a quarter of the prefill cluster's
      per-tile MMA work), and there is no cross-CTA barrier traffic.
    - ``SOFTMAX_WARPGROUPS=1``: the second warpgroup owned sub-tile 1, which no
      longer exists -- 12 warps (softmax 0-3, correction 4-7, MMA 8, TMA-LDG 9,
      TMA-STG 10, scheduler 11), the d256 flavor's layout.
    - ``STAGES_KV=3`` with the Q/O slab aliased: 32 (Q u O) + 3 x (32 K + 32 V)
      = 224 KiB, under the SM100 227 KiB cap. At cga1 every CTA streams full
      K/V tiles, so the deeper ring keeps more of the KV cache in flight per SM
      while the tile is memory-bound.

    Selected by the adapter for the (128, 128) f16/bf16 flavor whenever the
    plan's ``TILE_CGA_M`` knob is 1 (``api_dsl._load_sm100_kernel_module``); the
    heuristics propose cga=1 exactly when ``S_q * pack_g <= 128`` (``pack_g`` =
    the candidate's own packing: ``PACK_G`` packed -- the whole group G, or its
    largest divisor of 128 under partial PackGQA -- 1 unpacked).
    """

    CGA_M: int = 1
    CTA_MMA: int = 1
    QO_ALIAS: int = 1

    TILES_Q: int = 1
    # One live Q tile and no second KV stream: the inherited CfgD128 defaults say 2.
    Q_TILES_LIVE: int = 1
    DUAL_KV_STREAM: int = 0
    STAGES_KV: int = 3

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32

    # One softmax warpgroup @ warps 0-3, correction @ 4-7, single roles @ 8..11.
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4  # unused: the kernel dispatches no second warpgroup
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    # 4 softmax + 4 corr + 1 MMA + 1 TMALDG + 1 TMASTG = 11 arrivers (cga1).
    READ_TILE_ARRIVERS: int = 11

    # Ragged Q/O/Stats over paged K/V (TemplateParams.ragged_q): the TMA-LDG
    # warp reads the batch's Q ragged offset as the row coordinate over the
    # packed Q view; the combine places the final rows.  Split path mandatory.
    RAGGED_Q: int = 0


def _validate_cfg_d128_decode(cfg: CfgD128Decode) -> None:
    """Consistency checks on the d128 decode-tile geometry."""
    checks = (
        (
            cfg.Q_TILES_LIVE == cfg.TILES_Q == 1 and cfg.DUAL_KV_STREAM == 0,
            "d128 decode: one live Q tile, no second KV stream (Q_TILES_LIVE / DUAL_KV_STREAM must not inherit CfgD128's defaults)",
        ),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d128 decode: f16/bf16 only (the fp8 families keep the prefill tile)"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d128 decode: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128 decode: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128 decode: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128 decode: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 1 and cfg.CTA_MMA == 1, "d128 decode: one independent CTA per tile (cga1)"),
        (cfg.QO_ALIAS == 1, "d128 decode: Q and O share one SMEM slab (pays for the third KV stage)"),
        (cfg.TILES_Q == 1, "d128 decode: TILES_Q must be 1 (one 128-row Q tile per CTA)"),
        (cfg.TILE_M == 128 and cfg.TILE_N == 128 and cfg.TILE_K in (128, 192) and cfg.TILE_O == 128, "single-Q tile: QK width 128 or 192, V width 128"),
        (cfg.STAGES_KV == (3 if cfg.TILE_K == 128 else 2), "single-Q tile: KV stages must fit the QK width"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128 decode: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4, "d128 decode: one softmax warpgroup, four correction warps"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d128 decode: 12 warps / 384 threads"),
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and cfg.CORR_WARP_BASE == 4
            and cfg.MMA_WARP_ID == 8
            and cfg.TMALDG_WARP_ID == 9
            and cfg.TMASTG_WARP_ID == 10
            and cfg.SCHED_WARP_ID == 11,
            "d128 decode: role layout and warp count disagree",
        ),
        (cfg.READ_TILE_ARRIVERS == 11, f"d128 decode: expected READ_TILE_ARRIVERS=11, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d128 decode: f16 K=16 MMA phases"),
        (
            not cfg.THD_VARLEN or ((cfg.TILE_K == 128 and cfg.SPLIT_KV > 1) or (cfg.TILE_K == 192 and not cfg.PAGED_KV and not cfg.PACK_GQA)),
            "single-Q THD: D128 split or unpacked nonpaged D192",
        ),
        (
            cfg.RAGGED_Q == 0 or (cfg.SPLIT_KV >= 2 and cfg.PAGED_KV == 1 and cfg.SEQ_Q_LENS_PRESENT == 0),
            "d128 decode: RAGGED_Q rides the split path over paged K/V (SPLIT_KV >= 2, PAGED_KV, no dense Q-length trim)",
        ),
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128,
            "d128 decode: 128 B swizzle on every operand",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128_decode(params: TemplateParams) -> Tuple[CfgD128Decode, TmaIters]:
    """Config for the shared single-Q half pipeline at ``cta_mma=1``.

    Backstop, like every ``make_cfg_*``: the (128, 128) f16/bf16 row admits
    cga=1 on dense graphs and on the ragged-Q-over-paged-KV leg (``ragged_q``,
    S_q(max) == 1), and the adapter routes exactly those combinations here;
    D192/V128 reuses it for unpacked nonpaged THD, with two KV stages to
    accommodate the wider Q/K slabs. Anything else raising below is a gap
    in those gates.
    """
    _validate_params("d128", params)
    if params.cta_mma != 1:
        raise ValueError(f"d128 decode: the decode tile is cga1 only (cta_mma=1); got cta_mma={params.cta_mma}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"d128 decode: f16/bf16 inputs only (DTYPE_QKV 2/3); got {params.dtype_qkv}")
    d_qk = params.single_q_head_dim
    if d_qk not in (128, 192):
        raise ValueError("single-Q tile: QK width must be 128 or 192")
    if d_qk == 192 and not (params.thd_varlen and not params.paged_kv and not params.pack_gqa):
        raise ValueError("D192 single-Q tile requires unpacked nonpaged THD")
    if params.thd_varlen and not ((d_qk == 192 and not params.paged_kv) or (d_qk == 128 and params.split_kv > 1)):
        raise ValueError("single-Q THD: D128 split or unpacked nonpaged D192")
    if params.pv_bf16 or not params.emit_amax_o:
        raise ValueError("d128 decode: pv_bf16 / emit_amax_o are MXFP8-only experiment axes")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    cfg = CfgD128Decode(
        TILE_K=d_qk,
        STAGES_KV=3 if d_qk == 128 else 2,
        THD_VARLEN=int(params.thd_varlen),
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(d_qk, b),
        K_SWZ_BYTES=q_swz_bytes(d_qk, b),
        V_SWZ_BYTES=v_swz_bytes(128, 1, b),
        O_SWZ_BYTES=o_swz_bytes(128, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA, as on the d128 prefill tile: the kernel packs the
        # largest divisor of the group that divides the 128-row tile
        # (HEADS_PER_TILE = PACK_G; a group sharing no factor with it raises here).
        PACK_G=_pack_g(params, CfgD128Decode.TILE_M, partial=True),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        RAGGED_Q=int(params.ragged_q),
    )
    _validate_cfg_d128_decode(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d64 flavor — d_qk = d_v = 64, SM100 (Blackwell), gpt-oss-class models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD64(CfgD128):
    """gpt-oss flavor: the d128 pipeline at a native d_qk = d_v = 64 geometry.

    Until this flavor existed a d=64 graph was served by the d128 kernel's
    ENVELOPE (``_pick_flavor`` walks smallest-first and (128, 128) was the
    floor), so every tile box stayed 128 wide and the TMA zero-filled columns
    64..127.  Those columns are exact zero terms -- correct, but the MMA still
    issued them, and measured on B300 the d128 kernel took the SAME time at
    d=64 as at d=128 (gpt-oss B=2, H=128, SWA=128: 0.277 vs 0.281 ms at S=2K,
    i.e. halving the head dim bought nothing).

    Both tile extents are native here: ``TILE_K=64`` halves the Q/K slabs and
    runs QK^T in 4 K-phases instead of 8, and ``TILE_O=64`` halves the V/O
    slabs and the P.V accumulator so BMM2 stops accumulating a 128-wide O.

    NOTE for anyone re-tuning this: ``N_BMM2_CHUNKS`` / ``BMM2_CHUNK_SIZE`` are
    derived from **TILE_N**, not TILE_O -- the softmax reads S_acc in
    ``N_BMM2_CHUNKS`` slices of a hardcoded 64 columns along the KV axis and
    signals one ``mb_bmm2_ready`` per slice.  Scaling them with TILE_O makes
    the softmax read half of S and signal half the barriers (silent NaN), or
    desync against that hardcoded 64 (deadlock).  They are inherited from
    CfgD128 unchanged and must stay that way while TILE_N is 128.

    This matches the geometry of cuDNN's own native kernel for the shape,
    ``...flash_fprop_f16_knob_1_128x128x64_4x1x1_cga1x1x1`` (dumped on both
    B200/sm100 and B300/sm103): 128x128x64.
    """

    TILE_K: int = 64
    TILE_O: int = 64

    # N_BMM2_CHUNKS / BMM2_CHUNK_SIZE are deliberately NOT overridden: they are
    # TILE_N-derived (see the class docstring), and TILE_N is unchanged at 128.


def _validate_cfg_d64(cfg: CfgD64) -> None:
    """Consistency checks on the native d64 geometry."""
    checks = (
        (cfg.TILE_K == 64 and cfg.TILE_O == 64, "d64: d_qk = d_v = 64"),
        (cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16), "d64: DTYPE_QKV must be E4M3/E5M2/BF16/FP16"),
        (cfg.DTYPE_QKV <= 1 or cfg.DTYPE_O == cfg.DTYPE_QKV, "d64: DTYPE_O must equal DTYPE_QKV for half input"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d64: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d64: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d64: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d64 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d64: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILES_Q == 2, "d64: TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d64: SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d64: CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d64: 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d64: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if cfg.DTYPE_QKV <= 1 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if cfg.DTYPE_QKV <= 1 else 16)),
            "d64: TILE_K_HW must be 32 for FP8/MXFP8 inputs and 16 for f16/bf16",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d64: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d64: V/O swizzle out of range"),
        (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, "d64: N_BMM2_CHUNKS * BMM2_CHUNK_SIZE must equal TILE_N (KV axis), not TILE_O"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d64(params: TemplateParams) -> Tuple[CfgD64, TmaIters]:
    if _small_tile_form_requested(params):
        # A small-tile SEAM on the d64 record (a bias / kv_mask form, the pair item, a ring depth, a tail / quiet / max form, Q in TMEM, a
        # second S slot, a third softmax warp group, the 64-wide KV tile): the shared small-tile factory at SM100_ARCH.  The plain d64
        # record below is untouched -- its renderings (and the fp8 / mxfp8 d64 files') stay byte-identical.
        return _make_cfg_small_d_sm100(params, tile_k=64)
    _validate_params("d64", params)
    if params.decode_tile:
        raise ValueError("d64: decode_tile selects the decode tile (make_cfg_d64_decode), not the prefill one")
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= 1  # E4M3/E5M2 inputs: the FP8 / MXFP8 d128 kernel files at d_flavor=64
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    if fp8 and params.decode_tile:
        raise ValueError("d64: the decode tile is f16/bf16 only")
    # Same K-path rule as d128: FP8/MXFP8 pin the Blackwell K=32 QMMA path.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    b_v = 2 if params.pv_bf16 else b
    cfg = CfgD64(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=O_BLOCK_SCALE_BY_DTYPE[dtype_o],
        O_PACK_DIV=o_pack_div(dtype_o),
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # Mirror d128's rule rather than exploiting d64's smaller slabs: the
        # kernel body reads QO_ALIAS, so it is not a free SMEM knob, and only
        # the aliased arrangement is validated at cga1.
        QO_ALIAS=1 if params.cta_mma == 1 else 0,
        Q_SWZ_BYTES=q_swz_bytes(64, b),
        K_SWZ_BYTES=q_swz_bytes(64, b),
        V_SWZ_BYTES=v_swz_bytes(64, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(64, b_o, o_pack_div(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=16 if params.pv_bf16 else tile_k_hw_fp8,
        # Deeper KV pipeline than d128's 2. The halved d64 slabs pay for it
        # (cga1: 32 KiB Q u O + 64 K + 64 V = 160 KiB against the 227 KiB cap;
        # cga2: 128 KiB), and ncu says this is where the room is worth spending:
        # `long_scoreboard` -- warps blocked on a memory dependency -- is by far
        # the dominant stall for this kernel (56.6% at SWA=128, 58.4% at full
        # causal), against ~2% for barrier stalls.
        STAGES_KV=4,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # partial=False: the capability row leaves (64, 64) out of
        # pack_gqa_partial_d_shapes, so only a group that FULLY divides TILE_M
        # is admitted here. _pack_g raises on a group the gate would not pass,
        # which is the check this used to open-code after _validate_cfg_d64.
        PACK_G=_pack_g(params, CfgD64.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d64(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# The f16/bf16 SMALL head-dim tiles of the d128 FILE on SM100 -- TILE_K = TILE_O = 32 / 64 WITH the small-tile seams
# (the additive bias / kv_mask arms, the (n, n+1) pair item, the streamed ring, the row-max forms, the padded tail and dead-sub-tile
# forms, the double-buffered S, Q in TMEM, the third softmax warp group): the Rubin body's seams on the sm100 twin, behind the SAME
# Cfg fields (config_d128_family.D128FamilySmallTileFields).  The PLAIN d64 record keeps make_cfg_d64's own path below (byte-identical
# renderings; the fp8 / mxfp8 d64 files too); d32 always builds here.
# ---------------------------------------------------------------------------

# The SM100 BUDGET the shared d128-family budgets / validators take (config_d128_family.D128FamilyArch): the 227 KiB opt-in carveout (no
# oversized mode), the same 2 KiB fixed slack the Rubin validator spends (barriers + the scheduler ring + the TMEM pointer: 480 B measured
# plus 16 B per bias barrier row), 512 TMEM columns allocated WITHOUT is_exclusive (the 4.7.0 floor rejects the kwarg; a 512-column
# allocation must never trace it).  At this budget the D 64 @128 cga1 forms hold NO bias at STAGES_KV 4 (192 KiB of operands leave no
# 32 KiB tile beside the slack) where Rubin holds 128 / 384 columns -- the arithmetic decides, never a Rubin threshold.
SM100_ARCH = D128FamilyArch(name="SM100", smem_usable_bytes=_SM100_MAX_DYN_SMEM, smem_fixed_overhead=2 * 1024, tmem_total_cols=512, tmem_exclusive_alloc=False)
# The small head-dim shapes of the SM100 f16/bf16 d128 file and the cga widths their bodies are validated at (the sm107 sets' twins; the
# engine row's d_shapes / cgas_by_d_shape, the adapter's pool and the tests read these -- rule 8b').  cga1 is the default at both widths;
# d64 also admits cga2 (its per-CTA V operand is the 32-byte MN-major atom the shipped sm100 d64 cga2 build runs); at TILE_O 32 a cga2 V
# operand would be a 16-column atom no shipped kernel runs -- declined until measured.
SM100_F16_SMALL_D_SHAPES = frozenset({(32, 32), (64, 64)})
SM100_SMALL_D_CGAS = {(32, 32): frozenset({1}), (64, 64): frozenset({1, 2})}
# The small shape that carries the row's DENSE leg only on SM100 (the engine row's ``dense_only_d_shapes`` and the adapter's flavor pool read
# ONE constant, rule 8b'): the 32-wide tile -- a graph that asks for THD, paged KV, split-KV, PackGQA, the score side outputs or a cga outside
# {1} walks PAST it onto the native d64 flavor (which, unlike its Rubin twin, carries those legs itself) or the d128 one.  (64, 64) is NOT in
# the set on SM100: the native d64 flavor serves THD / paged / split / PackGQA natively and keeps every leg it served before the small-tile seams.
SM100_DENSE_ONLY_D_SHAPES = frozenset({(32, 32)})
# The KV tile widths each small shape renders (TemplateParams.tile_n): 64 / 128, as on Rubin; 256 is refused by the TMEM predicate.
SM100_TILE_NS_BY_D_SHAPE = (((32, 32), frozenset({64, 128})), ((64, 64), frozenset({64, 128})))
# The K / V ring depth of the SM100 small tiles: 4 = the shipped sm100 d64 flavor's depth (its ncu read long_scoreboard as the dominant
# stall; a quarter / half of d128's slabs, so 4 stages are 96 / 192 KiB at cga1) -- the depth the body is validated at on both archs.
SM100_SMALL_D_STAGES_KV_DEFAULT = 4
# PRIVATE, process-local sweep seam (``monkeypatch.setattr`` in a test or a sweep driver that resets it to None; never a plan path, an
# import or a config): pins the K / V ring depth of the SM100 32- / 64-wide f16/bf16 small-tile forms (2 / 3 / 4 = SMALL_D_STAGES_KV) -- the
# A/B arm of the D 64 @128 budget decision (STAGES_KV 2: operands 128 KiB, a 3-tile resident block / a 3-stage ring; vs TILE_N 64; vs
# TILE_M 64).  Read by the ONE helper the budgets and the factory call, so a pinned depth moves the bias budgets with it.  It moves Cfg
# fields WITHOUT moving the TemplateParams record, so api_dsl folds :func:`sm100_sweep_seam_state` into the template-module cache key and
# the source digest (two renderings that differ only by the depth never share a module or an artifact).  The PLAIN d64 record (no small-tile
# field set) keeps make_cfg_d64's own depth whatever this seam says.
_STAGES_KV_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm100_sweep_seam_state() -> Optional[tuple]:
    """The SM100 small tiles' private sweep seams as ONE hashable, ``None`` when none is set (every plan path): the sm100 twin of
    ``config_sm107.sweep_seam_state``, folded into the template-module key by ``api_dsl._load_kernel_template``.  The K / V ring depth
    seam alone keeps its two-field form; the bias pre-load and the ring-wait seams (both module constants the kernel reads at trace
    time, invisible to the TemplateParams record) append when set, so two renderings that differ only by one of them never share a
    module or an artifact in one process."""
    skv, pre, spin = _STAGES_KV_OVERRIDE_FOR_SWEEPS, _BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS, _SPIN_RING_WAITS_OVERRIDE_FOR_SWEEPS
    if skv is None and pre is None and spin is None:
        return None
    state = ("sm100_sweep_seams", None if skv is None else int(skv))
    if pre is not None or spin is not None:
        state = state + (("bias_preload_chunks", None if pre is None else int(pre)), ("spin_ring_waits", None if spin is None else bool(spin)))
    return state


# The SM100 kernel's ``BIAS_PRELOAD_CHUNKS`` per KV tile width (bias chunks pre-loaded ABOVE the ``bmm1_done`` wait): the Rubin table
# (``config_sm107.SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N``: 0 at the 128-wide tile, 1 at the 64-wide one) as the PRIOR until the GB200 sweep
# measures it -- said so here, not applied as a measurement.  The sweep seam below pins the constant every rendering of the process gets
# (``monkeypatch.setattr`` in a test or a sweep driver that resets it to None; never a plan path, an import or a config).
SM100_BIAS_PRELOAD_CHUNKS_BY_TILE_N = {128: 0, 64: 1}
_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm100_bias_preload_chunks(tile_n: int) -> int:
    """The SM100 kernel's ``BIAS_PRELOAD_CHUNKS`` for a rendering at KV tile width ``tile_n`` (:data:`SM100_BIAS_PRELOAD_CHUNKS_BY_TILE_N`,
    the Rubin PRIOR); a width outside the table keeps the one-chunk default so an unrelated rendering's instruction stream is unchanged.
    The sweep seam ``_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS`` (validated to the kernel's domain) wins over the table."""
    if _BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS)
        if want not in BIAS_PRELOAD_CHUNKS_DOMAIN:
            raise ValueError(
                f"_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS={want}: the kernel's BIAS_PRELOAD_CHUNKS domain is {sorted(BIAS_PRELOAD_CHUNKS_DOMAIN)}"
            )
        return want
    return SM100_BIAS_PRELOAD_CHUNKS_BY_TILE_N.get(int(tile_n), 1)


# The SM100 kernel's ``SPIN_RING_WAITS`` (the retry form of its per-kv-iteration ring waits: ``spin=True`` = the hint-less
# ``mbarrier.try_wait.parity`` spin, ``False`` = the DSL's ``.timelimit`` suspend loop): the shipped constant is ``d_flavor == 128`` -- MEASURED
# per flavor on the d128 / d64 bodies (the kernel's own comment); the 32- / 64-wide small-tile forms INHERITED the d64 flavor's sleeping form
# un-A/B'd (the Rubin twin opts its small tiles IN: +4.6 / +8.3 % on the one-batch-entry ring there, ledger A24, a per-kernel / per-arch sign).
# The sweep seam below pins the constant every small-tile rendering of the process gets (``monkeypatch.setattr`` in a test or a sweep driver that
# resets it to None; never a plan path, an import or a config); the d128 flavor keeps its measured form whatever the seam says.
_SPIN_RING_WAITS_OVERRIDE_FOR_SWEEPS: Optional[bool] = None
# MEASURED on GB200 (152 SMs @ 1830 MHz, cc 10.0, 2026-10-14; in-process A/B, 3 x 100 graph replays, a control twin, every window at 1830): the BIAS
# body of the 32- / 64-wide tiles runs FASTER with the hint-less ring waits -- D 64 S 640 / 768 bias 579.9 -> 526.7 us / 812.9 -> 738.1 us (+10.1 /
# +10.1 %), S 512 +9.6 %, S 256 +5.4 %; bias + kv_mask word +17.6 / +17.9 / +14.7 %; the padded leg +9.3 / +9.0 / +8.0 % (the Rubin twin's sign on
# the same forms, ledger A24) -- so the bias renderings take ``True``; the PLAIN small-tile forms and the kv_mask-ONLY body were not measured and
# keep the d64 flavor's sleeping form (a per-kernel / per-arch sign: never flipped without its own A/B).
SM100_BIAS_BODY_SPIN_RING_WAITS = True


def sm100_spin_ring_waits(d_flavor: int, *, bias_body: bool = False) -> bool:
    """The SM100 kernel's ``SPIN_RING_WAITS`` for a rendering of head-dim flavor ``d_flavor``: the shipped per-flavor constant (``True`` on the
    d128 flavor, ``False`` on the plain d64 flavor, the plain small tiles and the kv_mask-only body) or :data:`SM100_BIAS_BODY_SPIN_RING_WAITS` on
    the BIAS body (``bias_body``: ``CFG.BIAS_MODE`` set, with or without the kv_mask word) of the 32- / 64-wide tiles; on the 32- / 64-wide tiles
    the sweep seam ``_SPIN_RING_WAITS_OVERRIDE_FOR_SWEEPS`` wins when set (folded into :func:`sm100_sweep_seam_state`)."""
    if int(d_flavor) in (32, 64) and _SPIN_RING_WAITS_OVERRIDE_FOR_SWEEPS is not None:
        return bool(_SPIN_RING_WAITS_OVERRIDE_FOR_SWEEPS)
    if int(d_flavor) in (32, 64) and bias_body:
        return SM100_BIAS_BODY_SPIN_RING_WAITS
    return int(d_flavor) == 128


def sm100_small_d_stages_kv(tile_k: int, cta_mma: int) -> int:
    """The K / V ring depth of an SM100 small-tile form: the seam when set (validated to SMALL_D_STAGES_KV), else
    :data:`SM100_SMALL_D_STAGES_KV_DEFAULT`.  ONE helper for the budgets and the factory (rule 8b')."""
    if tile_k > 64:
        return CfgD128.STAGES_KV
    if _STAGES_KV_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_STAGES_KV_OVERRIDE_FOR_SWEEPS)
        if want not in SMALL_D_STAGES_KV:
            raise ValueError(
                f"_STAGES_KV_OVERRIDE_FOR_SWEEPS={want}: the SM100 32- / 64-wide tiles' K / V ring is validated at {sorted(SMALL_D_STAGES_KV)} stages "
                f"(SMEM fit + the ring-wrap axis of the fresh-process census); a depth outside it is unvalidated, not merely slow"
            )
        return want
    return SM100_SMALL_D_STAGES_KV_DEFAULT


def sm100_bias_resident_tiles(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """How many ``tile_n``-column bias tiles the SM100 carveout holds beside the d128 family's Q / O slabs and K / V rings at
    ``tile_k = tile_o`` and the small tiles' ring depth (``config_d128_family.bias_resident_tiles`` at ``SM100_ARCH``; the Q / O slabs
    are NOT aliased on the small-tile forms).  0 = no resident block fits.  Derived at 4 stages: 2 (one-batch-entry) / 4 (pair) at
    the 32-wide tile, 5 / 10 at its 64-wide KV tile; **0 at the 64-wide tile's cga1 on either item**, 1 at its cga2 one-batch-entry
    item, 3 / 6 at its 64-wide KV tile (cga1, one-batch-entry / pair).  Never a Rubin threshold."""
    stages = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128 if tile_m is None else int(tile_m)
    return bias_resident_tiles(SM100_ARCH, tile_k, cta_mma, stages_kv=stages, tile_m=tile_m, bpe_val=bpe(dtype_qkv), tile_n=tile_n, tiles_q=tiles_q, pair=pair)


def sm100_bias_resident_s_kv_max(d_flavor: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, pair: bool = False, tiles_q: int = 2) -> int:
    """The largest S_kv (in whole tiles) whose bias row-block the resident form serves at this width and cga on SM100 (0 = none):
    the engine row's mismatch, the adapter's check_support and the factory read ONE arithmetic (rule 8b')."""
    return sm100_bias_resident_tiles(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair) * tile_n


def sm100_bias_ring_stages(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """The deepest STREAMED bias ring the SM100 carveout holds beside the operands at this width (``config_d128_family.bias_ring_stages``
    at ``SM100_ARCH``): 3 / 2 stages, 0 when fewer than 2 fit.  Derived at 4 stages: 2 at the 32-wide tile's one-batch-entry item
    (its 64 KiB stages), 3 on its pair item; 0 at the 64-wide tile's 128-wide forms at cga1; 3 at its 64-wide KV tile."""
    stages = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128 if tile_m is None else int(tile_m)
    return bias_ring_stages(SM100_ARCH, tile_k, cta_mma, stages_kv=stages, tile_m=tile_m, bpe_val=bpe(dtype_qkv), tile_n=tile_n, tiles_q=tiles_q, pair=pair)


def sm100_bias_mode_pick(
    d_flavor: int, cta_mma: int, s_kv: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, pair: bool = False, request=None, tiles_q: int = 2
) -> int:
    """The ``TemplateParams.bias_mode`` of a BIAS plan on the SM100 small tiles at this width -- the BUDGET pick (``BIAS_MODE_RESIDENT`` when
    the resident block holds ``s_kv``, else ``BIAS_MODE_STREAMED`` when a ring of >= 2 stages fits, else 0 = no form at this width, a typed
    decline); ``request`` pins the form and returns 0 when THAT form does not fit here -- honoured or declined, never degraded.  The Rubin
    pick's MEASURED preferences (the one-batch-entry item streaming at the 32-wide tile, the pair ring's multicast default) are Rubin
    priors, NOT applied here: the SM100 column is the sweep's deliverable, and until it lands the budget decides."""
    resident_ok = s_kv <= sm100_bias_resident_s_kv_max(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair, tiles_q=tiles_q)
    ring_ok = sm100_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair) > 0
    if request is not None:
        want = int(request)
        if want == BIAS_MODE_RESIDENT:
            return BIAS_MODE_RESIDENT if resident_ok else BIAS_MODE_NONE
        if want == BIAS_MODE_STREAMED:
            return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE
        if want == BIAS_MODE_STREAMED_MULTICAST:
            return BIAS_MODE_STREAMED_MULTICAST if (ring_ok and pair and int(cta_mma) == 1) else BIAS_MODE_NONE
        return BIAS_MODE_NONE
    if resident_ok:
        return BIAS_MODE_RESIDENT
    return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE


def sm100_bias_stages_pick(
    d_flavor: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, pair: bool = False, request: int = 0, tiles_q: int = 2
) -> int:
    """The ``STAGES_BIAS`` of a STREAMED plan on the SM100 small tiles: the request when it is in ``BIAS_RING_STAGES`` and fits, else the
    derived deepest ring (both item forms -- the Rubin one-batch-entry item's measured 2-stage preference is a prior the sweep decides);
    0 = no ring fits (the factory raises, the row / adapter decline)."""
    deepest = sm100_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair)
    want = int(request) if request else 0
    if want:
        return want if (want in BIAS_RING_STAGES and want <= deepest) else 0
    return deepest


# The fp32 ADDITIVE kv_mask vector TOGETHER WITH a bias is served at the 64-wide KV tile only on this line too: the 128-wide rendering spills on
# the softmax role (STL / LDL 53 / 64 on the sm_100a rendering, 2026-10-14; the Rubin finding, config_sm107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N).
SM100_FP32_VECTOR_WITH_BIAS_TILE_N = 64
# The bias body on the PADDING leg (per-row kv_lengths) takes the 64-wide KV tile on SM100 WHILE the 64-wide resident block holds S_kv -- MEASURED on
# GB200 2026-10-14 (the 128-wide form pays a half-empty tail tile whenever the live length is off the 128 multiple: +21.8 % at D 32 S 256 f 0.75;
# the 64-wide RING above that budget loses -2.3 % at S 384), a bounded form of the Rubin width table's padding rule; the dense bias body keeps the
# 128-wide tile (the 64-wide one measured +1.2 / -3.3 % there).
SM100_PADDED_BIAS_TILE_N = 64


def sm100_tile_n_pick(d_flavor: int, s_kv: int, *, has_bias: bool, has_kv_mask: bool, kv_mask_fp32: bool = False, padded: bool = False, domain=None) -> int:
    """The KV tile width of a plan on the SM100 small tiles when the TILE_N knob is unset -- ONE arithmetic for the engine row
    (``engines._bias_arm_decline``), the heuristics (``_resident_arm_tile_n``) and the adapter (``_effective_tile_n``), rule 8b'.
    The 128-wide tile by default (the Rubin measured width table's other rules -- 64 for a bias with a kv_mask, for a bias at one or two kv
    tiles -- are Rubin PRIORS, NOT applied here), except: the one form the 128-wide rendering does not serve, the fp32 additive vector WITH
    a bias, lands on :data:`SM100_FP32_VECTOR_WITH_BIAS_TILE_N`; the bias body on the PADDING leg lands on :data:`SM100_PADDED_BIAS_TILE_N`
    while that tile's resident block holds ``s_kv`` (MEASURED on GB200 2026-10-14: the 128-wide form pays a half-empty tail tile off the 128
    multiple; the 64-wide ring above the resident budget loses); and the BUDGET rule -- 64 when no bias
    form holds ``s_kv`` at the 128-wide tile at any validated cga while one does at the 64-wide tile.  A width outside ``domain`` (the row's
    ``tile_ns_by_d_shape`` entry) falls back to the domain's minimum; a shape without a domain keeps 128."""
    dom = frozenset(domain) if domain is not None else frozenset({128})
    pick = SM100_FP32_VECTOR_WITH_BIAS_TILE_N if (has_bias and has_kv_mask and kv_mask_fp32) else 128
    if (
        has_bias and padded and SM100_PADDED_BIAS_TILE_N in dom and int(d_flavor) == 32
    ):  # the measured flavor; at D 64 the 64-wide tile is the budget default above S_kv 128 already
        # MEASURED (GB200, 152 SMs @ 1830 MHz, cc 10.0, 2026-10-14; both widths in one process, A/B/A x 3 x 100 with a control twin): the bias body on
        # the PADDING leg (per-row kv_lengths) runs faster at the 64-wide KV tile WHILE ITS RESIDENT BLOCK HOLDS S_kv -- the 128-wide form pays a
        # half-empty tail tile off the 128 multiple (D 32, S 256, kv_len 192: 49.15 -> 40.34 us = +21.8 %); above the 64-wide resident budget the
        # 64-wide RING loses to the 128-wide one (D 32, S 384, kv_len 288: -2.3 %; S 512, kv_len 384: -15.5 %), so the ring forms keep the 128-wide tile.
        if any(
            sm100_bias_resident_s_kv_max(int(d_flavor), cga, tile_n=SM100_PADDED_BIAS_TILE_N) >= int(s_kv)
            for cga in SM100_SMALL_D_CGAS[(int(d_flavor), int(d_flavor))]
        ):
            pick = SM100_PADDED_BIAS_TILE_N
    if has_bias and pick == 128 and 64 in dom and (int(d_flavor), int(d_flavor)) in SM100_SMALL_D_CGAS:
        # The BUDGET rule (the only non-default pick without a measurement): when no bias form -- the resident block or a >= 2-stage ring --
        # holds s_kv at the 128-wide tile at ANY validated cga of the flavor, while one does at the 64-wide tile, the plan takes the 64-wide
        # tile (a D 64 bias graph above S_kv 128 fits only there on this line: cga1 holds no 128-wide block, cga2 one tile and no ring).
        cgas = SM100_SMALL_D_CGAS[(int(d_flavor), int(d_flavor))]
        if (
            sm100_bias_auto_cga(int(d_flavor), int(s_kv), cgas, tile_n=128) is None
            and sm100_bias_auto_cga(int(d_flavor), int(s_kv), cgas, tile_n=64) is not None
        ):
            pick = 64
    return pick if pick in dom else min(dom)


# The PLAIN padded body's tail form on the SM100 small tiles (TemplateParams.padded_tail_form): 0 = the legacy second (software-max) body for
# the tail kv tile, 1 = the PRE-MASKED tail (one unmasked body; the tail tile's out-of-range S columns rewritten as -inf in TMEM).  The SM100
# DEFAULT is 0 -- the shipped plain padded legs of the 32- / 64-wide tiles keep the body they run today; the Rubin default (1 at the 128-wide
# tile, measured +17..+81 % there) is a PRIOR the GB200 sweep decides, reached through the sweep seam or an explicit request.
SM100_PADDED_TAIL_FORM_DEFAULT = 0
# The BIAS body on the padding leg (per-row kv_lengths; no kv_mask, no band) renders its -inf body with MASK_FLAGS == MASK_PADDED: the band words,
# the R2P pass and one FSEL per cell on EVERY kv tile (the padded bias cells measured +13..+20 % per kv tile over the dense bias body on GB200,
# 2026-10-14).  Form 1 on the bias body = the Rubin twin's arm (its ledger A44): the -inf body renders its band mask at MASK_NONE and each softmax
# lane pre-masks its own row's out-of-range S columns in TMEM on the tail tile only.  SERVED on SM100 behind a request or the sweep seam; its
# knob-less default per KV tile width is 0 until the GB200 A/B lands a rule (the entries below are that rule's landing place).
SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N = {128: 1, 64: 1}
_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS: Optional[int] = None

# The DEAD-SUB-TILE QUIET body on the SM100 small tiles' -inf body (TemplateParams.dead_subtile_quiet; a bias or a kv_mask, the one-batch-entry
# item): a softmax sub-tile with no live row (the batch entry's S_q ends inside the 256-row item: S_q % (256 x cga) in 1..128 -- S 384 / 640 of the
# triangle grid) runs the full -inf body on rows nobody stores (0, the legacy form) or the quiet body (1: the live body's barrier sequence, no load,
# no math).  MEASURED on GB200 (152 SMs @ 1830 MHz, cc 10.0, 2026-10-14; in-process A/B beside the shipped default, 3 x 100 graph replays, a control
# twin, every window at 1830; speed-up = t(full) / t(quiet) - 1): over the spinning bias body D 64 S 640 / 384 B 1 +3.5 / +5.6 %, S 384 B 4 +5.2 %, D 32 S 640 / 384 B 1 +3.7 / +6.9 %, S 384 B 4 +5.9 %; with the kv_mask word D 64 +8.8 / +12.1 / +11.7 %, D 32 +1.1 / +7.1 / +4.1 %; the 64-wide padded D 64 cells +3.5 / +5.7 / +6.3 %; the 128-wide padded D 32 cells a LOSS -7.9 / -7.0 / -9.5 % (S 640 / 384 B 1, S 384 B 4).  So the knob-less default is 1 on the DENSE-length -inf forms
# (the bias body with or without the kv_mask word) and 0 on the PADDED leg (per-row kv_lengths), where the pick returns 0; wherever every sub-tile is
# live (S 256 / 512 / 768 / 1024 / 2048 of the grid) the rows rule returns 0 and nothing moves; the 128-wide PADDED body keeps the full body (the pick's
# width rule: a LOSS there).  The PLAIN body's quiet form is not wired on this line
# (the kernel refuses it); the Rubin pick (`sm107_dead_subtile_quiet_pick`) serves the OTHER half -- its plain body -- so the two picks are disjoint.
SM100_INF_BODY_DEAD_SUBTILE_QUIET_DEFAULT = 1
_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm100_dead_subtile_quiet_pick(
    d_flavor: int,
    *,
    has_bias: bool,
    has_kv_mask: bool,
    request: Optional[int] = None,
    s_q: Optional[int] = None,
    cga: int = 1,
    tile_m: int = 128,
    padded: bool = False,
    tile_n: int = 128,
) -> int:
    """The ``TemplateParams.dead_subtile_quiet`` of a plan on the SM100 small tiles: 0 off the eligible set (the plain body -- its quiet form is
    not wired on this line -- the d128 flavor); on the -inf body (``has_bias`` or ``has_kv_mask``) ``request`` when the caller made one (the
    adapter's ``dead_subtile_quiet=``, validated there: honoured or declined, never degraded), else 0 when the plan's S_q (``s_q`` with its cluster
    width ``cga`` and Q tile height ``tile_m``) leaves NO dead sub-tile (the one-batch-entry item's cluster tile is 2 x tile_m x cga rows; a dead
    sub-tile exists iff ``s_q % rows`` is in 1..(rows - tile_m); without one the quiet rendering only costs its code), else the sweep seam, else
    :data:`SM100_INF_BODY_DEAD_SUBTILE_QUIET_DEFAULT` -- except the PADDED leg (``padded``: per-row kv_lengths) at the 128-wide tile (``tile_n``),
    where the quiet body measured a LOSS and the pick returns 0 (the arm stays reachable by request / the seam)."""
    if not (has_bias or has_kv_mask) or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if s_q is not None:
        rows = 2 * int(tile_m) * max(int(cga), 1)
        rem = int(s_q) % rows
        if rem == 0 or rem > rows - int(tile_m):
            return 0  # every softmax sub-tile of every item holds a live row: nothing to keep quiet
    if _DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS is not None:
        return int(_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS)
    if padded and int(tile_n) == 128:
        return 0
    return SM100_INF_BODY_DEAD_SUBTILE_QUIET_DEFAULT


def sm100_padded_tail_form_pick(
    d_flavor: int, *, padded: bool, has_bias: bool, has_kv_mask: bool, causal: bool = False, swa: bool = False, request: Optional[int] = None, tile_n: int = 128
) -> int:
    """The ``TemplateParams.padded_tail_form`` of a plan on the SM100 small tiles: 0 off the eligible set (a dense S_kv at a tile multiple, a
    kv_mask -- whose keep words fold the length in already -- a causal / SWA band, the d128 flavor); on it ``request`` when the caller made one
    (the adapter's ``padded_tail_form=``, validated there: honoured or declined, never degraded), else the sweep seam, else the default: on the
    PLAIN body :data:`SM100_PADDED_TAIL_FORM_DEFAULT` (0 at every width until measured); on the BIAS body (``has_bias``, served since the
    pre-mask entered the -inf body) :data:`SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N` at the plan's KV tile width ``tile_n``."""
    if not padded or has_kv_mask or causal or swa or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if _PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS is not None:
        return int(_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS)
    if has_bias:
        return SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N.get(int(tile_n), 0)
    return SM100_PADDED_TAIL_FORM_DEFAULT


# MEASURED on GB200 (152 SMs @ 1830 MHz, cc 10.0, 2026-10-14; in-process A/B, 3 x 100, a control twin): at the D 64 flavor's 64-wide KV tile the
# one-batch-entry item on the cga-1 3-stage STREAMED RING beats the cga-2 RESIDENT block by +77.5 % at S 256 B 1 (83.87 -> 47.26 us; the padded
# leg 73.89 -> 43.10 us = +71 %) -- the 2-CTA pair halves each CTA's rows on a 4-tile cell whose fixed cost dominates.  The cga pick therefore
# prefers cga 1 whenever a form fits there at this width (S_kv > 192 = the ring; <= 192 = the block, the narrowest-resident pick already).
SM100_D64_T64_PREFER_CGA1_RING = True


def sm100_bias_auto_cga(
    d_flavor: int, s_kv: int, cgas, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, pair: bool = False, tiles_q: int = 2
) -> Optional[int]:
    """The NARROWEST cga in ``cgas`` whose resident block holds ``s_kv`` on SM100, else the narrowest whose ring fits, else None (the
    sm107 pick's algorithm at the SM100 budget: a D 64 bias graph at TILE_N 128 lands on cga2 -- its cga1 holds no form at 4 stages),
    with ONE measured exception: at the D 64 flavor's 64-wide KV tile the cga-1 STREAMED RING is preferred over the cga-2 RESIDENT block
    (:data:`SM100_D64_T64_PREFER_CGA1_RING`; GB200 2026-10-14, S 256: the cga-2 block 83.87 us, the cga-1 ring 47.26 us = +77.5 %, which
    turns that cell from -35 % to +16 % against cuEq 0.12.0) -- the pair of CTAs shares K / V but halves each CTA's rows, and the item's
    fixed cost dominates a 4-tile cell; where cga 1 holds the block itself (S_kv <= 192) it is the narrowest-resident pick already."""
    if SM100_D64_T64_PREFER_CGA1_RING and int(d_flavor) == 64 and int(tile_n) == 64 and 1 in cgas:
        if sm100_bias_mode_pick(d_flavor, 1, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair, tiles_q=tiles_q) != BIAS_MODE_NONE:
            return 1
    for cga in sorted(cgas):
        if sm100_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1, tiles_q=tiles_q) == BIAS_MODE_RESIDENT:
            return cga
    for cga in sorted(cgas):
        if sm100_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1, tiles_q=tiles_q) != BIAS_MODE_NONE:
            return cga
    return None


# The small shapes that serve the bias at SOME validated cga -- DERIVED from the SM100 resident budget exactly as SM107_BIAS_D_SHAPES is on
# Rubin: (32, 32) through its cga1 blocks, (64, 64) through its cga2 one-batch-entry block (and its 64-wide KV tile); the d128 tile has no
# spare on either arch.  The engine row's ``bias_d_shapes`` and the adapter read this set (rule 8b').
SM100_BIAS_D_SHAPES = frozenset(
    shape for shape in SM100_F16_SMALL_D_SHAPES if any(sm100_bias_resident_tiles(shape[0], cga) > 0 for cga in SM100_SMALL_D_CGAS[shape])
)


@dataclass(frozen=True)
class CfgD32(CfgD128):
    """d_qk = d_v = 32 on the d128 body: ``TILE_K = TILE_O = 32`` (BMM1 in 2 k-steps, a 32-wide O accumulator, a quarter of
    the d128 SMEM; 64-byte Q/K/O swizzle atoms).  Serves every f16/bf16 head dim <= 32 with D % 8 == 0 by TMA zero fill.
    At this width the MMA is a quarter of d128's per score element while the softmax's exp / FMA work per element is unchanged,
    so the kernel's floor is the softmax issue rate (on cc 10.0 the MUFU.EX2 rate: 16 lanes/clk/SM), not the tensor core.
    Built by :func:`_make_cfg_small_d_sm100` only (the small-tile factory; no plain-record path: the d32 flavor IS a small tile)."""

    TILE_K: int = 32
    TILE_O: int = 32


_SMALL_TILE_RECORD_FIELDS = ("bias_mode", "kv_mask_form", "n_chunk", "bias_stages", "padded_tail_form", "dead_subtile_quiet", "bias_max_form", "q_in_tmem")


def _small_tile_form_requested(params: TemplateParams) -> bool:
    """Whether a d64 record asks for a small-tile SEAM (any of the appended small-tile fields off its default: a bias / kv_mask form, the pair
    item, a ring depth, a tail / quiet form, the row-max form, Q in TMEM, a second S slot, a third softmax warp group, the 64-wide KV tile).
    False = the plain d64 record every existing rendering is built from (``make_cfg_d64``'s own path, byte-identical)."""
    return (
        any(int(getattr(params, f, 0)) for f in _SMALL_TILE_RECORD_FIELDS)
        or int(getattr(params, "s_slots", 1)) != 1
        or int(getattr(params, "softmax_wgs", 2)) != 2
        or int(getattr(params, "tile_n", 128)) != 128
    )


def _make_cfg_small_d_sm100(params: TemplateParams, *, tile_k: int) -> Tuple[CfgD128, TmaIters]:
    """The f16/bf16 SMALL-tile factory of the SM100 d128 file (``TILE_K = TILE_O = tile_k`` in {32, 64}): the record's small-tile fields
    (bias / kv_mask / pair / ring / max form / tail / quiet / S_SLOTS / Q_IN_TMEM / softmax WGs / TILE_N) resolved against the shared
    validators at ``SM100_ARCH`` -- the sm107 factory's resolution with the SM100 budget, the SM100 ring depth, the 128-row tile and NO Q u O
    alias (the resident block needs the separate ``sO``; the plain d64 cga1 record keeps its alias on ``make_cfg_d64``'s path).  Every
    structural coupling raises here with the failure it would otherwise produce (frost-kernels.md s 5 / s 6); every rejection must also be
    a Capabilities / check_support decline."""
    flavor = f"d{tile_k}"
    tag = f"sm100 {flavor}"
    _validate_params(flavor, params)
    if params.dtype_qkv <= 1:
        raise ValueError(
            f"{tag}: the 32- / 64-wide small-tile forms are wired on the f16/bf16 d128 file only (dtype_qkv={params.dtype_qkv}); the fp8 / mxfp8 d64 flavors keep their own records"
        )
    if params.decode_tile or params.decode_q_tile:
        raise ValueError(f"{tag}: the decode tiles do not consume the small-tile record fields")
    if params.thd_varlen or params.split_kv != 1 or params.pack_gqa or params.paged_kv or params.epilogue_gate:
        raise ValueError(
            f"{tag}: the small-tile forms serve dense, unsplit, unpaged, non-PackGQA, ungated graphs (global or per-batch Q / KV lengths, the sm107 "
            f"factory's envelope) only -- THD / paged / split-KV / PackGQA ride the native d64 / d128 flavors, where those legs are wired"
        )
    cta_mma = params.cta_mma
    if cta_mma not in SM100_SMALL_D_CGAS[(tile_k, tile_k)]:
        raise ValueError(
            f"{tag}: cta_mma={cta_mma} is outside the validated widths {sorted(SM100_SMALL_D_CGAS[(tile_k, tile_k)])} for TILE_O={tile_k}; "
            f"at cga2 the per-CTA V operand is {tile_k // 2} columns = a {tile_k // 2 * 2}-byte MN-major swizzle atom"
        )
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_n = int(getattr(params, "tile_n", 128))
    if tile_n not in dict(SM100_TILE_NS_BY_D_SHAPE)[(tile_k, tile_k)]:
        raise ValueError(f"{tag}: TILE_N must be one of {sorted(dict(SM100_TILE_NS_BY_D_SHAPE)[(tile_k, tile_k)])} (got {tile_n})")
    s_slots = int(getattr(params, "s_slots", 1))
    q_in_tmem = int(getattr(params, "q_in_tmem", 0))
    if q_in_tmem not in SMALL_D_Q_IN_TMEM:
        raise ValueError(f"{tag}: q_in_tmem must be 0 (Q from SMEM) or 1 (Q resident in TMEM); got {q_in_tmem}")
    bias_mode = int(getattr(params, "bias_mode", 0))
    kv_mask_form = int(getattr(params, "kv_mask_form", 0))
    if (bias_mode or kv_mask_form) and (bias_mode not in BIAS_MODES or kv_mask_form not in KV_MASK_FORMS):
        raise ValueError(
            f"{tag}: bias_mode must be 0 / 1 (resident row-block) / 2 (streamed ring) / 3 (streamed ring, cluster multicast) and kv_mask_form "
            f"0 / 1 (bit word) / 2 / 3 (additive io-dtype / fp32 vector); got {bias_mode} / {kv_mask_form}"
        )
    n_chunk = int(getattr(params, "n_chunk", 0))
    pair = n_chunk > 0
    if pair and bias_mode == BIAS_MODE_NONE:
        raise ValueError(f"{tag}: n_chunk={n_chunk} is the (n, n+1) pair item of the BIAS arm; it needs bias_mode=1 / 2 (got {bias_mode})")
    bias_stages_req = int(getattr(params, "bias_stages", 0))
    if bias_stages_req and bias_mode not in BIAS_STREAMED_MODES:
        raise ValueError(f"{tag}: bias_stages={bias_stages_req} is the STREAMED ring's depth; it needs bias_mode=2 / 3 (got {bias_mode})")
    bias_max_form = int(getattr(params, "bias_max_form", 0))
    if bias_max_form not in BIAS_MAX_FORMS:
        raise ValueError(f"{tag}: bias_max_form must be 0 (the software row max) or 1 (the hardware upper bound); got {bias_max_form}")
    if bias_max_form == BIAS_MAX_HW_UPPER_BOUND:
        if not (bias_mode or kv_mask_form):
            raise ValueError(
                f"{tag}: bias_max_form=1 is the row-max form of the bias / kv_mask body; this record has neither arm (bias_mode=0, kv_mask_form=0)"
            )
        if kv_mask_form in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32):
            raise ValueError(
                f"{tag}: bias_max_form=1 (the hardware upper-bound row max) is served with the bit-word kv_mask or none; the additive kv_mask vector "
                f"(kv_mask_form={kv_mask_form}) has no per-tile max to bound -- keep bias_max_form=0 with it"
            )
    softmax_wgs = int(getattr(params, "softmax_wgs", 2))
    if softmax_wgs not in SMALL_D_SOFTMAX_WGS:
        raise ValueError(f"{tag}: softmax_wgs must be one of {sorted(SMALL_D_SOFTMAX_WGS)}; got {softmax_wgs}")
    if softmax_wgs == 3 and (tile_n != 64 or not pair or cta_mma != 1):
        raise ValueError(
            f"{tag}: softmax_wgs=3 serves the (n, n+1) pair item at TILE_N 64 and cga1 only (got tile_n={tile_n}, n_chunk={n_chunk}, cta_mma={cta_mma}): "
            f"its 128-register softmax holds one 64-column chunk and the one-batch-entry paths hand-unroll two sub-tiles"
        )
    stages_kv = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128
    softmax_regs, correction_regs = D128_REG_SPLIT[2 if softmax_wgs == 2 else (softmax_wgs, tile_k)]
    total_warps = d128_total_warps(softmax_wgs)
    corr_warp_base = softmax_wgs * D128_SOFTMAX_WG_WARPS
    mma_warp_id = corr_warp_base + D128_CORRECTION_WARPS
    bias_tiles = 0
    stages_bias = 0
    if bias_mode == BIAS_MODE_RESIDENT:
        bias_tiles = sm100_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, tiles_q=softmax_wgs, pair=pair)
        if bias_tiles == 0:
            raise ValueError(
                f"{tag}: no resident bias block fits beside the Q / O slabs and the {stages_kv}-deep K / V ring at TILE_K={tile_k}, TILE_N={tile_n}, cta_mma={cta_mma} "
                f"({SM100_ARCH.smem_usable_bytes // 1024} KiB usable on SM100: the 64-wide tile's 128-wide forms hold no block at cga1 at 4 stages); the bias is "
                f"served on the widths of SM100_BIAS_D_SHAPES={sorted(SM100_BIAS_D_SHAPES)} at the cga / KV width / ring depth whose budget holds a tile"
            )
    elif bias_mode in BIAS_STREAMED_MODES:
        stages_bias = sm100_bias_stages_pick(
            tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, pair=pair, request=bias_stages_req, tiles_q=softmax_wgs
        )
        if stages_bias == 0:
            raise ValueError(
                f"{tag}: no {sorted(BIAS_RING_STAGES)}-stage streamed bias ring of {bias_block_rows(softmax_wgs, tile_m, pair=pair)} x {tile_n} tiles "
                f"{'(requested bias_stages=' + str(bias_stages_req) + ') ' if bias_stages_req else ''}fits beside the Q / O slabs and the {stages_kv}-deep "
                f"K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} ({SM100_ARCH.smem_usable_bytes // 1024} KiB usable on SM100)"
            )
    mask_flags = _mask_flags_from(params)
    padded_tail_form = int(getattr(params, "padded_tail_form", 0))
    if padded_tail_form not in PADDED_TAIL_FORMS:
        raise ValueError(f"{tag}: padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {padded_tail_form}")
    if padded_tail_form == 1 and (mask_flags != MASK_PADDED or kv_mask_form):
        raise ValueError(
            f"{tag}: padded_tail_form=1 (the pre-masked tail) is wired on the f16/bf16 32- / 64-wide tiles' padded body without a kv_mask only "
            f"(MASK_FLAGS == MASK_PADDED alone; the plain or the bias body); got MASK_FLAGS={mask_flags}, kv_mask_form={kv_mask_form}"
        )
    dead_subtile_quiet = int(getattr(params, "dead_subtile_quiet", 0))
    if dead_subtile_quiet not in DEAD_SUBTILE_QUIET_FORMS:
        raise ValueError(f"{tag}: dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {dead_subtile_quiet}")
    if dead_subtile_quiet == 1 and (pair or not (bias_mode or kv_mask_form)):
        raise ValueError(
            f"{tag}: dead_subtile_quiet=1 (the dead-sub-tile quiet body) is wired on the -inf body (a bias or a kv_mask) of the one-batch-entry item only "
            f"on this line; got bias_mode={bias_mode}, kv_mask_form={kv_mask_form}, n_chunk={n_chunk}"
        )
    cls = CfgD64 if tile_k == 64 else CfgD32
    cfg = cls(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_k,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        PV_BF16=0,
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=0,
        O_PACK_DIV=1,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        # No Q u O alias on the small-tile forms: the resident block / the pair item's Q refill need the separate sO (the sm107 geometry every
        # seam's barrier analysis assumed); the plain d64 cga1 record keeps its alias on make_cfg_d64's own path.
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_k, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_k, b_o, 1),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=softmax_wgs,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=softmax_wgs,
        CORRECTION_WARPS=D128_CORRECTION_WARPS,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=correction_regs,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG_WARPS=D128_SOFTMAX_WG_WARPS,
        OTHER_WARPS=D128_OTHER_WARPS,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=D128_SOFTMAX_WG_WARPS,
        CORR_WARP_BASE=corr_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=d128_read_tile_arrivers(softmax_wgs),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        # The per-batch Q-length trim (the graph's seq_len_q under the padding mask) is the d128 body's own leg on every flavor of this file
        # (`SEQ_Q_LENS_PRESENT` folds the OOB-row store guard + the Q trim); a padded graph always binds it (engines: a dense padded graph
        # with seq_q_t compiles the trim specialization), so the small tiles take it like the sm107 factory does -- a refusal here was a
        # whole-row decline of every padded graph at D <= 32 (12 / 117 cells of the dev-box tier, 2026-10-13).
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=0,
        SPLIT_KV=1,
        PACK_GQA=0,
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=1,
        PAGED_KV=0,
        PAGE_SIZE=0,
        PADDED_TAIL_FORM=padded_tail_form,
        DEAD_SUBTILE_QUIET=dead_subtile_quiet,
        BIAS_MODE=bias_mode,
        BIAS_RESIDENT_TILES=bias_tiles,
        BIAS_BLOCK_ROWS=bias_block_rows(softmax_wgs, tile_m, pair=pair) if bias_mode else 0,
        BIAS_SWZ_BYTES=BIAS_SWZ_BYTES,
        BIAS_BOXES_PER_TILE=(tile_n * b) // BIAS_SWZ_BYTES if bias_mode else 0,
        BIAS_EMPTY_ARRIVERS=softmax_wgs * D128_SOFTMAX_WG_WARPS if (bias_mode or kv_mask_form) else 0,
        KV_MASK_FORM=kv_mask_form,
        N_CHUNK=n_chunk,
        STAGES_BIAS=stages_bias,
        BIAS_MAX_FORM=bias_max_form,
        S_SLOTS=s_slots,
        Q_IN_TMEM=q_in_tmem,
        BIAS_MCAST_CTAS=BIAS_MCAST_CTAS if bias_mode == BIAS_MODE_STREAMED_MULTICAST else 1,
    )
    # The SM100 facts the shared validators do not know: the f16 k-step and the half-precision O.
    if cfg.TILE_K_HW_BMM1 != 16 or cfg.TILE_K_HW_BMM2 != 16:
        raise ValueError(
            f"{tag}: the f16/bf16 small tiles run the 16-wide k-step (got {cfg.TILE_K_HW_BMM1} / {cfg.TILE_K_HW_BMM2}); the 2-chunk path is silently wrong on SM10x"
        )
    if cfg.DTYPE_O != cfg.DTYPE_QKV:
        raise ValueError(f"{tag}: DTYPE_O must equal DTYPE_QKV for half input")
    validate_regs(cfg, tag)
    validate_warp_layout(cfg, tag, expect_total=d128_total_warps(softmax_wgs), expect_arrivers=d128_read_tile_arrivers(softmax_wgs))
    validate_softmax_wgs(cfg, tag)
    validate_swizzles(cfg, tag)
    validate_head_dim_tile(cfg, tag, arch=SM100_ARCH)
    validate_tile_n(cfg, tag, arch=SM100_ARCH)
    validate_q_in_tmem(cfg, tag, arch=SM100_ARCH)
    validate_small_tile_masks(cfg, tag)
    validate_bias(cfg, tag, arch=SM100_ARCH)
    validate_smem(cfg, tag, arch=SM100_ARCH, qo_alias=False, bias_bytes=bias_resident_block_bytes(cfg))
    return cfg, _tma_iters(cfg)


def make_cfg_d32(params: TemplateParams) -> Tuple[CfgD32, TmaIters]:
    """The 32-wide head-dim tile of ``sm100/prefill_d128_f16.py`` (``TemplateParams.d_flavor=32``): every f16/bf16 head dim <= 32 with
    D % 8 == 0 by TMA zero fill; cga1 (SM100_SMALL_D_CGAS); dense / causal / bottom-right / SWA / padded / sink / GQA like d128 plus the
    small-tile seams; THD / split-KV / paged KV / PackGQA / the score side outputs ride the d128 flavor (the row's ``dense_only_d_shapes``)."""
    return _make_cfg_small_d_sm100(params, tile_k=32)


@dataclass(frozen=True)
class CfgD64Decode(CfgD64):
    """The d64 DECODE tile (``sm100/decode_d128_f16.py`` at ``d_flavor=64``).

    :class:`CfgD128Decode` for the narrow head geometry: same Q-tile reshape
    (one 128-row slab per independent CTA instead of the prefill cluster's
    512 rows), same masks, paged loader and split/epilogue contract, only the
    head-dim extents halved.  Every slab halves with it, so the footprint is
    16 (Q u O) + 3 x (16 K + 16 V) = 112 KiB against d128 decode's 224 KiB --
    the deepest headroom of any flavor on this line.

    Selected exactly as the d128 decode tile is: the adapter routes the
    (64, 64) f16/bf16 flavor here whenever the plan's ``TILE_CGA_M`` knob is 1
    (``api_dsl._load_sm100_kernel_module``), and ``heuristics._auto_sched_cga``
    proposes that knob when one 128-row tile covers a KV head's live Q rows.
    """

    # Inherited from CfgD64: TILE_K / TILE_O = 64. Everything below mirrors
    # CfgD128Decode's overrides -- the reshape is head-dim independent.
    CGA_M: int = 1
    CTA_MMA: int = 1
    QO_ALIAS: int = 1

    TILES_Q: int = 1
    # One live Q tile and no second KV stream: the inherited CfgD128 defaults say 2.
    Q_TILES_LIVE: int = 1
    DUAL_KV_STREAM: int = 0
    # Three 16 KiB K + 16 KiB V stages (96 KiB of KV in flight).  Six -- the
    # d128 tile's 192 KiB -- was measured 2026-10-08 on a b=128 x 128k-key
    # gpt-oss decode: B200 6.37 -> 6.19 ms (-3 %) but B300 4.99 -> 5.70 ms
    # (+14 %), so the narrow tile keeps 3; B200's 66 % of HBM against B300's
    # 86 % on this shape is not a ring-depth problem.
    STAGES_KV: int = 3

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4  # unused: the kernel dispatches no second warpgroup
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    READ_TILE_ARRIVERS: int = 11


def _validate_cfg_d64_decode(cfg: CfgD64Decode) -> None:
    """Consistency checks on the d64 decode-tile geometry.

    The d128 decode checks with the narrow extents: same reshape, same role
    layout, only TILE_K / TILE_O and the V swizzle differ (a 64-wide V row is
    128 B at cga1, so every operand still lands on the 128 B atom).
    """
    checks = (
        (
            cfg.Q_TILES_LIVE == cfg.TILES_Q == 1 and cfg.DUAL_KV_STREAM == 0,
            "d64 decode: one live Q tile, no second KV stream (Q_TILES_LIVE / DUAL_KV_STREAM must not inherit CfgD128's defaults)",
        ),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d64 decode: f16/bf16 only"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d64 decode: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d64 decode: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d64 decode: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d64 decode: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 1 and cfg.CTA_MMA == 1, "d64 decode: one independent CTA per tile (cga1)"),
        (cfg.QO_ALIAS == 1, "d64 decode: Q and O share one SMEM slab"),
        (cfg.TILES_Q == 1, "d64 decode: TILES_Q must be 1 (one 128-row Q tile per CTA)"),
        (cfg.TILE_M == 128 and cfg.TILE_N == 128 and cfg.TILE_K == 64 and cfg.TILE_O == 64, "d64 decode: 128x128 tiles, d_qk = d_v = 64"),
        (cfg.STAGES_KV == 3, "d64 decode: STAGES_KV must be 3"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d64 decode: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4, "d64 decode: one softmax warpgroup, four correction warps"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d64 decode: 12 warps / 384 threads"),
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and cfg.CORR_WARP_BASE == 4
            and cfg.MMA_WARP_ID == 8
            and cfg.TMALDG_WARP_ID == 9
            and cfg.TMASTG_WARP_ID == 10
            and cfg.SCHED_WARP_ID == 11,
            "d64 decode: role layout and warp count disagree",
        ),
        (cfg.READ_TILE_ARRIVERS == 11, f"d64 decode: expected READ_TILE_ARRIVERS=11, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d64 decode: f16 K=16 MMA phases"),
        (not cfg.THD_VARLEN or cfg.PAGED_KV, "d64 decode: THD requires paged KV"),
        (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, "d64 decode: BMM2 chunking is TILE_N-derived"),
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128,
            "d64 decode: 128 B swizzle on every operand",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d64_decode(params: TemplateParams) -> Tuple[CfgD64Decode, TmaIters]:
    """Config for ``sm100/decode_d128_f16.py`` at ``d_flavor=64`` -- the d64
    flavor at ``cta_mma=1``.  Backstop, like every ``make_cfg_*``."""
    _validate_params("d64", params)
    if not params.decode_tile:
        raise ValueError("d64 decode: this tile is selected by decode_tile=True (the d64 prefill leg also runs cga1)")
    if params.cta_mma != 1:
        raise ValueError(f"d64 decode: the decode tile is cga1 only (cta_mma=1); got cta_mma={params.cta_mma}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"d64 decode: f16/bf16 inputs only (DTYPE_QKV 2/3); got {params.dtype_qkv}")
    if params.thd_varlen and not params.paged_kv:
        # The THD leg of this tile rides the paged loader (page pools through block tables): a ragged K/V
        # needs the prefill tile's clamped descriptors.  Unsplit, the per-batch O descriptors place the
        # rows (LSE_KINDS token / head / padded); split, the combine does.
        raise ValueError("d64 decode: THD requires paged KV")
    if params.pv_bf16 or not params.emit_amax_o:
        raise ValueError("d64 decode: pv_bf16 / emit_amax_o are MXFP8-only experiment axes")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    cfg = CfgD64Decode(
        THD_VARLEN=int(params.thd_varlen),
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(64, b),
        K_SWZ_BYTES=q_swz_bytes(64, b),
        V_SWZ_BYTES=v_swz_bytes(64, 1, b),
        O_SWZ_BYTES=o_swz_bytes(64, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # partial=False, as on the d64 prefill tile: the capability row leaves
        # (64, 64) out of pack_gqa_partial_d_shapes.
        PACK_G=_pack_g(params, CfgD64Decode.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d64_decode(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d192/d128 flavor — DSv3 MLA logical d_qk = 192, d_v = 128, SM100, cga1/cga2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD192(CfgD128):
    TILE_K: int = 192
    TILE_O: int = 128
    QO_ALIAS: int = 1
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    # Separate K and V ring depths: the half-precision cga1 tile cannot afford two full K+V stages (256 KiB), but a
    # second V stage alone fits (208 KiB), so the V reload is double-buffered while K stays single-stage.  Every other
    # d192 configuration keeps both rings at STAGES_KV.
    STAGES_K: int = 2
    STAGES_V: int = 2
    # K sub-stages per tile: the TMA D-subtile count (3 x 16 KiB for half input) on the cga1 tile, so each subtile
    # is reloaded for the next tile as soon as BMM1 has read it, instead of after the whole 48 KiB tile; 1 elsewhere.
    K_SUB: int = 1
    # The L2-prefetching loader (TemplateParams.cga1_l2_prefetch on the half-input cga1 tile): the TMA-LDG warp
    # prefetches the K/V tiles L2_PREFETCH_KV kv steps ahead of their SMEM load and, with L2_PREFETCH_NEXT_TILE, the
    # next tile's Q slabs and first K/V tile; tiles with fewer than L2_PREFETCH_MIN_STEPS kv steps issue no prefetch.
    # Both distances are 0 elsewhere.
    L2_PREFETCH_KV: int = 0
    L2_PREFETCH_NEXT_TILE: int = 0
    L2_PREFETCH_MIN_STEPS: int = 4
    # Unroll factor of the TMA-LDG warp's kv loop (1 with the L2-prefetching loader, 6 otherwise).
    LDG_UNROLL: int = 6
    # The dense-causal exp2 tail emulation of the softmax (0 with the L2-prefetching loader, 1 otherwise).
    EXP2_TAIL_MIX: int = 1


def _d192_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d192 pipeline.

    Same shape as _d128_smem_bytes, but d_qk = 192 makes the Q and K slabs 1.5x
    the d128 ones, which is why this flavor needs a shallower KV pipeline:

        cga2, STAGES_KV=2 : 96(Q u O) + 48(K) + 32(V) = 176 KiB
        cga1, STAGES_KV=2 : 96        + 96    + 64    = 256 KiB  (over cap)
        cga1, STAGES_KV=1 : 96        + 48    + 32    = 176 KiB
        cga1, K=1 / V=2   : 96        + 48    + 64    = 208 KiB  (half input: STAGES_K=1, STAGES_V=2)
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV
    qo = cfg.TILES_Q * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_K * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_V * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
    return qo + k + v


def _validate_cfg_d192(cfg: CfgD192) -> None:
    """Consistency checks on the native DSv3 d192/d128 geometry."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    checks = (
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16) if fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d192: DTYPE_O must equal DTYPE_QKV for half input; FP8 allows an independent output dtype",
        ),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d192: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d192: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d192: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d192 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            cfg.STAGES_KV == ((3 if cfg.CTA_MMA == 2 else 1) if cfg.PV_BF16 else (2 if fp8 else 1) * cfg.CTA_MMA),
            "d192: STAGES_KV must be 3/1 for BF16 PV at cga2/cga1, otherwise follow the input-dtype pipeline depth",
        ),
        (
            cfg.STAGES_K == cfg.STAGES_KV and cfg.STAGES_V == (2 if (not fp8 and cfg.CTA_MMA == 1) else cfg.STAGES_KV),
            "d192: the K ring is STAGES_KV deep; the V ring is 2 deep on the half-input cga1 tile and STAGES_KV elsewhere",
        ),
        (
            cfg.K_SUB == ((cfg.TILE_K * cfg.BPE) // 128 if (not fp8 and cfg.CTA_MMA == 1) else 1),
            "d192: K is released per TMA D-subtile on the half-input cga1 tile only",
        ),
        (
            cfg.L2_PREFETCH_KV in (0, 1) and (cfg.L2_PREFETCH_KV == 0 or (not fp8 and cfg.CTA_MMA == 1)),
            "d192: the L2-prefetching loader (one kv step ahead) belongs to the half-input cga1 tile only",
        ),
        (cfg.L2_PREFETCH_NEXT_TILE == cfg.L2_PREFETCH_KV, "d192: the next-tile L2 prefetch comes with the K/V L2 prefetch"),
        (cfg.L2_PREFETCH_MIN_STEPS >= 1, "d192: L2_PREFETCH_MIN_STEPS must be at least 1"),
        (
            cfg.LDG_UNROLL == (1 if cfg.L2_PREFETCH_KV else 6),
            "d192: the TMA-LDG kv loop is unrolled 1x with the L2-prefetching loader and 6x otherwise",
        ),
        (
            cfg.EXP2_TAIL_MIX == (0 if cfg.L2_PREFETCH_KV else 1),
            "d192: the dense-causal exp2 tail emulation is folded out with the L2-prefetching loader and kept otherwise",
        ),
        (
            _d192_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d192: SMEM {_d192_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILE_K == 192 and cfg.TILE_O == 128, "d192: expected D_QK tile 192 and D_V tile 128"),
        (cfg.QO_ALIAS == (0 if fp8 else 1), "d192: Q/O SMEM alias must be disabled for FP8 and enabled for half input"),
        (cfg.TILES_Q == 2, "d192: TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d192: SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d192: CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d192: 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d192: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if fp8 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if fp8 else 16)),
            "d192: BMM1 uses FP8 K=32; BF16 PV uses BMM2 K=16",
        ),
        (
            cfg.Q_SWZ_BYTES == (64 if fp8 else 128) and cfg.K_SWZ_BYTES == (64 if fp8 else 128),
            "d192: Q/K swizzle must be 64B for FP8 and 128B for BF16/FP16",
        ),
        (
            cfg.V_SWZ_BYTES == v_swz_bytes(128, cfg.CTA_MMA, cfg.BPE_V) and cfg.O_SWZ_BYTES in ((64, 128) if fp8 else (128,)),
            "d192: V/O swizzle is inconsistent with the input/output dtype",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def d192_square_br_as_tl(params: TemplateParams, *, s_q: int, s_kv: int) -> bool:
    """Whether a D192 bottom-right mask is exactly top-left causal."""

    return (
        params.split_kv == 1
        and not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
        and 4096 < s_kv <= 8192
    )


def canonicalize_d192_lowering(
    params: TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
) -> TemplateParams:
    """Apply strictly equivalent D192 lowering canonicalizations."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    window_left = params.window_left
    window_right = params.window_right

    template_window_right = window_right
    if fp8 and pertensor and window_left is None and window_right is None and not params.seq_kv_lens_present and not params.kv_tail_mask:
        # CUTLASS DSL 4.7 does not finish lowering the large-shape FP8
        # MASK_NONE x32 path, so the dense plan is lowered as MASK_CAUSAL with a
        # right band no sequence reaches.  The band is a compile-time
        # `window_right` at the kernel's mask sites, so it must sit INSIDE the
        # bit-word mask op's Int32 domain: `apply_mask_chunk` raises at trace
        # time from MASK_BOUND_LIMIT (1 << 30) on, and a trace-time raise is a
        # typed decline at engine.build_plan -- the former `1 << 30` dropped the
        # FROST fp8 row out of every dense per-tensor d192 graph once the kernels
        # masked in the bit-word form.  MASK_BOUND_LIMIT - 1 still exceeds any dense D192
        # sequence that fits in SM100 memory while leaving signed-int32 headroom
        # for q + R, and keeps the module key independent of S_kv.  Imported here,
        # not at module level: tile_dsl.mask imports cutlass, which stays off the
        # eligibility path (this runs in the lowering, right before the compile).
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mask import MASK_BOUND_LIMIT

        template_window_right = MASK_BOUND_LIMIT - 1

    template_bottom_right = False if d192_square_br_as_tl(params, s_q=s_q, s_kv=s_kv) else params.bottom_right

    return replace(
        params,
        window_right=template_window_right,
        bottom_right=template_bottom_right,
    )


def derive_d192_internal_params(
    params: TemplateParams,
    *,
    pertensor: bool,
    batch_size: int,
    h_q: int,
    s_q: int,
    s_kv: int,
) -> TemplateParams:
    """Derive D192-private codegen fields after public knobs are fixed."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    pack_gqa_ratio = params.qh_per_kh if params.pack_gqa else 1
    groups = batch_size * h_q // pack_gqa_ratio
    lpt_head_group = 8 if fp8 and not params.thd_varlen and groups % 8 == 0 else 1
    q_rows_per_cluster = cga_tile_m(192, params.cta_mma)
    lpt_q_tiles = (s_q * pack_gqa_ratio + q_rows_per_cluster - 1) // q_rows_per_cluster if fp8 and not params.thd_varlen else 0

    lpt_l2_size_mib = 0
    lpt_l2_8k = params.sched_policy == SCHED_LPT_L2 and not params.thd_varlen and params.split_kv == 1 and s_q == 8192 and s_kv == 8192
    if lpt_l2_8k and pertensor and params.dtype_qkv == DTYPE_E4M3 and groups % 24 != 0 and groups % 16 == 0:
        # At 8K, 60 MiB groups 24 one-byte K/V heads; 40 MiB groups 16 and
        # avoids a short final group for these grids.
        lpt_l2_size_mib = 40
    elif lpt_l2_8k and not fp8 and not params.pack_gqa and groups % 16 == 0:
        # At 8K, each half-precision K/V head occupies 5 MiB. Grouping exactly
        # 16 heads avoids a short final LPT-L2 group on the model grids.
        lpt_l2_size_mib = 80

    return replace(
        params,
        lpt_head_group=lpt_head_group,
        lpt_q_tiles=lpt_q_tiles,
        lpt_l2_size_mib=lpt_l2_size_mib,
    )


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """The SM100 d192x128 configuration for one set of public knobs: the 2-CTA pair, or the single-CTA tile with its
    per-D-subtile K ring and second V stage on half input."""
    _validate_params("d192", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    mask_flags = _mask_flags_from(params)
    thd_swa = fp8 and params.thd_varlen and bool(mask_flags & MASK_SWA)
    e4_thd_swa = thd_swa and params.dtype_qkv == DTYPE_E4M3
    e5_dense_causal_regs = (
        params.dtype_qkv == DTYPE_E5M2
        and not params.thd_varlen
        and params.split_kv == 1
        and not params.bottom_right
        and mask_flags == MASK_CAUSAL
        and (params.window_right or 0) == 0
    )
    wide_role_regs = thd_swa or e5_dense_causal_regs
    b_v = 2 if params.pv_bf16 else b
    stages_kv = (3 if params.cta_mma == 2 else 1) if params.pv_bf16 else (2 if fp8 else 1) * params.cta_mma
    # Half input at cga1: one full K stage (48 KiB) plus TWO V stages (64 KiB) under the 96 KiB Q u O slab -- the V
    # reload then overlaps a whole iteration instead of the PV->PV window a single stage leaves it.
    stages_v = 2 if (not fp8 and params.cta_mma == 1) else stages_kv
    k_sub = (192 * b) // 128 if (not fp8 and params.cta_mma == 1) else 1  # one slot per 128 B TMA D-subtile
    half_input_cga1 = not fp8 and params.cta_mma == 1
    cga1_loader = half_input_cga1 and bool(params.cga1_l2_prefetch)
    cfg = CfgD192(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        SPLIT_KV=int(params.split_kv),
        QO_ALIAS=0 if fp8 else 1,
        Q_SWZ_BYTES=q_swz_bytes(192, b),
        K_SWZ_BYTES=q_swz_bytes(192, b),
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32 if fp8 else tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=16 if params.pv_bf16 else (32 if fp8 else tile_k_hw(params.dtype_qkv)),
        STAGES_KV=stages_kv,
        STAGES_K=stages_kv,
        STAGES_V=stages_v,
        K_SUB=k_sub,
        L2_PREFETCH_KV=1 if cga1_loader else 0,
        L2_PREFETCH_NEXT_TILE=1 if cga1_loader else 0,
        LDG_UNROLL=1 if cga1_loader else 6,
        EXP2_TAIL_MIX=0 if cga1_loader else 1,
        SCHEDULER_STAGES=3 if e4_thd_swa and not params.bottom_right else 2,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        L2_SIZE_MIB=params.lpt_l2_size_mib or 60,
        SCHEDULER_POLICY=params.sched_policy,
        SOFTMAX_REGS=192 if wide_role_regs else 184 if fp8 else 216 if mask_flags == MASK_NONE else 192,
        CORRECTION_REGS=88 if wide_role_regs else 104 if fp8 else 40 if mask_flags == MASK_NONE else 88,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD192.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d192(cfg)
    return cfg, _tma_iters(cfg)


class _CfgD576Geometry:
    """Native D576x512 launch geometry: the 2x2 datapath with one CTA pair (make_cfg_d512_2x2(cga_m=2))."""

    TILES_Q = CfgD512X2.TILES_Q
    TILE_M = CfgD512X2.TILE_M
    CTA_MMA = CfgD512X2.CTA_MMA
    CGA_M = 2


_GEOMETRY_BY_D_QK = {32: CfgD32, 64: CfgD64, 128: CfgD128, 192: CfgD192, 256: CfgD256, 512: CfgD512, 576: _CfgD576Geometry}


MAKE_CFG = {32: make_cfg_d32, 64: make_cfg_d64, 128: make_cfg_d128, 192: make_cfg_d192, 256: make_cfg_d256, 512: make_cfg_d512}
