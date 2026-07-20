# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/smem.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Shared-memory layout factory for the unified SM120 sparse-MLA backend.

Given a :class:`UnifiedMLATraits`, compute ``const_expr`` byte offsets for the
FlashInfer DSV4 ``DecodeDsv4Smem`` layout and build a ``cute.struct``
``SharedStorage`` allocatable via ``cutlass.utils.SmemAllocator`` (idiom:
one typed ``cute.struct`` allocation through ``SmemAllocator``).

The byte offsets are pinned to the AUTHORITATIVE FlashInfer DSV4 kernel
(``decode_dsv4_kernel.cuh:47`` ``DecodeDsv4Smem`` + ``common/fp8_quant.cuh`` +
``common/d2_load_b.cuh`` + ``common/kv_cache_io.cuh``). The decode-DSV4 math
addresses every smem region as a FLAT, LINEAR ``base + row*STRIDE + col`` array
(``ldmatrix_load_A_fp8`` / ``ldmatrix_load_B_fp8`` / ``d2_load_b_fp8`` all index
that way) -- it does NOT use the ``_permuted_offset_128b`` XOR bank-swizzle that
the GLM ``kernel_onepass`` QK path uses. The bank-conflict mitigation for DSV4
comes from the STRIDE choice instead (``KV_SMEM_STRIDE=464``: ``464/4 % 32 = 20``
-> clean; ``d2_load_b.cuh:44``). So every sub-region here is a plain linear
array at the documented stride; the IO loader (``io.py``) and the math write
and read it as ``entry * STRIDE + dim``.

Regions (DSV4 ``DecodeDsv4Smem`` order; offsets are const_expr ints):
  * ``q_rope``   HPB x D_ROPE bf16            (linear ``h*D_ROPE + d``)
  * ``q_fp8``    HPB x Q_NOPE_STRIDE staging  (FP8 normally; BF16 for NVFP4)
  * ``q_sc``     HPB x NUM_SCALES fp32        (power-of-2 FP32; converted to the
                                               UE8M0 ``sfa`` selector at use via
                                               ``pow2_ceil_ue8m0`` -- matches
                                               ``fp8_quant.cuh`` storing FP32)
  * ``kv_fp8[2]``  BI x KV_SMEM_STRIDE fp8    (NoPE only + 16B pad; double-buf)
  * ``kv_sc[2]``   BI x SCALE_BYTES_PER_TOKEN (DSV4 UE8M0 footer bytes; double-buf)
                                               -- DSV4-ONLY (has_extra_cache); for
                                               GLM the scales are inline in the
                                               ``kv_fp8`` ``KV_SMEM_STRIDE`` row.
  * ``kv_rope[2]`` BI x D_ROPE bf16           (linear ``entry*D_ROPE + d``)
  * ``mbar``       full[2] + empty[2] u64     (P6 warp-specialized pipeline: a
                                               full/empty mbarrier pair per KV
                                               buffer. full[s]: IO producer
                                               arrives + expect_tx, bulk
                                               completion drives tx; empty[s]:
                                               math consumer arrives when done
                                               reading buffer s. Order:
                                               full[0],full[1],empty[0],empty[1])
  * ``reduce``     2 x N_WARPS x HPB fp32     (warp_max | warp_sum cross-warp tree)
  * ``w_head_sc``  N_V_CHUNKS x HPB fp32      (per-vc per-head XV output scale)
  * ``w_fp8[2]``   HPB x (BI + 16) fp8        (re-quantized P; double-buf across vc)
  * ``token_idx[2]`` BI int32                 (staged raw topk index incl -1
                                               sentinel; gap #9 single source of
                                               truth for the consumer mask)
  * ``sm_p_full``  HPB x BI bf16              (normalized P; 128B-aligned for the
                                               S6b ldmatrix.x4 read)

Total dynamic smem stays < the SM120 carveout (~92KB DSV4, ~96KB GLM) and the
import-time asserts pin ``KV_SMEM_STRIDE`` to the traits.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .traits import ScaleFormat, UnifiedMLATraits


# SM120 dynamic-smem opt-in carveout cap: (100-1)*1024 (see cutlass
# SMEM_CAPACITY_MAP; proven launchable by .sm120port/probes/gate_g2_smem.py).
SM120_SMEM_CARVEOUT_BYTES = 101376

# Double-buffered KV gather (full/empty parity). The P5 synchronous loader only
# uses buf 0 per chunk, but the two-buffer sizing is pinned now so the P6
# mbarrier pipeline (w_fp8[vc&1]-style ping-pong) drops in without a relayout.
_KV_BUF_COUNT = 2

# w_fp8 is double-buffered across the XV vc loop (w_fp8[vc&1]) to drop the
# read-after-write bar_sync between consecutive vc iterations (FlashInfer
# decode_dsv4_kernel.cuh:112 OFF_W_FP8 + parity).
_W_FP8_BUF_COUNT = 2

# token-index validity buffer is staged per KV buffer (one int32 row of BI
# candidates per buf) so the consumer mask matches the buffered KV.
_TOKEN_IDX_BUF_COUNT = _KV_BUF_COUNT

# Number of math warps participating in the cross-warp softmax reduce (8 warps =
# 256 math threads). The IO warp (P6) is excluded from this reduce.
_MATH_WARPS = 8

# w_fp8 ldmatrix pad: the XV W A-operand is loaded with ldmatrix.x4 over BI
# columns; FlashInfer pads the row to (BI + 16) (W_FP8_STRIDE) for the load.
_W_FP8_PAD = 16

# Per-stage mbarrier cost: full[KV_BUF_COUNT] + empty[KV_BUF_COUNT], 8B (u64)
# each. The P6 warp-specialized pipeline needs a full/empty pair PER buffer
# (FlashInfer decode_dsv4_kernel.cuh:226 mbar_full(s)/mbar_empty(s) for s in
# [0, DSV4_KV_BUF_COUNT)) so the IO producer and math consumer ping-pong on
# independent parities. Layout order: full[0], full[1], empty[0], empty[1].
_MBAR_BYTES = 2 * _KV_BUF_COUNT * 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class SmemLayout:
    """Resolved const_expr byte offsets + region sizes for one specialization.

    All fields are plain Python ints (captured into the @cute.jit closure and
    constant-folded). ``total_bytes`` is the dynamic-smem request asserted
    against the SM120 carveout. Strides (``q_nope_stride`` / ``kv_smem_stride`` /
    ``q_sc_stride`` etc.) are echoed so the loader + math index the regions with
    the SAME const_expr the layout was built from.
    """

    # --- Q staging (single buffer; HPB heads). ---
    # q_rope FIRST to match FlashInfer DecodeDsv4Smem OFF_Q_ROPE = 0.
    q_rope_off: int
    q_rope_bytes: int
    q_rope_stride: int  # D_ROPE plus optional bank-conflict pad

    q_fp8_off: int
    q_fp8_bytes: int
    q_nope_stride: int  # Q_NOPE_STRIDE (fp8 bytes per head row; 448 + 16 pad)

    q_sc_off: int
    q_sc_bytes: int
    q_sc_stride: int  # NUM_SCALES (fp32 power-of-2 scales per head)

    # --- Double-buffered KV stage. ---
    # kv_fp8: NoPE fp8 ONLY at KV_SMEM_STRIDE (448 + 16 pad); GLM inline-scale
    # bytes also live in this stride; DSV4 footer scales are in kv_sc.
    kv_fp8_off: int
    kv_fp8_buf_bytes: int  # per buf == BI * KV_SMEM_STRIDE
    kv_smem_stride: int

    # kv_sc: DSV4 UE8M0 footer bytes (SCALE_BYTES_PER_TOKEN per entry).
    # DSV4-only (has_extra_cache); 0 bytes for GLM (scales inline in kv_fp8 row).
    kv_sc_off: int
    kv_sc_buf_bytes: int  # per buf == BI * SCALE_BYTES_PER_TOKEN (0 if inline)
    kv_sc_stride: int  # SCALE_BYTES_PER_TOKEN (0 if inline)

    # kv_rope: V/K rope bf16, linear entry*D_ROPE + d.
    kv_rope_off: int
    kv_rope_buf_bytes: int  # per buf == BI * D_ROPE * 2
    kv_rope_stride: int  # D_ROPE (bf16 elems per entry row)

    kv_bufs: int

    # --- P6 pipeline placeholder (full + empty mbarrier pair). ---
    mbar_off: int
    mbar_bytes: int

    # --- cross-warp softmax reduce scratch (warp_max | warp_sum). ---
    reduce_off: int
    reduce_bytes: int
    reduce_warp_max_off: int  # == reduce_off
    reduce_warp_sum_off: int  # == reduce_off + N_WARPS * HPB * 4

    # --- per-vc per-head output scale (fp32). ---
    w_head_sc_off: int
    w_head_sc_bytes: int
    w_head_sc_stride: int  # HPB (one row of HPB scales per vc)

    # --- re-quantized-P staging for the XV MMA (double-buffered across vc). ---
    w_fp8_off: int
    w_fp8_buf_bytes: int  # per buf == HPB * (BI + 16)
    w_fp8_stride: int  # BI + 16
    w_fp8_bufs: int

    # --- staged raw-token-index validity buffer (incl. -1 sentinel), int32. ---
    token_idx_off: int
    token_idx_buf_bytes: int  # per buf == BI * 4
    token_idx_bufs: int

    # --- static sm_p_full score staging (HPB x BI bf16; 128B aligned). ---
    sm_p_full_off: int
    sm_p_full_bytes: int
    sm_p_full_stride: int  # BI (bf16 elems per head row)

    # --- second-group w_head_sc for the native H16 two-group decode. ---
    # The base w_head_sc row packs scale [0,8) + reciprocal [8,16) for ONE
    # 8-head group, so the H16 kernel's second group gets its own region.
    # Appended at the TAIL so every pre-existing offset stays byte-identical.
    w_head_sc2_off: int
    w_head_sc2_bytes: int

    total_bytes: int


def make_smem_layout(traits: UnifiedMLATraits) -> SmemLayout:
    """Compute the const_expr smem byte layout for ``traits``.

    Region order + sizing mirror FlashInfer ``DecodeDsv4Smem`` exactly. Each
    region is a flat linear array at its documented stride. const_expr keys:
      * ``model_type``  -> d_nope/quant_tile/num_scales/n_v_chunks +
        q_nope_stride/kv_smem_stride (464 DSV4 / 528 GLM)
      * ``scale_format``/``has_extra_cache`` -> the separate ``kv_sc`` footer
        buffer (DSV4 UE8M0 footer) vs inline scales in the kv row (GLM)
      * ``v_has_rope`` does NOT change the layout (kv_rope is allocated for both;
        for GLM the XV-rope stage is dead-code-eliminated but K-rope still reads
        kv_rope, and Q-rope is always present)
    """
    bi = traits.bi
    hpb = traits.hpb
    d_rope = traits.d_rope
    num_scales = traits.num_scales
    n_v_chunks = traits.n_v_chunks
    kv_smem_stride = traits.kv_smem_stride
    q_nope_stride = traits.q_nope_stride
    bufs = _KV_BUF_COUNT

    off = 0

    # --- Q rope (bf16), FlashInfer OFF_Q_ROPE = 0. ---
    q_rope_off = off
    # DSV4's 64-bf16 rows otherwise repeat every 128 bytes and alias banks
    # across heads during the native H8/H16 scalar Q-RoPE loads.
    q_rope_stride = d_rope + (8 if traits.has_extra_cache else 0)
    q_rope_bytes = hpb * q_rope_stride * 2
    off = q_rope_off + q_rope_bytes  # FlashInfer packs Q regions back-to-back

    # --- Q-NoPE staging. FP8 normally; BF16 for native NVFP4 cache math. ---
    q_fp8_off = off
    q_element_bytes = 2 if traits.scale_format == ScaleFormat.NVFP4_E4M3 else 1
    q_fp8_bytes = hpb * q_nope_stride * q_element_bytes
    off = q_fp8_off + q_fp8_bytes

    # --- Q per-head scales: FP32 power-of-2 (converted to UE8M0 sfa at use). ---
    q_sc_off = off
    q_sc_stride = num_scales
    q_sc_bytes = hpb * num_scales * 4
    off = q_sc_off + q_sc_bytes

    # 128B-align the start of the (large) KV region for clean vectorized stores.
    off = _align_up(off, 128)

    # --- Double-buffered KV nope (fp8) at KV_SMEM_STRIDE. ---
    kv_fp8_off = off
    kv_fp8_buf_bytes = bi * kv_smem_stride
    off = kv_fp8_off + kv_fp8_buf_bytes * bufs

    # --- Double-buffered KV footer scales (DSV4 UE8M0; inline -> 0 for GLM). ---
    kv_sc_off = off
    if traits.has_extra_cache:
        # DSV4: 8 footer bytes/token (7 UE8M0 + 1 pad). Separately gathered.
        kv_sc_stride = 8  # SCALE_BYTES_PER_TOKEN
        kv_sc_buf_bytes = bi * kv_sc_stride
        off = kv_sc_off + kv_sc_buf_bytes * bufs
    else:
        # GLM: scales are INLINE in the kv_fp8 KV_SMEM_STRIDE row (no footer buf).
        kv_sc_stride = 0
        kv_sc_buf_bytes = 0

    # --- Double-buffered KV rope (bf16) linear entry*D_ROPE + d. ---
    kv_rope_off = off
    kv_rope_stride = d_rope
    kv_rope_buf_bytes = bi * d_rope * 2
    off = kv_rope_off + kv_rope_buf_bytes * bufs

    # --- P6 mbarrier pair placeholder (16B aligned; reserved, unused in P5). ---
    off = _align_up(off, 16)
    mbar_off = off
    mbar_bytes = _MBAR_BYTES
    off = mbar_off + mbar_bytes

    # --- cross-warp softmax reduce scratch: warp_max | warp_sum (fp32). ---
    reduce_off = off
    reduce_warp_max_off = reduce_off
    reduce_warp_sum_off = reduce_off + _MATH_WARPS * hpb * 4
    reduce_bytes = 2 * _MATH_WARPS * hpb * 4
    off = reduce_off + reduce_bytes

    # --- per-vc per-head output scale (fp32), N_V_CHUNKS rows of HPB. ---
    w_head_sc_off = off
    w_head_sc_stride = hpb
    w_head_sc_bytes = n_v_chunks * hpb * 4
    off = w_head_sc_off + w_head_sc_bytes

    # --- re-quantized-P staging for XV MMA: HPB x (BI + 16), double-buffered. ---
    w_fp8_off = off
    w_fp8_stride = bi + _W_FP8_PAD
    w_fp8_buf_bytes = hpb * w_fp8_stride
    w_fp8_bufs = _W_FP8_BUF_COUNT
    off = w_fp8_off + w_fp8_buf_bytes * w_fp8_bufs

    # --- staged raw-token-index validity buffer (incl -1 sentinel), int32. ---
    # 16-align for clean int32 indexing; double-buffered to track the KV buf.
    off = _align_up(off, 16)
    token_idx_off = off
    token_idx_buf_bytes = bi * 4
    token_idx_bufs = _TOKEN_IDX_BUF_COUNT
    off = token_idx_off + token_idx_buf_bytes * token_idx_bufs

    # --- static sm_p_full (bf16), 128B-aligned for the S6b ldmatrix.x4 read. ---
    off = _align_up(off, 128)
    sm_p_full_off = off
    sm_p_full_stride = bi + (8 if traits.has_extra_cache else 0)
    sm_p_full_bytes = hpb * sm_p_full_stride * 2
    off = sm_p_full_off + sm_p_full_bytes

    # --- native H16 group-1 w_head_sc (tail region; base offsets unchanged).
    #     DSV4-only: GLM has no two-group H16 mode and sits near the carveout.
    #     The extra 8*BI bf16 rows keep group 1's S6b ldmatrix.x4 A-loads (which
    #     always touch 16 sm_p rows from the group base at +8 rows) inside the
    #     allocation; those rows are read-garbage/compute-discarded, exactly
    #     like the H8 kernel's rows 8-15. ---
    off = _align_up(off, 16)
    w_head_sc2_off = off
    w_head_sc2_bytes = (
        n_v_chunks * hpb * 4 + 8 * sm_p_full_stride * 2 if traits.has_extra_cache else 0
    )
    off = w_head_sc2_off + w_head_sc2_bytes

    total_bytes = _align_up(off, 128)

    return SmemLayout(
        q_rope_off=q_rope_off,
        q_rope_bytes=q_rope_bytes,
        q_rope_stride=q_rope_stride,
        q_fp8_off=q_fp8_off,
        q_fp8_bytes=q_fp8_bytes,
        q_nope_stride=q_nope_stride,
        q_sc_off=q_sc_off,
        q_sc_bytes=q_sc_bytes,
        q_sc_stride=q_sc_stride,
        kv_fp8_off=kv_fp8_off,
        kv_fp8_buf_bytes=kv_fp8_buf_bytes,
        kv_smem_stride=kv_smem_stride,
        kv_sc_off=kv_sc_off,
        kv_sc_buf_bytes=kv_sc_buf_bytes,
        kv_sc_stride=kv_sc_stride,
        kv_rope_off=kv_rope_off,
        kv_rope_buf_bytes=kv_rope_buf_bytes,
        kv_rope_stride=kv_rope_stride,
        kv_bufs=bufs,
        mbar_off=mbar_off,
        mbar_bytes=mbar_bytes,
        reduce_off=reduce_off,
        reduce_bytes=reduce_bytes,
        reduce_warp_max_off=reduce_warp_max_off,
        reduce_warp_sum_off=reduce_warp_sum_off,
        w_head_sc_off=w_head_sc_off,
        w_head_sc_bytes=w_head_sc_bytes,
        w_head_sc_stride=w_head_sc_stride,
        w_fp8_off=w_fp8_off,
        w_fp8_buf_bytes=w_fp8_buf_bytes,
        w_fp8_stride=w_fp8_stride,
        w_fp8_bufs=w_fp8_bufs,
        token_idx_off=token_idx_off,
        token_idx_buf_bytes=token_idx_buf_bytes,
        token_idx_bufs=token_idx_bufs,
        sm_p_full_off=sm_p_full_off,
        sm_p_full_bytes=sm_p_full_bytes,
        sm_p_full_stride=sm_p_full_stride,
        w_head_sc2_off=w_head_sc2_off,
        w_head_sc2_bytes=w_head_sc2_bytes,
        total_bytes=total_bytes,
    )


def get_unified_shared_storage_cls(traits: UnifiedMLATraits):
    """Build the ``cute.struct`` ``SharedStorage`` for one specialization.

    Each region is a typed ``cute.struct.MemRange`` (128B/16B-aligned where the
    math/ldmatrix needs it) so a single
    ``SmemAllocator().allocate(SharedStorage)`` reserves the whole layout.
    Sub-regions are NAMED typed fields -- the kernel obtains a flat-linear
    ``cute.Tensor`` view via ``<field>.get_tensor(layout)`` (or a u32 smem
    address via ``shared_ptr_to_u32(<field>.data_ptr())`` for the
    ldmatrix/st.shared.v4 path). See the ``*_view`` / ``*_addr`` helpers below.

    Buffers that are double-buffered in the layout (``kv_fp8`` / ``kv_sc`` /
    ``kv_rope`` / ``w_fp8`` / ``token_idx``) are allocated as a single MemRange
    of ``bufs * buf_bytes``; the kernel selects buffer ``b`` via the const_expr
    ``*_buf_bytes`` offset. DSV4-only fields are omitted from GLM entirely so
    CUTLASS DSL 4.6's typed-allocation size and offsets remain byte-identical to
    :func:`make_smem_layout`.
    """
    layout = make_smem_layout(traits)

    class SharedStorage:
        pass

    kv_sc_field = (
        {
            "kv_sc": cute.struct.MemRange[
                cutlass.Uint8,
                int(layout.kv_sc_buf_bytes * layout.kv_bufs),
            ]
        }
        if traits.has_extra_cache
        else {}
    )
    w_head_sc2_field = (
        {
            "w_head_sc2": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, int(layout.w_head_sc2_bytes // 4)
                ],
                16,
            ]
        }
        if traits.has_extra_cache
        else {}
    )
    SharedStorage.__annotations__ = {
        # Q staging (single buffer). q_rope first (FlashInfer OFF_Q_ROPE = 0).
        "q_rope": cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, int(layout.q_rope_bytes // 2)],
            128,
        ],
        "q_fp8": cute.struct.MemRange[cutlass.Uint8, int(layout.q_fp8_bytes)],
        "q_sc": cute.struct.MemRange[cutlass.Float32, int(layout.q_sc_bytes // 4)],
        # Double-buffered KV. 128B-align the big nope region.
        "kv_fp8": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8, int(layout.kv_fp8_buf_bytes * layout.kv_bufs)
            ],
            128,
        ],
        **kv_sc_field,
        # 16B-align kv_rope for cp.async.bulk and ldmatrix. DSV4's footer is
        # already 128B-sized; GLM has no footer field at all.
        "kv_rope": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.BFloat16, int(layout.kv_rope_buf_bytes * layout.kv_bufs // 2)
            ],
            16,
        ],
        # P6 mbarrier array (u64 full[2]+empty[2]). 16B-aligned.
        "mbar": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint64, int(layout.mbar_bytes // 8)],
            16,
        ],
        # cross-warp reduce scratch (warp_max | warp_sum fp32).
        "reduce": cute.struct.MemRange[cutlass.Float32, int(layout.reduce_bytes // 4)],
        # per-vc per-head output scale (fp32).
        "w_head_sc": cute.struct.MemRange[
            cutlass.Float32, int(layout.w_head_sc_bytes // 4)
        ],
        # re-quantized-P (fp8), double-buffered across vc.
        "w_fp8": cute.struct.MemRange[
            cutlass.Uint8, int(layout.w_fp8_buf_bytes * layout.w_fp8_bufs)
        ],
        # staged raw-token-index validity buffer (incl -1), int32, double-buffered.
        "token_idx": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Int32,
                int(layout.token_idx_buf_bytes * layout.token_idx_bufs // 4),
            ],
            16,
        ],
        # static sm_p_full (bf16), 128B-aligned for ldmatrix.x4.
        "sm_p_full": cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, int(layout.sm_p_full_bytes // 2)],
            128,
        ],
        # Native DSV4 H16 group-1 w_head_sc tail (16B-aligned). GLM has no
        # two-group H16 mode, so it has no placeholder field or tail padding.
        **w_head_sc2_field,
    }
    storage_cls = cute.struct(SharedStorage)

    expected_offsets = {
        "q_rope": layout.q_rope_off,
        "q_fp8": layout.q_fp8_off,
        "q_sc": layout.q_sc_off,
        "kv_fp8": layout.kv_fp8_off,
        "kv_rope": layout.kv_rope_off,
        "mbar": layout.mbar_off,
        "reduce": layout.reduce_off,
        "w_head_sc": layout.w_head_sc_off,
        "w_fp8": layout.w_fp8_off,
        "token_idx": layout.token_idx_off,
        "sm_p_full": layout.sm_p_full_off,
    }
    if traits.has_extra_cache:
        expected_offsets["kv_sc"] = layout.kv_sc_off
        expected_offsets["w_head_sc2"] = layout.w_head_sc2_off
    actual_offsets = dict(storage_cls._offsets)
    if actual_offsets != expected_offsets:
        raise ValueError(
            "typed MLA SharedStorage offsets diverge from SmemLayout: "
            f"typed={actual_offsets}, logical={expected_offsets}"
        )
    typed_bytes = int(storage_cls.size_in_bytes())
    if typed_bytes != layout.total_bytes:
        raise ValueError(
            "typed MLA SharedStorage size diverges from SmemLayout: "
            f"typed={typed_bytes}, logical={layout.total_bytes}"
        )
    return storage_cls


# ---------------------------------------------------------------------------
# Module-level size + stride assertions for BOTH validated models. These run at
# import time (pure Python) and guard against a layout change blowing the SM120
# carveout or desyncing KV_SMEM_STRIDE / Q_NOPE_STRIDE from the traits.
# ---------------------------------------------------------------------------
def _assert_model(model_type: int, compute_mode: int, scale_format: int) -> int:
    # Imported lazily to keep this module's top-level import order simple.
    from .traits import make_unified_traits

    traits = make_unified_traits(model_type, compute_mode, scale_format)
    layout = make_smem_layout(traits)
    # This constructor validates every typed field offset and the exact total
    # that CUTLASS DSL 4.6 infers for the launch's dynamic shared memory.
    get_unified_shared_storage_cls(traits)
    assert layout.kv_smem_stride == traits.kv_smem_stride, (
        f"KV_SMEM_STRIDE mismatch: layout={layout.kv_smem_stride} "
        f"traits={traits.kv_smem_stride}"
    )
    assert layout.q_nope_stride == traits.q_nope_stride, (
        f"Q_NOPE_STRIDE mismatch: layout={layout.q_nope_stride} "
        f"traits={traits.q_nope_stride}"
    )
    # DSV4 has the separate UE8M0 footer buffer; GLM keeps scales inline.
    if traits.has_extra_cache:
        assert layout.kv_sc_buf_bytes == traits.bi * 8, (
            f"DSV4 kv_sc footer buf must be BI*8; got {layout.kv_sc_buf_bytes}"
        )
    else:
        assert layout.kv_sc_buf_bytes == 0, (
            f"GLM scales are inline; kv_sc footer buf must be 0, "
            f"got {layout.kv_sc_buf_bytes}"
        )
    # Sub-region offsets must be monotonic + non-overlapping (sanity on packing).
    assert layout.q_rope_off == 0
    assert layout.q_fp8_off >= layout.q_rope_off + layout.q_rope_bytes
    assert layout.kv_fp8_off >= layout.q_sc_off + layout.q_sc_bytes
    assert layout.total_bytes < SM120_SMEM_CARVEOUT_BYTES, (
        f"smem layout {layout.total_bytes}B exceeds SM120 carveout "
        f"{SM120_SMEM_CARVEOUT_BYTES}B for model_type={model_type}"
    )
    return layout.total_bytes


def _run_module_asserts() -> None:
    from .traits import ComputeMode, ModelType, ScaleFormat

    dsv4 = _assert_model(ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    _assert_model(ModelType.DSV4, ComputeMode.BF16, ScaleFormat.UE8M0_BYTE)
    glm = _assert_model(ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
    _assert_model(ModelType.GLM_NSA, ComputeMode.BF16, ScaleFormat.ARBITRARY_FP32)
    _assert_model(ModelType.GLM_NSA, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3)
    # Sanity on the expected magnitudes (~92KB DSV4, ~96KB GLM after the
    # double-buffered KV + footer + w_fp8 refinements).
    assert 80 * 1024 <= dsv4 <= 96 * 1024, f"DSV4 smem {dsv4}B outside ~92KB band"
    assert 88 * 1024 <= glm <= SM120_SMEM_CARVEOUT_BYTES, (
        f"GLM smem {glm}B outside ~96KB band"
    )


_run_module_asserts()
