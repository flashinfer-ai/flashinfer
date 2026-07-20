# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/decode_math.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""DSV4 sparse-MLA decode MATH stages (P5) -- the QK path (S0-S3).

100%-CuTeDSL single-CTA DSV4 decode for ONE query token over HPB=16 heads with
8 math warps (256 threads). This module currently implements the QK score path:

  * S0  Q-quant       : BF16 Q -> per-(QUANT_TILE=64) absmax -> pow2 UE8M0 scale
                        -> E4M3 quantize into ``q_fp8`` + FP32 pow2 scale into
                        ``q_sc``; Q-rope copied bf16 into ``q_rope``.
  * S1  QK-NoPE       : block-scaled FP8 MMA (mxfp8 m16n8k32, ue8m0 selectors)
                        with REAL sfa (Q pow2 exponent byte from ``q_sc``) and
                        sfb (K UE8M0 footer byte from ``kv_sc``). NUM_SCALES=7 x
                        (QUANT_TILE/32)=2 = 14 block-scaled MMAs.
  * S2  QK-RoPE       : bf16 m16n8k16 accumulating into the SAME qk regs over
                        D_ROPE=64 (4 K-steps) = 4 bf16 MMAs.
  * S3  mask + scale  : invalid candidates (idx<0 OR past split_cand_end) -> -inf
                        (-Float32.inf, the rs-1 base-2 merge sentinel); then
                        qk *= sm_scale * LOG2E.

The DATA PATH follows FlashInfer ``decode_dsv4_kernel.cuh`` (S0 ``fp8_quant.cuh``
``quantize_q_to_smem``; S1/S2 the chunk-loop QK block at decode_dsv4 :372-454).
The ldmatrix loaders mirror ``arch/ldmatrix_sm120.cuh`` (FLAT byte addressing on
the linear smem regions finalized in ``smem.py``) rather than the GLM onepass
``_permuted_offset_128b`` XOR swizzle (see the smem.py module docstring).

KEY DIVERGENCE from ``kernel_onepass.py`` (blueprint ORCHESTRATOR NOTE):
``kernel_onepass`` pre-dequantizes K to bf16 and passes ``unit_scale``
(0x7F7F7F7F) into the block-scaled MMA. DSV4 instead keeps the FP8 nope bytes RAW
in smem and feeds the REAL UE8M0 K footer byte + Q pow2 exponent byte as the
``sfa``/``sfb`` selectors -- the hardware ue8m0 path, zero-cost scaling.

Per-warp QK score fragment layout (single source of truth for the PV agent):
  * 8 math warps; warp ``w`` owns candidates ``[w*8, w*8+8)`` (DSV4_QK_N_TILES=1
    so one m16n8 N-tile per warp -> 8 candidate columns per warp; 8 warps cover
    BI=64).
  * The MMA M dimension is the HPB=16 heads; N is the candidate column.
  * Per lane, the m16n8 fragment is 4 FP32 accumulators ``qk[0..3]``:
      gid = lane >> 2 ;  tid = lane & 3
      qk[0] = score[head=gid     , cand=w*8 + tid*2    ]
      qk[1] = score[head=gid     , cand=w*8 + tid*2 + 1]
      qk[2] = score[head=gid + 8 , cand=w*8 + tid*2    ]
      qk[3] = score[head=gid + 8 , cand=w*8 + tid*2 + 1]
    (gid in [0,8) covers heads 0..15 across the two row groups; tid in [0,4)
    covers the 8 candidate columns in pairs.)
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flashinfer.experimental.sm12x.attention._shared.cute.ops import LOG2_E
from flashinfer.experimental.sm12x._lib.intrinsics import (
    atomic_max_shared_f32,
    byte_perm,
    cvt_f32_to_e4m3,
    cvt_e4m3_to_f32_via_f16,
    fabs_f32,
    f16x2_to_f32x2,
    fmax_f32,
    fmin_f32,
    fp8_e4m3_to_f32,
    fp4_decode_2,
    get_ptr_as_int64,
    ld_global_nc_v2_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    ld_shared_f32,
    ld_shared_u8_offset,
    ld_shared_u32,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    mma_m16n8k32_f32_e4m3,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    pow2_ceil_ue8m0,
    rcp_approx_ftz,
    shared_ptr_to_u32,
    st_shared_bf16_from_f32,
    st_shared_f32,
    st_shared_u32,
    st_shared_u8,
    st_global_v4_u32,
)


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """``exp2f``-equivalent base-2 exponential (``ex2.approx.ftz.f32``).

    Matches kernel.py ``_exp2_approx_ftz_f32`` and the decode-dsv4 base-2 online
    softmax (``exp2f``). FTZ is harmless at the score magnitudes here."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _log2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """``log2f``-equivalent (``lg2.approx.ftz.f32``) for the base-2 LSE epilogue."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_to_bits(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Reinterpret a Float32 as its raw u32 IEEE-754 bits (mov.b32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _u32_to_f32(a: Uint32, *, loc=None, ip=None) -> Float32:
    """Reinterpret a Uint32 as its raw Float32 IEEE-754 value (mov.b32).

    GLM ARBITRARY_FP32 path: the 4 inline K scale bytes are an arbitrary fp32
    (NOT a UE8M0 exponent byte), so the smem word is reinterpreted whole."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_f32x2_to_e4m3x2(
    v0: Float32,
    v1: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert two f32 values to packed E4M3 with one native conversion."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 fp8_pair;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, $2, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e4m3x2_to_f16x2(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert two packed E4M3 values to packed F16 with one instruction."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 $0, fp8_pair;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_u16(
    smem_addr: Int32,
    value: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store the low 16 bits of ``value`` to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u16 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ── DSV4 QK-path compile-time geometry (one CTA, one chunk) ──────────────────
_FP8_MAX = 448.0
_FP8_MIN = -448.0
# NVFP4 MLA latent record geometry inside the staged 288-byte smem row:
# 256B packed E2M1 data, then 32 E4M3 group-16 scale bytes at offset 256.
_NVFP4_SCALE_OFFSET = 256
_NVFP4_SCALE_GROUP = 16
_NEG_INF = float("-inf")
# Intra-kernel softmax mask sentinel (FlashInfer decode_dsv4 :443). Large FINITE
# negative so a fully-invalid warp's (qk - local_max) is 0, not inf - inf = nan.
# The merge -inf sentinel is emitted only at S7 (empty-block LSE).
_QK_MASK = -1e30

# 8 math warps, BI=64 candidates -> 8 candidates per warp = one m16n8 N-tile.
_N_WARPS = 8
_ENTRIES_PER_WARP = 8  # BI // _N_WARPS
_QK_N_TILES = 1  # _ENTRIES_PER_WARP // 8

# S7 epilogue modes (cutlass.Constexpr int keys). The decode/split path writes
# PER-SPLIT NORMALIZED partials + base-2 LSE into mid_out/mid_lse for the reused
# split.py merge (PARTIAL_WRITEBACK, the DEFAULT so decode stays byte-identical).
# The prefill single-pass path writes the FINAL normalized BF16 O directly into
# output[token,h,d] + a base-2 LSE with an optional attn_sink fold (FINAL_BF16).
EPILOGUE_PARTIAL_WRITEBACK = 0
EPILOGUE_FINAL_BF16 = 1


@cute.jit
def _smem_byte(base_addr: Int32, byte_off) -> Int32:
    """Flat smem byte address (base u32 + byte offset); no XOR swizzle."""
    return base_addr + Int32(byte_off)


# =============================================================================
# S0 -- Q quantization to smem (BF16 -> E4M3 + pow2 UE8M0 scale)
# Port of fp8_quant.cuh::quantize_q_to_smem. Outputs q_fp8 / q_sc / q_rope.
# =============================================================================
@cute.jit
def s0_quantize_q_to_smem(
    q_token: cute.Tensor,  # (NUM_HEADS, D_QK) bf16 view for this token
    q_fp8_base_addr: Int32,  # u32 smem addr of q_fp8 (HPB x Q_NOPE_STRIDE)
    q_sc_view: cute.Tensor,  # smem fp32 view (HPB*NUM_SCALES,) -- pow2 scales
    q_rope_base_addr: Int32,  # u32 smem addr of q_rope (HPB x D_ROPE bf16)
    amax_view: cute.Tensor,  # smem fp32 scratch (>= HPB*NUM_SCALES) reused as amax
    head_base: Int32,  # first head index of this CTA (h_start)
    valid_hpb: Int32,  # number of valid heads (<= HPB)
    tid: Int32,  # flat thread id in [0, MATH_THREADS)
    *,
    d_nope: cutlass.Constexpr,  # 448
    d_rope: cutlass.Constexpr,  # 64
    d_qk: cutlass.Constexpr,  # 512
    quant_tile: cutlass.Constexpr,  # 64
    num_scales: cutlass.Constexpr,  # 7
    hpb: cutlass.Constexpr,  # 16
    q_nope_stride: cutlass.Constexpr,  # 464
    q_rope_stride: cutlass.Constexpr,  # D_ROPE plus optional bank-layout pad
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,  # named-barrier slot for the math-only sync
    barrier_threads: cutlass.Constexpr = 0,  # barrier width override (0 -> num_threads)
    fused_subgroup_quant: cutlass.Constexpr = False,
    subgroup_amax: cutlass.Constexpr = False,
    vectorized_rope_copy: cutlass.Constexpr = False,
    packed_q_scale_words: cutlass.Constexpr = False,
):
    """S0: cooperative BF16->E4M3 Q quant with per-tile pow2 UE8M0 scale.

    Mirrors fp8_quant.cuh exactly:
      1. copy Q-rope bf16 -> smem (zero-fill invalid heads)
      2. init amax[HPB*NUM_SCALES] = 0
      3. per-(head,tile) absmax via atomicMax-on-int (atomic_max_shared_f32)
      4. raw = max(amax,1e-4)/FP8_MAX ; pow2-ceil -> q_sc (FP32 pow2 scale)
      5. quantize q_nope * (1/q_sc) -> E4M3 into q_fp8 (zero-fill invalid heads)
    The caller fences (named barrier) AFTER this returns so S1 reads a coherent
    Q stage. The internal barriers separate the 5 phases.

    ``barrier_threads`` widens the named barrier beyond ``num_threads`` for the
    native-H16 two-group decode (each 128-thread group strides its own staging
    with ``num_threads`` while both groups share one 256-wide barrier domain).
    0 keeps the historical num_threads-wide barrier (byte-identical).
    """
    bar_kw = dict(
        barrier_id=barrier_id,
        number_of_threads=(barrier_threads if barrier_threads else num_threads),
    )

    # --- Step 1: copy Q-rope bf16 -> smem; zero-fill invalid heads. ---
    if cutlass.const_expr(vectorized_rope_copy):
        chunks_per_head = d_rope // 4
        i = tid
        while i < Int32(hpb * chunks_per_head):
            h = i // Int32(chunks_per_head)
            chunk = i - h * Int32(chunks_per_head)
            d = chunk * Int32(4)
            q0 = Uint32(0)
            q1 = Uint32(0)
            if h < valid_hpb:
                q_off = cute.crd2idx((head_base + h, Int32(d_nope) + d), q_token.layout)
                q0, q1 = ld_global_nc_v2_u32(get_ptr_as_int64(q_token, q_off))
            s_addr = q_rope_base_addr + (h * Int32(q_rope_stride) + d) * Int32(2)
            st_shared_u32(s_addr, q0)
            st_shared_u32(s_addr + Int32(4), q1)
            i += Int32(num_threads)
    else:
        i = tid
        while i < Int32(hpb * d_rope):
            h = i // Int32(d_rope)
            d = i - h * Int32(d_rope)
            s_byte = (h * Int32(q_rope_stride) + d) * Int32(2)
            val = Float32(0.0)
            if h < valid_hpb:
                val = Float32(q_token[head_base + h, Int32(d_nope) + d])
            st_shared_bf16_from_f32(_smem_byte(q_rope_base_addr, s_byte), val)
            i += Int32(num_threads)

    # --- Steps 2-3: per-(head,tile) absmax over valid heads. ---
    # Keep the dynamic-loop index names distinct across const_expr branches:
    # CuTe joins same-named Python locals before pruning and rejects None->Int32.
    if cutlass.const_expr(fused_subgroup_quant):
        # Native DSV4 gives each 8-lane subgroup one complete 64-value tile.
        # Keep its eight values per lane live through the reduction so scale
        # construction and quantization need no shared amax round-trip or
        # second global Q load.
        lane8 = tid & Int32(7)
        fused_slot = tid >> Int32(3)
        while fused_slot < Int32(hpb * num_scales):
            h = fused_slot // Int32(num_scales)
            blk = fused_slot - h * Int32(num_scales)
            values = [Float32(0.0) for _ in range(quant_tile // 8)]
            amax = Float32(0.0)
            for k in cutlass.range_constexpr(quant_tile // 8):
                d = blk * Int32(quant_tile) + lane8 + Int32(k * 8)
                if h < valid_hpb:
                    values[k] = Float32(q_token[head_base + h, d])
                amax = fmax_f32(amax, fabs_f32(values[k]))
            for offset in (1, 2, 4):
                amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=offset))

            raw = fmax_f32(amax, Float32(1e-4)) * Float32(1.0 / _FP8_MAX)
            rounded, _ue8m0 = pow2_ceil_ue8m0(raw)
            if lane8 == Int32(0):
                q_sc_view[fused_slot] = rounded

            inv_scale = Float32(1.0) / rounded
            for k in cutlass.range_constexpr(quant_tile // 8):
                d = blk * Int32(quant_tile) + lane8 + Int32(k * 8)
                v = values[k] * inv_scale
                v = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), v))
                fp8_bits = cvt_f32_to_e4m3(v)
                out_byte = h * Int32(q_nope_stride) + d
                st_shared_u8(
                    _smem_byte(q_fp8_base_addr, out_byte),
                    (fp8_bits & Uint32(0xFF)).to(cutlass.Uint8),
                )
            fused_slot += Int32(num_threads // 8)

        cute.arch.barrier(**bar_kw)
        return

    if cutlass.const_expr(subgroup_amax):
        # Native DSV4 uses 64-value tiles. Four 8-lane subgroups per warp each
        # own one tile, so every max has a single writer and needs no atomic.
        lane8 = tid & Int32(7)
        subgroup_slot = tid >> Int32(3)
        while subgroup_slot < valid_hpb * Int32(num_scales):
            h = subgroup_slot // Int32(num_scales)
            blk = subgroup_slot - h * Int32(num_scales)
            amax = Float32(0.0)
            for k in cutlass.range_constexpr(quant_tile // 8):
                d = blk * Int32(quant_tile) + lane8 + Int32(k * 8)
                amax = fmax_f32(amax, fabs_f32(Float32(q_token[head_base + h, d])))
            for offset in (1, 2, 4):
                amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=offset))
            if lane8 == Int32(0):
                amax_view[subgroup_slot] = amax
            subgroup_slot += Int32(num_threads // 8)
    else:
        i = tid
        while i < Int32(hpb * num_scales):
            amax_view[i] = Float32(0.0)
            i += Int32(num_threads)
        cute.arch.barrier(**bar_kw)

        idx = tid
        while idx < valid_hpb * Int32(d_nope):
            h = idx // Int32(d_nope)
            d = idx - h * Int32(d_nope)
            blk = d // Int32(quant_tile)
            v = Float32(q_token[head_base + h, d])
            amax_slot = h * Int32(num_scales) + blk
            amax_addr = shared_ptr_to_u32(amax_view.iterator) + amax_slot * Int32(4)
            atomic_max_shared_f32(amax_addr, fabs_f32(v))
            idx += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    # --- Step 4: scale = pow2_ceil(max(amax,1e-4)/FP8_MAX) -> q_sc (FP32). ---
    i = tid
    while i < Int32(hpb * num_scales):
        raw = fmax_f32(amax_view[i], Float32(1e-4)) * Float32(1.0 / _FP8_MAX)
        rounded, _ue8m0 = pow2_ceil_ue8m0(raw)
        q_sc_view[i] = rounded
        if cutlass.const_expr(packed_q_scale_words):
            h = i // Int32(num_scales)
            blk = i - h * Int32(num_scales)
            st_shared_u8(
                q_fp8_base_addr + h * Int32(q_nope_stride) + Int32(d_nope) + blk,
                (_ue8m0 & Uint32(0xFF)).to(cutlass.Uint8),
            )
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    # --- Step 5: quantize q_nope to E4M3 into q_fp8; zero-fill invalid heads. ---
    idx = tid
    while idx < Int32(hpb * d_nope):
        h = idx // Int32(d_nope)
        d = idx - h * Int32(d_nope)
        blk = d // Int32(quant_tile)
        out_byte = h * Int32(q_nope_stride) + d
        fp8_bits = Uint32(0)
        if h < valid_hpb:
            si = Float32(1.0) / q_sc_view[h * Int32(num_scales) + blk]
            v = Float32(q_token[head_base + h, d]) * si
            v = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), v))
            fp8_bits = cvt_f32_to_e4m3(v)
        st_shared_u8(
            _smem_byte(q_fp8_base_addr, out_byte),
            (fp8_bits & Uint32(0xFF)).to(cutlass.Uint8),
        )
        idx += Int32(num_threads)
    cute.arch.barrier(**bar_kw)


@cute.jit
def s0_load_q_bf16_to_smem(
    q_token: cute.Tensor,  # (NUM_HEADS, D_QK) bf16 view for this token
    q_nope_bf16_base_addr: Int32,  # u32 smem addr of BF16 Q-NoPE
    q_rope_base_addr: Int32,  # u32 smem addr of q_rope (HPB x q_rope_stride bf16)
    head_base: Int32,  # first head index of this CTA (h_start)
    valid_hpb: Int32,  # number of valid heads (<= HPB)
    tid: Int32,  # flat thread id in [0, MATH_THREADS)
    *,
    d_nope: cutlass.Constexpr,  # 512
    d_rope: cutlass.Constexpr,  # 64
    hpb: cutlass.Constexpr,  # 16
    q_nope_bf16_stride: cutlass.Constexpr,  # D_NOPE + 8
    q_rope_stride: cutlass.Constexpr,  # D_ROPE plus optional bank-layout pad
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,
    barrier_threads: cutlass.Constexpr = 0,  # barrier width override (0 -> num_threads)
):
    """S0 (NVFP4): stage Q-NoPE and Q-RoPE as BF16.

    This is the decode-local counterpart to the MG prefill BF16-QK path: no Q
    FP8 quantization and no Q scale side buffer. Invalid tail heads are zeroed so
    the regular VALID_HPB-gated epilogue can reuse the same tile geometry.
    """
    bar_kw = dict(
        barrier_id=barrier_id,
        number_of_threads=(barrier_threads if barrier_threads else num_threads),
    )

    i = tid
    while i < Int32(hpb * d_nope):
        h = i // Int32(d_nope)
        d = i - h * Int32(d_nope)
        val = Float32(0.0)
        if h < valid_hpb:
            val = Float32(q_token[head_base + h, d])
        st_shared_bf16_from_f32(
            q_nope_bf16_base_addr + (h * Int32(q_nope_bf16_stride) + d) * Int32(2),
            val,
        )
        i += Int32(num_threads)

    i = tid
    while i < Int32(hpb * d_rope):
        h = i // Int32(d_rope)
        d = i - h * Int32(d_rope)
        val = Float32(0.0)
        if h < valid_hpb:
            val = Float32(q_token[head_base + h, Int32(d_nope) + d])
        st_shared_bf16_from_f32(
            _smem_byte(q_rope_base_addr, (h * Int32(q_rope_stride) + d) * Int32(2)),
            val,
        )
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)


# =============================================================================
# S0b -- REMOVED (P10f). The prior GLM K dequant+requant stage forced K/V back
# through e4m3 to use a UNIT block-scale selector, which DISCARDED the per-group
# e4m3 mantissa headroom (K-dequant cos 0.99968 -> 0.99627) and cost ~0.003 cos
# end-to-end. GLM now keeps RAW e4m3 K/V and applies its arbitrary fp32 group
# scale AFTER the MMA: S1 (QK) post-MMA FP32 scale (legacy
# _accumulate_scaled_score_frag), S6 (V) inline per-group fp32 V-scale. DSV4
# (UE8M0_BYTE) never ran S0b, so its trace/PTX are unchanged.
# =============================================================================


# =============================================================================
# S1 -- QK-NoPE block-scaled FP8 MMA (ue8m0 selectors, REAL sfa/sfb)
# =============================================================================
@cute.jit
def s1_qk_nope_block_scaled(
    qk,  # length-4 python list of Float32 (in/out accum)
    q_fp8_base_addr: Int32,  # u32 smem addr of q_fp8
    kv_fp8_base_addr: Int32,  # u32 smem addr of kv_fp8[buf]
    q_sc_view: cute.Tensor,  # smem fp32 view (HPB*NUM_SCALES,)
    kv_sc_base_addr: Int32,  # u32 smem addr of kv_sc[buf] (BI x 8 footer bytes)
    warp_first_cand: Int32,  # first candidate of this warp (warp_id * 8)
    lane: Int32,
    latent_scale: Float32,
    *,
    num_scales: cutlass.Constexpr,  # 7
    quant_tile: cutlass.Constexpr,  # 64
    q_nope_stride: cutlass.Constexpr,  # 464
    kv_smem_stride: cutlass.Constexpr,  # 464
    scale_bytes_per_token: cutlass.Constexpr,  # 8
    scale_format: cutlass.Constexpr = 0,  # ScaleFormat.UE8M0_BYTE (0) / ARBITRARY_FP32 (1)
    valid_hpb: cutlass.Constexpr = 16,
):
    """S1: accumulate Q_nope . K_nope into qk[0..3] via NUM_SCALES*(QUANT_TILE/32)
    block-scaled MMAs (14 DSV4 / 16 GLM).

    range_constexpr-unrolled NUM_SCALES x (QUANT_TILE/32) so the (default-0)
    bid/tid selectors fold to i16 immediates. sfa = the UE8M0 exponent byte of
    the Q pow2 scale (extracted from the FP32 q_sc) -- IDENTICAL for both models
    (Q is always quantized to a pow2 UE8M0 scale in S0).

    sfb DIVERGES on ``scale_format`` (const_expr):
      * UE8M0_BYTE (DSV4): sfb = the K footer UE8M0 byte (kv_sc[cand*8 + blk]).
        The raw e4m3 K is kept and the real pow2 K selector applies (zero cost).
        All NUM_SCALES groups accumulate DIRECTLY into qk (the K pow2 magnitude
        is folded by the hardware ue8m0 selector).
      * ARBITRARY_FP32 (GLM, P10f): sfb = UNIT (0x7F) on the RAW e4m3 K (NO S0b
        requant -- that threw away the per-group e4m3 mantissa headroom). The
        block-scaled MMA of group ``blk`` runs into a SEPARATE per-group temp
        accumulator (UNIT K, real pow2 sfa on Q), then the temp is multiplied by
        the per-CANDIDATE arbitrary fp32 group scale ``k_scale[cand, blk]`` and
        accumulated into qk in FP32 -- EXACTLY legacy kernel.py
        _accumulate_scaled_score_frag (:1308-1311) post-MMA FP32 scaling. The
        arbitrary fp32 K scale (~5e-5, NON-power-of-2) cannot be represented by a
        ue8m0 selector, so it must apply AFTER the MMA. The inline fp32 group
        scale lives at ``cand*KV_SMEM_STRIDE + D_NOPE + blk*4`` (4B), where
        D_NOPE == NUM_SCALES*QUANT_TILE.

      * NVFP4_E4M3: Q-NoPE is already staged as BF16; packed E2M1 K data and
        E4M3 group-16 scales are dequantized in registers, multiplied by the
        per-layer outer scale, and fed to the BF16 m16n8k16 MMA. No FP8
        block-scale selectors are used in this branch.

    A (Q) loaded via ldmatrix.x4 (FP8 A 16x32 for scale_format 0/1, BF16 A
    16x16 for scale_format 2); B follows the matching FP8 or BF16 path.
    """
    if cutlass.const_expr(scale_format == 2):
        return s1_qk_nope_nvfp4_bf16(
            qk,
            q_fp8_base_addr,
            kv_fp8_base_addr,
            warp_first_cand,
            lane,
            latent_scale,
            num_scales=num_scales,
            quant_tile=quant_tile,
            q_nope_bf16_stride=q_nope_stride,
            kv_smem_stride=kv_smem_stride,
        )

    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    # GLM only: this lane's two candidate columns (c0 -> qk[0]/qk[2],
    # c1 -> qk[1]/qk[3]); inline fp32 group-scale base byte D_NOPE.
    glm_c0 = warp_first_cand + tid * Int32(2)
    glm_c1 = glm_c0 + Int32(1)
    glm_scale_base = Int32(num_scales) * Int32(quant_tile)  # == D_NOPE

    # ldmatrix A (Q) row/col -- arch/ldmatrix_sm120.cuh ldmatrix_load_A_fp8.
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    # ldmatrix B (K) row/col -- ldmatrix_load_B_fp8.
    b_row = lane & Int32(7)
    b_col = ((lane >> Int32(3)) & Int32(1)) * Int32(16)

    # sfa head index: (gid + (lane&1)*8) -- matches decode_dsv4 :378.
    sfa_head = gid + (lane & Int32(1)) * Int32(8)
    # sfb candidate row = warp_first_cand + gid (the N-tile base + gid). nt=0.
    sfb_cand = warp_first_cand + gid
    if cutlass.const_expr(scale_format == 0):
        sfb_row_addr = kv_sc_base_addr + sfb_cand * Int32(scale_bytes_per_token)
    hi = cutlass.const_expr(valid_hpb > 8)

    for blk in cutlass.range_constexpr(num_scales):
        # sfa = exponent byte of the FP32 pow2 Q scale for this (head, blk).
        q_scale = q_sc_view[sfa_head * Int32(num_scales) + Int32(blk)]
        sfa = _fp32_to_ue8m0_byte(q_scale)
        if cutlass.const_expr(scale_format == 0):
            # DSV4 UE8M0_BYTE: sfb = K footer UE8M0 byte for this candidate + blk.
            sfb = ld_shared_u8_offset(sfb_row_addr, blk)
        else:
            # GLM ARBITRARY_FP32 (P10f): raw e4m3 K, UNIT block-scale; the per-group
            # arbitrary fp32 scale is applied AFTER the MMA (below).
            sfb = Uint32(0x7F)
        # GLM accumulates each group's partial into a SEPARATE temp so the
        # arbitrary fp32 group scale can be applied post-MMA; DSV4 accumulates
        # directly into qk (byte-identical to the pre-P10f trace).
        if cutlass.const_expr(scale_format == 0):
            acc0, acc1, acc2, acc3 = qk[0], qk[1], qk[2], qk[3]
        else:
            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
        for ks in cutlass.range_constexpr(quant_tile // 32):
            ko = Int32(blk) * Int32(quant_tile) + Int32(ks) * Int32(32)
            # A operand: Q (16 heads x 32 fp8 K-dims) at q_fp8 + ko.
            a_addr = _smem_byte(
                q_fp8_base_addr, a_row * Int32(q_nope_stride) + ko + a_col
            )
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
            # B operand: K (8 cands x 32 fp8 K-dims) at kv_fp8 + cand_base*stride + ko.
            b_addr = _smem_byte(
                kv_fp8_base_addr,
                warp_first_cand * Int32(kv_smem_stride)
                + b_row * Int32(kv_smem_stride)
                + ko
                + b_col,
            )
            b0, b1 = ldmatrix_m8n8x2_b16(b_addr)
            acc0, acc1, acc2, acc3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc0,
                acc1,
                acc2,
                acc3,
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                sfa,
                sfb,
            )
        if cutlass.const_expr(scale_format == 0):
            qk[0] = acc0
            qk[1] = acc1
            if cutlass.const_expr(hi):
                qk[2] = acc2
                qk[3] = acc3
        else:
            # GLM: post-MMA FP32 scale by the per-candidate arbitrary fp32 group
            # scale (legacy _accumulate_scaled_score_frag). c0 -> qk[0]/qk[2];
            # c1 -> qk[1]/qk[3]. Scale read from the inline fp32 footer (the same
            # bytes io.py gathers in the kv_fp8 nope bulk).
            ks0 = _u32_to_f32(
                ld_shared_u32(
                    kv_fp8_base_addr
                    + glm_c0 * Int32(kv_smem_stride)
                    + glm_scale_base
                    + Int32(blk) * Int32(4)
                )
            )
            ks1 = _u32_to_f32(
                ld_shared_u32(
                    kv_fp8_base_addr
                    + glm_c1 * Int32(kv_smem_stride)
                    + glm_scale_base
                    + Int32(blk) * Int32(4)
                )
            )
            qk[0] = qk[0] + acc0 * ks0
            qk[1] = qk[1] + acc1 * ks1
            if cutlass.const_expr(hi):
                qk[2] = qk[2] + acc2 * ks0
                qk[3] = qk[3] + acc3 * ks1
    return qk


@cute.jit
def s1_qk_nope_nvfp4_bf16(
    qk,
    q_nope_bf16_base_addr: Int32,
    kv_fp4_base_addr: Int32,
    warp_first_cand: Int32,
    lane: Int32,
    latent_scale: Float32,
    *,
    num_scales: cutlass.Constexpr,  # 8 logical 64-dim FP4 K steps
    quant_tile: cutlass.Constexpr,  # 64
    q_nope_bf16_stride: cutlass.Constexpr,  # 520 elems
    kv_smem_stride: cutlass.Constexpr,  # 288 bytes
):
    """S1 (NVFP4): BF16 QK-NoPE over in-register dequantized E2M1 K.

    Storage has 32 E4M3 scales per token (group-16). The outer loop keeps the
    logical FP4 K-step count at 512/64=8, and the inner loop feeds four BF16
    m16n8k16 MMAs per 64-dim step.
    """
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    kv_gid_row = warp_first_cand + gid

    for blk in cutlass.range_constexpr(num_scales):
        for ks in cutlass.range_constexpr(quant_tile // 16):
            ko = Int32(blk) * Int32(quant_tile) + Int32(ks) * Int32(16)
            b0 = _nvfp4_pair_bfloat2(
                kv_fp4_base_addr,
                kv_gid_row,
                ko + tid * Int32(2),
                latent_scale,
                kv_smem_stride=kv_smem_stride,
            )
            b1 = _nvfp4_pair_bfloat2(
                kv_fp4_base_addr,
                kv_gid_row,
                ko + tid * Int32(2) + Int32(8),
                latent_scale,
                kv_smem_stride=kv_smem_stride,
            )
            a_byte = a_row * Int32(q_nope_bf16_stride * 2) + (ko + a_col) * Int32(2)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_byte(q_nope_bf16_base_addr, a_byte)
            )
            qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(
                qk[0],
                qk[1],
                qk[2],
                qk[3],
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
            )
    return qk


@cute.jit
def s1_qk_nope_block_scaled_glm_h8_swap_ab(
    qk,
    q_fp8_base_addr: Int32,
    kv_fp8_base_addr: Int32,
    q_sc_view: cute.Tensor,
    warp_first_cand: Int32,
    lane: Int32,
    *,
    num_scales: cutlass.Constexpr,
    quant_tile: cutlass.Constexpr,
    q_nope_stride: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
):
    """GLM TP8 QK-NoPE with the logical operands swapped.

    Hardware A is a 16-candidate K tile and hardware B is Q transposed into the
    eight-column role.  The m16n8 output is therefore fully occupied instead of
    padding the eight query heads to sixteen rows.  Four warps cover BI=64.

    Fragment ownership is::

        qk[0] = score[cand=base+gid,   head=2*tid]
        qk[1] = score[cand=base+gid,   head=2*tid+1]
        qk[2] = score[cand=base+gid+8, head=2*tid]
        qk[3] = score[cand=base+gid+8, head=2*tid+1]

    GLM's arbitrary K scale remains a candidate-wise FP32 post-MMA multiply.
    Q's pow2 scale moves from SFA to SFB, matching its new hardware-B role.
    """
    gid = lane >> Int32(2)

    # A: sixteen candidate rows from K.
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    # B: eight query-head rows.  scale_vec::1X SFB consumes q_sc[gid].
    b_row = lane & Int32(7)
    b_col = ((lane >> Int32(3)) & Int32(1)) * Int32(16)
    sfb_head = gid

    cand0 = warp_first_cand + gid
    cand1 = cand0 + Int32(8)
    glm_scale_base = Int32(num_scales) * Int32(quant_tile)

    for blk in cutlass.range_constexpr(num_scales):
        sfa = Uint32(0x7F)  # raw GLM K; arbitrary scale is applied below
        sfb = _fp32_to_ue8m0_byte(q_sc_view[sfb_head * Int32(num_scales) + Int32(blk)])
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        for ks in cutlass.range_constexpr(quant_tile // 32):
            ko = Int32(blk) * Int32(quant_tile) + Int32(ks) * Int32(32)
            a_addr = _smem_byte(
                kv_fp8_base_addr,
                warp_first_cand * Int32(kv_smem_stride)
                + a_row * Int32(kv_smem_stride)
                + ko
                + a_col,
            )
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
            b_addr = _smem_byte(
                q_fp8_base_addr,
                b_row * Int32(q_nope_stride) + ko + b_col,
            )
            b0, b1 = ldmatrix_m8n8x2_b16(b_addr)
            acc0, acc1, acc2, acc3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc0,
                acc1,
                acc2,
                acc3,
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                sfa,
                sfb,
            )

        ks0 = _u32_to_f32(
            ld_shared_u32(
                kv_fp8_base_addr
                + cand0 * Int32(kv_smem_stride)
                + glm_scale_base
                + Int32(blk) * Int32(4)
            )
        )
        ks1 = _u32_to_f32(
            ld_shared_u32(
                kv_fp8_base_addr
                + cand1 * Int32(kv_smem_stride)
                + glm_scale_base
                + Int32(blk) * Int32(4)
            )
        )
        qk[0] = qk[0] + acc0 * ks0
        qk[1] = qk[1] + acc1 * ks0
        qk[2] = qk[2] + acc2 * ks1
        qk[3] = qk[3] + acc3 * ks1
    return qk


@cute.jit
def s1_qk_nope_block_scaled_dsv4_h8_swap_ab(
    qk,
    q_fp8_base_addr: Int32,
    kv_fp8_base_addr: Int32,
    q_sc_view: cute.Tensor,
    kv_sc_base_addr: Int32,
    warp_first_cand: Int32,
    lane: Int32,
    *,
    num_scales: cutlass.Constexpr,
    quant_tile: cutlass.Constexpr,
    q_nope_stride: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    scale_bytes_per_token: cutlass.Constexpr,
    packed_footer_words: cutlass.Constexpr = False,
    packed_q_scale_words: cutlass.Constexpr = False,
):
    """DSV4 H8 QK with candidates in hardware A and heads in hardware B.

    Four warps cover the 64-candidate window.  Unlike the H16 orientation, both
    operands are regular ldmatrix tiles; the per-candidate UE8M0 K scale moves to
    SFA and the per-head Q scale moves to SFB.  Fragment ownership matches the
    GLM H8 swapped path.
    """
    gid = lane >> Int32(2)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    b_row = lane & Int32(7)
    b_col = ((lane >> Int32(3)) & Int32(1)) * Int32(16)

    # scale_vec::1X selector ownership follows the physical operand tiles.
    sfa_cand = warp_first_cand + gid + (lane & Int32(1)) * Int32(8)
    sfb_head = gid

    footer_lo = Uint32(0)
    footer_hi = Uint32(0)
    if cutlass.const_expr(packed_footer_words):
        footer_lo, footer_hi = ld_shared_v2_u32(
            kv_sc_base_addr + sfa_cand * Int32(scale_bytes_per_token)
        )
    q_scale_lo = Uint32(0)
    q_scale_hi = Uint32(0)
    if cutlass.const_expr(packed_q_scale_words):
        q_scale_lo, q_scale_hi = ld_shared_v2_u32(
            q_fp8_base_addr
            + sfb_head * Int32(q_nope_stride)
            + Int32(num_scales * quant_tile)
        )

    for blk in cutlass.range_constexpr(num_scales):
        if cutlass.const_expr(packed_footer_words):
            sfa = _packed_u8(footer_lo, footer_hi, blk)
        else:
            sfa = _ld_u8_zext(
                kv_sc_base_addr,
                sfa_cand * Int32(scale_bytes_per_token) + Int32(blk),
            )
        if cutlass.const_expr(packed_q_scale_words):
            sfb = _packed_u8(q_scale_lo, q_scale_hi, blk)
        else:
            sfb = _fp32_to_ue8m0_byte(
                q_sc_view[sfb_head * Int32(num_scales) + Int32(blk)]
            )
        for ks in cutlass.range_constexpr(quant_tile // 32):
            ko = Int32(blk) * Int32(quant_tile) + Int32(ks) * Int32(32)
            a_addr = _smem_byte(
                kv_fp8_base_addr,
                warp_first_cand * Int32(kv_smem_stride)
                + a_row * Int32(kv_smem_stride)
                + ko
                + a_col,
            )
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
            b_addr = _smem_byte(
                q_fp8_base_addr,
                b_row * Int32(q_nope_stride) + ko + b_col,
            )
            b0, b1 = ldmatrix_m8n8x2_b16(b_addr)
            qk[0], qk[1], qk[2], qk[3] = mxfp8_mma_m16n8k32_f32_e4m3(
                qk[0],
                qk[1],
                qk[2],
                qk[3],
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                sfa,
                sfb,
            )
    return qk


# =============================================================================
# S2 -- QK-RoPE bf16 m16n8k16 MMA (accumulate into the SAME qk regs)
# =============================================================================
@cute.jit
def s2_qk_rope_bf16(
    qk,  # length-4 python list of Float32 (in/out accum)
    q_rope_base_addr: Int32,  # u32 smem addr of q_rope (HPB x D_ROPE bf16)
    kv_rope_base_addr: Int32,  # u32 smem addr of kv_rope[buf] (BI x D_ROPE bf16)
    warp_first_cand: Int32,
    lane: Int32,
    *,
    d_rope: cutlass.Constexpr,  # 64
    q_rope_stride: cutlass.Constexpr,
    valid_hpb: cutlass.Constexpr = 16,
    fp8_rope: cutlass.Constexpr = False,
):
    """S2: accumulate Q_rope . K_rope into qk[0..3] via D_ROPE/16=4 bf16 MMAs.

    A (Q-rope, 16x16 bf16) via ldmatrix.x4 (ldmatrix_load_A_bf16); B (K-rope)
    via per-lane scalar u32 reads -- the N-major rope smem layout can't feed
    ldmatrix.x2 here (decode_dsv4 :401-425).
    """
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    # Each lane's K-rope entry = warp_first_cand + gid (nt=0).
    entry = warp_first_cand + gid
    hi = cutlass.const_expr(valid_hpb > 8)

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        # A: Q-rope (bf16) at q_rope + ks*16 (elems).
        a_byte = a_row * Int32(q_rope_stride * 2) + (ko + a_col) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_byte(q_rope_base_addr, a_byte))
        # B: flag-off retains the exact BF16 scalar loads.  Flag-on interprets
        # the staged 80-byte tail as fp32 scale at +0, pad through +15, then
        # 64 E4M3 values at +16 and reconstructs each pair to BF16 registers.
        if cutlass.const_expr(fp8_rope):
            b0 = _fp8_rope_pair_bfloat2(
                kv_rope_base_addr,
                entry,
                ko + tid * Int32(2),
                d_rope=d_rope,
            )
            b1 = _fp8_rope_pair_bfloat2(
                kv_rope_base_addr,
                entry,
                ko + tid * Int32(2) + Int32(8),
                d_rope=d_rope,
            )
        else:
            row_byte = entry * Int32(d_rope * 2) + ko * Int32(2)
            b0 = _ld_u32(kv_rope_base_addr, row_byte + tid * Int32(2) * Int32(2))
            b1 = _ld_u32(
                kv_rope_base_addr, row_byte + (tid * Int32(2) + Int32(8)) * Int32(2)
            )
        d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
            qk[0],
            qk[1],
            qk[2],
            qk[3],
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
        )
        qk[0] = d0
        qk[1] = d1
        if cutlass.const_expr(hi):
            qk[2] = d2
            qk[3] = d3
    return qk


@cute.jit
def s2_qk_rope_bf16_glm_h8_swap_ab(
    qk,
    q_rope_base_addr: Int32,
    kv_rope_base_addr: Int32,
    warp_first_cand: Int32,
    lane: Int32,
    *,
    d_rope: cutlass.Constexpr,
    q_rope_stride: cutlass.Constexpr,
    kv_rope_stride_bytes: cutlass.Constexpr,
):
    """GLM TP8 RoPE dot product in the same swapped fragment layout as S1."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    q_head = gid

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        a_byte = (warp_first_cand + a_row) * Int32(kv_rope_stride_bytes) + (
            ko + a_col
        ) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_byte(kv_rope_base_addr, a_byte))
        row_byte = q_head * Int32(q_rope_stride * 2) + ko * Int32(2)
        b0 = _ld_u32(q_rope_base_addr, row_byte + tid * Int32(2) * Int32(2))
        b1 = _ld_u32(
            q_rope_base_addr,
            row_byte + (tid * Int32(2) + Int32(8)) * Int32(2),
        )
        qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(
            qk[0],
            qk[1],
            qk[2],
            qk[3],
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
        )
    return qk


# =============================================================================
# S3 -- mask invalid candidates -> -inf, then qk *= sm_scale * LOG2E
# =============================================================================
@cute.jit
def s3_mask_and_scale(
    qk,  # length-4 python list of Float32 (in/out)
    sTokenIdx: cute.Tensor,  # smem int32 validity buffer (BI,)
    warp_first_cand: Int32,
    split_cand_start: Int32,  # chunk_in_section * DSV4_CAND_WINDOW
    split_cand_end: Int32,  # min(start + CAND_WINDOW, section_len)
    section_len: Int32,
    sm_scale_log2: Float32,  # sm_scale * LOG2E
    lane: Int32,
):
    """S3: mask + base-2 prescale. Validity = staged token index (gap #9).

    The lane owns candidate columns c0 = warp_first_cand + tid*2 and c1 = c0+1
    (qk[0],qk[2] -> c0 ; qk[1],qk[3] -> c1). Invalid if the absolute candidate
    position is past split_cand_end OR the staged raw slot id is < 0.

    Masked qk is set to ``-1e30`` (NOT -inf) -- matches decode_dsv4 :443. After
    the ``* sm_scale_log2`` this is a large FINITE negative (~-6.38e28); the S4
    online softmax relies on this finiteness so a fully-invalid warp's
    ``qk - local_max`` is 0 (not the ``inf - inf = nan`` that -inf would give).
    The merge sentinel (-inf empty-block LSE) is emitted separately in S7."""
    tid = lane & Int32(3)
    c0 = warp_first_cand + tid * Int32(2)
    c1 = c0 + Int32(1)
    abs_c0 = c0 + split_cand_start
    abs_c1 = c1 + split_cand_start

    idx0 = Int32(-1)
    if abs_c0 < section_len:
        idx0 = Int32(sTokenIdx[c0])
    idx1 = Int32(-1)
    if abs_c1 < section_len:
        idx1 = Int32(sTokenIdx[c1])

    if abs_c0 >= split_cand_end or idx0 < Int32(0):
        qk[0] = Float32(_QK_MASK)
        qk[2] = Float32(_QK_MASK)
    if abs_c1 >= split_cand_end or idx1 < Int32(0):
        qk[1] = Float32(_QK_MASK)
        qk[3] = Float32(_QK_MASK)

    qk[0] = qk[0] * sm_scale_log2
    qk[1] = qk[1] * sm_scale_log2
    qk[2] = qk[2] * sm_scale_log2
    qk[3] = qk[3] * sm_scale_log2
    return qk


@cute.jit
def s3_mask_and_scale_glm_h8_swap_ab(
    qk,
    sTokenIdx: cute.Tensor,
    warp_first_cand: Int32,
    split_cand_start: Int32,
    split_cand_end: Int32,
    section_len: Int32,
    sm_scale_log2: Float32,
    lane: Int32,
):
    """Mask the two candidate rows owned by a swapped GLM TP8 fragment."""
    gid = lane >> Int32(2)
    c0 = warp_first_cand + gid
    c1 = c0 + Int32(8)
    abs_c0 = c0 + split_cand_start
    abs_c1 = c1 + split_cand_start

    idx0 = Int32(-1)
    if abs_c0 < section_len:
        idx0 = Int32(sTokenIdx[c0])
    idx1 = Int32(-1)
    if abs_c1 < section_len:
        idx1 = Int32(sTokenIdx[c1])

    if abs_c0 >= split_cand_end or idx0 < Int32(0):
        qk[0] = Float32(_QK_MASK)
        qk[1] = Float32(_QK_MASK)
    if abs_c1 >= split_cand_end or idx1 < Int32(0):
        qk[2] = Float32(_QK_MASK)
        qk[3] = Float32(_QK_MASK)

    for i in cutlass.range_constexpr(4):
        qk[i] = qk[i] * sm_scale_log2
    return qk


# ── small smem byte-access helpers (typed loads the imported ops don't cover) ─
@cute.jit
def _fp32_to_ue8m0_byte(scale: Float32) -> Uint32:
    """sfa = (float_as_uint(scale) >> 23) & 0xFF -- scale_convert.cuh fp32_to_ue8m0.

    ``scale`` is already a pow2 (mantissa 0) so this is its exponent byte; the
    block-scaled MMA uses only the low byte of the u32 (scale_vec::1X)."""
    bits = _f32_to_bits(scale)
    return (bits >> Uint32(23)) & Uint32(0xFF)


@cute.jit
def _ld_u8_zext(base_addr: Int32, byte_off: Int32) -> Uint32:
    """Load one u8 from smem (base+byte_off), zero-extended to u32."""
    word = byte_off & ~Int32(3)
    sh = (byte_off & Int32(3)) * Int32(8)
    val = ld_shared_u32(base_addr + word)
    return (val >> sh.to(Uint32)) & Uint32(0xFF)


@cute.jit
def _packed_u8(
    lo: Uint32,
    hi: Uint32,
    byte_idx: cutlass.Constexpr,
) -> Uint32:
    if cutlass.const_expr(byte_idx < 4):
        return (lo >> Uint32(byte_idx * 8)) & Uint32(0xFF)
    return (hi >> Uint32((byte_idx - 4) * 8)) & Uint32(0xFF)


@cute.jit
def _ld_u32(base_addr: Int32, byte_off: Int32) -> Uint32:
    return ld_shared_u32(base_addr + byte_off)


@cute.jit
def _ld_u16_zext(base_addr: Int32, byte_off: Int32) -> Uint32:
    """Load one u16 (bf16) from smem (base+byte_off), zero-extended to u32.

    ``byte_off`` is always 2-byte aligned (bf16 or packed FP8-pair element).
    Use the native halfword shared-memory instruction rather than loading the
    containing word and dynamically selecting its low/high half.  The latter
    costs address masking plus SHF/LOP3 on every BF16-QK K step.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [(base_addr + byte_off).ir_value()],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _fp8_rope_pair_bfloat2(
    kv_rope_base_addr: Int32,
    entry: Int32,
    dim_even: Int32,
    *,
    d_rope: cutlass.Constexpr,
) -> Uint32:
    """Dequant one E4M3 RoPE pair from the staged 368-byte-record tail."""
    row_byte = entry * Int32(d_rope * 2)
    scale = ld_shared_f32(kv_rope_base_addr + row_byte)
    packed = _ld_u16_zext(
        kv_rope_base_addr,
        row_byte + Int32(16) + dim_even,
    )
    v0 = cvt_e4m3_to_f32_via_f16(packed & Uint32(0xFF)) * scale
    v1 = cvt_e4m3_to_f32_via_f16((packed >> Uint32(8)) & Uint32(0xFF)) * scale
    return pack_f32x2_to_bfloat2(v0, v1)


@cute.jit
def _bf16x2_extract_lane_u16(packed: Uint32, lane: Int32) -> Uint32:
    sh = (lane & Int32(1)) * Int32(16)
    return (packed >> sh.to(Uint32)) & Uint32(0xFFFF)


@cute.jit
def _nvfp4_pair_bfloat2(
    kv_fp4_base_addr: Int32,
    entry: Int32,
    dim_even: Int32,
    latent_scale: Float32,
    *,
    kv_smem_stride: cutlass.Constexpr,
) -> Uint32:
    """Dequant E2M1 * inline E4M3 * per-layer outer scale to bf16x2."""
    data_byte = _ld_u8_zext(
        kv_fp4_base_addr,
        entry * Int32(kv_smem_stride) + (dim_even // Int32(2)),
    )
    vals_h2 = fp4_decode_2(data_byte)
    v0, v1 = f16x2_to_f32x2(vals_h2)
    scale_group = dim_even // Int32(_NVFP4_SCALE_GROUP)
    scale_byte = _ld_u8_zext(
        kv_fp4_base_addr,
        entry * Int32(kv_smem_stride) + Int32(_NVFP4_SCALE_OFFSET) + scale_group,
    )
    scale_f = cvt_e4m3_to_f32_via_f16(scale_byte)
    # This is the single NVFP4 decode dequant point shared by QK and P.V.
    # Apply s_l before BF16 packing so both MMAs consume the same true-magnitude
    # latent; the separate BF16 RoPE path never calls this helper.
    return pack_f32x2_to_bfloat2(
        (v0 * scale_f) * latent_scale,
        (v1 * scale_f) * latent_scale,
    )


@cute.jit
def _nvfp4_scalar_bf16_u16(
    kv_fp4_base_addr: Int32,
    entry: Int32,
    dim: Int32,
    latent_scale: Float32,
    *,
    kv_smem_stride: cutlass.Constexpr,
) -> Uint32:
    pair = _nvfp4_pair_bfloat2(
        kv_fp4_base_addr,
        entry,
        dim & ~Int32(1),
        latent_scale,
        kv_smem_stride=kv_smem_stride,
    )
    return _bf16x2_extract_lane_u16(pair, dim & Int32(1))


@cute.jit
def _ue8m0_byte_to_fp32(byte: Uint32) -> Float32:
    """value = 2^(byte - 127) -- scale_convert.cuh ue8m0_to_fp32 (the V-scale).

    Reconstructs the FP32 power-of-2 from its UE8M0 exponent byte by placing the
    byte in the IEEE-754 exponent field (zero mantissa)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(byte & Uint32(0xFF)).ir_value()],
            "{ .reg .b32 b; shl.b32 b, $1, 23; mov.b32 $0, b; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _ue8m0_zext_byte_to_fp32(byte: Uint32) -> Float32:
    """Expand an already zero-extended UE8M0 byte to its FP32 power of two.

    Native ``ld.shared.u8`` and the ``prmt(..., 0, 0x777x)`` scale-cache form
    both guarantee that the upper 24 bits are zero, so masking again only adds
    an instruction and diverges from the native prefill PTX.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(byte).ir_value()],
            "{ .reg .b32 b; shl.b32 b, $1, 23; mov.b32 $0, b; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# S4 -- online softmax (warp-local + cross-warp + cross-chunk rescale)
# Port of decode_dsv4 :456-566 + online_softmax.cuh; base-2 (exp2f).
# =============================================================================
@cute.jit
def s4_online_softmax(
    qk,  # length-4 Float32 (masked, base-2-prescaled)
    p,  # length-4 Float32 out: exp2(qk - local_max)
    acc_nope,  # list[N_V_CHUNKS] of length-4 Float32 lists (in/out)
    acc_rope,  # length-4 Float32 list (in/out)
    global_max,  # length-2 Float32 list (in/out): [row0, row1]
    global_sum,  # length-2 Float32 list (in/out)
    reduce_max_addr: Int32,  # u32 smem addr of warp_max (N_WARPS*HPB f32)
    reduce_sum_addr: Int32,  # u32 smem addr of warp_sum (N_WARPS*HPB f32)
    is_first_chunk: cutlass.Constexpr,  # True for the very first chunk in the loop
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,  # flat thread id [0, MATH_THREADS)
    *,
    n_v_chunks: cutlass.Constexpr,  # 7
    hpb: cutlass.Constexpr,  # 16
    n_warps: cutlass.Constexpr,  # 8
    valid_hpb: cutlass.Constexpr,  # <= HPB
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,  # math-only named-barrier slot
    n_acc_tiles: cutlass.Constexpr = None,  # len(acc_nope); defaults to n_v_chunks
):
    """S4: per-warp + cross-warp max/sum, exp2(qk-max), cross-chunk rescale.

    Returns ``(p, warp_rescale0, warp_rescale1)``; mutates ``acc_nope`` /
    ``acc_rope`` (rescaled by the cross-chunk alpha) and ``global_max`` /
    ``global_sum`` in place. ``p`` is exp2(qk - local_max) (the warp-local-frame
    probabilities); the caller multiplies by ``warp_rescale`` before storing
    sm_p_full (kills all-invalid-warp spurious 1s). Exactly decode_dsv4
    :456-587."""
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    # alpha "has-prev-max" guard threshold == decode_dsv4 :520 (> -1e29f).
    _NI = Float32(-1e29)

    # --- per-warp local max over the 2 physical rows (gid heads, gid+8 heads). ---
    local_max0 = fmax_f32(qk[0], qk[1])
    local_max1 = fmax_f32(qk[2], qk[3])
    # shfl_xor offsets {2,1} finish the 8-lane (tid in [0,4) over 2-lane pairs)
    # reduction across the candidate columns of this warp's N-tile.
    local_max0 = fmax_f32(local_max0, cute.arch.shuffle_sync_bfly(local_max0, offset=2))
    local_max1 = fmax_f32(local_max1, cute.arch.shuffle_sync_bfly(local_max1, offset=2))
    local_max0 = fmax_f32(local_max0, cute.arch.shuffle_sync_bfly(local_max0, offset=1))
    local_max1 = fmax_f32(local_max1, cute.arch.shuffle_sync_bfly(local_max1, offset=1))

    # --- p = exp2(qk - local_max); local sum over the 4 frag entries. ---
    p[0] = _exp2_approx_ftz_f32(qk[0] - local_max0)
    p[1] = _exp2_approx_ftz_f32(qk[1] - local_max0)
    p[2] = _exp2_approx_ftz_f32(qk[2] - local_max1)
    p[3] = _exp2_approx_ftz_f32(qk[3] - local_max1)
    local_sum0 = p[0] + p[1]
    local_sum1 = p[2] + p[3]
    local_sum0 = local_sum0 + cute.arch.shuffle_sync_bfly(local_sum0, offset=2)
    local_sum1 = local_sum1 + cute.arch.shuffle_sync_bfly(local_sum1, offset=2)
    local_sum0 = local_sum0 + cute.arch.shuffle_sync_bfly(local_sum0, offset=1)
    local_sum1 = local_sum1 + cute.arch.shuffle_sync_bfly(local_sum1, offset=1)

    # --- cross-warp reduce: stage per-warp (max,sum) for heads gid / gid+8. ---
    if tid == Int32(0):
        st_shared_f32(
            reduce_max_addr + (warp_id * Int32(hpb) + gid) * Int32(4), local_max0
        )
        st_shared_f32(
            reduce_max_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            local_max1,
        )
        st_shared_f32(
            reduce_sum_addr + (warp_id * Int32(hpb) + gid) * Int32(4), local_sum0
        )
        st_shared_f32(
            reduce_sum_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            local_sum1,
        )
    cute.arch.barrier(**bar_kw)

    # one thread per head folds the 8 warps' (max,sum) into a block reduction.
    if tid_flat < Int32(valid_hpb):
        h = tid_flat
        bmax = Float32(-1e30)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(reduce_max_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bmax = fmax_f32(bmax, wm)
        bsum = Float32(0.0)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(reduce_max_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            ws = ld_shared_f32(reduce_sum_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bsum = bsum + ws * _exp2_approx_ftz_f32(wm - bmax)
        st_shared_f32(reduce_max_addr + h * Int32(4), bmax)
        st_shared_f32(reduce_sum_addr + h * Int32(4), bsum)
    cute.arch.barrier(**bar_kw)

    block_local_max0 = ld_shared_f32(reduce_max_addr + gid * Int32(4))
    block_local_max1 = ld_shared_f32(reduce_max_addr + (gid + Int32(8)) * Int32(4))
    block_local_sum0 = ld_shared_f32(reduce_sum_addr + gid * Int32(4))
    block_local_sum1 = ld_shared_f32(reduce_sum_addr + (gid + Int32(8)) * Int32(4))

    # --- online softmax update (cross-chunk). ---
    new_gmax0 = fmax_f32(global_max[0], block_local_max0)
    new_gmax1 = fmax_f32(global_max[1], block_local_max1)
    alpha0 = Float32(0.0)
    alpha1 = Float32(0.0)
    if global_max[0] > _NI:
        alpha0 = _exp2_approx_ftz_f32(global_max[0] - new_gmax0)
    if global_max[1] > _NI:
        alpha1 = _exp2_approx_ftz_f32(global_max[1] - new_gmax1)
    block_rescale0 = _exp2_approx_ftz_f32(block_local_max0 - new_gmax0)
    block_rescale1 = _exp2_approx_ftz_f32(block_local_max1 - new_gmax1)
    warp_rescale0 = _exp2_approx_ftz_f32(local_max0 - new_gmax0)
    warp_rescale1 = _exp2_approx_ftz_f32(local_max1 - new_gmax1)

    _n_acc = cutlass.const_expr(n_acc_tiles if n_acc_tiles is not None else n_v_chunks)
    if cutlass.const_expr(not is_first_chunk):
        for vc in cutlass.range_constexpr(_n_acc):
            acc_nope[vc][0] = acc_nope[vc][0] * alpha0
            acc_nope[vc][1] = acc_nope[vc][1] * alpha0
            acc_nope[vc][2] = acc_nope[vc][2] * alpha1
            acc_nope[vc][3] = acc_nope[vc][3] * alpha1
        acc_rope[0] = acc_rope[0] * alpha0
        acc_rope[1] = acc_rope[1] * alpha0
        acc_rope[2] = acc_rope[2] * alpha1
        acc_rope[3] = acc_rope[3] * alpha1
        global_sum[0] = global_sum[0] * alpha0 + block_local_sum0 * block_rescale0
        global_sum[1] = global_sum[1] * alpha1 + block_local_sum1 * block_rescale1
    else:
        global_sum[0] = block_local_sum0 * block_rescale0
        global_sum[1] = block_local_sum1 * block_rescale1
    global_max[0] = new_gmax0
    global_max[1] = new_gmax1

    return p, warp_rescale0, warp_rescale1


@cute.jit
def s4_online_softmax_glm_h8_swap_ab(
    qk,
    p,
    acc_nope,
    acc_rope,
    global_max,
    global_sum,
    reduce_max_addr: Int32,
    reduce_sum_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    n_acc_tiles: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    rope_tiles_per_warp: cutlass.Constexpr = 0,
    barrier_threads: cutlass.Constexpr = 0,  # barrier width override (0 -> num_threads)
):
    """Online softmax for the swapped 16-candidate x 8-head score tile.

    ``global_{max,sum}[0:2]`` track the head pair ``(2*tid, 2*tid+1)``.
    PV accumulators instead use the ordinary output-row ownership (head=gid),
    so the cross-chunk alpha is shuffled from the lane owning that head pair
    before rescaling the accumulator rows.
    """
    bar_kw = dict(
        barrier_id=barrier_id,
        number_of_threads=(barrier_threads if barrier_threads else num_threads),
    )
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)
    _NI = Float32(-1e29)

    local_max0 = fmax_f32(qk[0], qk[2])
    local_max1 = fmax_f32(qk[1], qk[3])
    for off in (4, 8, 16):
        local_max0 = fmax_f32(
            local_max0, cute.arch.shuffle_sync_bfly(local_max0, offset=off)
        )
        local_max1 = fmax_f32(
            local_max1, cute.arch.shuffle_sync_bfly(local_max1, offset=off)
        )

    p[0] = _exp2_approx_ftz_f32(qk[0] - local_max0)
    p[1] = _exp2_approx_ftz_f32(qk[1] - local_max1)
    p[2] = _exp2_approx_ftz_f32(qk[2] - local_max0)
    p[3] = _exp2_approx_ftz_f32(qk[3] - local_max1)
    local_sum0 = p[0] + p[2]
    local_sum1 = p[1] + p[3]
    for off in (4, 8, 16):
        local_sum0 = local_sum0 + cute.arch.shuffle_sync_bfly(local_sum0, offset=off)
        local_sum1 = local_sum1 + cute.arch.shuffle_sync_bfly(local_sum1, offset=off)

    # Lanes 0..3 (gid==0) own the four head pairs for this warp.
    if gid == Int32(0):
        st_shared_f32(
            reduce_max_addr + (warp_id * Int32(hpb) + head0) * Int32(4),
            local_max0,
        )
        st_shared_f32(
            reduce_max_addr + (warp_id * Int32(hpb) + head1) * Int32(4),
            local_max1,
        )
        st_shared_f32(
            reduce_sum_addr + (warp_id * Int32(hpb) + head0) * Int32(4),
            local_sum0,
        )
        st_shared_f32(
            reduce_sum_addr + (warp_id * Int32(hpb) + head1) * Int32(4),
            local_sum1,
        )
    cute.arch.barrier(**bar_kw)

    if tid_flat < Int32(8):
        h = tid_flat
        bmax = Float32(-1e30)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(reduce_max_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bmax = fmax_f32(bmax, wm)
        bsum = Float32(0.0)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(reduce_max_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            ws = ld_shared_f32(reduce_sum_addr + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bsum = bsum + ws * _exp2_approx_ftz_f32(wm - bmax)
        st_shared_f32(reduce_max_addr + h * Int32(4), bmax)
        st_shared_f32(reduce_sum_addr + h * Int32(4), bsum)
    cute.arch.barrier(**bar_kw)

    block_max0 = ld_shared_f32(reduce_max_addr + head0 * Int32(4))
    block_max1 = ld_shared_f32(reduce_max_addr + head1 * Int32(4))
    block_sum0 = ld_shared_f32(reduce_sum_addr + head0 * Int32(4))
    block_sum1 = ld_shared_f32(reduce_sum_addr + head1 * Int32(4))

    new_gmax0 = fmax_f32(global_max[0], block_max0)
    new_gmax1 = fmax_f32(global_max[1], block_max1)
    alpha0 = Float32(0.0)
    alpha1 = Float32(0.0)
    if global_max[0] > _NI:
        alpha0 = _exp2_approx_ftz_f32(global_max[0] - new_gmax0)
    if global_max[1] > _NI:
        alpha1 = _exp2_approx_ftz_f32(global_max[1] - new_gmax1)

    block_rescale0 = _exp2_approx_ftz_f32(block_max0 - new_gmax0)
    block_rescale1 = _exp2_approx_ftz_f32(block_max1 - new_gmax1)
    warp_rescale0 = _exp2_approx_ftz_f32(local_max0 - new_gmax0)
    warp_rescale1 = _exp2_approx_ftz_f32(local_max1 - new_gmax1)

    # Source lanes 0..3 carry alpha for pairs (0,1), (2,3), (4,5), (6,7).
    pair_lane = gid >> Int32(1)
    row_alpha0 = cute.arch.shuffle_sync(alpha0, pair_lane)
    row_alpha1 = cute.arch.shuffle_sync(alpha1, pair_lane)
    row_alpha = row_alpha0
    if (gid & Int32(1)) != Int32(0):
        row_alpha = row_alpha1
    for at in cutlass.range_constexpr(n_acc_tiles):
        acc_nope[at][0] = acc_nope[at][0] * row_alpha
        acc_nope[at][1] = acc_nope[at][1] * row_alpha
    for rt in cutlass.range_constexpr(rope_tiles_per_warp):
        acc_rope[rt * 4 + 0] = acc_rope[rt * 4 + 0] * row_alpha
        acc_rope[rt * 4 + 1] = acc_rope[rt * 4 + 1] * row_alpha

    global_sum[0] = global_sum[0] * alpha0 + block_sum0 * block_rescale0
    global_sum[1] = global_sum[1] * alpha1 + block_sum1 * block_rescale1
    global_max[0] = new_gmax0
    global_max[1] = new_gmax1
    return p, warp_rescale0, warp_rescale1


# =============================================================================
# S5 -- sm_p_full fill (normalized P -> bf16) + w_head_sc zero-init
# Port of decode_dsv4 :568-593.
# =============================================================================
@cute.jit
def s5_fill_sm_p_full(
    w_pre,  # length-4 Float32 (= p * warp_rescale)
    sm_p_full_addr: Int32,  # u32 smem addr of sm_p_full (HPB x BI bf16)
    w_head_sc_view: cute.Tensor,  # smem fp32 view (N_V_CHUNKS*HPB,)
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    bi: cutlass.Constexpr,  # 64
    sm_p_stride: cutlass.Constexpr,
    n_v_chunks: cutlass.Constexpr,  # 7
    hpb: cutlass.Constexpr,  # 16
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,
):
    """S5: store warp-rescaled P (bf16) to sm_p_full; zero w_head_sc. The caller
    barriers AFTER this so S6/S6b read coherent sm_p_full / w_head_sc."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    cand_col_base = warp_id * Int32(8)  # ENTRIES_PER_WARP = 8 (DSV4_QK_N_TILES=1)
    c0 = cand_col_base + tid * Int32(2)
    c1 = c0 + Int32(1)
    # sm_p_full[head][cand] bf16.
    st_shared_bf16_from_f32(
        sm_p_full_addr + (gid * Int32(sm_p_stride) + c0) * Int32(2), w_pre[0]
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + (gid * Int32(sm_p_stride) + c1) * Int32(2), w_pre[1]
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + ((gid + Int32(8)) * Int32(sm_p_stride) + c0) * Int32(2),
        w_pre[2],
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + ((gid + Int32(8)) * Int32(sm_p_stride) + c1) * Int32(2),
        w_pre[3],
    )

    # zero-init w_head_sc (cooperative; covered by the caller's barrier).
    i = tid_flat
    while i < Int32(n_v_chunks * hpb):
        w_head_sc_view[i] = Float32(0.0)
        i += Int32(num_threads)


# =============================================================================
# S6 -- XV-NoPE: per-vc absmax + P re-quant + PLAIN m16n8k32.e4m3 MMA
# Port of decode_dsv4 :595-671.
# =============================================================================
@cute.jit
def s6_xv_nope(
    w_pre,  # length-4 Float32 (= p * warp_rescale)
    acc_nope,  # list[N_V_CHUNKS*NT_PER_WARP_XV] of length-4 (in/out)
    kv_fp8_base_addr: Int32,  # u32 smem addr of kv_fp8[buf]
    kv_sc_base_addr: Int32,  # u32 smem addr of kv_sc[buf] (BI x 8 footer)
    w_head_sc_view: cute.Tensor,  # smem fp32 view (N_V_CHUNKS*HPB,)
    w_fp8_base_addr: Int32,  # u32 smem addr of w_fp8 (2 x HPB x W_FP8_STRIDE)
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    latent_scale: Float32,
    *,
    n_v_chunks: cutlass.Constexpr,  # 7
    v_chunk: cutlass.Constexpr,  # QUANT_TILE = 64
    hpb: cutlass.Constexpr,  # 16
    bi: cutlass.Constexpr,  # 64
    kv_smem_stride: cutlass.Constexpr,  # 464
    w_fp8_stride: cutlass.Constexpr,  # BI + 16 = 80
    n_warps: cutlass.Constexpr,  # 8
    scale_bytes_per_token: cutlass.Constexpr,  # 8
    nt_per_warp_xv: cutlass.Constexpr = 1,  # 1 DSV4 / 2 GLM (V_CHUNK/N_WARPS/8)
    scale_format: cutlass.Constexpr = 0,  # UE8M0_BYTE (0) / ARBITRARY_FP32 (1)
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,
    valid_hpb: cutlass.Constexpr = 16,
    pack_hilo_rows: cutlass.Constexpr = False,
    sm_p_full_addr: Int32 = None,  # NVFP4 BF16 PV only
    sm_p_stride: cutlass.Constexpr = 0,  # NVFP4: bf16 elems per sm_p row (0 -> BI)
):
    """S6: accumulate W . V_nope into acc_nope[vc*NT+nt][0..3] via PLAIN fp8 MMAs
    (14 DSV4 / 16 GLM = N_V_CHUNKS * NT_PER_WARP_XV * (BI/32)).

    Per V-chunk vc: (1) absmax of |w_pre * V_scale| into w_head_sc[vc][head];
    (2) normalize w_head_sc -> max/FP8_MAX; (3) re-quant W to e4m3 into
    w_fp8[vc&1]; (4) per N-tile nt a PLAIN m16n8k32 MMA (W ldmatrix.x4 A, V via
    d2_load_b prmt B), post-scaled by the per-head w_head_sc. XV_KSTEPS = BI/32.

    ``scale_format`` (const_expr) selects the V scale:
      * UE8M0_BYTE (DSV4): V_scale = ue8m0_to_fp32(kv_sc footer byte for vc).
      * ARBITRARY_FP32 (GLM, P10f): V keeps RAW e4m3 (no S0b requant), so
        V_scale = the per-(candidate, vc) inline arbitrary fp32 group scale read
        from the kv_fp8 footer at ``cand*KV_SMEM_STRIDE + D_NOPE + vc*4`` (4B),
        where D_NOPE == N_V_CHUNKS*V_CHUNK (V-chunk vc == scale group vc since
        V_CHUNK == QUANT_TILE for GLM). This keeps V's per-group e4m3 mantissa
        headroom instead of requantizing it away.

    GLM 2-PASS W (P10f): a single e4m3 W (3-bit mantissa) is the dominant PV
    error for GLM (the arbitrary fp32 V scale spreads W's dynamic range, so the
    per-head W scale can't capture it -> ~0.9993 cos < the 0.9995 gate). GLM
    therefore quantizes W into a HIGH e4m3 byte + a LOW e4m3 residual byte
    (~7 effective mantissa bits) and runs the PLAIN MMA TWICE per (vc, nt),
    accumulating both into the SAME fp32 xv (W_PASSES=2). The legacy GLM kernel
    sidesteps this with a bf16 P.V MMA; the residual-split keeps the unified e4m3
    W.V MMA infrastructure (DSV4 shares it at W_PASSES=1, byte-identical). The
    fp32 V scale is folded into BOTH W passes via the w_head_sc normalizer.
    For an eight-head TP8 shard, ``pack_hilo_rows`` stores HIGH in MMA rows
    0..7 and LOW in the otherwise-unused rows 8..15. One m16 MMA then produces
    both components, preserving the residual-split math while replacing the
    two serial MMAs with one.
    ``nt_per_warp_xv`` (const_expr) tiles each V_CHUNK across N_WARPS*8 columns
    NT times: GLM's 128-dim V_CHUNK needs NT=2 (8 warps x 8 dims = 64 per nt).

    ``scale_format == NVFP4_E4M3`` (const_expr) bypasses the FP8 W machinery
    entirely: BF16 probabilities staged in sm_p (``sm_p_full_addr``) are the A
    operand and V is dequantized in registers from packed E2M1 + E4M3 group-16
    scales (see ``s6_xv_nope_nvfp4_bf16``)."""
    if cutlass.const_expr(scale_format == 2):
        return s6_xv_nope_nvfp4_bf16(
            acc_nope,
            sm_p_full_addr,
            kv_fp8_base_addr,
            warp_id,
            lane,
            latent_scale,
            n_v_chunks=n_v_chunks,
            v_chunk=v_chunk,
            bi=bi,
            kv_smem_stride=kv_smem_stride,
            n_warps=n_warps,
            nt_per_warp_xv=nt_per_warp_xv,
            sm_p_stride=(sm_p_stride if sm_p_stride else bi),
        )

    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    warp_first_cand = warp_id * Int32(8)
    cand_e0 = warp_first_cand + tid * Int32(2)
    cand_e1 = cand_e0 + Int32(1)
    hi = cutlass.const_expr(valid_hpb > 8)

    glm_scale_base = Int32(n_v_chunks) * Int32(v_chunk)  # == D_NOPE (GLM)

    def _vsc(cand: Int32, vc: int):
        if cutlass.const_expr(scale_format == 0):
            return _ue8m0_byte_to_fp32(
                _ld_u8_zext(
                    kv_sc_base_addr, cand * Int32(scale_bytes_per_token) + Int32(vc)
                )
            )
        # GLM ARBITRARY_FP32 (P10f): per-(candidate, vc) inline arbitrary fp32
        # group scale (V keeps raw e4m3). vc == scale group (V_CHUNK==QUANT_TILE).
        return _u32_to_f32(
            ld_shared_u32(
                kv_fp8_base_addr
                + cand * Int32(kv_smem_stride)
                + glm_scale_base
                + Int32(vc) * Int32(4)
            )
        )

    # --- (1) per-vc per-head absmax of |w_pre * V_scale| -> w_head_sc. ---
    for vc in cutlass.range_constexpr(n_v_chunks):
        vsc0 = _vsc(cand_e0, vc)
        vsc1 = _vsc(cand_e1, vc)
        m0 = fmax_f32(fabs_f32(w_pre[0] * vsc0), fabs_f32(w_pre[1] * vsc1))
        w_head_sc_base = shared_ptr_to_u32(w_head_sc_view.iterator)
        sc0_addr = w_head_sc_base + (Int32(vc) * Int32(hpb) + gid) * Int32(4)
        atomic_max_shared_f32(sc0_addr, m0)
        if cutlass.const_expr(hi):
            m1 = fmax_f32(fabs_f32(w_pre[2] * vsc0), fabs_f32(w_pre[3] * vsc1))
            sc1_addr = w_head_sc_base + (
                Int32(vc) * Int32(hpb) + gid + Int32(8)
            ) * Int32(4)
            atomic_max_shared_f32(sc1_addr, m1)
    cute.arch.barrier(**bar_kw)

    # --- (2) normalize w_head_sc -> max(.,1e-10)/FP8_MAX. ---
    i = tid_flat
    while i < Int32(n_v_chunks * valid_hpb):
        vc = i // Int32(valid_hpb)
        h = i - vc * Int32(valid_hpb)
        slot = vc * Int32(hpb) + h
        scale = fmax_f32(w_head_sc_view[slot], Float32(1e-10)) * Float32(1.0 / _FP8_MAX)
        w_head_sc_view[slot] = scale
        if cutlass.const_expr(pack_hilo_rows):
            # The upper eight head slots are unused by a TP8 shard. Cache the
            # reciprocal there once per (V chunk, head), replacing the same
            # reciprocal redundantly computed by every candidate lane/warp.
            w_head_sc_view[slot + Int32(8)] = Float32(1.0) / scale
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    # --- (3)+(4) per-vc re-quant + per-nt PLAIN MMA. w_fp8 double-buffered. ---
    if cutlass.const_expr(scale_format == 0):
        # DSV4 (UE8M0_BYTE): the ORIGINAL single-pass W path, kept VERBATIM so the
        # DSV4 trace/PTX is byte-identical.
        for vc in cutlass.range_constexpr(n_v_chunks):
            w_fp8_addr = w_fp8_base_addr + Int32(vc & 1) * Int32(hpb * w_fp8_stride)
            si0 = Float32(1.0) / w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
            vsc0 = _vsc(cand_e0, vc)
            vsc1 = _vsc(cand_e1, vc)
            f00 = _quant_e4m3_byte(w_pre[0] * vsc0 * si0)
            f01 = _quant_e4m3_byte(w_pre[1] * vsc1 * si0)
            st_shared_u8(
                w_fp8_addr + gid * Int32(w_fp8_stride) + cand_e0, f00.to(cutlass.Uint8)
            )
            st_shared_u8(
                w_fp8_addr + gid * Int32(w_fp8_stride) + cand_e1, f01.to(cutlass.Uint8)
            )
            if cutlass.const_expr(hi):
                si1 = (
                    Float32(1.0)
                    / w_head_sc_view[Int32(vc) * Int32(hpb) + gid + Int32(8)]
                )
                f10 = _quant_e4m3_byte(w_pre[2] * vsc0 * si1)
                f11 = _quant_e4m3_byte(w_pre[3] * vsc1 * si1)
                st_shared_u8(
                    w_fp8_addr + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e0,
                    f10.to(cutlass.Uint8),
                )
                st_shared_u8(
                    w_fp8_addr + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e1,
                    f11.to(cutlass.Uint8),
                )
            cute.arch.barrier(**bar_kw)

            sc0 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
            if cutlass.const_expr(hi):
                sc1 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid + Int32(8)]
            # A (W) ldmatrix.x4 row/col -- ldmatrix_load_A_fp8 (nt-invariant).
            a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            a_col = (lane >> Int32(4)) * Int32(16)
            for nt in cutlass.range_constexpr(nt_per_warp_xv):
                # dim = vc*V_CHUNK + (nt*N_WARPS + warp_id)*8 (covers the full V_CHUNK).
                dim = Int32(vc) * Int32(v_chunk) + (
                    Int32(nt) * Int32(n_warps) + warp_id
                ) * Int32(8)
                xv0 = Float32(0.0)
                xv1 = Float32(0.0)
                xv2 = Float32(0.0)
                xv3 = Float32(0.0)
                for kstep in cutlass.range_constexpr(bi // 32):
                    ko = Int32(kstep) * Int32(32)
                    a_addr = w_fp8_addr + a_row * Int32(w_fp8_stride) + ko + a_col
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                    b0, b1 = _d2_load_b_fp8(
                        kv_fp8_base_addr, ko, dim, lane, kv_smem_stride=kv_smem_stride
                    )
                    xv0, xv1, xv2, xv3 = mma_m16n8k32_f32_e4m3(
                        xv0, xv1, xv2, xv3, a0, a1, a2, a3, b0, b1
                    )
                at = vc * nt_per_warp_xv + nt
                acc_nope[at][0] = acc_nope[at][0] + xv0 * sc0
                acc_nope[at][1] = acc_nope[at][1] + xv1 * sc0
                if cutlass.const_expr(hi):
                    acc_nope[at][2] = acc_nope[at][2] + xv2 * sc1
                    acc_nope[at][3] = acc_nope[at][3] + xv3 * sc1
        return acc_nope

    if cutlass.const_expr(pack_hilo_rows):
        # GLM TP8: valid_hpb==8 leaves the upper half of every m16 A tile idle.
        # Put the residual for head h in row h+8, then add the corresponding
        # upper output fragment to the lower one. The same per-head scale
        # applies to both rows because LOW is a residual in normalized-W space.
        for vc in cutlass.range_constexpr(n_v_chunks):
            w_fp8_addr = w_fp8_base_addr + Int32(vc & 1) * Int32(hpb * w_fp8_stride)
            si0 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid + Int32(8)]
            sc0 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
            vsc0 = _vsc(cand_e0, vc)
            vsc1 = _vsc(cand_e1, vc)
            wn00 = w_pre[0] * vsc0 * si0
            wn01 = w_pre[1] * vsc1 * si0
            vc00 = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), wn00))
            vc01 = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), wn01))
            fhi2 = _cvt_f32x2_to_e4m3x2(vc00, vc01)
            hi00, hi01 = f16x2_to_f32x2(_cvt_e4m3x2_to_f16x2(fhi2))
            resid00 = vc00 - hi00
            resid01 = vc01 - hi01
            resid00 = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), resid00))
            resid01 = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), resid01))
            flo2 = _cvt_f32x2_to_e4m3x2(resid00, resid01)
            _st_shared_u16(
                w_fp8_addr + gid * Int32(w_fp8_stride) + cand_e0,
                fhi2,
            )
            _st_shared_u16(
                w_fp8_addr + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e0,
                flo2,
            )
            cute.arch.barrier(**bar_kw)

            a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            a_col = (lane >> Int32(4)) * Int32(16)
            for nt in cutlass.range_constexpr(nt_per_warp_xv):
                dim = Int32(vc) * Int32(v_chunk) + (
                    Int32(nt) * Int32(n_warps) + warp_id
                ) * Int32(8)
                xv0 = Float32(0.0)
                xv1 = Float32(0.0)
                xv2 = Float32(0.0)
                xv3 = Float32(0.0)
                for kstep in cutlass.range_constexpr(bi // 32):
                    ko = Int32(kstep) * Int32(32)
                    a_addr = w_fp8_addr + a_row * Int32(w_fp8_stride) + ko + a_col
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                    b0, b1 = _d2_load_b_fp8(
                        kv_fp8_base_addr,
                        ko,
                        dim,
                        lane,
                        kv_smem_stride=kv_smem_stride,
                    )
                    xv0, xv1, xv2, xv3 = mma_m16n8k32_f32_e4m3(
                        xv0, xv1, xv2, xv3, a0, a1, a2, a3, b0, b1
                    )
                at = vc * nt_per_warp_xv + nt
                acc_nope[at][0] = acc_nope[at][0] + (xv0 + xv2) * sc0
                acc_nope[at][1] = acc_nope[at][1] + (xv1 + xv3) * sc0
        return acc_nope

    # GLM (ARBITRARY_FP32, P10f): 2-pass W (HIGH + LOW e4m3 residual) -> ~7 mantissa
    # bits, run the PLAIN MMA twice per (vc, nt) into the SAME fp32 xv. This is a
    # SEPARATE const_expr branch, so DSV4 (above) is untouched.
    for vc in cutlass.range_constexpr(n_v_chunks):
        w_fp8_addr = w_fp8_base_addr + Int32(vc & 1) * Int32(hpb * w_fp8_stride)
        si0 = Float32(1.0) / w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
        sc0 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
        if cutlass.const_expr(hi):
            si1 = Float32(1.0) / w_head_sc_view[Int32(vc) * Int32(hpb) + gid + Int32(8)]
            sc1 = w_head_sc_view[Int32(vc) * Int32(hpb) + gid + Int32(8)]
        vsc0 = _vsc(cand_e0, vc)
        vsc1 = _vsc(cand_e1, vc)
        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col = (lane >> Int32(4)) * Int32(16)
        # normalized W (= w_pre * V_scale / w_head_sc), the e4m3 quant target.
        wn00 = w_pre[0] * vsc0 * si0
        wn01 = w_pre[1] * vsc1 * si0
        if cutlass.const_expr(hi):
            wn10 = w_pre[2] * vsc0 * si1
            wn11 = w_pre[3] * vsc1 * si1
        # per-nt fp32 accumulators, carried across the W hi/lo passes.
        xv = [
            [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
            for _ in range(nt_per_warp_xv)
        ]
        for wpass in cutlass.range_constexpr(2):
            if cutlass.const_expr(wpass > 0):
                # serialize: the prev pass's MMA reads of w_fp8 must finish before
                # we overwrite it with the residual bytes (same double-buffer slot).
                cute.arch.barrier(**bar_kw)
                # LOW residual = e4m3(Wn - dequant(hi_byte)); halves W's quant error.
                f00 = _quant_e4m3_residual_byte(wn00)
                f01 = _quant_e4m3_residual_byte(wn01)
                if cutlass.const_expr(hi):
                    f10 = _quant_e4m3_residual_byte(wn10)
                    f11 = _quant_e4m3_residual_byte(wn11)
            else:
                f00 = _quant_e4m3_byte(wn00)
                f01 = _quant_e4m3_byte(wn01)
                if cutlass.const_expr(hi):
                    f10 = _quant_e4m3_byte(wn10)
                    f11 = _quant_e4m3_byte(wn11)
            st_shared_u8(
                w_fp8_addr + gid * Int32(w_fp8_stride) + cand_e0, f00.to(cutlass.Uint8)
            )
            st_shared_u8(
                w_fp8_addr + gid * Int32(w_fp8_stride) + cand_e1, f01.to(cutlass.Uint8)
            )
            if cutlass.const_expr(hi):
                st_shared_u8(
                    w_fp8_addr + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e0,
                    f10.to(cutlass.Uint8),
                )
                st_shared_u8(
                    w_fp8_addr + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e1,
                    f11.to(cutlass.Uint8),
                )
            cute.arch.barrier(**bar_kw)

            for nt in cutlass.range_constexpr(nt_per_warp_xv):
                # dim = vc*V_CHUNK + (nt*N_WARPS + warp_id)*8 (covers the full V_CHUNK).
                dim = Int32(vc) * Int32(v_chunk) + (
                    Int32(nt) * Int32(n_warps) + warp_id
                ) * Int32(8)
                xv0 = xv[nt][0]
                xv1 = xv[nt][1]
                xv2 = xv[nt][2]
                xv3 = xv[nt][3]
                for kstep in cutlass.range_constexpr(bi // 32):
                    ko = Int32(kstep) * Int32(32)
                    a_addr = w_fp8_addr + a_row * Int32(w_fp8_stride) + ko + a_col
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                    b0, b1 = _d2_load_b_fp8(
                        kv_fp8_base_addr, ko, dim, lane, kv_smem_stride=kv_smem_stride
                    )
                    xv0, xv1, xv2, xv3 = mma_m16n8k32_f32_e4m3(
                        xv0, xv1, xv2, xv3, a0, a1, a2, a3, b0, b1
                    )
                xv[nt][0] = xv0
                xv[nt][1] = xv1
                xv[nt][2] = xv2
                xv[nt][3] = xv3
        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            at = vc * nt_per_warp_xv + nt
            acc_nope[at][0] = acc_nope[at][0] + xv[nt][0] * sc0
            acc_nope[at][1] = acc_nope[at][1] + xv[nt][1] * sc0
            if cutlass.const_expr(hi):
                acc_nope[at][2] = acc_nope[at][2] + xv[nt][2] * sc1
                acc_nope[at][3] = acc_nope[at][3] + xv[nt][3] * sc1
    return acc_nope


@cute.jit
def s6_xv_nope_nvfp4_bf16(
    acc_nope,
    sm_p_full_addr: Int32,
    kv_fp4_base_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    latent_scale: Float32,
    *,
    n_v_chunks: cutlass.Constexpr,  # 8
    v_chunk: cutlass.Constexpr,  # 64
    bi: cutlass.Constexpr,  # 64
    kv_smem_stride: cutlass.Constexpr,  # 288 bytes
    n_warps: cutlass.Constexpr,  # 8
    nt_per_warp_xv: cutlass.Constexpr,  # 1
    sm_p_stride: cutlass.Constexpr = 0,  # bf16 elems per sm_p row (0 -> BI)
):
    """S6 (NVFP4): BF16 P.V over in-register dequantized E2M1 V.

    V is the same 512-dim MLA latent as K-NoPE. The BF16 probabilities staged by
    S5 are used directly as the A operand; each B scalar is dequantized from the
    packed E2M1 byte and its E4M3 group-16 scale.
    """
    p_stride = cutlass.const_expr(sm_p_stride if sm_p_stride else bi)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)

    for vc in cutlass.range_constexpr(n_v_chunks):
        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            dim_base = Int32(vc) * Int32(v_chunk) + (
                Int32(nt) * Int32(n_warps) + warp_id
            ) * Int32(8)
            col = dim_base + gid
            xv0 = Float32(0.0)
            xv1 = Float32(0.0)
            xv2 = Float32(0.0)
            xv3 = Float32(0.0)
            for ks in cutlass.range_constexpr(bi // 16):
                k_base = Int32(ks) * Int32(16)
                a_byte = (a_row * Int32(p_stride) + (k_base + a_col)) * Int32(2)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sm_p_full_addr + a_byte)

                ent0 = k_base + tid * Int32(2)
                v0 = _nvfp4_scalar_bf16_u16(
                    kv_fp4_base_addr,
                    ent0,
                    col,
                    latent_scale,
                    kv_smem_stride=kv_smem_stride,
                )
                v1 = _nvfp4_scalar_bf16_u16(
                    kv_fp4_base_addr,
                    ent0 + Int32(1),
                    col,
                    latent_scale,
                    kv_smem_stride=kv_smem_stride,
                )
                v8 = _nvfp4_scalar_bf16_u16(
                    kv_fp4_base_addr,
                    ent0 + Int32(8),
                    col,
                    latent_scale,
                    kv_smem_stride=kv_smem_stride,
                )
                v9 = _nvfp4_scalar_bf16_u16(
                    kv_fp4_base_addr,
                    ent0 + Int32(9),
                    col,
                    latent_scale,
                    kv_smem_stride=kv_smem_stride,
                )
                b0 = v0 | (v1 << Uint32(16))
                b1 = v8 | (v9 << Uint32(16))
                xv0, xv1, xv2, xv3 = mma_m16n8k16_f32_bf16(
                    xv0,
                    xv1,
                    xv2,
                    xv3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            at = vc * nt_per_warp_xv + nt
            acc_nope[at][0] = acc_nope[at][0] + xv0
            acc_nope[at][1] = acc_nope[at][1] + xv1
            acc_nope[at][2] = acc_nope[at][2] + xv2
            acc_nope[at][3] = acc_nope[at][3] + xv3
    return acc_nope


@cute.jit
def s6_xv_nope_glm_h8_swap_ab(
    w_pre,
    acc_nope,
    kv_fp8_base_addr: Int32,
    w_head_sc_view: cute.Tensor,
    w_fp8_base_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    n_v_chunks: cutlass.Constexpr,
    v_chunk: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    bi: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    w_fp8_stride: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    nt_per_warp_xv: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
):
    """GLM TP8 packed HIGH/LOW PV fed directly by swapped-score fragments.

    Four warps each own sixteen candidates.  HIGH for each of the eight heads
    occupies W rows 0..7 and the e4m3 residual occupies rows 8..15, so one
    m16n8k32 MMA accumulates both precision passes.  The ordinary MMA output-row
    mapping is retained for the epilogue (lane gid owns head gid).
    """
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)
    cand0 = warp_id * Int32(16) + gid
    cand1 = cand0 + Int32(8)
    glm_scale_base = Int32(n_v_chunks) * Int32(v_chunk)

    def _vsc(cand: Int32, vc: int):
        return _u32_to_f32(
            ld_shared_u32(
                kv_fp8_base_addr
                + cand * Int32(kv_smem_stride)
                + glm_scale_base
                + Int32(vc) * Int32(4)
            )
        )

    # Clear both the lower scale slots and the upper reciprocal scratch slots.
    i = tid_flat
    while i < Int32(n_v_chunks * hpb):
        w_head_sc_view[i] = Float32(0.0)
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    for vc in cutlass.range_constexpr(n_v_chunks):
        vsc0 = _vsc(cand0, vc)
        vsc1 = _vsc(cand1, vc)
        m0 = fmax_f32(fabs_f32(w_pre[0] * vsc0), fabs_f32(w_pre[2] * vsc1))
        m1 = fmax_f32(fabs_f32(w_pre[1] * vsc0), fabs_f32(w_pre[3] * vsc1))
        scale_base = shared_ptr_to_u32(w_head_sc_view.iterator)
        atomic_max_shared_f32(
            scale_base + (Int32(vc) * Int32(hpb) + head0) * Int32(4), m0
        )
        atomic_max_shared_f32(
            scale_base + (Int32(vc) * Int32(hpb) + head1) * Int32(4), m1
        )
    cute.arch.barrier(**bar_kw)

    i = tid_flat
    while i < Int32(n_v_chunks * 8):
        vc = i // Int32(8)
        h = i - vc * Int32(8)
        slot = vc * Int32(hpb) + h
        scale = fmax_f32(w_head_sc_view[slot], Float32(1e-10)) * Float32(1.0 / _FP8_MAX)
        w_head_sc_view[slot] = scale
        w_head_sc_view[slot + Int32(8)] = Float32(1.0) / scale
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    for vc in cutlass.range_constexpr(n_v_chunks):
        w_fp8_addr = w_fp8_base_addr + Int32(vc & 1) * Int32(hpb * w_fp8_stride)
        si0 = w_head_sc_view[Int32(vc) * Int32(hpb) + head0 + Int32(8)]
        si1 = w_head_sc_view[Int32(vc) * Int32(hpb) + head1 + Int32(8)]
        vsc0 = _vsc(cand0, vc)
        vsc1 = _vsc(cand1, vc)

        wn00 = w_pre[0] * vsc0 * si0
        wn01 = w_pre[1] * vsc0 * si1
        wn10 = w_pre[2] * vsc1 * si0
        wn11 = w_pre[3] * vsc1 * si1

        f00 = _quant_e4m3_byte(wn00)
        f01 = _quant_e4m3_byte(wn01)
        f10 = _quant_e4m3_byte(wn10)
        f11 = _quant_e4m3_byte(wn11)
        r00 = _quant_e4m3_residual_byte(wn00)
        r01 = _quant_e4m3_residual_byte(wn01)
        r10 = _quant_e4m3_residual_byte(wn10)
        r11 = _quant_e4m3_residual_byte(wn11)

        st_shared_u8(
            w_fp8_addr + head0 * Int32(w_fp8_stride) + cand0,
            f00.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head1 * Int32(w_fp8_stride) + cand0,
            f01.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head0 * Int32(w_fp8_stride) + cand1,
            f10.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head1 * Int32(w_fp8_stride) + cand1,
            f11.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + (head0 + Int32(8)) * Int32(w_fp8_stride) + cand0,
            r00.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + (head1 + Int32(8)) * Int32(w_fp8_stride) + cand0,
            r01.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + (head0 + Int32(8)) * Int32(w_fp8_stride) + cand1,
            r10.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + (head1 + Int32(8)) * Int32(w_fp8_stride) + cand1,
            r11.to(cutlass.Uint8),
        )
        cute.arch.barrier(**bar_kw)

        # The PV MMA's output rows use the ordinary m16 fragment mapping:
        # this lane owns row/head ``gid`` irrespective of the score fragment's
        # ``2*tid`` head pair used above to populate W.
        sc_row = w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            dim = Int32(vc) * Int32(v_chunk) + (
                Int32(nt) * Int32(n_warps) + warp_id
            ) * Int32(8)
            xv0 = Float32(0.0)
            xv1 = Float32(0.0)
            xv2 = Float32(0.0)
            xv3 = Float32(0.0)
            for kstep in cutlass.range_constexpr(bi // 32):
                ko = Int32(kstep) * Int32(32)
                a_addr = w_fp8_addr + a_row * Int32(w_fp8_stride) + ko + a_col
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                b0, b1 = _d2_load_b_fp8(
                    kv_fp8_base_addr,
                    ko,
                    dim,
                    lane,
                    kv_smem_stride=kv_smem_stride,
                )
                xv0, xv1, xv2, xv3 = mma_m16n8k32_f32_e4m3(
                    xv0,
                    xv1,
                    xv2,
                    xv3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            at = vc * nt_per_warp_xv + nt
            acc_nope[at][0] = acc_nope[at][0] + (xv0 + xv2) * sc_row
            acc_nope[at][1] = acc_nope[at][1] + (xv1 + xv3) * sc_row
    return acc_nope


@cute.jit
def s6_xv_nope_dsv4_h8_swap_ab(
    w_pre,
    acc_nope,
    kv_fp8_base_addr: Int32,
    kv_sc_base_addr: Int32,
    w_head_sc_view: cute.Tensor,
    w_fp8_base_addr: Int32,
    sm_p_full_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    n_v_chunks: cutlass.Constexpr,
    v_chunk: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    bi: cutlass.Constexpr,
    sm_p_stride: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    w_fp8_stride: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    nt_per_warp_xv: cutlass.Constexpr,
    scale_bytes_per_token: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    barrier_threads: cutlass.Constexpr = 0,  # barrier width override (0 -> num_threads)
    packed_footer_words: cutlass.Constexpr = False,
):
    """DSV4 H8 PV fed directly by the swapped score fragment.

    The lower eight W rows hold the eight query heads.  Four warps each stage
    sixteen candidate probabilities, then two N tiles per warp cover every
    64-wide V group.  The same probability stores also populate the BF16 matrix
    consumed by the RoPE PV path.
    """
    bar_kw = dict(
        barrier_id=barrier_id,
        number_of_threads=(barrier_threads if barrier_threads else num_threads),
    )
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)
    cand0 = warp_id * Int32(16) + gid
    cand1 = cand0 + Int32(8)

    # Swapped score-fragment ownership -> ordinary [head, candidate] staging.
    st_shared_bf16_from_f32(
        sm_p_full_addr + (head0 * Int32(sm_p_stride) + cand0) * Int32(2), w_pre[0]
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + (head1 * Int32(sm_p_stride) + cand0) * Int32(2), w_pre[1]
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + (head0 * Int32(sm_p_stride) + cand1) * Int32(2), w_pre[2]
    )
    st_shared_bf16_from_f32(
        sm_p_full_addr + (head1 * Int32(sm_p_stride) + cand1) * Int32(2), w_pre[3]
    )

    i = tid_flat
    while i < Int32(n_v_chunks * hpb):
        w_head_sc_view[i] = Float32(0.0)
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    def _vsc(cand: Int32, vc: int):
        return _ue8m0_byte_to_fp32(
            _ld_u8_zext(
                kv_sc_base_addr,
                cand * Int32(scale_bytes_per_token) + Int32(vc),
            )
        )

    scale_base = shared_ptr_to_u32(w_head_sc_view.iterator)
    max_footer0_lo = Uint32(0)
    max_footer0_hi = Uint32(0)
    max_footer1_lo = Uint32(0)
    max_footer1_hi = Uint32(0)
    if cutlass.const_expr(packed_footer_words):
        max_footer0_lo, max_footer0_hi = ld_shared_v2_u32(
            kv_sc_base_addr + cand0 * Int32(scale_bytes_per_token)
        )
        max_footer1_lo, max_footer1_hi = ld_shared_v2_u32(
            kv_sc_base_addr + cand1 * Int32(scale_bytes_per_token)
        )
    for vc in cutlass.range_constexpr(n_v_chunks):
        if cutlass.const_expr(packed_footer_words):
            vsc0 = _ue8m0_zext_byte_to_fp32(
                _packed_u8(max_footer0_lo, max_footer0_hi, vc)
            )
            vsc1 = _ue8m0_zext_byte_to_fp32(
                _packed_u8(max_footer1_lo, max_footer1_hi, vc)
            )
        else:
            vsc0 = _vsc(cand0, vc)
            vsc1 = _vsc(cand1, vc)
        atomic_max_shared_f32(
            scale_base + (Int32(vc) * Int32(hpb) + head0) * Int32(4),
            fmax_f32(fabs_f32(w_pre[0] * vsc0), fabs_f32(w_pre[2] * vsc1)),
        )
        atomic_max_shared_f32(
            scale_base + (Int32(vc) * Int32(hpb) + head1) * Int32(4),
            fmax_f32(fabs_f32(w_pre[1] * vsc0), fabs_f32(w_pre[3] * vsc1)),
        )
    cute.arch.barrier(**bar_kw)

    # Lower slots hold the dequant scale; upper slots hold its reciprocal.
    i = tid_flat
    while i < Int32(n_v_chunks * 8):
        vc = i // Int32(8)
        h = i - vc * Int32(8)
        slot = vc * Int32(hpb) + h
        scale = fmax_f32(w_head_sc_view[slot], Float32(1e-10)) * Float32(1.0 / _FP8_MAX)
        w_head_sc_view[slot] = scale
        w_head_sc_view[slot + Int32(8)] = Float32(1.0) / scale
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    quant_footer0_lo = Uint32(0)
    quant_footer0_hi = Uint32(0)
    quant_footer1_lo = Uint32(0)
    quant_footer1_hi = Uint32(0)
    if cutlass.const_expr(packed_footer_words):
        quant_footer0_lo, quant_footer0_hi = ld_shared_v2_u32(
            kv_sc_base_addr + cand0 * Int32(scale_bytes_per_token)
        )
        quant_footer1_lo, quant_footer1_hi = ld_shared_v2_u32(
            kv_sc_base_addr + cand1 * Int32(scale_bytes_per_token)
        )
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    for vc in cutlass.range_constexpr(n_v_chunks):
        w_fp8_addr = w_fp8_base_addr + Int32(vc & 1) * Int32(hpb * w_fp8_stride)
        si0 = w_head_sc_view[Int32(vc) * Int32(hpb) + head0 + Int32(8)]
        si1 = w_head_sc_view[Int32(vc) * Int32(hpb) + head1 + Int32(8)]
        if cutlass.const_expr(packed_footer_words):
            vsc0 = _ue8m0_zext_byte_to_fp32(
                _packed_u8(quant_footer0_lo, quant_footer0_hi, vc)
            )
            vsc1 = _ue8m0_zext_byte_to_fp32(
                _packed_u8(quant_footer1_lo, quant_footer1_hi, vc)
            )
        else:
            vsc0 = _vsc(cand0, vc)
            vsc1 = _vsc(cand1, vc)

        f00 = _quant_e4m3_byte(w_pre[0] * vsc0 * si0)
        f01 = _quant_e4m3_byte(w_pre[1] * vsc0 * si1)
        f10 = _quant_e4m3_byte(w_pre[2] * vsc1 * si0)
        f11 = _quant_e4m3_byte(w_pre[3] * vsc1 * si1)
        st_shared_u8(
            w_fp8_addr + head0 * Int32(w_fp8_stride) + cand0,
            f00.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head1 * Int32(w_fp8_stride) + cand0,
            f01.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head0 * Int32(w_fp8_stride) + cand1,
            f10.to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_addr + head1 * Int32(w_fp8_stride) + cand1,
            f11.to(cutlass.Uint8),
        )
        cute.arch.barrier(**bar_kw)

        sc_row = w_head_sc_view[Int32(vc) * Int32(hpb) + gid]
        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            dim = Int32(vc) * Int32(v_chunk) + (
                Int32(nt) * Int32(n_warps) + warp_id
            ) * Int32(8)
            xv0 = Float32(0.0)
            xv1 = Float32(0.0)
            xv2 = Float32(0.0)
            xv3 = Float32(0.0)
            for kstep in cutlass.range_constexpr(bi // 32):
                ko = Int32(kstep) * Int32(32)
                a_addr = w_fp8_addr + a_row * Int32(w_fp8_stride) + ko + a_col
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                b0, b1 = _d2_load_b_fp8(
                    kv_fp8_base_addr,
                    ko,
                    dim,
                    lane,
                    kv_smem_stride=kv_smem_stride,
                )
                xv0, xv1, xv2, xv3 = mma_m16n8k32_f32_e4m3(
                    xv0, xv1, xv2, xv3, a0, a1, a2, a3, b0, b1
                )
            at = vc * nt_per_warp_xv + nt
            acc_nope[at][0] = acc_nope[at][0] + xv0 * sc_row
            acc_nope[at][1] = acc_nope[at][1] + xv1 * sc_row
    return acc_nope


# =============================================================================
# S6b -- XV-RoPE: bf16 m16n8k16 MMA (V_HAS_ROPE gate), V read from sm_kv_rope
# Port of decode_dsv4 :673-708.
# =============================================================================
@cute.jit
def s6b_xv_rope(
    acc_rope,  # length-4 Float32 (in/out)
    sm_p_full_addr: Int32,  # u32 smem addr of sm_p_full (HPB x BI bf16)
    kv_rope_base_addr: Int32,  # u32 smem addr of kv_rope[buf] (BI x D_ROPE bf16)
    warp_id: Int32,
    lane: Int32,
    *,
    bi: cutlass.Constexpr,  # 64
    sm_p_stride: cutlass.Constexpr,
    d_rope: cutlass.Constexpr,  # 64
    n_warps: cutlass.Constexpr,  # 8
):
    """S6b: acc_rope += P . V_rope via D_ROPE/N_WARPS->BI/16=4 bf16 m16n8k16 MMAs.

    A (P) via ldmatrix.x4 from sm_p_full (bf16, stride BI); B (V_rope) via
    per-lane scalar u16 reads from sm_kv_rope packed into b0/b1. Each warp owns
    ROPE_DIMS_PER_WARP = D_ROPE/N_WARPS = 8 rope dims (n_col = warp_id*8)."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    rope_dim_base = warp_id * Int32(d_rope // n_warps)  # ROPE_DIMS_PER_WARP = 8
    col = rope_dim_base + gid  # nt=0 -> n_col = rope_dim_base

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)

    for ks in cutlass.range_constexpr(bi // 16):
        k_base = Int32(ks) * Int32(16)
        # A: sm_p_full[0][ks*16].
        a_byte = (a_row * Int32(sm_p_stride) + (k_base + a_col)) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sm_p_full_addr + a_byte)
        # B: per-lane scalar reads of v[ent][col], packed (ent0,ent1)/(ent8,ent9).
        ent0 = k_base + tid * Int32(2)
        v0 = _ld_u16_zext(kv_rope_base_addr, (ent0 * Int32(d_rope) + col) * Int32(2))
        v1 = _ld_u16_zext(
            kv_rope_base_addr, ((ent0 + Int32(1)) * Int32(d_rope) + col) * Int32(2)
        )
        v8 = _ld_u16_zext(
            kv_rope_base_addr, ((ent0 + Int32(8)) * Int32(d_rope) + col) * Int32(2)
        )
        v9 = _ld_u16_zext(
            kv_rope_base_addr, ((ent0 + Int32(9)) * Int32(d_rope) + col) * Int32(2)
        )
        b0 = v0 | (v1 << Uint32(16))
        b1 = v8 | (v9 << Uint32(16))
        acc_rope[0], acc_rope[1], acc_rope[2], acc_rope[3] = mma_m16n8k16_f32_bf16(
            acc_rope[0],
            acc_rope[1],
            acc_rope[2],
            acc_rope[3],
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
        )
    return acc_rope


@cute.jit
def s6b_xv_rope_h8_swap_ab(
    acc_rope,
    sm_p_full_addr: Int32,
    kv_rope_base_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    *,
    bi: cutlass.Constexpr,
    sm_p_stride: cutlass.Constexpr,
    d_rope: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    tiles_per_warp: cutlass.Constexpr,
    kv_rope_stride_bytes: cutlass.Constexpr,
):
    """H8 RoPE PV: two 8-column N tiles per warp cover D_ROPE=64."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)

    for nt in cutlass.range_constexpr(tiles_per_warp):
        col = (Int32(nt) * Int32(n_warps) + warp_id) * Int32(8) + gid
        r = nt * 4
        for ks in cutlass.range_constexpr(bi // 16):
            k_base = Int32(ks) * Int32(16)
            a_byte = (a_row * Int32(sm_p_stride) + k_base + a_col) * Int32(2)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sm_p_full_addr + a_byte)
            ent0 = k_base + tid * Int32(2)
            v0 = _ld_u16_zext(
                kv_rope_base_addr,
                ent0 * Int32(kv_rope_stride_bytes) + col * Int32(2),
            )
            v1 = _ld_u16_zext(
                kv_rope_base_addr,
                (ent0 + Int32(1)) * Int32(kv_rope_stride_bytes) + col * Int32(2),
            )
            v8 = _ld_u16_zext(
                kv_rope_base_addr,
                (ent0 + Int32(8)) * Int32(kv_rope_stride_bytes) + col * Int32(2),
            )
            v9 = _ld_u16_zext(
                kv_rope_base_addr,
                (ent0 + Int32(9)) * Int32(kv_rope_stride_bytes) + col * Int32(2),
            )
            b0 = v0 | (v1 << Uint32(16))
            b1 = v8 | (v9 << Uint32(16))
            (
                acc_rope[r + 0],
                acc_rope[r + 1],
                acc_rope[r + 2],
                acc_rope[r + 3],
            ) = mma_m16n8k16_f32_bf16(
                acc_rope[r + 0],
                acc_rope[r + 1],
                acc_rope[r + 2],
                acc_rope[r + 3],
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
            )
    return acc_rope


# =============================================================================
# S7 -- epilogue: per-SPLIT NORMALIZED partial writeback in split.py's convention
# Port of decode_dsv4 :725-783, adapted to the rs-1 base-2 merge (split.py:1138).
# =============================================================================
@cute.jit
def s7_epilogue(
    acc_nope,  # list[N_V_CHUNKS*NT_PER_WARP_XV] of length-4
    acc_rope,  # length-4 Float32 (unused if not v_has_rope)
    global_max,  # length-2 Float32 list
    global_sum,  # length-2 Float32 list
    out_o: cute.Tensor,  # (HPB, D_V) bf16 O view (partial mid_out OR final output)
    out_lse: cute.Tensor,  # (HPB,) f32 base-2 LSE view (mid_lse OR out_lse)
    warp_id: Int32,
    lane: Int32,
    *,
    n_v_chunks: cutlass.Constexpr,  # 7
    v_chunk: cutlass.Constexpr,  # QUANT_TILE = 64
    d_nope: cutlass.Constexpr,  # 448
    d_rope: cutlass.Constexpr,  # 64
    n_warps: cutlass.Constexpr,  # 8
    valid_hpb: cutlass.Constexpr,  # <= HPB
    nt_per_warp_xv: cutlass.Constexpr = 1,  # 1 DSV4 / 2 GLM
    v_has_rope: cutlass.Constexpr = True,  # True DSV4 / False GLM (V == nope only)
    rope_tiles_per_warp: cutlass.Constexpr = 1,
    epilogue_mode: cutlass.Constexpr = EPILOGUE_PARTIAL_WRITEBACK,
    has_attn_sink: cutlass.Constexpr = False,  # FINAL_BF16 only: fold per-head sink
    attn_sink=None,  # (heads,) f32 view (FINAL_BF16 + has_attn_sink)
    head_base: Int32 = None,  # first head index of this CTA (sink indexing)
    fast_rcp: cutlass.Constexpr = False,
    staging_base_addr: Int32 = None,
    d_v: cutlass.Constexpr = 0,
    num_threads: cutlass.Constexpr = 0,
    barrier_id: cutlass.Constexpr = 0,
    coalesced_output: cutlass.Constexpr = False,
):
    """S7: normalized O + base-2 LSE epilogue. ``epilogue_mode`` (const_expr)
    selects the destination + normalizer convention; the (gid, d0) output-write
    geometry is IDENTICAL for both, so only the per-row inverse-normalizer and the
    LSE fold diverge.

    PARTIAL_WRITEBACK (DEFAULT, decode/split -- byte-identical to the prior decode
    epilogue): write this SPLIT's NORMALIZED partial O + base-2 LSE in the exact
    active split merge convention (SparseMLASplitDecodeMergeKernel in merge.py).
    The merge reduces over the split axis assuming each partial is the
    PER-SPLIT-NORMALIZED output (acc / this split's softmax sum) tagged with the
    split's base-2 LSE (log2(sum) + max). inv_g = 1/global_sum (0 if empty);
    O[head][dim] = acc * inv_g; base-2 LSE = log2(global_sum) + global_max. Empty
    split -> LSE = -inf sentinel (the merge's skip condition) and O = 0.
    ``num_splits=1`` is the trivial 1-split merge (partial == final O).

    FINAL_BF16 (prefill single-pass): the CTA processed ALL topk in one online
    softmax, so ``global_sum``/``global_max`` are the FINAL row sum l / max m and
    we write the FINAL normalized BF16 O directly into output[token,h,d] (this
    out_o view), plus the FINAL base-2 LSE (NO merge). With ``has_attn_sink`` the
    FlashMLA V4 sink folds into BOTH the normalizer and the LSE
    (prefill_kernel.cuh:485-560, base-2 domain):
        il  = 1 / (l + exp2(sink_log2 - m))          (vs 1/l without sink)
        lse = log2(l) + m ; lse += log2(1 + exp2(sink_log2 - lse))  (empty -> sink_log2)
    where sink_log2 = attn_sink[head] * LOG2E. This is algebraically the FlashMLA
    ``out *= sigmoid(lse_e - sink)`` + log-sum-exp LSE fold the prefill reference
    cross-checks."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)

    # ── per-physical-row inverse normalizer (gid heads, gid+8 heads). ──
    if cutlass.const_expr(epilogue_mode == EPILOGUE_FINAL_BF16 and has_attn_sink):
        # FINAL_BF16 + sink: il = 1/(l + exp2(sink_log2 - m)) (FlashInfer:494).
        s0 = Float32(attn_sink[head_base + gid]) * Float32(LOG2_E)
        denom0 = global_sum[0] + _exp2_approx_ftz_f32(s0 - global_max[0])
        inv_g0 = Float32(0.0)
        if denom0 > Float32(0.0):
            if cutlass.const_expr(fast_rcp):
                inv_g0 = rcp_approx_ftz(denom0)
            else:
                inv_g0 = Float32(1.0) / denom0
        inv_g1 = Float32(0.0)
        if cutlass.const_expr(valid_hpb > 8):
            s1 = Float32(attn_sink[head_base + gid + Int32(8)]) * Float32(LOG2_E)
            denom1 = global_sum[1] + _exp2_approx_ftz_f32(s1 - global_max[1])
            if denom1 > Float32(0.0):
                if cutlass.const_expr(fast_rcp):
                    inv_g1 = rcp_approx_ftz(denom1)
                else:
                    inv_g1 = Float32(1.0) / denom1
    else:
        # PARTIAL_WRITEBACK or FINAL_BF16-without-sink: il = 1/l (0 if empty).
        inv_g0 = Float32(0.0)
        inv_g1 = Float32(0.0)
        if global_sum[0] > Float32(0.0):
            if cutlass.const_expr(fast_rcp):
                inv_g0 = rcp_approx_ftz(global_sum[0])
            else:
                inv_g0 = Float32(1.0) / global_sum[0]
        if global_sum[1] > Float32(0.0):
            if cutlass.const_expr(fast_rcp):
                inv_g1 = rcp_approx_ftz(global_sum[1])
            else:
                inv_g1 = Float32(1.0) / global_sum[1]

    # NoPE dims: d0 = vc*V_CHUNK + (nt*N_WARPS + warp_id)*8 + tid*2 (per N-tile).
    # Cast to the output element type (bf16 for the split.py mid_out partials AND
    # the prefill final output; the merge consumes bf16 -- fp32 also works for the
    # standalone probe). Output-write geometry is mode-invariant.
    _ot = out_o.element_type
    for vc in cutlass.range_constexpr(n_v_chunks):
        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            at = vc * nt_per_warp_xv + nt
            d0 = (
                Int32(vc) * Int32(v_chunk)
                + (Int32(nt) * Int32(n_warps) + warp_id) * Int32(8)
                + tid * Int32(2)
            )
            if cutlass.const_expr(coalesced_output):
                st_shared_bf16_from_f32(
                    staging_base_addr + (gid * Int32(d_v) + d0) * Int32(2),
                    acc_nope[at][0] * inv_g0,
                )
                st_shared_bf16_from_f32(
                    staging_base_addr + (gid * Int32(d_v) + d0 + Int32(1)) * Int32(2),
                    acc_nope[at][1] * inv_g0,
                )
                if cutlass.const_expr(valid_hpb > 8):
                    st_shared_bf16_from_f32(
                        staging_base_addr
                        + ((gid + Int32(8)) * Int32(d_v) + d0) * Int32(2),
                        acc_nope[at][2] * inv_g1,
                    )
                    st_shared_bf16_from_f32(
                        staging_base_addr
                        + ((gid + Int32(8)) * Int32(d_v) + d0 + Int32(1)) * Int32(2),
                        acc_nope[at][3] * inv_g1,
                    )
            else:
                out_o[gid, d0] = (acc_nope[at][0] * inv_g0).to(_ot)
                out_o[gid, d0 + Int32(1)] = (acc_nope[at][1] * inv_g0).to(_ot)
                if cutlass.const_expr(valid_hpb > 8):
                    out_o[gid + Int32(8), d0] = (acc_nope[at][2] * inv_g1).to(_ot)
                    out_o[gid + Int32(8), d0 + Int32(1)] = (
                        acc_nope[at][3] * inv_g1
                    ).to(_ot)

    # RoPE dims (DSV4 only: V_HAS_ROPE). GLM V == nope-only so this is elided.
    if cutlass.const_expr(v_has_rope):
        for nt in cutlass.range_constexpr(rope_tiles_per_warp):
            r = nt * 4
            rd0 = (
                Int32(d_nope)
                + (Int32(nt) * Int32(n_warps) + warp_id) * Int32(8)
                + tid * Int32(2)
            )
            if cutlass.const_expr(coalesced_output):
                st_shared_bf16_from_f32(
                    staging_base_addr + (gid * Int32(d_v) + rd0) * Int32(2),
                    acc_rope[r + 0] * inv_g0,
                )
                st_shared_bf16_from_f32(
                    staging_base_addr + (gid * Int32(d_v) + rd0 + Int32(1)) * Int32(2),
                    acc_rope[r + 1] * inv_g0,
                )
                if cutlass.const_expr(valid_hpb > 8):
                    st_shared_bf16_from_f32(
                        staging_base_addr
                        + ((gid + Int32(8)) * Int32(d_v) + rd0) * Int32(2),
                        acc_rope[r + 2] * inv_g1,
                    )
                    st_shared_bf16_from_f32(
                        staging_base_addr
                        + ((gid + Int32(8)) * Int32(d_v) + rd0 + Int32(1)) * Int32(2),
                        acc_rope[r + 3] * inv_g1,
                    )
            else:
                out_o[gid, rd0] = (acc_rope[r + 0] * inv_g0).to(_ot)
                out_o[gid, rd0 + Int32(1)] = (acc_rope[r + 1] * inv_g0).to(_ot)
                if cutlass.const_expr(valid_hpb > 8):
                    out_o[gid + Int32(8), rd0] = (acc_rope[r + 2] * inv_g1).to(_ot)
                    out_o[gid + Int32(8), rd0 + Int32(1)] = (
                        acc_rope[r + 3] * inv_g1
                    ).to(_ot)

    if cutlass.const_expr(coalesced_output):
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
        stores_per_head = d_v // 8
        store_idx = warp_id * Int32(32) + lane
        while store_idx < Int32(valid_hpb * stores_per_head):
            h = store_idx // Int32(stores_per_head)
            d8 = (store_idx - h * Int32(stores_per_head)) * Int32(8)
            v0, v1, v2, v3 = ld_shared_v4_u32(
                staging_base_addr + (h * Int32(d_v) + d8) * Int32(2)
            )
            st_global_v4_u32(
                get_ptr_as_int64(out_o, h.to(Int64) * Int64(d_v) + d8.to(Int64)),
                v0,
                v1,
                v2,
                v3,
            )
            store_idx += Int32(num_threads)

    # base-2 LSE (warp 0, tid 0 owns one head pair via gid). The PARTIAL_WRITEBACK
    # path is the EXACT prior decode epilogue (byte-identical IR); the FINAL_BF16 +
    # sink fold is a const_expr-gated addition that is fully elided for decode.
    if warp_id == Int32(0) and tid == Int32(0):
        lse0 = Float32(_NEG_INF)
        lse1 = Float32(_NEG_INF)
        if global_sum[0] > Float32(0.0):
            lse0 = _log2_approx_ftz_f32(global_sum[0]) + global_max[0]
        if global_sum[1] > Float32(0.0):
            lse1 = _log2_approx_ftz_f32(global_sum[1]) + global_max[1]
        if cutlass.const_expr(epilogue_mode == EPILOGUE_FINAL_BF16 and has_attn_sink):
            # FlashMLA sink fold: lse += log2(1 + exp2(sink_log2 - lse)); empty ->
            # sink_log2 (prefill_kernel.cuh:548-560).
            sink0 = Float32(attn_sink[head_base + gid]) * Float32(LOG2_E)
            if lse0 > Float32(_NEG_INF):
                lse0 = lse0 + _log2_approx_ftz_f32(
                    Float32(1.0) + _exp2_approx_ftz_f32(sink0 - lse0)
                )
            else:
                lse0 = sink0
            if cutlass.const_expr(valid_hpb > 8):
                sink1 = Float32(attn_sink[head_base + gid + Int32(8)]) * Float32(LOG2_E)
                if lse1 > Float32(_NEG_INF):
                    lse1 = lse1 + _log2_approx_ftz_f32(
                        Float32(1.0) + _exp2_approx_ftz_f32(sink1 - lse1)
                    )
                else:
                    lse1 = sink1
        out_lse[gid] = lse0
        if cutlass.const_expr(valid_hpb > 8):
            out_lse[gid + Int32(8)] = lse1


# ── extra typed helpers for the PV path ───────────────────────────────────────
@cute.jit
def _quant_e4m3_byte(v: Float32) -> Uint32:
    """Clamp to [FP8_MIN, FP8_MAX] then cvt.rn.satfinite.e4m3 -> low byte."""
    vc = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), v))
    return cvt_f32_to_e4m3(vc) & Uint32(0xFF)


@cute.jit
def _quant_e4m3_bounded_byte(v: Float32) -> Uint32:
    """Quantize an already-bounded value with saturating E4M3 conversion.

    This is for paths whose scale construction proves ``abs(v) <= FP8_MAX``.
    The saturating conversion handles any rounding overshoot without a redundant
    min/max pair at every call site.
    """
    return cvt_f32_to_e4m3(v) & Uint32(0xFF)


@cute.jit
def _quant_e4m3_residual_byte(v: Float32) -> Uint32:
    """LOW e4m3 residual byte for the GLM 2-pass W (P10f): quantize ``v`` to e4m3
    (the HIGH byte), dequant it back to f32, and quantize the residual
    ``v - dequant(hi)`` to e4m3. (HIGH @ V + LOW @ V) recovers ~7 mantissa bits of
    W, halving the e4m3 W quant error that otherwise floors GLM PV at ~0.9993 cos.
    The HIGH byte itself is re-derived here (cheap) so the caller only stages bytes."""
    vc = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), v))
    hi_byte = cvt_f32_to_e4m3(vc) & Uint32(0xFF)
    resid = vc - fp8_e4m3_to_f32(hi_byte)
    resid = fmax_f32(Float32(_FP8_MIN), fmin_f32(Float32(_FP8_MAX), resid))
    return cvt_f32_to_e4m3(resid) & Uint32(0xFF)


@cute.jit
def _d2_load_b_fp8(
    kv_fp8_base_addr: Int32,
    entry_base: Int32,
    dim: Int32,
    lane: Int32,
    *,
    kv_smem_stride: cutlass.Constexpr,
):
    """Direct-B synthesizer for the PLAIN fp8 XV MMA (d2_load_b.cuh).

    Produces b0/b1 (QMMA.16832 B operand) from kv_fp8[entry][dim] without a
    transpose buffer: 8 LDS.32 + 6 prmt. b0 byte j = V[entry_base+tid*4+j][dim+gid]
    (K=0..15); b1 = the +16 entries (K=16..31)."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    d = dim + gid
    d_base = d & ~Int32(3)
    d_sel = d & Int32(3)
    sel = ((Int32(4) + d_sel) << Int32(4)) | d_sel

    r0 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + tid * Int32(4) + Int32(0)) * Int32(kv_smem_stride)
        + d_base
    )
    r1 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + tid * Int32(4) + Int32(1)) * Int32(kv_smem_stride)
        + d_base
    )
    r2 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + tid * Int32(4) + Int32(2)) * Int32(kv_smem_stride)
        + d_base
    )
    r3 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + tid * Int32(4) + Int32(3)) * Int32(kv_smem_stride)
        + d_base
    )
    t01 = byte_perm(r0, r1, sel)
    t23 = byte_perm(r2, r3, sel)
    b0 = byte_perm(t01, t23, Int32(0x5410))

    r0 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(0)) * Int32(kv_smem_stride)
        + d_base
    )
    r1 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(1)) * Int32(kv_smem_stride)
        + d_base
    )
    r2 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(2)) * Int32(kv_smem_stride)
        + d_base
    )
    r3 = ld_shared_u32(
        kv_fp8_base_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(3)) * Int32(kv_smem_stride)
        + d_base
    )
    t01 = byte_perm(r0, r1, sel)
    t23 = byte_perm(r2, r3, sel)
    b1 = byte_perm(t01, t23, Int32(0x5410))
    return b0, b1
