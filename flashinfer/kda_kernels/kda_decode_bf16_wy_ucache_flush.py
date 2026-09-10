"""KDA (Kimi Delta Attention) decode kernel with a history ring AND
per-request state flush — the KDA analogue of the GDN ReplaySSM kernel
(``flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py``).

Implements the ReplaySSM chunked delta-rule decode for speculative decoding
of Kimi K3. Each CTA reads its request's fill level P = hist_len[b] and
either

  - P <  flush_min: runs the verify path — computes the T draft-token
    outputs and appends the T new (normed-k, u, G) entries to the ring; or
  - P >= flush_min: additionally FOLDS the ring into the checkpoint state
    and RESTARTS the ring.

Target: SM90+ (H200 / B200). Fixed K == V == 128, H == HV (KDA has no GQA).
Native draft length T in {4, 8}.

KDA vs GDN — the one structural difference
------------------------------------------
KDA's decay gate is a PER-KEY-CHANNEL vector (g in R^128 per token per
head; Kimi K3 lower-bound gate g_log = lb * sigmoid(exp(A_log) * (g +
dt_bias)), lb < 0), not GDN's per-head scalar. Consequences carried through
this kernel:

  - the cumulative-log-decay ring g_cache is [pool, H, 32, K] fp32 (a
    K-vector per slot) instead of GDN's scalar-per-slot [pool, HV, 32];
  - decay factors no longer factor out of dot products, so they are folded
    INTO the MMA operands (the PR #4709 output-only kernel's scheme):
      khat_t = k̂_t * exp(cum_t)      (packed A-tile k rows; ring holds k̂)
      qhat_t = q̂_t * exp(cum_t)      (packed A-tile q rows)
      ktil_t = k̂_t * exp(-cum_t)     (Gram B operand, one extra SMEM tile)
    and GDN's post-GEMM exp(cum_r - cum_c) Phase-2 factors disappear;
  - the replay weights w_j = exp(G_P - G_j) become per-channel tiles
    applied ELEMENTWISE to the khist tile BEFORE the scores GEMM (GDN
    scaled the scores after the GEMM by the scalar w_j); rows >= P are
    zero-WRITTEN there (select, not multiply — NaN-safe), which also serves
    the fold (no separate w-scale over the u stage);
  - bdec = e^{G_P} becomes a per-channel COLUMN scale of the packed tile,
    and the fold's FMA epilogue takes per-column bdec pairs;
  - since the decay is fully baked into hw_k / hw_q, the R pass is
    R = V - hw_k and the output is y = hw_q + QT @ R (no e^{G} factors).

Flush semantics per (request b, head h) — identical contract to GDN
-------------------------------------------------------------------
  S_h[v,c] = e^{G_P[c]} * S0[v,c] + sum_{j<P} u_j[v] * k_j[c] * w_j[c],
  w_j[c] = e^{G_P[c] - G_j[c]}
  - S_h is stored back to the state pool (S0 <- S_h);
  - the draft outputs y are computed via the SAME route as the verify path;
  - the T current drafts are NOT folded: their fresh corrections U, normed
    k, and LOCAL per-channel cumulative log-decay (restarting at 0) are
    appended at ring slots (base+P+s) & RING_MASK — PAST the fold-source
    window [base, base+P) (RING_SLOTS = 32);
  - cursor commits (base slide, len reset) are CALLER-OWNED, outside the
    launch: flush rows base' = (base+P) & RING_MASK, len' = accepted.

Precisions
----------
bf16 only: q/k/v/g/b, the state pool, the u and k rings, and the output are
all bf16 (the KDA stack is bf16-enforced end to end; GDN's default arm).
A_log and dt_bias are fp32 (the KDA convention). g_cache is fp32 always.
All GEMM accumulation and the gate / log-decay math run in f32 internally.

Tensors (public entry point ``kda_delta_rule_mtp_ucache_flush``)
----------------------------------------------------------------
  Name                   Shape              Dtype   Dir      Meaning
  ---------------------  -----------------  ------  -------  ------------------------
  q, k                   [B, T, H,  K]      bf16    in       draft query / key
  v                      [B, T, H, V]       bf16    in       draft values
  g                      [B, T, H, K]       bf16    in       raw per-channel gate pre-activation
  b                      [B, T, H]          bf16    in       per-token beta logit (sigmoid in-kernel)
  A_log                  [H]                fp32    in       per-head log-decay rate
  dt_bias                [H*K]              fp32    in       per-(head, channel) gate bias
  initial_state_source   [pool, H, V, K]    bf16    in/out   checkpoint S0 (written on flush)
  initial_state_indices  [B]                int32   in       per-request pool slot (< 0: padded row, CTA retires)
  k_cache                [pool, H, 32, K]   bf16    in/out   ring: L2-normalized keys
  u_cache                [pool, H, 32, V]   bf16    in/out   ring: correction vectors u = beta*(v - S k̂)
  g_cache                [pool, H, 32, K]   f32     in/out   ring: per-channel cumulative log-decay
  hist_len               [B]                int32   in       filled ring slots P per request
  cache_base             [B]                int32   in       ring window origin
  output                 [B, T, H, V]       bf16    out      draft-token outputs (returned)

  Scalars: scale (float, default 1/sqrt(K)); lower_bound (float < 0, Kimi
  K3 default -5.0); flush_min (int, default W_RING - T + 1 = lazy flush).

Ring tensors are pool-indexed via initial_state_indices and MUST be
zero-initialized at allocation. Legal hist_len at call time: [0, 16];
cache_base: [0, 32).

High-level flow (per CTA = one (request, head))
-----------------------------------------------
  1. Load K/Q/G; L2-normalize k, q; load the g ring window + form w/bdec.
  2. Gate stage: per-channel cumsum; append (normed-k, G) to the ring;
     form khat/ktil/qhat in place.
  3. Grams khat@ktil^T and qhat@ktil^T; WY transform Tmat (triangular solve).
  4. w-scale the khist tile; scores GEMM; bdec-scale the packed tile;
     H GEMM against the streamed S0 + history contraction -> hw.
  5. R = V - hw_k; U = Tmat @ R; y = hw_q + (NegL@Tmat) @ R; append u.
  6. On flush (P >= flush_min): fold the window into S0 and write it back.
"""

import torch
import math
import os
import weakref
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cutlass.cute.experimental  # noqa: F401  # side effect: registers cute.experimental.jit
import cutlass.utils as utils
from cutlass.cute.arch import sync_threads
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T


device = torch.device("cuda:0")

# Problem dimensions. One CTA processes a full V tile per (request, head).
T = 16
K_DIM = 128
V_DIM_C = 128  # full V tile per CTA
EPS = 1e-6
LOG2_E = 1.4426950408889634

io = cutlass.BFloat16
IO_TORCH = torch.bfloat16
_CVT_F32_FROM_H = "cvt.f32.bf16"
_CVT_H2_FROM_F32 = "cvt.rn.bf16x2.f32"
_MMA_M16N8K16_HH_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"

f32 = cutlass.Float32
WARP = 32
THREADS = 128
T_PAD = 16
W_RING = 16  # default history WINDOW rows (per-instance: self._w_ring)
W_BLK = 16  # MMA j-block height — windows are processed in 16-row blocks
# Physical ring depth (Triton-ReplaySSM-compatible circular ring). The live
# window is [base, base+P) mod RING_SLOTS with P <= W_RING; appends land at
# (base+P+s) & RING_MASK — always PAST the window. Cursor commits are
# CALLER-OWNED, outside the launch.
RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1

K_HALF = K_DIM // 2  # 64 — K-half streamed per TMA copy
K_PADDED = K_DIM + 8  # 136 — padded row stride for sK / sQ / sKtil
V_PADDED = V_DIM_C + 8  # 136 — padded row stride for sV / OutStage

TT = T * T
BF_PAD = 24


# ---------------------------------------------------------------------------
# Small inline-PTX helpers (bf16-only forms of the GDN kernel's set, plus the
# per-channel additions: _mul2_bf16x2_f32, _cp_async_f32x4_cg,
# _fold_fma2_bf16x2).
# ---------------------------------------------------------------------------


def _smat_off(row, col):
    e = row * T + col
    return e ^ (
        ((e >> Int32(5)) & Int32(1)) | (((e >> Int32(6)) & Int32(1)) << Int32(3))
    )


def _ldmatrix_x4(smem_tensor, lane_id, row_stride_bytes=BF_PAD * 2, byte_off=0):
    addr = (
        smem_tensor.iterator.toint()
        + (lane_id % 16) * Int32(row_stride_bytes)
        + (lane_id // 16) * Int32(16)
        + Int32(byte_off)
    )
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [addr.ir_value()],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _dot_sq_bf16x2(packed_i32, acc):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [acc.ir_value(), packed_i32.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $2;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " fma.rn.f32 $0, _flo, _flo, $1;"
        " fma.rn.f32 $0, _fhi, _fhi, $0; }",
        "=f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _rsqrt_approx_f32(x):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [x.ir_value()],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _exp2_approx_f32(x):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [x.ir_value()],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _exp_approx_f32(x):
    return _exp2_approx_f32(x * f32(LOG2_E))


def _mul_bf16x2_f32(packed_i32, scalar):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        f" {_CVT_H2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _mul2_bf16x2_f32(packed_i32, s_lo, s_hi):
    """(kda) per-CHANNEL pair scale: unpack a bf16 pair, multiply lo/hi by
    TWO DIFFERENT f32 scalars, repack. The per-channel decay analogue of
    ``_mul_bf16x2_f32`` — used for the khist w-scale and the packed-tile
    bdec column scale, where adjacent K channels carry different factors."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value(), s_lo.ir_value(), s_hi.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $3;"
        f" {_CVT_H2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_bf16x8(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """cp.async.ca, 16 B (8 bf16). Uses .ca for K, Q and G (reuse stream)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.ca.shared.global [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_bf16x8_cg(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """cp.async.cg, 16 B (8 bf16). .cg = bypass L1 (cache only at L2).
    Used for the u ring — single-pass stream, keep L1 for K/Q/V."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.cg.shared.global [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_f32x4_cg(base_addr_i64, f32_elem_offset, smem_addr_i32):
    """(kda) cp.async.cg, 16 B (4 f32). Loads the per-channel g ring window
    (g_cache rows are K f32 wide here, vs GDN's one scalar per row)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            f32_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 4, $2;"
        " cp.async.cg.shared.global [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_commit_group():
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ cp.async.commit_group; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_wait_group_0():
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ cp.async.wait_group 0; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _exit_cta_if_neg(idx_i32):
    """Retire the calling thread iff idx < 0 (padded CUDA-graph row). Must be
    called CTA-uniformly at kernel entry, BEFORE any SMEM/mbarrier/TMA
    issue."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [idx_i32.ir_value()],
        "{ .reg .pred _pexit; setp.lt.s32 _pexit, $1, 0;"
        " @_pexit exit; mov.u32 $0, 0; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _st_global_f32(base_addr_i64, f32_elem_offset, val_f32):
    """STG.32 of one f32 (offset in f32 elements) — the g-ring append."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            f32_elem_offset.ir_value(),
            val_f32.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 4, $1;"
        " st.global.f32 [_a], $3; mov.u32 $0, 0; }",
        "=r,l,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _st_global_bf16(base_addr_i64, bf16_elem_offset, packed_lo16_i32):
    """(kda) STG.16 of one bf16 element (the low 16 bits of $3) at an element
    offset — the per-channel k-ring append (thread == channel)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            packed_lo16_i32.ir_value(),
        ],
        "{ .reg .u64 _a; .reg .b16 _lo, _hi; mad.wide.u32 _a, $2, 2, $1;"
        " mov.b32 {_lo, _hi}, $3;"
        " st.global.b16 [_a], _lo; mov.u32 $0, 0; }",
        "=r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _prefetch_l2_bf16(base_addr_i64, bf16_elem_offset):
    """prefetch.global.L2 of the 128-B line at base + 2*offset (ring rows)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), bf16_elem_offset.ir_value()],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 2, $1;"
        " prefetch.global.L2 [_a]; mov.u32 $0, 0; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _lds_b32(smem_addr_i32):
    """(flush) LDS.32: 4 B (2 bf16) from SMEM. Address must be 4-B aligned."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value()],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _fold_fma2_bf16x2(packed_i32, bdec_lo_f32, bdec_hi_f32, d0_f32, d1_f32):
    """(flush) per-pair state-fold epilogue with PER-CHANNEL bdec: unpack two
    bf16 S0 values (lo = col c, hi = col c+1), return (lo*bdec_lo + d0,
    hi*bdec_hi + d1) as f32 — S_h = e^{G_P[c]}*S0 + D with a single f32 fma
    per element. The two bdec scalars are the per-channel generalization of
    GDN's single bdec."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32(), mlir_T.f32()]),
        [
            packed_i32.ir_value(),
            bdec_lo_f32.ir_value(),
            bdec_hi_f32.ir_value(),
            d0_f32.ir_value(),
            d1_f32.ir_value(),
        ],
        "{ .reg .b32 _wl, _wh; .reg .f32 _fl, _fh;"
        " shl.b32 _wl, $2, 16;"
        " and.b32 _wh, $2, 0xFFFF0000;"
        " mov.b32 _fl, _wl;"
        " mov.b32 _fh, _wh;"
        " fma.rn.f32 $0, _fl, $3, $5; fma.rn.f32 $1, _fh, $4, $6; }",
        "=f,=f,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        f32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        f32(llvm.extractvalue(mlir_T.f32(), r, [1])),
    )


def _ldmatrix_x4_trans(addr_i32):
    """(flush) ldmatrix.x4.trans: A-fragments [m16, k16] for m16n8k16 read
    from a row-major [k16, m16] SMEM tile (the natural u-ring staging)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [addr_i32.ir_value()],
        "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _bar_sync_1_64():
    """Named barrier 1 over 64 threads (warps 2-3): orders the khist cp.async
    wave and w-scale pass before the scores GEMM's cross-warp ldmatrix
    reads, without stalling warps 0-1."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ bar.sync 1, 64; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _r_sub_bf16x2(packed_i32, neg_scale_f32, hw0_f32, hw1_f32):
    """R-pass pair op: unpack two v-values, fma each with neg_scale * hw,
    repack. Called with neg_scale = -1.0 (KDA's decay is baked into hw_k)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            packed_i32.ir_value(),
            neg_scale_f32.ir_value(),
            hw0_f32.ir_value(),
            hw1_f32.ir_value(),
        ],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " fma.rn.f32 _flo, $2, $3, _flo;"
        " fma.rn.f32 _fhi, $2, $4, _fhi;"
        f" {_CVT_H2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_bf16x2_f32(smem_addr_i32, lo_f32, hi_f32):
    """Packed FP32 -> bf16 pair cast + STS.32 to SMEM."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        f" {_CVT_H2_FROM_F32} _v, $3, $2;"
        " st.shared.b32 [$1], _v;"
        " mov.u32 $0, 0; }",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _lds_v4_b32(smem_addr_i32):
    """LDS.128: 16 B (8 bf16) from SMEM. Address must be 16-B aligned."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [smem_addr_i32.ir_value()],
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _st_global_v4_b32(base_addr_i64, bf16_elem_offset, v0, v1, v2, v3):
    """STG.128: 16 B (8 bf16) to global. Offset in bf16 elements, 16-B aligned."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 2, $1;"
        " st.global.v4.b32 [_a], {$3,$4,$5,$6}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _fused_ab_1mma(a_addr, b_addr, c0, c1, c2, c3):
    """ldmatrix.x4 A + ldmatrix.x2.trans B + 1 MMA."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 4),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            a_addr.ir_value(),
            b_addr.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8];"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$9];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3}; }",
        "=f,=f,=f,=f,0,1,2,3,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [2])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [3])),
    )


def _fused_ab_4mma_serial_brow(a_base, b_base, c0, c1, c2, c3):
    """Grams / scores GEMM — 4 sequential (ldmatrix_A + ldmatrix_B + MMA) at
    K-stride 32B, both operands bf16."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 4),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            a_base.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+32];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+32];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+64];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+64];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+96];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+96];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3}; }",
        "=f,=f,=f,=f,0,1,2,3,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [2])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [3])),
    )


def _qtv_4mma(a0, a1, a2, a3, b_base):
    """4 independent (ldmatrix_B_trans + MMA) sharing one A fragment, B
    stride 16 B. Used by the history contraction, y GEMM, U GEMM and fold."""
    zero = cutlass.Float32(0.0)
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 16),
        [zero.ir_value()] * 16
        + [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _b<2>;"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {$32,$33,$34,$35}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+16];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$4,$5,$6,$7}, {$32,$33,$34,$35}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+32];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$8,$9,$10,$11}, {$32,$33,$34,$35}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+48];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$12,$13,$14,$15}, {$32,$33,$34,$35}, {_b0,_b1}, {$12,$13,$14,$15}; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,"
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [i])) for i in range(16)
    )


def _h_gemm_4v(
    a_addr,
    b0_addr,
    b1_addr,
    b2_addr,
    b3_addr,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
):
    """ldmatrix_A.x4 (16x16 of K) + 4× ldmatrix_B.x2 (8x16 of V-rows × K-cols)
    + 4× MMA accumulating into 4 separate C-tiles (one per V-group). B comes
    from the SW128-swizzled state tile; callers MUST apply ``_sw128_xor``."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 16),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            c4.ir_value(),
            c5.ir_value(),
            c6.ir_value(),
            c7.ir_value(),
            c8.ir_value(),
            c9.ir_value(),
            c10.ir_value(),
            c11.ir_value(),
            c12.ir_value(),
            c13.ir_value(),
            c14.ir_value(),
            c15.ir_value(),
            a_addr.ir_value(),
            b0_addr.ir_value(),
            b1_addr.ir_value(),
            b2_addr.ir_value(),
            b3_addr.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$32];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$33];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$34];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$4,$5,$6,$7}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$35];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$8,$9,$10,$11}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$12,$13,$14,$15}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$12,$13,$14,$15}; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,"
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [i])) for i in range(16)
    )


def _wh_write(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, r):
    """Overwrite the 4 wh_acc fragments (4 slots each) from a flat 16-value
    MMA result tuple, as returned by _h_gemm_4v."""
    frags = (wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3)
    for f in range(4):
        for i in range(4):
            frags[f].iterator[i] = r[f * 4 + i]


def _wh_accumulate(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, r):
    """Add a flat 16-value MMA result tuple into the 4 wh_acc fragments (4
    slots each), as returned by _h_gemm_4v / _qtv_4mma."""
    frags = (wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3)
    for f in range(4):
        for i in range(4):
            frags[f].iterator[i] = frags[f].iterator[i] + r[f * 4 + i]


def _sw128_xor(addr_i32):
    """Apply SW128 (cute.make_swizzle(3, 4, 3)) XOR to a logical SMEM byte
    address: phys = L XOR ((L >> 3) & 0x70). Must match
    ``_make_sH_sw128_layout_half``."""
    return addr_i32 ^ ((addr_i32 >> Int32(3)) & Int32(0x70))


def _make_sH_sw128_layout_half():
    """SW128 K-major BF16 layout tiled to (V_DIM_C=128, K_HALF=64)."""
    sw = cute.make_swizzle(3, 4, 3)
    base = cute.make_layout((8, 64), stride=(64, 1))
    atom = cute.make_composed_layout(sw, 0, base)
    return cute.tile_to_shape(atom, (V_DIM_C, K_HALF), order=(1, 0))


class KdaDecodeUCacheFlushKernel:
    """CuTeDSL KDA decode output + u-cache + per-request state flush."""

    def __init__(
        self,
        min_blocks_per_mp=2,
        t_input=4,
        pdl_trigger=False,
        w_ring=16,
        tma_late=True,
    ):
        assert t_input in (4, 8), "KDA ucache kernel supports T in {4, 8} only"
        assert w_ring in (16, 32), "w_ring must be 16 or 32"
        self._min_blocks_per_mp = min_blocks_per_mp
        self._t_input = int(t_input)
        self._n_valid = int(t_input)  # native short-T always (T rows in gmem)
        self._pdl_trigger = bool(pdl_trigger)
        # (w32) deep-window mode: the logical history window spans w_ring
        # rows, processed as NJB 16-row j-blocks (scores GEMM + history
        # contraction get a second, P>16-gated block; the fold's second
        # block is unconditional — its khist rows are zero-written). The
        # 16-row whist staging buffer is REUSED across the two g-ring
        # rounds, so W32 costs no extra whist smem.
        self._w_ring = int(w_ring)
        self._njb = self._w_ring // W_BLK
        # (tma-late) TWO state-load schedules, chosen per batch size (exact
        # CTAs/SM depend on w_ring and T — see the wrapper's _mbp_cap):
        #   late (True):  the g-ring window stages in h_buf's bytes and the
        #     state TMA half-0 is issued at the Step-4 sync — deletes the
        #     dedicated whist tile, raising residency, but exposes
        #     state-load latency, which only pays when there are enough
        #     CTAs to hide it.
        #   early (False): dedicated whist tile, TMA at kernel entry —
        #     lower residency but wins at small batch where SMs are
        #     under-filled and latency cover beats residency.
        self._tma_late = bool(tma_late)

    @cute.experimental.jit
    def __call__(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        lower_bound: cutlass.Float32,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        flush_min: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        op = MmaF16BF16Op(io, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        B_val = gH0idx.layout.shape[0]
        # State TMA: reorder gH0 (pool, HV, V, K) modes to (V, K, HV, pool) so
        # the per-CTA tile is (V_DIM_C, K_HALF); SW128-swizzled SMEM target.
        gH0_vkhp = cute.make_tensor(
            gH0.iterator,
            cute.select(gH0.layout, mode=[2, 3, 1, 0]),
        )
        sH_tma_layout = _make_sH_sw128_layout_half()
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gH0_vkhp,
            sH_tma_layout,
            (V_DIM_C, K_HALF),
        )
        # One CTA per (b, hv) — full V tile per CTA.
        self.kernel(
            gQ,
            gK,
            gV,
            gG,
            gB,
            gAlog,
            gDtbias,
            gH0,
            gH0idx,
            gKC,
            gUC,
            gGC,
            gHlen,
            gBase,
            gOut,
            scale,
            lower_bound,
            tiled_mma,
            HV,
            V_DIM,
            H,
            tma_atom_h,
            tma_tensor_h,
            flush_min,
        ).launch(
            grid=(1, HV, B_val),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        lower_bound: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
        flush_min: cutlass.Int32,
    ):
        # Strides (contiguous native [B, T, ...] layout; T == n_valid rows).
        sq_h = K_DIM
        sq_t = H * K_DIM
        sq_b = self._n_valid * sq_t
        sk_h = K_DIM
        sk_t = H * K_DIM
        sk_b = self._n_valid * sk_t
        sv_hv = V_DIM
        sv_t = HV * V_DIM
        sv_b = self._n_valid * sv_t
        sg_hv = K_DIM
        sg_t = HV * K_DIM
        sg_b = self._n_valid * sg_t
        so_hv = V_DIM
        so_t = HV * V_DIM
        so_b = self._n_valid * so_t
        sb_hv = cutlass.Int32(1)
        sb_t = HV
        sb_b = self._n_valid * HV

        tidx, _, _ = cute.arch.thread_idx()
        # grid is (1, HV, B) — one CTA per (head, request); the leading dim
        # is always 1, so its block index is discarded.
        _, pid_hv, pid_b = cute.arch.block_idx()
        lane_id = tidx & 31
        warp_id = tidx // WARP
        WR = self._w_ring  # logical history window rows (16 or 32)
        NJB = self._njb  # 16-row j-blocks in the window (1 or 2)
        SW_PAD = WR + 8  # sWScores row stride (j runs over the window)

        # Query-head mapping (H == HV for KDA, so i_h == pid_hv; kept generic).
        i_h = pid_hv // (HV // H)
        if const_expr(self._pdl_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        cache_idx = gH0idx.iterator[pid_b]
        # Padded CUDA-graph rows carry cache_idx < 0: the whole CTA retires
        # here, before any SMEM/TMA/ring work.
        _exit_cta_if_neg(cache_idx)

        # Ring addressing + per-request history fill level.
        # k_cache [pool, H, 32, K] bf16 (L2-NORMED k),
        # u_cache [pool, HV, 32, V] bf16,
        # g_cache [pool, HV, 32, K] f32 (per-channel cumulative log-decay).
        P_hist = gHlen.iterator[pid_b]
        ring_base = gBase.iterator[pid_b]
        skc_pool = gKC.layout.stride[0]
        skc_h = gKC.layout.stride[1]
        suc_pool = gUC.layout.stride[0]
        suc_hv = gUC.layout.stride[1]
        sgc_pool = gGC.layout.stride[0]
        sgc_hv = gGC.layout.stride[1]
        cache_idx64 = Int64(cache_idx)
        _kc_pool_e64 = cache_idx64 * skc_pool + Int64(i_h) * skc_h
        _uc_pool_e64 = cache_idx64 * suc_pool + Int64(pid_hv) * suc_hv
        _gc_pool_e64 = cache_idx64 * sgc_pool + Int64(pid_hv) * sgc_hv
        # Ring byte bases with the pool offset absorbed.
        _gGC_base = gGC.iterator.toint() + _gc_pool_e64 * 4
        _gKC_base = gKC.iterator.toint() + _kc_pool_e64 * 2
        _gUC_base = gUC.iterator.toint() + _uc_pool_e64 * 2

        # Warp-3 beta LDG at kernel entry (maximum HBM latency hiding).
        _v7e_b_bf16 = f32(0.0)
        if warp_id == 3 and lane_id < Int32(self._n_valid):
            _v7e_b_bf16 = gB.iterator[
                pid_b * sb_b + lane_id * sb_t + pid_hv * sb_hv
            ].to(f32)
        # Per-head gate constants: A_log scalar (broadcast) and one dt_bias
        # element per K channel (thread == channel), both fp32 tensors.
        _exp_A = _exp_approx_f32(gAlog.iterator[pid_hv].to(f32))
        _dtb = gDtbias.iterator[pid_hv * K_DIM + tidx].to(f32)

        # (perf) L2-prefetch the live khist/u ring rows at kernel entry.
        # Threads [0,64) cover up to 32 k rows (2 halves each); [64,128)
        # cover the u rows.
        _pf_half = (tidx & Int32(1)) * Int32(64)
        if tidx < Int32(64):
            _pf_row = tidx >> 1
            _pf_ring = (ring_base + _pf_row) & Int32(RING_MASK)
            if _pf_row < P_hist:
                _prefetch_l2_bf16(
                    _gKC_base,
                    _pf_ring * K_DIM + _pf_half,
                )
        else:
            _pf_row = (tidx - Int32(64)) >> 1
            _pf_ring = (ring_base + _pf_row) & Int32(RING_MASK)
            if _pf_row < P_hist:
                _prefetch_l2_bf16(
                    _gUC_base,
                    _pf_ring * V_DIM + _pf_half,
                )
        # g ring rows for the w-scale rounds (f32: 4 x 128-B lines per row;
        # offsets doubled because the prefetch helper scales by 2 B).
        _pg_row = tidx >> 2
        if _pg_row < P_hist:
            _prefetch_l2_bf16(
                _gGC_base,
                (
                    ((ring_base + _pg_row) & Int32(RING_MASK)) * K_DIM
                    + (tidx & Int32(3)) * Int32(32)
                )
                * 2,
            )

        smem = utils.SmemAllocator()

        @cute.struct
        class SS:
            # mbarrier for the state-tile TMA load (8B Int64), placed first.
            h_load_mbar: cute.struct.MemRange[Int64, 1]
            # k_buf: K loads -> khat -> khat*e^{cum} (packed k rows) with the
            # packed q rows at [8:8+t); later OutStage; later the fold's u
            # reload stage ([WR, V_DIM_C] rows at w_ring=32).
            k_buf: cute.struct.Align[
                cute.struct.MemRange[io, max(T, self._w_ring) * K_PADDED], 128
            ]
            # qv_buf tenants: Q loads -> qhat -> qhat*e^{cum}; khist tile
            # (w-scaled, [WR, K]); u tile ([WR, V_DIM_C]); V tile; fold khist
            # restore.
            qv_buf: cute.struct.Align[
                cute.struct.MemRange[io, max(T, self._w_ring) * K_PADDED], 128
            ]
            # (kda) ktil = khat * e^{-cum}, the Gram B operand. 8 rows: with
            # T <= 8 the Grams only need N = 8 columns (Phase 2 masks
            # everything at r >= t or c >= t, t <= 8), so ktil rows [8:16)
            # would be dead weight. Rows [t:8) are explicitly zeroed.
            ktil_buf: cute.struct.Align[cute.struct.MemRange[io, 8 * K_PADDED], 128]
            # (kda) raw per-channel gate tile [T, K] bf16 (row-contiguous).
            # (smem diet) only t_input rows are ever loaded/read (the gate
            # stage is a constexpr loop over n_valid), so size the tile to T
            # actual rows: 1 KB @T=4 vs 4 KB at the full 16 — the saving is
            # what lifts residency from 4 to 5 CTAs/SM.
            g_buf: cute.struct.Align[
                cute.struct.MemRange[io, self._t_input * K_DIM], 128
            ]
            # State tile (V_DIM_C x K_HALF=64, SW128), single-buffered across
            # the 2 TMA half-loads via the mbarrier ping-pong.
            h_buf: cute.struct.Align[cute.struct.MemRange[io, V_DIM_C * K_HALF], 128]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            # TRANSPOSED scores tile [T_PAD, WR (+8 pad)] bf16 — the
            # A-operand of the MMA history contraction (w is baked into
            # khist, not here); at w_ring=32 the j axis spans two 16-col
            # blocks. Tenant #2: QT (bf16) staged by the y-GEMM prep.
            wscores_bf: cute.struct.Align[
                cute.struct.MemRange[io, T_PAD * (self._w_ring + 8)], 128
            ]
            # (tma-early) dedicated 16-row g-ring staging block; collapsed
            # to 8 elements in the tma-late schedule, where the block is a
            # TENANT of h_buf instead.
            whist_f32: cute.struct.Align[
                cute.struct.MemRange[f32, (W_BLK * K_DIM) if not self._tma_late else 8],
                128,
            ]
            # (kda) [0:128) = G_P[c]; [128:256) = bdec[c] = e^{G_P[c]}.
            # NOTE: there is deliberately NO smem tile for the g ring
            # window — G_j rows are read straight from (L2-prefetched)
            # gmem inside the khist w-scale pass, one visit per element.
            # A [W_RING, K] f32 staging tile (8 KB at W16, 16 KB at W32)
            # bought nothing and was the residency bottleneck.
            gvec_f32: cute.struct.Align[cute.struct.MemRange[f32, 2 * K_DIM], 128]

        st = smem.allocate(SS)
        sK = st.k_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        sQ = st.qv_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        sKtil = st.ktil_buf.get_tensor(
            cute.make_layout((8, K_PADDED), stride=(K_PADDED, 1))
        )
        sG = st.g_buf.get_tensor(
            cute.make_layout((self._t_input, K_DIM), stride=(K_DIM, 1))
        )
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sV = st.qv_buf.get_tensor(cute.make_layout((T, V_PADDED), stride=(V_PADDED, 1)))
        sTmat = st.tmat_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sBeta = st.beta.get_tensor(cute.make_layout((WARP,)))
        sMat = st.mat_fp32.get_tensor(cute.make_layout((T, T), stride=(T, 1)))
        sNegL = st.scratch_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sWScores = st.wscores_bf.get_tensor(
            cute.make_layout((T_PAD, WR), stride=(SW_PAD, 1))
        )
        # (tma-late) the g-ring staging block is a TENANT of h_buf's first
        # 8 KB: the state TMA half-0 is issued only at the Step-4 sync —
        # AFTER the w-scale pass consumed the g window — so the lifetimes
        # never overlap and the dedicated whist tile is deleted, raising
        # residency (exact CTAs/SM: see the wrapper's _mbp_cap).
        # (tma-early) dedicated tile; TMA at entry; lower residency.
        if const_expr(self._tma_late):
            _sH_flat_io = st.h_buf.get_tensor(cute.make_layout((V_DIM_C * K_HALF,)))
            sWhist = cute.recast_tensor(_sH_flat_io, f32)
        else:
            sWhist = st.whist_f32.get_tensor(cute.make_layout((W_BLK * K_DIM,)))
        _sWhist_base = sWhist.iterator.toint()
        sGvec = st.gvec_f32.get_tensor(cute.make_layout((2 * K_DIM,)))

        # mbarrier init for the state-tile TMA load.
        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()
        sync_threads()

        # Partition the TMA tensor for this CTA's (cache_idx, pid_hv).
        # flat_divide with (V_DIM_C, K_HALF) yields modes (V, KH, V_REST=1,
        # K_REST, HV, pool).
        gH_tiled = cute.flat_divide(tma_tensor_h, (V_DIM_C, K_HALF))
        gH_slice0 = gH_tiled[None, None, 0, 0, pid_hv, cache_idx]
        gH_slice1 = gH_tiled[None, None, 0, 1, pid_hv, cache_idx]
        gH_grp0 = cute.group_modes(gH_slice0, 0, 2)
        gH_grp1 = cute.group_modes(gH_slice1, 0, 2)
        sH_grp = cute.group_modes(sH, 0, 2)
        tHsH0, tHgH0 = cpasync.tma_partition(
            tma_atom_h,
            0,
            cute.make_layout(1),
            sH_grp,
            gH_grp0,
        )
        tHsH1, tHgH1 = cpasync.tma_partition(
            tma_atom_h,
            0,
            cute.make_layout(1),
            sH_grp,
            gH_grp1,
        )

        thr_mma = tiled_mma.get_slice(lane_id)
        tCsC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((T, 8)))
        acc = cute.make_fragment_like(tCsC)
        _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)

        EPT_TT = TT // THREADS

        # ============================================================
        # cp.async stage 1: K + Q + G (bf16, .ca) and the g ring window
        # (f32, .cg) — one commit group; the single wait below drains all.
        # ============================================================
        k_base = pid_b * sk_b + i_h * sk_h
        q_base = pid_b * sq_b + i_h * sq_h
        g_base = pid_b * sg_b + pid_hv * sg_hv
        _gK_gbase = gK.iterator.toint()
        _gQ_gbase = gQ.iterator.toint()
        _gG_gbase = gG.iterator.toint()
        _sK_i32 = cute.recast_tensor(sK, cutlass.Int32)
        _sQ_i32 = cute.recast_tensor(sQ, cutlass.Int32)
        _kpad_i32 = K_PADDED // 2
        _sK_base_async = sK.iterator.toint()
        _sQ_base_async = sQ.iterator.toint()
        _sG_base_async = sG.iterator.toint()
        for i in cutlass.range_constexpr(T * K_DIM // (THREADS * 8)):
            _kq_group = tidx + i * THREADS
            _kq_row = _kq_group // Int32(K_DIM // 8)
            _kq_col = (_kq_group % Int32(K_DIM // 8)) * Int32(8)
            _smem_byte_off = _kq_row * Int32(K_PADDED * 2) + _kq_col * Int32(2)
            _g_smem_byte_off = _kq_row * Int32(K_DIM * 2) + _kq_col * Int32(2)
            # Native short-T: the gmem tensors hold only n_valid rows; skip
            # rows >= n_valid (OOB). SMEM tails are zeroed after the wait; sG
            # tail rows are skipped by the constexpr gate-stage loop.
            if const_expr(self._n_valid < T):
                if _kq_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gK_gbase,
                        k_base + _kq_row * sk_t + _kq_col,
                        _sK_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gQ_gbase,
                        q_base + _kq_row * sq_t + _kq_col,
                        _sQ_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gG_gbase,
                        g_base + _kq_row * sg_t + _kq_col,
                        _sG_base_async + _g_smem_byte_off,
                    )
            else:
                _cp_async_bf16x8(
                    _gK_gbase,
                    k_base + _kq_row * sk_t + _kq_col,
                    _sK_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gQ_gbase,
                    q_base + _kq_row * sq_t + _kq_col,
                    _sQ_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gG_gbase,
                    g_base + _kq_row * sg_t + _kq_col,
                    _sG_base_async + _g_smem_byte_off,
                )
        # g ring round 1: rows [0, min(P, 16)) into the 16-row staging
        # block (ring-rotated at load). Round 2 (w_ring=32, rows [16, 32))
        # is issued mid-kernel by warps 2-3 into the SAME buffer.
        for _gh in cutlass.range_constexpr(W_BLK * K_DIM // (THREADS * 4)):
            _gh_chunk = tidx + _gh * THREADS
            _gh_row = _gh_chunk // Int32(K_DIM // 4)
            _gh_col = (_gh_chunk % Int32(K_DIM // 4)) * Int32(4)
            if _gh_row < P_hist:
                _cp_async_f32x4_cg(
                    _gGC_base,
                    ((ring_base + _gh_row) & Int32(RING_MASK)) * K_DIM + _gh_col,
                    _sWhist_base + (_gh_row * Int32(K_DIM) + _gh_col) * Int32(4),
                )
        _cp_async_commit_group()  # group 0 = K + Q + G + g-ring round 1

        # (tma-early) issue the FIRST state-tile half at entry — maximal
        # latency cover for the under-filled small-batch regime. The
        # tma-late schedule issues it at the Step-4 sync instead.
        if const_expr(not self._tma_late):
            if warp_id == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_h_ptr,
                        V_DIM_C * K_HALF * 2,
                    )
                cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # warp 3: beta (per-token scalar; sigmoid of the logit). KDA's decay
        # is per-channel and handled in the gate stage below, so warp 3 has
        # no gamma cumsum work here.
        # ============================================================
        if warp_id == 3:
            beta_val = f32(0.0)
            if lane_id < Int32(self._n_valid):
                beta_val = f32(1.0) / (
                    f32(1.0) + _exp_approx_f32(f32(0.0) - _v7e_b_bf16)
                )
            if lane_id < T:
                sBeta.iterator[lane_id] = beta_val

        # ============================================================
        # Wait for K+Q+G+ghist (group 0); zero the sK/sQ tail rows.
        # ============================================================
        _cp_async_wait_group_0()
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, T):
                sK.iterator[_zr * K_PADDED + tidx] = io(0.0)
                sQ.iterator[_zr * K_PADDED + tidx] = io(0.0)
        sync_threads()

        # ============================================================
        # L2 norm for K (warp 0) and Q (warp 2) — T <= 8 always here, so
        # warps 1 and 3 are free. Rows [t:8) are zeros (harmless no-op).
        # ============================================================
        if warp_id == Int32(0):
            norm_row = lane_id // 4
            norm_quarter = lane_id % 4
            _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
            partial = f32(0.0)
            if norm_row < Int32(self._t_input):
                for c in cutlass.range_constexpr(16):
                    packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
            for d in [1, 2]:
                other = cute.arch.shuffle_sync(
                    partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                )
                partial = partial + other
            inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
            if norm_row < Int32(self._t_input):
                for c in cutlass.range_constexpr(16):
                    _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        if warp_id == Int32(2):
            norm_row = lane_id // 4
            norm_quarter = lane_id % 4
            _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
            partial = f32(0.0)
            if norm_row < Int32(self._t_input):
                for c in cutlass.range_constexpr(16):
                    packed = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
            for d in [1, 2]:
                other = cute.arch.shuffle_sync(
                    partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                )
                partial = partial + other
            inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
            inv_norm = inv_norm * scale
            if norm_row < Int32(self._t_input):
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        sync_threads()

        # ============================================================
        # (kda) W STAGE — thread tidx owns K-channel c = tidx. Publish
        #   sGvec[c]        = G_P[c]           (0 if P == 0)
        #   sGvec[128 + c]  = e^{G_P[c]}       (bdec; 1 if P == 0)
        # G_P is the LAST LIVE ring row, read straight from gmem: the row
        # index is & RING_MASK'd, so even out-of-contract hist_len (the
        # host-unvalidated restart=False path) yields wrong-but-safe values
        # — same failure class as the GDN sibling's lane-shuffle anchor.
        # The clamp keeps the anchor inside the logical window.
        # ============================================================
        _p_src = P_hist - 1 if P_hist > 0 else Int32(0)
        _p_src = _p_src if _p_src < Int32(WR) else Int32(WR - 1)
        _gp_c = gGC.iterator[
            _gc_pool_e64
            + Int64(((ring_base + _p_src) & Int32(RING_MASK)) * K_DIM + tidx)
        ]
        _gp_c = _gp_c if P_hist > 0 else f32(0.0)
        _bdec_c = _exp_approx_f32(_gp_c) if P_hist > 0 else f32(1.0)
        sGvec.iterator[tidx] = _gp_c
        sGvec.iterator[K_DIM + tidx] = _bdec_c

        # ============================================================
        # (kda) GATE STAGE — thread tidx owns K-channel c = tidx. Serial
        # per-channel cumsum over the T token rows; per row:
        #   - append the L2-normed k (still in sK) and the cumulative
        #     log-decay G to the ring at (base+P+s) & RING_MASK — PAST the
        #     live window, so a sibling's fold never reads these rows.
        #     Flush rows append G with LOCAL decay (the fold absorbs
        #     e^{G_P} into the checkpoint);
        #   - then form in place: sK <- khat*e^{cum} (packed k rows),
        #     sKtil <- khat*e^{-cum}, sQ <- qhat*e^{cum}, and dual-store the
        #     scaled q row into packed row [8+s] of sK (Gram/H-GEMM A tile).
        # Row-wise SMEM access is conflict-free (128 threads x 128 channels).
        # ============================================================
        if const_expr(self._n_valid < 8):
            for _zt in cutlass.range_constexpr(self._n_valid, 8):
                sKtil.iterator[_zt * K_PADDED + tidx] = io(0.0)
        _g_app_base = f32(0.0) if P_hist >= flush_min else _gp_c
        _cum = f32(0.0)
        for t in cutlass.range_constexpr(self._n_valid):
            _graw = sG.iterator[t * K_DIM + tidx].to(f32)
            _x = _graw + _dtb
            _sig = f32(1.0) / (f32(1.0) + _exp_approx_f32(f32(0.0) - _exp_A * _x))
            _glog = lower_bound * _sig
            _cum = _cum + _glog
            _ring_row = (ring_base + P_hist + Int32(t)) & Int32(RING_MASK)
            # k-ring append: the L2-normed, UNDECAYED khat (sK not yet
            # rescaled at this point of the iteration).
            _koff = t * K_PADDED + tidx
            gKC.iterator[_kc_pool_e64 + Int64(_ring_row * K_DIM + tidx)] = sK.iterator[
                _koff
            ]
            # g-ring append (per-channel cumulative log-decay).
            _st_global_f32(
                _gGC_base,
                _ring_row * K_DIM + tidx,
                _g_app_base + _cum,
            )
            _ep = _exp_approx_f32(_cum)
            _en = _exp_approx_f32(f32(0.0) - _cum)
            _kval = sK.iterator[_koff].to(f32)
            _qval = sQ.iterator[_koff].to(f32)
            _qcum = (_qval * _ep).to(io)
            sK.iterator[_koff] = (_kval * _ep).to(io)
            sKtil.iterator[_koff] = (_kval * _en).to(io)
            sQ.iterator[_koff] = _qcum
            sK.iterator[(8 + t) * K_PADDED + tidx] = _qcum
        sync_threads()

        # ============================================================
        # KKT (warp 0): khat @ ktil^T -> sMat[:, 0:8]  ||  QKT (warp 2):
        # qhat @ ktil^T -> sNegL[:, 0:8]. N = 8 columns are SUFFICIENT at
        # T <= 8: Phase 2 reads only (r < t, c < t) and zero-writes every
        # other sNegL/sTmat entry itself, so Gram columns [8:16) (dead
        # masked garbage in the 16-wide version) are simply never computed
        # — half the Gram MMAs, and warps 1/3 stay free. The per-channel
        # decay is already folded into the operands (no post-GEMM factors).
        # ============================================================
        acc.fill(f32(0.0))
        _sK_int = sK.iterator.toint()
        _sQ_int = sQ.iterator.toint()
        _sKtil_int = sKtil.iterator.toint()
        _rs_kpad = Int32(K_PADDED * 2)
        _lane_mod16 = lane_id & Int32(15)
        _lane_hi = (lane_id >> Int32(4)) * Int32(16)
        _lane_mod8 = lane_id % Int32(8)
        _lane_b_col = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        for kk_group in cutlass.range_constexpr(K_DIM // 16 // 4):
            k_group_off = kk_group * 4 * 16 * Int32(2)
            if warp_id == Int32(0):
                _a_base = _sK_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sKtil_int + _lane_mod8 * _rs_kpad + k_group_off + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
            if warp_id == Int32(2):
                _a_base = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sKtil_int + _lane_mod8 * _rs_kpad + k_group_off + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2
        if warp_id == Int32(0):
            sMat.iterator[_smat_off(_r0, _c0)] = acc.iterator[0]
            sMat.iterator[_smat_off(_r0, _c0 + 1)] = acc.iterator[1]
            sMat.iterator[_smat_off(_r0 + 8, _c0)] = acc.iterator[2]
            sMat.iterator[_smat_off(_r0 + 8, _c0 + 1)] = acc.iterator[3]
        if warp_id == Int32(2):
            sNegL.iterator[_r0 * BF_PAD + _c0] = acc.iterator[0].to(io)
            sNegL.iterator[_r0 * BF_PAD + _c0 + 1] = acc.iterator[1].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + _c0] = acc.iterator[2].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + _c0 + 1] = acc.iterator[3].to(io)
        sync_threads()

        # ============================================================
        # tenant #2 of qv_buf: khist tile [W_RING, K_PADDED] via cp.async.ca.
        # sQ is dead after the Grams above. Issued AND awaited only by warps
        # 2-3 (the tile's sole consumers). Rows >= P stay unread (the w-scale
        # pass below zero-writes them).
        # ============================================================
        _kc_rd_base = Int32(0)
        if warp_id >= 2:
            for _kh in cutlass.range_constexpr(self._w_ring * K_DIM // (64 * 8)):
                _kh_group = (tidx - Int32(64)) + _kh * Int32(64)
                _kh_row = _kh_group // Int32(K_DIM // 8)
                _kh_col = (_kh_group % Int32(K_DIM // 8)) * Int32(8)
                if _kh_row < P_hist:
                    # ring-rotate at load: gmem row (base+j)&mask lands in
                    # smem/logical row j.
                    _cp_async_bf16x8(
                        _gKC_base,
                        _kc_rd_base
                        + ((ring_base + _kh_row) & Int32(RING_MASK)) * K_DIM
                        + _kh_col,
                        _sQ_base_async
                        + _kh_row * Int32(K_PADDED * 2)
                        + _kh_col * Int32(2),
                    )
            _cp_async_commit_group()

        # ============================================================
        # PHASE 2: form the pre-inverse M and the masked NegL. KDA: the
        # decay factors are already inside sMat/sNegL (operand folding), so
        # this is pure masking + beta scaling — no exp() here.
        # ============================================================
        for idx in cutlass.range_constexpr(EPT_TT):
            flat = tidx + idx * THREADS
            r = flat // T
            c = flat % T
            if r < Int32(self._t_input) and c < Int32(self._t_input):
                qkt = sNegL.iterator[r * BF_PAD + c].to(f32)
                sNegL.iterator[r * BF_PAD + c] = qkt.to(io) if r >= c else io(0.0)
                kkt_val = sMat.iterator[_smat_off(r, c)]
                negL_val = (
                    (f32(0.0) - sBeta.iterator[r] * kkt_val) if r > c else f32(0.0)
                )
                sTmat.iterator[r * BF_PAD + c] = negL_val.to(io)
            else:
                sNegL.iterator[r * BF_PAD + c] = io(0.0)
                sTmat.iterator[r * BF_PAD + c] = io(0.0)
        sync_threads()

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2

        # ============================================================
        # BLOCK INVERSE (T <= 8 path: only T00 is real work; T11 = diag(β1))
        # running on warps 0-1, CONCURRENT with the khist w-scale + scores
        # GEMM on warps 2-3.
        # ============================================================
        if warp_id >= 2 and P_hist > 0:
            # own-wave wait + named barrier: the khist tile AND g-ring round
            # 1 are fully landed for both warps 2 and 3 before the w-scale
            # pass RMWs the tile.
            _cp_async_wait_group_0()
            _bar_sync_1_64()
            _sc_r0 = lane_id // 4
            _sc_c0 = (lane_id & 3) * 2
            _sc_col_off = (warp_id - Int32(2)) * Int32(8)
            for _jb in cutlass.range_constexpr(NJB):
                # --- w-scale this j-block of the khist tile IN PLACE: row
                # j = 16*_jb + local, channel c gets khist * w with
                # w = exp(G_P[c] - G_j[c]) from the STAGED g block. Rows
                # >= P are zero-WRITTEN (their SMEM bytes are stale and the
                # scores GEMM + fold read the full window).
                # Round 2 runs ONLY when the window actually reaches it
                # (P > 16, CTA-uniform): at P <= 16 there is no consumer —
                # scores/contraction block 2 are P>16-gated and the fold
                # (which does read khist rows [16, 32)) requires
                # P >= flush_min = RING-2T+1 > 16 — so shallow rows skip the
                # entire round-2 pipeline (the deep window's iso-cost tax
                # collapses to the residency difference). ---
                if const_expr(_jb == 0):
                    _blk_run = warp_id >= 0  # always true; keeps types even
                else:
                    _blk_run = P_hist > Int32(W_BLK)
                if _blk_run:
                    for _ws in cutlass.range_constexpr(W_BLK * (K_DIM // 2) // 64):
                        _ws_g = (tidx - Int32(64)) + _ws * Int32(64)
                        _ws_lr = _ws_g // Int32(K_DIM // 2)
                        _ws_r = Int32(_jb * W_BLK) + _ws_lr
                        _ws_c = _ws_g % Int32(K_DIM // 2)
                        _w0 = _exp_approx_f32(
                            sGvec.iterator[_ws_c * 2]
                            - sWhist.iterator[_ws_lr * Int32(K_DIM) + _ws_c * 2]
                        )
                        _w1 = _exp_approx_f32(
                            sGvec.iterator[_ws_c * 2 + 1]
                            - sWhist.iterator[_ws_lr * Int32(K_DIM) + _ws_c * 2 + 1]
                        )
                        _kh_pair = _sQ_i32.iterator[_ws_r * _kpad_i32 + _ws_c]
                        _scaled = _mul2_bf16x2_f32(_kh_pair, _w0, _w1)
                        _sQ_i32.iterator[_ws_r * _kpad_i32 + _ws_c] = (
                            _scaled if _ws_r < P_hist else Int32(0)
                        )
                if const_expr(NJB == 2 and _jb == 0):
                    # issue g-ring round 2 (rows [16, 32)) into the SAME
                    # staging block — round 1 is fully consumed above; the
                    # transfer overlaps this round's scores GEMM and is
                    # awaited (with a publish barrier) before round 2's
                    # w-scale. The predicate on each load already skips
                    # rows >= P; nothing to issue at P <= 16.
                    for _g2 in cutlass.range_constexpr(W_BLK * K_DIM // (64 * 4)):
                        _g2_chunk = (tidx - Int32(64)) + _g2 * Int32(64)
                        _g2_row = _g2_chunk // Int32(K_DIM // 4)
                        _g2_col = (_g2_chunk % Int32(K_DIM // 4)) * Int32(4)
                        if Int32(W_BLK) + _g2_row < P_hist:
                            _cp_async_f32x4_cg(
                                _gGC_base,
                                (
                                    (ring_base + Int32(W_BLK) + _g2_row)
                                    & Int32(RING_MASK)
                                )
                                * K_DIM
                                + _g2_col,
                                _sWhist_base
                                + (_g2_row * Int32(K_DIM) + _g2_col) * Int32(4),
                            )
                    _cp_async_commit_group()
                _bar_sync_1_64()
                # --- scores GEMM over this j-block's khist rows:
                # scores[j, col] = (w_j (.) khist_j) . packed_col, staged
                # TRANSPOSED at j-column 16*_jb + local of sWScores. Block 2
                # is skipped (warp-uniformly) when the window fits block 1;
                # its khist rows are all zeros then anyway. ---
                if const_expr(_jb == 0):
                    _sc_run = warp_id >= 0  # always true, keeps types even
                else:
                    _sc_run = P_hist > Int32(W_BLK)
                if _sc_run:
                    acc.fill(f32(0.0))
                    for _sc_g in cutlass.range_constexpr(K_DIM // 16 // 4):
                        _sc_k_off = _sc_g * 4 * 16 * Int32(2)
                        _sc_a = (
                            _sQ_int
                            + Int32(_jb * W_BLK) * _rs_kpad
                            + _lane_mod16 * _rs_kpad
                            + _lane_hi
                            + _sc_k_off
                        )
                        _sc_b = (
                            _sK_int
                            + (_sc_col_off + _lane_mod8) * _rs_kpad
                            + _sc_k_off
                            + _lane_b_col
                        )
                        (
                            acc.iterator[0],
                            acc.iterator[1],
                            acc.iterator[2],
                            acc.iterator[3],
                        ) = _fused_ab_4mma_serial_brow(
                            _sc_a,
                            _sc_b,
                            acc.iterator[0],
                            acc.iterator[1],
                            acc.iterator[2],
                            acc.iterator[3],
                        )
                    _sj = Int32(_jb * W_BLK) + _sc_r0
                    sWScores.iterator[(_sc_col_off + _sc_c0) * SW_PAD + _sj] = (
                        acc.iterator[0]
                    ).to(io)
                    sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * SW_PAD + _sj] = (
                        acc.iterator[1]
                    ).to(io)
                    sWScores.iterator[(_sc_col_off + _sc_c0) * SW_PAD + _sj + 8] = (
                        acc.iterator[2]
                    ).to(io)
                    sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * SW_PAD + _sj + 8] = (
                        acc.iterator[3]
                    ).to(io)
                if const_expr(NJB == 2 and _jb == 0):
                    if P_hist > Int32(W_BLK):
                        # drain round 2's g block + publish across warps 2<->3
                        _cp_async_wait_group_0()
                        _bar_sync_1_64()
        # === T<=8 PATH: only T00 (warp 0) is real work; T11 = diag(β1) ===
        if warp_id == Int32(0):
            if lane_id < Int32(8):
                _col = lane_id
                _x_t00 = [None] * 8
                _x_t00[0] = sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                if const_expr(self._t_input <= 4):
                    for _r in cutlass.range_constexpr(1, 4):
                        _accum = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                            _accum = _accum + _m_rk * _x_t00[_k]
                        _x_t00[_r] = _accum
                    for _r in cutlass.range_constexpr(4, 8):
                        _x_t00[_r] = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                else:
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                            _accum = _accum + _m_rk * _x_t00[_k]
                        _x_t00[_r] = _accum
                for _r in cutlass.range_constexpr(8):
                    sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
        if warp_id == Int32(1):
            # T11 = diag(β1); T10 = 0 (rows/cols >= 8 are zero at T <= 8).
            if lane_id < Int32(8):
                _col = lane_id
                for _r in cutlass.range_constexpr(8):
                    _v = (
                        sBeta.iterator[Int32(8 + _r)] if _col == Int32(_r) else f32(0.0)
                    )
                    sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _v
                for _r in cutlass.range_constexpr(8):
                    sMat.iterator[_smat_off(8 + _r, _col)] = f32(0.0)
        sync_threads()

        # === Stage final Tmat (bf16) to sTmat, zero top-right ===
        _flat0_v11 = tidx
        _flat1_v11 = tidx + Int32(THREADS)
        _r0_v11 = _flat0_v11 // Int32(T)
        _c0_v11 = _flat0_v11 % Int32(T)
        _r1_v11 = _flat1_v11 // Int32(T)
        _c1_v11 = _flat1_v11 % Int32(T)
        _v0_v11 = sMat.iterator[_smat_off(_r0_v11, _c0_v11)]
        _v1_v11 = sMat.iterator[_smat_off(_r1_v11, _c1_v11)]
        _out0_v11 = (
            io(0.0) if (_r0_v11 < Int32(8) and _c0_v11 >= Int32(8)) else _v0_v11.to(io)
        )
        _out1_v11 = (
            io(0.0) if (_r1_v11 < Int32(8) and _c1_v11 >= Int32(8)) else _v1_v11.to(io)
        )
        sTmat.iterator[_r0_v11 * BF_PAD + _c0_v11] = _out0_v11
        sTmat.iterator[_r1_v11 * BF_PAD + _c1_v11] = _out1_v11
        sync_threads()

        # ============================================================
        # (tma-late) issue the FIRST state-tile half (K=0..63) INTO h_buf —
        # legal only now: the sync above orders every w-scale read of the
        # g-window tenant occupying these bytes. The load overlaps the
        # khist snapshot, the u-tile cp.async and the bdec pass; the extra
        # resident CTAs bought by deleting the whist tile hide the rest.
        # ============================================================
        if const_expr(self._tma_late):
            if warp_id == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_h_ptr,
                        V_DIM_C * K_HALF * 2,
                    )
                cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # (flush) khist REGISTER SNAPSHOT — flushing CTAs keep the whole
        # (w-SCALED, rows >= P zeroed) khist tile in registers across the
        # qv_buf tenant switch: the tail fold needs khist AND the u tile
        # simultaneously. The w-scaling being baked in here is exactly what
        # the fold wants: D = u^T @ (w ⊙ khist), so the fold has NO
        # separate w pass (GDN scaled the u side instead).
        # ============================================================
        _khs0 = Int32(0)
        _khs1 = Int32(0)
        _khs2 = Int32(0)
        _khs3 = Int32(0)
        _khs4 = Int32(0)
        _khs5 = Int32(0)
        _khs6 = Int32(0)
        _khs7 = Int32(0)
        # (w32) the wider window needs 16 i32/thread (32 rows x 64 i32 over
        # 128 threads; thread -> row tidx//4, i32 cols (tidx%4)*16..+16).
        if const_expr(WR == 32):
            _khs8 = Int32(0)
            _khs9 = Int32(0)
            _khs10 = Int32(0)
            _khs11 = Int32(0)
            _khs12 = Int32(0)
            _khs13 = Int32(0)
            _khs14 = Int32(0)
            _khs15 = Int32(0)
        if P_hist >= flush_min:
            if const_expr(WR == 16):
                _khsnap_off = (tidx // Int32(8)) * Int32(K_PADDED // 2) + (
                    tidx % Int32(8)
                ) * Int32(8)
                _khs0 = _sQ_i32.iterator[_khsnap_off + 0]
                _khs1 = _sQ_i32.iterator[_khsnap_off + 1]
                _khs2 = _sQ_i32.iterator[_khsnap_off + 2]
                _khs3 = _sQ_i32.iterator[_khsnap_off + 3]
                _khs4 = _sQ_i32.iterator[_khsnap_off + 4]
                _khs5 = _sQ_i32.iterator[_khsnap_off + 5]
                _khs6 = _sQ_i32.iterator[_khsnap_off + 6]
                _khs7 = _sQ_i32.iterator[_khsnap_off + 7]
            else:
                _khsnap_off = (tidx // Int32(4)) * Int32(K_PADDED // 2) + (
                    tidx % Int32(4)
                ) * Int32(16)
                _khs0 = _sQ_i32.iterator[_khsnap_off + 0]
                _khs1 = _sQ_i32.iterator[_khsnap_off + 1]
                _khs2 = _sQ_i32.iterator[_khsnap_off + 2]
                _khs3 = _sQ_i32.iterator[_khsnap_off + 3]
                _khs4 = _sQ_i32.iterator[_khsnap_off + 4]
                _khs5 = _sQ_i32.iterator[_khsnap_off + 5]
                _khs6 = _sQ_i32.iterator[_khsnap_off + 6]
                _khs7 = _sQ_i32.iterator[_khsnap_off + 7]
                _khs8 = _sQ_i32.iterator[_khsnap_off + 8]
                _khs9 = _sQ_i32.iterator[_khsnap_off + 9]
                _khs10 = _sQ_i32.iterator[_khsnap_off + 10]
                _khs11 = _sQ_i32.iterator[_khsnap_off + 11]
                _khs12 = _sQ_i32.iterator[_khsnap_off + 12]
                _khs13 = _sQ_i32.iterator[_khsnap_off + 13]
                _khs14 = _sQ_i32.iterator[_khsnap_off + 14]
                _khs15 = _sQ_i32.iterator[_khsnap_off + 15]
            sync_threads()

        # ============================================================
        # tenant #3 of qv_buf: the u tile [W_RING, V_DIM_C] bf16 via
        # cp.async.cg (staged at the V_PADDED row stride). Landing is
        # awaited in the TMA half-1 shadow.
        # ============================================================
        _uc_base = Int32(0)
        _sV_base_async_u = sV.iterator.toint()
        for _uh in cutlass.range_constexpr(self._w_ring * V_DIM_C // (THREADS * 8)):
            _uh_group = tidx + _uh * THREADS
            _uh_row = _uh_group // Int32(V_DIM_C // 8)
            _uh_col = (_uh_group % Int32(V_DIM_C // 8)) * Int32(8)
            if _uh_row < P_hist:
                _cp_async_bf16x8_cg(
                    _gUC_base,
                    _uc_base
                    + ((ring_base + _uh_row) & Int32(RING_MASK)) * V_DIM
                    + _uh_col,
                    _sV_base_async_u
                    + _uh_row * Int32(V_PADDED * 2)
                    + _uh_col * Int32(2),
                )
        _cp_async_commit_group()

        # ============================================================
        # (kda) bdec fold: scale the packed A-tile rows (k rows [0:t), q
        # rows [8:8+t)) PER CHANNEL by bdec[c] = e^{G_P[c]} so the H GEMM
        # directly yields the S0 term of S_h·x. Skipped at P=0 (bdec = 1).
        # Runs strictly AFTER the scores GEMM consumed the un-bdec'd packed
        # tile (the Step-4 sync above orders it).
        # ============================================================
        if P_hist > 0:
            for _bs in cutlass.range_constexpr(
                2 * self._t_input * (K_DIM // 2) // THREADS
            ):
                _bs_idx = tidx + _bs * THREADS
                _bs_rr = _bs_idx // Int32(K_DIM // 2)
                _bs_col = _bs_idx % Int32(K_DIM // 2)
                _bs_row = (
                    _bs_rr
                    if _bs_rr < Int32(self._t_input)
                    else Int32(8 - self._t_input) + _bs_rr
                )
                _bd0 = sGvec.iterator[K_DIM + _bs_col * 2]
                _bd1 = sGvec.iterator[K_DIM + _bs_col * 2 + 1]
                _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col] = _mul2_bf16x2_f32(
                    _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col], _bd0, _bd1
                )

        # ============================================================
        # Wait for the state tile half-0.
        # ============================================================
        cute.arch.mbarrier_wait(mbar_h_ptr, 0)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM: WH[16, 128] = A[16, 128] @ H^T, A = the packed [k; q] tile.
        # 4 warps x 4 V-groups (8 rows each) x 8 K-tiles (16 K each).
        # ============================================================
        wh_acc_0 = cute.make_fragment_like(tCsC)
        wh_acc_0.fill(f32(0.0))
        wh_acc_1 = cute.make_fragment_like(tCsC)
        wh_acc_1.fill(f32(0.0))
        wh_acc_2 = cute.make_fragment_like(tCsC)
        wh_acc_2.fill(f32(0.0))
        wh_acc_3 = cute.make_fragment_like(tCsC)
        wh_acc_3.fill(f32(0.0))

        _sK_base_vl = sK.iterator.toint()
        _sH_base_vl = sH.iterator.toint()
        _rs_a = Int32(K_PADDED * 2)
        _rs_b = Int32(K_HALF * 2)

        _b_lane_row = lane_id % Int32(8)
        _b_col_inner = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        _vg_base_row = warp_id * Int32(32)

        # H GEMM HALF-0 (ka=0..3, sH holds K=0..63).
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32(ka_local * 16 * 2)
            col_byte_off_b = Int32(ka_local * 16 * 2)
            _a_addr = _sK_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            _wh_write(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, _r)

        # Issue the SECOND state-tile half (K=64..127, overwrites sH).
        sync_threads()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,
                )
            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # HISTORY CONTRACTION — in the TMA half-1 shadow. hw[r, v] +=
        # sum_j sWScores[r, j] * u[j, v]; the w weights are baked into the
        # scores (via the w-scaled khist), so this is a plain contraction.
        # ============================================================
        _cp_async_wait_group_0()
        sync_threads()
        if P_hist > 0:
            _hc_a0, _hc_a1, _hc_a2, _hc_a3 = _ldmatrix_x4(sWScores, lane_id, SW_PAD * 2)
            _hc_b_base = (
                _sV_base_async_u + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
            )
            _hcr = _qtv_4mma(_hc_a0, _hc_a1, _hc_a2, _hc_a3, _hc_b_base)
            _wh_accumulate(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, _hcr)
            if const_expr(NJB == 2):
                # (w32) second j-block: scores cols [16, 32) x u rows
                # [16, 32). Skipped (CTA-uniformly) when the window fits
                # block 1 — those scores cols were never staged.
                if P_hist > Int32(W_BLK):
                    _h2a0, _h2a1, _h2a2, _h2a3 = _ldmatrix_x4(
                        sWScores, lane_id, SW_PAD * 2, byte_off=W_BLK * 2
                    )
                    _h2b = _hc_b_base + Int32(W_BLK * V_PADDED * 2)
                    _hcr2 = _qtv_4mma(_h2a0, _h2a1, _h2a2, _h2a3, _h2b)
                    _wh_accumulate(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, _hcr2)

        # Next tenant of qv_buf: the V tile.
        sync_threads()
        _gV_gbase = gV.iterator.toint()
        _v_base_bf16 = pid_b * sv_b + pid_hv * sv_hv
        for i in cutlass.range_constexpr(1):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col * Int32(2)
            if _v_row < Int32(self._n_valid):
                _cp_async_bf16x8(
                    _gV_gbase,
                    _v_base_bf16 + _v_row * sv_t + _v_col,
                    _sV_base_async_u + _smem_byte_off_v,
                )
        _cp_async_commit_group()

        # QT = sNegL @ sTmat, computed in the half-1 shadow on warps 0-1
        # into sWScores (dead after the contraction MMA above).
        if warp_id < 2:
            acc.fill(f32(0.0))
            _qt_col_off = warp_id * 8
            _qt_a_addr = (
                sNegL.iterator.toint() + _lane_mod16 * Int32(BF_PAD * 2) + _lane_hi
            )
            _qt_b_addr = (
                sTmat.iterator.toint()
                + _ldm_row * Int32(BF_PAD * 2)
                + _qt_col_off * Int32(2)
            )
            acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                _fused_ab_1mma(
                    _qt_a_addr,
                    _qt_b_addr,
                    acc.iterator[0],
                    acc.iterator[1],
                    acc.iterator[2],
                    acc.iterator[3],
                )
            )
            _qt_r0 = lane_id // 4
            _qt_c0 = (lane_id & 3) * 2
            _qt_sw_base = sWScores.iterator.toint()
            _sts_bf16x2_f32(
                _qt_sw_base + (_qt_r0 * SW_PAD + _qt_col_off + _qt_c0) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )
            _sts_bf16x2_f32(
                _qt_sw_base + ((_qt_r0 + 8) * SW_PAD + _qt_col_off + _qt_c0) * 2,
                acc.iterator[2],
                acc.iterator[3],
            )

        # Wait for the second state half before H GEMM half-1.
        cute.arch.mbarrier_wait(mbar_h_ptr, 1)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # H GEMM HALF-1 (ka=4..7, sH holds K=64..127; sH col offset resets).
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32((4 + ka_local) * 16 * 2)
            col_byte_off_b = Int32(ka_local * 16 * 2)
            _a_addr = _sK_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            _wh_write(wh_acc_0, wh_acc_1, wh_acc_2, wh_acc_3, _r)

        # ============================================================
        # TAIL: R → U (ring append) → y = hw_q + QT·R (decay fully baked
        # into hw, so no e^{G} factors here — unlike GDN).
        # ============================================================
        _sV_base = sV.iterator.toint()
        _gOut_gbase = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        # OutStage lives in k_buf (dead after H GEMM half-1) — keeping sH
        # untouched leaves the S0 half-1 tile RESIDENT for the tail fold.
        _sOutStage_base = sK.iterator.toint()

        _cp_async_wait_group_0()
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, 8):
                sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        sync_threads()

        # R = V − hw_k, in place over sV rows [0:t). hw_k lives in wh_acc
        # elements {0,1} (fragment rows r0 = packed k rows). beta is folded
        # into sTmat, NOT applied here.
        _y_r0 = lane_id // Int32(4)
        _y_c0 = (lane_id & Int32(3)) * Int32(2)
        _neg_one = f32(0.0) - f32(1.0)
        if _y_r0 < Int32(self._t_input):
            _r_i32 = (
                _y_r0 * Int32(V_PADDED // 2)
                + warp_id * Int32(16)
                + (lane_id & Int32(3))
            )
            _sQ_i32.iterator[_r_i32] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32],
                _neg_one,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 4] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 4],
                _neg_one,
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 8] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 8],
                _neg_one,
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 12] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 12],
                _neg_one,
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
            )
        sync_threads()

        # y = QT @ R (QT precomputed in sWScores) and U = sTmat @ R,
        # back-to-back on the same B operand.
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sWScores, lane_id, SW_PAD * 2)
        _qtv_base = _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        _qtvr = _qtv_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _qtv_base)
        _u_a0, _u_a1, _u_a2, _u_a3 = _ldmatrix_x4(sTmat, lane_id)
        _ur = _qtv_4mma(_u_a0, _u_a1, _u_a2, _u_a3, _qtv_base)

        for h_iter in cutlass.range_constexpr(4):
            h = warp_id * 4 + h_iter
            acc.iterator[0] = _qtvr[h_iter * 4]
            acc.iterator[1] = _qtvr[h_iter * 4 + 1]
            acc.iterator[2] = _qtvr[h_iter * 4 + 2]
            acc.iterator[3] = _qtvr[h_iter * 4 + 3]
            # + hw_q (fragment rows r0+8 = packed q rows; decay baked in).
            if h_iter == 0:
                acc.iterator[0] = acc.iterator[0] + wh_acc_0.iterator[2]
                acc.iterator[1] = acc.iterator[1] + wh_acc_0.iterator[3]
            if h_iter == 1:
                acc.iterator[0] = acc.iterator[0] + wh_acc_1.iterator[2]
                acc.iterator[1] = acc.iterator[1] + wh_acc_1.iterator[3]
            if h_iter == 2:
                acc.iterator[0] = acc.iterator[0] + wh_acc_2.iterator[2]
                acc.iterator[1] = acc.iterator[1] + wh_acc_2.iterator[3]
            if h_iter == 3:
                acc.iterator[0] = acc.iterator[0] + wh_acc_3.iterator[2]
                acc.iterator[1] = acc.iterator[1] + wh_acc_3.iterator[3]
            _out_r0 = lane_id // 4
            _out_c0 = (lane_id & 3) * 2
            _stg_col = h * 8 + _out_c0
            _sts_bf16x2_f32(
                _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )

        # Stage U rows [0:t) at OutStage rows [8:8+t): frag elements {0,1}
        # hold U row _y_r0.
        if _y_r0 < Int32(self._t_input):
            for _ug in cutlass.range_constexpr(4):
                _uu_col = (warp_id * Int32(4) + Int32(_ug)) * Int32(8) + _y_c0
                _sts_bf16x2_f32(
                    _sOutStage_base + ((8 + _y_r0) * V_PADDED + _uu_col) * 2,
                    _ur[_ug * 4],
                    _ur[_ug * 4 + 1],
                )

        # Coalesced output flush + u-ring append from the same staging tile.
        sync_threads()
        for _fl_pass in cutlass.range_constexpr(1):
            _fl_chunk = _fl_pass * 128 + tidx
            _fl_row = _fl_chunk // 16
            _fl_pos = _fl_chunk & 15
            _fl_lds = _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
            _fl_off = _out_base + _fl_row * so_t + _fl_pos * 8
            _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
            if const_expr(self._t_input >= 8):
                _st_global_v4_b32(_gOut_gbase, _fl_off, _v0, _v1, _v2, _v3)
            else:
                if _fl_row < Int32(self._t_input):
                    _st_global_v4_b32(_gOut_gbase, _fl_off, _v0, _v1, _v2, _v3)

        # u-ring append straight from OutStage rows [8:8+t).
        if tidx < Int32(self._t_input * (V_DIM_C // 8)):
            _uf_row = tidx // Int32(V_DIM_C // 8)
            _uf_pos = tidx % Int32(V_DIM_C // 8)
            _uv0, _uv1, _uv2, _uv3 = _lds_v4_b32(
                _sOutStage_base
                + (8 + _uf_row) * Int32(V_PADDED * 2)
                + _uf_pos * Int32(16)
            )
            _st_global_v4_b32(
                _gUC_base,
                _uc_base
                + ((ring_base + P_hist + _uf_row) & Int32(RING_MASK)) * V_DIM
                + _uf_pos * Int32(8),
                _uv0,
                _uv1,
                _uv2,
                _uv3,
            )

        # ============================================================
        # PER-REQUEST STATE FOLD + WRITE-BACK (ring semantics; cursor
        # slide/reset is the CALLER's commit). CTA-uniform predicate.
        #   1. reload the OLD u window rows into k_buf; restore the
        #      (w-scaled) khist snapshot into qv_buf.
        #   2. NO w pass over u (KDA: w lives in the khist tile).
        #   3. fold half-1 FIRST from the RESIDENT sH, then one re-TMA for
        #      half-0: D = u^T @ khist_w via MMA strips, then
        #      S_h = bdec[c]*S0 + D per element (per-channel bdec pairs),
        #      STS back to the swizzled sH bytes, coalesced STG to the pool.
        # ============================================================
        if P_hist >= flush_min:
            sync_threads()
            # --- 1) u reload (rows < P; .cg) into k_buf + khist snapshot
            #        restore into qv_buf ---
            for _fr in cutlass.range_constexpr(self._w_ring * V_DIM_C // (THREADS * 8)):
                _fr_group = tidx + _fr * THREADS
                _fr_row = _fr_group // Int32(V_DIM_C // 8)
                _fr_col = (_fr_group % Int32(V_DIM_C // 8)) * Int32(8)
                if _fr_row < P_hist:
                    _cp_async_bf16x8_cg(
                        _gUC_base,
                        _uc_base
                        + ((ring_base + _fr_row) & Int32(RING_MASK)) * V_DIM
                        + _fr_col,
                        _sK_base_async
                        + _fr_row * Int32(K_PADDED * 2)
                        + _fr_col * Int32(2),
                    )
                else:
                    # zero-fill rows >= P: at w32 the u-stage rows [16, 32)
                    # have NO prior full-write tenant (OutStage covers rows
                    # [0, 16) only), so stale bytes can be NaN and the
                    # fold's unconditional second j-block would compute
                    # 0 * NaN = NaN. A zero A-row is exact.
                    for _fz in cutlass.range_constexpr(4):
                        _sK_i32.iterator[
                            _fr_row * _kpad_i32 + _fr_col // 2 + Int32(_fz)
                        ] = Int32(0)
            _cp_async_commit_group()
            if const_expr(WR == 16):
                _kh_sts = (tidx // Int32(8)) * Int32(K_PADDED // 2) + (
                    tidx % Int32(8)
                ) * Int32(8)
                _sQ_i32.iterator[_kh_sts + 0] = _khs0
                _sQ_i32.iterator[_kh_sts + 1] = _khs1
                _sQ_i32.iterator[_kh_sts + 2] = _khs2
                _sQ_i32.iterator[_kh_sts + 3] = _khs3
                _sQ_i32.iterator[_kh_sts + 4] = _khs4
                _sQ_i32.iterator[_kh_sts + 5] = _khs5
                _sQ_i32.iterator[_kh_sts + 6] = _khs6
                _sQ_i32.iterator[_kh_sts + 7] = _khs7
            else:
                _kh_sts = (tidx // Int32(4)) * Int32(K_PADDED // 2) + (
                    tidx % Int32(4)
                ) * Int32(16)
                _sQ_i32.iterator[_kh_sts + 0] = _khs0
                _sQ_i32.iterator[_kh_sts + 1] = _khs1
                _sQ_i32.iterator[_kh_sts + 2] = _khs2
                _sQ_i32.iterator[_kh_sts + 3] = _khs3
                _sQ_i32.iterator[_kh_sts + 4] = _khs4
                _sQ_i32.iterator[_kh_sts + 5] = _khs5
                _sQ_i32.iterator[_kh_sts + 6] = _khs6
                _sQ_i32.iterator[_kh_sts + 7] = _khs7
                _sQ_i32.iterator[_kh_sts + 8] = _khs8
                _sQ_i32.iterator[_kh_sts + 9] = _khs9
                _sQ_i32.iterator[_kh_sts + 10] = _khs10
                _sQ_i32.iterator[_kh_sts + 11] = _khs11
                _sQ_i32.iterator[_kh_sts + 12] = _khs12
                _sQ_i32.iterator[_kh_sts + 13] = _khs13
                _sQ_i32.iterator[_kh_sts + 14] = _khs14
                _sQ_i32.iterator[_kh_sts + 15] = _khs15
            _cp_async_wait_group_0()
            sync_threads()
            # (kda) rows >= P of the u stage hold stale-but-finite OutStage
            # bytes; the khist tile's zeroed rows >= P kill their fold
            # contribution exactly (0 * finite = 0), so no u-side w pass.
            _gH0_st = (
                gH0.iterator.toint()
                + (
                    cache_idx64 * gH0.layout.stride[0]
                    + Int64(pid_hv) * gH0.layout.stride[1]
                )
                * 2
            )
            _h0_elem_base = Int32(0)
            _fa_row = (lane_id & Int32(7)) + ((lane_id >> Int32(4)) & Int32(1)) * Int32(
                8
            )
            _fa_colb = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
            _fc_r0 = lane_id // Int32(4)
            _fc_c0 = (lane_id & Int32(3)) * Int32(2)
            for _fh in (1, 0):
                # Barrier BEFORE anything in the iteration (half-1: publishes
                # the u stage; half-0: LOAD-BEARING against the re-TMA).
                sync_threads()
                if _fh == 0:
                    if warp_id == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_h_ptr,
                                V_DIM_C * K_HALF * 2,
                            )
                        cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)
                    cute.arch.mbarrier_wait(mbar_h_ptr, 0)
                    cute.arch.fence_view_async_shared()
                    sync_threads()
                for _fs2 in cutlass.range_constexpr(2):
                    _fs = warp_id * Int32(2) + Int32(_fs2)
                    for _fspan in cutlass.range_constexpr(2):
                        _fa0, _fa1, _fa2, _fa3 = _ldmatrix_x4_trans(
                            _sK_base_async
                            + _fa_row * Int32(K_PADDED * 2)
                            + _fs * Int32(32)
                            + _fa_colb
                        )
                        _frr = _qtv_4mma(
                            _fa0,
                            _fa1,
                            _fa2,
                            _fa3,
                            _sQ_base_async
                            + _ldm_row * Int32(K_PADDED * 2)
                            + Int32(_fh * 128 + _fspan * 64),
                        )
                        if const_expr(NJB == 2):
                            # (w32) second j-block: u rows [16, 32) (A) x
                            # khist rows [16, 32) (B). UNCONDITIONAL — rows
                            # >= P are exact zeros on both operands (u
                            # zero-fill above; khist zero-written in the
                            # w-scale rounds), so the extra MMAs add 0.
                            _f2a0, _f2a1, _f2a2, _f2a3 = _ldmatrix_x4_trans(
                                _sK_base_async
                                + Int32(W_BLK * K_PADDED * 2)
                                + _fa_row * Int32(K_PADDED * 2)
                                + _fs * Int32(32)
                                + _fa_colb
                            )
                            _frr2 = _qtv_4mma(
                                _f2a0,
                                _f2a1,
                                _f2a2,
                                _f2a3,
                                _sQ_base_async
                                + Int32(W_BLK * K_PADDED * 2)
                                + _ldm_row * Int32(K_PADDED * 2)
                                + Int32(_fh * 128 + _fspan * 64),
                            )
                            _frr = tuple(_frr[_fi] + _frr2[_fi] for _fi in range(16))
                        for _ft in cutlass.range_constexpr(4):
                            _f_col = Int32(_fspan * 32 + _ft * 8) + _fc_c0
                            _f_row0 = _fs * Int32(16) + _fc_r0
                            # per-channel bdec pair for global K columns
                            # (fh*64 + _f_col, +1).
                            _fb0 = sGvec.iterator[K_DIM + Int32(_fh * K_HALF) + _f_col]
                            _fb1 = sGvec.iterator[
                                K_DIM + Int32(_fh * K_HALF) + _f_col + 1
                            ]
                            _sh_a0 = _sw128_xor(
                                _sH_base_vl
                                + _f_row0 * Int32(K_HALF * 2)
                                + _f_col * Int32(2)
                            )
                            _s0p = _lds_b32(_sh_a0)
                            _sf0, _sf1 = _fold_fma2_bf16x2(
                                _s0p, _fb0, _fb1, _frr[_ft * 4], _frr[_ft * 4 + 1]
                            )
                            _sts_bf16x2_f32(_sh_a0, _sf0, _sf1)
                            _sh_a1 = _sw128_xor(
                                _sH_base_vl
                                + (_f_row0 + Int32(8)) * Int32(K_HALF * 2)
                                + _f_col * Int32(2)
                            )
                            _s1p = _lds_b32(_sh_a1)
                            _sf2, _sf3 = _fold_fma2_bf16x2(
                                _s1p, _fb0, _fb1, _frr[_ft * 4 + 2], _frr[_ft * 4 + 3]
                            )
                            _sts_bf16x2_f32(_sh_a1, _sf2, _sf3)
                sync_threads()
                # coalesced flush of the updated half.
                for _fc in cutlass.range_constexpr(8):
                    _f_chunk = tidx + _fc * THREADS
                    _f_row = _f_chunk >> 3
                    _f_pos = _f_chunk & Int32(7)
                    _cv0, _cv1, _cv2, _cv3 = _lds_v4_b32(
                        _sw128_xor(
                            _sH_base_vl
                            + _f_row * Int32(K_HALF * 2)
                            + _f_pos * Int32(16)
                        )
                    )
                    _st_global_v4_b32(
                        _gH0_st,
                        _h0_elem_base
                        + _f_row * K_DIM
                        + Int32(_fh * K_HALF)
                        + _f_pos * Int32(8),
                        _cv0,
                        _cv1,
                        _cv2,
                        _cv3,
                    )
            # No tail k restart: this step's normed k was appended at
            # (base+P+s)&mask in the gate stage — past every sibling's
            # fold-source window.


# ============================================================================
# Public entry point — kda_delta_rule_mtp_ucache_flush: the draft-token
# decode output PLUS the (k_cache, u_cache, g_cache, hist_len) history-ring
# append AND per-request state flush. Native T in {4, 8} only. bf16 only.
# Requires SM90+ (TMA + mbarrier); K == V == 128; H == HV.
# ============================================================================

_CACHE: dict = {}
_F32_CACHE: dict = {}


def _cached_f32(t):
    """Cast-cache the per-layer constant A_log / dt_bias to contiguous fp32,
    keyed by source-tensor identity (weakref-evicted; see the GDN kernel's
    _cached_bf16 for the data_ptr-staleness rationale)."""
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    key = id(t)
    c = _F32_CACHE.get(key)
    if c is None:
        c = t.to(torch.float32).contiguous()
        _F32_CACHE[key] = c
        weakref.finalize(t, _F32_CACHE.pop, key, None)
    return c


def kda_delta_rule_mtp_ucache_flush(
    A_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    u_cache: Optional[torch.Tensor] = None,
    g_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    flush_min: Optional[int] = None,
    restart_hist_on_flush: bool = True,
    pdl_trigger: bool = False,
    w_ring: Optional[int] = None,
    tma_late: Optional[bool] = None,
) -> torch.Tensor:
    """KDA decode output + u-cache append + PER-REQUEST state flush.

    The Kimi K3 analogue of ``gated_delta_rule_mtp_ucache_flush``: requests
    with hist_len[b] < flush_min take the verify path; requests with
    hist_len[b] >= flush_min additionally FOLD their ring into the state
    pool (S0[v,c] <- e^{G_P[c]} S0[v,c] + sum_j w_j[c] u_j[v] k_j[c],
    written in place at initial_state_indices[b]). RING SEMANTICS are
    identical to the GDN kernel: the live window is
    ``[cache_base[b], cache_base[b]+hist_len[b]) mod 32`` and the T new
    (u, normed-k, LOCAL per-channel-G) entries are appended at
    ``(cache_base+hist_len+s) & 31``. Cursor commits are CALLER-OWNED;
    ``restart_hist_on_flush=True`` applies the standalone commit here.

    Gate: Kimi K3 lower-bound form, computed in-kernel from the raw gate
    pre-activation ``g`` [B,T,H,K]:
        g_log = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h,:]))
    beta = sigmoid(b) is applied in-kernel (b carries logits).

    flush_min defaults to W_RING - T + 1 (lazy). Legal hist_len at call
    time: [0, 16]; cache_base: [0, 32).

    Returns ``output`` of shape ``[B, T, H, V]`` bf16.
    """
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None
    assert use_qk_l2norm_in_kernel, (
        "kda_delta_rule_mtp_ucache_flush: use_qk_l2norm_in_kernel=False is not "
        "supported (the kernel always applies Q/K L2 normalization)."
    )
    assert lower_bound is not None and lower_bound < 0.0, (
        f"lower_bound must be a negative float (Kimi K3 gate); got {lower_bound}."
    )

    B, T_in, H, K_dim = q.shape
    HV = v.shape[2]
    V_dim = v.shape[3]
    dev = q.device
    assert K_dim == K_DIM and V_dim == V_DIM_C, (
        f"this kernel requires K==V=={K_DIM}; got K={K_dim}, V={V_dim}."
    )
    assert H == HV, f"KDA has no GQA: H ({H}) must equal HV ({HV})."
    if T_in not in (4, 8):
        raise NotImplementedError(
            f"kda_delta_rule_mtp_ucache_flush: T={T_in} unsupported — native T in "
            "{4, 8} only."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(K_dim)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=dev)
    else:
        initial_state_indices = initial_state_indices.contiguous()
    assert (
        q.dtype == IO_TORCH
        and k.dtype == IO_TORCH
        and v.dtype == IO_TORCH
        and g.dtype == IO_TORCH
        and b.dtype == IO_TORCH
    ), (
        f"bf16-only kernel; got q={q.dtype} k={k.dtype} v={v.dtype} g={g.dtype} "
        f"b={b.dtype}."
    )
    assert tuple(g.shape) == (B, T_in, HV, K_dim), (
        f"g must be [B,T,H,K]={B, T_in, HV, K_dim}; got {tuple(g.shape)}"
    )
    assert tuple(b.shape) == (B, T_in, HV)
    assert initial_state_source.dtype == IO_TORCH, (
        f"initial_state_source must be {IO_TORCH} (pool, H, V, K); got "
        f"{initial_state_source.dtype}."
    )

    # A_log [H] fp32; dt_bias [H*K] fp32 (accept [H, K] and flatten).
    A_log = _cached_f32(A_log)
    dt_bias = _cached_f32(dt_bias.reshape(-1))
    assert A_log.numel() == HV, f"A_log must have H={HV} elements."
    assert dt_bias.numel() == HV * K_dim, f"dt_bias must have H*K={HV * K_dim} elems."

    def _inner_dense(t: torch.Tensor, name: str) -> bool:
        dense_inner = torch.empty(t.shape[1:], device="meta").stride()
        assert tuple(t.stride()[1:]) == tuple(dense_inner) and t.stride(0) >= (
            t.shape[1] * dense_inner[0] if t.dim() > 1 else 1
        ), (
            f"{name}: inner dims must be dense (block-strided dim 0 is OK); "
            f"got shape {tuple(t.shape)} strides {tuple(t.stride())}"
        )
        return t.is_contiguous()

    h0 = initial_state_source

    # --- ring validation -----------------------------------------------------
    assert (
        k_cache is not None
        and u_cache is not None
        and g_cache is not None
        and hist_len is not None
    ), "kda ucache: k_cache/u_cache/g_cache/hist_len are required."
    _pool = h0.shape[0]
    _pools_contig = _inner_dense(h0, "initial_state_source")
    assert k_cache.dtype == IO_TORCH
    _pools_contig &= _inner_dense(k_cache, "k_cache")
    assert tuple(k_cache.shape) == (_pool, H, RING_SLOTS, K_dim), (
        f"k_cache must be [pool={_pool}, H={H}, {RING_SLOTS}, K={K_dim}]; "
        f"got {tuple(k_cache.shape)}"
    )
    assert u_cache.dtype == IO_TORCH
    _pools_contig &= _inner_dense(u_cache, "u_cache")
    assert tuple(u_cache.shape) == (_pool, HV, RING_SLOTS, V_dim)
    assert g_cache.dtype == torch.float32
    _pools_contig &= _inner_dense(g_cache, "g_cache")
    assert tuple(g_cache.shape) == (_pool, HV, RING_SLOTS, K_dim), (
        f"g_cache must be [pool={_pool}, H={HV}, {RING_SLOTS}, K={K_dim}] fp32 "
        f"(per-channel cumulative log-decay); got {tuple(g_cache.shape)}"
    )
    assert hist_len.dtype == torch.int32 and hist_len.shape[0] == B
    # (w32) deep-window mode: logical window w_ring in {16, 32} over the
    # same 32-slot physical ring. Capacity algebra (same as the GDN W32
    # fork): flush_min <= min(w_ring - T + 1, RING - 2T + 1) — 13 @T=4 /
    # 9 @T=8 at w_ring=16; 25 @T=4 / 17 @T=8 at w_ring=32 — and
    # hist_len <= min(w_ring, RING - T).
    _wr_env = os.environ.get("KDA_UCACHE_WRING")
    if w_ring is None:
        w_ring = int(_wr_env) if _wr_env else W_RING
    assert w_ring in (16, 32), f"w_ring must be 16 or 32, got {w_ring}"
    _hist_cap = min(w_ring, RING_SLOTS - T_in)
    if restart_hist_on_flush:
        if cache_base is None:
            raise ValueError(
                "kda_delta_rule_mtp_ucache_flush: restart_hist_on_flush=True "
                "commits ring cursors in place and needs a caller-owned "
                "cache_base (got None)."
            )
        assert hist_len.is_contiguous() and cache_base.is_contiguous(), (
            "restart_hist_on_flush=True: hist_len and cache_base must be "
            "contiguous — .contiguous() would copy them and the in-place "
            "cursor commit would be lost on the caller's tensors."
        )
        if not torch.cuda.is_current_stream_capturing():
            _hl_min, _hl_max = (int(x.item()) for x in hist_len.aminmax())
            assert _hl_min >= 0 and _hl_max <= _hist_cap, (
                f"hist_len out of legal range [0, {_hist_cap}]: "
                f"min={_hl_min} max={_hl_max}"
            )
            _cb_min, _cb_max = (int(x.item()) for x in cache_base.aminmax())
            assert _cb_min >= 0 and _cb_max < RING_SLOTS, (
                f"cache_base out of legal range [0, {RING_SLOTS}): "
                f"min={_cb_min} max={_cb_max}"
            )
    hist_len = hist_len.contiguous()
    if cache_base is None:
        cache_base = torch.zeros_like(hist_len)
    assert cache_base.dtype == torch.int32 and cache_base.shape[0] == B
    cache_base = cache_base.contiguous()
    _fm_cap = min(w_ring - T_in + 1, RING_SLOTS - 2 * T_in + 1)
    if flush_min is None:
        flush_min = _fm_cap  # lazy: flush exactly when the window overflows
    assert 1 <= flush_min <= _fm_cap, (
        f"flush_min={flush_min} out of range [1, {_fm_cap}] for T={T_in}, "
        f"w_ring={w_ring} (tile cap w_ring-T+1, ring cap RING-2T+1)."
    )
    if w_ring == 32:
        # The deep-window kernel skips its round-2 pipeline (which zeroes
        # khist rows [16, 32)) CTA-uniformly at P <= 16; a fold there would
        # read stale bytes. A deep window with a shallow flush threshold is
        # self-defeating anyway — require the fold to live past block 1.
        assert flush_min > 16, (
            f"w_ring=32 requires flush_min > 16 (got {flush_min}); use "
            "w_ring=16 for shallow flush thresholds."
        )
    if not _pools_contig:
        raise NotImplementedError(
            "kda_delta_rule_mtp_ucache_flush: block-strided pools are not "
            "supported yet (contiguous pools required)."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    b = b.contiguous()

    _num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    _total_ctas = HV * B
    _needed = math.ceil(_total_ctas / _num_sms)
    # Per-CTA SMEM after the g_buf diet is small enough that the register
    # budget, not the 227 KB SMEM/SM limit, sets the achievable occupancy —
    # see the schedule-and-w_ring-dependent CTA/SM figures just below.
    # (tma-late) schedule choice by batch: measured crossover on B200 is
    # between B*HV = 384 (late loses ~5-8%) and 768 (late wins 3-17%) —
    # split at ~4 work-CTAs per SM. Env/arg override for experiments.
    _tl_env = os.environ.get("KDA_UCACHE_TMA_LATE")
    if tma_late is None:
        if _tl_env is not None:
            tma_late = _tl_env != "0"
        else:
            tma_late = 4 * _num_sms <= B * HV
    tma_late = bool(tma_late)
    # SMEM (w16): tma-late -> 7 CTAs/SM at T<=4, 6 at T=8; tma-early -> 5
    # regardless of T. w32: tma-late -> 5, tma-early -> 4.
    if w_ring == 32:
        _mbp_cap = 5 if tma_late else 4
    else:
        _mbp_cap = (7 if T_in <= 4 else 6) if tma_late else 5
    mbp = max(1, min(_needed + 1, _mbp_cap))
    _mbp_env = os.environ.get("KDA_UCACHE_MBP")
    if _mbp_env:
        mbp = int(_mbp_env)
    t_disc = 4 if T_in <= 4 else 8
    cache_key: tuple = (
        "kda-ucache-flush-v1",
        str(dev),
        mbp,
        t_disc,
        bool(pdl_trigger),
        HV,
        H,
        V_dim,
        w_ring,
        tma_late,
    )
    mk = from_dlpack

    def mk_dyn(t):
        return mk(t, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    _out_aliased = False
    if (
        output is not None
        and output.shape == (B, T_in, HV, V_dim)
        and output.dtype == IO_TORCH
        and output.is_contiguous()
    ):
        out_t = output
        _out_aliased = True
    else:
        out_t = torch.empty(B, T_in, HV, V_dim, dtype=IO_TORCH, device=dev)

    stream = cuda.CUstream(torch.cuda.current_stream(device=dev).cuda_stream)
    args = [
        mk_dyn(q),
        mk_dyn(k),
        mk_dyn(v),
        mk_dyn(g),
        mk_dyn(b),
        mk(A_log, 16),
        mk(dt_bias, 16),
        mk_dyn(h0),
        mk_dyn(initial_state_indices),
        mk_dyn(k_cache),
        mk_dyn(u_cache),
        mk_dyn(g_cache),
        mk_dyn(hist_len),
        mk_dyn(cache_base),
        mk_dyn(out_t),
        scale,
        float(lower_bound),
        HV,
        V_dim,
        H,
        int(flush_min),
        stream,
    ]

    if cache_key not in _CACHE:
        _CACHE[cache_key] = cute.compile(
            KdaDecodeUCacheFlushKernel(
                min_blocks_per_mp=mbp,
                t_input=t_disc,
                pdl_trigger=pdl_trigger,
                w_ring=w_ring,
                tma_late=tma_late,
            ),
            *args,
        )
    _CACHE[cache_key](*args)

    # Standalone ring cursor commit (Triton commit_*_replayssm_spec
    # semantics). restart_hist_on_flush=False hands the commit to the caller
    # (vLLM serving: N layers share one cursor set within a step).
    if restart_hist_on_flush:
        _flushed = hist_len >= flush_min
        cache_base.copy_((cache_base + hist_len * _flushed) & (RING_SLOTS - 1))
        hist_len.masked_fill_(_flushed, 0)

    if output is not None and not _out_aliased:
        output.copy_(out_t)
        return output
    return out_t


__all__ = [
    "kda_delta_rule_mtp_ucache_flush",
    "KdaDecodeUCacheFlushKernel",
    "K_DIM",
    "V_DIM_C",
    "W_RING",
    "RING_SLOTS",
    "RING_MASK",
]
