"""Gated Delta-Net (GDN) decode kernel with a history-ring cache (verify path).

Implements the ReplaySSM chunked delta-rule decode for speculative decoding.
For each (request, value-head) pair the kernel computes the draft-token
outputs from the checkpoint state and a small per-request history ring, and
appends the new correction vectors back to the ring so that later verify
steps can replay the recurrence. The checkpoint state is read-only here;
folding the ring into the state ("flush") is handled by the companion
``*_flush`` kernel.

Target: SM90+ (H200 / B200). Fixed K == V == 128.
Native draft length T in {4, 8}.

Precision
---------
Activations (q/k/v, a/b), the u and k rings, the checkpoint state, and the
output are all bf16; g_cache and the accumulators are f32. All GEMM
accumulation and the gate / log-decay math run in f32 internally.

Tensors (public entry point ``gated_delta_rule_mtp_ucache``)
------------------------------------------------------------
  Name                   Shape             Dtype   Dir    Meaning
  ---------------------  ----------------  ------  -----  ------------------------
  q, k                   [B, T, H,  K]     bf16    in     draft query / key
  v                      [B, T, HV, V]     bf16    in     draft values
  a, b                   [B, T, HV]        bf16    in     per-token gate / beta
  A_log                  [HV]              bf16    in     per-head log-decay (cast+cached)
  dt_bias                [HV]              bf16    in     per-head time-step bias (cast+cached)
  initial_state_source   [pool, HV, V, K]  bf16    in     checkpoint S0 (read-only here)
  initial_state_indices  [B]               int32   in     per-request pool slot
  k_cache                [pool, H,  16, K]  bf16   in/out  ring: L2-normalized keys
  u_cache                [pool, HV, 16, V]  bf16   in/out  ring: correction vectors
  g_cache                [pool, HV, 16]    f32     in/out  ring: cumulative log-decay
  hist_len               [B]               int32   in      filled ring slots P per request
  output                 [B, T, HV, V]     bf16    out     draft-token outputs (returned)

  Scalars: scale (float, default 1/sqrt(K)). softplus_beta / softplus_threshold
  are FIXED (the kernel uses beta=1 and no threshold); the wrapper rejects any
  non-default value rather than silently ignoring it.

Ring tensors are pool-indexed via initial_state_indices and MUST be
zero-initialized at allocation. The correction vector appended per token is
u = Tmat @ (V - e^G * (S0 @ k)). New entries are written speculatively to
slots [P, P+T); serving code rewinds after verification by setting
hist_len = P + accepted. The caller must keep P + T <= 16 and flush the ring
externally when it fills up.

High-level flow
---------------
  1. Load and L2-normalize k, q into shared memory.
  2. Form the T x T Gram matrices K@K^T and Q@K^T.
  3. Build the WY transform Tmat via a block triangular solve.
  4. GEMM the state S0 against the packed [k; q] tile, add the history-ring
     contribution, and produce the token outputs.
  5. Compute the new corrections u and append (normed k, u, G) to the ring.

Implementation notes
--------------------
- The state tile is streamed with TMA into an SW128-swizzled shared buffer,
  in two K-halves to halve its shared-memory footprint.
- All SMEM->register loads for the MMAs use ldmatrix; MMA accumulators are
  written straight to SMEM (no separate C-staging buffer).
- One shared-memory region is reused for several tiles (q, ring keys, ring
  corrections, v) whose lifetimes do not overlap.
- The Gram / inverse / GEMM phases are specialized at compile time for the
  effective draft length T in {4, 8, 16}.
"""

import torch
import math
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
BK_H = 16  # K-tile for the H GEMM (multiple of 16 for mma.k=16)
EPS = 1e-6
io = cutlass.BFloat16
f32 = cutlass.Float32
WARP = 32
THREADS = 128
T_PAD = 16
W_RING = 16  # history-ring slots per (request, head) — one 16-row MMA tile

# The state tile is streamed in two K-halves so its SMEM buffer is 16 KiB
# instead of 32 KiB. Shared-memory rows are padded by 8 bf16 to avoid bank
# conflicts on the ldmatrix / vectorized loads.
K_HALF = K_DIM // 2  # 64 — K-half streamed per TMA copy
K_PADDED = K_DIM + 8  # 136 — padded row stride for sK / sQ
V_PADDED = V_DIM_C + 8  # 136 — padded row stride for sV / sH (V rows, K cols)

TK = T * K_DIM
TK_PAD = T * K_PADDED
TT = T * T
BF_PAD = 24


# ---------------------------------------------------------------------------
# Small inline-PTX helpers used throughout the kernel.
# ---------------------------------------------------------------------------


def _smat_off(row, col):
    e = row * T + col
    return e ^ (
        ((e >> Int32(5)) & Int32(1)) | (((e >> Int32(6)) & Int32(1)) << Int32(3))
    )


def _ldmatrix_x4(smem_tensor, lane_id):
    addr = (
        smem_tensor.iterator.toint()
        + (lane_id % 16) * Int32(BF_PAD * 2)
        + (lane_id // 16) * Int32(16)
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
        " cvt.f32.bf16 _flo, _lo;"
        " cvt.f32.bf16 _fhi, _hi;"
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
    return _exp2_approx_f32(x * f32(1.4426950408889634))


def _mul_bf16x2_f32(packed_i32, scalar):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        " cvt.f32.bf16 _flo, _lo;"
        " cvt.f32.bf16 _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        " cvt.rn.bf16x2.f32 $0, _fhi, _flo; }",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_bf16x8(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """cp.async.ca, 16 B (8 bf16). Uses .ca for K and Q (small reuse stream)."""
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
    Used for H — single-pass stream, no L1 reuse keeps L1 capacity for K/Q/V.
    Note: ptxas on this rig rejects `.L1::no_allocate` on cp.async; .cg alone
    already implies skipping L1 caching."""
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
    """Retire the calling thread iff idx < 0 (PTX `exit`).

    Padded CUDA-graph rows carry a sentinel index < 0 (batch padding); the
    whole CTA retires at entry, BEFORE any SMEM/mbarrier/TMA/cp.async issue.
    Must be called CTA-uniformly right after pid/lane setup. Rows with
    idx >= 0 are untouched."""
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
    """STG.32 of one f32 (offset in f32 elements). Used for the g_cache ring
    append; all global writes in this kernel go through inline PTX."""
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


def _prefetch_l2_bf16(base_addr_i64, bf16_elem_offset):
    """prefetch.global.L2 of the 128-B line at base + 2*offset. Issued at
    kernel entry for the khist/u ring rows so the mid-kernel cp.async waves
    hit L2 instead of paying full DRAM latency."""
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


def _bar_sync_1_64():
    """Named barrier 1 over 64 threads (warps 2-3): orders the khist cp.async
    wave (issued and waited only by warps 2-3) before the scores GEMM's
    cross-warp ldmatrix reads, without stalling warps 0-1 which never touch
    the tile."""
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


def _r_sub_bf16x2(packed_i32, neg_eg_f32, hw0_f32, hw1_f32):
    """R-pass pair op: unpack two bf16 v-values, fma each with
    (-e^{G_s}) * hw_k, repack to bf16x2 in one i32 read-modify-write."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            packed_i32.ir_value(),
            neg_eg_f32.ir_value(),
            hw0_f32.ir_value(),
            hw1_f32.ir_value(),
        ],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        " cvt.f32.bf16 _flo, _lo;"
        " cvt.f32.bf16 _fhi, _hi;"
        " fma.rn.f32 _flo, $2, $3, _flo;"
        " fma.rn.f32 _fhi, $2, $4, _fhi;"
        " cvt.rn.bf16x2.f32 $0, _fhi, _flo; }",
        "=r,r,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_bf16x2_f32(smem_addr_i32, lo_f32, hi_f32):
    """Packed FP32 -> BF16x2 cast + STS.32 to SMEM. The cvt packs (hi, lo)
    into one 32-bit register; the store writes both bf16 values in a single
    4-byte SMEM transaction."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        " cvt.rn.bf16x2.f32 _v, $3, $2;"
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
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
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
    """KKT/QKT pattern — 4 sequential (ldmatrix_A + ldmatrix_B + MMA) at K-stride 32B."""
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
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+32];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+32];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+64];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+64];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+96];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+96];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
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
    """4 independent QT@V (ldmatrix_B_trans + MMA), B stride 16 B."""
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
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$0,$1,$2,$3}, {$32,$33,$34,$35}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+16];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$4,$5,$6,$7}, {$32,$33,$34,$35}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+32];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$8,$9,$10,$11}, {$32,$33,$34,$35}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+48];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
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
    """ldmatrix_A.x4 (16x16 of K) + 4× ldmatrix_B.x2 (8x16 of V-rows × K-cols, non-trans)
       + 4× MMA accumulating into 4 separate C-tiles (one per V-group).

    A is row-major bf16 [16, 16] (rows = T, cols = K-tile of 16). The lane-row-stride
    is row_stride_bytes (passed inside the SMEM address). For sK that's K_PADDED*2=272.
    B is row-major bf16 [8, 16] non-trans (rows = V, cols = K-tile of 16). B comes
    from the SW128-swizzled state tile (K_DIM*2=256 stride); callers MUST apply
    `_sw128_xor` to each per-lane B address before calling this.

    The A-fragment is shared across all 4 MMAs — 4 different B-fragments at b{0..3}_addr.
    """
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
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$34];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$4,$5,$6,$7}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$35];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$8,$9,$10,$11}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$36];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
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


def _sw128_xor(addr_i32):
    """Apply SW128 (cute.make_swizzle(3, 4, 3)) XOR to a logical SMEM byte
    address.

    SW128 spec: B=3 (3 bits XORed), M=4 (target = bits 4..6), S=3 (source =
    bits 7..9). Equivalent: phys = L XOR ((L >> 3) & 0x70).

    This MUST match the swizzle the SMEM tensor was built with (see
    `_make_sH_sw128_layout_half`).
    """
    return addr_i32 ^ ((addr_i32 >> Int32(3)) & Int32(0x70))


def _make_sH_sw128_layout_half():
    """SW128 K-major BF16 layout tiled to (V_DIM_C=128, K_HALF=64).

    K_HALF=64 BF16 = exactly one 128-byte row = one SW128 swizzle period.
    The base atom (8 rows x 64 cols = 1 swizzle period) tiles 16x in V and
    1x in K. The state buffer is reused across the 2 TMA-half loads
    (single-buffer streaming), halving its SMEM footprint.
    """
    sw = cute.make_swizzle(3, 4, 3)
    base = cute.make_layout((8, 64), stride=(64, 1))
    atom = cute.make_composed_layout(sw, 0, base)
    return cute.tile_to_shape(atom, (V_DIM_C, K_HALF), order=(1, 0))


class GdnDecodeUCacheKernel:
    """CuTeDSL GDN decode output + u-cache (ReplaySSM Alg. 8, no flush)."""

    def __init__(
        self,
        disable_state_update=False,
        min_blocks_per_mp=2,
        t_input=16,
        bv=None,
        n_valid=16,
        qkv_row_stride=0,
        ab_native=False,
        ab_t_stride=0,
        launch_pdl=False,
    ):
        assert disable_state_update, "State update not implemented in CuTeDSL kernel"
        # `bv` is accepted only for bench-script signature compatibility — this
        # kernel always consumes the full V=128 tile in one CTA so a V-split
        # path is not implemented here.
        self._disable_state_update = disable_state_update
        self._min_blocks_per_mp = min_blocks_per_mp
        self._t_input = int(t_input)
        # (native-short-T) number of valid token rows actually present in the q/k
        # gmem tensors. When n_valid < T the kernel loads only these rows via the
        # K/Q cp.async and zeros the sK/sQ[n_valid:T] smem tail itself, instead of
        # the host staging q/k into a T=16 zero-padded buffer. Default T (=16) keeps
        # the original behavior (host provides a full T-row, zero-padded tensor).
        self._n_valid = int(n_valid)
        # (strided-qkv) per-token row stride of the q/k/v gmem tensors, in elements.
        # 0 -> compact (token stride = H*K_DIM / HV*V_DIM, the staged/contiguous path).
        # >0 -> q/k/v are read directly from the fused conv-output column slices whose
        # token stride is conv_dim (= q_dim+k_dim+v_dim); the kernel loads from that
        # stride instead of requiring the host to .contiguous() them. Features within a
        # token stay contiguous (stride 1) so the smem layout / MMA path is unchanged.
        self._qkv_row_stride = int(qkv_row_stride)
        # (native-a/b) when True, a/b are the real [B, n_valid, HV] tensors (not staged into
        # T_KERNEL zero-padded buffers): batch stride = n_valid*HV and the warp-3 load/compute
        # gate uses n_valid instead of T. Tail lanes [n_valid:T] are not loaded; their gamma
        # (log_alpha=0) cannot reach the real rows through the causal prefix-sum.
        self._ab_native = bool(ab_native)
        # (strided-a/b) >0 -> a/b are regular strided views (packed a|b chunk:
        # token stride 2*HV, batch stride rows*token, feature stride 1) read
        # directly with NO host-side .contiguous(). 0 -> contiguous (HV).
        self._ab_t_stride = int(ab_t_stride)
        # Programmatic dependent launch (PDL): this kernel may begin while the
        # preceding kernel (which fires launch_dependents at entry) is still
        # running. No griddepcontrol wait needed: row disjointness means we
        # never read its writes.
        self._launch_pdl = bool(launch_pdl)

    @cute.experimental.jit
    def __call__(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        op = MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        B_val = gH0idx.layout.shape[0]
        # Build the TMA atom for the state tile. gH0 logical shape is
        # (pool, HV, V_DIM_C, K_DIM). cpasync.make_tiled_tma_atom tiles the
        # FIRST modes — we reorder modes to (V, K, HV, pool) by selecting
        # [2, 3, 1, 0] so the per-CTA tile is (V_DIM_C, K_DIM); the trailing
        # (HV, pool) modes survive tma_partition as outer iteration coords.
        # The SMEM target layout is SW128 swizzled — required for the TMA
        # descriptor.
        gH0_vkhp = cute.make_tensor(
            gH0.iterator,
            cute.select(gH0.layout, mode=[2, 3, 1, 0]),
        )
        # Half-K TMA atom — box = (V_DIM_C, K_HALF). Each CTA issues this atom
        # twice (once per K-half) into the SAME shared buffer via a 2-phase
        # mbarrier ping-pong, halving the state tile's SMEM footprint.
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
            gA,
            gB,
            gAlog,
            gDtbias,
            gH0,
            gH0idx,
            gKC,
            gUC,
            gGC,
            gHlen,
            gOut,
            scale,
            tiled_mma,
            HV,
            V_DIM,
            H,
            tma_atom_h,
            tma_tensor_h,
        ).launch(
            grid=(1, HV, B_val),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp,
            use_pdl=self._launch_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
    ):
        # Strides (contiguous layout assumed). q/k/v use the batch stride of their
        # ACTUAL row count: n_valid (== T in the default/staged path, so unchanged;
        # < T in the native-short-T path where q/k/v are the real [B,n_valid,...]
        # tensors and the kernel loads n_valid rows + zeros its smem tail).
        # (strided-qkv) when qkv_row_stride>0, q/k/v are the fused conv-output column
        # slices: per-token row stride is conv_dim (shared by q/k/v) rather than the
        # compact per-tensor H*K_DIM / HV*V_DIM. Head/element strides are unchanged
        # because each token's features stay contiguous within its slice.
        _rs = self._qkv_row_stride
        _qt = _rs if _rs > 0 else H * K_DIM
        _kt = _rs if _rs > 0 else H * K_DIM
        _vt = _rs if _rs > 0 else HV * V_DIM
        cutlass.Int32(1)
        sq_h = K_DIM
        sq_t = _qt
        sq_b = self._n_valid * _qt
        cutlass.Int32(1)
        sk_h = K_DIM
        sk_t = _kt
        sk_b = self._n_valid * _kt
        cutlass.Int32(1)
        sv_hv = V_DIM
        sv_t = _vt
        sv_b = self._n_valid * _vt
        cutlass.Int32(1)
        so_hv = V_DIM
        so_t = HV * V_DIM
        # Output batch stride matches the output tensor's row count: n_valid (== T in
        # the staged path -> [B,T_KERNEL] out; == T in the native path where out is
        # the compact [B,T] tensor, valid because native is gated to T==t_disc so the
        # t_input-gated STG writes exactly T rows). Removes the caller's reshape copy.
        so_b = self._n_valid * HV * V_DIM
        # (native-a/b) a/b rows actually present in the tensor: n_valid when native (real
        # [B,n_valid,HV] passed) else T (staged T_KERNEL-row zero-padded buffer). The warp-3
        # load/compute gate below uses the same count so tail lanes never read OOB.
        _ab_rows = self._n_valid if self._ab_native else T
        _ab_t = self._ab_t_stride if self._ab_t_stride > 0 else HV
        sa_hv = cutlass.Int32(1)
        sa_t = _ab_t
        sa_b = _ab_rows * _ab_t
        sb_hv = cutlass.Int32(1)
        # b carries the SAME token stride as a (wrapper enforces
        # tuple(b.stride()) == tuple(a.stride())), so use _ab_t, not a
        # hardcoded HV — else packed/chunk-view b (stride(1) != HV, the vLLM
        # strided-qkv path) reads the wrong rows. _ab_t == HV when compact.
        sb_t = _ab_t
        sb_b = _ab_rows * _ab_t
        # State pool natural layout: (pool, HV, V, K) bf16 contiguous. Addressing
        # is handled by the TMA descriptor, so no raw GMEM strides are needed here.

        tidx, _, _ = cute.arch.thread_idx()
        _pid_vt, pid_hv, pid_b = cute.arch.block_idx()
        lane_id = tidx & 31
        warp_id = tidx // WARP

        # GQA head mapping
        i_h = pid_hv // (HV // H)
        cache_idx = gH0idx.iterator[pid_b]
        # Padded CUDA-graph rows carry cache_idx < 0 (batch padding sentinel):
        # the whole CTA retires here, before any SMEM/TMA/ring work. Real rows
        # (idx >= 0) proceed unchanged.
        _exit_cta_if_neg(cache_idx)

        # Ring addressing + per-request history fill level.
        # k_cache [pool, H, W_RING, K] bf16 (L2-NORMED k),
        # u_cache [pool, HV, W_RING, V] bf16,
        # g_cache [pool, HV, W_RING] f32 (absolute log-decay since ckpt).
        P_hist = gHlen.iterator[pid_b]
        # Pool/head strides from the descriptor layouts, NOT dense shape
        # products: paged serving pools (vLLM) are block-strided views whose
        # dim-0 stride spans the whole multi-component page; inner dims stay
        # dense. With contiguous pools these reduce to the old products.
        skc_pool = gKC.layout.stride[0]
        skc_h = gKC.layout.stride[1]
        suc_pool = gUC.layout.stride[0]
        suc_hv = gUC.layout.stride[1]
        sgc_pool = gGC.layout.stride[0]
        sgc_hv = gGC.layout.stride[1]
        # 64-bit per-CTA pool ELEMENT offsets: block-strided paged pools can
        # exceed 2^31 elements (cache_idx * pool_stride wraps in 32-bit).
        cache_idx64 = Int64(cache_idx)
        _kc_pool_e64 = cache_idx64 * skc_pool + Int64(i_h) * skc_h
        _uc_pool_e64 = cache_idx64 * suc_pool + Int64(pid_hv) * suc_hv
        _gc_pool_e64 = cache_idx64 * sgc_pool + Int64(pid_hv) * sgc_hv
        # g-ring byte base with the pool offset absorbed (stores use small
        # intra-row offsets only; do NOT re-add _gc_pool_e64 at call sites).
        _gGC_base_st = gGC.iterator.toint() + _gc_pool_e64 * 4

        # Issue the per-token gate/bias (a, b, A_log, dt_bias) loads for warp 3
        # right after pid/lane setup so they are the first instructions warp 3
        # emits — maximizing the HBM round-trip hiding window before the
        # gamma/beta math consumes them.
        _v7e_a_bf16 = f32(0.0)
        _v7e_b_bf16 = f32(0.0)
        _v7e_alog_bf16 = f32(0.0)
        _v7e_dt_bf16 = f32(0.0)
        # Load only the _ab_rows present in the a/b tensors (n_valid native,
        # T staged) so tail lanes never index past the real [B,n_valid,HV] rows.
        if warp_id == 3 and lane_id < _ab_rows:
            _v7e_a_bf16 = gA.iterator[
                pid_b * sa_b + lane_id * sa_t + pid_hv * sa_hv
            ].to(f32)
            _v7e_b_bf16 = gB.iterator[
                pid_b * sb_b + lane_id * sb_t + pid_hv * sb_hv
            ].to(f32)
            _v7e_alog_bf16 = gAlog.iterator[pid_hv].to(f32)
            _v7e_dt_bf16 = gDtbias.iterator[pid_hv].to(f32)

        # Issue the g_cache ring LDG on warp 1 (idle until the norm phase, where
        # it computes w_j/bdec from these values). Lanes >= W_RING keep 0.0;
        # g_cache is f32 so the load is direct.
        _g_hist_f32 = f32(0.0)
        if warp_id == 1 and lane_id < W_RING:
            _g_hist_f32 = gGC.iterator[_gc_pool_e64 + Int64(lane_id)]

        # L2-prefetch the live khist/u ring rows (rows < P; 2 x 128-B lines per
        # row) at kernel entry — the mid-kernel cp.async waves then hit L2,
        # shrinking their exposed latency at small B.
        _pf_row = (tidx & Int32(31)) >> 1
        _pf_half = (tidx & Int32(1)) * Int32(64)
        if _pf_row < P_hist:
            if tidx < Int32(32):
                _prefetch_l2_bf16(
                    gKC.iterator.toint() + _kc_pool_e64 * 2,
                    _pf_row * K_DIM + _pf_half,
                )
            if tidx >= Int32(32) and tidx < Int32(64):
                _prefetch_l2_bf16(
                    gUC.iterator.toint() + _uc_pool_e64 * 2,
                    _pf_row * V_DIM + _pf_half,
                )

        smem = utils.SmemAllocator()

        @cute.struct
        class SS:
            # mbarrier for the state-tile TMA load (8B Int64). Placed first so
            # its natural 8B alignment is preserved by the prefix; the large
            # 128-aligned buffers follow. Arrival count = 1 (only one thread
            # issues the TMA + arrive; the TX-bytes complete it independently).
            h_load_mbar: cute.struct.MemRange[Int64, 1]
            k_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # q_buf and v_buf are ALIASED onto one region (qv_buf): sQ is fully
            # read before sV's cp.async fill writes the same bytes (a
            # sync_threads orders the handoff). K_PADDED == V_PADDED == 136.
            qv_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # State tile = V=128 rows x K_HALF=64 cols bf16, SW128 swizzled,
            # single-buffered and reused across the 2 TMA half-loads via the
            # mbarrier ping-pong.
            h_buf: cute.struct.Align[cute.struct.MemRange[io, V_DIM_C * K_HALF], 128]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            gamma: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            # w-scaled TRANSPOSED scores tile [W_RING, BF_PAD] bf16 — the
            # A-operand of the MMA history contraction
            # (sWScores[r_packed, j] = w_j * khist_j . packed_r).
            wscores_bf: cute.struct.Align[
                cute.struct.MemRange[io, W_RING * BF_PAD], 128
            ]
            # G-ring scratch: [0:16]=G_j, [16:32]=w_j, [32]=bdec.
            ghist_fp32: cute.struct.Align[cute.struct.MemRange[f32, 48], 128]

        st = smem.allocate(SS)
        sK = st.k_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # sQ and sV both view the same qv_buf SMEM region (alias).
        sQ = st.qv_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # State tile view: (V_DIM_C, K_HALF) SW128-swizzled (exactly one swizzle
        # period in K, so the _sw128_xor() helper applies unchanged).
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sV = st.qv_buf.get_tensor(cute.make_layout((T, V_PADDED), stride=(V_PADDED, 1)))
        sTmat = st.tmat_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sGamma = st.gamma.get_tensor(cute.make_layout((WARP,)))
        sBeta = st.beta.get_tensor(cute.make_layout((WARP,)))
        sMat = st.mat_fp32.get_tensor(cute.make_layout((T, T), stride=(T, 1)))
        sNegL = st.scratch_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sWScores = st.wscores_bf.get_tensor(
            cute.make_layout((W_RING, W_RING), stride=(BF_PAD, 1))
        )
        sGhist = st.ghist_fp32.get_tensor(cute.make_layout((48,)))

        # ============================================================
        # mbarrier init for the state-tile TMA load. Thread 0 issues the
        # bulk-tensor copy once; all 128 threads block in mbarrier_wait before
        # the H GEMM. Arrival count = 1 (the issuing thread is the sole
        # arriver; the TX-bytes complete the barrier independently).
        # ============================================================
        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()
        sync_threads()

        # Partition the TMA tensor for this CTA's (cache_idx, pid_hv).
        # tma_tensor_h logical shape (V, K, HV, pool) (mode-reordered on host).
        # flat_divide with (V_DIM_C, K_HALF) -> (V_TILE, K_TILE, V_REST=1, K_REST=2, HV, pool).
        # Two slices, one per K-half; both write to the SAME state buffer.
        gH_tiled = cute.flat_divide(tma_tensor_h, (V_DIM_C, K_HALF))
        gH_slice0 = gH_tiled[None, None, None, 0, pid_hv, cache_idx]
        gH_slice1 = gH_tiled[None, None, None, 1, pid_hv, cache_idx]
        gH_grp0 = cute.group_modes(gH_slice0, 0, 3)
        gH_grp1 = cute.group_modes(gH_slice1, 0, 3)
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
        # Build a register-only C fragment template (no SMEM).
        # partition_shape_C((T, 8)) returns the per-thread partition shape for
        # an (M=T, N=8) tile under the m16n8k16 MMA; make_fragment_C creates
        # a register fragment of that shape with the MMA's accumulator dtype (f32).
        tCsC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((T, 8)))
        acc = cute.make_fragment_like(tCsC)
        _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)

        EPT_TT = TT // THREADS

        # The warp-3 gamma/beta scalar loads were issued right after pid/lane
        # setup above; the _v7e_* registers are already in flight here.

        # ============================================================
        # cp.async stage 1: K + Q (8 bf16 / instr, .ca for L1 reuse)
        # ============================================================
        k_base = pid_b * sk_b + i_h * sk_h
        q_base = pid_b * sq_b + i_h * sq_h
        _gK_base = gK.iterator.toint()
        _gQ_base = gQ.iterator.toint()
        _sK_i32 = cute.recast_tensor(sK, cutlass.Int32)
        _sQ_i32 = cute.recast_tensor(sQ, cutlass.Int32)
        _kq_col_i32 = lane_id * Int32(2)
        _kpad_i32 = K_PADDED // 2
        _sK_base_async = sK.iterator.toint()
        _sQ_base_async = sQ.iterator.toint()
        for i in cutlass.range_constexpr(T * K_DIM // (THREADS * 8)):
            _kq_group = tidx + i * THREADS
            _kq_row = _kq_group // Int32(K_DIM // 8)
            _kq_col_bf16_async = (_kq_group % Int32(K_DIM // 8)) * Int32(8)
            _smem_byte_off = _kq_row * Int32(K_PADDED * 2) + _kq_col_bf16_async * Int32(
                2
            )
            # When n_valid < T the q/k gmem tensors hold only n_valid rows; skip
            # the cp.async for rows >= n_valid (those would read OOB). The
            # sK/sQ[n_valid:T] smem tail is zeroed after the wait below.
            if const_expr(self._n_valid < T):
                if _kq_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gK_base,
                        k_base + _kq_row * sk_t + _kq_col_bf16_async,
                        _sK_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gQ_base,
                        q_base + _kq_row * sq_t + _kq_col_bf16_async,
                        _sQ_base_async + _smem_byte_off,
                    )
            else:
                _cp_async_bf16x8(
                    _gK_base,
                    k_base + _kq_row * sk_t + _kq_col_bf16_async,
                    _sK_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gQ_base,
                    q_base + _kq_row * sq_t + _kq_col_bf16_async,
                    _sQ_base_async + _smem_byte_off,
                )
        _cp_async_commit_group()  # group 0 = K+Q

        # ============================================================
        # Issue the FIRST state-tile half (K=0..63); its load overlaps
        # Phase 1 + Phase 2. The SECOND half is issued later (right before
        # H GEMM half-1) once half-0 has finished reading the buffer.
        # mbarrier_arrive_and_expect_tx with V_DIM_C * K_HALF * 2 bytes.
        # cute.copy MUST stay OUTSIDE elect_one (else the GPU deadlocks).
        # ============================================================
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # warp 3 computes gamma/beta in parallel with the cp.async pipeline,
        # consuming the gate/bias registers loaded at kernel entry.
        # ============================================================
        if warp_id == 3:
            log_alpha = f32(0.0)
            beta_val = f32(0.0)
            # Gate by _ab_rows: tail lanes [n_valid:T] are not loaded (their
            # _v7e_a/b regs are undefined), so they must keep log_alpha=beta=0. The causal
            # prefix-sum makes their (zero) contribution invisible to rows 0..n_valid-1.
            if lane_id < _ab_rows:
                a_val = _v7e_a_bf16
                b_val = _v7e_b_bf16
                A_log_val = _v7e_alog_bf16
                dt_val = _v7e_dt_bf16
                x = a_val + dt_val
                sp = cute.math.log(f32(1.0) + _exp_approx_f32(x))
                log_alpha = (f32(0.0) - _exp_approx_f32(A_log_val)) * sp
                beta_val = f32(1.0) / (f32(1.0) + _exp_approx_f32(f32(0.0) - b_val))
            cumsum = log_alpha
            for d in [1, 2, 4, 8]:
                prev = cute.arch.shuffle_sync(
                    cumsum, Int32(lane_id - d), Int32(0x0000FFFF), Int32(0x1F)
                )
                if lane_id >= d:
                    cumsum = cumsum + prev
            if lane_id < T:
                exp_g = _exp_approx_f32(cumsum)
                sGamma.iterator[T + lane_id] = exp_g
                # Store the log-domain cumsum in the free sGamma[0:T] slots so the
                # decay matrix can be formed as exp(cumsum_r - cumsum_c) directly (bounded
                # <=1 for the causal r>=c region) instead of exp(cumsum_r)*exp(-cumsum_c),
                # whose exp(-cumsum_c) overflows to inf for strong real decay (large A_log)
                # -> 0*inf = NaN. Mathematically identical, but NaN-safe.
                sGamma.iterator[lane_id] = cumsum
                sBeta.iterator[lane_id] = beta_val
                sBeta.iterator[T + lane_id] = f32(1.0) / exp_g

        # ============================================================
        # Wait for K+Q (group 0); H load is now driven by mbarrier (TMA),
        # not cp.async commit groups, so wait_group(0) is sufficient.
        # ============================================================
        _cp_async_wait_group_0()
        # (native-short-T) zero the sK/sQ tail rows [n_valid:T] that were NOT loaded
        # from gmem (q/k held only n_valid rows). This restores the documented
        # sK/sQ[..:T]=0 invariant the t_input<=8 zero-propagation proof relies on
        # (see the QT@V comment), previously supplied by the host zero-padding the
        # staged buffer. tidx in [0,THREADS=128) covers K_DIM=128 cols exactly;
        # pad cols [K_DIM:K_PADDED) are never read. The following sync_threads()
        # publishes both the loaded rows and these zeros before the L2-norm reads.
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, T):
                sK.iterator[_zr * K_PADDED + tidx] = io(0.0)
                sQ.iterator[_zr * K_PADDED + tidx] = io(0.0)
        sync_threads()

        # L2 norm for K (warps 0,1) and Q (warps 2,3) — t-aware warp skip.
        # At T<=8, warps 1 and 3 (which normalize rows 8..15) are dead-elided.
        # Rows 8..15 of sK/sQ feed only outputs that Phase-2 already gates
        # (T11 = diag(beta1) with no read of sMat[8..15, 8..15], Y/T10 skipped,
        # and the Phase-4 STG bottom half is t_input-gated).
        # Warps 0 and 2 still process 8 lane-rows even at T=4 — rows 4..7 of
        # sK/sQ are wrapper-zero-padded, so L2-norm is a harmless no-op
        # (zero * any_inv_norm = zero). Predicating individual lanes would
        # break the shuffle_sync (mask 0xFFFFFFFF requires all 32 warp lanes).
        # `self._t_input` is a Python int fixed at JIT compile time, so
        # const_expr produces 2 specializations: t_input<=8 and t_input=16.
        if const_expr(self._t_input <= 8):
            if warp_id == Int32(0):
                norm_row = lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                # (perf) rows >= t are zeros — their L2-norm RMW is a no-op
                # on zeros (bit-exact to skip); saves 32 LSU ops per tail
                # lane. The quad shuffles below still run on all 32 lanes
                # (tail partial stays 0; tail inv_norm is never stored).
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
                # (perf) tail-lane skip as in the K norm above (bit-exact).
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
                        _qn_val = _mul_bf16x2_f32(
                            _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                        )
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _qn_val
                        # (perf) fused packed-A-tile store: duplicate the
                        # normed q rows [0:t) into sK rows [8:8+t) (dual
                        # store) — removes the separate pack pass and one
                        # CTA barrier; the norm-end sync publishes both.
                        _sK_i32.iterator[
                            _norm_off_i32 + 8 * (K_PADDED // 2) + 4 * c
                        ] = _qn_val
            if warp_id == Int32(1):
                # (u-cache) publish the G ring + w_j/bdec while warps 0/2
                # L2-normalize (warp 1 is idle here at t<=8). w_j =
                # e^{G_P - G_j} for j < P else 0 — the j >= P mask is
                # MANDATORY (stale slots -> inf * 0 = NaN). G_P is lane
                # P-1's value, shuffled (all 32 lanes execute the shuffle).
                if lane_id < Int32(W_RING):
                    sGhist.iterator[lane_id] = _g_hist_f32
                _p_src = P_hist - 1 if P_hist > 0 else Int32(0)
                _gp_lane = cute.arch.shuffle_sync(
                    _g_hist_f32, _p_src, Int32(0xFFFFFFFF), Int32(0x1F)
                )
                _gp = _gp_lane if P_hist > 0 else f32(0.0)
                if lane_id < Int32(W_RING):
                    _w_j = (
                        _exp_approx_f32(_gp - _g_hist_f32)
                        if lane_id < P_hist
                        else f32(0.0)
                    )
                    sGhist.iterator[W_RING + lane_id] = _w_j
                if lane_id == Int32(0):
                    _bdec_v = _exp_approx_f32(_gp) if P_hist > 0 else f32(1.0)
                    sGhist.iterator[32] = _bdec_v
                # G-ring append: g_new[s] = G_P + cumsum_s. sGamma[0:T]
                # (log-domain cumsum, warp 3) was published by the K/Q-wait
                # sync before this norm phase.
                if lane_id < Int32(self._t_input):
                    _st_global_f32(
                        _gGC_base_st,
                        P_hist + lane_id,
                        _gp + sGamma.iterator[lane_id],
                    )
        else:
            # T=16 path.
            if warp_id < 2:
                norm_row = warp_id * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                for c in cutlass.range_constexpr(16):
                    _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
            if warp_id >= 2:
                norm_row = (warp_id - 2) * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
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
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        sync_threads()

        # ============================================================
        # (u-cache) PACKED A-TILE: the normed q rows [0:t) were already
        # dual-stored into sK rows [8:8+t) inside the warp-2 norm pass
        # (published by the sync above) — the H GEMM yields S0·k (rows
        # 0..t-1) AND S0·q (rows 8..8+t-1) from ONE streamed S0 read.
        # Safe with native-a/b at t<=8: Gram pollution from the packed
        # rows lands only where beta=0 (rows >= t) or r<c masks it.
        # k-ring append (one CTA per k-group) runs here, BEFORE the bdec
        # row-scale corrupts the normed k rows; it is a pure global write
        # of sK rows [0:t) (read-only vs the Grams below), so NO extra
        # barrier is needed — it overlaps the Grams.
        # ============================================================
        if (pid_hv % (HV // H)) == 0:
            _gKC_base_st = gKC.iterator.toint() + _kc_pool_e64 * 2
            _kc_wr_base = P_hist * K_DIM
            if tidx < Int32(self._t_input * (K_DIM // 8)):
                _kc_row = tidx // Int32(K_DIM // 8)
                _kc_pos = tidx % Int32(K_DIM // 8)
                _kv0, _kv1, _kv2, _kv3 = _lds_v4_b32(
                    _sK_base_async
                    + _kc_row * Int32(K_PADDED * 2)
                    + _kc_pos * Int32(16)
                )
                _st_global_v4_b32(
                    _gKC_base_st,
                    _kc_wr_base + _kc_row * K_DIM + _kc_pos * Int32(8),
                    _kv0,
                    _kv1,
                    _kv2,
                    _kv3,
                )

        # KKT (warps 0-1) || QKT (warps 2-3) — direct acc → SMEM writes.
        acc.fill(f32(0.0))
        _sK_int = sK.iterator.toint()
        _sQ_int = sQ.iterator.toint()
        _rs_kpad = Int32(K_PADDED * 2)
        _lane_mod16 = lane_id & Int32(15)
        _lane_hi = (lane_id >> Int32(4)) * Int32(16)
        _lane_mod8 = lane_id % Int32(8)
        _lane_b_col = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        for kk_group in cutlass.range_constexpr(K_DIM // 16 // 4):
            k_group_off = kk_group * 4 * 16 * Int32(2)
            if warp_id < 2:
                col_off = warp_id * 8
                _a_base = _sK_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sK_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
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
            if warp_id >= 2:
                col_off = (warp_id - Int32(2)) * Int32(8)
                _a_base = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sK_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
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
        if warp_id < 2:
            col_off = warp_id * 8
            sMat.iterator[_smat_off(_r0, col_off + _c0)] = acc.iterator[0]
            sMat.iterator[_smat_off(_r0, col_off + _c0 + 1)] = acc.iterator[1]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0)] = acc.iterator[2]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0 + 1)] = acc.iterator[3]
        if warp_id >= 2:
            col_off = (warp_id - Int32(2)) * Int32(8)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[1].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[2].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = acc.iterator[3].to(
                io
            )
        sync_threads()

        # ============================================================
        # (u-cache) tenant #2 of qv_buf: khist tile [W_RING, K_PADDED] via
        # cp.async.ca (the tile is re-read by the 4 sibling CTAs of the
        # k-group). sQ is dead after the Grams above. Landing is awaited
        # (wait + one barrier) after Phase 2, before the scores GEMM.
        # ============================================================
        _gKC_base_rd = gKC.iterator.toint() + _kc_pool_e64 * 2
        _kc_rd_base = Int32(0)
        # (perf) the khist wave is ISSUED AND WAITED ONLY BY WARPS 2-3 (the
        # tile's sole consumers, in the scores GEMM): warps 0-1 enter the
        # block inverse without ever stalling on ring-load latency. The
        # cross-warp (2<->3) visibility is ordered by their own wait_group
        # + the 64-thread named barrier at the scores GEMM. Rows >= P stay
        # unread (w-masked downstream).
        if warp_id >= 2:
            for _kh in cutlass.range_constexpr(W_RING * K_DIM // (64 * 8)):
                _kh_group = (tidx - Int32(64)) + _kh * Int32(64)
                _kh_row = _kh_group // Int32(K_DIM // 8)
                _kh_col = (_kh_group % Int32(K_DIM // 8)) * Int32(8)
                if _kh_row < P_hist:
                    _cp_async_bf16x8(
                        _gKC_base_rd,
                        _kc_rd_base + _kh_row * K_DIM + _kh_col,
                        _sQ_base_async
                        + _kh_row * Int32(K_PADDED * 2)
                        + _kh_col * Int32(2),
                    )
            _cp_async_commit_group()

        # ============================================================
        # PHASE 2: log-depth Neumann inverse
        # ============================================================
        for idx in cutlass.range_constexpr(EPT_TT):
            flat = tidx + idx * THREADS
            r = flat // T
            c = flat % T
            # (perf) tail short-circuit: entries with r >= t or c >= t are
            # structurally ZERO in both outputs under the packed-tile
            # invariants (beta = 0 tail rows, zero k/q rows, r < c masking)
            # — write the zeros directly and skip their sNegL/sMat/sGamma
            # loads. Bit-exact: the skipped loads only ever fed values
            # that reduced to 0 for these entries.
            if r < Int32(self._t_input) and c < Int32(self._t_input):
                # Stable decay: exp(cumsum_r - cumsum_c)
                # directly (<=1 for r>c) instead of
                # exp(cumsum_r)*exp(-cumsum_c) (overflows). exp_gij is only
                # consumed for r>=c (below); r<c value is discarded.
                exp_gij = (
                    f32(1.0)
                    if r == c
                    else (
                        _exp_approx_f32(sGamma.iterator[r] - sGamma.iterator[c])
                        if r > c
                        else f32(0.0)
                    )
                )
                qkt = sNegL.iterator[r * BF_PAD + c].to(f32)
                sNegL.iterator[r * BF_PAD + c] = (
                    (qkt * exp_gij).to(io) if r >= c else io(0.0)
                )
                kkt_val = sMat.iterator[_smat_off(r, c)]
                negL_val = (
                    (f32(0.0) - sBeta.iterator[r] * exp_gij * kkt_val)
                    if r > c
                    else f32(0.0)
                )
                negL_bf = negL_val.to(io)
                sTmat.iterator[r * BF_PAD + c] = negL_bf
            else:
                sNegL.iterator[r * BF_PAD + c] = io(0.0)
                sTmat.iterator[r * BF_PAD + c] = io(0.0)
        # (u-cache) NOTE: the khist tile is NOT waited here — warps 2-3 own
        # the wave and wait for it themselves at the scores GEMM, so this
        # barrier only publishes phase-2's writes.
        sync_threads()

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2

        # ============================================================
        # BLOCK INVERSE for T=16 — register-resident forward substitution.
        # Each lane owns one column of the 8x8 result, holding all 8 row
        # values in registers (no per-row SMEM round-trip). Eliminates
        # the SMEM store-then-load critical path (~240 cyc → ~120 cyc).
        # Algorithm:
        #   T00 = solve(I - M00, diag(beta0))  [warp 0, X in regs]
        #   T11 = solve(I - M11, diag(beta1))  [warp 1, X in regs] (parallel)
        #   Y   = M10 @ T00                    [warp 0, scalar 8x8 product]
        #   T10 = solve(I - M11, Y)            [warp 1, X in regs]
        #   sTmat = [[T00, 0], [T10, T11]] (bf16)
        #
        # v14_t_aware Stage B.2: For t_input ≤ 8, M's nonzero region is the
        # top-left 8×8 only (rows/cols ≥ T are zero in K → M block-rows/cols
        # ≥ 8 are zero). Therefore at T≤8:
        #   - T11 = solve(I, diag(β1)) = diag(β1) directly (skip 28-MAC forward sub)
        #   - Y   = M10 @ T00 = 0 (skip entire 8×8 product)
        #   - T10 = solve(I, 0) = 0 (skip 28-MAC forward sub)
        # And for t_input ≤ 4: M00's bottom 4 rows are also zero, so T00 rows
        # 4..7 collapse to diag(β[r]) directly (skip 4 forward-sub iterations).
        # The const_expr branches are compile-time at JIT (3 specializations:
        # t_input ∈ {4, 8, 16}); at t_input=16 the SKIP branch is dead-code
        # eliminated. sync_threads() topology is preserved across paths.
        # ============================================================

        # === Step 1: parallel 8x8 diagonal forward substitutions (register-resident) ===
        if const_expr(self._t_input <= 8):
            # (u-cache) scores GEMM on warps 2-3, CONCURRENT with the
            # warp-0/1 block inverse: scores[j, col] = khist_j · packed_col
            # (cols 0..t-1 = ·k_hat, cols 8..8+t-1 = ·q_hat), staged
            # w-scaled + transposed + bf16 into sWScores (the contraction
            # MMA's A operand). Same MMA pattern as KKT with A = khist.
            # (perf) skipped entirely at P=0 (CTA-uniform branch): the
            # contraction is also skipped there, so sWScores has no reader.
            if warp_id >= 2 and P_hist > 0:
                # own-wave wait: each of the 64 issuing threads drains its
                # own cp.async groups, then the named barrier orders the
                # cross-warp (2<->3) SMEM visibility. Warps 0-1 are already
                # deep in the block inverse at this point.
                _cp_async_wait_group_0()
                _bar_sync_1_64()
                acc.fill(f32(0.0))
                _sc_col_off = (warp_id - Int32(2)) * Int32(8)
                for _sc_g in cutlass.range_constexpr(K_DIM // 16 // 4):
                    _sc_k_off = _sc_g * 4 * 16 * Int32(2)
                    _sc_a = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + _sc_k_off
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
                _sc_r0 = lane_id // 4
                _sc_c0 = (lane_id & 3) * 2
                # (perf) stage TRANSPOSED + w-scaled + bf16: acc element [e]
                # is C[j, col] with j = _sc_r0 (+8 for e in {2,3}); store
                # sWScores[col, j] = w_j * C[j, col]. Columns j >= P carry
                # w = 0, so the stale u rows they meet in the contraction
                # MMA contribute exact zeros.
                _sc_w_lo = sGhist.iterator[W_RING + _sc_r0]
                _sc_w_hi = sGhist.iterator[W_RING + _sc_r0 + 8]
                sWScores.iterator[(_sc_col_off + _sc_c0) * BF_PAD + _sc_r0] = (
                    acc.iterator[0] * _sc_w_lo
                ).to(io)
                sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * BF_PAD + _sc_r0] = (
                    acc.iterator[1] * _sc_w_lo
                ).to(io)
                sWScores.iterator[(_sc_col_off + _sc_c0) * BF_PAD + _sc_r0 + 8] = (
                    acc.iterator[2] * _sc_w_hi
                ).to(io)
                sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * BF_PAD + _sc_r0 + 8] = (
                    acc.iterator[3] * _sc_w_hi
                ).to(io)
            # === T≤8 PATH: only T00 (warp 0) is real work; T11 = diag(β1) ===
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t00 = [None] * 8
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    if const_expr(self._t_input <= 4):
                        # Real forward-sub for rows 0..3; rows 4..7 collapse to diag(β[r])
                        for _r in cutlass.range_constexpr(1, 4):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                        for _r in cutlass.range_constexpr(4, 8):
                            _x_t00[_r] = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                    else:
                        # T=8: real forward-sub all 8 rows
                        for _r in cutlass.range_constexpr(1, 8):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                    # Spill T00 column to sMat[0:8, col]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                # T11 = diag(β1) — 28 forward-sub MACs collapse to a single column write.
                # Step 4 (sTmat stage) reads sMat[8:16, 8:16] from this region.
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        _v = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _v
            # No barrier here: in the t<=8 path there are zero cross-warp
            # dependencies between the phase-2 barrier and the final solve
            # barrier — step 2 is a no-op skip and each warp writes disjoint
            # sMat regions that are only read after the CTA-wide barrier
            # before Step 4.

            # === Step 2 SKIP: Y = M10 @ T00 = 0 ===
            # sMat[0:8, 8:16] (top-right) — no need to write zeros: Step 4's
            # stage line `_out0_v11 = io(0.0) if (_r0_v11 < 8 and _c0_v11 >= 8)`
            # already forces sTmat top-right to 0 regardless of sMat content.

            # === Step 3 SKIP: T10 = solve(I, 0) = 0 → write zeros to sMat[8:16, 0:8] ===
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = f32(0.0)
            sync_threads()
        else:
            # === T=16 path (block inverse) ===
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    # X_t00[r] = T00[r, col] (lane-private, fp32 register)
                    _x_t00 = [None] * 8
                    # Row 0: T00[0, col] = (col==0) * beta0[0]
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            # M0[r, k] broadcast LDS (all 8 active lanes read same addr)
                            _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                            _accum = _accum + _m_rk * _x_t00[_k]  # register read
                        _x_t00[_r] = _accum
                    # Spill T00 column to sMat[0:8, col] for use by Y product
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t11 = [None] * 8
                    _x_t11[0] = (
                        sBeta.iterator[Int32(8)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t11[_k]
                        _x_t11[_r] = _accum
                    # Spill T11 to sMat[8:16, 8:16]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _x_t11[_r]
            sync_threads()

            # === Step 2: Y = M10 @ T00 → sMat[0:8, 8:16] ===
            # 64 outputs / 32 lanes, 2 passes × 4 rows per pass.
            if warp_id == Int32(0):
                for _p in cutlass.range_constexpr(2):
                    _i = Int32(_p * 4) + (lane_id >> Int32(3))
                    _j = lane_id & Int32(7)
                    _y_ij = f32(0.0)
                    for _k in cutlass.range_constexpr(8):
                        _m_ik = sTmat.iterator[
                            (Int32(8) + _i) * Int32(BF_PAD) + Int32(_k)
                        ].to(f32)
                        _t_kj = sMat.iterator[_smat_off(_k, _j)]
                        _y_ij = _y_ij + _m_ik * _t_kj
                    sMat.iterator[_smat_off(_i, 8 + _j)] = _y_ij
            sync_threads()

            # === Step 3: T10 = solve(I - M11, Y) → sMat[8:16, 0:8] (register-resident) ===
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t10 = [None] * 8
                    # Row 0: T10[0, col] = Y[0, col] = sMat[0 * T + 8 + col]
                    _x_t10[0] = sMat.iterator[_smat_off(0, 8 + _col)]
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = sMat.iterator[_smat_off(_r, 8 + _col)]
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t10[_k]
                        _x_t10[_r] = _accum
                    # Spill T10 to sMat[8:16, 0:8]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = _x_t10[_r]
            sync_threads()

        # === Step 4: stage final Tmat (bf16) to sTmat, zero top-right ===
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
        # (u-cache) tenant #3 of qv_buf: the u tile [W_RING, V_PADDED] bf16
        # via cp.async.cg (single consumer; keep L1 for k/q/v). khist's
        # last read was the scores GEMM (pre-Step-4 sync, so the tenant
        # handoff is ordered). Landing is awaited in the TMA half-1
        # shadow, right before the history contraction consumes it.
        # ============================================================
        _gUC_base = gUC.iterator.toint() + _uc_pool_e64 * 2
        _uc_base = Int32(0)
        _sV_base_async_u = sV.iterator.toint()
        for _uh in cutlass.range_constexpr(W_RING * V_DIM_C // (THREADS * 8)):
            _uh_group = tidx + _uh * THREADS
            _uh_row = _uh_group // Int32(V_DIM_C // 8)
            _uh_col = (_uh_group % Int32(V_DIM_C // 8)) * Int32(8)
            # (perf) same row < P predication as the khist wave — slots
            # >= P are w-masked; skip their DRAM reads.
            if _uh_row < P_hist:
                _cp_async_bf16x8_cg(
                    _gUC_base,
                    _uc_base + _uh_row * V_DIM + _uh_col,
                    _sV_base_async_u
                    + _uh_row * Int32(V_PADDED * 2)
                    + _uh_col * Int32(2),
                )
        _cp_async_commit_group()

        # ============================================================
        # bdec fold: scale the packed A-tile rows (k rows [0:t), q rows
        # [8:8+t)) by bdec = e^{G_P} so the H GEMM directly yields
        # bdec*(S0·x); the history term joins via the contraction below. The
        # sync at the H-half-0 mbarrier wait publishes these writes.
        # ============================================================
        # Skipped at P=0: bdec = 1 there and _mul_bf16x2_f32 by 1.0 is
        # value-preserving, so the pass is a pure no-op — save the RMWs.
        if P_hist > 0:
            _bdec = sGhist.iterator[32]
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
                _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col] = _mul_bf16x2_f32(
                    _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col], _bdec
                )

        # The V load is issued later, in the TMA half-1 shadow below — sV's
        # region (qv_buf) is occupied by the u tile until the history
        # contraction has read it.

        # ============================================================
        # Wait for the state tile to land in sH via mbarrier. The TMA store
        # uses the async proxy; ldmatrix uses the generic proxy —
        # fence_view_async_shared crosses the proxy boundary. We do NOT wait
        # for V here — its cp.async runs in parallel with the H GEMM below,
        # and wait_group(0) for V fires just before the QT@V consumer.
        # ============================================================
        cute.arch.mbarrier_wait(mbar_h_ptr, 0)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM: WH[16, 128] = A[16, 128] @ H^T, where A is the packed
        # [k; q] tile in sK. 4 warps x 4 V-groups (8 rows each) x 8 K-tiles
        # (16 K each) = 128 MMAs / 4 warps = 32 MMAs per warp.
        # ============================================================
        wh_acc_0 = cute.make_fragment_like(tCsC)
        wh_acc_0.fill(f32(0.0))
        wh_acc_1 = cute.make_fragment_like(tCsC)
        wh_acc_1.fill(f32(0.0))
        wh_acc_2 = cute.make_fragment_like(tCsC)
        wh_acc_2.fill(f32(0.0))
        wh_acc_3 = cute.make_fragment_like(tCsC)
        wh_acc_3.fill(f32(0.0))

        _sK_base_vl = sK.iterator.toint()  # A operand (packed tile in sK, K_PADDED stride)
        _sH_base_vl = sH.iterator.toint()  # B operand (state in sH, SW128-swizzled, half-K)
        _rs_a = Int32(K_PADDED * 2)  # 272 — sK row stride (padded, full K)
        _rs_b = Int32(K_HALF * 2)  # 128 — sH row stride (half-K, SW128)

        # Per-warp V-group base (warp_id * 32 V-rows). For B-fragment ldmatrix.x2:
        # lane_id (0..31) maps to (lane%8) row × ((lane//8)%2) 16-col group.
        _b_lane_row = lane_id % Int32(8)
        _b_col_inner = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)  # 0 or 16 bytes
        _vg_base_row = warp_id * Int32(32)

        # ============================================================
        # H GEMM HALF-0 (ka=0..3, uses sH for K=0..63). sH currently holds
        # the state's K=0..63 columns from the first TMA load.
        # ============================================================
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32(ka_local * 16 * 2)  # sK K=0..63
            col_byte_off_b = Int32(ka_local * 16 * 2)  # sH K=0..63
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
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]

        # ============================================================
        # Issue the SECOND state-tile half (K=64..127, overwrites sH). The
        # sync_threads ensures ALL warps finished reading half-0; then warp 0
        # issues the second TMA and the mbarrier parity flips to 1.
        # ============================================================
        sync_threads()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # HISTORY CONTRACTION — placed in the TMA half-1 shadow. Consumes the
        # u tile (viewed through sV) and the w-scaled transposed scores tile,
        # accumulating hw[r, v] += sum_j sWScores[r, j] * u[j, v] into the
        # wh_acc fragments via one _qtv_4mma. r = {r0 (k rows -> hw_k),
        # r0+8 (q rows -> hw_q)} matches the packed A-tile row map, so the H
        # GEMM and this MMA land in the same accumulator elements.
        # ============================================================
        _cp_async_wait_group_0()
        sync_threads()
        # MMA-based history contraction:
        #   hw[r, v] += sum_j sWScores[r, j] * u[j, v]
        # A = the w-scaled transposed scores tile (bf16, staged by the scores
        # GEMM), B = the u tile (in sV's region) — the _qtv_4mma pattern,
        # whose output fragments match wh_acc exactly. Skipped at P=0
        # (CTA-uniform; wh_acc keeps only the S0 term).
        if P_hist > 0:
            _hc_a0, _hc_a1, _hc_a2, _hc_a3 = _ldmatrix_x4(sWScores, lane_id)
            _hc_b_base = (
                _sV_base_async_u
                + _ldm_row * Int32(V_PADDED * 2)
                + warp_id * Int32(64)
            )
            _hcr = _qtv_4mma(_hc_a0, _hc_a1, _hc_a2, _hc_a3, _hc_b_base)
            wh_acc_0.iterator[0] = wh_acc_0.iterator[0] + _hcr[0]
            wh_acc_0.iterator[1] = wh_acc_0.iterator[1] + _hcr[1]
            wh_acc_0.iterator[2] = wh_acc_0.iterator[2] + _hcr[2]
            wh_acc_0.iterator[3] = wh_acc_0.iterator[3] + _hcr[3]
            wh_acc_1.iterator[0] = wh_acc_1.iterator[0] + _hcr[4]
            wh_acc_1.iterator[1] = wh_acc_1.iterator[1] + _hcr[5]
            wh_acc_1.iterator[2] = wh_acc_1.iterator[2] + _hcr[6]
            wh_acc_1.iterator[3] = wh_acc_1.iterator[3] + _hcr[7]
            wh_acc_2.iterator[0] = wh_acc_2.iterator[0] + _hcr[8]
            wh_acc_2.iterator[1] = wh_acc_2.iterator[1] + _hcr[9]
            wh_acc_2.iterator[2] = wh_acc_2.iterator[2] + _hcr[10]
            wh_acc_2.iterator[3] = wh_acc_2.iterator[3] + _hcr[11]
            wh_acc_3.iterator[0] = wh_acc_3.iterator[0] + _hcr[12]
            wh_acc_3.iterator[1] = wh_acc_3.iterator[1] + _hcr[13]
            wh_acc_3.iterator[2] = wh_acc_3.iterator[2] + _hcr[14]
            wh_acc_3.iterator[3] = wh_acc_3.iterator[3] + _hcr[15]

        # Next tenant of qv_buf: the V tile. The u tile's last read was the
        # loop above; one barrier orders the handoff, then the V load runs
        # here (its wait stays at the tail, so the transfer overlaps the H
        # GEMM half-1).
        sync_threads()
        _gV_base = gV.iterator.toint()
        _v_base_bf16 = pid_b * sv_b + pid_hv * sv_hv
        _v_iters = 1 if self._t_input <= 8 else (T * V_DIM_C // (THREADS * 8))
        for i in cutlass.range_constexpr(_v_iters):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col_bf16_async = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col_bf16_async * Int32(
                2
            )
            # v holds only n_valid rows; skip rows >= n_valid (OOB). Tail
            # zeroing happens after the V wait below.
            if const_expr(self._n_valid < T):
                if _v_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gV_base,
                        _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                        _sV_base_async_u + _smem_byte_off_v,
                    )
            else:
                _cp_async_bf16x8(
                    _gV_base,
                    _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                    _sV_base_async_u + _smem_byte_off_v,
                )
        _cp_async_commit_group()

        # QT = sNegL @ sTmat, computed in the half-1 shadow on warps 0-1 into
        # sWScores (dead after the contraction MMA above; the sync above
        # orders the handoff, and the R-pass barrier below publishes QT before
        # the y GEMM reads it). Associativity takes U off the output critical
        # path: y_intra = sNegL @ (sTmat @ R) = QT @ R.
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
            sWScores.iterator[_qt_r0 * BF_PAD + _qt_col_off + _qt_c0] = acc.iterator[
                0
            ].to(io)
            sWScores.iterator[_qt_r0 * BF_PAD + _qt_col_off + _qt_c0 + 1] = (
                acc.iterator[1].to(io)
            )
            sWScores.iterator[(_qt_r0 + 8) * BF_PAD + _qt_col_off + _qt_c0] = (
                acc.iterator[2].to(io)
            )
            sWScores.iterator[(_qt_r0 + 8) * BF_PAD + _qt_col_off + _qt_c0 + 1] = (
                acc.iterator[3].to(io)
            )

        # Wait for second half to land before H GEMM half-1.
        cute.arch.mbarrier_wait(mbar_h_ptr, 1)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM HALF-1 (ka=4..7, uses sH for K=64..127). sH was overwritten
        # by the second TMA, so the col offset into sH RESETS; the sK col
        # offset advances (sK still has the full K_DIM=128 layout).
        # ============================================================
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32((4 + ka_local) * 16 * 2)  # sK K=64..127
            col_byte_off_b = Int32(ka_local * 16 * 2)  # sH K=0..63 (reset!)
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
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]

        # ============================================================
        # (u-cache) TAIL: R → U (ring append) → y = e^G∘hw_q + F^T·U
        # ============================================================
        _sV_base = sV.iterator.toint()
        _gOut_base = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        _v_off_base = Int32(0)  # full V in one tile
        # h_buf staging (sH dead after the half-1 H GEMM): ONE [16, V_PADDED]
        # tile at +8192 — rows [0:8) stage the output, rows [8:8+t) stage the
        # U ring append (dead rows at t<=8), so a single barrier + a single
        # flush region covers both.
        _sOutStage_base = sH.iterator.toint() + Int32(8192)

        # Wait for the V cp.async (tenant #4, issued in the half-1 shadow;
        # the wait drains whatever didn't finish under the H GEMM half-1).
        _cp_async_wait_group_0()
        # Zero the sV working-set tail rows [n_valid:8]. Rows [8:16) stay
        # stale-but-finite — safe: the U GEMM's A (sTmat) has exact-zero
        # cols >= t, so those rows cannot propagate.
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, 8):
                sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        sync_threads()

        # R = V − e^{G_s} ∘ hw_k, in place over sV rows [0:t). hw_k lives
        # in wh_acc elements {0,1} (fragment rows r0 = packed k rows);
        # beta is folded into sTmat, NOT applied here.
        _y_r0 = lane_id // Int32(4)
        _y_c0 = (lane_id & Int32(3)) * Int32(2)
        if _y_r0 < Int32(self._t_input):
            _neg_eg = f32(0.0) - sGamma.iterator[T + _y_r0]
            # (perf) i32-paired RMW through the sQ i32 view (same buffer and
            # row stride as sV: K_PADDED == V_PADDED) — 4 load+store pairs
            # instead of 16 scalar bf16 chains; fma(-e^G, hw, v) is the
            # fused form of the previous v - e^G*hw.
            _r_i32 = (
                _y_r0 * Int32(V_PADDED // 2)
                + warp_id * Int32(16)
                + (lane_id & Int32(3))
            )
            _sQ_i32.iterator[_r_i32] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32],
                _neg_eg,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 4] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 4],
                _neg_eg,
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 8] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 8],
                _neg_eg,
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 12] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 12],
                _neg_eg,
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
            )
        sync_threads()

        # y FIRST, via associativity: y_intra = QT @ R with QT = sNegL @ sTmat
        # precomputed into sWScores in the half-1 shadow (B = R). U is computed
        # AFTER the output staging (off the critical path).
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sWScores, lane_id)
        _qtv_base = _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        _qtvr = _qtv_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _qtv_base)
        # U = sTmat @ R back-to-back with the y MMA (same B = sV(R)); its rows
        # [0:t) are staged into OutStage rows [8:8+t) below, so the single
        # pre-flush barrier publishes output AND ring-append data.
        _u_a0, _u_a1, _u_a2, _u_a3 = _ldmatrix_x4(sTmat, lane_id)
        _ur = _qtv_4mma(_u_a0, _u_a1, _u_a2, _u_a3, _qtv_base)

        _eg_yq = sGamma.iterator[T + _y_r0]
        for h_iter in cutlass.range_constexpr(4):
            h = warp_id * 4 + h_iter
            acc.iterator[0] = _qtvr[h_iter * 4]
            acc.iterator[1] = _qtvr[h_iter * 4 + 1]
            acc.iterator[2] = _qtvr[h_iter * 4 + 2]
            acc.iterator[3] = _qtvr[h_iter * 4 + 3]
            # + e^{G_s} ∘ hw_q: hw_q lives in wh_acc elements
            # {2,3} (fragment rows r0+8 = packed q rows); it lands on the
            # y elements {0,1} (output rows r0 = window tokens). Fragment
            # rows r0+8 of y are garbage — neither staged (t<=8) nor
            # stored (STG row-gated), so elements {2,3} get no add.
            if h_iter == 0:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_0.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_0.iterator[3]
            if h_iter == 1:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_1.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_1.iterator[3]
            if h_iter == 2:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_2.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_2.iterator[3]
            if h_iter == 3:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_3.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_3.iterator[3]
            # SMEM-staged epilogue: stage the [T,128] tile in SMEM (h_buf — sH
            # is dead after the half-1 H GEMM; the sync at the QT@V wait above
            # orders all warps past it), then flush with fully-coalesced 16-B
            # STGs below. The STS pattern (word = 68*r + 4*h + lane%4) is
            # bank-conflict-free, avoiding the uncoalesced fragment-direct
            # 4-B stores.
            _out_r0 = lane_id // 4
            _out_c0 = (lane_id & 3) * 2
            _stg_col = h * 8 + _out_c0
            _sts_bf16x2_f32(
                _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )
            if const_expr(self._t_input > 8):
                _sts_bf16x2_f32(
                    _sOutStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 2,
                    acc.iterator[2],
                    acc.iterator[3],
                )

        # Stage U rows [0:t) at OutStage rows [8:8+t) (dead rows at t<=8):
        # frag elements {0,1} hold U row _y_r0. Published by the same
        # pre-flush barrier as the output rows.
        if _y_r0 < Int32(self._t_input):
            for _ug in cutlass.range_constexpr(4):
                _uu_col = (warp_id * Int32(4) + Int32(_ug)) * Int32(8) + _y_c0
                _sts_bf16x2_f32(
                    _sOutStage_base + ((8 + _y_r0) * V_PADDED + _uu_col) * 2,
                    _ur[_ug * 4],
                    _ur[_ug * 4 + 1],
                )

        # Coalesced flush: consecutive lanes write consecutive 16-B chunks
        # (16 chunks per 256-B row), so each warp covers 512 B contiguous —
        # 100% sector utilization. LDS.128 quarter-warps read 32 consecutive
        # SMEM words (pos*4 spans a full bank period) — conflict-free.
        sync_threads()
        for _fl_pass in cutlass.range_constexpr(2 if self._t_input > 8 else 1):
            _fl_chunk = _fl_pass * 128 + tidx
            _fl_row = _fl_chunk // 16
            _fl_pos = _fl_chunk & 15
            _fl_lds = _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
            _fl_off = _out_base + _fl_row * so_t + _v_off_base + _fl_pos * 8
            # LDS hoisted out of the runtime guard: tuple-unpack inside an
            # if-region trips a DSL region-type error, and reading staged
            # garbage rows (>= t_input) is harmless — only the STG is gated.
            _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
            if const_expr(self._t_input >= 8):
                _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)
            else:
                if _fl_row < Int32(self._t_input):
                    _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)

        # (u-cache) ring append straight from OutStage rows [8:8+t) — same
        # barrier and staging tile as the output, so this overlaps the
        # output STGs above (no extra pass, no extra barrier).
        _uc_wr_base = _uc_base + P_hist * V_DIM
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
                _uc_wr_base + _uf_row * V_DIM + _uf_pos * Int32(8),
                _uv0,
                _uv1,
                _uv2,
                _uv3,
            )


# ============================================================================
# Public entry point — gated_delta_rule_mtp_ucache: the draft-token decode
# output PLUS the (k_cache, u_cache, g_cache, hist_len) history-ring append
# (see the module docstring for the ring contract). Native T in {4, 8} only.
#
# Consumes the state pool in its natural (pool, HV, V, K) bf16 layout (no
# external prepack) and computes the decode output. The checkpoint state is
# read-only (never written back); state-update / intermediate-cache / fused-
# recovery request flags raise NotImplementedError so a mis-routed caller
# fails loudly rather than silently returning wrong results.
#
# Input/output contract:
#   A_log [HV], a [B,T,HV], dt_bias [HV], q/k [B,T,H,K], v [B,T,HV,V], b [B,T,HV],
#   initial_state_source [pool,HV,V,K] bf16, initial_state_indices [B] int32,
#   scale, output [B,T,HV,V] -> returns output [B,T,HV,V] bf16.
# Requires SM90+ (TMA + mbarrier); K == V == 128.
# ============================================================================

_CACHE: dict = {}
# Persistent pre-zeroed T=16 input staging buffers for the T<16 path, keyed by
# (device, B, H, HK, HV, K, V, dtype, T). Reused across calls so short-T decode
# pays only a T-row copy-in (no per-call F.pad realloc/re-zero).
_STAGE: dict = {}
# When False, the T<16 path assumes the staging buffers already hold the current
# inputs and skips the per-call copy-in. Set this only when the producer writes
# q/k/v/a/b directly into the persistent T=16 buffers (the fixed-buffer serving
# pattern) or to benchmark the bare kernel. Default True = always safe drop-in.
_RESTAGE = True
# When set, the T<T_KERNEL path passes q/k to the kernel as the real [B,T,...]
# tensors (no host staging copy) and the kernel loads only those T rows + zeros
# its sK/sQ smem tail, removing the two big q/k gmem->gmem staging copies. v/a/b
# stay staged. Set SGLANG_GDN_WY_NATIVE_T=0 to restore full staging.
import os as _os

_NATIVE_T = _os.environ.get("SGLANG_GDN_WY_NATIVE_T", "1") != "0"
# (strided-qkv) read q/k/v directly from the fused conv-output column slices (token
# stride = conv_dim) instead of .contiguous()-materializing them. Removes the 3 big
# q/k/v copies from the verify region. Only valid on the native path (T in {4,8}).
_STRIDED_QKV = _os.environ.get("SGLANG_GDN_WY_STRIDED_QKV", "0") != "0"
# (native-a/b) read a/b directly from the real [B, n_valid, HV] tensors instead of staging
# them into T_KERNEL-row zero-padded buffers (removes the 2 a/b staging copies). Bit-exact on
# the compact [B,T] output: gamma is a causal prefix-sum, so the unloaded tail rows (which get
# log_alpha=0 instead of the staged-zero value) cannot affect rows 0..n_valid-1, and the tail
# output is discarded. Native path only (T in {4,8}).
# ON by default alongside _NATIVE_T (same validation); SGLANG_GDN_WY_NATIVE_AB=0
# restores a/b staging.
_NATIVE_AB = _os.environ.get("SGLANG_GDN_WY_NATIVE_AB", "1") != "0"
# Cache the bf16 cast of the per-layer CONSTANT weights A_log/dt_bias, keyed by
# storage identity (data_ptr, shape). They are persistent tensors passed every verify
# call; caching turns the per-call `.to(bf16)` into a one-time (warm-up) cast that does
# not appear in the captured CUDA graph. Safe for inference (weights never change).
_BF16_CACHE: dict = {}


def _cached_bf16(t):
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    # Key by the SOURCE TENSOR OBJECT's identity, evicted when the object dies —
    # NOT by data_ptr: the caching allocator recycles freed storage, so a
    # data_ptr key can return a STALE cast for a brand-new tensor that landed on
    # a recycled allocation (silent wrong A_log/dt_bias whenever the caller
    # recreates them, e.g. benches/tests). id() is safe here because the
    # weakref.finalize pop runs during the referent's destruction, before CPython
    # can reuse the id. Serving keeps the fast path: per-layer weights are
    # persistent objects, so hits return the same bf16 tensor (stable address —
    # required for CUDA-graph replay). In-place mutation of a cached source
    # tensor is not detected (same limitation as the previous key; these are
    # frozen inference weights).
    key = id(t)
    c = _BF16_CACHE.get(key)
    if c is None:
        c = t.to(torch.bfloat16).contiguous()
        _BF16_CACHE[key] = c
        weakref.finalize(t, _BF16_CACHE.pop, key, None)
    return c


def gated_delta_rule_mtp_ucache(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    accepted_steps: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    disable_output: bool = False,
    recovery_steps: int = 0,
    k_cache: Optional[torch.Tensor] = None,
    u_cache: Optional[torch.Tensor] = None,
    g_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> torch.Tensor:
    """GDN decode output + history-ring append (verify path, no state flush).

    Reads S0 (never written) and the (k_cache, u_cache, g_cache) ring at
    hist_len[b] filled slots, computes the T draft outputs, and appends the T
    new (normed-k, u, G) entries at ring slots [P, P+T) — speculatively;
    serving rewinds by setting hist_len = P + accepted. Ring contract in the
    module docstring; ring buffers must be ZERO-INITIALIZED at pool
    allocation. Native T in {4, 8} only.

    Returns ``output`` of shape ``[B, T, HV, V]`` (bf16).
    """
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    # --- reject request flags outside this kernel's contract ---
    if recovery_steps != 0:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache: recovery_steps>0 is not supported "
            "(fused recovery is not implemented)."
        )
    if disable_output:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache: disable_output=True (state-only mode) "
            "is not supported (this kernel always emits output)."
        )
    if intermediate_states_buffer is not None:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache: intermediate-state caching is not supported."
        )
    if accepted_steps is not None or ssm_state_indices is not None:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache: per-request K / FLA-scatter is not supported."
        )
    if output_state_indices is not None:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache: split-pool (output_state_indices) is not supported."
        )
    # softplus_beta / softplus_threshold are hardcoded in the kernel (beta=1,
    # overflow-safe exp); reject non-default values rather than silently ignoring.
    assert softplus_beta == 1.0, (
        f"softplus_beta={softplus_beta} not supported (kernel hardcodes beta=1.0)."
    )
    assert softplus_threshold == 20.0, (
        f"softplus_threshold={softplus_threshold} not supported (kernel ignores "
        "the threshold; pass 20.0 for signature compatibility)."
    )
    # The kernel always applies Q/K L2 normalization internally; reject False
    # rather than silently returning un-normalized-semantics results.
    assert use_qk_l2norm_in_kernel, (
        "gated_delta_rule_mtp_ucache: use_qk_l2norm_in_kernel=False is not supported "
        "(the kernel always applies Q/K L2 normalization)."
    )
    assert initial_state_source.dtype == torch.bfloat16, (
        f"initial_state_source must be bf16 (pool, HV, V, K); got {initial_state_source.dtype}."
    )

    B, T, H, K_dim = q.shape
    HV = v.shape[2]
    V_dim = v.shape[3]
    device = q.device
    assert K_dim == K_DIM and V_dim == V_DIM_C, (
        f"this kernel requires K==V=={K_DIM}; got K={K_dim}, V={V_dim}."
    )
    T_KERNEL = 16  # kernel is hardcoded T=16

    if scale is None:
        scale = 1.0 / math.sqrt(K_dim)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    else:
        initial_state_indices = initial_state_indices.contiguous()
    _io_dtype = q.dtype
    HK = k.shape[2]

    # A_log / dt_bias are read as bf16 by the kernel. They are per-layer constants, so
    # cache the bf16 cast by storage identity (one-time at warm-up; absent from the
    # captured graph). Falls back to a plain cast if already bf16-contiguous.
    A_log = _cached_bf16(A_log)
    dt_bias = _cached_bf16(dt_bias)
    def _inner_dense(t: torch.Tensor, name: str) -> bool:
        """Pools may be block-strided views (paged serving layouts, e.g. vLLM:
        dim-0 stride spans the whole multi-component page). Inner dims must be
        dense; returns whether the tensor is fully contiguous."""
        dense_inner = torch.empty(t.shape[1:], device="meta").stride()
        assert tuple(t.stride()[1:]) == tuple(dense_inner) and t.stride(0) >= (
            t.shape[1] * dense_inner[0] if t.dim() > 1 else 1
        ), (
            f"{name}: inner dims must be dense (block-strided dim 0 is OK); "
            f"got shape {tuple(t.shape)} strides {tuple(t.stride())}"
        )
        return t.is_contiguous()

    h0 = initial_state_source
    _pools_contig = _inner_dense(h0, "initial_state_source")

    # --- (u-cache) ring validation -----------------------------------------
    assert (
        k_cache is not None
        and u_cache is not None
        and g_cache is not None
        and hist_len is not None
    ), "gated_delta_rule_mtp_ucache: k_cache/u_cache/g_cache/hist_len are required."
    _pool = h0.shape[0]
    assert k_cache.dtype == torch.bfloat16
    _pools_contig &= _inner_dense(k_cache, "k_cache")
    assert tuple(k_cache.shape) == (_pool, HK, 16, K_dim), (
        f"k_cache must be [pool={_pool}, H={HK}, 16, K={K_dim}]; got {tuple(k_cache.shape)}"
    )
    assert u_cache.dtype == torch.bfloat16, (
        "v1 supports bf16 u_cache only (f32 is a documented extension point)."
    )
    _pools_contig &= _inner_dense(u_cache, "u_cache")
    assert tuple(u_cache.shape) == (_pool, HV, 16, V_dim)
    assert g_cache.dtype == torch.float32
    _pools_contig &= _inner_dense(g_cache, "g_cache")
    assert tuple(g_cache.shape) == (_pool, HV, 16)
    assert hist_len.dtype == torch.int32 and hist_len.shape[0] == B
    hist_len = hist_len.contiguous()
    if T not in (4, 8):
        raise NotImplementedError(
            f"gated_delta_rule_mtp_ucache: T={T} unsupported — native T in {{4, 8}} "
            "only (T=16 leaves no ring headroom; other T have no serving user)."
        )
    # n_valid = token rows actually present in the q/k tensors handed to the kernel.
    # Native-short-T (SGLANG_GDN_WY_NATIVE_T): pass q/k as the real [B,T,...] tensors
    # (n_valid=T); the kernel loads only those rows and zeros its sK/sQ smem tail,
    # skipping the two big q/k gmem->gmem staging copies. Otherwise q/k are staged
    # into a T_KERNEL-row zero-padded buffer (n_valid=T_KERNEL = original behavior).
    # Native-short-T is gated to T in {4, 8}: only there does T == t_disc, so (a) the
    # t_input-gated output STG writes exactly T rows (compact [B,T] output is safe) and
    # (b) the smem-tail zeroing aligns with the kernel's working-set masking. T=4 is the
    # draft-len-3 verify shape. Other T (1-3,5-7,9-15) fall back to full staging.
    # (u-cache) the kernel's ring math assumes t_input == T (native), so the
    # native path is FORCED here (SGLANG_GDN_WY_NATIVE_T is ignored).
    _native = True
    n_valid = T

    # Contiguity: tensors the kernel reads DIRECTLY from gmem need canonical-compact
    # strides for the CuTe descriptor; staged tensors get this for free from .copy_().
    _qkv_rs = 0  # >0 => strided q/k/v read (token stride = conv_dim); see _STRIDED_QKV
    _ab_native_flag = (
        False  # True => a/b read native [B,n_valid,HV] (no staging); _NATIVE_AB
    )
    _ab_ts = 0  # a/b token stride baked into the cubin (0 = contiguous HV)
    if T == T_KERNEL:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
    elif _native:
        # q/k/v are read natively (kernel loads n_valid rows + zeros its smem tail).
        # Default: .contiguous() so the kernel's compact-stride descriptor is valid.
        # Strided-qkv: q/k/v are the fused conv-output column slices (token stride =
        # conv_dim, features contiguous within a token). Pass that row stride to the
        # kernel and skip the copies — bit-identical (same values, strided gmem read).
        if _STRIDED_QKV and q.stride(1) == k.stride(1) == v.stride(1):
            _qkv_rs = q.stride(1)
        else:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

    # Block-strided (paged) pools bake their strides into the STATIC
    # descriptors of the strided mode; the compact-dynamic descriptors assume
    # contiguous pools. Same contract as the flush wrapper.
    if not _pools_contig and _qkv_rs == 0:
        raise NotImplementedError(
            "block-strided state pools require the static-descriptor mode: set "
            "SGLANG_GDN_WY_STRIDED_QKV=1 and pass q/k/v slices sharing a token "
            "stride (the compact-dynamic descriptors assume contiguous pools)."
        )

    # For T<T_KERNEL, stage the inputs into persistent, pre-zeroed T_KERNEL-row
    # buffers (keyed by shape AND T so rows [T:T_KERNEL) stay zero) and copy only the
    # T valid rows per call. Those zero rows are load-bearing: the kernel's
    # ldmatrix/MMA reads the full tile (NaN-tail probe confirmed). The native path
    # stages only a/b and moves the q/k/v zero-fill into the kernel (smem tail).
    if T < T_KERNEL:
        if (
            _native
            and _NATIVE_AB
            and a.stride(-1) == 1
            and tuple(b.stride()) == tuple(a.stride())
            and a.stride(0) == a.shape[1] * a.stride(1)
        ):
            # Native-a/b: pass the real [B, T(=n_valid), HV] tensors straight to the kernel
            # (batch stride = n_valid*HV, contiguous). The kernel gates the warp-3 load/compute
            # by n_valid, so no T_KERNEL zero-pad staging copy is needed. Bit-exact on the
            # compact [B,T] output (causal prefix-sum isolates the unloaded tail rows).
            _ab_native_flag = True
            _ab_ts = a.stride(1)
            # q, k, v and a, b all stay native [B, T, ...].
        elif _native:
            skey: tuple = (str(device), B, HV, str(_io_dtype), T, "ab")
            buf = _STAGE.get(skey)
            _fresh = buf is None
            if _fresh:
                with torch.inference_mode(False):
                    buf = (
                        torch.zeros(B, T_KERNEL, HV, dtype=_io_dtype, device=device),
                        torch.zeros(B, T_KERNEL, HV, dtype=_io_dtype, device=device),
                    )
                _STAGE[skey] = buf
            ab, bb = buf
            if _fresh or _RESTAGE:
                ab[:, :T].copy_(a)
                bb[:, :T].copy_(b)
            a, b = ab, bb
            # q, k, v stay as the native [B, T, ...] tensors.
        else:
            skey = (str(device), B, H, HK, HV, K_dim, V_dim, str(_io_dtype), T)
            buf = _STAGE.get(skey)
            _fresh = buf is None
            if _fresh:
                # Allocate OUTSIDE inference_mode so they are normal tensors;
                # otherwise the in-place .copy_() is rejected during CUDA-graph
                # capture ("Inplace update to inference tensor").
                with torch.inference_mode(False):
                    buf = (
                        torch.zeros(
                            B, T_KERNEL, H, K_dim, dtype=_io_dtype, device=device
                        ),
                        torch.zeros(
                            B, T_KERNEL, HK, K_dim, dtype=_io_dtype, device=device
                        ),
                        torch.zeros(
                            B, T_KERNEL, HV, V_dim, dtype=_io_dtype, device=device
                        ),
                        torch.zeros(B, T_KERNEL, HV, dtype=_io_dtype, device=device),
                        torch.zeros(B, T_KERNEL, HV, dtype=_io_dtype, device=device),
                    )
                _STAGE[skey] = buf
            qb, kb, vb, ab, bb = buf
            if _fresh or _RESTAGE:  # always fill a fresh buffer; else honor _RESTAGE
                qb[:, :T].copy_(q)
                kb[:, :T].copy_(k)
                vb[:, :T].copy_(v)
                ab[:, :T].copy_(a)
                bb[:, :T].copy_(b)
            q, k, v, a, b = qb, kb, vb, ab, bb

    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # One CTA per (b, hv) — full V tile per CTA. Per-CTA SMEM ~29.8 KB -> <=7 CTAs/SM.
    _total_ctas = HV * B
    _needed = math.ceil(_total_ctas / _num_sms)
    # min_blocks_per_mp (launch bounds) is capped at 8: mbp=8 makes the
    # compiler fit 64 regs/thread, unlocking the 7-CTA SMEM limit (29.8 KB/CTA).
    # Higher mbp gains no further occupancy (SMEM-capped at 7 CTAs) and is
    # slower — do not raise past 8.
    mbp = max(1, min(_needed + 1, 8))
    # GDN_WY_MBP overrides min_blocks_per_mp (launch bounds) for perf experiments.
    # mbp is part of cache_key, so each value compiles its own kernel.
    _mbp_env = _os.environ.get("GDN_WY_MBP")
    if _mbp_env:
        mbp = int(_mbp_env)
    # T-aware Phase-2 squaring depth.
    t_disc = 4 if T <= 4 else (8 if T <= 8 else 16)
    # n_valid in the key: native (n_valid<T) vs staged (n_valid=T_KERNEL) compile to
    # different kernels (different q/k batch stride + the smem-tail-zero path).
    # B / pool_size are NOT in the key: the batch (mode-0) dim of every per-batch
    # tensor is marked shape-dynamic below, so one cubin serves all batch and pool
    # sizes (grid derives B from gH0idx at launch; the H0 TMA descriptor takes the
    # pool extent at launch). mbp still varies with B, but only over <=4 buckets.
    # Exception: the strided-qkv opt-in path passes non-compact q/k/v whose
    # descriptors stay fully static, so it keeps B/pool in the key (fallback).
    # HV/H/V_dim MUST be in the key: they are runtime Int32 kernel args, but the
    # compiled artifact bakes in the captured tensors' layouts (H0 TMA descriptor,
    # q/k/v/out head+feature strides — only the batch mode-0 dim is dynamic), so a
    # process mixing HV values would otherwise reuse a cubin with the wrong strides.
    cache_key: tuple = (
        "ucache-v1",
        str(device),
        mbp,
        t_disc,
        n_valid,
        _qkv_rs,
        _ab_native_flag,
        _ab_ts,
        bool(use_pdl),
        HV,
        H,
        V_dim,
    )
    if _qkv_rs > 0:
        cache_key = cache_key + (B, h0.shape[0])
    mk = from_dlpack

    def mk_dyn(t):
        # Batch/pool-dynamic compact marking: mode-0 (leading) dim dynamic, inner
        # dims static so the kernel keeps constexpr tile geometry. Requires a
        # compact (contiguous) tensor — every tensor below is either staged into
        # a contiguous buffer or .contiguous()'d by this wrapper.
        if _qkv_rs > 0:
            return mk(t, 16)  # strided fallback: fully static descriptor
        return mk(t, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    # The kernel always writes a full T=16 output tile. If the caller did not
    # provide `output`, write into a fresh [B,16,HV,V] buffer and return a
    # [:, :T] VIEW (zero-copy). If the caller provided `output`, honor it:
    # T==16 writes straight in; T<16 uses a scratch tile and copies the T valid
    # rows back.
    _out_aliased = False
    if output is not None and T == T_KERNEL:
        out16 = output
        _out_aliased = True
    elif (
        output is not None
        and _native
        and output.shape == (B, T, HV, V_dim)
        and output.dtype == _io_dtype
        and output.is_contiguous()
    ):
        # (split) zero-copy alias: the kernel STGs write the caller's buffer
        # directly (retired pad-skip rows write nothing), so two masked
        # launches can share one output with no merge/copy.
        out16 = output
        _out_aliased = True
    elif _native:
        # Native (T in {4,8}): the STG writes exactly T rows, so a compact [B,T,HV,V]
        # output is correct. Returning it contiguous makes the caller's
        # reshape(1, B*T, ...) a free view instead of a ~5us materializing copy.
        out16 = torch.empty(B, T, HV, V_dim, dtype=_io_dtype, device=device)
    else:
        out16 = torch.empty(B, T_KERNEL, HV, V_dim, dtype=_io_dtype, device=device)

    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    args = [
        mk_dyn(q),
        mk_dyn(k),
        mk_dyn(v),
        mk_dyn(a),
        mk_dyn(b),
        mk(A_log, 16),
        mk(dt_bias, 16),
        mk_dyn(h0),
        mk_dyn(initial_state_indices),
        mk_dyn(k_cache),
        mk_dyn(u_cache),
        mk_dyn(g_cache),
        mk_dyn(hist_len),
        mk_dyn(out16),
        scale,
        HV,
        V_dim,
        H,
        stream,
    ]

    if cache_key not in _CACHE:
        _CACHE[cache_key] = cute.compile(
            GdnDecodeUCacheKernel(
                disable_state_update=True,
                min_blocks_per_mp=mbp,
                t_input=t_disc,
                n_valid=n_valid,
                qkv_row_stride=_qkv_rs,
                ab_native=_ab_native_flag,
                ab_t_stride=_ab_ts,
                launch_pdl=use_pdl,
            ),
            *args,
        )
    _CACHE[cache_key](*args)

    if output is None:
        return out16[:, :T]  # zero-copy view of the valid tokens
    if T < T_KERNEL and not _out_aliased:
        output.copy_(out16[:, :T])
    return output


__all__ = [
    "gated_delta_rule_mtp_ucache",
    "GdnDecodeUCacheKernel",
    "K_DIM",
    "V_DIM_C",
    "W_RING",
]
