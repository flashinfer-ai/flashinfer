"""CuTeDSL GDN raw-V-cache spec-decode kernel — verify + append + per-request flush.

Raw-v-cache sibling of the u-cache spec-decode kernels (PR #4081): the ring
caches RAW ``v``/``k`` (bf16, [pool, heads, 32, feat]) plus RAW ``a``/``b``
(**fp32**, [pool, HV, 32]) instead of the materialized ``u`` — the WY solve
reconstructs ``u`` on the fly. Ring and flush knobs match #4081: a 32-slot
Triton-ReplaySSM-compatible circular ring whose live window is
[cache_base, cache_base+hist_len) mod 32 with appends PAST the window,
``flush_min`` (threshold; P >= flush_min folds the window into the state),
and caller-owned cursor commits (``restart_hist_on_flush`` = standalone
convenience). IO/rings are bf16 (+fp32 a/b); the checkpoint STATE dtype is
bf16 by default or fp16 via GDN_VCACHE_STATE_DTYPE=fp16 (PR #4081's
fp16-state mode: higher-fidelity commits at identical bandwidth).

Per CTA (b, hv), the 16-row window [P committed | T draft | zero tail] is
assembled IN-KERNEL: committed rows stream from the ring caches via cp.async,
draft rows from the fresh [B, T, ...] tensors, dead-tail rows zero-fill via the
cp.async src-size operand. Committed a/b are read fp32 (full decay precision).
ALL ring appends (k/v/a/b) happen in-kernel: appends land past every live
window, so even the shared k ring can never race a sibling CTA's reads.
Draft outputs are written COMPACTED to [B, T, HV, V]. Fully CUDA-graph-safe.

PREDICTIVE flush model: flush_min, hist_len <= 16 - T so [committed | draft]
always fits one tile (draft outputs from S0 == S0' + draft); #4081's lazy
fold-before-output variant is a documented follow-up.

Pipeline structure (per CTA = one (request, v-head); one 16-row window):
  Phase 1  K/Q L2 norm; KKT / QKT (T x T GEMMs)
  Phase 2  log-depth Neumann inverse -> Tmat (the WY solve)
  Phase 3  A_full @ H0^T (TMA half-H, SW128-swizzled) + QT @ V -> outputs
  Phase 5  FLUSH (flushing CTAs only — per-CTA 0/1-trip dynamic loops):
    Step A  KHraw = k_norm @ H0^T      — piggybacks the Phase-3 H GEMM halves
    Step B  C = V - diag(exp(G_i))KHraw
    Step C  U = Tmat @ C               — T_P top-left = causal P-row solve
    Step D  Khat = diag(i<P ? exp(G_P - G_i) : 0) k_norm
    Step E  H0_new = exp(G_P) H0 + U^T Khat  — rank-P GEMM + in-place store
The WY math derives unchanged from the v18 wy_output_only decode kernel line
(inline-PTX ldmatrix/MMA, TMA half-H + mbarrier, SW128 swizzle, cp.async
staging); the vN tags in region comments record that lineage's measured
rationale. K == V == 128, 16-row window only. Requires SM90+.
"""

import math
import os
import weakref

import torch
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

# Fixed dims matching Triton v5
T = 16
K_DIM = 128
V_DIM_C = 128  # full V tile per CTA
BK_H = 16  # K-tile for H GEMM (must be a multiple of 16 for mma.k=16)
EPS = 1e-6
io = cutlass.BFloat16
f32 = cutlass.Float32
# Checkpoint-state (pool) element type: defaults to bf16 (the IO dtype). Set
# GDN_VCACHE_STATE_DTYPE=fp16 for the MIXED mode (PR #4081 parity): q/k/v
# inputs, the raw v/k/a/b rings, and the output stay bf16; only the checkpoint
# is stored fp16 (10 mantissa bits vs bf16's 7 at identical bandwidth).
# State-touching paths then run at higher fidelity: the H GEMM and the Step-A
# KH piggyback (B = the state tile) convert their four shared A-fragments
# bf16->f16 in registers (exact: every bf16 value is f16-representable in
# range — activations/normed-k never reach f16's 65504 limit) and issue
# .f16.f16 MMAs; the Step-E fold unpacks fp16 state pairs through f32 FMAs
# and repacks fp16.
_ST_ENV = os.environ.get("GDN_VCACHE_STATE_DTYPE", "").strip().lower()
if _ST_ENV in ("", "bf16", "bfloat16"):
    state_ty = cutlass.BFloat16
    ST_TORCH = torch.bfloat16
    _ST_MIXED = False
elif _ST_ENV in ("fp16", "float16", "half"):
    state_ty = cutlass.Float16
    ST_TORCH = torch.float16
    _ST_MIXED = True
else:
    raise ValueError(
        f"GDN_VCACHE_STATE_DTYPE={_ST_ENV!r} unsupported: use 'bf16' (default) "
        "or 'fp16'."
    )
if _ST_MIXED:
    _CVT_ST2_FROM_F32 = "cvt.rn.f16x2.f32"  # pack two f32 -> state pair
    _CVT_F32_FROM_ST = "cvt.f32.f16"
    _MMA_H_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
else:
    _CVT_ST2_FROM_F32 = "cvt.rn.bf16x2.f32"
    _CVT_F32_FROM_ST = "cvt.f32.bf16"
    _MMA_H_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
# H-GEMM A-fragment bf16->f16 in-register conversion (mixed mode only):
# widen each bf16 pair through f32 (shl/and bit tricks — exact) and repack
# f16x2, for all four shared A fragments. Empty in the bf16 mode.
_H_A_CVT_ASM = (
    "{ .reg .b32 _wl, _wh; .reg .f32 _fl, _fh;"
    " shl.b32 _wl, _a0, 16; and.b32 _wh, _a0, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a0, _fh, _fl;"
    " shl.b32 _wl, _a1, 16; and.b32 _wh, _a1, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a1, _fh, _fl;"
    " shl.b32 _wl, _a2, 16; and.b32 _wh, _a2, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a2, _fh, _fl;"
    " shl.b32 _wl, _a3, 16; and.b32 _wh, _a3, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a3, _fh, _fl; }"
    if _ST_MIXED
    else ""
)
WARP = 32
THREADS = 128
T_PAD = 16

# SMEM padding for sK/sQ — same as Path 1
K_HALF = K_DIM // 2  # 64 — v13 half-H streaming (sH = 16 KiB instead of 32 KiB)
K_PADDED = K_DIM + 8  # 136 — padded row stride for sK / sQ
V_PADDED = V_DIM_C + 8  # 136 — padded row stride for sV / sH (V rows, K cols)
W_RING_C = 16  # max history WINDOW rows (one 16-row MMA tile) — smem/tile constant
# Physical ring depth (Triton-ReplaySSM-compatible circular ring). The live
# window is [base, base+P) mod RING_SLOTS with P <= W_RING_C; appends land at
# (base+P+s) & RING_MASK — always PAST the window, so a flush never overwrites
# rows any sibling CTA is still reading (the old single-buffer restart race).
# Cursor commits (base slide / len reset) are CALLER-OWNED, outside the launch.
RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1
# sH layout: (V=128 rows, K_PADDED=136 cols) stored contiguous K-first
# → row_stride_bytes = K_PADDED * 2 = 272.

TK = T * K_DIM
TK_PAD = T * K_PADDED
TT = T * T
BF_PAD = 24


# ---------------------------------------------------------------------------
# PTX helpers (inherited unchanged from Path 1's kernel)
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


def _ldmatrix_x4_addr(addr):
    """ldmatrix.x4 from a caller-computed per-lane SMEM address (any row stride)."""
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


def _ld_global_v2_bf16(base_addr_i64, bf16_elem_offset):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32(), mlir_T.i32()]),
        [base_addr_i64.ir_value(), bf16_elem_offset.ir_value()],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2; ld.global.v2.b32 {$0,$1}, [_a]; }",
        "=r,=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
    )


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


def _cp_async_bf16x8_zfill(base_addr_i64, bf16_elem_offset, smem_addr_i32, src_sz_i32):
    """cp.async.ca, 16 B, with runtime src-size (16 = full copy, 0 = zero-fill).
    (vcache) Used for the in-kernel window assembly: every (row, col) slot issues
    exactly one cp.async — ring row, draft row, or zero-fill for the dead tail —
    keeping the issue loop uniform. When src_sz=0 the gmem address is still
    formed (PTX requires a valid address); callers pass a clamped default."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            src_sz_i32.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.ca.shared.global [$1], [_a], 16, $4;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r,r",
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


def _cp_async_bf16x8_cg_l2_128B(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """v5 (Hypothesis A): cp.async.cg with `.L2::128B` cache hint.
    The `.L2::128B` modifier asks the L2 to allocate a full 128-byte sector
    for this load. Since each cp.async is 16 B and 8 lanes contiguously cover
    a 128 B chunk of K, this should better-align L2 sector replacement and
    reduce cross-set contention seen as `set_conflicts` (120k cycles).
    If ptxas rejects this modifier, we fall back to plain .cg via the
    USE_L2_HINT toggle in the caller."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.cg.shared.global.L2::128B [$1], [_a], 16;"
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


def _cp_async_wait_group_n(n_const):
    """Wait until at most `n_const` cp.async groups remain in flight (constexpr int)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        f"{{ cp.async.wait_group {int(n_const)}; mov.u32 $0, 0; }}",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _st_global_bf16x2_f32(base_addr_i64, bf16_elem_offset, lo_f32, hi_f32):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            lo_f32.ir_value(),
            hi_f32.ir_value(),
        ],
        "{ .reg .u64 _addr; .reg .b32 _v;"
        " mad.wide.u32 _addr, $2, 2, $1;"
        " cvt.rn.bf16x2.f32 _v, $4, $3;"
        " st.global.b32 [_addr], _v;"
        " mov.u32 $0, 0; }",
        "=r,l,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_bf16x2_f32(smem_addr_i32, lo_f32, hi_f32):
    """v5 (H3): packed FP32 → BF16x2 cast + STS.32 to SMEM.
    Replaces a pair of (LDS f32 + F2FP.BF16 + STS.16) sequences with a
    single `cvt.rn.bf16x2.f32` + `st.shared.b32` for adjacent BF16 pairs.
    The cvt packs (hi, lo) into one 32-bit register; the store writes
    both bf16 values in a single 4-byte SMEM transaction.
    Halves the F2FP/STS instruction count for the Phase-2 sMat→sTmat
    refresh loops (NCU v4: 6 of top 10 short_scoreboard stalls were on
    F2FP.BF16 in this region)."""
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


def _ldg_v4_b32(base_addr_i64, bf16_elem_offset):
    """LDG.128: 16 B (8 bf16) from global. Offset in bf16 elements, 16-B aligned.
    (vcache) used by the k-ring append micro-kernel."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32()] * 4),
        [
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $5, 2, $4;"
        " ld.global.v4.b32 {$0,$1,$2,$3}, [_a]; }",
        "=r,=r,=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _lds_b32(smem_addr_i32):
    """ld.shared.b32 — one 4-byte (bf16x2) SMEM load at a raw address."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value()],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _bf16x2_to_f32x2(packed_i32):
    """Unpack a bf16x2 word into two f32 values (lo, hi)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 2),
        [packed_i32.ir_value()],
        "{ .reg .b16 _lo, _hi;"
        " mov.b32 {_lo, _hi}, $2;"
        " cvt.f32.bf16 $0, _lo;"
        " cvt.f32.bf16 $1, _hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
    )


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


def _afull_4mma(a0, a1, a2, a3, b_base):
    """4 independent (ldmatrix_B_trans + MMA) with shared A. B stride = 64 B (BK_H=32)."""
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
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+64];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$4,$5,$6,$7}, {$32,$33,$34,$35}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+128];"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "   {$8,$9,$10,$11}, {$32,$33,$34,$35}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+192];"
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
    B is row-major bf16 [8, 16] non-trans (rows = V, cols = K-tile of 16). Row-stride
    is row_stride_bytes_B. v6: B comes from SW128-swizzled sH (K_DIM*2=256 stride);
    callers MUST apply `_sw128_xor` to each per-lane B address before calling this.

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
        f"{_H_A_CVT_ASM}"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$33];"
        f" {_MMA_H_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$34];"
        f" {_MMA_H_F32}"
        "   {$4,$5,$6,$7}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$35];"
        f" {_MMA_H_F32}"
        "   {$8,$9,$10,$11}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_H_F32}"
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


def _st2_to_f32x2(packed_i32):
    """Unpack a STATE-dtype pair word into two f32 values (lo, hi).
    bf16 mode: identical to ``_bf16x2_to_f32x2``; fp16-state mode: cvt.f32.f16.
    Used only on Step E's H0 reads (the state tile)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 2),
        [packed_i32.ir_value()],
        "{ .reg .b16 _lo, _hi;"
        " mov.b32 {_lo, _hi}, $2;"
        f" {_CVT_F32_FROM_ST} $0, _lo;"
        f" {_CVT_F32_FROM_ST} $1, _hi; }}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
    )


def _sts_st2_f32(smem_addr_i32, lo_f32, hi_f32):
    """STS one STATE-dtype pair packed from two f32 (Step E H0_new STS-back).
    bf16 mode: identical to ``_sts_bf16x2_f32``; fp16-state: cvt.rn.f16x2.f32."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        f" {_CVT_ST2_FROM_F32} _v, $3, $2;"
        " st.shared.b32 [$1], _v; mov.u32 $0, 0; }"
        "",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sw128_xor(addr_i32):
    """Apply SW128 (cute.make_swizzle(3, 4, 3)) XOR to a logical SMEM byte
    address.

    SW128 spec: B=3 (3 bits XORed), M=4 (target = bits 4..6), S=3 (source =
    bits 7..9). Equivalent: phys = L XOR ((L >> 3) & 0x70).

    This MUST match the swizzle the SMEM tensor was built with (see
    `_make_sH_sw128_layout` and the SS h_buf alignment). Empirically verified
    against `debug/tma_standalone_repro_stage5.py` (max diff 0.0 on a
    natural-layout torch slice).
    """
    return addr_i32 ^ ((addr_i32 >> Int32(3)) & Int32(0x70))


def _make_sH_sw128_layout():
    """SW128 K-major BF16 SMEM layout for sH, tiled to (V_DIM_C=128, K_DIM=128).

    Pattern from `debug/tma_standalone_repro_stage5.py` and the canonical
    K_SW128 atom in cutlass.utils.blackwell_helpers. Required for both:
    1. TMA descriptor encoding (TMA dispatcher silently stalls without it
       on a 256-byte BF16 K-row box).
    2. Conflict-free SMEM layout for ldmatrix.x2 reads (the SW128 XOR
       eliminates the bank conflicts that a non-padded K=128 layout would
       have).
    """
    sw = cute.make_swizzle(3, 4, 3)  # SW128: <3, 4, 3>
    base = cute.make_layout((8, 64), stride=(64, 1))
    atom = cute.make_composed_layout(sw, 0, base)
    # order=(1, 0): tile inner-most along K first.
    return cute.tile_to_shape(atom, (V_DIM_C, K_DIM), order=(1, 0))


def _make_sH_sw128_layout_half():
    """v13 half-H: SW128 K-major BF16 layout tiled to (V_DIM_C=128, K_HALF=64).

    K_HALF=64 BF16 = exactly one 128-byte row = one SW128 swizzle period.
    The base atom (8 rows × 64 cols = 1 swizzle period) tiles 16x in V and
    1x in K. The XOR pattern operates at sub-lane granularity within each
    row, so the same `_sw128_xor()` helper works for both layouts.

    Saves 16 KiB SMEM vs full-H. sH is reused across 2 TMA-half loads
    (single-buffer streaming).
    """
    sw = cute.make_swizzle(3, 4, 3)
    base = cute.make_layout((8, 64), stride=(64, 1))
    atom = cute.make_composed_layout(sw, 0, base)
    return cute.tile_to_shape(atom, (V_DIM_C, K_HALF), order=(1, 0))


class GdnVcacheFlushKernel:
    """CuTeDSL GDN MTP decode + Phase-5 flush (state_and_output variant).

    Identical to GdnDecodeKernel (wy_output_only) through Phase 4, then adds
    Phase 5: fold the first P (per-request, runtime) window rows into H0 and
    write the updated state back to the pool in place. Launched only on flush
    iterations; t_input == n_valid == 16 (full left-packed window) only.
    """

    def __init__(
        self,
        min_blocks_per_mp=2,
        t_input=16,
        bv=None,
        n_valid=16,
        qkv_row_stride=0,
        ab_native=False,
        vcache_t_draft=0,
    ):
        # t_draft: the kernel assembles its 16-row window IN-KERNEL from the
        # ring caches (rows < P = hist_len[b]) and the [B, t_draft, ...] draft
        # tensors (rows [P, P+t_draft)), with rows >= P+t_draft zero-filled via
        # cp.async zfill. gP carries hist_len; the fold count is derived per
        # CTA (P if P >= flush_min else 0).
        self._vcache_t_draft = int(vcache_t_draft)
        assert self._vcache_t_draft > 0, (
            "this kernel is vcache-only (in-kernel ring window assembly); "
            "t_draft must be >= 1"
        )
        assert t_input == 16 and n_valid == 16, (
            "state_and_output variant supports only the full 16-row window"
        )
        assert qkv_row_stride == 0 and not ab_native, (
            "state_and_output variant supports only the compact/staged layout"
        )
        # `bv` is accepted only for bench-script signature compatibility — the
        # noprepack kernel always consumes the full V=128 tile in one CTA so
        # the V-split path of Path 1 is not implemented here.
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
        gP: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        gKC: cute.Tensor,
        gVC: cute.Tensor,
        gAC: cute.Tensor,
        gBC: cute.Tensor,
        scale: cutlass.Float32,
        flush_min: cutlass.Int32,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        op = MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        B_val = gH0idx.layout.shape[0]
        # v6: build TMA atom for the H tile. gH0 logical shape is
        # (pool, HV, V_DIM_C, K_DIM). cpasync.make_tiled_tma_atom tiles the
        # FIRST modes — we reorder modes to (V, K, HV, pool) by selecting
        # [2, 3, 1, 0] so the per-CTA tile is (V_DIM_C, K_DIM); the trailing
        # (HV, pool) modes survive tma_partition as outer iteration coords.
        # The SMEM target layout is SW128 swizzled — required for the TMA
        # descriptor (see results/v6_tma_debug_summary.md).
        gH0_vkhp = cute.make_tensor(
            gH0.iterator,
            cute.select(gH0.layout, mode=[2, 3, 1, 0]),
        )
        # v13: half-H TMA atom — box = (V_DIM_C, K_HALF) = 16 KiB.
        # Each CTA issues this atom TWICE (per K-half) into the SAME sH buffer
        # via a 2-phase mbarrier ping-pong. Saves 16 KiB SMEM/CTA → 5→6 CTAs/SM.
        sH_tma_layout = _make_sH_sw128_layout_half()
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gH0_vkhp,
            sH_tma_layout,
            (V_DIM_C, K_HALF),
        )
        # noprepack runs one CTA per (b, hv) — full V tile per CTA.
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
            gP,
            gBase,
            gOut,
            gKC,
            gVC,
            gAC,
            gBC,
            scale,
            flush_min,
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
        gP: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        gKC: cute.Tensor,
        gVC: cute.Tensor,
        gAC: cute.Tensor,
        gBC: cute.Tensor,
        scale: cutlass.Float32,
        flush_min: cutlass.Int32,
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
        sa_hv = cutlass.Int32(1)
        sa_t = HV
        sa_b = _ab_rows * HV
        sb_hv = cutlass.Int32(1)
        sb_t = HV
        sb_b = _ab_rows * HV
        # (vcache) draft tensors are the real [B, t_draft, ...] tensors — their
        # batch strides use t_draft, not n_valid/T. The 16-row window itself is
        # assembled in SMEM from ring + draft + zero rows.
        if const_expr(self._vcache_t_draft > 0):
            _T_D = Int32(self._vcache_t_draft)
            sq_b = self._vcache_t_draft * _qt
            sk_b = self._vcache_t_draft * _kt
            sv_b = self._vcache_t_draft * _vt
            sa_b = self._vcache_t_draft * HV
            sb_b = self._vcache_t_draft * HV
            # (vcache) COMPACT output: the epilogue writes only the T_D draft
            # rows (window rows [P, P+T_D)) at output rows [0, T_D) — the
            # caller gets [B, T_D, HV, V] directly, no gather.
            so_b = self._vcache_t_draft * HV * V_DIM
        # H0 natural layout: (pool, HV, V, K) bf16 contiguous → strides (HV*V*K, V*K, K, 1)
        # H0 layout strides — v6 unused (TMA handles addressing); kept for
        # clarity if future cycles need raw GMEM offsets.
        # sh_k = 1; sh_v = K_DIM; sh_hv = V_DIM * K_DIM; sh_pool = HV * V_DIM * K_DIM

        tidx, _, _ = cute.arch.thread_idx()
        _pid_vt, pid_hv, pid_b = cute.arch.block_idx()
        lane_id = tidx & 31
        warp_id = tidx // WARP

        # GQA head mapping
        i_h = pid_hv // (HV // H)
        cache_idx = gH0idx.iterator[pid_b]
        # Phase 5: number of committed window rows to fold into H0 for this
        # request. 0 => this CTA skips the H0_new store (compute still runs —
        # uniform control flow; the DSL rejects tuple-unpacking MMA helpers
        # inside runtime if-regions, so only scalar stores are predicated).
        p_val = gP.iterator[pid_b]
        # (vcache) gP carries hist_len: P_hist = committed ring rows (window
        # assembly always uses it); the FOLD count p_val is derived per CTA —
        # P_hist when this request flushes (P_hist >= flush_min), else 0 (the
        # Phase-5 store predication then skips it, state untouched).
        P_hist = p_val
        if const_expr(self._vcache_t_draft > 0):
            if P_hist < flush_min:
                p_val = Int32(0)
        # Per-CTA Phase-5 trip count: 1 for flushing CTAs, 0 otherwise. Every
        # Phase-5 compute block below is wrapped in `cutlass.range(_p5_iters)`
        # — a 0/1-trip DYNAMIC loop. Unlike runtime if-regions (which reject
        # tuple-unpacking MMA/ldmatrix helpers), dynamic loop bodies accept
        # them (probe-validated), so non-flushing CTAs genuinely SKIP the fold
        # MMAs instead of computing-and-discarding. p_val is CTA-uniform, so
        # sync_threads() inside the skipped regions stays uniform per CTA.
        _p5_iters = Int32(1)
        if p_val == Int32(0):
            _p5_iters = Int32(0)
        # Ring window origin (Triton cache_base semantics). All HISTORY ring
        # row addresses below are (ring_base + logical_row) & RING_MASK;
        # appends use (ring_base + P_hist + s) & RING_MASK — past the live
        # window. Smem/logical window rows stay 0..15 (unwrapped).
        ring_base = gBase.iterator[pid_b]

        # v12 EXP3h: hoist γβ LDGs to ABSOLUTE START of kernel (before SMEM setup).
        # Currently they're at line ~638, AFTER lots of SMEM struct definition and
        # TMA partition setup. Moving them to right after pid/lane setup lets the
        # compiler/scheduler emit them as the FIRST instructions warp 3 issues —
        # maximum HBM round-trip hiding window.
        _v7e_a_bf16 = f32(0.0)
        _v7e_b_bf16 = f32(0.0)
        _v7e_alog_bf16 = f32(0.0)
        _v7e_dt_bf16 = f32(0.0)
        # (native-a/b) load only the _ab_rows present in the a/b tensors (n_valid native,
        # T staged) so tail lanes never index past the real [B,n_valid,HV] rows.
        if warp_id == 3 and lane_id < _ab_rows:
            if const_expr(self._vcache_t_draft > 0):
                # (vcache) per-row source select. Committed rows (< P_hist) read
                # the fp32 a/b rings DIRECTLY — full decay/gate precision, no
                # bf16 round-trip. Draft rows read the fresh bf16 tensors. Rows
                # >= P_hist + T_D keep 0 (dead-tail; causal prefix-sum isolates
                # them). a_cache/b_cache layout: [pool, HV, 16] fp32.
                _abc_off = cache_idx * (HV * Int32(RING_SLOTS)) + pid_hv * Int32(
                    RING_SLOTS
                )
                if lane_id < P_hist:
                    _abc_ring = (ring_base + lane_id) & Int32(RING_MASK)
                    _v7e_a_bf16 = gAC.iterator[_abc_off + _abc_ring]
                    _v7e_b_bf16 = gBC.iterator[_abc_off + _abc_ring]
                if lane_id >= P_hist and lane_id < P_hist + _T_D:
                    _v7e_a_bf16 = gA.iterator[
                        pid_b * sa_b + (lane_id - P_hist) * sa_t + pid_hv * sa_hv
                    ].to(f32)
                    _v7e_b_bf16 = gB.iterator[
                        pid_b * sb_b + (lane_id - P_hist) * sb_t + pid_hv * sb_hv
                    ].to(f32)
            _v7e_alog_bf16 = gAlog.iterator[pid_hv].to(f32)
            _v7e_dt_bf16 = gDtbias.iterator[pid_hv].to(f32)

        smem = utils.SmemAllocator()

        @cute.struct
        class SS:
            # v6 TMA: mbarrier for H load. 8B Int64. Place first so its natural
            # 8B alignment is preserved by the prefix; large 128-aligned buffers
            # follow. ARRIVAL count = 1 (only one thread issues the TMA + arrive).
            h_load_mbar: cute.struct.MemRange[Int64, 1]
            k_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # v12.2: q_buf and v_buf ALIASED onto one 4352-B region (qv_buf).
            # K_PADDED == V_PADDED == 136 → T*K_PADDED == T*V_PADDED == 2176 bf16.
            # sQ last read = line ~961 (A_full RMW into sK); sV first cp.async
            # issue = line ~979. A sync_threads is added between them so all
            # threads finish reading sQ before any cp.async fill can write the
            # same bytes. Saves 4352 B per CTA.
            qv_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # v13 half-H: H tile = V=128 rows × K=64 cols bf16 = 16 KiB.
            # SW128 swizzled, single-buffered. Reused across 2 TMA loads
            # via mbarrier ping-pong. Saves 16 KiB SMEM/CTA vs v12.2.
            h_buf: cute.struct.Align[
                cute.struct.MemRange[state_ty, V_DIM_C * K_HALF], 128
            ]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            gamma: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            # v12.1: removed c_all (2 KiB) — was a sized template for partition_C
            # only, never written-to. Now we build tCsC via partition_shape_C +
            # make_fragment_C directly (no SMEM allocation needed).
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            scratch2_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            # Phase 5: snapshot of the L2-NORMED k rows (taken before the eK
            # scaling destroys them in sK; eK itself then dies to the A_full
            # overwrite). Used for the KH piggyback MMAs (A operand) and, row
            # scaled in place by exp(G_P - G_i), as Khat (Step E's B operand).
            k_snap_bf: cute.struct.Align[cute.struct.MemRange[io, T * K_PADDED], 128]

        st = smem.allocate(SS)
        sK = st.k_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # v12.2: sQ and sV both view the same qv_buf SMEM region (alias).
        sQ = st.qv_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # v13 half-H: sH = (V_DIM_C, K_HALF) SW128-swizzled = 16 KiB.
        # Same swizzle as full-H (exactly one swizzle period in K), so the
        # existing _sw128_xor() inline-PTX helper applies unchanged.
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sV = st.qv_buf.get_tensor(cute.make_layout((T, V_PADDED), stride=(V_PADDED, 1)))
        sTmat = st.tmat_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sGamma = st.gamma.get_tensor(cute.make_layout((WARP,)))
        sBeta = st.beta.get_tensor(cute.make_layout((WARP,)))
        # v12.1: sC removed — no SMEM C-staging buffer needed.
        sMat = st.mat_fp32.get_tensor(cute.make_layout((T, T), stride=(T, 1)))
        sNegL = st.scratch_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sPowk = st.scratch2_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        # Phase 5 views: k_snap [T, K_PADDED] (normed-k snapshot / later Khat);
        # sUT re-views the (dead by then) qv_buf as U^T [V_DIM_C, T] row-major
        # (2048 io <= TK_PAD): Step E's A operand (16-row V tiles x 16 P cols).
        sKsnap = st.k_snap_bf.get_tensor(
            cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1))
        )
        sUT = st.qv_buf.get_tensor(cute.make_layout((V_DIM_C, T), stride=(T, 1)))

        # ============================================================
        # v6 TMA: mbarrier init for the H tile load.
        # Single-shot: thread 0 issues cp.async.bulk.tensor once; all 128
        # threads block in mbarrier_wait(parity=0) before the H GEMM.
        # Arrival count = 1 (the issuing thread is the sole arriver; the
        # TX-bytes complete the barrier independently).
        # ============================================================
        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()
        sync_threads()

        # v13 half-H: partition the TMA tensor for this CTA's (cache_idx, pid_hv).
        # tma_tensor_h logical shape (V, K, HV, pool) (mode-reordered on host).
        # flat_divide with (V_DIM_C, K_HALF) → (V_TILE, K_TILE, V_REST=1, K_REST=2, HV, pool).
        # Two slices, one per K-half. Both halves write to the SAME sH (16 KiB).
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
        # v12.1: build register-only C fragment template (no SMEM).
        # partition_shape_C((T, 8)) returns the per-thread partition shape for
        # an (M=T, N=8) tile under the m16n8k16 MMA; make_fragment_C creates
        # a register fragment of that shape with the MMA's accumulator dtype (f32).
        tCsC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((T, 8)))
        acc = cute.make_fragment_like(tCsC)
        _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)

        EPT_TT = TT // THREADS

        # ============================================================
        # v7 variant E: hoist warp-3 γβ scalar LDG.E.U16 issues to the very
        # start of the kernel, BEFORE the K+Q cp.async issue. The BF16
        # loads (a_val, b_val, A_log_val, dt_val) are tiny (8 B/lane total)
        # but their LDG result-ready latency was previously serialized
        # inside the γβ block, costing ~600 cyc. By hoisting issue, the
        # results land in registers concurrent with cp.async issue/wait,
        # so when γβ math runs it consumes already-retired registers.
        # Materialized as a no-op (predicated to warp 3 + lane_id < T).
        # No SMEM writes here; results stay in regs and feed γβ block.
        # ============================================================
        # γβ LDGs were hoisted to top of __call__ (v12 EXP3h). _v7e_* vars live.

        # ============================================================
        # cp.async stage 1: K + Q (8 bf16 / instr, .ca for L1 reuse)
        # ============================================================
        k_base = pid_b * sk_b + i_h * sk_h
        q_base = pid_b * sq_b + i_h * sq_h
        _gK_base = gK.iterator.toint()
        _gQ_base = gQ.iterator.toint()
        if const_expr(self._vcache_t_draft > 0):
            # (vcache) ring bases. k_cache layout [pool, H, 16, K] bf16: rows of
            # this CTA's k-head i_h start at cache_idx*(H*16*K) + i_h*(16*K).
            _gKC_base = gKC.iterator.toint()
            _kc_ring_base = cache_idx * (H * Int32(RING_SLOTS * K_DIM)) + (
                i_h * Int32(RING_SLOTS * K_DIM)
            )
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
            # (vcache) per-row source select: committed rows (< P_hist) fill
            # from the k ring; draft rows ([P_hist, P_hist+T_D)) fill from the
            # fresh [B, T_D, ...] tensors; dead-tail rows zero-fill via the
            # cp.async src-size operand (address stays valid: draft row 0).
            # q has no ring (not cached): committed + dead rows zero-fill.
            if const_expr(self._vcache_t_draft > 0):
                _k_src_base = _gK_base
                _k_src_off = k_base + _kq_col_bf16_async
                _k_sz = Int32(0)
                if _kq_row < P_hist:
                    _k_src_base = _gKC_base
                    _k_src_off = (
                        _kc_ring_base
                        + ((ring_base + _kq_row) & Int32(RING_MASK)) * Int32(K_DIM)
                        + _kq_col_bf16_async
                    )
                    _k_sz = Int32(16)
                if _kq_row >= P_hist and _kq_row < P_hist + _T_D:
                    _k_src_off = (
                        k_base + (_kq_row - P_hist) * sk_t + _kq_col_bf16_async
                    )
                    _k_sz = Int32(16)
                _cp_async_bf16x8_zfill(
                    _k_src_base,
                    _k_src_off,
                    _sK_base_async + _smem_byte_off,
                    _k_sz,
                )
                _q_src_off = q_base + _kq_col_bf16_async
                _q_sz = Int32(0)
                if _kq_row >= P_hist and _kq_row < P_hist + _T_D:
                    _q_src_off = (
                        q_base + (_kq_row - P_hist) * sq_t + _kq_col_bf16_async
                    )
                    _q_sz = Int32(16)
                _cp_async_bf16x8_zfill(
                    _gQ_base,
                    _q_src_off,
                    _sQ_base_async + _smem_byte_off,
                    _q_sz,
                )
        _cp_async_commit_group()  # group 0 = K+Q

        # ============================================================
        # v13 half-H TMA: ISSUE FIRST HALF (K=0..63).
        # Same hiding window as v6 — overlaps with Phase-1 + Phase-2.
        # The SECOND half is issued later (right before H GEMM half-1)
        # after H GEMM half-0 finishes reading sH.
        # mbarrier_arrive_and_expect_tx with V_DIM_C * K_HALF * 2 = 16384 B.
        # cute.copy is OUTSIDE elect_one (v6 lesson — deadlocks the GPU).
        # ============================================================
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # warp 3 computes gamma/beta in parallel with cp.async pipeline
        # v7 variant E: scalar LDG.E.U16 loads were hoisted to kernel
        # entry; here we consume the already-retired registers.
        # ============================================================
        if warp_id == 3:
            log_alpha = f32(0.0)
            beta_val = f32(0.0)
            # (native-a/b) gate by _ab_rows: tail lanes [n_valid:T] are not loaded (their
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
                # (local fix) store log-domain cumsum in the free sGamma[0:T] slots so the
                # decay matrix can be formed as exp(cumsum_r - cumsum_c) directly (bounded
                # <=1 for the causal r>=c region) instead of exp(cumsum_r)*exp(-cumsum_c),
                # whose exp(-cumsum_c) overflows to inf for strong real decay (large A_log)
                # -> 0*inf = NaN. Keeps exact math; fixes the NaN that broke MTP verify.
                sGamma.iterator[lane_id] = cumsum
                sBeta.iterator[lane_id] = beta_val
                sBeta.iterator[T + lane_id] = f32(1.0) / exp_g
            # (vcache) IN-KERNEL a/b ring append: the draft a/b values already
            # sit in this warp's _v7e regs as fp32 (lane == logical window
            # row); store lanes [P_hist, P_hist+T_D) to the fp32 rings at
            # (ring_base + lane) & RING_MASK — past the live window (ring
            # semantics: appends never overwrite committed rows).
            if const_expr(self._vcache_t_draft > 0):
                _abc_st = cache_idx * (HV * Int32(RING_SLOTS)) + pid_hv * Int32(
                    RING_SLOTS
                )
                if lane_id >= P_hist and lane_id < P_hist + _T_D:
                    _abw_ring = (ring_base + lane_id) & Int32(RING_MASK)
                    gAC.iterator[_abc_st + _abw_ring] = _v7e_a_bf16
                    gBC.iterator[_abc_st + _abw_ring] = _v7e_b_bf16

        # ============================================================
        # Wait for K+Q (group 0); H load is now driven by mbarrier (TMA),
        # not cp.async commit groups, so wait_group(0) is sufficient.
        # ============================================================
        _cp_async_wait_group_0()
        sync_threads()

        # (vcache) IN-KERNEL k ring append, from SMEM: sK holds the RAW
        # assembled window here (post-wait, PRE-norm — the ring caches raw k;
        # the L2 norm below scales sK in place). The writer CTA of each
        # k-head group STGs the draft rows [P_hist, P_hist+T_D) to ring rows
        # (ring_base + P_hist + s) & RING_MASK — past EVERY sibling CTA's
        # live window [base, base+P), so the write can never race a
        # committed-row read (the ring-semantics guarantee that replaced the
        # old k-append micro-kernel launch). LDS is unconditional (row
        # wrapped mod T_D keeps the address in [P_hist, P_hist+T_D) ⊂
        # [0,16)); only the STG is predicated (scalar call — legal in a
        # runtime if-region, unlike the tuple-unpack LDS).
        if const_expr(self._vcache_t_draft > 0):
            _kap_row = (tidx // Int32(K_DIM // 8)) % _T_D
            _kap_pos = tidx % Int32(K_DIM // 8)
            _kv0, _kv1, _kv2, _kv3 = _lds_v4_b32(
                _sK_base_async
                + (P_hist + _kap_row) * Int32(K_PADDED * 2)
                + _kap_pos * Int32(16)
            )
            _kap_ring = (ring_base + P_hist + _kap_row) & Int32(RING_MASK)
            if (pid_hv % (HV // H)) == 0 and tidx < Int32(
                self._vcache_t_draft * (K_DIM // 8)
            ):
                _st_global_v4_b32(
                    _gKC_base,
                    _kc_ring_base + _kap_ring * Int32(K_DIM) + _kap_pos * Int32(8),
                    _kv0,
                    _kv1,
                    _kv2,
                    _kv3,
                )

        # L2 norm for K (warps 0,1) and Q (warps 2,3) — t-aware warp skip.
        # At T<=8, warps 1 and 3 (which normalize rows 8..15) are dead-elided.
        # Rows 8..15 of sK/sQ feed only outputs that Stage B's Phase-2 already
        # gates: T11 written as diag(beta1) at line 907-914 (no read of sMat
        # [8..15, 8..15]), Y/T10 skipped, and Phase-4 STG bottom half at line
        # 1235 is t_input-gated.
        # Warps 0 and 2 still process 8 lane-rows even at T=4 — rows 4..7 of
        # sK/sQ are wrapper-zero-padded, so L2-norm is a harmless no-op
        # (zero * any_inv_norm = zero). Predicating individual lanes would
        # break the shuffle_sync (mask 0xFFFFFFFF requires all 32 warp lanes).
        # `self._t_input` is a Python int set at JIT compile time (line 485);
        # const_expr produces 2 specializations: t_input<=8 and t_input=16.
        if const_expr(self._t_input <= 8):
            if warp_id == Int32(0):
                norm_row = lane_id // 4
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
            if warp_id == Int32(2):
                norm_row = lane_id // 4
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
        else:
            # T=16 ORIGINAL PATH — unchanged.
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
                        partial,
                        Int32(lane_id ^ d),
                        Int32(0xFFFFFFFF),
                        Int32(0x1F),
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                inv_norm = inv_norm * scale
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        sync_threads()

        # Phase 5: snapshot the L2-normed k rows into sKsnap NOW — sK is about
        # to be scaled to eK in place and then overwritten by A_full. 16 rows x
        # 64 i32 (128 bf16 data cols; the 8-col pad is never read) = 1024 words
        # across 128 threads. First consumer is the KH piggyback in the H GEMM,
        # which sits behind several sync_threads — no extra barrier needed.
        _sKsnap_i32 = cute.recast_tensor(sKsnap, cutlass.Int32)
        if p_val > Int32(0):  # (skip) snapshot feeds only the Phase-5 fold
            for _sn in cutlass.range_constexpr(8):
                _sn_flat = tidx + _sn * THREADS
                _sn_row = _sn_flat >> 6
                _sn_col = _sn_flat & Int32(63)
                _sKsnap_i32.iterator[_sn_row * (K_PADDED // 2) + _sn_col] = (
                    _sK_i32.iterator[_sn_row * (K_PADDED // 2) + _sn_col]
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
                _a_base = (
                    _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                )
                _b_direct = (
                    _sK_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
                )
                (
                    acc.iterator[0],
                    acc.iterator[1],
                    acc.iterator[2],
                    acc.iterator[3],
                ) = _fused_ab_4mma_serial_brow(
                    _a_base,
                    _b_direct,
                    acc.iterator[0],
                    acc.iterator[1],
                    acc.iterator[2],
                    acc.iterator[3],
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
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[
                1
            ].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[
                2
            ].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = (
                acc.iterator[3].to(io)
            )
        sync_threads()

        # ============================================================
        # PHASE 2: log-depth Neumann inverse
        # ============================================================
        for idx in cutlass.range_constexpr(EPT_TT):
            flat = tidx + idx * THREADS
            r = flat // T
            c = flat % T
            # (local fix) stable decay: exp(cumsum_r - cumsum_c) directly (<=1 for r>c),
            # instead of sGamma[T+r]*sBeta[T+c] = exp(cumsum_r)*exp(-cumsum_c) (overflows).
            # exp_gij is only consumed for r>=c (below); r<c value is discarded.
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
        sync_threads()

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2

        # ============================================================
        # v11r BLOCK INVERSE for T=16 — register-resident forward sub.
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
            sync_threads()

            # === Step 2 SKIP: Y = M10 @ T00 = 0 ===
            # sMat[0:8, 8:16] (top-right) — no need to write zeros: Step 4's
            # stage line `_out0_v11 = io(0.0) if (_r0_v11 < 8 and _c0_v11 >= 8)`
            # already forces sTmat top-right to 0 regardless of sMat content.
            # NO sync_threads needed here either — but we keep one to preserve
            # the same barrier topology as the T=16 path (cheap; cluster=1).
            sync_threads()

            # === Step 3 SKIP: T10 = solve(I, 0) = 0 → write zeros to sMat[8:16, 0:8] ===
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = f32(0.0)
            sync_threads()
        else:
            # === T=16 ORIGINAL PATH (v11 block-inverse) ===
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
        # PRECOMPUTE QT and A_full (V-tile independent)
        # ============================================================
        # QT = QKTm @ Tmat → sPowk   [output path]
        if warp_id < 2:
            acc.fill(f32(0.0))
            col_off = warp_id * 8
            _qt_a_addr = (
                sNegL.iterator.toint()
                + _lane_mod16 * Int32(BF_PAD * 2)
                + _lane_hi
            )
            _qt_b_addr = (
                sTmat.iterator.toint()
                + _ldm_row * Int32(BF_PAD * 2)
                + col_off * Int32(2)
            )
            (
                acc.iterator[0],
                acc.iterator[1],
                acc.iterator[2],
                acc.iterator[3],
            ) = _fused_ab_1mma(
                _qt_a_addr,
                _qt_b_addr,
                acc.iterator[0],
                acc.iterator[1],
                acc.iterator[2],
                acc.iterator[3],
            )
            _r0 = lane_id // 4
            _c0 = (lane_id & 3) * 2
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[
                1
            ].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[
                2
            ].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = (
                acc.iterator[3].to(io)
            )

        # eK = exp(gamma) * K_normed → sK
        _ek_iters = (self._t_input * K_DIM) // (THREADS * 4)
        for i in cutlass.range_constexpr(_ek_iters):
            _ek_row = warp_id + i * (THREADS // WARP)
            _ek_exp = sGamma.iterator[T + _ek_row]
            _ek_base = _ek_row * _kpad_i32 + lane_id
            _sK_i32.iterator[_ek_base] = _mul_bf16x2_f32(
                _sK_i32.iterator[_ek_base], _ek_exp
            )
            _sK_i32.iterator[_ek_base + WARP] = _mul_bf16x2_f32(
                _sK_i32.iterator[_ek_base + WARP], _ek_exp
            )
        sync_threads()

        # QT@eK → A_full residual contribution
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sPowk, lane_id)
        _sK_base_af = sK.iterator.toint()
        _af_b_base = (
            _sK_base_af + _ldm_row * Int32(K_PADDED * 2) + warp_id * Int32(16)
        )
        _afr = _afull_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _af_b_base)

        # A_full = eQ - QT@eK → sK (overwrite eK)
        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2
        _exp_eq_r0 = sGamma.iterator[T + _r0]
        _exp_eq_r8 = sGamma.iterator[T + _r0 + 8]
        BK_GROUPS = K_DIM // 32  # = 4
        for bk_idx in cutlass.range_constexpr(BK_GROUPS):
            k_col = bk_idx * 32 + warp_id * 8
            sK.iterator[_r0 * K_PADDED + k_col + _c0] = (
                _exp_eq_r0 * sQ.iterator[_r0 * K_PADDED + k_col + _c0].to(f32)
                - _afr[bk_idx * 4]
            ).to(io)
            sK.iterator[_r0 * K_PADDED + k_col + _c0 + 1] = (
                _exp_eq_r0
                * sQ.iterator[_r0 * K_PADDED + k_col + _c0 + 1].to(f32)
                - _afr[bk_idx * 4 + 1]
            ).to(io)
            sK.iterator[(_r0 + 8) * K_PADDED + k_col + _c0] = (
                _exp_eq_r8
                * sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0].to(f32)
                - _afr[bk_idx * 4 + 2]
            ).to(io)
            sK.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1] = (
                _exp_eq_r8
                * sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1].to(f32)
                - _afr[bk_idx * 4 + 3]
            ).to(io)

        # v12.2: CRITICAL barrier — sQ aliases sV in SMEM. All threads must
        # finish reading sQ above BEFORE any thread issues cp.async writes
        # to sV (same bytes). Without this, async fills can land in sV
        # while other lanes are still mid-read on sQ → silent corruption.
        sync_threads()

        # ============================================================
        # v10: Load V tile via cp.async (LDGSTS) so the V transfer
        # overlaps with the H GEMM below. Mirrors the K+Q LDGSTS
        # pattern at lines 657-666. sV is non-swizzled, 128-aligned,
        # so cp.async.ca writes directly with no swizzle transform.
        # 8 BF16 / instruction × 2 iters / thread × 128 threads = 2048 BF16.
        # ============================================================
        # v17 t-aware: at T<=8 skip iter 2 (rows 8..15). Safe because the
        # QT@V MMA reduction's k>=8 contribution is `sPowk[i, k] * sV[k, v]`
        # with sPowk[0..t_input-1, 8..15] proven zero (sK[8..15]=wrapper-zero
        # → QKT warp 3 (B=sK[8..15]) → sNegL[*, 8..15]=0 → sPowk[*, 8..15]=0
        # through QT=sNegL@sTmat). So sV[8..15] garbage * 0 = 0 — no
        # contamination of output rows 0..t_input-1. K cp.async is unchanged
        # so the sK[8..15]=wrapper-zero invariant that this proof relies on
        # is preserved.
        _gV_base = gV.iterator.toint()
        _sV_base_async = sV.iterator.toint()
        _v_base_bf16 = pid_b * sv_b + pid_hv * sv_hv
        if const_expr(self._vcache_t_draft > 0):
            # (vcache) v ring base. v_cache layout [pool, HV, 16, V] bf16.
            _gVC_base = gVC.iterator.toint()
            _vc_ring_base = cache_idx * (HV * Int32(RING_SLOTS * V_DIM_C)) + (
                pid_hv * Int32(RING_SLOTS * V_DIM_C)
            )
        _v_iters = 1 if self._t_input <= 8 else (T * V_DIM_C // (THREADS * 8))
        for i in cutlass.range_constexpr(_v_iters):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col_bf16_async = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col_bf16_async * Int32(
                2
            )
            # (vcache) committed rows from the v ring, draft rows from gV,
            # dead-tail rows zero-filled (Phase 5's C step reads sV rows < P,
            # so the tail must be literal zeros, not garbage).
            if const_expr(self._vcache_t_draft > 0):
                _v_src_base = _gV_base
                _v_src_off = _v_base_bf16 + _v_col_bf16_async
                _v_sz = Int32(0)
                if _v_row < P_hist:
                    _v_src_base = _gVC_base
                    _v_src_off = (
                        _vc_ring_base
                        + ((ring_base + _v_row) & Int32(RING_MASK)) * Int32(V_DIM_C)
                        + _v_col_bf16_async
                    )
                    _v_sz = Int32(16)
                if _v_row >= P_hist and _v_row < P_hist + _T_D:
                    _v_src_off = (
                        _v_base_bf16 + (_v_row - P_hist) * sv_t + _v_col_bf16_async
                    )
                    _v_sz = Int32(16)
                _cp_async_bf16x8_zfill(
                    _v_src_base,
                    _v_src_off,
                    _sV_base_async + _smem_byte_off_v,
                    _v_sz,
                )
        _cp_async_commit_group()  # group = V (K+Q's group already waited at line 717)

        # ============================================================
        # v6 TMA: wait for the H tile to land in sH via mbarrier.
        # The TMA store uses the async proxy; ldmatrix uses the generic
        # proxy — fence_view_async_shared crosses the proxy boundary.
        # NOTE: We do NOT wait for V here — V cp.async runs in parallel
        # with the H GEMM below. wait_group(0) for V fires just before
        # the QT@V consumer at line 1148.
        # ============================================================
        cute.arch.mbarrier_wait(mbar_h_ptr, 0)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM: WH[16, 128] = A_full[16, 128] @ H^T
        # 4 warps × 4 V-groups (8 rows each) × 8 K-tiles (16 K each)
        # = 128 MMAs / 4 warps = 32 MMAs per warp.
        # ============================================================
        wh_acc_0 = cute.make_fragment_like(tCsC)
        wh_acc_0.fill(f32(0.0))
        wh_acc_1 = cute.make_fragment_like(tCsC)
        wh_acc_1.fill(f32(0.0))
        wh_acc_2 = cute.make_fragment_like(tCsC)
        wh_acc_2.fill(f32(0.0))
        wh_acc_3 = cute.make_fragment_like(tCsC)
        wh_acc_3.fill(f32(0.0))

        # Phase 5 (Step A): KHraw = k_norm @ H0^T piggybacks on the H GEMM's
        # B-operand addresses while each sH half is resident — no extra H0
        # read. Row scale by exp(G_i) (-> KH = eK @ H0^T) happens later in the
        # C epilogue. Runs unconditionally (uniform control flow; P=0 CTAs
        # just never store).
        kh_acc_0 = cute.make_fragment_like(tCsC)
        kh_acc_0.fill(f32(0.0))
        kh_acc_1 = cute.make_fragment_like(tCsC)
        kh_acc_1.fill(f32(0.0))
        kh_acc_2 = cute.make_fragment_like(tCsC)
        kh_acc_2.fill(f32(0.0))
        kh_acc_3 = cute.make_fragment_like(tCsC)
        kh_acc_3.fill(f32(0.0))
        _sKsnap_int = sKsnap.iterator.toint()

        _sK_base_vl = sK.iterator.toint()  # A operand (A_full in sK, K_PADDED stride)
        _sH_base_vl = sH.iterator.toint()  # B operand (H in sH, SW128-swizzled, half-K)
        _rs_a = Int32(K_PADDED * 2)  # 272 — sK row stride (padded, full K)
        _rs_b = Int32(K_HALF * 2)  # 128 — v13: sH row stride (half-K, SW128)

        # Per-warp V-group base (warp_id * 32 V-rows). For B-fragment ldmatrix.x2:
        # lane_id (0..31) maps to (lane%8) row × ((lane//8)%2) 16-col group.
        _b_lane_row = lane_id % Int32(8)
        _b_col_inner = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)  # 0 or 16 bytes
        _vg_base_row = warp_id * Int32(32)

        # ============================================================
        # v13 half-H: H GEMM HALF-0 (ka=0..3, uses sH for K=0..63).
        # sH currently holds H[:, 0:64] from the first TMA load.
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

            for _p5a in cutlass.range(_p5_iters):  # (skip) Phase-5-only
                # Phase 5 Step A (half-0): KHraw += k_norm[:, K-tile] @ sH-half^T.
                _a_addr_kh = (
                    _sKsnap_int + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
                )
                _rkh = _h_gemm_4v(
                    _a_addr_kh,
                    _b0,
                    _b1,
                    _b2,
                    _b3,
                    kh_acc_0.iterator[0],
                    kh_acc_0.iterator[1],
                    kh_acc_0.iterator[2],
                    kh_acc_0.iterator[3],
                    kh_acc_1.iterator[0],
                    kh_acc_1.iterator[1],
                    kh_acc_1.iterator[2],
                    kh_acc_1.iterator[3],
                    kh_acc_2.iterator[0],
                    kh_acc_2.iterator[1],
                    kh_acc_2.iterator[2],
                    kh_acc_2.iterator[3],
                    kh_acc_3.iterator[0],
                    kh_acc_3.iterator[1],
                    kh_acc_3.iterator[2],
                    kh_acc_3.iterator[3],
                )
                kh_acc_0.iterator[0] = _rkh[0]
                kh_acc_0.iterator[1] = _rkh[1]
                kh_acc_0.iterator[2] = _rkh[2]
                kh_acc_0.iterator[3] = _rkh[3]
                kh_acc_1.iterator[0] = _rkh[4]
                kh_acc_1.iterator[1] = _rkh[5]
                kh_acc_1.iterator[2] = _rkh[6]
                kh_acc_1.iterator[3] = _rkh[7]
                kh_acc_2.iterator[0] = _rkh[8]
                kh_acc_2.iterator[1] = _rkh[9]
                kh_acc_2.iterator[2] = _rkh[10]
                kh_acc_2.iterator[3] = _rkh[11]
                kh_acc_3.iterator[0] = _rkh[12]
                kh_acc_3.iterator[1] = _rkh[13]
                kh_acc_3.iterator[2] = _rkh[14]
                kh_acc_3.iterator[3] = _rkh[15]

        # ============================================================
        # v13 half-H: ISSUE SECOND HALF TMA (K=64..127, overwrites sH).
        # sync_threads ensures ALL warps finished reading sH (half-0 done).
        # Then warp 0 issues the second TMA. Mbarrier parity flips to 1.
        # ============================================================
        sync_threads()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)

        # Wait for second half to land before H GEMM half-1.
        cute.arch.mbarrier_wait(mbar_h_ptr, 1)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # v13 half-H: H GEMM HALF-1 (ka=4..7, uses sH for K=64..127).
        # sH was overwritten by the second TMA — col offset into sH RESETS.
        # sK col offset advances (sK still has full K_DIM=128 layout).
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

            for _p5b in cutlass.range(_p5_iters):  # (skip) Phase-5-only
                # Phase 5 Step A (half-1): complete KHraw over K=64..127. The A
                # tile comes from sKsnap cols 64..127 (same col offset as sK's
                # A_full read); the B addresses already point at the re-filled sH.
                _a_addr_kh = (
                    _sKsnap_int + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
                )
                _rkh = _h_gemm_4v(
                    _a_addr_kh,
                    _b0,
                    _b1,
                    _b2,
                    _b3,
                    kh_acc_0.iterator[0],
                    kh_acc_0.iterator[1],
                    kh_acc_0.iterator[2],
                    kh_acc_0.iterator[3],
                    kh_acc_1.iterator[0],
                    kh_acc_1.iterator[1],
                    kh_acc_1.iterator[2],
                    kh_acc_1.iterator[3],
                    kh_acc_2.iterator[0],
                    kh_acc_2.iterator[1],
                    kh_acc_2.iterator[2],
                    kh_acc_2.iterator[3],
                    kh_acc_3.iterator[0],
                    kh_acc_3.iterator[1],
                    kh_acc_3.iterator[2],
                    kh_acc_3.iterator[3],
                )
                kh_acc_0.iterator[0] = _rkh[0]
                kh_acc_0.iterator[1] = _rkh[1]
                kh_acc_0.iterator[2] = _rkh[2]
                kh_acc_0.iterator[3] = _rkh[3]
                kh_acc_1.iterator[0] = _rkh[4]
                kh_acc_1.iterator[1] = _rkh[5]
                kh_acc_1.iterator[2] = _rkh[6]
                kh_acc_1.iterator[3] = _rkh[7]
                kh_acc_2.iterator[0] = _rkh[8]
                kh_acc_2.iterator[1] = _rkh[9]
                kh_acc_2.iterator[2] = _rkh[10]
                kh_acc_2.iterator[3] = _rkh[11]
                kh_acc_3.iterator[0] = _rkh[12]
                kh_acc_3.iterator[1] = _rkh[13]
                kh_acc_3.iterator[2] = _rkh[14]
                kh_acc_3.iterator[3] = _rkh[15]

        # v10: Wait for the V cp.async to land in sV. Needed by BOTH variants:
        # the QT@V ldmatrix (output path) and Phase-5's C = V - KH step.
        # Same proxy as K+Q LDGSTS — no fence_view_async_shared needed.
        _cp_async_wait_group_0()
        sync_threads()

        # (vcache) IN-KERNEL v ring append: the draft v rows are RESIDENT in
        # sV[P_hist : P_hist+T_D) (just waited above); copy them to the v ring
        # at (ring_base + P_hist + s) & RING_MASK — PAST the live window
        # (ring semantics: appends never overwrite committed rows). LDS is unconditional (row wrapped mod T_D, so
        # the address stays in [P_hist, P_hist+T_D) ⊂ [0,16) — reading a valid
        # row twice is harmless); only the STG is gated (tuple-unpack helpers
        # cannot sit inside a runtime if-region).
        if const_expr(self._vcache_t_draft > 0):
            _vap_row = (tidx // Int32(V_DIM_C // 8)) % _T_D
            _vap_pos = tidx % Int32(V_DIM_C // 8)
            _vap_addr = (
                sV.iterator.toint()
                + (P_hist + _vap_row) * Int32(V_PADDED * 2)
                + _vap_pos * Int32(16)
            )
            _va0, _va1, _va2, _va3 = _lds_v4_b32(_vap_addr)
            _gVC_st = gVC.iterator.toint()
            _vap_ring = (ring_base + P_hist + _vap_row) & Int32(RING_MASK)
            if tidx < Int32(self._vcache_t_draft * (V_DIM_C // 8)):
                _st_global_v4_b32(
                    _gVC_st,
                    _vc_ring_base + _vap_ring * Int32(V_DIM_C) + (
                        _vap_pos * Int32(8)
                    ),
                    _va0,
                    _va1,
                    _va2,
                    _va3,
                )

        # ============================================================
        # OUTPUT: out = WH + QT@V
        # ============================================================
        _sV_base = sV.iterator.toint()
        _gOut_base = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        _v_off_base = Int32(0)  # full V in one tile
        # v2 flush-variant: stage outputs through sK, NOT sH. A_full (sK) is
        # dead after the half-1 H GEMM and the [T, V_PADDED] tile is an exact
        # fit (V_PADDED == K_PADDED == 136). Keeping sH untouched preserves the
        # H0 half-1 tile through the output flush, so Phase-5 Step E processes
        # half-1 from RESIDENT SMEM (no TMA re-read; only half-0 re-fetches).
        # C is staged into sK afterwards (it is free again post output-flush).
        _sOutStage_base = _sK_base_vl

        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sPowk, lane_id)
        # 4 V-groups per warp at 8 V-cols each → byte stride = 8*2 = 16
        # within a warp, warp_id stride in V-cols = 32 → 64 B between warps.
        _qtv_base = (
            _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        )
        _qtvr = _qtv_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _qtv_base)

        for h_iter in cutlass.range_constexpr(4):
            h = warp_id * 4 + h_iter
            acc.iterator[0] = _qtvr[h_iter * 4]
            acc.iterator[1] = _qtvr[h_iter * 4 + 1]
            acc.iterator[2] = _qtvr[h_iter * 4 + 2]
            acc.iterator[3] = _qtvr[h_iter * 4 + 3]
            if h_iter == 0:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_0.iterator[j]
            if h_iter == 1:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_1.iterator[j]
            if h_iter == 2:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_2.iterator[j]
            if h_iter == 3:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_3.iterator[j]
            # SMEM-staged epilogue (NCU B=256 mbp8: the 8 fragment-direct 4-B
            # STG.32s were the top uncoalesced-global source — 50% of each
            # 32-B sector wasted, 2.1M of 2.6M excessive L2 sectors). Stage
            # the [T,128] tile in SMEM (h_buf — sH is dead after the half-1 H
            # GEMM; the sync at the QT@V wait above orders all warps past it),
            # then flush with fully-coalesced 16-B STGs below. STS pattern
            # (word = 68*r + 4*h + lane%4) is bank-conflict-free.
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

        # Coalesced flush: consecutive lanes write consecutive 16-B chunks
        # (16 chunks per 256-B row), so each warp covers 512 B contiguous —
        # 100% sector utilization. LDS.128 quarter-warps read 32 consecutive
        # SMEM words (pos*4 spans a full bank period) — conflict-free.
        sync_threads()
        _fl_passes = 2 if self._t_input > 8 else 1
        for _fl_pass in cutlass.range_constexpr(_fl_passes):
            _fl_chunk = _fl_pass * 128 + tidx
            _fl_row = _fl_chunk // 16
            _fl_pos = _fl_chunk & 15
            _fl_lds = _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
            _fl_off = _out_base + _fl_row * so_t + _v_off_base + _fl_pos * 8
            # LDS hoisted out of the runtime guard: tuple-unpack inside an
            # if-region trips a DSL region-type error, and reading staged
            # garbage rows (>= t_input) is harmless — only the STG is gated.
            _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
            # (vcache) compact-output STG: only draft rows ([P_hist, P_hist+T_D)
            # in the window) are written, shifted to output rows [0, T_D).
            if const_expr(self._vcache_t_draft > 0):
                _fl_off_vc = (
                    _out_base
                    + (_fl_row - P_hist) * so_t
                    + _v_off_base
                    + _fl_pos * 8
                )
                if _fl_row >= P_hist and _fl_row < P_hist + _T_D:
                    _st_global_v4_b32(_gOut_base, _fl_off_vc, _v0, _v1, _v2, _v3)
        # ============================================================
        # PHASE 5 — FLUSH: fold window rows [0:P) into H0, in place.
        #   H0_new = exp(G_P) * H0 + U^T @ Khat
        #   KH   = diag(exp(G_i)) * KHraw        (KHraw piggybacked on H GEMM)
        #   C    = V - KH                        (innovation)
        #   U    = Tmat @ C                      (top-left PxP = causal P-row
        #          solve; U rows >= P are garbage, killed by Khat rows = 0)
        #   Khat = diag(i < P ? exp(G_P - G_i) : 0) * k_norm
        # Outputs (Phases 1-4) never read H0_new — it exists only for the next
        # iteration. All compute is unconditional (uniform control flow; the
        # DSL rejects tuple-unpacking helpers inside runtime if-regions); only
        # the final STGs are predicated on p_val > 0.
        # ============================================================
        sync_threads()  # output flush done: sH (staging) and qv_buf are dead

        # G_P = log-domain decay cumsum through row P-1 (clamped for P == 0 —
        # the value is then unused because the store is predicated off).
        _pm1 = p_val - Int32(1)
        if p_val == Int32(0):
            _pm1 = Int32(0)
        _g_p = sGamma.iterator[_pm1]
        _exp_gp = _exp_approx_f32(_g_p)

        # --- Steps B+C: C = V - exp(G_i)*KHraw -> bf16 into sK (dead after the
        # H GEMM consumed A_full). Fragment coords mirror the output epilogue.
        _p5_r0 = lane_id // 4
        _p5_c0 = (lane_id & 3) * 2
        _eg_lo = sGamma.iterator[T + _p5_r0]
        _eg_hi = sGamma.iterator[T + _p5_r0 + 8]
        # Rows >= P are staged as LITERAL zeros in both C and U^T below: the
        # MMAs reduce over all 16 rows, and 0 * NaN = NaN, so a single NaN/Inf
        # in a garbage draft row would otherwise leak into H0_new even though
        # its weight is zero (campaign NaN-drafts probe). Both operands of
        # every dead product must be finite.
        for _p5c in cutlass.range(_p5_iters):  # (skip) Phase-5-only
            _kh_frags = (kh_acc_0, kh_acc_1, kh_acc_2, kh_acc_3)
            for _cg in cutlass.range_constexpr(4):
                _khf = _kh_frags[_cg]
                _c_col = (warp_id * 4 + _cg) * 8 + _p5_c0
                _cv0 = sV.iterator[_p5_r0 * V_PADDED + _c_col].to(f32)
                _cv1 = sV.iterator[_p5_r0 * V_PADDED + _c_col + 1].to(f32)
                _cv2 = sV.iterator[(_p5_r0 + 8) * V_PADDED + _c_col].to(f32)
                _cv3 = sV.iterator[(_p5_r0 + 8) * V_PADDED + _c_col + 1].to(f32)
                _cw0 = (_cv0 - _eg_lo * _khf.iterator[0]).to(io)
                _cw1 = (_cv1 - _eg_lo * _khf.iterator[1]).to(io)
                _cw2 = (_cv2 - _eg_hi * _khf.iterator[2]).to(io)
                _cw3 = (_cv3 - _eg_hi * _khf.iterator[3]).to(io)
                if _p5_r0 >= p_val:
                    _cw0 = io(0.0)
                    _cw1 = io(0.0)
                if _p5_r0 + Int32(8) >= p_val:
                    _cw2 = io(0.0)
                    _cw3 = io(0.0)
                sK.iterator[_p5_r0 * K_PADDED + _c_col] = _cw0
                sK.iterator[_p5_r0 * K_PADDED + _c_col + 1] = _cw1
                sK.iterator[(_p5_r0 + 8) * K_PADDED + _c_col] = _cw2
                sK.iterator[(_p5_r0 + 8) * K_PADDED + _c_col + 1] = _cw3
            sync_threads()

            # --- U = Tmat @ C (QT@V-shaped MMA; C staged in sK, K_PADDED ==
            # V_PADDED stride). U covers V-cols [warp_id*32, warp_id*32+32).
            _u_a0, _u_a1, _u_a2, _u_a3 = _ldmatrix_x4(sTmat, lane_id)
            _u_base = _sK_base_vl + _ldm_row * Int32(K_PADDED * 2) + warp_id * Int32(64)
            _ur = _qtv_4mma(_u_a0, _u_a1, _u_a2, _u_a3, _u_base)

            # --- stage U^T into sUT (qv_buf, dead: last sV read was the C step
            # before the sync above): [V_DIM_C, T] row-major, row stride 32 B.
            for _ug in cutlass.range_constexpr(4):
                _u_col = (warp_id * 4 + _ug) * 8 + _p5_c0
                _uw0 = _ur[_ug * 4 + 0].to(io)
                _uw1 = _ur[_ug * 4 + 1].to(io)
                _uw2 = _ur[_ug * 4 + 2].to(io)
                _uw3 = _ur[_ug * 4 + 3].to(io)
                if _p5_r0 >= p_val:  # U rows >= P: literal zeros (see C staging)
                    _uw0 = io(0.0)
                    _uw1 = io(0.0)
                if _p5_r0 + Int32(8) >= p_val:
                    _uw2 = io(0.0)
                    _uw3 = io(0.0)
                sUT.iterator[_u_col * T + _p5_r0] = _uw0
                sUT.iterator[(_u_col + 1) * T + _p5_r0] = _uw1
                sUT.iterator[_u_col * T + _p5_r0 + 8] = _uw2
                sUT.iterator[(_u_col + 1) * T + _p5_r0 + 8] = _uw3

            # --- Step D: Khat = diag(i < P ? exp(G_P - G_i) : 0) * k_norm, scaling
            # sKsnap in place (bf16x2 words). exp(G_P - G_i) <= 1 for i < P (G is a
            # non-increasing cumsum) — overflow-safe, same trick as the decay fix.
            for _kd in cutlass.range_constexpr(8):
                _kd_flat = tidx + _kd * THREADS
                _kd_row = _kd_flat >> 6
                _kd_col = _kd_flat & Int32(63)
                _kd_addr = _kd_row * (K_PADDED // 2) + _kd_col
                if _kd_row < p_val:
                    _kd_f = _exp_approx_f32(_g_p - sGamma.iterator[_kd_row])
                    _sKsnap_i32.iterator[_kd_addr] = _mul_bf16x2_f32(
                        _sKsnap_i32.iterator[_kd_addr], _kd_f
                    )
                else:
                    # Write a LITERAL zero for rows >= P. Multiplying by 0 would
                    # keep NaN/Inf from garbage draft rows alive (0*NaN = NaN) and
                    # leak them into H0_new through the U^T @ Khat reduction
                    # (caught by the correctness campaign's NaN-drafts probe).
                    _sKsnap_i32.iterator[_kd_addr] = Int32(0)
        sync_threads()

        # --- Step E (v2): H0_new[v,k] = exp(G_P)*H0[v,k] + (U^T @ Khat)[v,k].
        # Half-1 runs FIRST, from the still-resident sH (the output epilogue
        # now stages through sK, so H0 half-1 survives Phase 4) — no TMA;
        # only half-0 re-fetches (3rd mbarrier arrival -> parity 0, predicated
        # on p_val > 0 per v1.5). Store path (v2): each lane STS's its H0_new
        # fragment back to the SAME swizzled sH bytes it just read (thread-
        # local, hazard-free), then a coalesced 16-B LDS/STG flush mirrors the
        # output epilogue — replaces v1's 4-B uncoalesced state STGs.
        # M = 128 V-rows: warp handles tiles [warp*32, +16) and [+16, +32).
        _gH0_base = gH0.iterator.toint()
        _sh_hv_p5 = V_DIM * K_DIM
        _h0_off_base = cache_idx * (HV * _sh_hv_p5) + pid_hv * _sh_hv_p5
        _sUT_int = sUT.iterator.toint()
        _sH_int_p5 = sH.iterator.toint()
        for _eh in (1, 0):
            if _eh == 0:
                # Half-0 re-fetch. All lanes must be past half-1's coalesced
                # flush reads of sH before the TMA overwrites it.
                sync_threads()
                if p_val > Int32(0):
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

            for _p5e in cutlass.range(_p5_iters):  # (skip) Phase-5-only
                for _em in cutlass.range_constexpr(2):
                    _e_mbase = warp_id * Int32(32) + Int32(_em * 16)
                    _e_a_addr = (
                        _sUT_int
                        + (_e_mbase + (lane_id % 16)) * Int32(T * 2)
                        + (lane_id // 16) * Int32(16)
                    )
                    _e_a0, _e_a1, _e_a2, _e_a3 = _ldmatrix_x4_addr(_e_a_addr)
                    for _en in cutlass.range_constexpr(2):
                        _e_b_base = (
                            _sKsnap_int
                            + _ldm_row * Int32(K_PADDED * 2)
                            + Int32(_eh * K_HALF * 2)
                            + Int32(_en * 64)
                        )
                        _er = _qtv_4mma(_e_a0, _e_a1, _e_a2, _e_a3, _e_b_base)
                        for _eg in cutlass.range_constexpr(4):
                            _e_kcl = _en * 32 + _eg * 8 + _p5_c0  # col within half
                            _e_rv0 = _e_mbase + _p5_r0
                            _e_rv8 = _e_rv0 + Int32(8)
                            _e_sa0 = _sw128_xor(
                                _sH_int_p5 + _e_rv0 * _rs_b + _e_kcl * 2
                            )
                            _e_sa8 = _sw128_xor(
                                _sH_int_p5 + _e_rv8 * _rs_b + _e_kcl * 2
                            )
                            _h0l0, _h0h0 = _st2_to_f32x2(_lds_b32(_e_sa0))
                            _h0l8, _h0h8 = _st2_to_f32x2(_lds_b32(_e_sa8))
                            if p_val > Int32(0):  # P=0 CTAs skip the store path
                                _sts_st2_f32(
                                    _e_sa0,
                                    _exp_gp * _h0l0 + _er[_eg * 4 + 0],
                                    _exp_gp * _h0h0 + _er[_eg * 4 + 1],
                                )
                                _sts_st2_f32(
                                    _e_sa8,
                                    _exp_gp * _h0l8 + _er[_eg * 4 + 2],
                                    _exp_gp * _h0h8 + _er[_eg * 4 + 3],
                                )

            # Coalesced flush of this half's H0_new tile: 128 rows x 8 16-B
            # chunks; consecutive lanes cover consecutive chunks (full 32-B
            # sectors). LDS applies the same SW128 xor the STS used; the gmem
            # offset uses the LOGICAL (row, chunk) coords. The whole loop sits
            # inside the p_val guard so P=0 CTAs skip it entirely; the LDS uses
            # 4x single-word _lds_b32 (single-value returns are legal inside a
            # runtime if-region; the tuple-returning _lds_v4_b32 is not).
            sync_threads()
            if p_val > Int32(0):
                for _ep in cutlass.range_constexpr(8):
                    _e_chunk = tidx + _ep * THREADS
                    _e_frow = _e_chunk >> 3
                    _e_fpos = _e_chunk & Int32(7)
                    _e_lds = _sw128_xor(
                        _sH_int_p5 + _e_frow * _rs_b + _e_fpos * 16
                    )
                    _ev0 = _lds_b32(_e_lds)
                    _ev1 = _lds_b32(_e_lds + 4)
                    _ev2 = _lds_b32(_e_lds + 8)
                    _ev3 = _lds_b32(_e_lds + 12)
                    _e_goff = (
                        _h0_off_base
                        + _e_frow * K_DIM
                        + Int32(_eh * K_HALF)
                        + _e_fpos * 8
                    )
                    _st_global_v4_b32(_gH0_base, _e_goff, _ev0, _ev1, _ev2, _ev3)


# ============================================================================
# Public entry point.
# ============================================================================

_CACHE: dict = {}
import os as _os
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



def gated_delta_rule_mtp_vcache_flush(
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
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    a_cache: Optional[torch.Tensor] = None,
    b_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    flush_min: Optional[int] = None,
    restart_hist_on_flush: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN decode output + raw-v-cache append + per-request state flush.

    Raw-v-cache sibling of the u-cache spec-decode kernels (PR #4081): the ring
    caches RAW ``v/k`` (bf16) plus RAW ``a/b`` (fp32) instead of the materialized
    ``u``, and the WY solve reconstructs ``u`` on the fly. ``q`` is not cached.

    RING SEMANTICS (Triton-ReplaySSM-compatible): the live window of request
    ``b`` is ``[cache_base[b], cache_base[b]+hist_len[b]) mod 32`` and the T
    new (raw-k, raw-v, fp32 a, fp32 b) entries are appended at
    ``(cache_base+hist_len+s) & 31`` — past the window, for flush and verify
    rows alike, so an append can never overwrite rows a sibling CTA still
    reads. Cursor commits are CALLER-OWNED (outside the launch): flush rows
    ``base' = (base+len) & 31, len' = accepted``; verify rows
    ``len' = len + accepted``. ``restart_hist_on_flush=True`` applies that
    commit here as a convenience for standalone use (graph-capturable ops).

    Per request ``b`` with window length ``P = hist_len[b]``:
      - ``P <  flush_min``: verify only. Draft outputs come from S0 evolved
        through the P committed ring rows + the T drafts; state is UNCHANGED.
      - ``P >= flush_min``: additionally FOLD the committed rows into the state
        pool at ``initial_state_indices[b]`` (in place, WY Phase-5 math).

    PREDICTIVE flush model (differs from #4081's lazy variant): ``flush_min`` and
    ``hist_len`` are capped at ``W_RING - T`` so ``[committed | draft]`` always
    fits one 16-row tile — the draft outputs are computed from S0 in the same
    window (equal to S0' + draft for the draft rows), so no fold-before-output
    path is needed. ``flush_min`` defaults to ``W_RING - T``.

    Ring tensors (zero-initialize at pool allocation; RING_SLOTS = 32 deep):
      k_cache [pool, H,  32, K] bf16   v_cache [pool, HV, 32, V] bf16
      a_cache [pool, HV, 32]    f32    b_cache [pool, HV, 32]    f32
      hist_len   [B] int32  live-window length P (legal [0, W_RING-T])
      cache_base [B] int32  ring window origin (legal [0, 32))

    Draft inputs ``q/k/v/a/b`` are ``[B, T, ...]``. Returns ``output`` of shape
    ``[B, T, HV, V]`` (bf16) — the draft-token outputs, written COMPACTED by
    the kernel (window rows [P, P+T) land at output rows [0, T)).

    The 16-row window is assembled IN-KERNEL (committed rows stream from the
    ring caches via cp.async, draft rows from the fresh tensors, dead tail
    zero-filled) — fully CUDA-graph-capturable. Committed a/b are read fp32
    straight from the rings (no bf16 round-trip). The draft append + cursor
    reset + output gather are fixed-shape torch ops after the launch (also
    graph-safe; see the in-wrapper comment for why the append is host-side).
    """
    assert q is not None and k is not None and v is not None and b is not None
    assert initial_state_source is not None
    assert use_qk_l2norm_in_kernel, "kernel always applies Q/K L2 norm."
    assert softplus_beta == 1.0 and softplus_threshold == 20.0
    assert initial_state_source.dtype == ST_TORCH, (
        f"initial_state_source must be {ST_TORCH} (pool,HV,V,K) — module STATE "
        f"dtype (GDN_VCACHE_STATE_DTYPE={_ST_ENV!r}); got "
        f"{initial_state_source.dtype}."
    )
    assert initial_state_source.is_contiguous(), (
        "initial_state_source must be contiguous — the flush writes it in place."
    )

    B, T, H, K_dim = q.shape
    HV = v.shape[2]
    V_dim = v.shape[3]
    HK = k.shape[2]
    device = q.device
    assert K_dim == K_DIM and V_dim == V_DIM_C, (
        f"this kernel requires K==V=={K_DIM}; got K={K_dim}, V={V_dim}."
    )
    W_RING = 16
    T_KERNEL = 16
    assert 1 <= T <= W_RING
    if scale is None:
        scale = 1.0 / math.sqrt(K_dim)

    # --- ring validation (raw v/k bf16 + raw a/b fp32) ---
    assert (
        k_cache is not None and v_cache is not None and a_cache is not None
        and b_cache is not None and hist_len is not None
    ), "k_cache/v_cache/a_cache/b_cache/hist_len are required."
    pool = initial_state_source.shape[0]
    assert tuple(k_cache.shape) == (pool, HK, RING_SLOTS, K_dim), tuple(k_cache.shape)
    assert tuple(v_cache.shape) == (pool, HV, RING_SLOTS, V_dim), tuple(v_cache.shape)
    assert tuple(a_cache.shape) == (pool, HV, RING_SLOTS), tuple(a_cache.shape)
    assert tuple(b_cache.shape) == (pool, HV, RING_SLOTS), tuple(b_cache.shape)
    assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, (
        "k_cache/v_cache must be bf16."
    )
    assert a_cache.dtype == torch.float32 and b_cache.dtype == torch.float32, (
        "a_cache/b_cache must be fp32 (decay/gate precision preserved)."
    )
    assert hist_len.dtype == torch.int32 and hist_len.shape[0] == B
    if flush_min is None:
        flush_min = W_RING - T
    # W_RING - T + 1 = NEVER-FLUSH sentinel (hist_len is capped at W_RING - T
    # by the predictive contract, so no request can reach it): pure verify +
    # append, state untouched — used by the verify-only sibling wrapper.
    assert 1 <= flush_min <= W_RING - T + 1, (
        f"flush_min={flush_min} out of range [1, {W_RING - T + 1}] "
        "(predictive model; the top value = never-flush/verify-only)."
    )

    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    else:
        initial_state_indices = initial_state_indices.contiguous()
        assert initial_state_indices.shape[0] == B

    # Contract check (host-side; skipped during CUDA-graph capture, where
    # .item() would be illegal). hist_len + T must fit the 16-row window.
    if not torch.cuda.is_current_stream_capturing():
        assert int(hist_len.max().item()) + T <= W_RING, (
            "hist_len + T must be <= W_RING (predictive flush contract)."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    if restart_hist_on_flush:
        # The standalone commit at the end of this call mutates hist_len and
        # cache_base IN PLACE and must see the caller's own storage across
        # calls: cache_base=None would slide the base of a throwaway temp
        # (the next call re-defaults base to 0 and reads a stale window), and
        # a non-contiguous tensor would be silently replaced by its
        # .contiguous() copy below, absorbing the commit the same way.
        if cache_base is None:
            raise ValueError(
                "gated_delta_rule_mtp_vcache_flush: restart_hist_on_flush="
                "True commits ring cursors in place and needs a caller-owned "
                "cache_base (got None). Keep a persistent int32 [B] tensor "
                "(zeros initially) across calls, or pass "
                "restart_hist_on_flush=False and own the cursor commit."
            )
        assert hist_len.is_contiguous() and cache_base.is_contiguous(), (
            "restart_hist_on_flush=True: hist_len and cache_base must be "
            "contiguous — .contiguous() would copy them and the in-place "
            "cursor commit would be lost on the caller's tensors."
        )
        # Cursor range validation (host sync: standalone path only, never
        # while a CUDA graph is capturing). The ring masks all addressing
        # with & RING_MASK, so out-of-range cursors corrupt SILENTLY.
        if not torch.cuda.is_current_stream_capturing():
            _cb_min, _cb_max = (int(x.item()) for x in cache_base.aminmax())
            assert 0 <= _cb_min and _cb_max < RING_SLOTS, (
                f"cache_base out of legal range [0, {RING_SLOTS}): "
                f"min={_cb_min} max={_cb_max}"
            )
    hist_len = hist_len.contiguous()
    if cache_base is None:
        cache_base = torch.zeros_like(hist_len)
    assert cache_base.dtype == torch.int32 and cache_base.shape[0] == B
    cache_base = cache_base.contiguous()
    assert (
        k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and a_cache.is_contiguous()
        and b_cache.is_contiguous()
    ), "ring caches must be contiguous (written in place)."

    A_logb = _cached_bf16(A_log)
    dt_biasb = _cached_bf16(dt_bias)
    h0 = initial_state_source
    n_valid = T_KERNEL
    _qkv_rs = 0
    _ab_native_flag = False
    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    _needed = math.ceil((HV * B) / _num_sms)
    mbp = max(1, min(_needed + 1, 6))
    _mbp_env = _os.environ.get("GDN_WY_MBP")
    if _mbp_env:
        mbp = int(_mbp_env)
    t_disc = 16  # full 16-row window (committed + draft) drives Phase-2 depth
    cache_key: tuple = (
        str(device), mbp, t_disc, n_valid, _qkv_rs, _ab_native_flag,
        HV, H, V_dim, "vcache", T,
    )
    mk = from_dlpack

    def mk_dyn(t):
        return mk(t, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    # (compact output) the kernel writes ONLY the T draft rows, at output rows
    # [0, T) — allocate [B, T, HV, V] directly (no 16-row tile, no gather).
    out_c = output if (
        output is not None and output.is_contiguous()
        and tuple(output.shape) == (B, T, HV, V_dim)
        and output.dtype == torch.bfloat16
    ) else torch.empty(B, T, HV, V_dim, dtype=torch.bfloat16, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    args = [
        mk_dyn(q), mk_dyn(k), mk_dyn(v), mk_dyn(a), mk_dyn(b),
        mk(A_logb, 16), mk(dt_biasb, 16),
        mk_dyn(h0), mk_dyn(initial_state_indices),
        mk_dyn(hist_len), mk_dyn(cache_base),
        mk_dyn(out_c),
        mk_dyn(k_cache), mk_dyn(v_cache), mk_dyn(a_cache), mk_dyn(b_cache),
        scale, int(flush_min), HV, V_dim, H, stream,
    ]
    if cache_key not in _CACHE:
        _CACHE[cache_key] = cute.compile(
            GdnVcacheFlushKernel(
                min_blocks_per_mp=mbp, t_input=t_disc, n_valid=n_valid,
                qkv_row_stride=_qkv_rs, ab_native=_ab_native_flag,
                vcache_t_draft=T,
            ),
            *args,
        )
    _CACHE[cache_key](*args)

    # ------------------------------------------------------------------
    # APPEND drafts to the ring (verify: [P, P+T); flush: [0, T) restart),
    # cursor reset, and draft-output gather. All fixed-shape torch ops on
    # the SAME stream: graph-capturable, and stream-ordered AFTER the
    # kernel so ring writes can never race the kernel's ring reads (the
    # in-kernel append alternative races across CTAs on the flush path:
    # sibling hv-CTAs of a request read committed k rows [0, P) while the
    # writer CTA restarts rows [0, T) — no grid-wide sync exists).
    # ------------------------------------------------------------------
    # ALL ring appends (k/v/a/b) happen IN-KERNEL: appends land PAST every
    # request's live window ((base+len+s) & RING_MASK), so even the shared
    # k ring can never race a sibling CTA's committed-row reads.
    # Ring cursor commit for STANDALONE use: flush rows slide the base past
    # the folded window and reset the length (Triton
    # commit_gdn_replayssm_spec semantics; host-side, graph-capturable).
    # restart_hist_on_flush=False hands the commit to the caller (a serving
    # loop sharing one cursor set across N layers must commit once per step).
    if restart_hist_on_flush:
        _flushed = hist_len >= flush_min
        cache_base.copy_((cache_base + hist_len * _flushed) & RING_MASK)
        hist_len.masked_fill_(_flushed, 0)

    if output is not None and out_c is not output:
        output.copy_(out_c)
        return output
    return out_c


__all__ = [
    "gated_delta_rule_mtp_vcache_flush",
    "GdnVcacheFlushKernel",
    "K_DIM",
    "V_DIM_C",
    "W_RING_C",
    "RING_SLOTS",
    "RING_MASK",
]
