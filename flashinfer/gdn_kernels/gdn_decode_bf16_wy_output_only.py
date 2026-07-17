"""CuTeDSL GDN MTP Decode Kernel — WY-parallel, NOPREPACK variant.

v6 (TMA + SW128 swizzle on H tile):
- Replaces v5's 4-stage staggered cp.async H load with a single
  `cp.async.bulk.tensor.4d.shared.cluster.global` TMA issued once per CTA.
- sH SMEM tile is allocated as a SW128-swizzled tensor (mandatory for the
  TMA dispatcher to encode swizzle_mode=SW128 — empirically required for
  the (V=128, K=128) BF16 K-major box, see results/v6_tma_debug_summary.md).
- H GEMM B-fragment ldmatrix.x2 addresses get the SW128 XOR transform
  (`phys = L ^ ((L >> 3) & 0x70)`) applied per-lane to each of the 4
  per-V-group base addresses.

Algorithmic structure mirrors Path 1:
- Phase 1: KKT / QKT (T x T GEMMs)
- Phase 2: Log-depth Neumann inverse (6 GEMMs at T=16)
- Phase 3: A_full @ H^T + QT @ V → output

H0 layout differences from Path 1:
- Path 1: H0 loaded directly from prepacked GMEM into MMA B-fragments via
  ld.global.v2.b32. H0 layout (pool, HV, V//8, K//16, 8, 16).
- This file (Path 2): H0 layout (pool, HV, V, K). Per-CTA tile is V=128, K=128
  = 32 KB BF16. v6: TMA G2S issued once during the prologue, mbarrier
  signals completion before the H GEMM. SW128-swizzled SMEM target.

Other features inherited from Path 1:
- Inline PTX ldmatrix for all SMEM->register MMA loads
- Direct acc->SMEM writes (skip sC staging) for KKT/QKT
- Vectorized cp.async (16 B / instruction) for K, Q
- T-aware Phase-2 squaring depth (4 / 8 / 16 variants)
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

# Fixed dims matching Triton v5
T = 16
K_DIM = 128
V_DIM_C = 128  # full V tile per CTA
BK_H = 16  # K-tile for H GEMM (must be a multiple of 16 for mma.k=16)
EPS = 1e-6
io = cutlass.BFloat16
f32 = cutlass.Float32
WARP = 32
THREADS = 128
T_PAD = 16

# SMEM padding for sK/sQ — same as Path 1
K_HALF = K_DIM // 2  # 64 — v13 half-H streaming (sH = 16 KiB instead of 32 KiB)
K_PADDED = K_DIM + 8  # 136 — padded row stride for sK / sQ
V_PADDED = V_DIM_C + 8  # 136 — padded row stride for sV / sH (V rows, K cols)
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


class GdnDecodeKernel:
    """CuTeDSL GDN MTP decode (noprepack) — WY-parallel, inference only."""

    def __init__(
        self,
        disable_state_update=False,
        min_blocks_per_mp=2,
        t_input=16,
        bv=None,
        n_valid=16,
        qkv_row_stride=0,
        ab_native=False,
    ):
        assert disable_state_update, "State update not implemented in CuTeDSL kernel"
        # `bv` is accepted only for bench-script signature compatibility — the
        # noprepack kernel always consumes the full V=128 tile in one CTA so
        # the V-split path of Path 1 is not implemented here.
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
        sa_hv = cutlass.Int32(1)
        sa_t = HV
        sa_b = _ab_rows * HV
        sb_hv = cutlass.Int32(1)
        sb_t = HV
        sb_b = _ab_rows * HV
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
            _v7e_a_bf16 = gA.iterator[
                pid_b * sa_b + lane_id * sa_t + pid_hv * sa_hv
            ].to(f32)
            _v7e_b_bf16 = gB.iterator[
                pid_b * sb_b + lane_id * sb_t + pid_hv * sb_hv
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
            h_buf: cute.struct.Align[cute.struct.MemRange[io, V_DIM_C * K_HALF], 128]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            gamma: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            # v12.1: removed c_all (2 KiB) — was a sized template for partition_C
            # only, never written-to. Now we build tCsC via partition_shape_C +
            # make_fragment_C directly (no SMEM allocation needed).
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            scratch2_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]

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
            # (native-short-T) when n_valid < T the q/k gmem tensors hold only
            # n_valid rows; skip the cp.async for rows >= n_valid (those would read
            # OOB). The sK/sQ[n_valid:T] smem tail is zeroed after the wait below.
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
        # QT = QKTm @ Tmat → sPowk
        if warp_id < 2:
            acc.fill(f32(0.0))
            col_off = warp_id * 8
            _qt_a_addr = (
                sNegL.iterator.toint() + _lane_mod16 * Int32(BF_PAD * 2) + _lane_hi
            )
            _qt_b_addr = (
                sTmat.iterator.toint()
                + _ldm_row * Int32(BF_PAD * 2)
                + col_off * Int32(2)
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
            _r0 = lane_id // 4
            _c0 = (lane_id & 3) * 2
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[1].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[2].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = acc.iterator[3].to(
                io
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
        _af_b_base = _sK_base_af + _ldm_row * Int32(K_PADDED * 2) + warp_id * Int32(16)
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
                _exp_eq_r0 * sQ.iterator[_r0 * K_PADDED + k_col + _c0 + 1].to(f32)
                - _afr[bk_idx * 4 + 1]
            ).to(io)
            sK.iterator[(_r0 + 8) * K_PADDED + k_col + _c0] = (
                _exp_eq_r8 * sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0].to(f32)
                - _afr[bk_idx * 4 + 2]
            ).to(io)
            sK.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1] = (
                _exp_eq_r8 * sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1].to(f32)
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
        _v_iters = 1 if self._t_input <= 8 else (T * V_DIM_C // (THREADS * 8))
        for i in cutlass.range_constexpr(_v_iters):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col_bf16_async = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col_bf16_async * Int32(
                2
            )
            # (native-short-T) v holds only n_valid rows; skip rows >= n_valid (OOB).
            # The sV working-set tail [n_valid:8] is zeroed after the V wait below;
            # rows [8:16) stay garbage (proof-safe: sPowk[*,8:15]=0 -> garbage*0=0).
            if const_expr(self._n_valid < T):
                if _v_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gV_base,
                        _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                        _sV_base_async + _smem_byte_off_v,
                    )
            else:
                _cp_async_bf16x8(
                    _gV_base,
                    _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                    _sV_base_async + _smem_byte_off_v,
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

        # ============================================================
        # OUTPUT: out = WH + QT@V
        # ============================================================
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sPowk, lane_id)
        _sV_base = sV.iterator.toint()
        _gOut_base = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        _v_off_base = Int32(0)  # full V in one tile
        # Output staging tile [T, V_PADDED] bf16 aliased onto h_buf (16 KiB;
        # needs 4.25 KiB). sH's last read is the half-1 H GEMM; every warp is
        # past it once the sync below (before QT@V) has run.
        _sOutStage_base = sH.iterator.toint()

        # 4 V-groups per warp at 8 V-cols each → byte stride = 8*2 = 16 within a warp,
        # warp_id stride in V-cols = 32 → 64 bytes between warps.
        _qtv_base = _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        # v10: Wait for the V cp.async to land in sV before ldmatrix-ing
        # from it. V was issued before the H GEMM; the wait here drains
        # whatever didn't finish in the H GEMM's shadow. Same proxy as
        # K+Q LDGSTS — no fence_view_async_shared needed.
        _cp_async_wait_group_0()
        # (native-short-T) zero the sV working-set tail rows [n_valid:8] that were NOT
        # loaded (v held only n_valid rows). The QT@V reduction reads the 8-row working
        # set for t_input<=8; rows [n_valid:8] must be zero (staging supplied them
        # before). Rows [8:16) stay garbage — proof-safe (sPowk[*,8:15]=0). tidx in
        # [0,THREADS=128) covers V_DIM_C=128 cols; the following sync publishes it.
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, 8):
                sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        sync_threads()
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


# ============================================================================
# Public entry point — drop-in for
#   flashinfer.gdn_kernels.gdn_decode_bf16_state.gated_delta_rule_mtp
# restricted to the OUTPUT-ONLY (frozen-state) case.
#
# This is the bank-conflict-eliminated "v18" no-prepack kernel. It consumes H0
# in its natural (pool, HV, V, K) bf16 layout (NO external prepack) and computes
# the MTP decode output, keeping the running state h in FP32 registers across
# the T tokens (loaded/stored as BF16 at the boundary). It does NOT write state
# back, cache intermediates, or do fused recovery; those paths raise
# NotImplementedError so a mis-routed caller fails loudly rather than silently
# returning wrong results.
#
# Input/output contract matches the BF16-state MTP kernel for output-only use:
#   A_log [HV], a [B,T,HV], dt_bias [HV], q/k [B,T,H,K], v [B,T,HV,V], b [B,T,HV],
#   initial_state_source [pool,HV,V,K] bf16, initial_state_indices [B] int32,
#   scale, output [B,T,HV,V] -> returns output [B,T,HV,V] bf16.
# Requires SM90+ (TMA + mbarrier; validated on H200/SM90 and B200/GB300 SM100);
# K == V == 128.
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
# (native-short-T) When set, the T<T_KERNEL path passes q/k to the kernel as the
# real [B,T,...] tensors (no host staging copy) and the kernel loads only those T
# rows + zeros its sK/sQ smem tail. Removes the two big q/k gmem->gmem staging
# copies. v/a/b stay staged (v is already minimized by _v_iters=1 at t_input<=8;
# a/b feed the per-lane gamma path).
# ON by default since the _BF16_CACHE stale-cast fix: the earlier "unsafe"
# accuracy signal traced to that host-side bug, not this path. Validated on B200
# (T=4 and T=8, BS=1..256): max|d| <= 9.77e-04 vs branch kernel and vs the torch
# reference, full wy_output_only pytest green with native forced, and 2-6%
# faster kernel time at BS>=8 plus 2 fewer host staging copies per call.
# Set FLASHINFER_GDN_WY_NATIVE_T=0 to restore full staging.
import os as _os

_NATIVE_T = _os.environ.get("FLASHINFER_GDN_WY_NATIVE_T", "1") != "0"
# (strided-qkv) read q/k/v directly from the fused conv-output column slices (token
# stride = conv_dim) instead of .contiguous()-materializing them. Removes the 3 big
# q/k/v copies from the verify region. Only valid on the native path (T in {4,8}).
_STRIDED_QKV = _os.environ.get("FLASHINFER_GDN_WY_STRIDED_QKV", "0") != "0"
# (native-a/b) read a/b directly from the real [B, n_valid, HV] tensors instead of staging
# them into T_KERNEL-row zero-padded buffers (removes the 2 a/b staging copies). Bit-exact on
# the compact [B,T] output: gamma is a causal prefix-sum, so the unloaded tail rows (which get
# log_alpha=0 instead of the staged-zero value) cannot affect rows 0..n_valid-1, and the tail
# output is discarded. Native path only (T in {4,8}).
# ON by default alongside _NATIVE_T (same validation); FLASHINFER_GDN_WY_NATIVE_AB=0
# restores a/b staging.
_NATIVE_AB = _os.environ.get("FLASHINFER_GDN_WY_NATIVE_AB", "1") != "0"
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


def gated_delta_rule_mtp(
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
) -> torch.Tensor:
    """GDN MTP decode (BF16 state), bank-conflict-eliminated no-prepack kernel.

    Output-only / frozen-state: computes ``output`` for all T tokens and never
    writes ``initial_state_source`` back. Signature mirrors
    ``gdn_decode_bf16_state.gated_delta_rule_mtp`` so it is a drop-in for that
    kernel's output-only path; the state-update / recovery / cache / per-request
    features are not implemented and raise ``NotImplementedError``.

    Returns ``output`` of shape ``[B, T, HV, V]`` (bf16).
    """
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    # --- output-only kernel: reject the fused/state-update features ---
    if recovery_steps != 0:
        raise NotImplementedError(
            "gdn_decode_bf16_wy_output_only: recovery_steps>0 is not supported "
            "(this is an output-only / frozen-state kernel)."
        )
    if disable_output:
        raise NotImplementedError(
            "gdn_decode_bf16_wy_output_only: disable_output=True (state-only mode) "
            "is not supported (this kernel always emits output)."
        )
    if intermediate_states_buffer is not None:
        raise NotImplementedError(
            "gdn_decode_bf16_wy_output_only: intermediate-state caching is not supported."
        )
    if accepted_steps is not None or ssm_state_indices is not None:
        raise NotImplementedError(
            "gdn_decode_bf16_wy_output_only: per-request K / FLA-scatter is not supported."
        )
    if output_state_indices is not None:
        raise NotImplementedError(
            "gdn_decode_bf16_wy_output_only: split-pool (output_state_indices) is not supported."
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
        "gdn_decode_bf16_wy_output_only: use_qk_l2norm_in_kernel=False is not supported "
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
    h0 = initial_state_source.contiguous()
    # n_valid = token rows actually present in the q/k tensors handed to the kernel.
    # Native-short-T (FLASHINFER_GDN_WY_NATIVE_T): pass q/k as the real [B,T,...] tensors
    # (n_valid=T); the kernel loads only those rows and zeros its sK/sQ smem tail,
    # skipping the two big q/k gmem->gmem staging copies. Otherwise q/k are staged
    # into a T_KERNEL-row zero-padded buffer (n_valid=T_KERNEL = original behavior).
    # Native-short-T is gated to T in {4, 8}: only there does T == t_disc, so (a) the
    # t_input-gated output STG writes exactly T rows (compact [B,T] output is safe) and
    # (b) the smem-tail zeroing aligns with the kernel's working-set masking. T=4 is the
    # draft-len-3 verify shape. Other T (1-3,5-7,9-15) fall back to full staging.
    _native = _NATIVE_T and (T == 4 or T == 8)
    n_valid = T if _native else T_KERNEL

    # Contiguity: tensors the kernel reads DIRECTLY from gmem need canonical-compact
    # strides for the CuTe descriptor; staged tensors get this for free from .copy_().
    _qkv_rs = 0  # >0 => strided q/k/v read (token stride = conv_dim); see _STRIDED_QKV
    _ab_native_flag = (
        False  # True => a/b read native [B,n_valid,HV] (no staging); _NATIVE_AB
    )
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

    # For T<T_KERNEL, stage the inputs into persistent, pre-zeroed T_KERNEL-row
    # buffers (keyed by shape AND T so rows [T:T_KERNEL) stay zero) and copy only the
    # T valid rows per call. Those zero rows are load-bearing: the kernel's
    # ldmatrix/MMA reads the full tile (NaN-tail probe confirmed). The native path
    # stages only a/b and moves the q/k/v zero-fill into the kernel (smem tail).
    if T < T_KERNEL:
        if _native and _NATIVE_AB and a.is_contiguous() and b.is_contiguous():
            # Native-a/b: pass the real [B, T(=n_valid), HV] tensors straight to the kernel
            # (batch stride = n_valid*HV, contiguous). The kernel gates the warp-3 load/compute
            # by n_valid, so no T_KERNEL zero-pad staging copy is needed. Bit-exact on the
            # compact [B,T] output (causal prefix-sum isolates the unloaded tail rows).
            _ab_native_flag = True
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
                # (local patch) allocate OUTSIDE inference_mode so they are normal
                # tensors; otherwise the in-place .copy_() is rejected during
                # sglang CUDA-graph capture ("Inplace update to inference tensor").
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
    # One CTA per (b, hv) — full V tile per CTA. Per-CTA SMEM ~29.8 KB -> <=7 CTAs/SM (ncu, B200).
    _total_ctas = HV * B
    _needed = math.ceil(_total_ctas / _num_sms)
    # Cap raised 4 -> 8 (measured on B200, T=16/HV=64): launch bounds mbp=8 makes the
    # compiler fit 64 regs/thread (was 73 -> 80 allocated -> 6-CTA register limit),
    # unlocking the 7-CTA SMEM limit (29.8 KB/CTA): theoretical occupancy 37.5% -> 43.75%,
    # ~1-7% faster across BS=16..256 with bit-identical output. mbp=12 (40 regs) gains no
    # further occupancy (SMEM-capped at 7 CTAs) and is slower — do not raise past 8.
    mbp = max(1, min(_needed + 1, 8))
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
    # q/k/v/out head+feature strides — only the batch mode-0 dim is dynamic). A
    # process mixing HV values (e.g. HV=32 then HV=64) previously reused the first
    # compile and read H0 with the wrong strides -> ~3e-01 garbage outputs. Found
    # by the intense correctness sweep; invisible to the tests/benches, which use
    # one HV per process.
    cache_key: tuple = (
        str(device),
        mbp,
        t_disc,
        n_valid,
        _qkv_rs,
        _ab_native_flag,
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
    if output is not None and T == T_KERNEL:
        out16 = output
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
        mk_dyn(out16),
        scale,
        HV,
        V_dim,
        H,
        stream,
    ]

    if cache_key not in _CACHE:
        _CACHE[cache_key] = cute.compile(
            GdnDecodeKernel(
                disable_state_update=True,
                min_blocks_per_mp=mbp,
                t_input=t_disc,
                n_valid=n_valid,
                qkv_row_stride=_qkv_rs,
                ab_native=_ab_native_flag,
            ),
            *args,
        )
    _CACHE[cache_key](*args)

    if output is None:
        return out16[:, :T]  # zero-copy view of the valid tokens
    if T < T_KERNEL:
        output.copy_(out16[:, :T])
    return output


__all__ = ["gated_delta_rule_mtp", "GdnDecodeKernel", "K_DIM", "V_DIM_C"]
