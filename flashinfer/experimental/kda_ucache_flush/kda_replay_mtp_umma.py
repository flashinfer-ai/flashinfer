"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
KDA (Kimi K3) ReplaySSM verify + ring append + per-request flush on
tcgen05 / UMMA / TMEM (SM10x), the ``backend="umma"`` arm of
``kda_delta_rule_mtp_ucache_flush``.

A fork of the B200 GDN UMMA kernel
(``flashinfer/gdn_kernels/gdn_replay_mtp_umma.py``) for bf16 operands and the
per-channel KDA gate. Contract validation lives in the dispatcher
(``kda_decode_bf16_wy_ucache_flush.kda_delta_rule_mtp_ucache_flush``); this
module only builds and launches the kernel.

Math per (request b, head h), with the live window j < P (logical slot j at
physical ring row (base + j) & 31), G_j the window's per-channel cumulative
log-decay, G_P = G_{P-1} (0 at P = 0), bdec = e^{G_P}, w_j = e^{G_P - G_j} <= 1:

    S_h x  = S0 (bdec . x) + sum_j u_j (K~_j . x)          K~_j = bf16(w_j . k̂_j)
    khat_t = k̂_t e^{cum_t},  ktil_s = k̂_s e^{-cum_s},  qhat_t = q̂_t e^{cum_t}
    u_t    = beta_t (v_t - S_h khat_t - sum_{s<t} (khat_t . ktil_s) u_s)
    y_t    = S_h qhat_t + sum_{s<=t} (qhat_t . ktil_s) u_s

CTA = (head, request), grid (H, B), 192 threads, 4 CTAs / SM:
  warp 0   : mbarrier inits, TMA of the bf16 state (two 16 KB SW128 K halves),
             bulk copy of the P live u-window rows into SMEM.
  warp 1   : TMEM allocation at entry (128 columns), the UMMA issuer.
  warps 2-5: the consumers: row-layout L2 norm + k̂ ring append, channel-layout
             gate (g ring append, X_r / X_c / Ktil tiles), K~ build, the
             mma.sync Grams (Gq = X_c Ktil^T, C = X_c K~^T), the D tile
             (raw u, TMEM), the per-row forward substitution epilogue.
UMMAs (kind::f16 with BF16 A / B, fp32 accumulate in TMEM):
  readout  H  = S0 . X_r^T   M128 N16 K128   A: state (SMEM)  B: X_r (K-major SW128)
  history  H += D . C^T      M128 N16 K16    A: D (TMEM)      B: C (SW32)
  fold     F  = D . K~       M128 N64 K16    A: D (TMEM)      B: K~ half (MN-major SW128)
X_r = [khat; qhat] (.) bdec rows 0..2T-1 (zero rows after), so H holds S_h khat_t in
column t and S_h qhat_t in column T + t.

Flush rows (hist_len >= flush_min) also fold the window into the state, one K half at
a time: S_half <- bf16(bdec . S0_half + F) in place in SMEM (per-channel bdec), then a
TMA store of the half back to the pool slot. Consumer order on flush rows: RMW of half
0 -> epilogue -> RMW of half 1 (the fold UMMA of half 1 hides behind the epilogue).

w_ring = 32 (window up to 28 slots): a second 16-slot j-block (K~ + 4096 B, C + 512 B,
D cols 40..47) built and multiplied only when P > 16 (CTA-uniform), so shallow
windows issue exactly the w_ring = 16 work; u is read from the ring per row (ldg)
by default instead of the SMEM window copy.

T = 1..8 natively (STP T = 1 included). X rows 2T..15 of the operand tiles are zero;
at 2T % 8 != 0 the last row-layout pass also covers the dead rows 2T..8 NP - 1
(clamped in-range loads, zero stores) and the epilogue reads 2^ceil(log2 T) TMEM
columns per side, as the GDN UMMA kernel. T in {4, 8} builds are unchanged.

Status (stage S3c + T = 1..8): w_ring in {16, 32}, one head per CTA.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.gdn_kernels import _umma_helpers as U
from flashinfer.gdn_kernels.device_target import (
    gdn_compile_options,
    gdn_device_target,
)

RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1
K_DIM = 128
V_DIM = 128
XC_PITCH = 272  # bytes per row of the row-major X_c / Ktil tiles (136 bf16)
LOG2_E = 1.4426950408889634
_EPS = 1e-6  # inside the L2-norm rsqrt, as the HMMA kernel
_TMEM_COLS = 128
_MIN_BLOCKS_PER_SM = 4  # 128 of the 512 TMEM columns, ~54 KB SMEM, 80 registers
# TMEM columns: H = R + Y (readout + history) at 0..15, D (raw u, bf16 pairs) at
# 32..39 for window slots j 0..15 and 40..47 for j 16..31 (w_ring = 32).
_COL_H = 0
_COL_D = 32
_COL_F = 64  # fold F = D . K~ for the current state K half (64 columns)
# s_coef fp32 words: A[t][s] at 0, Bq[t][s] at 64, beta[t] at 128
_CO_A, _CO_BQ, _CO_BETA = 0, 64, 128

# Test hooks (default None = automatic choice). Not part of the API.
_FORCE_U_SRC: Optional[str] = None  # "smem" | "ldg"
_FORCE_HPC: Optional[int] = None
# None = automatic (T <= 4 and more (request, head) items than 4 CTAs x SMs); True / False
# force the persistent kernel (kda_replay_mtp_umma_persistent) on / off.
_FORCE_PERSISTENT: Optional[bool] = None
# Number of UMMA launches issued (dispatch tests read it).
_LAUNCHES = 0

# Stage-supported configurations; widened as the kernel grows.
_SUPPORTED_T: frozenset = frozenset(range(1, 9))
_SUPPORTED_W_RING: frozenset = frozenset({16, 32})
_KERNEL_IMPLEMENTED = True


# ----------------------------------------------------------------------------- helpers
# KDA-local variants of the GDN UMMA helpers (flashinfer/gdn_kernels/_umma_helpers.py is
# shared and stays untouched).


def _bf16_idesc(m: int, n: int, *, b_mn_major: bool = False):
    """kind::f16 instruction descriptor with BF16 A and B (a_format / b_format = 1, bits
    7 and 10), fp32 D. ``U.f16_idesc`` leaves both format fields 0 (= F16)."""
    return U.f16_idesc(m, n, b_mn_major=b_mn_major) | Uint32((1 << 7) | (1 << 10))


def _sel(cond, a, b, ty):
    """Branch-free typed select (arith.select)."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


@dsl_user_op
def _ldmatrix_x4(addr: Int32, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.x4.m8n8.shared.b16 from this lane's row address. For an
    m16n8k16 A fragment of a row-major tile: lane l addresses row (l % 16) at column
    offset (l // 16) * 8 elements; returns (a0, a1, a2, a3)."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldmatrix_x2(addr: Int32, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.x2.m8n8.shared.b16 from this lane's row address. For an
    m16n8k16 B fragment of an [n][k] row-major tile: lane l (l < 16 used) addresses row
    n = l % 8 at column offset ((l // 8) % 2) * 8 elements; returns (b0, b1)."""
    st = llvm.StructType.get_literal([_T.i32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _ex2(x: Float32, *, loc=None, ip=None) -> Float32:
    """ex2.approx.ftz.f32 (the HMMA kernel's _exp2_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


def _exp(x):
    """e^x as the HMMA kernel's _exp_approx_f32: ex2.approx.ftz(x * log2 e)."""
    return _ex2(x * Float32(LOG2_E))


@dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx.ftz.f32 (the HMMA kernel's _rsqrt_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _fold_fma2_bf16x2(
    w: Uint32,
    bd_lo: Float32,
    bd_hi: Float32,
    f0: Float32,
    f1: Float32,
    *,
    loc=None,
    ip=None,
):
    """Fold of one packed bf16 state pair with PER-CHANNEL bdec: (lo * bd_lo + f0,
    hi * bd_hi + f1) as fp32, one fma.rn each (the HMMA kernel's _fold_fma2_bf16x2)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([_T.f32(), _T.f32()]),
        [
            Uint32(w).ir_value(loc=loc, ip=ip),
            Float32(bd_lo).ir_value(loc=loc, ip=ip),
            Float32(bd_hi).ir_value(loc=loc, ip=ip),
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 _wl, _wh; .reg .f32 _fl, _fh;"
        " shl.b32 _wl, $2, 16;"
        " and.b32 _wh, $2, 0xFFFF0000;"
        " mov.b32 _fl, _wl;"
        " mov.b32 _fh, _wh;"
        " fma.rn.f32 $0, _fl, $3, $5; fma.rn.f32 $1, _fh, $4, $6; }",
        "=f,=f,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(_T.f32(), r, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(_T.f32(), r, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bar_sync_ua(barrier_id: int, number_of_threads: int, *, loc=None, ip=None) -> None:
    """Named-barrier sync without ``.aligned`` (``barrier.cta.sync``), for every named
    barrier used at more than one site (compute-sanitizer synccheck flags a multi-site
    ``.aligned`` barrier as divergent). Arrive side: ``cute.arch.barrier_arrive(...,
    aligned=False)``."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_u32(addr: Int32, w: Uint32, *, loc=None, ip=None) -> None:
    """st.shared.b32."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_u16_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Predicated ld.global.u16 (zero-extended); 0 where pred == 0."""
    r = llvm.inline_asm(
        _T.i32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tmov.u32 $0, 0;\n\t"
        "setp.ne.s32 p, $1, 0;\n\t@p ld.global.u16 h, [$2];\n\t"
        "@p cvt.u32.u16 $0, h;\n\t}",
        "=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def _ldg_f32x4_if(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """Predicated ld.global.v4.f32 (16 B aligned): four fp32 values, 0 where pred == 0."""
    st = llvm.StructType.get_literal([_T.f32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.f32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=f,=f,=f,=f,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _lds_f32x4(addr: Int32, *, loc=None, ip=None):
    """ld.shared.v4.f32 -> four fp32 values."""
    st = llvm.StructType.get_literal([_T.f32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


def _x_sw128_off(row, c):
    """Byte offset of element (row, k = c) in a 16-row K-major SW128 operand tile (X_r,
    and K̃ per 16-row j-block): two 64-element K halves 2048 B apart, 128 B rows, 16 B
    chunks XOR-swizzled with (row % 8). Works on Python ints and DSL integers."""
    return (
        (c // 64) * 2048
        + row * 128
        + ((((c % 64) // 8) ^ (row % 8)) * 16)
        + (c % 8) * 2
    )


def _c_sw32_off(xc, j):
    """Byte offset of C[x = xc][j] (j < 16) in one SW32 [16 x][16 j] bf16 j-block."""
    return xc * 32 + (((j // 8) ^ ((xc >> 2) & 1)) * 16) + (j % 8) * 2


def _u_window_copy(src_head, uw, mbar_addr, mbar, P, base):
    """Bulk-copy the P live u rows (logical j = 0..P-1 at physical (base + j) & 31) of
    one head into SMEM as [j][V]. The window wraps the ring at most once: two contiguous
    copies, the second predicated off without a wrap. Call under elect_one, P > 0."""
    rowb = V_DIM * 2
    n1 = _sel(base + P <= RING_SLOTS, P, Int32(RING_SLOTS) - base, Int32)
    n2 = P - n1
    cute.arch.mbarrier_arrive_and_expect_tx(mbar, P * rowb)
    U.bulk_g2s(uw, src_head + Int64(base * rowb), n1 * rowb, mbar_addr)
    _bulk_g2s_if(Int32(n2 > 0), uw + n1 * rowb, src_head, n2 * rowb, mbar_addr)


@dsl_user_op
def _bulk_g2s_if(
    pred: Int32,
    dst_smem: Int32,
    src_gmem: Int64,
    nbytes: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
):
    """Predicated ``U.bulk_g2s`` (cp.async.bulk global -> shared, complete_tx)."""
    llvm.inline_asm(
        _T.i32(),
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [$2], [$3], $4, [$5];\n\tmov.u32 $0, 0;\n\t}\n",
        "=r,r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _kt_block1(kc_row, gc_row, gp_addr, kt_addr, base, P, jr, c8):
    """w_ring = 32: K~ j-block 1 (window slots j = 16..31, zero rows past P) into
    s_kt + 4096, the same per-thread map and arithmetic as block 0. Call only when
    P > 16 (CTA-uniform); its ring loads are issued here, not hoisted, so the w16 /
    shallow-window path keeps its registers."""
    j = 16 + jr
    kpred = Int32(j < P)
    kslot = (base + j) & RING_MASK
    ka_ = kc_row + Int64((kslot * K_DIM + c8 * 8) * 2)
    kraw = [U.ldg_v4_if(kpred, ka_), U.ldg_v4_if(kpred, ka_ + Int64(128))]
    ga_ = gc_row + Int64((kslot * K_DIM + c8 * 8) * 4)
    gj = []
    for gk in range(2):
        for i4 in range(2):
            gj += _ldg_f32x4_if(kpred, ga_ + Int64(gk * 256 + i4 * 16))
    for gk in range(2):
        gpv = _lds_f32x4(gp_addr + (gk * 64 + c8 * 8) * 4) + _lds_f32x4(
            gp_addr + (gk * 64 + c8 * 8 + 4) * 4
        )
        kw = []
        for i in range(4):
            w0 = _exp(gpv[2 * i] - gj[gk * 8 + 2 * i])
            w1 = _exp(gpv[2 * i + 1] - gj[gk * 8 + 2 * i + 1])
            pk = U.pack_bf16x2(
                U.bf16x2_lo_f32(kraw[gk][i]) * w0,
                U.bf16x2_hi_f32(kraw[gk][i]) * w1,
            )
            kw.append(_sel(kpred != 0, pk, Uint32(0), Uint32))
        U.sts_v4(
            kt_addr + 4096 + gk * 2048 + jr * 128 + ((c8 ^ (jr % 8)) * 16),
            kw[0],
            kw[1],
            kw[2],
            kw[3],
        )


def _c_ntile_block1(nt, lane, a_ln, kt_addr, c_addr):
    """w_ring = 32: C n-tile nt in {2, 3} (window slots 8 nt .. 8 nt + 7, j-block 1) =
    X_c . K~^T by mma.sync, packed to bf16 into the SW32 j-block 1 of s_c (+512 B).
    Same fragment addressing as the block-0 tiles. Call only when P > 16."""
    grp = lane >> 2
    c4 = lane & 3
    acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    jl = 8 * (nt % 2) + (lane % 8)  # row of K~ j-block 1
    kt_row = kt_addr + 4096 + jl * 128
    for ks in range(8):
        fa = _ldmatrix_x4(a_ln + ks * 32)
        chunk = ks * 2 + ((lane // 8) % 2)
        fb = _ldmatrix_x2(
            kt_row + (chunk // 8) * 2048 + (((chunk % 8) ^ (jl % 8)) * 16)
        )
        acc = U.mma_bf16_16816(fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc)
    jc = 8 * (nt % 2) + 2 * c4
    cb = c_addr + 512
    _sts_u32(cb + _c_sw32_off(grp, jc), U.pack_bf16x2(acc[0], acc[1]))
    _sts_u32(cb + _c_sw32_off(grp + 8, jc), U.pack_bf16x2(acc[2], acc[3]))
    cute.arch.fence_view_async_shared()  # read by the j-block 1 history UMMA


def _d_block1(use_smem, u_ptr, base, P, ct, tbase):
    """w_ring = 32: D columns 40..47 = raw u of window slots j = 16..31 (zero past
    P), from the SMEM window (u_ptr = this row's s_uwin byte address) or the ring
    (u_ptr = this row's u_cache[sidx, h, 0, row] address). Call only when P > 16."""
    uv = []
    for j in range(16, 32):
        if use_smem:
            uv.append(U.lds_u16(u_ptr + j * (V_DIM * 2)))
        else:
            uv.append(
                _ldg_u16_if(
                    Int32(j < P),
                    u_ptr + Int64(((base + j) & RING_MASK) * (V_DIM * 2)),
                )
            )
    words = []
    for i in range(8):
        lo = _sel(16 + 2 * i < P, uv[2 * i], Uint32(0), Uint32)
        hi = _sel(16 + 2 * i + 1 < P, uv[2 * i + 1], Uint32(0), Uint32)
        words.append(lo | (hi << Uint32(16)))
    U.tmem_st_u32(ct, tbase + _COL_D + 8, words)


def _rmw(g, ct, row, tbase, st_addr, bdec_addr, f_epi_a, f_gdone):
    """Flush rows: S_half <- bf16(bdec . S0_half + F) in place on state K half g (TMEM
    F at cols 64..127, this thread's V row = TMEM lane), then publish the half to warp
    0's TMA store (f_gdone). bdec is per channel (s_bdec, a warp-uniform broadcast
    read). Half 0 releases the F columns (f_epi_a) after its last TMEM load so the fold
    UMMA of half 1 can overwrite them."""
    ldn = 32  # TMEM columns per load
    nch = ldn // 8  # 16-B (8-channel) chunks per load
    srow = st_addr + g * 16384 + row * 128
    for grp in range(64 // ldn):
        fv = U.tmem_ld(ct, tbase + _COL_F + grp * ldn, ldn)
        U.tmem_wait_ld()
        if g == 0 and grp == 64 // ldn - 1:
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(f_epi_a)
        for cc in range(nch):
            chunk = grp * nch + cc  # logical 8-channel chunk of this half
            ba = bdec_addr + (g * 64 + chunk * 8) * 4
            bd = _lds_f32x4(ba) + _lds_f32x4(ba + 16)
            sa = srow + ((chunk ^ (row % 8)) * 16)
            ow = U.lds_v4(sa)
            nw = []
            for i in range(4):
                lo, hi = _fold_fma2_bf16x2(
                    ow[i],
                    bd[2 * i],
                    bd[2 * i + 1],
                    fv[cc * 8 + 2 * i],
                    fv[cc * 8 + 2 * i + 1],
                )
                nw.append(U.pack_bf16x2(lo, hi))
            U.sts_v4(sa, nw[0], nw[1], nw[2], nw[3])
    cute.arch.fence_view_async_shared()  # the TMA store reads the half (async proxy)
    cute.arch.mbarrier_arrive(f_gdone)


# ----------------------------------------------------------------------------- kernel
class KdaReplayMtpUmma:
    def __init__(self, T: int, h: int, w_ring: int = 16, u_src: str = "smem"):
        assert 1 <= T <= 8, T
        assert w_ring in (16, 32), w_ring
        assert u_src in ("smem", "ldg"), u_src
        self.T = T
        self.h = h
        self.w_ring = w_ring
        self.u_src = u_src
        self.num_threads = 192

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op):
        elems = 64
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, 1, V_DIM, (elems, 1), 1),
                stride=(0, 0, elems, (1, V_DIM * elems), V_DIM * elems),
            ),
        )
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, elems)),
            smem_layout,
            cta_tiler=(1, 1, V_DIM, elems),
        )

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        g,
        b,
        a_log,
        dt_bias,
        state,
        state_idx,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        out,
        scale: Float32,
        lower_bound: Float32,
        flush_min: Int32,
        stream,
    ):
        tma_ld = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        tma_st = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel(
            q,
            k,
            v,
            g,
            b,
            a_log,
            dt_bias,
            state_idx,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            out,
            scale,
            lower_bound,
            flush_min,
            tma_ld,
            tma_st,
        ).launch(
            grid=(self.h, q.shape[0], 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=_MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        g: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_idx_t: cute.Tensor,
        k_cache: cute.Tensor,
        u_cache: cute.Tensor,
        g_cache: cute.Tensor,
        hist_len: cute.Tensor,
        cache_base: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        lower_bound: Float32,
        flush_min: Int32,
        tma_ld: cpasync.TmaInfo,
        tma_st: cpasync.TmaInfo,
    ):
        T = self.T
        T2 = 2 * T
        NP = (T2 + 7) // 8  # row-layout passes (8 X rows per pass)
        # T not in {4, 8}: the last pass also holds dead rows 2T..8 NP - 1 (zeroed)
        PARTIAL = T2 % 8 != 0
        WR = self.w_ring
        USE_SMEM_U = self.u_src == "smem"
        NJB = WR // 16  # 16-slot window j-blocks (2 at w_ring = 32)

        tid, _, _ = cute.arch.thread_idx()
        h, brow, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            BFloat16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        # X_r: readout B operand, 16 rows x 128 k bf16, K-major SW128 (_x_sw128_off)
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * K_DIM), byte_alignment=1024
        )
        # K~: window keys w_j . k̂_j (j < P, else 0), NJB 16-row j-blocks 4096 B apart,
        # same layout
        s_kt = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(NJB * 16 * K_DIM), byte_alignment=1024
        )
        # X_c (16 rows) and Ktil (8 rows), row-major, 272 B pitch (ldmatrix conflict-free)
        s_xc = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * XC_PITCH // 2), byte_alignment=128
        )
        s_ktil = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(8 * XC_PITCH // 2), byte_alignment=128
        )
        # C: history B operand, SW32 [16 x][16 j] bf16 per j-block (512 B apart)
        s_c = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(NJB * 16 * 16), byte_alignment=256
        )
        if cutlass.const_expr(USE_SMEM_U):
            # u window [j][V] bf16, rows j < P bulk-copied by warp 0
            s_uwin = smem.allocate_tensor(
                cutlass.Uint16, cute.make_layout(WR * V_DIM), byte_alignment=128
            )
        s_gram = smem.allocate_tensor(
            Float32, cute.make_layout(16 * 8), byte_alignment=16
        )
        s_coef = smem.allocate_tensor(Float32, cute.make_layout(136), byte_alignment=16)
        s_gp = smem.allocate_tensor(Float32, cute.make_layout(K_DIM), byte_alignment=16)
        # bdec = e^{G_P} per channel (flush RMW)
        s_bdec = smem.allocate_tensor(
            Float32, cute.make_layout(K_DIM), byte_alignment=16
        )
        st_full = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2)]
        u_full = smem.allocate_array(cutlass.Int64, 1)
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        d_ready = smem.allocate_array(cutlass.Int64, 1)
        mma_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        f_mma_a = smem.allocate_array(cutlass.Int64, 1)
        f_mma_b = smem.allocate_array(cutlass.Int64, 1)
        f_epi_a = smem.allocate_array(cutlass.Int64, 1)
        f_gdone = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2)]
        tmem_addr = smem.allocate(Int32, 4)

        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (GDN: allocating late spaced CTAs ~2.5 us apart).
        # Named barrier 2 (warp 1 arrive + 128 consumers) publishes the address.
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
            U.fence_before_sync()
            cute.arch.barrier_arrive(barrier_id=2, number_of_threads=160)

        # Cursors first (every thread), then the mbarrier inits while they are in flight.
        sidx = Int32(state_idx_t[brow])
        P = Int32(hist_len[brow])
        base = Int32(cache_base[brow])
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(st_full[0], 1)
                cute.arch.mbarrier_init(st_full[1], 1)
                cute.arch.mbarrier_init(u_full, 1)
                cute.arch.mbarrier_init(x_ready, 128)
                cute.arch.mbarrier_init(d_ready, 128)
                cute.arch.mbarrier_init(mma_done, 1)
                cute.arch.mbarrier_init(tmem_done, 128)
                cute.arch.mbarrier_init(f_mma_a, 1)
                cute.arch.mbarrier_init(f_mma_b, 1)
                cute.arch.mbarrier_init(f_epi_a, 128)
                cute.arch.mbarrier_init(f_gdone[0], 128)
                cute.arch.mbarrier_init(f_gdone[1], 128)
                cute.arch.mbarrier_init_fence()
            cute.arch.fence_acq_rel_cta()
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=64)
            cute.arch.barrier_arrive(barrier_id=4, number_of_threads=160)

        # Batch-row-only inputs, issued by EVERY thread ahead of the padded-row exit so
        # their latency overlaps the cursors' (warps 0 / 1 load in-range dummies).
        ct = (tid - 64) & 127  # consumer index: channel c / (row r, lane l) / TMEM lane
        row = ((warp & 3) << 5) | lane  # V row of this consumer thread (= TMEM lane)
        r8 = ct >> 4  # row layout: X row r8 + 8 p, 8 k values at l16 * 8
        l16 = ct & 15
        xw = []
        for p in cutlass.range_constexpr(NP):
            xr = r8 + 8 * p
            is_q = xr >= T
            t_k = _sel(is_q, 0, xr, Int32)
            t_q = _sel(is_q, xr - T, 0, Int32)
            if cutlass.const_expr(PARTIAL and p == NP - 1):
                # dead rows xr >= 2T: an in-range dummy load (q token 0), zeroed in P1
                t_q = _sel(xr >= T2, 0, t_q, Int32)
            ak = k.iterator.toint() + Int64(
                cute.crd2idx((brow, t_k, h, l16 * 8), k.layout)
            ) * Int64(2)
            aq = q.iterator.toint() + Int64(
                cute.crd2idx((brow, t_q, h, l16 * 8), q.layout)
            ) * Int64(2)
            xw.append(U.ldg_v4(_sel(is_q, aq, ak, Int64)))
        graw = []
        for t in cutlass.range_constexpr(T):
            graw.append(g[brow, t, h, ct])
        vv = []
        for t in cutlass.range_constexpr(T):
            vv.append(v[brow, t, h, row])
        tb = _sel(ct < T, ct, 0, Int32)
        braw = b[brow, tb, h]
        dtb = Float32(dt_bias[h * K_DIM + ct])
        alog = Float32(a_log[h])

        if warp == 1:
            if sidx < 0:
                # padded row: hand the TMEM columns back before the CTA exits
                cute.arch.sync_warp()
                U.tmem_dealloc(tmem_addr[0], _TMEM_COLS)
        U.exit_if_negative(sidx)
        is_flush = flush_min <= P

        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.iterator.toint()
        xc_addr = s_xc.iterator.toint()
        ktil_addr = s_ktil.iterator.toint()
        c_addr = s_c.iterator.toint()
        gp_addr = s_gp.iterator.toint()

        if warp == 0:
            # ============================================================ producer
            cpasync.prefetch_descriptor(tma_ld.atom)
            if cutlass.const_expr(USE_SMEM_U):
                # the P live u rows (window order) ahead of the 32 KB state: the D build
                # needs them before the readout needs the state
                src_head = u_cache.iterator.toint() + Int64(
                    cute.crd2idx((sidx, h, 0, 0), u_cache.layout)
                ) * Int64(2)
                uw = s_uwin.iterator.toint()
                if P > 0:
                    with cute.arch.elect_one():
                        _u_window_copy(src_head, uw, u_full.toint(), u_full, P, base)
            for gk in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(st_full[gk], 16384)
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[sidx, h, None, (None, gk)],
                    s_state[None, (None, gk)],
                    st_full[gk],
                )
            if is_flush:
                # store each folded K half back to the pool slot once its RMW is done
                cpasync.prefetch_descriptor(tma_st.atom)
                for gk in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_wait(f_gdone[gk], 0)
                    U.simple_tma_copy(
                        tma_st.atom,
                        s_state[None, (None, gk)],
                        tma_st.tma_tensor[sidx, h, None, (None, gk)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
        elif warp == 1:
            # ============================================================ UMMA issuer
            tbase = tmem_addr[0]
            sd = U.sdesc_sw128(V_DIM * 128)
            # descriptor offsets are added, not OR-ed (see the S1 notes: OR is only
            # correct while the base address has those bits clear)
            x_desc = sd + Uint64(x_addr >> 4)
            stage_desc = [
                sd + Uint64(st_addr >> 4),
                sd + Uint64((st_addr + 16384) >> 4),
            ]
            c_desc = U.sdesc_sw32(Uint64(c_addr >> 4))
            # MN-major SW128 K~: SBO = 1024 B between 8-slot K atoms, N halves 2048 B apart
            kt_desc = U.sdesc_mn_sw128(2048, 1024) + Uint64(kt_addr >> 4)
            id_r = _bf16_idesc(128, 16)
            id_f = _bf16_idesc(128, 64, b_mn_major=True)
            cute.arch.barrier(barrier_id=3, number_of_threads=64)  # warp 0's inits
            cute.arch.mbarrier_wait(x_ready, 0)
            U.fence_after_sync()
            # readout H = S0 . X_r^T (cols 0..15), per state K half as it lands
            for gk in cutlass.range_constexpr(2):
                cute.arch.mbarrier_wait(st_full[gk], 0)
                U.fence_after_sync()
                for ka in cutlass.range_constexpr(4):
                    U.umma_ss(
                        tbase + _COL_H,
                        stage_desc[gk] + Uint64((ka * 32) >> 4),
                        x_desc + Uint64((gk * 2048 + ka * 32) >> 4),
                        id_r,
                        (gk > 0) or (ka > 0),
                    )
            # history accumulates INTO H: H = S_h . X
            cute.arch.mbarrier_wait(d_ready, 0)
            U.fence_after_sync()
            if P > 0:
                U.umma_ts(tbase + _COL_H, tbase + _COL_D, c_desc, id_r, True)
            if cutlass.const_expr(WR == 32):
                # window slots 16..31: D cols 40..47 x C j-block 1
                if P > 16:
                    U.umma_ts(
                        tbase + _COL_H,
                        tbase + _COL_D + 8,
                        U.sdesc_sw32(Uint64((c_addr + 512) >> 4)),
                        id_r,
                        True,
                    )
            U.umma_commit(mma_done)
            if is_flush:
                # fold F = D . K~ per state K half into cols 64..127. The commit of half
                # 0 also covers the readout, so the consumers' RMW may overwrite s_state
                # once f_mma_a completes. Half 1 reuses the F columns after the RMW of
                # half 0 has read them (f_epi_a).
                U.umma_ts(tbase + _COL_F, tbase + _COL_D, kt_desc, id_f, False)
                if cutlass.const_expr(WR == 32):
                    # flush rows at w32 have P >= flush_min > 16 (dispatcher check),
                    # so j-block 1 (K~ + 4096 B, D cols 40..47) is always live here
                    U.umma_ts(
                        tbase + _COL_F,
                        tbase + _COL_D + 8,
                        kt_desc + Uint64(4096 >> 4),
                        id_f,
                        True,
                    )
                U.umma_commit(f_mma_a)
                cute.arch.mbarrier_wait(f_epi_a, 0)
                U.fence_after_sync()
                U.umma_ts(
                    tbase + _COL_F,
                    tbase + _COL_D,
                    kt_desc + Uint64(2048 >> 4),
                    id_f,
                    False,
                )
                if cutlass.const_expr(WR == 32):
                    U.umma_ts(
                        tbase + _COL_F,
                        tbase + _COL_D + 8,
                        kt_desc + Uint64((2048 + 4096) >> 4),
                        id_f,
                        True,
                    )
                U.umma_commit(f_mma_b)
            cute.arch.mbarrier_wait(tmem_done, 0)
            U.fence_after_sync()
            U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            # ============================================================ consumers
            cw = warp - 2
            gc_row = g_cache.iterator.toint() + Int64(
                cute.crd2idx((sidx, h, 0, 0), g_cache.layout)
            ) * Int64(4)
            kc_row = k_cache.iterator.toint() + Int64(
                cute.crd2idx((sidx, h, 0, 0), k_cache.layout)
            ) * Int64(2)
            # G_P of channel ct (last live window row; 0 at P = 0)
            gp_slot = (base + P - 1) & RING_MASK
            gp = Float32(g_cache[sidx, h, gp_slot, ct])
            gp = _sel(P > 0, gp, 0.0, Float32)
            # K~ inputs of this thread: window slot jr, 8 channels c8 * 8 of each K half
            jr = ct >> 3
            c8 = ct & 7
            kpred = Int32(jr < P)
            kslot = (base + jr) & RING_MASK
            ka_ = kc_row + Int64((kslot * K_DIM + c8 * 8) * 2)
            kraw = [U.ldg_v4_if(kpred, ka_), U.ldg_v4_if(kpred, ka_ + Int64(128))]
            ga_ = gc_row + Int64((kslot * K_DIM + c8 * 8) * 4)
            gj = []
            for gk in cutlass.range_constexpr(2):
                for i4 in cutlass.range_constexpr(2):
                    gj += _ldg_f32x4_if(kpred, ga_ + Int64(gk * 256 + i4 * 16))

            # -------- P1: L2 norm (row layout) + k̂ ring append + raw k̂ / q̂ into X_c
            for p in cutlass.range_constexpr(NP):
                xr = r8 + 8 * p
                ss = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    lo = U.bf16x2_lo_f32(xw[p][i])
                    hi = U.bf16x2_hi_f32(xw[p][i])
                    ss = ss + lo * lo
                    ss = ss + hi * hi
                for off in (8, 4, 2, 1):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, off)
                inv = _rsqrt_approx(ss + Float32(_EPS))
                inv = _sel(xr >= T, inv * scale, inv, Float32)
                nw = []
                for i in cutlass.range_constexpr(4):
                    nw.append(
                        U.pack_bf16x2(
                            U.bf16x2_lo_f32(xw[p][i]) * inv,
                            U.bf16x2_hi_f32(xw[p][i]) * inv,
                        )
                    )
                xcw = nw
                if cutlass.const_expr(PARTIAL and p == NP - 1):
                    # dead rows xr >= 2T: zero X_c here and X_r (P2 writes rows < 2T)
                    xcw = [_sel(xr < T2, w_, Uint32(0), Uint32) for w_ in nw]
                    if xr >= T2:
                        zd = Uint32(0)
                        U.sts_v4(
                            x_addr
                            + (l16 // 8) * 2048
                            + xr * 128
                            + (((l16 % 8) ^ (xr % 8)) * 16),
                            zd,
                            zd,
                            zd,
                            zd,
                        )
                U.sts_v4(
                    xc_addr + xr * XC_PITCH + l16 * 16, xcw[0], xcw[1], xcw[2], xcw[3]
                )
                if xr < T:
                    # ring append: the L2-normed, UNDECAYED k̂ (the ring contract)
                    kdst = kc_row + Int64(
                        (((base + P + xr) & RING_MASK) * K_DIM + l16 * 8) * 2
                    )
                    U.stg_v4(kdst, nw[0], nw[1], nw[2], nw[3])
            z = Uint32(0)
            for p in cutlass.range_constexpr(NP, 2):
                # X_c rows 8 NP..15 are zero (rows 2T..8 NP - 1: in P1 above)
                U.sts_v4(xc_addr + (r8 + 8 * p) * XC_PITCH + l16 * 16, z, z, z, z)
            # X_r rows 8 NP..15 and Ktil rows T..7 are zero
            for p in cutlass.range_constexpr(NP, 2):
                zr = r8 + 8 * p
                U.sts_v4(
                    x_addr
                    + (l16 // 8) * 2048
                    + zr * 128
                    + (((l16 % 8) ^ (zr % 8)) * 16),
                    z,
                    z,
                    z,
                    z,
                )
            if cutlass.const_expr(T < 8):
                if ct < (8 - T) * 16:
                    U.sts_v4(ktil_addr + (T + r8) * XC_PITCH + l16 * 16, z, z, z, z)
            _bar_sync_ua(1, 128)

            # -------- P2: gate + operand tiles (channel layout, c = ct)
            c = ct
            s_gp[c] = gp
            bdec = _sel(P > 0, _exp(gp), 1.0, Float32)
            s_bdec[c] = bdec
            gbase = _sel(is_flush, 0.0, gp, Float32)
            eA = _exp(alog)
            cum = Float32(0.0)
            for t in cutlass.range_constexpr(T):
                xg = Float32(graw[t]) + dtb
                sig = Float32(1.0) / (Float32(1.0) + _exp(Float32(0.0) - eA * xg))
                cum = cum + lower_bound * sig
                g_cache[sidx, h, (base + P + t) & RING_MASK, c] = gbase + cum
                ep = _exp(cum)
                en = _exp(Float32(0.0) - cum)
                eb = ep * bdec
                kh = U.u16_as_f32(U.lds_u16(xc_addr + t * XC_PITCH + c * 2), True)
                qh = U.u16_as_f32(U.lds_u16(xc_addr + (T + t) * XC_PITCH + c * 2), True)
                # X_c = [khat; qhat] (Gram / C A operand)
                U.sts_u16(xc_addr + t * XC_PITCH + c * 2, U.f32_to_u16(kh * ep, True))
                U.sts_u16(
                    xc_addr + (T + t) * XC_PITCH + c * 2, U.f32_to_u16(qh * ep, True)
                )
                # Ktil = k̂ e^{-cum} (<= e^{80}: bf16 holds it)
                U.sts_u16(ktil_addr + t * XC_PITCH + c * 2, U.f32_to_u16(kh * en, True))
                # X_r = X_c . bdec, rounded once from fp32 (readout B operand)
                U.sts_u16(x_addr + _x_sw128_off(t, c), U.f32_to_u16(kh * eb, True))
                U.sts_u16(x_addr + _x_sw128_off(T + t, c), U.f32_to_u16(qh * eb, True))
            if ct < T:
                s_coef[_CO_BETA + ct] = Float32(1.0) / (
                    Float32(1.0) + _exp(Float32(0.0) - Float32(braw))
                )
            cute.arch.barrier(barrier_id=4, number_of_threads=160)  # warp 0's inits
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(x_ready)
            _bar_sync_ua(1, 128)  # s_gp, X_c, Ktil visible to every consumer

            # -------- P3: K~ = bf16(k̂_j . e^{G_P - G_j}) (j < P, else 0)
            for gk in cutlass.range_constexpr(2):
                gpv = _lds_f32x4(gp_addr + (gk * 64 + c8 * 8) * 4) + _lds_f32x4(
                    gp_addr + (gk * 64 + c8 * 8 + 4) * 4
                )
                kw = []
                for i in cutlass.range_constexpr(4):
                    w0 = _exp(gpv[2 * i] - gj[gk * 8 + 2 * i])
                    w1 = _exp(gpv[2 * i + 1] - gj[gk * 8 + 2 * i + 1])
                    pk = U.pack_bf16x2(
                        U.bf16x2_lo_f32(kraw[gk][i]) * w0,
                        U.bf16x2_hi_f32(kraw[gk][i]) * w1,
                    )
                    kw.append(_sel(kpred != 0, pk, Uint32(0), Uint32))
                U.sts_v4(
                    kt_addr + gk * 2048 + jr * 128 + ((c8 ^ (jr % 8)) * 16),
                    kw[0],
                    kw[1],
                    kw[2],
                    kw[3],
                )
            if cutlass.const_expr(WR == 32):
                if P > 16:
                    _kt_block1(kc_row, gc_row, gp_addr, kt_addr, base, P, jr, c8)
            _bar_sync_ua(1, 128)

            # -------- P4: mma.sync Grams. warp cw 0: Gq = X_c . Ktil^T (fp32, s_gram);
            # warps cw 1, 2: C n-tiles (j 0..7, 8..15) = X_c . K~^T (bf16, SW32 s_c)
            grp = lane >> 2
            c4 = lane & 3
            a_ln = xc_addr + (lane % 16) * XC_PITCH + (lane // 16) * 16
            if cw == 0:
                acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                b_ln = ktil_addr + (lane % 8) * XC_PITCH + ((lane // 8) % 2) * 16
                for ks in cutlass.range_constexpr(8):
                    fa = _ldmatrix_x4(a_ln + ks * 32)
                    fb = _ldmatrix_x2(b_ln + ks * 32)
                    acc = U.mma_bf16_16816(
                        fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc
                    )
                s_gram[grp * 8 + 2 * c4] = acc[0]
                s_gram[grp * 8 + 2 * c4 + 1] = acc[1]
                s_gram[(grp + 8) * 8 + 2 * c4] = acc[2]
                s_gram[(grp + 8) * 8 + 2 * c4 + 1] = acc[3]
                if cutlass.const_expr(WR == 32):
                    if P > 16:
                        _c_ntile_block1(3, lane, a_ln, kt_addr, c_addr)
            elif cw < 3:
                nt = cw - 1
                acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                jl = 8 * nt + (lane % 8)
                kt_row = kt_addr + jl * 128
                for ks in cutlass.range_constexpr(8):
                    fa = _ldmatrix_x4(a_ln + ks * 32)
                    chunk = ks * 2 + ((lane // 8) % 2)
                    fb = _ldmatrix_x2(
                        kt_row + (chunk // 8) * 2048 + (((chunk % 8) ^ (jl % 8)) * 16)
                    )
                    acc = U.mma_bf16_16816(
                        fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc
                    )
                jc = 8 * nt + 2 * c4
                _sts_u32(c_addr + _c_sw32_off(grp, jc), U.pack_bf16x2(acc[0], acc[1]))
                _sts_u32(
                    c_addr + _c_sw32_off(grp + 8, jc), U.pack_bf16x2(acc[2], acc[3])
                )
                # C is read by the history UMMA (warp 1 issues it after d_ready)
                cute.arch.fence_view_async_shared()
            else:
                if cutlass.const_expr(WR == 32):
                    if P > 16:
                        _c_ntile_block1(2, lane, a_ln, kt_addr, c_addr)
            _bar_sync_ua(1, 128)

            # -------- P5: intra-step coefficient tables (fp32, masked by select)
            NT = T * T
            if ct < 2 * NT:
                tab = ct // NT
                tt = (ct // T) % T
                ss_ = ct % T
                gv = s_gram[(tab * T + tt) * 8 + ss_]
                bt = s_coef[_CO_BETA + tt]
                av = _sel(ss_ < tt, bt * gv, 0.0, Float32)
                bq = _sel(ss_ <= tt, gv, 0.0, Float32)
                if tab == 0:
                    s_coef[_CO_A + tt * 8 + ss_] = av
                else:
                    s_coef[_CO_BQ + tt * 8 + ss_] = bq

            # -------- P6: D = raw u_j (j < P, bf16 pairs) into TMEM (lane = V row)
            uv = []
            if cutlass.const_expr(USE_SMEM_U):
                if P > 0:
                    cute.arch.mbarrier_wait(u_full, 0)
                ub = s_uwin.iterator.toint() + row * 2
                u_ptr = ub
                for j in cutlass.range_constexpr(16):
                    uv.append(U.lds_u16(ub + j * (V_DIM * 2)))
            else:
                ucr = u_cache.iterator.toint() + Int64(
                    cute.crd2idx((sidx, h, 0, row), u_cache.layout)
                ) * Int64(2)
                u_ptr = ucr
                for j in cutlass.range_constexpr(16):
                    uv.append(
                        _ldg_u16_if(
                            Int32(j < P),
                            ucr + Int64(((base + j) & RING_MASK) * (V_DIM * 2)),
                        )
                    )
            words = []
            for i in cutlass.range_constexpr(8):
                lo = _sel(2 * i < P, uv[2 * i], Uint32(0), Uint32)
                hi = _sel(2 * i + 1 < P, uv[2 * i + 1], Uint32(0), Uint32)
                words.append(lo | (hi << Uint32(16)))
            cute.arch.barrier(barrier_id=2, number_of_threads=160)  # TMEM address
            U.fence_after_sync()
            tbase = tmem_addr[0]
            U.tmem_st_u32(ct, tbase + _COL_D, words)
            if cutlass.const_expr(WR == 32):
                if P > 16:
                    _d_block1(USE_SMEM_U, u_ptr, base, P, ct, tbase)
            U.tmem_wait_st()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(d_ready)
            _bar_sync_ua(1, 128)  # coefficient tables

            st_addr_c = s_state.iterator.toint()
            bdec_addr = s_bdec.iterator.toint()
            if is_flush:
                # fold RMW of K half 0 first: its UMMA was issued right after the history
                cute.arch.mbarrier_wait(f_mma_a, 0)
                U.fence_after_sync()
                _rmw(0, ct, row, tbase, st_addr_c, bdec_addr, f_epi_a, f_gdone[0])

            # -------- epilogue: per V row forward substitution
            cute.arch.mbarrier_wait(mma_done, 0)
            U.fence_after_sync()
            nld = 1 << (T - 1).bit_length()  # tcgen05.ld widths are powers of two
            hk = U.tmem_ld(ct, tbase + _COL_H, nld)
            hq = U.tmem_ld(ct, tbase + _COL_H + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            if not is_flush:
                # verify rows: release TMEM before the solve (warp 1 deallocates early)
                cute.arch.mbarrier_arrive(tmem_done)
            uu: list = []
            for t in cutlass.range_constexpr(T):
                bt = s_coef[_CO_BETA + t]
                acc_u = bt * (Float32(vv[t]) - hk[t])
                for s_ in cutlass.range_constexpr(t):
                    acc_u = acc_u - s_coef[_CO_A + t * 8 + s_] * uu[s_]
                uu.append(acc_u)
            for t in cutlass.range_constexpr(T):
                y = hq[t]
                for s_ in cutlass.range_constexpr(t + 1):
                    y = y + s_coef[_CO_BQ + t * 8 + s_] * uu[s_]
                out[brow, t, h, row] = BFloat16(y)
                u_cache[sidx, h, (base + P + t) & RING_MASK, row] = BFloat16(uu[t])
            if is_flush:
                # fold RMW of K half 1 (its UMMA ran behind the epilogue), then TMEM is free
                cute.arch.mbarrier_wait(f_mma_b, 0)
                U.fence_after_sync()
                _rmw(1, ct, row, tbase, st_addr_c, bdec_addr, f_epi_a, f_gdone[1])
                U.fence_before_sync()
                cute.arch.mbarrier_arrive(tmem_done)

    # ------------------------------------------------------------------ compile
    @staticmethod
    def compile(device: torch.device, T: int, h: int, w_ring: int, u_src: str):
        """Compile one specialisation for ``device``'s architecture (callers use
        ``_get_kernel``); batch and pool sizes are dynamic."""
        batch = cute.sym_int()
        pool = cute.sym_int()
        ft = U.make_fake_tensor_lead_dyn
        args = [
            ft(BFloat16, (batch, T, h, K_DIM), 8),  # q
            ft(BFloat16, (batch, T, h, K_DIM), 8),  # k
            ft(BFloat16, (batch, T, h, V_DIM), 8),  # v
            ft(BFloat16, (batch, T, h, K_DIM), 8),  # g (raw gate pre-activation)
            ft(BFloat16, (batch, T, h), 1),  # b (beta logits)
            ft(Float32, (h,), 1),  # A_log (fp32)
            ft(Float32, (h * K_DIM,), 1),  # dt_bias (fp32, [H * K])
            ft(BFloat16, (pool, h, V_DIM, K_DIM), 8),  # state pool (bf16)
            ft(Int32, (batch,), 1),  # state indices
            ft(BFloat16, (pool, h, RING_SLOTS, K_DIM), 8),  # k ring
            ft(BFloat16, (pool, h, RING_SLOTS, V_DIM), 8),  # u ring
            ft(Float32, (pool, h, RING_SLOTS, K_DIM), 4),  # g ring (per channel)
            ft(Int32, (batch,), 1),  # hist_len
            ft(Int32, (batch,), 1),  # cache_base
            ft(BFloat16, (batch, T, h, V_DIM), 8),  # out
        ]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile[gdn_compile_options(device, cute.EnableTVMFFI(True))](
            KdaReplayMtpUmma(T, h, w_ring, u_src),
            *args,
            Float32(1.0),
            Float32(-5.0),
            Int32(1),
            stream,
        )


_KERNELS: dict = {}


def _get_kernel(device: torch.device, T: int, h: int, w_ring: int, u_src: str):
    """The compiled kernel for this specialisation on ``device``'s architecture."""
    key = (gdn_device_target(device).compile_key, T, h, w_ring, u_src)
    fn = _KERNELS.get(key)
    if fn is None:
        fn = KdaReplayMtpUmma.compile(device, T, h, w_ring, u_src)
        _KERNELS[key] = fn
    return fn


def kda_umma_unavailable_reason(
    dev: torch.device, T: int, w_ring: int, pdl_trigger: bool
) -> Optional[str]:
    """``None`` when the UMMA backend can run this configuration, else why not."""
    if torch.cuda.get_device_capability(dev)[0] != 10:
        cc = torch.cuda.get_device_capability(dev)
        return f"needs SM10x (tcgen05), got SM{cc[0]}{cc[1]}"
    if not _KERNEL_IMPLEMENTED:
        return "the UMMA kernel is not implemented yet"
    if T not in _SUPPORTED_T:
        return f"T={T} not supported (supported: {sorted(_SUPPORTED_T)})"
    if w_ring not in _SUPPORTED_W_RING:
        return f"w_ring={w_ring} not supported (supported: {sorted(_SUPPORTED_W_RING)})"
    if pdl_trigger:
        return "pdl_trigger=True not supported yet"
    return None


def _u_src(w_ring: int) -> str:
    if _FORCE_U_SRC is not None:
        assert _FORCE_U_SRC in ("smem", "ldg"), _FORCE_U_SRC
        return _FORCE_U_SRC
    return "smem" if w_ring == 16 else "ldg"


def _use_persistent(B: int, T: int, H: int, w_ring: int, dev: torch.device) -> bool:
    """The persistent kernel where it measured faster (B200, H = 12, w_ring = 16, ncu
    kernel time vs one item per CTA): T <= 6 once the (request, head) items outnumber
    the resident CTA slots (4 per SM): T = 4 / 6 +9-20 % at B >= 512; T = 8 only with
    16+ items per slot (+2-3 % at B >= 1024, -3.5 % at B = 512). w_ring = 32 (u ring
    read per row, 32-slot K~) measured 30 % slower: never."""
    if _FORCE_PERSISTENT is not None:
        return bool(_FORCE_PERSISTENT)
    if w_ring != 16:
        return False
    slots = (
        _MIN_BLOCKS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count
    )
    items = B * H
    return items > slots if T <= 6 else items >= 16 * slots


def launch_umma(
    q,
    k,
    v,
    g,
    b,
    A_log,
    dt_bias,
    h0,
    initial_state_indices,
    k_cache,
    u_cache,
    g_cache,
    hist_len,
    cache_base,
    out_t,
    scale: float,
    lower_bound: float,
    flush_min: int,
    w_ring: int,
) -> None:
    """Launch the UMMA kernel on already-validated, contiguous inputs (the dispatcher's
    contract: bf16 io / state / k and u rings, fp32 g ring, fp32 A_log [H] and dt_bias
    [H * K], H == HV, K == V == 128)."""
    global _LAUNCHES
    T = q.shape[1]
    H = q.shape[2]
    if _use_persistent(q.shape[0], T, H, int(w_ring), q.device):
        from .kda_replay_mtp_umma_persistent import launch_umma_persistent

        launch_umma_persistent(
            q,
            k,
            v,
            g,
            b,
            A_log,
            dt_bias,
            h0,
            initial_state_indices,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            out_t,
            scale,
            lower_bound,
            flush_min,
            w_ring,
        )
        _LAUNCHES += 1
        return
    fn = _get_kernel(q.device, T, H, int(w_ring), _u_src(int(w_ring)))
    fn(
        q,
        k,
        v,
        g,
        b,
        A_log,
        dt_bias,
        h0,
        initial_state_indices,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        out_t,
        float(scale),
        float(lower_bound),
        int(flush_min),
    )
    _LAUNCHES += 1


__all__ = ["kda_umma_unavailable_reason", "launch_umma"]
