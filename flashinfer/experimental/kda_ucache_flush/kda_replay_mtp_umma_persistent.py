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
tcgen05 / UMMA / TMEM (SM10x): the PERSISTENT variant of ``kda_replay_mtp_umma``,
launched by its ``launch_umma`` for T <= 4 when the (request, head) items outnumber the
resident CTA slots (4 per SM). Same math, same results (bitwise).

Persistent layout (see /home/scratch.ameyn_gpu_3/kda_umma_dram_log.md for the study):
  * grid = 4 CTAs / SM, CTA c runs items c, c + grid, ... (item = request * H + head);
    every role walks the same list; mbarrier phases count processed items (f_* ones flush
    items); padded items are skipped by all roles.
  * 256 threads with the GDN register split (warps 0-3 setmaxnreg 24, consumers on warps
    4-7 at 104): warp 0 TMA producer, warp 1 UMMA issuer, warps 2-3 idle.
  * the next item's state load starts as soon as the current readout has consumed the stage
    (st_free; flush items: after warp 0's stores have read it); warp 0 also bulk-prefetches
    the next item's row inputs, ring window and state into L2; the consumers issue the next
    item's row inputs one item ahead.
  * it removes the per-CTA launch gap (~1 us with TMEM alloc / dealloc) and the cursor round
    trip of every item; it is then consumer-compute bound (P2 gate math, P3 K~ build), so it
    is used only where it wins (T <= 4; T = 8 measured 2-4 % slower than one item per CTA).

The original single-item description follows.

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

import os
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
from flashinfer.gdn_kernels._umma_helpers_rubin import (
    bulk_prefetch_l2_n as _bulk_prefetch_l2_n,
)
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
_REGSPLIT = (24, 104)  # setmaxnreg (warps 0-3, consumers) of the 256-thread layout
_REGSPLIT_HELP = (32, 96)  # the same with the K~ helper warps 2-3
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
_FORCE_PERSISTENT: Optional[bool] = None
# IKET warp-phase probes for profiling studies (pic-c --warp-phase-trace): KDA_UMMA_IKET=1
# at compile time emits cute.experimental.iket ranges (compile with
# CUTE_DSL_COMPILER_OPT=iket). Off by default; then no probe code is emitted.
_IKET = os.environ.get("KDA_UMMA_IKET", "0") == "1"
_FORCE_SPLIT: Optional[bool] = None
_HANG_DEBUG_ADDR = 0  # debug: pinned int32 buffer [CTAs, 8] for watchdog waits
_KT_CH = False  # K~ in channel layout, live window rows only (uniform skip of j >= P)
_HELP_P3 = False  # split layout: warps 2-3 build K~ (P3) for the consumers
_RING_PF = False  # consumers: next item's ring words in registers one item ahead
_PF_WHERE = "producer"  # debug: L2 prefetch issuer (producer / consumer)
_HANG_PROGRESS_ADDR = 0  # debug: pinned int32 [CTAs, 8] progress words
_L2_PREFETCH = True  # persistent: prefetch the next item into L2 (debug switch)
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
def _ldg_f32_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Float32:
    """Predicated ld.global.f32; 0 where pred == 0."""
    r = llvm.inline_asm(
        _T.f32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p ld.global.f32 $0, [$2];\n\t}",
        "=f,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


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


@dsl_user_op
def _stg_f32(addr: Int64, x: Float32, *, loc=None, ip=None) -> None:
    """st.global.f32."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(x).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


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
    copies, each predicated off when empty; P = 0 completes the phase with no bytes.
    Call under elect_one."""
    rowb = V_DIM * 2
    n1 = _sel(base + P <= RING_SLOTS, P, Int32(RING_SLOTS) - base, Int32)
    n2 = P - n1
    cute.arch.mbarrier_arrive_and_expect_tx(mbar, P * rowb)
    _bulk_g2s_if(Int32(n1 > 0), uw, src_head + Int64(base * rowb), n1 * rowb, mbar_addr)
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


def _cursors(state_idx_t, hist_len, cache_base, item, nit, nh):
    """(state slot, hist_len, cache_base) of work item ``item`` = request * nh + head
    (past the last item: request 0's, never processed). Raw loads, nothing consumes
    them here: the next item's cursors are issued one item ahead, and a select on
    them would wait for the round trip at once."""
    br = _sel(item < nit, item // nh, 0, Int32)
    return Int32(state_idx_t[br]), Int32(hist_len[br]), Int32(cache_base[br])


def _ik_push(on: bool, name: str, payload=None) -> None:
    """IKET range push (no code unless ``on``, a compile-time flag)."""
    if on:
        if payload is None:
            cute.experimental.iket.range_push(name)
        else:
            cute.experimental.iket.range_push(name, Int32(payload))


def _ik_pop(on: bool) -> None:
    if on:
        cute.experimental.iket.range_pop()


@dsl_user_op
def _mbwait_dbg(
    mbar_addr: Int32,
    parity: Int32,
    rec: Int64,
    tag: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Debug mbarrier wait (``_HANG_DEBUG_ADDR`` != 0): spin on try_wait.parity; after
    200 ms write ``tag`` to the record address (pinned host memory) once and keep
    waiting, so a hung launch can be inspected from the host."""
    llvm.inline_asm(
        None,
        [
            Int32(mbar_addr).ir_value(loc=loc, ip=ip),
            Int32(parity).ir_value(loc=loc, ip=ip),
            Int64(rec).ir_value(loc=loc, ip=ip),
            Int32(tag).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p, w;\n\t.reg .u64 t0, t1;\n\t"
        "mov.u64 t0, %globaltimer;\n\tsetp.eq.u32 w, 0, 0;\n"
        "WAIT${:uid}:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@p bra DONE${:uid};\n\t"
        "@!w bra WAIT${:uid};\n\t"
        "mov.u64 t1, %globaltimer;\n\tsub.u64 t1, t1, t0;\n\t"
        "setp.lt.u64 p, t1, 200000000;\n\t@p bra WAIT${:uid};\n\t"
        "st.global.u32 [$2], $3;\n\tfence.sc.sys;\n\tsetp.ne.u32 w, 0, 0;\n\t"
        "bra WAIT${:uid};\n"
        "DONE${:uid}:\n\t}",
        "r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prog_st(addr: Int64, v: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _mbw(mbar, parity, tag, cta, warp, kk):
    """mbarrier wait; with the hang-debug build a watchdog wait that records
    (site tag, item index) per (CTA, warp) in pinned host memory."""
    if _HANG_PROGRESS_ADDR != 0:
        _prog_st(
            Int64(_HANG_PROGRESS_ADDR)
            + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4),
            Int32(tag) + (Int32(kk) + 1) * 256,
        )
    if _HANG_DEBUG_ADDR == 0:
        cute.arch.mbarrier_wait(mbar, parity)
    else:
        rec = Int64(_HANG_DEBUG_ADDR) + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4)
        _mbwait_dbg(
            mbar.toint(), Int32(parity), rec, Int32(tag) + (Int32(kk) + 1) * 256
        )


@dsl_user_op
def _mat_i32(x: Int32, *, loc=None, ip=None) -> Int32:
    """Copy through a side-effecting mov: waits for x's load at this point and returns
    an unscoreboarded register. Loop-carried cursors are materialised before the next
    item's input loads are issued, else the loop-top branch on them waits for those
    loads too (ptxas groups the loads on one scoreboard)."""
    r = llvm.inline_asm(
        _T.i32(),
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(r)


def _next_cursors(persistent, state_idx_t, hist_len, cache_base, item, nit, nh, cur):
    """The next item's cursors (persistent grid); one item per CTA otherwise (no loads,
    the current cursors stand in and are never used)."""
    if persistent:
        return _cursors(state_idx_t, hist_len, cache_base, item, nit, nh)
    return cur


def _load_x(q, k, g, b, dt_bias, a_log, brow, hh, ct, r8, l16, T):
    """Row inputs of head hh for this consumer thread: the row-layout k / q words (X row
    r8 + 8 p, 8 k values at l16 * 8; dead rows 2T.. load q token 0), the raw gate of
    channel ct for every token, beta logit and dt_bias. (A_log is read from SMEM: a
    loop-carried CTA-uniform global load drew an early R2UR that waited on it.)"""
    T2 = 2 * T
    NP = (T2 + 7) // 8
    xw_ = []
    for p in range(NP):
        xr = r8 + 8 * p
        is_q = xr >= T
        t_k = _sel(is_q, 0, xr, Int32)
        t_q = _sel(is_q, xr - T, 0, Int32)
        if T2 % 8 != 0 and p == NP - 1:
            # dead rows xr >= 2T: an in-range dummy load (q token 0), zeroed in P1
            t_q = _sel(xr >= T2, 0, t_q, Int32)
        ak = k.iterator.toint() + Int64(
            cute.crd2idx((brow, t_k, hh, l16 * 8), k.layout)
        ) * Int64(2)
        aq = q.iterator.toint() + Int64(
            cute.crd2idx((brow, t_q, hh, l16 * 8), q.layout)
        ) * Int64(2)
        xw_.append(U.ldg_v4(_sel(is_q, aq, ak, Int64)))
    graw_ = []
    for t in range(T):
        graw_.append(Float32(g[brow, t, hh, ct]))
    tb_ = _sel(ct < T, ct, 0, Int32)
    braw_ = Float32(b[brow, tb_, hh])
    dtb_ = Float32(dt_bias[hh * K_DIM + ct])
    return xw_, graw_, braw_, dtb_


def _load_ring(g_cache, k_cache, sidx, hh, P, base, ct, jr, c8):
    """Ring inputs of item (sidx, hh) for this consumer thread: G_P of channel ct (0 at
    P = 0) and the raw k̂ / G words of window slot jr, 8 channels c8 * 8 of each K half
    (zero for jr >= P). sidx < 0: in-range dummy loads (slot 0, predicated off)."""
    ok = sidx >= 0
    sl = _sel(ok, sidx, 0, Int32)
    pp = _sel(ok, P, 0, Int32)
    gp_slot = (base + pp - 1) & RING_MASK
    gp = Float32(g_cache[sl, hh, gp_slot, ct])
    gp = _sel(pp > 0, gp, 0.0, Float32)
    kpred = Int32(jr < pp)
    kslot = (base + jr) & RING_MASK
    kc_row = k_cache.iterator.toint() + Int64(
        cute.crd2idx((sl, hh, 0, 0), k_cache.layout)
    ) * Int64(2)
    gc_row = g_cache.iterator.toint() + Int64(
        cute.crd2idx((sl, hh, 0, 0), g_cache.layout)
    ) * Int64(4)
    ka_ = kc_row + Int64((kslot * K_DIM + c8 * 8) * 2)
    kraw = [U.ldg_v4_if(kpred, ka_), U.ldg_v4_if(kpred, ka_ + Int64(128))]
    ga_ = gc_row + Int64((kslot * K_DIM + c8 * 8) * 4)
    gj = []
    for gk in range(2):
        for i4 in range(2):
            gj += _ldg_f32x4_if(kpred, ga_ + Int64(gk * 256 + i4 * 16))
    return gp, kraw, gj


def _load_v(v, brow, hh, row, T):
    """v[brow, t, hh, row] for every token (fp32)."""
    return [Float32(v[brow, t, hh, row]) for t in range(T)]


def _prefetch_rings_l2(k_cache, g_cache, u_cache, sidx, hh, P, base):
    """L2 prefetch (evict_last) of head hh's live ring window: the P k / g / u rows at
    physical (base + j) & 31, as at most two contiguous runs per ring. Warp 0, elect_one."""
    n1 = _sel(base + P <= RING_SLOTS, P, Int32(RING_SLOTS) - base, Int32)
    n2 = P - n1
    for ring, rb in ((k_cache, K_DIM * 2), (g_cache, K_DIM * 4), (u_cache, V_DIM * 2)):
        el = 4 if rb == K_DIM * 4 else 2
        head = ring.iterator.toint() + Int64(
            cute.crd2idx((sidx, hh, 0, 0), ring.layout)
        ) * Int64(el)
        _bulk_prefetch_l2_n(head + Int64(base * rb), n1 * rb, Int32(n1 > 0))
        _bulk_prefetch_l2_n(head, n2 * rb, Int32(n2 > 0))


def _issue_state(tma_ld, s_state, st_full, sidx, hh):
    """TMA both 16 KB K halves of head hh's bf16 state into the stage (warp 0)."""
    for gk in range(2):
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(st_full[gk], 16384)
        U.simple_tma_copy(
            tma_ld.atom,
            tma_ld.tma_tensor[sidx, hh, None, (None, gk)],
            s_state[None, (None, gk)],
            st_full[gk],
        )


# ----------------------------------------------------------------------------- kernel
class KdaReplayMtpUmma:
    def __init__(
        self,
        T: int,
        h: int,
        w_ring: int = 16,
        u_src: str = "smem",
        persistent: bool = False,
        split: bool = False,
    ):
        assert 1 <= T <= 8, T
        assert w_ring in (16, 32), w_ring
        assert u_src in ("smem", "ldg"), u_src
        self.T = T
        self.h = h
        self.w_ring = w_ring
        self.u_src = u_src
        # persistent: a fixed grid walks the (request, head) items, overlapping the next
        # item's state load with the current one and prefetching it into L2
        self.persistent = persistent
        # register split (GDN UMMA kernel's): 256 threads, consumers on warps 4-7 at
        # setmaxnreg 104, warps 0-3 (producer, issuer, two idle) at 24
        self.split = split
        self.num_threads = 256 if split else 192
        self.iket = _IKET

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
        ncta: Int32,
        stream,
    ):
        tma_ld = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        tma_st = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel(
            state,
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
            grid=(ncta, 1, 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=_MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Tensor,
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
        cta, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        NH = self.h
        CW0 = 4 if self.split else 2  # first consumer warp
        HELP = self.split and _HELP_P3  # warps 2-3 build K~
        KTC = _KT_CH and not HELP
        RS = _REGSPLIT_HELP if HELP else _REGSPLIT
        IK = self.iket
        # work items = (request, head), head fastest; CTA cta runs items cta, cta + ncta, ..
        # (one item per CTA when the grid covers them all)
        NIT = q.shape[0] * NH
        nk = (NIT - cta + ncta - 1) // ncta

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
        # A_log of every head (H <= 128), filled once by the consumers
        s_alog = smem.allocate_tensor(
            Float32, cute.make_layout(K_DIM), byte_alignment=16
        )
        # bdec = e^{G_P} per channel (flush RMW)
        s_bdec = smem.allocate_tensor(
            Float32, cute.make_layout(K_DIM), byte_alignment=16
        )
        # mbarriers: one phase per processed item (it), the f_* ones per flush item (fit)
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
        # the state stage is free for the next item once its readout UMMAs completed (and,
        # on flush items, warp 0's stores have read it); the u window once the D build has
        # read it
        st_free = smem.allocate_array(cutlass.Int64, 1)
        u_free = smem.allocate_array(cutlass.Int64, 1)
        kt_ready = smem.allocate_array(cutlass.Int64, 1)  # helper warps: K~ built
        tmem_addr = smem.allocate(Int32, 4)

        if cutlass.const_expr(_HANG_PROGRESS_ADDR != 0):
            if lane == 0:
                _prog_st(
                    Int64(_HANG_PROGRESS_ADDR)
                    + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4),
                    Int32(1),
                )
        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (GDN: allocating late spaced CTAs ~2.5 us apart).
        # Named barrier 2 (warp 1 arrive + 128 consumers) publishes the address.
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
            U.fence_before_sync()
            cute.arch.barrier_arrive(barrier_id=2, number_of_threads=160)

        # Cursors of the first item (every thread), then the mbarrier inits while they
        # are in flight. Every role walks the same item list with the same cursors.
        sidx, P, base = _cursors(state_idx_t, hist_len, cache_base, cta, NIT, NH)
        ct = (
            tid - 32 * CW0
        ) & 127  # consumer index: channel c / (row r, lane l) / TMEM lane
        row = ((warp & 3) << 5) | lane  # V row of this consumer thread (= TMEM lane)
        r8 = ct >> 4  # row layout: X row r8 + 8 p, 8 k values at l16 * 8
        l16 = ct & 15
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
                cute.arch.mbarrier_init(st_free, 1)
                cute.arch.mbarrier_init(u_free, 128)
                cute.arch.mbarrier_init(kt_ready, 64)
                cute.arch.mbarrier_init_fence()
            cute.arch.fence_acq_rel_cta()
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=64)
            cute.arch.barrier_arrive(barrier_id=4, number_of_threads=160)

        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.iterator.toint()
        xc_addr = s_xc.iterator.toint()
        ktil_addr = s_ktil.iterator.toint()
        c_addr = s_c.iterator.toint()
        gp_addr = s_gp.iterator.toint()
        st_addr = s_state.iterator.toint()

        if warp < CW0:
            if cutlass.const_expr(self.split):
                cute.arch.setmaxregister_decrease(RS[0])
            if warp == 0:
                # ============================================================ producer
                cpasync.prefetch_descriptor(tma_ld.atom)
                cpasync.prefetch_descriptor(tma_st.atom)
                it = Int32(0)
                fit = Int32(0)
                for kk in cutlass.range(nk, unroll=1):
                    item = cta + kk * ncta
                    brow = item // NH
                    hh = item - brow * NH
                    nitem = item + ncta
                    nsidx, nP, nbase = _next_cursors(
                        self.persistent,
                        state_idx_t,
                        hist_len,
                        cache_base,
                        nitem,
                        NIT,
                        NH,
                        (sidx, P, base),
                    )
                    if sidx >= 0:
                        ph = it & 1
                        is_flush = flush_min <= P
                        _ik_push(IK, "item", item)
                        _ik_push(IK, "p_wait_st")
                        if it > 0:
                            # the previous item's readout has consumed the stage (flush
                            # items: its stores were waited at the end of its iteration)
                            _mbw(st_free, ph ^ 1, 2, cta, warp, kk)
                        _ik_pop(IK)
                        if cutlass.const_expr(self.persistent):
                            _issue_state(tma_ld, s_state, st_full, sidx, hh)
                        if cutlass.const_expr(USE_SMEM_U):
                            # the P live u rows (window order), once the previous item's D
                            # build has read the window (u_free). Persistent: behind the
                            # state, which the readout needs first; one item per CTA:
                            # ahead of it (the D build needs the few rows long before the
                            # readout needs the 32 KB). P = 0 still completes the phase.
                            _ik_push(IK, "p_wait_u")
                            if it > 0:
                                _mbw(u_free, ph ^ 1, 3, cta, warp, kk)
                            _ik_pop(IK)
                            src_head = u_cache.iterator.toint() + Int64(
                                cute.crd2idx((sidx, hh, 0, 0), u_cache.layout)
                            ) * Int64(2)
                            with cute.arch.elect_one():
                                _u_window_copy(
                                    src_head,
                                    s_uwin.iterator.toint(),
                                    u_full.toint(),
                                    u_full,
                                    P,
                                    base,
                                )
                        if cutlass.const_expr(not self.persistent):
                            _issue_state(tma_ld, s_state, st_full, sidx, hh)
                        if cutlass.const_expr(
                            self.persistent and _L2_PREFETCH and _PF_WHERE == "producer"
                        ):
                            pn = Int32(nitem < NIT) * Int32(nsidx >= 0)
                            nb = _sel(pn != 0, nitem // NH, 0, Int32)
                            nh = _sel(pn != 0, nitem - (nitem // NH) * NH, 0, Int32)
                            nsl = _sel(pn != 0, nsidx, 0, Int32)
                            with cute.arch.elect_one():
                                for t in cutlass.range_constexpr(T):
                                    for src in (q, k, g, v):
                                        _bulk_prefetch_l2_n(
                                            src.iterator.toint()
                                            + Int64(
                                                cute.crd2idx((nb, t, nh, 0), src.layout)
                                            )
                                            * Int64(2),
                                            Int32(K_DIM * 2),
                                            pn,
                                        )
                                _prefetch_rings_l2(
                                    k_cache, g_cache, u_cache, nsl, nh, nP * pn, nbase
                                )
                                _bulk_prefetch_l2_n(
                                    state.iterator.toint()
                                    + Int64(cute.crd2idx((nsl, nh, 0, 0), state.layout))
                                    * Int64(2),
                                    Int32(V_DIM * K_DIM * 2),
                                    pn,
                                )
                        if is_flush:
                            # store each folded K half back to the pool slot once its RMW is
                            # done; the stage is reloaded only after the stores have read it
                            _ik_push(IK, "p_store")
                            fph = fit & 1
                            for gk in cutlass.range_constexpr(2):
                                _mbw(f_gdone[gk], fph, 4, cta, warp, kk)
                                U.simple_tma_copy(
                                    tma_st.atom,
                                    s_state[None, (None, gk)],
                                    tma_st.tma_tensor[sidx, hh, None, (None, gk)],
                                )
                                cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            _ik_pop(IK)
                            fit = fit + 1
                        _ik_pop(IK)  # item
                        it = it + 1
                    sidx = nsidx
                    P = nP
                    base = nbase
                if cutlass.const_expr(_HANG_PROGRESS_ADDR != 0):
                    _prog_st(
                        Int64(_HANG_PROGRESS_ADDR)
                        + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4),
                        Int32(16),
                    )
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
                it = Int32(0)
                fit = Int32(0)
                for kk in cutlass.range(nk, unroll=1):
                    item = cta + kk * ncta
                    brow = item // NH
                    hh = item - brow * NH
                    nsidx, nP, nbase = _next_cursors(
                        self.persistent,
                        state_idx_t,
                        hist_len,
                        cache_base,
                        item + ncta,
                        NIT,
                        NH,
                        (sidx, P, base),
                    )
                    if sidx >= 0:
                        ph = it & 1
                        is_flush = flush_min <= P
                        _ik_push(IK, "item", item)
                        _ik_push(IK, "i_wait_x")
                        _mbw(x_ready, ph, 5, cta, warp, kk)
                        if it > 0:
                            # the previous item's epilogue has read H (and its RMW F)
                            _mbw(tmem_done, ph ^ 1, 6, cta, warp, kk)
                        _ik_pop(IK)
                        U.fence_after_sync()
                        # readout H = S0 . X_r^T (cols 0..15), per state K half as it lands
                        for gk in cutlass.range_constexpr(2):
                            _ik_push(IK, "i_wait_st")
                            _mbw(st_full[gk], ph, 7, cta, warp, kk)
                            _ik_pop(IK)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + _COL_H,
                                    stage_desc[gk] + Uint64((ka * 32) >> 4),
                                    x_desc + Uint64((gk * 2048 + ka * 32) >> 4),
                                    id_r,
                                    (gk > 0) or (ka > 0),
                                )
                        U.umma_commit(st_free)  # the next item's state may land
                        # history accumulates INTO H: H = S_h . X
                        _ik_push(IK, "i_wait_d")
                        _mbw(d_ready, ph, 8, cta, warp, kk)
                        _ik_pop(IK)
                        U.fence_after_sync()
                        if P > 0:
                            U.umma_ts(
                                tbase + _COL_H, tbase + _COL_D, c_desc, id_r, True
                            )
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
                            # fold F = D . K~ per state K half into cols 64..127. The commit
                            # of half 0 also covers the readout, so the consumers' RMW may
                            # overwrite s_state once f_mma_a completes. Half 1 reuses the F
                            # columns after the RMW of half 0 has read them (f_epi_a).
                            U.umma_ts(
                                tbase + _COL_F, tbase + _COL_D, kt_desc, id_f, False
                            )
                            if cutlass.const_expr(WR == 32):
                                # flush rows at w32 have P >= flush_min > 16 (dispatcher
                                # check), so j-block 1 (K~ + 4096 B, D cols 40..47) is live
                                U.umma_ts(
                                    tbase + _COL_F,
                                    tbase + _COL_D + 8,
                                    kt_desc + Uint64(4096 >> 4),
                                    id_f,
                                    True,
                                )
                            U.umma_commit(f_mma_a)
                            _ik_push(IK, "i_wait_rmw0")
                            _mbw(f_epi_a, fit & 1, 9, cta, warp, kk)
                            _ik_pop(IK)
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
                            fit = fit + 1
                        _ik_pop(IK)  # item
                        it = it + 1
                    sidx = nsidx
                    P = nP
                    base = nbase
                if it > 0:
                    _mbw(tmem_done, (it - 1) & 1, 10, cta, warp, -1)
                U.fence_after_sync()
                U.tmem_dealloc(tbase, _TMEM_COLS)
                if cutlass.const_expr(_HANG_PROGRESS_ADDR != 0):
                    _prog_st(
                        Int64(_HANG_PROGRESS_ADDR)
                        + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4),
                        Int32(15),
                    )
                else:
                    if cutlass.const_expr(HELP):
                        # ======================================== K~ helpers (warps 2-3)
                        # K~ = bf16(k̂_j . e^{G_P - G_j}) for window slot hj, channel
                        # chunks c8 = 2 hq, 2 hq + 1 of both K halves (the consumers'
                        # P3 arithmetic), off the consumers' critical path. s_kt is free
                        # once the previous item's history UMMA (its C tiles read K~
                        # before d_ready) and fold UMMAs (flush) have completed.
                        hthr = tid - 64
                        hj = hthr >> 2
                        hq = hthr & 3
                        it = Int32(0)
                        fit = Int32(0)
                        pfl = Int32(0)
                        for kk in cutlass.range(nk, unroll=1):
                            item = cta + kk * ncta
                            brow = item // NH
                            hh = item - brow * NH
                            nsidx, nP, nbase = _next_cursors(
                                self.persistent,
                                state_idx_t,
                                hist_len,
                                cache_base,
                                item + ncta,
                                NIT,
                                NH,
                                (sidx, P, base),
                            )
                            if sidx >= 0:
                                is_flush = flush_min <= P
                                if it > 0:
                                    _mbw(mma_done, (it - 1) & 1, 15, cta, warp, kk)
                                    if pfl != 0:
                                        _mbw(f_mma_b, (fit - 1) & 1, 16, cta, warp, kk)
                                gp_slot = (base + P - 1) & RING_MASK
                                g0 = Float32(g_cache[sidx, hh, gp_slot, 2 * hthr])
                                g1 = Float32(g_cache[sidx, hh, gp_slot, 2 * hthr + 1])
                                s_gp[2 * hthr] = _sel(P > 0, g0, 0.0, Float32)
                                s_gp[2 * hthr + 1] = _sel(P > 0, g1, 0.0, Float32)
                                _bar_sync_ua(5, 64)
                                kc_row_h = k_cache.iterator.toint() + Int64(
                                    cute.crd2idx((sidx, hh, 0, 0), k_cache.layout)
                                ) * Int64(2)
                                gc_row_h = g_cache.iterator.toint() + Int64(
                                    cute.crd2idx((sidx, hh, 0, 0), g_cache.layout)
                                ) * Int64(4)
                                hpred = Int32(hj < P)
                                hslot = (base + hj) & RING_MASK
                                for ci in cutlass.range_constexpr(2):
                                    c8h = hq * 2 + ci
                                    ka_ = kc_row_h + Int64(
                                        (hslot * K_DIM + c8h * 8) * 2
                                    )
                                    kraw = [
                                        U.ldg_v4_if(hpred, ka_),
                                        U.ldg_v4_if(hpred, ka_ + Int64(128)),
                                    ]
                                    ga_ = gc_row_h + Int64(
                                        (hslot * K_DIM + c8h * 8) * 4
                                    )
                                    gj = []
                                    for gk in cutlass.range_constexpr(2):
                                        for i4 in cutlass.range_constexpr(2):
                                            gj += _ldg_f32x4_if(
                                                hpred, ga_ + Int64(gk * 256 + i4 * 16)
                                            )
                                    for gk in cutlass.range_constexpr(2):
                                        gpv = _lds_f32x4(
                                            gp_addr + (gk * 64 + c8h * 8) * 4
                                        ) + _lds_f32x4(
                                            gp_addr + (gk * 64 + c8h * 8 + 4) * 4
                                        )
                                        kw = []
                                        for i in cutlass.range_constexpr(4):
                                            w0 = _exp(gpv[2 * i] - gj[gk * 8 + 2 * i])
                                            w1 = _exp(
                                                gpv[2 * i + 1] - gj[gk * 8 + 2 * i + 1]
                                            )
                                            pk = U.pack_bf16x2(
                                                U.bf16x2_lo_f32(kraw[gk][i]) * w0,
                                                U.bf16x2_hi_f32(kraw[gk][i]) * w1,
                                            )
                                            kw.append(
                                                _sel(hpred != 0, pk, Uint32(0), Uint32)
                                            )
                                        U.sts_v4(
                                            kt_addr
                                            + gk * 2048
                                            + hj * 128
                                            + ((c8h ^ (hj % 8)) * 16),
                                            kw[0],
                                            kw[1],
                                            kw[2],
                                            kw[3],
                                        )
                                # the fold UMMA reads K~ through the async proxy
                                cute.arch.fence_view_async_shared()
                                cute.arch.mbarrier_arrive(kt_ready)
                                pfl = Int32(is_flush)
                                if is_flush:
                                    fit = fit + 1
                                it = it + 1
                            sidx = nsidx
                            P = nP
                            base = nbase
        else:
            if cutlass.const_expr(self.split):
                cute.arch.setmaxregister_increase(RS[1])
            # ============================================================ consumers
            cw = warp - CW0
            # K~ inputs of this thread: window slot jr, 8 channels c8 * 8 of each K half
            jr = ct >> 3
            c8 = ct & 7
            st_addr_c = s_state.iterator.toint()
            bdec_addr = s_bdec.iterator.toint()
            tbase = Int32(0)
            # the first item's row inputs; later items' are issued at the end of the
            # previous iteration
            # the first item's row inputs; later items' are issued one item ahead
            b0 = _sel(cta < NIT, cta // NH, 0, Int32)
            h0 = _sel(cta < NIT, cta - (cta // NH) * NH, 0, Int32)
            xw, graw, braw, dtb = _load_x(
                q, k, g, b, dt_bias, a_log, b0, h0, ct, r8, l16, T
            )
            vv = _load_v(v, b0, h0, row, T)
            if ct < NH:
                s_alog[ct] = Float32(a_log[ct])  # published by the first P1 barrier
            if cutlass.const_expr(KTC):
                # K~ rows past the window keep the previous item's (finite) values; zero
                # them once so the first item never meets uninitialised SMEM
                zz = Uint32(0)
                for zi in cutlass.range_constexpr(NJB * 2):
                    U.sts_v4(kt_addr + (zi * 128 + ct) * 16, zz, zz, zz, zz)
            # G_P and the K~ ring words of the first item (later items': one item ahead)
            if cutlass.const_expr(_RING_PF):
                gp, kraw, gj = _load_ring(
                    g_cache, k_cache, sidx, h0, P, base, ct, jr, c8
                )
            it = Int32(0)
            fit = Int32(0)
            for kk in cutlass.range(nk, unroll=1):
                item = cta + kk * ncta
                brow = item // NH
                hh = item - brow * NH
                nitem = item + ncta
                nsidx, nP, nbase = _next_cursors(
                    self.persistent,
                    state_idx_t,
                    hist_len,
                    cache_base,
                    nitem,
                    NIT,
                    NH,
                    (sidx, P, base),
                )
                nb = _sel(nitem < NIT, nitem // NH, 0, Int32)
                nh = _sel(nitem < NIT, nitem - (nitem // NH) * NH, 0, Int32)
                if sidx >= 0:
                    ph = it & 1
                    is_flush = flush_min <= P
                    _ik_push(IK, "item", item)
                    _ik_push(IK, "c_top")
                    if it > 0:
                        # every consumer is done with the previous item's tiles / tables
                        _bar_sync_ua(1, 128)
                    gc_row0 = g_cache.iterator.toint() + Int64(
                        cute.crd2idx((sidx, hh, 0, 0), g_cache.layout)
                    ) * Int64(4)
                    kc_row0 = k_cache.iterator.toint() + Int64(
                        cute.crd2idx((sidx, hh, 0, 0), k_cache.layout)
                    ) * Int64(2)
                    if cutlass.const_expr(KTC):
                        # channel layout: G_P and the window's raw k̂ / G of channel ct
                        # (rows j < P; the K~ rows past P keep stale finite values: they
                        # only meet zero D columns)
                        gp_c = Float32(
                            g_cache[sidx, hh, (base + P - 1) & RING_MASK, ct]
                        )
                        gp_c = _sel(P > 0, gp_c, 0.0, Float32)
                        gjc = []
                        krc = []
                        for j in cutlass.range_constexpr(16):
                            jp = Int32(j < P)
                            js = (base + j) & RING_MASK
                            gjc.append(
                                _ldg_f32_if(jp, gc_row0 + Int64((js * K_DIM + ct) * 4))
                            )
                            krc.append(
                                _ldg_u16_if(jp, kc_row0 + Int64((js * K_DIM + ct) * 2))
                            )
                    elif cutlass.const_expr(HELP):
                        # G_P of this channel (bdec / g append); K~ is the helpers'
                        gp_c = Float32(
                            g_cache[sidx, hh, (base + P - 1) & RING_MASK, ct]
                        )
                        gp_c = _sel(P > 0, gp_c, 0.0, Float32)
                    elif cutlass.const_expr(not _RING_PF):
                        gp_c, kraw_c, gj_c = _load_ring(
                            g_cache, k_cache, sidx, hh, P, base, ct, jr, c8
                        )
                    else:
                        gp_c = gp
                        kraw_c = kraw
                        gj_c = gj
                    kpred = Int32(jr < P)
                    gc_row = g_cache.iterator.toint() + Int64(
                        cute.crd2idx((sidx, hh, 0, 0), g_cache.layout)
                    ) * Int64(4)
                    kc_row = k_cache.iterator.toint() + Int64(
                        cute.crd2idx((sidx, hh, 0, 0), k_cache.layout)
                    ) * Int64(2)
                    _ik_pop(IK)
                    _ik_push(IK, "c_p1")
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
                            xc_addr + xr * XC_PITCH + l16 * 16,
                            xcw[0],
                            xcw[1],
                            xcw[2],
                            xcw[3],
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
                        U.sts_v4(
                            xc_addr + (r8 + 8 * p) * XC_PITCH + l16 * 16, z, z, z, z
                        )
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
                            U.sts_v4(
                                ktil_addr + (T + r8) * XC_PITCH + l16 * 16, z, z, z, z
                            )
                    _bar_sync_ua(1, 128)

                    _ik_pop(IK)
                    _ik_push(IK, "c_p2")
                    # -------- P2: gate + operand tiles (channel layout, c = ct)
                    c = ct
                    if cutlass.const_expr((not HELP and not KTC) or WR == 32):
                        s_gp[c] = gp_c
                    bdec = _sel(P > 0, _exp(gp_c), 1.0, Float32)
                    s_bdec[c] = bdec
                    gbase = _sel(is_flush, 0.0, gp_c, Float32)
                    eA = _exp(s_alog[hh])
                    # Independent work first so the T chains overlap: every token's
                    # sigmoid, the raw k̂ / q̂ of this channel, then the cumulative sum and
                    # its exponentials, then the stores (same arithmetic, same results)
                    sig = []
                    for t in cutlass.range_constexpr(T):
                        xg = graw[t] + dtb
                        sig.append(
                            Float32(1.0) / (Float32(1.0) + _exp(Float32(0.0) - eA * xg))
                        )
                    khs = []
                    qhs = []
                    for t in cutlass.range_constexpr(T):
                        khs.append(
                            U.u16_as_f32(
                                U.lds_u16(xc_addr + t * XC_PITCH + c * 2), True
                            )
                        )
                        qhs.append(
                            U.u16_as_f32(
                                U.lds_u16(xc_addr + (T + t) * XC_PITCH + c * 2), True
                            )
                        )
                    gstore = gc_row + Int64(c * 4)
                    cum = Float32(0.0)
                    for t in cutlass.range_constexpr(T):
                        cum = cum + lower_bound * sig[t]
                        _stg_f32(
                            gstore + Int64(((base + P + t) & RING_MASK) * (K_DIM * 4)),
                            gbase + cum,
                        )
                        ep = _exp(cum)
                        en = _exp(Float32(0.0) - cum)
                        eb = ep * bdec
                        kh = khs[t]
                        qh = qhs[t]
                        # X_c = [khat; qhat] (Gram / C A operand)
                        U.sts_u16(
                            xc_addr + t * XC_PITCH + c * 2, U.f32_to_u16(kh * ep, True)
                        )
                        U.sts_u16(
                            xc_addr + (T + t) * XC_PITCH + c * 2,
                            U.f32_to_u16(qh * ep, True),
                        )
                        # Ktil = k̂ e^{-cum} (<= e^{80}: bf16 holds it)
                        U.sts_u16(
                            ktil_addr + t * XC_PITCH + c * 2,
                            U.f32_to_u16(kh * en, True),
                        )
                        # X_r = X_c . bdec, rounded once from fp32 (readout B operand)
                        U.sts_u16(
                            x_addr + _x_sw128_off(t, c), U.f32_to_u16(kh * eb, True)
                        )
                        U.sts_u16(
                            x_addr + _x_sw128_off(T + t, c), U.f32_to_u16(qh * eb, True)
                        )
                    if ct < T:
                        s_coef[_CO_BETA + ct] = Float32(1.0) / (
                            Float32(1.0) + _exp(Float32(0.0) - braw)
                        )
                    if it == 0:
                        cute.arch.barrier(barrier_id=4, number_of_threads=160)  # inits
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(x_ready)
                    _bar_sync_ua(1, 128)  # s_gp, X_c, Ktil visible to every consumer
                    # the next item's row inputs (xw / graw / beta / dt_bias / A_log are
                    # consumed): in flight through the rest of this item. Its cursors
                    # (loaded at the top of this item) are materialised first.
                    if cutlass.const_expr(self.persistent):
                        nsidx = _mat_i32(nsidx)
                        nP = _mat_i32(nP)
                        nbase = _mat_i32(nbase)
                        xw, graw, braw, dtb = _load_x(
                            q, k, g, b, dt_bias, a_log, nb, nh, ct, r8, l16, T
                        )

                    _ik_pop(IK)
                    _ik_push(IK, "c_p3")
                    # -------- P3: K~ = bf16(k̂_j . e^{G_P - G_j}) (j < P, else 0)
                    if cutlass.const_expr(KTC):
                        for j in cutlass.range_constexpr(16):
                            if j < P:
                                wj = _exp(gp_c - gjc[j])
                                kv = U.u16_as_f32(krc[j], True) * wj
                                U.sts_u16(
                                    kt_addr + _x_sw128_off(j, ct),
                                    U.f32_to_u16(kv, True),
                                )
                    elif cutlass.const_expr(HELP):
                        _mbw(kt_ready, ph, 17, cta, warp, kk)
                    else:
                        for gk in cutlass.range_constexpr(2):
                            gpv = _lds_f32x4(
                                gp_addr + (gk * 64 + c8 * 8) * 4
                            ) + _lds_f32x4(gp_addr + (gk * 64 + c8 * 8 + 4) * 4)
                            kw = []
                            for i in cutlass.range_constexpr(4):
                                w0 = _exp(gpv[2 * i] - gj_c[gk * 8 + 2 * i])
                                w1 = _exp(gpv[2 * i + 1] - gj_c[gk * 8 + 2 * i + 1])
                                pk = U.pack_bf16x2(
                                    U.bf16x2_lo_f32(kraw_c[gk][i]) * w0,
                                    U.bf16x2_hi_f32(kraw_c[gk][i]) * w1,
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
                            _kt_block1(
                                kc_row, gc_row, gp_addr, kt_addr, base, P, jr, c8
                            )
                    _bar_sync_ua(1, 128)
                    if cutlass.const_expr(_RING_PF):
                        # the next item's G_P / K~ ring words (this item's are consumed)
                        gp, kraw, gj = _load_ring(
                            g_cache, k_cache, nsidx, nh, nP, nbase, ct, jr, c8
                        )

                    _ik_pop(IK)
                    _ik_push(IK, "c_p4")
                    # -------- P4: mma.sync Grams. warp cw 0: Gq = X_c . Ktil^T (fp32,
                    # s_gram); warps cw 1, 2: C n-tiles (j 0..7, 8..15) = X_c . K~^T (bf16,
                    # SW32 s_c)
                    grp = lane >> 2
                    c4 = lane & 3
                    a_ln = xc_addr + (lane % 16) * XC_PITCH + (lane // 16) * 16
                    if cw == 0:
                        acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                        b_ln = (
                            ktil_addr + (lane % 8) * XC_PITCH + ((lane // 8) % 2) * 16
                        )
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
                                kt_row
                                + (chunk // 8) * 2048
                                + (((chunk % 8) ^ (jl % 8)) * 16)
                            )
                            acc = U.mma_bf16_16816(
                                fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc
                            )
                        jc = 8 * nt + 2 * c4
                        _sts_u32(
                            c_addr + _c_sw32_off(grp, jc), U.pack_bf16x2(acc[0], acc[1])
                        )
                        _sts_u32(
                            c_addr + _c_sw32_off(grp + 8, jc),
                            U.pack_bf16x2(acc[2], acc[3]),
                        )
                        # C is read by the history UMMA (warp 1 issues it after d_ready)
                        cute.arch.fence_view_async_shared()
                    else:
                        if cutlass.const_expr(WR == 32):
                            if P > 16:
                                _c_ntile_block1(2, lane, a_ln, kt_addr, c_addr)
                        if cutlass.const_expr(
                            self.persistent and _L2_PREFETCH and _PF_WHERE == "consumer"
                        ):
                            # the idle Gram warp pulls the next item into L2 (no
                            # registers held, no scoreboard): its row inputs, ring window
                            # and state. (In warp 0 at 24 registers this hung under load.)
                            pn = Int32(nitem < NIT) * Int32(nsidx >= 0)
                            nsl = _sel(pn != 0, nsidx, 0, Int32)
                            with cute.arch.elect_one():
                                for t in cutlass.range_constexpr(T):
                                    for src in (q, k, g, v):
                                        _bulk_prefetch_l2_n(
                                            src.iterator.toint()
                                            + Int64(
                                                cute.crd2idx((nb, t, nh, 0), src.layout)
                                            )
                                            * Int64(2),
                                            Int32(K_DIM * 2),
                                            pn,
                                        )
                                _prefetch_rings_l2(
                                    k_cache, g_cache, u_cache, nsl, nh, nP * pn, nbase
                                )
                                _bulk_prefetch_l2_n(
                                    state.iterator.toint()
                                    + Int64(cute.crd2idx((nsl, nh, 0, 0), state.layout))
                                    * Int64(2),
                                    Int32(V_DIM * K_DIM * 2),
                                    pn,
                                )
                    _bar_sync_ua(1, 128)

                    _ik_pop(IK)
                    _ik_push(IK, "c_p56")
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
                        # waited on every item, P = 0 too (the producer completes the
                        # phase with no bytes): a skipped wait let the consumers arrive
                        # u_free for this item before warp 0 had waited the previous
                        # phase, and its parity wait then aliased two phases (hang)
                        _mbw(u_full, ph, 11, cta, warp, kk)
                        ub = s_uwin.iterator.toint() + row * 2
                        u_ptr = ub
                        for jj in cutlass.range_constexpr(16):
                            uv.append(U.lds_u16(ub + jj * (V_DIM * 2)))
                    else:
                        ucr = u_cache.iterator.toint() + Int64(
                            cute.crd2idx((sidx, hh, 0, row), u_cache.layout)
                        ) * Int64(2)
                        u_ptr = ucr
                        for jj in cutlass.range_constexpr(16):
                            uv.append(
                                _ldg_u16_if(
                                    Int32(jj < P),
                                    ucr
                                    + Int64(((base + jj) & RING_MASK) * (V_DIM * 2)),
                                )
                            )
                    words = []
                    for i in cutlass.range_constexpr(8):
                        lo = _sel(2 * i < P, uv[2 * i], Uint32(0), Uint32)
                        hi = _sel(2 * i + 1 < P, uv[2 * i + 1], Uint32(0), Uint32)
                        words.append(lo | (hi << Uint32(16)))
                    if it == 0:
                        cute.arch.barrier(
                            barrier_id=2, number_of_threads=160
                        )  # TMEM addr
                        U.fence_after_sync()
                    tbase = tmem_addr[0]
                    U.tmem_st_u32(ct, tbase + _COL_D, words)
                    if cutlass.const_expr(WR == 32):
                        if P > 16:
                            _d_block1(USE_SMEM_U, u_ptr, base, P, ct, tbase)
                    if cutlass.const_expr(USE_SMEM_U):
                        # the window has been read: warp 0 may copy the next head's
                        cute.arch.mbarrier_arrive(u_free)
                    U.tmem_wait_st()
                    U.fence_before_sync()
                    cute.arch.mbarrier_arrive(d_ready)
                    _bar_sync_ua(1, 128)  # coefficient tables
                    _ik_pop(IK)

                    if is_flush:
                        # fold RMW of K half 0 first: its UMMA was issued right after the
                        # history
                        _ik_push(IK, "c_rmw0")
                        _mbw(f_mma_a, fit & 1, 12, cta, warp, kk)
                        U.fence_after_sync()
                        _rmw(
                            0, ct, row, tbase, st_addr_c, bdec_addr, f_epi_a, f_gdone[0]
                        )
                        _ik_pop(IK)

                    # -------- epilogue: per V row forward substitution
                    _ik_push(IK, "c_wait_mma")
                    _mbw(mma_done, ph, 13, cta, warp, kk)
                    _ik_pop(IK)
                    _ik_push(IK, "c_epi")
                    U.fence_after_sync()
                    nld = (
                        1 << (T - 1).bit_length()
                    )  # tcgen05.ld widths are powers of two
                    hk = U.tmem_ld(ct, tbase + _COL_H, nld)
                    hq = U.tmem_ld(ct, tbase + _COL_H + T, nld)
                    U.tmem_wait_ld()
                    U.fence_before_sync()
                    if not is_flush:
                        # verify rows: release TMEM before the solve (warp 1 moves on early)
                        cute.arch.mbarrier_arrive(tmem_done)
                    uu: list = []
                    for t in cutlass.range_constexpr(T):
                        bt = s_coef[_CO_BETA + t]
                        acc_u = bt * (vv[t] - hk[t])
                        for s_ in cutlass.range_constexpr(t):
                            acc_u = acc_u - s_coef[_CO_A + t * 8 + s_] * uu[s_]
                        uu.append(acc_u)
                    for t in cutlass.range_constexpr(T):
                        y = hq[t]
                        for s_ in cutlass.range_constexpr(t + 1):
                            y = y + s_coef[_CO_BQ + t * 8 + s_] * uu[s_]
                        out[brow, t, hh, row] = BFloat16(y)
                        u_cache[sidx, hh, (base + P + t) & RING_MASK, row] = BFloat16(
                            uu[t]
                        )
                    _ik_pop(IK)
                    if is_flush:
                        # fold RMW of K half 1 (its UMMA ran behind the epilogue), then TMEM
                        # is free
                        _ik_push(IK, "c_rmw1")
                        _mbw(f_mma_b, fit & 1, 14, cta, warp, kk)
                        U.fence_after_sync()
                        _rmw(
                            1, ct, row, tbase, st_addr_c, bdec_addr, f_epi_a, f_gdone[1]
                        )
                        _ik_pop(IK)
                        U.fence_before_sync()
                        cute.arch.mbarrier_arrive(tmem_done)
                        fit = fit + 1
                    it = it + 1
                    if cutlass.const_expr(self.persistent):
                        vv = _load_v(v, nb, nh, row, T)
                    _ik_pop(IK)  # item
                else:
                    # skipped (padded) item: the next item's inputs here instead
                    nsidx = _mat_i32(nsidx)
                    nP = _mat_i32(nP)
                    nbase = _mat_i32(nbase)
                    xw, graw, braw, dtb = _load_x(
                        q, k, g, b, dt_bias, a_log, nb, nh, ct, r8, l16, T
                    )
                    if cutlass.const_expr(_RING_PF):
                        gp, kraw, gj = _load_ring(
                            g_cache, k_cache, nsidx, nh, nP, nbase, ct, jr, c8
                        )
                    vv = _load_v(v, nb, nh, row, T)
                sidx = nsidx
                P = nP
                base = nbase
            if cutlass.const_expr(_HANG_PROGRESS_ADDR != 0):
                _prog_st(
                    Int64(_HANG_PROGRESS_ADDR)
                    + (Int64(cta) * Int64(8) + Int64(warp)) * Int64(4),
                    Int32(17),
                )

    # ------------------------------------------------------------------ compile
    @staticmethod
    def compile(
        device: torch.device,
        T: int,
        h: int,
        w_ring: int,
        u_src: str,
        persistent: bool = False,
        split: bool = False,
    ):
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
            KdaReplayMtpUmma(T, h, w_ring, u_src, persistent, split),
            *args,
            Float32(1.0),
            Float32(-5.0),
            Int32(1),
            Int32(1),
            stream,
        )


_KERNELS: dict = {}


def _get_kernel(
    device: torch.device,
    T: int,
    h: int,
    w_ring: int,
    u_src: str,
    persistent: bool = False,
    split: bool = False,
):
    """The compiled kernel for this specialisation on ``device``'s architecture."""
    key = (
        gdn_device_target(device).compile_key,
        T,
        h,
        w_ring,
        u_src,
        persistent,
        split,
        _IKET,
        _L2_PREFETCH,
        _HANG_DEBUG_ADDR,
        _PF_WHERE,
        _HANG_PROGRESS_ADDR,
        _RING_PF,
        _HELP_P3,
        _KT_CH,
    )
    fn = _KERNELS.get(key)
    if fn is None:
        fn = KdaReplayMtpUmma.compile(device, T, h, w_ring, u_src, persistent, split)
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


def _grid(B: int, H: int, dev: torch.device):
    """(persistent, CTAs): one CTA per (request, head) item, or a persistent grid of
    _MIN_BLOCKS_PER_SM CTAs per SM when there are more items than that."""
    n = B * H
    slots = (
        _MIN_BLOCKS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count
    )
    pz = _FORCE_PERSISTENT if _FORCE_PERSISTENT is not None else True
    return bool(pz), (min(n, slots) if pz else n)


def launch_umma_persistent(
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
    B = q.shape[0]
    pz, ncta = _grid(B, H, q.device)
    split = _FORCE_SPLIT if _FORCE_SPLIT is not None else pz
    fn = _get_kernel(q.device, T, H, int(w_ring), _u_src(int(w_ring)), pz, split)
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
        int(ncta),
    )
    _LAUNCHES += 1


__all__ = ["launch_umma_persistent"]
