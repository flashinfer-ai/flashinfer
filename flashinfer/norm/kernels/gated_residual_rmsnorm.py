# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Fused gated residual RMSNorm (CuTe DSL, SM100/SM103).

    x      = f32(branch)                        (MODE == 0)
    x      = f32(branch) * f32(norm_weight)     (MODE == 1, weight before the norm)
    rms    = rsqrt(mean_h(x*x) + eps)
    normed = x * rms                            <- current upstream FP32 semantics
    out    = alpha * stream + beta * normed     <- FP32 residual stream

One GPU kernel body, one launch per non-empty invocation, on the caller's stream.
grid = (M, S): a CTA owns row `row = blockIdx.x` and column segment
`seg = blockIdx.y`.  (VEC, S) are the shape-dependent tuning parameters.

* L1 residency is steered by hand.  `norm_weight`, `alpha` and `beta` are the
  three operands whose lines are re-read by *every* CTA of the launch, so they
  are loaded `evict_last`; `branch`, `stream` and the output plane are touched
  once per CTA and never again, so they are `evict_first` and stop displacing
  the shared vectors.  With 148 % S == 0 the two co-resident CTAs of an SM
  carry the same segment, so the second one finds alpha/beta -- and in MODE 1
  the whole 16 KiB weight row, which sits on the reduction's critical path --
  already in L1.

* The reduction reads the whole row through a *fixed* map
  `col = k*(RV*T) + t*RV + v`, RV = 8 BF16 lanes, i.e. one 128-bit load per chunk
  and 512 contiguous bytes per warp instruction.  RV / T / RK never change with
  M, MODE or S, so a row's sum of squares is bit-identical at any batch size or
  row permutation.  Every CTA of a row
  recomputes that same sum, so there is no cross-CTA traffic, no atomics and no
  second launch; the redundant re-reads merge in L1/L2 and leave HBM traffic
  unchanged.

* The epilogue map is *segment-major*: `col = seg*(H/S) + c*(VEC*T) + t*VEC + v`,
  so a CTA owns the single contiguous column block [seg*H/S, (seg+1)*H/S) and
  walks it with a thread-contiguous FP32 vector of width VEC.  H/S is always a
  whole number of RV-wide reduction blocks, so that column block is exactly
  `H/(RV*S)` of the reduction map's per-thread pieces, all inside chunk
  `k = seg/(S/2)`.  The threads holding them push their already-weighted FP32 x
  into shared memory before the barrier the block reduction already needs, and
  every thread reads its own VEC slice back after it.  The epilogue therefore
  re-reads neither `branch` nor -- in MODE 1 -- `norm_weight`, and does not
  repeat the FP32 weighting, at no cost in barriers, launches or access width.

  Nesting CJ *inside* the segment rather than outside it is what makes that
  hand-off line up when a CTA needs more than one vector per thread, which is
  what puts the four-way rung (M = 17..64) on the staged path at VEC = 4.  Taking
  the block straight out of the reduction registers instead is possible there --
  the two maps coincide -- but it forces VEC = RV, striding the residual planes
  by 32 B, and measured 3.648 against 3.600 on m64-attn.  One CTA per row keeps
  the direct epilogue: the residual planes, not the duplicate read, set its pace.

* The launch carries `cluster = (1, CX, 1)` with CX = S for S <= 4 and CX = 2
  for the eight-way split.  Adjacent segments of a row are exactly the CTAs that
  re-read the same 16 KiB row (32 KiB in MODE 1) for the sum, so co-scheduling
  them inside one GPC keeps those duplicate lines on the near L2 slice instead
  of spraying them across the whole device.  The cluster is a *placement hint*
  only -- no cross-CTA data is exchanged, so the reduction tree, the round point
  and the launch count are unchanged.  Keeping it narrow matters: an eight-CTA
  cluster cannot start until one GPC has eight free SMs, and that launch
  serialisation costs more on the decode shapes than the extra locality buys
  (m1-attn measures 2.127 us at CX = 8 against 1.920 us at CX = 2).  A
  partitioned cluster reduction that exchanges the eight partials through
  `mapa`/`st.shared::cluster` removes the duplicate reads outright, but the
  ordered cluster barrier it needs costs ~0.7 us of launch skew -- far more than
  the traffic it saves -- so the reduction stays replicated.

Everything that does not depend on the row sum -- every global load, the FP32
weighting and the squares -- is issued before the barrier, so the latency-bound
decode shapes cost one memory round trip plus the reduction tail.  MODE 1 rows
17..32 use a low-register aligned shared broadcast fold.  The exact
`mean = sum * 2^-13` and accurate rsqrt are preserved, and the normalized value
stays FP32 through the residual blend.
"""

import functools
import struct

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import SmemAllocator

from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

EP = cute.nvgpu.CacheEvictionPriority
FIRST = EP.EVICT_FIRST
LAST = EP.EVICT_LAST


def _btree(vals):
    """Balanced binary fold of a compile-time list (adjacent pairs first)."""
    while len(vals) > 1:
        vals = [vals[2 * i] + vals[2 * i + 1] for i in range(len(vals) // 2)]
    return vals[0]


def _bfly_fold(vals):
    """Fold a compile-time list in exactly the order `shfl.bfly` would.

    `cute.arch.warp_reduction` walks offsets n/2, n/4, ... 1, so lane 0 ends up
    with (((v0+v[n/2]) + (v[n/4]+v[3n/4])) + ...).  Reproducing that association
    in registers keeps the row sum bit-identical to the butterfly it replaces.
    """
    cur = list(vals)
    n = len(cur)
    while n > 1:
        n //= 2
        cur = [cur[i] + cur[i + n] for i in range(n)]
    return cur[0]


H = 8192  # hidden size, fixed by the definition
RV = 8  # reduction columns per thread per chunk -> 128-bit BF16 loads
T = 512  # threads per CTA; fixed for all M so the reduction tree is invariant
RK = H // (RV * T)  # reduction chunks per thread
NW = T // 32  # warps per CTA
F32 = cutlass.Float32


@cute.kernel
def _gated_residual_rmsnorm_kernel(
    mBranch: cute.Tensor,
    mStream: cute.Tensor,
    mWeight: cute.Tensor,
    mAlpha: cute.Tensor,
    mBeta: cute.Tensor,
    mOut: cute.Tensor,
    VEC: cutlass.Constexpr,
    S: cutlass.Constexpr,
    CJ: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    LOWREG: cutlass.Constexpr,
    GAIN_CENTER: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    ENABLE_PDL: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, seg, _ = cute.arch.block_idx()

    M = cute.size(mBranch, mode=[0])
    # reduction map (segment independent) and segment-major epilogue map
    rrl = cute.make_layout((RV, T, RK, M), stride=(1, RV, RV * T, H))
    rvl = cute.make_layout((RV, T, RK), stride=(1, RV, RV * T))
    el = cute.make_layout((VEC, T, CJ, S, M), stride=(1, VEC, VEC * T, VEC * T * CJ, H))
    evl = cute.make_layout((VEC, T, CJ, S), stride=(1, VEC, VEC * T, VEC * T * CJ))
    STAGED: cutlass.Constexpr = S >= 2

    if cutlass.const_expr(ENABLE_PDL):
        cute.arch.griddepcontrol_wait()

    # ---- issue every global load first: one memory round trip for the kernel ----
    gBa = cute.make_tensor(mBranch.iterator, rrl)[(None, tidx, None, row)]
    fBa = cute.make_fragment_like(gBa)
    cute.autovec_copy(gBa, fBa, l1c_evict_priority=FIRST)

    if cutlass.const_expr(MODE == 1):
        gWa = cute.make_tensor(mWeight.iterator, rvl)[(None, tidx, None)]
        fWa = cute.make_fragment_like(gWa)
        cute.autovec_copy(gWa, fWa, l1c_evict_priority=LAST)

    if cutlass.const_expr(not STAGED):
        gB = cute.make_tensor(mBranch.iterator, el)[(None, tidx, None, seg, row)]
        fB = cute.make_fragment_like(gB)
        cute.autovec_copy(gB, fB, l1c_evict_priority=FIRST)
        if cutlass.const_expr(MODE == 1):
            gW = cute.make_tensor(mWeight.iterator, evl)[(None, tidx, None, seg)]
            fW = cute.make_fragment_like(gW)
            cute.autovec_copy(gW, fW, l1c_evict_priority=LAST)

    gS = cute.make_tensor(mStream.iterator, el)[(None, tidx, None, seg, row)]
    fS = cute.make_fragment_like(gS)
    cute.autovec_copy(gS, fS, l1c_evict_priority=FIRST)

    gA = cute.make_tensor(mAlpha.iterator, evl)[(None, tidx, None, seg)]
    fA = cute.make_fragment_like(gA)
    cute.autovec_copy(gA, fA, l1c_evict_priority=LAST)

    gBe = cute.make_tensor(mBeta.iterator, evl)[(None, tidx, None, seg)]
    fBe = cute.make_fragment_like(gBe)
    cute.autovec_copy(gBe, fBe, l1c_evict_priority=LAST)

    xa = fBa.load().to(F32)
    if cutlass.const_expr(MODE == 1):
        wa = fWa.load().to(F32)
        if cutlass.const_expr(GAIN_CENTER != 0.0):
            wa = wa + F32(GAIN_CENTER)
        xa = xa * wa  # weight before the norm
    if cutlass.const_expr(STAGED):
        fXa = cute.make_fragment_like(fBa, F32)
        fXa.store(xa)
    else:
        x = fB.load().to(F32)
        if cutlass.const_expr(MODE == 1):
            w = fW.load().to(F32)
            if cutlass.const_expr(GAIN_CENTER != 0.0):
                w = w + F32(GAIN_CENTER)
            x = x * w

    alloc = SmemAllocator()
    if cutlass.const_expr(STAGED):
        # The reduction fragment already holds the exact FP32 x this segment
        # needs.  Its owner threads hand it over through the same barrier the
        # block reduction uses, so no second branch/weight read is issued.
        stage = alloc.allocate_tensor(F32, cute.make_layout(H // S), byte_alignment=16)
        OT: cutlass.Constexpr = H // (RV * S)  # owner threads per segment
        sw = cute.make_tensor(
            stage.iterator, cute.make_layout((RV, OT), stride=(1, RV))
        )
        HALF: cutlass.Constexpr = S // 2  # segments carved out of one chunk
        if seg < HALF:
            owner = seg * OT
            if tidx >= owner and tidx < owner + OT:
                cute.autovec_copy(fXa[(None, 0)], sw[(None, tidx - owner)])
        else:
            owner = (seg - HALF) * OT
            if tidx >= owner and tidx < owner + OT:
                cute.autovec_copy(fXa[(None, 1)], sw[(None, tidx - owner)])

    # ---- FP32 row reduction, fixed association tree ----
    # The squares stay a packed f32x2 vector op; folding them as an explicit
    # depth-4 tree instead of a linear accumulate turns 17 dependent adds on
    # the critical path into 4.
    fSq = cute.make_fragment_like(fBa, F32)
    fSq.store(xa * xa)
    part = _btree([fSq[(e, k)] for k in range(RK) for e in range(RV)])

    warp_sum = cute.arch.warp_reduction_sum(part)
    red = alloc.allocate_tensor(F32, cute.make_layout(NW), byte_alignment=16)
    lane = cute.arch.lane_idx()
    # blockDim is 1-D, so the warp index is a shift; cute.arch.warp_idx() would
    # linearise tid.y/tid.z and broadcast it through an extra shuffle.
    warp = tidx // 32
    if lane == 0:
        red[warp] = warp_sum
    # `alpha * stream` is the half of the residual that does not depend on the
    # row sum, so it is folded here, on the far side of the barrier's shadow.
    # That takes one multiply off the post-barrier critical path and retires the
    # alpha and stream fragments into a single one, which is VEC*CJ registers
    # less live state across the barrier (16 on the whole-row S = 1 rung).
    pre = fA.load() * fS.load()
    cute.arch.sync_threads()
    if cutlass.const_expr(VEC * CJ <= 2):
        # One broadcast 128-bit read of all NW warp partials, folded in register
        # in the butterfly's own association: four independent LDS and a depth-4
        # tree instead of five dependent shuffles on the critical path.  It costs
        # NW live registers, so it is only taken when the epilogue fragments are
        # narrow enough (VEC*CJ <= 2) to leave room under a two-block budget.
        fRed = cute.make_fragment_like(red)
        cute.autovec_copy(red, fRed)
        total = _bfly_fold([fRed[i] for i in range(NW)])
    elif cutlass.const_expr(LOWREG):
        # Four aligned FP32x4 broadcasts avoid the wide path's four shuffles,
        # while folding each opposite pair before issuing the next pair caps
        # live state below the full 16-register broadcast path.
        r0 = cute.make_tensor(red.iterator, cute.make_layout(4))
        r8 = cute.make_tensor(red.iterator + 8, cute.make_layout(4))
        f0 = cute.make_fragment_like(r0)
        f8 = cute.make_fragment_like(r8)
        cute.autovec_copy(r0, f0)
        cute.autovec_copy(r8, f8)
        a0 = f0[0] + f8[0]
        a1 = f0[1] + f8[1]
        a2 = f0[2] + f8[2]
        a3 = f0[3] + f8[3]

        r4 = cute.make_tensor(red.iterator + 4, cute.make_layout(4))
        r12 = cute.make_tensor(red.iterator + 12, cute.make_layout(4))
        f4 = cute.make_fragment_like(r4)
        f12 = cute.make_fragment_like(r12)
        cute.autovec_copy(r4, f4)
        cute.autovec_copy(r12, f12)
        b0 = a0 + (f4[0] + f12[0])
        b1 = a1 + (f4[1] + f12[1])
        b2 = a2 + (f4[2] + f12[2])
        b3 = a3 + (f4[3] + f12[3])
        total = (b0 + b2) + (b1 + b3)
    else:
        # Duplicate the 16 initialized warp partials into both half-warps.  A
        # 16-lane butterfly preserves the fixed association while removing the
        # leading shuffle/add against zero and halving the scratch allocation.
        v = red[lane & (NW - 1)]
        total = cute.arch.warp_reduction_sum(v, threads_in_group=NW)

    mean = total * F32(1.0 / H) + F32(EPS)
    inv = cute.math.rsqrt(mean)

    # ---- epilogue ----
    if cutlass.const_expr(STAGED):
        se = cute.make_tensor(
            stage.iterator, cute.make_layout((VEC, T, CJ), stride=(1, VEC, VEC * T))
        )
        fX = cute.make_fragment_like(se[(None, tidx, None)])
        cute.autovec_copy(se[(None, tidx, None)], fX)
        x = fX.load()
    normed = x * inv
    y = pre + fBe.load() * normed

    gO = cute.make_tensor(mOut.iterator, el)[(None, tidx, None, seg, row)]
    fO = cute.make_fragment_like(gO)
    fO.store(y)
    cute.autovec_copy(fO, gO, l1c_evict_priority=FIRST)

    if cutlass.const_expr(ENABLE_PDL):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _gated_residual_rmsnorm_launch(
    mBranch: cute.Tensor,
    mStream: cute.Tensor,
    mWeight: cute.Tensor,
    mAlpha: cute.Tensor,
    mBeta: cute.Tensor,
    mOut: cute.Tensor,
    cuda_stream: _cuda.CUstream,
    VEC: cutlass.Constexpr,
    S: cutlass.Constexpr,
    CJ: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    LOWREG: cutlass.Constexpr,
    GAIN_CENTER: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    SMEM: cutlass.Constexpr,
    MBPM: cutlass.Constexpr,
    CX: cutlass.Constexpr,
    ENABLE_PDL: cutlass.Constexpr,
):
    M = cute.size(mBranch, mode=[0])
    if cutlass.const_expr(MBPM > 0):
        _gated_residual_rmsnorm_kernel(
            mBranch,
            mStream,
            mWeight,
            mAlpha,
            mBeta,
            mOut,
            VEC,
            S,
            CJ,
            MODE,
            LOWREG,
            GAIN_CENTER,
            EPS,
            ENABLE_PDL,
        ).launch(
            grid=[M, S, 1],
            block=[T, 1, 1],
            smem=SMEM,
            stream=cuda_stream,
            cluster=[1, CX, 1],
            use_pdl=ENABLE_PDL,
            min_blocks_per_mp=MBPM,
        )
    else:
        _gated_residual_rmsnorm_kernel(
            mBranch,
            mStream,
            mWeight,
            mAlpha,
            mBeta,
            mOut,
            VEC,
            S,
            CJ,
            MODE,
            LOWREG,
            GAIN_CENTER,
            EPS,
            ENABLE_PDL,
        ).launch(
            grid=[M, S, 1],
            block=[T, 1, 1],
            smem=SMEM,
            stream=cuda_stream,
            cluster=[1, CX, 1],
            use_pdl=ENABLE_PDL,
        )


def _config(m, mode):
    """(epilogue FP32 vector width, column split S): enough CTAs to cover the SMs.

    Every CTA re-reads the whole row for the sum, so a segment costs a fixed
    16 KiB (32 KiB in MODE 1) however narrow it is; splitting past roughly one
    CTA per SM only multiplies that fixed cost.  Below that a narrower residual
    vector is what buys the extra segments.  Past 64 rows the rows fill the
    machine on their own and one CTA owns a whole row.

    The knee sits at 16 rows: an eight-way split is what covers the machine up
    to there, and beyond it the four-way split's 16 B residual vectors and
    halved duplicate-read bill win in both modes (m32-attn measures 2.833 us at
    (2, 8) against 2.752 us at (4, 4)).  Splitting only two ways at 64 rows
    halves the threads that carry the residual planes and loses more than the
    traffic it saves (3.552 us against 3.232 us).
    """
    if m <= 16:
        return (2, 8)
    if m <= 64:
        return (4, 4)
    return (4, 1)


def _float_key(value: float) -> str:
    return struct.pack("!d", value).hex()


@functools.cache
def _get_compiled(
    vec,
    s,
    mode,
    lowreg,
    gain_center,
    eps,
    mbpm,
    cx,
    enable_pdl,
):
    sym_m = cute.sym_int()
    branch = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, H), stride_order=(1, 0), assumed_align=16
    )
    residual = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (sym_m, H), stride_order=(1, 0), assumed_align=16
    )
    weight = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (H,), assumed_align=16
    )
    gate = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (H,), assumed_align=16
    )
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (sym_m, H), stride_order=(1, 0), assumed_align=16
    )
    cuda_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_name = "_".join(
        (
            f"v{vec}",
            f"s{s}",
            f"m{mode}",
            f"lr{int(lowreg)}",
            f"gc{_float_key(gain_center)}",
            f"eps{_float_key(eps)}",
            f"mb{mbpm}",
            f"cx{cx}",
            f"pdl{int(enable_pdl)}",
        )
    )
    return build_and_load_cute_dsl_kernel(
        "gated_residual_rmsnorm",
        kernel_name,
        lambda: cute.compile(
            _gated_residual_rmsnorm_launch,
            branch,
            residual,
            weight,
            gate,
            gate,
            output,
            cuda_stream,
            vec,
            s,
            H // (vec * T * s),
            mode,
            lowreg,
            gain_center,
            eps,
            128 + (4 * H // s if s >= 2 else 0),
            mbpm,
            cx,
            enable_pdl,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def run(
    branch,
    stream,
    norm_weight,
    alpha,
    beta,
    eps,
    gain_center,
    mode,
    output,
    enable_pdl=False,
):
    if branch.shape[0] == 0:  # empty invocation: nothing to launch
        return
    mode_i = int(mode)
    m = branch.shape[0]
    vec, s = _config(m, mode_i)
    # The aligned low-register fold wins on the weighted quarter-wave rung.
    lowreg = mode_i == 1 and m > 16 and m <= 32
    # Two co-resident blocks per SM only pay off once the grid is wave-rich; on
    # a sub-one-wave grid the target buys no warps and only halves the
    # per-thread register budget.  A whole-row CTA (S = 1) also carries CJ
    # times the epilogue fragments, so it only tolerates the halved budget once
    # the grid is many waves deep.
    mbpm = 2 if m * s > 148 else 0
    # Co-locate CTAs of a row in *small* clusters.  A cluster is a placement
    # hint only, but it is the CTAs of one row that re-read the same 16 KiB
    # (32 KiB in MODE 1) for the sum, so pinning a few of them to one GPC keeps
    # those duplicate lines on the near L2 slice.  Binding the whole eight-way
    # split costs more than it saves -- the launch cannot start the row until a
    # single GPC has eight free SMs -- so wide splits are clustered pairwise.
    cx = s if s <= 4 else 2
    fn = _get_compiled(
        vec,
        s,
        mode_i,
        lowreg,
        float(gain_center),
        float(eps),
        mbpm,
        cx,
        bool(enable_pdl),
    )
    # Launch on the caller's current stream so the kernel is graph capturable.
    fn(branch, stream, norm_weight, alpha, beta, output)


def gated_residual_rmsnorm(
    branch: torch.Tensor,
    stream: torch.Tensor,
    norm_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    gain_center: float,
    mode: int,
    output: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run the fixed-width fused gated residual RMSNorm kernel."""
    if output is None:
        output = torch.empty_like(stream)
    run(
        branch,
        stream,
        norm_weight,
        alpha,
        beta,
        eps,
        gain_center,
        mode,
        output,
        enable_pdl,
    )
    return output
