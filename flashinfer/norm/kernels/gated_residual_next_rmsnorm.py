# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused gated residual and the RMSNorm that consumes it."""

import functools
import struct

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import Float32, Int32, const_expr

from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

H = 8192
CLUSTER_N = 8
THREADS = 128
VEC = H // (CLUSTER_N * THREADS)
NWARPS = THREADS // 32
NSLOT = NWARPS * CLUSTER_N
INV_H = 1.0 / H

assert NSLOT <= 32
assert CLUSTER_N * THREADS * VEC == H


@cute.jit
def _store4_remote(ptr, mbar, peer, x0, x1, x2, x3):
    """One transaction-counted 16-byte store into a peer CTA's SMEM."""
    remote_ptr = cute.arch.map_dsmem_ptr(ptr, peer)
    remote_bar = cute.arch.map_dsmem_ptr(mbar, peer)
    asm = (
        "{\n\t.reg .v4 .f32 q;\n\t"
        "mov.f32 q.x, {$r2};\n\t"
        "mov.f32 q.y, {$r3};\n\t"
        "mov.f32 q.z, {$r4};\n\t"
        "mov.f32 q.w, {$r5};\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[{$r0}], q, [{$r1}];\n}"
    )
    cute.arch.inline_ptx(
        asm,
        read_only_args=[
            remote_ptr.ir_value(),
            remote_bar.ir_value(),
            x0,
            x1,
            x2,
            x3,
        ],
    )


@cute.jit
def _warp_sum4(v0, v1, v2, v3):
    """Interleave four independent butterfly trees to expose shuffle ILP."""
    w0, w1, w2, w3 = v0, v1, v2, v3
    for offset in (16, 8, 4, 2, 1):
        w0 = w0 + cute.arch.shuffle_sync_bfly(w0, offset=offset)
        w1 = w1 + cute.arch.shuffle_sync_bfly(w1, offset=offset)
        w2 = w2 + cute.arch.shuffle_sync_bfly(w2, offset=offset)
        w3 = w3 + cute.arch.shuffle_sync_bfly(w3, offset=offset)
    return w0, w1, w2, w3


@cute.jit
def _row_sum4(v0, v1, v2, v3, buf, mbar, tidx, lane, warp, rank):
    w0, w1, w2, w3 = _warp_sum4(v0, v1, v2, v3)
    if const_expr(CLUSTER_N > 1):
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, NSLOT * 16)
        slot = rank * NWARPS + warp
        if lane < CLUSTER_N:
            p = buf[(None, slot)].iterator
            _store4_remote(p, mbar, lane, w0, w1, w2, w3)
        cute.arch.mbarrier_wait(mbar, 0)
        f = cute.make_fragment_like(buf[(None, lane)])
        cute.autovec_copy(buf[(None, lane)], f)
        x = f.load()
        return _warp_sum4(x[0], x[1], x[2], x[3])
    else:
        if lane == 0:
            buf[(0, warp)] = w0
            buf[(1, warp)] = w1
            buf[(2, warp)] = w2
            buf[(3, warp)] = w3
        cute.arch.barrier()
        x0 = Float32(0.0)
        x1 = Float32(0.0)
        x2 = Float32(0.0)
        x3 = Float32(0.0)
        if lane < NWARPS:
            x0 = buf[(0, lane)]
            x1 = buf[(1, lane)]
            x2 = buf[(2, lane)]
            x3 = buf[(3, lane)]
        return _warp_sum4(x0, x1, x2, x3)


@cute.kernel
def _dual_norm_kernel(
    gBr: cute.Tensor,
    gSt: cute.Tensor,
    gAl: cute.Tensor,
    gBe: cute.Tensor,
    gWt: cute.Tensor,
    gRo: cute.Tensor,
    gFo: cute.Tensor,
    gPw: cute.Tensor,
    post_eps: cutlass.Constexpr,
    next_eps: cutlass.Constexpr,
    gain: cutlass.Constexpr,
    has_post_weight: cutlass.Constexpr,
    post_gain: cutlass.Constexpr,
    alpha_scalar: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cy, _ = cute.arch.block_idx()

    smem = cu_utils.SmemAllocator()
    if const_expr(CLUSTER_N > 1):
        mbar = smem.allocate_array(cutlass.Int64, num_elems=1)
    else:
        mbar = None
    buf = smem.allocate_tensor(
        Float32,
        cute.make_layout((4, NSLOT), stride=(1, 4)),
        byte_alignment=16,
    )

    col = cy * THREADS + tidx
    rc = ((None, None), (bidx, col))
    wc = ((None, None), (0, col))

    tBr, tSt, tRo, tFo = gBr[rc], gSt[rc], gRo[rc], gFo[rc]
    tBe, tWt = gBe[wc], gWt[wc]

    fBr = cute.make_fragment_like(tBr)
    fSt = cute.make_fragment_like(tSt)
    fBe = cute.make_fragment_like(tBe)
    fWt = cute.make_fragment_like(tWt)
    fRo = cute.make_fragment_like(tRo)
    fFo = cute.make_fragment_like(tFo)

    cute.autovec_copy(tBr, fBr)
    cute.autovec_copy(tSt, fSt)
    if const_expr(alpha_scalar is None):
        tAl = gAl[wc]
        fAl = cute.make_fragment_like(tAl)
        cute.autovec_copy(tAl, fAl)
    cute.autovec_copy(tBe, fBe)
    cute.autovec_copy(tWt, fWt)
    if const_expr(has_post_weight):
        tPw = gPw[wc]
        fPw = cute.make_fragment_like(tPw)
        cute.autovec_copy(tPw, fPw)

    if const_expr(CLUSTER_N > 1):
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    if const_expr(CLUSTER_N > 1):
        rank = cute.arch.block_idx_in_cluster()
    else:
        rank = Int32(0)

    b = fBr.load().to(Float32)
    if const_expr(has_post_weight):
        b = b * (fPw.load().to(Float32) + post_gain)
    if const_expr(alpha_scalar is None):
        u = fAl.load().to(Float32) * fSt.load().to(Float32)
    elif const_expr(alpha_scalar == 1.0):
        u = fSt.load().to(Float32)
    else:
        u = Float32(alpha_scalar) * fSt.load().to(Float32)
    v = fBe.load().to(Float32) * b

    pA = (b * b).reduce(cute.ReductionOp.ADD, 0.0, 0)
    pB = (u * u).reduce(cute.ReductionOp.ADD, 0.0, 0)
    pC = (u * v).reduce(cute.ReductionOp.ADD, 0.0, 0)
    pD = (v * v).reduce(cute.ReductionOp.ADD, 0.0, 0)

    if const_expr(CLUSTER_N > 1):
        cute.arch.cluster_wait()

    sA, sB, sC, sD = _row_sum4(pA, pB, pC, pD, buf, mbar, tidx, lane, warp, rank)

    r1 = cute.math.rsqrt(sA * INV_H + post_eps, fastmath=True)
    res = u + v * r1
    fRo.store(res)
    cute.autovec_copy(fRo, tRo)

    s2 = cute.arch.fmax(sB + 2.0 * r1 * sC + (r1 * r1) * sD, Float32(0.0))
    r2 = cute.math.rsqrt(s2 * INV_H + next_eps, fastmath=True)

    w = fWt.load().to(Float32)
    out = (res * r2) * (w + gain)
    fFo.store(out.to(gFo.element_type))
    cute.autovec_copy(fFo, tFo)


@cute.jit
def _launch(
    mBr: cute.Tensor,
    mSt: cute.Tensor,
    mAl: cute.Tensor,
    mBe: cute.Tensor,
    mWt: cute.Tensor,
    mRo: cute.Tensor,
    mFo: cute.Tensor,
    mPw: cute.Tensor,
    post_eps: cutlass.Constexpr,
    next_eps: cutlass.Constexpr,
    gain: cutlass.Constexpr,
    has_post_weight: cutlass.Constexpr,
    post_gain: cutlass.Constexpr,
    alpha_scalar: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    tiler = (1, VEC)
    gBr = cute.zipped_divide(mBr, tiler)
    gSt = cute.zipped_divide(mSt, tiler)
    gAl = cute.zipped_divide(mAl, tiler)
    gBe = cute.zipped_divide(mBe, tiler)
    gWt = cute.zipped_divide(mWt, tiler)
    gRo = cute.zipped_divide(mRo, tiler)
    gFo = cute.zipped_divide(mFo, tiler)
    gPw = cute.zipped_divide(mPw, tiler)

    _dual_norm_kernel(
        gBr,
        gSt,
        gAl,
        gBe,
        gWt,
        gRo,
        gFo,
        gPw,
        post_eps,
        next_eps,
        gain,
        has_post_weight,
        post_gain,
        alpha_scalar,
    ).launch(
        grid=[mBr.shape[0], CLUSTER_N, 1],
        block=[THREADS, 1, 1],
        cluster=[1, CLUSTER_N, 1] if const_expr(CLUSTER_N > 1) else None,
        smem=1024,
        stream=stream,
    )


def _float_key(value: float) -> str:
    return struct.pack("!d", value).hex()


@functools.cache
def _get_compiled(
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    has_post_weight: bool,
    post_gain_center: float,
    alpha_scalar: float | None = None,
):
    sym_m = cute.sym_int()
    branch = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, H),
        stride_order=(1, 0),
        assumed_align=32,
    )
    residual = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_m, H),
        stride_order=(1, 0),
        assumed_align=32,
    )
    gate = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1, H),
        stride_order=(1, 0),
        assumed_align=32,
    )
    next_weight = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, H),
        stride_order=(1, 0),
        assumed_align=32,
    )
    ffn_input = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, H),
        stride_order=(1, 0),
        assumed_align=32,
    )
    cuda_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_name = "_".join(
        (
            f"peps{_float_key(post_eps)}",
            f"neps{_float_key(next_eps)}",
            f"gc{_float_key(next_gain_center)}",
            f"pw{int(has_post_weight)}",
            f"pgc{_float_key(post_gain_center)}",
            "alpha_tensor"
            if alpha_scalar is None
            else f"alpha{_float_key(alpha_scalar)}",
        )
    )
    return build_and_load_cute_dsl_kernel(
        "gated_residual_next_rmsnorm",
        kernel_name,
        lambda: cute.compile(
            _launch,
            branch,
            residual,
            gate,
            gate,
            next_weight,
            residual,
            ffn_input,
            next_weight,
            post_eps,
            next_eps,
            next_gain_center,
            has_post_weight,
            post_gain_center,
            alpha_scalar,
            cuda_stream,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def run(
    branch: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    post_weight: torch.Tensor | None = None,
    post_gain_center: float = 0.0,
    alpha_scalar: float | None = None,
) -> None:
    if branch.shape[0] == 0:
        return
    fn = _get_compiled(
        float(post_eps),
        float(next_eps),
        float(next_gain_center),
        post_weight is not None,
        float(post_gain_center) if post_weight is not None else 0.0,
        alpha_scalar,
    )
    fn(
        branch,
        residual,
        alpha.unsqueeze(0),
        beta.unsqueeze(0),
        next_weight.unsqueeze(0),
        residual_out,
        ffn_input,
        (post_weight if post_weight is not None else next_weight).unsqueeze(0),
    )


def gated_residual_next_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    post_weight: torch.Tensor | None = None,
    post_gain_center: float = 0.0,
    alpha_scalar: float | None = None,
) -> None:
    """Run the fixed-width fused gated-residual plus next-RMSNorm kernel."""
    run(
        branch,
        residual,
        alpha,
        beta,
        next_weight,
        post_eps,
        next_eps,
        next_gain_center,
        residual_out,
        ffn_input,
        post_weight,
        post_gain_center,
        alpha_scalar,
    )
