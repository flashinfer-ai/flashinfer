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
    gRouter: cute.Tensor,
    post_eps: cutlass.Constexpr,
    next_eps: cutlass.Constexpr,
    gain: cutlass.Constexpr,
    has_post_weight: cutlass.Constexpr,
    post_gain: cutlass.Constexpr,
    alpha_value: Float32,
    alpha_mode: cutlass.Constexpr,
    enable_pdl: cutlass.Constexpr[bool],
    has_router: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cy, _ = cute.arch.block_idx()

    smem = cu_utils.SmemAllocator()
    if const_expr(CLUSTER_N > 1):
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
    else:
        mbar = None
    buf = smem.allocate_tensor(
        Float32,
        cute.make_layout((4, NSLOT * 2), stride=(1, 4)),
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

    if enable_pdl:
        cute.arch.griddepcontrol_wait()

    cute.autovec_copy(tBr, fBr)
    cute.autovec_copy(tSt, fSt)
    if const_expr(alpha_mode == 0):
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
            cute.arch.mbarrier_init(mbar + 1, 1)
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
    if const_expr(alpha_mode == 0):
        u = fAl.load().to(Float32) * fSt.load().to(Float32)
    elif const_expr(alpha_mode == 1):
        u = fSt.load().to(Float32)
    else:
        u = alpha_value * fSt.load().to(Float32)
    v = fBe.load().to(Float32) * b

    pA = (b * b).reduce(cute.ReductionOp.ADD, 0.0, 0)

    if const_expr(CLUSTER_N > 1):
        cute.arch.cluster_wait()

    first = cute.make_tensor(buf.iterator, cute.make_layout((4, NSLOT), stride=(1, 4)))
    second = cute.make_tensor(
        buf.iterator + 4 * NSLOT, cute.make_layout((4, NSLOT), stride=(1, 4))
    )
    zero = Float32(0.0)
    sA, _, _, _ = _row_sum4(pA, zero, zero, zero, first, mbar, tidx, lane, warp, rank)

    r1 = cute.math.rsqrt(sA * INV_H + post_eps, fastmath=True)
    res = u + v * r1
    fRo.store(res)
    cute.autovec_copy(fRo, tRo)

    # Reduce the rounded FP32 residual itself: expanding its square loses
    # the small result when the residual and normalized branch cancel.
    p2 = (res * res).reduce(cute.ReductionOp.ADD, 0.0, 0)
    s2, _, _, _ = _row_sum4(
        p2, zero, zero, zero, second, mbar + 1, tidx, lane, warp, rank
    )
    # No CTA may exit while a peer still reads its distributed shared memory.
    if const_expr(CLUSTER_N > 1):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
    r2 = cute.math.rsqrt(s2 * INV_H + next_eps, fastmath=True)

    w = fWt.load().to(Float32)
    out = (res * r2) * (w + gain)
    fFo.store(out.to(gFo.element_type))
    cute.autovec_copy(fFo, tFo)
    if const_expr(has_router):
        tRouter = gRouter[rc]
        fRouter = cute.make_fragment_like(tRouter)
        fRouter.store(out)
        cute.autovec_copy(fRouter, tRouter)

    if enable_pdl:
        cute.arch.griddepcontrol_launch_dependents()


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
    mRouter: cute.Tensor,
    post_eps: cutlass.Constexpr,
    next_eps: cutlass.Constexpr,
    gain: cutlass.Constexpr,
    has_post_weight: cutlass.Constexpr,
    post_gain: cutlass.Constexpr,
    alpha_value: Float32,
    alpha_mode: cutlass.Constexpr,
    enable_pdl: cutlass.Constexpr[bool],
    has_router: cutlass.Constexpr,
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
    gRouter = cute.zipped_divide(mRouter, tiler)

    _dual_norm_kernel(
        gBr,
        gSt,
        gAl,
        gBe,
        gWt,
        gRo,
        gFo,
        gPw,
        gRouter,
        post_eps,
        next_eps,
        gain,
        has_post_weight,
        post_gain,
        alpha_value,
        alpha_mode,
        enable_pdl,
        has_router,
    ).launch(
        grid=[mBr.shape[0], CLUSTER_N, 1],
        block=[THREADS, 1, 1],
        cluster=[1, CLUSTER_N, 1] if const_expr(CLUSTER_N > 1) else None,
        stream=stream,
        use_pdl=enable_pdl,
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
    alpha_mode: int = 0,
    enable_pdl: bool = False,
    has_router: bool = False,
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
            f"alpha_mode{alpha_mode}",
            f"pdl{int(enable_pdl)}",
            f"router{int(has_router)}",
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
            residual,
            post_eps,
            next_eps,
            next_gain_center,
            has_post_weight,
            post_gain_center,
            Float32(1.0),
            alpha_mode,
            enable_pdl,
            has_router,
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
    enable_pdl: bool = False,
    router_input: torch.Tensor | None = None,
) -> None:
    """Optionally write the next norm's unrounded FP32 result to router_input."""
    if router_input is not None and (
        router_input.dtype != torch.float32
        or router_input.shape != branch.shape
        or router_input.device != branch.device
        or not router_input.is_contiguous()
        or router_input.data_ptr() % 32
    ):
        raise ValueError("router_input must be aligned contiguous FP32 like branch")
    if branch.shape[0] == 0:
        return
    fn = _get_compiled(
        float(post_eps),
        float(next_eps),
        float(next_gain_center),
        post_weight is not None,
        float(post_gain_center) if post_weight is not None else 0.0,
        0 if alpha_scalar is None else 1 if alpha_scalar == 1.0 else 2,
        enable_pdl,
        router_input is not None,
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
        router_input if router_input is not None else residual_out,
        1.0 if alpha_scalar is None else float(alpha_scalar),
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
    enable_pdl: bool = False,
    alpha_scalar: float | None = None,
    router_input: torch.Tensor | None = None,
) -> None:
    """Run dual norm with an optional unrounded FP32 router destination."""
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
        enable_pdl,
        router_input=router_input,
    )
