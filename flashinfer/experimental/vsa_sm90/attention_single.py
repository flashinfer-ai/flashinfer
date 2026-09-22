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
"""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16
from cutlass.cute.nvgpu import warpgroup, cpasync, OperandMajorMode
from cutlass.utils import SmemAllocator, LayoutEnum
import cutlass.utils.hopper_helpers as hopper
from cutlass.cute.runtime import from_dlpack
from .attention64 import softmax_update


@cute.kernel
def single_block(
    TQ: cute.Tensor,
    TK: cute.Tensor,
    TV: cute.Tensor,
    tq_atom: cute.CopyAtom,
    tk_atom: cute.CopyAtom,
    tv_atom: cute.CopyAtom,
    I: cute.Tensor,
    O: cute.Tensor,
    scale: Float32,
):
    linear, _, _ = cute.arch.block_idx()
    row = linear % (O.shape[1] // 64)
    head = linear // (O.shape[1] // 64)
    tid, _, _ = cute.arch.thread_idx()
    shared = SmemAllocator()
    lq = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (64, 64, 128), BFloat16, 1)
    lv = hopper.make_smem_layout_b(LayoutEnum.COL_MAJOR, (64, 128, 64), BFloat16, 1)
    sq = shared.allocate_tensor(
        BFloat16, lq.outer, byte_alignment=1024, swizzle=lq.inner
    )
    sk = shared.allocate_tensor(
        BFloat16, lq.outer, byte_alignment=1024, swizzle=lq.inner
    )
    sv = shared.allocate_tensor(
        BFloat16, lv.outer, byte_alignment=1024, swizzle=lv.inner
    )
    bars = shared.allocate_array(cutlass.Int64, 2)
    if tid < 2:
        cute.arch.mbarrier_init(bars + tid, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if tid < 32:
        if tid == 0:
            cpasync.prefetch_descriptor(tq_atom)
            cpasync.prefetch_descriptor(tk_atom)
            cpasync.prefetch_descriptor(tv_atom)
        qtiles = cute.local_tile(TQ[None, None, head], (64, 128), (None, 0))
        ktiles = cute.local_tile(TK[None, None, head], (64, 128), (None, 0))
        vtiles = cute.local_tile(TV[None, None, head], (128, 64), (0, None))
        qs, qg = cpasync.tma_partition(
            tq_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sq, 0, 2),
            cute.group_modes(qtiles, 0, 2),
        )
        ks, kg = cpasync.tma_partition(
            tk_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sk, 0, 2),
            cute.group_modes(ktiles, 0, 2),
        )
        vs, vg = cpasync.tma_partition(
            tv_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sv, 0, 2),
            cute.group_modes(vtiles, 0, 2),
        )
        idx = I[head, row, 0]
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bars, 32768)
            cute.arch.mbarrier_arrive_and_expect_tx(bars + 1, 16384)
        cute.copy(tq_atom, qg[None, row], qs[None, 0], tma_bar_ptr=bars)
        cute.copy(tk_atom, kg[None, idx], ks[None, 0], tma_bar_ptr=bars)
        cute.copy(tv_atom, vg[None, idx], vs[None, 0], tma_bar_ptr=bars + 1)
    qkmma = hopper.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        Float32,
        (1, 1, 1),
        (64, 64),
    )
    pvmma = hopper.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        Float32,
        (1, 1, 1),
        (64, 128),
        warpgroup.OperandSource.RMEM,
    )
    tq = qkmma.get_slice(tid)
    tp = pvmma.get_slice(tid)
    aq = qkmma.make_fragment_A(tq.partition_A(sq[None, None, 0]))
    bk = qkmma.make_fragment_B(tq.partition_B(sk[None, None, 0]))
    bv = pvmma.make_fragment_B(tp.partition_B(sv[None, None, 0]))
    scores = cute.make_rmem_tensor(tq.partition_shape_C((64, 64)), Float32)
    accum = cute.make_rmem_tensor(tp.partition_shape_C((64, 128)), Float32)
    s = cute.make_tensor(
        scores.iterator, cute.make_layout((2, (2, 8)), stride=(2, (1, 4)))
    )
    a = cute.make_tensor(
        accum.iterator, cute.make_layout((2, (2, 16)), stride=(2, (1, 4)))
    )
    probs = cute.make_rmem_tensor(cute.make_layout(((2, 2, 2), 1, 4)), BFloat16)
    pc = cute.make_tensor(probs.iterator, scores.layout)
    mx = cute.make_rmem_tensor((2,), Float32)
    den = cute.make_rmem_tensor((2,), Float32)
    mx.fill(-Float32.inf)
    den.fill(0.0)
    cute.arch.mbarrier_wait(bars, 0)
    warpgroup.fence()
    for kk in cutlass.range_constexpr(8):
        qkmma.set(warpgroup.Field.ACCUMULATE, kk != 0)
        cute.gemm(qkmma, scores, aq[None, None, kk], bk[None, None, kk], scores)
    warpgroup.commit_group()
    warpgroup.wait_group(0)
    softmax_update(s, mx, den, scale, True)
    pc.store(scores.load().to(BFloat16))
    accum.fill(0.0)
    cute.arch.mbarrier_wait(bars + 1, 0)
    warpgroup.fence()
    pvmma.set(warpgroup.Field.ACCUMULATE, True)
    cute.gemm(pvmma, accum, probs, bv, accum)
    warpgroup.commit_group()
    warpgroup.wait_group(0)
    for r in cutlass.range_constexpr(2):
        total = cute.arch.warp_reduction_sum(den[r], threads_in_group=4)
        inv = cute.arch.rcp_approx(total)
        for col in cutlass.range_constexpr(32):
            a[r, col] = a[r, col] * inv
    sout = sq[None, None, 0]
    dst = tp.partition_C(sout)
    result = cute.make_rmem_tensor(accum.shape, BFloat16)
    result.store(accum.load().to(BFloat16))
    cute.autovec_copy(result, dst)
    cute.arch.sync_threads()
    qbase = cute.assume(
        cutlass.Int64(head) * O.shape[1] * 128 + cutlass.Int64(row) * 8192, divby=8
    )
    otile = cute.make_tensor(
        O.iterator + qbase, cute.make_layout((64, 128), stride=(128, 1))
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
    )
    store_copy = cute.make_tiled_copy_tv(
        store_atom, cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 8))
    )
    thr_store = store_copy.get_slice(tid)
    src_vec = thr_store.partition_S(sout)
    dst_vec = thr_store.partition_D(otile)
    registers = cute.make_rmem_tensor_like(src_vec)
    cute.copy(store_copy, src_vec, registers)
    cute.copy(store_copy, registers, dst_vec)


@cute.jit
def launch(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    I: cute.Tensor,
    O: cute.Tensor,
    scale: Float32,
    stream: cuda.CUstream,
):
    qview = cute.make_tensor(
        Q.iterator,
        cute.make_layout(
            (Q.shape[1], 128, Q.shape[0]),
            stride=(128, 1, cute.assume(Q.shape[1] * 128, divby=128)),
        ),
    )
    kview = cute.make_tensor(
        K.iterator,
        cute.make_layout(
            (K.shape[1], 128, K.shape[0]),
            stride=(128, 1, cute.assume(K.shape[1] * 128, divby=128)),
        ),
    )
    vview = cute.make_tensor(
        V.iterator,
        cute.make_layout(
            (128, V.shape[1], V.shape[0]),
            stride=(1, 128, cute.assume(V.shape[1] * 128, divby=128)),
        ),
    )
    lq = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (64, 64, 128), BFloat16, 1)
    lv = hopper.make_smem_layout_b(LayoutEnum.COL_MAJOR, (64, 128, 64), BFloat16, 1)
    tq_atom, TQ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        qview,
        cute.slice_(lq, (None, None, 0)),
        (64, 128),
    )
    tk_atom, TK = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        kview,
        cute.slice_(lq, (None, None, 0)),
        (64, 128),
    )
    tv_atom, TV = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        vview,
        cute.slice_(lv, (None, None, 0)),
        (128, 64),
    )
    single_block(TQ, TK, TV, tq_atom, tk_atom, tv_atom, I, O, scale).launch(
        grid=((Q.shape[1] // 64) * Q.shape[0], 1, 1),
        block=(128, 1, 1),
        min_blocks_per_mp=4,
        stream=stream,
    )


_compiled = {}  # Code only; descriptors are owned by the wrapper plan.


def run_prepared(q, k, v, idx_gpu, cnt_gpu, order, sm_scale, out):
    sm_scale = float(sm_scale) * 1.4426950216293335
    caller_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(caller_stream.cuda_stream)
    tensors = (q, k, v, idx_gpu, out)
    ck = (str(q.device), str(q.dtype), str(idx_gpu.dtype))
    if ck not in _compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the VSA kernel before CUDA graph capture")
        views = [
            from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
                leading_dim=t.ndim - 1
            )
            for t in tensors
        ]
        _compiled[ck] = cute.compile(
            launch, *views, Float32(sm_scale), stream, options="--enable-tvm-ffi"
        )
    _compiled[ck](*tensors, float(sm_scale), stream)
