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


@cute.jit
def softmax_update(
    s: cute.Tensor,
    mx: cute.Tensor,
    den: cute.Tensor,
    logscale: Float32,
    first: cutlass.Constexpr,
    valid_cols: cutlass.Int32,
):
    factors = cute.make_rmem_tensor((2,), Float32)
    for r in cutlass.range_constexpr(2):
        prev = mx[r]
        maximum = prev
        for col in cutlass.range_constexpr(24):
            if first and (
                (col // 2) * 8 + (cute.arch.lane_idx() % 4) * 2 + col % 2 >= valid_cols
            ):
                s[r, col] = -Float32.inf
            maximum = cute.arch.fmax(maximum, s[r, col], nan=True, ftz=False)
        maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=4)
        alpha = cute.math.exp2((prev - maximum) * logscale, fastmath=True)
        if cutlass.const_expr(first):
            alpha = Float32(0.0)
        factors[r] = alpha
        mx[r] = maximum
        scaled_max = maximum * logscale
        for col in cutlass.range_constexpr(24):
            prob = cute.math.exp2(
                cute.math.fma(s[r, col], logscale, -scaled_max), fastmath=True
            )
            s[r, col] = prob
        den[r] = den[r] * alpha
        for col in cutlass.range_constexpr(24):
            den[r] = den[r] + s[r, col]
    return factors


@cute.kernel
def sparse_attn_tma(
    TQ: cute.Tensor,
    TK: cute.Tensor,
    TV: cute.Tensor,
    tq_atom: cute.CopyAtom,
    tk_atom: cute.CopyAtom,
    tv_atom: cute.CopyAtom,
    I: cute.Tensor,
    C: cute.Tensor,
    Ord: cute.Tensor,
    O: cute.Tensor,
    scale: Float32,
    PERM: cutlass.Constexpr,
):
    linear, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(PERM):
        linear = Ord[linear]
    row = linear % (O.shape[1] // 64)
    head = linear // (O.shape[1] // 64)
    tid, _, _ = cute.arch.thread_idx()
    shared = SmemAllocator()
    lq = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (64, 64, 128), BFloat16, 1)
    lk = hopper.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (64, 96, 128), BFloat16, 2)
    lv = hopper.make_smem_layout_b(LayoutEnum.COL_MAJOR, (64, 128, 96), BFloat16, 2)
    sq0 = shared.allocate_tensor(
        BFloat16, lq.outer, byte_alignment=1024, swizzle=lq.inner
    )
    sk0 = shared.allocate_tensor(
        BFloat16, lk.outer, byte_alignment=1024, swizzle=lk.inner
    )
    sv0 = shared.allocate_tensor(
        BFloat16, lv.outer, byte_alignment=1024, swizzle=lv.inner
    )
    barriers = shared.allocate_array(cutlass.Int64, 9)
    if tid < 2:
        cute.arch.mbarrier_init(barriers + tid, 1)
        cute.arch.mbarrier_init(barriers + 2 + tid, 1)
        cute.arch.mbarrier_init(barriers + 4 + tid, 4)
        cute.arch.mbarrier_init(barriers + 6 + tid, 4)
    if tid == 2:
        cute.arch.mbarrier_init(barriers + 8, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    block_count = C[head, row]
    halves = 2 * block_count
    count = (halves + 2) // 3
    first_cols = (halves - (count - 1) * 3) * 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if tid >= 128:
        cute.arch.setmaxregister_decrease(32)
        qtiles = cute.local_tile(TQ[None, None, head], (64, 128), (None, 0))
        ktiles = cute.local_tile(TK[None, None, head], (32, 128), (None, 0))
        vtiles = cute.local_tile(TV[None, None, head], (128, 32), (0, None))
        qs, qg = cpasync.tma_partition(
            tq_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sq0, 0, 2),
            cute.group_modes(qtiles, 0, 2),
        )
        skparts = cute.flat_divide(sk0, (32, 128))
        svparts = cute.flat_divide(sv0, (128, 32))
        ks, kg = cpasync.tma_partition(
            tk_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(skparts, 0, 2),
            cute.group_modes(ktiles, 0, 2),
        )
        vs, vg = cpasync.tma_partition(
            tv_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(svparts, 0, 2),
            cute.group_modes(vtiles, 0, 2),
        )
        if warp_idx == 4 or warp_idx == 5:
            if warp_idx == 4:
                cpasync.prefetch_descriptor(tq_atom)
                cpasync.prefetch_descriptor(tk_atom)
                cpasync.prefetch_descriptor(tv_atom)
            lane = cute.arch.lane_idx()
            idx_lo = cutlass.Int32(0)
            idx_hi = cutlass.Int32(0)
            if lane < I.shape[2]:
                idx_lo = I[head, row, lane]
            if lane + 32 < I.shape[2]:
                idx_hi = I[head, row, lane + 32]
            if warp_idx == 4:
                if tid == 128:
                    cute.arch.mbarrier_arrive_and_expect_tx(barriers + 8, 16384)
                cute.copy(tq_atom, qg[None, row], qs[None, 0], tma_bar_ptr=barriers + 8)
            for slot in cutlass.range(count):
                stage = slot % 2
                phase = (slot // 2) % 2
                if warp_idx == 4:
                    cute.arch.mbarrier_wait(barriers + 4 + stage, phase ^ 1)
                else:
                    cute.arch.mbarrier_wait(barriers + 6 + stage, phase ^ 1)
                tile = count - 1 - slot
                if warp_idx == 4:
                    if tid == 128:
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, 24576)
                else:
                    if tid == 160:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barriers + 2 + stage, 24576
                        )
                for piece in cutlass.range_constexpr(3):
                    half = tile * 3 + piece
                    idx = TK.shape[0] // 32
                    if half < halves:
                        src_i = half // 2
                        src = idx_lo
                        if src_i >= 32:
                            src = idx_hi
                        block_idx = cute.arch.shuffle_sync_op(src, src_i % 32)
                        if src_i >= 64:
                            block_idx = I[head, row, src_i]
                        idx = block_idx * 2 + half % 2
                    if warp_idx == 4:
                        cute.copy(
                            tk_atom,
                            kg[None, idx],
                            ks[None, piece, 0, stage],
                            tma_bar_ptr=barriers + stage,
                        )
                    else:
                        cute.copy(
                            tv_atom,
                            vg[None, idx],
                            vs[None, 0, piece, stage],
                            tma_bar_ptr=barriers + 2 + stage,
                        )

    else:
        cute.arch.setmaxregister_increase(224)
        qkmma = hopper.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (64, 96),
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
        aq = qkmma.make_fragment_A(tq.partition_A(sq0[None, None, 0]))
        bk = qkmma.make_fragment_B(tq.partition_B(sk0))
        bv = pvmma.make_fragment_B(tp.partition_B(sv0))
        scores = cute.make_rmem_tensor(tq.partition_shape_C((64, 96)), Float32)
        accum = cute.make_rmem_tensor(tp.partition_shape_C((64, 128)), Float32)
        accum.fill(0.0)
        s = cute.make_tensor(
            scores.iterator, cute.make_layout((2, (2, 12)), stride=(2, (1, 4)))
        )
        a = cute.make_tensor(
            accum.iterator, cute.make_layout((2, (2, 16)), stride=(2, (1, 4)))
        )
        probs = cute.make_rmem_tensor(cute.make_layout(((2, 2, 2), 1, 6)), BFloat16)
        pc = cute.make_tensor(probs.iterator, scores.layout)
        mx = cute.make_rmem_tensor((2,), Float32)
        den = cute.make_rmem_tensor((2,), Float32)
        mx.fill(-Float32.inf)
        den.fill(0.0)
        logscale = scale
        qbase = cute.assume(
            cutlass.Int64(head) * O.shape[1] * 128 + cutlass.Int64(row) * 8192, divby=8
        )
        otile = cute.make_tensor(
            O.iterator + qbase, cute.make_layout((64, 128), stride=(128, 1))
        )
        cute.arch.mbarrier_wait(barriers + 8, 0)
        cute.arch.mbarrier_wait(barriers, 0)
        warpgroup.fence()
        for kk in cutlass.range_constexpr(8):
            qkmma.set(warpgroup.Field.ACCUMULATE, kk != 0)
            cute.gemm(qkmma, scores, aq[None, None, kk], bk[None, None, kk, 0], scores)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        if cute.arch.lane_idx() == 0:
            cute.arch.mbarrier_arrive(barriers + 4)
        alpha = softmax_update(s, mx, den, logscale, True, first_cols)
        pc.store(scores.load().to(BFloat16))

        for slot in cutlass.range(1, count):
            read_stage = slot % 2
            previous_stage = (slot - 1) % 2
            phase = (slot // 2) % 2
            cute.arch.mbarrier_wait(barriers + read_stage, phase)
            warpgroup.fence()
            for kk in cutlass.range_constexpr(8):
                qkmma.set(warpgroup.Field.ACCUMULATE, kk != 0)
                cute.gemm(
                    qkmma,
                    scores,
                    aq[None, None, kk],
                    bk[None, None, kk, read_stage],
                    scores,
                )
            warpgroup.commit_group()
            cute.arch.mbarrier_wait(
                barriers + 2 + previous_stage, ((slot - 1) // 2) % 2
            )
            pvmma.set(warpgroup.Field.ACCUMULATE, True)
            cute.gemm(pvmma, accum, probs, bv[None, None, None, previous_stage], accum)
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            if cute.arch.lane_idx() == 0:
                cute.arch.mbarrier_arrive(barriers + 4 + read_stage)
            alpha = softmax_update(s, mx, den, logscale, False, cutlass.Int32(96))
            warpgroup.wait_group(0)
            if cute.arch.lane_idx() == 0:
                cute.arch.mbarrier_arrive(barriers + 6 + previous_stage)
            for r in cutlass.range_constexpr(2):
                for col in cutlass.range_constexpr(32):
                    a[r, col] = a[r, col] * alpha[r]
            pc.store(scores.load().to(BFloat16))

        cute.arch.mbarrier_wait(barriers + 2 + (count - 1) % 2, ((count - 1) // 2) % 2)
        warpgroup.fence()
        pvmma.set(warpgroup.Field.ACCUMULATE, True)
        cute.gemm(pvmma, accum, probs, bv[None, None, None, (count - 1) % 2], accum)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        if cute.arch.lane_idx() == 0:
            cute.arch.mbarrier_arrive(barriers + 6 + (count - 1) % 2)
        for r in cutlass.range_constexpr(2):
            total = cute.arch.warp_reduction_sum(den[r], threads_in_group=4)
            inv = cute.arch.rcp_approx(total)
            for col in cutlass.range_constexpr(32):
                a[r, col] = a[r, col] * inv
        # Q is dead after the final QK. Reuse its shared tile to turn
        # the MMA fragment's four-byte stores into contiguous 16-byte stores.
        sout = sq0[None, None, 0]
        dst = tp.partition_C(sout)
        result = cute.make_rmem_tensor(accum.shape, BFloat16)
        result.store(accum.load().to(BFloat16))
        cute.autovec_copy(result, dst)
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        store_copy = cute.make_tiled_copy_tv(
            store_atom,
            cute.make_layout((8, 16), stride=(16, 1)),
            cute.make_layout((1, 8)),
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
    C: cute.Tensor,
    Ord: cute.Tensor,
    O: cute.Tensor,
    scale: Float32,
    stream: cuda.CUstream,
    PERM: cutlass.Constexpr,
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
    lk = hopper.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (64, 96, 128), BFloat16, 2)
    lv = hopper.make_smem_layout_b(LayoutEnum.COL_MAJOR, (64, 128, 96), BFloat16, 2)
    tq_atom, TQ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        qview,
        cute.slice_(lq, (None, None, 0)),
        (64, 128),
    )
    tk_atom, TK = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        kview,
        cute.slice_(cute.flat_divide(lk, (32, 128)), (None, None, 0, 0, 0)),
        (32, 128),
    )
    tv_atom, TV = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        vview,
        cute.slice_(cute.flat_divide(lv, (128, 32)), (None, None, 0, 0, 0)),
        (128, 32),
    )
    sparse_attn_tma(
        TQ, TK, TV, tq_atom, tk_atom, tv_atom, I, C, Ord, O, scale, PERM
    ).launch(
        grid=((Q.shape[1] // 64) * Q.shape[0], 1, 1),
        block=(256, 1, 1),
        min_blocks_per_mp=2,
        stream=stream,
    )


_compiled = {}  # Code only; descriptors are owned by the wrapper plan.


def run_prepared(q, k, v, idx_gpu, cnt_gpu, order, sm_scale, out):
    sm_scale = float(sm_scale) * 1.4426950216293335
    caller_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(caller_stream.cuda_stream)
    use_order = order.numel() > 1
    tensors = (q, k, v, idx_gpu, cnt_gpu, order, out)
    ck = (
        use_order,
        str(q.device),
        str(q.dtype),
        str(idx_gpu.dtype),
        str(cnt_gpu.dtype),
    )
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
            launch,
            *views,
            Float32(sm_scale),
            stream,
            use_order,
            options="--enable-tvm-ffi",
        )
    _compiled[ck](*tensors, float(sm_scale), stream)
