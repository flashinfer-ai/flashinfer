# SPDX-License-Identifier: Apache-2.0
# Algorithm translated from FlashQLA5538c35,MIT,Qwen team2026.
import torch
import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import KeyedCompileMixin
from .primitives import (
    _acc,
    _coords,
    _gemm,
    _store,
    _sync,
    _transpose,
)
from .preprocess import launch, write_tile, load_tile, async_load32


class Warmup(KeyedCompileMixin):
    def __init__(self, heads, parts, index_dtype, threshold, g_dtype):
        self.heads, self.parts, self.index_dtype, self.threshold, self.g_dtype = (
            heads,
            parts,
            index_dtype,
            threshold,
            g_dtype,
        )
        self.manual_cache_key("heads", "parts", "index_dtype", "threshold", "g_dtype")

    @cute.jit
    def __call__(self, g, segments, counts, fallback, stream):
        self.kernel(g, segments, counts, fallback).launch(
            grid=(self.parts, 1, 1),
            block=(((self.heads + 31) // 32) * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, g, segments, counts, fallback):
        part, _, _ = cute.arch.block_idx()
        head, _, _ = cute.arch.thread_idx()
        if head < self.heads:
            start = cutlass.Int32(segments[part * 4])
            end = cutlass.Int32(segments[part * 4 + 1])
            last = cutlass.Int32(segments[part * 4 + 3])
            n = (end - start) // 32
            count = n
            fb = True
            if last:
                count = cutlass.Int32(0)
                fb = False
            else:
                total = cutlass.Float32(0.0)
                for i in range(n):
                    total += cutlass.Float32(g[(end - i * 32 - 1) * self.heads + head])
                    if total < self.threshold and count == n:
                        count = i + 1
                        fb = False
            counts[part * self.heads + head] = count
            fallback[part * self.heads + head] = fb


class PrepareH(KeyedCompileMixin):
    def __init__(self, dtype, hk, heads, parts, index_dtype, g_dtype, b_dtype):
        (
            self.dtype,
            self.hk,
            self.heads,
            self.parts,
            self.index_dtype,
            self.g_dtype,
            self.b_dtype,
        ) = dtype, hk, heads, parts, index_dtype, g_dtype, b_dtype
        self.manual_cache_key(
            "dtype", "hk", "heads", "parts", "index_dtype", "g_dtype", "b_dtype"
        )

    @cute.jit
    def __call__(self, k, v, a, g, b, segments, counts, ht, mt, stream):
        dtype = self.dtype

        @cute.struct
        class Storage:
            k: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 136], 128]
            v: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 128], 128]
            a: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 40], 128]
            g: cute.struct.MemRange[cutlass.Float32, 2 * 32]
            b: cute.struct.MemRange[cutlass.Float32, 2 * 32]
            h: cute.struct.Align[cute.struct.MemRange[dtype, 128 * 136], 128]
            x: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 136], 128]
            y: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 136], 128]
            z0: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 72], 128]
            z1: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 72], 128]
            bars: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 8], 8]

        self.storage = Storage
        # Match FlashQLA's launch_bounds(512, 1). On the validated SM120
        # toolchain, omitting this prevents CTA register-pool allocation and
        # spills the recurrent state despite the setmaxregister calls below.
        self.kernel(k, v, a, g, b, segments, counts, ht, mt).launch(
            grid=(self.parts, self.heads, 1),
            block=(512, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, k, v, a, g, b, segments, counts, ht, mt):
        part, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 128
        begin = cutlass.Int32(segments[part * 4])
        end = cutlass.Int32(segments[part * 4 + 1])
        n = cutlass.Int32(counts[part * self.heads + head])
        start = max(begin, end - n * 32)
        calc = n >= (end - begin + 31) // 32
        kh = head * self.hk // self.heads
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        sk = sm.k.get_tensor(cute.make_layout((32, 128, 2), stride=(136, 1, 32 * 136)))
        sv = sm.v.get_tensor(cute.make_layout((32, 128, 2), stride=(128, 1, 32 * 128)))
        sa = sm.a.get_tensor(cute.make_layout((32, 32, 2), stride=(40, 1, 32 * 40)))
        sg = sm.g.get_tensor(cute.make_layout((32, 2), stride=(1, 32)))
        sb = sm.b.get_tensor(cute.make_layout((32, 2), stride=(1, 32)))
        sh = sm.h.get_tensor(cute.make_layout((128, 128), stride=(136, 1)))
        sx = sm.x.get_tensor(cute.make_layout((32, 128), stride=(136, 1)))
        sy = sm.y.get_tensor(cute.make_layout((32, 128), stride=(136, 1)))
        sz0 = sm.z0.get_tensor(cute.make_layout((32, 64), stride=(72, 1)))
        sz1 = sm.z1.get_tensor(cute.make_layout((32, 64), stride=(72, 1)))
        bars = sm.bars.data_ptr()
        if tid == 0:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(bars + i, 128)
                cute.arch.mbarrier_init(bars + 2 + i, 384)
        if tid == 0:
            cute.arch.mbarrier_init(bars + 4, 384)
            cute.arch.mbarrier_init(bars + 5, 256)
            cute.arch.mbarrier_init(bars + 6, 384)
            cute.arch.mbarrier_init(bars + 7, 128)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tid >= 384:
            cute.arch.setmaxregister_decrease(24)
            for chunk in range(n):
                buf = chunk % 2
                if chunk >= 2:
                    cute.arch.mbarrier_wait(bars + 2 + buf, (chunk // 2 - 1) % 2)
                token = start + chunk * 32
                async_load32(
                    k,
                    sk[None, None, buf],
                    (token * self.hk + kh) * 128,
                    self.hk * 128,
                    128,
                    lane,
                )
                async_load32(
                    v,
                    sv[None, None, buf],
                    (token * self.heads + head) * 128,
                    self.heads * 128,
                    128,
                    lane,
                )
                async_load32(
                    a,
                    sa[None, None, buf],
                    (token * self.heads + head) * 32,
                    self.heads * 32,
                    32,
                    lane,
                )
                cute.arch.cp_async_commit_group()
                if lane < 32:
                    sg[lane, buf] = cutlass.Float32(
                        g[(token + lane) * self.heads + head]
                    )
                    sb[lane, buf] = cutlass.Float32(
                        b[(token + lane) * self.heads + head]
                    )
                cute.arch.cp_async_wait_group(0)
                cute.arch.mbarrier_arrive(bars + buf)
        elif tid < 128:
            cute.arch.setmaxregister_increase(168)
            state = _acc(self.dtype, 128, 128)
            for chunk in range(n):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                _store(state, sh, 128, 128)
                cute.arch.mbarrier_arrive(bars + 5)
                cute.arch.mbarrier_wait(bars + 5, chunk % 2)
                decay = cute.math.exp2(
                    sg[31, buf] * cutlass.Float32(1.442695), fastmath=True
                )
                for i in cutlass.range_constexpr(cute.size(state)):
                    state[i] *= decay
                cute.arch.mbarrier_arrive(bars + 6)
                cute.arch.mbarrier_wait(bars + 6, chunk % 2)
                _gemm(_transpose(sy), _transpose(sx), state, 128, 128, 32, True)
                cute.arch.mbarrier_arrive(bars + 7)
                cute.arch.mbarrier_arrive(bars + 2 + buf)
            write_tile(
                state, ht, (part * self.heads + head) * 16384, 128, 128, 128, self.dtype
            )
        elif tid < 256:
            cute.arch.setmaxregister_increase(160)
            matrix = _acc(self.dtype, 128, 64)
            mxy = _coords(self.dtype, 128, 64)
            xy = _coords(self.dtype, 32, 128)
            sumg = cutlass.Float32(0.0)
            if calc:
                for i in cutlass.range_constexpr(cute.size(matrix)):
                    row, col = mxy[i]
                    if row == col + 64:
                        matrix[i] = cutlass.Float32(1.0)
            for chunk in range(n):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, chunk // 2 % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                values = _acc(self.dtype, 32, 128)
                _gemm(
                    _transpose(sa[None, None, buf]),
                    _transpose(sk[None, None, buf]),
                    values,
                    32,
                    128,
                    32,
                    True,
                )
                for i in cutlass.range_constexpr(cute.size(values)):
                    row, col = xy[i]
                    values[i] *= -sb[row, buf]
                _store(values, sx, 32, 128)
                cute.arch.mbarrier_arrive(bars + 6)
                if calc:
                    cute.arch.mbarrier_wait(bars + 6, chunk % 2)
                if calc:
                    _store(matrix, cute.local_tile(sh, (128, 64), (0, 1)), 128, 64)
                if calc:
                    _sync(barrier_id=5, number_of_threads=128)
                    cute.arch.mbarrier_wait(bars + 7, chunk % 2)
                if calc:
                    z = _acc(self.dtype, 32, 64)
                    _gemm(
                        sk[None, None, buf],
                        _transpose(cute.local_tile(sh, (128, 64), (0, 1))),
                        z,
                        32,
                        64,
                        128,
                        True,
                    )
                    _store(z, sz1, 32, 64)
                    _sync(barrier_id=5, number_of_threads=128)
                    _gemm(_transpose(sx), _transpose(sz1), matrix, 128, 64, 32, True)
                    sumg += sg[31, buf]
                cute.arch.mbarrier_arrive(bars + 2 + buf)
            decay = cute.math.exp2(sumg * cutlass.Float32(1.442695), fastmath=True)
            for i in cutlass.range_constexpr(cute.size(matrix)):
                if calc:
                    matrix[i] *= decay
                else:
                    matrix[i] = cutlass.Float32(0.0)
            write_tile(
                matrix,
                mt,
                (part * self.heads + head) * 16384 + 64,
                128,
                64,
                128,
                self.dtype,
            )
        else:
            cute.arch.setmaxregister_increase(160)
            matrix = _acc(self.dtype, 128, 64)
            mxy = _coords(self.dtype, 128, 64)
            xy = _coords(self.dtype, 32, 128)
            sumg = cutlass.Float32(0.0)
            if calc:
                for i in cutlass.range_constexpr(cute.size(matrix)):
                    row, col = mxy[i]
                    if row == col:
                        matrix[i] = cutlass.Float32(1.0)
            for chunk in range(n):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, chunk // 2 % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                values = _acc(self.dtype, 32, 128)
                cute.arch.mbarrier_arrive(bars + 5)
                cute.arch.mbarrier_wait(bars + 5, chunk % 2)
                _gemm(sk[None, None, buf], sh, values, 32, 128, 128, True)
                decay = cute.math.exp2(
                    sg[31, buf] * cutlass.Float32(1.442695), fastmath=True
                )
                for i in cutlass.range_constexpr(cute.size(values)):
                    row, col = xy[i]
                    values[i] = values[i] * decay - cutlass.Float32(
                        sv[row, col, buf]
                    ) * cute.math.exp2(
                        (sg[31, buf] - sg[row, buf]) * cutlass.Float32(1.442695),
                        fastmath=True,
                    )
                _store(values, sy, 32, 128)
                cute.arch.mbarrier_arrive(bars + 6)
                if calc:
                    cute.arch.mbarrier_wait(bars + 6, chunk % 2)
                if calc:
                    _store(matrix, cute.local_tile(sh, (128, 64), (0, 0)), 128, 64)
                if calc:
                    _sync(barrier_id=6, number_of_threads=128)
                    cute.arch.mbarrier_wait(bars + 7, chunk % 2)
                if calc:
                    z = _acc(self.dtype, 32, 64)
                    _gemm(
                        sk[None, None, buf],
                        _transpose(cute.local_tile(sh, (128, 64), (0, 0))),
                        z,
                        32,
                        64,
                        128,
                        True,
                    )
                    _store(z, sz0, 32, 64)
                    _sync(barrier_id=6, number_of_threads=128)
                    _gemm(_transpose(sx), _transpose(sz0), matrix, 128, 64, 32, True)
                    sumg += sg[31, buf]
                cute.arch.mbarrier_arrive(bars + 2 + buf)
            decay = cute.math.exp2(sumg * cutlass.Float32(1.442695), fastmath=True)
            for i in cutlass.range_constexpr(cute.size(matrix)):
                if calc:
                    matrix[i] *= decay
                else:
                    matrix[i] = cutlass.Float32(0.0)
            write_tile(
                matrix, mt, (part * self.heads + head) * 16384, 128, 64, 128, self.dtype
            )


class CorrectH(KeyedCompileMixin):
    def __init__(
        self,
        dtype,
        heads,
        sequences,
        index_dtype,
        initial_dtype,
        has_initial,
        initial_v_first,
    ):
        self.dtype, self.heads, self.sequences, self.index_dtype = (
            dtype,
            heads,
            sequences,
            index_dtype,
        )
        self.initial_dtype, self.has_initial, self.initial_v_first = (
            initial_dtype,
            has_initial,
            initial_v_first,
        )
        self.manual_cache_key(
            "dtype",
            "heads",
            "sequences",
            "index_dtype",
            "initial_dtype",
            "has_initial",
            "initial_v_first",
        )

    @cute.jit
    def __call__(self, ht, mt, fb, ranges, initial, result, stream):
        dtype = self.dtype

        @cute.struct
        class Storage:
            h: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 136], 128]
            m: cute.struct.Align[cute.struct.MemRange[dtype, 128 * 136], 128]

        self.storage = Storage
        self.kernel(ht, mt, fb, ranges, initial, result).launch(
            grid=(4, self.heads, self.sequences), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, ht, mt, fb, ranges, initial, result):
        row = cutlass.Int32(0)
        col = cutlass.Int32(0)
        offset = cutlass.Int32(0)
        vb, head, raw = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        start = cutlass.Int32(ranges[raw * 2])
        end = cutlass.Int32(ranges[raw * 2 + 1])
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        sh = sm.h.get_tensor(cute.make_layout((32, 128), stride=(136, 1)))
        mm = sm.m.get_tensor(cute.make_layout((128, 128), stride=(136, 1)))
        state = _acc(self.dtype, 32, 128)
        xy = _coords(self.dtype, 32, 128)
        if cutlass.const_expr(self.has_initial):
            for i in cutlass.range_constexpr(cute.size(state)):
                row, col = xy[i]
                offset = (raw * self.heads + head) * 16384
                if cutlass.const_expr(self.initial_v_first):
                    offset += (vb * 32 + row) * 128 + col
                else:
                    offset += col * 128 + vb * 32 + row
                state[i] = cutlass.Float32(initial[offset])
        for part in range(start, end):
            write_tile(
                state,
                result,
                ((part * self.heads + head) * 128 + vb * 32) * 128,
                32,
                128,
                128,
                self.dtype,
            )
            if part + 1 < end:
                _store(state, sh, 32, 128)
                load_tile(
                    mt, mm, (part * self.heads + head) * 16384, 128, 128, 128, tid
                )
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(cute.size(state)):
                    row, col = xy[i]
                    state[i] = cutlass.Float32(
                        ht[
                            ((part * self.heads + head) * 128 + vb * 32 + row) * 128
                            + col
                        ]
                    )
                if fb[part * self.heads + head]:
                    _gemm(sh, mm, state, 32, 128, 128, True)
                cute.arch.sync_threads()


def boundary_states(
    k, v, a, g, b, segments, ranges, initial, *, state_v_first=True, threshold=-10.0
):
    parts = segments.shape[0]
    heads = v.shape[1]
    dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[k.dtype]
    counts = torch.empty((parts, heads), device=k.device, dtype=segments.dtype)
    fallback = torch.empty((parts, heads), device=k.device, dtype=torch.bool)
    ht = torch.empty((parts, heads, 128, 128), device=k.device, dtype=k.dtype)
    mt = torch.empty_like(ht)
    launch(
        Warmup(heads, parts, segments.dtype, threshold, g.dtype),
        (g, segments, counts, fallback),
        k.device,
    )
    launch(
        PrepareH(dtype, k.shape[1], heads, parts, segments.dtype, g.dtype, b.dtype),
        (k, v, a, g, b, segments, counts, ht, mt),
        k.device,
    )
    has_initial = initial is not None
    if initial is None:
        initial = torch.empty((1,), device=k.device, dtype=torch.float32)
    result = torch.empty((parts, heads, 128, 128), device=k.device, dtype=initial.dtype)
    launch(
        CorrectH(
            dtype,
            heads,
            ranges.shape[0],
            ranges.dtype,
            initial.dtype,
            has_initial,
            state_v_first,
        ),
        (ht, mt, fallback, ranges, initial, result),
        k.device,
    )
    return result, counts, fallback, ht, mt
