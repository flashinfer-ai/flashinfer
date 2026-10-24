# SPDX-License-Identifier: Apache-2.0
# Dataflow adapted from FlashQLA5538c35 SM120 fused_fwd,MIT,Qwen team2026.
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
from .preprocess import async_load32, write_tile


class FusedForward32(KeyedCompileMixin):
    def __init__(
        self,
        dtype,
        hk,
        heads,
        segments,
        index_dtype,
        initial_dtype,
        has_initial,
        initial_v_first,
        final_v_first,
        store_final,
        g_dtype,
        b_dtype,
    ):
        self.dtype, self.hk, self.heads, self.segments = dtype, hk, heads, segments
        self.index_dtype, self.initial_dtype, self.has_initial = (
            index_dtype,
            initial_dtype,
            has_initial,
        )
        self.g_dtype, self.b_dtype = g_dtype, b_dtype
        self.initial_v_first, self.final_v_first, self.store_final = (
            initial_v_first,
            final_v_first,
            store_final,
        )
        self.manual_cache_key(
            "dtype",
            "hk",
            "heads",
            "segments",
            "index_dtype",
            "initial_dtype",
            "has_initial",
            "initial_v_first",
            "final_v_first",
            "store_final",
            "g_dtype",
            "b_dtype",
        )

    @cute.jit
    def __call__(
        self, q, k, v, a, g, b, segments, initial, output, final, scale, stream
    ):
        dtype = self.dtype

        @cute.struct
        class Storage:
            q: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 136], 128]
            k: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 136], 128]
            v: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 72], 128]
            a: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * 40], 128]
            g: cute.struct.MemRange[cutlass.Float32, 2 * 32]
            b: cute.struct.MemRange[cutlass.Float32, 2 * 32]
            h: cute.struct.Align[cute.struct.MemRange[dtype, 64 * 136], 128]
            vd: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 72], 128]
            vn: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 72], 128]
            exp: cute.struct.MemRange[cutlass.Float32, 32]
            rev: cute.struct.MemRange[cutlass.Float32, 32]
            p: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 40], 128]
            bars: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 9], 8]

        self.storage = Storage
        self.kernel(q, k, v, a, g, b, segments, initial, output, final, scale).launch(
            grid=(2, self.heads, self.segments), block=(512, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self, q, k, v, a, g, b, segments, initial, output, final, scale: cutlass.Float32
    ):
        vb, head, seq = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 128
        row = cutlass.Int32(0)
        col = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        start = cutlass.Int32(segments[seq * 4])
        end = cutlass.Int32(segments[seq * 4 + 1])
        raw = cutlass.Int32(segments[seq * 4 + 2])
        last = cutlass.Int32(segments[seq * 4 + 3])
        count = (end - start + 31) // 32
        kh = head * self.hk // self.heads
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        sq = sm.q.get_tensor(cute.make_layout((32, 128, 2), stride=(136, 1, 32 * 136)))
        sk = sm.k.get_tensor(cute.make_layout((32, 128, 2), stride=(136, 1, 32 * 136)))
        sv = sm.v.get_tensor(cute.make_layout((32, 64, 2), stride=(72, 1, 32 * 72)))
        sa = sm.a.get_tensor(cute.make_layout((32, 32, 2), stride=(40, 1, 32 * 40)))
        sg = sm.g.get_tensor(cute.make_layout((32, 2), stride=(1, 32)))
        sb = sm.b.get_tensor(cute.make_layout((32, 2), stride=(1, 32)))
        sh = sm.h.get_tensor(cute.make_layout((64, 128), stride=(136, 1)))
        sd = sm.vd.get_tensor(cute.make_layout((32, 64), stride=(72, 1)))
        sn = sm.vn.get_tensor(cute.make_layout((32, 64), stride=(72, 1)))
        sp = sm.p.get_tensor(cute.make_layout((32, 32), stride=(40, 1)))
        ge = sm.exp.get_tensor(cute.make_layout(32))
        gr = sm.rev.get_tensor(cute.make_layout(32))
        bars = sm.bars.data_ptr()
        if tid == 0:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(bars + i, 128)
                cute.arch.mbarrier_init(bars + 2 + i, 384)
        if tid == 0:
            for j in cutlass.range_constexpr(5):
                expected = 384
                if j == 1:
                    expected = 256
                if j == 2 or j == 3:
                    expected = 128
                cute.arch.mbarrier_init(bars + 4 + j, expected)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tid >= 384:
            cute.arch.setmaxregister_decrease(32)
            for chunk in range(count):
                buf = chunk % 2
                if chunk >= 2:
                    cute.arch.mbarrier_wait(bars + 2 + buf, (chunk // 2 - 1) % 2)
                token = start + chunk * 32
                if token + 32 <= end:
                    async_load32(
                        q,
                        sq[None, None, buf],
                        (token * self.hk + kh) * 128,
                        self.hk * 128,
                        128,
                        lane,
                    )
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
                        (token * self.heads + head) * 128 + vb * 64,
                        self.heads * 128,
                        64,
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
                else:
                    for row in cutlass.range_constexpr(32):
                        qv, kv = self.dtype(0.0), self.dtype(0.0)
                        if token + row < end:
                            qv = q[((token + row) * self.hk + kh) * 128 + lane]
                            kv = k[((token + row) * self.hk + kh) * 128 + lane]
                        sq[row, lane, buf] = qv
                        sk[row, lane, buf] = kv
                    for j in cutlass.range_constexpr(16):
                        idx = lane + j * 128
                        row, col = idx // 64, idx % 64
                        value = self.dtype(0.0)
                        if token + row < end:
                            value = v[
                                ((token + row) * self.heads + head) * 128
                                + vb * 64
                                + col
                            ]
                        sv[row, col, buf] = value
                    for j in cutlass.range_constexpr(8):
                        idx = lane + j * 128
                        row, col = idx // 32, idx % 32
                        value = self.dtype(0.0)
                        if token + row < end:
                            value = a[((token + row) * self.heads + head) * 32 + col]
                        sa[row, col, buf] = value
                cute.arch.cp_async_commit_group()
                if lane < 32:
                    pos = min(token + lane, end - 1)
                    sg[lane, buf] = cutlass.Float32(g[pos * self.heads + head])
                    value = cutlass.Float32(0.0)
                    if token + lane < end:
                        value = cutlass.Float32(b[(token + lane) * self.heads + head])
                    sb[lane, buf] = value
                cute.arch.cp_async_wait_group(0)
                cute.arch.mbarrier_arrive(bars + buf)
        elif tid < 128:
            cute.arch.setmaxregister_increase(128)
            state = _acc(self.dtype, 64, 128)
            xy = _coords(self.dtype, 64, 128)
            if cutlass.const_expr(self.has_initial):
                for i in cutlass.range_constexpr(cute.size(state)):
                    row, col = xy[i]
                    idx = (seq * self.heads + head) * 16384
                    if cutlass.const_expr(self.initial_v_first):
                        idx += (vb * 64 + row) * 128 + col
                    else:
                        idx += col * 128 + vb * 64 + row
                    state[i] = cutlass.Float32(initial[idx])
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                _store(state, sh, 64, 128)
                cute.arch.mbarrier_arrive(bars + 5)
                cute.arch.mbarrier_wait(bars + 5, chunk % 2)
                decay = ge[31]
                for i in cutlass.range_constexpr(cute.size(state)):
                    state[i] *= decay
                cute.arch.mbarrier_arrive(bars + 8)
                cute.arch.mbarrier_wait(bars + 8, chunk % 2)
                _gemm(
                    _transpose(sn),
                    _transpose(sk[None, None, buf]),
                    state,
                    64,
                    128,
                    32,
                    True,
                )
                cute.arch.mbarrier_arrive(bars + 2 + buf)
            if cutlass.const_expr(self.store_final):
                if last:
                    for i in cutlass.range_constexpr(cute.size(state)):
                        row, col = xy[i]
                        idx = (raw * self.heads + head) * 16384
                        if cutlass.const_expr(self.final_v_first):
                            idx += (vb * 64 + row) * 128 + col
                        else:
                            idx += col * 128 + vb * 64 + row
                        final[idx] = state[i]
        elif tid < 256:
            cute.arch.setmaxregister_increase(128)
            xy = _coords(self.dtype, 32, 64)
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                if lane < 32:
                    ge[lane] = cute.math.exp2(
                        sg[lane, buf] * cutlass.Float32(1.442695), fastmath=True
                    )
                    value = cutlass.Float32(0.0)
                    if start + chunk * 32 + lane < end:
                        value = cute.math.exp2(
                            (sg[31, buf] - sg[lane, buf]) * cutlass.Float32(1.442695),
                            fastmath=True,
                        )
                    gr[lane] = value
                cute.arch.mbarrier_arrive(bars + 5)
                cute.arch.mbarrier_wait(bars + 5, chunk % 2)
                residual = _acc(self.dtype, 32, 64)
                _gemm(sk[None, None, buf], sh, residual, 32, 64, 128, True)
                for i in cutlass.range_constexpr(cute.size(residual)):
                    row, col = xy[i]
                    residual[i] = (
                        cutlass.Float32(sv[row, col, buf]) - ge[row] * residual[i]
                    )
                _store(residual, sv[None, None, buf], 32, 64)
                _sync(barrier_id=6, number_of_threads=128)
                cute.arch.mbarrier_wait(bars + 6, chunk % 2)
                delta = _acc(self.dtype, 32, 64)
                _gemm(
                    sa[None, None, buf],
                    _transpose(sv[None, None, buf]),
                    delta,
                    32,
                    64,
                    32,
                    True,
                )
                _store(delta, sd, 32, 64)
                cute.arch.mbarrier_arrive(bars + 7)
                for i in cutlass.range_constexpr(cute.size(delta)):
                    row, col = xy[i]
                    value = cutlass.Float32(0.0)
                    if start + chunk * 32 + row < end:
                        value = delta[i] * gr[row]
                    delta[i] = value
                _store(delta, sn, 32, 64)
                cute.arch.mbarrier_arrive(bars + 8)
                cute.arch.mbarrier_wait(bars + 8, chunk % 2)
                cute.arch.mbarrier_arrive(bars + 2 + buf)
        else:
            cute.arch.setmaxregister_increase(128)
            pxy = _coords(self.dtype, 32, 32)
            oxy = _coords(self.dtype, 32, 64)
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.mbarrier_arrive(bars + 4)
                cute.arch.mbarrier_wait(bars + 4, chunk % 2)
                p = _acc(self.dtype, 32, 32)
                ag = _acc(self.dtype, 32, 32)
                _gemm(sq[None, None, buf], sk[None, None, buf], p, 32, 32, 128, True)
                for i in cutlass.range_constexpr(cute.size(p)):
                    row, col = pxy[i]
                    gate = cutlass.Float32(0.0)
                    if row >= col:
                        gate = cute.math.exp2(
                            (sg[row, buf] - sg[col, buf]) * cutlass.Float32(1.442695),
                            fastmath=True,
                        )
                    ag[i] = cutlass.Float32(sa[row, col, buf]) * gate * sb[col, buf]
                    p[i] *= scale * gate
                _store(ag, sa[None, None, buf], 32, 32)
                _store(p, sp, 32, 32)
                cute.arch.mbarrier_wait(bars + 5, chunk % 2)
                out = _acc(self.dtype, 32, 64)
                _gemm(sq[None, None, buf], sh, out, 32, 64, 128, True)
                cute.arch.mbarrier_arrive(bars + 6)
                for i in cutlass.range_constexpr(cute.size(out)):
                    row, col = oxy[i]
                    out[i] *= scale * ge[row]
                cute.arch.mbarrier_wait(bars + 7, chunk % 2)
                _gemm(sp, _transpose(sd), out, 32, 64, 32, True)
                cute.arch.mbarrier_arrive(bars + 8)
                cute.arch.mbarrier_wait(bars + 8, chunk % 2)
                token = start + chunk * 32
                if token + 32 <= end:
                    write_tile(
                        out,
                        output,
                        (token * self.heads + head) * 128 + vb * 64,
                        32,
                        64,
                        self.heads * 128,
                        self.dtype,
                    )
                else:
                    for i in cutlass.range_constexpr(cute.size(out)):
                        row, col = oxy[i]
                        position = token + row
                        if position < end:
                            output[
                                (position * self.heads + head) * 128 + vb * 64 + col
                            ] = self.dtype(out[i])
                cute.arch.mbarrier_arrive(bars + 2 + buf)


def fused_forward(
    q,
    k,
    v,
    a,
    g,
    b,
    segments,
    initial,
    *,
    scale=None,
    state_v_first=True,
    initial_v_first=True,
    output_final_state=True,
    num_raw_sequences=1,
):
    dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[k.dtype]
    output = torch.empty_like(v)
    final = torch.empty(
        (num_raw_sequences, v.shape[1], 128, 128), device=v.device, dtype=torch.float32
    )
    has_initial = initial is not None
    if initial is None:
        initial = torch.empty((1,), device=k.device, dtype=torch.float32)
    # Scale is a one-element device tensor here so the shared launch helper stays tensor-only.
    # Main launch wraps scale as a native Float32 scalar separately.
    from .preprocess import wrap
    from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
        cached_compile,
    )
    import cuda.bindings.driver as driver

    kernel = FusedForward32(
        dtype,
        k.shape[1],
        v.shape[1],
        segments.shape[0],
        segments.dtype,
        initial.dtype,
        has_initial,
        initial_v_first,
        state_v_first,
        output_final_state,
        g.dtype,
        b.dtype,
    )
    args = tuple(map(wrap, (q, k, v, a, g, b, segments, initial, output, final))) + (
        cutlass.Float32(scale if scale is not None else 128**-0.5),
        driver.CUstream(torch.cuda.current_stream(k.device).cuda_stream),
    )
    fn = cached_compile(kernel, *args, compile_options=(cute.GPUArch("sm_120a"),))
    fn(*args)
    return output, final if output_final_state else None
