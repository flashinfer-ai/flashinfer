"""Parallel SSD chunk states, state propagation, and tensor-core outputs."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._kernel import _Scan, Program

CHUNK = 128


def chunk_capacity(q):
    return q.max_tokens // CHUNK + q.max_seqs


def workspace_elements(q):
    return chunk_capacity(q) * (
        q.heads * (q.head_dim * q.dstate + 2 * CHUNK) + q.groups * CHUNK * CHUNK
    )


class _Chunked(_Scan):
    def __init__(self, q, config):
        super().__init__(q)
        self.chunks = chunk_capacity(q)
        self.prepare_threads = config.prepare_warps * 32
        self.prepare_column_warps = config.prepare_warps // 4
        self.carry_elements = config.carry_elements
        self.output_rows = config.output_rows
        self.output_threads = self.output_rows * 8
        self.partial_elements = self.chunks * self.h * self.d * self.n
        self.cb_offset = self.partial_elements + self.chunks * self.h * 2 * CHUNK

    @cute.jit
    def partial_index(self, chunk: Int32, head: Int32, element: Int32):
        return (Int64(chunk) * self.h + Int64(head)) * (self.d * self.n) + Int64(
            element
        )

    @cute.jit
    def locate(self, chunk: Int32, cu: cute.Pointer, ns: cute.Pointer):
        lo, hi = Int32(0), Int32(ns[0])
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if Int32(cu[mid]) // CHUNK + mid <= chunk:
                lo = mid
            else:
                hi = mid
        start = Int32(cu[lo])
        local = chunk - (start // CHUNK + lo)
        tile = start + local * CHUNK
        end = cutlass.min(Int32(cu[lo + 1]), tile + CHUNK)
        return lo, tile, end

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        dt: cute.Pointer,
        A: cute.Pointer,
        B: cute.Pointer,
        C: cute.Pointer,
        D: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        cu: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        fresh: cute.Pointer,
        num_seqs: cute.Pointer,
        num_tokens: cute.Pointer,
        out: cute.Pointer,
        work: cute.Pointer,
        seq_capacity: Int32,
        token_capacity: Int32,
        slot_capacity: Int64,
        xs: Int64,
        bs: Int64,
        cs: Int64,
        dts: Int64,
        ss: Int64,
        stream: cuda.CUstream,
    ):
        self.prepare(
            x,
            dt,
            A,
            B,
            bias,
            cu,
            src,
            dst,
            num_seqs,
            num_tokens,
            work,
            token_capacity,
            slot_capacity,
            xs,
            bs,
            dts,
        ).launch(
            grid=(self.chunks, self.h, 1),
            block=(self.prepare_threads, 1, 1),
            stream=stream,
        )
        self.carry(
            state,
            cu,
            src,
            dst,
            fresh,
            num_seqs,
            num_tokens,
            work,
            token_capacity,
            slot_capacity,
            ss,
        ).launch(
            grid=(self.d * self.n // (256 * self.carry_elements), self.h, seq_capacity),
            block=(256, 1, 1),
            stream=stream,
        )
        self.grouped_cb(
            B,
            C,
            cu,
            src,
            dst,
            num_seqs,
            num_tokens,
            work,
            token_capacity,
            slot_capacity,
            bs,
            cs,
        ).launch(grid=(self.chunks, self.g, 1), block=(512, 1, 1), stream=stream)
        self.output(
            x,
            C,
            D,
            cu,
            src,
            dst,
            num_seqs,
            num_tokens,
            out,
            work,
            token_capacity,
            slot_capacity,
            xs,
            cs,
        ).launch(
            grid=(self.chunks, self.h, CHUNK // self.output_rows),
            block=(self.output_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def prepare(
        self,
        x: cute.Pointer,
        dt: cute.Pointer,
        A: cute.Pointer,
        B: cute.Pointer,
        bias: cute.Pointer,
        cu: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        ns: cute.Pointer,
        nt: cute.Pointer,
        work: cute.Pointer,
        token_capacity: Int32,
        slot_capacity: Int64,
        xs: Int64,
        bs: Int64,
        dts: Int64,
    ):
        chunk, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, tid = Int32(chunk), Int32(head), Int32(tid)
        warp, lane = tid // 32, tid % 32
        if ns[0] > 0:
            seq, tile, end = self.locate(chunk, cu, ns)
            if (
                tile >= 0
                and end > tile
                and end <= nt[0]
                and end <= token_capacity
                and src[seq] >= 0
                and src[seq] < slot_capacity
                and dst[seq] >= 0
                and dst[seq] < slot_capacity
            ):
                smem = cutlass.utils.SmemAllocator()
                sb = smem.allocate_tensor(
                    BFloat16, cute.make_layout((self.n * CHUNK,)), byte_alignment=16
                )
                sx = smem.allocate_tensor(
                    BFloat16, cute.make_layout((self.d * CHUNK,)), byte_alignment=16
                )
                factors = smem.allocate_tensor(
                    Float32, cute.make_layout((2 * CHUNK + 4,)), byte_alignment=16
                )
                step = Float32(0)
                if tid < CHUNK and tile + tid < end:
                    step = Float32(dt[Int64(tile + tid) * dts + Int64(head)]) + Float32(
                        bias[head]
                    )
                    if cutlass.const_expr(self.softplus):
                        if step <= Float32(20):
                            step = cute.math.log1p(
                                cute.math.exp(step, fastmath=True), fastmath=True
                            )
                prefix = step * Float32(A[head])
                for shift in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_up(prefix, 1 << shift)
                    if lane >= (1 << shift):
                        prefix += other
                if tid < CHUNK and lane == 31:
                    factors[2 * CHUNK + warp] = prefix
                cute.arch.sync_threads()
                base = (Int64(chunk) * self.h + Int64(head)) * (
                    2 * CHUNK
                ) + self.partial_elements
                if tid < CHUNK:
                    for w in cutlass.range_constexpr(3):
                        if warp > w:
                            prefix += factors[2 * CHUNK + w]
                    factors[tid], factors[CHUNK + tid] = step, prefix
                    work[base + Int64(tid)] = step
                    work[base + CHUNK + Int64(tid)] = prefix
                cute.arch.sync_threads()
                group = head // (self.h // self.g)
                for i in range(self.n * CHUNK // self.prepare_threads):
                    idx = tid + i * self.prepare_threads
                    t, n = idx // self.n, idx % self.n
                    value = Float32(0)
                    if tile + t < end:
                        value = Float32(
                            B[Int64(tile + t) * bs + Int64(group * self.n + n)]
                        )
                    value *= factors[t] * cute.math.exp(
                        factors[2 * CHUNK - 1] - factors[CHUNK + t], fastmath=True
                    )
                    sb[self.offset(n, t, CHUNK)] = value.to(BFloat16)
                for i in range(self.d * CHUNK // self.prepare_threads):
                    idx = tid + i * self.prepare_threads
                    t, d = idx // self.d, idx % self.d
                    value = Float32(0)
                    if tile + t < end:
                        value = Float32(
                            x[Int64(tile + t) * xs + Int64(head * self.d + d)]
                        )
                    sx[self.offset(d, t, CHUNK)] = value.to(BFloat16)
                cute.arch.sync_threads()
                for v in cutlass.range_constexpr(self.d // 64):
                    for j in cutlass.range_constexpr(
                        self.n // (8 * self.prepare_column_warps)
                    ):
                        a0, a1, a2, a3 = Float32(0), Float32(0), Float32(0), Float32(0)
                        for k in cutlass.range_constexpr(0, CHUNK, 16):
                            a0, a1, a2, a3 = self.mma(
                                sx,
                                sb,
                                (warp // self.prepare_column_warps) * 16 + v * 64,
                                (warp % self.prepare_column_warps)
                                * (self.n // self.prepare_column_warps)
                                + j * 8,
                                Int32(k),
                                CHUNK,
                                lane,
                                a0,
                                a1,
                                a2,
                                a3,
                            )
                        vals = cute.make_rmem_tensor((4,), Float32)
                        vals[0], vals[1], vals[2], vals[3] = a0, a1, a2, a3
                        for z in cutlass.range_constexpr(4):
                            r = (
                                (warp // self.prepare_column_warps) * 16
                                + v * 64
                                + lane // 4
                                + (z // 2) * 8
                            )
                            c = (
                                (warp % self.prepare_column_warps)
                                * (self.n // self.prepare_column_warps)
                                + j * 8
                                + (lane % 4) * 2
                                + z % 2
                            )
                            work[self.partial_index(chunk, head, r * self.n + c)] = (
                                vals[z]
                            )

    @cute.kernel
    def carry(
        self,
        state: cute.Pointer,
        cu: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        fresh: cute.Pointer,
        ns: cute.Pointer,
        nt: cute.Pointer,
        work: cute.Pointer,
        token_capacity: Int32,
        slot_capacity: Int64,
        ss: Int64,
    ):
        block, head, seq = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        seq, head, element = (
            Int32(seq),
            Int32(head),
            Int32(block) * (256 * self.carry_elements) + Int32(tid),
        )
        if seq < ns[0]:
            start, end = Int32(cu[seq]), Int32(cu[seq + 1])
            source, destination = Int64(src[seq]), Int64(dst[seq])
            if (
                start >= 0
                and end > start
                and end <= nt[0]
                and end <= token_capacity
                and source >= 0
                and source < slot_capacity
                and destination >= 0
                and destination < slot_capacity
            ):
                offset = Int64(head) * (self.d * self.n) + Int64(element)
                acc = cute.make_rmem_tensor((self.carry_elements,), Float32)
                acc.fill(0)
                if not fresh[seq]:
                    for j in cutlass.range_constexpr(self.carry_elements):
                        acc[j] = Float32(state[source * ss + offset + j * 256])
                first = start // CHUNK + seq
                count = (end - start + CHUNK - 1) // CHUNK
                for i in range(count):
                    index = Int64(first + i) * self.h + Int64(head)
                    pos = self.partial_index(first + i, head, element)
                    decay = cute.math.exp(
                        work[
                            self.partial_elements + index * (2 * CHUNK) + 2 * CHUNK - 1
                        ],
                        fastmath=True,
                    )
                    update = cute.make_rmem_tensor((self.carry_elements,), Float32)
                    for j in cutlass.range_constexpr(self.carry_elements):
                        update[j] = work[pos + j * 256]
                    for j in cutlass.range_constexpr(self.carry_elements):
                        work[pos + j * 256] = acc[j]
                        acc[j] = acc[j] * decay + update[j]
                for j in cutlass.range_constexpr(self.carry_elements):
                    state[destination * ss + offset + j * 256] = acc[j].to(
                        self.state_type
                    )

    @cute.kernel
    def grouped_cb(
        self,
        B: cute.Pointer,
        C: cute.Pointer,
        cu: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        ns: cute.Pointer,
        nt: cute.Pointer,
        work: cute.Pointer,
        token_capacity: Int32,
        slot_capacity: Int64,
        bs: Int64,
        cs: Int64,
    ):
        chunk, group, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        chunk, group, tid = Int32(chunk), Int32(group), Int32(tid)
        warp, lane = tid // 32, tid % 32
        query_warp, value_warp = warp // 2, warp % 2
        if ns[0] > 0:
            seq, tile, end = self.locate(chunk, cu, ns)
            if (
                tile >= 0
                and end > tile
                and end <= nt[0]
                and end <= token_capacity
                and src[seq] >= 0
                and src[seq] < slot_capacity
                and dst[seq] >= 0
                and dst[seq] < slot_capacity
            ):
                smem = cutlass.utils.SmemAllocator()
                sb = smem.allocate_tensor(
                    BFloat16, cute.make_layout((CHUNK * self.n,)), byte_alignment=16
                )
                sc = smem.allocate_tensor(
                    BFloat16, cute.make_layout((CHUNK * self.n,)), byte_alignment=16
                )
                for i in range(CHUNK * self.n // 512):
                    idx = tid + i * 512
                    t, n = idx // self.n, idx % self.n
                    bv, cv = Float32(0), Float32(0)
                    if tile + t < end:
                        bv = Float32(
                            B[Int64(tile + t) * bs + Int64(group * self.n + n)]
                        )
                        cv = Float32(
                            C[Int64(tile + t) * cs + Int64(group * self.n + n)]
                        )
                    sb[self.offset(t, n, self.n)] = bv.to(BFloat16)
                    sc[self.offset(t, n, self.n)] = cv.to(BFloat16)
                cute.arch.sync_threads()
                base = (
                    self.cb_offset
                    + (Int64(chunk) * self.g + Int64(group)) * CHUNK * CHUNK
                )
                for j in cutlass.range_constexpr(CHUNK // 16):
                    a0, a1, a2, a3 = Float32(0), Float32(0), Float32(0), Float32(0)
                    for k in cutlass.range_constexpr(0, self.n, 16):
                        a0, a1, a2, a3 = self.mma(
                            sc,
                            sb,
                            query_warp * 16,
                            value_warp * 64 + j * 8,
                            Int32(k),
                            self.n,
                            lane,
                            a0,
                            a1,
                            a2,
                            a3,
                        )
                    vals = cute.make_rmem_tensor((4,), Float32)
                    vals[0], vals[1], vals[2], vals[3] = a0, a1, a2, a3
                    for z in cutlass.range_constexpr(4):
                        row = query_warp * 16 + lane // 4 + (z // 2) * 8
                        column = value_warp * 64 + j * 8 + (lane % 4) * 2 + z % 2
                        work[base + Int64(row * CHUNK + column)] = vals[z]

    @cute.kernel
    def output(
        self,
        x: cute.Pointer,
        C: cute.Pointer,
        D: cute.Pointer,
        cu: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        ns: cute.Pointer,
        nt: cute.Pointer,
        out: cute.Pointer,
        work: cute.Pointer,
        token_capacity: Int32,
        slot_capacity: Int64,
        xs: Int64,
        cs: Int64,
    ):
        chunk, head, half = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, half, tid = Int32(chunk), Int32(head), Int32(half), Int32(tid)
        warp, lane = tid // 32, tid % 32
        query_warp, value_warp = warp // 4, warp % 4
        if ns[0] > 0:
            seq, tile, end = self.locate(chunk, cu, ns)
            if (
                tile >= 0
                and end > tile + half * self.output_rows
                and end <= nt[0]
                and end <= token_capacity
                and src[seq] >= 0
                and src[seq] < slot_capacity
                and dst[seq] >= 0
                and dst[seq] < slot_capacity
            ):
                smem = cutlass.utils.SmemAllocator()
                sb = smem.allocate_tensor(
                    BFloat16,
                    cute.make_layout((max(self.d * self.n, self.d * CHUNK),)),
                    byte_alignment=16,
                )
                sc = smem.allocate_tensor(
                    BFloat16,
                    cute.make_layout(
                        (max(self.output_rows * self.n, 2 * self.output_rows * CHUNK),)
                    ),
                    byte_alignment=16,
                )
                score = sc
                factors = smem.allocate_tensor(
                    Float32, cute.make_layout((2 * CHUNK,)), byte_alignment=16
                )
                index = Int64(chunk) * self.h + Int64(head)
                fbase = self.partial_elements + index * (2 * CHUNK)
                if tid < CHUNK:
                    factors[tid], factors[CHUNK + tid] = (
                        work[fbase + Int64(tid)],
                        work[fbase + CHUNK + Int64(tid)],
                    )
                group = head // (self.h // self.g)
                for i in range(self.output_rows * self.n // self.output_threads):
                    idx = tid + i * self.output_threads
                    t, n = idx // self.n, idx % self.n
                    value = Float32(0)
                    if tile + half * self.output_rows + t < end:
                        value = Float32(
                            C[
                                Int64(tile + half * self.output_rows + t) * cs
                                + Int64(group * self.n + n)
                            ]
                        )
                    sc[self.offset(t, n, self.n)] = value.to(BFloat16)
                cute.arch.sync_threads()
                acc = cute.make_rmem_tensor((self.d // 32, 4), Float32)
                acc.fill(0)
                for part in cutlass.range_constexpr(2):
                    for i in range(self.d * self.n // self.output_threads):
                        idx = tid + i * self.output_threads
                        d, n = idx // self.n, idx % self.n
                        value = work[self.partial_index(chunk, head, idx)]
                        high = value.to(BFloat16)
                        sb[self.offset(d, n, self.n)] = (
                            high
                            if cutlass.const_expr(part == 0)
                            else (value - Float32(high)).to(BFloat16)
                        )
                    cute.arch.sync_threads()
                    for j in cutlass.range_constexpr(self.d // 32):
                        a0, a1, a2, a3 = acc[j, 0], acc[j, 1], acc[j, 2], acc[j, 3]
                        for k in cutlass.range_constexpr(0, self.n, 16):
                            a0, a1, a2, a3 = self.mma(
                                sc,
                                sb,
                                query_warp * 16,
                                value_warp * (self.d // 4) + j * 8,
                                Int32(k),
                                self.n,
                                lane,
                                a0,
                                a1,
                                a2,
                                a3,
                            )
                        acc[j, 0], acc[j, 1], acc[j, 2], acc[j, 3] = a0, a1, a2, a3
                    cute.arch.sync_threads()
                for j in cutlass.range_constexpr(self.d // 32):
                    decay0 = cute.math.exp(
                        factors[
                            CHUNK
                            + half * self.output_rows
                            + query_warp * 16
                            + lane // 4
                        ],
                        fastmath=True,
                    )
                    decay1 = cute.math.exp(
                        factors[
                            CHUNK
                            + half * self.output_rows
                            + query_warp * 16
                            + lane // 4
                            + 8
                        ],
                        fastmath=True,
                    )
                    acc[j, 0] *= decay0
                    acc[j, 1] *= decay0
                    acc[j, 2] *= decay1
                    acc[j, 3] *= decay1
                cb = (
                    self.cb_offset
                    + (Int64(chunk) * self.g + Int64(group)) * CHUNK * CHUNK
                )
                for i in range(self.output_rows * CHUNK // self.output_threads):
                    idx = tid + i * self.output_threads
                    t, column = idx // CHUNK, idx % CHUNK
                    row = half * self.output_rows + t
                    value = Float32(0)
                    if column <= row:
                        value = (
                            work[cb + Int64(row * CHUNK + column)]
                            * factors[column]
                            * cute.math.exp(
                                factors[CHUNK + row] - factors[CHUNK + column],
                                fastmath=True,
                            )
                        )
                    high = value.to(BFloat16)
                    score[self.offset(t, column, CHUNK)] = high
                    score[self.offset(t + self.output_rows, column, CHUNK)] = (
                        value - Float32(high)
                    ).to(BFloat16)
                for i in range(self.d * CHUNK // self.output_threads):
                    idx = tid + i * self.output_threads
                    t, d = idx // self.d, idx % self.d
                    value = Float32(0)
                    if tile + t < end:
                        value = Float32(
                            x[Int64(tile + t) * xs + Int64(head * self.d + d)]
                        )
                    sb[self.offset(d, t, CHUNK)] = value.to(BFloat16)
                cute.arch.sync_threads()
                for j in cutlass.range_constexpr(self.d // 32):
                    a0, a1, a2, a3 = acc[j, 0], acc[j, 1], acc[j, 2], acc[j, 3]
                    for k in cutlass.range_constexpr(0, CHUNK, 16):
                        a0, a1, a2, a3 = self.mma(
                            score,
                            sb,
                            query_warp * 16,
                            value_warp * (self.d // 4) + j * 8,
                            Int32(k),
                            CHUNK,
                            lane,
                            a0,
                            a1,
                            a2,
                            a3,
                        )
                        a0, a1, a2, a3 = self.mma(
                            score,
                            sb,
                            query_warp * 16 + self.output_rows,
                            value_warp * (self.d // 4) + j * 8,
                            Int32(k),
                            CHUNK,
                            lane,
                            a0,
                            a1,
                            a2,
                            a3,
                        )
                    vals = cute.make_rmem_tensor((4,), Float32)
                    vals[0], vals[1], vals[2], vals[3] = a0, a1, a2, a3
                    for z in cutlass.range_constexpr(4):
                        t = (
                            half * self.output_rows
                            + query_warp * 16
                            + lane // 4
                            + (z // 2) * 8
                        )
                        d = value_warp * (self.d // 4) + j * 8 + (lane % 4) * 2 + z % 2
                        if tile + t < end:
                            result = vals[z] + Float32(D[head]) * Float32(
                                sb[self.offset(d, t, CHUNK)]
                            )
                            out[
                                Int64(tile + t) * (self.h * self.d)
                                + Int64(head * self.d + d)
                            ] = result.to(BFloat16)


@program_cache
def compile_chunked(geometry, config_payload, ordinal):
    from ._tuning import MambaQuery, MambaConfig

    q = MambaQuery(**dict(geometry))
    types = (
        BFloat16,
        BFloat16 if q.dt_dtype == "bfloat16" else Float32,
        Float32,
        BFloat16,
        BFloat16,
        BFloat16 if q.d_dtype == "bfloat16" else Float32,
        BFloat16 if q.bias_dtype == "bfloat16" else Float32,
        BFloat16 if q.state_dtype == "bfloat16" else Float32,
        Int32,
        Int64 if q.index_dtype == "int64" else Int32,
        Int64 if q.index_dtype == "int64" else Int32,
        Boolean,
        Int32,
        Int32,
        BFloat16,
        Float32,
    )
    entry, key = (
        _Chunked(q, MambaConfig(**dict(config_payload))),
        (geometry, config_payload, ordinal),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=max(1, t.width // 8))
        for t in types
    )
    with torch.cuda.device(ordinal):
        raw = compile_cute(
            entry,
            *pointers,
            Int32(1),
            Int32(1),
            Int64(1),
            *[Int64(1) for _ in range(5)],
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "sequence.mamba_ssd.chunked", 2, key
            ),
        )
    return attach_programs(Program(raw, types), raw)
