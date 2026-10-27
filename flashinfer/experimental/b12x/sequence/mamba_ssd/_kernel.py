"""Warp-MMA Mamba-2 scan with an FP32 state resident across 16-token tiles."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Boolean

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.intrinsics.memory import (
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x2_b16,
    shared_ptr_to_u32,
)
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr


class _Scan:
    def __init__(self, q):
        self.h, self.g, self.d, self.n = q.heads, q.groups, q.head_dim, q.dstate
        self.softplus = q.dt_softplus
        self.threads = q.head_dim * 2
        self.state_type = BFloat16 if q.state_dtype == "bfloat16" else Float32

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
        self.kernel(
            x,
            dt,
            A,
            B,
            C,
            D,
            bias,
            state,
            cu,
            src,
            dst,
            fresh,
            num_seqs,
            num_tokens,
            out,
            seq_capacity,
            token_capacity,
            slot_capacity,
            xs,
            bs,
            cs,
            dts,
            ss,
        ).launch(
            grid=(seq_capacity, self.h, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.jit
    def offset(self, row: Int32, col: Int32, width: cutlass.Constexpr):
        return row * width + (col ^ ((row & min(7, width // 8 - 1)) * 8))

    @cute.jit
    def mma(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        ar: Int32,
        bc: Int32,
        k: Int32,
        width: cutlass.Constexpr,
        lane: Int32,
        c0: Float32,
        c1: Float32,
        c2: Float32,
        c3: Float32,
    ):
        ap, bp = shared_ptr_to_u32(a.iterator), shared_ptr_to_u32(b.iterator)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
            ap + self.offset(ar + lane % 16, k + (lane // 16) * 8, width) * 2
        )
        b0, b1 = ldmatrix_m8n8x2_b16(
            bp + self.offset(bc + lane % 8, k + ((lane // 8) % 2) * 8, width) * 2
        )
        return bf16_mma_m16n8k16_f32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1)

    @cute.kernel
    def kernel(
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
        seq_capacity: Int32,
        token_capacity: Int32,
        slot_capacity: Int64,
        xs: Int64,
        bs: Int64,
        cs: Int64,
        dts: Int64,
        ss: Int64,
    ):
        seq, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        seq, head, tid = Int32(seq), Int32(head), Int32(tid)
        warp, lane = tid // 32, tid % 32
        if seq < num_seqs[0] and seq < seq_capacity:
            start, end = Int32(cu[seq]), Int32(cu[seq + 1])
            source, destination = Int64(src[seq]), Int64(dst[seq])
            if (
                start >= 0
                and end > start
                and end <= num_tokens[0]
                and end <= token_capacity
                and source >= 0
                and source < slot_capacity
                and destination >= 0
                and destination < slot_capacity
            ):
                smem = cutlass.utils.SmemAllocator()
                sb = smem.allocate_tensor(
                    BFloat16, cute.make_layout((16 * self.n,)), byte_alignment=16
                )
                sc = smem.allocate_tensor(
                    BFloat16, cute.make_layout((16 * self.n,)), byte_alignment=16
                )
                sx = smem.allocate_tensor(
                    BFloat16, cute.make_layout((self.d * 16,)), byte_alignment=16
                )
                score = smem.allocate_tensor(
                    BFloat16, cute.make_layout((32 * 16,)), byte_alignment=16
                )
                shadow = smem.allocate_tensor(
                    BFloat16, cute.make_layout((self.d * self.n,)), byte_alignment=16
                )
                residual = smem.allocate_tensor(
                    BFloat16, cute.make_layout((self.d * self.n,)), byte_alignment=16
                )
                factors = smem.allocate_tensor(
                    Float32, cute.make_layout((48,)), byte_alignment=16
                )
                acc = cute.make_rmem_tensor((self.n // 8, 4), Float32)
                acc.fill(0)
                r, col = warp * 16 + lane // 4, (lane % 4) * 2
                group = head // (self.h // self.g)
                state_head = Int64(head) * (self.d * self.n)
                if not fresh[seq]:
                    for j in cutlass.range_constexpr(self.n // 8):
                        for z in cutlass.range_constexpr(4):
                            rr, cc = r + (z // 2) * 8, col + j * 8 + z % 2
                            acc[j, z] = Float32(
                                state[
                                    source * ss
                                    + state_head
                                    + Int64(rr) * self.n
                                    + Int64(cc)
                                ]
                            )
                for tile in range(start, end, 16):
                    rows = cutlass.min(Int32(16), end - tile)
                    if tid < 16:
                        step = Float32(0)
                        if tid < rows:
                            step = Float32(
                                dt[Int64(tile + tid) * dts + Int64(head)]
                            ) + Float32(bias[head])
                            if cutlass.const_expr(self.softplus):
                                if step <= Float32(20):
                                    step = cute.math.log1p(
                                        cute.math.exp(step, fastmath=True),
                                        fastmath=True,
                                    )
                        factors[tid] = step
                        factors[16 + tid] = step * Float32(A[head])
                    for i in cutlass.range_constexpr(
                        (16 * self.n + self.threads - 1) // self.threads
                    ):
                        idx = tid + i * self.threads
                        if idx < 16 * self.n:
                            t, n = idx // self.n, idx % self.n
                            vb, vc = Float32(0), Float32(0)
                            if t < rows:
                                vb = Float32(
                                    B[Int64(tile + t) * bs + Int64(group * self.n + n)]
                                )
                                vc = Float32(
                                    C[Int64(tile + t) * cs + Int64(group * self.n + n)]
                                )
                            sb[self.offset(t, n, self.n)] = vb.to(BFloat16)
                            sc[self.offset(t, n, self.n)] = vc.to(BFloat16)
                    for i in cutlass.range_constexpr(16 * self.d // self.threads):
                        idx = tid + i * self.threads
                        d, t = idx // 16, idx % 16
                        value = Float32(0)
                        if t < rows:
                            value = Float32(
                                x[Int64(tile + t) * xs + Int64(head * self.d + d)]
                            )
                        sx[self.offset(d, t, 16)] = value.to(BFloat16)
                    cute.arch.sync_threads()
                    if tid < 16:
                        prefix = Float32(0)
                        for j in cutlass.range_constexpr(16):
                            if j <= tid:
                                prefix += factors[16 + j]
                        factors[32 + tid] = prefix
                    cute.arch.sync_threads()
                    if warp < 2:
                        s0, s1, s2, s3 = Float32(0), Float32(0), Float32(0), Float32(0)
                        for k in cutlass.range_constexpr(0, self.n, 16):
                            s0, s1, s2, s3 = self.mma(
                                sc,
                                sb,
                                Int32(0),
                                warp * 8,
                                Int32(k),
                                self.n,
                                lane,
                                s0,
                                s1,
                                s2,
                                s3,
                            )
                        vals = cute.make_rmem_tensor((4,), Float32)
                        vals[0], vals[1], vals[2], vals[3] = s0, s1, s2, s3
                        for z in cutlass.range_constexpr(4):
                            t, j = lane // 4 + (z // 2) * 8, warp * 8 + col + z % 2
                            value = Float32(0)
                            if j <= t:
                                value = (
                                    vals[z]
                                    * factors[j]
                                    * cute.math.exp(
                                        factors[32 + t] - factors[32 + j], fastmath=True
                                    )
                                )
                            high = value.to(BFloat16)
                            score[self.offset(t, j, 16)] = high
                            score[self.offset(t + 16, j, 16)] = (
                                value - Float32(high)
                            ).to(BFloat16)
                    cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(
                        (16 * self.n + self.threads - 1) // self.threads
                    ):
                        idx = tid + i * self.threads
                        if idx < 16 * self.n:
                            t, n = idx // self.n, idx % self.n
                            off = self.offset(t, n, self.n)
                            sb[off] = (
                                Float32(sb[off])
                                * factors[t]
                                * cute.math.exp(
                                    factors[47] - factors[32 + t], fastmath=True
                                )
                            ).to(BFloat16)
                    for j in cutlass.range_constexpr(self.n // 8):
                        for z in cutlass.range_constexpr(4):
                            off = self.offset(
                                r + (z // 2) * 8, col + j * 8 + z % 2, self.n
                            )
                            high = acc[j, z].to(BFloat16)
                            shadow[off] = high
                            residual[off] = (acc[j, z] - Float32(high)).to(BFloat16)
                    cute.arch.sync_threads()
                    for v in cutlass.range_constexpr(2):
                        o0, o1, o2, o3 = Float32(0), Float32(0), Float32(0), Float32(0)
                        for k in cutlass.range_constexpr(0, self.n, 16):
                            o0, o1, o2, o3 = self.mma(
                                sc,
                                shadow,
                                Int32(0),
                                warp * 16 + v * 8,
                                Int32(k),
                                self.n,
                                lane,
                                o0,
                                o1,
                                o2,
                                o3,
                            )
                            o0, o1, o2, o3 = self.mma(
                                sc,
                                residual,
                                Int32(0),
                                warp * 16 + v * 8,
                                Int32(k),
                                self.n,
                                lane,
                                o0,
                                o1,
                                o2,
                                o3,
                            )
                        decay0 = cute.math.exp(factors[32 + lane // 4], fastmath=True)
                        decay1 = cute.math.exp(factors[40 + lane // 4], fastmath=True)
                        o0, o1, o2, o3 = self.mma(
                            score,
                            sx,
                            Int32(0),
                            warp * 16 + v * 8,
                            Int32(0),
                            16,
                            lane,
                            o0 * decay0,
                            o1 * decay0,
                            o2 * decay1,
                            o3 * decay1,
                        )
                        o0, o1, o2, o3 = self.mma(
                            score,
                            sx,
                            Int32(16),
                            warp * 16 + v * 8,
                            Int32(0),
                            16,
                            lane,
                            o0,
                            o1,
                            o2,
                            o3,
                        )
                        vals = cute.make_rmem_tensor((4,), Float32)
                        vals[0], vals[1], vals[2], vals[3] = o0, o1, o2, o3
                        for z in cutlass.range_constexpr(4):
                            t, d = (
                                lane // 4 + (z // 2) * 8,
                                warp * 16 + v * 8 + col + z % 2,
                            )
                            if t < rows:
                                value = vals[z] + Float32(D[head]) * Float32(
                                    sx[self.offset(d, t, 16)]
                                )
                                out[
                                    Int64(tile + t) * (self.h * self.d)
                                    + Int64(head * self.d + d)
                                ] = value.to(BFloat16)
                    decay = cute.math.exp(factors[47], fastmath=True)
                    # The state update needs B transposed relative to the score operand.
                    cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(
                        (16 * self.n + self.threads - 1) // self.threads
                    ):
                        idx = tid + i * self.threads
                        if idx < 16 * self.n:
                            t, n = idx // self.n, idx % self.n
                            sc[self.offset(n, t, 16)] = sb[self.offset(t, n, self.n)]
                    cute.arch.sync_threads()
                    for j in cutlass.range_constexpr(self.n // 8):
                        a0, a1, a2, a3 = self.mma(
                            sx,
                            sc,
                            warp * 16,
                            Int32(j * 8),
                            Int32(0),
                            16,
                            lane,
                            acc[j, 0] * decay,
                            acc[j, 1] * decay,
                            acc[j, 2] * decay,
                            acc[j, 3] * decay,
                        )
                        acc[j, 0], acc[j, 1], acc[j, 2], acc[j, 3] = a0, a1, a2, a3
                    cute.arch.sync_threads()
                for j in cutlass.range_constexpr(self.n // 8):
                    for z in cutlass.range_constexpr(4):
                        rr, cc = r + (z // 2) * 8, col + j * 8 + z % 2
                        state[
                            destination * ss
                            + state_head
                            + Int64(rr) * self.n
                            + Int64(cc)
                        ] = acc[j, z].to(self.state_type)


@dataclass(frozen=True)
class Program:
    raw: object
    types: tuple


@program_cache
def compile_scan(geometry, ordinal):
    from ._tuning import MambaQuery

    q = MambaQuery(**dict(geometry))
    types = (
        BFloat16,
        getattr(cutlass, "BFloat16" if q.dt_dtype == "bfloat16" else "Float32"),
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
    )
    entry = _Scan(q)
    key = (geometry, ordinal)
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
            compile_spec=KernelCompileSpec.from_key("sequence.mamba_ssd", 1, key),
        )
    return attach_programs(Program(raw, types), raw)


def launch(program, q, tensors):
    pointers = tuple(
        make_ptr(
            t, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=max(1, t.width // 8)
        )
        for t, x in zip(program.types, tensors, strict=True)
    )
    with torch.cuda.device(tensors[0].device):
        run_compiled(
            program.raw,
            (
                *pointers,
                Int32(q.max_seqs),
                Int32(q.max_tokens),
                Int64(q.max_slots),
                Int64(q.x_stride),
                Int64(q.b_stride),
                Int64(q.c_stride),
                Int64(q.dt_stride),
                Int64(q.state_stride),
                current_cuda_stream(),
            ),
        )
