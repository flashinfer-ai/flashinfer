# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer team
"""Opt-in 32-token, four-warp-group GDN for SM120.

Algorithm/scheduling reference: QwenLM/FlashQLA blackwell_sm120/fused_fwd.py.
Independent CuTe implementation; no TileLang runtime dependency.

WG0 owns FP32 state, WG1 computes delta values, WG2 computes output, WG3
prefetches two shared-memory slots. V is split into 64-column CTAs so the
shared-memory footprint fits SM120. All tensor-core accumulations are FP32.
This source requires GPU compilation, numerical and sanitizer acceptance tests.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

from .collective_inverse_hmma import CollectiveInverse
from .custom_compile_cache import KeyedCompileMixin, cached_compile
from .sm120_pipeline32_layout import workspace_shapes


@cute.jit
def _transpose(t: cute.Tensor):
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (cute.size(t, mode=[1]), cute.size(t, mode=[0])),
            stride=(t.layout.stride[1], t.layout.stride[0]),
        ),
    )


@cute.jit
def _mma(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    return cute.make_tiled_mma(
        warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
        cute.make_layout((2, 2, 1)),
        permutation_mnk=(m, n, 16),
    )


@cute.jit
def _gemm(
    a: cute.Tensor,
    b_nk: cute.Tensor,
    acc: cute.Tensor,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    use_ldmatrix: cutlass.Constexpr,
):
    """acc += A[M,K] B[N,K]^T, all 128 threads in the calling WG.

    Load one K=16 fragment at a time to bound register lifetimes. LDSM moves
    8x8 matrices directly into MMA register layout. Transposed shared views use
    the transposing atom. Pitches and K-tile offsets remain 16-byte aligned.
    The generic branch reproduces the original loader for paired ablations.
    """
    tid, _, _ = cute.arch.thread_idx()
    mma = _mma(a.element_type, m, n)
    thr = mma.get_slice(tid % 128)
    for block in cutlass.range_constexpr(k // 16):
        sa = cute.local_tile(a, (m, 16), (0, block))
        sb = cute.local_tile(b_nk, (n, 16), (0, block))
        ra = thr.make_fragment_A(thr.partition_A(sa))
        rb = thr.make_fragment_B(thr.partition_B(sb))
        if cutlass.const_expr(use_ldmatrix):
            atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=(a.layout.stride[0] == 1), num_matrices=4
                ),
                a.element_type,
            )
            atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=(b_nk.layout.stride[0] == 1), num_matrices=4
                ),
                b_nk.element_type,
            )
            copy_a = cute.make_tiled_copy_A(atom_a, mma)
            copy_b = cute.make_tiled_copy_B(atom_b, mma)
            ta, tb = copy_a.get_slice(tid % 128), copy_b.get_slice(tid % 128)
            cute.copy(copy_a, ta.partition_S(sa), ta.retile(ra))
            cute.copy(copy_b, tb.partition_S(sb), tb.retile(rb))
        else:
            cute.autovec_copy(thr.partition_A(sa), ra)
            cute.autovec_copy(thr.partition_B(sb), rb)
        cute.gemm(mma, acc, ra, rb, acc)


@cute.jit
def _acc(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    thr = _mma(dtype, m, n).get_slice(tid % 128)
    out = cute.make_rmem_tensor(thr.partition_shape_C((m, n)), cutlass.Float32)
    out.fill(0.0)
    return out


@cute.jit
def _coords(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    return (
        _mma(dtype, m, n)
        .get_slice(tid % 128)
        .partition_C(cute.make_identity_tensor((m, n)))
    )


@cute.jit
def _store(
    acc: cute.Tensor, dst: cute.Tensor, m: cutlass.Constexpr, n: cutlass.Constexpr
):
    coords = _coords(dst.element_type, m, n)
    for i in cutlass.range_constexpr(cute.size(acc)):
        r, c = coords[i]
        dst[r, c] = dst.element_type(acc[i])


class _Precompute32(KeyedCompileMixin):
    def __init__(
        self,
        dtype,
        hk,
        heads,
        seqs,
        has_alpha,
        has_beta,
        use_ldmatrix,
        pad,
        cu_seqlens_dtype,
    ):
        self.dtype, self.hk, self.heads, self.seqs = dtype, hk, heads, seqs
        self.has_alpha, self.has_beta = has_alpha, has_beta
        self.use_ldmatrix, self.pad = use_ldmatrix, pad
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.manual_cache_key(
            "dtype",
            "hk",
            "heads",
            "seqs",
            "has_alpha",
            "has_beta",
            "use_ldmatrix",
            "pad",
            "cu_seqlens_dtype",
        )

    @cute.jit
    def __call__(self, k, alpha, beta, cu, matrix, logs, capacity: int, stream):
        dtype = self.dtype
        pad = self.pad

        @cute.struct
        class Storage:
            key: cute.struct.Align[cute.struct.MemRange[dtype, 32 * (128 + pad)], 128]
            inverse: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, 32 * (32 + pad)], 16
            ]
            gates: cute.struct.MemRange[cutlass.Float32, 32]
            beta: cute.struct.MemRange[cutlass.Float32, 32]

        self.storage = Storage
        self.kernel(k, alpha, beta, cu, matrix, logs).launch(
            grid=(capacity, self.heads, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, k, alpha, beta, cu, matrix, logs):
        slot, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(self.seqs)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cutlass.Int32(cu[mid]) // 32 + mid <= slot:
                lo = mid
            else:
                hi = mid
        start = cutlass.Int32(cu[lo])
        end = cutlass.Int32(cu[lo + 1])
        token = start + (slot - start // 32 - lo) * 32
        if token < end:
            sm = cutlass.utils.SmemAllocator().allocate(self.storage)
            sk = sm.key.get_tensor(
                cute.make_layout((32, 128), stride=(128 + self.pad, 1))
            )
            inv = sm.inverse.get_tensor(
                cute.make_layout((32, 32), stride=(32 + self.pad, 1))
            )
            g = sm.gates.get_tensor(cute.make_layout(32))
            b = sm.beta.get_tensor(cute.make_layout(32))
            kh = head * self.hk // self.heads
            for row in cutlass.range_constexpr(32):
                val = self.dtype(0.0)
                if token + row < end:
                    val = k[((token + row) * self.hk + kh) * 128 + tid]
                sk[row, tid] = val
            if tid < 32:
                # Prefix is local to this chunk. Tail lanes repeat the last gate.
                log = cutlass.Float32(0.0)
                if cutlass.const_expr(self.has_alpha):
                    for j in range(32):
                        if j <= tid and token + j < end:
                            log += cute.math.log2(
                                cutlass.Float32(alpha[(token + j) * self.heads + head])
                                + cutlass.Float32(1e-10),
                                fastmath=True,
                            )
                g[tid] = log
                bv = cutlass.Float32(0.0)
                if token + tid < end:
                    bv = cutlass.Float32(1.0)
                    if cutlass.const_expr(self.has_beta):
                        bv = cutlass.Float32(beta[(token + tid) * self.heads + head])
                b[tid] = bv
                logs[(slot * self.heads + head) * 32 + tid] = log
            cute.arch.sync_threads()
            kk = _acc(self.dtype, 32, 32)
            _gemm(sk, sk, kk, 32, 32, 128, self.use_ldmatrix)
            xy = _coords(self.dtype, 32, 32)
            for i in cutlass.range_constexpr(cute.size(kk)):
                r, c = xy[i]
                val = cutlass.Float32(0.0)
                if r > c:
                    val = kk[i] * b[r]
                inv[r, c] = cutlass.Float16(val)
            cute.arch.sync_threads()
            # One warp, one row/lane: exact triangular elimination in FP32
            # registers, rounded to FP16 at entry/exit, reusing FI's inverse helper.
            if tid < 32:
                CollectiveInverse().compute_diagonal_inverse_NxN(inv, tid, 32)
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr(8):
                linear = tid + j * 128
                r, c = linear // 32, linear % 32
                value = cutlass.Float32(0.0)
                if r >= c:
                    value = (
                        cutlass.Float32(inv[r, c])
                        * b[c]
                        * cute.math.exp2(g[r] - g[c], fastmath=True)
                    )
                matrix[(slot * self.heads + head) * 1024 + linear] = self.dtype(value)


class _Pipeline32(KeyedCompileMixin):
    def __init__(
        self,
        dtype,
        hq,
        hk,
        hv,
        heads,
        seqs,
        has_initial,
        checkpoints,
        use_ldmatrix,
        pad,
        cu_seqlens_dtype,
        checkpoint_cu_starts_dtype,
    ):
        self.dtype = dtype
        self.hq, self.hk, self.hv = hq, hk, hv
        self.heads, self.seqs = heads, seqs
        self.has_initial, self.checkpoints = has_initial, checkpoints
        self.use_ldmatrix, self.pad = use_ldmatrix, pad
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.checkpoint_cu_starts_dtype = checkpoint_cu_starts_dtype
        self.manual_cache_key(
            "dtype",
            "hq",
            "hk",
            "hv",
            "heads",
            "seqs",
            "has_initial",
            "checkpoints",
            "use_ldmatrix",
            "pad",
            "cu_seqlens_dtype",
            "checkpoint_cu_starts_dtype",
        )

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        cu,
        initial,
        output,
        final,
        matrix,
        logs,
        scale: cutlass.Float32,
        checkpoint,
        cp_starts,
        stream,
    ):
        dtype = self.dtype
        pad = self.pad

        @cute.struct
        class Storage:
            q: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * (128 + pad)], 128]
            k: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * (128 + pad)], 128]
            v: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * (64 + pad)], 128]
            a: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 32 * (32 + pad)], 128]
            g: cute.struct.MemRange[cutlass.Float32, 2 * 32]
            state: cute.struct.Align[cute.struct.MemRange[dtype, 64 * (128 + pad)], 128]
            delta: cute.struct.Align[cute.struct.MemRange[dtype, 32 * (64 + pad)], 128]
            update: cute.struct.Align[cute.struct.MemRange[dtype, 32 * (64 + pad)], 128]
            p: cute.struct.Align[cute.struct.MemRange[dtype, 32 * (32 + pad)], 128]
            barriers: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 8]

        self.storage = Storage
        self.kernel(
            q,
            k,
            v,
            cu,
            initial,
            output,
            final,
            matrix,
            logs,
            scale,
            checkpoint,
            cp_starts,
        ).launch(grid=(2, self.heads, self.seqs), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        cu,
        initial,
        output,
        final,
        matrix,
        logs,
        scale: cutlass.Float32,
        checkpoint,
        cp_starts,
    ):
        vb, head, seq = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 128
        start, end = cutlass.Int32(cu[seq]), cutlass.Int32(cu[seq + 1])
        count = (end - start + 31) // 32
        slot0 = start // 32 + seq
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        sq = sm.q.get_tensor(
            cute.make_layout(
                (32, 128, 2),
                stride=(128 + self.pad, 1, 32 * (128 + self.pad)),
            )
        )
        sk = sm.k.get_tensor(
            cute.make_layout(
                (32, 128, 2),
                stride=(128 + self.pad, 1, 32 * (128 + self.pad)),
            )
        )
        sv = sm.v.get_tensor(
            cute.make_layout(
                (32, 64, 2),
                stride=(64 + self.pad, 1, 32 * (64 + self.pad)),
            )
        )
        sa = sm.a.get_tensor(
            cute.make_layout(
                (32, 32, 2),
                stride=(32 + self.pad, 1, 32 * (32 + self.pad)),
            )
        )
        sg = sm.g.get_tensor(cute.make_layout((32, 2), stride=(1, 32)))
        sh = sm.state.get_tensor(
            cute.make_layout((64, 128), stride=(128 + self.pad, 1))
        )
        sd = sm.delta.get_tensor(cute.make_layout((32, 64), stride=(64 + self.pad, 1)))
        sn = sm.update.get_tensor(cute.make_layout((32, 64), stride=(64 + self.pad, 1)))
        sp = sm.p.get_tensor(cute.make_layout((32, 32), stride=(32 + self.pad, 1)))
        bars = sm.barriers.data_ptr()
        if tid == 0:
            for b in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(bars + b, 128)  # ready: producer WG
                cute.arch.mbarrier_init(bars + 2 + b, 384)  # free: three consumers
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if tid >= 384:
            # Producer runs independently up to two chunks ahead. Generic global
            # loads handle unaligned ragged starts/tails; ready is release/acquire.
            cute.arch.setmaxregister_decrease(32)
            for chunk in range(count):
                buf = chunk % 2
                if chunk >= 2:
                    cute.arch.mbarrier_wait(bars + 2 + buf, (chunk // 2 - 1) % 2)
                token = start + chunk * 32
                qh = head * self.hq // self.heads
                kh = head * self.hk // self.heads
                vh = head * self.hv // self.heads
                for r in cutlass.range_constexpr(32):
                    qv, kv = self.dtype(0.0), self.dtype(0.0)
                    if token + r < end:
                        qv = q[((token + r) * self.hq + qh) * 128 + lane]
                        kv = k[((token + r) * self.hk + kh) * 128 + lane]
                    sq[r, lane, buf], sk[r, lane, buf] = qv, kv
                for j in cutlass.range_constexpr(16):
                    idx = lane + j * 128
                    r, c = idx // 64, idx % 64
                    value = self.dtype(0.0)
                    if token + r < end:
                        value = v[((token + r) * self.hv + vh) * 128 + vb * 64 + c]
                    sv[r, c, buf] = value
                for j in cutlass.range_constexpr(8):
                    idx = lane + j * 128
                    sa[idx // 32, idx % 32, buf] = matrix[
                        ((slot0 + chunk) * self.heads + head) * 1024 + idx
                    ]
                if lane < 32:
                    sg[lane, buf] = logs[
                        ((slot0 + chunk) * self.heads + head) * 32 + lane
                    ]
                cute.arch.mbarrier_arrive(bars + buf)
        elif tid < 128:
            cute.arch.setmaxregister_increase(128)
            state = _acc(self.dtype, 64, 128)
            xy = _coords(self.dtype, 64, 128)
            if cutlass.const_expr(self.has_initial):
                for i in cutlass.range_constexpr(cute.size(state)):
                    r, c = xy[i]
                    state[i] = cutlass.Float32(
                        initial[
                            ((seq * self.heads + head) * 128 + vb * 64 + r) * 128 + c
                        ]
                    )
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                _store(state, sh, 64, 128)
                # 1: state snapshot published; all three math WGs participate.
                cute.arch.barrier(barrier_id=1, number_of_threads=384)
                decay = cute.math.exp2(sg[31, buf], fastmath=True)
                for i in cutlass.range_constexpr(cute.size(state)):
                    state[i] *= decay
                # 2: V' ready AND output has finished using shared snapshot/delta.
                cute.arch.barrier(barrier_id=2, number_of_threads=384)
                _gemm(
                    _transpose(sn),
                    _transpose(sk[None, None, buf]),
                    state,
                    64,
                    128,
                    32,
                    self.use_ldmatrix,
                )
                if cutlass.const_expr(self.checkpoints > 0):
                    completed = (chunk + 1) * 32
                    if completed <= end - start and completed % self.checkpoints == 0:
                        cp = (
                            cutlass.Int32(cp_starts[seq])
                            + completed // self.checkpoints
                            - 1
                        )
                        for i in cutlass.range_constexpr(cute.size(state)):
                            r, c = xy[i]
                            checkpoint[
                                ((cp * self.heads + head) * 128 + vb * 64 + r) * 128 + c
                            ] = state[i]
                cute.arch.mbarrier_arrive(bars + 2 + buf)
            # Also handles empty sequences: unchanged initial state, or zeros.
            for i in cutlass.range_constexpr(cute.size(state)):
                r, c = xy[i]
                final[((seq * self.heads + head) * 128 + vb * 64 + r) * 128 + c] = (
                    state[i]
                )
        elif tid < 256:
            cute.arch.setmaxregister_increase(128)
            xy = _coords(self.dtype, 32, 64)
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.barrier(barrier_id=1, number_of_threads=384)
                residual = _acc(self.dtype, 32, 64)
                _gemm(
                    sk[None, None, buf],
                    sh,
                    residual,
                    32,
                    64,
                    128,
                    self.use_ldmatrix,
                )
                for i in cutlass.range_constexpr(cute.size(residual)):
                    r, c = xy[i]
                    residual[i] = (
                        cutlass.Float32(sv[r, c, buf])
                        - cute.math.exp2(sg[r, buf], fastmath=True) * residual[i]
                    )
                _store(residual, sv[None, None, buf], 32, 64)
                cute.arch.barrier(barrier_id=4, number_of_threads=128)
                delta = _acc(self.dtype, 32, 64)
                _gemm(
                    sa[None, None, buf],
                    _transpose(sv[None, None, buf]),
                    delta,
                    32,
                    64,
                    32,
                    self.use_ldmatrix,
                )
                _store(delta, sd, 32, 64)
                for i in cutlass.range_constexpr(cute.size(delta)):
                    r, c = xy[i]
                    value = cutlass.Float32(0.0)
                    if start + chunk * 32 + r < end:
                        value = delta[i] * cute.math.exp2(
                            sg[31, buf] - sg[r, buf], fastmath=True
                        )
                    sn[r, c] = self.dtype(value)
                # 3: publish delta to output WG; 256 threads (V + O).
                cute.arch.barrier(barrier_id=3, number_of_threads=256)
                cute.arch.barrier(barrier_id=2, number_of_threads=384)
                cute.arch.mbarrier_arrive(bars + 2 + buf)
        else:
            cute.arch.setmaxregister_increase(128)
            pxy, oxy = _coords(self.dtype, 32, 32), _coords(self.dtype, 32, 64)
            for chunk in range(count):
                buf = chunk % 2
                cute.arch.mbarrier_wait(bars + buf, (chunk // 2) % 2)
                cute.arch.barrier(barrier_id=1, number_of_threads=384)
                p = _acc(self.dtype, 32, 32)
                _gemm(
                    sq[None, None, buf],
                    sk[None, None, buf],
                    p,
                    32,
                    32,
                    128,
                    self.use_ldmatrix,
                )
                for i in cutlass.range_constexpr(cute.size(p)):
                    r, c = pxy[i]
                    value = cutlass.Float32(0.0)
                    if r >= c:
                        value = (
                            p[i]
                            * scale
                            * cute.math.exp2(sg[r, buf] - sg[c, buf], fastmath=True)
                        )
                    sp[r, c] = self.dtype(value)
                out = _acc(self.dtype, 32, 64)
                _gemm(
                    sq[None, None, buf],
                    sh,
                    out,
                    32,
                    64,
                    128,
                    self.use_ldmatrix,
                )
                for i in cutlass.range_constexpr(cute.size(out)):
                    r, c = oxy[i]
                    out[i] *= scale * cute.math.exp2(sg[r, buf], fastmath=True)
                cute.arch.barrier(barrier_id=3, number_of_threads=256)
                _gemm(sp, _transpose(sd), out, 32, 64, 32, self.use_ldmatrix)
                for i in cutlass.range_constexpr(cute.size(out)):
                    r, c = oxy[i]
                    token = start + chunk * 32 + r
                    if token < end:
                        output[(token * self.heads + head) * 128 + vb * 64 + c] = (
                            self.dtype(out[i])
                        )
                cute.arch.barrier(barrier_id=2, number_of_threads=384)
                cute.arch.mbarrier_arrive(bars + 2 + buf)


def delta_rule_prefill_pipeline32(
    output,
    output_state,
    q,
    k,
    v,
    initial_state,
    alpha,
    beta,
    cu_seqlens,
    scale,
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
    *,
    workspace=None,
    mma_load="ldmatrix",
    shared_padding=True,
):
    """Enqueue precompute + 512-thread consumer on the current stream.

    D=128, FP16/BF16, contiguous token-major tensors. State is FP32 [B,H,V,K].
    Supports GVA/GQA, ragged and empty sequences, optional gates/initial state,
    and checkpoints at positive multiples of 32. ``alpha`` is multiplicative
    (not log decay), matching FI's low-level SM120 entrypoint.

    ``mma_load="generic", shared_padding=False`` reproduces the original
    pipeline32 data path. Default LDSM + eight-element row padding targets the
    bank conflicts observed in fi-pipeline32-both.ncu-repz. See
    benchmarks/README_gdn_sm120_pipeline32.md for correctness and performance
    evidence; speed depends on SKU and shape.

    Caller supplies valid monotonic cu_seqlens spanning [0,T]; metadata values
    remain on the GPU. Workspace is (matrix, logs), private to this invocation
    or stream, and must not alias inputs/outputs. Explicit preallocation supports
    graph capture after warmup. No fallback to the old fused kernel is used.
    """
    import cuda.bindings.driver as cuda_driver

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must have shape [tokens, heads, 128]")
    if mma_load not in ("generic", "ldmatrix") or not isinstance(shared_padding, bool):
        raise ValueError(
            "mma_load must be generic/ldmatrix and shared_padding must be bool"
        )
    use_ldmatrix, pad = mma_load == "ldmatrix", 8 if shared_padding else 0
    t, hq, d = q.shape
    hk, hv = k.shape[1], v.shape[1]
    heads, seqs = max(hq, hv), cu_seqlens.numel() - 1
    if (
        min(hq, hk, hv) <= 0
        or seqs <= 0
        or d != 128
        or k.shape != (t, hk, 128)
        or v.shape != (t, hv, 128)
    ):
        raise ValueError(
            "pipeline32 requires D=128, matching tokens and positive heads/sequences"
        )
    if not ((hq == hk and hv % hq == 0) or (hk == hv and hq % hv == 0)):
        raise ValueError("unsupported GVA/GQA head mapping")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("q/k/v must share FP16 or BF16 dtype")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("pipeline32 requires an SM120 CUDA device")
    expected_state = (seqs, heads, 128, 128)
    if (
        output.shape != (t, heads, 128)
        or output.dtype != q.dtype
        or output_state.shape != expected_state
        or output_state.dtype != torch.float32
    ):
        raise ValueError("invalid output/output_state shape or dtype")
    if initial_state is not None and (
        initial_state.shape != expected_state or initial_state.dtype != torch.float32
    ):
        raise ValueError("initial_state must be FP32 [sequences, heads, 128, 128]")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be an int32/int64 vector")
    for gate in (alpha, beta):
        if gate is not None and (
            gate.numel() != t * heads or gate.dtype != torch.float32
        ):
            raise ValueError("gates must contain tokens*output_heads FP32 values")
    interval = checkpoint_every_n_tokens
    if interval < 0 or interval % 32:
        raise ValueError(
            "checkpoint interval must be zero or a positive multiple of 32"
        )
    if interval:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError("checkpoint buffers and starts are required")
        if (
            state_checkpoints.ndim != 4
            or state_checkpoints.shape[1:] != (heads, 128, 128)
            or state_checkpoints.dtype != torch.float32
        ):
            raise ValueError("invalid checkpoint buffer")
        # Packed checkpoint starts must describe non-overlapping sequence ranges.
        # Reserve a conservative token-count bound without reading GPU metadata,
        # so validation also works during CUDA graph capture.
        required_checkpoints = (t + interval - 1) // interval
        if state_checkpoints.shape[0] < required_checkpoints:
            raise ValueError(
                f"checkpoint buffer needs at least {required_checkpoints} entries, "
                f"got {state_checkpoints.shape[0]}"
            )
        if (
            checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.numel() < seqs
            or checkpoint_cu_starts.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid checkpoint starts")
    tensors = (
        q,
        k,
        v,
        output,
        output_state,
        initial_state,
        alpha,
        beta,
        cu_seqlens,
        state_checkpoints,
        checkpoint_cu_starts,
    )
    if any(
        a is not None and (a.device != q.device or not a.is_contiguous())
        for a in tensors
    ):
        raise ValueError("all tensors must be contiguous and on the same CUDA device")
    matrix_shape, log_shape = workspace_shapes(t, seqs, heads)
    if workspace is None:
        workspace = (
            torch.empty(matrix_shape, device=q.device, dtype=q.dtype),
            torch.empty(log_shape, device=q.device, dtype=torch.float32),
        )
    if not isinstance(workspace, (tuple, list)) or len(workspace) != 2:
        raise ValueError("workspace must be (matrix, log_prefix)")
    for tensor, shape, dtype in zip(
        workspace, (matrix_shape, log_shape), (q.dtype, torch.float32), strict=False
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"workspace tensor must be contiguous {shape}, {dtype}, {q.device}"
            )
    # Avoid narrowing flat offsets in device arithmetic beyond signed int32.
    if any(a is not None and a.numel() >= 2**31 for a in (*tensors, *workspace)):
        raise ValueError(
            "pipeline32 currently requires int32-addressable element offsets"
        )
    if t == 0:
        if initial_state is None:
            output_state.zero_()
        else:
            output_state.copy_(initial_state)
        return

    def wrap(a):
        if a is None:
            return None
        return cute.runtime.from_dlpack(
            a.view(-1), assumed_align=a.element_size(), enable_tvm_ffi=True
        ).mark_layout_dynamic()

    with torch.cuda.device(q.device):
        dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
            q.dtype
        ]
        stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        mat, log = map(wrap, workspace)
        pre = _Precompute32(
            dtype,
            hk,
            heads,
            seqs,
            alpha is not None,
            beta is not None,
            use_ldmatrix,
            pad,
            cu_seqlens.dtype,
        )
        pre_args = (
            wrap(k),
            wrap(alpha),
            wrap(beta),
            wrap(cu_seqlens),
            mat,
            log,
            matrix_shape[0],
            stream,
        )
        opts = (cute.GPUArch("sm_120a"),)
        pre_compiled = cached_compile(pre, *pre_args, compile_options=opts)
        consumer = _Pipeline32(
            dtype,
            hq,
            hk,
            hv,
            heads,
            seqs,
            initial_state is not None,
            interval,
            use_ldmatrix,
            pad,
            cu_seqlens.dtype,
            checkpoint_cu_starts.dtype if interval else None,
        )
        args = (
            wrap(q),
            wrap(k),
            wrap(v),
            wrap(cu_seqlens),
            wrap(initial_state),
            wrap(output),
            wrap(output_state),
            mat,
            log,
            cutlass.Float32(scale),
            wrap(state_checkpoints) if interval else None,
            wrap(checkpoint_cu_starts) if interval else None,
            stream,
        )
        compiled = cached_compile(consumer, *args, compile_options=opts)
        pre_compiled(*pre_args)
        compiled(*args)
