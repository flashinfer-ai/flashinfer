"""CuTe KDA verification and accepted-prefix state recovery.

Verification leaves the FP32 checkpoint unchanged and stores rank-one updates
plus the original BF16 key/gate inputs. Recovery applies only accepted updates,
in forward order, with the same FP32 recurrence. No state quantization is used.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import warp_reduce
from b12x._lib.utils import current_cuda_stream

from ._cute_kernels import _add, _fake_pointer, _numeric_type, _pointer


class _Verify:
    def __init__(self, query, block_v):
        self.heads = query.value_heads
        self.window = query.state_index_columns
        self.block_v = block_v
        self.null = query.null_state_index if query.null_state_index is not None else -1
        self.strides = query.kda_strides

    @cute.jit
    def __call__(
        self,
        mixed: cute.Pointer,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        starts: cute.Pointer,
        indices: cute.Pointer,
        live_seqs: cute.Pointer,
        output: cute.Pointer,
        corrections: cute.Pointer,
        kg: cute.Pointer,
        index_stride: Int64,
        state_stride: Int64,
        correction_stride: Int64,
        kg_stride: Int64,
        beta_row_stride: Int64,
        beta_head_stride: Int64,
        capacity: Int32,
        scale: Float32,
        lower: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mixed,
            gate,
            beta,
            a_log,
            bias,
            state,
            starts,
            indices,
            live_seqs,
            output,
            corrections,
            kg,
            index_stride,
            state_stride,
            correction_stride,
            kg_stride,
            beta_row_stride,
            beta_head_stride,
            capacity,
            scale,
            lower,
        ).launch(
            grid=(128 // self.block_v, capacity * self.heads, 1),
            block=(self.block_v * 8, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mixed: cute.Pointer,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        starts: cute.Pointer,
        indices: cute.Pointer,
        live_seqs: cute.Pointer,
        output: cute.Pointer,
        corrections: cute.Pointer,
        kg: cute.Pointer,
        index_stride: Int64,
        state_stride: Int64,
        correction_stride: Int64,
        kg_stride: Int64,
        beta_row_stride: Int64,
        beta_head_stride: Int64,
        capacity: Int32,
        scale: Float32,
        lower: Float32,
    ):
        tile, request_head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        request = Int32(request_head) // self.heads
        head = Int32(request_head) % self.heads
        tid = Int32(tid)
        lane = tid % 32
        key_lane = tid % 8
        row = Int32(tile) * self.block_v + tid // 8
        if request < live_seqs[0] and request < capacity:
            start = Int32(starts[request])
            end = Int32(starts[request + 1])
            source = Int64(indices[request.to(Int64) * index_stride])
            if source == self.null:
                for pos in cutlass.range(self.window):
                    if pos < end - start and key_lane == 0:
                        off = (
                            Int64(start + pos) * self.strides[9]
                            + head.to(Int64) * self.strides[10]
                            + row.to(Int64)
                        )
                        output[off] = BFloat16(0.0)
            else:
                base = (
                    source * state_stride
                    + head.to(Int64) * self.strides[7]
                    + row.to(Int64) * self.strides[8]
                )
                s = cute.make_rmem_tensor((16,), Float32)
                for j in cutlass.range_constexpr(16):
                    s[j] = Float32(state[base + Int64(key_lane + j * 8)])
                smem = cutlass.utils.SmemAllocator()
                vectors = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((3, 128), stride=(128, 1)),
                    byte_alignment=16,
                )
                a = cute.math.exp(Float32(a_log[head]), fastmath=False)
                for pos in cutlass.range(self.window):
                    if pos < end - start:
                        token = Int64(start + pos)
                        q_base = token * self.strides[0] + head.to(Int64) * 128
                        k_base = q_base + self.heads * 128
                        kg_base = (
                            source * kg_stride
                            + (head.to(Int64) * self.window + Int64(pos)) * 256
                        )
                        if tid < 32:
                            qsum = Float32(0.0)
                            ksum = Float32(0.0)
                            for j in cutlass.range_constexpr(4):
                                k = lane + j * 32
                                qv = Float32(mixed[q_base + k.to(Int64)])
                                kv = Float32(mixed[k_base + k.to(Int64)])
                                vectors[0, k] = qv
                                vectors[1, k] = kv
                                qsum += qv * qv
                                ksum += kv * kv
                                if tile == 0:
                                    kg[kg_base + k.to(Int64)] = BFloat16(kv)
                            qinv = cute.math.rsqrt(
                                warp_reduce(qsum, _add) + 1.0e-6, fastmath=False
                            )
                            kinv = cute.math.rsqrt(
                                warp_reduce(ksum, _add) + 1.0e-6, fastmath=False
                            )
                            for j in cutlass.range_constexpr(4):
                                k = lane + j * 32
                                vectors[0, k] = Float32(vectors[0, k]) * qinv * scale
                                vectors[1, k] = Float32(vectors[1, k]) * kinv
                        if tid < 128:
                            raw = Float32(
                                gate[
                                    token * self.strides[1]
                                    + head.to(Int64) * self.strides[2]
                                    + tid.to(Int64)
                                ]
                            )
                            biased = raw + Float32(
                                bias[head.to(Int64) * self.strides[5] + tid.to(Int64)]
                            )
                            activated = lower / (
                                1.0 + cute.math.exp(-a * biased, fastmath=False)
                            )
                            vectors[2, tid] = cute.math.exp(activated, fastmath=False)
                            if tile == 0:
                                kg[kg_base + 128 + tid.to(Int64)] = BFloat16(raw)
                        cute.arch.sync_threads()
                        dot = Float32(0.0)
                        for j in cutlass.range_constexpr(16):
                            k = key_lane + j * 8
                            s[j] = s[j] * Float32(vectors[2, k])
                            dot += s[j] * Float32(vectors[1, k])
                        dot = warp_reduce(dot, _add, 8)
                        value = Float32(
                            mixed[
                                token * self.strides[0]
                                + (2 * self.heads + head.to(Int64)) * 128
                                + row.to(Int64)
                            ]
                        )
                        b = Float32(
                            beta[
                                token * beta_row_stride
                                + head.to(Int64) * beta_head_stride
                            ]
                        )
                        delta = (value - dot) / (
                            1.0 + cute.math.exp(-b, fastmath=False)
                        )
                        decoded = Float32(0.0)
                        for j in cutlass.range_constexpr(16):
                            k = key_lane + j * 8
                            s[j] = s[j] + delta * Float32(vectors[1, k])
                            decoded += s[j] * Float32(vectors[0, k])
                        decoded = warp_reduce(decoded, _add, 8)
                        if key_lane == 0:
                            out = (
                                token * self.strides[9]
                                + head.to(Int64) * self.strides[10]
                                + row.to(Int64)
                            )
                            output[out] = BFloat16(decoded)
                            record = (
                                source * correction_stride
                                + (head.to(Int64) * self.window + Int64(pos)) * 128
                                + row.to(Int64)
                            )
                            corrections[record] = delta
                        cute.arch.sync_threads()


class _Commit:
    def __init__(self, query, block_v):
        self.heads = query.value_heads
        self.window = query.state_index_columns
        self.block_v = block_v
        self.null = query.null_state_index if query.null_state_index is not None else -1
        self.state_head_stride = query.kda_strides[7]
        self.state_row_stride = query.kda_strides[8]

    @cute.jit
    def __call__(
        self,
        state_addrs: cute.Pointer,
        state_strides: cute.Pointer,
        correction_addrs: cute.Pointer,
        correction_strides: cute.Pointer,
        kg_addrs: cute.Pointer,
        kg_strides: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        sources: cute.Pointer,
        lengths: cute.Pointer,
        destinations: cute.Pointer,
        boundaries: cute.Pointer,
        boundary_lengths: cute.Pointer,
        source_stride: Int64,
        batch: Int32,
        layers: Int32,
        lower: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            state_addrs,
            state_strides,
            correction_addrs,
            correction_strides,
            kg_addrs,
            kg_strides,
            a_log,
            bias,
            sources,
            lengths,
            destinations,
            boundaries,
            boundary_lengths,
            source_stride,
            lower,
        ).launch(
            grid=(128 // self.block_v, batch, layers * self.heads),
            block=(self.block_v * 8, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state_addrs: cute.Pointer,
        state_strides: cute.Pointer,
        correction_addrs: cute.Pointer,
        correction_strides: cute.Pointer,
        kg_addrs: cute.Pointer,
        kg_strides: cute.Pointer,
        a_log: cute.Pointer,
        bias: cute.Pointer,
        sources: cute.Pointer,
        lengths: cute.Pointer,
        destinations: cute.Pointer,
        boundaries: cute.Pointer,
        boundary_lengths: cute.Pointer,
        source_stride: Int64,
        lower: Float32,
    ):
        tile, request, layer_head = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        tid = Int32(tid)
        layer = Int32(layer_head) // self.heads
        head = Int32(layer_head) % self.heads
        source = Int64(sources[Int64(request) * source_stride])
        destination = Int64(destinations[request])
        length = Int32(lengths[request])
        if source != self.null and destination != self.null and length > 0:
            state = cute.make_ptr(
                Float32,
                Int64(state_addrs[layer]),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            corrections = cute.make_ptr(
                Float32,
                Int64(correction_addrs[layer]),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            kg = cute.make_ptr(
                BFloat16,
                Int64(kg_addrs[layer]),
                cute.AddressSpace.gmem,
                assumed_align=2,
            )
            state_stride = Int64(state_strides[layer])
            c_base = (
                source * Int64(correction_strides[layer])
                + head.to(Int64) * self.window * 128
            )
            kg_base = (
                source * Int64(kg_strides[layer]) + head.to(Int64) * self.window * 256
            )
            row = Int32(tile) * self.block_v + tid // 8
            key_lane = tid % 8
            head_row = (
                head.to(Int64) * self.state_head_stride
                + row.to(Int64) * self.state_row_stride
            )
            s = cute.make_rmem_tensor((16,), Float32)
            for j in cutlass.range_constexpr(16):
                s[j] = Float32(
                    state[source * state_stride + head_row + Int64(key_lane + j * 8)]
                )
            smem = cutlass.utils.SmemAllocator()
            vectors = smem.allocate_tensor(
                Float32, cute.make_layout((2, 128), stride=(128, 1)), byte_alignment=16
            )
            a = cute.math.exp(Float32(a_log[Int64(layer_head)]), fastmath=False)
            boundary = Int64(boundaries[request])
            boundary_length = Int32(boundary_lengths[request])
            for pos in cutlass.range(length):
                if tid < 32:
                    ksum = Float32(0.0)
                    for j in cutlass.range_constexpr(4):
                        k = tid + j * 32
                        kv = Float32(kg[kg_base + Int64(pos) * 256 + k.to(Int64)])
                        vectors[0, k] = kv
                        ksum += kv * kv
                    kinv = cute.math.rsqrt(
                        warp_reduce(ksum, _add) + 1.0e-6, fastmath=False
                    )
                    for j in cutlass.range_constexpr(4):
                        k = tid + j * 32
                        vectors[0, k] = Float32(vectors[0, k]) * kinv
                if tid < 128:
                    raw = Float32(kg[kg_base + Int64(pos) * 256 + 128 + tid.to(Int64)])
                    biased = raw + Float32(
                        bias[Int64(layer_head) * 128 + tid.to(Int64)]
                    )
                    activated = lower / (
                        1.0 + cute.math.exp(-a * biased, fastmath=False)
                    )
                    vectors[1, tid] = cute.math.exp(activated, fastmath=False)
                cute.arch.sync_threads()
                delta = Float32(corrections[c_base + Int64(pos) * 128 + row.to(Int64)])
                for j in cutlass.range_constexpr(16):
                    k = key_lane + j * 8
                    s[j] = s[j] * Float32(vectors[1, k]) + delta * Float32(
                        vectors[0, k]
                    )
                if boundary != self.null and pos + 1 == boundary_length:
                    for j in cutlass.range_constexpr(16):
                        state[
                            boundary * state_stride + head_row + Int64(key_lane + j * 8)
                        ] = s[j]
                cute.arch.sync_threads()
            for j in cutlass.range_constexpr(16):
                state[
                    destination * state_stride + head_row + Int64(key_lane + j * 8)
                ] = s[j]


def compile_recovery(query, config):
    """Prepare both executables without live pool pointers or sequence lengths."""
    a_type = _numeric_type(getattr(torch, query.a_log_dtype))
    bias_type = _numeric_type(getattr(torch, query.dt_bias_dtype))
    index_type = _numeric_type(getattr(torch, query.state_indices_dtype))
    # Pool size and pool stride cannot influence generated instructions. Strides
    # that belong to projections or static head geometry remain specialization.
    strides = (*query.kda_strides[:6], 0, *query.kda_strides[7:])
    key = (
        query.value_heads,
        query.state_index_columns,
        config.recurrent_block_v,
        query.null_state_index,
        query.a_log_dtype,
        query.dt_bias_dtype,
        query.state_indices_dtype,
        strides,
    )
    vtypes = (
        BFloat16,
        BFloat16,
        BFloat16,
        a_type,
        bias_type,
        Float32,
        Int32,
        index_type,
        Int32,
        BFloat16,
        Float32,
        BFloat16,
    )
    verifier = b12x_compile(
        _Verify(query, config.recurrent_block_v),
        *(_fake_pointer(t) for t in vtypes),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int32(1),
        Float32(1.0),
        Float32(-5.0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_decode.kda_verify_records", 1, key
        ),
    )
    ctypes = (Int64,) * 6 + (a_type, bias_type, index_type) + (Int32,) * 4
    commit = b12x_compile(
        _Commit(query, config.recurrent_block_v),
        *(_fake_pointer(t) for t in ctypes),
        Int64(1),
        Int32(1),
        Int32(1),
        Float32(-5.0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_decode.kda_commit_records", 1, key
        ),
    )

    def verify(binding, *, scale, lower_bound):
        tensors = (
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.A_log,
            binding.dt_bias,
            binding.recurrent_state,
            binding.query_start_loc,
            binding.state_indices,
            binding.num_seqs,
            binding.output,
            binding.correction_cache,
            binding.kg_cache,
        )
        verifier(
            *(_pointer(x, t) for x, t in zip(tensors, vtypes, strict=False)),
            binding.state_indices.stride(0),
            binding.recurrent_state.stride(0),
            binding.correction_cache.stride(0),
            binding.kg_cache.stride(0),
            binding.raw_beta.stride(0),
            binding.raw_beta.stride(1),
            binding.state_indices.shape[0],
            scale,
            lower_bound,
            current_cuda_stream(),
        )

    def recover(binding, *, lower_bound):
        commit(
            *(_pointer(x, t) for x, t in zip(binding.tensors, ctypes, strict=False)),
            binding.tensors[8].stride(0),
            binding.batch,
            binding.layers,
            lower_bound,
            current_cuda_stream(),
        )

    attach_programs(verify, verifier)
    attach_programs(recover, commit)
    return verify, recover
