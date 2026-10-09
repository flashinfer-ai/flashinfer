"""Prepared FP32 GDN recurrence over packed prefill sequences.

The arithmetic follows the CuTe decode recurrence, while state reads/exports
and empty sequences follow the prefill contract. A CTA retains its state rows
across a dynamic token loop; no token-dependent host launches or state staging
are required. Model activations and the sigmoid beta boundary remain BF16.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import warp_reduce
from b12x._lib.program_cache import register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_PROGRAMS: dict[tuple, object] = {}
register_program_cache(_PROGRAMS)


def _add(a: Float32, b: Float32) -> Float32:
    return a + b


def _type(dtype):
    return {
        torch.bfloat16: BFloat16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }[dtype]


def _pointer(tensor, dtype):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


def _fake(dtype):
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)


class _RecurrentPrefill:
    def __init__(self, caps):
        self.key_heads = caps.key_heads
        self.heads = caps.value_heads
        self.qk_l2norm = caps.qk_l2norm
        self.checkpoint_export = caps.checkpoint_export
        self.has_null = caps.null_state_index is not None
        self.null = 0 if caps.null_state_index is None else caps.null_state_index
        self.ctas = min(1536, caps.max_seqs * caps.value_heads * 4)

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        state_pool: cute.Pointer,
        output: cute.Pointer,
        cu: cute.Pointer,
        initial: cute.Pointer,
        final: cute.Pointer,
        checkpoint: cute.Pointer,
        offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        num_tokens: cute.Pointer,
        token_capacity: Int32,
        seq_capacity: Int32,
        q_stride: Int64,
        k_stride: Int64,
        v_stride: Int64,
        a_stride: Int64,
        b_stride: Int64,
        b_head_stride: Int64,
        state_stride: Int64,
        output_stride: Int64,
        final_stride: Int64,
        scale: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state_pool,
            output,
            cu,
            initial,
            final,
            checkpoint,
            offsets,
            num_seqs,
            num_tokens,
            token_capacity,
            seq_capacity,
            q_stride,
            k_stride,
            v_stride,
            a_stride,
            b_stride,
            b_head_stride,
            state_stride,
            output_stride,
            final_stride,
            scale,
            eps,
        ).launch(grid=(self.ctas, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def _is_null(self, slot: Int64):
        if cutlass.const_expr(self.has_null):
            return slot == Int64(self.null)
        return cutlass.Boolean(False)

    @cute.jit
    def _store(
        self,
        pool: cute.Pointer,
        state: cute.Tensor,
        slot: Int64,
        state_stride: Int64,
        head: Int32,
        row: Int32,
        key_lane: Int32,
    ):
        if not self._is_null(slot):
            base = slot * state_stride + head.to(Int64) * Int64(128 * 128)
            base += row.to(Int64) * Int64(128)
            for item in cutlass.range_constexpr(16):
                column = key_lane + Int32(item * 8)
                pool[base + column.to(Int64)] = state[item]

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        state_pool: cute.Pointer,
        output: cute.Pointer,
        cu: cute.Pointer,
        initial: cute.Pointer,
        final: cute.Pointer,
        checkpoint: cute.Pointer,
        offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        num_tokens: cute.Pointer,
        token_capacity: Int32,
        seq_capacity: Int32,
        q_stride: Int64,
        k_stride: Int64,
        v_stride: Int64,
        a_stride: Int64,
        b_stride: Int64,
        b_head_stride: Int64,
        state_stride: Int64,
        output_stride: Int64,
        final_stride: Int64,
        scale: Float32,
        eps: Float32,
    ):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        block, thread = Int32(block), Int32(thread)
        lane = thread % Int32(32)
        key_lane = thread % Int32(8)
        row_leader = lane - key_lane
        live_seqs = cutlass.max(Int32(0), cutlass.min(Int32(num_seqs[0]), seq_capacity))
        live_tokens = cutlass.max(
            Int32(0), cutlass.min(Int32(num_tokens[0]), token_capacity)
        )

        tail = block * Int32(256) + thread
        tail_elements = (token_capacity - live_tokens) * Int32(self.heads * 128)
        while tail < tail_elements:
            token = live_tokens + tail // Int32(self.heads * 128)
            column = tail % Int32(self.heads * 128)
            output[token.to(Int64) * output_stride + column.to(Int64)] = BFloat16(0)
            tail += Int32(self.ctas * 256)

        shared = cutlass.utils.SmemAllocator()
        q_shared = shared.allocate_tensor(
            Float32, cute.make_layout((128,)), byte_alignment=16
        )
        k_shared = shared.allocate_tensor(
            Float32, cute.make_layout((128,)), byte_alignment=16
        )
        params = shared.allocate_tensor(
            Float32, cute.make_layout((2,)), byte_alignment=8
        )
        work = block
        while work < live_seqs * Int32(self.heads * 4):
            value_tile = work % Int32(4)
            head = (work // Int32(4)) % Int32(self.heads)
            seq = work // Int32(self.heads * 4)
            key_head = head // Int32(3)
            row = value_tile * Int32(32) + thread // Int32(8)
            start, end = Int32(cu[seq]), Int32(cu[seq + Int32(1)])
            source = Int64(initial[seq])
            state = cute.make_rmem_tensor((16,), Float32)
            source_base = source * state_stride + head.to(Int64) * Int64(128 * 128)
            source_base += row.to(Int64) * Int64(128)
            for item in cutlass.range_constexpr(16):
                state[item] = Float32(0)
                if not self._is_null(source):
                    column = key_lane + Int32(item * 8)
                    state[item] = Float32(state_pool[source_base + column.to(Int64)])
            rate, bias = Float32(0), Float32(0)
            if thread == Int32(0):
                rate = cute.math.exp(Float32(A_log[head]), fastmath=False)
                bias = Float32(dt_bias[head])
            token = start
            while token < end:
                if thread < Int32(32):
                    q_sum, k_sum = Float32(0), Float32(0)
                    for item in cutlass.range_constexpr(4):
                        column = lane + Int32(item * 32)
                        hcol = key_head.to(Int64) * Int64(128) + column.to(Int64)
                        qv = Float32(q[token.to(Int64) * q_stride + hcol])
                        kv = Float32(k[token.to(Int64) * k_stride + hcol])
                        q_shared[column], k_shared[column] = qv, kv
                        q_sum += qv * qv
                        k_sum += kv * kv
                    q_sum = warp_reduce(q_sum, _add)
                    k_sum = warp_reduce(k_sum, _add)
                    q_norm, k_norm = Float32(1), Float32(1)
                    if cutlass.const_expr(self.qk_l2norm):
                        q_norm = cute.math.rsqrt(q_sum + eps, fastmath=False)
                        k_norm = cute.math.rsqrt(k_sum + eps, fastmath=False)
                    for item in cutlass.range_constexpr(4):
                        column = lane + Int32(item * 32)
                        q_shared[column] = Float32(q_shared[column]) * q_norm * scale
                        k_shared[column] = Float32(k_shared[column]) * k_norm
                if thread == Int32(0):
                    gate = (
                        Float32(a[token.to(Int64) * a_stride + head.to(Int64)]) + bias
                    )
                    softplus = gate
                    if gate <= Float32(20):
                        softplus = cute.math.log1p(
                            cute.math.exp(gate, fastmath=False), fastmath=False
                        )
                    params[0] = cute.math.exp(-rate * softplus, fastmath=False)
                    beta_arg = Float32(
                        b[token.to(Int64) * b_stride + head.to(Int64) * b_head_stride]
                    )
                    beta = cute.arch.rcp_approx(
                        Float32(1) + cute.math.exp(-beta_arg, fastmath=False)
                    )
                    params[1] = Float32(BFloat16(beta))
                cute.arch.sync_threads()
                value = Float32(0)
                if key_lane == Int32(0):
                    index = (
                        token.to(Int64) * v_stride
                        + head.to(Int64) * Int64(128)
                        + row.to(Int64)
                    )
                    value = Float32(v[index])
                value = Float32(cute.arch.shuffle_sync(value, row_leader))
                dot = Float32(0)
                decay, beta = Float32(params[0]), Float32(params[1])
                for item in cutlass.range_constexpr(16):
                    column = key_lane + Int32(item * 8)
                    state[item] = state[item] * decay
                    dot += state[item] * Float32(k_shared[column])
                dot = warp_reduce(dot, _add, 8)
                delta = (value - dot) * beta
                result = Float32(0)
                for item in cutlass.range_constexpr(16):
                    column = key_lane + Int32(item * 8)
                    state[item] = state[item] + delta * Float32(k_shared[column])
                    result += state[item] * Float32(q_shared[column])
                result = warp_reduce(result, _add, 8)
                if key_lane == Int32(0):
                    index = (
                        token.to(Int64) * output_stride
                        + head.to(Int64) * Int64(128)
                        + row.to(Int64)
                    )
                    output[index] = BFloat16(result)
                if cutlass.const_expr(self.checkpoint_export):
                    offset = Int32(offsets[seq])
                    if (offset > Int32(0)) & (token - start + Int32(1) == offset):
                        self._store(
                            state_pool,
                            state,
                            Int64(checkpoint[seq]),
                            state_stride,
                            head,
                            row,
                            key_lane,
                        )
                token += Int32(1)
                cute.arch.sync_threads()
            self._store(
                state_pool,
                state,
                Int64(final[seq.to(Int64) * final_stride]),
                state_stride,
                head,
                row,
                key_lane,
            )
            work += Int32(self.ctas)


def _key(binding):
    caps = binding._state.caps
    return (
        caps.max_seqs,
        caps.key_heads,
        caps.value_heads,
        caps.qk_l2norm,
        caps.checkpoint_export,
        caps.null_state_index,
        binding.A_log.dtype,
        binding.dt_bias.dtype,
        binding.initial_state_indices.dtype,
        binding.output.device.index,
    )


def compile_binding(binding):
    """Compile without reading device metadata or allocating CUDA storage."""
    key = _key(binding)
    if key in _PROGRAMS:
        return (_PROGRAMS[key],)
    kernel = _RecurrentPrefill(binding._state.caps)
    a_type, bias_type, index_type = (
        _type(binding.A_log.dtype),
        _type(binding.dt_bias.dtype),
        _type(binding.initial_state_indices.dtype),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        *(_fake(BFloat16) for _ in range(5)),
        _fake(a_type),
        _fake(bias_type),
        _fake(Float32),
        _fake(BFloat16),
        _fake(Int32),
        *(_fake(index_type) for _ in range(3)),
        *(_fake(Int32) for _ in range(3)),
        Int32(1),
        Int32(1),
        *(Int64(1) for _ in range(9)),
        Float32(1),
        Float32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_prefill.recurrent_fp32", 1, key
        ),
    )

    def launch(active, scale, eps):
        if _key(active) != key:
            raise ValueError("compiled FP32 GDN prefill does not match the binding")
        raw(
            *(
                _pointer(value, BFloat16)
                for value in (
                    active.q,
                    active.k,
                    active.v,
                    active.raw_g,
                    active.raw_beta,
                )
            ),
            _pointer(active.A_log, a_type),
            _pointer(active.dt_bias, bias_type),
            _pointer(active.recurrent_state, Float32),
            _pointer(active.output, BFloat16),
            _pointer(active.cu_seqlens, Int32),
            *(
                _pointer(value, index_type)
                for value in (
                    active.initial_state_indices,
                    active.final_state_indices,
                    active.checkpoint_state_indices,
                )
            ),
            *(
                _pointer(value, Int32)
                for value in (
                    active.checkpoint_offsets,
                    active.num_seqs,
                    active.num_tokens,
                )
            ),
            active.token_capacity,
            active.seq_capacity,
            active.q.stride(0),
            active.k.stride(0),
            active.v.stride(0),
            active.raw_g.stride(0),
            active.raw_beta.stride(0),
            active.raw_beta.stride(1),
            active.recurrent_state.stride(0),
            active.output.stride(0),
            active.final_state_indices.stride(0),
            float(scale),
            float(eps),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _PROGRAMS[key] = launch
    return (launch,)
