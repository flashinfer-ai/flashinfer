"""CuTeDSL launches for Qwen GDN decode stages.

The public Qwen decode uses the CuTe recurrent stage. A CuTe gated RMSNorm
variant remains available for direct testing while the public output norm uses
Triton.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._impl import Binding


_KEY_DIM = 128
_VALUE_DIM = 128
_VALUE_ROWS_PER_CTA = 32
_KEY_LANES_PER_ROW = 8
_MAX_WORK_CTAS = 1536
_KEYS_PER_THREAD = _KEY_DIM // _KEY_LANES_PER_ROW
_THREADS = _VALUE_ROWS_PER_CTA * _KEY_LANES_PER_ROW
_WARPS_PER_CTA = _THREADS // 32
_NORM_THREADS = 256
_NORM_WARPS_PER_CTA = _NORM_THREADS // 32
_NORM_MAX_CTAS = 192
_GROUPED_VALUE_HEADS = 2

_KERNEL_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: set[tuple[object, ...]] = set()
_NORM_CACHE: dict[tuple[object, ...], Callable[[Binding, float], None]] = {}
_NORM_WARMED: set[tuple[object, ...]] = set()
register_program_cache(_KERNEL_CACHE)
register_program_cache(_NORM_CACHE)


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


def _numeric_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return BFloat16
    if dtype == torch.float32:
        return Float32
    if dtype == torch.int32:
        return Int32
    if dtype == torch.int64:
        return Int64
    raise TypeError(f"unsupported CuTe GDN dtype {dtype}")


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        16,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _pointer(
    tensor: torch.Tensor, dtype: type[cutlass.Numeric]
) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


class _GatedRmsNormKernel:
    def __init__(
        self,
        *,
        max_tokens: int,
        value_heads: int,
        sigmoid_gate: bool,
        norm_weight_type: type[cutlass.Numeric],
        norm_fp32: bool,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.value_heads = int(value_heads)
        self.sigmoid_gate = bool(sigmoid_gate)
        self.norm_weight_type = norm_weight_type
        self.norm_fp32 = bool(norm_fp32)
        total_rows = self.max_tokens * self.value_heads
        natural_grid = (
            total_rows + _NORM_WARPS_PER_CTA - 1
        ) // _NORM_WARPS_PER_CTA
        self.grid_ctas = min(_NORM_MAX_CTAS, natural_grid)

    @cute.jit
    def __call__(
        self,
        output: cute.Pointer,
        z: cute.Pointer,
        norm_weight: cute.Pointer,
        num_tokens: cute.Pointer,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            output,
            z,
            norm_weight,
            num_tokens,
            eps,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(_NORM_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        output: cute.Pointer,
        z: cute.Pointer,
        norm_weight: cute.Pointer,
        num_tokens: cute.Pointer,
        eps: Float32,
    ):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp = Int32(thread) // Int32(32)
        lane = Int32(thread) % Int32(32)
        token_value_head = (
            Int32(block) * Int32(_NORM_WARPS_PER_CTA) + warp
        )
        row_stride = Int32(grid) * Int32(_NORM_WARPS_PER_CTA)
        total_rows = Int32(self.max_tokens * self.value_heads)
        live_tokens = num_tokens[Int32(0)].to(Int32)
        bounded_tokens = cutlass.max(
            Int32(0), cutlass.min(live_tokens, Int32(self.max_tokens))
        )
        while token_value_head < total_rows:
            token = token_value_head // Int32(self.value_heads)
            value_head = token_value_head % Int32(self.value_heads)
            base = (
                token.to(Int64) * Int64(self.value_heads * _VALUE_DIM)
                + value_head.to(Int64) * Int64(_VALUE_DIM)
            )
            if token >= bounded_tokens:
                for lane_element in cutlass.range_constexpr(
                    _VALUE_DIM // 32
                ):
                    column = lane + Int32(lane_element * 32)
                    output[base + column.to(Int64)] = BFloat16(0.0)
            else:
                values = cute.make_rmem_tensor(
                    (_VALUE_DIM // 32,), Float32
                )
                square_sum = Float32(0.0)
                for lane_element in cutlass.range_constexpr(
                    _VALUE_DIM // 32
                ):
                    column = lane + Int32(lane_element * 32)
                    value = Float32(output[base + column.to(Int64)])
                    values[lane_element] = value
                    square_sum += value * value
                square_sum = warp_reduce(square_sum, _add)
                inv_rms = cute.math.rsqrt(
                    square_sum / Float32(_VALUE_DIM) + eps,
                    fastmath=False,
                )
                for lane_element in cutlass.range_constexpr(
                    _VALUE_DIM // 32
                ):
                    column = lane + Int32(lane_element * 32)
                    normalized = values[lane_element] * inv_rms
                    weighted = Float32(0.0)
                    if cutlass.const_expr(self.norm_fp32):
                        weighted = normalized * Float32(norm_weight[column])
                    else:
                        normalized_bf16 = BFloat16(normalized)
                        if cutlass.const_expr(
                            self.norm_weight_type is Float32
                        ):
                            weighted = Float32(normalized_bf16) * Float32(
                                norm_weight[column]
                            )
                        else:
                            weighted = Float32(
                                BFloat16(
                                    normalized_bf16
                                    * BFloat16(norm_weight[column])
                                )
                            )
                    gate_input = Float32(z[base + column.to(Int64)])
                    gate = cute.arch.rcp_approx(
                        Float32(1.0)
                        + cute.math.exp(-gate_input, fastmath=False)
                    )
                    if cutlass.const_expr(not self.sigmoid_gate):
                        gate *= gate_input
                    output[base + column.to(Int64)] = BFloat16(
                        weighted * gate
                    )
            token_value_head += row_stride


class _PackedRecurrentQwenKernel:
    def __init__(
        self,
        *,
        max_seqs: int,
        state_index_columns: int,
        key_heads: int,
        value_heads: int,
        qk_l2norm: bool,
        null_state_index: int | None,
        state_type: type[cutlass.Numeric],
    ) -> None:
        self.max_seqs = int(max_seqs)
        self.state_index_columns = int(state_index_columns)
        self.key_heads = int(key_heads)
        self.value_heads = int(value_heads)
        self.head_ratio = self.value_heads // self.key_heads
        if self.head_ratio != 3:
            raise ValueError("CuTe Qwen GDN requires three value heads per key head")
        self.work_ctas = min(
            _MAX_WORK_CTAS,
            self.max_seqs * self.value_heads * (_VALUE_DIM // _VALUE_ROWS_PER_CTA),
        )
        self.packed_qkv_width = (
            2 * self.key_heads * _KEY_DIM + self.value_heads * _VALUE_DIM
        )
        self.qk_l2norm = bool(qk_l2norm)
        self.has_null_state_index = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)
        self.state_type = state_type

    @cute.jit
    def __call__(
        self,
        mixed_qkv: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        recurrent_state: cute.Pointer,
        query_start_loc: cute.Pointer,
        num_accepted_tokens: cute.Pointer,
        state_indices: cute.Pointer,
        num_seqs: cute.Pointer,
        output: cute.Pointer,
        state_slot_stride: Int64,
        mixed_token_stride: Int64,
        a_token_stride: Int64,
        b_token_stride: Int64,
        output_token_stride: Int64,
        state_index_request_stride: Int64,
        state_index_column_stride: Int64,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            recurrent_state,
            query_start_loc,
            num_accepted_tokens,
            state_indices,
            num_seqs,
            output,
            state_slot_stride,
            mixed_token_stride,
            a_token_stride,
            b_token_stride,
            output_token_stride,
            state_index_request_stride,
            state_index_column_stride,
            scale,
        ).launch(
            grid=(self.work_ctas, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _zero_request(
        self,
        output: cute.Pointer,
        start: Int32,
        end: Int32,
        value_head: Int32,
        value_row: Int32,
        output_token_stride: Int64,
    ):
        for relative_token in cutlass.range_constexpr(self.state_index_columns):
            if Int32(relative_token) < end - start:
                token = start + Int32(relative_token)
                output_offset = (
                    token.to(Int64) * output_token_stride
                    + value_head.to(Int64) * Int64(_VALUE_DIM)
                    + value_row.to(Int64)
                )
                output[output_offset] = BFloat16(0.0)

    @cute.jit
    def _run_request(
        self,
        mixed_qkv: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        recurrent_state: cute.Pointer,
        state_indices: cute.Pointer,
        output: cute.Pointer,
        request: Int32,
        value_head: Int32,
        key_head: Int32,
        value_row: Int32,
        start: Int32,
        end: Int32,
        source_index: Int64,
        state_slot_stride: Int64,
        mixed_token_stride: Int64,
        a_token_stride: Int64,
        b_token_stride: Int64,
        output_token_stride: Int64,
        state_index_request_stride: Int64,
        state_index_column_stride: Int64,
        scale: Float32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        lane = Int32(cute.arch.lane_idx())
        key_lane = thread % Int32(_KEY_LANES_PER_ROW)
        row_leader = lane - key_lane

        # All pool-scaled products are widened before multiplication. A valid
        # recycled slot can place the first live state element past 2^31.
        state_base = (
            source_index * state_slot_stride
            + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
            + value_row.to(Int64) * Int64(_KEY_DIM)
        )
        state = cute.make_rmem_tensor((_KEYS_PER_THREAD,), Float32)
        for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
            key_column = key_lane + Int32(key_element * _KEY_LANES_PER_ROW)
            state[key_element] = Float32(
                recurrent_state[state_base + key_column.to(Int64)]
            )

        allocator = cutlass.utils.SmemAllocator()
        shared_q = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_KEY_DIM,), stride=(1,)),
            byte_alignment=16,
        )
        shared_k = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_KEY_DIM,), stride=(1,)),
            byte_alignment=16,
        )
        shared_params = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((2,), stride=(1,)),
            byte_alignment=8,
        )
        a_log_scale = Float32(0.0)
        dt_bias_value = Float32(0.0)
        if thread == Int32(0):
            a_log_scale = cute.math.exp(Float32(A_log[value_head]), fastmath=False)
            dt_bias_value = Float32(dt_bias[value_head])

        for relative_token in cutlass.range_constexpr(self.state_index_columns):
            if Int32(relative_token) < end - start:
                token = start + Int32(relative_token)
                token_base = token.to(Int64) * mixed_token_stride
                q_base = token_base + key_head.to(Int64) * Int64(_KEY_DIM)
                k_base = (
                    token_base
                    + Int64(self.key_heads * _KEY_DIM)
                    + key_head.to(Int64) * Int64(_KEY_DIM)
                )

                if thread < Int32(32):
                    q_square_sum = Float32(0.0)
                    k_square_sum = Float32(0.0)
                    for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                        key_column = lane + Int32(lane_element * 32)
                        q_value = Float32(mixed_qkv[q_base + key_column.to(Int64)])
                        k_value = Float32(mixed_qkv[k_base + key_column.to(Int64)])
                        shared_q[key_column] = q_value
                        shared_k[key_column] = k_value
                        if cutlass.const_expr(self.qk_l2norm):
                            q_square_sum += q_value * q_value
                            k_square_sum += k_value * k_value
                    if cutlass.const_expr(self.qk_l2norm):
                        q_square_sum = warp_reduce(q_square_sum, _add)
                        k_square_sum = warp_reduce(k_square_sum, _add)
                        q_inv_norm = cute.math.rsqrt(
                            q_square_sum + Float32(1.0e-6), fastmath=False
                        )
                        k_inv_norm = cute.math.rsqrt(
                            k_square_sum + Float32(1.0e-6), fastmath=False
                        )
                        for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                            key_column = lane + Int32(lane_element * 32)
                            shared_q[key_column] = (
                                Float32(shared_q[key_column]) * q_inv_norm * scale
                            )
                            shared_k[key_column] = (
                                Float32(shared_k[key_column]) * k_inv_norm
                            )
                    else:
                        for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                            key_column = lane + Int32(lane_element * 32)
                            shared_q[key_column] = Float32(shared_q[key_column]) * scale

                if thread == Int32(0):
                    a_offset = token.to(Int64) * a_token_stride + value_head.to(Int64)
                    b_offset = token.to(Int64) * b_token_stride + value_head.to(Int64)
                    a_value = Float32(a[a_offset])
                    b_value = Float32(b[b_offset])
                    softplus_input = a_value + dt_bias_value
                    softplus = softplus_input
                    if softplus_input <= Float32(20.0):
                        softplus = cute.math.log1p(
                            cute.math.exp(softplus_input, fastmath=False),
                            fastmath=False,
                        )
                    shared_params[Int32(0)] = cute.math.exp(
                        -a_log_scale * softplus,
                        fastmath=False,
                    )
                    beta = cute.arch.rcp_approx(
                        Float32(1.0) + cute.math.exp(-b_value, fastmath=False)
                    )
                    # Qwen rounds beta through BF16 before the state update.
                    shared_params[Int32(1)] = Float32(BFloat16(beta))
                cute.arch.sync_threads()

                decay = Float32(shared_params[Int32(0)])
                beta = Float32(shared_params[Int32(1)])
                value = Float32(0.0)
                if key_lane == Int32(0):
                    value_offset = (
                        token_base
                        + Int64(2 * self.key_heads * _KEY_DIM)
                        + value_head.to(Int64) * Int64(_VALUE_DIM)
                        + value_row.to(Int64)
                    )
                    value = Float32(mixed_qkv[value_offset])
                value = Float32(cute.arch.shuffle_sync(value, row_leader))

                state_dot_k = Float32(0.0)
                for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                    local_key_column = key_lane + Int32(
                        key_element * _KEY_LANES_PER_ROW
                    )
                    state_value = state[key_element] * decay
                    state[key_element] = state_value
                    state_dot_k += state_value * Float32(shared_k[local_key_column])
                state_dot_k = warp_reduce(state_dot_k, _add, _KEY_LANES_PER_ROW)
                delta = (value - state_dot_k) * beta
                decoded = Float32(0.0)
                for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                    local_key_column = key_lane + Int32(
                        key_element * _KEY_LANES_PER_ROW
                    )
                    state_value = state[key_element] + delta * Float32(
                        shared_k[local_key_column]
                    )
                    state[key_element] = state_value
                    decoded += state_value * Float32(shared_q[local_key_column])
                decoded = warp_reduce(decoded, _add, _KEY_LANES_PER_ROW)

                if key_lane == Int32(0):
                    output_offset = (
                        token.to(Int64) * output_token_stride
                        + value_head.to(Int64) * Int64(_VALUE_DIM)
                        + value_row.to(Int64)
                    )
                    output[output_offset] = BFloat16(decoded)
                destination_index_offset = (
                    request.to(Int64) * state_index_request_stride
                    + Int64(relative_token) * state_index_column_stride
                )
                destination_index = state_indices[destination_index_offset].to(Int64)
                if cutlass.const_expr(self.has_null_state_index):
                    if destination_index != Int64(self.null_state_index):
                        destination_base = (
                            destination_index * state_slot_stride
                            + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
                            + value_row.to(Int64) * Int64(_KEY_DIM)
                        )
                        for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                            local_key_column = key_lane + Int32(
                                key_element * _KEY_LANES_PER_ROW
                            )
                            recurrent_state[
                                destination_base + local_key_column.to(Int64)
                            ] = self.state_type(state[key_element])
                else:
                    destination_base = (
                        destination_index * state_slot_stride
                        + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
                        + value_row.to(Int64) * Int64(_KEY_DIM)
                    )
                    for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                        local_key_column = key_lane + Int32(
                            key_element * _KEY_LANES_PER_ROW
                        )
                        recurrent_state[
                            destination_base + local_key_column.to(Int64)
                        ] = self.state_type(state[key_element])
                if Int32(relative_token + 1) < end - start:
                    cute.arch.sync_threads()

    @cute.jit
    def _run_grouped_request(
        self,
        mixed_qkv: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        recurrent_state: cute.Pointer,
        state_indices: cute.Pointer,
        output: cute.Pointer,
        request: Int32,
        key_head: Int32,
        value_row: Int32,
        start: Int32,
        end: Int32,
        source_index: Int64,
        state_slot_stride: Int64,
        mixed_token_stride: Int64,
        a_token_stride: Int64,
        b_token_stride: Int64,
        output_token_stride: Int64,
        state_index_request_stride: Int64,
        state_index_column_stride: Int64,
        scale: Float32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        lane = Int32(cute.arch.lane_idx())
        key_lane = thread % Int32(_KEY_LANES_PER_ROW)
        row_leader = lane - key_lane

        allocator = cutlass.utils.SmemAllocator()
        shared_q = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (self.state_index_columns * _KEY_DIM,), stride=(1,)
            ),
            byte_alignment=16,
        )
        shared_k = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (self.state_index_columns * _KEY_DIM,), stride=(1,)
            ),
            byte_alignment=16,
        )
        shared_params = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (_GROUPED_VALUE_HEADS * self.state_index_columns * 2,),
                stride=(1,),
            ),
            byte_alignment=16,
        )

        a_log_scales = cute.make_rmem_tensor((_GROUPED_VALUE_HEADS,), Float32)
        dt_bias_values = cute.make_rmem_tensor((_GROUPED_VALUE_HEADS,), Float32)
        for value_head_offset in cutlass.range_constexpr(_GROUPED_VALUE_HEADS):
            a_log_scales[value_head_offset] = Float32(0.0)
            dt_bias_values[value_head_offset] = Float32(0.0)
        if thread == Int32(0):
            for value_head_offset in cutlass.range_constexpr(_GROUPED_VALUE_HEADS):
                value_head = key_head * Int32(self.head_ratio) + Int32(
                    value_head_offset
                )
                a_log_scales[value_head_offset] = cute.math.exp(
                    Float32(A_log[value_head]), fastmath=False
                )
                dt_bias_values[value_head_offset] = Float32(dt_bias[value_head])

        for relative_token in cutlass.range_constexpr(self.state_index_columns):
            if Int32(relative_token) < end - start:
                token = start + Int32(relative_token)
                token_base = token.to(Int64) * mixed_token_stride
                q_base = token_base + key_head.to(Int64) * Int64(_KEY_DIM)
                k_base = (
                    token_base
                    + Int64(self.key_heads * _KEY_DIM)
                    + key_head.to(Int64) * Int64(_KEY_DIM)
                )
                shared_token_base = Int32(relative_token * _KEY_DIM)

                if thread < Int32(32):
                    q_square_sum = Float32(0.0)
                    k_square_sum = Float32(0.0)
                    for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                        key_column = lane + Int32(lane_element * 32)
                        q_value = Float32(mixed_qkv[q_base + key_column.to(Int64)])
                        k_value = Float32(mixed_qkv[k_base + key_column.to(Int64)])
                        shared_offset = shared_token_base + key_column
                        shared_q[shared_offset] = q_value
                        shared_k[shared_offset] = k_value
                        if cutlass.const_expr(self.qk_l2norm):
                            q_square_sum += q_value * q_value
                            k_square_sum += k_value * k_value
                    if cutlass.const_expr(self.qk_l2norm):
                        q_square_sum = warp_reduce(q_square_sum, _add)
                        k_square_sum = warp_reduce(k_square_sum, _add)
                        q_inv_norm = cute.math.rsqrt(
                            q_square_sum + Float32(1.0e-6), fastmath=False
                        )
                        k_inv_norm = cute.math.rsqrt(
                            k_square_sum + Float32(1.0e-6), fastmath=False
                        )
                        for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                            key_column = lane + Int32(lane_element * 32)
                            shared_offset = shared_token_base + key_column
                            shared_q[shared_offset] = (
                                Float32(shared_q[shared_offset]) * q_inv_norm * scale
                            )
                            shared_k[shared_offset] = (
                                Float32(shared_k[shared_offset]) * k_inv_norm
                            )
                    else:
                        for lane_element in cutlass.range_constexpr(_KEY_DIM // 32):
                            key_column = lane + Int32(lane_element * 32)
                            shared_offset = shared_token_base + key_column
                            shared_q[shared_offset] = (
                                Float32(shared_q[shared_offset]) * scale
                            )

                if thread == Int32(0):
                    for value_head_offset in cutlass.range_constexpr(
                        _GROUPED_VALUE_HEADS
                    ):
                        value_head = key_head * Int32(self.head_ratio) + Int32(
                            value_head_offset
                        )
                        a_offset = token.to(Int64) * a_token_stride + value_head.to(
                            Int64
                        )
                        b_offset = token.to(Int64) * b_token_stride + value_head.to(
                            Int64
                        )
                        softplus_input = (
                            Float32(a[a_offset]) + dt_bias_values[value_head_offset]
                        )
                        softplus = softplus_input
                        if softplus_input <= Float32(20.0):
                            softplus = cute.math.log1p(
                                cute.math.exp(softplus_input, fastmath=False),
                                fastmath=False,
                            )
                        param_offset = Int32(
                            (
                                value_head_offset * self.state_index_columns
                                + relative_token
                            )
                            * 2
                        )
                        shared_params[param_offset] = cute.math.exp(
                            -a_log_scales[value_head_offset] * softplus,
                            fastmath=False,
                        )
                        beta = cute.arch.rcp_approx(
                            Float32(1.0)
                            + cute.math.exp(-Float32(b[b_offset]), fastmath=False)
                        )
                        shared_params[param_offset + Int32(1)] = Float32(BFloat16(beta))

        cute.arch.sync_threads()

        for value_head_offset in cutlass.range(Int32(_GROUPED_VALUE_HEADS), unroll=1):
            value_head = key_head * Int32(self.head_ratio) + value_head_offset
            # All pool-scaled products are widened before multiplication.
            state_base = (
                source_index * state_slot_stride
                + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
                + value_row.to(Int64) * Int64(_KEY_DIM)
            )
            state = cute.make_rmem_tensor((_KEYS_PER_THREAD,), Float32)
            for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                key_column = key_lane + Int32(key_element * _KEY_LANES_PER_ROW)
                state[key_element] = Float32(
                    recurrent_state[state_base + key_column.to(Int64)]
                )

            for relative_token in cutlass.range_constexpr(self.state_index_columns):
                if Int32(relative_token) < end - start:
                    token = start + Int32(relative_token)
                    token_base = token.to(Int64) * mixed_token_stride
                    param_offset = (
                        value_head_offset * Int32(self.state_index_columns)
                        + Int32(relative_token)
                    ) * Int32(2)
                    decay = Float32(shared_params[param_offset])
                    beta = Float32(shared_params[param_offset + Int32(1)])
                    value = Float32(0.0)
                    if key_lane == Int32(0):
                        value_offset = (
                            token_base
                            + Int64(2 * self.key_heads * _KEY_DIM)
                            + value_head.to(Int64) * Int64(_VALUE_DIM)
                            + value_row.to(Int64)
                        )
                        value = Float32(mixed_qkv[value_offset])
                    value = Float32(cute.arch.shuffle_sync(value, row_leader))

                    shared_token_base = Int32(relative_token * _KEY_DIM)
                    state_dot_k = Float32(0.0)
                    for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                        key_column = key_lane + Int32(key_element * _KEY_LANES_PER_ROW)
                        state_value = state[key_element] * decay
                        state[key_element] = state_value
                        state_dot_k += state_value * Float32(
                            shared_k[shared_token_base + key_column]
                        )
                    state_dot_k = warp_reduce(state_dot_k, _add, _KEY_LANES_PER_ROW)
                    delta = (value - state_dot_k) * beta
                    decoded = Float32(0.0)
                    for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                        key_column = key_lane + Int32(key_element * _KEY_LANES_PER_ROW)
                        state_value = state[key_element] + delta * Float32(
                            shared_k[shared_token_base + key_column]
                        )
                        state[key_element] = state_value
                        decoded += state_value * Float32(
                            shared_q[shared_token_base + key_column]
                        )
                    decoded = warp_reduce(decoded, _add, _KEY_LANES_PER_ROW)

                    if key_lane == Int32(0):
                        output_offset = (
                            token.to(Int64) * output_token_stride
                            + value_head.to(Int64) * Int64(_VALUE_DIM)
                            + value_row.to(Int64)
                        )
                        output[output_offset] = BFloat16(decoded)
                    destination_index_offset = (
                        request.to(Int64) * state_index_request_stride
                        + Int64(relative_token) * state_index_column_stride
                    )
                    destination_index = state_indices[destination_index_offset].to(
                        Int64
                    )
                    if cutlass.const_expr(self.has_null_state_index):
                        if destination_index != Int64(self.null_state_index):
                            destination_base = (
                                destination_index * state_slot_stride
                                + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
                                + value_row.to(Int64) * Int64(_KEY_DIM)
                            )
                            for key_element in cutlass.range_constexpr(
                                _KEYS_PER_THREAD
                            ):
                                key_column = key_lane + Int32(
                                    key_element * _KEY_LANES_PER_ROW
                                )
                                recurrent_state[
                                    destination_base + key_column.to(Int64)
                                ] = self.state_type(state[key_element])
                    else:
                        destination_base = (
                            destination_index * state_slot_stride
                            + value_head.to(Int64) * Int64(_VALUE_DIM * _KEY_DIM)
                            + value_row.to(Int64) * Int64(_KEY_DIM)
                        )
                        for key_element in cutlass.range_constexpr(_KEYS_PER_THREAD):
                            key_column = key_lane + Int32(
                                key_element * _KEY_LANES_PER_ROW
                            )
                            recurrent_state[destination_base + key_column.to(Int64)] = (
                                self.state_type(state[key_element])
                            )

    @cute.kernel
    def kernel(
        self,
        mixed_qkv: cute.Pointer,
        a: cute.Pointer,
        b: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        recurrent_state: cute.Pointer,
        query_start_loc: cute.Pointer,
        num_accepted_tokens: cute.Pointer,
        state_indices: cute.Pointer,
        num_seqs: cute.Pointer,
        output: cute.Pointer,
        state_slot_stride: Int64,
        mixed_token_stride: Int64,
        a_token_stride: Int64,
        b_token_stride: Int64,
        output_token_stride: Int64,
        state_index_request_stride: Int64,
        state_index_column_stride: Int64,
        scale: Float32,
    ):
        work_block, _, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        work_block = Int32(work_block)

        live_seqs = num_seqs[Int32(0)].to(Int32)
        bounded_seqs = cutlass.max(
            Int32(0), cutlass.min(live_seqs, Int32(self.max_seqs))
        )
        value_tiles = Int32(_VALUE_DIM // _VALUE_ROWS_PER_CTA)
        total_work = bounded_seqs * Int32(self.value_heads) * value_tiles
        work_iterations = cutlass.max(
            Int32(0),
            (total_work - work_block + Int32(self.work_ctas - 1))
            // Int32(self.work_ctas),
        )
        for work_iteration in cutlass.range(work_iterations, unroll=1):
            work = work_block + work_iteration * Int32(self.work_ctas)
            value_tile = work % value_tiles
            request_value_head = work // value_tiles
            value_head = request_value_head % Int32(self.value_heads)
            request = request_value_head // Int32(self.value_heads)
            key_head = value_head // Int32(self.head_ratio)
            value_row = value_tile * Int32(_VALUE_ROWS_PER_CTA) + Int32(
                lane
            ) // Int32(_KEY_LANES_PER_ROW)
            start = query_start_loc[request].to(Int32)
            end = query_start_loc[request + Int32(1)].to(Int32)
            if end > start:
                accepted_column = num_accepted_tokens[request].to(Int32) - Int32(1)
                source_index_offset = (
                    request.to(Int64) * state_index_request_stride
                    + accepted_column.to(Int64) * state_index_column_stride
                )
                source_index = state_indices[source_index_offset].to(Int64)
                grouped_heads = (end - start > Int32(1)) & (bounded_seqs > Int32(1))
                value_head_in_group = value_head % Int32(self.head_ratio)
                group_leader = value_head_in_group == Int32(0)
                ungrouped_tail = value_head_in_group == Int32(_GROUPED_VALUE_HEADS)
                if cutlass.const_expr(self.has_null_state_index):
                    if source_index == Int64(self.null_state_index):
                        if Int32(lane) % Int32(_KEY_LANES_PER_ROW) == Int32(0):
                            if grouped_heads:
                                if group_leader:
                                    for (
                                        value_head_offset
                                    ) in cutlass.range_constexpr(
                                        _GROUPED_VALUE_HEADS
                                    ):
                                        self._zero_request(
                                            output,
                                            start,
                                            end,
                                            value_head + Int32(value_head_offset),
                                            value_row,
                                            output_token_stride,
                                        )
                                elif ungrouped_tail:
                                    self._zero_request(
                                        output,
                                        start,
                                        end,
                                        value_head,
                                        value_row,
                                        output_token_stride,
                                    )
                            else:
                                self._zero_request(
                                    output,
                                    start,
                                    end,
                                    value_head,
                                    value_row,
                                    output_token_stride,
                                )
                    else:
                        if grouped_heads:
                            if group_leader:
                                self._run_grouped_request(
                                    mixed_qkv,
                                    a,
                                    b,
                                    A_log,
                                    dt_bias,
                                    recurrent_state,
                                    state_indices,
                                    output,
                                    request,
                                    key_head,
                                    value_row,
                                    start,
                                    end,
                                    source_index,
                                    state_slot_stride,
                                    mixed_token_stride,
                                    a_token_stride,
                                    b_token_stride,
                                    output_token_stride,
                                    state_index_request_stride,
                                    state_index_column_stride,
                                    scale,
                                )
                            elif ungrouped_tail:
                                self._run_request(
                                    mixed_qkv,
                                    a,
                                    b,
                                    A_log,
                                    dt_bias,
                                    recurrent_state,
                                    state_indices,
                                    output,
                                    request,
                                    value_head,
                                    key_head,
                                    value_row,
                                    start,
                                    end,
                                    source_index,
                                    state_slot_stride,
                                    mixed_token_stride,
                                    a_token_stride,
                                    b_token_stride,
                                    output_token_stride,
                                    state_index_request_stride,
                                    state_index_column_stride,
                                    scale,
                                )
                        else:
                            self._run_request(
                                mixed_qkv,
                                a,
                                b,
                                A_log,
                                dt_bias,
                                recurrent_state,
                                state_indices,
                                output,
                                request,
                                value_head,
                                key_head,
                                value_row,
                                start,
                                end,
                                source_index,
                                state_slot_stride,
                                mixed_token_stride,
                                a_token_stride,
                                b_token_stride,
                                output_token_stride,
                                state_index_request_stride,
                                state_index_column_stride,
                                scale,
                            )
                else:
                    if grouped_heads:
                        if group_leader:
                            self._run_grouped_request(
                                mixed_qkv,
                                a,
                                b,
                                A_log,
                                dt_bias,
                                recurrent_state,
                                state_indices,
                                output,
                                request,
                                key_head,
                                value_row,
                                start,
                                end,
                                source_index,
                                state_slot_stride,
                                mixed_token_stride,
                                a_token_stride,
                                b_token_stride,
                                output_token_stride,
                                state_index_request_stride,
                                state_index_column_stride,
                                scale,
                            )
                        elif ungrouped_tail:
                            self._run_request(
                                mixed_qkv,
                                a,
                                b,
                                A_log,
                                dt_bias,
                                recurrent_state,
                                state_indices,
                                output,
                                request,
                                value_head,
                                key_head,
                                value_row,
                                start,
                                end,
                                source_index,
                                state_slot_stride,
                                mixed_token_stride,
                                a_token_stride,
                                b_token_stride,
                                output_token_stride,
                                state_index_request_stride,
                                state_index_column_stride,
                                scale,
                            )
                    else:
                        self._run_request(
                            mixed_qkv,
                            a,
                            b,
                            A_log,
                            dt_bias,
                            recurrent_state,
                            state_indices,
                            output,
                            request,
                            value_head,
                            key_head,
                            value_row,
                            start,
                            end,
                            source_index,
                            state_slot_stride,
                            mixed_token_stride,
                            a_token_stride,
                            b_token_stride,
                            output_token_stride,
                            state_index_request_stride,
                            state_index_column_stride,
                            scale,
                        )


def _binding_key(binding: Binding) -> tuple[object, ...]:
    if not isinstance(binding, Binding):
        raise TypeError(f"binding must be GDN Binding, got {type(binding)!r}")
    caps = binding._state.caps
    if caps.key_head_dim != _KEY_DIM or caps.value_head_dim != _VALUE_DIM:
        raise ValueError("CuTe Qwen GDN requires key/value head dimensions of 128")
    if tuple(binding.recurrent_state.stride()[1:]) != (
        _VALUE_DIM * _KEY_DIM,
        _KEY_DIM,
        1,
    ):
        raise ValueError("recurrent state must be contiguous within each slot")
    return (
        binding.output.device.index,
        caps.max_seqs,
        caps.state_index_columns,
        caps.key_heads,
        caps.value_heads,
        caps.qk_l2norm,
        caps.null_state_index,
        binding.recurrent_state.dtype,
        binding.state_indices.dtype,
        binding.A_log.dtype,
        binding.dt_bias.dtype,
    )


def _compile(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _binding_key(binding)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return key, cached

    caps = binding._state.caps
    state_type = _numeric_type(binding.recurrent_state.dtype)
    index_type = _numeric_type(binding.state_indices.dtype)
    a_log_type = _numeric_type(binding.A_log.dtype)
    dt_bias_type = _numeric_type(binding.dt_bias.dtype)
    kernel = _PackedRecurrentQwenKernel(
        max_seqs=caps.max_seqs,
        state_index_columns=caps.state_index_columns,
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
        qk_l2norm=caps.qk_l2norm,
        null_state_index=caps.null_state_index,
        state_type=state_type,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=kernel,
        cache_key=key,
    )

    def fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
        return make_ptr(
            dtype,
            16,
            cute.AddressSpace.gmem,
            assumed_align=max(1, dtype.width // 8),
        )

    raw = b12x_compile(
        kernel,
        fake_pointer(BFloat16),
        fake_pointer(BFloat16),
        fake_pointer(BFloat16),
        fake_pointer(a_log_type),
        fake_pointer(dt_bias_type),
        fake_pointer(state_type),
        fake_pointer(Int32),
        fake_pointer(Int32),
        fake_pointer(index_type),
        fake_pointer(Int32),
        fake_pointer(BFloat16),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1.0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_decode.packed_recurrent_qwen",
            3,
            key,
        ),
    )

    def pointer(tensor: torch.Tensor, dtype: type[cutlass.Numeric]) -> cute.Pointer:
        return make_ptr(
            dtype,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=max(1, dtype.width // 8),
        )

    def launch(
        mixed_qkv, a, b, A_log, dt_bias, recurrent_state,
        query_start_loc, num_accepted_tokens, state_indices, num_seqs,
        output, scale,
    ) -> None:
        raw(
            pointer(mixed_qkv, BFloat16),
            pointer(a, BFloat16),
            pointer(b, BFloat16),
            pointer(A_log, a_log_type),
            pointer(dt_bias, dt_bias_type),
            pointer(recurrent_state, state_type),
            pointer(query_start_loc, Int32),
            pointer(num_accepted_tokens, Int32),
            pointer(state_indices, index_type),
            pointer(num_seqs, Int32),
            pointer(output, BFloat16),
            int(recurrent_state.stride(0)),
            int(mixed_qkv.stride(0)),
            int(a.stride(0)),
            int(b.stride(0)),
            int(output.stride(0)),
            int(state_indices.stride(0)),
            int(state_indices.stride(1)),
            float(scale),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _KERNEL_CACHE[key] = launch
    return key, launch


def precompile_packed_recurrent_qwen(binding: Binding) -> None:
    """Compile the binding specialization without mutating runtime tensors."""
    with torch.cuda.device(binding.output.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe GDN compilation is forbidden during CUDA capture"
            )
        _compile(binding)


def run_packed_recurrent_qwen(
    binding: Binding,
    *,
    scale: float | None = None,
) -> None:
    """Launch the Qwen recurrent stage without the output norm.

    A normal-stream launch is required before capture so both compilation and
    CUDA module loading have completed. This function has no fallback path.
    """
    caps = binding._state.caps
    scale_value = caps.key_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    with torch.cuda.device(binding.output.device):
        key = _binding_key(binding)
        capturing = torch.cuda.is_current_stream_capturing()
        launch = _KERNEL_CACHE.get(key)
        if capturing and (launch is None or key not in _WARMED):
            raise RuntimeError(
                "CuTe GDN specialization must be compiled and warm-run before "
                "CUDA graph capture"
            )
        if launch is None:
            key, launch = _compile(binding)
        launch(
            binding.mixed_qkv, binding.a, binding.b, binding.A_log, binding.dt_bias,
            binding.recurrent_state, binding.query_start_loc, binding.num_accepted_tokens,
            binding.state_indices, binding.num_seqs, binding.output, scale_value,
        )
        if not capturing:
            _WARMED.add(key)


def _norm_key(
    binding: Binding, *, norm_fp32: bool
) -> tuple[object, ...]:
    caps = binding._state.caps
    return (
        binding.output.device.index,
        caps.max_tokens,
        caps.value_heads,
        caps.gate_activation,
        binding.norm_weight.dtype,
        bool(norm_fp32),
    )


def _compile_norm(
    binding: Binding, *, norm_fp32: bool
) -> tuple[tuple[object, ...], Callable[[Binding, float], None]]:
    key = _norm_key(binding, norm_fp32=norm_fp32)
    cached = _NORM_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding._state.caps
    norm_weight_type = _numeric_type(binding.norm_weight.dtype)
    kernel = _GatedRmsNormKernel(
        max_tokens=caps.max_tokens,
        value_heads=caps.value_heads,
        sigmoid_gate=caps.gate_activation == "sigmoid",
        norm_weight_type=norm_weight_type,
        norm_fp32=norm_fp32,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=key
    )
    raw = b12x_compile(
        kernel,
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(norm_weight_type),
        _fake_pointer(Int32),
        Float32(1.0e-6),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_decode.gated_rmsnorm", 2, key
        ),
    )

    def launch(active_binding: Binding, eps: float) -> None:
        if _norm_key(active_binding, norm_fp32=norm_fp32) != key:
            raise ValueError(
                "compiled CuTe GDN RMSNorm does not match the binding"
            )
        raw(
            _pointer(active_binding.output, BFloat16),
            _pointer(active_binding.z, BFloat16),
            _pointer(active_binding.norm_weight, norm_weight_type),
            _pointer(active_binding.num_tokens, Int32),
            float(eps),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _NORM_CACHE[key] = launch
    return key, launch


def run_gated_rmsnorm(
    binding: Binding, *, eps: float, norm_fp32: bool = False
) -> None:
    """Apply the graph-safe gated RMSNorm to the bound output rows."""
    with torch.cuda.device(binding.output.device):
        key = _norm_key(binding, norm_fp32=norm_fp32)
        capturing = torch.cuda.is_current_stream_capturing()
        launch = _NORM_CACHE.get(key)
        if capturing and (launch is None or key not in _NORM_WARMED):
            raise RuntimeError(
                "CuTe GDN RMSNorm must be compiled and warm-run before capture"
            )
        if launch is None:
            key, launch = _compile_norm(binding, norm_fp32=norm_fp32)
        launch(binding, eps)
        if not capturing:
            _NORM_WARMED.add(key)


def clear_packed_recurrent_qwen_cache() -> None:
    """Clear process-local launcher state for focused tests."""
    _KERNEL_CACHE.clear()
    _WARMED.clear()
    _NORM_CACHE.clear()
    _NORM_WARMED.clear()


__all__ = [
    "clear_packed_recurrent_qwen_cache",
    "precompile_packed_recurrent_qwen",
    "run_gated_rmsnorm",
    "run_packed_recurrent_qwen",
]
