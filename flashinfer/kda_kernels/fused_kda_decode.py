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

Fused Kimi KDA decode kernel for SM100.

The kernel combines the width-four depthwise causal convolution, SiLU,
per-key-dimension gated delta-rule recurrence, and gated RMSNorm into one
launch. It is specialized for head dimension 128 and 12, 24, 32, 48, or 96
heads.
"""

import functools
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.utils import SmemAllocator
import tvm_ffi  # noqa: F401 -- TVM FFI is required for kernel dispatch

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

_HEAD_DIM = 128
_NUM_WARPS = 16
_NUM_THREADS = _NUM_WARPS * 32
_ROWS_PER_WARP = _HEAD_DIM // _NUM_WARPS
_CONV_THREADS = 3 * _HEAD_DIM // 4
_Q_SCALE = _HEAD_DIM**-0.5
_L2_EPS = 1.0e-6
_SUPPORTED_HEADS = (12, 24, 32, 48, 96)
_CUTE_DSL_MODULE = "fused_kda_decode"
_SOURCE_FILES = (str(Path(__file__).resolve()),)


def _aligned_tensor(tensor, alignment):
    """Return a view carrying the alignment known by this kernel."""
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            pointer.dtype,
            pointer.toint(),
            pointer.memspace,
            assumed_align=alignment,
        ),
        tensor.layout,
    )


def _sigmoid(value):
    """Evaluate a cancellation-safe approximate sigmoid."""
    return cute.rcp(cute.exp(-value, fastmath=True) + 1.0, approx=True, ftz=True)


@cute.kernel
def _fused_kda_decode_kernel(
    x,
    weight,
    conv_state,
    raw_gate,
    dt_bias,
    A_log,
    raw_beta,
    state_indices,
    state,
    output_gate,
    norm_weight,
    output,
    state_is_bf16: cutlass.Constexpr,
    use_lower_bound: cutlass.Constexpr,
    lower_bound: cutlass.Constexpr,
    norm_eps: cutlass.Constexpr,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    head_idx, row_idx, _ = cute.arch.block_idx()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    smem = SmemAllocator()
    mixed = smem.allocate_tensor(
        F32, cute.make_layout((3 * _HEAD_DIM,)), byte_alignment=16
    )
    recurrence_output = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,)), byte_alignment=16
    )
    output_scale = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,)), byte_alignment=16
    )
    gate_decay = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,)), byte_alignment=16
    )
    beta_smem = smem.allocate_tensor(F32, cute.make_layout((4,)), byte_alignment=16)

    mixed_warp = cute.make_tensor(
        mixed.iterator, cute.make_layout((4, _CONV_THREADS), stride=(1, 4))
    )
    mixed_qkv = cute.make_tensor(
        mixed.iterator, cute.make_layout((4, 32, 3), stride=(1, 4, _HEAD_DIM))
    )
    recurrence_output_warp = cute.make_tensor(
        recurrence_output.iterator, cute.make_layout((4, 32), stride=(1, 4))
    )
    output_scale_warp = cute.make_tensor(
        output_scale.iterator, cute.make_layout((4, 32), stride=(1, 4))
    )
    gate_decay_warp = cute.make_tensor(
        gate_decay.iterator, cute.make_layout((4, 32), stride=(1, 4))
    )

    copy_f32x4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128
    )
    copy_bf16x4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=64
    )

    # Slot zero is a read-only null slot. Non-positive indices produce a zero
    # output and suppress both cache updates while still following a uniform
    # control-flow path through the CTA.
    requested_slot = state_indices[row_idx]
    is_live = requested_slot > 0
    slot = cute.max(requested_slot, cutlass.Int32(0))

    # Stage 1: width-four depthwise causal convolution followed by SiLU.
    if thread_idx < _CONV_THREADS:
        channel_in_warp = thread_idx % 32
        vector_idx = thread_idx // 32
        weight_registers = cute.make_rmem_tensor((4, 4), F32)
        conv_registers = cute.make_rmem_tensor((4, 3), BF16)
        for width_idx in cutlass.range_constexpr(4):
            cute.copy(
                copy_f32x4,
                _aligned_tensor(
                    weight[(None, channel_in_warp, width_idx, vector_idx, head_idx)],
                    16,
                ),
                weight_registers[(None, width_idx)],
            )
        for history_idx in cutlass.range_constexpr(3):
            cute.copy(
                copy_bf16x4,
                _aligned_tensor(
                    conv_state[
                        (
                            None,
                            channel_in_warp,
                            history_idx,
                            vector_idx,
                            head_idx,
                            slot,
                        )
                    ],
                    8,
                ),
                conv_registers[(None, history_idx)],
            )

        mixed_registers = cute.make_rmem_tensor((4,), F32)
        for qkv_idx in cutlass.range_constexpr(4):
            current = x[(qkv_idx, channel_in_warp, vector_idx, head_idx, row_idx)]
            accumulator = (
                conv_registers[(qkv_idx, 0)].to(F32) * weight_registers[(qkv_idx, 0)]
                + conv_registers[(qkv_idx, 1)].to(F32) * weight_registers[(qkv_idx, 1)]
                + conv_registers[(qkv_idx, 2)].to(F32) * weight_registers[(qkv_idx, 2)]
                + current.to(F32) * weight_registers[(qkv_idx, 3)]
            )
            mixed_registers[qkv_idx] = (
                (accumulator * _sigmoid(accumulator)).to(BF16).to(F32)
            )
            conv_registers[(qkv_idx, 0)] = conv_registers[(qkv_idx, 1)]
            conv_registers[(qkv_idx, 1)] = conv_registers[(qkv_idx, 2)]
            conv_registers[(qkv_idx, 2)] = current

        cute.copy(
            copy_f32x4,
            mixed_registers,
            _aligned_tensor(mixed_warp[(None, thread_idx)], 16),
        )
        if is_live:
            for history_idx in cutlass.range_constexpr(3):
                cute.copy(
                    copy_bf16x4,
                    conv_registers[(None, history_idx)],
                    _aligned_tensor(
                        conv_state[
                            (
                                None,
                                channel_in_warp,
                                history_idx,
                                vector_idx,
                                head_idx,
                                slot,
                            )
                        ],
                        8,
                    ),
                )
    elif thread_idx >= _NUM_THREADS - _HEAD_DIM:
        # Four otherwise-idle warps build the per-channel recurrence decay and
        # gated RMSNorm scale once, rather than recomputing them in every warp.
        channel_idx = thread_idx - 3 * _HEAD_DIM
        channel_vector = channel_idx % 4
        channel_lane = channel_idx // 4
        A = cute.exp(A_log[head_idx])
        gate = (
            raw_gate[(channel_vector, channel_lane, head_idx, row_idx)].to(F32)
            + dt_bias[(channel_vector, channel_lane, head_idx)]
        )
        if cutlass.const_expr(use_lower_bound):
            gate_value = lower_bound * _sigmoid(A * gate)
        else:
            softplus = gate
            if gate <= 20.0:
                softplus = cute.log(
                    F32(1.0) + cute.exp(gate, fastmath=True), fastmath=True
                )
            gate_value = -A * softplus
        gate_decay[channel_idx] = cute.exp(gate_value)
        output_gate_value = output_gate[
            (channel_vector, channel_lane, head_idx, row_idx)
        ].to(F32)
        output_scale[channel_idx] = norm_weight[
            (channel_vector, channel_lane)
        ] * _sigmoid(output_gate_value)
        if channel_idx == 0:
            beta_smem[0] = _sigmoid(raw_beta[(head_idx, row_idx)].to(F32))

    global_state = _aligned_tensor(
        state[(None, lane_idx, None, warp_idx, head_idx, slot)], 16
    )
    state_registers = cute.make_rmem_tensor((4, _ROWS_PER_WARP), F32)
    if cutlass.const_expr(state_is_bf16):
        for local_row in cutlass.range_constexpr(_ROWS_PER_WARP):
            state_bf16 = cute.make_rmem_tensor((4,), BF16)
            cute.copy(
                copy_bf16x4,
                _aligned_tensor(global_state[(None, local_row)], 8),
                state_bf16,
            )
            for channel_idx in cutlass.range_constexpr(4):
                state_registers[(channel_idx, local_row)] = state_bf16[channel_idx].to(
                    F32
                )
    else:
        cute.copy(copy_f32x4, global_state, state_registers)

    cute.arch.barrier()

    # Each lane owns four contiguous K channels. A warp owns eight complete
    # state rows, keeping the full 128x128 state matrix in registers per CTA.
    query = cute.make_rmem_tensor((4,), F32)
    key = cute.make_rmem_tensor((4,), F32)
    cute.copy(copy_f32x4, _aligned_tensor(mixed_qkv[(None, lane_idx, 0)], 16), query)
    cute.copy(copy_f32x4, _aligned_tensor(mixed_qkv[(None, lane_idx, 1)], 16), key)

    query_norm = F32(0.0)
    key_norm = F32(0.0)
    query_key_dot = F32(0.0)
    for channel_idx in cutlass.range_constexpr(4):
        query_norm += query[channel_idx] * query[channel_idx]
        key_norm += key[channel_idx] * key[channel_idx]
        query_key_dot += query[channel_idx] * key[channel_idx]
    query_norm = cute.arch.warp_reduction_sum(query_norm)
    key_norm = cute.arch.warp_reduction_sum(key_norm)
    query_key_dot = cute.arch.warp_reduction_sum(query_key_dot)

    query_scale = cute.rsqrt(query_norm + _L2_EPS) * _Q_SCALE
    key_scale = cute.rsqrt(key_norm + _L2_EPS)
    normalized_query_key_dot = query_key_dot * query_scale * key_scale

    decay = cute.make_rmem_tensor((4,), F32)
    cute.copy(copy_f32x4, _aligned_tensor(gate_decay_warp[(None, lane_idx)], 16), decay)
    beta = beta_smem[0]

    # Stage 2: gated delta-rule state update and recurrent output.
    for local_row in cutlass.range_constexpr(_ROWS_PER_WARP):
        for channel_idx in cutlass.range_constexpr(4):
            state_registers[(channel_idx, local_row)] *= decay[channel_idx]

    for local_row in cutlass.range_constexpr(_ROWS_PER_WARP):
        state_key_dot = F32(0.0)
        state_query_dot = F32(0.0)
        for channel_idx in cutlass.range_constexpr(4):
            state_key_dot += (
                state_registers[(channel_idx, local_row)] * key[channel_idx]
            )
            state_query_dot += (
                state_registers[(channel_idx, local_row)] * query[channel_idx]
            )
        state_key_dot = cute.arch.warp_reduction_sum(state_key_dot) * key_scale
        state_query_dot = cute.arch.warp_reduction_sum(state_query_dot) * query_scale
        output_row = warp_idx * _ROWS_PER_WARP + local_row
        value = mixed[2 * _HEAD_DIM + output_row]
        delta = (value - state_key_dot) * beta
        recurrent_value = state_query_dot + delta * normalized_query_key_dot
        delta_key_scale = delta * key_scale
        for channel_idx in cutlass.range_constexpr(4):
            state_registers[(channel_idx, local_row)] += (
                delta_key_scale * key[channel_idx]
            )
        if is_live:
            if cutlass.const_expr(state_is_bf16):
                updated_state_bf16 = cute.make_rmem_tensor((4,), BF16)
                for channel_idx in cutlass.range_constexpr(4):
                    updated_state_bf16[channel_idx] = state_registers[
                        (channel_idx, local_row)
                    ].to(BF16)
                cute.copy(
                    copy_bf16x4,
                    updated_state_bf16,
                    _aligned_tensor(global_state[(None, local_row)], 8),
                )
            else:
                cute.copy(
                    copy_f32x4,
                    state_registers[(None, local_row)],
                    _aligned_tensor(global_state[(None, local_row)], 16),
                )
        if lane_idx == 0:
            recurrence_output[output_row] = recurrent_value.to(BF16).to(F32)

    cute.arch.barrier()

    # Stage 3: gated RMSNorm and BF16 output conversion.
    if thread_idx < 32:
        output_registers = cute.make_rmem_tensor((4,), F32)
        cute.copy(
            copy_f32x4,
            _aligned_tensor(recurrence_output_warp[(None, lane_idx)], 16),
            output_registers,
        )
        sum_squares = F32(0.0)
        for channel_idx in cutlass.range_constexpr(4):
            sum_squares += output_registers[channel_idx] ** 2
        sum_squares = cute.arch.warp_reduction_sum(sum_squares)
        inverse_rms = cute.rsqrt(sum_squares * (1.0 / _HEAD_DIM) + norm_eps)
        scale = cute.make_rmem_tensor((4,), F32)
        cute.copy(
            copy_f32x4,
            _aligned_tensor(output_scale_warp[(None, lane_idx)], 16),
            scale,
        )
        final_output = cute.make_rmem_tensor((4,), BF16)
        for channel_idx in cutlass.range_constexpr(4):
            final_output[channel_idx] = (
                output_registers[channel_idx] * inverse_rms * scale[channel_idx]
            ).to(BF16)
        if is_live:
            cute.copy(
                copy_bf16x4,
                final_output,
                _aligned_tensor(output[(None, lane_idx, head_idx, row_idx)], 8),
            )
        else:
            zero_output = cute.make_rmem_tensor((4,), BF16)
            for channel_idx in cutlass.range_constexpr(4):
                zero_output[channel_idx] = BF16(0.0)
            cute.copy(
                copy_bf16x4,
                zero_output,
                _aligned_tensor(output[(None, lane_idx, head_idx, row_idx)], 8),
            )


@cute.jit
def _fused_kda_decode_launch(
    x,
    weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state_indices,
    state,
    output_gate,
    norm_weight,
    output,
    stream: cuda.CUstream,
    state_is_bf16: cutlass.Constexpr,
    use_lower_bound: cutlass.Constexpr,
    lower_bound: cutlass.Constexpr,
    norm_eps: cutlass.Constexpr,
):
    num_heads = A_log.shape[0]
    num_rows = state_indices.shape[0]
    hidden_size = num_heads * _HEAD_DIM
    qkv_size = 3 * hidden_size

    x_layout = cute.make_tensor(
        x.iterator,
        cute.make_layout(
            (4, 32, 3, num_heads, num_rows),
            stride=(1, 4, hidden_size, _HEAD_DIM, x.stride[0]),
        ),
    )
    weight_layout = cute.make_tensor(
        weight.iterator,
        cute.make_layout(
            (4, 32, 4, 3, num_heads),
            stride=(1, 4, hidden_size, 4 * hidden_size, _HEAD_DIM),
        ),
    )
    conv_state_layout = cute.make_tensor(
        conv_state.iterator,
        cute.make_layout(
            (4, 32, 3, 3, num_heads, conv_state.shape[0]),
            stride=(
                1,
                4,
                qkv_size,
                hidden_size,
                _HEAD_DIM,
                conv_state.stride[0],
            ),
        ),
    )
    gate_layout = cute.make_tensor(
        raw_gate.iterator,
        cute.make_layout(
            (4, 32, num_heads, num_rows), stride=(1, 4, _HEAD_DIM, hidden_size)
        ),
    )
    dt_bias_layout = cute.make_tensor(
        dt_bias.iterator,
        cute.make_layout((4, 32, num_heads), stride=(1, 4, _HEAD_DIM)),
    )
    beta_layout = cute.make_tensor(
        raw_beta.iterator,
        cute.make_layout((num_heads, num_rows), stride=(1, raw_beta.stride[1])),
    )
    state_layout = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            (4, 32, _ROWS_PER_WARP, _NUM_WARPS, num_heads, state.shape[0]),
            stride=(
                1,
                4,
                _HEAD_DIM,
                _ROWS_PER_WARP * _HEAD_DIM,
                _HEAD_DIM * _HEAD_DIM,
                state.stride[0],
            ),
        ),
    )
    output_gate_layout = cute.make_tensor(
        output_gate.iterator,
        cute.make_layout(
            (4, 32, num_heads, num_rows),
            stride=(1, 4, _HEAD_DIM, output_gate.stride[0]),
        ),
    )
    norm_weight_layout = cute.make_tensor(
        norm_weight.iterator, cute.make_layout((4, 32), stride=(1, 4))
    )
    output_layout = cute.make_tensor(
        output.iterator,
        cute.make_layout(
            (4, 32, num_heads, num_rows),
            stride=(1, 4, _HEAD_DIM, hidden_size),
        ),
    )

    _fused_kda_decode_kernel(
        x_layout,
        weight_layout,
        conv_state_layout,
        gate_layout,
        dt_bias_layout,
        A_log,
        beta_layout,
        state_indices,
        state_layout,
        output_gate_layout,
        norm_weight_layout,
        output_layout,
        state_is_bf16,
        use_lower_bound,
        lower_bound,
        norm_eps,
    ).launch(
        grid=[num_heads, num_rows, 1],
        block=[_NUM_THREADS, 1, 1],
        smem=4 * (6 * _HEAD_DIM) + 256,
        stream=stream,
    )


def _make_compile_inputs(state_dtype):
    """Build shape- and stride-dynamic inputs for CuTe compilation."""
    num_rows = cute.sym_int()
    num_heads = cute.sym_int()
    conv_slots = cute.sym_int()
    state_slots = cute.sym_int()
    hidden_size = num_heads * _HEAD_DIM
    qkv_size = 3 * hidden_size

    def compact(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=16,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    x = cute.runtime.make_fake_tensor(
        BF16,
        shape=(num_rows, qkv_size),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    conv_state = cute.runtime.make_fake_tensor(
        BF16,
        shape=(conv_slots, qkv_size, 3),
        stride=(cute.sym_int64(), 1, qkv_size),
        assumed_align=16,
    )
    raw_beta = cute.runtime.make_fake_tensor(
        BF16,
        shape=(1, num_rows, num_heads),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
        assumed_align=16,
    )
    state = cute.runtime.make_fake_tensor(
        state_dtype,
        shape=(state_slots, num_heads, _HEAD_DIM, _HEAD_DIM),
        stride=(
            cute.sym_int64(),
            _HEAD_DIM * _HEAD_DIM,
            _HEAD_DIM,
            1,
        ),
        assumed_align=16,
    )
    output_gate = cute.runtime.make_fake_tensor(
        BF16,
        shape=(num_rows, num_heads, _HEAD_DIM),
        stride=(cute.sym_int64(), _HEAD_DIM, 1),
        assumed_align=16,
    )
    return (
        x,
        compact((3, 4, hidden_size), F32),
        conv_state,
        compact((1, num_rows, num_heads, _HEAD_DIM), BF16),
        raw_beta,
        compact((num_heads,), F32),
        compact((hidden_size,), F32),
        compact((num_rows,), cutlass.Int32),
        state,
        output_gate,
        compact((_HEAD_DIM,), F32),
        compact((1, num_rows, num_heads, _HEAD_DIM), BF16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.cache
def _get_compiled_kernel(state_dtype, lower_bound, norm_eps):
    """Get a shape-dynamic specialization from the two-level CuTe DSL cache."""
    state_is_bf16 = state_dtype == torch.bfloat16
    compile_state_dtype = BF16 if state_is_bf16 else F32
    state_name = "bf16" if state_is_bf16 else "f32"
    use_lower_bound = lower_bound is not None
    gate_name = (
        f"lb{str(float(lower_bound)).replace('.', '_').replace('-', 'm')}"
        if use_lower_bound
        else "softplus"
    )
    compile_lower_bound = float(lower_bound) if use_lower_bound else -5.0
    kernel_name = (
        f"d128_w4_{gate_name}_state{state_name}"
        f"_eps{str(float(norm_eps)).replace('.', '_').replace('-', 'm')}"
    )
    return build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        kernel_name,
        lambda: cute.compile(
            _fused_kda_decode_launch,
            *_make_compile_inputs(compile_state_dtype),
            state_is_bf16,
            use_lower_bound,
            compile_lower_bound,
            float(norm_eps),
            options="--enable-tvm-ffi --generate-line-info",
        ),
        extra_key_files=_SOURCE_FILES,
    )


def _check_cuda_tensor(name, tensor, dtype):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    expected_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in expected_dtypes:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


@torch.no_grad()
def run_fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: float | None = -5.0,
    norm_eps: float = 1e-5,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused Kimi decode and update both cache tensors in-place."""
    _check_cuda_tensor("x", x, torch.bfloat16)
    _check_cuda_tensor("weight", weight, torch.float32)
    _check_cuda_tensor("conv_state", conv_state, torch.bfloat16)
    _check_cuda_tensor("raw_gate", raw_gate, torch.bfloat16)
    _check_cuda_tensor("raw_beta", raw_beta, torch.bfloat16)
    _check_cuda_tensor("A_log", A_log, torch.float32)
    _check_cuda_tensor("dt_bias", dt_bias, torch.float32)
    _check_cuda_tensor("state_indices", state_indices, torch.int32)
    _check_cuda_tensor("state", state, (torch.float32, torch.bfloat16))
    _check_cuda_tensor("output_gate", output_gate, torch.bfloat16)
    _check_cuda_tensor("norm_weight", norm_weight, torch.float32)
    for name, tensor in (
        ("weight", weight),
        ("conv_state", conv_state),
        ("raw_gate", raw_gate),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("state_indices", state_indices),
        ("state", state),
        ("output_gate", output_gate),
        ("norm_weight", norm_weight),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same device as x")

    if x.ndim != 2 or x.shape[1] % (3 * _HEAD_DIM) != 0:
        raise ValueError("x must have shape [num_rows, 3 * num_heads * 128]")
    num_rows = x.shape[0]
    num_heads = x.shape[1] // (3 * _HEAD_DIM)
    hidden_size = num_heads * _HEAD_DIM
    if num_rows == 0:
        raise ValueError("x must contain at least one decode row")
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(
            f"num_heads must be one of {_SUPPORTED_HEADS}, got {num_heads}"
        )
    if x.stride(1) != 1:
        raise ValueError("x must be contiguous in its channel dimension")
    if weight.shape != (3, 4, hidden_size) or not weight.is_contiguous():
        raise ValueError("weight must be contiguous with shape [3, 4, H * 128]")
    if (
        conv_state.ndim != 3
        or conv_state.shape[1:] != (3 * hidden_size, 3)
        or conv_state.stride(0) < 9 * hidden_size
        or conv_state.stride(1) != 1
        or conv_state.stride(2) != 3 * hidden_size
    ):
        raise ValueError(
            "conv_state must have shape [slots, 3 * H * 128, 3] "
            "and use the SD cache layout"
        )
    if raw_gate.shape != (1, num_rows, num_heads, _HEAD_DIM):
        raise ValueError("raw_gate must have shape [1, num_rows, H, 128]")
    if not raw_gate.is_contiguous():
        raise ValueError("raw_gate must be contiguous")
    if raw_beta.shape != (1, num_rows, num_heads) or raw_beta.stride(2) != 1:
        raise ValueError("raw_beta must have shape [1, num_rows, H]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must be contiguous with shape [H]")
    if dt_bias.shape != (hidden_size,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must be contiguous with shape [H * 128]")
    if state_indices.shape != (num_rows,) or not state_indices.is_contiguous():
        raise ValueError("state_indices must be contiguous with shape [num_rows]")
    if (
        state.ndim != 4
        or state.shape[1:] != (num_heads, _HEAD_DIM, _HEAD_DIM)
        or state.stride(0) < num_heads * _HEAD_DIM * _HEAD_DIM
        or state.stride(1) != _HEAD_DIM * _HEAD_DIM
        or state.stride(2) != _HEAD_DIM
        or state.stride(3) != 1
    ):
        raise ValueError(
            "state must have shape [slots, H, 128, 128] with contiguous slot contents"
        )
    if conv_state.shape[0] != state.shape[0]:
        raise ValueError(
            "conv_state and state must have the same number of cache slots"
        )
    if output_gate.ndim == 4:
        if output_gate.shape != (1, num_rows, num_heads, _HEAD_DIM):
            raise ValueError(
                "output_gate must have shape [num_rows, H, 128] "
                "or [1, num_rows, H, 128]"
            )
        output_gate = output_gate[0]
    if output_gate.shape != (num_rows, num_heads, _HEAD_DIM):
        raise ValueError(
            "output_gate must have shape [num_rows, H, 128] or [1, num_rows, H, 128]"
        )
    if output_gate.stride(2) != 1 or output_gate.stride(1) != _HEAD_DIM:
        raise ValueError("output_gate must have contiguous head rows")
    if norm_weight.shape != (_HEAD_DIM,) or not norm_weight.is_contiguous():
        raise ValueError("norm_weight must be contiguous with shape [128]")
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError("lower_bound must be negative")
    if norm_eps < 0:
        raise ValueError("norm_eps must be non-negative")

    expected_output_shape = (1, num_rows, num_heads, _HEAD_DIM)
    if output is None:
        output = torch.empty(expected_output_shape, dtype=x.dtype, device=x.device)
    else:
        _check_cuda_tensor("output", output, torch.bfloat16)
        if output.device != x.device:
            raise ValueError("output must be on the same device as x")
        if output.shape != expected_output_shape or not output.is_contiguous():
            raise ValueError(
                "output must be contiguous with shape [1, num_rows, H, 128]"
            )

    kernel = _get_compiled_kernel(state.dtype, lower_bound, float(norm_eps))
    kernel(
        x,
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        output,
    )
    return output
