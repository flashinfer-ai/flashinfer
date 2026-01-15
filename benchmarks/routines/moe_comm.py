# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MoE Communication Benchmark Routine

This module provides benchmarking for MoE All-to-All communication operations
using FlashInfer's MoeAlltoAll interface. Designed to run with mpirun for
multi-GPU benchmarking.

Launch examples:
    # Basic (no quantization)
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine moe_a2a_dispatch_combine \
        --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8

    # With FP8 quantization
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine moe_a2a_dispatch_combine \
        --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
        --quant_dtype fp8

    # With NVFP4 quantization
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine moe_a2a_dispatch_combine \
        --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
        --quant_dtype nvfp4

    # With validation (recommended for first run)
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine moe_a2a_dispatch_combine \
        --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
        --validate

    # With per-phase timing (less accurate but shows dispatch and combine times separately)
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine moe_a2a_dispatch_combine \
        --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
        --per_phase_timing

Options:
    --quant_dtype fp8    : FP8 (float8_e4m3fn) with float32 per-tensor scale
    --quant_dtype nvfp4  : NVFP4 (4-bit) with float8_e4m3fn block scales
    --validate           : Run correctness validation before benchmarking.
                           Uses a deterministic fake MoE to verify round-trip
                           communication. For non-quantized mode, performs exact
                           comparison. For quantized mode, validates output
                           shape and numerical validity.
    --per_phase_timing   : Enable per-phase timing (dispatch/combine). Adds slight
                           overhead from CUDA events.
                           This is less accurate than the total timing but shows
                           dispatch and combine times separately.
    --nvtx               : Enable NVTX markers for Nsight Systems profiling.
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch

from mpi4py import MPI

from flashinfer.comm import MoeAlltoAll
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory
from flashinfer import fp4_quantize
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    print_perf_metrics,
)

# Constants for FP4 quantization
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


@contextmanager
def cuda_event_timer(events_list: list, enabled: bool = True):
    """
    Context manager for deferred CUDA event timing.

    Records start/end events and appends them to events_list.
    Does NOT synchronize - caller must sync before reading elapsed times.

    Args:
        events_list: List to append (start, end) event tuple to.
        enabled: If False, skip event recording (no-op).

    Example:
        dispatch_events = []
        for _ in range(num_iters):
            with cuda_event_timer(dispatch_events):
                moe_a2a.dispatch(...)

        torch.cuda.synchronize()  # Single sync at end
        times = [s.elapsed_time(e) for s, e in dispatch_events]
    """
    if not enabled:
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        events_list.append((start, end))


@contextmanager
def nvtx_range(name: str, enabled: bool = True):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def print_ordered(comm, rank: int, msg, condition: bool = True):
    """
    Print messages one rank at a time in order, synchronized with MPI barriers.

    Args:
        comm: MPI communicator
        rank: Current rank
        msg: string to print.
        condition: If False, skip printing but still participate in barriers.
    """
    for i in range(comm.Get_size()):
        if i == rank and condition:
            print(msg, flush=True)
        comm.Barrier()


def run_moe_comm_test(args):
    """
    Run a MoE communication test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.routine == "moe_a2a_dispatch_combine":
        return test_moe_a2a_dispatch_combine(args)
    # TODO: add a2a_dispatch + moe + a2a_combine
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_moe_comm_args(line, parser):
    """
    Parse command line arguments for MoE communication test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="Number of tokens per rank (local batch size).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="Hidden dimension size.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        required=True,
        help="Total number of experts across all ranks.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Number of experts to route each token to.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for hidden states payload (before quantization if quant_dtype is set).",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        required=False,
        default=None,
        choices=["fp8", "nvfp4"],
        help="Quantization format for hidden states. If set, hidden states are quantized and block-scale scale factors are communicated.",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        required=False,
        default=None,
        help="Max tokens per rank for workspace allocation. Defaults to num_tokens.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run correctness validation before benchmarking. Uses a deterministic fake MoE to verify round-trip communication.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX markers for Nsight Systems profiling.",
    )
    parser.add_argument(
        "--per_phase_timing",
        action="store_true",
        help="Enable per-phase timing (dispatch/combine). Adds slight overhead from CUDA events.",
    )

    args = parser.parse_args(line)

    # Default max_num_tokens to num_tokens if not specified
    if args.max_num_tokens is None:
        args.max_num_tokens = args.num_tokens

    # Derive scale_dtype from quant_dtype
    if args.quant_dtype == "nvfp4":
        args.scale_dtype = torch.float8_e4m3fn
    elif args.quant_dtype == "fp8":
        args.scale_dtype = torch.float32
    else:
        args.scale_dtype = None

    if args.verbose >= 1:
        print(f"[INFO] {args = }")

    return args


def _setup_mpi_and_device() -> Tuple[MPI.Comm, int, int, int]:
    """
    Setup MPI communicator and CUDA device based on local rank.

    Returns:
        Tuple of (comm, rank, world_size, local_rank)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Get local rank within the node for device assignment
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    torch.cuda.set_device(local_rank)

    return comm, rank, world_size, local_rank


def _calculate_fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate global scale factor for FP4 quantization."""
    tensor_amax = tensor.abs().max().to(torch.float32)
    global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / tensor_amax if tensor_amax != 0.0 else 0.0
    )
    return global_scale


def _quantize_to_fp8(
    hidden_states: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to FP8 (per-tensor scale).

    Args:
        hidden_states: Input tensor to quantize
        scale: Optional pre-computed scale. If None, computed from hidden_states.

    Returns:
        Tuple of (quantized_hidden_states, scale_factor)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if scale is None:
        amax = hidden_states.abs().max().float().clamp(min=1e-6)
        scale = amax / fp8_max
    inv_scale = 1.0 / scale if scale != 0.0 else 0.0
    quantized = (
        (hidden_states.float() * inv_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale.view(1)


def _dequantize_fp8_to_dtype(
    tensor_fp8: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to high precision.

    Args:
        tensor_fp8: FP8 quantized tensor (float8_e4m3fn)
        scale: Per-tensor scale factor
        dtype: Output dtype

    Returns:
        Dequantized tensor in specified dtype
    """
    return (tensor_fp8.float() * scale.float()).to(dtype)


def _quantize_to_nvfp4(
    hidden_states: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to NVFP4 (block scale).

    Args:
        hidden_states: Input tensor to quantize
        global_scale: Optional pre-computed global scale. If None, computed from hidden_states.

    Returns:
        Tuple of (quantized_hidden_states, block_scale_factors, global_scale_factor)
        - quantized_hidden_states: uint8 tensor, packed (2 FP4 values per byte)
        - block_scale_factors: float8_e4m3fn tensor, shape [num_tokens, hidden_size // 16]
        - global_scale_factor: float32 scalar
    """
    if global_scale is None:
        global_scale = _calculate_fp4_global_scale(hidden_states)
    sf_vec_size = 16
    use_ue8m0 = False

    # Activation always uses linear (i.e., non-swizzled) layout
    is_sf_swizzled_layout = False

    # Returns (quantized_data, block_scales)
    quantized, block_scales = fp4_quantize(
        hidden_states, global_scale, sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    # Reshape quantized data: pack 2 FP4 values into 1 byte
    num_tokens, hidden_size = hidden_states.shape
    quantized_packed = quantized.view(torch.uint8).reshape(num_tokens, hidden_size // 2)

    # Block scales are float8_e4m3fn
    block_scales_reshaped = block_scales.view(torch.float8_e4m3fn).reshape(
        num_tokens, hidden_size // sf_vec_size
    )

    return quantized_packed, block_scales_reshaped, global_scale


# Copied/adapted from tests/moe/test_trtllm_cutlass_fused_moe.py
def _dequantize_nvfp4_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
):
    """Dequantize the fp4 tensor back to high precision."""

    def break_fp4_bytes(a, dtype):
        assert a.dtype == torch.uint8
        m, n = a.shape
        # Vectorized nibble processing
        a_flat = a.flatten()
        high = (a_flat & 0xF0) >> 4  # Upper nibbles
        low = a_flat & 0x0F  # Lower nibbles
        # Combine nibbles for batch processing
        combined = torch.stack((low, high), dim=1).flatten()
        # Vectorized sign and magnitude extraction
        signs = (combined & 0x08).to(torch.bool)  # Sign bits
        abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices
        # Device-aware lookup and sign application
        kE2M1ToFloat = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
        )
        kE2M1 = kE2M1ToFloat.to(device=a.device)
        values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
        # Reshape to final form
        return values.reshape(m, n * 2).to(dtype=dtype)

    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def _create_moe_inputs(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    input_dtype: torch.dtype,
    quant_dtype: Optional[str],
    device: torch.device,
    comm: MPI.Comm,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    List[torch.Tensor],
]:
    """
    Create input tensors for MoE A2A benchmark.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        num_experts: Total number of experts
        top_k: Number of experts per token
        input_dtype: Data type for hidden states (before quantization)
        quant_dtype: None, "fp8", or "nvfp4"
        device: CUDA device
        comm: MPI communicator for syncing global scale

    Returns:
        Tuple of (hidden_states, hidden_states_original, token_selected_experts, token_final_scales, scale_factor, global_scale, input_payloads)
        - hidden_states: The tensor to communicate (may be quantized)
        - hidden_states_original: The original unquantized hidden states for validation purpose
        - token_selected_experts: Expert indices
        - token_final_scales: Routing weights
        - scale_factor: Block scale factor tensor for NVFP4 (in A2A payloads), None otherwise
        - global_scale: Global/per-tensor scale factor for quantized dtype (synced via MPI max, same across ranks), None otherwise
        - input_payloads: List of all payloads for dispatch
    """
    # Generate original hidden states in input_dtype
    hidden_states_original = torch.randn(
        num_tokens, hidden_size, dtype=input_dtype, device=device
    )

    # Expert selection: random experts for each token
    token_selected_experts = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device=device)[:top_k]
            for _ in range(num_tokens)
        ],
        dim=0,
    ).contiguous()

    # Routing weights
    token_final_scales = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device=device
    )

    # Handle quantization
    # Global scale is synced via MPI max reduction to mimic the fact that it is part of model ckpt
    scale_factor = None
    global_scale = None
    if quant_dtype == "nvfp4":
        # Compute local global scale, sync via max, then quantize
        local_global_scale = _calculate_fp4_global_scale(hidden_states_original)
        synced_global_scale = comm.allreduce(
            local_global_scale.cpu().item(), op=MPI.MAX
        )
        global_scale = torch.tensor(
            synced_global_scale, dtype=torch.float32, device=device
        )
        hidden_states, scale_factor, global_scale = _quantize_to_nvfp4(
            hidden_states_original, global_scale
        )
    elif quant_dtype == "fp8":
        # Compute local amax, sync via max, then quantize with synced scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        local_amax = hidden_states_original.abs().max().float().item()
        synced_amax = comm.allreduce(local_amax, op=MPI.MAX)
        synced_scale = torch.tensor(
            synced_amax / fp8_max, dtype=torch.float32, device=device
        )
        hidden_states, global_scale = _quantize_to_fp8(
            hidden_states_original, synced_scale
        )
    else:
        # No quantization
        hidden_states = hidden_states_original

    # Build payload list for dispatch
    # Base payloads: hidden states, expert IDs, routing weights
    input_payloads = [hidden_states, token_selected_experts, token_final_scales]

    # For post-quant communication: include block scale factors so experts can dequantize
    if scale_factor is not None:
        input_payloads.append(scale_factor)

    return (
        hidden_states,
        hidden_states_original,
        token_selected_experts,
        token_final_scales,
        scale_factor,
        global_scale,
        input_payloads,
    )


def _calculate_comm_bandwidth(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    time_ms: float,
    input_dtype: torch.dtype,
    quant_dtype: Optional[str] = None,
    phase: str = "dispatch_combine",
) -> float:
    """
    Calculate memory bandwidth for MoE A2A communication in TB/sec.

    Args:
        num_tokens: Number of tokens per rank
        hidden_size: Hidden dimension size
        top_k: Number of experts per token
        ep_size: Expert parallel size (number of ranks)
        time_ms: Time in milliseconds
        input_dtype: Data type of hidden states (before quantization)
        quant_dtype: None, "fp8", or "nvfp4"
        phase: "dispatch", "combine", or "dispatch_combine"

    Returns:
        Bandwidth in TB/sec
    """
    # Calculate hidden states payload size based on quant_dtype
    if quant_dtype == "nvfp4":
        # NVFP4: 0.5 bytes per element (packed uint8) + block scales
        hidden_states_bytes = num_tokens * hidden_size // 2  # packed FP4
        # Block scales: float8_e4m3fn, one per 16 elements
        scale_bytes = num_tokens * (hidden_size // 16) * 1  # float8_e4m3fn = 1 byte
    elif quant_dtype == "fp8":
        # FP8: 1 byte per element, no scale payload in A2A
        hidden_states_bytes = num_tokens * hidden_size * 1  # float8_e4m3fn = 1 byte
        scale_bytes = 0
    else:
        # No quantization
        element_size = torch.tensor([], dtype=input_dtype).element_size()
        hidden_states_bytes = num_tokens * hidden_size * element_size
        scale_bytes = 0

    # Dispatch phase: send hidden_states, expert_ids, scales, and quant scales
    dispatch_bytes = (
        hidden_states_bytes
        + num_tokens * top_k * 4  # token_selected_experts (int32)
        + num_tokens * top_k * 4  # token_final_scales (float32)
        + scale_bytes
    )

    # Combine phase: receive processed hidden_states back
    # Combine ALWAYS uses activation dtype (bfloat16/float16), not quantized format
    # This matches the real MoE flow: quantized dispatch -> expert compute -> activation dtype combine
    element_size = torch.tensor([], dtype=input_dtype).element_size()
    combine_bytes = num_tokens * hidden_size * element_size

    if phase == "dispatch":
        total_bytes = dispatch_bytes
    elif phase == "combine":
        total_bytes = combine_bytes
    else:  # dispatch_combine
        total_bytes = dispatch_bytes + combine_bytes

    # Account for multi-rank communication: data crosses ranks
    total_bytes *= (ep_size - 1) / ep_size if ep_size > 1 else 1

    tb_per_sec = total_bytes / (time_ms * 1e-3) / 1e12
    return tb_per_sec


# Copied from tests/comm/test_trtllm_moe_alltoall.py
def fake_moe(
    hidden_states: torch.Tensor,
    token_selected_experts: torch.Tensor,
    num_experts: int,
    is_ep: bool = False,
    ep_rank: Optional[int] = None,
    num_experts_per_rank: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply a deterministic fake MoE transformation for validation.

    Each expert applies a predictable scale: (expert_id + 1.0) / num_experts + 0.5
    This allows verifying that communication correctly routes tokens to experts
    and combines results.

    Args:
        hidden_states: Input tensor [num_tokens, hidden_size] or [world_size, num_tokens, hidden_size]
        token_selected_experts: Expert assignments [num_tokens, top_k] or [world_size, num_tokens, top_k]
        num_experts: Total number of experts
        is_ep: If True, only process experts assigned to this rank
        ep_rank: Rank for expert parallel filtering
        num_experts_per_rank: Number of experts per rank

    Returns:
        Processed tensor with same shape as hidden_states
    """
    target_shape = hidden_states.shape
    hidden_states = hidden_states.flatten(end_dim=-2)
    token_selected_experts = token_selected_experts.flatten(end_dim=-2)
    num_tokens, _ = hidden_states.shape
    _, top_k = token_selected_experts.shape

    if is_ep:
        assert ep_rank is not None and num_experts_per_rank is not None

    # Initialize output
    processed_states = torch.zeros_like(hidden_states)

    # Process each token
    for token_idx in range(num_tokens):
        results = []
        for k in range(top_k):
            expert_id = token_selected_experts[token_idx, k].item()
            if is_ep and not (
                ep_rank * num_experts_per_rank
                <= expert_id
                < (ep_rank + 1) * num_experts_per_rank
            ):
                continue

            # Deterministic scale based on expert_id
            scale = (expert_id + 1.0) / num_experts + 0.5
            results.append(hidden_states[token_idx] * scale)

        # Sum results with higher precision to match actual implementation
        if results:
            processed_states[token_idx] = torch.sum(
                torch.stack(results, dim=0), dim=0, dtype=torch.float32
            ).to(processed_states.dtype)

    return processed_states.view(target_shape)


def _validate_moe_a2a(
    moe_a2a: MoeAlltoAll,
    hidden_states: torch.Tensor,
    hidden_states_original: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    input_payloads: List[torch.Tensor],
    runtime_max_tokens_per_rank: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    quant_dtype: Optional[str],
    global_scale: Optional[torch.Tensor],
    num_experts: int,
    ep_size: int,
    rank: int,
    comm,
    verbose: int = 0,
) -> bool:
    """
    Validate MoE A2A communication correctness with a round-trip test.

    Runs dispatch -> fake_moe -> combine and compares with local reference.

    Args:
        moe_a2a: MoeAlltoAll instance
        hidden_states: Original hidden states (before quantization)
        token_selected_experts: Expert assignments
        token_final_scales: Routing weights
        input_payloads: Payloads for dispatch
        runtime_max_tokens_per_rank: Max tokens per rank
        hidden_size: Hidden dimension
        input_dtype: Data type for hidden states
        quant_dtype: Quantization format
        global_scale: Per-tensor scale factor for quantized dtype (e.g., fp8 or nvfp4), None otherwise
        num_experts: Total number of experts
        ep_size: Expert parallel size
        rank: Current rank
        comm: MPI communicator
        verbose: Verbosity level

    Returns:
        True if validation passes, False otherwise
    """
    num_experts_per_rank = num_experts // ep_size

    # Dispatch phase
    recv_tensors = moe_a2a.dispatch(
        token_selected_experts,
        input_payloads,
        runtime_max_tokens_per_rank,
    )

    # Tuck away comm and rank for the print_ordered function
    def _invoke_print_ordered(msg, condition=True):
        print_ordered(comm, rank, msg, condition)

    if verbose >= 2:
        _invoke_print_ordered(
            f"[VVERBOSE][VALIDATE][Rank {rank}] hidden_states shape: {hidden_states.shape} [num_tokens, hidden_size]:\n{hidden_states[:8, :5]}\n"
            f"[VVERBOSE][VALIDATE][Rank {rank}] token_selected_experts shape: {token_selected_experts.shape} [num_tokens, top_k]:\n{token_selected_experts[:8, :]}"
        )

    # Unpack recv_tensors
    recv_hidden = recv_tensors[0]
    recv_experts = recv_tensors[1]
    _ = recv_tensors[2]  # recv_token_final_scales
    recv_scale_factor = recv_tensors[3] if len(recv_tensors) > 3 else None

    # Note: For quantized dispatch, recv_tensors[0] is quantized.
    # Per-tensor scale factor is part of model ckpts, not in A2A payloads.
    recv_hidden_dequant = torch.zeros(
        (ep_size, runtime_max_tokens_per_rank, hidden_size),
        dtype=input_dtype,
        device=recv_hidden.device,
    )
    if quant_dtype == "nvfp4":
        for i in range(recv_hidden.shape[0]):
            recv_hidden_dequant[i] = _dequantize_nvfp4_to_dtype(
                recv_hidden[i],
                recv_scale_factor[i],
                global_scale,
                block_size=16,
                dtype=input_dtype,
            )
    elif quant_dtype == "fp8":
        for i in range(recv_hidden.shape[0]):
            recv_hidden_dequant[i] = _dequantize_fp8_to_dtype(
                recv_hidden[i],
                global_scale,
                dtype=input_dtype,
            )
    else:
        recv_hidden_dequant = recv_hidden

    if verbose >= 2:
        _invoke_print_ordered(
            f"[VVERBOSE][VALIDATE][Rank {rank}] recv_hidden shape: {recv_hidden.shape} [ep_size, max_tokens, hidden_size]:\n{recv_hidden[:, :8, :5]}\n"
            f"[VVERBOSE][VALIDATE][Rank {rank}] recv_experts shape: {recv_experts.shape} [ep_size, max_tokens, top_k]:\n{recv_experts[:, :8, :]}",
        )

    # Apply fake MoE (each expert scales by (expert_id + 1) / num_experts + 0.5)
    processed = fake_moe(
        recv_hidden_dequant,
        recv_experts,
        num_experts,
        is_ep=True,
        ep_rank=rank,
        num_experts_per_rank=num_experts_per_rank,
    )

    # Get combine payload workspace
    combine_payload = moe_a2a.get_combine_payload_tensor_in_workspace(
        runtime_max_tokens_per_rank,
        hidden_size,
        input_dtype,
    )

    combine_payload.copy_(processed)

    # Combine phase
    output_tensor = moe_a2a.combine(
        combine_payload,
        runtime_max_tokens_per_rank,
        payload_in_workspace=True,
    )

    # Verify output dtype
    assert output_tensor.dtype == input_dtype, "Output dtype mismatch"

    num_tokens = hidden_states_original.shape[0]

    # Compute exact reference using original (unquantized) hidden states
    # Gather all hidden_states_original and token_selected_experts across ranks
    # Note: numpy doesn't support bfloat16, so convert to float32 for allgather
    all_hidden_states = comm.allgather(hidden_states_original.cpu().float().numpy())
    all_token_selected_experts = comm.allgather(token_selected_experts.cpu().numpy())

    # Stack into global tensors
    global_hidden_states = (
        torch.from_numpy(np.concatenate(all_hidden_states, axis=0))
        .to(hidden_states.device)
        .to(input_dtype)
    )
    global_token_selected_experts = torch.from_numpy(
        np.concatenate(all_token_selected_experts, axis=0)
    ).to(token_selected_experts.device)

    # Compute expected result locally
    expected = fake_moe(
        global_hidden_states,
        global_token_selected_experts,
        num_experts,
        is_ep=False,
    )

    # Extract this rank's portion
    expected_local = expected[rank * num_tokens : (rank + 1) * num_tokens]

    # Print verbose debug info
    if verbose >= 2:
        _invoke_print_ordered(
            f"[VVERBOSE][VALIDATE][Rank {rank}] output_tensor shape: {output_tensor.shape} [num_tokens, hidden_size]:\n{output_tensor[:8, :5]}\n"
            f"[VVERBOSE][VALIDATE][Rank {rank}] expected_local shape: {expected_local.shape} [num_tokens, hidden_size]:\n{expected_local[:8, :5]}",
        )

    # Compare tensors with tolerance based on quantization
    # FP4 has very limited precision, so we need larger tolerances
    if quant_dtype == "nvfp4":
        atol, rtol = 2.0, 0.5  # FP4: very loose tolerance due to 4-bit precision
    elif quant_dtype == "fp8":
        atol, rtol = 0.1, 0.1  # FP8: moderate tolerance
    else:
        atol, rtol = 1e-2, 1e-2  # Non-quantized: tight tolerance

    error_msg = None
    try:
        torch.testing.assert_close(
            output_tensor,
            expected_local,
            atol=atol,
            rtol=rtol,
        )
        passed = True
    except AssertionError as e:
        passed = False
        error_msg = str(e)

    # Print errors rank-by-rank; all ranks must participate to avoid deadlock
    _invoke_print_ordered(
        f"[VALIDATE][Rank {rank}] ERROR: {error_msg}",
        condition=(error_msg is not None),
    )

    # All-reduce pass/fail status
    all_passed = comm.allreduce(passed, op=MPI.LAND)

    if rank == 0:
        if all_passed:
            print("[VALIDATE] PASSED: All ranks validated successfully")
        else:
            print("[VALIDATE] FAILED: Validation errors detected")

    return all_passed


def test_moe_a2a_dispatch_combine(args):
    """
    Benchmark MoE A2A dispatch + combine cycle.

    This benchmarks the full round-trip communication pattern:
    1. Dispatch: send tokens to expert ranks
    2. Combine: gather processed tokens back

    Args:
        args: Parsed command line arguments

    Returns:
        list: List of result dictionaries
    """
    # Setup MPI
    comm, rank, world_size, local_rank = _setup_mpi_and_device()
    ep_size = world_size
    device = torch.device("cuda")

    if rank == 0 and args.verbose >= 1:
        print("[INFO] Running test_moe_a2a_dispatch_combine")
        print(f"[INFO] ep_size={ep_size}, rank={rank}")

    # Initialize MNNVL
    try:
        MnnvlMemory.initialize()
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] MNNVL initialization failed: {e}")
        return []

    # Parse parameters
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    num_experts = args.num_experts
    top_k = args.top_k
    max_num_tokens = args.max_num_tokens
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    quant_dtype = args.quant_dtype

    res = []

    # Validate parameters
    if num_experts % ep_size != 0:
        if rank == 0:
            print(
                f"[ERROR] num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
            )
        return res

    # Create mapping
    mapping = Mapping(
        rank=rank,
        tp_size=ep_size,
        moe_ep_size=ep_size,
        world_size=world_size,
    )

    # Create MoeAlltoAll instance
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        hidden_size=hidden_size,
    )

    # Synchronize all_num_tokens across ranks
    all_num_tokens = comm.allgather(num_tokens)
    runtime_max_tokens_per_rank = max(all_num_tokens)

    # Create input data
    torch.manual_seed(args.random_seed + rank)
    (
        hidden_states,
        hidden_states_original,
        token_selected_experts,
        token_final_scales,
        scale_factor,
        global_scale,
        input_payloads,
    ) = _create_moe_inputs(
        num_tokens,
        hidden_size,
        num_experts,
        top_k,
        input_dtype,
        quant_dtype,
        device,
        comm,
    )

    # Run validation if requested
    if getattr(args, "validate", False):
        if rank == 0:
            print("[INFO] Running validation before benchmarking...")

        validation_passed = _validate_moe_a2a(
            moe_a2a=moe_a2a,
            hidden_states=hidden_states,
            hidden_states_original=hidden_states_original,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            input_payloads=input_payloads,
            runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            hidden_size=hidden_size,
            input_dtype=input_dtype,
            quant_dtype=quant_dtype,
            global_scale=global_scale,
            num_experts=num_experts,
            ep_size=ep_size,
            rank=rank,
            comm=comm,
            verbose=args.verbose,
        )
        if not validation_passed:
            if rank == 0:
                print("[ERROR] Validation failed. Aborting benchmark.")
            return res

    # Storage for per-phase CUDA events to be populated later during benchmark
    # Deferred timing: collect events during iterations, compute times after single sync
    dispatch_events = []
    combine_events = []
    enable_nvtx = getattr(args, "nvtx", False)
    enable_per_phase_timing = getattr(args, "per_phase_timing", False)

    # Define benchmark function that accepts tensors as arguments
    # This enables automatic buffer rotation by bench_gpu_time
    def run_dispatch_combine(sel_experts, *payloads):
        # Dispatch phase: send (possibly quantized) hidden states to experts
        with (
            nvtx_range("moe_a2a_dispatch", enable_nvtx),
            cuda_event_timer(dispatch_events, enable_per_phase_timing),
        ):
            _ = moe_a2a.dispatch(
                sel_experts,
                list(payloads),
                runtime_max_tokens_per_rank,
            )

        # Simulate expert processing output
        # Dispatch sends quantized data (e.g., fp8/nvfp4),
        # Combine receives data in activation dtype (e.g., bfloat16/float16)
        with nvtx_range("moe_a2a_fake_math", enable_nvtx):
            combine_payload = moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank,
                hidden_size,
                input_dtype,
            )
            # TODO: add real math here if user prefers

        # Combine phase: gather processed outputs from all ranks
        with (
            nvtx_range("moe_a2a_combine", enable_nvtx),
            cuda_event_timer(combine_events, enable_per_phase_timing),
        ):
            output = moe_a2a.combine(
                combine_payload,
                runtime_max_tokens_per_rank,
                payload_in_workspace=True,
            )

        return output

    # Synchronize before benchmarking
    comm.Barrier()
    torch.cuda.synchronize()

    # Use bench_gpu_time with cold L2 cache
    total_times = bench_gpu_time(
        fn=run_dispatch_combine,
        input_args=(token_selected_experts, *input_payloads),
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        # Note: disable use_cuda_graph when per_phase_timing=True, which inserts CUDA events in the middle
        use_cuda_graph=(not args.no_cuda_graph and not enable_per_phase_timing),
        cold_l2_cache=True,
    )

    num_measure_iters = len(total_times)

    # Compute per-phase times if enabled
    if enable_per_phase_timing:
        # Per-phase events include dry runs; only use the last num_measure_iters entries
        dispatch_events_measure = dispatch_events[-num_measure_iters:]
        combine_events_measure = combine_events[-num_measure_iters:]

        # Convert events to times (no additional sync needed - bench_gpu_time already synced)
        dispatch_times = [s.elapsed_time(e) for s, e in dispatch_events_measure]
        combine_times = [s.elapsed_time(e) for s, e in combine_events_measure]
    else:
        dispatch_times = []
        combine_times = []

    # Gather times from all ranks
    all_total_times = comm.allgather(total_times)
    all_dispatch_times = comm.allgather(dispatch_times)
    all_combine_times = comm.allgather(combine_times)

    # Compute statistics from rank 0
    if rank == 0:
        # Use max time across ranks as the benchmark result
        # since communication is synchronous
        total_per_iter_max = [
            max(t[i] for t in all_total_times) for i in range(num_measure_iters)
        ]
        median_time = np.median(total_per_iter_max)
        std_time = np.std(total_per_iter_max)

        # Calculate total bandwidth
        tb_per_sec_total = _calculate_comm_bandwidth(
            num_tokens,
            hidden_size,
            top_k,
            ep_size,
            median_time,
            input_dtype,
            quant_dtype,
            phase="dispatch_combine",
        )

        # Per-phase statistics if enabled --per_phase_timing flag
        median_time_dispatch, std_time_dispatch = np.nan, np.nan
        median_time_combine, std_time_combine = np.nan, np.nan
        tb_per_sec_dispatch, tb_per_sec_combine = np.nan, np.nan
        if enable_per_phase_timing:
            dispatch_per_iter_max = [
                max(t[i] for t in all_dispatch_times) for i in range(num_measure_iters)
            ]
            combine_per_iter_max = [
                max(t[i] for t in all_combine_times) for i in range(num_measure_iters)
            ]
            median_time_dispatch = np.median(dispatch_per_iter_max)
            std_time_dispatch = np.std(dispatch_per_iter_max)
            median_time_combine = np.median(combine_per_iter_max)
            std_time_combine = np.std(combine_per_iter_max)

            tb_per_sec_dispatch = _calculate_comm_bandwidth(
                num_tokens,
                hidden_size,
                top_k,
                ep_size,
                median_time_dispatch,
                input_dtype,
                quant_dtype,
                phase="dispatch",
            )
            tb_per_sec_combine = _calculate_comm_bandwidth(
                num_tokens,
                hidden_size,
                top_k,
                ep_size,
                median_time_combine,
                input_dtype,
                quant_dtype,
                phase="combine",
            )

            # Print per-phase metrics
            print_perf_metrics(
                "a2a_dispatch",
                median_time_dispatch,
                std_time_dispatch,
                torch.nan,
                tb_per_sec_dispatch,
            )
            print_perf_metrics(
                "a2a_combine",
                median_time_combine,
                std_time_combine,
                torch.nan,
                tb_per_sec_combine,
            )

        # Always print total
        print_perf_metrics(
            "a2a_total", median_time, std_time, torch.nan, tb_per_sec_total
        )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["dispatch_time"] = median_time_dispatch
            cur_res["dispatch_std"] = std_time_dispatch
            cur_res["combine_time"] = median_time_combine
            cur_res["combine_std"] = std_time_combine
            cur_res["tflops"] = "N/A"
            cur_res["tb_per_sec"] = tb_per_sec_total
            cur_res["dispatch_tb_sec"] = tb_per_sec_dispatch
            cur_res["combine_tb_sec"] = tb_per_sec_combine
            cur_res["backend"] = "moe_a2a"
            cur_res["num_tokens"] = num_tokens
            cur_res["hidden_size"] = hidden_size
            cur_res["num_experts"] = num_experts
            cur_res["top_k"] = top_k
            cur_res["ep_size"] = ep_size
            cur_res["input_dtype"] = str(input_dtype)
            cur_res["quant_dtype"] = quant_dtype if quant_dtype else ""
            cur_res["max_num_tokens"] = max_num_tokens
            res.append(cur_res)

    return res
