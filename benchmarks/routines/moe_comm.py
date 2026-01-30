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
    --quant_dtype fp8           : FP8 (float8_e4m3fn) with float32 per-tensor scale
    --quant_dtype nvfp4         : NVFP4 (4-bit) with float8_e4m3fn block scales
    --quant_dtype fp8_block_scale : FP8 with block scales (128 elements per block)
    --real_math                 : Run actual MoE kernels (trtllm_fp4/fp8_block_scale_moe)
                                  Supported quant_dtype: nvfp4 and fp8_block_scale only.
    --intermediate_size N       : FFN intermediate size; must be specified if real_math=True
    --validate                  : Run correctness validation for A2A before benchmarking.
                                  Uses a deterministic fake MoE to verify round-trip
                                  communication. For non-quantized mode, performs exact
                                  comparison. For quantized mode, validates output
                                  shape and numerical validity.
    --per_phase_timing          : Enable per-phase timing (dispatch/combine). Adds slight
                                  overhead from CUDA events.
                                  This is less accurate than the total timing but shows
                                  dispatch and combine times separately.
    --nvtx                      : Enable NVTX markers for Nsight Systems profiling.
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
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    print_perf_metrics,
)
from .moe_utils import (
    add_common_moe_args,
    calculate_fp4_global_scale,
    quantize_fp4,
    dequantize_nvfp4,
    quantize_fp8,
    quantize_fp8_block_scale,
    dequantize_fp8,
    dequantize_fp8_block_scale,
    pack_topk_ids_triton,
    calculate_moe_tflops,
    calculate_moe_kernel_bandwidth,
    generate_moe_weights,
    create_moe_output_scale_scalars,
    quantize_and_pack_nvfp4,
)


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
    # Parse num_tokens/hidden_size/num_experts/top_k/input_dtype in add_common_moe_args
    add_common_moe_args(parser)
    parser.add_argument(
        "--quant_dtype",
        type=str,
        required=False,
        default=None,
        choices=["fp8", "nvfp4", "fp8_block_scale"],
        help="Quantization format for hidden states. If set, hidden states are quantized and scale factors are communicated. fp8_block_scale: FP8 with block scales (128 elements per block).",
    )
    parser.add_argument(
        "--real_math",
        action="store_true",
        help="Runs actual MoE kernels (trtllm_(fp4|fp8)_block_scale_moe).",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=False,
        default=None,
        help="Intermediate size for each expert in MoE. Must be specified if real_math=True.",
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
        help="Enable NVTX markers for Nsight Systems profiling. This also turns on --use_cuda_events.",
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

    if args.real_math:
        # Must specify intermediate_size if real_math=True
        assert args.intermediate_size is not None, (
            "intermediate_size must be specified if real_math=True"
        )
        # Must specify quant_dtype as one of the following: nvfp4, fp8_block_scale
        # Other quant_dtype support is TBD
        assert args.quant_dtype in [
            "nvfp4",
            "fp8_block_scale",
        ], (
            f"real_math=True requires quant_dtype 'nvfp4' or 'fp8_block_scale', got '{args.quant_dtype}'"
        )

    # Derive scale_dtype from quant_dtype
    if args.quant_dtype == "nvfp4":
        args.scale_dtype = torch.float8_e4m3fn
    elif args.quant_dtype == "fp8":
        args.scale_dtype = torch.float32
    elif args.quant_dtype == "fp8_block_scale":
        args.scale_dtype = torch.float32  # Block scales are float32
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


def _init_moe_weights(
    num_experts_local: int,
    hidden_size: int,
    intermediate_size: int,
    quant_dtype: str,
    device: torch.device,
) -> dict:
    """
    Initialize MoE weights for MoE.
    The function does not do weight shuffling required for TRTLLM-Gen MoE as it serves benchmark purposes only.
    However, A2A with fake MoE kernel can be validated through flags (--validate).

    Args:
        num_experts_local: Number of local experts on this rank
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate FFN size
        quant_dtype: "nvfp4" or "fp8_block_scale"
        device: CUDA device

    Returns:
        Dictionary containing weights and scales for MoE computation
    """
    weights = {}

    # Create quantized weights
    # Note: Generate and quantize one expert at a time to avoid OOM with large expert counts
    # gemm1: [num_experts, 2*intermediate_size, hidden_size]
    # gemm2: [num_experts, hidden_size, intermediate_size]
    if quant_dtype == "nvfp4":
        # Create FP4 quantized weights

        # Quantize to FP4 using swizzled layout for weights
        sf_vec_size = 16
        use_ue8m0 = False
        is_sf_swizzled_layout = True

        gemm1_fp4_list = []
        gemm1_sf_list = []
        gemm1_global_sf_list = []
        gemm2_fp4_list = []
        gemm2_sf_list = []
        gemm2_global_sf_list = []
        for _ in range(num_experts_local):
            # Generate bf16 weights for this expert using shared utility
            w1_batch, w2_batch = generate_moe_weights(
                1, hidden_size, intermediate_size, device, dtype=torch.bfloat16
            )
            expert_w1_bf16 = w1_batch.squeeze(0)
            expert_w2_bf16 = w2_batch.squeeze(0)
            del w1_batch, w2_batch

            # Quantize gemm1 weights using moe_utils.quantize_fp4
            quantized, sf, global_sf = quantize_fp4(
                expert_w1_bf16,
                global_scale=None,
                use_ue8m0=use_ue8m0,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
            )
            gemm1_fp4_list.append(quantized.view(torch.uint8))
            gemm1_sf_list.append(sf.view(torch.float8_e4m3fn))
            gemm1_global_sf_list.append(global_sf)
            del expert_w1_bf16

            # Quantize gemm2 weights using moe_utils.quantize_fp4
            quantized, sf, global_sf = quantize_fp4(
                expert_w2_bf16,
                global_scale=None,
                use_ue8m0=use_ue8m0,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
            )

            # NOTE: the script chooses not to do weight shuffling as it is intended for benchmarks;
            # only A2A with fake MoE kernel is validated

            gemm2_fp4_list.append(quantized.view(torch.uint8))
            gemm2_sf_list.append(sf.view(torch.float8_e4m3fn))
            gemm2_global_sf_list.append(global_sf)
            del expert_w2_bf16

        # Stack and reshape
        weights["gemm1_weights"] = torch.stack(gemm1_fp4_list).reshape(
            num_experts_local, 2 * intermediate_size, hidden_size // 2
        )
        weights["gemm1_weights_scale"] = torch.stack(gemm1_sf_list).reshape(
            num_experts_local, 2 * intermediate_size, hidden_size // sf_vec_size
        )
        weights["gemm2_weights"] = torch.stack(gemm2_fp4_list).reshape(
            num_experts_local, hidden_size, intermediate_size // 2
        )
        weights["gemm2_weights_scale"] = torch.stack(gemm2_sf_list).reshape(
            num_experts_local, hidden_size, intermediate_size // sf_vec_size
        )

        # Scale scalars for output using shared utility
        (
            weights["output1_scale_scalar"],
            weights["output1_scale_gate_scalar"],
            weights["output2_scale_scalar"],
        ) = create_moe_output_scale_scalars(num_experts_local, device)

    elif quant_dtype == "fp8_block_scale":
        # Create FP8 block-scaled weights

        # Optionally shuffle weights using shared utility
        gemm1_weights = []
        gemm2_weights = []
        for _ in range(num_experts_local):
            # Generate bf16 weights for this expert using shared utility
            w1_batch, w2_batch = generate_moe_weights(
                1, hidden_size, intermediate_size, device, dtype=torch.bfloat16
            )
            expert_w1_bf16 = w1_batch.squeeze(0)
            expert_w2_bf16 = w2_batch.squeeze(0)
            del w1_batch, w2_batch

            expert_w1_fp8 = expert_w1_bf16.to(torch.float8_e4m3fn)
            expert_w2_fp8 = expert_w2_bf16.to(torch.float8_e4m3fn)
            del expert_w1_bf16, expert_w2_bf16  # Free memory immediately

            # NOTE: the script chooses not to do weight shuffling as it is intended for benchmarks;
            # only A2A with fake MoE kernel is validated

            gemm1_weights.append(expert_w1_fp8)
            gemm2_weights.append(expert_w2_fp8)
            del expert_w1_fp8, expert_w2_fp8  # Free memory immediately
        weights["gemm1_weights"] = torch.stack(gemm1_weights)
        weights["gemm2_weights"] = torch.stack(gemm2_weights)

        # Block scales: [num_experts, out_dim // 128, in_dim // 128]
        weights["gemm1_weights_scale"] = 2.0 * torch.ones(
            (num_experts_local, 2 * intermediate_size // 128, hidden_size // 128),
            device=device,
            dtype=torch.float32,
        )
        weights["gemm2_weights_scale"] = 2.0 * torch.ones(
            (num_experts_local, hidden_size // 128, intermediate_size // 128),
            device=device,
            dtype=torch.float32,
        )

    else:
        raise ValueError(f"Unsupported quant_dtype for real computation: {quant_dtype}")

    return weights


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
        quant_dtype: None, "fp8", "nvfp4", or "fp8_block_scale"
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
        local_global_scale = calculate_fp4_global_scale(hidden_states_original)
        synced_global_scale = comm.allreduce(
            local_global_scale.cpu().item(), op=MPI.MAX
        )
        global_scale = torch.tensor(
            synced_global_scale, dtype=torch.float32, device=device
        )
        hidden_states, scale_factor, global_scale = quantize_and_pack_nvfp4(
            hidden_states_original,
            global_scale,
            use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
    elif quant_dtype == "fp8":
        # Compute local amax, sync via max, then quantize with synced scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        local_amax = hidden_states_original.abs().max().float().item()
        synced_amax = comm.allreduce(local_amax, op=MPI.MAX)
        synced_scale = torch.tensor(
            synced_amax / fp8_max, dtype=torch.float32, device=device
        )
        hidden_states, global_scale = quantize_fp8(hidden_states_original, synced_scale)
    elif quant_dtype == "fp8_block_scale":
        # FP8 with block scales (128 elements per block)
        # Block scales shape: [hidden_size // 128, num_tokens]
        hidden_states, scale_factor = quantize_fp8_block_scale(
            hidden_states_original, block_size=128
        )
        # Transpose scale_factor to [num_tokens, hidden_size // 128] for A2A payload
        # A2A expects [num_tokens, *] shape for payloads
        scale_factor = scale_factor.transpose(0, 1).contiguous()
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


def _calculate_exact_comm_traffic(
    all_token_selected_experts: List[np.ndarray],
    num_experts: int,
    ep_size: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    quant_dtype: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Calculate exact inter-rank traffic from actual expert assignments.

    For MoE A2A, each token is sent to the rank that owns the selected expert.
    Assuming that expert `e` is owned by rank `e // num_experts_per_rank`.

    If a token selects multiple experts on the same remote rank,
    the data should be sent only once and counted accordingly.

    Args:
        all_token_selected_experts: List of expert assignments from all ranks [ep_size][num_tokens, top_k]
        num_experts: Total number of experts
        ep_size: Expert parallel size (number of ranks)
        hidden_size: Hidden dimension size
        input_dtype: Data type of hidden states (before quantization)
        quant_dtype: None, "fp8", or "nvfp4"

    Returns:
        Tuple of (dispatch_bytes, combine_bytes) for actual inter-rank traffic
    """
    num_experts_per_rank = num_experts // ep_size

    # Calculate per-element sizes based on quant_dtype
    if quant_dtype == "nvfp4":
        # NVFP4: 0.5 bytes per element + block scales
        hidden_bytes_per_token = hidden_size // 2
        scale_bytes_per_token = (hidden_size // 16) * 1  # float8_e4m3fn
    elif quant_dtype == "fp8":
        # FP8: 1 byte per element
        hidden_bytes_per_token = hidden_size * 1
        scale_bytes_per_token = 0
    else:
        # No quantization
        element_size = torch.tensor([], dtype=input_dtype).element_size()
        hidden_bytes_per_token = hidden_size * element_size
        scale_bytes_per_token = 0

    # Activation dtype element size for combine phase
    combine_element_bytes = torch.tensor([], dtype=input_dtype).element_size()
    # Expert IDs/scales are sent for each selection
    token_topk_id_and_weight_bytes = 4 + 4

    # Count unique inter-rank transfers
    total_dispatch_bytes = 0
    total_combine_bytes = 0

    for src_rank, experts_on_rank in enumerate(all_token_selected_experts):
        # experts_on_rank: [num_tokens, top_k]
        num_tokens = experts_on_rank.shape[0]
        top_k = experts_on_rank.shape[1]

        for token_idx in range(num_tokens):
            # Find unique destination ranks for this token
            dst_ranks_for_token = set()
            num_expert_ids_per_dst = {}  # Count expert selections per dst rank

            for k in range(top_k):
                expert_id = experts_on_rank[token_idx, k]
                dst_rank = expert_id // num_experts_per_rank

                if dst_rank != src_rank:
                    dst_ranks_for_token.add(dst_rank)
                    num_expert_ids_per_dst[dst_rank] = (
                        num_expert_ids_per_dst.get(dst_rank, 0) + 1
                    )

            # For each unique dst_rank, token is sent once
            for dst_rank in dst_ranks_for_token:
                num_experts_to_dst = num_expert_ids_per_dst[dst_rank]

                # Dispatch: hidden states sent once
                total_dispatch_bytes += (
                    hidden_bytes_per_token
                    + scale_bytes_per_token
                    + num_experts_to_dst * token_topk_id_and_weight_bytes
                )
                # Combine: one output per token per dst_rank
                total_combine_bytes += hidden_size * combine_element_bytes

    return total_dispatch_bytes, total_combine_bytes


def _calculate_comm_bandwidth(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    time_ms: float,
    input_dtype: torch.dtype,
    quant_dtype: Optional[str] = None,
    phase: str = "dispatch_combine",
    actual_traffic: Optional[Tuple[int, int]] = None,
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
        quant_dtype: None, "fp8", "nvfp4", or "fp8_block_scale"
        phase: "dispatch", "combine", or "dispatch_combine"
        actual_traffic: Optional tuple of (dispatch_bytes, combine_bytes) from actual routing.
                       If provided, uses exact traffic instead of uniform distribution estimate.

    Returns:
        Bandwidth in TB/sec
    """
    if actual_traffic is not None:
        # Use actual traffic from expert routing analysis
        dispatch_bytes, combine_bytes = actual_traffic
        if phase == "dispatch":
            total_bytes = dispatch_bytes
        elif phase == "combine":
            total_bytes = combine_bytes
        else:  # dispatch_combine
            total_bytes = dispatch_bytes + combine_bytes
    else:
        # Estimate assuming uniform distribution (fallback)
        if quant_dtype == "nvfp4":
            # NVFP4: 0.5 bytes per element (packed uint8) + block scales
            hidden_states_bytes = num_tokens * hidden_size // 2  # packed FP4
            # Block scales: float8_e4m3fn, one per 16 elements
            scale_bytes = num_tokens * (hidden_size // 16) * 1  # float8_e4m3fn = 1 byte
        elif quant_dtype == "fp8":
            # FP8: 1 byte per element, no scale payload in A2A
            hidden_states_bytes = num_tokens * hidden_size * 1  # float8_e4m3fn = 1 byte
            scale_bytes = 0
        elif quant_dtype == "fp8_block_scale":
            # FP8 with block scales: 1 byte per element + block scales (128 elements per block)
            hidden_states_bytes = num_tokens * hidden_size * 1  # float8_e4m3fn = 1 byte
            # Block scales: float32, one per 128 elements
            scale_bytes = num_tokens * (hidden_size // 128) * 4  # float32 = 4 bytes
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
        element_size = torch.tensor([], dtype=input_dtype).element_size()
        combine_bytes = num_tokens * hidden_size * element_size

        if phase == "dispatch":
            total_bytes = dispatch_bytes
        elif phase == "combine":
            total_bytes = combine_bytes
        else:  # dispatch_combine
            total_bytes = dispatch_bytes + combine_bytes

        # Account for multi-rank communication: assume uniform distribution
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
            results.append(hidden_states[token_idx].to(torch.float32) * scale)

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
            recv_hidden_dequant[i] = dequantize_nvfp4(
                recv_hidden[i],
                recv_scale_factor[i],
                global_scale,
                block_size=16,
                dtype=input_dtype,
            )
    elif quant_dtype == "fp8":
        for i in range(recv_hidden.shape[0]):
            recv_hidden_dequant[i] = dequantize_fp8(
                recv_hidden[i],
                global_scale,
                dtype=input_dtype,
            )
    elif quant_dtype == "fp8_block_scale":
        for i in range(recv_hidden.shape[0]):
            # Transpose scales from [max_tokens, hidden_size // block_size] to
            # [hidden_size // block_size, max_tokens]
            scales_transposed = recv_scale_factor[i].transpose(0, 1).contiguous()
            recv_hidden_dequant[i] = dequantize_fp8_block_scale(
                recv_hidden[i],
                scales_transposed,
                block_size=128,
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
    elif quant_dtype == "fp8_block_scale":
        atol, rtol = 0.5, 0.1  # FP8 block scale: slightly worse than per-tensor FP8
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
    sum_all_num_tokens = sum(all_num_tokens)

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

    # Gather expert assignments from all ranks to calculate actual traffic
    all_token_selected_experts = comm.allgather(token_selected_experts.cpu().numpy())
    dispatch_bytes, combine_bytes = _calculate_exact_comm_traffic(
        all_token_selected_experts,
        num_experts,
        ep_size,
        hidden_size,
        input_dtype,
        quant_dtype,
    )
    # Compute total active experts across all ranks
    total_active_experts = int(
        np.unique(np.concatenate(all_token_selected_experts).flatten()).size
    )
    if rank == 0 and args.verbose >= 1:
        print(
            f"[INFO] Inter-rank traffic: dispatch={dispatch_bytes / 1024**2:.3f} MiB, combine={combine_bytes / 1024**2:.3f} MiB"
        )

    # Storage for per-phase CUDA events to be populated later during benchmark
    # Deferred timing: collect events during iterations, compute times after single sync
    dispatch_events = []
    combine_events = []
    moe_events = []  # For MoE kernel timing (excluding packing)
    enable_nvtx = getattr(args, "nvtx", False)
    if enable_nvtx:
        # CUPTI complains subscribers when using CUPTI for timing and nsys profiling at the same time
        args.use_cuda_events = True
    enable_per_phase_timing = getattr(args, "per_phase_timing", False)
    enable_real_math = getattr(args, "real_math", False)
    intermediate_size = getattr(args, "intermediate_size", None)
    num_experts_local = num_experts // ep_size

    if enable_real_math:
        assert intermediate_size is not None, (
            "intermediate_size must be specified if -real_math=True"
        )

    # Initialize MoE weights for real computation mode
    moe_weights = None
    if enable_real_math:
        if quant_dtype not in ["nvfp4", "fp8_block_scale"]:
            if rank == 0:
                print(
                    f"[ERROR] Real MoE math requires quant_dtype 'nvfp4' or 'fp8_block_scale', got '{quant_dtype}'"
                )
            return res
        moe_weights = _init_moe_weights(
            num_experts_local=num_experts_local,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_dtype=quant_dtype,
            device=device,
        )

    # Define benchmark function that accepts tensors as arguments
    # This enables automatic buffer rotation by bench_gpu_time
    def run_dispatch_combine(sel_experts, *payloads):
        # Dispatch phase: send (possibly quantized) hidden states to experts
        with (
            nvtx_range("moe_a2a_dispatch", enable_nvtx),
            cuda_event_timer(dispatch_events, enable_per_phase_timing),
        ):
            recv_tensors = moe_a2a.dispatch(
                sel_experts,
                list(payloads),
                runtime_max_tokens_per_rank,
            )

        # Expert processing in benchmark runs either no-op or real MoE kernel depending on --real_math flag
        with nvtx_range("moe_compute", enable_nvtx):
            combine_payload = moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank,
                hidden_size,
                input_dtype,
            )

            if enable_real_math and moe_weights is not None:
                # Real computation using actual MoE kernels
                # recv_tensors[0]: the received hidden states [ep_size, max_tokens, hidden_size]
                # recv_tensors[1]: the received expert IDs [ep_size, max_tokens, top_k]
                # recv_tensors[2]: the received token final scales [ep_size, max_tokens]
                # recv_tensors[3]: the received scale factor [ep_size, max_tokens, hidden_size // block_size]
                recv_hidden = recv_tensors[0]
                recv_experts = recv_tensors[1]
                recv_token_final_scales = recv_tensors[2]
                recv_scale_factor = recv_tensors[3] if len(recv_tensors) > 3 else None

                # Flatten for MoE kernel: [ep_size * max_tokens, hidden_size]
                total_tokens = ep_size * runtime_max_tokens_per_rank

                if quant_dtype == "nvfp4":
                    # Reshape hidden states for FP4 kernel
                    hidden_flat = recv_hidden.reshape(total_tokens, -1)
                    # Reshape scale factors
                    scale_flat = (
                        recv_scale_factor.reshape(total_tokens, -1)
                        if recv_scale_factor is not None
                        else None
                    )

                    # Pack expert IDs with actual routing weights using fused Triton kernel
                    local_expert_offset = rank * num_experts_local
                    recv_experts_flat = recv_experts.reshape(total_tokens, top_k)
                    recv_weights_flat = recv_token_final_scales.reshape(
                        total_tokens, top_k
                    )
                    packed_topk_ids = pack_topk_ids_triton(
                        recv_experts_flat,
                        recv_weights_flat,
                        local_expert_offset,
                    )

                    # Run block scale routed MoE
                    with cuda_event_timer(moe_events, enable_per_phase_timing):
                        trtllm_fp4_block_scale_routed_moe(
                            topk_ids=packed_topk_ids,
                            routing_bias=None,
                            hidden_states=hidden_flat,
                            hidden_states_scale=scale_flat,
                            gemm1_weights=moe_weights["gemm1_weights"],
                            gemm1_weights_scale=moe_weights["gemm1_weights_scale"],
                            gemm1_bias=None,
                            gemm1_alpha=None,
                            gemm1_beta=None,
                            gemm1_clamp_limit=None,
                            gemm2_weights=moe_weights["gemm2_weights"],
                            gemm2_weights_scale=moe_weights["gemm2_weights_scale"],
                            gemm2_bias=None,
                            output1_scale_scalar=moe_weights["output1_scale_scalar"],
                            output1_scale_gate_scalar=moe_weights[
                                "output1_scale_gate_scalar"
                            ],
                            output2_scale_scalar=moe_weights["output2_scale_scalar"],
                            num_experts=num_experts_local,
                            top_k=top_k,
                            n_group=None,
                            topk_group=None,
                            intermediate_size=intermediate_size,
                            local_expert_offset=0,
                            local_num_experts=num_experts_local,
                            routed_scaling_factor=None,
                            routing_method_type=1,  # Renormalize: TopK -> Softmax
                            output=combine_payload.view(total_tokens, hidden_size),
                        )

                elif quant_dtype == "fp8_block_scale":
                    # Reshape for FP8 block scale kernel
                    hidden_flat = recv_hidden.reshape(total_tokens, hidden_size)
                    # Transpose scale for kernel: [hidden_size // 128, total_tokens]
                    if recv_scale_factor is not None:
                        scale_flat = (
                            recv_scale_factor.reshape(total_tokens, -1)
                            .transpose(0, 1)
                            .contiguous()
                        )

                    # Pack expert IDs with actual routing weights using fused Triton kernel
                    local_expert_offset = rank * num_experts_local
                    recv_experts_flat = recv_experts.reshape(total_tokens, top_k)
                    recv_weights_flat = recv_token_final_scales.reshape(
                        total_tokens, top_k
                    )
                    packed_topk_ids = pack_topk_ids_triton(
                        recv_experts_flat,
                        recv_weights_flat,
                        local_expert_offset,
                    )

                    # Convert hidden states to FP8
                    hidden_fp8 = hidden_flat.to(torch.float8_e4m3fn)

                    # Run block scale routed MoE
                    with cuda_event_timer(moe_events, enable_per_phase_timing):
                        trtllm_fp8_block_scale_routed_moe(
                            topk_ids=packed_topk_ids,
                            routing_bias=None,
                            hidden_states=hidden_fp8,
                            hidden_states_scale=scale_flat,
                            gemm1_weights=moe_weights["gemm1_weights"],
                            gemm1_weights_scale=moe_weights["gemm1_weights_scale"],
                            gemm2_weights=moe_weights["gemm2_weights"],
                            gemm2_weights_scale=moe_weights["gemm2_weights_scale"],
                            num_experts=num_experts_local,
                            top_k=top_k,
                            n_group=0,
                            topk_group=0,
                            intermediate_size=intermediate_size,
                            local_expert_offset=0,
                            local_num_experts=num_experts_local,
                            routed_scaling_factor=None,
                            routing_method_type=1,  # Renormalize: TopK -> Softmax
                            use_shuffled_weight=False,
                            weight_layout=int(WeightLayout.MajorK),
                            enable_pdl=True,
                            output=combine_payload.view(total_tokens, hidden_size),
                        )

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
        moe_events_measure = moe_events[-num_measure_iters:] if moe_events else []

        # Convert events to times (no additional sync needed - bench_gpu_time already synced)
        dispatch_times = [s.elapsed_time(e) for s, e in dispatch_events_measure]
        combine_times = [s.elapsed_time(e) for s, e in combine_events_measure]
        moe_times = (
            [s.elapsed_time(e) for s, e in moe_events_measure]
            if moe_events_measure
            else []
        )
    else:
        dispatch_times = []
        combine_times = []
        moe_times = []

    # Gather times from all ranks
    all_total_times = comm.allgather(total_times)
    all_dispatch_times = comm.allgather(dispatch_times)
    all_combine_times = comm.allgather(combine_times)
    all_moe_times = comm.allgather(moe_times)

    # Compute statistics from rank 0
    if rank == 0:
        # Use max time across ranks as the benchmark result
        # since communication is synchronous
        total_per_iter_max = [
            max(t[i] for t in all_total_times) for i in range(num_measure_iters)
        ]
        median_time = np.median(total_per_iter_max)
        std_time = np.std(total_per_iter_max)

        # Calculate total bandwidth using actual traffic from expert routing
        tb_per_sec_total = _calculate_comm_bandwidth(
            num_tokens,
            hidden_size,
            top_k,
            ep_size,
            median_time,
            input_dtype,
            quant_dtype,
            phase="dispatch_combine",
            actual_traffic=(dispatch_bytes, combine_bytes),
        )

        # Per-phase statistics if enabled --per_phase_timing flag
        median_time_dispatch, std_time_dispatch = np.nan, np.nan
        median_time_combine, std_time_combine = np.nan, np.nan
        median_time_moe, std_time_moe = np.nan, np.nan
        tb_per_sec_dispatch, tb_per_sec_combine = np.nan, np.nan
        tflops_moe, tb_per_sec_moe = np.nan, np.nan
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

            # MoE timing is only available when real_math is enabled
            if all_moe_times and all_moe_times[0]:
                moe_per_iter_max = [
                    max(t[i] for t in all_moe_times) for i in range(num_measure_iters)
                ]
                median_time_moe = np.median(moe_per_iter_max)
                std_time_moe = np.std(moe_per_iter_max)

            tb_per_sec_dispatch = _calculate_comm_bandwidth(
                num_tokens,
                hidden_size,
                top_k,
                ep_size,
                median_time_dispatch,
                input_dtype,
                quant_dtype,
                phase="dispatch",
                actual_traffic=(dispatch_bytes, combine_bytes),
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
                actual_traffic=(dispatch_bytes, combine_bytes),
            )

            # Print per-phase metrics
            print_perf_metrics(
                "a2a_dispatch",
                median_time_dispatch,
                std_time_dispatch,
                torch.nan,
                tb_per_sec_dispatch,
            )
            # Only print MoE timing when real_math is enabled
            if args.real_math:
                # This is the total FLOPS of all ranks, not per rank
                tflops_moe = calculate_moe_tflops(
                    sum_all_num_tokens,
                    hidden_size,
                    intermediate_size,
                    num_experts,  # Actually not used
                    top_k,
                    median_time_moe,
                )
                # This is the total bandwidth of all ranks, not per rank
                tb_per_sec_moe = calculate_moe_kernel_bandwidth(
                    sum_all_num_tokens,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    top_k,
                    median_time_moe,
                    input_dtype,
                    input_dtype,
                    input_format=quant_dtype,
                    weight_format=quant_dtype,
                    routing_logits_dtype=None,  # No routing logits in routed MoE
                    active_experts=total_active_experts,
                )
                print_perf_metrics(
                    "moe_kernel",
                    median_time_moe,
                    std_time_moe,
                    tflops_moe,
                    tb_per_sec_moe,
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
        print(
            "[INFO] The reported achieved tflops/tb_per_sec is the aggregate FLOPS/bandwidth of all participating ranks based on timing results of rank 0. Could observe rank-to-rank variations."
        )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["dispatch_time"] = median_time_dispatch
            cur_res["dispatch_std"] = std_time_dispatch
            cur_res["moe_time"] = median_time_moe
            cur_res["moe_std"] = std_time_moe
            cur_res["moe_tflops"] = tflops_moe
            cur_res["moe_tb_per_sec"] = tb_per_sec_moe
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
