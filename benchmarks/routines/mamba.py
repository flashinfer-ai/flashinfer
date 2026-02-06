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
"""

# ==============================================================================
# Triton reference implementation for selective_state_update.
# Copied from tests/mamba/selective_state_update_triton.py
# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/mamba_ssm.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
# ==============================================================================

from collections import defaultdict

import numpy as np
import torch
import triton
import triton.language as tl
from packaging import version

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    print_perf_metrics,
    filter_backends_by_compute_capability,
)

# ---- Triton reference kernel ----

PAD_SLOT_ID = -1

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {
        "CACHE_INTERMEDIATE_STATES": lambda args: args["intermediate_states_buffer"]
        is not None
    }
)
@triton.heuristics(
    {
        "HAS_EAGLE_TREE_CUSTOM_ATTN_MASK": lambda args: args[
            "retrieve_parent_token_ptr"
        ]
        is not None
    }
)
@triton.heuristics(
    {
        "HAS_INTERMEDIATE_STATE_INDICES": lambda args: args[
            "intermediate_state_indices_ptr"
        ]
        is not None
    }
)
@triton.jit(do_not_specialize=["T"])
def _selective_scan_update_kernel_reference(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    intermediate_states_buffer,
    cache_steps,
    retrieve_parent_token_ptr,
    intermediate_state_indices_ptr,
    # Matrix dimensions
    batch,
    T,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    stride_retrieve_parent_token_batch,
    stride_retrieve_parent_token_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    HAS_INTERMEDIATE_STATE_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        state_batch_idx = pid_b
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        if HAS_INTERMEDIATE_STATE_INDICES:
            intermediate_state_idx = tl.load(intermediate_state_indices_ptr + pid_b).to(
                tl.int64
            )
            cache_idx = intermediate_state_idx
        elif HAS_STATE_BATCH_INDICES:
            cache_idx = state_batch_idx
        else:
            cache_idx = pid_b

    current_step_idx = 0
    for _ in range(T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            if current_step_idx != 0 and cache_idx >= 0:
                parent_ptr = (
                    retrieve_parent_token_ptr
                    + pid_b * stride_retrieve_parent_token_batch
                    + current_step_idx * stride_retrieve_parent_token_T
                )
                parent_step_idx = tl.load(parent_ptr).to(tl.int32)

                if parent_step_idx >= 0 and parent_step_idx < T:
                    step_offset = parent_step_idx * nheads * dim * dstate
                    cache_ptr = (
                        intermediate_states_buffer
                        + cache_idx * cache_steps * nheads * dim * dstate
                        + step_offset
                        + pid_h * dim * dstate
                        + offs_m[:, None] * dstate
                        + offs_n[None, :]
                    )
                    state = tl.load(cache_ptr, mask=mask, other=0.0).to(tl.float32)

        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        if CACHE_INTERMEDIATE_STATES:
            if state_batch_idx != pad_slot_id:
                cache_ptr_base = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * nheads * dim * dstate
                    + current_step_idx * nheads * dim * dstate
                    + pid_h * dim * dstate
                )
                cache_ptrs = cache_ptr_base + (
                    offs_m[:, None] * dstate + offs_n[None, :]
                )
                tl.store(cache_ptrs, state.to(cache_ptrs.dtype.element_ty), mask=mask)

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        current_step_idx += 1  # noqa: SIM113

        x_ptr += stride_x_T
        dt_ptr += stride_dt_T
        B_ptr += stride_B_T
        C_ptr += stride_C_T
        out_ptr += stride_out_T
        if HAS_Z:
            z_ptr += stride_z_T

    if not DISABLE_STATE_UPDATE:
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)


def selective_state_update_triton_reference(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
):
    """
    Triton reference implementation of selective_state_update.

    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim) for single-token
           or (batch, T, nheads, dim) for multi-token
        dt: (batch, dim) or (batch, nheads, dim) for single-token
            or (batch, T, nheads, dim) for multi-token
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate) for single-token
           or (batch, T, ngroups, dstate) for multi-token
        C: (batch, dstate) or (batch, ngroups, dstate) for single-token
           or (batch, T, ngroups, dstate) for multi-token
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim) for single-token
           or (batch, T, nheads, dim) for multi-token
        dt_bias: (dim,) or (nheads, dim)
    """
    # Track original x dimensionality to squeeze output appropriately
    x_orig_dim = x.dim()

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is None:
        out = torch.empty_like(x)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out.shape == x.shape

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else (
            (16, 4)
            if dstate <= 32
            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8))))
        )
    )
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )

    retrieve_parent_token_strides = (
        (retrieve_parent_token.stride(0), retrieve_parent_token.stride(1))
        if retrieve_parent_token is not None
        else (0, 0)
    )

    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel_reference[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            pad_slot_id,
            intermediate_states_buffer,
            cache_steps if cache_steps is not None else 0,
            retrieve_parent_token,
            intermediate_state_indices,
            batch,
            T,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            retrieve_parent_token_strides[0],
            retrieve_parent_token_strides[1],
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            DISABLE_STATE_UPDATE=disable_state_update,
            num_warps=num_warps,
        )
    # Squeeze T dimension if original x didn't have it (was 2D or 3D)
    if x_orig_dim < 4:
        out = out.squeeze(1)
    return out


# ==============================================================================
# Benchmark infrastructure
# ==============================================================================


def run_mamba_test(args):
    """
    Run a mamba test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "selective_state_update":
        return testSelectiveStateUpdate(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_mamba_args(line, parser):
    """
    Parse command line arguments for mamba test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size (number of sequences).",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        required=True,
        help="Number of SSM heads.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Head dimension (headdim).",
    )
    parser.add_argument(
        "--dstate",
        type=int,
        required=True,
        help="SSM state size.",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        required=False,
        default=8,
        help="Number of groups for B and C matrices. nheads must be divisible by ngroups.",
    )
    parser.add_argument(
        "--cache_steps",
        type=int,
        required=False,
        default=0,
        help="Number of steps/tokens for multi-token prediction. 0 = single-token prediction.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16"],
        help="Data type for input tensors (x, B, C, z). Only bfloat16 is supported.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for the SSM state cache.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["bfloat16", "float32"],
        help="Data type for weight tensors (dt, D, dt_bias).",
    )
    parser.add_argument(
        "--has_z",
        action="store_true",
        default=False,
        help="Include z tensor for gating (z * sigmoid(z) applied to output).",
    )
    parser.add_argument(
        "--dt_softplus",
        action="store_true",
        default=False,
        help="Apply softplus to dt before use.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["flashinfer"],
        choices=["flashinfer", "triton"],
        help="Kernel backends to benchmark. Default: flashinfer",
    )

    args = parser.parse_args(line)

    # Validate nheads divisibility
    if args.nheads % args.ngroups != 0:
        raise ValueError(
            f"nheads ({args.nheads}) must be divisible by ngroups ({args.ngroups})."
        )

    # Validate nheads/ngroups ratio is supported by the CUDA kernel
    supported_ratios = [1, 8, 16]
    ratio = args.nheads // args.ngroups
    if ratio not in supported_ratios:
        raise ValueError(
            f"nheads/ngroups ratio ({ratio} = {args.nheads}/{args.ngroups}) is not supported by the FlashInfer kernel. "
            f"Supported ratios: {supported_ratios}."
        )

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testSelectiveStateUpdate(args):
    """
    Test selective_state_update API for Mamba layers.

    This test:
    1. Generates random input tensors for SSM state update
    2. Runs selective_state_update with the requested backend(s)
       - 'flashinfer': FlashInfer CUDA kernel (architecture-specific: base/SM90/SM100+)
       - 'triton': Triton reference implementation
    3. Optionally runs reference check (compares against Triton reference)
    4. Measures performance metrics (memory bandwidth)

    Supports both single-token prediction (STP, cache_steps=0) and
    multi-token prediction (MTP, cache_steps>=1) modes.

    Note: selective_state_update is memory-bandwidth bound, so TB/sec is the
    primary performance metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSelectiveStateUpdate")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    nheads = args.nheads
    dim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    cache_steps = args.cache_steps
    has_z = args.has_z
    dt_softplus = args.dt_softplus
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)
    ## Done parsing input arguments

    ## Determine STP vs MTP mode
    is_mtp = cache_steps >= 1
    T = cache_steps if is_mtp else None

    ## Prepare input tensors (mirrors tests/mamba/utils.py::create_test_inputs)
    ssm_state_cache_size = max(384, batch_size * 10)

    # State cache: (total_entries, nheads, dim, dstate) - contiguous
    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=state_dtype, device=device
    )

    # Input x: (batch_size, [T,] nheads, dim)
    if T is not None:
        x = torch.randn(batch_size, T, nheads, dim, dtype=input_dtype, device=device)
    else:
        x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    # dt: broadcasting across dim (one value per head)
    if T is not None:
        dt_base = torch.randn(batch_size, T, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided(
            (batch_size, T, nheads, dim), (T * nheads, nheads, 1, 0)
        )
    else:
        dt_base = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided((batch_size, nheads, dim), (nheads, 1, 0))

    # A: (nheads, dim, dstate) - negative values, broadcasting (one value per head)
    A_base = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # B, C: (batch_size, [T,] ngroups, dstate)
    if T is not None:
        B = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
        C = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
    else:
        B = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
        C = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)

    # D: (nheads, dim) - broadcasting (one value per head)
    D_base = torch.randn(nheads, dtype=weight_dtype, device=device)
    D = D_base.as_strided((nheads, dim), (1, 0))

    # dt_bias: (nheads, dim) - broadcasting (one value per head)
    dt_bias_base = torch.rand(nheads, dtype=weight_dtype, device=device) - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    # Slot indices for state batching
    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device=device)[
        :batch_size
    ]

    # Optional z tensor for gating
    z = None
    if has_z:
        if T is not None:
            z = torch.randn(
                batch_size, T, nheads, dim, dtype=input_dtype, device=device
            )
        else:
            z = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] Mode: {'MTP' if is_mtp else 'STP'}")
        print(f"[VVERBOSE] {state_cache.shape = }, {state_cache.dtype = }")
        print(f"[VVERBOSE] {x.shape = }, {x.dtype = }")
        print(f"[VVERBOSE] {dt.shape = }, {dt.dtype = }")
        print(f"[VVERBOSE] {A.shape = }, {A.dtype = }")
        print(f"[VVERBOSE] {B.shape = }, {B.dtype = }")
        print(f"[VVERBOSE] {C.shape = }, {C.dtype = }")
        print(f"[VVERBOSE] {D.shape = }, {D.dtype = }")
        print(f"[VVERBOSE] {dt_bias.shape = }, {dt_bias.dtype = }")
        print(f"[VVERBOSE] {slot_idx.shape = }")
        print(f"[VVERBOSE] {has_z = }, {dt_softplus = }")
        if z is not None:
            print(f"[VVERBOSE] {z.shape = }, {z.dtype = }")

    # Cache steps for Triton reference (None for STP, integer for MTP)
    triton_cache_steps = cache_steps if cache_steps > 0 else None

    def run_backend(backend, state, x, dt, A, B, C, D):
        if backend == "flashinfer":
            return flashinfer.mamba.selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=cache_steps,
            )
        elif backend == "triton":
            return selective_state_update_triton_reference(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=triton_cache_steps,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check: use Triton as golden reference
    # Save a clean snapshot of state_cache before any benchmarking, because
    # bench_gpu_time mutates state_cache in-place across many iterations.
    # All refcheck clones must come from this clean snapshot.
    has_reference_output = False
    clean_state_snapshot = state_cache.clone() if run_refcheck else None
    if run_refcheck:
        ref_state = clean_state_snapshot.clone()
        reference_output = (
            selective_state_update_triton_reference(
                ref_state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=triton_cache_steps,
            )
            .detach()
            .clone()
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck and cur_backend != "triton":
            # Always clone from the clean snapshot, not from state_cache
            # (which may have been mutated by previous backend's bench_gpu_time)
            fresh_state = clean_state_snapshot.clone()
            outputs[cur_backend] = (
                run_backend(cur_backend, fresh_state, x, dt, A, B, C, D)
                .detach()
                .clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, state_cache, x, dt, A, B, C, D),
        )

    # Compare outputs against Triton reference
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(
                    reference_output.float(),
                    tested_outputs[i].float(),
                    rtol=1e-2,
                    atol=1e-3,
                )
                # Allow up to 0.01% of elements to differ (floating-point edge cases)
                mismatch_threshold_pct = 0.01
                if num_different_elements_percentage > mismatch_threshold_pct:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
                        f"(threshold: {mismatch_threshold_pct}%)"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )
                elif num_different_elements > 0:
                    if args.verbose >= 1:
                        print(
                            f"[REFCHECK] Backend {tested_backends[i]}: PASSED "
                            f"({num_different_elements}/{num_elements} elements differ "
                            f"({num_different_elements_percentage:.4f}%), within {mismatch_threshold_pct}% threshold)"
                        )
                else:
                    if args.verbose >= 1:
                        print(
                            f"[REFCHECK] Backend {tested_backends[i]}: PASSED (all {num_elements} elements match)"
                        )

    # Compute and report performance metrics
    T_val = cache_steps if cache_steps > 0 else 1

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation (physical bytes accessed)
            # Read:
            #   state: batch_size * nheads * dim * dstate (via slot_idx indirection)
            #   x: batch_size * T * nheads * dim
            #   dt: batch_size * T * nheads (broadcast across dim)
            #   A: nheads (broadcast across dim and dstate)
            #   B: batch_size * T * ngroups * dstate
            #   C: batch_size * T * ngroups * dstate
            #   D: nheads (broadcast across dim)
            #   dt_bias: nheads (broadcast across dim)
            #   z (optional): batch_size * T * nheads * dim
            # Write:
            #   state: batch_size * nheads * dim * dstate
            #   output: batch_size * T * nheads * dim
            read_bytes = (
                batch_size * nheads * dim * dstate * state_dtype.itemsize  # state
                + batch_size * T_val * nheads * dim * input_dtype.itemsize  # x
                + batch_size * T_val * nheads * weight_dtype.itemsize  # dt (broadcast)
                + nheads * 4  # A (float32, broadcast)
                + batch_size * T_val * ngroups * dstate * input_dtype.itemsize  # B
                + batch_size * T_val * ngroups * dstate * input_dtype.itemsize  # C
                + nheads * weight_dtype.itemsize  # D (broadcast)
                + nheads * weight_dtype.itemsize  # dt_bias (broadcast)
            )
            if has_z:
                read_bytes += batch_size * T_val * nheads * dim * input_dtype.itemsize

            write_bytes = (
                batch_size * nheads * dim * dstate * state_dtype.itemsize  # state
                + batch_size * T_val * nheads * dim * input_dtype.itemsize  # output
            )

            problem_bytes = read_bytes + write_bytes

            # FLOPs estimate (TIE_HDIM case, where dt/A/D/dt_bias broadcast across dim):
            # Per (dim, dstate) element per (batch, T, head):
            #   state * dA: 1 mul, dB * x[:, None]: 1 mul, state + ...: 1 add,
            #   state * C[None, :]: 1 mul, sum reduction: ~1 add => 5 FLOPs/element
            problem_flops = batch_size * T_val * nheads * dim * dstate * 5
            tflops = problem_flops / (10**9 * median_time)  # TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                # Mamba-specific columns
                cur_res["nheads"] = nheads
                cur_res["dim"] = dim
                cur_res["dstate"] = dstate
                cur_res["ngroups"] = ngroups
                cur_res["cache_steps"] = cache_steps
                cur_res["state_dtype"] = str(state_dtype)
                cur_res["weight_dtype"] = str(weight_dtype)
                cur_res["has_z"] = has_z
                cur_res["dt_softplus"] = dt_softplus
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
