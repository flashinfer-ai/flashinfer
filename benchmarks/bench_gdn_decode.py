"""
Copyright (c) 2025 by FlashInfer team.

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

"""
GDN Decode Benchmark

This benchmark supports:
1. All layouts comparison (default for decode): FlashInfer/Triton x pretranspose/nontranspose + bf16_state
2. Single layout comparison: FlashInfer (CuTe DSL) vs Triton kernel (--compare)
3. MTP benchmark (--version mtp)
4. BF16 state benchmark (--version bf16_state) for T=1 and MTP T>=1

Kernels benchmarked:
- FlashInfer Pretranspose [B, HV, V, K] (V-major layout)
- FlashInfer Nontranspose [B, HV, K, V] (K-major layout)
- Triton Pretranspose [B, HV, V, K]
- Triton Nontranspose [B, HV, K, V]
- BF16 State [B, HV, V, K] (K-fast layout, bf16 state, T=1 + MTP)

Usage:
    # Default: All layouts comparison
    python benchmarks/bench_gdn_decode.py --batch-size 1 4 8 16 32 64 128 256 512

    # Single layout comparison: FlashInfer vs Triton
    python benchmarks/bench_gdn_decode.py --compare --batch-size 1 4 8 16 32 64 128 256 512

    # MTP benchmark (FlashInfer only)
    python benchmarks/bench_gdn_decode.py --version mtp --batch-size 1 32 128

    # MTP comparison: FlashInfer vs Triton
    python benchmarks/bench_gdn_decode.py --version mtp --compare --batch-size 1 32 128

    # BF16 state benchmark (T=1 and MTP)
    python benchmarks/bench_gdn_decode.py --version bf16_state --batch-size 1 32 128 512

    # Use Qwen3-Next preset (q=k=16, v=32, d=128)
    python benchmarks/bench_gdn_decode.py --preset qwen3-next --batch-size 1 32 128 512
"""

import argparse
import numpy as np
import torch

from flashinfer.gdn_decode import (
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_decode,
    gated_delta_rule_mtp,
)
from flashinfer.testing import bench_gpu_time

# Import BF16 state kernels for benchmarking
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as gdn_decode_bf16_state,
        gated_delta_rule_mtp as gdn_decode_bf16_state_mtp,
    )

    GDN_DECODE_BF16_STATE_AVAILABLE = True
except ImportError:
    GDN_DECODE_BF16_STATE_AVAILABLE = False

# ============================================================================
# Utility Functions
# ============================================================================


def gdn_decode_flops(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int = 1,
) -> int:
    """
    Calculate FLOPs for Gated Delta Rule (GDN).

    Supports both decode (seq_len=1) and MTP (seq_len>1).

    Delta Rule formula (per token):
        g = -exp(A_log) * softplus(a + dt_bias)           # Log-space decay
        beta = sigmoid(b)                                  # Update gate
        state = state * exp(g)                             # State decay
        v_new = v - k @ state                              # Prediction error
        state = state + beta * k^T @ v_new                 # State update
        output = q @ state                                 # Output projection

    Matrix multiplications per token per head:
    1. k @ state: 2 * K * V FLOPs (for each head)
    2. k^T @ v_new (outer product): 2 * K * V FLOPs
    3. q @ state: 2 * K * V FLOPs

    Total per head: 6 * K * V FLOPs
    Note: K = V = head_size for GDN
    """
    num_o_heads = max(num_q_heads, num_v_heads)

    # Per token per head: 6 * d^2 FLOPs (d = head_size)
    # Total: seq_len * batch_size * num_heads * 6 * d^2
    total_flops = 6 * seq_len * batch_size * num_o_heads * head_size * head_size
    return total_flops


def gdn_decode_bytes(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seq_len: int = 1,
    disable_state_update: bool = False,
    state_dtype_bytes: int = 4,  # 4 for FP32, 2 for BF16
    cache_intermediate_states: bool = False,
) -> int:
    """
    Calculate memory bytes for GDN.

    Supports both decode (seq_len=1) and MTP (seq_len>1).

    Includes:
    - Q, K, V tensors (input): [B, T, H, K] - dtype
    - State tensor (input/output): [B, HV, K, V] - state_dtype_bytes (FP32=4 or BF16=2)
    - Intermediate states (MTP only): [B, T, HV, K, V] - state_dtype_bytes
    - GDN parameters: A_log (float32), a (dtype), dt_bias (dtype), b (dtype)
    - Output tensor: [B, T, HV, V] - dtype

    Note: When disable_state_update=True, state is only read, not written back.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = dtype.itemsize

    # Input tensors: [B, T, H, K]
    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_k_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size

    # Output tensor: [B, T, HV, V]
    o_bytes = batch_size * seq_len * num_o_heads * head_size * elem_size

    # State tensor: [B, HV, K, V]
    # If disable_state_update=True: only read initial state
    # If disable_state_update=False: read initial + write final state
    if disable_state_update:
        # Read only (e.g., MTP verify mode)
        state_bytes = (
            batch_size * num_sab_heads * head_size * head_size * state_dtype_bytes
        )
    else:
        # Read + write (e.g., normal decode)
        state_bytes = (
            2 * batch_size * num_sab_heads * head_size * head_size * state_dtype_bytes
        )

    # GDN parameters
    # A_log: [HV] - float32
    A_log_bytes = num_sab_heads * 4
    # a: [B, T, HV] - dtype
    a_bytes = batch_size * seq_len * num_sab_heads * elem_size
    # dt_bias: [HV] - dtype
    dt_bias_bytes = num_sab_heads * elem_size
    # b: [B, T, HV] - dtype
    b_bytes = batch_size * seq_len * num_sab_heads * elem_size

    # Intermediate states: [B, T, HV, K, V] - only written when MTP
    # intermediate-state caching is enabled
    intermediate_bytes = 0
    if cache_intermediate_states and seq_len > 1:
        intermediate_bytes = (
            batch_size
            * seq_len
            * num_sab_heads
            * head_size
            * head_size
            * state_dtype_bytes
        )

    total_bytes = (
        q_bytes
        + k_bytes
        + v_bytes
        + o_bytes
        + state_bytes
        + intermediate_bytes
        + A_log_bytes
        + a_bytes
        + dt_bias_bytes
        + b_bytes
    )
    return total_bytes


# ============================================================================
# Triton Kernels for comparison benchmarks
# ============================================================================
# The Triton GDN kernels live in benchmarks/gdn_triton_reference.py (same
# directory as this script) so they can be shared with the
# flashinfer_benchmark.py GDN routines (benchmarks/routines/gdn.py).

from gdn_triton_reference import (
    TRITON_AVAILABLE,
    triton_gdn_decode,
    triton_gdn_decode_pretranspose,
    triton_gdn_mtp,
)


# ============================================================================
# FlashInfer-only Benchmark Functions
# ============================================================================


def bench_gdn_decode(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    version: str = "nontranspose",
    use_alpha: bool = True,
    use_beta: bool = True,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark GDN decode kernel using bench_gpu_time with CUPTI.

    Args:
        version: 'pretranspose' or 'nontranspose'
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_alpha
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_beta
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )

    # Initial state - layout depends on version
    # Both versions use [B, HV, head_size, head_size]
    # Pretranspose interprets as [B, HV, V, K] (v-major)
    # Nontranspose interprets as [B, HV, K, V] (k-major)
    state = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # Pre-allocate output
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Select API function based on version
    if version == "pretranspose":
        decode_func = gated_delta_rule_decode_pretranspose
    elif version == "nontranspose":
        decode_func = gated_delta_rule_decode
    else:
        raise ValueError(f"Unknown version: {version}")

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: decode_func(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size
    )
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len=1,
        disable_state_update=False,  # Decode mode: state is read + written
    )

    kernel_tflops = flops / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    kernel_tb_per_sec = (
        bytes_accessed / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "kernel_median_us": kernel_median_ms * 1000,
        "kernel_tflops": kernel_tflops,
        "kernel_tb_per_sec": kernel_tb_per_sec,
    }


def bench_gdn_mtp(
    batch_size: int,
    seq_len: int,  # T > 1
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_alpha: bool = True,
    use_beta: bool = True,
    use_qk_l2norm: bool = True,
    cache_intermediate_states: bool = True,
    disable_state_update: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    ssm_state_indices_mode: str = "none",
):
    """Benchmark GDN MTP kernel using bench_gpu_time with CUPTI."""
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T > 1 for MTP)
    T = seq_len
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_alpha
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_beta
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )

    # FLA-style per-token pool scatter (vLLM API compat).
    assert ssm_state_indices_mode in ("none", "unique"), (
        f"ssm_state_indices_mode must be 'none' or 'unique', got "
        f"{ssm_state_indices_mode!r}"
    )
    fla_scatter = ssm_state_indices_mode != "none"
    if fla_scatter:
        assert T >= 2, f"--ssm-state-indices requires seq_len>=2, got T={T}"
        assert not cache_intermediate_states, (
            "--ssm-state-indices is mutex with --cache (no intermediate buffer)"
        )
        assert not disable_state_update, (
            "--ssm-state-indices requires state writes; pass --update-state"
        )

    # Initial state: [pool_size, HV, V, K] (K-last layout for MTP)
    # FLA scatter needs an extra B*T slots for the per-token scatter destinations.
    base_pool_size = batch_size
    fla_extra_slots = batch_size * T if fla_scatter else 0
    pool_size = base_pool_size + fla_extra_slots
    initial_state = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    # FLA-style scatter destinations: B*T fresh slots at the tail of the pool.
    ssm_state_indices_tensor = None
    if fla_scatter:
        ssm_state_indices_tensor = torch.arange(
            base_pool_size,
            base_pool_size + batch_size * T,
            dtype=torch.int32,
            device="cuda",
        ).reshape(batch_size, T)

    # Intermediate states buffer (optional)
    if cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            pool_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        intermediate_states_buffer = None

    # Pre-allocate output
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: gated_delta_rule_mtp(
            q,
            k,
            v,
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            output,
            intermediate_states_buffer,
            ssm_state_indices=ssm_state_indices_tensor,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, seq_len
    )
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
    )

    kernel_tflops = flops / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    kernel_tb_per_sec = (
        bytes_accessed / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "kernel_median_us": kernel_median_ms * 1000,
        "kernel_tflops": kernel_tflops,
        "kernel_tb_per_sec": kernel_tb_per_sec,
    }


# ============================================================================
# Comparison Benchmark Functions (FlashInfer vs Triton)
# ============================================================================


def bench_comparison(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark both FlashInfer and Triton implementations."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # ========== FlashInfer Benchmark ==========
    # State for FlashInfer (K-major layout) [B, HV, K, V]
    state_fi = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_fi = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    flashinfer_times = bench_gpu_time(
        lambda: gated_delta_rule_decode(
            q, k, v, state_fi, A_log, a, dt_bias, b, scale, output_fi, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    flashinfer_median_us = np.median(flashinfer_times) * 1000

    # ========== Triton Benchmark ==========
    # State [B, HV, K, V]
    state_tr = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_tr = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    triton_times = bench_gpu_time(
        lambda: triton_gdn_decode(
            q, k, v, state_tr, A_log, a, dt_bias, b, scale, output_tr, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    triton_median_us = np.median(triton_times) * 1000

    # Calculate metrics
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size
    )

    flashinfer_tflops = (
        flops / (flashinfer_median_us / 1000) / 1e9 if flashinfer_median_us > 0 else 0
    )
    triton_tflops = (
        flops / (triton_median_us / 1000) / 1e9 if triton_median_us > 0 else 0
    )

    speedup = triton_median_us / flashinfer_median_us if flashinfer_median_us > 0 else 0

    return {
        "batch_size": batch_size,
        "flashinfer_us": flashinfer_median_us,
        "triton_us": triton_median_us,
        "flashinfer_tflops": flashinfer_tflops,
        "triton_tflops": triton_tflops,
        "speedup": speedup,
    }


def bench_comparison_pretranspose(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark both FlashInfer and Triton pretranspose implementations."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # ========== FlashInfer Benchmark ==========
    # State for FlashInfer pretranspose (V-major layout) [B, HV, V, K]
    state_fi = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_fi = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    flashinfer_times = bench_gpu_time(
        lambda: gated_delta_rule_decode_pretranspose(
            q, k, v, state_fi, A_log, a, dt_bias, b, scale, output_fi, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    flashinfer_median_us = np.median(flashinfer_times) * 1000

    # ========== Triton Benchmark ==========
    # State [B, HV, V, K] - pretranspose layout
    state_tr = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_tr = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    triton_times = bench_gpu_time(
        lambda: triton_gdn_decode_pretranspose(
            q, k, v, state_tr, A_log, a, dt_bias, b, scale, output_tr, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    triton_median_us = np.median(triton_times) * 1000

    # Calculate metrics
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size
    )

    flashinfer_tflops = (
        flops / (flashinfer_median_us / 1000) / 1e9 if flashinfer_median_us > 0 else 0
    )
    triton_tflops = (
        flops / (triton_median_us / 1000) / 1e9 if triton_median_us > 0 else 0
    )

    speedup = triton_median_us / flashinfer_median_us if flashinfer_median_us > 0 else 0

    return {
        "batch_size": batch_size,
        "flashinfer_us": flashinfer_median_us,
        "triton_us": triton_median_us,
        "flashinfer_tflops": flashinfer_tflops,
        "triton_tflops": triton_tflops,
        "speedup": speedup,
    }


def bench_mtp_comparison(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    cache_intermediate_states: bool = False,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark both FlashInfer and Triton MTP implementations."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs
    T = seq_len
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Pool size = batch size for this benchmark
    pool_size = batch_size

    # ========== FlashInfer Benchmark ==========
    # State for FlashInfer (K-last layout) [pool_size, HV, V, K]
    state_fi = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_fi = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    # Intermediate states buffer
    if cache_intermediate_states:
        intermediate_fi = torch.zeros(
            pool_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        intermediate_fi = None

    flashinfer_times = bench_gpu_time(
        lambda: gated_delta_rule_mtp(
            q,
            k,
            v,
            state_fi,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            output_fi,
            intermediate_fi,
            disable_state_update=False,
            use_qk_l2norm=use_qk_l2norm,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    flashinfer_median_us = np.median(flashinfer_times) * 1000

    # ========== Triton Benchmark ==========
    # State for Triton [pool_size, HV, V, K]
    state_tr = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output_tr = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    if cache_intermediate_states:
        intermediate_tr = torch.zeros(
            pool_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        intermediate_tr = None

    triton_times = bench_gpu_time(
        lambda: triton_gdn_mtp(
            q,
            k,
            v,
            state_tr,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            output_tr,
            intermediate_tr,
            disable_state_update=False,
            use_qk_l2norm=use_qk_l2norm,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )
    triton_median_us = np.median(triton_times) * 1000

    # Calculate metrics
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, seq_len
    )

    flashinfer_tflops = (
        flops / (flashinfer_median_us / 1000) / 1e9 if flashinfer_median_us > 0 else 0
    )
    triton_tflops = (
        flops / (triton_median_us / 1000) / 1e9 if triton_median_us > 0 else 0
    )

    speedup = triton_median_us / flashinfer_median_us if flashinfer_median_us > 0 else 0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "flashinfer_us": flashinfer_median_us,
        "triton_us": triton_median_us,
        "flashinfer_tflops": flashinfer_tflops,
        "triton_tflops": triton_tflops,
        "speedup": speedup,
    }


def verify_correctness(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """Verify FlashInfer and Triton produce similar results."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Same initial state for both
    state_init = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # FlashInfer
    state_fi = state_init.clone()
    output_fi = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    gated_delta_rule_decode(
        q, k, v, state_fi, A_log, a, dt_bias, b, scale, output_fi, use_qk_l2norm
    )

    # Triton
    state_tr = state_init.clone()
    output_tr = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    triton_gdn_decode(
        q, k, v, state_tr, A_log, a, dt_bias, b, scale, output_tr, use_qk_l2norm
    )

    # Compare outputs using torch.testing.assert_close
    try:
        torch.testing.assert_close(
            output_fi.float(), output_tr.float(), rtol=rtol, atol=atol
        )
        output_close = True
    except AssertionError as e:
        output_close = False
        print(f"  Output mismatch: {e}")

    try:
        torch.testing.assert_close(
            state_fi.float(), state_tr.float(), rtol=rtol, atol=atol
        )
        state_close = True
    except AssertionError as e:
        state_close = False
        print(f"  State mismatch: {e}")

    return output_close and state_close


def verify_correctness_pretranspose(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """Verify FlashInfer and Triton pretranspose produce similar results."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Same initial state for both [B, HV, V, K] - pretranspose layout
    state_init = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # FlashInfer
    state_fi = state_init.clone()
    output_fi = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    gated_delta_rule_decode_pretranspose(
        q, k, v, state_fi, A_log, a, dt_bias, b, scale, output_fi, use_qk_l2norm
    )

    # Triton
    state_tr = state_init.clone()
    output_tr = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    triton_gdn_decode_pretranspose(
        q, k, v, state_tr, A_log, a, dt_bias, b, scale, output_tr, use_qk_l2norm
    )

    # Compare outputs using torch.testing.assert_close
    try:
        torch.testing.assert_close(
            output_fi.float(), output_tr.float(), rtol=rtol, atol=atol
        )
        output_close = True
    except AssertionError as e:
        output_close = False
        print(f"  Output mismatch: {e}")

    try:
        torch.testing.assert_close(
            state_fi.float(), state_tr.float(), rtol=rtol, atol=atol
        )
        state_close = True
    except AssertionError as e:
        state_close = False
        print(f"  State mismatch: {e}")

    return output_close and state_close


# ============================================================================
# All Layouts Comparison Benchmark
# ============================================================================


def gdn_decode_bf16_state_wrapper(
    q: torch.Tensor,  # [B, T, H_Q, K]
    k: torch.Tensor,  # [B, T, H_K, K]
    v: torch.Tensor,  # [B, T, HV, V]
    state: torch.Tensor,  # [pool_size, HV, V, K] BF16 (K-last layout)
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T, HV]
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, T, HV]
    scale: float,
    output: torch.Tensor,  # [B, T, HV, V] - unused, kernel returns output directly
    use_qk_l2norm: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    intermediate_states_buffer=None,
    accepted_steps=None,
    ssm_state_indices=None,
    disable_state_update: bool = False,
    disable_output: bool = False,
    recovery_steps: int = 0,
    initial_state_indices=None,
    output_state_indices=None,
):
    """
    Wrapper for gdn_decode_bf16_state GDN kernel.
    Supports T=1 (calls gated_delta_rule) and T>1 (calls gated_delta_rule_mtp).
    Both pool-only paths require initial_state_indices to be passed by the
    caller. When output_state_indices is non-None and differs from
    initial_state_indices, the call exercises the split-pool dispatch.

    Note: The kernel returns output directly, no copy needed.
    """
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        raise RuntimeError("gdn_decode_bf16_state kernel is not available")

    # Dispatch to T=1 or MTP kernel
    T = q.shape[1]
    if T == 1:
        return gdn_decode_bf16_state(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            scale=scale,
        )
    else:
        return gdn_decode_bf16_state_mtp(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=intermediate_states_buffer,
            accepted_steps=accepted_steps,
            ssm_state_indices=ssm_state_indices,
            disable_state_update=disable_state_update,
            disable_output=disable_output,
            recovery_steps=recovery_steps,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            scale=scale,
            output=output,
        )


def format_time(t):
    """Format time value, returning 'N/A' if None."""
    return f"{t:>8.2f}" if t is not None else "     N/A"


def format_speedup(base, other):
    """Calculate and format speedup."""
    if base is None or other is None or base == 0:
        return "    N/A"
    return f"{other / base:>7.2f}x"


def bench_all_layouts(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark all 4 implementations: FlashInfer/Triton x pretranspose/nontranspose."""
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    scale = 1.0 / (head_size**0.5)

    results = {"batch_size": batch_size}

    # ========== FlashInfer Pretranspose ==========
    state = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    try:
        times = bench_gpu_time(
            lambda: gated_delta_rule_decode_pretranspose(
                q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
            ),
            enable_cupti=True,
            dry_run_iters=warmup_iters,
            repeat_iters=bench_iters,
        )
        results["fi_pretrans_us"] = np.median(times) * 1000
    except Exception as e:
        results["fi_pretrans_us"] = None
        print(f"  FlashInfer pretranspose failed: {type(e).__name__}")

    # ========== FlashInfer Nontranspose ==========
    state = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    try:
        times = bench_gpu_time(
            lambda: gated_delta_rule_decode(
                q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
            ),
            enable_cupti=True,
            dry_run_iters=warmup_iters,
            repeat_iters=bench_iters,
        )
        results["fi_nontrans_us"] = np.median(times) * 1000
    except Exception as e:
        results["fi_nontrans_us"] = None
        print(f"  FlashInfer nontranspose failed: {type(e).__name__}")

    # ========== Triton Pretranspose ==========
    if TRITON_AVAILABLE:
        state = torch.randn(
            batch_size,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        output = torch.empty(
            batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
        )

        try:
            times = bench_gpu_time(
                lambda: triton_gdn_decode_pretranspose(
                    q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
                ),
                enable_cupti=True,
                dry_run_iters=warmup_iters,
                repeat_iters=bench_iters,
            )
            results["tr_pretrans_us"] = np.median(times) * 1000
        except Exception as e:
            results["tr_pretrans_us"] = None
            print(f"  Triton pretranspose failed: {type(e).__name__}")

        # ========== Triton Nontranspose ==========
        state = torch.randn(
            batch_size,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        output = torch.empty(
            batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
        )

        try:
            times = bench_gpu_time(
                lambda: triton_gdn_decode(
                    q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
                ),
                enable_cupti=True,
                dry_run_iters=warmup_iters,
                repeat_iters=bench_iters,
            )
            results["tr_nontrans_us"] = np.median(times) * 1000
        except Exception as e:
            results["tr_nontrans_us"] = None
            print(f"  Triton nontranspose failed: {type(e).__name__}")
    else:
        results["tr_pretrans_us"] = None
        results["tr_nontrans_us"] = None

    # ========== gdn_decode_bf16_state Kernel (K-fast/pretranspose layout) ==========
    if GDN_DECODE_BF16_STATE_AVAILABLE:
        # gdn_decode_bf16_state uses [B, HV, V, K] layout (K-fast, same as pretranspose)
        state = torch.randn(
            batch_size,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,  # gdn_decode_bf16_state uses BF16 state
            device="cuda",
        )
        output = torch.empty(
            batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
        )
        # The BF16 state kernels are pool-only: treat the [B, HV, V, K] state
        # as a pool of size B with sequential indices (read == write).
        initial_state_indices = torch.arange(
            batch_size, dtype=torch.int32, device="cuda"
        )

        try:
            times = bench_gpu_time(
                lambda: gdn_decode_bf16_state_wrapper(
                    q,
                    k,
                    v,
                    state,
                    A_log,
                    a,
                    dt_bias,
                    b,
                    scale,
                    output,
                    use_qk_l2norm,
                    initial_state_indices=initial_state_indices,
                ),
                enable_cupti=True,
                dry_run_iters=warmup_iters,
                repeat_iters=bench_iters,
            )
            results["gdn_decode_bf16_state_us"] = np.median(times) * 1000
        except Exception as e:
            results["gdn_decode_bf16_state_us"] = None
            print(f"  gdn_decode_bf16_state kernel failed: {type(e).__name__}: {e}")
    else:
        results["gdn_decode_bf16_state_us"] = None

    return results


def run_all_layouts_benchmark(args, dtype, use_qk_l2norm):
    """Run benchmark comparing all layouts: FlashInfer/Triton x pretranspose/nontranspose + CuTe-DSL."""
    # Verify correctness first if requested
    if args.verify and TRITON_AVAILABLE:
        print("\n=== Correctness Verification ===")
        for batch_size in [8, 16, 32, 64]:
            print(f"Batch={batch_size}:")
            # Pretranspose
            try:
                passed = verify_correctness_pretranspose(
                    batch_size=batch_size,
                    num_q_heads=args.num_q_heads,
                    num_k_heads=args.num_k_heads,
                    num_v_heads=args.num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    use_qk_l2norm=use_qk_l2norm,
                )
                print(f"  Pretranspose: {'PASS' if passed else 'FAIL'}")
            except Exception as e:
                print(f"  Pretranspose: ERROR - {type(e).__name__}")
            # Nontranspose
            try:
                passed = verify_correctness(
                    batch_size=batch_size,
                    num_q_heads=args.num_q_heads,
                    num_k_heads=args.num_k_heads,
                    num_v_heads=args.num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    use_qk_l2norm=use_qk_l2norm,
                )
                print(f"  Nontranspose: {'PASS' if passed else 'FAIL'}")
            except Exception as e:
                print(f"  Nontranspose: ERROR - {type(e).__name__}")
        print()

    print("\n" + "=" * 160)
    print("GDN Decode Benchmark (T=1): FlashInfer vs Triton vs gdn_decode_bf16_state")
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, "
        f"dtype={args.dtype}, qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
    )
    print("=" * 160)
    print()
    print(
        f"{'batch':>6} | {'FI-PreTr':>8} {'FI-NonTr':>8} | {'TR-PreTr':>8} {'TR-NonTr':>8} | {'Bf16State':>9} | "
        f"{'FI/TR-Pre':>9} {'Bf16State/FI':>11} {'Bf16State/TR':>11}"
    )
    print(
        f"{'':>6} | {'(us)':>8} {'(us)':>8} | {'(us)':>8} {'(us)':>8} | {'(us)':>8} | "
        f"{'speedup':>9} {'speedup':>10} {'speedup':>10}"
    )
    print("-" * 160)

    all_results = []
    for batch_size in args.batch_size:
        result = bench_all_layouts(
            batch_size=batch_size,
            num_q_heads=args.num_q_heads,
            num_k_heads=args.num_k_heads,
            num_v_heads=args.num_v_heads,
            head_size=args.head_size,
            dtype=dtype,
            use_qk_l2norm=use_qk_l2norm,
            warmup_iters=args.warmup,
            bench_iters=args.iters,
        )
        all_results.append(result)

        fi_pre = result.get("fi_pretrans_us")
        fi_non = result.get("fi_nontrans_us")
        tr_pre = result.get("tr_pretrans_us")
        tr_non = result.get("tr_nontrans_us")
        bf16_state_us = result.get("gdn_decode_bf16_state_us")

        # FI/TR speedup (>1 means FI faster)
        fi_tr_pre = format_speedup(fi_pre, tr_pre)

        # BF16 state vs FI-PreTr speedup (>1 means BF16 state faster)
        bf16_fi_speedup = format_speedup(bf16_state_us, fi_pre)

        # BF16 state vs TR-PreTr speedup (>1 means BF16 state faster)
        bf16_tr_speedup = format_speedup(bf16_state_us, tr_pre)

        print(
            f"{batch_size:>6} | {format_time(fi_pre)} {format_time(fi_non)} | "
            f"{format_time(tr_pre)} {format_time(tr_non)} | {format_time(bf16_state_us)} | "
            f"{fi_tr_pre} {bf16_fi_speedup:>10} {bf16_tr_speedup:>10}"
        )

    print("-" * 160)
    print()
    print("Legend:")
    print("  FI-PreTr  = FlashInfer Pretranspose [B, HV, V, K]")
    print("  FI-NonTr  = FlashInfer Nontranspose [B, HV, K, V]")
    print("  TR-PreTr  = Triton Pretranspose [B, HV, V, K]")
    print("  TR-NonTr  = Triton Nontranspose [B, HV, K, V]")
    print("  Bf16State = BF16 state kernel [B, HV, V, K] (bf16 state, T=1 + MTP)")
    print("  FI/TR speedup > 1.0 means FlashInfer is faster than Triton")
    print(
        "  Bf16State/FI speedup > 1.0 means BF16 state is faster than FlashInfer Pretranspose"
    )
    print(
        "  Bf16State/TR speedup > 1.0 means BF16 state is faster than Triton Pretranspose"
    )
    print()

    # Summary statistics
    fi_pre_times = [r["fi_pretrans_us"] for r in all_results if r.get("fi_pretrans_us")]
    tr_pre_times = [r["tr_pretrans_us"] for r in all_results if r.get("tr_pretrans_us")]
    bf16_state_times = [
        r["gdn_decode_bf16_state_us"]
        for r in all_results
        if r.get("gdn_decode_bf16_state_us")
    ]

    if fi_pre_times and tr_pre_times:
        speedups = [tr / fi for fi, tr in zip(fi_pre_times, tr_pre_times, strict=False)]
        print(
            f"FlashInfer vs Triton (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )

    if bf16_state_times and fi_pre_times and len(bf16_state_times) == len(fi_pre_times):
        speedups = [
            fi / t for t, fi in zip(bf16_state_times, fi_pre_times, strict=False)
        ]
        print(
            f"BF16 state vs FlashInfer (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )

    if bf16_state_times and tr_pre_times and len(bf16_state_times) == len(tr_pre_times):
        speedups = [
            tr / t for t, tr in zip(bf16_state_times, tr_pre_times, strict=False)
        ]
        print(
            f"BF16 state vs Triton (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )


# ============================================================================
# BF16 State Multi-Token Benchmark
# ============================================================================


def bench_gdn_decode_bf16_state(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    cache_intermediate_states: bool = False,
    disable_state_update: bool = False,
    disable_output: bool = False,
    recovery_steps: int = 0,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    pool_mode: str = "single",
    accepted_steps_mode: str = "none",
    accepted_steps_target_ar: float = -1.0,
    ssm_state_indices_mode: str = "none",
):
    """Benchmark BF16 state kernel.

    pool_mode:
      - "single": [B, HV, V, K] state, indices = arange(B), output_indices = None
        (read == write). Exercises wide_vec single-pool / mtp fallbacks.
      - "split": [2B, HV, V, K] state, read indices = arange(B),
        write indices = arange(B, 2B). Exercises the split-pool dispatch
        (speculative-decoding / MTP-verify shape).
    """
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        raise RuntimeError("gdn_decode_bf16_state kernel is not available")

    assert seq_len >= 1, f"seq_len must be >= 1, got T={seq_len}"
    assert pool_mode in ("single", "split"), (
        f"BF16 state path supports pool_mode in {{single, split}}, got {pool_mode}"
    )
    assert ssm_state_indices_mode in ("none", "unique"), (
        f"ssm_state_indices_mode must be 'none' or 'unique', got "
        f"{ssm_state_indices_mode!r}"
    )
    # FLA-style per-token scatter requires T >= 2 (asserted by the wrapper)
    # and is mutex with cache_intermediate_states and recovery_steps.
    fla_scatter = ssm_state_indices_mode != "none"
    if fla_scatter:
        assert seq_len >= 2, f"--ssm-state-indices requires seq_len>=2, got T={seq_len}"
        assert not cache_intermediate_states, (
            "--ssm-state-indices is mutex with --cache (no intermediate buffer)"
        )
        assert not disable_state_update, (
            "--ssm-state-indices requires state writes; pass --update-state"
        )
        assert recovery_steps == 0, (
            "--ssm-state-indices does not support --recovery-steps yet"
        )

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs
    T = seq_len
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")

    # Pool sized for the indexing mode: split needs 2B slots so write slots
    # are distinct from read slots. FLA scatter needs an extra B*T slots for
    # the per-token scatter destinations.
    base_pool_size = 2 * batch_size if pool_mode == "split" else batch_size
    fla_extra_slots = batch_size * T if fla_scatter else 0
    pool_size = base_pool_size + fla_extra_slots
    state = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Intermediate states buffer (MTP only, when caching is enabled).
    # The kernel keys the cache by READ indices (cache_idx * T * HV + ...),
    # and we use read_indices = arange(B) in both pool modes, so the buffer's
    # first dim is always B regardless of pool_mode (the upper half of the
    # pool, used only for split-mode writes, doesn't appear in the cache).
    intermediate_states_buffer = None
    if cache_intermediate_states and T > 1:
        intermediate_states_buffer = torch.zeros(
            batch_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

    # Pre-allocate output and state indices (avoid per-call torch.arange
    # overhead in CUPTI). For split-pool, write indices point into the upper
    # half of the pool so reads and writes don't alias.
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    if pool_mode == "split":
        output_state_indices = torch.arange(
            batch_size, 2 * batch_size, dtype=torch.int32, device="cuda"
        )
    else:
        output_state_indices = None

    # FLA-style scatter destinations: B*T fresh slots at the tail of the pool.
    # Layout: [h0_slots 0..B][split-pool out slots if any][FLA per-token slots]
    ssm_state_indices_tensor = None
    if fla_scatter:
        ssm_state_indices_tensor = torch.arange(
            base_pool_size,
            base_pool_size + batch_size * T,
            dtype=torch.int32,
            device="cuda",
        ).reshape(batch_size, T)

    # Build accepted_steps tensor based on mode. None means uniform-T (legacy
    # path, no per-request K support compiled in).
    # `--accepted-steps-target-ar AR` (when >= 0) takes precedence over mode
    # and generates binomial-sampled K values with mean AR per request.
    accepted_steps_tensor = None
    if accepted_steps_target_ar >= 0.0:
        # Binomial sampling: K_tokens ~ B(n=T, p=AR/T) so E[K_tokens] = AR.
        # Map to accepted_step (0-indexed last accepted): K_tokens - 1.
        # Clamp at [0, T-1]. K_tokens=0 → accepted_step=-1 would mean "no
        # tokens", but the kernel needs accepted_step in [0, T-1] for the
        # loop bound to be valid. We clamp to 0 (= 1 token) to match the
        # FLA convention.
        if not (0.0 < accepted_steps_target_ar <= T):
            raise ValueError(
                f"--accepted-steps-target-ar={accepted_steps_target_ar} must "
                f"be in (0, T={T}]"
            )
        torch.manual_seed(42)
        # n=T-1, p=(AR-1)/(T-1) gives E[accepted_step] = AR - 1.
        # Handle edge case T=1 (degenerate).
        if T == 1:
            accepted_steps_tensor = torch.zeros(
                batch_size, dtype=torch.int32, device="cuda"
            )
        else:
            p = (accepted_steps_target_ar - 1.0) / (T - 1)
            p = max(0.0, min(1.0, p))
            n_tensor = torch.full((batch_size,), float(T - 1), device="cuda")
            p_tensor = torch.full((batch_size,), p, device="cuda")
            accepted_steps_tensor = (
                torch.binomial(n_tensor, p_tensor).clamp(0, T - 1).to(torch.int32)
            )
    elif accepted_steps_mode == "uniform":
        # All requests process all T tokens (K = T-1 each). Verifies early-break
        # check overhead is negligible when the break never fires.
        accepted_steps_tensor = torch.full(
            (batch_size,), T - 1, dtype=torch.int32, device="cuda"
        )
    elif accepted_steps_mode == "uniform-half":
        # All requests process ~T/2 tokens. Verifies wallclock scales with K.
        kval = max(0, (T // 2) - 1)
        accepted_steps_tensor = torch.full(
            (batch_size,), kval, dtype=torch.int32, device="cuda"
        )
    elif accepted_steps_mode == "random":
        # Uniform random K ∈ [0, T-1]. Realistic spec-decode acceptance mix.
        torch.manual_seed(42)
        accepted_steps_tensor = torch.randint(
            0, T, (batch_size,), dtype=torch.int32, device="cuda"
        )
    elif accepted_steps_mode == "one-outlier":
        # One request at K=T-1, rest at K=0 (1 token). The "stress" case
        # — early-break should reduce wallclock from T*sum-cost to ~1*sum-cost
        # for the majority + T*1 for the outlier.
        accepted_steps_tensor = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        if batch_size > 0:
            accepted_steps_tensor[0] = T - 1
    elif accepted_steps_mode != "none":
        raise ValueError(
            f"Unknown accepted_steps_mode: {accepted_steps_mode!r}. "
            f"Choose from: none, uniform, uniform-half, random, one-outlier"
        )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: gdn_decode_bf16_state_wrapper(
            q,
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            output,
            use_qk_l2norm,
            intermediate_states_buffer=intermediate_states_buffer,
            accepted_steps=accepted_steps_tensor,
            ssm_state_indices=ssm_state_indices_tensor,
            disable_state_update=disable_state_update,
            disable_output=disable_output,
            recovery_steps=recovery_steps,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, seq_len
    )
    # gdn_decode_bf16_state uses BF16 state (2 bytes), not FP32 (4 bytes)
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len,
        disable_state_update=disable_state_update,
        state_dtype_bytes=2,  # BF16 state for gdn_decode_bf16_state
        cache_intermediate_states=cache_intermediate_states,
    )

    kernel_tflops = flops / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    kernel_tb_per_sec = (
        bytes_accessed / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "kernel_median_us": kernel_median_ms * 1000,
        "kernel_tflops": kernel_tflops,
        "kernel_tb_per_sec": kernel_tb_per_sec,
    }


def run_gdn_decode_bf16_state_benchmark(args, dtype, use_qk_l2norm):
    """Run BF16 state benchmark for T=1 and MTP T>=1."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        print("Error: BF16 state kernel is not available.")
        print("Make sure flashinfer.gdn_kernels.gdn_decode_bf16_state is importable.")
        return

    valid_seq_lens = [t for t in args.seq_len if t >= 1]
    if not valid_seq_lens:
        print("Error: --seq-len must include values >= 1")
        return

    cache_intermediate = getattr(args, "cache_intermediate_states", False)
    disable_state_update = not getattr(args, "update_state", False)
    disable_output = getattr(args, "no_output", False)
    recovery_steps = getattr(args, "recovery_steps", 0)
    pool_mode = getattr(args, "pool_mode", "single")
    accepted_steps_mode = getattr(args, "accepted_steps_mode", "none")
    accepted_steps_target_ar = getattr(args, "accepted_steps_target_ar", -1.0)
    ssm_state_indices_mode = getattr(args, "ssm_state_indices", "none")

    print("\n" + "=" * 100)
    print(f"BF16 State GDN Benchmark (T={valid_seq_lens})")
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, "
        f"dtype={args.dtype}, qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}, "
        f"cache_intermediate={'ON' if cache_intermediate else 'OFF'}, "
        f"update_state={'ON' if not disable_state_update else 'OFF'}, "
        f"output={'OFF' if disable_output else 'ON'}, "
        f"recovery_steps={recovery_steps}, "
        f"pool_mode={pool_mode}, "
        f"accepted_steps={accepted_steps_mode}, "
        f"ssm_state_indices={ssm_state_indices_mode}"
    )
    print("=" * 100)
    print()
    print(f"{'batch':>6} {'T':>4} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10}")
    print("-" * 100)

    all_results = []
    for batch_size in args.batch_size:
        for seq_len in valid_seq_lens:
            try:
                result = bench_gdn_decode_bf16_state(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=args.num_q_heads,
                    num_k_heads=args.num_k_heads,
                    num_v_heads=args.num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    use_qk_l2norm=use_qk_l2norm,
                    cache_intermediate_states=cache_intermediate,
                    disable_state_update=disable_state_update,
                    disable_output=disable_output,
                    recovery_steps=min(recovery_steps, seq_len),
                    warmup_iters=args.warmup,
                    bench_iters=args.iters,
                    pool_mode=pool_mode,
                    accepted_steps_mode=accepted_steps_mode,
                    accepted_steps_target_ar=accepted_steps_target_ar,
                    ssm_state_indices_mode=ssm_state_indices_mode,
                )
                all_results.append(result)

                print(
                    f"{result['batch_size']:>6} {result['seq_len']:>4} "
                    f"{result['kernel_median_us']:>10.2f} "
                    f"{result['kernel_tflops']:>10.2f} "
                    f"{result['kernel_tb_per_sec']:>10.2f}"
                )
            except Exception as e:
                print(
                    f"{batch_size:>6} {seq_len:>4} {'ERROR':>10} - {type(e).__name__}: {e}"
                )

    print("-" * 100)
    print()

    # Summary by T value
    for t in valid_seq_lens:
        t_results = [r for r in all_results if r["seq_len"] == t]
        if t_results:
            avg_time = np.mean([r["kernel_median_us"] for r in t_results])
            avg_tflops = np.mean([r["kernel_tflops"] for r in t_results])
            print(
                f"T={t}: Average time={avg_time:.2f}us, Average TFLOPS={avg_tflops:.2f}"
            )


def run_gdn_decode_bf16_wy_output_only_benchmark(args, dtype, use_qk_l2norm):
    """OUTPUT-ONLY head-to-head: branch bf16_state MTP vs no-prepack v18 kernel.

    Both kernels share the same bf16 (pool, HV, V, K) state and inputs and are
    timed output-only (disable_state_update=True, disable_output=False,
    recovery_steps=0) via cuda-graph replay = true kernel GPU time (the
    production decode pattern; excludes per-call Python wrapper overhead), with a
    cold-L2 cuda-event fallback if a config can't be captured.

    v18 is a fixed T=16-tile kernel, so for T<16 it stages inputs into persistent
    T=16 buffers. v18 is timed KERNEL-ONLY (the eager correctness call pre-stages
    the buffers, then _RESTAGE=False skips the per-call copy) — i.e. assuming the
    fixed-buffer serving pattern where the producer writes into the T=16 buffers.
    This matches how the avo harness measures (time_one(compiled(*cargs))). The
    branch needs no padding and is timed its natural way. Reports the v18 speedup,
    the max abs output diff (<=1 bf16 ULP cross-check), and the timing mode.
    """
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        print("Error: branch BF16 state kernel is not available.")
        return
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule_mtp as _branch_mtp,
    )

    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
            gated_delta_rule_mtp as _v18_mtp,
        )
        import flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only as _v18mod
    except (ImportError, RuntimeError) as e:
        print(f"Error: gdn_decode_bf16_wy_output_only not available: {e}")
        return

    valid_seq_lens = [t for t in args.seq_len if t >= 1]
    if not valid_seq_lens:
        print("Error: --seq-len must include values >= 1")
        return

    num_o_heads = max(args.num_q_heads, args.num_v_heads)
    num_sab = num_o_heads
    scale = 1.0 / (args.head_size**0.5)

    print("\n" + "=" * 100)
    print(
        f"BF16 State OUTPUT-ONLY head-to-head: branch vs wy_output_only (v18)  T={valid_seq_lens}"
    )
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, dtype={args.dtype}, "
        f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
    )
    print("=" * 100)
    print(
        f"{'batch':>6} {'T':>4} {'branch(us)':>12} {'v18(us)':>12} "
        f"{'speedup':>10} {'max|d|':>11} {'mode':>6}"
    )
    print("-" * 100)

    for batch_size in args.batch_size:
        for T in valid_seq_lens:
            try:
                _v18mod._RESTAGE = True  # default: stage inputs; flipped off only for the v18 timed run
                q = torch.randn(
                    batch_size,
                    T,
                    args.num_q_heads,
                    args.head_size,
                    dtype=dtype,
                    device="cuda",
                )
                k = torch.randn(
                    batch_size,
                    T,
                    args.num_k_heads,
                    args.head_size,
                    dtype=dtype,
                    device="cuda",
                )
                v = torch.randn(
                    batch_size,
                    T,
                    args.num_v_heads,
                    args.head_size,
                    dtype=dtype,
                    device="cuda",
                )
                # Scale gating inputs by 0.1 (matches the correctness test): raw
                # std-1 randn drives the gated recurrence unstable over many
                # tokens (-> Inf/NaN at large T), which would pollute the max|d|
                # cross-check. Magnitude does not affect kernel timing.
                A_log = torch.randn(num_sab, dtype=torch.float32, device="cuda") * 0.1
                dt_bias = torch.randn(num_sab, dtype=torch.float32, device="cuda") * 0.1
                a = (
                    torch.randn(batch_size, T, num_sab, dtype=dtype, device="cuda")
                    * 0.1
                )
                b = torch.randn(batch_size, T, num_sab, dtype=dtype, device="cuda")
                state = torch.randn(
                    batch_size,
                    num_sab,
                    args.head_size,
                    args.head_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

                common = dict(
                    A_log=A_log,
                    a=a,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    b=b,
                    initial_state_indices=idx,
                    disable_state_update=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm,
                    scale=scale,
                )

                # output=None exercises each kernel's natural allocation path;
                # v18 then returns a zero-copy [:, :T] view (its optimized path),
                # and the T<16 inputs go through v18's persistent staging buffers.
                def _time(fn, graph):
                    return (
                        np.median(
                            bench_gpu_time(
                                lambda: fn(initial_state_source=state, **common),
                                dry_run_iters=(5 if graph else args.warmup),
                                repeat_iters=args.iters,
                                use_cuda_graph=graph,
                                enable_cupti=False,
                                cold_l2_cache=(not graph),
                            )
                        )
                        * 1000
                    )

                # Eager correctness cross-check (expect <= 1 bf16 ULP).
                try:
                    o_br = _branch_mtp(initial_state_source=state.clone(), **common)
                except Exception as e:
                    o_br = None
                    print(f"  [branch B={batch_size} T={T}] {type(e).__name__}: {e}")
                try:
                    o_v18 = _v18_mtp(initial_state_source=state.clone(), **common)
                except Exception as e:
                    o_v18 = None
                    print(f"  [v18 B={batch_size} T={T}] {type(e).__name__}: {e}")
                torch.cuda.synchronize()
                max_d = (
                    (o_br.float() - o_v18.float()).abs().max().item()
                    if (o_br is not None and o_v18 is not None)
                    else float("nan")
                )

                # Prefer cuda-graph replay = true kernel GPU time (the production
                # decode pattern; excludes the per-call Python wrapper overhead
                # — 10x from_dlpack, contiguous, alloc, cache lookup — that
                # otherwise dominates v18 at small batch). Fall back to cold-L2
                # event timing for BOTH kernels if either can't be captured
                # (v18's T<16 pad path allocates during capture), so the row's
                # speedup stays apples-to-apples. 'mode' tags which was used.
                # Time v18 KERNEL-ONLY: the eager calls above already staged the
                # inputs into v18's persistent T=16 buffers, so skip the per-call
                # restage copy (the fixed-buffer serving pattern, and exactly how
                # avo's time_one(compiled(*cargs)) measures). The branch ignores
                # this flag and is timed its natural way (native T, no padding).
                _v18mod._RESTAGE = False
                t_br = t_v18 = None
                mode = "graph"
                if o_br is not None and o_v18 is not None:
                    try:
                        t_br = _time(_branch_mtp, True)
                        t_v18 = _time(_v18_mtp, True)
                    except Exception:
                        mode = "event"
                        try:
                            t_br = _time(_branch_mtp, False)
                        except Exception:
                            t_br = None
                        try:
                            t_v18 = _time(_v18_mtp, False)
                        except Exception:
                            t_v18 = None
                else:
                    mode = "event"
                    if o_br is not None:
                        try:
                            t_br = _time(_branch_mtp, False)
                        except Exception:
                            t_br = None
                    if o_v18 is not None:
                        try:
                            t_v18 = _time(_v18_mtp, False)
                        except Exception:
                            t_v18 = None

                spd = (t_br / t_v18) if (t_br and t_v18) else float("nan")
                br_s = f"{t_br:>12.2f}" if t_br is not None else f"{'ERR':>12}"
                v18_s = f"{t_v18:>12.2f}" if t_v18 is not None else f"{'ERR':>12}"
                print(
                    f"{batch_size:>6} {T:>4} {br_s} {v18_s} "
                    f"{spd:>9.2f}x {max_d:>11.2e} {mode:>6}"
                )
            except Exception as e:
                print(f"{batch_size:>6} {T:>4} {'ERROR':>12} - {type(e).__name__}: {e}")
    print("-" * 100)
    print()


# ============================================================================
# Main Entry Points
# ============================================================================


def run_flashinfer_only_benchmark(args, dtype, use_qk_l2norm):
    """Run FlashInfer-only benchmarks."""
    # Determine which versions to benchmark
    if args.version == "all":
        versions_to_bench = ["pretranspose", "nontranspose", "mtp"]
    else:
        versions_to_bench = [args.version]

    for version in versions_to_bench:
        if version == "mtp":
            # Benchmark MTP version
            print(
                f"\nGDN MTP Benchmark "
                f"(heads: q={args.num_q_heads}, k={args.num_k_heads}, "
                f"v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype}, "
                f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}, "
                f"cache_intermediate={'ON' if args.cache_intermediate_states else 'OFF'}, "
                f"update_state={'ON' if args.update_state else 'OFF'})"
            )
            print("-" * 100)
            print(
                f"{'batch':>6} {'seq_len':>8} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10}"
            )
            print("-" * 100)

            for batch_size in args.batch_size:
                for seq_len in args.seq_len:
                    result = bench_gdn_mtp(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_q_heads=args.num_q_heads,
                        num_k_heads=args.num_k_heads,
                        num_v_heads=args.num_v_heads,
                        head_size=args.head_size,
                        dtype=dtype,
                        use_qk_l2norm=use_qk_l2norm,
                        cache_intermediate_states=args.cache_intermediate_states,
                        disable_state_update=not args.update_state,
                        warmup_iters=args.warmup,
                        bench_iters=args.iters,
                        ssm_state_indices_mode=getattr(
                            args, "ssm_state_indices", "none"
                        ),
                    )

                    kernel_time_us = result["kernel_median_us"]

                    print(
                        f"{result['batch_size']:>6} {result['seq_len']:>8} {kernel_time_us:>10.2f} "
                        f"{result['kernel_tflops']:>10.2f} {result['kernel_tb_per_sec']:>10.2f}"
                    )

            print("-" * 100)
            continue

        # Benchmark decode versions (pretranspose/nontranspose)
        print(
            f"\nGDN Decode Benchmark - {version.upper()} version "
            f"(heads: q={args.num_q_heads}, k={args.num_k_heads}, "
            f"v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype}, "
            f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'})"
        )
        print("-" * 90)
        print(
            f"{'batch':>6} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10} {'kernel':>15}"
        )
        print("-" * 90)

        for batch_size in args.batch_size:
            result = bench_gdn_decode(
                batch_size=batch_size,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
                version=version,
                use_qk_l2norm=use_qk_l2norm,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )

            # Determine which kernel variant was used (based on batch size threshold)
            if version == "pretranspose":
                kernel_variant = "SmallBatch" if batch_size <= 32 else "LargeBatch"
            elif version == "nontranspose":
                kernel_variant = "SmallBatch" if batch_size < 32 else "LargeBatch"

            # Time in microseconds
            kernel_time_us = result["kernel_median_us"]

            print(
                f"{result['batch_size']:>6} {kernel_time_us:>10.2f} "
                f"{result['kernel_tflops']:>10.2f} {result['kernel_tb_per_sec']:>10.2f} "
                f"{kernel_variant:>15}"
            )

        print("-" * 90)


def run_comparison_benchmark(args, dtype, use_qk_l2norm):
    """Run comparison benchmarks (FlashInfer vs Triton)."""
    if not TRITON_AVAILABLE:
        print("Error: Triton is not available. Install with: pip install triton")
        return

    # Verify correctness first if requested
    if args.verify:
        version_name = args.version.upper() if args.version != "all" else "NONTRANSPOSE"
        print(f"\n=== Correctness Verification ({version_name}) ===")
        # Use larger batch sizes to avoid alignment issues with small batches
        for batch_size in [8, 16, 32, 64]:
            try:
                if args.version == "pretranspose":
                    passed = verify_correctness_pretranspose(
                        batch_size=batch_size,
                        num_q_heads=args.num_q_heads,
                        num_k_heads=args.num_k_heads,
                        num_v_heads=args.num_v_heads,
                        head_size=args.head_size,
                        dtype=dtype,
                        use_qk_l2norm=use_qk_l2norm,
                    )
                else:
                    passed = verify_correctness(
                        batch_size=batch_size,
                        num_q_heads=args.num_q_heads,
                        num_k_heads=args.num_k_heads,
                        num_v_heads=args.num_v_heads,
                        head_size=args.head_size,
                        dtype=dtype,
                        use_qk_l2norm=use_qk_l2norm,
                    )
                status = "PASS" if passed else "FAIL"
                print(f"Batch={batch_size}: {status}")
            except Exception as e:
                print(f"Batch={batch_size}: ERROR - {type(e).__name__}")
        print()

    if args.version == "mtp":
        # MTP comparison
        print("\nGDN MTP Comparison: FlashInfer (CuTe DSL) vs Triton")
        print(
            f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
            f"v_heads={args.num_v_heads}, head_size={args.head_size}, dtype={args.dtype}, "
            f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}, "
            f"cache_intermediate={'ON' if args.cache_intermediate_states else 'OFF'}"
        )
        print("-" * 110)
        print(
            f"{'batch':>6} {'seq_len':>8} {'FlashInfer(us)':>14} {'Triton(us)':>12} "
            f"{'FI TFLOPS':>10} {'TR TFLOPS':>10} {'Speedup':>10}"
        )
        print("-" * 110)

        results = []
        for batch_size in args.batch_size:
            for seq_len in args.seq_len:
                result = bench_mtp_comparison(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=args.num_q_heads,
                    num_k_heads=args.num_k_heads,
                    num_v_heads=args.num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    use_qk_l2norm=use_qk_l2norm,
                    cache_intermediate_states=args.cache_intermediate_states,
                    warmup_iters=args.warmup,
                    bench_iters=args.iters,
                )
                results.append(result)

                print(
                    f"{result['batch_size']:>6} {result['seq_len']:>8} "
                    f"{result['flashinfer_us']:>14.2f} {result['triton_us']:>12.2f} "
                    f"{result['flashinfer_tflops']:>10.2f} {result['triton_tflops']:>10.2f} "
                    f"{result['speedup']:>10.2f}x"
                )

        print("-" * 110)
    elif args.version == "pretranspose":
        # Pretranspose decode comparison
        print("\nGDN Decode Comparison (PRETRANSPOSE): FlashInfer (CuTe DSL) vs Triton")
        print(
            f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
            f"v_heads={args.num_v_heads}, head_size={args.head_size}, dtype={args.dtype}, "
            f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
        )
        print("-" * 100)
        print(
            f"{'batch':>6} {'FlashInfer(us)':>14} {'Triton(us)':>12} "
            f"{'FI TFLOPS':>10} {'TR TFLOPS':>10} {'Speedup':>10}"
        )
        print("-" * 100)

        results = []
        for batch_size in args.batch_size:
            result = bench_comparison_pretranspose(
                batch_size=batch_size,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
                use_qk_l2norm=use_qk_l2norm,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            results.append(result)

            print(
                f"{result['batch_size']:>6} {result['flashinfer_us']:>14.2f} "
                f"{result['triton_us']:>12.2f} {result['flashinfer_tflops']:>10.2f} "
                f"{result['triton_tflops']:>10.2f} {result['speedup']:>10.2f}x"
            )

        print("-" * 100)
    else:
        # Nontranspose decode comparison
        print("\nGDN Decode Comparison (NONTRANSPOSE): FlashInfer (CuTe DSL) vs Triton")
        print(
            f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
            f"v_heads={args.num_v_heads}, head_size={args.head_size}, dtype={args.dtype}, "
            f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
        )
        print("-" * 100)
        print(
            f"{'batch':>6} {'FlashInfer(us)':>14} {'Triton(us)':>12} "
            f"{'FI TFLOPS':>10} {'TR TFLOPS':>10} {'Speedup':>10}"
        )
        print("-" * 100)

        results = []
        for batch_size in args.batch_size:
            result = bench_comparison(
                batch_size=batch_size,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
                use_qk_l2norm=use_qk_l2norm,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            results.append(result)

            print(
                f"{result['batch_size']:>6} {result['flashinfer_us']:>14.2f} "
                f"{result['triton_us']:>12.2f} {result['flashinfer_tflops']:>10.2f} "
                f"{result['triton_tflops']:>10.2f} {result['speedup']:>10.2f}x"
            )

        print("-" * 100)

    print("Speedup > 1.0 means FlashInfer is faster")

    # Print summary
    speedups = [r["speedup"] for r in results]
    min_idx = speedups.index(min(speedups))
    max_idx = speedups.index(max(speedups))
    print("\nSummary:")
    print(f"  Average speedup: {np.mean(speedups):.2f}x")
    if args.version == "mtp":
        print(
            f"  Min speedup: {speedups[min_idx]:.2f}x "
            f"(batch={results[min_idx]['batch_size']}, T={results[min_idx]['seq_len']})"
        )
        print(
            f"  Max speedup: {speedups[max_idx]:.2f}x "
            f"(batch={results[max_idx]['batch_size']}, T={results[max_idx]['seq_len']})"
        )
    else:
        print(
            f"  Min speedup: {speedups[min_idx]:.2f}x (batch={results[min_idx]['batch_size']})"
        )
        print(
            f"  Max speedup: {speedups[max_idx]:.2f}x (batch={results[max_idx]['batch_size']})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="GDN Decode Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: All layouts comparison (FlashInfer/Triton x pretranspose/nontranspose + Improved CuTe-DSL)
  python benchmarks/bench_gdn_decode.py --batch-size 1 4 8 16 32 64 128 256 512

  # Single layout comparison: FlashInfer vs Triton (nontranspose)
  python benchmarks/bench_gdn_decode.py --compare --batch-size 1 4 8 16 32 64 128 256 512

  # Single layout comparison: FlashInfer vs Triton (pretranspose)
  python benchmarks/bench_gdn_decode.py --compare --version pretranspose --batch-size 1 4 8 16 32 64 128 256 512

  # MTP benchmark (FlashInfer only)
  python benchmarks/bench_gdn_decode.py --version mtp --batch-size 1 32 128

  # MTP comparison: FlashInfer vs Triton
  python benchmarks/bench_gdn_decode.py --version mtp --compare --batch-size 1 32 128

  # BF16 state benchmark (T=1 and MTP)
  python benchmarks/bench_gdn_decode.py --version bf16_state --batch-size 1 32 128 512
""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to benchmark (number of concurrent decode requests)",
    )
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["qwen3-next", "custom"],
        default="custom",
        help="Use preset config. qwen3-next: q=k=16, v=32, d=128",
    )
    parser.add_argument(
        "--no-qk-l2norm",
        action="store_true",
        help="Disable Q/K L2 normalization",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=[
            "pretranspose",
            "nontranspose",
            "mtp",
            "bf16_state",
            "bf16_wy_output_only",
            "all",
        ],
        default="nontranspose",
        help="Kernel version: pretranspose, nontranspose, mtp, bf16_state, "
        "bf16_wy_output_only (v18 head-to-head), or all",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Sequence lengths: for MTP use T>1, for bf16_state use any T>=1",
    )
    parser.add_argument(
        "--cache-intermediate-states",
        action="store_true",
        help="Cache intermediate states for MTP benchmark",
    )
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Update final state (disable_state_update=False) for MTP benchmark",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help=(
            "Skip the per-token output projection (state-only mode). "
            "Sets disable_output=True; the kernel still runs the recurrence "
            "(state update) but skips the second inner product (h_new @ q), "
            "the butterfly reduce of o, and the per-token output STG."
        ),
    )
    parser.add_argument(
        "--accepted-steps-mode",
        choices=("none", "uniform", "uniform-half", "random", "one-outlier"),
        default="none",
        help=(
            "Per-request K (accepted_steps) mode for the recovery kernel. "
            "'none': legacy uniform-T (no accepted_steps tensor; no kernel "
            "code change). 'uniform': all K=T-1 (verifies zero overhead of "
            "the runtime loop bound). 'uniform-half': all K=T/2-1. 'random': "
            "uniform random K∈[0,T-1] (realistic spec-decode mix). "
            "'one-outlier': K[0]=T-1, rest=0 — early-break stress case."
        ),
    )
    parser.add_argument(
        "--accepted-steps-target-ar",
        type=float,
        default=-1.0,
        help=(
            "Per-request K with binomial-sampled accepted_steps targeting an "
            "average # accepted tokens (AR) per request. Sampled as "
            "B(n=T-1, p=(AR-1)/(T-1)) with seed=42 (deterministic). When >= 0, "
            "overrides --accepted-steps-mode. Example: --accepted-steps-target-ar 5.0 "
            "at T=8 → per-request K averaging 5 tokens."
        ),
    )
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=0,
        help=(
            "Fused recovery+decode mode. Of the T total tokens, the first "
            "K=recovery-steps run state-only (no output); the remaining "
            "T-K run with output. State h_K is asynchronously written to "
            "GMEM at the boundary (overlapped with decode compute). The "
            "post-decode state h_T is discarded. K must be in [0, T]."
        ),
    )
    parser.add_argument(
        "--pool-mode",
        choices=("single", "split"),
        default="single",
        help=(
            "Pool indexing mode for the BF16 state benchmark. "
            "'single' (default): treat the [B, HV, V, K] state as a pool of "
            "size B with sequential indices arange(B) (read==write). "
            "'split': allocate a pool of size 2*B; reads from slots [0..B), "
            "writes to slots [B..2B), exercising the split-pool dispatch "
            "(speculative-decoding / MTP-verify shape)."
        ),
    )
    parser.add_argument(
        "--ssm-state-indices",
        choices=("none", "unique"),
        default="none",
        help=(
            "FLA-style per-token pool scatter. 'none' (default): legacy "
            "behavior (dense intermediate buffer or none). 'unique': "
            "allocate B*T extra pool slots and pass ssm_state_indices=[B,T] "
            "int32 to the kernel, so each h_{t+1} writes directly to "
            "pool[ssm_state_indices[i, t]] (matches FLA's Triton API). "
            "Requires T>=2 and is mutex with --cache."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison benchmark: FlashInfer vs Triton",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run correctness verification before comparison benchmarking",
    )
    args = parser.parse_args()

    # Apply preset configurations
    if args.preset == "qwen3-next":
        # Qwen3-Next-80B-A3B linear attention config (GVA)
        args.num_q_heads = 16
        args.num_k_heads = 16
        args.num_v_heads = 32
        args.head_size = 128

    # Check SM90 support
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 9:
        print(f"Current device capability: {device_capability}")
        print("GDN requires SM90 (Hopper) or later. Exiting...")
        return

    dtype = getattr(torch, args.dtype)
    use_qk_l2norm = not args.no_qk_l2norm

    if args.version == "mtp":
        # MTP mode: use comparison or flashinfer-only
        if args.compare:
            run_comparison_benchmark(args, dtype, use_qk_l2norm)
        else:
            run_flashinfer_only_benchmark(args, dtype, use_qk_l2norm)
    elif args.version == "bf16_state":
        # BF16 state benchmark: T=1 and MTP T>=2 vs FP32 MTP
        run_gdn_decode_bf16_state_benchmark(args, dtype, use_qk_l2norm)
    elif args.version == "bf16_wy_output_only":
        # OUTPUT-ONLY head-to-head: branch bf16_state MTP vs no-prepack v18
        run_gdn_decode_bf16_wy_output_only_benchmark(args, dtype, use_qk_l2norm)
    else:
        # Non-MTP: always run all layouts comparison (FlashInfer/Triton x pretranspose/nontranspose + gdn_decode_bf16_state)
        run_all_layouts_benchmark(args, dtype, use_qk_l2norm)


if __name__ == "__main__":
    main()
