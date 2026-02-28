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
1. All layouts comparison (default for decode): FlashInfer/Triton x pretranspose/nontranspose + gdn_decode_klast_bf16_state
2. Single layout comparison: FlashInfer (CuTe DSL) vs Triton kernel (--compare)
3. MTP benchmark (--version mtp)
4. gdn_decode_klast_bf16_state benchmark (--version gdn_decode_klast_bf16_state) for T=1,2,3,4

Kernels benchmarked:
- FlashInfer Pretranspose [B, HV, V, K] (V-major layout)
- FlashInfer Nontranspose [B, HV, K, V] (K-major layout)
- Triton Pretranspose [B, HV, V, K]
- Triton Nontranspose [B, HV, K, V]
- gdn_decode_klast_bf16_state [B, HV, V, K] (K-fast layout, T=1..4, bf16 state)
  from flashinfer.cute_dsl.gated_delta_rule

Usage:
    # Default: All layouts comparison (FlashInfer/Triton x pretranspose/nontranspose + gdn_decode_klast_bf16_state)
    python benchmarks/bench_gdn_decode.py --batch-size 1 4 8 16 32 64 128 256 512

    # Single layout comparison: FlashInfer vs Triton
    python benchmarks/bench_gdn_decode.py --compare --batch-size 1 4 8 16 32 64 128 256 512

    # MTP benchmark (FlashInfer only)
    python benchmarks/bench_gdn_decode.py --version mtp --batch-size 1 32 128

    # MTP comparison: FlashInfer vs Triton
    python benchmarks/bench_gdn_decode.py --version mtp --compare --batch-size 1 32 128

    # gdn_decode_klast_bf16_state benchmark (T=1,2,3,4)
    python benchmarks/bench_gdn_decode.py --version gdn_decode_klast_bf16_state --batch-size 1 32 128 512

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

# Import the gdn_decode_klast_bf16_state kernel for benchmarking (T=1..4, bf16 state, K-last)
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as gdn_decode_klast_bf16_state,
    )

    GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = True
except ImportError:
    GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = False


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

    # Intermediate states: [B, T, HV, K, V] - only for MTP (seq_len > 1)
    # Write all T steps of intermediate states
    intermediate_bytes = 0
    if seq_len > 1:
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

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    @triton.jit
    def fused_sigmoid_gating_delta_rule_kernel(
        # Pointers to matrices
        Q,
        K,
        V,
        O,
        H,  # Hidden state [B, HV, K, V]
        A_LOG,  # Log decay [HV]
        A,  # Input-dependent decay [B, HV]
        DT_BIAS,  # Decay bias [HV]
        B_GATE,  # Update gate [B, HV]
        # Strides
        stride_qb,
        stride_qh,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vv,
        stride_ob,
        stride_oh,
        stride_ov,
        stride_hb,
        stride_hh,
        stride_hk,
        stride_hv,
        # Parameters
        softplus_beta: tl.constexpr,
        softplus_threshold: tl.constexpr,
        scale: tl.constexpr,
        use_qk_l2norm: tl.constexpr,
        B: tl.constexpr,
        HV: tl.constexpr,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """
        Triton kernel for fused sigmoid gating delta rule update.

        Follows SGLang's implementation:
        1. g = -exp(A_log) * softplus(a + dt_bias)
        2. beta = sigmoid(b)
        3. h *= exp(g)
        4. v_new = v - k @ h
        5. v_new *= beta
        6. h += outer(k, v_new)
        7. o = q @ h
        """
        # Block indices
        i_bh = tl.program_id(0)
        i_k = tl.program_id(1)
        i_v = tl.program_id(2)

        i_b = i_bh // HV
        i_hv = i_bh % HV

        # GVA head mapping (num_v_heads > num_q_heads)
        h_ratio_q = HV // H_Q
        h_ratio_k = HV // H_K
        i_hq = i_hv // h_ratio_q
        i_hk = i_hv // h_ratio_k

        # Load A_log and dt_bias for this head
        b_A_log = tl.load(A_LOG + i_hv).to(tl.float32)
        b_dt_bias = tl.load(DT_BIAS + i_hv).to(tl.float32)

        # Load a (input-dependent decay) for this batch and head
        b_a = tl.load(A + i_b * HV + i_hv).to(tl.float32)

        # Load b (update gate) for this batch and head
        b_b = tl.load(B_GATE + i_b * HV + i_hv).to(tl.float32)

        # Compute softplus: softplus(x) = (1/beta) * log(1 + exp(beta*x))
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Block offsets
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        # Load q, k, v
        p_q = Q + i_b * stride_qb + i_hq * stride_qh + o_k * stride_qk
        p_k = K + i_b * stride_kb + i_hk * stride_kh + o_k * stride_kk
        p_v = V + i_b * stride_vb + i_hv * stride_vh + o_v * stride_vv

        b_q = tl.load(p_q, mask=o_k < K_DIM, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=o_k < K_DIM, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=o_v < V_DIM, other=0.0).to(tl.float32)

        # Apply L2 normalization (if enabled)
        if use_qk_l2norm:
            # Compute L2 norm across K dimension (need reduction across blocks)
            # For simplicity, assume single K block
            q_norm = tl.sqrt(tl.sum(b_q * b_q) + 1e-8)
            k_norm = tl.sqrt(tl.sum(b_k * b_k) + 1e-8)
            b_q = b_q / q_norm
            b_k = b_k / k_norm

        # Apply scale to q
        b_q = b_q * scale

        # Load hidden state h[K, V] from state[B, HV, K, V]
        p_h = (
            H
            + i_b * stride_hb
            + i_hv * stride_hh
            + o_k[:, None] * stride_hk
            + o_v[None, :] * stride_hv
        )
        b_h = tl.load(
            p_h, mask=(o_k[:, None] < K_DIM) & (o_v[None, :] < V_DIM), other=0.0
        ).to(tl.float32)

        # Step 1: Apply decay to hidden state: h *= exp(g)
        b_h = b_h * tl.exp(b_g)

        # Step 2: Delta rule: v -= sum(h * k, dim=0) = k @ h
        # b_h is [BK, BV], b_k is [BK]
        # We need to compute k @ h = sum(k[:, None] * h, dim=0)
        b_v = b_v - tl.sum(b_h * b_k[:, None], 0)

        # Step 3: Apply beta gating: v *= beta
        b_v = b_v * b_beta

        # Step 4: Update hidden state: h += outer(k, v) = k[:, None] * v[None, :]
        b_h = b_h + b_k[:, None] * b_v[None, :]

        # Step 5: Compute output: o = q @ h = sum(q[:, None] * h, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)

        # Store output
        p_o = O + i_b * stride_ob + i_hv * stride_oh + o_v * stride_ov
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_v < V_DIM)

        # Store updated hidden state
        tl.store(
            p_h,
            b_h.to(p_h.dtype.element_ty),
            mask=(o_k[:, None] < K_DIM) & (o_v[None, :] < V_DIM),
        )

    @triton.jit
    def fused_sigmoid_gating_delta_rule_mtp_kernel(
        # Pointers to matrices
        Q,  # [B, T, H_Q, K]
        K,  # [B, T, H_K, K]
        V,  # [B, T, HV, V]
        O,  # [B, T, HV, V]
        H,  # Hidden state [pool_size, HV, V, K] (K-last layout)
        INTERMEDIATE,  # Intermediate states [pool_size, T, HV, V, K]
        H0_INDICES,  # [B]
        A_LOG,  # Log decay [HV]
        A,  # Input-dependent decay [B, T, HV]
        DT_BIAS,  # Decay bias [HV]
        B_GATE,  # Update gate [B, T, HV]
        # Strides for Q, K, V, O [B, T, H, dim]
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qk,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kk,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vv,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_ov,
        # Strides for hidden state [pool_size, HV, V, K]
        stride_hp,
        stride_hh,
        stride_hv,
        stride_hk,
        # Strides for intermediate states [pool_size, T, HV, V, K]
        stride_ip,
        stride_it,
        stride_ih,
        stride_iv,
        stride_ik,
        # Strides for A [B, T, HV]
        stride_ab,
        stride_at,
        stride_ah,
        # Parameters
        softplus_beta: tl.constexpr,
        softplus_threshold: tl.constexpr,
        scale: tl.constexpr,
        use_qk_l2norm: tl.constexpr,
        disable_state_update: tl.constexpr,
        cache_intermediate_states: tl.constexpr,
        B: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """
        Triton kernel for MTP (Multiple Token Processing) delta rule update.
        Processes T tokens sequentially, updating state after each token.

        Note: The delta rule operations are fundamentally GEMV (matrix-vector) and
        rank-1 updates, which don't directly benefit from tensor cores. Tensor cores
        are optimized for GEMM (matrix-matrix). To use tensor cores, we would need
        to batch across multiple tokens/heads to form proper GEMM operations.
        """
        # Block indices
        i_bh = tl.program_id(0)
        i_k = tl.program_id(1)
        i_v = tl.program_id(2)

        i_b = i_bh // HV
        i_hv = i_bh % HV

        # GVA head mapping
        h_ratio_q = HV // H_Q
        h_ratio_k = HV // H_K
        i_hq = i_hv // h_ratio_q
        i_hk = i_hv // h_ratio_k

        # Load initial state index for this batch
        i_pool = tl.load(H0_INDICES + i_b)

        # Load A_log and dt_bias for this head
        b_A_log = tl.load(A_LOG + i_hv).to(tl.float32)
        b_dt_bias = tl.load(DT_BIAS + i_hv).to(tl.float32)

        # Block offsets
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        # Load initial hidden state h[V, K] from state[pool, HV, V, K]
        p_h = (
            H
            + i_pool * stride_hp
            + i_hv * stride_hh
            + o_v[:, None] * stride_hv
            + o_k[None, :] * stride_hk
        )
        b_h = tl.load(
            p_h, mask=(o_v[:, None] < V_DIM) & (o_k[None, :] < K_DIM), other=0.0
        ).to(tl.float32)  # [BV, BK]

        # Process each token
        for t in range(T):
            # Load a for this batch, time, head
            b_a = tl.load(A + i_b * stride_ab + t * stride_at + i_hv * stride_ah).to(
                tl.float32
            )
            b_b = tl.load(
                B_GATE + i_b * stride_ab + t * stride_at + i_hv * stride_ah
            ).to(tl.float32)

            # Compute softplus and decay
            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            # Load q, k, v for this timestep
            p_q = (
                Q + i_b * stride_qb + t * stride_qt + i_hq * stride_qh + o_k * stride_qk
            )
            p_k = (
                K + i_b * stride_kb + t * stride_kt + i_hk * stride_kh + o_k * stride_kk
            )
            p_v = (
                V + i_b * stride_vb + t * stride_vt + i_hv * stride_vh + o_v * stride_vv
            )

            b_q = tl.load(p_q, mask=o_k < K_DIM, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=o_k < K_DIM, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=o_v < V_DIM, other=0.0).to(tl.float32)

            # Apply L2 normalization
            if use_qk_l2norm:
                q_norm = tl.sqrt(tl.sum(b_q * b_q) + 1e-8)
                k_norm = tl.sqrt(tl.sum(b_k * b_k) + 1e-8)
                b_q = b_q / q_norm
                b_k = b_k / k_norm

            b_q = b_q * scale

            # Step 1: Apply decay: h *= exp(g)
            b_h = b_h * tl.exp(b_g)

            # Step 2: Delta rule: v -= h @ k (h is [BV, BK], k is [BK])
            # This is GEMV, which doesn't directly use tensor cores efficiently
            # h @ k = sum(h * k[None, :], axis=1) -> [BV]
            b_v = b_v - tl.sum(b_h * b_k[None, :], 1)

            # Step 3: Apply beta gating
            b_v = b_v * b_beta

            # Step 4: Update state: h += outer(v, k) = v[:, None] * k[None, :]
            # This is a rank-1 update
            b_h = b_h + b_v[:, None] * b_k[None, :]

            # Step 5: Compute output: o = h @ q = sum(h * q[None, :], axis=1) -> [BV]
            # This is also GEMV
            b_o = tl.sum(b_h * b_q[None, :], 1)

            # Store output for this timestep
            p_o = (
                O + i_b * stride_ob + t * stride_ot + i_hv * stride_oh + o_v * stride_ov
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_v < V_DIM)

            # Cache intermediate state if needed
            if cache_intermediate_states:
                p_inter = (
                    INTERMEDIATE
                    + i_pool * stride_ip
                    + t * stride_it
                    + i_hv * stride_ih
                    + o_v[:, None] * stride_iv
                    + o_k[None, :] * stride_ik
                )
                tl.store(
                    p_inter,
                    b_h.to(p_inter.dtype.element_ty),
                    mask=(o_v[:, None] < V_DIM) & (o_k[None, :] < K_DIM),
                )

        # Store final state if state update is enabled
        if not disable_state_update:
            tl.store(
                p_h,
                b_h.to(p_h.dtype.element_ty),
                mask=(o_v[:, None] < V_DIM) & (o_k[None, :] < K_DIM),
            )

    def triton_gdn_decode(
        q: torch.Tensor,  # [B, 1, H_Q, K]
        k: torch.Tensor,  # [B, 1, H_K, K]
        v: torch.Tensor,  # [B, 1, HV, V]
        state: torch.Tensor,  # [B, HV, K, V]
        A_log: torch.Tensor,  # [HV]
        a: torch.Tensor,  # [B, 1, HV]
        dt_bias: torch.Tensor,  # [HV]
        b: torch.Tensor,  # [B, 1, HV]
        scale: float,
        output: torch.Tensor,  # [B, 1, HV, V]
        use_qk_l2norm: bool = True,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        """
        Triton-based GDN decode matching SGLang's implementation.
        """
        B, T, H_Q, K_DIM = q.shape
        _, _, H_K, _ = k.shape
        _, _, HV, V_DIM = v.shape

        assert T == 1, "Triton kernel only supports decode (T=1)"

        # Reshape inputs for kernel
        q_flat = q.squeeze(1)  # [B, H_Q, K]
        k_flat = k.squeeze(1)  # [B, H_K, K]
        v_flat = v.squeeze(1)  # [B, HV, V]
        a_flat = a.squeeze(1)  # [B, HV]
        b_flat = b.squeeze(1)  # [B, HV]
        o_flat = output.squeeze(1)  # [B, HV, V]

        # Block sizes
        BK = triton.next_power_of_2(K_DIM)
        BV = triton.next_power_of_2(V_DIM)

        # Limit block sizes (BV smaller to allow more V blocks)
        BV = min(BV, 32)

        # Number of blocks
        NK = triton.cdiv(K_DIM, BK)
        NV = triton.cdiv(V_DIM, BV)

        assert NK == 1, f"Multi-block K not supported: NK={NK}"

        # Launch kernel
        grid = (B * HV, NK, NV)

        fused_sigmoid_gating_delta_rule_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            state,
            A_log,
            a_flat,
            dt_bias,
            b_flat,
            # Strides for q [B, H_Q, K]
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            # Strides for k [B, H_K, K]
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            # Strides for v [B, HV, V]
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            # Strides for o [B, HV, V]
            o_flat.stride(0),
            o_flat.stride(1),
            o_flat.stride(2),
            # Strides for h [B, HV, K, V]
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            # Parameters
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            B=B,
            HV=HV,
            H_Q=H_Q,
            H_K=H_K,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            BK=BK,
            BV=BV,
        )

        return output, state

    @triton.jit
    def fused_sigmoid_gating_delta_rule_kernel_pretranspose(
        # Pointers to matrices
        Q,
        K,
        V,
        O,
        H,  # Hidden state [B, HV, V, K] - V-major (pretranspose) layout
        A_LOG,  # Log decay [HV]
        A,  # Input-dependent decay [B, HV]
        DT_BIAS,  # Decay bias [HV]
        B_GATE,  # Update gate [B, HV]
        # Strides
        stride_qb,
        stride_qh,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vv,
        stride_ob,
        stride_oh,
        stride_ov,
        stride_hb,
        stride_hh,
        stride_hv,  # V dimension stride
        stride_hk,  # K dimension stride
        # Parameters
        softplus_beta: tl.constexpr,
        softplus_threshold: tl.constexpr,
        scale: tl.constexpr,
        use_qk_l2norm: tl.constexpr,
        B: tl.constexpr,
        HV: tl.constexpr,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """
        Triton kernel for pretranspose layout [B, HV, V, K].

        Key difference from nontranspose:
        - State layout: [B, HV, V, K] instead of [B, HV, K, V]
        - h is [BV, BK] instead of [BK, BV]
        - h @ k = sum(h * k[None, :], axis=1) -> [BV]
        - h += outer(v, k) = v[:, None] * k[None, :]
        - o = h @ q = sum(h * q[None, :], axis=1) -> [BV]
        """
        # Block indices
        i_bh = tl.program_id(0)
        i_k = tl.program_id(1)
        i_v = tl.program_id(2)

        i_b = i_bh // HV
        i_hv = i_bh % HV

        # GVA head mapping (num_v_heads > num_q_heads)
        h_ratio_q = HV // H_Q
        h_ratio_k = HV // H_K
        i_hq = i_hv // h_ratio_q
        i_hk = i_hv // h_ratio_k

        # Load A_log and dt_bias for this head
        b_A_log = tl.load(A_LOG + i_hv).to(tl.float32)
        b_dt_bias = tl.load(DT_BIAS + i_hv).to(tl.float32)

        # Load a (input-dependent decay) for this batch and head
        b_a = tl.load(A + i_b * HV + i_hv).to(tl.float32)

        # Load b (update gate) for this batch and head
        b_b = tl.load(B_GATE + i_b * HV + i_hv).to(tl.float32)

        # Compute softplus: softplus(x) = (1/beta) * log(1 + exp(beta*x))
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Block offsets
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        # Load q, k, v
        p_q = Q + i_b * stride_qb + i_hq * stride_qh + o_k * stride_qk
        p_k = K + i_b * stride_kb + i_hk * stride_kh + o_k * stride_kk
        p_v = V + i_b * stride_vb + i_hv * stride_vh + o_v * stride_vv

        b_q = tl.load(p_q, mask=o_k < K_DIM, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=o_k < K_DIM, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=o_v < V_DIM, other=0.0).to(tl.float32)

        # Apply L2 normalization (if enabled)
        if use_qk_l2norm:
            q_norm = tl.sqrt(tl.sum(b_q * b_q) + 1e-8)
            k_norm = tl.sqrt(tl.sum(b_k * b_k) + 1e-8)
            b_q = b_q / q_norm
            b_k = b_k / k_norm

        # Apply scale to q
        b_q = b_q * scale

        # Load hidden state h[V, K] from state[B, HV, V, K] - pretranspose layout
        p_h = (
            H
            + i_b * stride_hb
            + i_hv * stride_hh
            + o_v[:, None] * stride_hv
            + o_k[None, :] * stride_hk
        )
        b_h = tl.load(
            p_h, mask=(o_v[:, None] < V_DIM) & (o_k[None, :] < K_DIM), other=0.0
        ).to(tl.float32)  # [BV, BK]

        # Step 1: Apply decay to hidden state: h *= exp(g)
        b_h = b_h * tl.exp(b_g)

        # Step 2: Delta rule: v -= h @ k = sum(h * k[None, :], axis=1)
        # b_h is [BV, BK], b_k is [BK]
        b_v = b_v - tl.sum(b_h * b_k[None, :], 1)

        # Step 3: Apply beta gating: v *= beta
        b_v = b_v * b_beta

        # Step 4: Update hidden state: h += outer(v, k) = v[:, None] * k[None, :]
        b_h = b_h + b_v[:, None] * b_k[None, :]

        # Step 5: Compute output: o = h @ q = sum(h * q[None, :], axis=1)
        b_o = tl.sum(b_h * b_q[None, :], 1)

        # Store output
        p_o = O + i_b * stride_ob + i_hv * stride_oh + o_v * stride_ov
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_v < V_DIM)

        # Store updated hidden state
        tl.store(
            p_h,
            b_h.to(p_h.dtype.element_ty),
            mask=(o_v[:, None] < V_DIM) & (o_k[None, :] < K_DIM),
        )

    def triton_gdn_decode_pretranspose(
        q: torch.Tensor,  # [B, 1, H_Q, K]
        k: torch.Tensor,  # [B, 1, H_K, K]
        v: torch.Tensor,  # [B, 1, HV, V]
        state: torch.Tensor,  # [B, HV, V, K] - pretranspose layout
        A_log: torch.Tensor,  # [HV]
        a: torch.Tensor,  # [B, 1, HV]
        dt_bias: torch.Tensor,  # [HV]
        b: torch.Tensor,  # [B, 1, HV]
        scale: float,
        output: torch.Tensor,  # [B, 1, HV, V]
        use_qk_l2norm: bool = True,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        """
        Triton-based GDN decode for pretranspose layout [B, HV, V, K].
        """
        B, T, H_Q, K_DIM = q.shape
        _, _, H_K, _ = k.shape
        _, _, HV, V_DIM = v.shape

        assert T == 1, "Triton kernel only supports decode (T=1)"

        # Reshape inputs for kernel
        q_flat = q.squeeze(1)  # [B, H_Q, K]
        k_flat = k.squeeze(1)  # [B, H_K, K]
        v_flat = v.squeeze(1)  # [B, HV, V]
        a_flat = a.squeeze(1)  # [B, HV]
        b_flat = b.squeeze(1)  # [B, HV]
        o_flat = output.squeeze(1)  # [B, HV, V]

        # Block sizes
        BK = triton.next_power_of_2(K_DIM)
        BV = triton.next_power_of_2(V_DIM)

        # Limit block sizes (BV smaller to allow more V blocks)
        BV = min(BV, 32)

        # Number of blocks
        NK = triton.cdiv(K_DIM, BK)
        NV = triton.cdiv(V_DIM, BV)

        assert NK == 1, f"Multi-block K not supported: NK={NK}"

        # Launch kernel
        grid = (B * HV, NK, NV)

        fused_sigmoid_gating_delta_rule_kernel_pretranspose[grid](
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            state,
            A_log,
            a_flat,
            dt_bias,
            b_flat,
            # Strides for q [B, H_Q, K]
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            # Strides for k [B, H_K, K]
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            # Strides for v [B, HV, V]
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            # Strides for o [B, HV, V]
            o_flat.stride(0),
            o_flat.stride(1),
            o_flat.stride(2),
            # Strides for h [B, HV, V, K] - pretranspose layout
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            # Parameters
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            B=B,
            HV=HV,
            H_Q=H_Q,
            H_K=H_K,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            BK=BK,
            BV=BV,
        )

        return output, state

    def triton_gdn_mtp(
        q: torch.Tensor,  # [B, T, H_Q, K]
        k: torch.Tensor,  # [B, T, H_K, K]
        v: torch.Tensor,  # [B, T, HV, V]
        initial_state: torch.Tensor,  # [pool_size, HV, V, K]
        initial_state_indices: torch.Tensor,  # [B]
        A_log: torch.Tensor,  # [HV]
        a: torch.Tensor,  # [B, T, HV]
        dt_bias: torch.Tensor,  # [HV]
        b: torch.Tensor,  # [B, T, HV]
        scale: float,
        output: torch.Tensor,  # [B, T, HV, V]
        intermediate_states_buffer: torch.Tensor = None,  # [pool_size, T, HV, V, K]
        disable_state_update: bool = True,
        use_qk_l2norm: bool = True,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        """
        Triton-based GDN MTP matching SGLang's implementation.
        """
        B, T, H_Q, K_DIM = q.shape
        _, _, H_K, _ = k.shape
        _, _, HV, V_DIM = v.shape

        # Block sizes (BV smaller to allow more V blocks)
        BK = triton.next_power_of_2(K_DIM)
        BV = triton.next_power_of_2(V_DIM)
        BV = min(BV, 32)

        NK = triton.cdiv(K_DIM, BK)
        NV = triton.cdiv(V_DIM, BV)

        assert NK == 1, f"Multi-block K not supported: NK={NK}"

        cache_intermediate_states = intermediate_states_buffer is not None
        if cache_intermediate_states:
            intermediate = intermediate_states_buffer
        else:
            intermediate = torch.zeros(
                1, 1, 1, 1, 1, dtype=torch.float32, device=q.device
            )

        # Launch kernel
        grid = (B * HV, NK, NV)

        fused_sigmoid_gating_delta_rule_mtp_kernel[grid](
            q,
            k,
            v,
            output,
            initial_state,
            intermediate,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            # K strides
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            # V strides
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            # O strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            # H strides [pool_size, HV, V, K]
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            # Intermediate strides [pool_size, T, HV, V, K]
            intermediate.stride(0),
            intermediate.stride(1),
            intermediate.stride(2) if cache_intermediate_states else 0,
            intermediate.stride(3) if cache_intermediate_states else 0,
            intermediate.stride(4) if cache_intermediate_states else 0,
            # A strides [B, T, HV]
            a.stride(0),
            a.stride(1),
            a.stride(2),
            # Parameters
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            disable_state_update=disable_state_update,
            cache_intermediate_states=cache_intermediate_states,
            B=B,
            T=T,
            HV=HV,
            H_Q=H_Q,
            H_K=H_K,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            BK=BK,
            BV=BV,
        )

        return output, initial_state


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
    warmup_iters: int = 10,
    bench_iters: int = 100,
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

    # Initial state: [pool_size, HV, V, K] (K-last layout for MTP)
    pool_size = batch_size
    initial_state = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")

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
            disable_state_update=True,
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
        disable_state_update=True,  # MTP mode: state is not written back
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
            disable_state_update=True,
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
            disable_state_update=True,
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


def gdn_decode_klast_bf16_state_wrapper(
    q: torch.Tensor,  # [B, T, H_Q, K] where T=1,2,3,4
    k: torch.Tensor,  # [B, T, H_K, K]
    v: torch.Tensor,  # [B, T, HV, V]
    state: torch.Tensor,  # [B, HV, V, K] - K-last layout (pretranspose)
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T, HV]
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, T, HV]
    scale: float,
    output: torch.Tensor,  # [B, T, HV, V] - unused, kernel returns output directly
    use_qk_l2norm: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    """
    Wrapper for gdn_decode_klast_bf16_state GDN kernel.
    Supports T=1,2,3,4 (sequence lengths up to 4).
    Adapts the interface to match the benchmark's calling convention.

    Note: The kernel returns output directly, no copy needed.
    """
    if not GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        raise RuntimeError("gdn_decode_klast_bf16_state kernel is not available")

    # Call gdn_decode_klast_bf16_state kernel directly - no wrapper overhead
    # Kernel modifies state in-place and returns output tensor
    return gdn_decode_klast_bf16_state(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        scale=scale,
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

    # ========== gdn_decode_klast_bf16_state Kernel (K-fast/pretranspose layout) ==========
    if GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        # gdn_decode_klast_bf16_state uses [B, HV, V, K] layout (K-fast, same as pretranspose)
        state = torch.randn(
            batch_size,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,  # gdn_decode_klast_bf16_state uses BF16 state
            device="cuda",
        )
        output = torch.empty(
            batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
        )

        try:
            times = bench_gpu_time(
                lambda: gdn_decode_klast_bf16_state_wrapper(
                    q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
                ),
                enable_cupti=True,
                dry_run_iters=warmup_iters,
                repeat_iters=bench_iters,
            )
            results["gdn_decode_klast_bf16_state_us"] = np.median(times) * 1000
        except Exception as e:
            results["gdn_decode_klast_bf16_state_us"] = None
            print(
                f"  gdn_decode_klast_bf16_state kernel failed: {type(e).__name__}: {e}"
            )
    else:
        results["gdn_decode_klast_bf16_state_us"] = None

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
    print(
        "GDN Decode Benchmark (T=1): FlashInfer vs Triton vs gdn_decode_klast_bf16_state"
    )
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, "
        f"dtype={args.dtype}, qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
    )
    print("=" * 160)
    print()
    print(
        f"{'batch':>6} | {'FI-PreTr':>8} {'FI-NonTr':>8} | {'TR-PreTr':>8} {'TR-NonTr':>8} | {'KlastBf16':>9} | "
        f"{'FI/TR-Pre':>9} {'KlastBf16/FI':>11} {'KlastBf16/TR':>11}"
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
        klast_bf16_us = result.get("gdn_decode_klast_bf16_state_us")

        # FI/TR speedup (>1 means FI faster)
        fi_tr_pre = format_speedup(fi_pre, tr_pre)

        # gdn_decode_klast_bf16_state vs FI-PreTr speedup (>1 means klast_bf16 faster)
        klast_bf16_fi_speedup = format_speedup(klast_bf16_us, fi_pre)

        # gdn_decode_klast_bf16_state vs TR-PreTr speedup (>1 means klast_bf16 faster)
        klast_bf16_tr_speedup = format_speedup(klast_bf16_us, tr_pre)

        print(
            f"{batch_size:>6} | {format_time(fi_pre)} {format_time(fi_non)} | "
            f"{format_time(tr_pre)} {format_time(tr_non)} | {format_time(klast_bf16_us)} | "
            f"{fi_tr_pre} {klast_bf16_fi_speedup:>10} {klast_bf16_tr_speedup:>10}"
        )

    print("-" * 160)
    print()
    print("Legend:")
    print("  FI-PreTr  = FlashInfer Pretranspose [B, HV, V, K]")
    print("  FI-NonTr  = FlashInfer Nontranspose [B, HV, K, V]")
    print("  TR-PreTr  = Triton Pretranspose [B, HV, V, K]")
    print("  TR-NonTr  = Triton Nontranspose [B, HV, K, V]")
    print(
        "  KlastBf16 = gdn_decode_klast_bf16_state [B, HV, V, K] (K-fast layout, T=1..4, bf16 state)"
    )
    print("  FI/TR speedup > 1.0 means FlashInfer is faster than Triton")
    print(
        "  KlastBf16/FI speedup > 1.0 means gdn_decode_klast_bf16_state is faster than FlashInfer Pretranspose"
    )
    print(
        "  KlastBf16/TR speedup > 1.0 means gdn_decode_klast_bf16_state is faster than Triton Pretranspose"
    )
    print()

    # Summary statistics
    fi_pre_times = [r["fi_pretrans_us"] for r in all_results if r.get("fi_pretrans_us")]
    tr_pre_times = [r["tr_pretrans_us"] for r in all_results if r.get("tr_pretrans_us")]
    klast_bf16_times = [
        r["gdn_decode_klast_bf16_state_us"]
        for r in all_results
        if r.get("gdn_decode_klast_bf16_state_us")
    ]

    if fi_pre_times and tr_pre_times:
        speedups = [tr / fi for fi, tr in zip(fi_pre_times, tr_pre_times, strict=False)]
        print(
            f"FlashInfer vs Triton (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )

    if klast_bf16_times and fi_pre_times and len(klast_bf16_times) == len(fi_pre_times):
        speedups = [
            fi / t for t, fi in zip(klast_bf16_times, fi_pre_times, strict=False)
        ]
        print(
            f"gdn_decode_klast_bf16_state vs FlashInfer (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )

    if klast_bf16_times and tr_pre_times and len(klast_bf16_times) == len(tr_pre_times):
        speedups = [
            tr / t for t, tr in zip(klast_bf16_times, tr_pre_times, strict=False)
        ]
        print(
            f"gdn_decode_klast_bf16_state vs Triton (Pretranspose) - Average speedup: {np.mean(speedups):.2f}x"
        )


# ============================================================================
# gdn_decode_klast_bf16_state Multi-Token Benchmark (T=1,2,3,4)
# ============================================================================


def bench_gdn_decode_klast_bf16_state(
    batch_size: int,
    seq_len: int,  # T=1,2,3,4
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark gdn_decode_klast_bf16_state kernel for T=1,2,3,4."""
    if not GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        raise RuntimeError("gdn_decode_klast_bf16_state kernel is not available")

    assert seq_len in [1, 2, 3, 4], (
        f"gdn_decode_klast_bf16_state supports T=1,2,3,4, got T={seq_len}"
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

    # Initial state: [B, HV, V, K] (K-fast layout, BF16)
    state = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Pre-allocate output
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: gdn_decode_klast_bf16_state_wrapper(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
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
    # gdn_decode_klast_bf16_state uses BF16 state (2 bytes), not FP32 (4 bytes)
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len,
        disable_state_update=False,
        state_dtype_bytes=2,  # BF16 state for gdn_decode_klast_bf16_state
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


def run_gdn_decode_klast_bf16_state_benchmark(args, dtype, use_qk_l2norm):
    """Run gdn_decode_klast_bf16_state benchmark for T=1,2,3,4."""
    if not GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        print("Error: gdn_decode_klast_bf16_state kernel is not available.")
        print("Make sure flashinfer.cute_dsl.gated_delta_rule is importable.")
        return

    # Filter seq_len to only valid values (1,2,3,4)
    valid_seq_lens = [t for t in args.seq_len if t in [1, 2, 3, 4]]
    if not valid_seq_lens:
        print("Error: --seq-len must include values from [1, 2, 3, 4]")
        return

    print("\n" + "=" * 100)
    print(f"gdn_decode_klast_bf16_state GDN Benchmark (T={valid_seq_lens})")
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, "
        f"dtype={args.dtype}, qk_l2norm={'ON' if use_qk_l2norm else 'OFF'}"
    )
    print("=" * 100)
    print()
    print(f"{'batch':>6} {'T':>4} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10}")
    print("-" * 100)

    all_results = []
    for batch_size in args.batch_size:
        for seq_len in valid_seq_lens:
            try:
                result = bench_gdn_decode_klast_bf16_state(
                    batch_size=batch_size,
                    seq_len=seq_len,
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
                f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'})"
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
                        warmup_iters=args.warmup,
                        bench_iters=args.iters,
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

  # gdn_decode_klast_bf16_state benchmark (T=1,2,3,4)
  python benchmarks/bench_gdn_decode.py --version gdn_decode_klast_bf16_state --batch-size 1 32 128 512
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
            "gdn_decode_klast_bf16_state",
            "all",
        ],
        default="nontranspose",
        help="Kernel version: pretranspose (V-major state), nontranspose (K-major state), mtp (Multiple Token Processing), gdn_decode_klast_bf16_state (T=1..4, bf16 state, K-last), or all",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Sequence lengths: for MTP use T>1, for gdn_decode_klast_bf16_state use T=1,2,3,4",
    )
    parser.add_argument(
        "--cache-intermediate-states",
        action="store_true",
        help="Cache intermediate states for MTP benchmark",
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
    elif args.version == "gdn_decode_klast_bf16_state":
        # gdn_decode_klast_bf16_state benchmark for T=1,2,3,4
        run_gdn_decode_klast_bf16_state_benchmark(args, dtype, use_qk_l2norm)
    else:
        # Non-MTP: always run all layouts comparison (FlashInfer/Triton x pretranspose/nontranspose + gdn_decode_klast_bf16_state)
        run_all_layouts_benchmark(args, dtype, use_qk_l2norm)


if __name__ == "__main__":
    main()
