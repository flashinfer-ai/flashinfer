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

"""
Triton reference implementations of the Gated Delta Rule (GDN) decode/MTP
kernels, following SGLang's fused_sigmoid_gating_delta_rule implementation.

Shared by:
- bench_gdn_decode.py (standalone GDN decode benchmark)
- routines/gdn.py (flashinfer_benchmark.py GDN routines)

Exports:
- TRITON_AVAILABLE: bool, whether triton could be imported
- triton_gdn_decode: nontranspose [B, HV, K, V] state layout, T=1
- triton_gdn_decode_pretranspose: pretranspose [B, HV, V, K] state layout, T=1
- triton_gdn_mtp: MTP (T>=1) with state pool + indices, [pool, HV, V, K] layout
"""

import torch

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
else:
    # Allow `from gdn_triton_reference import ...` to succeed without Triton;
    # all call sites guard on TRITON_AVAILABLE before using these.
    triton_gdn_decode = None
    triton_gdn_decode_pretranspose = None
    triton_gdn_mtp = None
