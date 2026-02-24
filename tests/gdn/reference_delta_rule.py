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

from typing import Optional

import torch
import torch.nn.functional as F


def exclusive_cumsum(a: list[int]):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r


def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    assert b.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    if (
        a.dtype == torch.float16
        or b.dtype == torch.float16
        or a.dtype == torch.bfloat16
        or b.dtype == torch.bfloat16
    ):
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
        c_f32 = a_f32 @ b_f32
        if a.dtype == torch.bfloat16:
            return c_f32
        else:
            return c_f32.to(torch.float16)
    else:
        return a @ b


def LambdaQ(decay_factor, valid_nrows, block_size, device, offset=0):
    e = (
        F.pad(
            torch.arange(valid_nrows, device=device) + offset,
            (0, block_size - valid_nrows),
        )
        .unsqueeze(1)
        .unsqueeze(0)
    )
    return torch.pow(decay_factor, e)


def LambdaK(decay_factor, valid_nrows, block_size, device, offset=0):
    # NOTE: IT IS valid_nrows - ..., NOT block_size - ..., this is crucial for tail blocks
    e = (
        (
            (valid_nrows - offset)
            - F.pad(
                torch.arange(valid_nrows, device=device),
                (0, block_size - valid_nrows),
                value=block_size,
            )
        )
        .unsqueeze(1)
        .unsqueeze(0)
    )
    return torch.pow(decay_factor, e)


# sequence/block level linear attention
def _linear_attention(
    q: torch.Tensor,  # [seq_len, num_heads, head_size]
    k: torch.Tensor,  # [seq_len, num_heads, head_size]
    v: torch.Tensor,  # [seq_len, num_heads, head_size]
    *,
    decay_factor: torch.Tensor | None = None,
    qk_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # Compute Q @ K^T
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    assert num_qo_heads == num_kv_heads
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(q.shape, k.shape, v.shape)
    scores = matmul(q, k.transpose(-2, -1))

    # Create causal mask
    seq_len = q.size(-2)
    mask = torch.tril(
        torch.ones(num_qo_heads, seq_len, seq_len, dtype=q.dtype, device=q.device)
    )
    if decay_factor is not None and (decay_factor != 1.0).any():
        _, sq, sk = mask.shape
        with torch.device(q.device):
            e = (
                torch.arange(sq).unsqueeze(1) - torch.arange(sk).unsqueeze(0)
            ).unsqueeze(0)
            M = torch.pow(decay_factor, e)
            M[mask == 0.0] = 0.0
    elif qk_weight is not None:
        M = qk_weight.clone()
        M[mask == 0.0] = 0.0
    else:
        M = mask

    # Apply mask (Q @ K^T \odot M)
    masked_scores = scores * M

    # Apply to values (Q @ K^T \odot M) V
    out = matmul(masked_scores, v)
    out = out.transpose(0, 1)

    return out


@torch.inference_mode
def blockwise_linear_attention(
    q: torch.Tensor,  # [total_seq_len, num_qo_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    seq_lens: list[int],  # sequence length for each sequence
    block_size: int = 32,
    scale_factor=1.0,
    decay_factor: float
    | torch.Tensor = 1.0,  # float or tensor with num_elems == num_qo_heads
    decay_exponent_offset=0,
    state_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    num_qo_heads = q.size(1)
    head_size = q.size(2)
    num_kv_heads = k.size(1)

    if scale_factor != 1.0:
        k = k * scale_factor
    if isinstance(decay_factor, float):
        decay_factor = torch.ones(num_qo_heads) * decay_factor
        decay_factor = decay_factor.to(q.device)
    assert decay_factor.numel() == num_qo_heads
    decay_factor = decay_factor.reshape(num_qo_heads, 1, 1)

    k = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)

    KVs = []  # FIXME: kernel debug only
    kv = torch.zeros(
        (len(seq_lens), num_qo_heads, head_size, head_size),
        dtype=state_dtype,
        device=q.device,
    )
    output = torch.zeros_like(q)

    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        blk_offset = seq_start
        carried_kv = torch.zeros(
            (num_qo_heads, head_size, head_size), dtype=state_dtype, device=q.device
        )
        while blk_offset < seq_end:
            is_full_block = seq_end - blk_offset >= block_size
            valid_len = block_size if is_full_block else seq_end - blk_offset
            o_t = output[blk_offset : min(seq_end, blk_offset + block_size)]
            if is_full_block:
                q_t = q[blk_offset : blk_offset + block_size]
                k_t = k[blk_offset : blk_offset + block_size]
                v_t = v[blk_offset : blk_offset + block_size]
            else:
                q_t = torch.zeros(
                    (block_size, num_qo_heads, head_size),
                    dtype=q.dtype,
                    device=q.device,
                )
                k_t = torch.zeros(
                    (block_size, num_qo_heads, head_size),
                    dtype=q.dtype,
                    device=q.device,
                )
                v_t = torch.zeros(
                    (block_size, num_qo_heads, head_size),
                    dtype=q.dtype,
                    device=q.device,
                )
                q_t[: seq_end - blk_offset] = q[blk_offset:seq_end]
                k_t[: seq_end - blk_offset] = k[blk_offset:seq_end]
                v_t[: seq_end - blk_offset] = v[blk_offset:seq_end]

            Lq = LambdaQ(
                decay_factor,
                valid_len,
                block_size,
                device=q.device,
                offset=decay_exponent_offset,
            )

            o_inter = (
                matmul(
                    q_t.transpose(0, 1).to(torch.float32) * Lq,
                    carried_kv.to(torch.float32),
                )
                .transpose(0, 1)
                .to(q.dtype)
            )
            o_intra = _linear_attention(q_t, k_t, v_t, decay_factor=decay_factor)
            if is_full_block:
                # print(seq_idx, blk_offset, seq_end, o_t.shape, o_inter.shape, o_intra.shape)
                o_t[:] = o_inter + o_intra
            else:
                # print(seq_idx, blk_offset, seq_end, o_t.shape, o_inter.shape, o_intra.shape)
                o_t[:] = (o_inter + o_intra)[: o_t.shape[0]]

            if (decay_factor == 1.0).all():
                inc_kv = matmul(
                    k_t.transpose(0, 1).transpose(-2, -1).to(torch.float32),
                    v_t.transpose(0, 1).to(torch.float32),
                )
                carried_kv = (carried_kv.to(torch.float32) + inc_kv).to(state_dtype)
            else:
                Lk = LambdaK(
                    decay_factor,
                    valid_len,
                    block_size,
                    device=q.device,
                    offset=decay_exponent_offset,
                )
                inc_kv = matmul(
                    (k_t.transpose(0, 1) * Lk).transpose(-2, -1).to(torch.float32),
                    v_t.transpose(0, 1).to(torch.float32),
                )
                block_decay = decay_factor**valid_len
                carried_kv = (block_decay * carried_kv.to(torch.float32) + inc_kv).to(
                    state_dtype
                )
            KVs.append(carried_kv.clone())

            blk_offset += block_size

        # print(kv.shape, carried_kv.shape)
        kv[seq_idx, :, :] = carried_kv

    return output, kv, KVs


def delta_rule(
    q: torch.Tensor,  # [total_seq_len, num_qo_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    seq_lens: list[int],  # sequence length for each sequence
    *,
    alpha: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    beta: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    scale_factor=1.0,
    state_dtype: torch.dtype = torch.float32,
):
    o = []
    kv = []
    total_seqlen = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size = k.size(2)

    if alpha is None:
        alpha = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )
    if beta is None:
        beta = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )

    if num_q_heads > num_v_heads:  # GQA
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_v_heads, dim=1)
    else:  # GVA
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        seq_len = seq_end - seq_start
        s = slice(seq_start, seq_end)

        # slices
        qs = q[s]
        ks = k[s]
        vs = v[s]
        alphas = alpha[s]
        betas = beta[s]

        state_HKV = torch.zeros(
            num_q_heads, head_size, head_size, dtype=state_dtype, device=q.device
        )
        for i in range(seq_len):
            # var_DS where var is variable basename and DS is the dimensional semantics.
            # Q/K/V are Dq/Dk/Dv respectively
            q_H1Q = qs[i].unsqueeze(1)
            k_H1K = ks[i].unsqueeze(1)
            v_H1V = vs[i].unsqueeze(1)
            alpha_H11 = alphas[i].unsqueeze(1).unsqueeze(2)
            beta_H11 = betas[i].unsqueeze(1).unsqueeze(2)

            ### listed at the bottom of page3 of section 2.2 DELTA NETWORKS: LINEAR ATTENTION WITH DELTA RULE

            # state update rule, use the middle version for clearer dimensional semantics
            # Read state in fp32, compute in fp32, store back in state_dtype
            old_state_HKV = alpha_H11 * state_HKV.to(torch.float32)
            old_v_H1V = matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum("htv,htk->hkv", old_v_H1V, k_H1K)
            state_update = torch.einsum("htv,htk->hkv", new_v_H1V, k_H1K)
            state_HKV[:] = (old_state_HKV - state_remove + state_update).to(state_dtype)

            o_H1V = scale_factor * matmul(q_H1Q, state_HKV.to(torch.float32))
            o.append(o_H1V.squeeze(1))

        kv.append(state_HKV.clone())

    return torch.stack(o), torch.stack(kv)


def identity_add_strict_lower_diagonal(m: torch.Tensor):
    SIZE = m.size(-1)
    assert m.size(-2) == SIZE
    with torch.device(m.device):
        m = m.clone()
        mask = torch.arange(SIZE).unsqueeze(1) <= torch.arange(SIZE)
        m[:, mask] = 0.0
        # m[mask.unsqueeze(0)] = 0.0
        m = m + torch.eye(SIZE).unsqueeze(0)
    return m


def to_logspace_Gamma_and_gamma(alpha_HS: torch.Tensor, epsilon=1e-10):
    g = torch.log(alpha_HS + epsilon)
    cu_g = torch.cumsum(g, dim=-1)
    cu_g_HSS = cu_g.unsqueeze(2) - cu_g.unsqueeze(1)
    cu_g_HS1 = cu_g.unsqueeze(2)
    return cu_g_HSS, cu_g_HS1


@torch.inference_mode
def blockwise_delta_rule(
    q: torch.Tensor,  # [total_seq_len, num_qo_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    seq_lens: list[int],  # sequence length for each sequence
    alpha: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    beta: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    block_size: int = 32,
    scale_factor=1.0,
    state_dtype: torch.dtype = torch.float32,
    # intermediate_outputs = None,  # debug output
) -> torch.Tensor:
    total_seqlen = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size = q.size(2)

    if alpha is None:
        alpha = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )
    if beta is None:
        beta = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )

    if num_q_heads > num_v_heads:  # GQA
        num_qkv_heads = num_q_heads
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_v_heads, dim=1)
    else:  # GVA
        num_qkv_heads = num_v_heads
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    kv = torch.zeros(
        (len(seq_lens), num_sab_heads, head_size, head_size),
        dtype=state_dtype,
        device=q.device,
    )
    output = torch.zeros_like(q)

    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        blk_offset = seq_start
        state_HKV = torch.zeros(
            (num_sab_heads, head_size, head_size), dtype=state_dtype, device=q.device
        )
        while blk_offset < seq_end:
            is_full_block = seq_end - blk_offset >= block_size
            valid_len = block_size if is_full_block else seq_end - blk_offset
            o_t = output[blk_offset : min(seq_end, blk_offset + block_size)]
            if is_full_block:
                q_SHQ = q[blk_offset : blk_offset + block_size]
                k_SHK = k[blk_offset : blk_offset + block_size]
                v_SHV = v[blk_offset : blk_offset + block_size]
                alpha_SH = alpha[blk_offset : blk_offset + block_size]
                beta_SH = beta[blk_offset : blk_offset + block_size]
            else:
                q_SHQ = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=q.dtype,
                    device=q.device,
                )
                k_SHK = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=k.dtype,
                    device=k.device,
                )
                v_SHV = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=v.dtype,
                    device=v.device,
                )
                alpha_SH = torch.ones(
                    block_size, num_sab_heads, dtype=alpha.dtype, device=alpha.device
                )
                beta_SH = torch.zeros(
                    block_size, num_sab_heads, dtype=beta.dtype, device=beta.device
                )
                q_SHQ[:valid_len] = q[blk_offset:seq_end]
                k_SHK[:valid_len] = k[blk_offset:seq_end]
                v_SHV[:valid_len] = v[blk_offset:seq_end]
                alpha_SH[:valid_len] = alpha[blk_offset:seq_end]
                beta_SH[:valid_len] = beta[blk_offset:seq_end]

            alpha_HS = alpha_SH.transpose(0, 1)
            beta_HS1 = beta_SH.transpose(0, 1).unsqueeze(2)
            Gamma_HSS, gamma_HS1 = to_logspace_Gamma_and_gamma(alpha_HS)
            block_gamma = gamma_HS1[:, [valid_len - 1], :]

            q_HSQ = q_SHQ.transpose(0, 1)
            k_HSK = k_SHK.transpose(0, 1)
            v_HSV = v_SHV.transpose(0, 1)

            IKK = identity_add_strict_lower_diagonal(
                beta_HS1 * torch.exp(Gamma_HSS) * matmul(k_HSK, k_HSK.transpose(-2, -1))
            )  # NOTE: beta scale row-wise
            T = torch.inverse(IKK) * beta_HS1.transpose(
                1, 2
            )  # NOTE: beta scale col-wise
            T = T.to(q.dtype)
            # new_v_HSV = matmul(T, (v_HSV - matmul(torch.exp(gamma_HS1) * k_HSK, state_HKV)))
            u_HSV = matmul(T, v_HSV)
            w_HSK = matmul(T, torch.exp(gamma_HS1) * k_HSK)
            new_v_HSV = u_HSV - matmul(
                w_HSK.to(torch.float32), state_HKV.to(torch.float32)
            ).to(u_HSV.dtype)
            new_v_SHV = new_v_HSV.transpose(0, 1)

            # if intermediate_outputs is not None:
            #     intermediate_outputs["G"].append(Gamma_HSS.clone())
            #     intermediate_outputs["g"].append(gamma_HS1.clone())
            #     intermediate_outputs["IKK"].append(IKK.clone())
            #     intermediate_outputs["T"].append(T.clone())
            #     intermediate_outputs["u"].append(u_HSV.clone())
            #     intermediate_outputs["w"].append(w_HSK.clone())
            #     intermediate_outputs["new_v"].append(new_v_HSV.clone())

            o_inter = (
                matmul(
                    torch.exp(gamma_HS1) * q_HSQ.to(torch.float32),
                    state_HKV.to(torch.float32),
                )
                .transpose(0, 1)
                .to(q.dtype)
            )
            o_intra = _linear_attention(
                q_SHQ, k_SHK, new_v_SHV, qk_weight=torch.exp(Gamma_HSS)
            )

            if is_full_block:
                o_t[:] = scale_factor * (o_inter + o_intra)
            else:
                o_t[:] = scale_factor * (o_inter + o_intra)[: o_t.shape[0]]

            inc_HKV = matmul(
                (torch.exp(block_gamma - gamma_HS1) * k_HSK)
                .transpose(-2, -1)
                .to(torch.float32),
                new_v_HSV.to(torch.float32),
            )
            state_HKV = (
                torch.exp(block_gamma) * state_HKV.to(torch.float32) + inc_HKV
            ).to(state_dtype)

            blk_offset += block_size

        kv[seq_idx, :, :, :] = state_HKV

    return output, kv


@torch.inference_mode
def decode_delta_rule(
    q: torch.Tensor,  # [B, num_q_heads, K]
    k: torch.Tensor,  # [B, num_k_heads, K]
    v: torch.Tensor,  # [B, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for single-step decode with GDN formula.

    Strictly follows the Triton kernel logic from fused_sigmoid_gating_recurrent.py:
        1. Compute g = -exp(A_log) * softplus(a + dt_bias)
        2. Compute beta = sigmoid(b)
        3. Apply L2 norm to q and k (if enabled)
        4. h *= exp(g)                    # Apply decay to state
        5. v_new = v - k^T @ h            # Delta rule (h is [K,V], k is [K])
        6. v_new *= beta                  # Apply update gate
        7. h += k @ v_new^T               # Update state (outer product)
        8. o = q^T @ h                    # Compute output (h is [K,V], q is [K])

    Args:
        q: Query [B, num_q_heads, K]
        k: Key [B, num_k_heads, K]
        v: Value [B, num_v_heads, V]
        state: Input state [B, num_heads, K, V], where num_heads = num_v_heads
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, num_heads]
        scale_factor: Scale factor for q
        softplus_beta: Beta parameter for softplus activation
        softplus_threshold: Threshold for softplus numerical stability
        use_l2_norm: Whether to apply L2 normalization to q and k
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: [B, num_heads, V]
        new_state: [B, num_heads, K, V]
    """
    B = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    K = q.size(2)
    V = v.size(2)

    # State and output are always based on num_v_heads (matches kernel's HV dimension)
    num_heads = num_v_heads

    device = q.device
    dtype = torch.float32

    # Convert to float32 for computation
    A_log = A_log.to(dtype).to(device)
    a = a.to(dtype).to(device)
    dt_bias = dt_bias.to(dtype).to(device)
    b = b.to(dtype).to(device)

    # ============================================
    # Compute gating values (following Triton kernel exactly)
    # ============================================

    # Step 1: Compute g = -exp(A_log) * softplus(a + dt_bias)
    # Triton kernel lines 100-109
    x = a + dt_bias  # [B, num_heads]
    beta_x = softplus_beta * x

    # Apply softplus with numerical stability
    # softplus(x) = (1/beta) * log(1 + exp(beta*x)) if beta*x <= threshold, else x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (log-space decay gate)
    # Triton kernel line 109: b_g = -tl.exp(b_A_log) * softplus_x
    g = -torch.exp(A_log) * softplus_x  # [B, num_heads]

    # Step 2: Compute beta = sigmoid(b)
    # Triton kernel line 112: b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, num_heads]

    # Expand heads if needed (for GQA/GVA)
    # The reference works at v_heads level
    # For GQA (num_q_heads > num_v_heads): k and q need to be averaged/pooled per v_head
    # For GVA (num_v_heads > num_q_heads): q and k need to be repeated
    if num_k_heads < num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    if num_q_heads < num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    elif num_q_heads > num_v_heads:
        # GQA: multiple q_heads per v_head, reshape and average
        # [B, num_q_heads, K] -> [B, num_v_heads, num_q_heads//num_v_heads, K]
        q = q.reshape(B, num_v_heads, num_q_heads // num_v_heads, K).mean(dim=2)
        if num_k_heads == num_q_heads:
            k = k.reshape(B, num_v_heads, num_k_heads // num_v_heads, K).mean(dim=2)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    state = state.to(dtype)

    # Apply L2 normalization if requested
    if use_l2_norm:
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

    # Apply scale to q
    q = q * scale_factor

    # ============================================
    # Process each batch and head
    # ============================================
    new_state = torch.zeros(B, num_heads, K, V, device=device, dtype=state_dtype)
    output = torch.zeros(B, num_heads, V, device=device, dtype=dtype)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            # Get current vectors
            q_h = q[b_idx, h_idx]  # [K]
            k_h = k[b_idx, h_idx]  # [K]
            v_h = v[b_idx, h_idx]  # [V]
            h_state = (
                state[b_idx, h_idx].clone().to(torch.float32)
            )  # [K, V] read as fp32

            # Get gating values for this batch and head
            g_val = g[b_idx, h_idx]  # scalar
            beta_val = beta[b_idx, h_idx]  # scalar

            # ============================================
            # Recurrent update (following Triton kernel lines 121-134)
            # ============================================

            # Step 1: Apply gating to hidden state: h *= exp(g)
            # Triton kernel line 122: b_h *= tl.exp(b_g)
            h_state = h_state * torch.exp(g_val)

            # Step 2: Delta rule: v -= sum(h * k, dim=0)
            # Triton kernel line 125: b_v -= tl.sum(b_h * b_k[:, None], 0)
            # Triton: b_h is [BK, BV], b_k is [BK]
            # b_k[:, None] makes it [BK, 1]
            # b_h * b_k[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: k^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            v_new = v_h - (k_h @ h_state)

            # Step 3: Apply beta gating: v *= beta
            # Triton kernel line 128: b_v *= b_beta
            v_new = v_new * beta_val

            # Step 4: Update hidden state: h += k[:, None] * v[None, :]
            # Triton kernel line 131: b_h += b_k[:, None] * b_v[None, :]
            # Triton: [BK, BV] += [BK, 1] * [1, BV]
            # This is outer product: k @ v^T
            # [K, V] += [K, 1] @ [1, V]
            h_state = h_state + k_h.unsqueeze(1) @ v_new.unsqueeze(0)

            # Step 5: Compute output: o = sum(h * q, dim=0)
            # Triton kernel line 134: b_o = tl.sum(b_h * b_q[:, None], 0)
            # Triton: b_h is [BK, BV], b_q is [BK]
            # b_q[:, None] makes it [BK, 1]
            # b_h * b_q[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: q^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            output[b_idx, h_idx] = q_h @ h_state

            # Store updated state (cast back to state_dtype)
            new_state[b_idx, h_idx] = h_state.to(state_dtype)

    return output, new_state


@torch.inference_mode
def verify_delta_rule(
    q: torch.Tensor,  # [B, T, num_q_heads, K]
    k: torch.Tensor,  # [B, T, num_k_heads, K]
    v: torch.Tensor,  # [B, T, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, T, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, T, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    cache_intermediate_states: bool = False,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Reference implementation for multi-token (verify mode) delta rule.

    Processes T tokens sequentially, updating the state after each token.
    Optionally caches intermediate states for rollback in speculative decoding.

    Args:
        q: Query tensor [B, T, num_q_heads, K]
        k: Key tensor [B, T, num_k_heads, K]
        v: Value tensor [B, T, num_v_heads, V]
        state: Initial state tensor [B, num_heads, K, V]
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, T, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, T, num_heads]
        scale_factor: Scaling factor for queries
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus approximation
        use_l2_norm: Whether to apply L2 normalization
        cache_intermediate_states: Whether to cache state at each time step
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: Output tensor [B, T, num_heads, V]
        new_state: Final state tensor [B, num_heads, K, V]
        intermediate_states: Cached intermediate states [B, T, num_heads, K, V] or None
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = state.shape[1]

    # Handle GQA/GVA: expand or average heads
    if num_q_heads != num_heads:
        # Expand q heads to match num_heads (num_v_heads)
        assert num_heads % num_q_heads == 0
        repeat_factor = num_heads // num_q_heads
        q = q.repeat_interleave(repeat_factor, dim=2)  # [B, T, num_heads, K]

    if num_k_heads != num_heads:
        # Expand k heads to match num_heads (num_v_heads)
        assert num_heads % num_k_heads == 0
        repeat_factor = num_heads // num_k_heads
        k = k.repeat_interleave(repeat_factor, dim=2)  # [B, T, num_heads, K]

    # Convert to float32 for computation
    q = q.float()
    k = k.float()
    v = v.float()
    state = state.float()
    A_log = A_log.float()
    a = a.float()
    dt_bias = dt_bias.float()
    b = b.float()

    # Pre-compute gating values for all time steps
    # Shape: [B, T, num_heads]
    x = a + dt_bias.unsqueeze(0).unsqueeze(0)  # [B, T, num_heads]
    beta_x = softplus_beta * x

    # Softplus with threshold
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (decay factor, already includes exp)
    g = torch.exp(
        -torch.exp(A_log.unsqueeze(0).unsqueeze(0)) * softplus_x
    )  # [B, T, num_heads]

    # Compute beta (update gate)
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, T, num_heads]

    # Apply L2 normalization if needed
    if use_l2_norm:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

    # Apply scaling to q
    q = q * scale_factor

    # Initialize output and intermediate states
    output = torch.zeros(B, T, num_heads, V, dtype=torch.float32, device=q.device)
    current_state = state.clone().to(
        state_dtype
    )  # [B, num_heads, K, V] stored in state_dtype

    if cache_intermediate_states:
        intermediate_states = torch.zeros(
            B, T, num_heads, K, V, dtype=state_dtype, device=q.device
        )
    else:
        intermediate_states = None

    # Process each time step sequentially
    for t in range(T):
        q_t = q[:, t]  # [B, num_heads, K]
        k_t = k[:, t]  # [B, num_heads, K]
        v_t = v[:, t]  # [B, num_heads, V]
        g_t = g[:, t]  # [B, num_heads]
        beta_t = beta[:, t]  # [B, num_heads]

        # Process each batch and head
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_h = q_t[b_idx, h_idx]  # [K]
                k_h = k_t[b_idx, h_idx]  # [K]
                v_h = v_t[b_idx, h_idx]  # [V]
                h_state = (
                    current_state[b_idx, h_idx].clone().to(torch.float32)
                )  # [K, V] read as fp32
                g_val = g_t[b_idx, h_idx]
                beta_val = beta_t[b_idx, h_idx]

                # Recurrent update (following Triton kernel)
                # 1. Apply decay
                h_state = h_state * g_val

                # 2. Compute prediction error: v - k^T @ h
                v_pred = k_h @ h_state  # [K] @ [K, V] = [V]
                v_new = v_h - v_pred

                # 3. Apply gating
                v_new = v_new * beta_val

                # 4. Update state: h = h + k ⊗ v_new
                h_state = h_state + k_h.unsqueeze(1) @ v_new.unsqueeze(
                    0
                )  # [K, V] + [K, 1] @ [1, V]

                # 5. Compute output: o = q^T @ h
                output[b_idx, t, h_idx] = q_h @ h_state  # [K] @ [K, V] = [V]

                # Update current state (cast back to state_dtype)
                current_state[b_idx, h_idx] = h_state.to(state_dtype)

                # Cache intermediate state if requested
                if cache_intermediate_states:
                    intermediate_states[b_idx, t, h_idx] = h_state.to(state_dtype)

    return output, current_state, intermediate_states
