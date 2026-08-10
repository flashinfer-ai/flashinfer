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

Reference implementation of Gated DeltaProduct (arXiv:2502.10297).

Layout note: the householder axis sits immediately after the token axis, so
``x.reshape(total_seq_len * n_h, H, D)`` yields the ``(t n) h d`` ordering the
kernel wrappers expand into.
"""

from __future__ import annotations

import torch

from .reference_delta_rule import exclusive_cumsum, matmul


def delta_product(
    q: torch.Tensor,  # [total_seq_len,      num_q_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, n_h, num_k_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, n_h, num_v_heads, head_size]
    seq_lens: list[int],
    *,
    alpha: torch.Tensor | None = None,  # [total_seq_len,      num_sab_heads]
    beta: torch.Tensor | None = None,  # [total_seq_len, n_h, num_sab_heads]
    scale_factor: float = 1.0,
    state_dtype: torch.dtype = torch.float32,
):
    """Returns (output, final_state).

    output      [total_seq_len, num_o_heads, head_size]   -- one row per REAL token
    final_state [num_seqs, num_sab_heads, head_size, head_size]   (K-major, [.., K, V])
    """
    assert k.dim() == 4 and v.dim() == 4, (
        "k/v must have a householder axis: [T, n_h, H, D]"
    )
    n_h = k.size(1)
    assert v.size(1) == n_h, (
        f"k/v num_householders must match, got {n_h} != {v.size(1)}"
    )

    total_seqlen = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(2)
    num_v_heads = v.size(2)
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size = k.size(3)

    if alpha is None:
        # same alpha for all householders for each real token
        alpha = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )
    if beta is None:
        # beta is per-householder
        beta = torch.ones(
            total_seqlen, n_h, num_sab_heads, dtype=torch.float32, device=q.device
        )

    # Broadcast heads exactly as reference_delta_rule does, but k/v have the
    # extra householder axis at dim 1, so head expansion happens at dim 2.
    if num_q_heads > num_v_heads:  # GQA
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=2)
        v = v.repeat_interleave(num_q_heads // num_v_heads, dim=2)
    else:  # GVA
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)
    num_state_heads = q.size(1)

    o = []
    kv = []
    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        seq_len = seq_end - seq_start
        s = slice(seq_start, seq_end)

        qs, ks, vs = q[s], k[s], v[s]
        alphas, betas = alpha[s], beta[s]

        # state size remains fixed between GDN and GDP
        state_HKV = torch.zeros(
            num_state_heads, head_size, head_size, dtype=state_dtype, device=q.device
        )

        for i in range(seq_len):
            alpha_H11 = alphas[i].unsqueeze(1).unsqueeze(2)
            q_H1Q = qs[i].unsqueeze(1)

            # apply alpha gating ONCE per real token
            old_state_HKV = alpha_H11 * state_HKV.to(torch.float32)
            for j in range(n_h):
                k_H1K = ks[i, j].unsqueeze(1)
                v_H1V = vs[i, j].unsqueeze(1)
                beta_H11 = betas[i, j].unsqueeze(1).unsqueeze(2)

                old_v_H1V = matmul(k_H1K, old_state_HKV)
                new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
                state_remove = torch.einsum("htv,htk->hkv", old_v_H1V, k_H1K)
                state_update = torch.einsum("htv,htk->hkv", new_v_H1V, k_H1K)
                old_state_HKV = old_state_HKV - state_remove + state_update

            state_HKV[:] = old_state_HKV.to(state_dtype)
            o_H1V = scale_factor * matmul(q_H1Q, state_HKV.to(torch.float32))
            o.append(o_H1V.squeeze(1))

        kv.append(state_HKV.clone())

    return torch.stack(o), torch.stack(kv)


def expand_to_flat_sequence(
    k: torch.Tensor,  # [T, n_h, H, D]
    v: torch.Tensor,  # [T, n_h, H, D]
    alpha: torch.Tensor | None,  # [T, H]
    beta: torch.Tensor | None,  # [T, n_h, H]
    q: torch.Tensor,  # [T, H, D]
    cu_seqlens: torch.Tensor,
):
    """The Phase-1 expansion: GDP as GDN on a sequence n_h times longer.

    Returns (q, k, v, alpha, beta, cu_seqlens) all on the expanded token axis,
    ready to hand to ``chunk_gated_delta_rule``. Slice the result with
    ``out[n_h - 1 :: n_h]`` to recover one row per real token.

    NOTE: flashinfer's ``g`` is MULTIPLICATIVE alpha (neutral value 1.0), not the
    log-space decay FLA uses (neutral value 0.0).
    """
    raise NotImplementedError("step 2 -- write this when you add the wrapper")
