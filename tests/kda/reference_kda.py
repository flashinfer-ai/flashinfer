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

import torch


def exclusive_cumsum(int_list: list[int]) -> list[int]:
    result = [0]
    for i in int_list:
        result.append(result[-1] + i)
    return result


def recurrent_kda_ref(
    q: torch.Tensor,  # [total_seq_len, num_q_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, num_k_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, num_v_heads, head_size]
    seq_lens: list[int],  # sequence length for each sequence
    g: torch.Tensor
    | None = None,  # [total_seq_len, num_sab_heads, head_size], LOG space
    beta: torch.Tensor | None = None,  # [total_seq_len, num_sab_heads] scalar
    scale_factor: float = 1.0,
    initial_state: torch.Tensor | None = None,  # [num_seqs, num_sab_heads, K, V]
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token fp32 KDA reference (adapted from recurrent_gdn2 with
    w = beta broadcast over channels).

    Recurrence per token t (g channel-wise on the key axis, LOG space; beta a
    per-token scalar broadcast over both key and value axes):

        S     = S * exp(g_t)[:, None]
        v_new = beta_t * (v_t - k_t^T @ S)
        S     = S + k_t (x) v_new
        o_t   = scale_factor * q_t @ S

    Returns ``(output [total, num_sab_heads, V], state [num_seqs,
    num_sab_heads, K, V])`` -- the state is K-major; the kernel's k-last
    output must be transposed before comparison.
    """
    total_seqlen = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size = q.size(2)

    if g is None:
        g = torch.zeros(
            total_seqlen, num_sab_heads, head_size, dtype=torch.float32, device=q.device
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

    qf = q.float()
    kf = k.float()
    vf = v.float()
    af = g.float().exp()
    bf = beta.float()

    kv = torch.zeros(
        (len(seq_lens), num_sab_heads, head_size, head_size),
        dtype=state_dtype,
        device=q.device,
    )
    output = torch.zeros(
        total_seqlen, num_sab_heads, head_size, dtype=torch.float32, device=q.device
    )

    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        if initial_state is not None:
            state = initial_state[seq_idx].float().clone()  # [H, K, V]
        else:
            state = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=q.device,
            )
        for t in range(seq_start, seq_end):
            state = state * af[t].unsqueeze(-1)
            erase = torch.einsum("hk,hkv->hv", kf[t], state)
            v_new = bf[t].unsqueeze(-1) * (vf[t] - erase)
            state = state + torch.einsum("hk,hv->hkv", kf[t], v_new)
            output[t] = scale_factor * torch.einsum("hk,hkv->hv", qf[t], state)
        kv[seq_idx] = state.to(state_dtype)

    return output.to(q.dtype), kv
