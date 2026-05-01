# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for Gated Delta Net (GDN) operations."""

import math

import torch
import torch.nn.functional as F

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── GDN decode ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).

    State layout: [B, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = (
                state_f32[b_idx, h_idx].clone().transpose(-1, -2)
            )  # [V,K] -> [K,V]
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K]

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state


gated_delta_rule_decode_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_decode",
    description=(
        "Gated Delta Net decode with GVA configuration and k-last state layout. "
        "Single-token generation with recurrent state update."
    ),
    axes={
        "batch_size": Var(
            description="Number of sequences being decoded concurrently."
        ),
        "seq_len": Const(
            description="Sequence length (always 1 for single-token decode).", abbrev=""
        ),
        "num_q_heads": Const(
            description="Number of query heads (same as key heads in GVA mode).",
            abbrev="qk",
        ),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(
            description="Number of value heads (GVA: more value heads than query heads).",
            abbrev="v",
        ),
        "head_size": Const(
            description="Dimension of each attention head (K dimension in query/key space, V dimension in value space).",
            abbrev="d",
        ),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "seq_len", "num_q_heads", "head_size"],
            description="Query tensor for single token decode.",
        ),
        "k": Tensor(
            ["batch_size", "seq_len", "num_k_heads", "head_size"],
            description="Key tensor for single token decode.",
        ),
        "v": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            description="Value tensor for single token decode.",
        ),
        "state": Tensor(
            ["batch_size", "num_v_heads", "head_size", "head_size"],
            optional=True,
            description="Recurrent state in k-last layout [B, H, V, K].",
        ),
        "A_log": Tensor(
            ["num_v_heads"],
            description="Log decay parameter (learnable). Used to compute g = exp(-exp(A_log) * softplus(a + dt_bias)).",
        ),
        "a": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Input-dependent decay from projection.",
        ),
        "dt_bias": Tensor(
            ["num_v_heads"],
            description="Decay bias (learnable). Added to 'a' before softplus.",
        ),
        "b": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Update gate input from projection. beta = sigmoid(b).",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Scale factor. Default is 1/sqrt(head_size).",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            dtype="bfloat16",
            description="Attention output. Shape follows num_v_heads in GVA mode.",
        ),
        "new_state": Tensor(
            ["batch_size", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Updated recurrent state in k-last layout [B, H, V, K].",
        ),
    },
    constraints=[
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "num_k_heads == num_q_heads",
    ],
    tags=["stage:decode", "status:verified"],
    reference=_gdn_decode_reference,
)

# ── GDN prefill ───────────────────────────────────────────────────────────────


@torch.no_grad()
def _gdn_prefill_reference(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill reference implementation (k-last layout).

    State layout: [H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()  # [total_seq_len, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_seq_len, HV]
    beta = torch.sigmoid(b.float())  # [total_seq_len, HV]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = (
                state[seq_idx].clone().float().transpose(-1, -2)
            )  # [H,V,K] -> [H,K,V]
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = q_H1K.float() @ old_state_HKV  # reuse shape pattern
            old_v_H1V = k_H1K @ old_state_HKV
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum(
                "hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V
            )
            state_update = torch.einsum(
                "hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V
            )
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * (q_H1K @ state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state


gdn_prefill_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_prefill",
    description=(
        "Gated Delta Net prefill with GVA configuration and k-last state layout. "
        "The state is in k-last layout [N, H, V, K]."
    ),
    axes={
        "total_seq_len": Var(
            description="Total number of tokens across all sequences in the batch."
        ),
        "num_seqs": Var(description="Number of sequences in the batch."),
        "num_q_heads": Const(
            description="Number of query heads (same as key heads in GVA mode).",
            abbrev="qk",
        ),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(
            description="Number of value heads (GVA: more value heads than query heads).",
            abbrev="v",
        ),
        "head_size": Const(
            description="Dimension of each attention head (K dimension in query/key space, V dimension in value space).",
            abbrev="d",
        ),
        "len_cu_seqlens": Var(description="Length of cu_seqlens array (num_seqs + 1)."),
    },
    inputs={
        "q": Tensor(
            ["total_seq_len", "num_q_heads", "head_size"],
            description="Query tensor.",
        ),
        "k": Tensor(
            ["total_seq_len", "num_k_heads", "head_size"],
            description="Key tensor.",
        ),
        "v": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            description="Value tensor.",
        ),
        "state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            param="initial_state",
            optional=True,
            description="Recurrent state in k-last layout [N, H, V, K].",
        ),
        "A_log": Tensor(
            ["num_v_heads"],
            optional=True,
            description="Log decay parameter (conceptual; not passed directly — precomputed into g).",
        ),
        "a": Tensor(
            ["total_seq_len", "num_v_heads"],
            param="g",
            description="Precomputed gate values (g = exp(-exp(A_log) * softplus(a + dt_bias))).",
        ),
        "dt_bias": Tensor(
            ["num_v_heads"],
            optional=True,
            description="Decay bias (conceptual; not passed directly — precomputed into g).",
        ),
        "b": Tensor(
            ["total_seq_len", "num_v_heads"],
            param="beta",
            description="Update gate values (beta = sigmoid(b)).",
        ),
        "cu_seqlens": Tensor(
            ["len_cu_seqlens"],
            description="Cumulative sequence lengths for variable-length batching.",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Scale factor. Default is 1/sqrt(head_size).",
        ),
    },
    outputs={
        "output": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            dtype="bfloat16",
            description="Attention output. Shape follows num_v_heads in GVA mode.",
        ),
        "new_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Updated recurrent state in k-last layout [N, H, V, K].",
        ),
    },
    constraints=[
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "num_k_heads == num_q_heads",
        "len_cu_seqlens == num_seqs + 1",
        "total_seq_len == cu_seqlens[-1].item()",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_gdn_prefill_reference,
)

# ── GDN MTP (Multi-Token Prediction) ─────────────────────────────────────────


@torch.no_grad()
def _gdn_mtp_reference(
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
    intermediate_states_buffer=None,
):
    """
    Gated Delta Net MTP (Multi-Token Prediction) reference implementation.

    State layout: [pool_size, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    For each token t in sequence:
        state_new = g_t * state_old + k_t^T @ (beta_t * v_t + (1-beta_t) * k_t @ state_old) - k_t^T @ (k_t @ state_old)
        output_t = scale * q_t @ state_new
        state_old = state_new  # Update for next token
    """
    B, T, num_q_heads, head_size = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, _ = v.shape
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()  # [B, T, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, T, HV]
    beta = torch.sigmoid(b.float())  # [B, T, HV]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=2)  # [B, T, HV, K]
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)  # [B, T, HV, K]

    output = torch.zeros(
        (B, T, num_v_heads, head_size), dtype=torch.bfloat16, device=device
    )
    cache_intermediate = intermediate_states_buffer is not None
    final_state = initial_state.clone().float()

    for b_idx in range(B):
        state_idx = int(initial_state_indices[b_idx].item())
        state_HVK = (
            initial_state[state_idx].clone().float().transpose(-1, -2)
        )  # [H,V,K] -> [H,K,V]

        for t in range(T):
            q_HK = q_exp[b_idx, t].float()  # [HV, K]
            k_HK = k_exp[b_idx, t].float()  # [HV, K]
            v_HV = v[b_idx, t].float()  # [HV, V]
            g_H = g[b_idx, t]  # [HV]
            beta_H = beta[b_idx, t]  # [HV]

            for h_idx in range(num_v_heads):
                q_h = q_HK[h_idx]
                k_h = k_HK[h_idx]
                v_h = v_HV[h_idx]
                h_state = state_HVK[h_idx]
                g_val = g_H[h_idx]
                beta_val = beta_H[h_idx]

                old_state = g_val * h_state
                old_v = k_h @ old_state
                new_v = beta_val * v_h + (1 - beta_val) * old_v
                state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
                state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
                h_state = old_state - state_remove + state_update

                output[b_idx, t, h_idx] = (scale * (q_h @ h_state)).to(torch.bfloat16)
                state_HVK[h_idx] = h_state

            if cache_intermediate:
                intermediate_states_buffer[state_idx, t] = state_HVK.transpose(
                    -1, -2
                )  # [H,K,V] -> [H,V,K]

        # Commit accumulated state back to the pool slot [H,K,V] -> [H,V,K].
        final_state[state_idx] = state_HVK.transpose(-1, -2)

    return output, final_state


gdn_mtp_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_mtp",
    description=(
        "Gated Delta Net Multi-Token Prediction (MTP) with GVA configuration. "
        "Used for speculative decoding verification where multiple tokens (T > 1) "
        "need to be processed in sequence. State layout is k-last [pool_size, H, V, K]."
    ),
    axes={
        "batch_size": Var(
            description="Number of sequences being verified concurrently."
        ),
        "seq_len": Var(description="Number of tokens to process (T > 1 for MTP)."),
        "num_q_heads": Const(
            description="Number of query heads (same as key heads in GVA mode).",
            abbrev="qk",
        ),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(
            description="Number of value heads (GVA: more value heads than query heads).",
            abbrev="v",
        ),
        "head_size": Const(
            description="Dimension of each attention head (K dimension in query/key space, V dimension in value space).",
            abbrev="d",
        ),
        "pool_size": Var(description="Size of the state pool for efficient batching."),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "seq_len", "num_q_heads", "head_size"],
            description="Query tensor for multiple tokens.",
        ),
        "k": Tensor(
            ["batch_size", "seq_len", "num_k_heads", "head_size"],
            description="Key tensor for multiple tokens.",
        ),
        "v": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            description="Value tensor for multiple tokens.",
        ),
        "initial_state": Tensor(
            ["pool_size", "num_v_heads", "head_size", "head_size"],
            description="Initial recurrent state pool in k-last layout [pool_size, H, V, K].",
        ),
        "initial_state_indices": Tensor(
            ["batch_size"],
            description="Indices mapping each batch to its initial state in the pool.",
        ),
        "A_log": Tensor(
            ["num_v_heads"],
            description="Log decay parameter (learnable). Used to compute g = exp(-exp(A_log) * softplus(a + dt_bias)).",
        ),
        "a": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Input-dependent decay from projection.",
        ),
        "dt_bias": Tensor(
            ["num_v_heads"],
            description="Decay bias (learnable). Added to 'a' before softplus.",
        ),
        "b": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Update gate input from projection. beta = sigmoid(b).",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Scale factor. Default is 1/sqrt(head_size).",
        ),
        "intermediate_states_buffer": Tensor(
            ["pool_size", "seq_len", "num_v_heads", "head_size", "head_size"],
            optional=True,
            description="Optional buffer for caching intermediate states for potential rollback.",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            dtype="bfloat16",
            description="Attention output for all T tokens. Shape follows num_v_heads in GVA mode.",
        ),
        "final_state": Tensor(
            ["pool_size", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Updated recurrent state pool in k-last layout [pool_size, H, V, K].",
        ),
    },
    constraints=[
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "num_k_heads == num_q_heads",
        "seq_len > 1",
    ],
    tags=["stage:mtp", "status:verified"],
    reference=_gdn_mtp_reference,
)
