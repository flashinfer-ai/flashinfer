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


def _gdn_decode_init(
    *,
    batch_size: int,
    seq_len: int = 1,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdn_decode.gated_delta_rule_decode``.

    Sourced from ``tests/gdn/test_decode_delta_rule.py`` (``gated_delta_rule_decode``
    fixture): k is L2-normalized for numerical stability; ``A_log``,
    ``dt_bias``, ``a`` are scaled by 0.1; state is full ``randn`` (not zeros).
    """
    torch.manual_seed(seed)
    q = torch.randn(
        batch_size, seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch_size, seq_len, num_k_heads, head_size, dtype=torch.bfloat16, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)  # numerical stability
    v = torch.randn(
        batch_size, seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device
    )
    state = torch.randn(
        batch_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(
            batch_size, seq_len, num_v_heads, dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(
        batch_size, seq_len, num_v_heads, dtype=torch.bfloat16, device=device
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
    }


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
    init=_gdn_decode_init,
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


def _gdn_prefill_init(
    *,
    total_seq_len: int,
    num_seqs: int = 4,
    len_cu_seqlens: int = 0,  # derived
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdn_prefill.chunk_gated_delta_rule``.

    Sourced from ``tests/gdn/test_prefill_delta_rule.py`` + the
    ``gen_qkv`` fixture in ``tests/gdn/conftest.py``: each row of Q/K/V
    is drawn from its own ``Uniform(mean - 0.25, mean + 0.25)`` where
    the means come from ``Normal(0, 0.05)`` — see ``multidist_randu``.
    ``k`` is then L2-normalized for numerical stability; ``g``/``beta``
    are ``rand`` in [0, 1] (the kernel takes precomputed gate values).
    """
    del len_cu_seqlens
    torch.manual_seed(seed)

    def _multidist_randu(num_dists: int, dim: int) -> torch.Tensor:
        # Mirrors tests/gdn/conftest.py::multidist_randu(mean_std=0.05,
        # lower=-0.25, upper=0.25): per-row mean drawn from N(0, 0.05),
        # samples drawn from Uniform(mean - 0.25, mean + 0.25).
        means = torch.distributions.Normal(0.0, 0.05).sample((num_dists,))
        data = torch.distributions.Uniform(means - 0.25, means + 0.25).sample((dim,))
        return data.T.contiguous()

    q = (
        _multidist_randu(total_seq_len * num_q_heads, head_size)
        .reshape(total_seq_len, num_q_heads, head_size)
        .to(torch.bfloat16)
        .contiguous()
        .to(device)
    )
    k = (
        _multidist_randu(total_seq_len * num_k_heads, head_size)
        .reshape(total_seq_len, num_k_heads, head_size)
        .to(torch.bfloat16)
        .contiguous()
        .to(device)
    )
    v = (
        _multidist_randu(total_seq_len * num_v_heads, head_size)
        .reshape(total_seq_len, num_v_heads, head_size)
        .to(torch.bfloat16)
        .contiguous()
        .to(device)
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    base = total_seq_len // max(1, num_seqs)
    rem = total_seq_len % max(1, num_seqs)
    cum = [0]
    for i in range(num_seqs):
        cum.append(cum[-1] + base + (1 if i < rem else 0))
    cu_seqlens = torch.tensor(cum, dtype=torch.int64, device=device)
    # Trace template uses param="g"/"beta": kernel sees precomputed gate values.
    num_sab_heads = max(num_q_heads, num_v_heads)
    g = torch.rand(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
    beta = torch.rand(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu_seqlens": cu_seqlens}


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
    init=_gdn_prefill_init,
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


def _gdn_mtp_init(
    *,
    batch_size: int,
    seq_len: int = 4,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    pool_size: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdn_decode.gated_delta_rule_mtp``.

    Sourced from ``tests/gdn/test_decode_delta_rule.py`` (MTP fixture):
    same per-token distributions as the decode path (k L2-normalized,
    A_log/dt_bias/a scaled by 0.1, ``b`` and ``initial_state`` from
    ``randn``). ``initial_state_indices`` maps each batch row to a
    distinct slot in the state pool.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        batch_size, seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        batch_size, seq_len, num_k_heads, head_size, dtype=torch.bfloat16, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(
        batch_size, seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device
    )
    init_state = torch.randn(
        pool_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    init_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(
            batch_size, seq_len, num_v_heads, dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(
        batch_size, seq_len, num_v_heads, dtype=torch.bfloat16, device=device
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": init_state,
        "initial_state_indices": init_idx,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
    }


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
    init=_gdn_mtp_init,
)

# ── GDN fused decode step ─────────────────────────────────────────────────────


@torch.no_grad()
def _gdn_fused_decode_reference(
    hidden_states,
    w_ba,
    mixed_qkv,
    conv_weight,
    conv_bias,
    conv_state,
    A_log,
    dt_bias,
    scale,
    ssm_state,
    state_indices,
    use_qk_l2norm=True,
    out=None,
):
    """Reference for the fused GDN decode step (one token, paged pools).

    The whole serving chain in one op, in the order the kernels fuse it:

    1. ``ba = hidden_states @ w_ba`` in fp32, **rounded through bf16** — the
       reference materializes ``ba`` as a bf16 tensor, so a kernel that keeps
       fp32 here is more precise than the operation it implements;
    2. depthwise causal conv1d update (width 4, silu) over ``mixed_qkv``, with
       the paged bf16 conv-state rows at ``state_indices`` shifting left and
       appending the raw input;
    3. q/k/v head split of the activated conv output;
    4. gated delta-rule update with qk-L2-norm on the paged fp32 state pool.

    ``softplus`` is the threshold form on purpose: ``log(1 + exp(x))``
    overflows to ``+inf`` above ~88.7 in fp32 and silently collapses the decay
    gate to zero.

    Both pools are updated **in place** and returned alongside the output.
    """
    B = hidden_states.shape[0]
    hv = A_log.shape[0]
    qkv_dim = mixed_qkv.shape[1]
    d = ssm_state.shape[-1]
    h_q = (qkv_dim - hv * d) // (2 * d)
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(d)
    idx = state_indices.to(torch.long)

    ba = (hidden_states.float() @ w_ba.float()).to(torch.bfloat16)
    b_gate = ba[:, :hv]
    a_gate = ba[:, hv:]

    st = conv_state.index_select(0, idx)
    x_t = mixed_qkv.to(conv_state.dtype)
    window = torch.cat([st, x_t.unsqueeze(-1)], dim=-1)
    y = (window.float() * conv_weight.float().unsqueeze(0)).sum(dim=-1)
    y = y + conv_bias.float()
    y = y * torch.sigmoid(y)
    conv_out = y.to(torch.bfloat16)
    conv_state.index_copy_(0, idx, window[..., 1:])

    q = conv_out[:, : h_q * d].view(B, h_q, d).float()
    k = conv_out[:, h_q * d : 2 * h_q * d].view(B, h_q, d).float()
    v = conv_out[:, 2 * h_q * d :].view(B, hv, d).float()

    if use_qk_l2norm:
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    group = hv // h_q
    q = q.repeat_interleave(group, dim=1)
    k = k.repeat_interleave(group, dim=1)

    g = torch.exp(
        -torch.exp(A_log.float()) * F.softplus(a_gate.float() + dt_bias.float())
    )
    beta = torch.sigmoid(b_gate.float())

    state = ssm_state[idx]
    state = state * g[:, :, None, None]
    old_v = torch.einsum("bhk,bhvk->bhv", k, state)
    delta = beta[:, :, None] * (v - old_v)
    state = state + delta[..., None] * k[:, :, None, :]
    attn_out = scale * torch.einsum("bhk,bhvk->bhv", q, state)

    ssm_state[idx] = state
    result = attn_out.unsqueeze(1).to(torch.bfloat16)
    if out is not None:
        out.copy_(result)
        result = out
    return result, conv_state, ssm_state


def _gdn_fused_decode_init(
    *,
    batch_size: int,
    num_pages: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdn_fused_decode_step``.

    The layer geometry is baked in rather than exposed as kwargs: it is the
    registry's dispatch surface (``hidden=5120``, ``qkv_dim=10240``, 16 qk
    heads / 48 v heads, ``d=128``, captured from
    ``nvidia/Qwen3.6-27B-NVFP4``), and the fused op only accelerates
    geometries a registry row lists — a scaled-down variant would trace an
    op that never dispatches.

    Distributions follow ``tests/gdn/test_fused_decode.py::_make_inputs``:
    small-scale ``randn`` activations and weights (the b/a GEMV sums over
    5120 terms), a negative ``A_log`` so the decay gate stays in range, and
    distinct pool slots walked downwards from ``num_pages - 1``.
    """
    torch.manual_seed(seed)
    hidden_size = 5120
    n_ba = 96
    qkv_dim = 10240
    num_v_heads = 48
    head_dim = 128
    conv_width = 4
    conv_state_len = conv_width - 1
    if num_pages < batch_size:
        raise ValueError("num_pages must be at least batch_size")

    hidden_states = (
        torch.randn(batch_size, hidden_size, dtype=torch.float32, device=device) * 0.05
    ).to(torch.bfloat16)
    w_ba = (
        torch.randn(hidden_size, n_ba, dtype=torch.float32, device=device) * 0.05
    ).to(torch.bfloat16)
    mixed_qkv = (
        torch.randn(batch_size, qkv_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)
    conv_weight = (
        torch.randn(qkv_dim, conv_width, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)
    conv_bias = (torch.randn(qkv_dim, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    # The vLLM pool is physically (conv_state_len, qkv_dim) per page; the op
    # consumes its transposed [P, qkv_dim, conv_state_len] view ("SD").
    conv_state = (
        (
            torch.randn(
                num_pages, conv_state_len, qkv_dim, dtype=torch.float32, device=device
            )
            * 0.5
        )
        .to(torch.bfloat16)
        .transpose(-1, -2)
    )
    A_log = -torch.rand(num_v_heads, dtype=torch.float32, device=device) * 2.0 - 1.0
    dt_bias = (torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    ssm_state = (
        torch.randn(
            num_pages,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    state_indices = torch.arange(
        num_pages - 1, num_pages - 1 - batch_size, -1, dtype=torch.int32, device=device
    )
    out = torch.empty(
        batch_size, 1, num_v_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    return {
        "hidden_states": hidden_states,
        "w_ba": w_ba,
        "mixed_qkv": mixed_qkv,
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "conv_state": conv_state,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": head_dim**-0.5,
        "ssm_state": ssm_state,
        "state_indices": state_indices,
        "use_qk_l2norm": True,
        "out": out,
    }


gdn_fused_decode_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_fused_decode",
    description=(
        "Fused single-token GDN decode step over paged conv/ssm state pools. "
        "Folds the per-layer serving chain -- the b/a projection GEMV, the "
        "depthwise causal conv1d state update, the q/k/v head split and the "
        "gated delta-rule decode with qk-L2-norm -- into one op. Both pools "
        "are updated in place. The input contract is one architecture's layer "
        "geometry rather than a reusable tensor primitive, so the const axes "
        "below are the dispatch surface, not tuning knobs."
    ),
    axes={
        "batch_size": Var(
            description="Number of sequences decoded in this step (one token each)."
        ),
        "num_pages": Var(
            description="Slots in the paged conv/ssm state pools; state_indices selects one per sequence."
        ),
        "seq_len": Const(
            description="Tokens per sequence (always 1 for single-token decode).",
            abbrev="",
            value=1,
        ),
        "hidden_size": Const(description="Layer input width.", abbrev="h"),
        "n_ba": Const(
            description="Fused b/a projection width (2 * num_v_heads).", abbrev=""
        ),
        "qkv_dim": Const(
            description="Fused q/k/v channel count, (2 * num_qk_heads + num_v_heads) * head_dim.",
            abbrev="",
        ),
        "num_v_heads": Const(
            description="Number of value heads (GVA: more value heads than query heads).",
            abbrev="v",
        ),
        "head_dim": Const(
            description="Dimension of each head (K in query/key space, V in value space).",
            abbrev="d",
        ),
        "conv_width": Const(description="Causal conv1d kernel width.", abbrev=""),
        "conv_state_len": Const(
            description="Raw inputs kept per channel in the conv pool, conv_width - 1.",
            abbrev="",
        ),
    },
    inputs={
        "hidden_states": Tensor(
            ["batch_size", "hidden_size"],
            description="Layer input for the decoded token of each sequence.",
        ),
        "w_ba": Tensor(
            ["hidden_size", "n_ba"],
            description="Fused b/a projection weight; columns [:num_v_heads] feed the beta gate, [num_v_heads:] the decay gate.",
        ),
        "mixed_qkv": Tensor(
            ["batch_size", "qkv_dim"],
            description="Raw (pre-conv) fused q/k/v channels. Rows may be strided (a view into a wider fused projection).",
        ),
        "conv_weight": Tensor(
            ["qkv_dim", "conv_width"], description="Depthwise causal conv1d weight."
        ),
        "conv_bias": Tensor(["qkv_dim"], description="Depthwise conv1d bias."),
        "conv_state": Tensor(
            ["num_pages", "qkv_dim", "conv_state_len"],
            description="Paged conv-state pool as its logical [P, qkv_dim, conv_state_len] view; updated in place.",
        ),
        "A_log": Tensor(
            ["num_v_heads"],
            description="Log decay parameter. g = exp(-exp(A_log) * softplus(a + dt_bias)).",
        ),
        "dt_bias": Tensor(
            ["num_v_heads"], description="Decay bias, added to 'a' before softplus."
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Query scale. Default is 1/sqrt(head_dim).",
        ),
        "ssm_state": Tensor(
            ["num_pages", "num_v_heads", "head_dim", "head_dim"],
            description="Paged recurrent-state pool in V-major [P, HV, V, K] layout; updated in place.",
        ),
        "state_indices": Tensor(
            ["batch_size"], description="Pool slot index of each sequence."
        ),
        "use_qk_l2norm": Scalar(
            "int32",
            optional=True,
            description="Apply L2 normalization to q and k. Default true.",
        ),
        "out": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_dim"],
            dtype="bfloat16",
            optional=True,
            description="Pre-allocated attention output, written in place when provided.",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_dim"],
            dtype="bfloat16",
            param="out",
            description="Attention output for the decoded token.",
        ),
        "conv_state_out": Tensor(
            ["num_pages", "qkv_dim", "conv_state_len"],
            dtype="bfloat16",
            param="conv_state",
            description="The conv-state pool, mutated in place and returned.",
        ),
        "ssm_state_out": Tensor(
            ["num_pages", "num_v_heads", "head_dim", "head_dim"],
            dtype="float32",
            param="ssm_state",
            description="The recurrent-state pool, mutated in place and returned.",
        ),
    },
    constraints=[
        "num_pages >= batch_size",
        "n_ba == 2 * num_v_heads",
        "conv_state_len == conv_width - 1",
        "qkv_dim > num_v_heads * head_dim",
        "(qkv_dim - num_v_heads * head_dim) % (2 * head_dim) == 0",
    ],
    tags=["stage:decode", "status:verified"],
    reference=_gdn_fused_decode_reference,
    init=_gdn_fused_decode_init,
)
