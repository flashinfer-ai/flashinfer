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
from copy import deepcopy
from typing import cast

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


_gdn_prefill_default_trace = TraceTemplate(
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


@torch.no_grad()
def _gdn_prefill_gates_reference(
    q,
    k,
    v,
    state,
    A_log,
    a,
    dt_bias,
    b,
    cu_seqlens,
    scale,
    gate_domain="linear",
    use_gate_in_kernel=False,
    beta_is_logit=False,
    use_qk_l2norm_in_kernel=False,
    output_final_state=False,
    state_indices=None,
    output_state=None,
):
    """Independent token recurrence for transformed GDN gate inputs."""
    if beta_is_logit and b is None:
        raise ValueError("beta_is_logit requires beta")
    heads, dim = max(q.shape[1], v.shape[1]), q.shape[-1]
    scale = scale or dim**-0.5
    if use_qk_l2norm_in_kernel:
        qf, kf = q.float(), k.float()
        q = (qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k = (kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    qf = q.float().repeat_interleave(heads // q.shape[1], dim=1)
    kf = k.float().repeat_interleave(heads // k.shape[1], dim=1)
    vf = v.float().repeat_interleave(heads // v.shape[1], dim=1)
    if use_gate_in_kernel:
        alpha = torch.exp(
            -torch.exp(A_log.float())
            * torch.nn.functional.softplus(a.float() + dt_bias.float())
        )
    elif a is None:
        alpha = torch.ones(q.shape[0], heads, device=q.device)
    elif gate_domain == "log":
        alpha = a.float().exp()
    else:
        alpha = a.float()
    beta = torch.ones_like(alpha) if b is None else b.float()
    if beta_is_logit:
        beta = beta.sigmoid().to(b.dtype).float()
    output = torch.empty(q.shape[0], heads, v.shape[-1], dtype=q.dtype, device=q.device)
    if output_state is not None:
        final = output_state.clone()
    elif state_indices is not None:
        final = state.clone()
    else:
        final = torch.empty(
            cu_seqlens.numel() - 1,
            heads,
            v.shape[-1],
            dim,
            dtype=state.dtype if state is not None else torch.float32,
            device=q.device,
        )
    for seq in range(cu_seqlens.numel() - 1):
        slot = seq if state_indices is None else int(state_indices[seq])
        hidden = (
            state[slot].float().clone()
            if state is not None
            else torch.zeros_like(final[slot], dtype=torch.float32)
        )
        for token in range(int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])):
            hidden *= alpha[token, :, None, None]
            delta = beta[token, :, None] * (
                vf[token] - torch.einsum("hvk,hk->hv", hidden, kf[token])
            )
            hidden += delta[:, :, None] * kf[token, :, None, :]
            output[token] = (scale * torch.einsum("hvk,hk->hv", hidden, qf[token])).to(
                q.dtype
            )
        final[slot] = hidden
    return output, final if output_final_state else None


def _gdn_prefill_gates_init(
    *,
    total_seq_len: int,
    num_seqs: int = 4,
    state_pool_rows: int = 0,
    len_cu_seqlens: int = 0,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    inputs = _gdn_prefill_init(
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        len_cu_seqlens=len_cu_seqlens,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        device=device,
        seed=seed,
    )
    heads = max(num_q_heads, num_v_heads)
    rows = state_pool_rows or num_seqs
    if rows < num_seqs:
        raise ValueError("state_pool_rows must be at least num_seqs")
    inputs.update(
        g=torch.randn(total_seq_len, heads, device=device),
        beta=torch.randn(total_seq_len, heads, dtype=torch.bfloat16, device=device),
        A_log=torch.zeros(heads, device=device),
        dt_bias=torch.full((heads,), -2.0, device=device),
        use_gate_in_kernel=True,
        beta_is_logit=True,
        output_final_state=True,
        initial_state=torch.zeros(rows, heads, head_size, head_size, device=device),
        output_state=torch.zeros(rows, heads, head_size, head_size, device=device),
        state_indices=(
            torch.arange(num_seqs, device=device, dtype=torch.int32)
            if rows != num_seqs
            else None
        ),
    )
    return inputs


_gdn_prefill_gates_init._trace_init_dependencies = (_gdn_prefill_init,)  # type: ignore[attr-defined]
_gdn_prefill_gates_trace = deepcopy(_gdn_prefill_default_trace)
_gdn_prefill_gates_trace.name_prefix = "gdn_prefill_gates"
_gdn_prefill_gates_trace.description = (
    "GDN prefill with explicit raw/log gate semantics and optional fused sigmoid."
)
_gdn_prefill_gates_trace.reference = _gdn_prefill_gates_reference
_gdn_prefill_gates_trace.init = _gdn_prefill_gates_init
_gdn_prefill_gates_trace.axes["state_pool_rows"] = Var(
    description="Rows in the recurrent state pool or compact state."
)
_gdn_prefill_gates_trace.axes["num_o_heads"] = Const(
    description="max(num_q_heads, num_v_heads).", abbrev=""
)
cast(Const, _gdn_prefill_gates_trace.axes["num_q_heads"]).abbrev = "q"
cast(Const, _gdn_prefill_gates_trace.axes["num_k_heads"]).abbrev = "k"
for _name in ("a", "b"):
    cast(Tensor, _gdn_prefill_gates_trace.inputs[_name]).dim_names = [
        "total_seq_len",
        "num_o_heads",
    ]
for _name in ("A_log", "dt_bias"):
    cast(Tensor, _gdn_prefill_gates_trace.inputs[_name]).dim_names = ["num_o_heads"]
cast(Tensor, _gdn_prefill_gates_trace.inputs["state"]).dim_names = [
    "state_pool_rows",
    "num_o_heads",
    "head_size",
    "head_size",
]
cast(Tensor, _gdn_prefill_gates_trace.outputs["output"]).dim_names = [
    "total_seq_len",
    "num_o_heads",
    "head_size",
]
cast(Tensor, _gdn_prefill_gates_trace.outputs["output"]).dtype_from = "q"
_gdn_prefill_gates_trace.outputs["new_state"] = Tensor(
    ["state_pool_rows", "num_o_heads", "head_size", "head_size"],
    optional=True,
    dtype_from="output_state",
    dtype="float32",
)
_gdn_prefill_gates_trace.constraints = [
    "num_o_heads == max(num_q_heads, num_v_heads)",
    "num_o_heads % num_q_heads == 0",
    "num_o_heads % num_v_heads == 0",
    "num_k_heads in (num_q_heads, num_v_heads)",
    "len_cu_seqlens == num_seqs + 1",
    "total_seq_len == cu_seqlens[-1].item()",
]
_gdn_prefill_gates_trace.inputs[
    "a"
].description = (
    "Gate inputs in gate_domain, or raw logits when use_gate_in_kernel is set."
)
_gdn_prefill_gates_trace.inputs[
    "b"
].description = "Precomputed beta, or sigmoid logits when beta_is_logit is set."
_gdn_prefill_gates_trace.inputs[
    "A_log"
].description = "Per-head log decay parameter for raw gates."
_gdn_prefill_gates_trace.inputs["dt_bias"].description = "Per-head bias for raw gates."
_gdn_prefill_gates_trace.inputs.update(
    gate_domain=Scalar("str", optional=True),
    use_gate_in_kernel=Scalar("int32", optional=True),
    beta_is_logit=Scalar("int32", optional=True),
    use_qk_l2norm_in_kernel=Scalar("int32", optional=True),
    output_final_state=Scalar("int32", optional=True),
    state_indices=Tensor(["num_seqs"], optional=True),
    output_state=Tensor(
        ["state_pool_rows", "num_o_heads", "head_size", "head_size"], optional=True
    ),
)


def gdn_prefill_trace(**kwargs):
    # The slot arguments (output_state_indices, checkpoint_state_indices,
    # prefix_lens) have no template here: such a call is not traced, and Apply
    # keeps the original call. The paged APIs (gdn_prefill_paged_trace) carry
    # their own definitions.
    if any(
        kwargs.get(name) is not None
        for name in ("output_state_indices", "checkpoint_state_indices", "prefix_lens")
    ):
        return None
    if (
        kwargs.get("gate_domain", "linear") != "linear"
        or kwargs.get("use_gate_in_kernel", False)
        or kwargs.get("beta_is_logit", False)
        or kwargs.get("state_indices") is not None
    ):
        return _gdn_prefill_gates_trace
    return _gdn_prefill_default_trace


gdn_prefill_trace.templates = [_gdn_prefill_default_trace, _gdn_prefill_gates_trace]  # type: ignore[attr-defined]

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
    scale=None,
    intermediate_states_buffer=None,
    *,
    cache_replayssm=False,
    disable_state_update=True,
    use_qk_l2norm=True,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
):
    """Direct GDN recurrence, including frozen-state raw-window verification."""
    B, T, H, K = q.shape
    HV = v.shape[2]
    scale = K**-0.5 if scale is None else scale
    raw_q, raw_k, raw_v, raw_a, raw_b = (
        x.to(torch.bfloat16).double() for x in (q, k, v, a, b)
    )
    query, key = raw_q, raw_k
    if use_qk_l2norm:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query.repeat_interleave(HV // H, dim=2) * scale
    key = key.repeat_interleave(HV // H, dim=2)
    log_g = -A_log.double().exp() * F.softplus(raw_a + dt_bias.double())
    beta = raw_b.sigmoid()
    output = torch.zeros_like(v)
    final_state = initial_state.clone()
    for row, slot in enumerate(initial_state_indices.tolist()):
        if slot < 0:
            continue
        if cache_replayssm:
            replayssm_rawv[slot] = raw_v[row].transpose(0, 1).to(torch.bfloat16)
            replayssm_rawk[slot] = raw_k[row].transpose(0, 1).to(torch.bfloat16)
            replayssm_g[slot] = log_g[row].transpose(0, 1).float()
            replayssm_beta[slot] = beta[row].transpose(0, 1).float()
        state = initial_state[slot].double()
        for step in range(T):
            kt = key[row, step]
            state *= log_g[row, step, :, None, None].exp()
            prediction = (state * kt[:, None, :]).sum(-1)
            delta = (raw_v[row, step] - prediction) * beta[row, step, :, None]
            state += delta[..., None] * kt[:, None, :]
            output[row, step] = (
                (state * query[row, step, :, None, :]).sum(-1).to(output.dtype)
            )
            if intermediate_states_buffer is not None:
                intermediate_states_buffer[row, step] = state.float()
        if not disable_state_update:
            final_state[slot] = state.float()
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
        "cache_replayssm": Scalar("bool", optional=True),
        "disable_state_update": Scalar("bool", optional=True),
        "use_qk_l2norm": Scalar("bool", optional=True),
        "replayssm_rawv": Tensor(
            ["pool_size", "num_v_heads", "seq_len", "head_size"],
            dtype="bfloat16",
            optional=True,
        ),
        "replayssm_rawk": Tensor(
            ["pool_size", "num_k_heads", "seq_len", "head_size"],
            dtype="bfloat16",
            optional=True,
        ),
        "replayssm_g": Tensor(
            ["pool_size", "num_v_heads", "seq_len"], dtype="float32", optional=True
        ),
        "replayssm_beta": Tensor(
            ["pool_size", "num_v_heads", "seq_len"], dtype="float32", optional=True
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
        "replayssm_rawv": Tensor(
            ["pool_size", "num_v_heads", "seq_len", "head_size"],
            dtype="bfloat16",
            param="replayssm_rawv",
            optional=True,
        ),
        "replayssm_rawk": Tensor(
            ["pool_size", "num_k_heads", "seq_len", "head_size"],
            dtype="bfloat16",
            param="replayssm_rawk",
            optional=True,
        ),
        "replayssm_g": Tensor(
            ["pool_size", "num_v_heads", "seq_len"],
            dtype="float32",
            param="replayssm_g",
            optional=True,
        ),
        "replayssm_beta": Tensor(
            ["pool_size", "num_v_heads", "seq_len"],
            dtype="float32",
            param="replayssm_beta",
            optional=True,
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


@torch.no_grad()
def _gdn_replayssm_commit_reference(
    checkpoint_state,
    rawv_cache,
    rawk_cache,
    g_cache,
    beta_cache,
    state_indices,
    accept_lens,
    *,
    track_state_indices=None,
    track_steps=None,
    use_qk_l2norm=True,
    null_block_id=-1,
    backend="auto",
):
    """Direct recurrence over accepted tokens; mutates the checkpoint pool."""
    hv, h = rawv_cache.shape[2], rawk_cache.shape[2]
    for row, slot in enumerate(state_indices.tolist()):
        count = int(accept_lens[row])
        if slot <= null_block_id or count <= 0:
            continue
        state = checkpoint_state[:, slot].double()
        for step in range(count):
            key = (
                rawk_cache[:, slot, :, step].double().repeat_interleave(hv // h, dim=1)
            )
            if use_qk_l2norm:
                key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            state *= g_cache[:, slot, :, step, None, None].double().exp()
            delta = (
                rawv_cache[:, slot, :, step].double()
                - (state * key[..., None, :]).sum(-1)
            ) * beta_cache[:, slot, :, step, None].double()
            state += delta[..., None] * key[..., None, :]
            if track_state_indices is not None and track_steps is not None:
                track = int(track_state_indices[row])
                if track > null_block_id and int(track_steps[row]) == step:
                    checkpoint_state[:, track] = state.float()
        checkpoint_state[:, slot] = state.float()
    return checkpoint_state


def _gdn_replayssm_commit_init(
    *,
    batch_size=4,
    num_layers=2,
    pool_size=8,
    num_k_heads=2,
    num_v_heads=8,
    seq_len=8,
    head_size=128,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)
    return dict(
        checkpoint_state=torch.randn(
            num_layers, pool_size, num_v_heads, head_size, head_size, device=device
        )
        * 0.1,
        rawv_cache=torch.randn(
            num_layers,
            pool_size,
            num_v_heads,
            seq_len,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1,
        rawk_cache=torch.randn(
            num_layers,
            pool_size,
            num_k_heads,
            seq_len,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1,
        g_cache=-torch.rand(num_layers, pool_size, num_v_heads, seq_len, device=device),
        beta_cache=torch.rand(
            num_layers, pool_size, num_v_heads, seq_len, device=device
        ),
        state_indices=torch.arange(batch_size, device=device, dtype=torch.int32),
        accept_lens=torch.full(
            (batch_size,), seq_len, device=device, dtype=torch.int32
        ),
    )


gdn_replayssm_commit_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_replayssm_commit",
    description="Commit the accepted raw-input prefix into all FP32 GDN checkpoints.",
    axes={
        "num_layers": Var(),
        "pool_size": Var(),
        "batch_size": Var(),
        "seq_len": Var(),
        "num_k_heads": Const(abbrev="k"),
        "num_v_heads": Const(abbrev="v"),
        "head_size": Const(abbrev="d"),
    },
    inputs={
        "checkpoint_state": Tensor(
            ["num_layers", "pool_size", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
        ),
        "rawv_cache": Tensor(
            ["num_layers", "pool_size", "num_v_heads", "seq_len", "head_size"],
            dtype="bfloat16",
        ),
        "rawk_cache": Tensor(
            ["num_layers", "pool_size", "num_k_heads", "seq_len", "head_size"],
            dtype="bfloat16",
        ),
        "g_cache": Tensor(
            ["num_layers", "pool_size", "num_v_heads", "seq_len"], dtype="float32"
        ),
        "beta_cache": Tensor(
            ["num_layers", "pool_size", "num_v_heads", "seq_len"], dtype="float32"
        ),
        "state_indices": Tensor(["batch_size"], dtype="int32"),
        "accept_lens": Tensor(["batch_size"], dtype="int32"),
        "track_state_indices": Tensor(["batch_size"], dtype="int32", optional=True),
        "track_steps": Tensor(["batch_size"], dtype="int32", optional=True),
        "use_qk_l2norm": Scalar("bool", optional=True),
        "null_block_id": Scalar("int32", optional=True),
        "backend": Scalar("str", optional=True),
    },
    outputs={
        "checkpoint_state": Tensor(
            ["num_layers", "pool_size", "num_v_heads", "head_size", "head_size"],
            param="checkpoint_state",
            dtype="float32",
        ),
    },
    constraints=[
        "head_size == 128",
        "num_v_heads % num_k_heads == 0",
        "batch_size <= pool_size",
    ],
    tags=["stage:commit"],
    reference=_gdn_replayssm_commit_reference,
    init=_gdn_replayssm_commit_init,
)

# ── GDN on a paged recurrent-state pool ───────────────────────────────────────


@torch.no_grad()
def _gdn_prefill_paged_reference(
    q,
    k,
    v,
    g=None,
    beta=None,
    scale=None,
    *,
    cu_seqlens,
    state_pool,
    page_indices,
    page_indptr,
    page_size,
    last_page_len,
    use_qk_l2norm_in_kernel=False,
    gate_domain="linear",
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    beta_is_logit=False,
):
    """Token recurrence of ``chunk_gated_delta_rule_paged`` on a paged pool.

    Sequence ``b`` owns pages ``page_indices[page_indptr[b]:page_indptr[b+1]]``
    and holds ``total = (num_pages_b - 1) * page_size + last_page_len[b]``
    tokens after the call. Its new tokens are the last ``seq_len`` of them,
    so ``prefix = total - seq_len``. Page ``j`` holds the state after token
    ``(j + 1) * page_size`` (the last page: after the last token).

    - The initial state is the page holding token ``prefix - 1``, or zero
      when ``prefix == 0``.
    - After each new token whose absolute position ``prefix + n``
      (``1 <= n <= seq_len``) is a multiple of ``page_size``, the state is
      written to the page that ends there.
    - The final state is written to the last page, unless ``seq_len == 0``
      (the pool is unchanged) or ``total % page_size == 0`` (the boundary
      write already stored it there).

    Pages that end at or before the prefix are only read. The gates and
    the optional Q/K L2 norm follow ``chunk_gated_delta_rule``: ``g`` is the
    linear decay (``None``: no decay), or its natural log with
    ``gate_domain="log"``, or the raw input of
    ``exp(-exp(A_log) * softplus(g + dt_bias))`` with ``use_gate_in_kernel``;
    ``beta`` is the update gate (``None``: one), or its logit with
    ``beta_is_logit`` (sigmoid rounded to beta's dtype). The recurrent state
    stays in FP32 between tokens; ``state_pool`` is updated in place and
    returned with the output.
    """
    heads, dim = max(q.shape[1], v.shape[1]), q.shape[-1]
    if scale is None or scale == 0.0:
        scale = dim**-0.5
    if use_qk_l2norm_in_kernel:
        # FP32 L2 norm with eps 1e-6, rounded back to the input dtype.
        qf, kf = q.float(), k.float()
        q = (qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k = (kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    qf = q.float().repeat_interleave(heads // q.shape[1], dim=1)
    kf = k.float().repeat_interleave(heads // k.shape[1], dim=1)
    vf = v.float().repeat_interleave(heads // v.shape[1], dim=1)
    if use_gate_in_kernel:
        alpha = torch.exp(
            -torch.exp(A_log.float()) * F.softplus(g.float() + dt_bias.float())
        )
    elif g is None:
        alpha = torch.ones(q.shape[0], heads, device=q.device)
    elif gate_domain == "log":
        alpha = g.float().exp()
    else:
        alpha = g.float()
    if beta is None:
        beta_f = torch.ones(q.shape[0], heads, device=q.device)
    elif beta_is_logit:
        beta_f = beta.float().sigmoid().to(beta.dtype).float()
    else:
        beta_f = beta.float()

    output = torch.zeros(q.shape[0], heads, v.shape[-1], dtype=q.dtype, device=q.device)
    for seq in range(cu_seqlens.numel() - 1):
        start, end = int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])
        pages = page_indices[int(page_indptr[seq]) : int(page_indptr[seq + 1])]
        pages = [int(page) for page in pages]
        total = (len(pages) - 1) * page_size + int(last_page_len[seq])
        prefix = total - (end - start)
        if prefix > 0:
            hidden = state_pool[pages[(prefix - 1) // page_size]].float().clone()
        else:
            hidden = torch.zeros(heads, v.shape[-1], dim, device=q.device)
        for token in range(start, end):
            hidden *= alpha[token, :, None, None]
            delta = beta_f[token, :, None] * (
                vf[token] - torch.einsum("hvk,hk->hv", hidden, kf[token])
            )
            hidden += delta[:, :, None] * kf[token, :, None, :]
            output[token] = (scale * torch.einsum("hvk,hk->hv", hidden, qf[token])).to(
                q.dtype
            )
            position = prefix + token - start + 1
            if position % page_size == 0:
                state_pool[pages[position // page_size - 1]] = hidden
        if end > start and total % page_size != 0:
            state_pool[pages[-1]] = hidden
    return output, state_pool


def _gdn_prefill_paged_init(
    *,
    total_seq_len: int,
    num_seqs: int = 4,
    len_cu_seqlens: int = 0,  # derived: num_seqs + 1
    num_pages: int = 0,  # minimum pool rows; grown to the pages used + 3
    num_page_indices: int = 0,  # derived from the page table
    num_q_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    page_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
):
    """Build a paged batch for ``flashinfer.gdn_paged.chunk_gated_delta_rule_paged``.

    The ``total_seq_len`` new tokens are split raggedly over ``num_seqs``
    sequences (sequence 0 gets at least one; others may get none). The
    prefixes cycle through no prefix, ``page_size + 37`` tokens (inside a
    page, not 64-aligned), two full pages and 5 tokens, plus ``seq // 4``
    pages. Each sequence owns ``ceil((prefix + seq_len) / page_size)`` page
    ids, drawn without replacement from a shuffled pool with spare rows.
    Page tables and lengths are int32. Keys use the query heads (GVA) or
    the value heads (GQA): ``num_k_heads = min(num_q_heads, num_v_heads)``.
    Q/K/V follow ``_gdn_prefill_init``; the gates are linear, ``g`` in
    (0.9, 1) and ``beta`` in (0, 1); pool rows are ``0.1 * randn`` FP32.
    """
    del len_cu_seqlens, num_page_indices
    if total_seq_len < 1:
        raise ValueError("total_seq_len must be at least 1")
    inputs = _gdn_prefill_init(
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_k_heads=min(num_q_heads, num_v_heads),
        num_v_heads=num_v_heads,
        head_size=head_size,
        device=device,
        seed=seed,
    )
    heads = max(num_q_heads, num_v_heads)
    weights = [1 + (3 * seq + 1) % 5 for seq in range(num_seqs)]
    weight_sum = sum(weights)
    seq_lens = [total_seq_len * w // weight_sum for w in weights]
    seq_lens[0] += total_seq_len - sum(seq_lens)
    prefix_cycle = [0, page_size + 37, 2 * page_size, 5]
    prefixes = [
        prefix_cycle[seq % 4] + (seq // 4) * page_size for seq in range(num_seqs)
    ]
    totals = [p + n for p, n in zip(prefixes, seq_lens, strict=True)]
    pages_per_seq = [-(-total // page_size) for total in totals]
    used = sum(pages_per_seq)
    rows = max(num_pages, used + 3)
    gen = torch.Generator().manual_seed(seed)
    page_ids = torch.randperm(rows, generator=gen)[:used]

    def _int32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    cu_seqlens = [0]
    page_indptr = [0]
    for n, pages in zip(seq_lens, pages_per_seq, strict=True):
        cu_seqlens.append(cu_seqlens[-1] + n)
        page_indptr.append(page_indptr[-1] + pages)
    inputs.update(
        g=0.9 + 0.1 * torch.rand(total_seq_len, heads, device=device),
        beta=torch.rand(total_seq_len, heads, device=device),
        cu_seqlens=_int32(cu_seqlens),
        state_pool=0.1 * torch.randn(rows, heads, head_size, head_size, device=device),
        page_indices=page_ids.to(torch.int32).to(device),
        page_indptr=_int32(page_indptr),
        page_size=page_size,
        last_page_len=_int32(
            [
                total - (pages - 1) * page_size
                for total, pages in zip(totals, pages_per_seq, strict=True)
            ]
        ),
    )
    return inputs


_gdn_prefill_paged_init._trace_init_dependencies = (_gdn_prefill_init,)  # type: ignore[attr-defined]

gdn_prefill_paged_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_prefill_paged",
    description=(
        "Chunked GDN prefill on a paged recurrent-state pool [num_pages, H, V, K]. "
        "Each sequence owns a CSR list of page ids; page j holds the state after "
        "token (j + 1) * page_size. The call reads the page holding the last "
        "prefix token (zero state without a prefix), writes the state at every "
        "page boundary its new tokens cross and the final state to the last "
        "page. Pages that end at or before the prefix are only read."
    ),
    axes={
        "total_seq_len": Var(description="Total new tokens across all sequences."),
        "num_seqs": Var(description="Number of sequences in the batch."),
        "len_cu_seqlens": Var(
            description="num_seqs + 1 (length of cu_seqlens and page_indptr)."
        ),
        "num_q_heads": Const(description="Number of query heads.", abbrev="q"),
        "num_k_heads": Const(description="Number of key heads.", abbrev="k"),
        "num_v_heads": Const(description="Number of value heads.", abbrev="v"),
        "num_o_heads": Const(
            description="max(num_q_heads, num_v_heads): output and state heads.",
            abbrev="",
        ),
        "head_size": Const(
            description="Head dimension (K in query/key space, V in value space).",
            abbrev="d",
        ),
        "page_size": Const(
            description="Tokens per page, a positive multiple of 64.", abbrev="ps"
        ),
        "num_pages": Var(description="Rows of the recurrent-state pool."),
        "num_page_indices": Var(description="Length of page_indices."),
    },
    inputs={
        "q": Tensor(["total_seq_len", "num_q_heads", "head_size"]),
        "k": Tensor(["total_seq_len", "num_k_heads", "head_size"]),
        "v": Tensor(["total_seq_len", "num_v_heads", "head_size"]),
        "g": Tensor(
            ["total_seq_len", "num_o_heads"],
            optional=True,
            description=(
                "Decay gate: linear alpha (default), log alpha with "
                "gate_domain='log', or raw input with use_gate_in_kernel. "
                "None means no decay."
            ),
        ),
        "beta": Tensor(
            ["total_seq_len", "num_o_heads"],
            optional=True,
            description="Update gate, or its logit with beta_is_logit. None means one.",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Query scale. Default is 1/sqrt(head_size).",
        ),
        "cu_seqlens": Tensor(
            ["len_cu_seqlens"],
            description="Offsets of each sequence's new tokens in q/k/v.",
        ),
        "state_pool": Tensor(
            ["num_pages", "num_o_heads", "head_size", "head_size"],
            description="Recurrent-state pool in k-last layout [num_pages, H, V, K].",
        ),
        "page_indices": Tensor(
            ["num_page_indices"],
            description="Page ids of all sequences, in any order (CSR values).",
        ),
        "page_indptr": Tensor(
            ["len_cu_seqlens"],
            description="CSR offsets: sequence b owns page_indices[page_indptr[b]:page_indptr[b + 1]].",
        ),
        "page_size": Scalar("int32", description="Tokens per page."),
        "last_page_len": Tensor(
            ["num_seqs"],
            description="Tokens in each sequence's last page after the call, in [1, page_size].",
        ),
        "use_qk_l2norm_in_kernel": Scalar(
            "bool", optional=True, description="L2-normalize q and k (eps 1e-6)."
        ),
        "gate_domain": Scalar(
            "str", optional=True, description="'linear' (default) or 'log'."
        ),
        "use_gate_in_kernel": Scalar(
            "bool",
            optional=True,
            description="g is raw: alpha = exp(-exp(A_log) * softplus(g + dt_bias)).",
        ),
        "A_log": Tensor(
            ["num_o_heads"],
            optional=True,
            description="Per-head log decay parameter for raw gates.",
        ),
        "dt_bias": Tensor(
            ["num_o_heads"], optional=True, description="Per-head bias for raw gates."
        ),
        "beta_is_logit": Scalar(
            "bool", optional=True, description="beta = sigmoid(beta)."
        ),
    },
    outputs={
        "output": Tensor(
            ["total_seq_len", "num_o_heads", "head_size"],
            param="output",
            dtype_from="q",
            description="Attention output of the new tokens.",
        ),
        "state_pool": Tensor(
            ["num_pages", "num_o_heads", "head_size", "head_size"],
            param="state_pool",
            dtype_from="state_pool",
            description="The state pool, updated in place.",
        ),
    },
    constraints=[
        "num_o_heads == max(num_q_heads, num_v_heads)",
        "num_o_heads % num_q_heads == 0",
        "num_o_heads % num_v_heads == 0",
        "num_k_heads in (num_q_heads, num_v_heads)",
        "len_cu_seqlens == num_seqs + 1",
        "total_seq_len == cu_seqlens[-1].item()",
        "num_page_indices == page_indptr[-1].item()",
        "page_size % 64 == 0",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_gdn_prefill_paged_reference,
    init=_gdn_prefill_paged_init,
)


@torch.no_grad()
def _gdn_decode_paged_reference(
    q,
    k,
    v,
    A_log,
    a,
    dt_bias,
    b,
    scale=None,
    *,
    use_qk_l2norm=True,
    state_pool,
    page_indices,
    page_indptr,
    last_page_len,
):
    """One-token GDN decode of ``gated_delta_rule_decode_paged`` on a paged pool.

    The decoded token of sequence ``b`` is the last of its tokens, at
    position ``last_page_len[b]`` of its last page. When it opens that page
    (``last_page_len[b] == 1``) the state is read from the previous page,
    which stays unchanged; otherwise from the last page. The updated state
    is written to the last page.

    Gates: ``g = exp(-exp(A_log) * softplus(a + dt_bias))``,
    ``beta = sigmoid(b)``. With ``use_qk_l2norm``, q and k are scaled by
    ``rsqrt(sum(x^2) + 1e-6)`` in FP32. ``state_pool`` is updated in place
    and returned with the output ``[B, 1, HV, V]``.
    """
    batch, _, num_q_heads, dim = q.shape
    heads = v.shape[2]
    if scale is None:
        scale = dim**-0.5
    qf, kf, vf = q[:, 0].float(), k[:, 0].float(), v[:, 0].float()
    if use_qk_l2norm:
        qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    qf = qf.repeat_interleave(heads // num_q_heads, dim=1)
    kf = kf.repeat_interleave(heads // k.shape[2], dim=1)
    alpha = torch.exp(
        -torch.exp(A_log.float()) * F.softplus(a[:, 0].float() + dt_bias.float())
    )
    beta = torch.sigmoid(b[:, 0].float())

    output = torch.zeros(batch, 1, heads, v.shape[-1], dtype=q.dtype, device=q.device)
    for seq in range(batch):
        last = int(page_indptr[seq + 1]) - 1
        read = last - 1 if int(last_page_len[seq]) == 1 else last
        hidden = state_pool[int(page_indices[read])].float().clone()
        hidden *= alpha[seq, :, None, None]
        delta = beta[seq, :, None] * (
            vf[seq] - torch.einsum("hvk,hk->hv", hidden, kf[seq])
        )
        hidden += delta[:, :, None] * kf[seq, :, None, :]
        output[seq, 0] = (scale * torch.einsum("hvk,hk->hv", hidden, qf[seq])).to(
            q.dtype
        )
        state_pool[int(page_indices[last])] = hidden
    return output, state_pool


def _gdn_decode_paged_init(
    *,
    batch_size: int,
    batch_size_plus_1: int = 0,  # derived
    num_pages: int = 0,  # minimum pool rows; grown to the pages used + 3
    num_page_indices: int = 0,  # derived from the page table
    num_q_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    page_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
):
    """Build a paged batch for ``flashinfer.gdn_paged.gated_delta_rule_decode_paged``.

    Sequences cycle through four (pages owned, last_page_len) cases, plus
    ``seq // 4`` pages: a token that opens the second page (reads page 0),
    the 17th token of a single page, the last token of a third full page,
    and a token that opens the third page. Page ids are drawn without
    replacement from a shuffled pool with spare rows; page tables and
    lengths are int32. ``num_k_heads == num_q_heads`` as the decode kernel
    requires. Q/K/V and the gate inputs follow ``_gdn_decode_init``; pool
    rows are ``randn`` FP32.
    """
    del batch_size_plus_1, num_page_indices
    inputs = _gdn_decode_init(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        device=device,
        seed=seed,
    )
    del inputs["state"]
    cases = [(2, 1), (1, 17), (3, page_size), (3, 1)]
    pages_per_seq = [cases[seq % 4][0] + seq // 4 for seq in range(batch_size)]
    used = sum(pages_per_seq)
    rows = max(num_pages, used + 3)
    gen = torch.Generator().manual_seed(seed)
    page_ids = torch.randperm(rows, generator=gen)[:used]
    page_indptr = [0]
    for pages in pages_per_seq:
        page_indptr.append(page_indptr[-1] + pages)
    inputs.update(
        state_pool=torch.randn(
            rows, num_v_heads, head_size, head_size, dtype=torch.float32, device=device
        ),
        page_indices=page_ids.to(torch.int32).to(device),
        page_indptr=torch.tensor(page_indptr, dtype=torch.int32, device=device),
        last_page_len=torch.tensor(
            [cases[seq % 4][1] for seq in range(batch_size)],
            dtype=torch.int32,
            device=device,
        ),
    )
    return inputs


_gdn_decode_paged_init._trace_init_dependencies = (_gdn_decode_init,)  # type: ignore[attr-defined]

gdn_decode_paged_trace = TraceTemplate(
    op_type="gdn",
    name_prefix="gdn_decode_paged",
    description=(
        "Single-token GDN decode on a paged recurrent-state pool [num_pages, HV, V, K] "
        "with GVA heads. The decoded token is the last token of its sequence: when "
        "it opens a new page (last_page_len == 1) the state is read from the "
        "previous page, else from the last page; the updated state is written "
        "to the last page."
    ),
    axes={
        "batch_size": Var(description="Number of sequences, one token each."),
        "batch_size_plus_1": Var(description="batch_size + 1 (length of page_indptr)."),
        "seq_len": Const(
            description="Tokens per sequence (always 1).", abbrev="", value=1
        ),
        "num_q_heads": Const(
            description="Number of query heads (same as key heads).", abbrev="qk"
        ),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(description="Number of value heads.", abbrev="v"),
        "head_size": Const(
            description="Head dimension (K in query/key space, V in value space).",
            abbrev="d",
        ),
        "num_pages": Var(description="Rows of the recurrent-state pool."),
        "num_page_indices": Var(description="Length of page_indices."),
    },
    inputs={
        "q": Tensor(["batch_size", "seq_len", "num_q_heads", "head_size"]),
        "k": Tensor(["batch_size", "seq_len", "num_k_heads", "head_size"]),
        "v": Tensor(["batch_size", "seq_len", "num_v_heads", "head_size"]),
        "A_log": Tensor(
            ["num_v_heads"],
            description="Log decay: g = exp(-exp(A_log) * softplus(a + dt_bias)).",
        ),
        "a": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Input-dependent decay.",
        ),
        "dt_bias": Tensor(["num_v_heads"], description="Decay bias."),
        "b": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Update gate input: beta = sigmoid(b).",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Query scale. Default is 1/sqrt(head_size).",
        ),
        "use_qk_l2norm": Scalar(
            "bool",
            optional=True,
            description="L2-normalize q and k (eps 1e-6). Default true.",
        ),
        "state_pool": Tensor(
            ["num_pages", "num_v_heads", "head_size", "head_size"],
            description="Recurrent-state pool in k-last layout [num_pages, HV, V, K].",
        ),
        "page_indices": Tensor(
            ["num_page_indices"],
            description="Page ids of all sequences, in any order (CSR values).",
        ),
        "page_indptr": Tensor(
            ["batch_size_plus_1"],
            description="CSR offsets: sequence b owns page_indices[page_indptr[b]:page_indptr[b + 1]].",
        ),
        "last_page_len": Tensor(
            ["batch_size"],
            description="Tokens in the last page including the decoded token, in [1, page_size].",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            param="output",
            dtype_from="q",
            description="Attention output of the decoded token.",
        ),
        "state_pool": Tensor(
            ["num_pages", "num_v_heads", "head_size", "head_size"],
            param="state_pool",
            dtype_from="state_pool",
            description="The state pool, updated in place.",
        ),
    },
    constraints=[
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "num_k_heads == num_q_heads",
        "batch_size_plus_1 == batch_size + 1",
        "num_page_indices == page_indptr[-1].item()",
    ],
    tags=["stage:decode", "status:verified"],
    reference=_gdn_decode_paged_reference,
    init=_gdn_decode_paged_init,
)
