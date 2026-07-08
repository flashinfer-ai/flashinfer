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

"""TraceTemplates for KDA (Kimi Delta Attention) operations."""

import math

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── KDA Prefill ──────────────────────────────────────────────────────────────


@torch.no_grad()
def _kda_prefill_reference(q, k, v, g, b, cu_seqlens, scale):
    """
    KDA prefill reference implementation (k-last layout).

    State layout: [N, H, V, K] (k-last, K dimension at the end)

    KDA recurrence (g channel-wise on the key axis, LOG space; b a per-token
    scalar broadcast over both key and value axes):

    S = S * exp(g_t)[:, None]           (per-k-channel decay)
    v_new = b_t * (v_t - k_t^T @ S)
    S = S + k_t (x) v_new
    output_t = scale * q_t @ S
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    q_exp = q.repeat_interleave(num_sab_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_sab_heads // num_k_heads, dim=1)

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
        if seq_end <= seq_start:
            continue
        state_HKV = torch.zeros(
            (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
        )
        for t in range(seq_start, seq_end):
            state_HKV = state_HKV * g[t].float().exp().unsqueeze(-1)
            erase = torch.einsum("hk,hkv->hv", k_exp[t].float(), state_HKV)
            v_new = b[t].float().unsqueeze(-1) * (v[t].float() - erase)
            state_HKV = state_HKV + torch.einsum("hk,hv->hkv", k_exp[t].float(), v_new)
            output[t] = (
                scale * torch.einsum("hk,hkv->hv", q_exp[t].float(), state_HKV)
            ).to(torch.bfloat16)
        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state


def _kda_prefill_init(
    *,
    total_seq_len: int,
    num_seqs: int = 4,
    len_cu_seqlens: int = 0,  # derived
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 4,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.kda_prefill.chunk_kda``.

    Same Q/K/V distributions as the GDN prefill init (``multidist_randu`` +
    L2-normalized k); ``g`` is a channel-wise log-space decay in [-1, 0)
    float32, ``beta`` is a per-token scalar in [0, 1] (post-sigmoid space).
    """
    del len_cu_seqlens
    torch.manual_seed(seed)

    def _multidist_randu(num_dists: int, dim: int) -> torch.Tensor:
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
    num_sab_heads = max(num_q_heads, num_v_heads)
    g = -torch.rand(
        total_seq_len, num_sab_heads, head_size, dtype=torch.float32, device=device
    )
    beta = (
        torch.rand(total_seq_len, num_sab_heads, device=device)
        .sigmoid()
        .to(torch.bfloat16)
    )
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu_seqlens": cu_seqlens}


kda_prefill_trace = TraceTemplate(
    op_type="kda",
    name_prefix="kda_prefill",
    description=(
        "KDA (Kimi Delta Attention) prefill with channel-wise log-space forget "
        "gate, per-token scalar beta and k-last state layout. The state is in "
        "k-last layout [N, H, V, K]."
    ),
    axes={
        "total_seq_len": Var(
            description="Total number of tokens across all sequences in the batch."
        ),
        "num_seqs": Var(description="Number of sequences in the batch."),
        "num_q_heads": Const(
            description="Number of query heads.",
            abbrev="qk",
        ),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(
            description="Number of value heads.",
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
        "g": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            dtype="float32",
            description=("Channel-wise forget gate on the key axis in LOG space."),
        ),
        "b": Tensor(
            ["total_seq_len", "num_v_heads"],
            param="beta",
            description=(
                "Per-token scalar erase/write gate (post-sigmoid space, "
                "typical range [0, 1])."
            ),
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
            description="Attention output.",
        ),
        "new_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Updated recurrent state in k-last layout [N, H, V, K].",
        ),
    },
    constraints=[
        "num_k_heads == num_q_heads",
        "len_cu_seqlens == num_seqs + 1",
        "total_seq_len == cu_seqlens[-1].item()",
    ],
    tags=["stage:prefill", "status:experimental"],
    reference=_kda_prefill_reference,
    init=_kda_prefill_init,
)
