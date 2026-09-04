# Copyright (c) 2026 by FlashInfer team.
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

"""TraceTemplate for Gated Delta Net 2 (GDN-2) prefill."""

import math

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _gdn2_prefill_reference(q, k, v, g, beta, w, initial_state, cu_seqlens, scale):
    """Gated Delta Net 2 prefill reference, state V-major as ``[N, H, V, K]``.

    All three gates are per channel: ``g`` and ``beta`` on the key dimension,
    ``w`` on the value dimension. Per token,

        S = diag(g) S
        v_new = w * v - S (beta * k)
        S += v_new (x) k
        o = scale * S q
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    q_exp = q.float().repeat_interleave(num_sab_heads // num_q_heads, dim=1)
    k_exp = k.float().repeat_interleave(num_sab_heads // num_k_heads, dim=1)
    v_exp = v.float().repeat_interleave(num_sab_heads // num_v_heads, dim=1)
    g_f32 = g.float()
    beta_f32 = beta.float()
    w_f32 = w.float()

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=q.dtype, device=device
    )
    final_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            if initial_state is not None:
                final_state[seq_idx] = initial_state[seq_idx].float()
            continue
        if initial_state is not None:
            state = initial_state[seq_idx].clone().float()  # [H, V, K]
        else:
            state = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )
        for t in range(start, end):
            key = k_exp[t]
            state = state * g_f32[t][:, None, :]
            erased = beta_f32[t] * key
            v_new = w_f32[t] * v_exp[t] - torch.einsum("hvk,hk->hv", state, erased)
            state = state + v_new[:, :, None] * key[:, None, :]
            projected = torch.einsum("hvk,hk->hv", state, q_exp[t])
            output[t] = (scale * projected).to(q.dtype)
        final_state[seq_idx] = state

    return output, final_state


def _gdn2_prefill_init(
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
    """Build inputs for ``flashinfer.gdn2_prefill.chunk_gated_delta_rule2``."""
    del len_cu_seqlens
    torch.manual_seed(seed)
    num_sab_heads = max(num_q_heads, num_v_heads)

    def qkv(num_heads, normalize):
        x = torch.randn(
            total_seq_len, num_heads, head_size, dtype=torch.float32, device=device
        )
        if normalize:
            x = torch.nn.functional.normalize(x, p=2.0, dim=-1)
        return x.to(torch.bfloat16).contiguous()

    def channel_gate(dtype):
        return (
            torch.empty(
                total_seq_len,
                num_sab_heads,
                head_size,
                dtype=torch.float32,
                device=device,
            )
            .uniform_(0.1, 1.0)
            .to(dtype)
        )

    base = total_seq_len // max(1, num_seqs)
    rem = total_seq_len % max(1, num_seqs)
    cum = [0]
    for i in range(num_seqs):
        cum.append(cum[-1] + base + (1 if i < rem else 0))
    return {
        "q": qkv(num_q_heads, True),
        "k": qkv(num_k_heads, True),
        "v": qkv(num_v_heads, False),
        "g": channel_gate(torch.float32),
        "beta": channel_gate(torch.bfloat16),
        "w": channel_gate(torch.bfloat16),
        "cu_seqlens": torch.tensor(cum, dtype=torch.int64, device=device),
    }


gdn2_prefill_trace = TraceTemplate(
    op_type="gdn2",
    name_prefix="gdn2_prefill",
    description=(
        "Gated Delta Net 2 prefill: GDN's per-head scalar forget and erase "
        "gates become per key channel and a third write gate scales the "
        "incoming value per value channel. The state is in k-last layout "
        "[N, H, V, K]."
    ),
    axes={
        "total_seq_len": Var(
            description="Total number of tokens across all sequences in the batch."
        ),
        "num_seqs": Var(description="Number of sequences in the batch."),
        "num_q_heads": Const(description="Number of query heads.", abbrev="qk"),
        "num_k_heads": Const(description="Number of key heads.", abbrev=""),
        "num_v_heads": Const(
            description="Number of value heads (GVA: more value heads than query heads).",
            abbrev="v",
        ),
        "head_size": Const(
            description="Dimension of each attention head (K in query/key space, V in value space).",
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
        "g": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            description="Channel-wise forget gate in linear space, per key channel.",
        ),
        "beta": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            description="Channel-wise erase gate, post-sigmoid, per key channel.",
        ),
        "w": Tensor(
            ["total_seq_len", "num_v_heads", "head_size"],
            description="Channel-wise write gate, per value channel.",
        ),
        "initial_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            optional=True,
            description="Incoming recurrent state in k-last layout [N, H, V, K].",
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
            dtype_from="q",
            description="Attention output. Shape follows num_v_heads in GVA mode.",
        ),
        "final_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Outgoing recurrent state in k-last layout [N, H, V, K].",
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
    reference=_gdn2_prefill_reference,
    init=_gdn2_prefill_init,
)
