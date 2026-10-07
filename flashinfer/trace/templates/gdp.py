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

"""TraceTemplate for Gated DeltaProduct (GDP) prefill."""

import math
from copy import deepcopy

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _gdp_prefill_reference(
    q, k, v, g, beta, initial_state, cu_seqlens, scale, num_householder
):
    """Gated DeltaProduct prefill reference, state V-major as ``[N, H, V, K]``.

    Per real token, with ``n = num_householder`` and sub-token rows
    ``t*n .. t*n + n - 1`` of k/v/beta,

        S = alpha_t S
        for j in 0..n-1:
            v_new = beta_{t,j} * (v_{t,j} - S k_{t,j})
            S += v_new (x) k_{t,j}
        o_t = scale * S q_t
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    n = int(num_householder)
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    q_exp = q.float().repeat_interleave(num_sab_heads // num_q_heads, dim=1)
    k_exp = k.float().repeat_interleave(num_sab_heads // num_k_heads, dim=1)
    v_exp = v.float().repeat_interleave(num_sab_heads // num_v_heads, dim=1)
    g_f32 = (
        torch.ones(total_seq_len, num_sab_heads, dtype=torch.float32, device=device)
        if g is None
        else g.float()
    )
    beta_f32 = (
        torch.ones(total_seq_len * n, num_sab_heads, dtype=torch.float32, device=device)
        if beta is None
        else beta.float()
    )

    output = torch.zeros(
        (total_seq_len, num_sab_heads, v.shape[2]), dtype=q.dtype, device=device
    )
    final_state = torch.zeros(
        (num_seqs, num_sab_heads, v.shape[2], head_size),
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
                (num_sab_heads, v.shape[2], head_size),
                dtype=torch.float32,
                device=device,
            )
        for t in range(start, end):
            state = state * g_f32[t][:, None, None]
            for j in range(t * n, t * n + n):
                key = k_exp[j]
                v_new = v_exp[j] - torch.einsum("hvk,hk->hv", state, key)
                v_new = beta_f32[j][:, None] * v_new
                state = state + v_new[:, :, None] * key[:, None, :]
            projected = torch.einsum("hvk,hk->hv", state, q_exp[t])
            output[t] = (scale * projected).to(q.dtype)
        final_state[seq_idx] = state

    return output, final_state


def _gdp_prefill_init(
    *,
    total_seq_len: int,
    expanded_seq_len: int = 0,  # derived
    num_seqs: int = 4,
    len_cu_seqlens: int = 0,  # derived
    num_householder: int = 2,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdp_prefill.chunk_gated_delta_product``."""
    del expanded_seq_len, len_cu_seqlens
    torch.manual_seed(seed)
    num_sab_heads = max(num_q_heads, num_v_heads)
    expanded = total_seq_len * num_householder

    def qkv(rows, num_heads, normalize):
        x = torch.randn(rows, num_heads, head_size, dtype=torch.float32, device=device)
        if normalize:
            x = torch.nn.functional.normalize(x, p=2.0, dim=-1)
        return x.to(torch.bfloat16).contiguous()

    base = total_seq_len // max(1, num_seqs)
    rem = total_seq_len % max(1, num_seqs)
    cum = [0]
    for i in range(num_seqs):
        cum.append(cum[-1] + base + (1 if i < rem else 0))
    return {
        "q": qkv(total_seq_len, num_q_heads, True),
        "k": qkv(expanded, num_k_heads, True),
        "v": qkv(expanded, num_v_heads, False),
        "g": torch.empty(
            total_seq_len, num_sab_heads, dtype=torch.float32, device=device
        ).uniform_(0.1, 1.0),
        "beta": torch.empty(
            expanded, num_sab_heads, dtype=torch.float32, device=device
        ).uniform_(0.1, 1.0),
        "num_householder": num_householder,
        "cu_seqlens": torch.tensor(cum, dtype=torch.int64, device=device),
    }


_gdp_prefill_default_trace = TraceTemplate(
    op_type="gdp",
    name_prefix="gdp_prefill",
    description=(
        "Gated DeltaProduct prefill: num_householder beta-gated Householder "
        "updates per token with one per-head scalar decay per token. k/v/beta "
        "ride the expanded sub-token timeline; q, g and the outputs live at "
        "real-token rows. The state is in k-last layout [N, H, V, K]."
    ),
    axes={
        "total_seq_len": Var(
            description="Total number of real tokens across all sequences in the batch."
        ),
        "expanded_seq_len": Var(
            description="Total sub-token rows, total_seq_len * num_householder."
        ),
        "num_seqs": Var(description="Number of sequences in the batch."),
        "num_householder": Const(
            description="Householder updates per token.", abbrev="n"
        ),
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
            description="Query tensor at real-token rows.",
        ),
        "k": Tensor(
            ["expanded_seq_len", "num_k_heads", "head_size"],
            description="Key tensor on the expanded sub-token timeline.",
        ),
        "v": Tensor(
            ["expanded_seq_len", "num_v_heads", "head_size"],
            description="Value tensor on the expanded sub-token timeline.",
        ),
        "g": Tensor(
            ["total_seq_len", "num_v_heads"],
            optional=True,
            description="Per-head forget gate in linear space, at real-token rows.",
        ),
        "beta": Tensor(
            ["expanded_seq_len", "num_v_heads"],
            optional=True,
            description="Per-head, per-Householder update gate, post-sigmoid.",
        ),
        "num_householder": Scalar(
            "int32",
            description="Householder updates per token.",
        ),
        "initial_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            optional=True,
            description="Incoming recurrent state in k-last layout [N, H, V, K].",
        ),
        "cu_seqlens": Tensor(
            ["len_cu_seqlens"],
            description="Cumulative real-token sequence lengths for variable-length batching.",
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
            description="Attention output at real-token rows. Shape follows num_v_heads in GVA mode.",
        ),
        "final_state": Tensor(
            ["num_seqs", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Outgoing recurrent state in k-last layout [N, H, V, K].",
        ),
    },
    constraints=[
        "expanded_seq_len == total_seq_len * num_householder",
        "num_householder >= 1",
        "num_k_heads == num_q_heads or num_k_heads == num_v_heads",
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "len_cu_seqlens == num_seqs + 1",
        "total_seq_len == cu_seqlens[-1].item()",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_gdp_prefill_reference,
    init=_gdp_prefill_init,
)


@torch.no_grad()
def _gdp_prefill_pool_reference(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    num_householder,
    state_indices,
    output_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Independent compact recurrence, gathering and scattering only pool slots."""
    dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q.float(), dim=-1)
        k = torch.nn.functional.normalize(k.float(), dim=-1)
    selected = initial_state.index_select(0, state_indices.long())
    output, final = _gdp_prefill_reference(
        q, k, v, g, beta, selected, cu_seqlens, scale, num_householder
    )
    output = output.to(dtype)
    if not output_final_state:
        return output, None
    destination = output_state.clone()
    destination.index_copy_(0, state_indices.long(), final.to(destination.dtype))
    return output, destination


_gdp_prefill_pool_reference._trace_reference_dependencies = (  # type: ignore[attr-defined]
    _gdp_prefill_reference,
)


def _gdp_prefill_pool_init(
    *,
    total_seq_len: int,
    num_seqs: int = 4,
    state_pool_rows: int = 0,
    len_cu_seqlens: int = 0,
    expanded_seq_len: int = 0,
    num_householder: int = 2,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    inputs = _gdp_prefill_init(
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        len_cu_seqlens=len_cu_seqlens,
        expanded_seq_len=expanded_seq_len,
        num_householder=num_householder,
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
        initial_state=torch.zeros(rows, heads, head_size, head_size, device=device),
        output_state=torch.zeros(rows, heads, head_size, head_size, device=device),
        output_final_state=True,
        state_indices=torch.arange(num_seqs, device=device, dtype=torch.int32),
    )
    return inputs


_gdp_prefill_pool_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _gdp_prefill_init,
)
_gdp_prefill_pool_trace = deepcopy(_gdp_prefill_default_trace)
_gdp_prefill_pool_trace.name_prefix = "gdp_prefill_pool"
_gdp_prefill_pool_trace.reference = _gdp_prefill_pool_reference
_gdp_prefill_pool_trace.init = _gdp_prefill_pool_init
_gdp_prefill_pool_trace.axes["state_pool_rows"] = Var(
    description="Rows in the recurrent state pool."
)
_gdp_prefill_pool_trace.axes["num_q_heads"].abbrev = "q"
_gdp_prefill_pool_trace.axes["num_k_heads"].abbrev = "k"
_gdp_prefill_pool_trace.axes["num_o_heads"] = Const(
    description="max(num_q_heads, num_v_heads).", abbrev=""
)
# Pool traces cover both grouped-query and grouped-value head layouts.
for _name, _spec in _gdp_prefill_pool_trace.inputs.items():
    if isinstance(_spec, Tensor) and _name not in ("q", "k", "v"):
        _spec.dim_names = [
            "num_o_heads" if d == "num_v_heads" else d for d in _spec.dim_names
        ]
_gdp_prefill_pool_trace.inputs["initial_state"].dim_names[0] = "state_pool_rows"
_gdp_prefill_pool_trace.inputs.update(
    state_indices=Tensor(["num_seqs"]),
    output_state=Tensor(
        ["state_pool_rows", "num_o_heads", "head_size", "head_size"], optional=True
    ),
    output_final_state=Scalar("int32", optional=True),
    use_qk_l2norm_in_kernel=Scalar("int32", optional=True),
)
_gdp_prefill_pool_trace.outputs["output"].dim_names[1] = "num_o_heads"
_gdp_prefill_pool_trace.outputs["final_state"] = Tensor(
    ["state_pool_rows", "num_o_heads", "head_size", "head_size"],
    optional=True,
    dtype_from="output_state",
    dtype="float32",
)
_gdp_prefill_pool_trace.constraints = [
    "num_o_heads == max(num_q_heads, num_v_heads)",
    "num_o_heads % num_q_heads == 0",
    "num_o_heads % num_v_heads == 0",
    "num_k_heads in (num_q_heads, num_v_heads)",
    "len_cu_seqlens == num_seqs + 1",
    "total_seq_len == cu_seqlens[-1].item()",
    "expanded_seq_len == total_seq_len * num_householder",
    "num_householder >= 1",
]


def gdp_prefill_trace(**kwargs):
    if kwargs.get("state_indices") is not None:
        return _gdp_prefill_pool_trace
    return _gdp_prefill_default_trace


gdp_prefill_trace.templates = [  # type: ignore[attr-defined]
    _gdp_prefill_default_trace,
    _gdp_prefill_pool_trace,
]
