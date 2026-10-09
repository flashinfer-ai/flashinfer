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


gdp_prefill_trace = TraceTemplate(
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
def _gdp_decode_reference(
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
    output=None,
    ssm_state_indices=None,
    disable_state_update=None,
    use_qk_l2norm=True,
    output_state_indices=None,
):
    """Direct GDP decode recurrence, in fp64.

    Per real token the state decays once, then takes ``n_h`` Householder
    updates; the readout follows the last one.  ``k``/``v``/``b`` carry the
    householder axis at dim 2, ``q``/``a`` do not.
    """
    B, T, H, K = q.shape
    n_h = k.shape[2]
    HV = v.shape[3]
    scale = K**-0.5 if scale is None else scale
    raw_q, raw_k, raw_v, raw_a, raw_b = (
        x.to(torch.bfloat16).double() for x in (q, k, v, a, b)
    )
    query, key = raw_q, raw_k
    if use_qk_l2norm:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query.repeat_interleave(HV // H, dim=2) * scale
    key = key.repeat_interleave(HV // H, dim=3)
    log_g = -A_log.double().exp() * torch.nn.functional.softplus(
        raw_a + dt_bias.double()
    )
    beta = raw_b.sigmoid()
    out = torch.zeros(B, T, HV, v.shape[-1], dtype=v.dtype, device=v.device)
    final_state = initial_state.clone()
    for row, slot in enumerate(initial_state_indices.tolist()):
        if slot < 0:
            continue
        state = initial_state[slot].double()
        for step in range(T):
            # the gate models time passing, so it applies once per real token
            state *= log_g[row, step, :, None, None].exp()
            for j in range(n_h):
                kt = key[row, step, j]
                prediction = (state * kt[:, None, :]).sum(-1)
                delta = (raw_v[row, step, j] - prediction) * beta[row, step, j, :, None]
                state += delta[..., None] * kt[:, None, :]
            out[row, step] = (
                (state * query[row, step, :, None, :]).sum(-1).to(out.dtype)
            )
            if ssm_state_indices is not None:
                token_slot = int(ssm_state_indices[row, step])
                if token_slot >= 0:
                    final_state[token_slot] = state.float()
        if not disable_state_update and ssm_state_indices is None:
            write_slot = slot
            if output_state_indices is not None:
                write_slot = int(output_state_indices[row])
            if write_slot >= 0:
                final_state[write_slot] = state.float()
    return out, final_state


def _gdp_decode_init(
    *,
    batch_size: int,
    seq_len: int = 2,
    num_householder: int = 2,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    pool_size: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gdp_decode.gated_delta_product_mtp``.

    Mirrors the GDN MTP fixture, with a householder axis on ``k``/``v``/``b``:
    ``k`` L2-normalized, ``A_log``/``dt_bias``/``a`` scaled by 0.1, and
    ``initial_state_indices`` mapping each batch row to a distinct pool slot.
    """
    torch.manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    fp32 = dict(dtype=torch.float32, device=device)
    q = torch.randn(batch_size, seq_len, num_q_heads, head_size, **bf16)
    k = torch.randn(
        batch_size, seq_len, num_householder, num_k_heads, head_size, **bf16
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(
        batch_size, seq_len, num_householder, num_v_heads, head_size, **bf16
    )
    initial_state = torch.randn(pool_size, num_v_heads, head_size, head_size, **fp32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": initial_state,
        "initial_state_indices": torch.arange(
            batch_size, dtype=torch.int32, device=device
        ),
        "A_log": torch.randn(num_v_heads, **fp32) * 0.1,
        "a": torch.randn(batch_size, seq_len, num_v_heads, **bf16) * 0.1,
        "dt_bias": torch.randn(num_v_heads, **fp32) * 0.1,
        "b": torch.randn(batch_size, seq_len, num_householder, num_v_heads, **bf16),
        "scale": head_size**-0.5,
    }


gdp_decode_trace = TraceTemplate(
    op_type="gdp",
    name_prefix="gdp_decode",
    description=(
        "Gated DeltaProduct decode / MTP: num_householder Householder updates "
        "per real token on the GDN MTP kernel. k/v/b carry the householder "
        "axis next to the token axis; q, the decay logits and the output carry "
        "one row per real token. State layout is k-last [pool_size, H, V, K]."
    ),
    axes={
        "batch_size": Var(description="Number of sequences decoded concurrently."),
        "seq_len": Var(
            description="Real tokens per sequence (T > 1 under speculative decoding)."
        ),
        "num_householder": Const(
            description="Householder updates per real token. 1 is plain GDN.",
            abbrev="nh",
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
            description="Dimension of each attention head (K in query/key space, V in value space).",
            abbrev="d",
        ),
        "pool_size": Var(description="Size of the state pool for efficient batching."),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "seq_len", "num_q_heads", "head_size"],
            description="Query tensor, one row per real token.",
        ),
        "k": Tensor(
            ["batch_size", "seq_len", "num_householder", "num_k_heads", "head_size"],
            description="Key tensor, one row per (token, Householder).",
        ),
        "v": Tensor(
            ["batch_size", "seq_len", "num_householder", "num_v_heads", "head_size"],
            description="Value tensor, one row per (token, Householder).",
        ),
        "initial_state": Tensor(
            ["pool_size", "num_v_heads", "head_size", "head_size"],
            description="Initial recurrent state pool in k-last layout [pool_size, H, V, K].",
        ),
        "initial_state_indices": Tensor(
            ["batch_size"],
            description="Pool slot each batch row reads. Negative rows are skipped.",
        ),
        "A_log": Tensor(
            ["num_v_heads"],
            description="Log decay parameter (learnable). Used to compute g = exp(-exp(A_log) * softplus(a + dt_bias)).",
        ),
        "a": Tensor(
            ["batch_size", "seq_len", "num_v_heads"],
            description="Input-dependent decay from projection, one row per real token.",
        ),
        "dt_bias": Tensor(
            ["num_v_heads"],
            description="Decay bias (learnable). Added to 'a' before softplus.",
        ),
        "b": Tensor(
            ["batch_size", "seq_len", "num_householder", "num_v_heads"],
            description="Update gate input per (token, Householder). beta = sigmoid(b).",
        ),
        "scale": Scalar(
            "float32",
            optional=True,
            description="Scale factor. Default is 1/sqrt(head_size).",
        ),
        "ssm_state_indices": Tensor(
            ["batch_size", "seq_len"],
            dtype="int32",
            optional=True,
            description=(
                "Per-real-token scatter slot, written after the final Householder "
                "update. Negative entries skip the write."
            ),
        ),
        "disable_state_update": Scalar("bool", optional=True),
        "use_qk_l2norm": Scalar("bool", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "seq_len", "num_v_heads", "head_size"],
            dtype="bfloat16",
            description="Attention output, one row per real token. Shape follows num_v_heads in GVA mode.",
        ),
        "final_state": Tensor(
            ["pool_size", "num_v_heads", "head_size", "head_size"],
            dtype="float32",
            description="Updated recurrent state pool in k-last layout [pool_size, H, V, K].",
        ),
    },
    constraints=[
        "num_householder >= 1",
        "num_v_heads >= num_q_heads",
        "num_v_heads % num_q_heads == 0",
        "num_k_heads == num_q_heads",
    ],
    tags=["stage:mtp", "status:verified"],
    reference=_gdp_decode_reference,
    init=_gdp_decode_init,
)
