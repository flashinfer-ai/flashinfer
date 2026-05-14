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

"""TraceTemplates for sampling operations."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── Top-k sampling ────────────────────────────────────────────────────────────


@torch.no_grad()
def _top_k_sampling_reference(probs, top_k):
    """Top-k sampling: keep only the k highest probability tokens, renormalize, then sample."""
    batch_size, vocab_size = probs.shape
    device = probs.device
    probs = probs.to(torch.float32)
    samples = torch.empty(batch_size, dtype=torch.int64, device=device)
    for i in range(batch_size):
        row = probs[i]
        k = int(top_k[i].item())
        if 0 < k < vocab_size:
            idx_sorted = torch.argsort(row, descending=True)
            keep_idx = idx_sorted[:k]
            filtered = torch.zeros_like(row)
            filtered[keep_idx] = row[keep_idx]
            row = filtered / filtered.sum()
        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)
    return samples


top_k_sampling_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_sampling",
    description=(
        "Top-k sampling from probabilities. Keeps only the k highest probability tokens, "
        "renormalizes, then samples from the filtered distribution."
    ),
    axes={
        "batch_size": Var(description="Number of sequences to sample from"),
        "vocab_size": Const(description="Vocabulary size.", abbrev="v"),
    },
    inputs={
        "probs": Tensor(
            ["batch_size", "vocab_size"],
            description="Probability distributions (after softmax)",
        ),
        "top_k": Tensor(
            ["batch_size"],
            description="Number of top tokens to consider for sampling per sequence",
        ),
    },
    outputs={
        "samples": Tensor(
            ["batch_size"],
            dtype="int64",
            description="Sampled token indices",
        ),
    },
    tags=["status:verified"],
    reference=_top_k_sampling_reference,
)

# ── Top-p sampling ────────────────────────────────────────────────────────────


@torch.no_grad()
def _top_p_sampling_reference(probs, top_p):
    """Top-p (nucleus) sampling: filter by cumulative probability threshold, then sample."""
    batch_size, vocab_size = probs.shape
    device = probs.device
    probs = probs.to(torch.float32)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)
    for i in range(batch_size):
        row = probs[i]
        p = float(top_p[i].item())
        if p <= 0.0:
            out[i] = torch.argmax(row).to(torch.int64)
            continue
        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)
            to_remove = cdf > p
            to_remove[1:] = to_remove[:-1].clone()
            to_remove[0] = False
            keep_idx = idx[~to_remove]
            filtered = torch.zeros_like(row)
            filtered[keep_idx] = row[keep_idx]
            row = filtered / filtered.sum()
        out[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)
    return out


top_p_sampling_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_p_sampling",
    description=(
        "Top-p (nucleus) sampling from probabilities. Filters probabilities using "
        "cumulative probability threshold, then samples from the filtered distribution."
    ),
    axes={
        "batch_size": Var(description="Number of sequences to sample from"),
        "vocab_size": Const(description="Vocabulary size.", abbrev="v"),
    },
    inputs={
        "probs": Tensor(
            ["batch_size", "vocab_size"],
            description="Probability distributions (after softmax)",
        ),
        "top_p": Tensor(
            ["batch_size"],
            description="Cumulative probability threshold for nucleus sampling per sequence",
        ),
    },
    outputs={
        "samples": Tensor(
            ["batch_size"],
            dtype="int64",
            description="Sampled token indices",
        ),
    },
    tags=["status:verified"],
    reference=_top_p_sampling_reference,
)

# ── Top-k + Top-p sampling ────────────────────────────────────────────────────


@torch.no_grad()
def _top_k_top_p_sampling_reference(probs, top_k, top_p):
    """Top-k then top-p (nucleus) sampling: apply both filters, then sample."""
    batch_size, vocab_size = probs.shape
    device = probs.device
    probs = probs.to(torch.float32)
    samples = torch.empty(batch_size, dtype=torch.int64, device=device)
    for i in range(batch_size):
        row = probs[i]
        k = int(top_k[i].item())
        p = float(top_p[i].item())
        if 0 < k < vocab_size:
            idx_sorted = torch.argsort(row, descending=True)
            keep_idx_k = idx_sorted[:k]
            filtered_k = torch.zeros_like(row)
            filtered_k[keep_idx_k] = row[keep_idx_k]
            row = filtered_k / filtered_k.sum()
        if p <= 0.0:
            samples[i] = torch.argmax(row).to(torch.int64)
            continue
        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)
            to_remove = cdf > p
            if vocab_size > 1:
                to_remove[1:] = to_remove[:-1].clone()
                to_remove[0] = False
            keep_idx_p = idx[~to_remove]
            filtered_p = torch.zeros_like(row)
            filtered_p[keep_idx_p] = row[keep_idx_p]
            row = filtered_p / filtered_p.sum()
        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)
    return samples


top_k_top_p_sampling_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_top_p_sampling",
    description=(
        "Top-k top-p (nucleus) sampling from probabilities. Filters probabilities using "
        "top-k and top-p constraints, then samples from the filtered distribution."
    ),
    axes={
        "batch_size": Var(description="Number of sequences to sample from"),
        "vocab_size": Const(description="Vocabulary size.", abbrev="v"),
    },
    inputs={
        "probs": Tensor(
            ["batch_size", "vocab_size"],
            description="Probability distributions (after softmax)",
        ),
        "top_k": Tensor(
            ["batch_size"],
            description="Number of top tokens to consider for sampling per sequence",
        ),
        "top_p": Tensor(
            ["batch_size"],
            description="Cumulative probability threshold for nucleus sampling per sequence",
        ),
    },
    outputs={
        "samples": Tensor(
            ["batch_size"],
            dtype="int64",
            description="Sampled token indices",
        ),
    },
    tags=["status:verified"],
    reference=_top_k_top_p_sampling_reference,
)


# ── Free-function sampling utilities ─────────────────────────────────────────


@torch.no_grad()
def _softmax_reference(logits, temperature=None, **_unused):
    """Online safe softmax with optional temperature scaling."""
    x = logits.to(torch.float32)
    if temperature is not None:
        if isinstance(temperature, torch.Tensor):
            t = temperature.to(torch.float32).reshape(-1, 1)
        else:
            t = float(temperature)
        x = x / t
    return torch.softmax(x, dim=-1).to(logits.dtype)


softmax_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="softmax",
    description="Fused online safe softmax with optional temperature scaling.",
    axes={
        "batch_size": Var(),
        "vocab_size": Const(abbrev="v"),
    },
    inputs={
        "logits": Tensor(["batch_size", "vocab_size"]),
        "temperature": Scalar(
            "float32",
            optional=True,
            description="Per-tensor or per-row temperature.",
        ),
    },
    outputs={
        "output": Tensor(["batch_size", "vocab_size"], dtype_from="logits"),
    },
    tags=["status:verified"],
    reference=_softmax_reference,
)


@torch.no_grad()
def _sampling_from_probs_reference(probs, indices=None, **_unused):
    """Categorical sampling from probabilities (deterministic: argmax)."""
    p = probs.to(torch.float32)
    if indices is not None:
        p = p[indices.to(torch.long)]
    return p.argmax(dim=-1).to(torch.int32)


_sampling_common_axes: dict[str, Var | Const] = {
    "batch_size": Var(),
    "vocab_size": Const(abbrev="v"),
    "num_indices": Var(description="Length of optional indices tensor."),
}

sampling_from_probs_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="sampling_from_probs",
    description=(
        "Fused categorical sampling from [batch_size, vocab_size] probs. "
        "Reference uses argmax (matches deterministic=True)."
    ),
    axes=dict(_sampling_common_axes),
    inputs={
        "probs": Tensor(["batch_size", "vocab_size"]),
        "indices": Tensor(
            ["num_indices"],
            dtype="int32",
            optional=True,
        ),
    },
    outputs={"samples": Tensor(["batch_size"], dtype="int32")},
    tags=["status:verified"],
    reference=_sampling_from_probs_reference,
)


@torch.no_grad()
def _sampling_from_logits_reference(logits, indices=None, **_unused):
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    return _sampling_from_probs_reference(probs, indices=indices)


sampling_from_logits_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="sampling_from_logits",
    description=(
        "Fused sampling from logits (equivalent to softmax + sampling). "
        "Reference uses softmax + argmax."
    ),
    axes=dict(_sampling_common_axes),
    inputs={
        "logits": Tensor(["batch_size", "vocab_size"]),
        "indices": Tensor(
            ["num_indices"],
            dtype="int32",
            optional=True,
        ),
    },
    outputs={"samples": Tensor(["batch_size"], dtype="int32")},
    tags=["status:verified"],
    reference=_sampling_from_logits_reference,
)


@torch.no_grad()
def _min_p_sampling_reference(probs, min_p, indices=None, **_unused):
    """Min-p sampling: keep probs >= min_p * max_prob, renormalise, then argmax."""
    p = probs.to(torch.float32)
    if indices is not None:
        p = p[indices.to(torch.long)]
    if isinstance(min_p, torch.Tensor):
        mp = min_p.to(torch.float32).reshape(-1, 1)
    else:
        mp = float(min_p)
    threshold = p.max(dim=-1, keepdim=True).values * mp
    mask = p >= threshold
    p_masked = torch.where(mask, p, torch.zeros_like(p))
    p_masked = p_masked / (p_masked.sum(dim=-1, keepdim=True) + 1e-20)
    return p_masked.argmax(dim=-1).to(torch.int32)


min_p_sampling_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="min_p_sampling",
    description=(
        "Fused min-p sampling: keep probs >= min_p * max_prob, renormalise, "
        "categorical sample."
    ),
    axes=dict(_sampling_common_axes),
    inputs={
        "probs": Tensor(["batch_size", "vocab_size"]),
        "min_p": Scalar(
            "float32",
            description="Min-p threshold (scalar or per-row tensor).",
        ),
        "indices": Tensor(
            ["num_indices"],
            dtype="int32",
            optional=True,
        ),
    },
    outputs={"samples": Tensor(["batch_size"], dtype="int32")},
    tags=["status:verified"],
    reference=_min_p_sampling_reference,
)


@torch.no_grad()
def _top_p_renorm_probs_reference(probs, top_p, **_unused):
    """Renormalise probs by top-p thresholding."""
    p = probs.to(torch.float32)
    if isinstance(top_p, torch.Tensor):
        tp = top_p.to(torch.float32).reshape(-1, 1)
    else:
        tp = float(top_p)
    sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)
    cumsum = sorted_p.cumsum(dim=-1)
    keep_sorted = (cumsum - sorted_p) < tp
    keep = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, sorted_idx, keep_sorted)
    p_masked = torch.where(keep, p, torch.zeros_like(p))
    return (p_masked / (p_masked.sum(dim=-1, keepdim=True) + 1e-20)).to(probs.dtype)


top_p_renorm_probs_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_p_renorm_probs",
    description="Renormalise probabilities by top-p thresholding.",
    axes={"batch_size": Var(), "vocab_size": Const(abbrev="v")},
    inputs={
        "probs": Tensor(["batch_size", "vocab_size"]),
        "top_p": Scalar("float32"),
    },
    outputs={
        "renormalized": Tensor(["batch_size", "vocab_size"], dtype_from="probs"),
    },
    tags=["status:verified"],
    reference=_top_p_renorm_probs_reference,
)


@torch.no_grad()
def _top_k_renorm_probs_reference(probs, top_k, **_unused):
    """Renormalise probs by top-k thresholding."""
    p = probs.to(torch.float32)
    if isinstance(top_k, torch.Tensor):
        k = int(top_k.max().item())
    else:
        k = int(top_k)
    _, topk_idx = torch.topk(p, k=k, dim=-1)
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    p_masked = torch.where(mask, p, torch.zeros_like(p))
    return (p_masked / (p_masked.sum(dim=-1, keepdim=True) + 1e-20)).to(probs.dtype)


top_k_renorm_probs_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_renorm_probs",
    description="Renormalise probabilities by top-k thresholding.",
    axes={"batch_size": Var(), "vocab_size": Const(abbrev="v")},
    inputs={
        "probs": Tensor(["batch_size", "vocab_size"]),
        "top_k": Scalar("int32"),
    },
    outputs={
        "renormalized": Tensor(["batch_size", "vocab_size"], dtype_from="probs"),
    },
    tags=["status:verified"],
    reference=_top_k_renorm_probs_reference,
)


@torch.no_grad()
def _top_k_mask_logits_reference(logits, top_k, **_unused):
    """Mask logits outside the top-k to -inf."""
    x = logits.to(torch.float32)
    if isinstance(top_k, torch.Tensor):
        k = int(top_k.max().item())
    else:
        k = int(top_k)
    _, topk_idx = torch.topk(x, k=k, dim=-1)
    mask = torch.full_like(x, float("-inf"))
    mask.scatter_(-1, topk_idx, 0.0)
    return (x + mask).to(logits.dtype)


top_k_mask_logits_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_mask_logits",
    description="Mask out-of-top-k logits to -inf.",
    axes={"batch_size": Var(), "vocab_size": Const(abbrev="v")},
    inputs={
        "logits": Tensor(["batch_size", "vocab_size"]),
        "top_k": Scalar("int32"),
    },
    outputs={
        "masked_logits": Tensor(["batch_size", "vocab_size"], dtype_from="logits"),
    },
    tags=["status:verified"],
    reference=_top_k_mask_logits_reference,
)


@torch.no_grad()
def _top_k_top_p_sampling_from_logits_reference(
    logits, top_k, top_p, indices=None, filter_apply_order="top_k_first", **_unused
):
    """top-k + top-p sampling from logits (deterministic: argmax)."""
    x = logits.to(torch.float32)
    if filter_apply_order == "top_k_first":
        x = _top_k_mask_logits_reference(x, top_k)
        probs = torch.softmax(x, dim=-1)
        probs = _top_p_renorm_probs_reference(probs, top_p)
    else:  # "joint"
        probs = torch.softmax(x, dim=-1)
        probs = _top_k_renorm_probs_reference(probs, top_k)
        probs = _top_p_renorm_probs_reference(probs, top_p)
    if indices is not None:
        probs = probs[indices.to(torch.long)]
    return probs.argmax(dim=-1).to(torch.int32)


top_k_top_p_sampling_from_logits_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_top_p_sampling_from_logits",
    description=(
        "Fused top-k + top-p sampling starting from logits. "
        "Reference: softmax + top_k_mask + top_p_renorm + argmax."
    ),
    axes=dict(_sampling_common_axes),
    inputs={
        "logits": Tensor(["batch_size", "vocab_size"]),
        "top_k": Scalar("int32"),
        "top_p": Scalar("float32"),
        "indices": Tensor(
            ["num_indices"],
            dtype="int32",
            optional=True,
        ),
    },
    outputs={"samples": Tensor(["batch_size"], dtype="int32")},
    tags=["status:verified"],
    reference=_top_k_top_p_sampling_from_logits_reference,
)


@torch.no_grad()
def _chain_speculative_sampling_reference(
    draft_probs,
    draft_token_ids,
    target_probs,
    **_unused,
):
    """Deterministic chain speculative sampling: accept draft[i] iff
    target_prob[draft[i]] >= draft_prob[draft[i]]; emit argmax of the
    first rejecting target distribution (or last step)."""
    B, S = draft_token_ids.shape
    dp = draft_probs.to(torch.float32)
    tp = target_probs.to(torch.float32)
    out = torch.full(
        (B, S + 1),
        -1,
        dtype=torch.int32,
        device=draft_token_ids.device,
    )
    for b in range(B):
        for s in range(S):
            tok = int(draft_token_ids[b, s].item())
            if tp[b, s, tok] >= dp[b, s, tok]:
                out[b, s] = tok
            else:
                out[b, s] = int(tp[b, s].argmax().item())
                break
        else:
            out[b, S] = int(tp[b, S].argmax().item())
    return out


chain_speculative_sampling_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="chain_speculative_sampling",
    description=(
        "Chain speculative sampling: accept/reject draft tokens against target "
        "distribution and emit the accepted prefix + one sampled final token."
    ),
    axes={
        "batch_size": Var(),
        "num_speculative": Var(description="Draft tokens per step."),
        "num_speculative_plus_1": Var(
            description="num_speculative + 1 (draft_probs axis)."
        ),
        "vocab_size": Const(abbrev="v"),
    },
    inputs={
        "draft_probs": Tensor(
            ["batch_size", "num_speculative_plus_1", "vocab_size"],
        ),
        "draft_token_ids": Tensor(
            ["batch_size", "num_speculative"],
            dtype="int32",
        ),
        "target_probs": Tensor(
            ["batch_size", "num_speculative_plus_1", "vocab_size"],
        ),
    },
    outputs={
        "accepted_token_ids": Tensor(
            ["batch_size", "num_speculative_plus_1"], dtype="int32"
        ),
    },
    tags=["status:verified", "speculative"],
    reference=_chain_speculative_sampling_reference,
)


# ── Top-K + ragged index transform (sparse attention helper) ─────────────────


@torch.no_grad()
def _top_k_ragged_transform_reference(
    input: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    deterministic: bool = False,
    tie_break: int = 0,
    dsa_graph_safe: bool = False,
    row_starts=None,
    **_unused,
) -> torch.Tensor:
    """Reference for top_k_ragged_transform: per-row top-k selection on the
    leading ``lengths[i]`` valid entries, then add per-row ``offsets[i]`` to
    the selected indices. Used as the second stage of sparse attention to
    produce ragged page indices.
    """
    num_rows = input.shape[0]
    out = torch.zeros(num_rows, int(k), dtype=torch.int32, device=input.device)
    for i in range(num_rows):
        L = int(lengths[i].item())
        off = int(offsets[i].item())
        if L <= 0:
            continue
        row = input[i, :L].to(torch.float32)
        kk = min(int(k), L)
        _, idx = torch.topk(row, kk, sorted=True)
        out[i, :kk] = idx.to(torch.int32) + off
    return out


top_k_ragged_transform_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_ragged_transform",
    description=(
        "Fused per-row top-k selection on a ragged input plus index "
        "rebasing: for each row i, picks the top-k indices over "
        "input[i, :lengths[i]] and emits (selected_idx + offsets[i]). "
        "Used in sparse-attention page selection."
    ),
    axes={
        "num_rows": Var(),
        "max_len": Var(description="Padded row length of `input`."),
        "k": Const(abbrev="k"),
    },
    inputs={
        "input": Tensor(["num_rows", "max_len"]),
        "offsets": Tensor(["num_rows"], dtype="int32"),
        "lengths": Tensor(["num_rows"], dtype="int32"),
        "k": Scalar("int32"),
        "deterministic": Scalar("int32", optional=True),
        "tie_break": Scalar("int32", optional=True),
        "dsa_graph_safe": Scalar("int32", optional=True),
    },
    outputs={
        "indices": Tensor(["num_rows", "k"], dtype="int32"),
    },
    tags=["status:verified", "sparse"],
    reference=_top_k_ragged_transform_reference,
)


# ── DeepSeek-V3 fused expert routing (top-k) ─────────────────────────────────


@torch.no_grad()
def _fused_topk_deepseek_reference(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    **_unused,
) -> None:
    """Reference for DeepSeek-V3 fused expert routing.

    1. Compute biased scores: sigmoid(scores) + bias per expert.
    2. Group experts (n_group groups), score each group as the sum of its
       top-2 biased scores.
    3. Pick top ``topk_group`` groups.
    4. Within those groups, pick top ``topk`` experts by biased score.
    5. Output normalized weights = sigmoid_score / sum(sigmoid_scores) *
       routed_scaling_factor and the selected expert indices.

    Mutates ``topk_values`` and ``topk_indices`` in place.
    """
    T, E = scores.shape
    sig = torch.sigmoid(scores.to(torch.float32))
    biased = sig + bias.to(torch.float32).unsqueeze(0)
    # Group scores: top-2 per group, then sum.
    biased_g = biased.reshape(T, int(n_group), E // int(n_group))
    top2_per_group, _ = biased_g.topk(min(2, biased_g.shape[-1]), dim=-1)
    group_score = top2_per_group.sum(dim=-1)  # [T, n_group]
    _, top_groups = group_score.topk(int(topk_group), dim=-1)
    # Build a mask for the selected groups.
    mask = torch.zeros(T, int(n_group), dtype=torch.bool, device=scores.device)
    mask.scatter_(1, top_groups, True)
    mask = mask.unsqueeze(-1).expand_as(biased_g).reshape(T, E)
    masked = torch.where(mask, biased, torch.full_like(biased, -float("inf")))
    top_vals, top_idx = masked.topk(int(topk), dim=-1)
    # Re-normalize using the (un-biased) sigmoid values at the selected idx.
    sig_at = sig.gather(1, top_idx)
    norm = sig_at / sig_at.sum(dim=-1, keepdim=True) * float(routed_scaling_factor)
    topk_values.copy_(norm.to(topk_values.dtype))
    topk_indices.copy_(top_idx.to(topk_indices.dtype))


fused_topk_deepseek_trace = TraceTemplate(
    op_type="moe_routing",
    name_prefix="fused_topk_deepseek",
    description=(
        "DeepSeek-V3 fused expert routing: sigmoid+bias → group score "
        "(sum of top-2) → top-k groups → top-k experts → normalize by "
        "sum of selected sigmoid scores * routed_scaling_factor. Outputs "
        "topk_values and topk_indices in-place."
    ),
    axes={
        "num_tokens": Var(),
        "num_experts": Const(abbrev="e"),
        "topk": Const(abbrev="k"),
    },
    inputs={
        "scores": Tensor(["num_tokens", "num_experts"]),
        "bias": Tensor(["num_experts"]),
        "n_group": Scalar("int32"),
        "topk_group": Scalar("int32"),
        "topk": Scalar("int32"),
        "routed_scaling_factor": Scalar("float32"),
        "topk_values": Tensor(["num_tokens", "topk"], description="In-place output."),
        "topk_indices": Tensor(
            ["num_tokens", "topk"], dtype="int32", description="In-place output."
        ),
    },
    outputs={
        "topk_values": Tensor(["num_tokens", "topk"], dtype_from="scores"),
        "topk_indices": Tensor(["num_tokens", "topk"], dtype="int32"),
    },
    tags=["status:verified", "moe"],
    reference=_fused_topk_deepseek_reference,
)


# ── Top-K + page-table transform (sparse attention helper) ───────────────────


@torch.no_grad()
def _top_k_page_table_transform_reference(
    input: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_to_batch=None,
    deterministic: bool = False,
    tie_break: int = 0,
    dsa_graph_safe: bool = False,
    row_starts=None,
    **_unused,
) -> torch.Tensor:
    """Reference for top_k_page_table_transform: per-row top-k selection on
    the leading ``lengths[i]`` valid entries, then translate the selected
    indices through ``src_page_table[row_to_batch[i]]``. Used in sparse
    attention's second stage to produce per-row page-id sequences.
    """
    num_rows = input.shape[0]
    out = torch.zeros(num_rows, int(k), dtype=torch.int32, device=input.device)
    for i in range(num_rows):
        L = int(lengths[i].item())
        if L <= 0:
            continue
        b = int(row_to_batch[i].item()) if row_to_batch is not None else i
        row = input[i, :L].to(torch.float32)
        kk = min(int(k), L)
        _, idx = torch.topk(row, kk, sorted=True)
        out[i, :kk] = src_page_table[b, idx.to(torch.long)].to(torch.int32)
    return out


top_k_page_table_transform_trace = TraceTemplate(
    op_type="sampling",
    name_prefix="top_k_page_table_transform",
    description=(
        "Fused per-row top-k selection plus page-table translation. For "
        "each row i: pick top-k indices over input[i, :lengths[i]] and "
        "translate them via src_page_table[row_to_batch[i]] to produce "
        "per-row page-id sequences for sparse attention."
    ),
    axes={
        "num_rows": Var(),
        "max_len": Var(),
        "batch_size": Var(),
        "max_pages_per_seq": Var(),
        "k": Const(abbrev="k"),
    },
    inputs={
        "input": Tensor(["num_rows", "max_len"]),
        "src_page_table": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
        ),
        "lengths": Tensor(["num_rows"], dtype="int32"),
        "k": Scalar("int32"),
        "row_to_batch": Tensor(["num_rows"], dtype="int32", optional=True),
    },
    outputs={
        "indices": Tensor(["num_rows", "k"], dtype="int32"),
    },
    tags=["status:verified", "sparse"],
    reference=_top_k_page_table_transform_reference,
)
