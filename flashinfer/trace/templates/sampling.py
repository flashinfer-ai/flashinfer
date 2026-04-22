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
