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

from ..template import Const, Tensor, TraceTemplate, Var

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
