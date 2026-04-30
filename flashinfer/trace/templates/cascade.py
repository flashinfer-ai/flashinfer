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

"""TraceTemplates for cascade/merge attention state operations."""

import math

import torch

from ..template import Const, Tensor, TraceTemplate, Var

# ── Merge State ───────────────────────────────────────────────────────────────


@torch.no_grad()
def _merge_state_reference(v_a, s_a, v_b, s_b):
    """Merge two attention (V, S) states via numerically stable log-sum-exp."""
    # s_a, s_b are log2-scale logsumexp values; convert to natural scale
    s_a = s_a.to(torch.float32) * math.log(2.0)
    s_b = s_b.to(torch.float32) * math.log(2.0)
    v_a = v_a.to(torch.float32)
    v_b = v_b.to(torch.float32)
    s_max = torch.maximum(s_a, s_b)
    exp_a = torch.exp(s_a - s_max)
    exp_b = torch.exp(s_b - s_max)
    exp_sum = exp_a + exp_b
    v_merged = (
        v_a * exp_a.unsqueeze(-1) + v_b * exp_b.unsqueeze(-1)
    ) / exp_sum.unsqueeze(-1)
    s_merged = (s_max + torch.log(exp_sum)) / math.log(2.0)
    return v_merged.to(v_a.dtype), s_merged.to(torch.float32)


merge_state_trace = TraceTemplate(
    op_type="cascade_merge",
    name_prefix="merge_state",
    description="Merge two attention (V, S) states for cascade/speculative attention.",
    axes={
        "seq_len": Var(description="Number of query tokens."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "v_a": Tensor(
            ["seq_len", "num_heads", "head_dim"],
            description="Attention output from KV segment A.",
        ),
        "s_a": Tensor(
            ["seq_len", "num_heads"],
            dtype="float32",
            description="Logsumexp (base-2) from KV segment A.",
        ),
        "v_b": Tensor(
            ["seq_len", "num_heads", "head_dim"],
            description="Attention output from KV segment B.",
        ),
        "s_b": Tensor(
            ["seq_len", "num_heads"],
            dtype="float32",
            description="Logsumexp (base-2) from KV segment B.",
        ),
    },
    outputs={
        "v_merged": Tensor(["seq_len", "num_heads", "head_dim"], dtype_from="v_a"),
        "s_merged": Tensor(["seq_len", "num_heads"], dtype="float32"),
    },
    tags=["status:verified"],
    reference=_merge_state_reference,
)

# ── Merge State In-Place ──────────────────────────────────────────────────────


@torch.no_grad()
def _merge_state_in_place_reference(v, s, v_other, s_other, mask=None):
    """In-place LSE-weighted merge of (v, s) with (v_other, s_other).

    When ``mask`` is provided, only rows where mask is True are merged;
    other rows are returned unchanged. Scales are base-2 logsumexp as in
    ``_merge_state_reference``.
    """
    s_a = s.to(torch.float32) * math.log(2.0)
    s_b = s_other.to(torch.float32) * math.log(2.0)
    v_a = v.to(torch.float32)
    v_b = v_other.to(torch.float32)
    s_max = torch.maximum(s_a, s_b)
    exp_a = torch.exp(s_a - s_max)
    exp_b = torch.exp(s_b - s_max)
    exp_sum = exp_a + exp_b
    v_merged = (
        v_a * exp_a.unsqueeze(-1) + v_b * exp_b.unsqueeze(-1)
    ) / exp_sum.unsqueeze(-1)
    s_merged = (s_max + torch.log(exp_sum)) / math.log(2.0)
    if mask is not None:
        m = mask.to(torch.bool)
        v_merged = torch.where(m[:, None, None], v_merged, v_a)
        s_merged = torch.where(m[:, None], s_merged, s.to(torch.float32))
    return v_merged.to(v.dtype), s_merged.to(torch.float32)


merge_state_in_place_trace = TraceTemplate(
    op_type="cascade_merge",
    name_prefix="merge_state_in_place",
    description="Merge attention (V, S) states in-place. v and s are updated with merged result.",
    axes={
        "seq_len": Var(description="Number of query tokens."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "v": Tensor(
            ["seq_len", "num_heads", "head_dim"],
            description="Attention output (updated in-place with merged result).",
        ),
        "s": Tensor(
            ["seq_len", "num_heads"],
            dtype="float32",
            description="Logsumexp (base-2) (updated in-place).",
        ),
        "v_other": Tensor(
            ["seq_len", "num_heads", "head_dim"],
            description="Other attention output to merge in.",
        ),
        "s_other": Tensor(
            ["seq_len", "num_heads"],
            dtype="float32",
            description="Other logsumexp (base-2) to merge in.",
        ),
        "mask": Tensor(
            ["seq_len"],
            optional=True,
            description="Boolean mask; if set, only merge where mask is True.",
        ),
    },
    outputs={
        "v": Tensor(
            ["seq_len", "num_heads", "head_dim"],
            dtype_from="v",
            description="Updated v (in-place).",
        ),
        "s": Tensor(
            ["seq_len", "num_heads"],
            dtype="float32",
            description="Updated s (in-place).",
        ),
    },
    tags=["status:verified"],
    reference=_merge_state_in_place_reference,
)

# ── Merge States ──────────────────────────────────────────────────────────────


@torch.no_grad()
def _merge_states_reference(v, s):
    """Merge num_states attention (V, S) states via numerically stable log-sum-exp."""
    # v: [seq_len, num_states, num_heads, head_dim]
    # s: [seq_len, num_states, num_heads]  (log2 scale)
    s_nat = s.to(torch.float32) * math.log(2.0)
    v_f32 = v.to(torch.float32)
    s_max, _ = s_nat.max(dim=1, keepdim=True)
    exp_s = torch.exp(s_nat - s_max)  # [seq_len, num_states, num_heads]
    exp_sum = exp_s.sum(dim=1, keepdim=True)
    weights = exp_s / exp_sum  # [seq_len, num_states, num_heads]
    v_merged = (v_f32 * weights.unsqueeze(-1)).sum(dim=1)
    s_merged = (s_max.squeeze(1) + torch.log(exp_sum.squeeze(1))) / math.log(2.0)
    return v_merged.to(v.dtype), s_merged.to(torch.float32)


merge_states_trace = TraceTemplate(
    op_type="cascade_merge",
    name_prefix="merge_states",
    description="Merge multiple (num_states) attention (V, S) states.",
    axes={
        "seq_len": Var(description="Number of query tokens."),
        "num_states": Var(description="Number of KV segments to merge."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "v": Tensor(
            ["seq_len", "num_states", "num_heads", "head_dim"],
            description="Attention outputs from all KV segments.",
        ),
        "s": Tensor(
            ["seq_len", "num_states", "num_heads"],
            dtype="float32",
            description="Logsumexp (base-2) values from all KV segments.",
        ),
    },
    outputs={
        "v_merged": Tensor(["seq_len", "num_heads", "head_dim"], dtype_from="v"),
        "s_merged": Tensor(["seq_len", "num_heads"], dtype="float32"),
    },
    tags=["status:verified"],
    reference=_merge_states_reference,
)
