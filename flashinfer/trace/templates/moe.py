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

"""TraceTemplates for Mixture-of-Experts operations."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ---------------------------------------------------------------------------
# Shared GEMM computation helper
# ---------------------------------------------------------------------------

H = 7168
I = 2048
BLOCK = 128


@torch.no_grad()
def _fp8_moe_run_experts(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    weights,
    topk_idx,
    local_expert_offset,
    E_global,
):
    """FP8 block-scale dequantization + SwiGLU + GEMM for all routing types.

    ``weights``   : [T, TOP_K] float32 — per-token expert weights (already normalised)
    ``topk_idx``  : [T, TOP_K] int64   — selected global expert indices
    """
    T = hidden_states.shape[0]
    E_local = gemm1_weights.shape[0]
    device = hidden_states.device

    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H).contiguous()
    )
    A = A_fp32 * A_scale_expanded

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)
    W13 = W13_fp32 * S13_expanded

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)
    W2 = W2_fp32 * S2_expanded

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        # tokens that selected this expert
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)
        G1 = A_e.matmul(W13[le].t())
        X1, X2 = G1[:, :I], G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        O = (silu_X2 * X1).matmul(W2[le].t())
        # per-expert contribution weight for each token
        w_tok = weights.index_select(0, token_idx)
        # find which slot in topk_idx[token_idx] corresponds to ge
        match = (topk_idx.index_select(0, token_idx) == ge).float()
        w_e = (w_tok * match).sum(dim=1)
        output.index_add_(0, token_idx, O * w_e.unsqueeze(1))

    return output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Per-routing-type reference implementations
# ---------------------------------------------------------------------------


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_ds_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with DeepSeek-V3 routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalised and
        scaled by routed_scaling_factor
    """
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    )

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    raw_w = s * M
    weights_sum = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / weights_sum) * routed_scaling_factor

    # Gather per-row weights into [T, TOP_K] for the shared GEMM helper
    w_topk = weights.gather(1, topk_idx)

    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_default_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Default routing: Softmax → TopK.
    routing_bias is added to logits before softmax when provided.
    """
    TOP_K = 8
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    s = torch.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(s, k=TOP_K, dim=1, largest=True, sorted=False)
    weights = s.gather(1, topk_idx) * routed_scaling_factor
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        weights,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_renormalize_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Renormalize routing: TopK → Softmax.
    TopK is applied on raw logits; weights are then derived by softmax
    over the selected logits.
    """
    TOP_K = 8
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    _, topk_idx = torch.topk(logits, k=TOP_K, dim=1, largest=True, sorted=False)
    gathered = logits.gather(1, topk_idx)
    weights = torch.softmax(gathered, dim=-1) * routed_scaling_factor
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        weights,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_llama4_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Llama4 routing: Top1 → Sigmoid.
    Single expert selected per token; weight derived from sigmoid of its logit.
    """
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    topk_idx = logits.argmax(dim=-1, keepdim=True)  # [T, 1]
    top1_logit = logits.gather(1, topk_idx)
    weights = (1.0 / (1.0 + torch.exp(-top1_logit))) * routed_scaling_factor
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        weights,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_renormalize_naive_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with RenormalizeNaive routing: Softmax → TopK → Renormalize.
    Same as Default but the selected weights are re-normalised to sum to 1.
    """
    TOP_K = 8
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    s = torch.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(s, k=TOP_K, dim=1, largest=True, sorted=False)
    gathered = s.gather(1, topk_idx)
    weights = gathered / (gathered.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        weights,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_moe_topk_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with TopK-only routing: TopK, uniform weights.
    No softmax or sigmoid; all selected experts receive equal weight.
    """
    TOP_K = 8
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    _, topk_idx = torch.topk(logits, k=TOP_K, dim=1, largest=True, sorted=False)
    T = logits.shape[0]
    weights = torch.full(
        (T, TOP_K),
        routed_scaling_factor / TOP_K,
        dtype=torch.float32,
        device=logits.device,
    )
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        weights,
        topk_idx,
        local_expert_offset,
        E_global,
    )


# ---------------------------------------------------------------------------
# Template factory: shared axes/inputs/outputs for all routing types
# ---------------------------------------------------------------------------

_STANDARD_AXES = {
    "seq_len": Var(description="Sequence length (number of tokens)"),
    "num_experts": Const(description="Total number of experts.", abbrev=""),
    "top_k": Const(
        description="Number of experts to route to per token.", abbrev="topk"
    ),
    "num_local_experts": Const(description="Number of local experts.", abbrev="e"),
    "hidden_size": Const(description="Hidden dimension size.", abbrev="h"),
    "intermediate_size": Const(description="MoE intermediate layer size.", abbrev="i"),
    "gemm1_out_size": Const(
        description="Output size of the first GEMM (W13). Should be 2 * intermediate_size.",
        abbrev="",
    ),
    "num_hidden_blocks": Const(
        description="Number of quantized blocks along the hidden_size dimension (block_size=128).",
        abbrev="",
    ),
    "num_intermediate_blocks": Const(
        description="Number of quantized blocks along the intermediate_size dimension (block_size=128).",
        abbrev="",
    ),
    "num_gemm1_out_blocks": Const(
        description="Number of quantized blocks along the gemm1_out_size dimension (block_size=128).",
        abbrev="",
    ),
}

_STANDARD_INPUTS = {
    "routing_logits": Tensor(
        ["seq_len", "num_experts"],
        description="Routing logits for expert selection.",
    ),
    "routing_bias": Tensor(
        ["num_experts"],
        description="Bias added to logits before routing. Pass None for no bias.",
        optional=True,
    ),
    "hidden_states": Tensor(
        ["seq_len", "hidden_size"],
        description="Input hidden states tensor (FP8 quantized).",
    ),
    "hidden_states_scale": Tensor(
        ["num_hidden_blocks", "seq_len"],
        description="Block-wise scaling factors for hidden states.",
    ),
    "gemm1_weights": Tensor(
        ["num_local_experts", "gemm1_out_size", "hidden_size"],
        description="First GEMM weights for all local experts (gate and up projections).",
    ),
    "gemm1_weights_scale": Tensor(
        ["num_local_experts", "num_gemm1_out_blocks", "num_hidden_blocks"],
        description="Block-wise scaling factors for first GEMM weights.",
    ),
    "gemm2_weights": Tensor(
        ["num_local_experts", "hidden_size", "intermediate_size"],
        description="Second GEMM weights for all local experts (down projection).",
    ),
    "gemm2_weights_scale": Tensor(
        ["num_local_experts", "num_hidden_blocks", "num_intermediate_blocks"],
        description="Block-wise scaling factors for second GEMM weights.",
    ),
    "local_expert_offset": Scalar(
        "int32",
        description="Offset of local experts in global expert space.",
    ),
    "routed_scaling_factor": Scalar(
        "float32",
        description="Scaling factor applied to routing weights.",
    ),
}

_STANDARD_OUTPUTS = {
    "output": Tensor(
        ["seq_len", "hidden_size"],
        dtype="bfloat16",
        description="Final MoE output tensor.",
    ),
}

_STANDARD_TAGS = ["status:verified", "quantization:float8_e4m3fn"]


def _make_standard_moe_trace(name_prefix, description, reference):
    """Factory for standard (non-DS) routing templates (same inputs/axes)."""
    return TraceTemplate(
        op_type="moe",
        name_prefix=name_prefix,
        description=description,
        axes=dict(_STANDARD_AXES),
        inputs=dict(_STANDARD_INPUTS),
        outputs=dict(_STANDARD_OUTPUTS),
        tags=_STANDARD_TAGS,
        reference=reference,
    )


# ---------------------------------------------------------------------------
# Template instances — one per RoutingMethodType value
# ---------------------------------------------------------------------------

# RoutingMethodType.DeepSeekV3 = 2
# Uses additional n_group / topk_group axes and requires routing_bias.
trtllm_fp8_block_scale_moe_ds_routing_trace = TraceTemplate(
    op_type="moe",
    name_prefix="moe_fp8_block_scale_ds_routing",
    description="FP8 block scale MoE with DeepSeek-V3 routing. Includes grouped sigmoid routing and two grouped-GEMM.",
    axes={
        "seq_len": Var(description="Sequence length (number of tokens)"),
        "num_experts": Const(description="Total number of experts.", abbrev=""),
        "top_k": Const(
            description="Number of experts to route to per token.", abbrev="topk"
        ),
        "n_group": Const(
            description="Number of expert groups for group routing.", abbrev="ng"
        ),
        "topk_group": Const(
            description="Number of groups to select for top-k routing.", abbrev="kg"
        ),
        "num_local_experts": Const(description="Number of local experts.", abbrev="e"),
        "hidden_size": Const(description="Hidden dimension size.", abbrev="h"),
        "intermediate_size": Const(
            description="MoE intermediate layer size.", abbrev="i"
        ),
        "gemm1_out_size": Const(
            description="Output size of the first GEMM (W13). Should be 2 * intermediate_size.",
            abbrev="",
        ),
        "num_hidden_blocks": Const(
            description="Number of quantized blocks along the hidden_size dimension (block_size=128).",
            abbrev="",
        ),
        "num_intermediate_blocks": Const(
            description="Number of quantized blocks along the intermediate_size dimension (block_size=128).",
            abbrev="",
        ),
        "num_gemm1_out_blocks": Const(
            description="Number of quantized blocks along the gemm1_out_size dimension (block_size=128).",
            abbrev="",
        ),
    },
    inputs={
        "routing_logits": Tensor(
            ["seq_len", "num_experts"],
            description="Routing logits for expert selection.",
        ),
        "routing_bias": Tensor(
            ["num_experts"],
            description="Bias tensor for routing. Pass all zeros for no bias.",
        ),
        "hidden_states": Tensor(
            ["seq_len", "hidden_size"],
            description="Input hidden states tensor (FP8 quantized).",
        ),
        "hidden_states_scale": Tensor(
            ["num_hidden_blocks", "seq_len"],
            description="Block-wise scaling factors for hidden states.",
        ),
        "gemm1_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "hidden_size"],
            description="First GEMM weights for all local experts (gate and up projections).",
        ),
        "gemm1_weights_scale": Tensor(
            ["num_local_experts", "num_gemm1_out_blocks", "num_hidden_blocks"],
            description="Block-wise scaling factors for first GEMM weights.",
        ),
        "gemm2_weights": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            description="Second GEMM weights for all local experts (down projection).",
        ),
        "gemm2_weights_scale": Tensor(
            ["num_local_experts", "num_hidden_blocks", "num_intermediate_blocks"],
            description="Block-wise scaling factors for second GEMM weights.",
        ),
        "local_expert_offset": Scalar(
            "int32",
            description="Offset of local experts in global expert space.",
        ),
        "routed_scaling_factor": Scalar(
            "float32",
            description="Scaling factor for routing weights.",
        ),
    },
    outputs={
        "output": Tensor(
            ["seq_len", "hidden_size"],
            dtype="bfloat16",
            description="Final MoE output tensor.",
        ),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_trtllm_fp8_block_scale_moe_ds_routing_reference,
)

# Backward-compatible alias (the original name used in fused_moe/core.py import).
trtllm_fp8_block_scale_moe_trace = trtllm_fp8_block_scale_moe_ds_routing_trace

# RoutingMethodType.Default = 0 — Softmax → TopK
trtllm_fp8_block_scale_moe_default_routing_trace = _make_standard_moe_trace(
    name_prefix="moe_fp8_block_scale_default_routing",
    description="FP8 block scale MoE with Default routing (Softmax → TopK).",
    reference=_trtllm_fp8_block_scale_moe_default_routing_reference,
)

# RoutingMethodType.Renormalize = 1 — TopK → Softmax
trtllm_fp8_block_scale_moe_renormalize_routing_trace = _make_standard_moe_trace(
    name_prefix="moe_fp8_block_scale_renormalize_routing",
    description="FP8 block scale MoE with Renormalize routing (TopK → Softmax).",
    reference=_trtllm_fp8_block_scale_moe_renormalize_routing_reference,
)

# RoutingMethodType.Llama4 = 3 — Top1 → Sigmoid
trtllm_fp8_block_scale_moe_llama4_routing_trace = _make_standard_moe_trace(
    name_prefix="moe_fp8_block_scale_llama4_routing",
    description="FP8 block scale MoE with Llama4 routing (Top1 → Sigmoid).",
    reference=_trtllm_fp8_block_scale_moe_llama4_routing_reference,
)

# RoutingMethodType.RenormalizeNaive = 4 — Softmax → TopK → Renormalize
trtllm_fp8_block_scale_moe_renormalize_naive_routing_trace = _make_standard_moe_trace(
    name_prefix="moe_fp8_block_scale_renormalize_naive_routing",
    description="FP8 block scale MoE with RenormalizeNaive routing (Softmax → TopK → Renormalize).",
    reference=_trtllm_fp8_block_scale_moe_renormalize_naive_routing_reference,
)

# RoutingMethodType.TopK = 5 — TopK only (no softmax), uniform weights
trtllm_fp8_block_scale_moe_topk_routing_trace = _make_standard_moe_trace(
    name_prefix="moe_fp8_block_scale_topk_routing",
    description="FP8 block scale MoE with TopK-only routing (no softmax, uniform weights).",
    reference=_trtllm_fp8_block_scale_moe_topk_routing_reference,
)

# ---------------------------------------------------------------------------
# Dispatch function — maps routing_method_type → TraceTemplate
# ---------------------------------------------------------------------------

_MOE_TRACE_BY_ROUTING_TYPE = {
    0: trtllm_fp8_block_scale_moe_default_routing_trace,  # Default
    1: trtllm_fp8_block_scale_moe_renormalize_routing_trace,  # Renormalize
    2: trtllm_fp8_block_scale_moe_ds_routing_trace,  # DeepSeekV3
    3: trtllm_fp8_block_scale_moe_llama4_routing_trace,  # Llama4
    4: trtllm_fp8_block_scale_moe_renormalize_naive_routing_trace,  # RenormalizeNaive
    5: trtllm_fp8_block_scale_moe_topk_routing_trace,  # TopK
    # 6 = Unspecified: no trace
}


def trtllm_fp8_block_scale_moe_trace_dispatch(**kwargs):
    """Return the appropriate TraceTemplate for the given ``routing_method_type``.

    Pass this as ``trace=trtllm_fp8_block_scale_moe_trace_dispatch`` to
    ``@flashinfer_api`` so the correct template is selected at call time::

        @flashinfer_api(trace=trtllm_fp8_block_scale_moe_trace_dispatch)
        def trtllm_fp8_block_scale_moe(..., routing_method_type: int = 0, ...):
            ...

    Returns ``None`` for ``RoutingMethodType.Unspecified`` (6), which
    suppresses trace generation.
    """
    routing_method_type = int(kwargs.get("routing_method_type", 0))
    return _MOE_TRACE_BY_ROUTING_TYPE.get(routing_method_type)


# Expose all possible templates so _attach_fi_trace can auto-register them
# in _TRACE_REGISTRY for consistency testing.
trtllm_fp8_block_scale_moe_trace_dispatch.templates = list(  # type: ignore[attr-defined]
    _MOE_TRACE_BY_ROUTING_TYPE.values()
)


# ---------------------------------------------------------------------------
# FP4 block-scale MoE (trtllm_fp4_block_scale_moe)
# ---------------------------------------------------------------------------
# NvFP4: block_size=16, weights packed as uint8 (2 fp4 per byte).
#   hidden_states       : [seq_len, hidden_size // 2]   uint8
#   hidden_states_scale : [seq_len, hidden_size // 16]  float8  (optional for bf16 input)
#   gemm1_weights       : [E_loc, 2*I, hidden_size // 2]         uint8
#   gemm1_weights_scale : [E_loc, 2*I, hidden_size // 16]        float8
#   gemm2_weights       : [E_loc, hidden_size, I // 2]            uint8
#   gemm2_weights_scale : [E_loc, hidden_size, I // 16]           float8
# ---------------------------------------------------------------------------

_FP4_STANDARD_AXES: dict[str, Var | Const] = {
    "seq_len": Var(description="Number of tokens."),
    "num_experts": Const(description="Total number of experts.", abbrev=""),
    "top_k": Const(description="Number of experts selected per token.", abbrev="topk"),
    "num_local_experts": Const(description="Number of local experts.", abbrev="e"),
    "hidden_size": Const(description="Hidden dimension size.", abbrev="h"),
    "intermediate_size": Const(description="MoE intermediate layer size.", abbrev="i"),
    # Derived / block-count axes (abbrev="" → omitted from filename)
    "gemm1_out_size": Const(
        description="Output size of FC1 (2 × intermediate_size for SwiGLU).",
        abbrev="",
    ),
    "num_packed_hidden": Const(
        description="Packed hidden dimension (hidden_size // 2 for NvFP4).",
        abbrev="",
    ),
    "num_fp4_hidden_blocks": Const(
        description="Number of FP4 scale blocks along hidden_size (hidden_size // 16 for NvFP4).",
        abbrev="",
    ),
    "num_packed_intermediate": Const(
        description="Packed intermediate dimension (intermediate_size // 2 for NvFP4).",
        abbrev="",
    ),
    "num_fp4_intermediate_blocks": Const(
        description="Number of FP4 scale blocks along intermediate_size (intermediate_size // 16 for NvFP4).",
        abbrev="",
    ),
}

_FP4_STANDARD_INPUTS: dict[str, Tensor | Scalar] = {
    "routing_logits": Tensor(
        ["seq_len", "num_experts"],
        description="Routing logits for expert selection.",
    ),
    "routing_bias": Tensor(
        ["num_experts"],
        description="Bias added to routing logits. Pass None when not used.",
        optional=True,
    ),
    # Packed NvFP4 hidden states (2 values per uint8 byte).
    "hidden_states": Tensor(
        ["seq_len", "num_packed_hidden"],
        description="Input hidden states, NvFP4-packed (uint8, 2 fp4 per byte).",
    ),
    "hidden_states_scale": Tensor(
        ["seq_len", "num_fp4_hidden_blocks"],
        description="Block-wise scale factors for hidden_states (float8). None for bf16 input.",
        optional=True,
    ),
    "gemm1_weights": Tensor(
        ["num_local_experts", "gemm1_out_size", "num_packed_hidden"],
        description="FC1 weights, NvFP4-packed (uint8). Shape includes gate+up for SwiGLU.",
    ),
    "gemm1_weights_scale": Tensor(
        ["num_local_experts", "gemm1_out_size", "num_fp4_hidden_blocks"],
        description="Block-wise scale factors for gemm1_weights (float8).",
    ),
    "gemm1_bias": Tensor(
        ["num_local_experts", "gemm1_out_size"],
        description="FC1 bias (float32). Optional.",
        optional=True,
    ),
    "gemm1_alpha": Tensor(
        ["num_local_experts"],
        description="Per-expert SwiGLU alpha (float32). Optional.",
        optional=True,
    ),
    "gemm1_beta": Tensor(
        ["num_local_experts"],
        description="Per-expert SwiGLU beta (float32). Optional.",
        optional=True,
    ),
    "gemm1_clamp_limit": Tensor(
        ["num_local_experts"],
        description="Per-expert SwiGLU clamp limit (float32). Optional.",
        optional=True,
    ),
    "gemm2_weights": Tensor(
        ["num_local_experts", "hidden_size", "num_packed_intermediate"],
        description="FC2 weights, NvFP4-packed (uint8).",
    ),
    "gemm2_weights_scale": Tensor(
        ["num_local_experts", "hidden_size", "num_fp4_intermediate_blocks"],
        description="Block-wise scale factors for gemm2_weights (float8).",
    ),
    "gemm2_bias": Tensor(
        ["num_local_experts", "hidden_size"],
        description="FC2 bias (float32). Optional.",
        optional=True,
    ),
    "output1_scale_scalar": Tensor(
        ["num_local_experts"],
        description="Per-expert output scale for FC1 activation (float32). Optional.",
        optional=True,
    ),
    "output1_scale_gate_scalar": Tensor(
        ["num_local_experts"],
        description="Per-expert output scale for FC1 gate (float32). Optional.",
        optional=True,
    ),
    "output2_scale_scalar": Tensor(
        ["num_local_experts"],
        description="Per-expert output scale for FC2 (float32). Optional.",
        optional=True,
    ),
    "local_expert_offset": Scalar(
        "int32",
        description="Offset of local experts in the global expert array.",
    ),
    "routed_scaling_factor": Scalar(
        "float32",
        optional=True,
        description="Scaling factor applied to routing weights. None for some routing methods.",
    ),
}

_FP4_STANDARD_OUTPUTS = {
    "output": Tensor(
        ["seq_len", "hidden_size"],
        dtype="bfloat16",
        description="Final MoE output tensor.",
    ),
}

_FP4_STANDARD_TAGS = ["status:experimental", "quantization:nvfp4"]


def _make_standard_fp4_moe_trace(name_prefix, description):
    """Factory for FP4 MoE templates that share the standard (non-DS) axis set."""
    return TraceTemplate(
        op_type="moe",
        name_prefix=name_prefix,
        description=description,
        axes=dict(_FP4_STANDARD_AXES),
        inputs=dict(_FP4_STANDARD_INPUTS),
        outputs=dict(_FP4_STANDARD_OUTPUTS),
        tags=_FP4_STANDARD_TAGS,
        reference=None,
    )


# RoutingMethodType.Default = 0 — Softmax → TopK
trtllm_fp4_block_scale_moe_default_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_default_routing",
    description="NvFP4 block-scale MoE with Default routing (Softmax → TopK).",
)

# RoutingMethodType.Renormalize = 1 — TopK → Softmax
trtllm_fp4_block_scale_moe_renormalize_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_renormalize_routing",
    description="NvFP4 block-scale MoE with Renormalize routing (TopK → Softmax).",
)

# RoutingMethodType.DeepSeekV3 = 2 — Sigmoid → group selection → TopK
trtllm_fp4_block_scale_moe_ds_routing_trace = TraceTemplate(
    op_type="moe",
    name_prefix="moe_fp4_block_scale_ds_routing",
    description="NvFP4 block-scale MoE with DeepSeekV3 routing (Sigmoid → group selection → top_k).",
    axes={
        **_FP4_STANDARD_AXES,
        "n_group": Const(
            description="Number of expert groups for group routing.", abbrev="ng"
        ),
        "topk_group": Const(
            description="Number of groups selected in top-k routing.", abbrev="kg"
        ),
    },
    inputs=dict(_FP4_STANDARD_INPUTS),
    outputs=dict(_FP4_STANDARD_OUTPUTS),
    tags=_FP4_STANDARD_TAGS,
    reference=None,
)

# RoutingMethodType.Llama4 = 3 — Top1 → Sigmoid
trtllm_fp4_block_scale_moe_llama4_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_llama4_routing",
    description="NvFP4 block-scale MoE with Llama4 routing (Top1 → Sigmoid).",
)

# RoutingMethodType.RenormalizeNaive = 4 — Softmax → TopK → Renormalize
trtllm_fp4_block_scale_moe_renormalize_naive_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_renormalize_naive_routing",
    description="NvFP4 block-scale MoE with RenormalizeNaive routing (Softmax → TopK → Renormalize).",
)

# RoutingMethodType.TopK = 5 — plain TopK, uniform weights
trtllm_fp4_block_scale_moe_topk_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_topk_routing",
    description="NvFP4 block-scale MoE with TopK-only routing (no softmax, uniform weights).",
)

_FP4_MOE_TRACE_BY_ROUTING_TYPE = {
    0: trtllm_fp4_block_scale_moe_default_routing_trace,
    1: trtllm_fp4_block_scale_moe_renormalize_routing_trace,
    2: trtllm_fp4_block_scale_moe_ds_routing_trace,
    3: trtllm_fp4_block_scale_moe_llama4_routing_trace,
    4: trtllm_fp4_block_scale_moe_renormalize_naive_routing_trace,
    5: trtllm_fp4_block_scale_moe_topk_routing_trace,
    # 6 = Unspecified: no trace
}


def trtllm_fp4_block_scale_moe_trace_dispatch(**kwargs):
    """Return the FP4 TraceTemplate for the given ``routing_method_type``.

    Pass this as ``trace=trtllm_fp4_block_scale_moe_trace_dispatch`` to
    ``@flashinfer_api`` so the correct template is selected at call time::

        @flashinfer_api(trace=trtllm_fp4_block_scale_moe_trace_dispatch)
        def trtllm_fp4_block_scale_moe(..., routing_method_type: int = 0, ...):
            ...

    Returns ``None`` for ``RoutingMethodType.Unspecified`` (6).
    """
    routing_method_type = int(kwargs.get("routing_method_type", 0))
    return _FP4_MOE_TRACE_BY_ROUTING_TYPE.get(routing_method_type)


trtllm_fp4_block_scale_moe_trace_dispatch.templates = list(  # type: ignore[attr-defined]
    _FP4_MOE_TRACE_BY_ROUTING_TYPE.values()
)
