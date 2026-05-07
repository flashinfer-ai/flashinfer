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
    T, H = hidden_states.shape
    E_local, gemm1_out_size, _ = gemm1_weights.shape
    I = gemm1_out_size // 2
    BLOCK = 128
    if gemm1_out_size != 2 * I:
        raise ValueError(
            f"gemm1_weights.shape[1]={gemm1_out_size} is not 2*intermediate_size; "
            "SwiGLU requires gemm1_out_size == 2 * intermediate_size."
        )
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
    top_k,
    n_group,
    topk_group,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with DeepSeek-V3 routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group; per group take top-2 sum → pick topk_group groups
        on the kept groups, take global top_k experts
        combine with weights derived from s (without bias), normalised and
        scaled by routed_scaling_factor
    """
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = int(top_k)
    N_GROUP = int(n_group)
    TOPK_GROUP = int(topk_group)

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
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Default routing: Softmax → TopK.
    routing_bias is added to logits before softmax when provided.
    """
    TOP_K = int(top_k)
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
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Renormalize routing: TopK → Softmax.
    TopK is applied on raw logits; weights are then derived by softmax
    over the selected logits.
    """
    TOP_K = int(top_k)
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
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with Llama4 routing: Top1 → Sigmoid.
    Single expert selected per token; weight derived from sigmoid of its logit.
    By definition Llama4 routing uses top_k=1; the parameter is accepted for
    schema consistency with the other routing methods.
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
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with RenormalizeNaive routing: Softmax → TopK → Renormalize.
    Same as Default but the selected weights are re-normalised to sum to 1.
    """
    TOP_K = int(top_k)
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
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    FP8 block-scale MoE with TopK-only routing: TopK, uniform weights.
    No softmax or sigmoid; all selected experts receive equal weight.
    """
    TOP_K = int(top_k)
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
    "top_k": Scalar(
        "int32",
        description="Number of experts to route to per token.",
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
        "top_k": Scalar(
            "int32",
            description="Number of experts to route to per token (DeepSeek-V3 uses 8).",
        ),
        "n_group": Scalar(
            "int32",
            description="Number of expert groups (DeepSeek-V3 uses 8).",
        ),
        "topk_group": Scalar(
            "int32",
            description="Number of groups to keep after group-level top-k (DeepSeek-V3 uses 4).",
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


# FP4 e2m1fn magnitudes. The 4-bit code is {sign(1), exponent(2), mantissa(1)};
# this table maps the 16 possible nibble values to the corresponding float32
# magnitude so dequantization is a single gather.
_E2M1_LUT_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


@torch.no_grad()
def _unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a uint8 tensor of packed e2m1fn FP4 values into float32.

    Each byte stores two 4-bit values (low nibble = first element along the
    last axis). The returned tensor has twice the last-dim size of *packed*.
    """
    lut = torch.tensor(_E2M1_LUT_VALUES, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8).to(torch.int64)
    lo = lut[p & 0x0F]
    hi = lut[(p >> 4) & 0x0F]
    stacked = torch.stack([lo, hi], dim=-1)  # pairs along a new last axis
    return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@torch.no_grad()
def _ue8m0_to_float32(scales: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 (uint8, unsigned exponent-only) scales to float32."""
    e = scales.view(torch.uint8).to(torch.int64)
    return torch.pow(torch.tensor(2.0, device=scales.device), (e - 127).float())


@torch.no_grad()
def _decode_block_scales(scales: torch.Tensor, is_ue8m0: bool) -> torch.Tensor:
    """Decode block scales: UE8M0 for MX formats, float8_e4m3fn otherwise."""
    if is_ue8m0:
        return _ue8m0_to_float32(scales)
    # fp8_e4m3fn (or already float): plain cast.
    return scales.to(torch.float32)


@torch.no_grad()
def _dequantize_fp4_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    is_ue8m0_scales: bool,
) -> torch.Tensor:
    """Unpack an FP4 tensor and apply its per-block scales along the last dim.

    The packed tensor has half the logical last-dim size of the output; the
    scale tensor has last-dim size = (output last dim) / block_size.
    block_size is inferred from the shape ratio.
    """
    unpacked = _unpack_fp4_e2m1(packed)  # float32, last dim = packed.last * 2
    block_size = unpacked.shape[-1] // scales.shape[-1]
    decoded_scales = _decode_block_scales(scales, is_ue8m0_scales)
    expanded = decoded_scales.repeat_interleave(block_size, dim=-1)
    return unpacked * expanded


@torch.no_grad()
def _dequantize_fp4_hidden_states(
    hidden_states: torch.Tensor,
    hidden_states_scale,
    is_weights_mxfp4: bool,
) -> torch.Tensor:
    """Dequantize hidden_states to float32.

    Three cases by dtype:
      * bfloat16 — pass-through (no scale).
      * float8_e4m3fn — MXFP8 activation with UE8M0 per-32 scales.
      * uint8 — NvFP4/MXFP4 packed activation with per-block scales (fp8_e4m3fn
        for NvFP4, UE8M0 for MXFP4; here both are treated as fp8_e4m3fn since
        the runtime FP4 path uses fp8_e4m3fn scales for activations).
    """
    if hidden_states.dtype == torch.bfloat16:
        return hidden_states.to(torch.float32)
    if hidden_states.dtype == torch.float8_e4m3fn:
        # MXFP8 hidden states: UE8M0 scales, block size 32.
        scales = _ue8m0_to_float32(hidden_states_scale)
        block_size = hidden_states.shape[-1] // scales.shape[-1]
        expanded = scales.repeat_interleave(block_size, dim=-1)
        return hidden_states.to(torch.float32) * expanded
    # uint8-packed FP4. For NvFP4 activation + NvFP4 weights the scales are
    # fp8_e4m3fn; for MXFP4 weights (and bf16-packed-as-fp4 corner cases) they
    # are UE8M0. Use the weight mode as the tiebreaker since activation scale
    # format tracks weight format in the trtllm-gen kernel.
    return _dequantize_fp4_tensor(
        hidden_states, hidden_states_scale, is_ue8m0_scales=is_weights_mxfp4
    )


@torch.no_grad()
def _fp4_moe_run_experts(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    gemm1_bias,
    gemm2_bias,
    weights,
    topk_idx,
    local_expert_offset,
    E_global,
):
    """FP4 dequantize + SwiGLU + GEMM for all routing types.

    ``weights``   : [T, TOP_K] float32 — per-token expert weights (normalised)
    ``topk_idx``  : [T, TOP_K] int64   — selected global expert indices

    Detects MXFP4 vs NvFP4 weight format from whether gemm1_weights_scale is
    fp8_e4m3fn (NvFP4) or uint8 (UE8M0, MXFP4). Block size is inferred from
    the ratio of unpacked K to scale K.
    """
    is_mxfp4 = gemm1_weights_scale.dtype == torch.uint8
    device = gemm1_weights.device

    # Dequantize both expert-weight tensors in one shot.
    W1 = _dequantize_fp4_tensor(
        gemm1_weights, gemm1_weights_scale, is_ue8m0_scales=is_mxfp4
    )  # [E_local, 2*I, H]
    W2 = _dequantize_fp4_tensor(
        gemm2_weights, gemm2_weights_scale, is_ue8m0_scales=is_mxfp4
    )  # [E_local, H, I]

    E_local, gemm1_out_size, H = W1.shape
    I = gemm1_out_size // 2
    if gemm1_out_size != 2 * I:
        raise ValueError(
            f"gemm1 output size {gemm1_out_size} is not 2*intermediate_size; "
            "FP4 MoE requires SwiGLU (gate + up)."
        )

    A = _dequantize_fp4_hidden_states(hidden_states, hidden_states_scale, is_mxfp4)
    T = A.shape[0]
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)  # [N, H]
        G1 = A_e.matmul(W1[le].t())  # [N, 2*I]
        if gemm1_bias is not None:
            G1 = G1 + gemm1_bias[le].to(torch.float32)
        # SwiGLU uses the trtllm-gen convention: silu(X2) * X1 with X1 first.
        X1, X2 = G1[:, :I], G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        activated = silu_X2 * X1
        O = activated.matmul(W2[le].t())  # [N, H]
        if gemm2_bias is not None:
            O = O + gemm2_bias[le].to(torch.float32)
        # Fold per-token expert weight.
        w_tok = weights.index_select(0, token_idx)
        match = (topk_idx.index_select(0, token_idx) == ge).float()
        w_e = (w_tok * match).sum(dim=1)
        output.index_add_(0, token_idx, O * w_e.unsqueeze(1))

    return output.to(torch.bfloat16)


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_default_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with Default routing (Softmax → TopK)."""
    TOP_K = int(top_k)
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    s = torch.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(s, k=TOP_K, dim=1, largest=True, sorted=False)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    w_topk = s.gather(1, topk_idx) * scale
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_renormalize_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with Renormalize routing (TopK on logits → Softmax)."""
    TOP_K = int(top_k)
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    _, topk_idx = torch.topk(logits, k=TOP_K, dim=1, largest=True, sorted=False)
    gathered = logits.gather(1, topk_idx)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    w_topk = torch.softmax(gathered, dim=-1) * scale
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_ds_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    n_group,
    topk_group,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with DeepSeek-V3 routing: sigmoid + groups + top_k."""
    TOP_K = int(top_k)
    N_GROUP = int(n_group)
    TOPK_GROUP = int(topk_group)
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

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
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    full_weights = (raw_w / weights_sum) * scale
    w_topk = full_weights.gather(1, topk_idx)

    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_llama4_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with Llama4 routing (Top1 → Sigmoid). top_k is fixed at 1."""
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    topk_idx = logits.argmax(dim=-1, keepdim=True)
    top1_logit = logits.gather(1, topk_idx)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    w_topk = (1.0 / (1.0 + torch.exp(-top1_logit))) * scale
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_renormalize_naive_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with RenormalizeNaive routing (Softmax → TopK → sum-to-1)."""
    TOP_K = int(top_k)
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    s = torch.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(s, k=TOP_K, dim=1, largest=True, sorted=False)
    gathered = s.gather(1, topk_idx)
    w_topk = gathered / (gathered.sum(dim=1, keepdim=True) + 1e-20)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    w_topk = w_topk * scale
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_moe_topk_routing_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    top_k,
    local_expert_offset,
    routed_scaling_factor,
):
    """FP4 MoE with TopK-only routing (uniform weights)."""
    TOP_K = int(top_k)
    E_global = routing_logits.shape[1]
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    _, topk_idx = torch.topk(logits, k=TOP_K, dim=1, largest=True, sorted=False)
    T = logits.shape[0]
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    w_topk = torch.full(
        (T, TOP_K), scale / TOP_K, dtype=torch.float32, device=logits.device
    )
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_idx,
        local_expert_offset,
        E_global,
    )


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


def _make_standard_fp4_moe_trace(name_prefix, description, reference=None):
    """Factory for FP4 MoE templates that share the standard (non-DS) axis set."""
    return TraceTemplate(
        op_type="moe",
        name_prefix=name_prefix,
        description=description,
        axes=dict(_FP4_STANDARD_AXES),
        inputs=dict(_FP4_STANDARD_INPUTS),
        outputs=dict(_FP4_STANDARD_OUTPUTS),
        tags=_FP4_STANDARD_TAGS,
        reference=reference,
    )


# RoutingMethodType.Default = 0 — Softmax → TopK
trtllm_fp4_block_scale_moe_default_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_default_routing",
    description="NvFP4 block-scale MoE with Default routing (Softmax → TopK).",
    reference=_trtllm_fp4_block_scale_moe_default_routing_reference,
)

# RoutingMethodType.Renormalize = 1 — TopK → Softmax
trtllm_fp4_block_scale_moe_renormalize_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_renormalize_routing",
    description="NvFP4 block-scale MoE with Renormalize routing (TopK → Softmax).",
    reference=_trtllm_fp4_block_scale_moe_renormalize_routing_reference,
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
    reference=_trtllm_fp4_block_scale_moe_ds_routing_reference,
)

# RoutingMethodType.Llama4 = 3 — Top1 → Sigmoid
trtllm_fp4_block_scale_moe_llama4_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_llama4_routing",
    description="NvFP4 block-scale MoE with Llama4 routing (Top1 → Sigmoid).",
    reference=_trtllm_fp4_block_scale_moe_llama4_routing_reference,
)

# RoutingMethodType.RenormalizeNaive = 4 — Softmax → TopK → Renormalize
trtllm_fp4_block_scale_moe_renormalize_naive_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_renormalize_naive_routing",
    description="NvFP4 block-scale MoE with RenormalizeNaive routing (Softmax → TopK → Renormalize).",
    reference=_trtllm_fp4_block_scale_moe_renormalize_naive_routing_reference,
)

# RoutingMethodType.TopK = 5 — plain TopK, uniform weights
trtllm_fp4_block_scale_moe_topk_routing_trace = _make_standard_fp4_moe_trace(
    name_prefix="moe_fp4_block_scale_topk_routing",
    description="NvFP4 block-scale MoE with TopK-only routing (no softmax, uniform weights).",
    reference=_trtllm_fp4_block_scale_moe_topk_routing_reference,
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


# ---------------------------------------------------------------------------
# Additional MoE variants (CUTLASS fused MoE, bf16, routed, per-tensor, mxint4)
# ---------------------------------------------------------------------------

_MOE_COMMON_AXES: dict[str, Var | Const] = {
    "seq_len": Var(description="Number of input tokens."),
    "num_experts": Const(abbrev="", description="Total number of experts."),
    "top_k": Const(abbrev="topk"),
    "num_local_experts": Const(abbrev="e", description="Number of local experts."),
    "hidden_size": Const(abbrev="h"),
    "intermediate_size": Const(abbrev="i"),
}

# ---------------------------------------------------------------------------
# References for the additional MoE variants (bf16 / per-tensor FP8 / routed /
# mxint4). Each reference assumes inputs are already in their declared dtypes.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _moe_bf16_run_experts(
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    weights,
    topk_idx,
    local_expert_offset,
    E_global,
):
    """Un-quantized (bf16) MoE expert computation with SwiGLU."""
    T, H = hidden_states.shape
    E_local, gemm1_out, _ = gemm1_weights.shape
    I = gemm1_out // 2
    device = hidden_states.device
    A = hidden_states.to(torch.float32)
    W1 = gemm1_weights.to(torch.float32)
    W2 = gemm2_weights.to(torch.float32)
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)
        G1 = A_e.matmul(W1[le].t())
        X1, X2 = G1[:, :I], G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        O = (silu_X2 * X1).matmul(W2[le].t())
        w_tok = weights.index_select(0, token_idx)
        match = (topk_idx.index_select(0, token_idx) == ge).float()
        w_e = (w_tok * match).sum(dim=1)
        output.index_add_(0, token_idx, O * w_e.unsqueeze(1))
    return output.to(torch.bfloat16)


@torch.no_grad()
def _default_routing_weights(routing_logits, routing_bias, top_k, scale):
    logits = routing_logits.to(torch.float32)
    if routing_bias is not None:
        logits = logits + routing_bias.to(torch.float32).reshape(-1)
    s = torch.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(s, k=int(top_k), dim=1, largest=True, sorted=False)
    return s.gather(1, topk_idx) * float(scale or 1.0), topk_idx


@torch.no_grad()
def _cutlass_fused_moe_reference(
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
    **_unused,
):
    """Reference for CUTLASS fused MoE with precomputed routing."""
    E_global = fc1_expert_weights.shape[0]
    return _moe_bf16_run_experts(
        input,
        fc1_expert_weights,
        fc2_expert_weights,
        token_final_scales,
        token_selected_experts.to(torch.int64),
        local_expert_offset=0,
        E_global=E_global,
    )


@torch.no_grad()
def _trtllm_bf16_moe_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM BF16 MoE (Default routing)."""
    w_topk, topk_idx = _default_routing_weights(
        routing_logits, routing_bias, top_k, routed_scaling_factor
    )
    return _moe_bf16_run_experts(
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        w_topk,
        topk_idx,
        local_expert_offset,
        int(num_experts),
    )


@torch.no_grad()
def _trtllm_bf16_routed_moe_reference(
    topk_ids,
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM BF16 MoE with precomputed topk_ids."""
    T = topk_ids.shape[0]
    scale = float(routed_scaling_factor or 1.0)
    # Uniform weight per selected expert (real routing scales not available).
    w_topk = torch.full(
        (T, int(top_k)),
        scale / float(top_k),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return _moe_bf16_run_experts(
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        w_topk,
        topk_ids.to(torch.int64),
        local_expert_offset,
        int(num_experts),
    )


@torch.no_grad()
def _trtllm_fp8_per_tensor_scale_moe_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    gemm1_weights,
    output1_scales_scalar,
    output1_scales_gate_scalar,
    gemm2_weights,
    output2_scales_scalar,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM FP8 per-tensor scale MoE. Dequantizes per-expert."""
    E_local = gemm1_weights.shape[0]
    w_topk, topk_idx = _default_routing_weights(
        routing_logits, routing_bias, top_k, routed_scaling_factor
    )
    # Per-expert dequant: each expert has its own scalar scale for FC1 gate,
    # FC1 up, and FC2. Scale broadcasts over the non-expert dims.
    W1 = gemm1_weights.to(torch.float32)
    W2 = gemm2_weights.to(torch.float32)
    s1 = output1_scales_scalar.to(torch.float32).view(E_local, 1, 1)
    s1g = output1_scales_gate_scalar.to(torch.float32).view(E_local, 1, 1)
    s2 = output2_scales_scalar.to(torch.float32).view(E_local, 1, 1)
    I = W1.shape[1] // 2
    # W1 is [E, 2I, H]: first half is gate, second half is up — apply scales.
    W1 = torch.cat([W1[:, :I] * s1g, W1[:, I:] * s1], dim=1)
    W2 = W2 * s2
    return _moe_bf16_run_experts(
        hidden_states.to(torch.float32),
        W1,
        W2,
        w_topk,
        topk_idx,
        local_expert_offset,
        int(num_experts),
    )


@torch.no_grad()
def _trtllm_fp8_block_scale_routed_moe_reference(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM FP8 block-scale routed MoE (precomputed topk_ids).

    Reuses ``_fp8_moe_run_experts`` for the dequant + SwiGLU path, and builds
    a uniform per-token weight tensor (real routing scales are not available
    from topk_ids alone).
    """
    T = topk_ids.shape[0]
    TOP_K = int(top_k)
    scale = float(routed_scaling_factor or 1.0)
    w_topk = torch.full(
        (T, TOP_K),
        scale / TOP_K,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return _fp8_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        w_topk,
        topk_ids.to(torch.int64),
        local_expert_offset,
        int(num_experts),
    )


@torch.no_grad()
def _trtllm_fp4_block_scale_routed_moe_reference(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM FP4 block-scale routed MoE (precomputed topk_ids)."""
    T = topk_ids.shape[0]
    TOP_K = int(top_k)
    scale = float(routed_scaling_factor or 1.0)
    w_topk = torch.full(
        (T, TOP_K),
        scale / TOP_K,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return _fp4_moe_run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gemm1_bias,
        gemm2_bias,
        w_topk,
        topk_ids.to(torch.int64),
        local_expert_offset,
        int(num_experts),
    )


@torch.no_grad()
def _trtllm_mxint4_block_scale_moe_reference(
    routing_logits,
    routing_bias,
    hidden_states,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts,
    top_k,
    local_expert_offset,
    routed_scaling_factor=None,
    **_unused,
):
    """Reference for TRT-LLM MxInt4 block-scale MoE.

    Weights are int4 packed as uint8 with bf16 per-32 block scales. Hidden
    states are bf16 (no activation quantization).
    """

    # Unpack int4: low nibble is first element, values are 4-bit signed (-8..7).
    def _unpack_int4(packed):
        lo = (packed & 0x0F).to(torch.int64)
        hi = ((packed >> 4) & 0x0F).to(torch.int64)
        # Sign-extend from 4-bit.
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        stacked = torch.stack([lo, hi], dim=-1)
        return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(
            torch.float32
        )

    W1 = _unpack_int4(gemm1_weights)  # [E, 2I, H]
    W2 = _unpack_int4(gemm2_weights)  # [E, H, I]
    # Scales are bf16, broadcast per-32 along last axis.
    s1 = gemm1_weights_scale.to(torch.float32)
    s2 = gemm2_weights_scale.to(torch.float32)
    block1 = W1.shape[-1] // s1.shape[-1]
    block2 = W2.shape[-1] // s2.shape[-1]
    W1 = W1 * s1.repeat_interleave(block1, dim=-1)
    W2 = W2 * s2.repeat_interleave(block2, dim=-1)

    w_topk, topk_idx = _default_routing_weights(
        routing_logits, routing_bias, top_k, routed_scaling_factor
    )
    return _moe_bf16_run_experts(
        hidden_states,
        W1,
        W2,
        w_topk,
        topk_idx,
        local_expert_offset,
        int(num_experts),
    )


# CUTLASS fused MoE: precomputed token_selected_experts + token_final_scales
cutlass_fused_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="cutlass_fused_moe",
    description="CUTLASS fused MoE. Accepts precomputed per-token expert selections.",
    axes={
        "seq_len": Var(description="Number of input tokens."),
        "num_local_experts": Const(abbrev="e"),
        "hidden_size": Const(abbrev="h"),
        "intermediate_size": Const(abbrev="i"),
        "top_k": Const(abbrev="topk"),
    },
    inputs={
        "input": Tensor(
            ["seq_len", "hidden_size"],
            description="Input hidden states (bf16/fp8/fp4 depending on quant config).",
        ),
        "token_selected_experts": Tensor(
            ["seq_len", "top_k"],
            dtype="int32",
            description="Precomputed top-k expert ids per token.",
        ),
        "token_final_scales": Tensor(
            ["seq_len", "top_k"],
            dtype="float32",
            description="Precomputed per-token expert scales.",
        ),
        "fc1_expert_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "hidden_size"],
            description="FC1 weights per expert.",
        ),
        "fc2_expert_weights": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            description="FC2 weights per expert.",
        ),
    },
    outputs={
        "output": Tensor(["seq_len", "hidden_size"], dtype="bfloat16"),
    },
    tags=["status:verified", "backend:cutlass"],
    reference=_cutlass_fused_moe_reference,
)
cutlass_fused_moe_trace.axes["gemm1_out_size"] = Const(
    abbrev="", description="FC1 output size (typically 2 * intermediate_size)."
)

# Shared factory for the remaining trtllm_* variants
_TRTLLM_MOE_COMMON_INPUTS: dict[str, Tensor | Scalar] = {
    "routing_logits": Tensor(
        ["seq_len", "num_experts"], description="Routing logits for expert selection."
    ),
    "routing_bias": Tensor(
        ["num_experts"], optional=True, description="Optional routing bias."
    ),
    "hidden_states": Tensor(
        ["seq_len", "hidden_size"],
        description="Input hidden states (dtype depends on variant).",
    ),
    "gemm1_weights": Tensor(
        ["num_local_experts", "gemm1_out_size", "hidden_size"],
        description="FC1 weights (gate+up).",
    ),
    "gemm2_weights": Tensor(
        ["num_local_experts", "hidden_size", "intermediate_size"],
        description="FC2 weights (down).",
    ),
    "top_k": Scalar("int32", description="Number of experts to route per token."),
    "n_group": Scalar(
        "int32", optional=True, description="Expert groups (DeepSeek-V3)."
    ),
    "topk_group": Scalar(
        "int32", optional=True, description="Groups to keep (DeepSeek-V3)."
    ),
    "local_expert_offset": Scalar(
        "int32", description="Offset of local experts in global expert space."
    ),
    "routed_scaling_factor": Scalar(
        "float32", optional=True, description="Scaling factor for routing weights."
    ),
    "routing_method_type": Scalar(
        "int32",
        optional=True,
        description="0=Default, 1=Renormalize, 2=DeepSeekV3, 3=Llama4, 4=RenormalizeNaive, 5=TopK.",
    ),
}

_TRTLLM_MOE_COMMON_AXES: dict[str, Var | Const] = {
    **_MOE_COMMON_AXES,
    "gemm1_out_size": Const(abbrev="", description="2 * intermediate_size."),
}

_TRTLLM_MOE_COMMON_OUTPUTS: dict[str, Tensor | Scalar] = {
    "output": Tensor(
        ["seq_len", "hidden_size"], dtype="bfloat16", description="MoE output."
    ),
}

# BF16 MoE (no quantization)
trtllm_bf16_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_bf16_moe",
    description="TRT-LLM BF16 MoE (no quantization).",
    axes=dict(_TRTLLM_MOE_COMMON_AXES),
    inputs=dict(_TRTLLM_MOE_COMMON_INPUTS),
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:verified", "backend:trtllm"],
    reference=_trtllm_bf16_moe_reference,
)

# BF16 routed MoE (accepts precomputed topk_ids instead of routing_logits)
# num_experts / intermediate_size become Var in routed variants because they
# are passed as scalar kwargs (no routing_logits tensor to resolve from).
_TRTLLM_MOE_ROUTED_AXES: dict[str, Var | Const] = {
    **_TRTLLM_MOE_COMMON_AXES,
    "num_experts": Var(description="Total number of experts (passed as kwarg)."),
    "intermediate_size": Var(
        description="MoE intermediate layer size (passed as kwarg)."
    ),
}
trtllm_bf16_routed_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_bf16_routed_moe",
    description="TRT-LLM BF16 MoE with precomputed topk_ids.",
    axes=dict(_TRTLLM_MOE_ROUTED_AXES),
    inputs={
        "topk_ids": Tensor(
            ["seq_len", "top_k"],
            dtype="int32",
            description="Precomputed top-k expert ids per token.",
        ),
        "hidden_states": _TRTLLM_MOE_COMMON_INPUTS["hidden_states"],
        "gemm1_weights": _TRTLLM_MOE_COMMON_INPUTS["gemm1_weights"],
        "gemm2_weights": _TRTLLM_MOE_COMMON_INPUTS["gemm2_weights"],
        "num_experts": Scalar("int32", description="Total number of experts."),
        "top_k": _TRTLLM_MOE_COMMON_INPUTS["top_k"],
        "local_expert_offset": _TRTLLM_MOE_COMMON_INPUTS["local_expert_offset"],
        "routed_scaling_factor": _TRTLLM_MOE_COMMON_INPUTS["routed_scaling_factor"],
    },
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:verified", "backend:trtllm"],
    reference=_trtllm_bf16_routed_moe_reference,
)

# FP8 per-tensor scale MoE
trtllm_fp8_per_tensor_scale_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_fp8_per_tensor_scale_moe",
    description="TRT-LLM FP8 MoE with per-tensor activation/weight scales.",
    axes=dict(_TRTLLM_MOE_COMMON_AXES),
    inputs={
        **_TRTLLM_MOE_COMMON_INPUTS,
        "output1_scales_scalar": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC1 output scale.",
        ),
        "output1_scales_gate_scalar": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC1 gate scale.",
        ),
        "output2_scales_scalar": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC2 output scale.",
        ),
    },
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:verified", "backend:trtllm", "quantization:float8_e4m3fn"],
    reference=_trtllm_fp8_per_tensor_scale_moe_reference,
)

# FP8 block-scale routed (precomputed topk_ids)
trtllm_fp8_block_scale_routed_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_fp8_block_scale_routed_moe",
    description="TRT-LLM FP8 block-scale MoE with precomputed topk_ids.",
    axes={
        **_TRTLLM_MOE_ROUTED_AXES,
        "num_hidden_blocks": Const(abbrev=""),
        "num_intermediate_blocks": Const(abbrev=""),
        "num_gemm1_out_blocks": Const(abbrev=""),
    },
    inputs={
        "topk_ids": Tensor(
            ["seq_len", "top_k"], dtype="int32", description="Precomputed top-k."
        ),
        "routing_bias": Tensor(
            ["num_experts"], optional=True, description="Optional routing bias."
        ),
        "hidden_states": Tensor(
            ["seq_len", "hidden_size"],
            description="FP8-quantized hidden states.",
        ),
        "hidden_states_scale": Tensor(
            ["num_hidden_blocks", "seq_len"],
            description="Block-wise hidden_states scale.",
        ),
        "gemm1_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "hidden_size"],
            description="FC1 FP8 weights.",
        ),
        "gemm1_weights_scale": Tensor(
            ["num_local_experts", "num_gemm1_out_blocks", "num_hidden_blocks"],
            description="FC1 block-wise scale.",
        ),
        "gemm2_weights": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            description="FC2 FP8 weights.",
        ),
        "gemm2_weights_scale": Tensor(
            ["num_local_experts", "num_hidden_blocks", "num_intermediate_blocks"],
            description="FC2 block-wise scale.",
        ),
        "num_experts": Scalar("int32", description="Total number of experts."),
        "top_k": Scalar("int32"),
        "local_expert_offset": Scalar("int32"),
        "routed_scaling_factor": Scalar("float32", optional=True),
    },
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:verified", "backend:trtllm", "quantization:float8_e4m3fn"],
    reference=_trtllm_fp8_block_scale_routed_moe_reference,
)

# FP4 block-scale routed (precomputed topk_ids)
trtllm_fp4_block_scale_routed_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_fp4_block_scale_routed_moe",
    description="TRT-LLM NvFP4 block-scale MoE with precomputed topk_ids.",
    axes={
        **_TRTLLM_MOE_ROUTED_AXES,
        "num_packed_hidden": Const(abbrev=""),
        # Var rather than Const because hidden_states_scale is optional and the
        # other tensors using this axis may have different shapes in routed mode.
        "num_fp4_hidden_blocks": Var(
            description="NvFP4 block count along hidden_size."
        ),
        "num_packed_intermediate": Const(abbrev=""),
        "num_fp4_intermediate_blocks": Const(abbrev=""),
    },
    inputs={
        "topk_ids": Tensor(
            ["seq_len", "top_k"], dtype="int32", description="Precomputed top-k."
        ),
        "routing_bias": Tensor(
            ["num_experts"], optional=True, description="Optional routing bias."
        ),
        "hidden_states": Tensor(
            ["seq_len", "num_packed_hidden"],
            description="NvFP4-packed hidden states.",
        ),
        "hidden_states_scale": Tensor(
            ["seq_len", "num_fp4_hidden_blocks"],
            optional=True,
            description="NvFP4 hidden_states scale.",
        ),
        "gemm1_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_packed_hidden"],
            description="FC1 NvFP4 weights.",
        ),
        "gemm1_weights_scale": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_fp4_hidden_blocks"],
            description="FC1 NvFP4 scale.",
        ),
        "gemm2_weights": Tensor(
            ["num_local_experts", "hidden_size", "num_packed_intermediate"],
            description="FC2 NvFP4 weights.",
        ),
        "gemm2_weights_scale": Tensor(
            ["num_local_experts", "hidden_size", "num_fp4_intermediate_blocks"],
            description="FC2 NvFP4 scale.",
        ),
        "num_experts": Scalar("int32", description="Total number of experts."),
        "top_k": Scalar("int32"),
        "local_expert_offset": Scalar("int32"),
        "routed_scaling_factor": Scalar("float32", optional=True),
    },
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:experimental", "backend:trtllm", "quantization:nvfp4"],
    reference=_trtllm_fp4_block_scale_routed_moe_reference,
)

# MxInt4 block-scale MoE
trtllm_mxint4_block_scale_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="trtllm_mxint4_block_scale_moe",
    description="TRT-LLM MxInt4 block-scale MoE.",
    axes={
        **_TRTLLM_MOE_COMMON_AXES,
        "intermediate_size": Var(description="MoE intermediate size (kwarg)."),
        "num_packed_hidden": Const(abbrev=""),
        "num_mxint4_hidden_blocks": Const(abbrev=""),
        "num_packed_intermediate": Const(abbrev=""),
        "num_mxint4_intermediate_blocks": Const(abbrev=""),
    },
    inputs={
        "routing_logits": Tensor(
            ["seq_len", "num_experts"], description="Routing logits."
        ),
        "routing_bias": Tensor(
            ["num_experts"], optional=True, description="Optional routing bias."
        ),
        "hidden_states": Tensor(
            ["seq_len", "hidden_size"],
            description="BF16/FP16 hidden states (quantized internally).",
        ),
        "gemm1_weights": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_packed_hidden"],
            description="FC1 MxInt4-packed weights.",
        ),
        "gemm1_weights_scale": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_mxint4_hidden_blocks"],
            description="FC1 MxInt4 scales.",
        ),
        "gemm2_weights": Tensor(
            ["num_local_experts", "hidden_size", "num_packed_intermediate"],
            description="FC2 MxInt4-packed weights.",
        ),
        "gemm2_weights_scale": Tensor(
            ["num_local_experts", "hidden_size", "num_mxint4_intermediate_blocks"],
            description="FC2 MxInt4 scales.",
        ),
        "top_k": Scalar("int32"),
        "n_group": Scalar("int32", optional=True),
        "topk_group": Scalar("int32", optional=True),
        "local_expert_offset": Scalar("int32"),
        "routed_scaling_factor": Scalar("float32", optional=True),
        "routing_method_type": Scalar("int32", optional=True),
    },
    outputs=dict(_TRTLLM_MOE_COMMON_OUTPUTS),
    tags=["status:experimental", "backend:trtllm", "quantization:mxint4"],
    reference=_trtllm_mxint4_block_scale_moe_reference,
)


# ---------------------------------------------------------------------------
# CuteDSL MoE variants (precomputed routing, NvFP4 weights on SM100+)
# ---------------------------------------------------------------------------

cute_dsl_fused_moe_nvfp4_trace = TraceTemplate(
    op_type="moe",
    name_prefix="cute_dsl_fused_moe_nvfp4",
    description=(
        "CuteDSL NVFP4 fused MoE (SM100/SM103). Accepts NvFP4-packed input + "
        "scales with precomputed top-k routing (token_selected_experts + "
        "token_final_scales) and per-expert alpha scales."
    ),
    axes={
        "num_tokens": Var(description="Total tokens across the batch."),
        "num_experts": Const(abbrev="", description="Total number of experts."),
        "top_k": Const(abbrev="topk"),
        "num_local_experts": Const(abbrev="e"),
        "hidden_size": Const(abbrev="h"),
        "intermediate_size": Var(description="MoE intermediate size (kwarg)."),
        "num_packed_hidden": Var(description="hidden_size // 2 (NvFP4 packed)."),
        "num_packed_intermediate": Var(
            description="intermediate_size // 2 (NvFP4 packed)."
        ),
        "num_fp4_hidden_blocks": Var(
            description="NvFP4 scale-factor count along hidden_size."
        ),
        "num_fp4_intermediate_blocks": Var(
            description="NvFP4 scale-factor count along intermediate_size."
        ),
        "gemm1_out_size": Const(abbrev="", description="2 * intermediate_size."),
    },
    inputs={
        "x": Tensor(
            ["num_tokens", "num_packed_hidden"],
            description="NvFP4-packed input (uint8, 2 fp4 per byte).",
        ),
        "x_sf": Tensor(
            ["num_tokens", "num_fp4_hidden_blocks"],
            description="NvFP4 scale factors for x (float8_e4m3fn).",
        ),
        "token_selected_experts": Tensor(
            ["num_tokens", "top_k"],
            dtype="int32",
            description="Precomputed top-k expert ids per token.",
        ),
        "token_final_scales": Tensor(
            ["num_tokens", "top_k"],
            dtype="float32",
            description="Precomputed per-token routing scales.",
        ),
        "w1_weight": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_packed_hidden"],
            description="FC1 weights, NvFP4-packed.",
        ),
        "w1_weight_sf": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_fp4_hidden_blocks"],
            description="FC1 NvFP4 scales.",
        ),
        "w1_alpha": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC1 global scale.",
        ),
        "fc2_input_scale": Tensor(
            ["one"],
            dtype="float32",
            description="Global scale for FC2 input quantization.",
        ),
        "w2_weight": Tensor(
            ["num_local_experts", "hidden_size", "num_packed_intermediate"],
            description="FC2 weights, NvFP4-packed.",
        ),
        "w2_weight_sf": Tensor(
            ["num_local_experts", "hidden_size", "num_fp4_intermediate_blocks"],
            description="FC2 NvFP4 scales.",
        ),
        "w2_alpha": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC2 global scale.",
        ),
        "num_experts": Scalar("int32", description="Total number of experts."),
        "top_k": Scalar("int32", description="Number of experts per token."),
        "local_expert_offset": Scalar(
            "int32", optional=True, description="Offset of local experts."
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "hidden_size"],
            dtype="bfloat16",
            description="MoE output.",
        ),
    },
    tags=["status:experimental", "backend:cute-dsl", "quantization:nvfp4"],
)
cute_dsl_fused_moe_nvfp4_trace.axes["one"] = Var(
    description="Placeholder for shape [1] scalars."
)

_cute_dsl_wrapper_inputs = dict(cute_dsl_fused_moe_nvfp4_trace.inputs)
# num_experts / top_k live on the wrapper instance (set in __init__), not on run().
_cute_dsl_wrapper_inputs["num_experts"] = Scalar(
    "int32",
    optional=True,
    description="Set at wrapper __init__, not passed to run().",
)
_cute_dsl_wrapper_inputs["top_k"] = Scalar(
    "int32",
    optional=True,
    description="Set at wrapper __init__, not passed to run().",
)

_cute_dsl_wrapper_axes = dict(cute_dsl_fused_moe_nvfp4_trace.axes)
# num_experts / top_k are set at __init__ time — no tensor on run() has a
# num_experts dim, so the axis must be a Var here.
_cute_dsl_wrapper_axes["num_experts"] = Var(description="Total number of experts.")
_cute_dsl_wrapper_axes["top_k"] = Var(description="Experts per token.")

cute_dsl_moe_wrapper_run_trace = TraceTemplate(
    op_type="moe",
    name_prefix="cute_dsl_moe_wrapper",
    description=(
        "CuteDslMoEWrapper.run(): stateful version of cute_dsl_fused_moe_nvfp4 "
        "(same schema; wrapper persists autotuning state across calls)."
    ),
    axes=_cute_dsl_wrapper_axes,
    inputs=_cute_dsl_wrapper_inputs,
    outputs=dict(cute_dsl_fused_moe_nvfp4_trace.outputs),
    tags=cute_dsl_fused_moe_nvfp4_trace.tags,
)


# ---------------------------------------------------------------------------
# B12x MoE (SM120/SM121 CuTe-DSL, bf16 input + FP4 packed weights)
# ---------------------------------------------------------------------------

b12x_fused_moe_trace = TraceTemplate(
    op_type="moe",
    name_prefix="b12x_fused_moe",
    description=(
        "B12x CuTe-DSL fused MoE (SM120/SM121). BF16 input, FP4-packed "
        "weights, precomputed top-k routing; fuses quant + FC1 + activation + "
        "FC2 + scatter."
    ),
    axes={
        "num_tokens": Var(),
        "num_experts": Const(abbrev="", description="Total number of experts."),
        "top_k": Const(abbrev="topk"),
        "num_local_experts": Const(abbrev="e"),
        "hidden_size": Const(abbrev="h"),
        "intermediate_size": Var(description="MoE intermediate size (kwarg)."),
        "num_packed_hidden": Var(description="hidden_size // 2."),
        "num_packed_intermediate": Var(description="intermediate_size // 2."),
        "num_fp4_hidden_blocks": Var(),
        "num_fp4_intermediate_blocks": Var(),
        "gemm1_out_size": Const(
            abbrev="",
            description="2*I (SwiGLU) or I (ReLU2).",
        ),
    },
    inputs={
        "x": Tensor(
            ["num_tokens", "hidden_size"], description="BF16 input activations."
        ),
        "w1_weight": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_packed_hidden"],
            description="FC1 weights, FP4-packed.",
        ),
        "w1_weight_sf": Tensor(
            ["num_local_experts", "gemm1_out_size", "num_fp4_hidden_blocks"],
            description="FC1 FP4 scales.",
        ),
        "w2_weight": Tensor(
            ["num_local_experts", "hidden_size", "num_packed_intermediate"],
            description="FC2 weights, FP4-packed.",
        ),
        "w2_weight_sf": Tensor(
            ["num_local_experts", "hidden_size", "num_fp4_intermediate_blocks"],
            description="FC2 FP4 scales.",
        ),
        "token_selected_experts": Tensor(
            ["num_tokens", "top_k"],
            dtype="int32",
            description="Precomputed top-k expert ids per token.",
        ),
        "token_final_scales": Tensor(
            ["num_tokens", "top_k"],
            dtype="float32",
            description="Precomputed per-token routing scales.",
        ),
        "num_experts": Scalar("int32", description="Total experts."),
        "top_k": Scalar("int32"),
        "w1_alpha": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC1 global scale.",
        ),
        "w2_alpha": Tensor(
            ["num_local_experts"],
            dtype="float32",
            description="Per-expert FC2 global scale.",
        ),
        "fc2_input_scale": Tensor(
            ["one"],
            dtype="float32",
            description="Global scale for FC2 input quantization.",
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "hidden_size"],
            dtype="bfloat16",
            description="MoE output.",
        ),
    },
    tags=["status:experimental", "backend:cute-dsl", "quantization:fp4"],
)
b12x_fused_moe_trace.axes["one"] = Var(description="Placeholder for shape [1].")

_b12x_wrapper_inputs = dict(b12x_fused_moe_trace.inputs)
_b12x_wrapper_inputs["num_experts"] = Scalar(
    "int32",
    optional=True,
    description="Set at wrapper __init__, not passed to run().",
)
_b12x_wrapper_inputs["top_k"] = Scalar(
    "int32",
    optional=True,
    description="Set at wrapper __init__, not passed to run().",
)

_b12x_wrapper_axes = dict(b12x_fused_moe_trace.axes)
_b12x_wrapper_axes["num_experts"] = Var(description="Total number of experts.")
_b12x_wrapper_axes["top_k"] = Var(description="Experts per token.")


@torch.no_grad()
def _cute_dsl_fused_moe_nvfp4_reference(
    x,
    x_sf,
    token_selected_experts,
    token_final_scales,
    w1_weight,
    w1_weight_sf,
    w1_alpha,
    fc2_input_scale,
    w2_weight,
    w2_weight_sf,
    w2_alpha,
    num_experts,
    top_k,
    **_unused,
):
    """Reference for CuteDSL NvFP4 fused MoE — bridges to the FP4
    block-scale kernel with alpha scales folded into the dequantized
    weights."""
    E_local = w1_weight.shape[0]
    # Dequantize input and weights with alpha factors.
    hs_deq = _dequantize_fp4_tensor(x, x_sf, is_ue8m0_scales=False)
    W1 = _dequantize_fp4_tensor(w1_weight, w1_weight_sf, is_ue8m0_scales=False)
    W2 = _dequantize_fp4_tensor(w2_weight, w2_weight_sf, is_ue8m0_scales=False)
    W1 = W1 * w1_alpha.to(torch.float32).view(E_local, 1, 1)
    W2 = W2 * w2_alpha.to(torch.float32).view(E_local, 1, 1)
    return _moe_bf16_run_experts(
        hs_deq,
        W1,
        W2,
        token_final_scales,
        token_selected_experts.to(torch.int64),
        local_expert_offset=0,
        E_global=int(num_experts),
    )


@torch.no_grad()
def _b12x_fused_moe_reference(
    x,
    w1_weight,
    w1_weight_sf,
    w2_weight,
    w2_weight_sf,
    token_selected_experts,
    token_final_scales,
    num_experts,
    top_k,
    w1_alpha=None,
    w2_alpha=None,
    fc2_input_scale=None,
    **_unused,
):
    """Reference for B12x CuTe-DSL fused MoE (bf16 input, FP4 weights)."""
    E_local = w1_weight.shape[0]
    W1 = _dequantize_fp4_tensor(w1_weight, w1_weight_sf, is_ue8m0_scales=False)
    W2 = _dequantize_fp4_tensor(w2_weight, w2_weight_sf, is_ue8m0_scales=False)
    if w1_alpha is not None:
        W1 = W1 * w1_alpha.to(torch.float32).view(E_local, 1, 1)
    if w2_alpha is not None:
        W2 = W2 * w2_alpha.to(torch.float32).view(E_local, 1, 1)
    return _moe_bf16_run_experts(
        x,
        W1,
        W2,
        token_final_scales,
        token_selected_experts.to(torch.int64),
        local_expert_offset=0,
        E_global=int(num_experts),
    )


cute_dsl_fused_moe_nvfp4_trace.reference = _cute_dsl_fused_moe_nvfp4_reference
cute_dsl_moe_wrapper_run_trace.reference = _cute_dsl_fused_moe_nvfp4_reference
b12x_fused_moe_trace.reference = _b12x_fused_moe_reference


b12x_moe_wrapper_run_trace = TraceTemplate(
    op_type="moe",
    name_prefix="b12x_moe_wrapper",
    description="B12xMoEWrapper.run(): wrapper form of b12x_fused_moe.",
    axes=_b12x_wrapper_axes,
    inputs=_b12x_wrapper_inputs,
    outputs=dict(b12x_fused_moe_trace.outputs),
    tags=b12x_fused_moe_trace.tags,
    reference=_b12x_fused_moe_reference,
)
