"""Non-fused SM90 MoE baseline built from local grouped FP8 GEMMs."""

from __future__ import annotations

import weakref
from typing import Any

import torch

COMPACT_EMPTY_EXPERTS = False
EXPERT_PAD = 1

_runner = None
_QuantizedWeights = tuple[torch.Tensor, ...]
_WeightCacheEntry = tuple[
    weakref.ReferenceType[torch.Tensor],
    weakref.ReferenceType[torch.Tensor],
    _QuantizedWeights,
]
_weight_cache: dict[tuple[int, int], _WeightCacheEntry] = {}


def get_runner():
    """Return one cached grouped-GEMM runner for the current process."""
    global _runner
    if _runner is None:
        from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import (
            create_sm90_push_fp8_moe_gemm_runner,
        )

        _runner = create_sm90_push_fp8_moe_gemm_runner()
    return _runner


def quant_weights(w13: torch.Tensor, w2: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Quantize expert weights and cache results for the exact source tensors."""
    key = (w13.data_ptr(), w2.data_ptr())
    cached = _weight_cache.get(key)
    if cached is not None:
        cached_w13, cached_w2, result = cached
        if cached_w13() is w13 and cached_w2() is w2:
            return result
    from flashinfer.testing.utils import per_block_cast_to_fp8

    num_experts, two_intermediate, hidden = w13.shape
    _, weight_hidden, intermediate = w2.shape
    assert weight_hidden == hidden
    w13_fp8 = torch.empty_like(w13, dtype=torch.float8_e4m3fn)
    w13_scales = torch.empty(
        num_experts,
        two_intermediate // 128,
        hidden // 128,
        device=w13.device,
        dtype=torch.float32,
    )
    w2_fp8 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    w2_scales = torch.empty(
        num_experts,
        hidden // 128,
        intermediate // 128,
        device=w2.device,
        dtype=torch.float32,
    )
    for expert in range(num_experts):
        quantized, scales = per_block_cast_to_fp8(w13[expert])
        w13_fp8[expert].copy_(quantized)
        w13_scales[expert].copy_(scales)
        quantized, scales = per_block_cast_to_fp8(w2[expert])
        w2_fp8[expert].copy_(quantized)
        w2_scales[expert].copy_(scales)
    result = (w13_fp8, w13_scales, w2_fp8, w2_scales)
    _weight_cache[key] = (weakref.ref(w13), weakref.ref(w2), result)
    return result


def padded_offset(offset: int, group: int) -> int:
    """Mirror the grouped GEMM's 32-row alignment with per-group skew."""
    return (offset + group * 31) // 32 * 32


def quant_act_grouped(
    activations: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize expert-contiguous rows and pack their external activation scales."""
    rows, columns = activations.shape
    num_k_blocks = columns // 128
    groups = offsets.numel() - 1
    host_offsets = offsets.tolist()
    padded_rows = max(padded_offset(host_offsets[-1], groups), 1)
    blocks = activations.float().reshape(rows, num_k_blocks, 128)
    amax = blocks.abs().amax(dim=-1)
    scales = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    quantized = (
        (blocks / scales.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .reshape(rows, columns)
        .to(torch.float8_e4m3fn)
    )
    packed_scales = torch.zeros(
        num_k_blocks,
        padded_rows,
        dtype=torch.float32,
        device=activations.device,
    )
    for group in range(groups):
        start, end = host_offsets[group], host_offsets[group + 1]
        if end > start:
            packed_start = padded_offset(start, group)
            packed_scales[:, packed_start : packed_start + end - start] = scales[
                start:end
            ].T
    return quantized, packed_scales.contiguous(), scales


def grouped_ffn(
    runner,
    repeated_x: torch.Tensor,
    offsets: torch.Tensor,
    w13_fp8: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    """Run FC1, SwiGLU, and FC2 over expert-contiguous activation rows."""
    from flashinfer.activation import silu_and_mul

    rows, hidden = repeated_x.shape
    groups, two_intermediate, _ = w13_fp8.shape
    intermediate = two_intermediate // 2
    if rows == 0:
        return torch.empty(0, hidden, device=repeated_x.device, dtype=torch.bfloat16)

    workspace_size = runner.get_moe_workspace_size(
        rows,
        rows,
        max(two_intermediate, hidden),
        max(hidden, intermediate),
        groups,
        True,
        True,
    )
    runner.configure_workspace(
        torch.empty(
            max(int(workspace_size), 1),
            device=repeated_x.device,
            dtype=torch.uint8,
        )
    )

    a1, a1_scales, _ = quant_act_grouped(repeated_x, offsets)
    fc1 = torch.empty(
        rows,
        two_intermediate,
        device=repeated_x.device,
        dtype=torch.bfloat16,
    )
    runner.moe_gemm(
        fc1,
        a1,
        w13_fp8,
        offsets,
        two_intermediate,
        hidden,
        a1_scales,
        w13_scales,
        False,
    )
    gated = silu_and_mul(fc1)
    a2, a2_scales, _ = quant_act_grouped(gated, offsets)
    output = torch.empty(rows, hidden, device=repeated_x.device, dtype=torch.bfloat16)
    runner.moe_gemm(
        output,
        a2,
        w2_fp8,
        offsets,
        hidden,
        intermediate,
        a2_scales,
        w2_scales,
        False,
    )
    return output


def build_expert_contiguous(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token-route pairs by expert and return their cumulative offsets."""
    num_tokens, top_k = topk_ids.shape
    flattened_ids = topk_ids.reshape(-1).long()
    order = torch.argsort(flattened_ids, stable=True)
    token_indices = torch.arange(num_tokens, device=topk_ids.device).repeat_interleave(
        top_k
    )[order]
    route_weights = topk_weights.reshape(-1)[order]
    counts = torch.bincount(flattened_ids, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=topk_ids.device)
    offsets[1:] = counts.cumsum(0)
    return token_indices, route_weights, counts, offsets


def sm90_moe_baseline_local(inputs: Any) -> torch.Tensor:
    """Run the local grouped-GEMM baseline for a tensor namespace."""
    runner = get_runner()
    hidden_states, w13, w2 = inputs.hidden_states, inputs.w13, inputs.w2
    num_tokens, hidden = hidden_states.shape
    num_experts = w13.shape[0]
    w13_fp8, w13_scales, w2_fp8, w2_scales = quant_weights(w13, w2)
    token_indices, route_weights, counts, offsets = build_expert_contiguous(
        inputs.topk_ids, inputs.topk_weights, num_experts
    )

    if EXPERT_PAD > 1:
        raise NotImplementedError("expert padding is not implemented")
    if COMPACT_EMPTY_EXPERTS:
        nonempty = (counts > 0).nonzero(as_tuple=True)[0]
        offsets = torch.zeros(
            nonempty.numel() + 1,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        offsets[1:] = counts[nonempty].cumsum(0)
        w13_fp8, w13_scales = w13_fp8[nonempty], w13_scales[nonempty]
        w2_fp8, w2_scales = w2_fp8[nonempty], w2_scales[nonempty]

    repeated_x = hidden_states[token_indices]
    expert_output = grouped_ffn(
        runner,
        repeated_x,
        offsets,
        w13_fp8,
        w13_scales,
        w2_fp8,
        w2_scales,
    )
    output = torch.zeros(
        num_tokens, hidden, device=hidden_states.device, dtype=torch.float32
    )
    output.index_add_(
        0,
        token_indices,
        expert_output.float() * route_weights.unsqueeze(1).float(),
    )
    return output
