"""BF16/MXFP8 input staging for the SM120 W4A8 split backend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .weights import SCALE_BLOCK, SCALE_DTYPE, ceil_div, round_up


ACTIVATION_DTYPE = torch.float8_e4m3fn


@triton.jit
def _stage_bf16_inputs_kernel(
    hidden_states,
    topk_weights,
    topk_ids,
    x_fp8,
    x_sf_packed,
    staged_topk_weights,
    staged_topk_ids,
    num_tokens,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    staged_weights_stride_m: tl.constexpr,
    staged_weights_stride_k: tl.constexpr,
    staged_ids_stride_m: tl.constexpr,
    staged_ids_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)
    active = token_id < num_tokens
    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = active & (k_offsets < hidden_size)
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.maximum(tl.max(hidden_groups, axis=1), 1.0e-4)
    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + (
        (scale_bits & 0x7FFFFF) != 0
    ).to(tl.uint32)
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
    scaled = tl.reshape(hidden, [num_groups, GROUP_K]) * (
        1.0 / rounded_scale
    )[:, None]
    fp8 = tl.reshape(scaled, [BLOCK_K]).to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf_packed
        + token_id * x_sf_stride_m
        + k_block_id * x_sf_stride_k,
        packed_scale,
        mask=active,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        live_mask = active & topk_mask
        ids = tl.load(
            topk_ids
            + token_id * topk_ids_stride_m
            + topk_offsets * topk_ids_stride_k,
            mask=live_mask,
            other=-1,
        ).to(tl.int64)
        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=live_mask,
            other=0.0,
        )
        tl.store(
            staged_topk_ids
            + token_id * staged_ids_stride_m
            + topk_offsets * staged_ids_stride_k,
            tl.where(active, ids, -1),
            mask=topk_mask,
        )
        tl.store(
            staged_topk_weights
            + token_id * staged_weights_stride_m
            + topk_offsets * staged_weights_stride_k,
            tl.where(active, weights, 0.0),
            mask=topk_mask,
        )


def _stage_bf16_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    staged_topk_ids: torch.Tensor,
    staged_topk_weights: torch.Tensor,
) -> None:
    tokens, hidden = hidden_states.shape
    if hidden % 128:
        raise ValueError("SM120 W4A8 staging requires hidden size divisible by 128")
    packed_scale = x_scale.view(torch.int32)
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    grid = (x.shape[0], hidden // 128)
    _stage_bf16_inputs_kernel[grid](
        hidden_states,
        topk_weights,
        topk_ids,
        x,
        packed_scale,
        staged_topk_weights,
        staged_topk_ids,
        tokens,
        hidden_states.stride(0),
        hidden_states.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        x.stride(0),
        x.stride(1),
        packed_scale.stride(0),
        packed_scale.stride(1),
        staged_topk_weights.stride(0),
        staged_topk_weights.stride(1),
        staged_topk_ids.stride(0),
        staged_topk_ids.stride(1),
        hidden,
        topk_ids.shape[1],
        BLOCK_K=128,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    staged_topk_ids: torch.Tensor,
    staged_topk_weights: torch.Tensor,
    *,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> None:
    tokens, hidden = hidden_states.shape
    if tokens > x.shape[0]:
        raise ValueError(f"{tokens} tokens exceed workspace capacity {x.shape[0]}")
    if quantize_input:
        _stage_bf16_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            x,
            x_scale,
            staged_topk_ids,
            staged_topk_weights,
        )
        return
    else:
        if hidden_states.dtype != ACTIVATION_DTYPE or scales is None:
            raise ValueError(
                f"pre-quantized input requires {ACTIVATION_DTYPE} data and E8M0 scales"
            )
        quantized, raw_scale = hidden_states, scales

    valid_scale_columns = ceil_div(hidden, SCALE_BLOCK)
    padded_scale_columns = round_up(valid_scale_columns, 4)
    if raw_scale.dtype != SCALE_DTYPE:
        if raw_scale.dtype != torch.uint8:
            raise ValueError(f"activation scales must be {SCALE_DTYPE} or uint8")
        raw_scale = raw_scale.view(SCALE_DTYPE)
    if raw_scale.shape[1] < valid_scale_columns:
        raise ValueError(
            f"activation scale width {raw_scale.shape[1]} is smaller than {valid_scale_columns}"
        )

    x[:tokens].view(torch.uint8).copy_(quantized.view(torch.uint8))
    x_scale[:tokens].zero_()
    x_scale[:tokens, :valid_scale_columns].view(torch.uint8).copy_(
        raw_scale[:, :valid_scale_columns].view(torch.uint8)
    )
    if x_scale.shape[1] < padded_scale_columns:
        raise ValueError("workspace activation-scale row is not padded to four bytes")
    staged_topk_ids[:tokens].copy_(topk_ids)
    staged_topk_weights[:tokens].copy_(topk_weights)
    if tokens < x.shape[0]:
        staged_topk_ids[tokens:].fill_(-1)


__all__ = ["ACTIVATION_DTYPE", "stage_inputs"]
