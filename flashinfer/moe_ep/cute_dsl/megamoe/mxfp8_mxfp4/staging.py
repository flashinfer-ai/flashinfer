# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation and routing staging for MXFP8 x MXFP4 MegaMoE.

The compute kernel always sees E4M3 data and raw E8M0 scales, one scale per
consecutive K32 block.  This module deliberately does not know about expert
weights or the persistent kernel: it only populates the four source buffers
used by token dispatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


MXFP8_BLOCK_SIZE = 32
MXFP8_DATA_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.float8_e8m0fnu


@dataclass(frozen=True)
class StagedActivation:
    """Views containing the live rows produced by :func:`stage_inputs`."""

    data: torch.Tensor
    scales: torch.Tensor
    route_ids: torch.Tensor
    route_scores: torch.Tensor
    num_tokens: int


def _shim_helpers():
    # Keep the CuTeDSL/vendor import lazy: validation and host tests stay
    # usable without importing cutlass or constructing a CUDA context.
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        fused_quant_stage,
        fused_quant_stage_supported,
        mxfp8_quantize_per_block_32,
        note_staged_tokens,
    )

    return (
        fused_quant_stage,
        fused_quant_stage_supported,
        mxfp8_quantize_per_block_32,
        note_staged_tokens,
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    ndim: int = 2,
    device: torch.device | None = None,
) -> None:
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in dtypes:
        expected = ", ".join(str(value) for value in dtypes)
        raise ValueError(f"{name} must have dtype {expected}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its trailing dimension")


def validate_staging_inputs(
    hidden_states: torch.Tensor,
    scales: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    data_out: torch.Tensor,
    scales_out: torch.Tensor,
    route_ids_out: torch.Tensor,
    route_scores_out: torch.Tensor,
    *,
    quantize_input: bool,
) -> tuple[int, int, int]:
    """Validate staging tensors and return ``(tokens, hidden, topk)``."""

    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be 2D, got shape {tuple(hidden_states.shape)}"
        )
    num_tokens, hidden = hidden_states.shape
    if hidden <= 0 or hidden % MXFP8_BLOCK_SIZE:
        raise ValueError(f"hidden must be a positive multiple of 32, got {hidden}")
    input_dtype = torch.bfloat16 if quantize_input else MXFP8_DATA_DTYPE
    _check_tensor("hidden_states", hidden_states, dtype=input_dtype)
    device = hidden_states.device

    # DataPreprocess consumes Int32 and widens to the kernel's Int64 routing
    # workspace.  Accepting Int64 here would compile a different CuTe tensor
    # type against that fixed kernel contract.
    _check_tensor("topk_ids", topk_ids, dtype=torch.int32, device=device)
    _check_tensor("topk_scores", topk_scores, dtype=torch.float32, device=device)
    if topk_ids.shape != topk_scores.shape or topk_ids.shape[0] != num_tokens:
        raise ValueError(
            "topk_ids and topk_scores must have identical [tokens, topk] shapes"
        )
    topk = topk_ids.shape[1]
    if topk <= 0:
        raise ValueError("topk must be positive")

    _check_tensor("data_out", data_out, dtype=MXFP8_DATA_DTYPE, device=device)
    _check_tensor("scales_out", scales_out, dtype=MXFP8_SCALE_DTYPE, device=device)
    _check_tensor("route_ids_out", route_ids_out, dtype=torch.int64, device=device)
    _check_tensor(
        "route_scores_out", route_scores_out, dtype=torch.float32, device=device
    )
    capacity = data_out.shape[0]
    sf_cols = hidden // MXFP8_BLOCK_SIZE
    if capacity < num_tokens:
        raise ValueError(f"staging capacity {capacity} is smaller than {num_tokens}")
    if data_out.shape[1] != hidden:
        raise ValueError(f"data_out must have shape [capacity, {hidden}]")
    if scales_out.shape[0] != capacity or scales_out.shape[1] < sf_cols:
        raise ValueError(
            f"scales_out must have shape [capacity, >= {sf_cols}], got "
            f"{tuple(scales_out.shape)}"
        )
    expected_routes = (capacity, topk)
    if (
        route_ids_out.shape != expected_routes
        or route_scores_out.shape != expected_routes
    ):
        raise ValueError(f"routing outputs must have shape {expected_routes}")

    if quantize_input:
        if scales is not None:
            raise ValueError("scales must be None when quantize_input=True")
    else:
        if scales is None:
            raise ValueError("prequantized MXFP8 input requires E8M0 scales")
        _check_tensor("scales", scales, dtype=MXFP8_SCALE_DTYPE, device=device)
        if scales.shape[0] != num_tokens or scales.shape[1] < sf_cols:
            raise ValueError(
                f"scales must have shape [{num_tokens}, >= {sf_cols}], got "
                f"{tuple(scales.shape)}"
            )
    return num_tokens, hidden, topk


def stage_inputs(
    hidden_states: torch.Tensor,
    scales: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    data_out: torch.Tensor,
    scales_out: torch.Tensor,
    route_ids_out: torch.Tensor,
    route_scores_out: torch.Tensor,
    *,
    quantize_input: bool,
) -> StagedActivation:
    """Stage BF16 or caller-prequantized MXFP8 inputs into dispatch buffers.

    BF16 uses the existing one-launch CuTeDSL quantize-and-route stager when
    its alignment constraints are met.  The Torch fallback implements the
    same E4M3 + raw-E8M0/K32 representation.  Prequantized input is copied
    byte-for-byte; it is never requantized.
    """

    num_tokens, hidden, _ = validate_staging_inputs(
        hidden_states,
        scales,
        topk_ids,
        topk_scores,
        data_out,
        scales_out,
        route_ids_out,
        route_scores_out,
        quantize_input=quantize_input,
    )
    sf_cols = hidden // MXFP8_BLOCK_SIZE
    capacity = data_out.shape[0]
    fused_stage, fused_supported, quantize, note_tokens = _shim_helpers()

    if num_tokens == 0:
        route_ids_out.fill_(-1)
        note_tokens(route_ids_out, 0)
    elif (
        quantize_input
        and os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and fused_supported(hidden_states, quant_type="mxfp8_e4m3")
    ):
        fused_stage(
            hidden_states,
            topk_ids,
            topk_scores,
            data_out,
            scales_out[:, :sf_cols],
            route_ids_out,
            route_scores_out,
            quant_type="mxfp8_e4m3",
        )
    else:
        if quantize_input:
            quantized, input_scales = quantize(
                hidden_states.to(torch.float32), MXFP8_DATA_DTYPE
            )
        else:
            quantized = hidden_states
            assert scales is not None
            input_scales = scales[:, :sf_cols]

        data_out[:num_tokens].view(torch.uint8).copy_(quantized.view(torch.uint8))
        scales_out[:num_tokens].zero_()
        scales_out[:num_tokens, :sf_cols].view(torch.uint8).copy_(
            input_scales.view(torch.uint8)
        )
        route_ids_out[:num_tokens].copy_(topk_ids)
        route_scores_out[:num_tokens].copy_(topk_scores)
        if num_tokens < capacity:
            route_ids_out[num_tokens:capacity].fill_(-1)
        note_tokens(route_ids_out, num_tokens)

    return StagedActivation(
        data_out[:num_tokens],
        scales_out[:num_tokens, :sf_cols],
        route_ids_out[:num_tokens],
        route_scores_out[:num_tokens],
        num_tokens,
    )


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "MXFP8_DATA_DTYPE",
    "MXFP8_SCALE_DTYPE",
    "StagedActivation",
    "stage_inputs",
    "validate_staging_inputs",
]
