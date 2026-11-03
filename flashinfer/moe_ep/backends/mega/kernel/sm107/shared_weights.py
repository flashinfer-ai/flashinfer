# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare one replicated E4M3 shared expert for Rubin FC12 execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .....weights import PrequantizedMoEWeights


@dataclass(frozen=True)
class Sm107PrequantizedMoEWeightsWithSharedExpert(PrequantizedMoEWeights):
    """Routed MXFP4 weights and one replicated E4M3 shared expert."""

    shared_w13: torch.Tensor
    shared_w2: torch.Tensor
    shared_w13_scale: torch.Tensor
    shared_w2_scale: torch.Tensor

    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)


@dataclass(frozen=True)
class Sm107SharedMegaWeights:
    """Prepared routed and shared weights, retained independently of EPLB slots."""

    routed: TransformedMegaWeights
    shared: TransformedMegaWeights


def validate_shared_backend(weights, config) -> None:
    if isinstance(
        weights,
        (Sm107PrequantizedMoEWeightsWithSharedExpert, Sm107SharedMegaWeights),
    ) and (
        getattr(config, "kernel_name", None) != "sm107_mxfp8_mxfp4_bf16_cutedsl"
        or not getattr(config, "shared_expert", False)
    ):
        raise ValueError(
            "SM107 shared weights require the MXFP4/MXFP8 backend with shared_expert=True"
        )


def bind_prepared_weights(workspace, weights):
    """Bind shared ownership once; leave the routed-only payload unchanged."""
    if isinstance(weights, Sm107SharedMegaWeights):
        workspace.bind_shared_weights(weights.shared)
        return weights.routed
    if getattr(getattr(workspace, "config", None), "shared_quant_kind", None):
        raise ValueError("Shared execution requires prepared routed and shared weights")
    return weights


if TYPE_CHECKING:
    import torch

    from .mxfp8_mxfp8_bf16_cutedsl.weights import TransformedMegaWeights


def preprocess_shared_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> TransformedMegaWeights:
    """Preserve E4M3 payloads and per-32 E8M0 scales while changing layout.

    Inputs describe one unsharded expert: gate/up rows are consecutive halves
    of ``w13[2 * intermediate_size, hidden_size]`` and ``w2`` has shape
    ``[hidden_size, intermediate_size]``. The returned tensors carry a singleton
    expert dimension; it does not represent a routed or EPLB expert slot.
    """
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.kernel_helpers import (
        preprocess_prequantized_block_scaled_weights,
    )

    if any(t.ndim != 2 for t in (w13, w2, w13_scale, w2_scale)):
        raise ValueError("shared weights and scale planes must be two-dimensional")
    return preprocess_prequantized_block_scaled_weights(
        w13.unsqueeze(0),
        w2.unsqueeze(0),
        w13_scale.unsqueeze(0),
        w2_scale.unsqueeze(0),
        quant_kind="mxfp8_e4m3",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
