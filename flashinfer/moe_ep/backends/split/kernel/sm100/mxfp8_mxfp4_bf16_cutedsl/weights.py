"""Quantize canonical bf16 expert weights into the CuTeDSL W4A8 layout.

The ``cute_dsl_fused_moe_mxfp8_mxfp4`` kernel consumes packed MXFP4 uint8
weights (E2M1 pairs) with block-32 UE8M0 scale factors in the CuTeDSL MMA
layout, and gemm1 weights with the SwiGLU linear/gate 64-row interleave
(matching ``flashinfer.fused_moe.prepare`` and the kernel's own tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TransformedSplitWeights:
    """Kernel-ready W4A8 weights plus the geometry derived from them."""

    w1_weight: "torch.Tensor"
    w1_weight_sf: "torch.Tensor"
    w1_alpha: "torch.Tensor"
    w2_weight: "torch.Tensor"
    w2_weight_sf: "torch.Tensor"
    w2_alpha: "torch.Tensor"
    num_local_experts: int
    hidden_size: int
    intermediate_size: int


def _interleave_linear_and_gate(w13: "torch.Tensor") -> "torch.Tensor":
    """Interleave the linear/gate halves of gemm1 weights in 64-row groups.

    Same reorder as ``flashinfer.fused_moe.prepare`` uses for the CuTeDSL
    NVFP4 path (``group_size=64`` on the ``2*intermediate`` dim).
    """
    experts, rows, k = w13.shape
    intermediate = rows // 2
    return (
        w13.view(experts, 2, intermediate // 64, 64, k)
        .transpose(1, 2)
        .contiguous()
        .view(experts, rows, k)
    )


def _quantize_mxfp4_grouped(
    weights: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """MXFP4-quantize ``[E, rows, k]`` bf16 weights to packed data + MMA scales."""
    import torch

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    experts, rows, k = weights.shape
    packed, scale = fp4_quantize(
        weights.reshape(experts * rows, k).contiguous(),
        global_scale=torch.ones(1, dtype=torch.float32, device=weights.device),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=True,
    )
    return packed.view(experts, rows, k // 2), convert_sf_to_mma_layout(
        scale,
        m=rows,
        k=k,
        num_groups=experts,
        sf_vec_size=32,
    )


def preprocess_split_weights(weights: MoEWeightPack) -> TransformedSplitWeights:
    """Quantize canonical bf16 ``w13``/``w2`` into the kernel-ready W4A8 pack."""
    import torch

    if isinstance(weights, PrequantizedMoEWeights):
        raise NotImplementedError(
            "sm100_mxfp8_mxfp4_bf16_cutedsl quantizes from canonical bf16 "
            "weights; pre-quantized MXFP4 weight packs are not wired yet."
        )
    w13, w2 = weights.w13, weights.w2
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError(
            "expected 3D [local_experts, 2*intermediate, hidden] w13 and "
            f"[local_experts, hidden, intermediate] w2; got {tuple(w13.shape)} "
            f"and {tuple(w2.shape)}"
        )
    num_local, two_i, hidden = w13.shape
    intermediate = two_i // 2
    if tuple(w2.shape) != (num_local, hidden, intermediate):
        raise ValueError(
            f"w2 shape {tuple(w2.shape)} is inconsistent with w13 "
            f"{tuple(w13.shape)} (expected "
            f"{(num_local, hidden, intermediate)})"
        )
    # The mixed kernel requires both GEMM K dims to be multiples of 128 and
    # the gemm1 interleave needs 64-row gate/linear groups.
    if hidden % 128 or intermediate % 128:
        raise ValueError(
            "sm100_mxfp8_mxfp4_bf16_cutedsl requires hidden and intermediate "
            f"sizes to be multiples of 128; got hidden={hidden}, "
            f"intermediate={intermediate}"
        )

    w13 = w13.to(torch.bfloat16)
    w2 = w2.to(torch.bfloat16)
    w1_weight, w1_weight_sf = _quantize_mxfp4_grouped(_interleave_linear_and_gate(w13))
    w2_weight, w2_weight_sf = _quantize_mxfp4_grouped(w2)
    ones = torch.ones(num_local, dtype=torch.float32, device=w13.device)
    return TransformedSplitWeights(
        w1_weight=w1_weight,
        w1_weight_sf=w1_weight_sf,
        w1_alpha=ones,
        w2_weight=w2_weight,
        w2_weight_sf=w2_weight_sf,
        w2_alpha=ones,
        num_local_experts=num_local,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
