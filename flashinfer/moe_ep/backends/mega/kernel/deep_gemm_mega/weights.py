"""Mega-path weight preprocessing."""

from __future__ import annotations

from typing import Tuple

import torch

from .....weights import MoEWeightPack

TransformedMegaWeights = Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)


def _weight_sf_to_float(w_sf: torch.Tensor) -> torch.Tensor:
    if w_sf.dtype == torch.uint8:
        return _ue8m0_uint8_to_float(w_sf)
    return w_sf.contiguous()


def _as_fp4_weight(w: torch.Tensor) -> torch.Tensor:
    if w.dtype in (torch.uint8, torch.int8):
        return w.view(torch.int8).contiguous()
    raise TypeError(
        "pre-quantized MoEWeightPack weights must be fp4 (torch.int8 or torch.uint8); "
        f"got {w.dtype}. Pass w13_scale/w2_scale together, or supply bf16 weights "
        "without scales to quantize during preprocess."
    )


def _quantize_grouped_fp4(
    bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm.utils import per_token_cast_to_fp4

    num_groups, n, k = bf16.shape
    w = torch.empty((num_groups, n, k // 2), device=bf16.device, dtype=torch.int8)
    w_sf = torch.empty(
        (num_groups, n, k // 32), device=bf16.device, dtype=torch.float32
    )
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16[i], use_ue8m0=True, gran_k=32)
    return w, w_sf


def _transform_weight_sf(
    w: torch.Tensor,
    w_sf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_groups, n, k_fp4 = w.shape
    k = k_fp4 * 2
    return w, deep_gemm.transform_sf_into_required_layout(
        w_sf, n, k, (1, 32), num_groups
    )


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """User fp4+scale (or bf16) weights → layout expected by ``fp8_fp4_mega_moe``."""
    import deep_gemm

    if weights.w13_scale is None or weights.w2_scale is None:
        l1 = _transform_weight_sf(*_quantize_grouped_fp4(weights.w13))
        l2 = _transform_weight_sf(*_quantize_grouped_fp4(weights.w2))
    else:
        l1 = _transform_weight_sf(
            _as_fp4_weight(weights.w13),
            _weight_sf_to_float(weights.w13_scale),
        )
        l2 = _transform_weight_sf(
            _as_fp4_weight(weights.w2),
            _weight_sf_to_float(weights.w2_scale),
        )

    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


__all__ = ["MoEWeightPack", "TransformedMegaWeights", "preprocess_mega_weights"]
