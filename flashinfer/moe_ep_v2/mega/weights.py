"""Mega-path weight preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..weights import MoEWeightPack

if TYPE_CHECKING:
    import torch

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """bf16 (or fp4+scale) weights → layout expected by ``fp8_fp4_mega_moe``."""
    import torch
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    def _to_fp4(bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16.shape
        w = torch.empty((num_groups, n, k // 2), device=bf16.device, dtype=torch.int8)
        w_sf = torch.empty(
            (num_groups, n, k // 32), device=bf16.device, dtype=torch.float32
        )
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(
                bf16[i], use_ue8m0=True, gran_k=32
            )
        w_sf = deep_gemm.transform_sf_into_required_layout(
            w_sf, n, k, (1, 32), num_groups
        )
        return w, w_sf

    if weights.w13_scale is None or weights.w2_scale is None:
        l1 = _to_fp4(weights.w13)
        l2 = _to_fp4(weights.w2)
    else:
        l1 = (weights.w13, weights.w13_scale)
        l2 = (weights.w2, weights.w2_scale)

    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


__all__ = ["MoEWeightPack", "TransformedMegaWeights", "preprocess_mega_weights"]
