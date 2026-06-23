"""Materialize canonical EP weights into fused_moe native views."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig, MoEWeightPack as FusedMoEWeightPack


def _block_major_k(w: torch.Tensor) -> torch.Tensor:
    """BlockMajorK shuffle for the trtllm bf16 routed runner (per-expert)."""
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe.core import convert_to_block_layout

    epilogue_tile_m = 64
    block_k = 128
    shuffled = []
    for i in range(w.shape[0]):
        s = shuffle_matrix_a(w[i].view(torch.uint8), epilogue_tile_m)
        s = convert_to_block_layout(s, block_k)
        shuffled.append(s)
    return torch.stack(shuffled).view(torch.bfloat16)


def materialize_fused_moe_weights(
    weights: MoEWeightPack,
    moe_config: "MoEConfig",
) -> "FusedMoEWeightPack":
    """Convert canonical :class:`MoEWeightPack` into a fused_moe weight pack."""
    from ......fused_moe.api import (
        CuteDslConfig,
        MoEWeightPack as FusedMoEWeightPack,
        QuantVariant,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    variant = moe_config.quant.variant
    experts = moe_config.experts
    routing = moe_config.routing
    num_local = experts.local_num_experts or routing.num_experts
    hidden = weights.w13.shape[-1]
    intermediate = weights.w13.shape[-2] // 2

    pack = FusedMoEWeightPack()

    for backend_cfg in moe_config.backend:
        if variant == QuantVariant.BF16 and isinstance(backend_cfg, TrtllmBf16Config):
            pack.prepare_for(
                "trtllm_bf16_routed",
                {
                    "gemm1_weights": _block_major_k(weights.w13),
                    "gemm2_weights": _block_major_k(weights.w2),
                },
            )
            return pack

        if variant == QuantVariant.NVFP4 and isinstance(backend_cfg, TrtllmFp4Config):
            view = TrtllmFp4Config.prepare_weights(
                weights.w13,
                weights.w2,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_fp4_routed", view)
            return pack

        if variant == QuantVariant.NVFP4 and isinstance(backend_cfg, CuteDslConfig):
            view = CuteDslConfig.prepare_weights(
                weights.w13,
                weights.w2,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                device=weights.w13.device,
            )
            pack.prepare_for("cute_dsl_nvfp4", view)
            return pack

    raise ValueError(
        f"No fused_moe backend in MoEConfig matches quant variant {variant!r}. "
        f"Configured backends: {[type(c).__name__ for c in moe_config.backend]}"
    )
