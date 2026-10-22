"""Materialize canonical EP weights into fused_moe native views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig, MoEWeightPack as FusedMoEWeightPack


def materialize_fused_moe_weights(
    weights: MoEWeightPack,
    moe_config: "MoEConfig",
) -> "FusedMoEWeightPack":
    """Convert canonical :class:`MoEWeightPack` into a fused_moe weight pack."""
    from ......fused_moe.api import (
        CuteDslConfig,
        MoEWeightPack as FusedMoEWeightPack,
        QuantFormat,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    quant = moe_config.quant
    experts = moe_config.experts
    routing = moe_config.routing
    num_local = experts.local_num_experts or routing.num_experts
    hidden = weights.w13.shape[-1]
    intermediate = experts.intermediate_size

    pack = FusedMoEWeightPack()

    for backend_cfg in moe_config.backend:
        if quant.pair == (QuantFormat.BF16, QuantFormat.BF16) and isinstance(
            backend_cfg, TrtllmBf16Config
        ):
            view = TrtllmBf16Config.prepare_weights(
                weights.w13,
                weights.w2,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_bf16_routed", view)
            return pack

        if quant.pair == (QuantFormat.NVFP4, QuantFormat.NVFP4) and isinstance(
            backend_cfg, TrtllmFp4Config
        ):
            view = TrtllmFp4Config.prepare_weights(
                weights.w13,
                weights.w2,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_fp4_routed", view)
            return pack

        if quant.pair in (
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
            (QuantFormat.MXFP4, QuantFormat.MXFP8),
            (QuantFormat.NVFP4, QuantFormat.BF16),
        ) and isinstance(backend_cfg, CuteDslConfig):
            view = CuteDslConfig.prepare_weights(
                weights.w13,
                weights.w2,
                quant=quant,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("cute_dsl", view)
            return pack

    raise ValueError(
        f"No fused_moe backend in MoEConfig matches quant {quant!r}. "
        f"Configured backends: {[type(c).__name__ for c in moe_config.backend]}"
    )
