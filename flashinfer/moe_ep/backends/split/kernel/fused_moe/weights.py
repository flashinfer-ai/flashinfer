"""Materialize canonical EP weights into fused_moe native views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig, MoEWeightPack as FusedMoEWeightPack


def materialize_fused_moe_weights(
    weights: MoEWeightPack,
    moe_config: "MoEConfig",
) -> "FusedMoEWeightPack":
    """Prepare each configured EP compute backend's native weight view.

    MoELayer compares the eligible candidates at execution time, so every
    candidate needs its view even when it is not first in the search order.
    Repeated configurations of the same backend share one weight view.
    """
    if isinstance(weights, PrequantizedMoEWeights):
        raise NotImplementedError(
            "The split fused_moe kernel prepares every backend's weights from "
            "canonical BF16; PrequantizedMoEWeights are not supported"
        )

    from ......fused_moe.api import (
        CuteDslConfig,
        MoEWeightPack as FusedMoEWeightPack,
        PrimsTsConfig,
        QuantFormat,
        TrtllmBf16Config,
        TrtllmFp4Config,
        TrtllmFp8BlockConfig,
    )

    quant = moe_config.quant
    # FP8PerTensor needs calibrated global scales, which MoEWeightPack does
    # not carry.
    prims_ts_pairs = (
        (QuantFormat.NVFP4, QuantFormat.NVFP4),
        (QuantFormat.MXFP4, QuantFormat.MXFP8),
        (QuantFormat.MXFP4, QuantFormat.BF16),
        (QuantFormat.BF16, QuantFormat.BF16),
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
    )
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
            if "trtllm_bf16_routed" in pack.native_views:
                continue
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
            continue

        if quant.pair in (
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
            (QuantFormat.MXFP4, QuantFormat.MXFP8),
        ) and isinstance(backend_cfg, TrtllmFp4Config):
            if "trtllm_fp4_routed" in pack.native_views:
                continue
            view = TrtllmFp4Config.prepare_weights(
                weights.w13,
                weights.w2,
                quant=quant,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_fp4_routed", view)
            continue

        if quant.pair == (
            QuantFormat.DeepSeekFp8,
            QuantFormat.DeepSeekFp8,
        ) and isinstance(backend_cfg, TrtllmFp8BlockConfig):
            if "trtllm_fp8_block" in pack.native_views:
                continue
            view = TrtllmFp8BlockConfig.prepare_weights(
                weights.w13,
                weights.w2,
                quant=quant,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_fp8_block", view)
            continue

        if quant.pair in (
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
            (QuantFormat.MXFP4, QuantFormat.MXFP8),
            (QuantFormat.NVFP4, QuantFormat.BF16),
        ) and isinstance(backend_cfg, CuteDslConfig):
            if "cute_dsl" in pack.native_views:
                continue
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
            continue

        if quant.pair in prims_ts_pairs and isinstance(backend_cfg, PrimsTsConfig):
            if "prims_ts" in pack.native_views:
                continue
            view = PrimsTsConfig.prepare_weights(
                weights.w13,
                weights.w2,
                quant=quant,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("prims_ts", view)

    if pack.native_views:
        return pack

    raise ValueError(
        f"No fused_moe backend in MoEConfig matches quant {quant!r}. "
        f"Configured backends: {[type(c).__name__ for c in moe_config.backend]}"
    )
