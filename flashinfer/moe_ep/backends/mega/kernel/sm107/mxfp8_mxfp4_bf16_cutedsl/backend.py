"""Rubin MegaMoE with MXFP4 weights and MXFP8 E4M3 activations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mxfp4_mxfp4_bf16_cutedsl.config import (
        Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

from ......core.kernel.registry import register_mega_kernel
from ..mxfp8_mxfp8_bf16_cutedsl.backend import Sm107Mxfp8BlockScaledMegaKernelBackend
from .config import Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights
from ..shared_weights import (
    Sm107PrequantizedMoEWeightsWithSharedExpert,
    Sm107SharedMegaWeights,
    preprocess_shared_weights,
)


@register_mega_kernel("sm107_mxfp8_mxfp4_bf16_cutedsl")
class Sm107Mxfp8Mxfp4MegaKernelBackend(Sm107Mxfp8BlockScaledMegaKernelBackend):
    """Reuse MXFP8 staging and communication with packed FP4 expert weights."""

    def __init__(
        self,
        config: Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        | Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    ) -> None:
        super().__init__(config)

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_mxfp8_mxfp4_bf16_cutedsl"

    def preprocess_weights(self, weights, fleet_params):
        shared = isinstance(weights, Sm107PrequantizedMoEWeightsWithSharedExpert)
        if shared != getattr(self._kernel_config, "shared_expert", False):
            raise ValueError("shared_expert must match the supplied weight container")
        routed = preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )
        if not shared:
            return routed
        return Sm107SharedMegaWeights(
            routed,
            preprocess_shared_weights(
                weights.shared_w13,
                weights.shared_w2,
                weights.shared_w13_scale,
                weights.shared_w2_scale,
                hidden_size=fleet_params.token_hidden_size,
                intermediate_size=self._kernel_config.intermediate_size,
            ),
        )

    def validate_transformed_weights(
        self, transformed_weights, bootstrap, fleet_params
    ):
        shared = isinstance(transformed_weights, Sm107SharedMegaWeights)
        if shared != getattr(self._kernel_config, "shared_expert", False):
            raise ValueError("shared_expert must match the prepared weight container")
        validate_transformed_mega_weights(
            transformed_weights.routed if shared else transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )
        if shared:
            from ..mxfp8_mxfp8_bf16_cutedsl.weights import (
                validate_transformed_mega_weights as validate_shared,
            )

            validate_shared(
                transformed_weights.shared,
                intermediate_size=self._kernel_config.intermediate_size,
                hidden_size=fleet_params.token_hidden_size,
                kind="mxfp8_e4m3",
                world_size=1,
                num_experts=1,
            )
