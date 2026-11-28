"""Rubin MegaMoE with MXFP4 weights and MXFP8 E4M3 activations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ......config import FleetParams
    from ......tensors import EplbRouting, MoEEpTensors
    from ..mxfp4_mxfp4_bf16_cutedsl.config import (
        Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

from ......core.kernel.registry import register_mega_kernel
from ......core.validation.common import MoEEpConfigError
from ..mxfp8_mxfp8_bf16_cutedsl.backend import Sm107Mxfp8BlockScaledMegaKernelBackend
from ..shared_expert import (
    prepare_shared_weights,
    validate_shared_weights,
)
from .config import Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights


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

    def validate_eplb_routing(
        self,
        t: MoEEpTensors,
        fleet_params: FleetParams,
        eplb_routing: EplbRouting,
        *,
        quantize_input: bool,
    ) -> None:
        if (
            not quantize_input
            or self._kernel_config.kind != "mxfp4_mxfp8"
            or self._kernel_config.knobs == "auto"
            or self._kernel_config.batch_invariant
            or os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") == "0"
            or os.environ.get("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0") != "0"
        ):
            raise MoEEpConfigError(
                "EPLB routing requires fixed SM107 fused MXFP8 input staging with "
                "batch invariance and route checks disabled"
            )
        if (
            eplb_routing.expert_load.numel() != fleet_params.num_experts
            or eplb_routing.expert_load.device != t.hidden_states.device
        ):
            raise MoEEpConfigError(
                "EPLB routing requires one load counter per physical expert "
                "on the input CUDA device"
            )

    def stage_inputs_with_eplb(
        self,
        t: MoEEpTensors,
        workspace: Any,
        eplb_routing: EplbRouting,
    ) -> None:
        self.stage_inputs(t, workspace, quantize_input=True, eplb_routing=eplb_routing)

    def preprocess_weights(self, weights, fleet_params):
        routed = preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )
        return prepare_shared_weights(
            self._kernel_config, weights, routed, fleet_params
        )

    def validate_transformed_weights(
        self, transformed_weights, bootstrap, fleet_params
    ):
        transformed_weights = validate_shared_weights(
            self._kernel_config, transformed_weights, fleet_params
        )
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )
