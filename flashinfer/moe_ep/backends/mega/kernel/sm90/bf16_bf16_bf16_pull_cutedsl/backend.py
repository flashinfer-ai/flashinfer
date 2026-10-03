"""SM90 (Hopper) pull-style BF16 mega-MoE kernel backend.

The FP8 pull-style megakernel compiled with BF16 operands (BF16 WGMMA, FP32
accumulation, BF16 FC1 output).  Workspace, compute, autotune and pooling
are the FP8 backend's: the config pins ``kind="bf16"`` and the per-tensor
ABI with unit scales.  Only the weight layout, staging and forward
validation differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......weights import MoEWeightPack
from ...sm100.common.bf16_staging import validate_bf16_forward_inputs
from ..fp8_fp8_bf16_pull_cutedsl.backend import Sm90PullFp8MegaKernelBackend
from .config import Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig
from .staging import stage_mega_moe_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


@register_mega_kernel("sm90_bf16_bf16_bf16_pull_cutedsl")
class Sm90PullBf16MegaKernelBackend(Sm90PullFp8MegaKernelBackend):
    def __init__(self, config: Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_bf16_bf16_bf16_pull_cutedsl"

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        # BF16 activations are copied, never quantized: quantize_input is
        # ignored, as on the other BF16-activation backends.
        validate_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            scales=t.scales,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        stage_mega_moe_inputs(
            t.hidden_states,
            t.topk_weights,
            t.topk_ids,
            workspace.x,
            workspace.topk_idx,
            workspace.topk_weights,
        )
