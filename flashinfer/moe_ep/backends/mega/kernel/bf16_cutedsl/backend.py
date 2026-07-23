"""CuTeDSL BF16 MegaMoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import bf16_cutedsl_runtime_requirements
from .....core.validation.common import validate_mega_arch, validate_mega_fleet_params
from .....weights import MoEWeightPack
from .config import Bf16CutedslMegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_bf16_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


def _clamp(config: Bf16CutedslMegaMoeConfig) -> float | None:
    return (
        config.gate_up_clamp
        if config.gate_up_clamp is not None
        else config.activation_clamp
    )


@register_mega_kernel("bf16_cutedsl")
class Bf16CutedslMegaKernelBackend(MegaKernelBackend):
    @classmethod
    def kernel_name(cls) -> str:
        return "bf16_cutedsl"

    def __init__(self, config: Bf16CutedslMegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config
        self._autotune_pending = config.knobs == "auto"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return bf16_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
        )
        if fleet_params.token_hidden_size % 32:
            raise ValueError("BF16 MegaMoE requires hidden size divisible by 32.")
        if self._kernel_config.intermediate_size % 64:
            raise ValueError("BF16 MegaMoE requires intermediate size divisible by 64.")

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
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

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from .....kernel_src.cutedsl_megamoe import get_symm_buffer_for_bf16_mega_moe

        config = self._kernel_config
        return get_symm_buffer_for_bf16_mega_moe(
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=_clamp(config),
            in_kernel_fc2_reduce=config.in_kernel_fc2_reduce,
            token_back_mode=config.token_back_mode,
            knobs=config.knobs if isinstance(config.knobs, dict) else None,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def stage_inputs(
        self, t: "MoEEpTensors", workspace: Any, *, quantize_input: bool
    ) -> None:
        del quantize_input
        stage_mega_moe_inputs(
            t.hidden_states,
            t.topk_weights,
            t.topk_ids,
            workspace.x,
            workspace.topk_idx,
            workspace.topk_weights,
        )

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from .....kernel_src.cutedsl_megamoe import bf16_mega_moe

        if self._autotune_pending:
            from .....kernel_src.cutedsl_megamoe import autotune_bf16_mega_moe

            autotune_bf16_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=output.shape[0],
                gate_up_clamp=_clamp(self._kernel_config),
                activation_clamp=self._kernel_config.activation_clamp,
            )
            self._autotune_pending = False
        bf16_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=output.shape[0],
            gate_up_clamp=_clamp(self._kernel_config),
            fast_math=self._kernel_config.fast_math,
        )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is not None:
            workspace.destroy()
