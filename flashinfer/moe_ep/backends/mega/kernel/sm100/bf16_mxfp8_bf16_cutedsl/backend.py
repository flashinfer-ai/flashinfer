"""CuTeDSL mixed MXFP8-weight/BF16-activation MegaMoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import bf16_mxfp8_cutedsl_runtime_requirements
from ......core.validation.common import validate_mega_arch, validate_mega_fleet_params
from ......weights import MoEWeightPack
from ..common.bf16_staging import (
    stage_mega_moe_inputs,
    validate_bf16_forward_inputs,
)
from .config import Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _clamp(config: Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig) -> float | None:
    return (
        config.gate_up_clamp
        if config.gate_up_clamp is not None
        else config.activation_clamp
    )


@register_mega_kernel("sm100_bf16_mxfp8_bf16_cutedsl")
class Bf16Mxfp8CutedslMegaKernelBackend(MegaKernelBackend):
    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_bf16_mxfp8_bf16_cutedsl"

    def __init__(self, config: Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config
        self._autotune_pending = config.knobs == "auto"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return bf16_mxfp8_cutedsl_runtime_requirements(bootstrap)

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
            raise ValueError("mixed MegaMoE requires hidden size divisible by 32.")
        if self._kernel_config.intermediate_size % 64:
            raise ValueError(
                "mixed MegaMoE requires intermediate size divisible by 64."
            )

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedMegaWeights:
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            kind=self._kernel_config.kind,
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
            kind=self._kernel_config.kind,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.cutedsl_megamoe import (
            get_symm_buffer_for_bf16_mxfp8_mega_moe,
        )

        config = self._kernel_config
        return get_symm_buffer_for_bf16_mxfp8_mega_moe(
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            kind=config.kind,
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
        del quantize_input
        validate_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
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
        from ......kernel_src.cutedsl_megamoe import bf16_mxfp8_mega_moe

        if self._autotune_pending:
            raise NotImplementedError("mixed MegaMoE autotuning is not implemented.")
        bf16_mxfp8_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=output.shape[0],
            gate_up_clamp=_clamp(self._kernel_config),
            fast_math=self._kernel_config.fast_math,
        )
        return output

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        config = self._kernel_config
        if config.knobs == "auto":
            return None

        from ......core.kernel.workspace_pool import knobs_pool_key

        return (
            "sm100_bf16_mxfp8_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.kind,
            _clamp(config),
            config.in_kernel_fc2_reduce,
            config.token_back_mode,
            knobs_pool_key(config.knobs),
        )
