"""DeepGEMM mega-MoE kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.validation.common import (
    validate_mega_arch,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
)
from .....weights import MoEWeightPack
from .config import DeepGemmMegaMoeConfig
from .staging import stage_mega_moe_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


@register_mega_kernel("deep_gemm_mega")
class DeepGemmMegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: DeepGemmMegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: DeepGemmMegaMoeConfig = config

    @classmethod
    def kernel_name(cls) -> str:
        return "deep_gemm_mega"

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
        )

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

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        import deep_gemm

        k = self._kernel_config
        fp = fleet_params
        return deep_gemm.get_symm_buffer_for_mega_moe(
            self.ep_comm_group,
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_mega_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        num_tokens = t.hidden_states.shape[0]
        if quantize_input:
            x_slot = workspace.x[:num_tokens]
            if x_slot.dtype != torch.float8_e4m3fn:
                x_slot = x_slot.view(torch.float8_e4m3fn)
            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                x_slot,
                workspace.x_sf[:num_tokens],
                workspace.topk_idx[:num_tokens],
                workspace.topk_weights[:num_tokens],
            )
            return

        workspace.x[:num_tokens].copy_(t.hidden_states)
        assert t.scales is not None
        workspace.x_sf[:num_tokens].copy_(t.scales)
        workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
        workspace.topk_weights[:num_tokens].copy_(t.topk_weights)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm

        kcfg = self._kernel_config
        deep_gemm.fp8_fp4_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            activation_clamp=kcfg.activation_clamp,
            fast_math=kcfg.fast_math,
        )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is not None:
            workspace.destroy()
