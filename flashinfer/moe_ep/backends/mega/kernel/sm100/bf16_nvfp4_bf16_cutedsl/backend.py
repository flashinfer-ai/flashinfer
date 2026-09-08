"""Fused EP with packed NVFP4 weights and BF16 token transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import bf16_cutedsl_runtime_requirements
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
)
from ......weights import MoEWeightPack
from ..bf16_bf16_bf16_cutedsl.staging import stage_mega_moe_inputs
from .config import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


@register_mega_kernel("sm100_bf16_nvfp4_bf16_cutedsl")
class W4A16CutedslMegaKernelBackend(MegaKernelBackend):
    supports_global_weight_scales = True

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_bf16_nvfp4_bf16_cutedsl"

    def __init__(self, config: Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return bf16_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc not in ((10, 0), (10, 3)):
                raise MoEEpArchError("W4A16 MegaMoE requires SM100 or SM103")
        config = self._kernel_config
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=config.intermediate_size,
            top_k=config.top_k,
            alignment=32,
        )
        if config.intermediate_size % 64:
            raise MoEEpConfigError(
                "W4A16 MegaMoE requires intermediate size divisible by 64"
            )
        if config.top_k > min(32, fleet_params.num_experts):
            raise MoEEpConfigError(
                "W4A16 MegaMoE top_k must not exceed 32 or num_experts"
            )

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
            world_size=bootstrap.world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.cutedsl_megamoe import get_symm_buffer_for_w4a16_mega_moe

        config = self._kernel_config
        return get_symm_buffer_for_w4a16_mega_moe(
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=config.gate_up_clamp,
            knobs=config.knobs,
        )

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        config = self._kernel_config
        return (
            self.kernel_name(),
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self.ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.top_k,
            config.gate_up_clamp,
            tuple(sorted((config.knobs or {}).items())),
        )

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        if not quantize_input:
            raise MoEEpConfigError(
                "W4A16 MegaMoE accepts BF16 activations; keep MegaConfig.quantize_input=True"
            )
        if any(
            v is not None
            for v in (t.scales, t.fc1_alpha, t.fc2_alpha, t.fc1_norm_const)
        ):
            raise MoEEpConfigError(
                "W4A16 MegaMoE does not accept activation quantization fields; "
                "pass weight global scales in PrequantizedMoEWeights"
            )
        if t.hidden_states.ndim != 2 or t.topk_ids.ndim != 2:
            raise MoEEpConfigError("W4A16 activations and routing must be 2D")
        if t.hidden_states.dtype != torch.bfloat16:
            raise MoEEpConfigError("W4A16 MegaMoE hidden_states must be BF16")
        if t.topk_ids.dtype not in (torch.int32, torch.int64):
            raise MoEEpConfigError("W4A16 MegaMoE topk_ids must be int32 or int64")
        if t.topk_weights.dtype != torch.float32:
            raise MoEEpConfigError("W4A16 MegaMoE topk_weights must be FP32")
        if (
            t.topk_ids.device != t.hidden_states.device
            or t.topk_weights.device != t.hidden_states.device
        ):
            raise MoEEpConfigError("W4A16 activations and routing must share a device")
        validate_mega_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=True,
        )

    def stage_inputs(
        self, t: MoEEpTensors, workspace: Any, *, quantize_input: bool
    ) -> None:
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
        from ......kernel_src.cutedsl_megamoe import w4a16_mega_moe

        w4a16_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=output.shape[0],
            gate_up_clamp=self._kernel_config.gate_up_clamp,
            fast_math=self._kernel_config.fast_math,
        )
        return output
