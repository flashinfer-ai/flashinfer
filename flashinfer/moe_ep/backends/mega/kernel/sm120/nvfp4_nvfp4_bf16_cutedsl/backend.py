"""FlashInfer MegaKernelBackend for SM120 NVFP4 x NVFP4 Split-MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import nvfp4_cutedsl_runtime_requirements
from ......core.validation.common import (
    validate_mega_arch_sm120,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
from .staging import validate_forward_inputs
from .weights import (
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors
    from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        TransformedWeights,
    )


def _effective_clamp(
    config: Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None and config.activation_clamp is not None:
        if config.gate_up_clamp != config.activation_clamp:
            raise ValueError("gate_up_clamp and activation_clamp disagree")
    return (
        config.gate_up_clamp
        if config.gate_up_clamp is not None
        else config.activation_clamp
    )


@register_mega_kernel("sm120_nvfp4_nvfp4_bf16_cutedsl")
class Sm120Nvfp4Nvfp4CutedslMegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config
        if config.intermediate_size <= 0 or config.top_k <= 0:
            raise ValueError("intermediate_size and top_k must be positive")
        if not config.fast_math:
            raise NotImplementedError("fast_math=False is not supported")

    @classmethod
    def kernel_name(cls) -> str:
        return "sm120_nvfp4_nvfp4_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return nvfp4_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch_sm120()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=32,
        )

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> "TransformedWeights":
        return preprocess_mega_weights(
            weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: "TransformedWeights",
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            local_experts=fleet_params.num_experts // self.ep_world_size,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
            MegaMoESm120Nvfp4Config,
            allocate_workspace,
        )

        config = self._kernel_config
        return allocate_workspace(
            MegaMoESm120Nvfp4Config(
                rank=self.ep_rank,
                world_size=self.ep_world_size,
                max_tokens_per_rank=fleet_params.max_tokens_per_rank,
                num_topk=config.top_k,
                num_total_experts=fleet_params.num_experts,
                hidden=fleet_params.token_hidden_size,
                intermediate=config.intermediate_size,
                gate_up_clamp=_effective_clamp(config),
                input_norm_const=config.input_norm_const,
                data_parallel_size=config.data_parallel_size,
                tensor_parallel_size=config.tensor_parallel_size,
                knobs=config.knobs,
            ),
            # MEGA_NO_DIST=1 single-rank sessions intentionally have no
            # process group. Multi-rank prepare_workspace() has already
            # resolved this optional group through _ensure_ep_bootstrap().
            control_group=self._ep_comm_group,
        )

    def validate_forward(
        self,
        tensors: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_forward_inputs(
            tensors.hidden_states,
            tensors.topk_ids,
            tensors.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=tensors.scales,
        )

    def set_compile_tokens_per_rank(
        self,
        workspace: Any,
        compile_tokens_per_rank: int | None,
    ) -> None:
        from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
            set_compile_tokens_per_rank,
        )

        set_compile_tokens_per_rank(workspace, compile_tokens_per_rank)

    def stage_inputs(
        self,
        tensors: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import stage_inputs

        compile_bucket = workspace._compile_bucket
        stage_inputs(
            tensors.hidden_states,
            tensors.topk_weights,
            tensors.topk_ids,
            workspace.x[:compile_bucket],
            workspace.x_scale[:compile_bucket],
            workspace.topk_ids[:compile_bucket],
            workspace.topk_weights[:compile_bucket],
            quantize_input=quantize_input,
            scales=tensors.scales,
            norm_const=self._kernel_config.input_norm_const,
        )
        for name, source in (
            ("fc1_alpha", tensors.fc1_alpha),
            ("fc2_alpha", tensors.fc2_alpha),
            ("fc1_norm_const", tensors.fc1_norm_const),
        ):
            target = getattr(workspace, name)
            if source is None:
                target.fill_(1.0)
            else:
                if source.shape != target.shape or source.dtype != torch.float32:
                    raise ValueError(
                        f"{name} must be float32 with shape {tuple(target.shape)}"
                    )
                target.copy_(source)
        workspace._staged_tokens = tensors.hidden_states.shape[0]

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return workspace.output

    def compute(
        self,
        workspace: Any,
        transformed_weights: "TransformedWeights",
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
            run_split_mega_moe,
        )

        direct_output = (
            output
            if output is not None
            and output.shape[0] >= workspace.config.max_tokens_per_rank
            else None
        )
        full_output = run_split_mega_moe(
            workspace,
            transformed_weights,
            output=direct_output,
        )
        live = full_output[: workspace._staged_tokens]
        if output is not None and direct_output is None:
            output.copy_(live)
            return output
        return live

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        from ......core.kernel.workspace_pool import knobs_pool_key

        config = self._kernel_config
        return (
            "sm120_nvfp4_nvfp4_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            config.top_k,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            _effective_clamp(config),
            config.input_norm_const,
            config.data_parallel_size,
            config.tensor_parallel_size,
            knobs_pool_key(config.knobs),
        )


__all__ = ["Sm120Nvfp4Nvfp4CutedslMegaKernelBackend"]
