"""SM90 (Hopper) pull-style W4A16 (BF16 activations x NVFP4 weights) mega-MoE backend.

The BF16 pull-style backend with the kernel's ``weight_format="nvfp4"`` path:
packed NVFP4 weights decoded into register-sourced BF16 WGMMA (swap-AB).
Workspace, staging, compute and autotune are the BF16 backend's; this adds
the NVFP4 weight layout and the per-expert global scales (``fc*_alpha``),
which ride the per-tensor weight dequant slot of each leg.  Runtime
``MoEEpTensors.fc1_alpha`` / ``fc2_alpha`` are staged into workspace buffers
(stable pointers, so CUDA-graph safe) and override the config values for that
forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......core.validation.common import MoEEpConfigError
from ......weights import MoEWeightPack
from ..bf16_bf16_bf16_pull_cutedsl.backend import Sm90PullBf16MegaKernelBackend
from .config import Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


@register_mega_kernel("sm90_bf16_nvfp4_bf16_pull_cutedsl")
class Sm90PullW4A16MegaKernelBackend(Sm90PullBf16MegaKernelBackend):
    def __init__(self, config: Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        # Typed handle for the W4A16-only fields (fc1_alpha / fc2_alpha).
        self._w4a16_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_bf16_nvfp4_bf16_pull_cutedsl"

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        k = self._w4a16_config
        return preprocess_mega_weights(
            weights,
            intermediate_size=k.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            fc1_alpha=k.fc1_alpha,
            fc2_alpha=k.fc2_alpha,
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
        super().validate_forward(t, fleet_params, quantize_input=quantize_input)
        local_experts = fleet_params.num_experts // self.ep_world_size
        for name in ("fc1_alpha", "fc2_alpha"):
            alpha = getattr(t, name)
            if alpha is not None and (
                alpha.dtype != torch.float32
                or tuple(alpha.shape) != (local_experts,)
                or alpha.device != t.hidden_states.device
            ):
                raise MoEEpConfigError(
                    f"{name} must be FP32 [local_experts] on the activation device"
                )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        workspace = super()._allocate_workspace(fleet_params)
        local_experts = fleet_params.num_experts // self.ep_world_size
        workspace.w4a16_alpha = tuple(
            torch.ones(local_experts, dtype=torch.float32, device="cuda")
            for _ in range(2)
        )
        workspace.w4a16_runtime_alpha = (False, False)
        return workspace

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        super().stage_inputs(t, workspace, quantize_input=quantize_input)
        staged = []
        for source, destination in zip(
            (t.fc1_alpha, t.fc2_alpha), workspace.w4a16_alpha, strict=True
        ):
            if source is not None:
                destination.copy_(source)
            staged.append(source is not None)
        workspace.w4a16_runtime_alpha = tuple(staged)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        legs = tuple(
            (leg[0], leg[1], leg[2], staged_alpha) if use_staged else leg
            for leg, staged_alpha, use_staged in zip(
                transformed_weights,
                workspace.w4a16_alpha,
                workspace.w4a16_runtime_alpha,
                strict=True,
            )
        )
        return super().compute(workspace, legs, output=output)  # type: ignore[arg-type]
