"""CuTeDSL NVFP4 mega-MoE kernel backend.

The fused kernel consumes NVFP4 expert weights in kernel-ready layout (packed
weights + atom-swizzled scale factors). ``MoEWeightPack`` supplies canonical
bf16 ``w13``/``w2`` by default; ``preprocess_weights()`` quantizes and swizzles
them. Pass pre-quantized NVFP4 weights via ``w13``/``w2`` + ``w13_scale``/``w2_scale``
to skip re-quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import nvfp4_cutedsl_runtime_requirements
from .....core.validation.common import (
    validate_mega_arch,
    validate_mega_fleet_params,
)
from .....weights import MoEWeightPack
from .config import Nvfp4CutedslMegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_nvfp4_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


def _resolve_gate_up_clamp(config: Nvfp4CutedslMegaMoeConfig) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("nvfp4_cutedsl")
class Nvfp4CutedslMegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Nvfp4CutedslMegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Nvfp4CutedslMegaMoeConfig = config

    @classmethod
    def kernel_name(cls) -> str:
        return "nvfp4_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return nvfp4_cutedsl_runtime_requirements(bootstrap)

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
            gate_up_clamp=_resolve_gate_up_clamp(self._kernel_config),
            activation_clamp=self._kernel_config.activation_clamp,
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
        from ..cutedsl_backend_kernels.frontend import get_symm_buffer_for_mega_moe

        k = self._kernel_config
        fp = fleet_params
        return get_symm_buffer_for_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            2 * k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            apply_topk_in_fc1=k.apply_topk_in_fc1,
            fc1_alpha=k.fc1_alpha,
            fc2_alpha=k.fc2_alpha,
            fc1_norm_const=k.fc1_norm_const,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_nvfp4_forward_inputs(
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
            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
                norm_const=self._kernel_config.input_norm_const,
            )
        else:
            from common.megamoe_constants import Nvfp4BlockSize
            from moe_nvfp4_swapab.runner_common import ceil_div, round_up

            hidden = workspace.hidden
            hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
            hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

            workspace.x[:num_tokens].copy_(t.hidden_states)
            assert t.scales is not None
            workspace.x_sf[:num_tokens].zero_()
            workspace.x_sf[:num_tokens, :hidden_sf_cols].copy_(
                t.scales[:num_tokens, :hidden_sf_cols]
            )
            if t.scales.shape[1] >= hidden_sf_cols_padded:
                workspace.x_sf[
                    :num_tokens, hidden_sf_cols:hidden_sf_cols_padded
                ].zero_()
            workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
            workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
            capacity = workspace.x.shape[0]
            if num_tokens < capacity:
                workspace.topk_idx[num_tokens:capacity].fill_(-1)

        if t.fc1_alpha is not None:
            workspace.fc1_alpha.copy_(t.fc1_alpha)
        if t.fc2_alpha is not None:
            workspace.fc2_alpha.copy_(t.fc2_alpha)
        if t.fc1_norm_const is not None:
            workspace.fc1_norm_const.copy_(t.fc1_norm_const)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from ..cutedsl_backend_kernels.frontend import nvfp4_mega_moe

        kcfg = self._kernel_config
        nvfp4_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=output.shape[0],
            gate_up_clamp=_resolve_gate_up_clamp(kcfg),
            activation_clamp=kcfg.activation_clamp,
            fast_math=kcfg.fast_math,
        )
        return output

    def destroy(self, workspace: Any) -> None:
        if workspace is not None:
            workspace.destroy()
