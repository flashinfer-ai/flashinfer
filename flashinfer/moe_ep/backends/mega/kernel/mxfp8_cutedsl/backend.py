"""CuTeDSL MXFP8 mega-MoE kernel backend.

The fused kernel consumes MXFP8 expert weights in kernel-ready layout (fp8
weights + atom-swizzled E8M0 scale factors). ``MoEWeightPack`` supplies
canonical bf16 ``w13``/``w2`` by default; ``preprocess_weights()`` quantizes
and swizzles them. Pass pre-quantized MXFP8 weights via ``w13``/``w2`` plus
``w13_scale``/``w2_scale`` to skip re-quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import mxfp8_cutedsl_runtime_requirements
from .....core.validation.common import (
    validate_mega_arch,
    validate_mega_fleet_params,
)
from .....weights import MoEWeightPack
from .config import Mxfp8CutedslMegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_mxfp8_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


def _resolve_gate_up_clamp(config: Mxfp8CutedslMegaMoeConfig) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("mxfp8_cutedsl")
class Mxfp8CutedslMegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Mxfp8CutedslMegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Mxfp8CutedslMegaMoeConfig = config
        # knobs="auto": tune at the first compute() (weights + staged inputs
        # exist there), then keep the winner for the session.
        self._autotune_pending = config.knobs == "auto"

    @classmethod
    def kernel_name(cls) -> str:
        return "mxfp8_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return mxfp8_cutedsl_runtime_requirements(bootstrap)

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
            kind=self._kernel_config.kind,
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
            kind=self._kernel_config.kind,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from .....kernel_src.cutedsl_megamoe import (
            get_symm_buffer_for_mxfp8_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        return get_symm_buffer_for_mxfp8_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            kind=k.kind,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            in_kernel_fc2_reduce=k.in_kernel_fc2_reduce,
            token_back_by_dispatch=k.token_back_by_dispatch,
            knobs=k.knobs if isinstance(k.knobs, dict) else None,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_mxfp8_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            kind=self._kernel_config.kind,
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
                kind=self._kernel_config.kind,
            )
        else:
            # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
            from .....kernel_src.cutedsl_megamoe import (
                Mxfp8BlockSize,
                ceil_div,
                round_up,
            )

            hidden = workspace.hidden
            hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
            hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

            workspace.x[:num_tokens].view(torch.uint8).copy_(
                t.hidden_states[:num_tokens].view(torch.uint8)
            )
            assert t.scales is not None
            workspace.x_sf[:num_tokens].zero_()
            workspace.x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(
                t.scales[:num_tokens, :hidden_sf_cols].view(torch.uint8)
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

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from .....kernel_src.cutedsl_megamoe import mxfp8_mega_moe

        kcfg = self._kernel_config
        if self._autotune_pending:
            # COLLECTIVE: every EP rank reaches this first compute() together,
            # so the candidate sweep stays in lockstep (see shim/autotune.py).
            from .....kernel_src.cutedsl_megamoe import autotune_mxfp8_mega_moe

            autotune_mxfp8_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=output.shape[0],
                gate_up_clamp=_resolve_gate_up_clamp(kcfg),
                activation_clamp=kcfg.activation_clamp,
            )
            # Cleared only on success: if the collective tune raises, a retried
            # compute() re-attempts it (all ranks fail together, so lockstep holds).
            self._autotune_pending = False
        mxfp8_mega_moe(
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
