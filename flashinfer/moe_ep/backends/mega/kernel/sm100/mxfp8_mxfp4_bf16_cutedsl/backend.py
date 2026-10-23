# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend adapter for the independent native MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.runtime import mxfp8_cutedsl_runtime_requirements
from ......core.validation.common import validate_mega_arch, validate_mega_fleet_params
from ......weights import MoEWeightPack
from ..mxfp8_mxfp8_bf16_cutedsl.staging import (
    stage_mega_moe_inputs,
    validate_mxfp8_forward_inputs,
)
from .config import (
    KERNEL_NAME,
    Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    resolve_tactic,
)
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


class Mxfp8Mxfp4CutedslMegaKernelBackend(MegaKernelBackend):
    """Backend adapter for the FlashInfer-owned MXFP8 x MXFP4 kernel."""

    supports_output_view = True

    def __init__(self, config: Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return mxfp8_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=128,
        )
        if fleet_params.token_hidden_size % 128:
            raise ValueError("MXFP8 x MXFP4 MegaMoE requires hidden multiple of 128")

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

    def _resolved_tactic(self, fleet_params: FleetParams) -> dict | None:
        """Pinned tactic, or None when the workspace resolves it (knob cache
        then default for ``knobs=None``; the autotune winner for ``"auto"``)."""
        knobs = self._kernel_config.knobs
        if knobs is None or knobs == "auto":
            return None
        return dict(
            resolve_tactic(
                fleet_params.max_tokens_per_rank,
                knobs,
                hidden=fleet_params.token_hidden_size,
                intermediate=self._kernel_config.intermediate_size,
            )
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        # Component 11 owns this integration facade; importing lazily keeps the
        # backend/config/weight package testable while that facade is assembled.
        from ......cute_dsl.megamoe.mxfp8_mxfp4.integration import (
            get_symm_buffer_for_mxfp8_mxfp4_mega_moe,
        )

        cfg = self._kernel_config
        knobs = self._resolved_tactic(fleet_params)
        capacity = None
        if cfg.knobs is None or cfg.knobs == "auto":
            from .config import candidate_tactics

            # General-autotuner cache lookup occurs with weights/live tokens
            # at first compute. Keep room for any admitted winner, including
            # when a caller enters autotune() after workspace allocation.
            if cfg.knobs == "auto":
                knobs = {}
            capacity = candidate_tactics(
                hidden=fleet_params.token_hidden_size,
                intermediate=cfg.intermediate_size,
            )
        return get_symm_buffer_for_mxfp8_mxfp4_mega_moe(
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            cfg.top_k,
            fleet_params.token_hidden_size,
            cfg.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=cfg.gate_up_clamp,
            knobs=knobs,
            capacity_tactics=capacity,
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
            kind="mxfp8_e4m3",
            scales=t.scales,
        )

    def stage_inputs(
        self, t: "MoEEpTensors", workspace: Any, *, quantize_input: bool
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
                kind="mxfp8_e4m3",
            )
            return

        from ......kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

        hidden_sf_cols = workspace.hidden // 32
        padded_sf_cols = ((hidden_sf_cols + 3) // 4) * 4
        workspace.x[:num_tokens].view(torch.uint8).copy_(
            t.hidden_states[:num_tokens].view(torch.uint8)
        )
        assert t.scales is not None
        workspace.x_sf[:num_tokens].zero_()
        workspace.x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(
            t.scales[:num_tokens, :hidden_sf_cols].view(torch.uint8)
        )
        if workspace.x_sf.shape[1] < padded_sf_cols:
            raise ValueError("workspace MXFP8 scale row is smaller than its padded ABI")
        workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
        workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
        if num_tokens < workspace.x.shape[0]:
            workspace.topk_idx[num_tokens:].fill_(-1)
        note_staged_tokens(workspace.topk_idx, num_tokens)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......cute_dsl.megamoe.mxfp8_mxfp4.integration import (
            autotune_mxfp8_mxfp4_mega_moe,
            mxfp8_mxfp4_mega_moe,
        )
        from ......kernel_src.sm100.cutedsl_megamoe import staged_tokens

        cfg = self._kernel_config
        unpinned = cfg.knobs is None or cfg.knobs == "auto"
        pending = unpinned and not workspace._autotune_selected

        if output is not None:
            num_tokens = output.shape[0]
        else:
            if pending and cfg.knobs == "auto":
                raise ValueError("knobs='auto' requires a caller-owned output tensor")
            num_tokens = staged_tokens(workspace.topk_idx)
            if num_tokens is None:
                raise ValueError("stage_inputs() must run before compute(output=None)")

        if unpinned:
            from flashinfer.autotuner import AutoTuner

            tuning = AutoTuner.get().is_tuning_mode
            if pending and cfg.knobs == "auto":
                autotune_mxfp8_mxfp4_mega_moe(
                    output,
                    transformed_weights[0],
                    transformed_weights[1],
                    workspace,
                    num_tokens=num_tokens,
                    gate_up_clamp=cfg.gate_up_clamp,
                    fast_math=cfg.fast_math,
                )
            elif pending or (tuning and not torch.cuda.is_current_stream_capturing()):
                from ......cute_dsl.megamoe.mxfp8_mxfp4.autotune import select_tactic

                select_tactic(
                    transformed_weights[0],
                    transformed_weights[1],
                    workspace,
                    num_tokens=num_tokens,
                    gate_up_clamp=cfg.gate_up_clamp,
                    fast_math=cfg.fast_math,
                )
        view = mxfp8_mxfp4_mega_moe(
            # Bind the reducer to stable workspace storage. The public layer's
            # owned-output path allocates a new tensor on every invocation,
            # including graph capture; rebinding DLPack views there is unsafe.
            None,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=num_tokens,
            gate_up_clamp=self._kernel_config.gate_up_clamp,
            fast_math=self._kernel_config.fast_math,
        )
        if output is not None:
            output.copy_(view)
            return output
        return view

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        if self._kernel_config.knobs == "auto":
            return None
        from ......core.kernel.workspace_pool import knobs_pool_key

        cfg = self._kernel_config
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            cfg.top_k,
            fleet_params.token_hidden_size,
            cfg.intermediate_size,
            cfg.gate_up_clamp,
            cfg.fast_math,
            knobs_pool_key(self._resolved_tactic(fleet_params)),
        )

    def _forget_workspace_state(self, workspace: Any) -> None:
        import sys

        quant_stage = sys.modules.get(
            "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.quant_stage"
        )
        topk_idx = getattr(workspace, "topk_idx", None)
        if quant_stage is not None and topk_idx is not None:
            quant_stage.forget_staged_tokens(topk_idx)


__all__ = ["Mxfp8Mxfp4CutedslMegaKernelBackend"]


# Do not advertise a backend whose persistent device assembler has not landed.
# The device-kernel component provides this module and class; registration then
# becomes automatic without making CPU-only config imports depend on CuTe DSL.
try:
    from ......cute_dsl.megamoe.mxfp8_mxfp4.persistent_kernel import (
        ASSEMBLY_READY as _ASSEMBLY_READY,
    )
except ImportError:
    _ASSEMBLY_READY = False

if _ASSEMBLY_READY:
    from ......core.kernel.registry import register_mega_kernel

    register_mega_kernel(KERNEL_NAME)(Mxfp8Mxfp4CutedslMegaKernelBackend)
