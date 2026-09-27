"""SM107 NVFP4 MegaMoE backend.

Calls the Rubin package API for fused dispatch, both GEMMs, and combine.
CUDA and CuTe DSL imports are deferred until the backend is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import sm107_block_scaled_runtime_requirements
from ......core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch_sm107,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from ..validation import validate_routing_values, validate_unit_scalars
from .config import Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_sm107_nvfp4_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


@register_mega_kernel("sm107_nvfp4_nvfp4_bf16_cutedsl")
class Sm107Nvfp4BlockScaledMegaKernelBackend(MegaKernelBackend):
    """Fused Rubin nvfp4 block-scaled inference MoE over the NVLink symmetric heap."""

    # compute(output=None) returns a view into the workspace.
    supports_output_view = True

    def __init__(self, config: Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_nvfp4_nvfp4_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig):
        return sm107_block_scaled_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch_sm107()
        from ......kernel_src.sm107.next_cutedsl_megamoe import require_sm107_dsl

        self._ensure_ep_bootstrap(bootstrap)
        self._resolved_config(fleet_params)
        if torch.cuda.is_available():
            require_sm107_dsl()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=64,
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
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _resolved_config(self, fleet_params: FleetParams):
        from ..validation import make_workspace_config

        return make_workspace_config(
            self._kernel_config,
            fleet_params,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            quant_kind="nvfp4",
            gate_up_clamp=self._kernel_config.gate_up_clamp,
            overrides=self._knob_overrides(fleet_params),
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.sm107.next_cutedsl_megamoe import (
            Sm107BlockScaledSymmBuffer,
        )

        return Sm107BlockScaledSymmBuffer(self._resolved_config(fleet_params))

    def _knob_overrides(self, fleet_params: FleetParams) -> dict:
        """Resolve the config's ``knobs`` field into shim-kwarg overrides.

        ``None`` -> {} (the explicit config fields stand); a dict ->
        validated explicit overrides; ``"cache"`` -> knob-cache lookup with
        the built-in heuristic fallback.  The online ``"auto"`` sweep is not
        supported on the engine path — the SM107 session bakes knobs at
        construction; tune offline with ``python -m flashinfer.moe_ep.tune``.
        """
        from ......kernel_src.sm107.next_cutedsl_megamoe import KNOB_KEYS, resolve_knobs

        k = self._kernel_config
        if k.knobs is None:
            return {}
        if isinstance(k.knobs, dict):
            unknown = set(k.knobs) - set(KNOB_KEYS)
            if unknown:
                raise MoEEpConfigError(
                    f"unknown SM107 knob keys: {sorted(unknown)} "
                    f"(valid: {sorted(KNOB_KEYS)})"
                )
            return dict(k.knobs)
        if k.knobs == "cache":
            knobs, source = resolve_knobs(
                dtype="nvfp4",
                world_size=self.ep_world_size,
                hidden=fleet_params.token_hidden_size,
                intermediate=k.intermediate_size,
                num_experts=fleet_params.num_experts,
                topk=k.top_k,
                max_tokens=fleet_params.max_tokens_per_rank,
                allow_nondeterministic=k.in_kernel_fc2_reduce,
                apply_topk_at_fc1=k.apply_topk_in_fc1,
            )
            if self.ep_rank == 0:
                print(
                    f"[{self.kernel_name()}] knobs='cache' resolved via "
                    f"{source}: {knobs}",
                    flush=True,
                )
            return knobs
        raise MoEEpConfigError(
            "knobs='auto' (online sweep) is not supported by the SM107 "
            "backends; run the offline tuner (python -m flashinfer.moe_ep.tune)"
            " and use knobs=None, knobs='cache', or an explicit knob dict."
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_unit_scalars(t)
        validate_sm107_nvfp4_forward_inputs(
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
        validate_routing_values(
            t.topk_ids, t.topk_weights, workspace.config.num_total_experts
        )
        if quantize_input:
            staged = stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
            )
            workspace.note_staged_tokens(staged)
            return

        num_tokens = t.hidden_states.shape[0]
        capacity = workspace.x.shape[0]
        if num_tokens > 0:
            workspace.x[:num_tokens].view(torch.uint8).copy_(
                t.hidden_states.view(torch.uint8)
            )
            sf_cols = t.scales.shape[1]
            workspace.x_sf[:num_tokens].view(torch.uint8).zero_()
            workspace.x_sf[:num_tokens, :sf_cols].view(torch.uint8).copy_(
                t.scales.view(torch.uint8)
            )
            workspace.topk_idx[:num_tokens].copy_(t.topk_ids.to(torch.int32))
            workspace.topk_weights[:num_tokens].copy_(t.topk_weights.to(torch.float32))
        if num_tokens < capacity:
            workspace.topk_idx[num_tokens:capacity].fill_(-1)
        workspace.note_staged_tokens(num_tokens)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......kernel_src.sm107.next_cutedsl_megamoe import (
            sm107_block_scaled_mega_moe,
        )

        if output is not None:
            num_tokens = int(output.shape[0])
        else:
            num_tokens = workspace.staged_tokens()
            if num_tokens is None:
                raise RuntimeError(
                    "compute() called before stage_inputs(); no token count is staged."
                )

        view = sm107_block_scaled_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=num_tokens,
            fast_math=self._kernel_config.fast_math,
        )
        return output if output is not None else view

    def _workspace_pool_key(self, fleet_params: FleetParams):
        from ......core.kernel.workspace_pool import knobs_pool_key

        k = self._kernel_config
        fp = fleet_params
        return (
            "sm107_nvfp4_nvfp4_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            k.gate_up_clamp,
            k.in_kernel_fc2_reduce,
            k.token_back_mode,
            k.apply_topk_in_fc1,
            k.schedule_policy,
            k.work_id_mode,
            k.fc2_use_bulk,
            k.fc2_tma_stages,
            k.epi_flag_batches,
            k.token_in_flag_batch,
            k.mma_tiler_mnk,
            k.cluster_shape_mn,
            k.fallback_cluster_shape_mn,
            k.max_sm_count,
            knobs_pool_key(k.knobs),
        )

    def _forget_workspace_state(self, workspace: Any) -> None:
        workspace.note_staged_tokens(0)
        workspace._staged_tokens = None
