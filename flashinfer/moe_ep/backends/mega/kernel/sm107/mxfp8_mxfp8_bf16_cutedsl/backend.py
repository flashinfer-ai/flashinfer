"""SM107 (Rubin) mxfp8 block-scaled mega-MoE kernel backend.

Wraps the vendored ``kernel_src/next_cutedsl_megamoe`` drop's fused
dispatch + FC1 + SwiGLU + FC2 + combine inference kernel
(``BlockScaledSwapAbMegaMoeKernel`` at quant kind mxfp8) behind the
``MegaKernelBackend`` contract.  The backend talks only to the drop's package
``__init__`` (never ``src/`` directly), keeping ``import flashinfer.moe_ep``
CPU-safe.
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
from .config import Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_sm107_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _resolve_gate_up_clamp(
    config: Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("sm107_mxfp8_mxfp8_bf16_cutedsl")
class Sm107Mxfp8BlockScaledMegaKernelBackend(MegaKernelBackend):
    """Fused Rubin mxfp8 block-scaled inference MoE over the NVLink symmetric heap."""

    def __init__(self, config: Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_mxfp8_mxfp8_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig):
        return sm107_block_scaled_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch_sm107()
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
        # Backend talks only to the next_cutedsl_megamoe shim (never src/
        # directly).
        from ......kernel_src.next_cutedsl_megamoe import (
            get_symm_buffer_for_sm107_block_scaled_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        # The tuning knobs the offline tuner sweeps; the config's `knobs`
        # field (dict / "cache") overrides the explicit fields.
        tuning: dict = {
            "mma_tiler_mnk": k.mma_tiler_mnk,
            "cluster_shape_mn": k.cluster_shape_mn,
            "fallback_cluster_shape_mn": k.fallback_cluster_shape_mn,
            "schedule_policy": k.schedule_policy,
            "work_id_mode": k.work_id_mode,
            "fc2_use_bulk": k.fc2_use_bulk,
            "fc2_tma_stages": k.fc2_tma_stages,
            "epi_flag_batches": k.epi_flag_batches,
            "token_in_flag_batch": k.token_in_flag_batch,
            "token_back_mode": k.token_back_mode,
            "reduce_topk_in_kernel": k.in_kernel_fc2_reduce,
        }
        tuning.update(self._knob_overrides(fp))
        # Unset optional keys fall back to the allocator defaults.
        for key in (
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "fallback_cluster_shape_mn",
            "fc2_tma_stages",
        ):
            if tuning[key] is None:
                del tuning[key]
        return get_symm_buffer_for_sm107_block_scaled_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            quant_kind=k.kind,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            apply_topk_at_fc1=k.apply_topk_in_fc1,
            max_sm_count=k.max_sm_count,
            **tuning,
        )

    def _knob_overrides(self, fleet_params: FleetParams) -> dict:
        """Resolve the config's ``knobs`` field into shim-kwarg overrides.

        ``None`` -> {} (the explicit config fields stand); a dict ->
        validated explicit overrides; ``"cache"`` -> knob-cache lookup with
        the built-in heuristic fallback.  The online ``"auto"`` sweep is not
        supported on the engine path — the SM107 session bakes knobs at
        construction; tune offline with ``python -m flashinfer.moe_ep.tune``.
        """
        from ......kernel_src.next_cutedsl_megamoe import KNOB_KEYS, resolve_knobs

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
                dtype=k.kind,
                world_size=self.ep_world_size,
                hidden=fleet_params.token_hidden_size,
                intermediate=k.intermediate_size,
                num_experts=fleet_params.num_experts,
                topk=k.top_k,
                max_tokens=fleet_params.max_tokens_per_rank,
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
        validate_sm107_forward_inputs(
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
        self, t: "MoEEpTensors", workspace: Any, *, quantize_input: bool
    ) -> None:
        if quantize_input:
            staged = stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
                kind=self._kernel_config.kind,
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
        # Backend talks only to the next_cutedsl_megamoe shim (never src/
        # directly).
        from ......kernel_src.next_cutedsl_megamoe import sm107_block_scaled_mega_moe

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
            "sm107_mxfp8_mxfp8_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            k.kind,
            _resolve_gate_up_clamp(k),
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
