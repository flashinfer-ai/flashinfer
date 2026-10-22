"""SM90 pull-style NVFP4 x FP8 MegaMoE backend."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import sm90_pull_fp8_runtime_requirements
from ......core.validation.common import (
    validate_mega_arch_sm90,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig
from .staging import (
    stage_mega_moe_inputs,
    staged_tokens,
    validate_sm90_nvfp4_forward_inputs,
)
from .weights import (
    NVFP4_EPILOGUE_COMPENSATION,
    NVFP4_FOLD_K,
    NVFP4_FOLD_M,
    NVFP4_GROUP_SIZE,
    NVFP4_GATE_UP_INTERLEAVE,
    TransformedMegaWeights,
    copy_nvfp4_kperm_hidden,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


_NVFP4_WEIGHT_FORMAT_ID = "nvfp4_e2m1"
_NVFP4_ACTIVATION_FORMAT_ID = "fp8_e4m3_per_token_full_hidden"
# Versioned packed-layout identity; "single" denotes one FP8 weight operand.
_NVFP4_LAYOUT_ID = "nvfp4_sm90_m64_k128_gateup8_global_x8_single_v1"
_FUSED_EXECUTION_MODE = "fused"


def _resolve_gate_up_clamp(
    config: Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("sm90_fp8_nvfp4_bf16_pull_cutedsl")
class Sm90PullNvfp4MegaKernelBackend(MegaKernelBackend):
    """Fused NVFP4 MegaMoE on Hopper."""

    supports_output_view = True

    def __init__(
        self,
        config: Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
    ) -> None:
        super().__init__(config)
        self._kernel_config = config
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "NVFP4 knobs='auto' runs a COLLECTIVE compile+timing sweep "
                "at the first forward. Every EP rank must participate; "
                "knobs=None selects a tuned default profile when applicable.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_fp8_nvfp4_bf16_pull_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return sm90_pull_fp8_runtime_requirements(bootstrap)

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch_sm90()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=NVFP4_FOLD_K,
        )
        k = self._kernel_config
        hidden = fleet_params.token_hidden_size
        intermediate = k.intermediate_size

        manual_selector = any(
            value is not None
            for value in (
                k.swap_ab,
                k.pingpong,
                k.mma_tiler_mnk,
                k.cluster_shape_mnk,
                k.load_balance_mode,
                k.dedup_dispatch,
                k.active_dispatch_warps,
                k.fc1_store_offload,
                k.fc1_early_done_publish,
                k.fold_producer_warps,
            )
        )
        if manual_selector:
            fused_tile = k.mma_tiler_mnk or (128, 32, 128)
        else:
            from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg

            tactic = pkg.resolve_hopper_nvfp4_knobs(
                k.knobs if isinstance(k.knobs, dict) else None,
                world_size=bootstrap.world_size,
                hidden=hidden,
                intermediate=intermediate,
                num_total_experts=fleet_params.num_experts,
                num_topk=k.top_k,
                num_max_tokens=fleet_params.max_tokens_per_rank,
                gate_up_clamp=_resolve_gate_up_clamp(k),
                routing_profile=k.routing_profile,
                nvfp4_kperm=k.kperm,
                nvfp4_fc1_k256_overlap=k.nvfp4_fc1_k256_overlap,
            )
            fused_tile = tactic.get("mma_tiler_mnk", (128, 32, 128))
        fused_tile_k = int(fused_tile[2])
        if hidden % fused_tile_k or intermediate % fused_tile_k:
            raise ValueError(
                f"hidden/intermediate ({hidden}/{intermediate}) must both be "
                f"divisible by resolved fused MMA K={fused_tile_k}"
            )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        k = self._kernel_config
        return preprocess_mega_weights(
            weights,
            intermediate_size=k.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            expert_chunk_size=k.preprocess_expert_chunk_size,
            kperm=k.kperm,
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

    def _fused_frontend_config_kwargs(
        self, fleet_params: FleetParams
    ) -> dict[str, Any]:
        """Return the single complete fused frontend request."""

        k = self._kernel_config
        return {
            "num_total_experts": fleet_params.num_experts,
            "num_max_tokens": fleet_params.max_tokens_per_rank,
            "num_topk": k.top_k,
            "hidden": fleet_params.token_hidden_size,
            "intermediate": k.intermediate_size,
            "rank": self.ep_rank,
            "world_size": self.ep_world_size,
            "kind": k.kind,
            "fp8_scale_mode": k.fp8_scale_mode,
            "fp8_accum_mode": k.fp8_accum_mode,
            # "auto" is a backend lifecycle request, not a static tactic.
            # Allocate the initial profile, then tune on first compute.
            "knobs": k.knobs if isinstance(k.knobs, dict) else None,
            "swap_ab": k.swap_ab,
            "pingpong": k.pingpong,
            "mma_tiler_mnk": k.mma_tiler_mnk,
            "cluster_shape_mnk": k.cluster_shape_mnk,
            "load_balance_mode": k.load_balance_mode,
            "gate_up_clamp": _resolve_gate_up_clamp(k),
            "activation_clamp": k.activation_clamp,
            "in_kernel_fc2_reduce": k.enable_in_kernel_fc2_reduce,
            "token_back_mode": k.token_back_mode,
            "dedup_dispatch": k.dedup_dispatch,
            "grouped_token_back": k.grouped_token_back,
            "combine_format": k.combine_format,
            "active_dispatch_warps": k.active_dispatch_warps,
            "fc1_store_offload": k.fc1_store_offload,
            "fc1_early_done_publish": k.fc1_early_done_publish,
            "fold_producer_warps": k.fold_producer_warps,
            "nvfp4_kperm": k.kperm,
            "nvfp4_fc1_k256_overlap": k.nvfp4_fc1_k256_overlap,
            "routing_profile": k.routing_profile,
        }

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            get_symm_buffer_for_hopper_nvfp4_mega_moe,
        )

        return get_symm_buffer_for_hopper_nvfp4_mega_moe(
            **self._fused_frontend_config_kwargs(fleet_params)
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_nvfp4_forward_inputs(
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
        kperm = self._kernel_config.kperm
        stage_mega_moe_inputs(
            t.hidden_states,
            t.topk_weights,
            t.topk_ids,
            workspace.x,
            workspace.x_sf,
            workspace.topk_idx,
            workspace.topk_weights,
            quantize_input=quantize_input,
            scales=t.scales,
            safe_quantization=True,
            # K-permuted layouts read FC1 activations in the weights' K order;
            # the per-token scale is permutation invariant.
            write_payload=(
                functools.partial(copy_nvfp4_kperm_hidden, kperm=kperm)
                if kperm
                else None
            ),
        )

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        k = self._kernel_config
        if output is None:
            if self._autotune_pending:
                raise ValueError(
                    "compute(output=None) is incompatible with NVFP4 knobs='auto'"
                )
            num_tokens = staged_tokens(workspace.topk_idx)
            if num_tokens is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to stage "
                    "this NVFP4 workspace first"
                )
        else:
            num_tokens = output.shape[0]

        launch: Callable[..., Any]
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            hopper_nvfp4_mega_moe,
        )

        launch = hopper_nvfp4_mega_moe

        if self._autotune_pending:
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                autotune_hopper_nvfp4_mega_moe,
            )

            self._autotune_winner = autotune_hopper_nvfp4_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                gate_up_clamp=_resolve_gate_up_clamp(k),
                activation_clamp=k.activation_clamp,
                process_group=(
                    self._ep_comm_group
                    if self._ep_comm_group is not None
                    else (self.ep_comm_group if self.ep_world_size > 1 else None)
                ),
            )
            self._autotune_pending = False

        view = launch(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=num_tokens,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            fast_math=k.fast_math,
        )
        if output is not None:
            return output
        if view is None:
            raise RuntimeError("NVFP4 zero-copy launch did not return an output view")
        return view

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        request = self._workspace_pool_request(fleet_params)
        return request[0] if request is not None else None

    def _workspace_pool_key_from_config(self, resolved_config: Any) -> Any:
        return (
            "sm90_fp8_nvfp4_bf16_pull_cutedsl",
            _FUSED_EXECUTION_MODE,
            _NVFP4_WEIGHT_FORMAT_ID,
            _NVFP4_ACTIVATION_FORMAT_ID,
            _NVFP4_LAYOUT_ID,
            NVFP4_GROUP_SIZE,
            NVFP4_FOLD_M,
            NVFP4_FOLD_K,
            NVFP4_GATE_UP_INTERLEAVE,
            NVFP4_EPILOGUE_COMPENSATION,
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            resolved_config,
        )

    def _workspace_pool_request(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        if k.knobs == "auto":
            # Autotuning mutates the frontend; keep these sessions unpooled.
            return None
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            _get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config,
            _resolve_hopper_nvfp4_mega_moe_config,
        )

        resolved_config = _resolve_hopper_nvfp4_mega_moe_config(
            **self._fused_frontend_config_kwargs(fleet_params)
        )
        key = self._workspace_pool_key_from_config(resolved_config)
        return (
            key,
            lambda: _get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config(
                resolved_config
            ),
        )


__all__ = ["Sm90PullNvfp4MegaKernelBackend"]
