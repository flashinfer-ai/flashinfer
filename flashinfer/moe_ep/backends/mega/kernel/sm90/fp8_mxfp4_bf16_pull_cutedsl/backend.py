"""Production SM90 pull-style Humming MXFP4 x FP8 MegaMoE backend."""

from __future__ import annotations

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
from .config import Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
from .staging import (
    stage_mega_moe_inputs,
    staged_tokens,
    validate_sm90_mxfp4_forward_inputs,
)
from .weights import (
    HUMMING_EPILOGUE_COMPENSATION,
    HUMMING_FOLD_K,
    HUMMING_FOLD_M,
    HUMMING_GROUP_SIZE,
    MXFP4_GATE_UP_INTERLEAVE,
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


_MXFP4_WEIGHT_FORMAT_ID = "mxfp4_e2m1"
_MXFP4_ACTIVATION_FORMAT_ID = "fp8_e4m3_per_token_full_hidden"
_HUMMING_LAYOUT_ID = "humming_sm90_m64_k128_gateup8_residual_x64_v1"
_FUSED_EXECUTION_MODE = "fused"


def _resolve_gate_up_clamp(
    config: Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


def _resolve_split_tactic(
    config: Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    fleet_params: FleetParams,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Resolve one complete split session tactic without fused fallback."""

    if (
        config.knobs is None
        and config.split_k1_sm_count is not None
        and config.split_k2_sm_count is not None
    ):
        # Backward-compatible explicit split_* representation.
        return {
            "counter_epoch_banks": config.split_counter_epoch_banks,
            "enable_iket": config.split_enable_iket,
            "graph_variant": config.split_graph_variant,
            "k1_cluster_shape_mnk": config.split_k1_cluster_shape_mnk,
            "k1_group_hint": config.split_k1_group_hint,
            "k1_mma_tiler_mnk": config.split_k1_mma_tiler_mnk,
            "k1_num_sched_stages": config.split_k1_num_sched_stages,
            "k1_sm_count": config.split_k1_sm_count,
            "k2_cluster_shape_mnk": config.split_k2_cluster_shape_mnk,
            "k2_group_hint": config.split_k2_group_hint,
            "k2_mma_tiler_mnk": config.split_k2_mma_tiler_mnk,
            "k2_num_sched_stages": config.split_k2_num_sched_stages,
            "k2_sm_count": config.split_k2_sm_count,
        }

    from ......kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4_split import (
        _resolve_mxfp4_split_tactic,
    )

    # auto starts from the heuristic; first compute measures and replaces it.
    return _resolve_mxfp4_split_tactic(
        config.knobs if isinstance(config.knobs, dict) else None,
        world_size=world_size,
        hidden=fleet_params.token_hidden_size,
        intermediate=config.intermediate_size,
        num_total_experts=fleet_params.num_experts,
        num_topk=config.top_k,
        num_max_tokens=fleet_params.max_tokens_per_rank,
        gate_up_clamp=_resolve_gate_up_clamp(config),
        routing_profile=config.routing_profile,
    )


@register_mega_kernel("sm90_fp8_mxfp4_bf16_pull_cutedsl")
class Sm90PullMxfp4MegaKernelBackend(MegaKernelBackend):
    """Fused or concurrent-Green Humming MXFP4 MegaMoE on Hopper."""

    supports_output_view = True

    def __init__(
        self,
        config: Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    ) -> None:
        super().__init__(config)
        self._kernel_config = config
        # knobs="auto": the fused workspace starts from its token-bucket
        # heuristic, then the first compute collectively measures only the
        # compact offline candidate union and applies the winner in place.
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "MXFP4 knobs='auto' runs a COLLECTIVE compile+timing sweep at "
                "the first forward; tune offline for serving "
                "(python -m flashinfer.moe_ep.tune --dtype sm90_mxfp4). "
                "Rank 0 persists the winner and knobs=None performs only a "
                "cache lookup with manifest-heuristic fallback.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_fp8_mxfp4_bf16_pull_cutedsl"

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
            alignment=HUMMING_FOLD_K,
        )
        k = self._kernel_config
        hidden = fleet_params.token_hidden_size
        intermediate = k.intermediate_size
        if k.execution_mode == "split":
            tactic = _resolve_split_tactic(
                k,
                fleet_params,
                world_size=bootstrap.world_size,
            )
            k1_tile_k = int(tactic["k1_mma_tiler_mnk"][2])
            k2_tile_k = int(tactic["k2_mma_tiler_mnk"][2])
            if hidden % k1_tile_k:
                raise ValueError(
                    f"token_hidden_size ({hidden}) must be divisible by "
                    f"resolved split K1 MMA K={k1_tile_k}"
                )
            if intermediate % k2_tile_k:
                raise ValueError(
                    f"intermediate_size ({intermediate}) must be divisible by "
                    f"resolved split K2 MMA K={k2_tile_k}"
                )
            return

        manual_geometry = any(
            value is not None
            for value in (
                k.swap_ab,
                k.pingpong,
                k.mma_tiler_mnk,
                k.cluster_shape_mnk,
            )
        )
        if manual_geometry:
            fused_tile = k.mma_tiler_mnk or (128, 32, 128)
        else:
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
                _resolve_mxfp4_knobs,
            )

            tactic = _resolve_mxfp4_knobs(
                k.knobs if isinstance(k.knobs, dict) else None,
                world_size=bootstrap.world_size,
                hidden=hidden,
                intermediate=intermediate,
                num_total_experts=fleet_params.num_experts,
                num_topk=k.top_k,
                num_max_tokens=fleet_params.max_tokens_per_rank,
                gate_up_clamp=_resolve_gate_up_clamp(k),
                routing_profile=k.routing_profile,
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
            humming_max_range=k.humming_max_range,
            expert_chunk_size=k.preprocess_expert_chunk_size,
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
        k = self._kernel_config
        if k.execution_mode == "split":
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                get_symm_buffer_for_hopper_mxfp4_split_mega_moe,
            )

            tactic = _resolve_split_tactic(
                k,
                fleet_params,
                world_size=self.ep_world_size,
            )

            return get_symm_buffer_for_hopper_mxfp4_split_mega_moe(
                fleet_params.num_experts,
                fleet_params.max_tokens_per_rank,
                k.top_k,
                fleet_params.token_hidden_size,
                k.intermediate_size,
                self.ep_rank,
                self.ep_world_size,
                split_k1_mma_tiler_mnk=tactic["k1_mma_tiler_mnk"],
                split_k2_mma_tiler_mnk=tactic["k2_mma_tiler_mnk"],
                split_k1_cluster_shape_mnk=tactic["k1_cluster_shape_mnk"],
                split_k2_cluster_shape_mnk=tactic["k2_cluster_shape_mnk"],
                split_k1_group_hint=tactic["k1_group_hint"],
                split_k2_group_hint=tactic["k2_group_hint"],
                split_k1_num_sched_stages=tactic["k1_num_sched_stages"],
                split_k2_num_sched_stages=tactic["k2_num_sched_stages"],
                split_k1_sm_count=tactic["k1_sm_count"],
                split_k2_sm_count=tactic["k2_sm_count"],
                split_counter_epoch_banks=tactic["counter_epoch_banks"],
                split_graph_variant=tactic["graph_variant"],
                gate_up_clamp=_resolve_gate_up_clamp(k),
                split_enable_iket=tactic["enable_iket"],
                routing_profile=k.routing_profile,
                process_group=(
                    self._ep_comm_group
                    if self._ep_comm_group is not None
                    else (self.ep_comm_group if self.ep_world_size > 1 else None)
                ),
            )

        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            get_symm_buffer_for_hopper_mxfp4_mega_moe,
        )

        return get_symm_buffer_for_hopper_mxfp4_mega_moe(
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            k.top_k,
            fleet_params.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            kind=k.kind,
            fp8_scale_mode=k.fp8_scale_mode,
            fp8_accum_mode=k.fp8_accum_mode,
            # "auto" is a backend lifecycle request, not a static tactic.
            # Allocate from cache/heuristic, then tune on first compute.
            knobs=k.knobs if isinstance(k.knobs, dict) else None,
            swap_ab=k.swap_ab,
            pingpong=k.pingpong,
            mma_tiler_mnk=k.mma_tiler_mnk,
            cluster_shape_mnk=k.cluster_shape_mnk,
            load_balance_mode=k.load_balance_mode,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            in_kernel_fc2_reduce=k.in_kernel_fc2_reduce,
            token_back_mode=k.token_back_mode,
            routing_profile=k.routing_profile,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_mxfp4_forward_inputs(
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
                    "compute(output=None) is incompatible with knobs='auto' "
                    "(the collective MXFP4 sweep needs a caller output buffer)"
                )
            num_tokens = staged_tokens(workspace.topk_idx)
            if num_tokens is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to stage "
                    "this MXFP4 workspace first"
                )
        else:
            num_tokens = output.shape[0]

        launch: Callable[..., Any]
        if k.execution_mode == "split":
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                hopper_mxfp4_split_mega_moe,
            )

            launch = hopper_mxfp4_split_mega_moe
        else:
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                hopper_mxfp4_mega_moe,
            )

            launch = hopper_mxfp4_mega_moe

        if self._autotune_pending:
            # COLLECTIVE: every EP rank reaches this first compute together.
            # Both wrappers use rank-local median then all-rank MAX. Split
            # candidates additionally own fresh fixed-pointer Green sessions.
            if k.execution_mode == "split":
                from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                    autotune_hopper_mxfp4_split_mega_moe,
                )

                autotune = autotune_hopper_mxfp4_split_mega_moe
            else:
                from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                    autotune_hopper_mxfp4_mega_moe,
                )

                autotune = autotune_hopper_mxfp4_mega_moe

            autotune(
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
            # Clear only after a successful collective sweep so a coordinated
            # retry does not silently continue with an unmeasured tactic.
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
            raise RuntimeError("MXFP4 zero-copy launch did not return an output view")
        return view

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        # Split graph executables own fixed pointers/Green resources/counter
        # epochs. Auto sessions mutate their frontend tactic at first compute.
        # Neither session kind may borrow a pooled workspace.
        if k.execution_mode == "split" or k.knobs == "auto":
            return None

        from ......core.kernel.workspace_pool import knobs_pool_key
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
            _resolve_mxfp4_knobs,
        )

        manual_geometry = any(
            value is not None
            for value in (
                k.swap_ab,
                k.pingpong,
                k.mma_tiler_mnk,
                k.cluster_shape_mnk,
            )
        )
        if manual_geometry:
            # Manual fields are already separate key axes below. Name their
            # source so they cannot collide with a cache-resolved tactic that
            # happens to have the same partial user representation.
            effective_knobs: Any = "manual_geometry"
        else:
            # Resolve before pooling: raw knobs=None is not a tactic identity.
            # The allocation path performs the same pure lookup immediately
            # afterwards, while this key prevents an old compiled workspace
            # from surviving a cache-winner change.
            effective_knobs = _resolve_mxfp4_knobs(
                k.knobs,
                world_size=self.ep_world_size,
                hidden=fleet_params.token_hidden_size,
                intermediate=k.intermediate_size,
                num_total_experts=fleet_params.num_experts,
                num_topk=k.top_k,
                num_max_tokens=fleet_params.max_tokens_per_rank,
                gate_up_clamp=_resolve_gate_up_clamp(k),
                routing_profile=k.routing_profile,
            )
        return (
            "sm90_fp8_mxfp4_bf16_pull_cutedsl",
            _FUSED_EXECUTION_MODE,
            k.routing_profile,
            _MXFP4_WEIGHT_FORMAT_ID,
            _MXFP4_ACTIVATION_FORMAT_ID,
            k.fp8_scale_mode,
            _HUMMING_LAYOUT_ID,
            k.humming_max_range,
            HUMMING_GROUP_SIZE,
            HUMMING_FOLD_M,
            HUMMING_FOLD_K,
            MXFP4_GATE_UP_INTERLEAVE,
            HUMMING_EPILOGUE_COMPENSATION,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            k.top_k,
            fleet_params.token_hidden_size,
            k.intermediate_size,
            k.kind,
            k.fp8_accum_mode,
            k.swap_ab,
            k.pingpong,
            k.mma_tiler_mnk,
            k.cluster_shape_mnk,
            k.load_balance_mode,
            _resolve_gate_up_clamp(k),
            k.in_kernel_fc2_reduce,
            k.token_back_mode,
            knobs_pool_key(effective_knobs),
        )


__all__ = ["Sm90PullMxfp4MegaKernelBackend"]
