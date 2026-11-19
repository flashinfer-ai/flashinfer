"""SM100 / SM103 NVLinkOneSided MegaMoE: MXFP8 activations, MXFP4 weights.

Wraps ``flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided``. The engine
configuration is static: ``nvlink_one_sided_tiling`` picks the token tile, cluster,
schedule and protocols from the workspace capacity, EP size and routing
shape, and ``knobs`` may override any of those fields. On an SM100 or SM103
device ``validate_init`` builds the engine for that configuration (no
allocation, no compile; the CuTe DSL targets the device's own architecture),
so its geometry and shared-memory checks run at layer construction.
Workspaces are torch symmetric-memory allocations on the EP process group,
shared by every layer with the same engine configuration. The kernel compiles
on a workspace's first launch, so a workspace is captured into a CUDA graph
only after an eager launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import (
    KERNEL_NAME,
    Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig,
)
from .staging import stage_inputs, validate_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

# The top-k reduce runs under both combine protocols. Each of its 128 threads
# owns 8 BF16 columns of an output row, and a landing-row reset checks one
# route per lane of a warp.
_REDUCE_HIDDEN_ALIGNMENT = 8 * 128
_MAX_TOP_K = 32

# Engine configurations whose engine has been built on this process.
_BUILT_ENGINE_CONFIGS: set = set()

# Device capabilities the engine runs on: Blackwell (B200) and Blackwell Ultra
# (B300), which share the block-scaled tcgen05 atoms and the TMEM capacity.
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _frontend():
    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided import frontend

    return frontend


def _build_engine(config) -> None:
    """Construct the engine for ``config`` without allocating or compiling.

    Runs the engine's geometry and shared-memory plan checks; imports the CuTe
    DSL and needs an SM100 or SM103 device. Each configuration is built once
    per process.
    """
    if config in _BUILT_ENGINE_CONFIGS:
        return
    buffer_type = _frontend().Sm100GenphaseSymmBuffer
    buffer_type._configure_dsl()
    try:
        buffer_type._build_kernel(config)
    except (ValueError, NotImplementedError) as exc:
        raise MoEEpConfigError(
            f"{KERNEL_NAME} cannot build its engine for hidden {config.hidden}, "
            f"intermediate {config.intermediate}, top_k {config.num_topk}, "
            f"{config.max_tokens_per_rank} tokens per rank and tile "
            f"{config.resolved_mma_tiler_mnk}: {exc}"
        ) from exc
    _BUILT_ENGINE_CONFIGS.add(config)


def _distributed() -> bool:
    """Whether workspaces are symmetric allocations (not ``MEGA_NO_DIST=1``)."""
    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import comm

    return not comm._no_dist()


@register_mega_kernel(KERNEL_NAME)
class Sm103Mxfp8Mxfp4NVLinkOneSidedMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(
        self, config: Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig
    ) -> None:
        super().__init__(config)
        self._kernel_config = config
        self._engine_configs: dict[tuple, Any] = {}
        self._knob_overrides()

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def _knob_overrides(self) -> dict[str, Any]:
        """Validated engine overrides; lists (from JSON) become tuples."""
        knobs = self._kernel_config.knobs
        if knobs is None:
            return {}
        if not isinstance(knobs, dict):
            raise MoEEpConfigError(
                f"{KERNEL_NAME} knobs must be None or a dictionary (no autotuner)"
            )
        unknown = set(knobs) - _frontend().TUNING_FIELDS
        if unknown:
            raise MoEEpConfigError(
                f"unknown {KERNEL_NAME} knobs {sorted(unknown)}; "
                f"allowed: {sorted(_frontend().TUNING_FIELDS)}"
            )
        return {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in knobs.items()
        }

    def _engine_config(self, fleet_params: FleetParams):
        """Engine config for one capacity: ``nvlink_one_sided_tiling`` merged with ``knobs``."""
        key = (
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            self.ep_rank,
            self.ep_world_size,
        )
        config = self._engine_configs.get(key)
        if config is not None:
            return config
        k = self._kernel_config
        if fleet_params.token_hidden_size % _REDUCE_HIDDEN_ALIGNMENT:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} requires token_hidden_size to be a multiple of "
                f"{_REDUCE_HIDDEN_ALIGNMENT}, got {fleet_params.token_hidden_size}"
            )
        if k.top_k > _MAX_TOP_K:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} supports top_k up to {_MAX_TOP_K}, got {k.top_k}"
            )
        frontend = _frontend()
        # A fused shared expert's FC1 tiles fill the dispatch's window, which the
        # tiling weighs in its dispatch choice.
        shared_shape = (
            dict(
                hidden=fleet_params.token_hidden_size,
                shared_intermediate=k.intermediate_size,
            )
            if k.shared_expert
            else {}
        )
        fields = frontend.nvlink_one_sided_tiling(
            fleet_params.max_tokens_per_rank,
            self.ep_world_size,
            fleet_params.num_experts,
            k.top_k,
            **shared_shape,
        )
        overrides = self._knob_overrides()
        fields.update(overrides)
        # The tiling's pull dispatch needs the counter combine; knobs that switch
        # the combine to Sentinel take the counter dispatch unless they also set
        # the dispatch.
        if (
            "dispatch_protocol" not in overrides
            and fields["dispatch_protocol"] == "pull"
            and fields["combine_protocol"] != "counter"
        ):
            fields["dispatch_protocol"] = "counter"
        # Dispatch rounds need both counter protocols. Unless knobs set them, they
        # follow the capacity wherever both protocols end up counter, also where
        # knobs pick the counter dispatch over the tiling's pull dispatch, and are
        # one elsewhere.
        if "dispatch_rounds" not in overrides:
            both_counter = (
                fields["dispatch_protocol"] == fields["combine_protocol"] == "counter"
            )
            fields["dispatch_rounds"] = (
                frontend.counter_dispatch_rounds(
                    fleet_params.max_tokens_per_rank,
                    self.ep_world_size,
                    fleet_params.num_experts,
                )
                if both_counter
                else 1
            )
        try:
            config = frontend.Sm100GenphaseMoeConfig(
                num_total_experts=fleet_params.num_experts,
                max_tokens_per_rank=fleet_params.max_tokens_per_rank,
                num_topk=k.top_k,
                hidden=fleet_params.token_hidden_size,
                intermediate=k.intermediate_size,
                rank=self.ep_rank,
                world_size=self.ep_world_size,
                quant_kind="mxfp4_mxfp8",
                gate_up_clamp=k.resolved_gate_up_clamp,
                die_local_weights=k.die_local_weights,
                # The counter combine's tail barrier publishes the in-kernel adds;
                # the Sentinel combine keeps its slots and the separate reduce.
                reduce_topk_in_kernel=(
                    k.reduce_topk_in_kernel and fields["combine_protocol"] == "counter"
                ),
                fc1_token_pool_capacity=k.fc1_token_pool_capacity,
                shared_expert=k.shared_expert,
                shared_gate_up_clamp=k.shared_gate_up_clamp,
                shared_bf16_rounding=k.shared_bf16_rounding,
                apply_routing_weights_before_fc2=not k.routing_weights_after_fc2,
                **fields,
            )
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(str(exc)) from exc
        self._engine_configs[key] = config
        return config

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        on_device = torch.cuda.is_available()
        if on_device:
            cc = torch.cuda.get_device_capability()
            if cc not in _SUPPORTED_CAPABILITIES:
                raise MoEEpArchError(
                    f"{KERNEL_NAME} requires sm_100 or sm_103 (Blackwell); host has "
                    f"sm_{cc[0]}{cc[1]}"
                )
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=128,
        )
        config = self._engine_config(fleet_params)
        if on_device:
            _build_engine(config)

    def _check_selection(self, config) -> None:
        """Require the same engine configuration on every EP rank."""
        if self.ep_world_size > 1 and _distributed():
            import dataclasses

            import torch.distributed as dist

            proposal = dataclasses.asdict(config)
            proposal.pop("rank")
            proposals = [None] * self.ep_world_size
            dist.all_gather_object(proposals, proposal, group=self.ep_comm_group)
            if any(p != proposal for p in proposals):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME} engine configuration differs across EP ranks"
                )

    def _create_workspace(self, config) -> Any:
        try:
            return _frontend().Sm100GenphaseSymmBuffer(
                config, process_group=self._ep_comm_group
            )
        except ValueError as exc:
            raise MoEEpConfigError(str(exc)) from exc

    def _workspace_pool_request(self, fleet_params: FleetParams):
        config = self._engine_config(fleet_params)
        self._check_selection(config)
        key = (
            KERNEL_NAME,
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            config,
        )
        return key, lambda: self._create_workspace(config)

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        config = self._engine_config(fleet_params)
        self._check_selection(config)
        return self._create_workspace(config)

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        config = self._engine_config(fleet_params)
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            weight_layout=config.resolved_weight_layout,
            die_local_weights=config.resolved_die_local_weights,
            shared_expert=self._kernel_config.shared_expert,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights, self._engine_config(fleet_params)
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_forward_inputs(
            t,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        stage_inputs(t, workspace, quantize_input=quantize_input)

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return workspace.output_activation

    def validate_capture_ready(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> None:
        """Reject capture before staging unless the workspace's kernel is compiled.

        Compilation runs on a workspace's first launch and cannot run during
        capture; rebinding a compiled kernel to other weights or another stream
        is host-only and stays legal.
        """
        if workspace._destroyed:
            raise MoEEpConfigError(f"{KERNEL_NAME} workspace has been destroyed")
        if workspace._compiled is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} workspace is not compiled for CUDA graph capture; "
                "run layer.warmup(...) (with workspace= for an explicit handle) "
                "on every EP rank first"
            )

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Launch on the staged rows; ``output=None`` returns a workspace view."""
        view = _frontend().sm100_genphase_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=workspace.staged_tokens(),
            shared_weights=tuple(transformed_weights[2:]) or None,
        )
        return output if output is not None else view

    def _forget_workspace_state(self, workspace: Any) -> None:
        """Drain every EP rank before the last release frees symmetric storage.

        Peers push into this rank's landing buffers, so no rank may free its
        copy while a peer's launch can still write to it.
        """
        torch.cuda.synchronize()
        if self.ep_world_size > 1 and _distributed():
            import torch.distributed as dist

            dist.barrier(group=self._ep_comm_group)
