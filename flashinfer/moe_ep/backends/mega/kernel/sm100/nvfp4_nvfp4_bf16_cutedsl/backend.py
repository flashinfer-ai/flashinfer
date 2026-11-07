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

from ......config import BootstrapConfig, FinalizeMode, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import nvfp4_cutedsl_runtime_requirements
from ......core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
from .staging import (
    stage_mega_moe_inputs,
    validate_fc1_activation_per_token_scale,
    validate_nvfp4_forward_inputs,
)
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _resolve_gate_up_clamp(
    config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


def _cake_reducer_supported(k, fleet_params: FleetParams) -> bool:
    from flashinfer.jit.cake_megamoe_topk_reduce import supports_device

    return (
        fleet_params.max_tokens_per_rank in (256, 4096)
        and fleet_params.token_hidden_size == 4096
        and k.top_k == 6
        and not k.enable_in_kernel_fc2_reduce
        and k.combine_dtype == "bf16"
        and k.apply_routing_weights_before_fc2
        and supports_device()
    )


@register_mega_kernel(
    "sm100_nvfp4_nvfp4_bf16_cutedsl", deprecated_aliases=("nvfp4_cutedsl",)
)
class Nvfp4CutedslMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        """Initialize legacy inference state and reject next-only GenPhase."""
        super().__init__(config)
        if config.kernel_variant != "inference":
            raise MoEEpConfigError(
                "NVFP4 GenPhase requires kernel_name='sm100_nvfp4_nvfp4_bf16_cutedsl_next'."
            )
        self._kernel_config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = config
        self._gin_device: int | None = None
        self.requires_collective_forward_validation = config.internode
        self.supports_live_token_tuning = not config.internode
        self.supports_unfinalized_output = config.finalize_mode is FinalizeMode.NONE
        self._thunk_states: dict[tuple, tuple] = {}
        # knobs="auto": tune at the first compute() (weights + staged inputs
        # exist there), then keep the winner for the session.
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "knobs='auto' runs a COLLECTIVE multi-minute compile+timing "
                "sweep at the first forward — never use it inside a serving "
                "engine. Tune offline instead (python -m flashinfer.moe_ep.tune"
                "); winners persist in the knob cache and knobs=None then "
                "resolves them with a pure lookup.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_nvfp4_nvfp4_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        if self._kernel_config.internode:
            from ......core.runtime.bootstrap import TORCH_DIST

            return frozenset({TORCH_DIST})
        return nvfp4_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if not self._kernel_config.internode:
            return self._validate_init_local(bootstrap, fleet_params)
        import torch.distributed as dist
        from ......kernel_src.sm100.cutedsl_megamoe import collective_gin_preflight

        k = self._kernel_config
        self._gin_fleet_params = fleet_params
        error = None
        try:
            self._validate_init_local(bootstrap, fleet_params)
            self._gin_device = torch.cuda.current_device()
            if (
                k.activation != "swiglu"
                or k.swiglu_alpha is not None
                or k.swiglu_beta is not None
                or k.gate_up_clamp is not None
                or k.activation_clamp is not None
                or k.situ_beta is not None
                or k.situ_linear_beta is not None
                or k.enable_in_kernel_fc2_reduce
                or k.combine_dtype != "bf16"
                or k.enable_fc1_activation_per_token_scale
                or any(
                    value is not None
                    for value in (k.fc1_alpha, k.fc2_alpha, k.fc1_norm_const)
                )
                or not k.apply_routing_weights_before_fc2
                or (
                    bootstrap.world_size,
                    k.ranks_per_node,
                    fleet_params.max_tokens_per_rank,
                    fleet_params.token_hidden_size,
                    k.intermediate_size,
                    fleet_params.num_experts,
                    k.top_k,
                )
                != (4, 2, 32, 7168, 1536, 48, 4)
            ):
                raise ValueError(
                    "GIN v1 requires the qualified N2R2 NVFP4/BF16 standard-SwiGLU case; no activation/scale/combine overrides."
                )
            if (
                k.knobs is not None
                or k.finalize_mode not in (FinalizeMode.AUTO, FinalizeMode.DEFAULT)
                or k.num_valid_tokens_tensor is not None
                or k.mma_tiler_mnk is not None
                or k.cluster_shape_mn is not None
                or k.schedule_policy is not None
                or k.input_norm_const != 1.0
                or not k.fast_math
            ):
                raise ValueError(
                    "GIN v1 does not support tuning/custom finalize/shape or math overrides."
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        collective_gin_preflight(
            dist,
            phase="backend-init",
            signature=(
                bootstrap.world_size,
                k.ranks_per_node,
                fleet_params.max_tokens_per_rank,
                fleet_params.token_hidden_size,
                k.intermediate_size,
                fleet_params.num_experts,
                k.top_k,
            ),
            local_error=error,
            requested_group=bootstrap.process_group,
        )

    def _validate_init_local(
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
            # cutedsl tiles are tail-safe (ceil-div + predicated epilogue);
            # the binding bound is TMA row alignment, not the dg SF word.
            alignment=64,
        )
        if (
            self._kernel_config.finalize_mode is FinalizeMode.CAKE
            and not self._cake_reducer_supported(fleet_params)
        ):
            raise ValueError(
                "finalize_mode=CAKE requires a supported device, "
                "max_tokens_per_rank in (256, 4096), hidden=4096, top_k=6, "
                "enable_in_kernel_fc2_reduce=False, combine_dtype='bf16' and "
                "apply_routing_weights_before_fc2=True."
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

    supports_live_token_tuning = True
    _knob_cache_module = (
        "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache"
    )

    def _tuning_key(self, fleet_params: FleetParams) -> dict[str, Any]:
        k = self._kernel_config
        return dict(
            dtype="nvfp4",
            world_size=self.ep_world_size,
            hidden=fleet_params.token_hidden_size,
            intermediate=2 * k.intermediate_size,
            num_experts=fleet_params.num_experts,
            topk=k.top_k,
            max_tokens=fleet_params.max_tokens_per_rank,
            combine_dtype=k.combine_dtype,
            enable_in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
            apply_routing_weights_before_fc2=k.apply_routing_weights_before_fc2,
            use_custom_finalize=self._finalize_mode(fleet_params)
            in (FinalizeMode.NONE, FinalizeMode.CAKE),
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.sm100.cutedsl_megamoe import get_symm_buffer_for_mega_moe

        k = self._kernel_config
        fp = fleet_params
        mode = self._finalize_mode(fleet_params)
        workspace = get_symm_buffer_for_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            2 * k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            swiglu_alpha=k.swiglu_alpha,
            swiglu_beta=k.swiglu_beta,
            activation_clamp=k.activation_clamp,
            activation=k.activation,
            situ_beta=k.situ_beta,
            situ_linear_beta=k.situ_linear_beta,
            apply_routing_weights_before_fc2=k.apply_routing_weights_before_fc2,
            enable_in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
            use_custom_finalize=mode in (FinalizeMode.NONE, FinalizeMode.CAKE),
            use_persistent_finalize_kernel=mode is FinalizeMode.PERSISTENT,
            num_valid_tokens_tensor=k.num_valid_tokens_tensor,
            combine_dtype=k.combine_dtype,
            fc1_alpha=k.fc1_alpha,
            fc2_alpha=k.fc2_alpha,
            fc1_norm_const=k.fc1_norm_const,
            enable_fc1_activation_per_token_scale=(
                k.enable_fc1_activation_per_token_scale
            ),
            knobs=None if k.internode else self._workspace_knobs(fleet_params),
            internode=k.internode,
            ranks_per_node=k.ranks_per_node,
            process_group=self._ep_bootstrap.process_group,
        )
        if mode is FinalizeMode.NONE:
            workspace._unfinalized_route_map = torch.arange(
                fp.max_tokens_per_rank * k.top_k,
                dtype=torch.int32,
                device=workspace.topk_idx.device,
            ).view(fp.max_tokens_per_rank, k.top_k)
            workspace._unfinalized_weights = (
                torch.ones_like(workspace.topk_weights)
                if workspace._frontend.config.apply_routing_weights_before_fc2
                else workspace.topk_weights
            )
        return workspace

    def _finalize_mode(self, fleet_params: FleetParams) -> FinalizeMode:
        k = self._kernel_config
        if k.internode:
            return FinalizeMode.DEFAULT
        if k.finalize_mode is not FinalizeMode.AUTO:
            return k.finalize_mode
        if self._cake_reducer_supported(fleet_params):
            return FinalizeMode.CAKE
        if k.num_valid_tokens_tensor is not None:
            return FinalizeMode.PERSISTENT
        return FinalizeMode.DEFAULT

    def _uses_native_topk_reduce(self, fleet_params: FleetParams) -> bool:
        return self._finalize_mode(fleet_params) is FinalizeMode.CAKE

    def _cake_reducer_supported(self, fleet_params: FleetParams) -> bool:
        return _cake_reducer_supported(self._kernel_config, fleet_params)

    def validate_forward(
        self, t: "MoEEpTensors", fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        if not self._kernel_config.internode:
            return self._validate_forward_local(
                t, fleet_params, quantize_input=quantize_input
            )
        self.validate_collective_forward(
            t, fleet_params, quantize_input=quantize_input, compile_tokens_per_rank=None
        )

    def validate_collective_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
        compile_tokens_per_rank: int | None,
    ) -> None:
        capturing = torch.cuda.is_current_stream_capturing()
        import torch.distributed as dist
        from ......kernel_src.sm100.cutedsl_megamoe import collective_gin_preflight

        error = None
        try:
            if (
                compile_tokens_per_rank is not None
                and compile_tokens_per_rank < t.num_tokens
            ):
                raise ValueError(
                    "compile_tokens_per_rank cannot be smaller than the live token count"
                )
            if compile_tokens_per_rank is not None:
                raise ValueError(
                    "GIN does not support compile token buckets; use the default "
                    "fixed workspace capacity (compile_tokens_per_rank=None)."
                )
            if capturing and not self._thunk_states:
                raise ValueError(
                    "GIN capture requires warmup on all ranks before capture."
                )
            self._validate_forward_local(
                t,
                fleet_params,
                quantize_input=quantize_input,
                check_route_values=not capturing,
            )
            if t.num_tokens > fleet_params.max_tokens_per_rank:
                raise ValueError("GIN live token count exceeds capacity.")
            if t.output is not None and (
                t.output.dtype != torch.bfloat16
                or t.output.device != t.hidden_states.device
                or t.output.ndim != 2
                or t.output.shape[0] < t.num_tokens
                or t.output.shape[1] != fleet_params.token_hidden_size
            ):
                raise ValueError("GIN caller output shape, dtype or device is invalid.")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        collective_gin_preflight(
            dist,
            phase="backend-forward",
            # Mixed eager/capture would launch on only the eager ranks.
            # CPU agreement runs during capture, never during graph replay.
            signature=("nvfp4-gin-v1", capturing, compile_tokens_per_rank),
            local_error=error,
        )

    def _validate_forward_local(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
        check_route_values: bool = True,
    ) -> None:
        if self._kernel_config.internode:
            if any(
                value is not None
                for value in (t.fc1_alpha, t.fc2_alpha, t.fc1_norm_const)
            ):
                raise ValueError(
                    "GIN runtime epilogue-scale overrides are not qualified."
                )
            expected_dtype = (
                torch.bfloat16 if quantize_input else torch.float4_e2m1fn_x2
            )
            if t.hidden_states.dtype != expected_dtype:
                raise ValueError(f"GIN hidden_states must have dtype {expected_dtype}.")
            if t.topk_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("GIN route IDs must be int32 or int64.")
            if t.topk_weights.dtype != torch.float32:
                raise ValueError("GIN route weights must be FP32.")
            tensors = [t.hidden_states, t.topk_ids, t.topk_weights]
            if not quantize_input and t.scales is not None:
                if t.scales.dtype != torch.float8_e4m3fn:
                    raise ValueError("GIN activation SF must be float8_e4m3fn.")
                tensors.append(t.scales)
            current_device = torch.cuda.current_device()
            if current_device != self._gin_device:
                raise ValueError("GIN inputs must use the session's bound CUDA device.")
            device = torch.device("cuda", current_device)
            if any(
                tensor.device != device or not tensor.is_contiguous()
                for tensor in tensors
            ):
                raise ValueError(
                    "GIN inputs must be contiguous on the bound CUDA device."
                )
        validate_nvfp4_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )
        validate_fc1_activation_per_token_scale(
            t.fc1_activation_per_token_scale,
            num_tokens=t.hidden_states.shape[0],
            enabled=self._kernel_config.enable_fc1_activation_per_token_scale,
        )
        if self._kernel_config.internode and check_route_values:
            ids = t.topk_ids
            if torch.any((ids < -1) | (ids >= fleet_params.num_experts)).item():
                raise ValueError(
                    "GIN route IDs must be -1 or a valid global expert ID."
                )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        num_tokens = t.hidden_states.shape[0]
        # GIN validates eagerly on every public call; capture checks metadata
        # only and requires valid route values on every replay. Other backends
        # run validate_forward() only for the first forward of a
        # workspace (later forwards are gated on a stable input signature that
        # does not cover this optional tensor).  Presence and metadata of the
        # per-token scale are per-forward facts -- a forward that omits it
        # would otherwise silently reuse the previous batch's scales -- so
        # re-check them here on every staging.  Metadata only; no device work.
        validate_fc1_activation_per_token_scale(
            t.fc1_activation_per_token_scale,
            num_tokens=num_tokens,
            enabled=self._kernel_config.enable_fc1_activation_per_token_scale,
        )
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
            # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
            from ......kernel_src.sm100.cutedsl_megamoe import (
                Nvfp4BlockSize,
                ceil_div,
                round_up,
            )

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
            from ......kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

            note_staged_tokens(workspace.topk_idx, num_tokens)

        destinations = []
        sources = []
        for source, destination in (
            (t.fc1_alpha, workspace.fc1_alpha),
            (t.fc2_alpha, workspace.fc2_alpha),
            (t.fc1_norm_const, workspace.fc1_norm_const),
        ):
            if source is not None:
                destinations.append(destination)
                sources.append(source)
        if t.fc1_activation_per_token_scale is not None:
            # Validated above (shape (num_tokens,), fp32, enabled); only the live
            # [:num_tokens] rows of the per-token plane are staged.  Strided
            # sources are fine here: copy_ honours the source stride and the
            # workspace plane is contiguous.
            destinations.append(workspace.fc1_activation_per_token_scale[:num_tokens])
            sources.append(t.fc1_activation_per_token_scale)
        if sources:
            # Workspace aliases preserve the ordering of the individual copies.
            if any(
                torch._C._overlaps(source, destination)
                for source in sources
                for destination in destinations
            ):
                for destination, source in zip(destinations, sources, strict=True):
                    destination.copy_(source)
            else:
                # Batch the common contiguous CUDA copies into one launch.
                torch._foreach_copy_(destinations, sources)

    def validate_capture_ready(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> None:
        frontend = workspace._frontend
        if frontend.config.internode:
            error = None
            try:
                self._prepared_thunk_state(workspace, transformed_weights)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            frontend._gin_agree(error, "capture-ready")
            return
        mega = frontend._mega
        if mega is None or mega.compiled is None:
            raise RuntimeError(
                "MegaMoE workspace is not warmed for CUDA graph capture; "
                "call layer.warmup(..., workspace=workspace) first"
            )
        if frontend.config.use_custom_finalize and not self.supports_unfinalized_output:
            from flashinfer.jit.cake_megamoe_topk_reduce import (
                is_cake_megamoe_topk_reduce_module_loaded,
            )

            if not is_cake_megamoe_topk_reduce_module_loaded():
                raise RuntimeError(
                    "MegaMoE terminal reducer is not warmed for CUDA graph capture; "
                    "call layer.warmup(..., workspace=workspace) first"
                )
        # Materialize the thunk for the current capture stream before staging
        # mutates workspace buffers or records any graph node.
        self._prepared_thunk_state(workspace, transformed_weights)

    @staticmethod
    def _mega_inputs(
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> Any:
        from ......kernel_src.sm100.cutedsl_megamoe import MegaMoENvfp4Inputs

        return MegaMoENvfp4Inputs(
            activation=workspace.x,
            activation_sf=workspace.x_sf,
            topk_idx=workspace.topk_idx,
            topk_weights=workspace.topk_weights,
            fc1_weight=transformed_weights[0][0],
            fc1_weight_sf=transformed_weights[0][1],
            fc2_weight=transformed_weights[1][0],
            fc2_weight_sf=transformed_weights[1][1],
            fc1_alpha=workspace.fc1_alpha,
            fc2_alpha=workspace.fc2_alpha,
            fc1_norm_const=workspace.fc1_norm_const,
            output_activation=workspace.output_activation,
            fc1_activation_per_token_scale=workspace.fc1_activation_per_token_scale,
            num_valid_tokens=workspace.num_valid_tokens,
        )

    def _prepared_thunk_state(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> tuple:
        kcfg = self._kernel_config
        fe = workspace._frontend
        fe.set_swiglu_params(kcfg.swiglu_alpha, kcfg.swiglu_beta)
        clamp = _resolve_gate_up_clamp(kcfg)
        if clamp is not None:
            fe.set_gate_up_clamp(clamp)
        mega = fe._mega
        # GIN preparation is collective. A stream change on only one rank
        # must not create a local cache miss and a different collective order.
        # Its session thunk binds the current stream at execution/capture.
        stream = (
            None if fe.config.internode else torch.cuda.current_stream().cuda_stream
        )
        weight_identity = tuple(
            id(tensor) for transformed in transformed_weights for tensor in transformed
        )
        key = (
            id(workspace),
            weight_identity,
            id(mega.compiled) if mega is not None and mega.compiled else None,
            stream,
        )
        state = self._thunk_states.get(key)
        if state is None or key[2] is None:
            if fe.config.internode and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GIN capture requires a prepared launch; call layer.warmup() "
                    "with this workspace and weights before capture."
                )
            # A pooled frontend can be recompiled by another layer. Release
            # thunks retaining the old compiled session and its freed scratch.
            self._thunk_states = {
                cached_key: cached_state
                for cached_key, cached_state in self._thunk_states.items()
                if cached_key[0] != key[0] or cached_key[2] == key[2]
            }
            inputs = self._mega_inputs(workspace, transformed_weights)
            # Full validation happens inside make_launch_thunk's
            # _prepare_launch_inputs (run()'s slow-path validator).
            thunk = fe.make_launch_thunk(inputs)
            mega = fe._mega
            assert mega is not None and mega.compiled is not None
            key = (key[0], key[1], id(mega.compiled), stream)
            state = (key, thunk, workspace.output_activation)
            self._thunk_states[key] = state
        return state

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......kernel_src.sm100.cutedsl_megamoe import staged_tokens

        if self.supports_unfinalized_output:
            raise ValueError("finalize_mode=NONE requires compute_unfinalized()")
        if output is not None:
            num_tokens = output.shape[0]
        else:
            if self._autotune_pending:
                raise ValueError(
                    "compute(output=None) is incompatible with knobs='auto' "
                    "(the autotune sweep needs a caller output buffer)"
                )
            staged = staged_tokens(workspace.topk_idx)
            if staged is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to have "
                    "staged this workspace first"
                )
            num_tokens = staged

        kcfg = self._kernel_config
        if self._autotune_pending:
            # COLLECTIVE: every EP rank reaches this first compute() together,
            # so the candidate sweep stays in lockstep (see shim/autotune.py).
            from ......kernel_src.sm100.cutedsl_megamoe import autotune_nvfp4_mega_moe

            self._autotune_winner = autotune_nvfp4_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                swiglu_alpha=kcfg.swiglu_alpha,
                swiglu_beta=kcfg.swiglu_beta,
                gate_up_clamp=_resolve_gate_up_clamp(kcfg),
                activation_clamp=kcfg.activation_clamp,
            )
            # Cleared only on success: if the collective tune raises, a retried
            # compute() re-attempts it (all ranks fail together, so lockstep holds).
            self._autotune_pending = False
        # Steady-state launch thunk: nvfp4_mega_moe re-validates, re-resolves
        # the clamp, and rebuilds the 12-field inputs bundle on every call.
        # Build once per workspace, weights and compiled session. Off thunks
        # also key on STREAM because their launch
        # kwargs bind it at build time — a graph capture runs on a capture
        # stream and must get its own thunk or the kernel launch escapes the
        # graph. A knobs/activation change nulls the compiled session,
        # changing the key and forcing a rebuild through the validated path.
        state = self._prepared_thunk_state(workspace, transformed_weights)
        key, thunk, out_buf = state
        reducer_state = None
        if workspace._frontend.config.use_custom_finalize:
            partials, workspace_root, _region = (
                workspace._frontend.custom_finalize_workspace()
            )
            # Resolve/load the native module and borrowed view before the
            # upstream launch.  The terminal interval below must contain only
            # the two same-stream device launches—no lazy compilation,
            # allocation, copy, event, or synchronization.
            from flashinfer.jit.cake_megamoe_topk_reduce import (
                get_cake_megamoe_topk_reduce_module,
            )

            reducer_state = (
                get_cake_megamoe_topk_reduce_module(partials.device),
                partials,
                workspace_root,
                key[3],
            )
        thunk()
        if reducer_state is not None:
            reducer, partials, workspace_root, stream = reducer_state
            reducer.run(partials, out_buf, num_tokens, stream)
            del workspace_root
        if output is not None:
            output.copy_(out_buf[:num_tokens])
            return output
        # Zero-copy: the caller consumes the [:n] view under stream ordering
        # (valid until the next launch on this session's buffers).
        return out_buf[:num_tokens]

    def compute_unfinalized(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ......kernel_src.sm100.cutedsl_megamoe import staged_tokens

        if not self.supports_unfinalized_output:
            raise ValueError("compute_unfinalized() requires finalize_mode=NONE")
        num_tokens = staged_tokens(workspace.topk_idx)
        if num_tokens is None:
            raise ValueError("compute_unfinalized() requires stage_inputs() first")
        _, thunk, _ = self._prepared_thunk_state(workspace, transformed_weights)
        partials, _workspace_root, _region = (
            workspace._frontend.custom_finalize_workspace()
        )
        top_k, hidden = partials.shape[1:]
        gemm2_permuted = partials[:num_tokens].view(num_tokens * top_k, hidden)
        weights = workspace._unfinalized_weights[:num_tokens]
        expanded_idx_to_permuted_idx = workspace._unfinalized_route_map[:num_tokens]
        # The full-capacity communication launch is required even for T=0.
        thunk()
        return gemm2_permuted, weights, expanded_idx_to_permuted_idx

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        if k.internode:
            return None
        if k.knobs == "auto":
            # Autotune retunes (and recompiles) the workspace's shared
            # frontend at first compute; give each session its own buffer.
            return None
        import torch

        from ......core.kernel.workspace_pool import epilogue_pool_key, knobs_pool_key

        fp = fleet_params
        return (
            "sm100_nvfp4_nvfp4_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            2 * k.intermediate_size,
            _resolve_gate_up_clamp(k),
            k.swiglu_alpha,
            k.swiglu_beta,
            k.activation,
            k.situ_beta,
            k.situ_linear_beta,
            k.apply_routing_weights_before_fc2,
            k.enable_in_kernel_fc2_reduce,
            self._finalize_mode(fleet_params),
            epilogue_pool_key(k.num_valid_tokens_tensor),
            k.combine_dtype,
            epilogue_pool_key(k.fc1_alpha),
            epilogue_pool_key(k.fc2_alpha),
            epilogue_pool_key(k.fc1_norm_const),
            k.enable_fc1_activation_per_token_scale,
            knobs_pool_key(self._workspace_knobs(fleet_params)),
        )

    def _forget_local_workspace_state(self, workspace) -> None:
        # Launch thunks close over this backend owner's workspace tensors.
        # Drop them on every release, not only the final pooled release.
        workspace_id = id(workspace)
        self._thunk_states = {
            key: state
            for key, state in self._thunk_states.items()
            if key[0] != workspace_id
        }

    def _forget_workspace_state(self, workspace) -> None:
        # The fused-stage memo is process-global and keys on
        # topk_idx.data_ptr(); evict only before the final physical free.
        import sys

        quant_stage = sys.modules.get(
            "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.quant_stage"
        )
        topk_idx = getattr(workspace, "topk_idx", None)
        if quant_stage is not None and topk_idx is not None:
            quant_stage.forget_staged_tokens(topk_idx)
