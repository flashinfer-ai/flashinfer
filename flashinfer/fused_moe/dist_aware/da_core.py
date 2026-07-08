"""Typed orchestration boundary for distribution-aware fused MoE."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

import torch

from ...autotuner import AutoTuner, DynamicValueSpec, OptimizationProfile, StaticDim
from ...tllm_enums import DtypeTrtllmGen
from . import da_capture, da_profile, da_state
from .da_config import DAConfig
from .da_utils import generate_da_distribution_assignments, pack_expert_assignments


@dataclass(frozen=True)
class DABackend:
    """Core-owned operations consumed by DA orchestration."""

    get_moe_op: Callable[[], Any]
    capture_backend: Callable[[], da_capture.DACaptureBackend]


@dataclass(frozen=True)
class DAInputIndices:
    routing_logits: int
    topk_ids: int
    hidden_states: int


@dataclass(frozen=True)
class DAInvocation:
    """All wrapper-owned state needed by a single DA transaction."""

    da_context: da_state.DAMoeContext
    runner: Any
    tuning_inputs: Sequence[Optional[torch.Tensor]]
    input_indices: DAInputIndices
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    routing_logits: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    expert_weights: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: Optional[torch.Tensor]
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm1_clamp_limit: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: Optional[torch.Tensor]
    gemm2_bias: Optional[torch.Tensor]
    output1_scale_scalar: Optional[torch.Tensor]
    output1_scale_gate_scalar: Optional[torch.Tensor]
    output2_scale_scalar: Optional[torch.Tensor]
    output: torch.Tensor
    num_experts: int
    top_k: int
    n_group: Optional[int]
    topk_group: Optional[int]
    intermediate_size: int
    local_expert_offset: int
    num_local_experts: int
    routed_scaling_factor: Optional[float]
    routing_method_type: int
    activation_type: int
    num_tokens: int
    tune_max_num_tokens: int
    dtype_act: DtypeTrtllmGen
    norm_topk_prob: bool
    use_routing_scales_on_input: bool
    precomputed_topk_ids_are_packed: bool
    routing_input_mode: int
    enable_pdl: bool


@dataclass(frozen=True)
class DAExecution:
    """Immutable adapters, invocation, and configuration for one DA call."""

    backend: DABackend
    invocation: DAInvocation
    config: DAConfig


def create_config(config: Optional[DAConfig] = None) -> DAConfig:
    """Adopt a supplied snapshot or resolve one at the DA boundary."""

    return DAConfig() if config is None else config


def create_execution(
    backend: DABackend,
    invocation: DAInvocation,
    config: Optional[DAConfig] = None,
) -> DAExecution:
    """Create the immutable DA execution grouping for one wrapper call."""

    return DAExecution(backend, invocation, create_config(config))


def _bucketed_profile(
    invocation: DAInvocation, num_tokens_bucket: int
) -> OptimizationProfile:
    """Build the runner profile used to query a DA token bucket."""

    shapes = [
        [StaticDim(int(dim)) for dim in tensor.shape]
        if isinstance(tensor, torch.Tensor)
        else [StaticDim(0)]
        for tensor in invocation.tuning_inputs
    ]
    hidden_state_dims = shapes[invocation.input_indices.hidden_states]
    if not hidden_state_dims:
        raise ValueError("DA hidden_states input must have at least one dimension")
    hidden_state_dims[0] = StaticDim(max(int(num_tokens_bucket), 1))
    return OptimizationProfile(shapes, [None] * len(shapes))


def candidate_tile_sizes(
    execution: DAExecution, num_tokens_bucket: int
) -> tuple[int, ...]:
    """Return the tile values reported by this runner for a token bucket."""

    tactics = execution.invocation.runner.get_valid_tactics(
        list(execution.invocation.tuning_inputs),
        _bucketed_profile(execution.invocation, num_tokens_bucket),
    )
    tiles = tuple(
        sorted(
            {
                int(tactic[0])
                for tactic in tactics
                if hasattr(tactic, "__getitem__") and len(tactic) > 0
            }
        )
    )
    call = execution.invocation
    if call.dtype_act != DtypeTrtllmGen.E2m1 or execution.config.max_knn_tile is None:
        return tiles
    return tuple(tile for tile in tiles if tile <= execution.config.max_knn_tile)


def normalize_tactic(execution: DAExecution, tactic: Any) -> Any:
    """Reject a stale tactic whose tile is not valid for this runner profile."""

    if tactic == -1 or not hasattr(tactic, "__getitem__") or len(tactic) == 0:
        return tactic
    valid_tiles = candidate_tile_sizes(execution, execution.invocation.num_tokens)
    return tactic if int(tactic[0]) in valid_tiles else -1


def upload_bucket(num_tokens: int, tune_max_num_tokens: int) -> int:
    """Map a runtime token count to the populated, power-of-two DA bucket."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        return 1
    bucket = 1 << (num_tokens - 1).bit_length()
    return min(bucket, int(tune_max_num_tokens))


def debug_log(message: str, config: DAConfig) -> None:
    """Append a worker-local DA debug message when explicitly requested."""

    path = config.debug_file
    if not path:
        return
    try:
        pid = os.getpid()
        path = path.format(pid=pid)
        if path.endswith(os.sep):
            path = os.path.join(path, f"da_debug_{pid}.log")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"pid={pid} {message}\n")
    except OSError:
        pass


def switch_tile_sizes(
    execution: DAExecution, num_tokens_bucket: int
) -> tuple[int, ...]:
    """Return the exact DA selector body order for a runtime token bucket."""

    tiles = candidate_tile_sizes(execution, num_tokens_bucket)
    tile_map = da_state.PER_TILE_TACTICS.get(
        da_state.cache_key(execution.invocation.da_context, int(num_tokens_bucket))
    )
    if tile_map is None:
        return tiles
    return tuple(tile for tile in tiles if tile in tile_map)


def _subset_value_tensor_generator(
    bucket_id: int, profiled_tensor: torch.Tensor, original_tensor: torch.Tensor
) -> torch.Tensor:
    del bucket_id
    if not isinstance(original_tensor, torch.Tensor):
        return profiled_tensor
    if profiled_tensor.shape == original_tensor.shape:
        return original_tensor
    if (
        profiled_tensor.dim() == original_tensor.dim()
        and profiled_tensor.shape[1:] == original_tensor.shape[1:]
        and profiled_tensor.shape[0] <= original_tensor.shape[0]
    ):
        return original_tensor[: profiled_tensor.shape[0]].contiguous()
    return profiled_tensor


class DADistributionTensorGenerator:
    """Generate DA tuning inputs from immutable distribution policy."""

    def __init__(self, execution: DAExecution, *, pack_topk_ids: bool) -> None:
        self.execution = execution
        self.pack_topk_ids = pack_topk_ids

    def __call__(
        self,
        bucket_id: int,
        profiled_tensor: torch.Tensor,
        original_tensor: torch.Tensor,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        del original_tensor, inputs
        call = self.execution.invocation
        dist = da_profile.active_auto_distributions(self.execution.config)[
            int(bucket_id)
        ]
        if not isinstance(profiled_tensor, torch.Tensor) or profiled_tensor.dim() < 2:
            return profiled_tensor
        num_tokens = int(profiled_tensor.shape[0])
        device = profiled_tensor.device
        assignments = generate_da_distribution_assignments(
            dist,
            torch.zeros(num_tokens, call.top_k, dtype=torch.int32, device=device),
            call.num_local_experts,
            call.num_experts,
            call.top_k,
            call.local_expert_offset,
        )
        if (
            profiled_tensor.shape[-1] == call.top_k
            and not profiled_tensor.is_floating_point()
        ):
            if not self.pack_topk_ids:
                return assignments.to(dtype=profiled_tensor.dtype)
            weights = torch.full(
                (num_tokens, call.top_k),
                1.0 / max(call.top_k, 1),
                dtype=torch.bfloat16,
                device=device,
            )
            return pack_expert_assignments(assignments, weights, top_k=call.top_k).to(
                dtype=profiled_tensor.dtype
            )
        if profiled_tensor.is_floating_point():
            logits = torch.full(
                (num_tokens, call.num_experts),
                -20.0,
                dtype=torch.float32,
                device=device,
            )
            values = torch.linspace(
                20.0, 19.0, steps=max(call.top_k, 1), dtype=torch.float32, device=device
            ).reshape(1, call.top_k)
            logits.scatter_(
                1, assignments.to(torch.long), values.expand(num_tokens, -1)
            )
            return logits.to(dtype=profiled_tensor.dtype)
        return profiled_tensor


def should_attach_value_specs(execution: DAExecution, tuner: AutoTuner) -> bool:
    if not execution.config.enabled or torch.cuda.is_current_stream_capturing():
        return False
    if not tuner.is_tuning_mode:
        return False
    context = execution.invocation.da_context
    return not (
        da_state.context_supports_knn_capture(context)
        and da_profile.bundle_has_tactics()
        and (
            not da_state.BUNDLE_TACTIC_CONTEXTS
            or context in da_state.BUNDLE_TACTIC_CONTEXTS
        )
    )


def make_value_specs(
    execution: DAExecution, *, pack_topk_ids: bool
) -> tuple[DynamicValueSpec, ...]:
    """Create value-aware tile and distribution specs for one DA execution."""

    call = execution.invocation
    tile_sizes = candidate_tile_sizes(execution, int(call.hidden_states.shape[0]))
    if not tile_sizes:
        return ()

    def tile_buckets(profile: OptimizationProfile) -> tuple[int, ...]:
        num_tokens = profile.get_opt_shapes()[call.input_indices.hidden_states][0]
        return candidate_tile_sizes(execution, int(num_tokens))

    value_input_idx = (
        call.input_indices.routing_logits
        if call.routing_logits is not None
        else call.input_indices.topk_ids
    )
    return (
        DynamicValueSpec(
            input_idx=value_input_idx,
            gen_value_buckets=tile_buckets,
            map_to_value_bucket=lambda _tensor, _inputs, _kwargs: tile_sizes[0],
            tensor_value_generator=_subset_value_tensor_generator,
        ),
        DynamicValueSpec(
            input_idx=value_input_idx,
            gen_value_buckets=tuple(
                range(len(da_profile.active_auto_distributions(execution.config)))
            ),
            map_to_value_bucket=lambda _tensor: 0,
            tensor_value_generator=DADistributionTensorGenerator(
                execution, pack_topk_ids=pack_topk_ids
            ),
        ),
    )


def tuning_config_kwargs(
    execution: DAExecution,
    tuner: AutoTuner,
    *,
    pack_topk_ids: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Return DA-specific value-profile kwargs when distribution-aware tuning is active."""

    kwargs: dict[str, Any] = {}
    if not should_attach_value_specs(execution, tuner):
        return kwargs, False
    value_specs = make_value_specs(execution, pack_topk_ids=pack_topk_ids)
    if not value_specs:
        return kwargs, False
    distribution_bucket_index = len(value_specs) - 1

    def value_sample_count(value_buckets, default_sample_count):
        if len(value_buckets) <= distribution_bucket_index:
            return default_sample_count
        try:
            distribution = da_profile.active_auto_distributions(execution.config)[
                int(value_buckets[distribution_bucket_index])
            ]
        except (IndexError, TypeError, ValueError):
            return default_sample_count
        return (
            1
            if distribution[1] == "single"
            else execution.config.distribution_sample_count
        )

    kwargs["value_specs"] = value_specs
    kwargs["value_sample_count"] = value_sample_count
    return kwargs, True


def resolve_static_fallback(
    execution: DAExecution,
    tuner: Any,
    *,
    custom_op: str,
    runner: Any,
    tactic: Any,
) -> Any:
    """Use the normal static profile only when DA tuning has no winner."""

    if tactic != -1 or not execution.config.enabled or tuner.is_tuning_mode:
        return tactic
    call = execution.invocation
    profile_bucket = min(
        1 << (max(int(call.num_tokens), 1).bit_length() - 1),
        int(call.tune_max_num_tokens),
    )
    fallback = da_profile.best_static_tactic_from_profiles(
        tuner,
        custom_op,
        runner.__class__.__name__,
        hash(runner),
        profile_bucket,
        da_context=call.da_context,
    )
    if fallback is None:
        return tactic
    tactic = fallback[0]
    if execution.config.verbose:
        print(
            f"[DA static-fallback] op={custom_op} bucket={profile_bucket} "
            f"tactic={tactic} time_ms={fallback[1]:.6f}",
            flush=True,
        )
    return tactic


def _profile_backend(execution: DAExecution) -> da_profile.DAProfileBackend:
    """Adapt the wrapper-owned FFI accessor to profile publication."""

    return da_profile.DAProfileBackend(
        get_ffi_moe_op=execution.backend.get_moe_op,
        supported_tile_sizes=lambda bucket, **_kwargs: switch_tile_sizes(
            execution, bucket
        ),
    )


def publish_tactics_from_autotune(
    execution: DAExecution,
    tuner: Any,
    *,
    custom_op: str,
    runner: Any,
    da_value_specs_active: bool,
) -> None:
    """Publish DA per-tile tactics after a real value-aware tuning pass."""

    if not (da_value_specs_active and tuner.is_tuning_mode):
        return
    da_profile.populate_per_tile_tactics_from_autotune(
        _profile_backend(execution),
        tuner,
        custom_op,
        da_context=execution.invocation.da_context,
        runner_hash=hash(runner),
        config=execution.config,
    )


def maybe_prepare_bundle(execution: DAExecution, tuner: Any) -> None:
    """Load a compatible bundle or register one immutable live profile."""

    call = execution.invocation
    backend = _profile_backend(execution)

    def register_live_profile() -> None:
        da_profile.register_auto_profile_callback(
            backend,
            tuner,
            execution.backend.get_moe_op(),
            da_context=call.da_context,
            hidden_states=call.hidden_states,
            hidden_states_scale=call.hidden_states_scale,
            gemm1_weights=call.gemm1_weights,
            gemm1_weights_scale=call.gemm1_weights_scale,
            gemm2_weights=call.gemm2_weights,
            gemm2_weights_scale=call.gemm2_weights_scale,
            output1_scale_scalar=call.output1_scale_scalar,
            output1_scale_gate_scalar=call.output1_scale_gate_scalar,
            output2_scale_scalar=call.output2_scale_scalar,
            num_experts=call.num_experts,
            top_k=call.top_k,
            n_group=call.n_group,
            topk_group=call.topk_group,
            intermediate_size=call.intermediate_size,
            local_expert_offset=call.local_expert_offset,
            num_local_experts=call.num_local_experts,
            routed_scaling_factor=call.routed_scaling_factor,
            routing_method_type=call.routing_method_type,
            activation_type=call.activation_type,
            tune_max_num_tokens=call.tune_max_num_tokens,
            config=execution.config,
        )

    da_profile.maybe_prepare_bundle(
        tuner,
        backend,
        da_context=call.da_context,
        hidden_states=call.hidden_states,
        top_k=call.top_k,
        intermediate_size=call.intermediate_size,
        num_local_experts=call.num_local_experts,
        activation_type=call.activation_type,
        register_live_profile=register_live_profile,
        debug_log=lambda message: debug_log(message, execution.config),
        config=execution.config,
    )


def try_capture_dispatch(
    execution: DAExecution,
    run_from_routing_metadata: Callable[
        [da_capture.RoutingMetadataBundle, Sequence[int]], None
    ],
    direct_body: Optional[Callable[[Sequence[int], dict[str, Any]], None]] = None,
) -> Optional[list[torch.Tensor]]:
    """Delegate capture dispatch with the execution's immutable configuration."""

    call = execution.invocation
    capture_backend = replace(
        execution.backend.capture_backend(),
        supported_tile_sizes=lambda bucket, **_kwargs: switch_tile_sizes(
            execution, bucket
        ),
    )
    return da_capture.try_trtllm_capture_aware_da(
        backend=capture_backend,
        upload_bucket=upload_bucket,
        debug_log=lambda message: debug_log(message, execution.config),
        da_context=call.da_context,
        run_from_routing_metadata=run_from_routing_metadata,
        direct_body=direct_body,
        precomputed_topk_ids_are_packed=call.precomputed_topk_ids_are_packed,
        routing_input_mode=call.routing_input_mode,
        hidden_states=call.hidden_states,
        hidden_states_scale=call.hidden_states_scale,
        routing_logits=call.routing_logits,
        topk_ids=call.topk_ids,
        expert_weights=call.expert_weights,
        routing_bias=call.routing_bias,
        gemm1_weights=call.gemm1_weights,
        gemm1_weights_scale=call.gemm1_weights_scale,
        gemm1_bias=call.gemm1_bias,
        gemm1_alpha=call.gemm1_alpha,
        gemm1_beta=call.gemm1_beta,
        gemm1_clamp_limit=call.gemm1_clamp_limit,
        gemm2_weights=call.gemm2_weights,
        gemm2_weights_scale=call.gemm2_weights_scale,
        gemm2_bias=call.gemm2_bias,
        output1_scale_scalar=call.output1_scale_scalar,
        output1_scale_gate_scalar=call.output1_scale_gate_scalar,
        output2_scale_scalar=call.output2_scale_scalar,
        output=call.output,
        num_experts=call.num_experts,
        top_k=call.top_k,
        n_group=call.n_group,
        topk_group=call.topk_group,
        intermediate_size=call.intermediate_size,
        local_expert_offset=call.local_expert_offset,
        num_local_experts=call.num_local_experts,
        routed_scaling_factor=call.routed_scaling_factor,
        routing_method_type=call.routing_method_type,
        enable_pdl=call.enable_pdl,
        activation_type=call.activation_type,
        num_tokens=call.num_tokens,
        tune_max_num_tokens=call.tune_max_num_tokens,
        dtype_act=call.dtype_act,
        norm_topk_prob=call.norm_topk_prob,
        use_routing_scales_on_input=call.use_routing_scales_on_input,
        config=execution.config,
    )
