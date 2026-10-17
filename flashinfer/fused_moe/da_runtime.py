"""Automatic tuning, preparation, and capture dispatch for TRTLLM DA MoE."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import torch

from flashinfer.autotuner import AutoTuner, TuningConfig, TunableRunner
from flashinfer.fused_moe.da_config import TrtllmDaConfig
from flashinfer.fused_moe.da_moe import (
    DABody,
    DAMoEDispatcher,
    DAPlan,
    DAPlanMode,
    DAResourceLeaseConflict,
    tensor_binding_signature,
)
from flashinfer.fused_moe.da_tuner import (
    DAPlanCompiler,
    DAProfileSelection,
    DADistribution,
    FactorizedSearch,
    FactorizedTactic,
    FullOpMeasurementCache,
    RoutingRealization,
    RoutingRealizationFactory,
    RoutingRealizationKey,
    publish_compiled_plan,
)
from flashinfer.jit.core import logger
from flashinfer.tllm_enums import RoutingInputMode


_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class TrtllmDaRoutingAdapter:
    """Stage and restore one public precomputed-routing representation."""

    # Fused-MoE flat input containing raw or packed expert IDs.
    routing_id_index: int
    # Fused-MoE flat input containing caller or launcher routing weights.
    routing_weight_index: int
    # Native routing representation enum used by the launcher.
    routing_input_mode: int

    def snapshot(self, inputs: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Clone the two mutable routing slots before value-aware profiling."""
        expert_ids = inputs[self.routing_id_index]
        routing_weights = inputs[self.routing_weight_index]
        if not isinstance(expert_ids, torch.Tensor) or not isinstance(
            routing_weights, torch.Tensor
        ):
            raise TypeError("DA routing adapter requires two tensor slots")
        return expert_ids.clone(), routing_weights.clone()

    def stage(self, inputs: Sequence[Any], realization: RoutingRealization) -> None:
        """Stage raw or packed expert IDs plus changing BF16 weights in place."""
        # Resolve the two declared mutable slots once; every other operation input retains
        # identical values and storage throughout value-aware profiling.
        expert_ids = inputs[self.routing_id_index]
        routing_weights = inputs[self.routing_weight_index]
        if not isinstance(expert_ids, torch.Tensor) or not isinstance(
            routing_weights, torch.Tensor
        ):
            raise TypeError("DA routing adapter requires two tensor slots")
        # Encode the canonical realization according to the exact public routing ABI while
        # preserving the caller's tensor addresses.
        if self.routing_input_mode == RoutingInputMode.UnpackedPrecomputed:
            expert_ids.copy_(realization.expert_ids)
        elif self.routing_input_mode == RoutingInputMode.PackedPrecomputed:
            weight_bits = (
                realization.routing_weights.contiguous()
                .view(torch.int16)
                .to(torch.int32)
                .bitwise_and_(0xFFFF)
            )
            packed = realization.expert_ids.bitwise_left_shift(16).bitwise_or_(
                weight_bits
            )
            expert_ids.copy_(packed)
        elif self.routing_input_mode == RoutingInputMode.FromLogits:
            expert_ids.fill_(-32)
            selected_logits = (
                realization.routing_weights.float().clamp_min_(1e-6).log().add_(32)
            )
            expert_ids.scatter_(
                1,
                realization.expert_ids.to(torch.int64),
                selected_logits.to(expert_ids),
            )
        else:
            raise ValueError("Unsupported DA routing representation")
        if routing_weights.numel() != 0:
            routing_weights.copy_(realization.routing_weights)

    def restore(
        self,
        inputs: Sequence[Any],
        snapshot: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Restore both public routing slots after value-aware profiling."""
        expert_ids = inputs[self.routing_id_index]
        routing_weights = inputs[self.routing_weight_index]
        assert isinstance(expert_ids, torch.Tensor)
        assert isinstance(routing_weights, torch.Tensor)
        expert_ids.copy_(snapshot[0])
        routing_weights.copy_(snapshot[1])


@dataclass(frozen=True)
class TrtllmDaOperationKey:
    """Process-local identity of one shape/static-configuration DA domain."""

    # Public custom-operation name sharing the ordinary AutoTuner namespace.
    custom_op: str
    # CUDA device owning profiling, preparation, and captured resources.
    device_index: int
    # Shape and dtype identity for every positional runner input.
    input_identity: tuple[tuple[tuple[int, ...], str], ...]
    # Deterministic dtype family and immutable inner-runner configuration.
    runner_identity: str
    # Concrete token count represented by this runtime/capture domain.
    num_tokens: int
    # Global expert domain represented by selector spectra.
    num_experts: int
    # First global expert ID owned by the local rank.
    local_expert_offset: int
    # Local expert count sampled by distribution realizations.
    num_local_experts: int
    # Number of selected experts in each token row.
    top_k: int
    # Public routing-method enum affecting the operation contract.
    routing_method_type: int
    # Precomputed routing representation consumed during replay.
    routing_input_mode: int
    # Flat input position containing live global expert IDs.
    routing_id_index: int
    # Flat input position containing live BF16 routing weights.
    routing_weight_index: int
    # Public scaling applied to the mutable routing weights.
    routed_scaling_factor: float
    # Stable JSON search/profiling configuration identity.
    config_identity: str

    def cache_key(self) -> str:
        """Return a deterministic JSON spelling for namespaced cache records."""
        # Device index is omitted because AutoTuner's outer environment identity already
        # partitions records by accelerator.
        # Compact sorted JSON makes logically identical operation domains byte-for-byte equal.
        return json.dumps(
            {
                "custom_op": self.custom_op,
                "input_identity": self.input_identity,
                "runner_identity": self.runner_identity,
                "num_tokens": self.num_tokens,
                "num_experts": self.num_experts,
                "local_expert_offset": self.local_expert_offset,
                "num_local_experts": self.num_local_experts,
                "top_k": self.top_k,
                "routing_method_type": self.routing_method_type,
                "routing_input_mode": self.routing_input_mode,
                "routing_id_index": self.routing_id_index,
                "routing_weight_index": self.routing_weight_index,
                "routed_scaling_factor": self.routed_scaling_factor,
                "config_identity": self.config_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def make_trtllm_da_operation_key(
    custom_op: str,
    runner: TunableRunner,
    inputs: Sequence[Any],
    config: TrtllmDaConfig,
    *,
    num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    top_k: int,
    routing_method_type: int,
    routing_input_mode: int,
    routing_id_index: int,
    routing_weight_index: int,
    routed_scaling_factor: float | None,
) -> TrtllmDaOperationKey:
    """Build one automatic DA registry identity from public operation inputs."""
    # Derive the device and concrete token bucket from the first tensor in the exact runner ABI.
    tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    # Mutable routing contents deliberately do not split the shape/static-configuration domain.
    input_identity = tuple(
        (tuple(value.shape), str(value.dtype))
        if isinstance(value, torch.Tensor)
        else ((0,), type(value).__name__)
        for value in inputs
    )
    return TrtllmDaOperationKey(
        custom_op=custom_op,
        device_index=device_index,
        input_identity=input_identity,
        runner_identity=_stable_runner_identity(runner),
        num_tokens=int(tensor.shape[0]),
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        top_k=top_k,
        routing_method_type=routing_method_type,
        routing_input_mode=routing_input_mode,
        routing_id_index=routing_id_index,
        routing_weight_index=routing_weight_index,
        routed_scaling_factor=(
            1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
        ),
        config_identity=json.dumps(
            config.cache_identity(), sort_keys=True, separators=(",", ":")
        ),
    )


def _stable_runner_identity(runner: TunableRunner) -> str:
    """Serialize immutable runner configuration without process-local hashes."""

    # Normalize enums, dtypes, and nested sequences into deterministic JSON-compatible values.
    def normalize(value: Any) -> Any:
        """Convert one immutable runner value into deterministic JSON data."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if hasattr(value, "value"):
            return normalize(value.value)
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        return repr(value)

    # Runtime caches contain process-local details and must not affect persistent tactic identity.
    fields = {
        name: normalize(value)
        for name, value in runner.__dict__.items()
        if not name.endswith("_cache")
    }
    return json.dumps(
        {"class": runner.__class__.__qualname__, "fields": fields},
        sort_keys=True,
        separators=(",", ":"),
    )


def collect_trtllm_da_bindings(
    inputs: Sequence[Any], runner_kwargs: Mapping[str, Any]
) -> tuple[torch.Tensor, ...]:
    """Collect every stable tensor address captured by inputs or exact-ABI kwargs."""
    bindings: list[torch.Tensor] = []
    seen: set[int] = set()
    for value in (*inputs, *runner_kwargs.values()):
        if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
            bindings.append(value)
            seen.add(value.data_ptr())
    return tuple(bindings)


def run_dist_aware_tactic(
    *,
    custom_op: str,
    tuner: AutoTuner,
    config: TrtllmDaConfig,
    runner: TunableRunner,
    runtime: Any,
    tuning_config: TuningConfig,
    inputs: list[Any],
    runner_kwargs: Mapping[str, Any],
    baseline_tactic: Any,
    routing_input_mode: int,
    routing_id_index: int,
    routing_weight_index: int,
    routing_precomputed_id_index: int | None = None,
    num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    top_k: int,
    routing_method_type: int,
    routed_scaling_factor: float | None,
    run_fixed_tactic: Callable[[Any], _ResultT],
    finish_switch: Callable[[Any], _ResultT],
) -> _ResultT:
    """Run the shared automatic DA lifecycle around one exact ordinary ABI."""
    # The adapter declares the only content-mutable inputs used by value-aware profiling and
    # later device-side distribution selection.
    routing_adapter = TrtllmDaRoutingAdapter(
        routing_id_index=routing_id_index,
        routing_weight_index=routing_weight_index,
        routing_input_mode=routing_input_mode,
    )
    # Scaling changes routed values, output semantics, and tactic timing, so it partitions DA state.
    key = make_trtllm_da_operation_key(
        custom_op,
        runner,
        inputs,
        config,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        top_k=top_k,
        routing_method_type=routing_method_type,
        routing_input_mode=routing_input_mode,
        routing_id_index=routing_id_index,
        routing_weight_index=routing_weight_index,
        routed_scaling_factor=routed_scaling_factor,
    )
    # Exclude representation-inactive scratch slots from the CUDA Graph binding signature while
    # retaining every address consumed by the active logits/packed/unpacked ABI.
    inactive_binding_indices: set[int] = set()
    if routing_input_mode == RoutingInputMode.PackedPrecomputed:
        inactive_binding_indices.add(routing_weight_index)
    elif routing_input_mode == RoutingInputMode.FromLogits:
        if routing_precomputed_id_index is None:
            raise ValueError("FromLogits DA requires its inactive precomputed-ID slot")
        inactive_binding_indices.update(
            (routing_precomputed_id_index, routing_weight_index)
        )
    binding_inputs = [
        value
        for index, value in enumerate(inputs)
        if index not in inactive_binding_indices
    ]
    bindings = collect_trtllm_da_bindings(binding_inputs, runner_kwargs)
    # Generic AutoTuner context owns profiling. Ordinary calls restore published state and never
    # tune implicitly.
    if tuner.is_tuning_mode:
        state = TRTLLM_DA_REGISTRY.get_or_create(key)
        state.tune_and_prepare(
            tuner=tuner,
            config=config,
            runner=runner,
            runtime=runtime,
            tuning_config=tuning_config,
            inputs=inputs,
            routing_adapter=routing_adapter,
            baseline_tactic=baseline_tactic,
            runner_kwargs=runner_kwargs,
            bindings=bindings,
        )
        TRTLLM_DA_REGISTRY.publish_cache(tuner)
    else:
        state = TRTLLM_DA_REGISTRY.find_or_restore(key, tuner)
    # Warmup prepares graph-stable resources outside capture; a missing or rejected plan keeps the
    # exact ordinary baseline path.
    if state is None:
        return run_fixed_tactic(baseline_tactic)
    if not torch.cuda.is_current_stream_capturing():
        state.prepare(
            runtime=runtime,
            inputs=inputs,
            runner_kwargs=runner_kwargs,
            bindings=bindings,
        )

    def capture_switch(
        plan: DAPlan,
        resources: Any,
        expected_capture_id: int,
        previous_conditional_node_handle: int,
    ) -> Any:
        """Inject one exact-ABI SWITCH and retain runtime topology evidence."""
        # Recover only the live routing representation consumed by selector and fused preamble.
        topk_ids = inputs[routing_id_index]
        if not isinstance(topk_ids, torch.Tensor):
            raise TypeError("DA capture requires a tensor expert-ID input")
        topk_weights = (
            inputs[routing_weight_index]
            if routing_input_mode == RoutingInputMode.UnpackedPrecomputed
            else None
        )
        if topk_weights is not None and not isinstance(topk_weights, torch.Tensor):
            raise TypeError("DA capture requires tensor routing weights")
        # The dtype-specific runtime injects bodies; this shared layer records topology and
        # constructs the unchanged public result after graph mutation succeeds.
        capture_result = runtime.capture_switch(
            plan,
            resources,
            inputs,
            topk_ids,
            expected_capture_id=expected_capture_id,
            previous_conditional_node_handle=previous_conditional_node_handle,
            topk_weights=topk_weights,
            **runner_kwargs,
        )
        if capture_result is None:
            return None
        topology, conditional_node_handle = capture_result
        state.record_topology(topology)
        from flashinfer.fused_moe.da_moe import DACaptureOutcome

        return DACaptureOutcome(
            finish_switch(resources), topology, conditional_node_handle
        )

    return state.dispatch(
        bindings,
        run_fallback=lambda: run_fixed_tactic(baseline_tactic),
        run_body=lambda body: run_fixed_tactic([body.tile_n, body.tactic]),
        capture_switch=capture_switch,
    )


class TrtllmDaOperationState:
    """Own one automatic DA domain from tuning through graph lease creation."""

    def __init__(self, key: TrtllmDaOperationKey) -> None:
        """Create pristine fallback state for one immutable operation domain."""
        # Immutable process-local registry identity.
        self.key = key
        # Sole plan, resource, dispatch, and graph-lease policy owner.
        self.dispatcher = DAMoEDispatcher(key.num_experts)
        # Exact full-operation timings reused within this domain.
        self._measurements = FullOpMeasurementCache()
        # Cached mutable routing realizations generated outside tactic timing.
        self._realizations = RoutingRealizationFactory()
        # Production typed runtime retained after successful tuning.
        self._runtime: Any | None = None
        # True after tuning admitted or rejected one complete policy.
        self._tuned = False
        # Compiled policy spelling retained for cache and benchmark diagnostics.
        self._published_policy: str | None = None
        # Host-selected body used when CUDA Graph replay is unavailable.
        self._eager_body: DABody | None = None
        # Preferred measured distribution that supplied the host-selected body.
        self._eager_distribution: str | None = None
        # Sticky preparation failures isolated by plan generation and exact bindings.
        self._preparation_failures: dict[tuple[int, tuple[int, ...]], str] = {}
        # Latest transient capacity conflict, cleared automatically by a successful retry.
        self._transient_preparation_failure: str | None = None
        # Deliberate policy-level fallback reason independent of binding preparation failures.
        self._policy_fallback_reason: str | None = None
        # Latest runtime-inspected SWITCH topology for diagnostics and benchmarks.
        self._last_topology: Any | None = None
        # Reentrant state lock serializing tuning and resource mutation.
        self._lock = threading.RLock()

    @property
    def tuned(self) -> bool:
        """Return whether this operation domain has compiled one policy."""
        return self._tuned

    @property
    def last_topology(self) -> Any | None:
        """Return the latest runtime-inspected conditional graph topology."""
        return self._last_topology

    def tune_and_prepare(
        self,
        *,
        tuner: AutoTuner,
        config: TrtllmDaConfig,
        runner: TunableRunner,
        runtime: Any,
        tuning_config: TuningConfig,
        inputs: list[Any],
        routing_adapter: TrtllmDaRoutingAdapter,
        baseline_tactic: Any,
        runner_kwargs: Mapping[str, Any],
        bindings: Sequence[torch.Tensor],
    ) -> DAPlan | None:
        """Profile distributions once, publish a plan, and prepare stable resources."""
        # Serialize the one-time tuning/publication transition for this operation domain.
        with self._lock:
            if not self._tuned:
                self._compile_plan(
                    tuner=tuner,
                    config=config,
                    runner=runner,
                    tuning_config=tuning_config,
                    inputs=inputs,
                    routing_adapter=routing_adapter,
                    baseline_tactic=baseline_tactic,
                    runner_kwargs=runner_kwargs,
                    runtime=runtime,
                )
                # Retain the exact dtype runtime only after policy compilation succeeds.
                self._runtime = runtime
                self._tuned = True
            # Resource preparation is binding-specific and may refresh during later warmup calls.
            self._prepare_resources(inputs, runner_kwargs, bindings)
            return self.dispatcher.plan

    def _compile_plan(
        self,
        *,
        tuner: AutoTuner,
        config: TrtllmDaConfig,
        runner: TunableRunner,
        tuning_config: TuningConfig,
        inputs: list[Any],
        routing_adapter: TrtllmDaRoutingAdapter,
        baseline_tactic: Any,
        runner_kwargs: Mapping[str, Any],
        runtime: Any,
    ) -> None:
        """Measure factorized candidates and compile the confirmed baseline guard."""
        normalized_baseline = tuple(int(value) for value in baseline_tactic)
        if len(normalized_baseline) != 2 or normalized_baseline[1] < 0:
            raise RuntimeError(
                "DA tuning requires one concrete ordinary baseline tactic"
            )

        # FromLogits bodies require the fused multi-tile preamble during candidate profiling.
        # Reject unsupported large-token shapes before staging any routing values or allocating
        # profiling arenas, and publish an explicit ordinary fixed-tactic fallback policy.
        if self.key.routing_input_mode == RoutingInputMode.FromLogits:
            max_tokens = runtime.max_multi_tile_tokens(self.key.num_experts)
            if self.key.num_tokens > max_tokens:
                self.dispatcher.clear_plan()
                self._published_policy = DAPlanMode.DA_FALLBACK.value
                self._eager_body = DABody(
                    tile_n=normalized_baseline[0], tactic=normalized_baseline[1]
                )
                self._eager_distribution = None
                self._policy_fallback_reason = (
                    "FromLogits DA requires fused multi-tile routing metadata, "
                    f"which supports at most {max_tokens} tokens for "
                    f"{self.key.num_experts} experts"
                )
                return

        # Snapshot caller routing before staging synthetic realizations; the finally block below
        # restores it even when profiling or plan compilation fails.
        original_routing = routing_adapter.snapshot(inputs)
        factorized_space = runner.get_factorized_tactic_space(inputs)  # type: ignore[attr-defined]

        # Logits routing is canonicalized outside body timing so every tactic profiles the same
        # precomputed mutable expert-ID/weight pair.
        profile_inputs = inputs
        profile_kwargs = dict(runner_kwargs)
        profile_tuning_config = tuning_config
        profile_runner = runner
        canonical_profile = None
        if self.key.routing_input_mode == RoutingInputMode.FromLogits:
            canonical_profile, profile_inputs, profile_kwargs = (
                runtime.prepare_from_logits_profile(
                    inputs, runner_kwargs, normalized_baseline[0]
                )
            )
            profile_tuning_config = runtime.make_from_logits_profile_tuning_config(
                profile_inputs, self.key.num_tokens
            )
            profile_runner = runtime.make_from_logits_profile_runner(canonical_profile)

        # Candidate selection and guard admission share measurements but remain separate phases.
        compiler = DAPlanCompiler(
            num_experts=self.key.num_experts,
            guard_enabled=config.baseline_guard_enabled,
            margin=config.baseline_guard_margin,
            control_overhead_us=config.control_overhead_us,
        )
        selections: list[DAProfileSelection] = []
        # Each generated realization is staged once per profile point and reused across every
        # compared complete tactic and its matched baseline.
        try:
            sample_index = 0
            for distribution in config.distributions:
                for _ in range(config.samples_per_distribution):
                    realization_key = self._realization_key(distribution, sample_index)
                    realization = self._realizations.get_or_create(realization_key)
                    routing_adapter.stage(inputs, realization)
                    selection_ids = realization.expert_ids
                    if canonical_profile is not None:
                        runtime.refresh_canonical_routing(
                            canonical_profile,
                            inputs,
                            runner_kwargs,
                        )
                        selection_ids = canonical_profile.routing_replay_ids.clone()
                    effective_config, batches = tuner.prepare_tactic_profile(
                        profile_inputs, profile_tuning_config
                    )

                    def measure(tactic: FactorizedTactic, decisive: bool) -> float:
                        """Measure one complete factorization on the shared input schedule."""
                        _ = decisive
                        identity = tuple(int(value) for value in tactic.tactic)

                        def profile_candidate() -> float:
                            """Prepare all lanes, then time preamble and typed body together."""
                            if canonical_profile is not None:
                                profile_runner.prepare_batches(  # type: ignore[attr-defined]
                                    batches, identity, **profile_kwargs
                                )
                            return tuner.profile_tactic(
                                profile_runner,
                                profile_inputs,
                                list(identity),
                                effective_config,
                                batches,
                                **profile_kwargs,
                            )

                        return self._measurements.measure(
                            (realization_key, "da", identity), profile_candidate
                        )

                    if config.factorized_search:
                        selected = FactorizedSearch(max_sweeps=2).search(
                            factorized_space, measure
                        )
                    else:
                        selected = min(
                            factorized_space.all_tactics(),
                            key=lambda tactic: (
                                measure(tactic, True),
                                repr(tactic.tactic),
                            ),
                        )
                    candidate_latency = measure(selected, True)
                    if not config.baseline_guard_enabled:
                        baseline_latency = None
                    elif (
                        canonical_profile is None
                        and tuple(selected.tactic) == normalized_baseline
                    ):
                        baseline_latency = candidate_latency
                    else:
                        baseline_config, baseline_batches = (
                            tuner.prepare_tactic_profile(inputs, tuning_config)
                        )
                        baseline_latency = self._measurements.measure(
                            (realization_key, "noda", normalized_baseline),
                            lambda: tuner.profile_tactic(
                                runner,
                                inputs,
                                list(normalized_baseline),
                                baseline_config,
                                baseline_batches,
                                **runner_kwargs,
                            ),
                        )
                    selections.append(
                        DAProfileSelection(
                            realization_key=realization_key,
                            expert_ids=selection_ids,
                            selected_tactic=selected,
                            candidate_latency_ms=candidate_latency,
                            baseline_latency_ms=baseline_latency,
                        )
                    )
                    sample_index += 1

            # Measure each retained body against every exemplar only when a guarded multi-body
            # plan could profitably collapse to one body and remove control overhead.
            candidate_bodies = tuple(
                dict.fromkeys(selection.selected_tactic for selection in selections)
            )
            candidate_latencies: dict[tuple[RoutingRealizationKey, Any], float] = {}
            if config.baseline_guard_enabled and len(candidate_bodies) > 1:
                for selection in selections:
                    realization = self._realizations.get_or_create(
                        selection.realization_key
                    )
                    routing_adapter.stage(inputs, realization)
                    if canonical_profile is not None:
                        runtime.refresh_canonical_routing(
                            canonical_profile,
                            inputs,
                            runner_kwargs,
                        )
                    effective_config, batches = tuner.prepare_tactic_profile(
                        profile_inputs, profile_tuning_config
                    )
                    for body in candidate_bodies:
                        identity = tuple(int(value) for value in body.tactic)

                        def profile_candidate() -> float:
                            """Prepare retained-body lanes before full-operation timing."""
                            if canonical_profile is not None:
                                profile_runner.prepare_batches(  # type: ignore[attr-defined]
                                    batches, identity, **profile_kwargs
                                )
                            return tuner.profile_tactic(
                                profile_runner,
                                profile_inputs,
                                list(identity),
                                effective_config,
                                batches,
                                **profile_kwargs,
                            )

                        candidate_latencies[
                            (selection.realization_key, body.tactic)
                        ] = self._measurements.measure(
                            (selection.realization_key, "da", identity),
                            profile_candidate,
                        )
                selections = list(
                    compiler.prefer_control_aware_singleton(
                        selections, candidate_latencies
                    )
                )
        finally:
            routing_adapter.restore(inputs, original_routing)

        # Publish only after routing restoration and complete host-side compilation succeed.
        compiled = compiler.compile(selections, normalized_baseline)
        publish_compiled_plan(self.dispatcher, compiled)
        self._published_policy = compiled.policy.value
        self._eager_body = (
            None if compiled.eager_tactic is None else compiled.eager_tactic.to_body()
        )
        self._eager_distribution = compiled.eager_distribution

    def prepare(
        self,
        *,
        runtime: Any,
        inputs: list[Any],
        runner_kwargs: Mapping[str, Any],
        bindings: Sequence[torch.Tensor],
    ) -> None:
        """Prepare this tuned state's current bindings during ordinary warmup."""
        with self._lock:
            if not self._tuned:
                return
            self._runtime = runtime
            self._prepare_resources(inputs, runner_kwargs, bindings)

    def _realization_key(
        self, distribution: DADistribution, sample_index: int
    ) -> RoutingRealizationKey:
        """Build one cached mutable-routing realization identity."""
        return RoutingRealizationKey(
            device=torch.device("cuda", self.key.device_index),
            num_tokens=self.key.num_tokens,
            distribution=distribution.name,
            sample_index=sample_index,
            local_expert_offset=self.key.local_expert_offset,
            num_local_experts=self.key.num_local_experts,
            top_k=self.key.top_k,
            routing_rule_fingerprint=(
                f"mode={self.key.routing_input_mode};method={self.key.routing_method_type}"
            ),
            routed_scaling_factor=self.key.routed_scaling_factor,
        )

    def _prepare_resources(
        self,
        inputs: list[Any],
        runner_kwargs: Mapping[str, Any],
        bindings: Sequence[torch.Tensor],
    ) -> None:
        """Prepare exact-generation resources with retryable live-lease conflicts."""
        # Only a multi-body SWITCH plan needs selector, fused routing metadata, and child-body
        # workspaces; fallback and singleton remain ordinary capture paths.
        plan = self.dispatcher.plan
        runtime = self._runtime
        if plan is None or runtime is None or len(plan.bodies) < 2:
            return
        signature = tensor_binding_signature(bindings)
        failure_key = (plan.generation, signature)
        # A failure is sticky only for its exact plan generation and binding signature; a new
        # generation or address set receives one fresh preparation attempt.
        if failure_key in self._preparation_failures:
            return
        # Strip orchestration-owned keywords before delegating dtype-specific allocation and
        # workspace binding to the production runtime.
        try:
            runtime_kwargs = dict(runner_kwargs)
            for consumed_name in (
                "routing_input_mode",
                "num_experts",
                "local_expert_offset",
            ):
                runtime_kwargs.pop(consumed_name, None)
            metadata_weights = (
                None
                if self.key.routing_input_mode == RoutingInputMode.PackedPrecomputed
                else inputs[self.key.routing_weight_index]
            )
            self.dispatcher.prepare(
                bindings,
                resource_factory=lambda published: runtime.prepare(
                    published,
                    inputs,
                    inputs[self.key.routing_id_index],
                    num_experts=self.key.num_experts,
                    top_k=self.key.top_k,
                    local_expert_offset=self.key.local_expert_offset,
                    num_local_experts=self.key.num_local_experts,
                    routing_input_mode=self.key.routing_input_mode,
                    topk_weights=metadata_weights,
                    **runtime_kwargs,
                ),
            )
            self._transient_preparation_failure = None
        except DAResourceLeaseConflict as error:
            # Capacity occupied exclusively by live graphs is transient. Do not poison this
            # binding: the next ordinary warmup call retries after graph teardown releases pins.
            self._transient_preparation_failure = str(error)
            logger.warning(
                f"TRTLLM DA resources are leased; capture will temporarily use NoDA: {error}"
            )
        except Exception as error:  # noqa: BLE001
            self._preparation_failures[failure_key] = str(error)
            logger.warning(
                f"TRTLLM DA preparation failed; capture will use NoDA: {error}"
            )

    def dispatch(
        self,
        bindings: Sequence[torch.Tensor],
        *,
        run_fallback: Callable[[], _ResultT],
        run_body: Callable[[Any], _ResultT],
        capture_switch: Callable[[DAPlan, Any, int, int], Any],
    ) -> _ResultT:
        """Use the host-selected eager body or delegate CUDA Graph capture policy."""
        if (
            not torch.cuda.is_current_stream_capturing()
            and self._eager_body is not None
        ):
            return run_body(self._eager_body)
        return self.dispatcher.dispatch(
            bindings,
            run_fallback=run_fallback,
            run_body=run_body,
            capture_switch=capture_switch,
        )

    def record_topology(self, topology: Any) -> None:
        """Retain cumulative SWITCH count and latest outer-graph topology for one capture."""
        # Native injection inspects one SWITCH at a time. Consecutive injections sharing the
        # outer capture ID contribute independently to that graph's conditional-node total.
        if (
            self._last_topology is not None
            and self._last_topology.capture_id == topology.capture_id
        ):
            topology = replace(
                topology,
                conditional_node_count=(
                    self._last_topology.conditional_node_count
                    + topology.conditional_node_count
                ),
                is_workspace_lane_serialized=(
                    self._last_topology.is_workspace_lane_serialized
                    and topology.is_workspace_lane_serialized
                ),
                workspace_lane_invocation_count=(
                    self._last_topology.workspace_lane_invocation_count
                    + topology.workspace_lane_invocation_count
                ),
            )
        self._last_topology = topology

    def cache_record(self) -> dict[str, Any] | None:
        """Serialize the current plan into the shared AutoTuner namespace."""
        # Untuned domains publish nothing; guard fallback publishes policy plus the independent
        # graph-free selection without fabricating selector state.
        plan = self.dispatcher.plan
        if not self._tuned:
            return None
        eager_record = self._serialize_eager_body()
        if plan is None:
            return {
                "schema": 1,
                "policy": DAPlanMode.DA_FALLBACK.value,
                "fallback_reason": self._policy_fallback_reason,
                **eager_record,
            }
        # Serialize only populated exemplar rows and deduplicated bodies; fixed-capacity device
        # padding is reconstructed during restore.
        return {
            "schema": 1,
            "policy": self._published_policy,
            **eager_record,
            "num_selector_exemplars": plan.num_selector_exemplars,
            "exemplar_spectra": plan.exemplar_spectra[: plan.num_selector_exemplars]
            .cpu()
            .tolist(),
            "exemplar_body_indices": plan.exemplar_body_indices[
                : plan.num_selector_exemplars
            ]
            .cpu()
            .tolist(),
            "bodies": [
                {"tile_n": body.tile_n, "tactic": body.tactic} for body in plan.bodies
            ],
        }

    def restore_cache_record(self, record: Mapping[str, Any]) -> None:
        """Validate, stage, and atomically publish one trusted-local JSON record."""
        # Stage schema, eager dispatch, and fallback policy under the state lock before allocating
        # selector tensors.
        with self._lock:
            if self._tuned:
                return
            if record.get("schema") != 1:
                raise ValueError("Unsupported current DA tuning-cache schema")
            policy = record.get("policy")
            self._restore_eager_body(record)
            if policy == DAPlanMode.DA_FALLBACK.value:
                fallback_reason = record.get("fallback_reason")
                if fallback_reason is not None and not isinstance(fallback_reason, str):
                    raise ValueError("Cached DA fallback reason must be a string")
                self._policy_fallback_reason = fallback_reason
                self._published_policy = policy
                self._tuned = True
                return
            # Decode variable-length payloads into temporary values before dispatcher publication.
            raw_spectra = record.get("exemplar_spectra")
            raw_body_indices = record.get("exemplar_body_indices")
            raw_bodies = record.get("bodies")
            if not isinstance(raw_spectra, list):
                raise ValueError("Cached DA exemplar spectra must be a list")
            if not isinstance(raw_body_indices, list):
                raise ValueError("Cached DA exemplar body indices must be a list")
            if not isinstance(raw_bodies, list):
                raise ValueError("Cached DA bodies must be a list")
            spectra = torch.tensor(
                raw_spectra,
                dtype=torch.float32,
                device=torch.device("cuda", self.key.device_index),
            )
            body_indices = tuple(int(index) for index in raw_body_indices)
            bodies = tuple(
                DABody(tile_n=int(body["tile_n"]), tactic=int(body["tactic"]))
                for body in raw_bodies
            )
            expected_policy = (
                DAPlanMode.DA_SINGLE_BODY.value
                if len(bodies) == 1
                else DAPlanMode.DA_SWITCH.value
            )
            if policy != expected_policy:
                raise ValueError(
                    "Cached DA policy does not match its deduplicated body count"
                )
            # Publication is the only plan mutation point; a trailing count mismatch explicitly
            # clears the staged plan rather than accepting a partial cache record.
            plan = self.dispatcher.publish_cached_plan(
                spectra,
                body_indices,
                bodies,
            )
            if plan.num_selector_exemplars != int(
                record.get("num_selector_exemplars", -1)
            ):
                self.dispatcher.clear_plan()
                raise ValueError("Cached DA exemplar count is inconsistent")
            self._published_policy = policy
            self._tuned = True

    def _serialize_eager_body(self) -> dict[str, Any]:
        """Serialize the optional host-selected eager body and its provenance."""
        body = self._eager_body
        return {
            "eager_distribution": self._eager_distribution,
            "eager_body": (
                None if body is None else {"tile_n": body.tile_n, "tactic": body.tactic}
            ),
        }

    def _restore_eager_body(self, record: Mapping[str, Any]) -> None:
        """Restore one cache-selected host body or reject an incomplete record."""
        if "eager_distribution" not in record or "eager_body" not in record:
            raise ValueError("Cached DA record lacks eager-dispatch selection")
        distribution = record["eager_distribution"]
        raw_body = record["eager_body"]
        if distribution is None and raw_body is None:
            self._eager_distribution = None
            self._eager_body = None
            return
        if distribution not in ("ddist:1.1", "uniform") or not isinstance(
            raw_body, Mapping
        ):
            raise ValueError("Cached DA eager-dispatch selection is inconsistent")
        self._eager_distribution = str(distribution)
        self._eager_body = DABody(
            tile_n=int(raw_body["tile_n"]), tactic=int(raw_body["tactic"])
        )

    def acquire_graph_lease(self, graph: torch.cuda.CUDAGraph) -> Any:
        """Commit this state's completed outer capture to a graph-owned lease."""
        return self.dispatcher.acquire_graph_lease(graph)

    def diagnostics(self) -> dict[str, Any]:
        """Return synchronized benchmark diagnostics for this operation domain."""
        # Snapshot host policy and prepared resource state into JSON-compatible scalar values.
        plan = self.dispatcher.plan
        resources = self.dispatcher.resources
        preparation_failure = (
            self._transient_preparation_failure
            or self._policy_fallback_reason
            or (
                None
                if not self._preparation_failures
                else next(reversed(self._preparation_failures.values()))
            )
        )
        selected_body = None
        if resources is not None and hasattr(resources, "selected_body"):
            selected_body = int(resources.selected_body.item())
        # Preserve the inspected parallel-root proof alongside policy and selected-body data.
        topology = self._last_topology
        return {
            "operation_key": self.key.cache_key(),
            "tuned": self._tuned,
            "policy": self._published_policy,
            "eager_distribution": self._eager_distribution,
            "eager_body": (
                None
                if self._eager_body is None
                else {
                    "tile_n": self._eager_body.tile_n,
                    "tactic": self._eager_body.tactic,
                }
            ),
            "num_selector_exemplars": (
                0 if plan is None else plan.num_selector_exemplars
            ),
            "bodies": (
                []
                if plan is None
                else [
                    {"tile_n": body.tile_n, "tactic": body.tactic}
                    for body in plan.bodies
                ]
            ),
            "selected_body": selected_body,
            "binding_record_count": self.dispatcher.prepared_binding_count,
            "prepared_workspace_lane_count": (
                self.dispatcher.prepared_workspace_lane_count
            ),
            "leased_workspace_lane_count": (
                self.dispatcher.leased_workspace_lane_count
            ),
            "prepared_body_workspace_count": sum(
                hasattr(resource, "body_workspace")
                for resource in self.dispatcher.prepared_resources
            ),
            "capture_stream_count": len(
                {
                    resource.body_workspace.capture_stream.handle
                    for resource in self.dispatcher.prepared_resources
                    if hasattr(resource, "body_workspace")
                }
            ),
            "capture_fallback_reason": preparation_failure,
            "topology": (
                None
                if topology is None
                else {
                    "outer_node_count": topology.outer_node_count,
                    "outer_edge_count": topology.outer_edge_count,
                    "conditional_node_count": topology.conditional_node_count,
                    "body_count": topology.body_count,
                    "body_node_counts": topology.body_node_counts,
                    "is_selector_preamble_parallelizable": (
                        topology.is_selector_preamble_parallelizable
                    ),
                    "is_workspace_lane_serialized": (
                        topology.is_workspace_lane_serialized
                    ),
                    "workspace_lane_invocation_count": (
                        topology.workspace_lane_invocation_count
                    ),
                }
            ),
        }


class TrtllmDaRegistry:
    """Own process-local automatic DA operation states and cache publication."""

    def __init__(self) -> None:
        """Create an empty thread-safe state registry."""
        # State indexed by immutable operation domain.
        self._states: dict[TrtllmDaOperationKey, TrtllmDaOperationState] = {}
        # Registry lock protecting lookup and cache snapshots.
        self._lock = threading.RLock()

    def get_or_create(self, key: TrtllmDaOperationKey) -> TrtllmDaOperationState:
        """Return one stable state object for an operation domain."""
        with self._lock:
            return self._states.setdefault(key, TrtllmDaOperationState(key))

    def find(self, key: TrtllmDaOperationKey) -> TrtllmDaOperationState | None:
        """Return a previously tuned operation state without creating one."""
        with self._lock:
            return self._states.get(key)

    def find_or_restore(
        self, key: TrtllmDaOperationKey, tuner: AutoTuner
    ) -> TrtllmDaOperationState | None:
        """Return process state or transactionally restore its cache record."""
        state = self.find(key)
        if state is not None:
            return state
        record = tuner.get_namespaced_records("trtllm_moe_da").get(key.cache_key())
        if record is None:
            return None
        candidate = TrtllmDaOperationState(key)
        try:
            candidate.restore_cache_record(record)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            logger.warning(
                f"Ignoring invalid TRTLLM DA tuning-cache record; retuning is required: {error}"
            )
            return None
        with self._lock:
            return self._states.setdefault(key, candidate)

    def publish_cache(self, tuner: AutoTuner) -> None:
        """Merge tuned states into the shared fused-MoE cache namespace."""
        records = tuner.get_namespaced_records("trtllm_moe_da")
        with self._lock:
            records.update(
                {
                    key.cache_key(): record
                    for key, state in self._states.items()
                    if (record := state.cache_record()) is not None
                }
            )
        tuner.publish_namespaced_records("trtllm_moe_da", records)

    def acquire_graph_leases(self, graph: torch.cuda.CUDAGraph) -> tuple[Any, ...]:
        """Lease every DA state injected into the just-completed outer graph."""
        with self._lock:
            states = tuple(self._states.values())
        leases = []
        for state in states:
            if state.dispatcher.pending_capture_generation is not None:
                leases.append(state.acquire_graph_lease(graph))
        return tuple(leases)

    def release_idle_resources(self) -> int:
        """Release all prepared resource bindings not pinned by live CUDA Graphs."""
        with self._lock:
            states = tuple(self._states.values())
        return sum(state.dispatcher.release_idle_resources() for state in states)

    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic diagnostics for every automatic DA domain."""
        with self._lock:
            states = tuple(
                state
                for _, state in sorted(
                    self._states.items(), key=lambda item: item[0].cache_key()
                )
            )
        return tuple(state.diagnostics() for state in states)


TRTLLM_DA_REGISTRY = TrtllmDaRegistry()


def trtllm_moe_acquire_da_graph_leases(
    graph: torch.cuda.CUDAGraph,
) -> tuple[Any, ...]:
    """Commit every automatic TRTLLM DA injection in one completed outer graph."""
    return TRTLLM_DA_REGISTRY.acquire_graph_leases(graph)


def trtllm_moe_da_diagnostics() -> tuple[dict[str, Any], ...]:
    """Return synchronized automatic DA plan and topology benchmark diagnostics."""
    return TRTLLM_DA_REGISTRY.diagnostics()


def trtllm_moe_release_da_resources() -> int:
    """Release every idle TRTLLM DA binding while preserving live-graph resources."""
    return TRTLLM_DA_REGISTRY.release_idle_resources()
