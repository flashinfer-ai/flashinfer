"""Minimum viable reference design for distribution-aware MoE graph dispatch.

The mock composes DA orchestration around a dtype-agnostic ordinary MoE runner so the
production capture topology can be understood without the full TRTLLM kernel ABI.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flashinfer.autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.fused_moe.da_moe import (
    DA_MAX_EXEMPLARS,
    DA_MAX_EXPERTS,
    DACaptureOutcome,
    DABody,
    DAGraphLease,
    DAGraphTopology,
    DAMoEDispatcher,
    DAPlan,
    DAPlanMode,
    DAResourceLeaseConflict,
    DAResources,
    tensor_content_fingerprint,
)
from flashinfer.jit.core import gen_jit_spec


@functools.cache
def _get_mock_da_moe_module() -> Any:
    """Build and load the test-only native mock MoE module."""
    source = Path(__file__).parent / "csrc" / "mock_da_moe.cu"
    return gen_jit_spec("mock_da_moe", [source]).build_and_load()


@dataclass(frozen=True)
class MockMoEInputs:
    """Stable public tensors for one mock MoE layer invocation."""

    # Public hidden-state input read by every mock body implementation.
    hidden_states: torch.Tensor
    # Live token-to-expert assignments consumed by routing and selection.
    expert_ids: torch.Tensor
    # Live routing weights included in the value-aware tuning identity.
    expert_weights: torch.Tensor
    # Public canonical result written after body-specific finalization.
    output: torch.Tensor
    # Device-visible tactic observation written by the executed body.
    body_trace: torch.Tensor

    @classmethod
    def allocate(
        cls,
        num_tokens: int,
        hidden_size: int,
        top_k: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        expert_weight_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
        device: torch.device | str = "cuda",
    ) -> MockMoEInputs:
        """Allocate a graph-stable tensor set for one mock MoE invocation."""
        # Resolve dtype-specific public and routing contracts before allocating any stable tensor.
        resolved_weight_dtype = expert_weight_dtype or dtype
        resolved_output_dtype = output_dtype or dtype
        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.float32, device=device
        ).to(dtype)
        expert_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
        expert_weights = torch.randn(
            num_tokens, top_k, dtype=resolved_weight_dtype, device=device
        )
        # Allocate only layer-visible tensors; the runner owns reusable internal workspaces.
        return cls(
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            expert_weights=expert_weights,
            output=torch.empty(
                num_tokens,
                hidden_size,
                dtype=resolved_output_dtype,
                device=device,
            ),
            body_trace=torch.full((1,), -1, dtype=torch.int32, device=device),
        )

    def as_list(self) -> list[torch.Tensor]:
        """Return tensors in the positional order consumed by the autotuner runner."""
        return [
            self.hidden_states,
            self.expert_ids,
            self.expert_weights,
            self.output,
            self.body_trace,
        ]

    @classmethod
    def from_list(cls, tensors: list[torch.Tensor]) -> MockMoEInputs:
        """Construct a typed tensor bundle from an autotuner positional list."""
        if len(tensors) != 5:
            raise ValueError(
                f"MockMoEInputs requires 5 tensors, received {len(tensors)}"
            )
        return cls(*tensors)


class MockMoERunner(TunableRunner):
    """Dtype-agnostic tunable mock of the existing TRTLLM MoERunner."""

    # Maximum runtime expert extent supported by the production DA selector.
    MAX_NUM_EXPERTS = DA_MAX_EXPERTS
    # Complete fixed-body tactics exposed through the ordinary runner API.
    VALID_TACTICS = (0, 1, 2)
    # Hidden-state dtypes with concrete mock backend adapters.
    SUPPORTED_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

    def __init__(
        self,
        num_experts: int = MAX_NUM_EXPERTS,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Create one runner instance configured for a supported operation dtype."""
        if not 0 < num_experts <= self.MAX_NUM_EXPERTS:
            raise ValueError(
                f"num_experts must be in [1, {self.MAX_NUM_EXPERTS}], "
                f"received {num_experts}"
            )
        if dtype not in self.SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported mock MoE dtype: {dtype}")
        # Runtime expert extent bounded by the compiled native mock capacity.
        self._num_experts = num_experts
        # Operation dtype used to choose the concrete native body adapter.
        self._dtype = dtype
        # Shape-keyed activation/intermediate buffers reused by ordinary serial invocations.
        self._ordinary_workspaces: dict[
            tuple[torch.device, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def num_experts(self) -> int:
        """Return the runtime expert count represented by this runner."""
        return self._num_experts

    @property
    def dtype(self) -> torch.dtype:
        """Return the hidden-state dtype configured for this runner instance."""
        return self._dtype

    @property
    def operation_family(self) -> str:
        """Return the dtype-qualified autotuner namespace for this operation."""
        suffix = "bf16" if self._dtype == torch.bfloat16 else "fp8"
        return f"mock_moe.{suffix}"

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[int]:
        """Include immutable expert-domain configuration in persistent tuning identity."""
        del inputs
        return (self._num_experts,)

    def allocate_inputs(
        self,
        num_tokens: int,
        hidden_size: int,
        top_k: int,
        *,
        device: torch.device | str = "cuda",
    ) -> MockMoEInputs:
        """Allocate inputs using this runner instance's operation dtype."""
        return MockMoEInputs.allocate(
            num_tokens,
            hidden_size,
            top_k,
            dtype=self._dtype,
            expert_weight_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            device=device,
        )

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[int]:
        """Return every mock body tactic supported by the tensor contract."""
        del profile
        self.validate_inputs(MockMoEInputs.from_list(inputs))
        return list(self.VALID_TACTICS)

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run one complete mock body and finalize it into canonical layout."""
        # Decode and validate the ordinary runner ABI before preparation or execution.
        del kwargs
        bundle = MockMoEInputs.from_list(inputs)
        self.validate_inputs(bundle)
        if do_preparation:
            _get_mock_da_moe_module()
            return bundle.output
        # Resolve the ordinary default, then invoke the dtype-selected native body adapter.
        resolved_tactic = 0 if tactic == -1 else tactic
        if resolved_tactic not in self.VALID_TACTICS:
            raise ValueError(f"Unsupported mock MoE tactic: {resolved_tactic}")
        activation_workspace, intermediate_workspace = self._workspaces_for(bundle)
        _get_mock_da_moe_module().run_mock_moe(
            bundle.hidden_states,
            bundle.expert_ids,
            bundle.expert_weights,
            bundle.output,
            activation_workspace,
            intermediate_workspace,
            bundle.body_trace,
            self._num_experts,
            resolved_tactic,
        )
        return bundle.output

    def _workspaces_for(
        self, inputs: MockMoEInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordinary activation/intermediate buffers shared by serial bindings."""
        key = (inputs.output.device, tuple(inputs.output.shape))
        workspaces = self._ordinary_workspaces.get(key)
        if workspaces is None:
            workspaces = (
                torch.empty_like(inputs.output),
                torch.empty_like(inputs.output),
            )
            self._ordinary_workspaces[key] = workspaces
        return workspaces

    def validate_inputs(self, inputs: MockMoEInputs) -> None:
        """Validate tensors against this runner instance's operation contract."""
        # CUDA residency and contiguity are shared graph-stability requirements for every field.
        tensors = inputs.as_list()
        if not all(tensor.is_cuda for tensor in tensors):
            raise ValueError("Mock MoE tensors must all reside on CUDA")
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Mock MoE tensors must all be contiguous")
        # Dtype validation deliberately allows body input ABI to differ across runner families.
        if inputs.hidden_states.dtype != self._dtype:
            raise TypeError(
                f"MockMoERunner requires hidden_states dtype {self._dtype}, "
                f"received {inputs.hidden_states.dtype}"
            )
        if inputs.expert_weights.dtype != torch.bfloat16:
            raise TypeError(
                "MockMoERunner requires expert_weights dtype torch.bfloat16"
            )
        # Canonical output shape is stable even though body-internal layouts differ.
        if inputs.output.shape != inputs.hidden_states.shape:
            raise ValueError("output must match hidden_states shape")
        if inputs.output.dtype != torch.bfloat16:
            raise TypeError("MockMoERunner requires output dtype torch.bfloat16")
        if inputs.expert_ids.dtype != torch.int32:
            raise TypeError("expert_ids must use int32")
        if inputs.expert_ids.shape != inputs.expert_weights.shape:
            raise ValueError("expert_ids and expert_weights must have identical shapes")
        if inputs.body_trace.dtype != torch.int32 or inputs.body_trace.numel() != 1:
            raise ValueError("body_trace must be a single int32 value")


class MockDAMoERunner:
    """Distribution-aware orchestration composed around one MockMoERunner."""

    # Maximum runtime expert extent supported by the composed runner.
    MAX_NUM_EXPERTS = DA_MAX_EXPERTS
    # Immutable maximum number of selector exemplars in a published plan.
    MAX_EXEMPLARS = DA_MAX_EXEMPLARS
    # Mock routing tile associated with each complete executable tactic.
    _TILE_N_BY_TACTIC = {0: 16, 1: 32, 2: 64}

    def __init__(
        self,
        *,
        num_experts: int = MockMoERunner.MAX_NUM_EXPERTS,
        dtype: torch.dtype = torch.bfloat16,
        tuning_config: TuningConfig | None = None,
        max_workspace_lanes: int = 1,
    ) -> None:
        """Layer DA orchestration around one dtype-agnostic MockMoERunner."""
        # Standalone fixed-body runner reused for eager, tuning, and DA bodies.
        self._moe_runner = MockMoERunner(num_experts=num_experts, dtype=dtype)
        # DA state owner layered around the stable fixed-body runner.
        self._dispatcher = DAMoEDispatcher(
            num_experts=self.num_experts,
            max_workspace_lanes=max_workspace_lanes,
        )
        # Generic autotuner controls reused by normal and value-aware profiling.
        self._tuning_config = tuning_config or TuningConfig()
        # Latest ordinary eager tactic selected outside DA capture dispatch.
        self._last_normal_tactic: int | None = None
        # Latest retryable resource conflict observed by the public warmup path.
        self._last_preparation_conflict: str | None = None

    @property
    def moe_runner(self) -> MockMoERunner:
        """Return the standalone fixed-body runner wrapped by DA orchestration."""
        return self._moe_runner

    @property
    def valid_tactics(self) -> tuple[int, ...]:
        """Return tactics owned by the composed MoE runner."""
        return self._moe_runner.VALID_TACTICS

    @property
    def normal_operation_name(self) -> str:
        """Return this operation's ordinary shape-only tuning identity."""
        return f"{self._moe_runner.operation_family}.normal_dispatch"

    @property
    def num_experts(self) -> int:
        """Return the runtime expert count owned by the composed MoE runner."""
        return self._moe_runner.num_experts

    @property
    def dispatcher(self) -> DAMoEDispatcher:
        """Return the production dispatcher used by this test runner."""
        return self._dispatcher

    @property
    def plan(self) -> DAPlan | None:
        """Return the currently published immutable plan, if one exists."""
        return self._dispatcher.plan

    @property
    def last_normal_tactic(self) -> int | None:
        """Return the tactic selected by the latest ordinary autotuner lookup."""
        return self._last_normal_tactic

    @property
    def last_preparation_conflict(self) -> str | None:
        """Return the latest transient resource conflict, if retry has not succeeded."""
        return self._last_preparation_conflict

    def resource_counts(self) -> dict[str, int]:
        """Return user-visible prepared and live-graph resource ownership counts."""
        return {
            "capacity": self._dispatcher.max_workspace_lanes,
            "binding_records": self._dispatcher.prepared_binding_count,
            "workspace_lanes": self._dispatcher.prepared_workspace_lane_count,
            "leased_workspace_lanes": self._dispatcher.leased_workspace_lane_count,
        }

    def publish_plan(
        self,
        exemplar_expert_ids: list[torch.Tensor],
        exemplar_bodies: Sequence[int | DABody],
    ) -> DAPlan:
        """Publish complete body specs or convenience tactic identifiers."""
        bodies: list[DABody] = []
        for body in exemplar_bodies:
            if isinstance(body, DABody):
                bodies.append(body)
            elif body in self._TILE_N_BY_TACTIC:
                bodies.append(DABody(tactic=body, tile_n=self._TILE_N_BY_TACTIC[body]))
            else:
                raise ValueError("DA plan contains an unsupported mock MoE tactic")
        if any(body.tactic not in self.valid_tactics for body in bodies):
            raise ValueError("DA plan contains an unsupported mock MoE tactic")
        return self._dispatcher.publish_plan(exemplar_expert_ids, bodies)

    def _select_normal_tactic(self, inputs: list[torch.Tensor]) -> int:
        """Resolve the ordinary shape-only tactic without entering DA dispatch."""
        _, tactic = AutoTuner.get().choose_one(
            self.normal_operation_name,
            [self._moe_runner],
            self._tuning_config,
            inputs,
        )
        self._last_normal_tactic = tactic
        return tactic

    def _run_warmup_autotuning(
        self,
        inputs: MockMoEInputs,
        exemplar_expert_ids: list[torch.Tensor],
    ) -> DAPlan:
        """Profile internal warmup exemplars and publish the resulting DA plan."""
        # Change both expert IDs and BF16 weights for every value-aware exemplar while reusing all
        # graph-stable public output tensors.
        selected_tactics: list[int] = []
        tuner = AutoTuner.get()
        for exemplar_index, expert_ids in enumerate(exemplar_expert_ids):
            if expert_ids.shape != inputs.expert_ids.shape:
                raise ValueError("Every tuning exemplar must match expert_ids shape")
            changed_weights = inputs.expert_weights + float(exemplar_index + 1)
            profile_inputs = MockMoEInputs(
                hidden_states=inputs.hidden_states,
                expert_ids=expert_ids,
                expert_weights=changed_weights,
                output=inputs.output,
                body_trace=inputs.body_trace,
            )
            operation_name = self._value_aware_operation_name(
                expert_ids, changed_weights
            )
            _, tactic = tuner.choose_one(
                operation_name,
                [self._moe_runner],
                self._tuning_config,
                profile_inputs.as_list(),
            )
            selected_tactics.append(tactic)
        # Publish only after every exemplar has completed ordinary AutoTuner selection.
        return self.publish_plan(exemplar_expert_ids, selected_tactics)

    def _default_tuning_exemplars(self, inputs: MockMoEInputs) -> list[torch.Tensor]:
        """Create deterministic spread and concentrated warmup distributions."""
        active_experts = max(1, min(inputs.expert_ids.numel(), self.num_experts))
        assignments = torch.arange(
            inputs.expert_ids.numel(),
            dtype=torch.int32,
            device=inputs.expert_ids.device,
        )
        spread = assignments.remainder(active_experts).view_as(inputs.expert_ids)
        concentrated = torch.zeros_like(inputs.expert_ids)
        return [spread, concentrated]

    def prepare(self, inputs: MockMoEInputs) -> DAResources | None:
        """Bind production dispatcher resources during the mock warmup phase."""
        self._moe_runner.validate_inputs(inputs)
        resources = self._dispatcher.prepare(inputs.as_list())
        self._last_preparation_conflict = None
        return resources

    def release_idle_resources(self) -> int:
        """Release prepared mock resources that are not pinned by a live graph lease."""
        return self._dispatcher.release_idle_resources()

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run eager fallback or realize the production DA capture decision."""
        # Preserve the ordinary runner's preparation/default-tactic behavior before layering DA.
        bundle = MockMoEInputs.from_list(inputs)
        self._moe_runner.validate_inputs(bundle)
        if do_preparation:
            _get_mock_da_moe_module()
            return bundle.output

        normal_tactic = tactic
        if normal_tactic == -1:
            normal_tactic = self._select_normal_tactic(inputs)

        # Public warmup autotunes and publishes automatically; no DA-specific tune method exists.
        if (
            self.plan is None
            and AutoTuner.get().is_tuning_mode
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._run_warmup_autotuning(bundle, self._default_tuning_exemplars(bundle))

        # Prepare multi-body resources only during eager warmup, never during outer capture.
        if (
            not torch.cuda.is_current_stream_capturing()
            and self.plan is not None
            and self.plan.mode is DAPlanMode.DA_SWITCH
        ):
            try:
                self.prepare(bundle)
            except DAResourceLeaseConflict as error:
                # A live graph owns the bounded slot. Eager still uses the ordinary runner, and
                # the next warmup invocation retries after that lease has been released.
                self._last_preparation_conflict = str(error)
        # Start a fresh topology record only for the first invocation in a new outer capture.
        if (
            torch.cuda.is_current_stream_capturing()
            and self._dispatcher.pending_capture_generation is None
        ):
            _get_mock_da_moe_module().reset_graph_inspection()
        # Delegate the final fallback/singleton/SWITCH choice to the production dispatcher.
        return self._dispatcher.dispatch(
            inputs,
            run_fallback=lambda: self._moe_runner.forward(
                inputs, tactic=normal_tactic, **kwargs
            ),
            run_body=lambda body: self._moe_runner.forward(
                inputs, tactic=body.tactic, **kwargs
            ),
            capture_switch=lambda plan,
            resources,
            capture_id,
            previous_node: self._capture_switch(
                bundle, plan, resources, capture_id, previous_node
            ),
        )

    def acquire_graph_lease(self, graph: torch.cuda.CUDAGraph) -> DAGraphLease:
        """Acquire production graph ownership for the latest mock capture."""
        return self._dispatcher.acquire_graph_lease(graph)

    def inspect_last_graph(self) -> DAGraphTopology:
        """Inspect actual nodes and edges created by the native test injector."""
        values = [
            int(value) for value in _get_mock_da_moe_module().inspect_last_graph()
        ]
        return DAGraphTopology.from_native(values)

    def inspect_last_workspace_bindings(
        self,
    ) -> tuple[tuple[int, int, int], ...]:
        """Return captured weight, activation, and intermediate addresses by layer."""
        values = [
            int(value)
            for value in _get_mock_da_moe_module().inspect_last_workspace_bindings()
        ]
        if not values:
            return ()
        binding_count = values[0]
        pointer_values = values[1:]
        if len(pointer_values) != binding_count * 3:
            raise RuntimeError("Native mock workspace-binding inspection is malformed")
        return tuple(
            tuple(pointer_values[offset : offset + 3])
            for offset in range(0, len(pointer_values), 3)
        )

    def selected_body_tensor(self) -> torch.Tensor:
        """Return the graph-stable selector output of the prepared plan."""
        resources = self._require_resources()
        return resources.selected_body

    def parallel_work_tensor(self) -> torch.Tensor:
        """Return input-dependent routing metadata for every unique body."""
        resources = self._require_resources()
        return resources.parallel_work

    def _capture_switch(
        self,
        inputs: MockMoEInputs,
        plan: DAPlan,
        resources: DAResources,
        expected_capture_id: int,
        previous_conditional_node_handle: int,
    ) -> DACaptureOutcome | None:
        """Ask the test backend to inject concrete mock bodies into the DA SWITCH."""
        if expected_capture_id == 0:
            _get_mock_da_moe_module().reset_graph_inspection()
        # Preserve deduplicated body order when encoding tactics for native child-graph capture.
        body_tactic_ids = [body.tactic for body in plan.bodies]
        # Pass only stable tensors and immutable plan metadata across the test TVM-FFI boundary.
        conditional_node_handle = int(
            _get_mock_da_moe_module().capture_mock_da_moe(
                inputs.hidden_states,
                inputs.expert_ids,
                inputs.expert_weights,
                inputs.output,
                resources.activation_workspace,
                resources.intermediate_workspace,
                inputs.body_trace,
                plan.exemplar_spectra,
                plan.exemplar_body_indices,
                plan.body_tile_ns,
                resources.selected_body,
                resources.parallel_work,
                self.num_experts,
                plan.num_selector_exemplars,
                body_tactic_ids,
                expected_capture_id,
                previous_conditional_node_handle,
            )
        )
        if conditional_node_handle == 0:
            return None
        topology = self.inspect_last_graph()
        return DACaptureOutcome(inputs.output, topology, conditional_node_handle)

    def _require_resources(self) -> DAResources:
        """Return prepared resources or report an invalid test observation."""
        resources = self._dispatcher.resources
        if resources is None:
            raise RuntimeError("No multi-body DA resources have been prepared")
        return resources

    def _value_aware_operation_name(
        self, expert_ids: torch.Tensor, expert_weights: torch.Tensor
    ) -> str:
        """Build a tuning identity from distribution and weight tensor contents."""
        fingerprint = tensor_content_fingerprint(expert_ids, expert_weights)
        return f"{self._moe_runner.operation_family}.value_{fingerprint[:20]}"
