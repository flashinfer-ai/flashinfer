"""Distribution-aware MoE plan, resource, and CUDA Graph lifecycle primitives."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import torch


_ResultT = TypeVar("_ResultT")

# Maximum local-expert domain supported by the DA selector implementation.
DA_MAX_EXPERTS = 512
# Immutable maximum number of distribution exemplar rows in one DA plan.
DA_MAX_EXEMPLARS = 8
# Immutable maximum number of unique conditional child bodies.
DA_MAX_BODIES = 8
# Maximum number of CUDA Graphs that may concurrently replay one operation domain.
DA_MAX_WORKSPACE_LANES = 8


class DAResourceLeaseConflict(RuntimeError):
    """Report a transient resource-capacity conflict caused only by live graph leases."""


class DAPlanMode(Enum):
    """Host admission outcome and corresponding CUDA Graph realization."""

    # Reject DA capture and execute the matched ordinary monolithic baseline.
    DA_FALLBACK = "da_fallback"
    # Capture one selected fixed body without selector or SWITCH machinery.
    DA_SINGLE_BODY = "da_single_body"
    # Capture a device-selected conditional graph containing multiple bodies.
    DA_SWITCH = "da_switch"


@dataclass(frozen=True)
class DABody:
    """One unique CUDA Graph body and its implementation-specific routing tile."""

    # Complete backend tactic identifier executed by this graph body.
    tactic: int
    # Routing tile-N used to prepare this body's input-dependent metadata.
    tile_n: int


@dataclass(frozen=True)
class DAPlan:
    """Immutable exemplar-to-body mapping published for CUDA Graph capture."""

    # Monotonic publication generation owning every tensor in this plan.
    generation: int
    # Number of populated exemplar rows within the fixed-capacity tensors.
    num_selector_exemplars: int
    # Device-resident normalized load spectra padded to exemplar capacity.
    exemplar_spectra: torch.Tensor
    # Device mapping from each populated exemplar row to a deduplicated body.
    exemplar_body_indices: torch.Tensor
    # Deduplicated complete body descriptions in conditional-switch order.
    bodies: tuple[DABody, ...]
    # Device-resident tactic identifier for each deduplicated body.
    body_tactics: torch.Tensor
    # Device-resident routing tile-N for each deduplicated body.
    body_tile_ns: torch.Tensor

    @property
    def mode(self) -> DAPlanMode:
        """Return the pruned graph realization required by this plan."""
        if not self.bodies:
            return DAPlanMode.DA_FALLBACK
        if len(self.bodies) == 1:
            return DAPlanMode.DA_SINGLE_BODY
        return DAPlanMode.DA_SWITCH


@dataclass(frozen=True)
class DAResources:
    """Graph-stable mock workspace owned by one concurrent replay lane."""

    # Published plan generation for which these resources were prepared.
    generation: int
    # Device scalar written by the selector with the chosen body index.
    selected_body: torch.Tensor
    # Device workspace populated by implementation-specific pre-body work.
    parallel_work: torch.Tensor
    # Maximum-sized post-FC1 activation shared by serial bindings and exclusive bodies.
    activation_workspace: torch.Tensor
    # Maximum-sized post-FC2 intermediate shared by serial bindings and exclusive bodies.
    intermediate_workspace: torch.Tensor


@dataclass
class DABindingRecord:
    """Retain one public pointer signature and its latest capture diagnostics."""

    # Stable public tensor-address signature used to admit this binding at capture.
    signature: tuple[int, ...]
    # Number of successful SWITCH injections recorded for this binding.
    capture_count: int = 0
    # CUDA capture identifier from this binding's latest successful injection.
    last_capture_id: int | None = None
    # Workspace lane identifier used by this binding's latest successful injection.
    last_workspace_lane_id: int | None = None


@dataclass
class DAWorkspaceLane:
    """Own one graph-stable workspace reusable by serial bindings in one graph."""

    # Immutable lane identifier used by diagnostics and graph leases.
    lane_id: int
    # Published plan generation for which this workspace was allocated.
    generation: int
    # Dtype-specific graph-stable workspace shared by serial bindings in this lane.
    workspace: Any
    # Outer capture identifier currently constructing a graph with this lane.
    pending_capture_id: int | None = None
    # Native conditional node that must precede the next serial invocation.
    previous_conditional_node_handle: int = 0
    # Successful SWITCH injections sharing this lane in the pending outer graph.
    pending_invocation_count: int = 0
    # Live graph lease currently pinning this lane, if any.
    leased: bool = False

    def reset_capture_state(self) -> None:
        """Return a released or abandoned lane to graph-independent idle state."""
        self.pending_capture_id = None
        self.previous_conditional_node_handle = 0
        self.pending_invocation_count = 0


@dataclass(frozen=True)
class DAGraphTopology:
    """Runtime-inspected topology facts for one DA graph injection."""

    # CUDA stream-capture identifier associated with this inspection.
    # Native FFI[0]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.capture_id.
    capture_id: int
    # Number of nodes in the outer captured graph after DA injection.
    # Native FFI[1]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.outer_node_count.
    outer_node_count: int
    # Number of dependency edges in the outer captured graph.
    # Native FFI[2]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.outer_edge_count.
    outer_edge_count: int
    # Number of conditional SWITCH nodes injected into the outer graph.
    # Native FFI[3]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.conditional_node_count.
    conditional_node_count: int
    # Number of deduplicated conditional body graphs.
    # Native FFI[4]; sync with tests/moe/csrc/mock_da_moe.cu:InspectLastGraph.
    body_count: int
    # Number of predecessor dependencies inherited by the selector node.
    # Native FFI[5]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.selector_dependency_count.
    selector_dependency_count: int
    # Number of predecessor dependencies inherited by the pre-body-work node.
    # Native FFI[6]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.parallel_work_dependency_count.
    parallel_work_dependency_count: int
    # True when runtime inspection proves the selector and preamble can execute in parallel.
    # Native FFI[7]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.is_selector_preamble_parallelizable.
    is_selector_preamble_parallelizable: bool
    # True when this invocation is ordered after the prior user of its workspace lane.
    # Native FFI[8]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.is_workspace_lane_serialized.
    is_workspace_lane_serialized: bool
    # Number of serial SWITCH invocations represented by this topology observation.
    # Native FFI[9]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.workspace_lane_invocation_count.
    workspace_lane_invocation_count: int
    # Node count for each conditional body graph in body-index order.
    # Native FFI[10:]; sync with include/flashinfer/fused_moe/da_moe.cuh:GraphTopology.body_node_counts.
    body_node_counts: tuple[int, ...]

    @classmethod
    def empty(cls) -> DAGraphTopology:
        """Return an inspection result representing a graph without DA injection."""
        return cls(0, 0, 0, 0, 0, 0, 0, False, False, 0, ())

    @classmethod
    def from_native(cls, values: Sequence[int]) -> DAGraphTopology:
        """Decode a compact native graph-inspection record."""
        # The empty record is the explicit no-DA topology returned for fallback and singleton
        # capture, so it must not be confused with a truncated SWITCH record.
        if not values:
            return cls.empty()
        if len(values) < 10:
            raise RuntimeError(f"Incomplete DA topology record: {values}")

        # Validate the variable child-graph tail before exposing named fields to diagnostics.
        body_count = values[4]
        body_node_counts = tuple(values[10:])
        if len(body_node_counts) != body_count:
            raise RuntimeError(
                "DA topology body count does not match inspected child graphs"
            )
        return cls(
            capture_id=values[0],
            outer_node_count=values[1],
            outer_edge_count=values[2],
            conditional_node_count=values[3],
            body_count=body_count,
            selector_dependency_count=values[5],
            parallel_work_dependency_count=values[6],
            is_selector_preamble_parallelizable=bool(values[7]),
            is_workspace_lane_serialized=bool(values[8]),
            workspace_lane_invocation_count=values[9],
            body_node_counts=body_node_counts,
        )


@dataclass(frozen=True)
class DACaptureOutcome:
    """Return one successful SWITCH result plus its inspected lane topology."""

    # Public result produced by the operation wrapper after graph injection.
    result: Any
    # Runtime-inspected topology for the newly injected SWITCH.
    topology: DAGraphTopology
    # Native conditional node that terminates this workspace-using invocation.
    conditional_node_handle: int


def tensor_content_fingerprint(*tensors: torch.Tensor) -> str:
    """Return a stable SHA-256 fingerprint of tensor metadata and contents."""
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_binding_signature(bindings: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Return the stable tensor addresses constraining one captured graph."""
    return tuple(tensor.data_ptr() for tensor in bindings)


class DAGraphLease:
    """Strong ownership lease for one graph and immutable DA plan generation."""

    def __init__(
        self,
        dispatcher: DAMoEDispatcher,
        graph: torch.cuda.CUDAGraph,
        generation: int,
        workspace_lane: DAWorkspaceLane,
    ) -> None:
        """Retain one graph generation and its exclusive concurrent replay lane."""
        # Dispatcher whose mutation remains blocked while this lease is live.
        self._dispatcher: DAMoEDispatcher | None = dispatcher
        # Strong reference to the captured graph retained until release.
        self._graph: torch.cuda.CUDAGraph | None = graph
        # Immutable DA plan generation captured by the retained graph.
        self.generation = generation
        # Strong reference pinning the lane workspace until graph teardown.
        self._workspace_lane: DAWorkspaceLane | None = workspace_lane
        # Idempotent release state used by dispatcher mutation checks.
        self.released = False

    @property
    def resource_count(self) -> int:
        """Return the single workspace lane pinned by this graph."""
        return 0 if self._workspace_lane is None else 1

    @property
    def workspace_lane_id(self) -> int:
        """Return the immutable lane identifier pinned by this graph."""
        if self._workspace_lane is None:
            raise RuntimeError("The DA graph lease has been released")
        return self._workspace_lane.lane_id

    @property
    def graph(self) -> torch.cuda.CUDAGraph:
        """Return the retained graph while the lease remains active."""
        if self._graph is None:
            raise RuntimeError("The DA graph lease has been released")
        return self._graph

    def replay(self) -> None:
        """Replay the retained CUDA Graph without host-side dispatch."""
        self.graph.replay()

    def release(self) -> None:
        """Release ownership after the caller has reset or synchronized graph work."""
        if self.released:
            return
        dispatcher = self._dispatcher
        self.released = True
        self._graph = None
        self._dispatcher = None
        if dispatcher is not None:
            dispatcher._release_graph_lease(self)
        self._workspace_lane = None


class DAMoEDispatcher:
    """Own DA plan pruning, capture decisions, resources, and graph lifetimes."""

    def __init__(
        self,
        num_experts: int,
        *,
        max_workspace_lanes: int = DA_MAX_WORKSPACE_LANES,
    ) -> None:
        """Create an unpublished dispatcher with immutable capacity settings."""
        if not 0 < num_experts <= DA_MAX_EXPERTS:
            raise ValueError(
                f"num_experts must be in [1, {DA_MAX_EXPERTS}], received {num_experts}"
            )
        if max_workspace_lanes <= 0:
            raise ValueError("max_workspace_lanes must be positive")
        # Immutable expert count defining every selector spectrum width.
        self._num_experts = num_experts
        # Immutable maximum number of exemplar rows in a published plan.
        self._max_exemplars = DA_MAX_EXEMPLARS
        # Immutable maximum number of unique bodies in a published plan.
        self._max_bodies = DA_MAX_BODIES
        # Immutable concurrent-graph capacity bounding graph-stable workspace lanes.
        self._max_workspace_lanes = max_workspace_lanes
        # Currently published immutable DA plan, or None before publication.
        self._plan: DAPlan | None = None
        # Lightweight pointer signatures and capture diagnostics for warmed public bindings.
        self._bindings: OrderedDict[tuple[int, ...], DABindingRecord] = OrderedDict()
        # Graph-stable workspaces indexed by immutable concurrent replay lane identifier.
        self._workspace_lanes: OrderedDict[int, DAWorkspaceLane] = OrderedDict()
        # Most recently prepared or captured lane used by diagnostics.
        self._latest_workspace_lane_id: int | None = None
        # Monotonic identifier assigned when a new workspace lane is allocated.
        self._next_workspace_lane_id = 0
        # Monotonic counter assigned to each successfully published plan.
        self._generation = 0
        # Generation injected into the outer capture that next requires successful graph commit.
        self._pending_capture_generation: int | None = None
        # Workspace lane injected into the outer capture awaiting graph commit.
        self._pending_workspace_lane_id: int | None = None
        # Strong collection of graph leases that block state mutation until explicit release.
        self._leases: set[DAGraphLease] = set()

    @property
    def num_experts(self) -> int:
        """Return the immutable expert capacity used by spectra and metadata."""
        return self._num_experts

    @property
    def max_exemplars(self) -> int:
        """Return the immutable exemplar capacity of every published plan."""
        return self._max_exemplars

    @property
    def max_bodies(self) -> int:
        """Return the immutable unique-body capacity of every published plan."""
        return self._max_bodies

    @property
    def max_workspace_lanes(self) -> int:
        """Return the concurrent replay capacity for this operation domain."""
        return self._max_workspace_lanes

    @property
    def plan(self) -> DAPlan | None:
        """Return the currently published immutable plan, if one exists."""
        return self._plan

    @property
    def resources(self) -> Any | None:
        """Return the most recently used lane workspace, if any."""
        if self._latest_workspace_lane_id is None:
            return None
        lane = self._workspace_lanes.get(self._latest_workspace_lane_id)
        return None if lane is None else lane.workspace

    @property
    def prepared_binding_count(self) -> int:
        """Return the number of lightweight warmed pointer signatures."""
        return len(self._bindings)

    @property
    def prepared_workspace_lane_count(self) -> int:
        """Return the number of allocated graph-stable concurrent replay lanes."""
        return len(self._workspace_lanes)

    @property
    def prepared_resources(self) -> tuple[Any, ...]:
        """Return a stable snapshot of retained lane workspaces."""
        return tuple(lane.workspace for lane in self._workspace_lanes.values())

    @property
    def leased_workspace_lane_count(self) -> int:
        """Return the number of lanes pinned by live graph leases."""
        return sum(lane.leased for lane in self._workspace_lanes.values())

    @property
    def pending_capture_generation(self) -> int | None:
        """Return the uncommitted SWITCH generation from the latest outer capture."""
        return self._pending_capture_generation

    def publish_plan(
        self,
        exemplar_expert_ids: Sequence[torch.Tensor],
        exemplar_bodies: Sequence[DABody],
    ) -> DAPlan:
        """Validate, prune, upload, and atomically publish an exemplar plan."""
        # Reject the complete host description before computing spectra or allocating device
        # storage; publication must remain pristine on every validation failure.
        self._ensure_no_live_leases()
        if len(exemplar_expert_ids) != len(exemplar_bodies):
            raise ValueError("Every exemplar must have exactly one selected body")
        if not exemplar_expert_ids:
            raise ValueError("A DA plan requires at least one exemplar")
        if len(exemplar_expert_ids) > self._max_exemplars:
            raise ValueError(
                f"DA plan capacity is {self._max_exemplars} exemplars, "
                f"received {len(exemplar_expert_ids)}"
            )
        if any(body.tile_n <= 0 for body in exemplar_bodies):
            raise ValueError("Every DA body must have a positive routing tile")
        if any(body.tactic < 0 for body in exemplar_bodies):
            raise ValueError("Every DA body must have a concrete backend tactic")

        # Deduplicate only exact complete bodies while preserving every distinct selector
        # exemplar and the first-seen conditional-body order.
        unique_bodies: list[DABody] = []
        exemplar_body_indices: list[int] = []
        for body in exemplar_bodies:
            if body not in unique_bodies:
                unique_bodies.append(body)
            exemplar_body_indices.append(unique_bodies.index(body))
        if len(unique_bodies) > self._max_bodies:
            raise ValueError(
                f"DA plan capacity is {self._max_bodies} unique bodies, "
                f"received {len(unique_bodies)}"
            )

        spectra = torch.stack([self._load_spectrum(ids) for ids in exemplar_expert_ids])
        return self.publish_cached_plan(
            spectra,
            exemplar_body_indices,
            unique_bodies,
        )

    def publish_cached_plan(
        self,
        exemplar_spectra: torch.Tensor,
        exemplar_body_indices: Sequence[int],
        bodies: Sequence[DABody],
    ) -> DAPlan:
        """Validate and atomically publish a current-schema cached DA plan."""
        # Treat cached tensors as untrusted staged data: validate shape, capacity, device,
        # normalization, uniqueness, and mapping consistency before mutating dispatcher state.
        self._ensure_no_live_leases()
        if exemplar_spectra.ndim != 2:
            raise ValueError("DA exemplar spectra must be a rank-two tensor")
        num_selector_exemplars, num_experts = exemplar_spectra.shape
        if not 0 < num_selector_exemplars <= self._max_exemplars:
            raise ValueError(
                f"DA plan requires [1, {self._max_exemplars}] exemplar rows"
            )
        if num_experts != self._num_experts:
            raise ValueError(
                f"DA spectrum width must be {self._num_experts}, received {num_experts}"
            )
        if exemplar_spectra.device.type != "cuda":
            raise ValueError("DA exemplar spectra must reside on CUDA")
        if exemplar_spectra.dtype != torch.float32:
            raise ValueError("DA exemplar spectra must use float32")
        if not torch.isfinite(exemplar_spectra).all().item():
            raise ValueError("DA exemplar spectra must be finite")
        row_norms = torch.linalg.vector_norm(exemplar_spectra, dim=1)
        if not torch.allclose(row_norms, torch.ones_like(row_norms)):
            raise ValueError("DA exemplar spectra must have unit L2 norm")
        serialized_spectra = [tuple(row.cpu().tolist()) for row in exemplar_spectra]
        if len(set(serialized_spectra)) != len(serialized_spectra):
            raise ValueError("DA exemplar spectra must be unique")
        if len(exemplar_body_indices) != num_selector_exemplars:
            raise ValueError("Every cached exemplar must map to exactly one body")
        if not 0 < len(bodies) <= self._max_bodies:
            raise ValueError(f"DA plan requires [1, {self._max_bodies}] unique bodies")
        if len(set(bodies)) != len(bodies):
            raise ValueError("Cached DA bodies must already be deduplicated")
        if any(body.tile_n <= 0 or body.tactic < 0 for body in bodies):
            raise ValueError("Cached DA bodies require concrete tactics and tiles")
        if any(index < 0 or index >= len(bodies) for index in exemplar_body_indices):
            raise ValueError("Cached DA exemplar body index is out of range")

        # Materialize immutable fixed-capacity selector and body tensors only after the entire
        # logical record has passed validation.
        device = exemplar_spectra.device
        padded_spectra = torch.zeros(
            self._max_exemplars,
            self._num_experts,
            dtype=torch.float32,
            device=device,
        )
        padded_spectra[:num_selector_exemplars].copy_(exemplar_spectra)
        exemplar_body_tensor = torch.zeros(
            self._max_exemplars, dtype=torch.int32, device=device
        )
        exemplar_body_tensor[:num_selector_exemplars] = torch.tensor(
            exemplar_body_indices, dtype=torch.int32, device=device
        )
        body_tactics = torch.tensor(
            [body.tactic for body in bodies], dtype=torch.int32, device=device
        )
        body_tile_ns = torch.tensor(
            [body.tile_n for body in bodies], dtype=torch.int32, device=device
        )

        # Commit the new generation last so a failure above leaves the previous plan and its
        # graph-owned resources intact.
        next_generation = self._generation + 1
        plan = DAPlan(
            generation=next_generation,
            num_selector_exemplars=num_selector_exemplars,
            exemplar_spectra=padded_spectra,
            exemplar_body_indices=exemplar_body_tensor,
            bodies=tuple(bodies),
            body_tactics=body_tactics,
            body_tile_ns=body_tile_ns,
        )
        self._generation = next_generation
        self._plan = plan
        self._bindings.clear()
        self._workspace_lanes.clear()
        self._latest_workspace_lane_id = None
        self._pending_capture_generation = None
        self._pending_workspace_lane_id = None
        return plan

    def clear_plan(self) -> None:
        """Publish pristine fallback state without allocating DA device resources."""
        self._ensure_no_live_leases()
        self._generation += 1
        self._plan = None
        self._bindings.clear()
        self._workspace_lanes.clear()
        self._latest_workspace_lane_id = None
        self._pending_capture_generation = None
        self._pending_workspace_lane_id = None

    def prepare(
        self,
        bindings: Sequence[torch.Tensor],
        resource_factory: Callable[[DAPlan], Any] | None = None,
    ) -> Any | None:
        """Register a binding and ensure one idle workspace lane during warmup."""
        # Fallback and singleton plans deliberately own no DA replay resources because their
        # captures contain an ordinary fixed-tactic operation.
        if self._plan is None or self._plan.mode is not DAPlanMode.DA_SWITCH:
            self.release_idle_resources()
            return None
        if not bindings:
            raise ValueError("DA resource preparation requires tensor bindings")
        device = bindings[0].device
        if device.type != "cuda" or any(
            binding.device != device for binding in bindings
        ):
            raise ValueError("DA capture bindings must share one CUDA device")
        # Retain only the public pointer identity. Every same-domain binding may use any idle lane
        # because its captured body nodes bind layer weights independently from workspace storage.
        signature = tensor_binding_signature(bindings)
        if signature not in self._bindings:
            self._bindings[signature] = DABindingRecord(signature)
        self._bindings.move_to_end(signature)
        lane = self._find_idle_workspace_lane()
        if lane is not None:
            self._latest_workspace_lane_id = lane.lane_id
            return lane.workspace

        # Allocate another lane only when every existing lane is pinned by a graph or pending
        # capture. Capacity therefore bounds concurrent replay, never layer or binding count.
        if len(self._workspace_lanes) >= self._max_workspace_lanes:
            raise DAResourceLeaseConflict(
                "DA workspace lane capacity is occupied by live CUDA Graph leases; "
                "retry after release"
            )
        if resource_factory is None:
            workspace = DAResources(
                generation=self._plan.generation,
                selected_body=torch.full((1,), -1, dtype=torch.int32, device=device),
                parallel_work=torch.zeros(
                    len(self._plan.bodies),
                    self._num_experts,
                    dtype=torch.int32,
                    device=device,
                ),
                activation_workspace=torch.empty(
                    bindings[0].shape,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                intermediate_workspace=torch.empty(
                    bindings[0].shape,
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
        else:
            # Production runtimes provide dtype-specific lane workspaces whose generation remains
            # a dispatcher invariant and whose pointers are independent of public bindings.
            workspace = resource_factory(self._plan)
            if workspace.generation != self._plan.generation:
                raise RuntimeError(
                    "Prepared DA workspace has the wrong plan generation"
                )
        lane = DAWorkspaceLane(
            lane_id=self._next_workspace_lane_id,
            generation=self._plan.generation,
            workspace=workspace,
        )
        self._next_workspace_lane_id += 1
        self._workspace_lanes[lane.lane_id] = lane
        self._latest_workspace_lane_id = lane.lane_id
        return workspace

    def release_idle_resources(self) -> int:
        """Release every idle lane while retaining lightweight binding records."""
        released = 0
        for lane_id, lane in tuple(self._workspace_lanes.items()):
            if lane.leased or lane.pending_capture_id is not None:
                continue
            del self._workspace_lanes[lane_id]
            released += 1
        if self._latest_workspace_lane_id not in self._workspace_lanes:
            self._latest_workspace_lane_id = next(reversed(self._workspace_lanes), None)
        return released

    def _find_idle_workspace_lane(self) -> DAWorkspaceLane | None:
        """Return the least-recent lane not owned by a live or pending graph."""
        for lane_id, lane in tuple(self._workspace_lanes.items()):
            if not lane.leased and lane.pending_capture_id is None:
                self._workspace_lanes.move_to_end(lane_id)
                return lane
        return None

    def _pending_workspace_lane(self) -> DAWorkspaceLane | None:
        """Return the lane constructing the uncommitted outer graph, if any."""
        if self._pending_workspace_lane_id is None:
            return None
        return self._workspace_lanes.get(self._pending_workspace_lane_id)

    def dispatch(
        self,
        bindings: Sequence[torch.Tensor],
        *,
        run_fallback: Callable[[], _ResultT],
        run_body: Callable[[DABody], _ResultT],
        capture_switch: Callable[[DAPlan, Any, int, int], DACaptureOutcome | None],
    ) -> _ResultT:
        """Choose ordinary, fixed-body, or SWITCH capture without eager DA dispatch."""
        # Eager execution never injects or replays a hidden graph; it follows the caller's
        # ordinary fallback callback.
        if not torch.cuda.is_current_stream_capturing():
            return run_fallback()

        # Pruned host policy decides whether capture records the baseline, one fixed body, or
        # the multi-body device selector topology.
        plan = self._plan
        if plan is None or plan.mode is DAPlanMode.DA_FALLBACK:
            return run_fallback()
        if plan.mode is DAPlanMode.DA_SINGLE_BODY:
            return run_body(plan.bodies[0])

        # SWITCH capture admits only warmed public bindings. Workspace selection is independent of
        # pointer identity so serial same-domain layers share one lane in the outer graph.
        signature = tensor_binding_signature(bindings)
        binding = self._bindings.get(signature)
        if binding is None:
            return run_fallback()
        lane = self._pending_workspace_lane()
        if lane is None:
            lane = self._find_idle_workspace_lane()
            if lane is None or lane.generation != plan.generation:
                return run_fallback()
            self._pending_workspace_lane_id = lane.lane_id

        # Native preflight proves that a repeated lane use belongs to the same capture and is
        # ordered after the prior conditional/finalize. None means the current invocation remains
        # pristine and must use the ordinary body instead.
        outcome = capture_switch(
            plan,
            lane.workspace,
            lane.pending_capture_id or 0,
            lane.previous_conditional_node_handle,
        )
        if outcome is None:
            if lane.pending_invocation_count == 0:
                lane.reset_capture_state()
                self._pending_workspace_lane_id = None
            return run_fallback()
        topology = outcome.topology
        if not topology.is_workspace_lane_serialized:
            raise RuntimeError(
                "Native DA topology did not prove sequential workspace use"
            )

        # Commit capture identity and the terminating conditional node only after native topology
        # inspection succeeds, then annotate the lightweight binding record.
        if lane.pending_capture_id not in (None, topology.capture_id):
            raise RuntimeError("DA workspace lane crossed outer capture generations")
        lane.pending_capture_id = topology.capture_id
        lane.previous_conditional_node_handle = outcome.conditional_node_handle
        lane.pending_invocation_count += 1
        self._bindings.move_to_end(signature)
        binding.capture_count += 1
        binding.last_capture_id = topology.capture_id
        binding.last_workspace_lane_id = lane.lane_id
        self._latest_workspace_lane_id = lane.lane_id
        self._pending_capture_generation = plan.generation
        return outcome.result

    def acquire_graph_lease(self, graph: torch.cuda.CUDAGraph) -> DAGraphLease:
        """Commit a completed outer capture and retain its DA generation."""
        # A pending generation is created only by successful SWITCH injection and is consumed by
        # this single post-capture commit attempt.
        capture_generation = self._pending_capture_generation
        if capture_generation is None:
            raise RuntimeError("No successful DA capture is available to lease")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("A DA graph lease requires a completed outer capture")
        graph_exec_getter = getattr(graph, "raw_cuda_graph_exec", None)
        lane = self._pending_workspace_lane()
        self._pending_capture_generation = None
        self._pending_workspace_lane_id = None
        if graph_exec_getter is None:
            raise RuntimeError(
                "A DA graph lease requires an instantiated outer CUDA Graph"
            )
        # Require an instantiated executable graph rather than merely a Python graph object;
        # otherwise no replay can own the injected child graphs safely.
        try:
            graph_exec = graph_exec_getter()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "A DA graph lease requires an instantiated outer CUDA Graph"
            ) from error
        if not graph_exec:
            raise RuntimeError(
                "A DA graph lease requires an instantiated outer CUDA Graph"
            )
        # Recheck publication generation after capture completion before installing the strong
        # lease that blocks subsequent plan/resource mutation.
        if self._plan is None or capture_generation != self._plan.generation:
            raise RuntimeError(
                "Captured graph generation no longer matches the published plan"
            )
        if lane is None or lane.generation != capture_generation:
            raise RuntimeError("Captured DA workspace lane is no longer available")
        if lane.pending_capture_id is None or lane.pending_invocation_count == 0:
            raise RuntimeError(
                "Captured DA workspace lane has no successful injections"
            )
        lane.leased = True
        lease = DAGraphLease(self, graph, capture_generation, lane)
        self._leases.add(lease)
        return lease

    def _release_graph_lease(self, lease: DAGraphLease) -> None:
        """Unregister one explicitly released graph lease."""
        self._leases.discard(lease)
        # The graph stops pinning its exclusive lane, which remains idle and reusable until the
        # explicit idle-resource teardown path releases its storage.
        lane = lease._workspace_lane
        if lane is not None:
            lane.leased = False
            lane.reset_capture_state()
            self._workspace_lanes.move_to_end(lane.lane_id)

    def _ensure_no_live_leases(self) -> None:
        """Reject plan or resource mutation while a captured graph remains live."""
        if any(not lease.released for lease in self._leases):
            raise RuntimeError("Cannot mutate DA state while a graph lease is active")

    def _load_spectrum(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Return the normalized sorted expert-load spectrum used by cosine k=1."""
        if not expert_ids.is_cuda or expert_ids.dtype not in (
            torch.int16,
            torch.int32,
        ):
            raise TypeError(
                "DA exemplars must be CUDA int16 or int32 expert ID tensors"
            )
        if expert_ids.numel() == 0:
            raise ValueError("DA exemplars must contain at least one expert assignment")
        minimum = int(expert_ids.min().item())
        maximum = int(expert_ids.max().item())
        if minimum < 0 or maximum >= self._num_experts:
            raise ValueError(
                f"DA exemplar expert IDs must be in [0, {self._num_experts})"
            )
        counts = torch.bincount(
            expert_ids.flatten().to(torch.int64), minlength=self._num_experts
        ).to(torch.float32)
        spectrum = torch.sort(counts, descending=True).values
        return spectrum / torch.linalg.vector_norm(spectrum)
