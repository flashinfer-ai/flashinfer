"""MoEEpSplitLayer — dispatch/combine EP with a pluggable inner kernel."""

from __future__ import annotations

import contextlib
import weakref
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobFaultTolerance,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobTopologyCapacity,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    _index_knobs,
)
from ..config import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    DispatchOutput,
    FleetParams,
    HandleParams,
)
from ..core.comm.fleet import Fleet, create_fleet
from ..core.kernel.base import SplitKernelBackend, SplitKernelContext
from ..core.kernel.registry import create_split_kernel
from ..core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
    split_comm_runtime_requirements,
)
from ..core.validation.common import (
    MoEEpConfigError,
    ensure_bootstrap_dist_validated,
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_fleet_weights,
    validate_split_forward_inputs,
)
from ..weights import MoEWeightPack
from .config import SplitConfig
from ..backends.split.kernel.identity.config import IdentityConfig

if TYPE_CHECKING:
    from ..tensors import MoEEpTensors
    from ..core.comm.handle import Handle


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class MoEEpSplitGraphState:
    """Persistent EP state that makes :meth:`MoEEpSplitLayer.forward` capturable.

    Create with :meth:`MoEEpSplitLayer.create_graph_state` **outside** any
    capture; never construct directly.

    The default forward creates a ``Handle`` per call and destroys it in a
    ``finally``. A CUDA graph records the device pointers it sees at capture
    time, so that handle's buffers are freed the moment capture ends and every
    replay dereferences dead memory -- which is silent at capture and faults
    (or worse, reads garbage) at replay. This object holds one long-lived
    handle instead: the allocating half (``ncclEpInitHandle``, via
    ``create_handle``) runs here, outside the capture, and the per-step half
    (``Handle.update``) is recorded inside it.

    It also pins the tensors the graph binds. ``topk_weights`` is bound into
    the handle at creation and ``out`` is the address combine writes, so both
    must be stable buffers whose *contents* the caller overwrites between
    replays -- the standard CUDA-graph static-input discipline.
    ``forward`` re-checks the addresses on every call, because a caller that
    silently passes a fresh tensor would otherwise get a graph that keeps
    serving the old one.
    """

    __slots__ = (
        "_layer_ref",
        "_handle",
        "_hidden_states",
        "_topk_ids",
        "_topk_weights",
        "_out",
        "_destroyed",
    )

    def __init__(
        self,
        layer: "MoEEpSplitLayer",
        handle: "Handle",
        t: "MoEEpTensors",
        out: torch.Tensor,
    ) -> None:
        # weakref, matching MoEEpMegaWorkspace: the layer holds the state,
        # so a strong back-reference would make the pair uncollectable.
        self._layer_ref = weakref.ref(layer)
        self._handle = handle
        self._hidden_states = t.hidden_states
        self._topk_ids = t.topk_ids
        self._topk_weights = t.topk_weights
        self._out = out
        self._destroyed = False

    @property
    def out(self) -> torch.Tensor:
        """The buffer combine writes on every replay."""
        return self._out

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def _check(self, t: "MoEEpTensors") -> None:
        if self._destroyed:
            raise RuntimeError(
                "MoEEpSplitGraphState has been destroyed; create a new one "
                "with MoEEpSplitLayer.create_graph_state()."
            )
        for name, bound, got in (
            ("hidden_states", self._hidden_states, t.hidden_states),
            ("topk_ids", self._topk_ids, t.topk_ids),
            ("topk_weights", self._topk_weights, t.topk_weights),
        ):
            if got.data_ptr() != bound.data_ptr() or got.shape != bound.shape:
                raise ValueError(
                    f"MoEEpSplitGraphState: {name} must be the same buffer the "
                    f"state was created with (a captured graph binds addresses "
                    f"once). Expected data_ptr=0x{bound.data_ptr():x} "
                    f"shape={tuple(bound.shape)}, got "
                    f"0x{got.data_ptr():x} shape={tuple(got.shape)}. Copy the "
                    "new values into the registered tensor in place instead of "
                    "rebinding it."
                )

    def destroy(self) -> None:
        """Release the persistent handle. Idempotent.

        Only safe once every graph that captured this state has been retired
        and the device synchronized; the graphs hold pointers into the
        handle's buffers.
        """
        if self._destroyed:
            return
        self._destroyed = True
        with contextlib.suppress(Exception):
            self._handle.destroy()
        layer = self._layer_ref()
        if layer is not None and layer._graph_state is self:
            layer._graph_state = None


class MoEEpSplitLayer(nn.Module):
    """Expert-Parallel layer: dispatch → inner kernel → combine."""

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, SplitConfig, object] = "nccl_ep",
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._weights: Optional[MoEWeightPack] = weights
        self._fleet_knobs = list(fleet_knobs)
        if isinstance(backend, SplitConfig):
            self._comm_backend = backend.comm
            self._kernel_config = backend.kernel
        else:
            self._comm_backend = backend
            self._kernel_config = IdentityConfig()

        self._kernel: SplitKernelBackend = create_split_kernel(self._kernel_config)

        ensure_moe_ep_cuda_device(bootstrap)

        self._runtime = None
        if bootstrap.auto_bootstrap:
            self._runtime = bootstrap_moe_ep_runtime(
                bootstrap,
                split_comm_runtime_requirements(self._comm_backend_name()),
            )

        self._validate_at_init()
        self._kernel.validate_init(bootstrap, fleet_params)

        if type(self._kernel).requires_weights():
            self._kernel.preprocess_weights(self._weights, fleet_params)
        # Source pack is only needed for init-time validation and kernel
        # preprocessing; the kernel retains what it needs. Release it so the
        # layer does not pin a per-layer weight copy for the model lifetime
        # (same OOM pattern as the mega path — see MoEEpMegaLayer docstring).
        self._weights = None

        self._fleet: Fleet | None = None
        # Set by create_graph_state(); the layer keeps at most one live state
        # so destroy() can tear it down with the fleet it borrows buffers from.
        self._graph_state: MoEEpSplitGraphState | None = None

        # Opt-in per-stage profiling. When True, forward() records CUDA events
        # around dispatch / compute / combine and stores elapsed GPU time (ms)
        # in ``last_timings_ms`` after a device sync. Off by default (zero
        # overhead on the hot path). Used by benchmarks/bench_moe_ep.py.
        self.enable_timing = False
        self.last_timings_ms: dict[str, float] = {}

    def _comm_backend_name(self) -> str:
        name = getattr(self._comm_backend, "backend_name", self._comm_backend)
        if not isinstance(name, str):
            raise TypeError(
                f"comm backend must be a string or have a .backend_name str attr; "
                f"got {self._comm_backend!r}"
            )
        return name

    def _validate_at_init(self) -> None:
        backend_name = self._comm_backend_name()
        validate_bootstrap_world_size(self._bootstrap)
        validate_fleet_weights(
            self._weights, self._fleet_params, self._bootstrap.world_size
        )
        # nixl_ep rendezvous-store validation is deferred to fleet creation
        # (first forward): layers are routinely constructed before
        # torch.distributed is initialized, and NixlEpFleet._resolve_store
        # raises the same actionable error when neither tcp_store nor an
        # initialized default group is available.
        if backend_name not in ("nccl_ep", "nixl_ep"):
            return
        validate_arch_for_backend(backend_name)
        fleet_knobs = _index_knobs(self._fleet_knobs)
        cap_knob = fleet_knobs.get(FleetAlgoKnobTopologyCapacity)
        topology_capacity = (
            int(cap_knob.n) if cap_knob is not None else None  # type: ignore[attr-defined]
        )
        validate_fleet_params(
            self._fleet_params,
            backend=backend_name,
            world_size=self._bootstrap.world_size,
            quant=fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
            topology_capacity=topology_capacity,
            fault_tolerance=fleet_knobs.get(FleetAlgoKnobFaultTolerance),  # type: ignore[arg-type]
        )

    def _ensure_fleet(self) -> Fleet:
        if self._fleet is None:
            self._fleet = create_fleet(
                self._bootstrap,
                self._fleet_params,
                self._fleet_knobs,
                backend=self._comm_backend,
            )
        return self._fleet

    def create_graph_state(
        self,
        t: "MoEEpTensors",
        *,
        out: Optional[torch.Tensor] = None,
    ) -> MoEEpSplitGraphState:
        """Build the persistent state a CUDA-graph capture of ``forward`` needs.

        Call once per (shape, buffer set) on **every** EP rank, outside any
        capture, then pass the result to ``forward``::

            state = layer.create_graph_state(t)
            layer.forward(t, graph_state=state)      # warmup, still eager
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = layer.forward(t, graph_state=state)
            ...
            t.hidden_states.copy_(new_x)             # in place, same buffers
            t.topk_ids.copy_(new_ids)
            g.replay()                               # y now holds the result

        ``t``'s tensors become the graph's bound buffers, so update them in
        place between replays rather than rebinding. The warmup forward is not
        optional in practice: it is what compiles/autotunes the inner kernel
        and establishes the transport's steady state, neither of which can
        happen during capture.

        The routing SHAPE is fixed here -- ``top_k`` and the token count are
        baked into the handle (and into the graph). A different batch size
        needs its own state and its own graph, the usual multi-size-graph
        pattern.
        """
        if _is_capturing():
            raise MoEEpConfigError(
                "MoEEpSplitLayer.create_graph_state() allocates transport "
                "buffers and cannot run during CUDA graph capture; call it "
                "(and one warmup forward) on all EP ranks before capturing."
            )
        ensure_bootstrap_dist_validated(self._bootstrap)
        validate_split_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            self._fleet_params,
        )
        if self._graph_state is not None and not self._graph_state.destroyed:
            raise MoEEpConfigError(
                "this MoEEpSplitLayer already has a live graph state; destroy "
                "it before creating another (one persistent handle per layer)."
            )
        if out is None:
            out = torch.empty_like(t.hidden_states)
        elif out.shape != t.hidden_states.shape or out.dtype != t.hidden_states.dtype:
            raise ValueError(
                f"out must match hidden_states: expected shape "
                f"{tuple(t.hidden_states.shape)} dtype {t.hidden_states.dtype}, "
                f"got {tuple(out.shape)} {out.dtype}."
            )

        fleet = self._ensure_fleet()
        handle = fleet.create_handle(
            HandleParams(topk_ids=t.topk_ids),
            algo_knobs=[
                # Bound once, to the stream creation runs on. Under capture the
                # handle issues transport work on the capture stream instead --
                # see NcclEpHandle._op_stream.
                HandleAlgoKnobUserStream(
                    stream=torch.cuda.current_stream().cuda_stream
                ),
                # Bound to the STATIC weights buffer: combine reads it from
                # this address on every replay.
                HandleAlgoKnobTopKWeights(weights=t.topk_weights),
            ],
        )
        # One update outside the capture. Backends that order their captured
        # work after InitHandle require this (it cannot be established from
        # inside a capture), and the handle raises a clear error if the first
        # update it ever sees is the captured one.
        try:
            handle.update(HandleParams(topk_ids=t.topk_ids))
        except NotImplementedError as e:
            handle.destroy()
            raise MoEEpConfigError(
                f"comm backend {self._comm_backend_name()!r} does not implement "
                "Handle.update(), so its forward cannot be CUDA-graph captured: "
                "a handle created per forward leaves the replay pointing at "
                "freed memory."
            ) from e
        except Exception:
            handle.destroy()
            raise

        state = MoEEpSplitGraphState(self, handle, t, out)
        self._graph_state = state
        return state

    def _inner_compute(self, dispatch: DispatchOutput) -> torch.Tensor:
        ctx = SplitKernelContext(
            expert_tensors=dispatch.expert_tensors,
            num_tokens=dispatch.get_num_tokens(),
            fleet_params=self._fleet_params,
            recv_topk_idx=dispatch.recv_topk_idx,
            recv_topk_weights=dispatch.recv_topk_weights,
        )
        return self._kernel.compute(ctx)

    def _round_trip(self, handle, t: "MoEEpTensors", out: torch.Tensor) -> torch.Tensor:
        """dispatch -> inner kernel -> combine, on an already-bound handle."""
        if not self.enable_timing:
            dispatch = handle.dispatch(
                DispatchInputParams(
                    x=[self._kernel.pack_dispatch_payload(t.hidden_states)]
                )
            )
            expert_out = self._inner_compute(dispatch)
            combine = handle.combine(CombineInputParams(x=[expert_out], out=out))
            return combine.x

        ev = {
            k: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for k in ("dispatch", "compute", "combine")
        }
        ev["dispatch"][0].record()
        dispatch = handle.dispatch(
            DispatchInputParams(x=[self._kernel.pack_dispatch_payload(t.hidden_states)])
        )
        ev["dispatch"][1].record()
        ev["compute"][0].record()
        expert_out = self._inner_compute(dispatch)
        ev["compute"][1].record()
        ev["combine"][0].record()
        combine = handle.combine(CombineInputParams(x=[expert_out], out=out))
        ev["combine"][1].record()
        torch.cuda.synchronize()
        self.last_timings_ms = {
            k: start.elapsed_time(end) for k, (start, end) in ev.items()
        }
        return combine.x

    def forward(
        self,
        t: "MoEEpTensors",
        *,
        graph_state: Optional[MoEEpSplitGraphState] = None,
    ) -> torch.Tensor:
        """Run one EP round trip.

        With ``graph_state`` (from :meth:`create_graph_state`) the layer reuses
        that state's persistent handle and static output buffer instead of
        creating and destroying a handle per call, which is what makes the call
        safe to record into a CUDA graph. Without it, behaviour is unchanged.
        """
        ensure_bootstrap_dist_validated(self._bootstrap)
        validate_split_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            self._fleet_params,
        )

        if graph_state is not None:
            graph_state._check(t)
            if self.enable_timing and _is_capturing():
                raise MoEEpConfigError(
                    "enable_timing synchronizes the device to read its CUDA "
                    "events, which is illegal during graph capture. Turn it "
                    "off to capture, and time the replay instead."
                )
            handle = graph_state._handle
            # The per-step half: recompute routing from the (rewritten) ids
            # into the handle's existing buffers. Recorded inside the capture.
            handle.update(HandleParams(topk_ids=t.topk_ids))
            try:
                return self._round_trip(handle, t, graph_state._out)
            finally:
                # complete() only waits on staged work; the handle deliberately
                # survives, so no destroy() here.
                handle.complete()

        if _is_capturing():
            raise MoEEpConfigError(
                "MoEEpSplitLayer.forward() creates a Handle per call and "
                "destroys it, so capturing it would leave the replay pointing "
                "at freed memory. Pass graph_state=layer.create_graph_state(t) "
                "(built outside the capture) to capture this layer."
            )

        fleet = self._ensure_fleet()
        handle_knobs: list[AlgoKnob] = [
            HandleAlgoKnobUserStream(stream=torch.cuda.current_stream().cuda_stream),
            HandleAlgoKnobTopKWeights(weights=t.topk_weights),
        ]
        handle = fleet.create_handle(
            HandleParams(topk_ids=t.topk_ids),
            algo_knobs=handle_knobs,
        )
        try:
            return self._round_trip(handle, t, torch.empty_like(t.hidden_states))
        finally:
            handle.complete()
            handle.destroy()

    def destroy(self) -> None:
        # Before the fleet: the persistent handle holds buffers the fleet owns.
        if self._graph_state is not None:
            self._graph_state.destroy()
            self._graph_state = None
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
