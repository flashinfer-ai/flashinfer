"""MoEEpSplitLayer — dispatch/combine EP with a pluggable inner kernel."""

from __future__ import annotations

import contextlib
import logging
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
    EpAlgorithm,
    EpLayout,
    FleetParams,
    HandleParams,
)
from ..core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    create_communication,
    is_communication_backend,
)
from ..core.comm.fleet import Fleet, create_fleet, is_fleet_backend
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

_logger = logging.getLogger(__name__)


def _is_capturing() -> bool:
    """Whether the current stream is recording into a CUDA graph."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _graph_tensor_signature(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


class MoEEpSplitGraphState:
    """Persistent EP state that makes :meth:`MoEEpSplitLayer.forward` capturable.

    Create with :meth:`MoEEpSplitLayer.create_graph_state` **outside** any
    capture; never construct directly.

    A CUDA graph records the device pointers it sees at capture time, so
    everything a captured forward touches must outlive the capture. What that
    takes depends on the layer's comm path:

    * Fleet/Handle path: the eager forward creates a ``Handle`` per call and
      destroys it in a ``finally``, so that handle's buffers are freed the
      moment capture ends and every replay dereferences dead memory -- which
      is silent at capture and faults (or worse, reads garbage) at replay.
      This object holds one long-lived handle instead: the allocating half
      (``ncclEpInitHandle``, via ``create_handle``) runs here, outside the
      capture, and the per-step half (``Handle.update``) is recorded inside it.
    * MoEEpCommunication path: the :class:`MoEEpCommunication` already
      outlives every forward, so the state carries ``handle=None``.

    On both paths it pins the tensors the graph binds. ``out`` is the address
    combine writes and the inputs are the addresses dispatch reads (on the
    Fleet/Handle path ``topk_weights`` is moreover bound into the handle at
    creation), so all must be stable buffers whose *contents* the caller
    overwrites between replays -- the standard CUDA-graph static-input
    discipline.
    ``forward`` re-checks the owning layer and buffer metadata on every call: a
    caller that silently passes a fresh tensor would otherwise get a graph that
    keeps serving the old one, and a caller that passes another layer's state
    would run this layer's tokens over that layer's transport and write that
    layer's output buffer.
    """

    __slots__ = (
        "_layer_ref",
        "_handle",
        "_hidden_states",
        "_topk_ids",
        "_topk_weights",
        "_out",
        "_buffer_signatures",
        "_destroyed",
    )

    def __init__(
        self,
        layer: "MoEEpSplitLayer",
        handle: "Handle | None",
        t: "MoEEpTensors",
        out: torch.Tensor,
    ) -> None:
        """Bind one layer, its persistent handle (if any) and one static buffer set.

        Not public: everything the graph will record has to be established
        outside a capture, which is what
        :meth:`MoEEpSplitLayer.create_graph_state` does before constructing the
        state.
        """
        # weakref, matching MoEEpMegaWorkspace: the layer holds the state,
        # so a strong back-reference would make the pair uncollectable.
        self._layer_ref = weakref.ref(layer)
        self._handle = handle
        self._hidden_states = t.hidden_states
        self._topk_ids = t.topk_ids
        self._topk_weights = t.topk_weights
        self._out = out
        # Tensor metadata is mutable: retaining the tensor alone does not
        # preserve the binding if the caller uses set_(), resize_(), or .data.
        self._buffer_signatures = {
            name: _graph_tensor_signature(tensor)
            for name, tensor in (
                ("hidden_states", self._hidden_states),
                ("topk_ids", self._topk_ids),
                ("topk_weights", self._topk_weights),
                ("out", self._out),
            )
        }
        self._destroyed = False

    @property
    def out(self) -> torch.Tensor:
        """The buffer combine writes on every replay."""
        return self._out

    @property
    def destroyed(self) -> bool:
        """Whether :meth:`destroy` has already run.

        A destroyed state can no longer back a ``forward``, and it no longer
        occupies its layer's one graph-state slot.
        """
        return self._destroyed

    def _check(self, layer: "MoEEpSplitLayer", t: "MoEEpTensors") -> None:
        """Gate one ``forward``: right layer, live state, unchanged buffers.

        The owner and buffer checks matter because of capture specifically,
        and neither can be caught downstream (a destroyed state is simply
        wrong at any time, and does surface later as a faulting handle). The
        state belongs to the creating layer's transport, so a state used by
        another layer routes that layer's tokens over a foreign transport and
        writes a foreign ``out`` -- and with the usual
        homogeneous FleetParams no shape disagrees anywhere, so it produces
        wrong results in silence. A rebound tensor is likewise fine eagerly and
        silently wrong once captured, because the graph keeps serving the
        address it saw at capture time.
        """
        if self._destroyed:
            raise RuntimeError(
                "MoEEpSplitGraphState has been destroyed; create a new one "
                "with MoEEpSplitLayer.create_graph_state()."
            )
        owner = self._layer_ref()
        if owner is None:
            # Deliberately not folded into the mismatch below: the state is
            # unusable because its layer -- and the fleet its persistent handle
            # borrows buffers from -- is gone, not because two live layers were
            # mixed up. Calling that a mismatch would send the reader hunting
            # for a second layer that no longer exists.
            raise MoEEpConfigError(
                "the MoEEpSplitLayer that created this MoEEpSplitGraphState "
                "has been garbage collected, so the transport it is bound to "
                "has been destroyed. Keep the layer alive for at least as "
                "long as the state and any graph captured from it."
            )
        if owner is not layer:
            raise MoEEpConfigError(
                "this MoEEpSplitGraphState belongs to a different "
                "MoEEpSplitLayer. It is bound to that layer's transport -- a "
                "different communicator and a different set of transport "
                "buffers -- so running this layer's round trip on it would "
                "send this layer's tokens over the other layer's transport and "
                "write the result into the other state's `out` buffer. Pass "
                "the state returned by THIS layer's create_graph_state()."
            )
        for name, bound, got in (
            ("hidden_states", self._hidden_states, t.hidden_states),
            ("topk_ids", self._topk_ids, t.topk_ids),
            ("topk_weights", self._topk_weights, t.topk_weights),
            ("out", self._out, self._out),
        ):
            expected = self._buffer_signatures[name]
            actual = _graph_tensor_signature(got)
            # The handle also retains the original weights tensor; passing an
            # unchanged alias must not hide a mutation of that original.
            if actual == expected and bound is not got:
                actual = _graph_tensor_signature(bound)
            if actual != expected:
                raise ValueError(
                    f"MoEEpSplitGraphState: {name} must be the same buffer the "
                    "state was created with, with unchanged shape, strides, "
                    "dtype and device. Expected "
                    f"(data_ptr, shape, strides, dtype, device)={expected}, "
                    f"got {actual}. Copy new values into the registered tensor "
                    "in place instead of rebinding it or changing its metadata."
                )

    def destroy(self) -> None:
        """Release the persistent handle, if any. Idempotent.

        Only safe once every graph that captured this state has been retired
        and the device synchronized; the graphs hold pointers into the
        handle's buffers.
        """
        if self._destroyed:
            return
        self._destroyed = True
        # Best effort, and deliberately so. Neither shipped backend makes this
        # retryable: NcclEpHandle.destroy() already swallows the C++ release
        # failure itself and latches its own destroyed flag, and
        # NixlEpHandle.destroy() only drops Python references. A raise here
        # could therefore only come from a future backend -- and it would
        # escape through MoEEpSplitLayer.destroy(), which calls this FIRST, so
        # it would permanently skip the fleet and the comm-runtime teardown
        # below it and leak both. A stranded handle is the smaller leak.
        try:
            if self._handle is not None:
                self._handle.destroy()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "moe_ep split graph-state handle release failed: %s",
                exc,
                exc_info=True,
            )
        layer = self._layer_ref()
        if layer is not None and layer._graph_state is self:
            layer._graph_state = None


_TOKEN_DTYPE_BY_BYTES = {1: torch.uint8, 2: torch.bfloat16, 4: torch.float32}


class _StageTimer:
    """Records a CUDA event pair around one forward stage into ``timings``."""

    __slots__ = ("_timings", "_stage", "_start")

    def __init__(
        self,
        timings: "dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]",
        stage: str,
    ) -> None:
        self._timings = timings
        self._stage = stage

    def __enter__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def __exit__(self, *exc_info) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._timings[self._stage] = (self._start, end)


class MoEEpSplitLayer(nn.Module):
    """Expert-Parallel layer: dispatch → inner kernel → combine.

    The comm backend runs on one of two peer paths, each over its own
    transport interface:

    * **Fleet/Handle path** (``nccl_ep``, ``nixl_ep``): a long-lived
      :class:`Fleet` creates a :class:`Handle` that carries one step's routing,
      and dispatch and combine run on that handle.
    * **MoEEpCommunication path** (``nvlink_one_sided``, ``cake``,
      ``nvlink_two_sided``): one long-lived :class:`MoEEpCommunication` takes
      the routing with each dispatch. It exchanges tokens rank-major, so it
      requires the low-latency RANK_MAJOR layout.

    Each path turns its dispatch output into the same
    :class:`SplitKernelContext`, so the inner kernel is shared.
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, SplitConfig, object] = "nccl_ep",
    ) -> None:
        """Validate the EP config and hand ``weights`` to the inner kernel.

        The transport is built lazily on the first forward, so construction
        allocates no transport buffers. It is not comm-free by default, though:
        ``BootstrapConfig.auto_bootstrap`` defaults to True, and this
        constructor then calls ``bootstrap_moe_ep_runtime()``, which for every
        registered comm backend initializes ``torch.distributed`` (and, for
        ``world_size > 1``, raises unless it is already initialized or
        RANK/WORLD_SIZE are set). Pass ``auto_bootstrap=False`` to keep
        construction legal before ``torch.distributed`` is up. ``weights`` is
        released once the kernel has preprocessed it, so the layer never pins a
        second copy for the model's lifetime.
        """
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

        # The transport of whichever path the comm backend runs on; the other
        # stays None.
        self._fleet: Fleet | None = None
        self._communication: MoEEpCommunication | None = None
        # Set by create_graph_state(); the layer keeps at most one live state
        # so destroy() can tear it down before the transport it borrows
        # buffers from.
        self._graph_state: MoEEpSplitGraphState | None = None
        # Flipped by the first successful EAGER forward on this layer, with or
        # without a graph state. Everything a capture needs warmed is a
        # property of the layer and its transport, not of one state: the inner
        # MoE kernel's lazy build and backend selection (which captures a graph
        # of its own -- nested capture is illegal) and the transport's cold
        # first dispatch/combine. Both eager forwards run the same dispatch →
        # compute → combine over the same transport, so either one warms them.
        # See the guard in forward().
        self._warmed = False
        # Set by destroy(). Without it a later forward() silently builds a
        # brand-new transport on an already-finalized comm runtime
        # (_ensure_fleet and _ensure_communication only test for None) -- the
        # same hole MoEEpMegaLayer closes with its own flag.
        self._destroyed = False

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

    def _uses_communication(self) -> bool:
        """Whether the comm backend runs on the MoEEpCommunication path rather
        than the Fleet/Handle path."""
        on_fleet = is_fleet_backend(self._comm_backend)
        on_communication = is_communication_backend(self._comm_backend)
        if on_fleet == on_communication:
            raise MoEEpConfigError(
                f"comm backend {self._comm_backend_name()!r} must be registered "
                "as exactly one of a Fleet backend or a MoEEpCommunication "
                f"backend; it is registered as {'both' if on_fleet else 'neither'}."
            )
        return on_communication

    def _validate_at_init(self) -> None:
        validate_bootstrap_world_size(self._bootstrap)
        validate_fleet_weights(
            self._weights, self._fleet_params, self._bootstrap.world_size
        )
        if self._uses_communication():
            self._validate_communication_config()
        else:
            self._validate_fleet_config()

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
            layer.forward(t, graph_state=state)      # warmup, still eager -- REQUIRED
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
        optional: ``forward`` refuses to capture on a layer that has never
        completed an eager forward. It is what compiles/autotunes the inner
        kernel and establishes the transport's steady state, neither of which
        can happen during capture.

        The routing SHAPE is fixed here -- ``top_k`` and the token count are
        baked into the graph (and, on the Fleet/Handle path, into the handle).
        A different batch size needs its own state and its own graph, the usual
        multi-size-graph pattern -- and, today, its own layer: a layer holds at
        most one live state, and destroying the live one to make room frees the
        buffers the graph already captured from it replays against.
        """
        if self._destroyed:
            raise MoEEpConfigError(
                "this MoEEpSplitLayer has been destroyed; its transport and "
                "comm runtime are gone."
            )
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
                "this MoEEpSplitLayer already has a live graph state (one "
                "per layer). Destroying it frees the buffers every graph "
                "captured from it replays against, so a second shape needs a "
                "second layer, not a second state on this one."
            )
        if out is None:
            out = torch.empty_like(
                t.hidden_states, memory_format=torch.contiguous_format
            )
        elif (
            out.shape != t.hidden_states.shape
            or out.dtype != t.hidden_states.dtype
            or out.device != t.hidden_states.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous and match hidden_states: expected "
                f"shape {tuple(t.hidden_states.shape)} dtype {t.hidden_states.dtype} "
                f"device {t.hidden_states.device}, got shape {tuple(out.shape)} "
                f"dtype {out.dtype} device {out.device} strides {out.stride()}."
            )

        if self._uses_communication():
            state = self._create_communication_graph_state(t, out)
        else:
            state = self._create_fleet_graph_state(t, out)
        self._graph_state = state
        return state

    def forward(
        self,
        t: "MoEEpTensors",
        *,
        graph_state: Optional[MoEEpSplitGraphState] = None,
    ) -> torch.Tensor:
        """Run one EP forward: dispatch → inner kernel → combine.

        With ``graph_state`` (from :meth:`create_graph_state`) the forward
        writes that state's static output buffer and, on the Fleet/Handle path,
        reuses its persistent handle instead of creating and destroying one
        per call, which is what makes the call safe to record into a CUDA
        graph. Without it, behaviour is unchanged.
        """
        if self._destroyed:
            raise MoEEpConfigError(
                "this MoEEpSplitLayer has been destroyed; its transport and "
                "comm runtime are gone."
            )
        ensure_bootstrap_dist_validated(self._bootstrap)
        validate_split_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            self._fleet_params,
        )

        capturing = _is_capturing()
        if graph_state is not None:
            if not isinstance(graph_state, MoEEpSplitGraphState):
                # Without this the mistyped argument reaches _check() and dies
                # with a bare AttributeError that names neither the argument
                # nor the call that produces a valid one.
                raise TypeError(
                    "graph_state must be created by "
                    "MoEEpSplitLayer.create_graph_state(); got "
                    f"{type(graph_state).__name__}."
                )
            graph_state._check(self, t)
            if self.enable_timing and capturing:
                raise MoEEpConfigError(
                    "enable_timing synchronizes the device to read its CUDA "
                    "events, which is illegal during graph capture. Turn it "
                    "off to capture, and time the replay instead."
                )
            if capturing and not self._warmed:
                # create_graph_state() builds the transport (on the
                # Fleet/Handle path also the handle, plus the one update
                # outside the capture that nccl_ep requires) -- but runs no
                # forward, so the inner kernel is still unbuilt and the
                # transport has no steady state. Caught here because the
                # symptom otherwise surfaces from inside the inner kernel's
                # backend selection (a nested capture plus a device sync) and
                # names nothing in this file.
                raise MoEEpConfigError(
                    "this MoEEpSplitLayer has never run an eager forward, so "
                    "the capture would be the first round trip it ever sees: "
                    "the inner MoE kernel is still unbuilt and selecting its "
                    "backend captures a graph of its own (nested capture is "
                    "illegal), and the transport has no steady state. Run one "
                    "eager forward on this layer -- layer.forward(t) or "
                    "layer.forward(t, graph_state=state), either warms the "
                    "same transport and kernel -- outside the capture, on "
                    "every EP rank, first."
                )

        if self._uses_communication():
            out = self._communication_forward(t, graph_state)
        else:
            out = self._fleet_forward(t, graph_state)
        if not capturing:
            # A completed eager forward is the warmup a later capture needs:
            # it built the inner kernel and ran the transport.
            self._warmed = True
        return out

    def destroy(self) -> None:
        """Tear down graph state, then the transport, then comm runtime.

        The order is load-bearing rather than stylistic: the graph state's
        persistent handle holds buffers the fleet owns, and the transport
        lives on the comm runtime. Collective on every EP rank, idempotent, and
        safe only once every graph captured from this layer has been retired
        and the device synchronized.
        """
        if self._destroyed:
            return
        # Before the fleet: the persistent handle holds buffers the fleet owns.
        if self._graph_state is not None:
            self._graph_state.destroy()
            self._graph_state = None
        # finally, not a plain sequence: NcclEpFleet.destroy() tears down the
        # C++ group with no suppression of its own (NixlEpFleet suppresses its
        # own buffer teardown), and this has to reach the runtime release even
        # when that raises, or the bootstrap ref_count leaks and the layer keeps
        # serving forward() on a half-destroyed transport.
        try:
            if self._communication is not None:
                self._communication.destroy()
                self._communication = None
            if self._fleet is not None:
                self._fleet.destroy()
                self._fleet = None
        finally:
            if self._runtime is not None:
                finalize_moe_ep_runtime(self._runtime)
                self._runtime = None
            self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()

    def _stage(
        self,
        timings: "dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]",
        stage: str,
    ) -> "contextlib.AbstractContextManager[None]":
        """Time one forward stage into ``timings`` when ``enable_timing`` is set."""
        if self.enable_timing:
            return _StageTimer(timings, stage)
        return contextlib.nullcontext()

    def _publish_timings(
        self, timings: "dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]"
    ) -> None:
        if timings:
            torch.cuda.synchronize()
            self.last_timings_ms = {
                stage: start.elapsed_time(end)
                for stage, (start, end) in timings.items()
            }

    # ------------------------------------------------- Fleet/Handle path

    def _validate_fleet_config(self) -> None:
        backend_name = self._comm_backend_name()
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

    def _create_fleet_graph_state(
        self, t: "MoEEpTensors", out: torch.Tensor
    ) -> MoEEpSplitGraphState:
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
        return MoEEpSplitGraphState(self, handle, t, out)

    def _fleet_forward(
        self, t: "MoEEpTensors", graph_state: Optional[MoEEpSplitGraphState]
    ) -> torch.Tensor:
        if graph_state is not None:
            handle = graph_state._handle
            # The per-step half: recompute routing from the (rewritten) ids
            # into the handle's existing buffers. Recorded inside the capture.
            handle.update(HandleParams(topk_ids=t.topk_ids))
            try:
                # The state owns this buffer for its lifetime, so the
                # transport may cache per-address state for it.
                return self._fleet_dispatch_compute_combine(
                    handle, t, graph_state._out, out_is_stable=True
                )
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
            # A fresh buffer per call: out_is_stable stays False so the
            # transport does not pin an address it will never see again.
            return self._fleet_dispatch_compute_combine(
                handle, t, torch.empty_like(t.hidden_states)
            )
        finally:
            handle.complete()
            handle.destroy()

    def _fleet_dispatch_compute_combine(
        self,
        handle: "Handle",
        t: "MoEEpTensors",
        out: torch.Tensor,
        *,
        out_is_stable: bool = False,
    ) -> torch.Tensor:
        """Dispatch, run the inner kernel and combine on a bound handle."""
        timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        with self._stage(timings, "dispatch"):
            dispatched = handle.dispatch(
                DispatchInputParams(
                    x=[self._kernel.pack_dispatch_payload(t.hidden_states)]
                )
            )
        with self._stage(timings, "compute"):
            expert_out = self._kernel.compute(self._fleet_kernel_context(dispatched))
        with self._stage(timings, "combine"):
            combined = handle.combine(
                CombineInputParams(x=[expert_out], out=out, out_is_stable=out_is_stable)
            )
        self._publish_timings(timings)
        return combined.x

    def _fleet_kernel_context(self, dispatched: DispatchOutput) -> SplitKernelContext:
        return SplitKernelContext(
            expert_tensors=dispatched.expert_tensors,
            num_tokens=dispatched.get_num_tokens(),
            fleet_params=self._fleet_params,
            recv_topk_idx=dispatched.recv_topk_idx,
            recv_topk_weights=dispatched.recv_topk_weights,
            recv_count=dispatched.expert_counts,
        )

    # ------------------------------------------ MoEEpCommunication path

    def _validate_communication_config(self) -> None:
        backend_name = self._comm_backend_name()
        if (
            self._fleet_params.algorithm is not EpAlgorithm.LOW_LATENCY
            or self._fleet_params.layout is not EpLayout.RANK_MAJOR
        ):
            raise MoEEpConfigError(
                f"comm backend {backend_name!r} exchanges tokens rank-major; "
                "set FleetParams(algorithm=EpAlgorithm.LOW_LATENCY, "
                "layout=EpLayout.RANK_MAJOR)."
            )
        if self._fleet_params.dtype_bytes not in _TOKEN_DTYPE_BY_BYTES:
            raise MoEEpConfigError(
                f"comm backend {backend_name!r} supports dtype_bytes in "
                f"{sorted(_TOKEN_DTYPE_BY_BYTES)}, got "
                f"{self._fleet_params.dtype_bytes}."
            )

    def _ensure_communication(self, top_k: int) -> MoEEpCommunication:
        if self._communication is None:
            fp = self._fleet_params
            self._communication = create_communication(
                self._bootstrap,
                MoEEpCommParams(
                    num_experts=fp.num_experts,
                    top_k=top_k,
                    max_tokens_per_rank=fp.max_tokens_per_rank,
                    hidden_size=fp.token_hidden_size,
                    dtype=_TOKEN_DTYPE_BY_BYTES[fp.dtype_bytes],
                ),
                self._comm_backend,
            )
        elif self._communication.params.top_k != top_k:
            raise MoEEpConfigError(
                f"top_k changed from {self._communication.params.top_k} to "
                f"{top_k}; a MoEEpSplitLayer serves a single top_k."
            )
        return self._communication

    def _create_communication_graph_state(
        self, t: "MoEEpTensors", out: torch.Tensor
    ) -> MoEEpSplitGraphState:
        # The communication outlives every forward, so the state only pins
        # the buffers; creating it here keeps its collective setup outside the
        # capture.
        comm = self._ensure_communication(t.topk_ids.shape[1])
        if not comm.supports_cuda_graph:
            raise MoEEpConfigError(
                f"comm backend {self._comm_backend_name()!r} cannot be "
                "captured into a CUDA graph."
            )
        return MoEEpSplitGraphState(self, None, t, out)

    def _communication_forward(
        self, t: "MoEEpTensors", graph_state: Optional[MoEEpSplitGraphState]
    ) -> torch.Tensor:
        if graph_state is not None:
            # The communication is persistent and the state owns `out`, so
            # the forward records as is.
            return self._communication_dispatch_compute_combine(t, graph_state._out)

        if _is_capturing():
            raise MoEEpConfigError(
                "capturing MoEEpSplitLayer.forward() needs "
                "graph_state=layer.create_graph_state(t), built outside the "
                "capture, to pin the buffers the graph binds."
            )
        return self._communication_dispatch_compute_combine(
            t, torch.empty_like(t.hidden_states)
        )

    def _communication_dispatch_compute_combine(
        self, t: "MoEEpTensors", out: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch, run the inner kernel and combine over the communication."""
        comm = self._ensure_communication(t.topk_ids.shape[1])
        timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        with self._stage(timings, "dispatch"):
            received = comm.dispatch(
                self._kernel.pack_dispatch_payload(t.hidden_states),
                t.topk_ids,
                t.topk_weights,
            )
        with self._stage(timings, "compute"):
            ctx = self._communication_kernel_context(comm, received)
            expert_out = self._kernel.compute(ctx)
        with self._stage(timings, "combine"):
            combined = comm.combine(expert_out.reshape(ctx.num_tokens, -1), output=out)
        self._publish_timings(timings)
        return combined

    def _communication_kernel_context(
        self, comm: MoEEpCommunication, received: MoEEpDispatchResult
    ) -> SplitKernelContext:
        # The inner kernels take RANK_MAJOR routing as ids local to this rank,
        # with -1 for picks this rank does not own.
        first_local = comm.ep_rank * comm.num_local_experts
        ids = received.topk_ids
        is_local = (ids >= first_local) & (ids < first_local + comm.num_local_experts)
        return SplitKernelContext(
            expert_tensors=received.hidden_states.view(
                comm.ep_size, received.tokens_per_rank, -1
            ),
            num_tokens=comm.ep_size * received.tokens_per_rank,
            fleet_params=self._fleet_params,
            recv_topk_idx=torch.where(
                is_local, ids - first_local, torch.full_like(ids, -1)
            ),
            recv_topk_weights=received.topk_weights,
        )
