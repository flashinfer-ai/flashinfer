"""Fleet — abstract Expert-Parallel transport endpoint.

Backends register themselves in :data:`_BACKEND_REGISTRY` at import time
(see :mod:`flashinfer.moe_ep.backends.split.comm.nccl_ep` and
:mod:`flashinfer.moe_ep.backends.split.comm.nixl_ep`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, NoReturn, Sequence

if TYPE_CHECKING:
    import torch

    from ...algo_knobs import AlgoKnob
    from ...config import BootstrapConfig, HandleParams, FleetParams
    from .handle import Handle


_BACKEND_REGISTRY: "dict[str, Callable[..., Fleet]]" = {}


class Fleet(ABC):
    @abstractmethod
    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: "FleetParams",
        algo_knobs: Sequence["AlgoKnob"] = (),
    ) -> None:
        """Backend Fleet ctors share this signature for registry lookup."""

    @abstractmethod
    def create_handle(
        self,
        params: "HandleParams",
        algo_knobs: Sequence["AlgoKnob"] = (),
    ) -> "Handle":
        """Return a Handle bound to this Fleet for one dispatch/combine pair."""

    @abstractmethod
    def update_topology(
        self,
        bootstrap: "BootstrapConfig",
        algo_knobs: Sequence["AlgoKnob"] = (),
    ) -> None:
        """Re-register the Fleet over a new rank set / process group."""

    @abstractmethod
    def destroy(self) -> None:
        """Release transport resources. Idempotent."""

    # ------------------------------------------------------- fault tolerance
    #
    # Optional capability, enabled per-Fleet with FleetAlgoKnobFaultTolerance.
    # These are deliberately CONCRETE raising defaults rather than
    # @abstractmethod: making them abstract would break every existing Fleet
    # implementation (including the test suite's _StubFleet) the moment this
    # lands. Backends that support rank masking override them.
    #
    # Mask convention across the whole package: int32, length world_size,
    # 1 = active, 0 = masked. nixl_ep's inverted, capacity-length buffer is
    # normalized inside its own Fleet and never escapes.

    @property
    def supports_fault_tolerance(self) -> bool:
        """True when this Fleet was built with rank masking enabled."""
        return False

    def query_active_mask(self, out: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Per-rank liveness, ``[world_size]`` int32 CUDA, **1 = active**.

        Stream-ordered against the current stream (both transports issue a
        device copy or kernel). Returns a Fleet-owned buffer when ``out`` is
        None — the caller must not retain it across calls. Must not be issued
        during CUDA-graph capture.
        """
        self._no_fault_tolerance("query_active_mask")

    def query_fault(self) -> bool:
        """True when a rank has been masked since the last ``clear_faults``.

        Host-side. Cheap on nccl_ep (a pinned-flag read); costs one small D2H
        sync on nixl_ep, which has no host error flag.

        Must not be called during CUDA-graph capture — *because* it is a host
        read rather than stream work, it cannot be captured, so it would
        return the capture-time answer and freeze the branch taken on it into
        the graph forever.
        """
        self._no_fault_tolerance("query_fault")

    def set_active_mask(self, mask: "Sequence[int] | torch.Tensor") -> None:
        """Apply an *absolute* ``[world_size]`` mask (1 = active).

        LOCAL, not collective — every rank must apply the same vector for the
        fleet to stay coherent, so prefer ``reconcile_active_mask`` unless you
        have an out-of-band agreement. ``mask[rank]`` must be 1: a rank cannot
        mask itself. Stream-ordered, and legal only between iterations — both
        transports read the mask live from the dispatch/combine kernels.
        """
        self._no_fault_tolerance("set_active_mask")

    def reconcile_active_mask(
        self, *, timeout_s: "float | None" = None
    ) -> "torch.Tensor":
        """Agree a mask across survivors via the bootstrap store, then apply it.

        Gathers each rank's local view through the rendezvous store, ANDs the
        views that arrive, treats a key still missing after ``timeout_s`` as a
        dead rank, has the lowest-numbered survivor publish the single agreed
        vector, applies it via ``set_active_mask``, and returns it.

        Deliberately NOT a ``torch.distributed`` allreduce: that would hang on
        exactly the rank being masked out.
        """
        self._no_fault_tolerance("reconcile_active_mask")

    def clear_faults(self, *, readmit: bool = False) -> None:
        """Re-arm fault detection; optionally re-admit masked ranks.

        ``readmit=False`` — LOCAL and cheap. Clears the transport's sticky
        error flag so the next fault is detectable. Masks are untouched:
        serving continues in degraded mode.

        ``readmit=True`` — **COLLECTIVE over survivors** and host-blocking.
        Resets masks and transport staging buffers so a delayed rank can
        rejoin. All survivors must call it simultaneously with no dispatch or
        combine in flight.
        """
        self._no_fault_tolerance("clear_faults")

    @property
    def active_mask_epoch(self) -> int:
        """Monotone generation counter of the currently applied mask."""
        return 0

    def _no_fault_tolerance(self, op: str) -> "NoReturn":
        from ...errors import MoEEpFaultToleranceUnsupportedError

        raise MoEEpFaultToleranceUnsupportedError(
            f"{type(self).__name__}.{op}() requires fault tolerance, which this "
            "Fleet was not built with. Pass FleetAlgoKnobFaultTolerance() in "
            "fleet_knobs at construction, and check "
            "flashinfer.moe_ep.supports_fault_tolerance(<backend>) first."
        )


def create_fleet(
    bootstrap: "BootstrapConfig",
    params: "FleetParams",
    algo_knobs: Sequence["AlgoKnob"] = (),
    backend: str | object = "nccl_ep",
) -> Fleet:
    """Instantiate the registered Fleet class for ``backend``.

    Expert weights are not validated here — they are a layer concern
    (validated in the MoEEpLayer ctor); the Fleet transport never reads them.
    """
    name = getattr(backend, "backend_name", backend)
    if not isinstance(name, str):
        raise TypeError(
            f"backend must be a string or have a .backend_name str attr; got {backend!r}"
        )
    if name not in _BACKEND_REGISTRY:
        available = sorted(_BACKEND_REGISTRY)
        raise KeyError(f"unknown backend {name!r}; available: {available}")
    return _BACKEND_REGISTRY[name](bootstrap, params, algo_knobs)
