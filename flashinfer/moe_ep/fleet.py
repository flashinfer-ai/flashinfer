"""Fleet — abstract Expert-Parallel transport endpoint.

A Fleet owns the durable resources of an EP transport: the NCCL communicator,
the RDMA buffer pool, the rank-set bookkeeping. Per-iteration state lives on
:class:`flashinfer.moe_ep.Handle` instances created by
:meth:`Fleet.create_handle`.

Backends register themselves in :data:`_BACKEND_REGISTRY` at import time
(see :mod:`flashinfer.moe_ep.nccl_ep` and :mod:`flashinfer.moe_ep.nixl_ep`).
Construction is mediated by :func:`create_fleet` so the public surface stays
backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence

# from ..api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from .algo_knobs import AlgoKnob
    from .config import BootstrapConfig, FleetParams, HandleParams
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
        """Backend Fleet ctors share this signature so the registry lookup
        in :func:`create_fleet` is type-safe."""

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


# @flashinfer_api  # disabled per PR #3453 review
def create_fleet(
    bootstrap: "BootstrapConfig",
    params: "FleetParams",
    algo_knobs: Sequence["AlgoKnob"] = (),
    backend: str | object = "nccl_ep",
) -> Fleet:
    """Instantiate the registered Fleet class for `backend`.

    `backend` accepts either a string name (``"nccl_ep"`` / ``"nixl_ep"``) or a
    config object with a ``backend_name`` attribute (e.g. :class:`NcclEpConfig`
    from :mod:`flashinfer.moe_ep.split_backends`).
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
