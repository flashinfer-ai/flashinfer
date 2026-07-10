"""Fleet — abstract Expert-Parallel transport endpoint.

Backends register themselves in :data:`_BACKEND_REGISTRY` at import time
(see :mod:`flashinfer.moe_ep.backends.split.comm.nccl_ep` and
:mod:`flashinfer.moe_ep.backends.split.comm.nixl_ep`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
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
