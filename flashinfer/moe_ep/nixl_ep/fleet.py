"""NixlEpFleet — NIXL-EP-backed Fleet implementation.

NIXL's ``Buffer`` collapses Fleet + Handle into one object. We surface the
design's Fleet / Handle split by:

* Constructing the Buffer at NixlEpFleet.__init__, running
  ``update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes)``
  and ``connect_ranks([0..world_size))`` once.
* Reusing the same Buffer for every Handle created via ``create_handle``.
* Stashing the per-dispatch ``(handle, event, hook)`` tuple from NIXL inside
  the NixlEpHandle for combine and complete to read.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Sequence

from .. import MoEEpNotBuiltError, _require_built
from .._validators import (
    MoEEpConfigError,
    validate_arch_for_backend,
    validate_fleet_params,
)
from ..algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobTopologyCapacity,
    _index_knobs,
)
from ..config import FleetParams, QuantType
from ..fleet import Fleet, _BACKEND_REGISTRY
# from ...api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from ..config import BootstrapConfig
    from ..handle import Handle


def _load_nixl_ep():
    """Import the vendored ``nixl_ep`` Python module + load the .so."""
    try:
        from . import _load_nixl_ep_cpp  # noqa: F401
    except ImportError as e:
        raise MoEEpNotBuiltError(
            "nixl_ep loaders not staged; rebuild with BUILD_NIXL_EP=1"
        ) from e
    _load_nixl_ep_cpp()
    try:
        # The vendored module is staged under flashinfer/moe_ep/nixl_ep/_vendored/nixl_ep
        # by Part A's _build_nixl_ep step. Add that to sys.path so plain
        # `import nixl_ep` resolves.
        import os
        import sys

        vendored = os.path.join(os.path.dirname(__file__), "_vendored")
        if vendored not in sys.path and os.path.isdir(vendored):
            sys.path.insert(0, vendored)
        import nixl_ep  # type: ignore[import-not-found]
    except ImportError as e:
        raise MoEEpNotBuiltError(
            "nixl_ep python module not importable; rebuild with BUILD_NIXL_EP=1"
        ) from e
    return nixl_ep


class NixlEpFleet(Fleet):
    """Owns a ``nixl_ep.Buffer`` for one rank."""

    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: FleetParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        _require_built("nixl_ep")
        validate_arch_for_backend("nixl_ep")

        if bootstrap.tcp_store is None:
            raise ValueError(
                "NixlEpFleet requires bootstrap.tcp_store to be set; "
                "construct a torch.distributed.TCPStore and pass it in."
            )

        self._params = params
        self._fleet_knobs = _index_knobs(algo_knobs)
        validate_fleet_params(
            params,
            backend="nixl_ep",
            world_size=bootstrap.world_size,
            quant=self._fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
        )
        self._bootstrap = bootstrap

        nixl_ep = _load_nixl_ep()
        # Topology capacity: how many ranks can join later via update_topology.
        cap_knob = self._fleet_knobs.get(FleetAlgoKnobTopologyCapacity)
        cap = int(cap_knob.n) if cap_knob is not None else bootstrap.world_size  # type: ignore[attr-defined]
        self._capacity = cap

        # `cap` is the num_ranks the experts are sharded across. validate_fleet_params
        # only checks num_experts % world_size; when a TopologyCapacity knob sets
        # cap != world_size, num_experts // cap below would silently truncate
        # experts. Enforce divisibility (and a positive per-rank count) here.
        if cap <= 0 or params.num_experts % cap != 0:
            raise MoEEpConfigError(
                f"nixl_ep: num_experts ({params.num_experts}) must be a positive "
                f"multiple of topology capacity ({cap})"
            )

        # num_rdma_bytes — size the per-rank RDMA buffer via the upstream hint.
        num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            params.max_tokens_per_rank,
            params.token_hidden_size,
            cap,
            params.num_experts,
        )
        self._buffer = nixl_ep.Buffer(
            rank=bootstrap.rank,
            low_latency_mode=True,  # MVP: LL only
            tcp_store_group=bootstrap.tcp_store,
        )
        num_experts_per_rank = params.num_experts // cap
        self._buffer.update_memory_buffers(cap, num_experts_per_rank, num_rdma_bytes)
        self._buffer.connect_ranks(list(range(bootstrap.world_size)))
        self._destroyed = False

    @property
    def use_fp8(self) -> bool:
        q = self._fleet_knobs.get(FleetAlgoKnobQuantization)
        if not q:
            return False
        fp8_types = {QuantType.FP8E4M3, QuantType.FP8E5M2, QuantType.NVFP8}
        return bool(q.quants & fp8_types)  # type: ignore[attr-defined]

    @property
    def use_ue8m0(self) -> bool:
        q = self._fleet_knobs.get(FleetAlgoKnobQuantization)
        return bool(q and QuantType.UE8M0 in q.quants)  # type: ignore[attr-defined]

    @property
    def buffer(self):
        return self._buffer

    @property
    def params(self) -> FleetParams:
        return self._params

    # @flashinfer_api  # disabled per PR #3453 review
    def create_handle(self, params, algo_knobs: Sequence[AlgoKnob] = ()) -> "Handle":
        from .handle import NixlEpHandle

        return NixlEpHandle(self, params, algo_knobs)

    # @flashinfer_api  # disabled per PR #3453 review
    def update_topology(
        self,
        bootstrap: "BootstrapConfig",
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        """Diff new vs current rank set, disconnect removed + connect added."""
        old_ranks = set(range(self._bootstrap.world_size))
        new_ranks = set(range(bootstrap.world_size))
        removed = sorted(old_ranks - new_ranks)
        added = sorted(new_ranks - old_ranks)
        if removed:
            self._buffer.disconnect_ranks(removed)
        if added:
            self._buffer.connect_ranks(added)
        self._bootstrap = bootstrap
        if algo_knobs:
            self._fleet_knobs = _index_knobs(algo_knobs)

    # @flashinfer_api  # disabled per PR #3453 review
    def destroy(self) -> None:
        if self._destroyed:
            return
        # Buffer destructor runs at GC; explicit destroy guarded behind ctor
        # flag we didn't set, so we let Python's __del__ handle it.
        self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


# Module-load side effect: register the backend.
_BACKEND_REGISTRY["nixl_ep"] = NixlEpFleet
