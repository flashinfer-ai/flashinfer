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
import hashlib
from typing import TYPE_CHECKING, Sequence

from ..... import _require_built
from .....errors import MoEEpNotBuiltError
from .....core.validation.common import (
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
)
from .....algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobTopologyCapacity,
    _index_knobs,
)
from .....config import FleetParams, QuantType
from .....core.comm.fleet import Fleet, _BACKEND_REGISTRY
# from .....api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from .....config import BootstrapConfig
    from .....core.comm.handle import Handle


def _load_nixl_ep():
    """Import the vendored ``nixl_ep`` Python module (staged by the build).

    The base NIXL libs are ctypes-preloaded (RTLD_GLOBAL) so the extension
    resolves its symbols; the extension itself is then loaded exactly once,
    as the package submodule (buffer.py does ``from . import nixl_ep_cpp``)
    — NOT dlopen'd separately, which would double-load its static state.
    """
    from . import _libs_dir, _preload_libnixl

    if not list(_libs_dir.glob("nixl_ep_cpp*.so")):
        raise MoEEpNotBuiltError(
            "nixl_ep loaders not staged; rebuild with `pip install -e .` "
            "(BUILD_NIXL_EP=1 makes missing build deps a hard error)"
        )
    _preload_libnixl()
    try:
        # The vendored package is staged under _vendored/nixl_ep/ by
        # build_backend._build_nixl_ep, with nixl_ep_cpp*.so inside it.
        import os
        import sys

        _here = os.path.dirname(__file__)
        vendored = os.path.abspath(os.path.join(_here, "_vendored"))
        if os.path.isdir(vendored) and vendored not in sys.path:
            sys.path.insert(0, vendored)
        import nixl_ep  # type: ignore[import-not-found]
    except ImportError as e:
        raise MoEEpNotBuiltError(
            "nixl_ep python module not importable; rebuild with `pip install -e .` "
            "(BUILD_NIXL_EP=1 makes missing build deps a hard error)"
        ) from e
    return nixl_ep


# Per-GROUP generation counters namespacing derived rendezvous stores, keyed
# by the group's sorted global-rank tuple. Fleet creation is collective over
# the EP group, so each group's counter agrees across its ranks; re-created
# fleets then never reuse a prior fleet's keys. A single process-wide counter
# would diverge when a process belongs to several EP subgroups and creates
# their fleets in a different interleaving than its peers.
_STORE_GENS: dict = {}


def _resolve_store(bootstrap: "BootstrapConfig"):
    """Return the rendezvous store the NIXL ``Buffer`` bootstraps over.

    Resolution order (the NIXL analogue of nccl_ep's ``_resolve_comm``):

    1. ``bootstrap.tcp_store`` set — use it as-is (previous behavior).
    2. otherwise — derive a ``PrefixStore`` from torch.distributed's default
       store, so hosts that pass only ``process_group`` (e.g. vLLM's EP group,
       the same ``BootstrapConfig`` shape the nccl_ep backend consumes) work
       without constructing a second TCPStore on a sibling port. The prefix is
       namespaced by the EP group's global ranks plus that group's generation
       counter so disjoint EP subgroups and re-created fleets never collide on
       store keys.
    """
    if bootstrap.tcp_store is not None:
        return bootstrap.tcp_store

    import torch.distributed as dist

    if not dist.is_initialized():
        raise ValueError(
            "NixlEpFleet needs a rendezvous store: set bootstrap.tcp_store "
            "(a torch.distributed.TCPStore), or initialize torch.distributed "
            "so one can be derived from the default store."
        )

    from .....core.bootstrap_utils import bootstrap_comm_group

    # torch exposes no public accessor for the default store;
    # _get_default_store has been stable across torch 2.x but is private, so
    # fail with the explicit-tcp_store escape hatch rather than a raw
    # AttributeError if it ever moves.
    try:
        base_store = dist.distributed_c10d._get_default_store()
    except (AttributeError, RuntimeError) as e:
        raise ValueError(
            "Could not derive a rendezvous store from torch.distributed's "
            "default store; set bootstrap.tcp_store (a "
            "torch.distributed.TCPStore) explicitly."
        ) from e

    ranks = tuple(sorted(dist.get_process_group_ranks(bootstrap_comm_group(bootstrap))))
    gen = _STORE_GENS.get(ranks, 0)
    _STORE_GENS[ranks] = gen + 1
    # Encode the FULL group identity: a min×len-style prefix collides for
    # overlapping groups like (0,1,2,3) vs (0,2,4,6). Digest the rank tuple
    # (not hash(), which is per-process randomized) to keep the prefix
    # bounded for large groups while staying identical across ranks.
    group_id = hashlib.sha1("-".join(map(str, ranks)).encode()).hexdigest()[:12]
    prefix = f"flashinfer/moe_ep/nixl_ep/{group_id}/{gen}"
    return dist.PrefixStore(prefix, base_store)


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
        validate_bootstrap_world_size(bootstrap)
        store = _resolve_store(bootstrap)

        self._params = params
        self._fleet_knobs = _index_knobs(algo_knobs)
        cap_knob = self._fleet_knobs.get(FleetAlgoKnobTopologyCapacity)
        cap = int(cap_knob.n) if cap_knob is not None else bootstrap.world_size  # type: ignore[attr-defined]
        validate_fleet_params(
            params,
            backend="nixl_ep",
            world_size=bootstrap.world_size,
            quant=self._fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
            topology_capacity=cap,
        )
        self._bootstrap = bootstrap
        self._capacity = cap

        nixl_ep = _load_nixl_ep()
        self._nixl_ep = nixl_ep  # handles read topk_idx_t off it
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
            tcp_store_group=store,
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

    @property
    def capacity(self) -> int:
        """Rank capacity the Buffer was sized to (≥ current world_size)."""
        return self._capacity

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
        if self._buffer is not None:
            with contextlib.suppress(Exception):
                self._buffer.destroy()
            self._buffer = None
        self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


# Module-load side effect: register the backend.
_BACKEND_REGISTRY["nixl_ep"] = NixlEpFleet
