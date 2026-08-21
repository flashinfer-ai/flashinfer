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

from ..... import _require_built
from .....errors import MoEEpFaultToleranceUnsupportedError, MoEEpNotBuiltError
from .....core.validation.common import (
    MoEEpConfigError,
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
)
from .....algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobFaultTolerance,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobTopologyCapacity,
    _index_knobs,
)
from .....config import FleetParams, QuantType
from .....core.bootstrap_utils import resolve_rendezvous_store
from .....core.comm.fault_tolerance import (
    ACTIVE,
    MASKED,
    FaultToleranceMixin,
    reject_graph_capture as _reject_graph_capture_impl,
)
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


def _resolve_store(bootstrap: "BootstrapConfig"):
    """Return the rendezvous store the NIXL ``Buffer`` bootstraps over.

    Thin alias over the shared resolver, which fault tolerance also uses (with
    ``subsystem="ft"``) since it needs a store on both transports.
    """
    return resolve_rendezvous_store(bootstrap, subsystem="nixl_ep")


class NixlEpFleet(FaultToleranceMixin, Fleet):
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
        ft = self._fleet_knobs.get(FleetAlgoKnobFaultTolerance)
        self._ft = ft if (ft is not None and ft.enabled) else None  # type: ignore[attr-defined]
        validate_fleet_params(
            params,
            backend="nixl_ep",
            world_size=bootstrap.world_size,
            quant=self._fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
            topology_capacity=cap,
            fault_tolerance=self._ft,  # type: ignore[arg-type]
        )
        self._bootstrap = bootstrap
        self._capacity = cap
        self._ft_epoch = 0
        self._ft_store_obj = None
        # Mirrors the mask we have applied, so set_active_mask can push only
        # the diff. connect_ranks below unmasks exactly [0, world_size).
        self._ft_applied = [ACTIVE] * bootstrap.world_size

        nixl_ep = _load_nixl_ep()
        self._nixl_ep = nixl_ep  # handles read topk_idx_t off it
        # num_rdma_bytes — size the per-rank RDMA buffer via the upstream hint.
        num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            params.max_tokens_per_rank,
            params.token_hidden_size,
            cap,
            params.num_experts,
        )
        buf_kwargs = {}
        if self._ft is not None and self._ft.timeout_ms:  # type: ignore[attr-defined]
            # LL: a timeout masks the offending rank. (HT traps instead, which
            # is why validate_fleet_params rejects FT+HT.) The mask buffer
            # itself is allocated unconditionally by update_memory_buffers
            # below, so timeout_ms is the only ctor-level FT knob.
            buf_kwargs["timeout_ms"] = int(self._ft.timeout_ms)  # type: ignore[attr-defined]
        try:
            self._buffer = nixl_ep.Buffer(
                rank=bootstrap.rank,
                low_latency_mode=True,  # MVP: LL only
                tcp_store_group=store,
                **buf_kwargs,
            )
        except TypeError as e:
            if not buf_kwargs:
                raise
            raise MoEEpFaultToleranceUnsupportedError(
                "the staged nixl_ep build's Buffer() does not accept "
                "`timeout_ms`; rebuild against a newer NIXL (BUILD_NIXL_EP=1), "
                "or drop FleetAlgoKnobFaultTolerance.timeout_ms to use the "
                "transport default."
            ) from e
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
        """Diff new vs current rank set, disconnect removed + connect added.

        Growing past the topology capacity is rejected: every per-rank array
        (RDMA, mask, sync) was sized to the capacity at construction, so
        connecting a rank beyond it writes out of bounds inside the transport
        rather than failing cleanly. Size the capacity for the largest world
        you intend to reach via ``FleetAlgoKnobTopologyCapacity``.

        Note the transport also requires removed ranks to form a *suffix* of
        the connected set. The diff below is suffix-only by construction
        (both sides are ``range(world_size)``), which is why shrinking works
        here but evicting a rank from the *middle* after a fault does not —
        that case stays masked-and-degraded instead.
        """
        if bootstrap.world_size > self._capacity:
            raise MoEEpConfigError(
                f"nixl_ep: cannot grow to world_size {bootstrap.world_size} on a "
                f"Fleet whose topology capacity is {self._capacity}; the "
                "transport's per-rank buffers were sized to that capacity. "
                "Pass FleetAlgoKnobTopologyCapacity(n=<max ranks>) at "
                "construction."
            )
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
        self._ft_applied = [ACTIVE] * bootstrap.world_size
        self._hot_ft_bufs = None

    # ------------------------------------------------------- fault tolerance

    def _require_ft(self, op: str):
        if self._ft is None:
            self._no_fault_tolerance(op)
        return self._ft

    @staticmethod
    def _reject_graph_capture(op: str) -> None:
        _reject_graph_capture_impl(op)

    def _ft_raw(self):
        """`[capacity]` int32 scratch for query_mask_buffer.

        The transport asserts the out tensor is exactly ``max_num_ranks``
        long, which is the topology CAPACITY, not the live world size — hence
        the trim in query_active_mask.
        """
        import torch

        buf = getattr(self, "_hot_ft_bufs", None)
        if buf is None or buf.numel() != self._capacity:
            buf = torch.empty(self._capacity, dtype=torch.int32, device="cuda")
            self._hot_ft_bufs = buf
        return buf

    def _normalize_mask(self, mask) -> list[int]:
        import torch

        if isinstance(mask, torch.Tensor):
            values = [int(v) for v in mask.detach().cpu().flatten().tolist()]
        else:
            values = [int(v) for v in mask]
        n = self._bootstrap.world_size
        if len(values) != n:
            raise ValueError(f"mask has {len(values)} entries, expected {n}")
        out = [ACTIVE if v == ACTIVE else MASKED for v in values]
        if out[self._bootstrap.rank] != ACTIVE:
            raise ValueError(
                f"rank {self._bootstrap.rank} cannot mask itself "
                "(mask[rank] must be 1); nixl_ep rejects masking the local rank"
            )
        return out

    def _ft_store(self):
        if self._ft_store_obj is None:
            self._ft_store_obj = resolve_rendezvous_store(
                self._bootstrap, subsystem="ft"
            )
        return self._ft_store_obj

    def _ft_knob(self):
        return self._ft

    @property
    def supports_fault_tolerance(self) -> bool:
        return self._ft is not None

    @property
    def active_mask_epoch(self) -> int:
        return self._ft_epoch

    def query_active_mask(self, out=None):
        import torch

        self._require_ft("query_active_mask")
        self._reject_graph_capture("query_active_mask")
        raw = self._ft_raw()
        self._buffer.query_mask_buffer(raw)  # asserts numel == capacity
        ws = self._bootstrap.world_size
        # NIXL's convention is "NONZERO means masked", not "1 means masked":
        # the buffer is 0xFF-memset at allocation (so an untouched entry reads
        # back as -1) and the kernels test `mask_buffer[r] != 0`. `== 0` is
        # therefore the only correct normalization to our 1 = active — a
        # `1 - raw` inversion would yield 2 for never-connected capacity-tail
        # ranks and silently poison every downstream sum()/bool().
        active = (raw[:ws] == 0).to(torch.int32)
        return active if out is None else out.copy_(active)

    def query_fault(self) -> bool:
        self._require_ft("query_fault")
        # NIXL has no host-side error flag (nccl_ep's pinned flag has no
        # equivalent here), so diff against what we last applied. Costs one
        # small D2H sync; documented on the ABC.
        cur = self.query_active_mask()
        return cur.cpu().tolist() != list(self._ft_applied)

    def set_active_mask(self, mask) -> None:
        self._require_ft("set_active_mask")
        self._reject_graph_capture("set_active_mask")
        want = self._normalize_mask(mask)
        # Push only the DIFF: unlike nccl_ep's single vector Update, each
        # nixl update_mask_buffer is a kernel launch, so a blind
        # range(world_size) loop would inject world_size launches into the
        # steady state where nothing changed.
        for r, (a, b) in enumerate(zip(want, self._ft_applied, strict=True)):
            if a != b and r != self._bootstrap.rank:
                self._buffer.update_mask_buffer(r, mask=(a == MASKED))
        self._ft_applied = list(want)
        self._ft_epoch += 1

    def clear_faults(self, *, readmit: bool = False) -> None:
        self._require_ft("clear_faults")
        if not readmit:
            # No sticky host error flag to re-arm on this transport; the mask
            # itself is the state, and query_fault diffs against it.
            return
        self._reject_graph_capture("clear_faults")
        ws = self._bootstrap.world_size
        # clean_mask_buffer zeroes ALL `capacity` entries, which marks the
        # never-connected tail [world_size, capacity) as ACTIVE — a real
        # correctness bug on any fleet sized above its live world. Re-mask it
        # immediately.
        self._buffer.clean_mask_buffer()
        self._ft_applied = [ACTIVE] * ws
        for r in range(ws, self._capacity):
            self._buffer.update_mask_buffer(r, mask=True)
        self._ft_epoch += 1

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
