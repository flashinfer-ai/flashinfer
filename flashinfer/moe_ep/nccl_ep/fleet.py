"""NcclEpFleet — NCCL-EP-backed Fleet implementation.

Rewritten for the ``nccl-ep-v0.1.0`` Pythonic API (``nccl.ep``): the legacy
flat ``nccl_ep`` ctypes module and the in-tree ``libnccl_ep.so`` dlopen path
are gone.  We now build an ``nccl.core.Communicator`` (via
``nccl.ep.interop.torch.get_nccl_comm_from_group``) and an ``nccl.ep.Group``
(``Group.create(comm, GroupConfig(...))``), and hand the live ``Group`` to each
:class:`NcclEpHandle`.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Sequence

from .. import MoEEpNotBuiltError, _require_built
from .._validators import validate_arch_for_backend, validate_fleet_params
from ..algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobNumQpsPerRank,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobRdmaBufferSize,
    _index_knobs,
)
from ..config import EpAlgorithm, FleetParams
from ..fleet import Fleet, _BACKEND_REGISTRY

# from ...api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from ..config import BootstrapConfig
    from ..handle import Handle


# ``GroupConfig`` fields left at 0 forward as NCCL_EP_AUTO.
NCCL_EP_AUTO = 0


def _import_nccl_ep():
    """Import the ``nccl.ep`` package or raise an actionable build error."""
    try:
        import nccl.ep as nccl_ep  # type: ignore[import-not-found]

        return nccl_ep
    except ImportError as e:  # pragma: no cover - exercised only without build
        raise MoEEpNotBuiltError(
            "nccl.ep (nccl-ep-v0.1.0) python package unavailable. Rebuild with "
            "BUILD_NCCL_EP=1 (which builds the nccl4py bindings), or install the "
            "nccl4py wheel that ships nccl.ep."
        ) from e


def _resolve_comm(bootstrap: "BootstrapConfig"):
    """Return an ``nccl.core.Communicator`` mirroring the default torch PG.

    ``nccl.ep.interop.torch.get_nccl_comm_from_group`` always creates a fresh
    communicator (vLLM's robust-across-torch-versions pattern), so we ignore the
    legacy ``bootstrap.nccl_comm`` int.
    """
    from nccl.ep.interop.torch import (  # type: ignore[import-not-found]
        get_nccl_comm_from_group,
    )

    return get_nccl_comm_from_group(group=None)


def _map_algorithm(algo: EpAlgorithm):
    from nccl.ep import Algorithm  # type: ignore[import-not-found]

    return {
        EpAlgorithm.LOW_LATENCY: Algorithm.LOW_LATENCY,
        EpAlgorithm.HIGH_THROUGHPUT: Algorithm.HIGH_THROUGHPUT,
    }[algo]


class NcclEpFleet(Fleet):
    """Owns the ``nccl.ep.Group`` lifecycle for one process."""

    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: FleetParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        _require_built("nccl_ep")
        validate_arch_for_backend("nccl_ep")

        self._params = params
        self._fleet_knobs = _index_knobs(algo_knobs)
        validate_fleet_params(
            params,
            backend="nccl_ep",
            world_size=bootstrap.world_size,
            quant=self._fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
        )
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._nccl_ep = _import_nccl_ep()
        self._comm = _resolve_comm(bootstrap)  # keepalive: Group borrows it

        self._group = self._nccl_ep.Group.create(self._comm, self._build_group_config())
        self._destroyed = False

    def _knob_or_auto(self, knob_cls: type, field: str) -> int:
        k = self._fleet_knobs.get(knob_cls)
        return int(getattr(k, field)) if k is not None else NCCL_EP_AUTO

    def _build_group_config(self):
        """Build :class:`nccl.ep.GroupConfig` from params + fleet knobs."""
        p = self._params
        kwargs = dict(
            algorithm=_map_algorithm(p.algorithm),
            num_experts=p.num_experts,
            max_dispatch_tokens_per_rank=p.max_tokens_per_rank,
            max_token_bytes=p.token_hidden_size * p.dtype_bytes,
            rdma_buffer_size=self._knob_or_auto(FleetAlgoKnobRdmaBufferSize, "bytes_"),
            num_qp_per_rank=self._knob_or_auto(FleetAlgoKnobNumQpsPerRank, "n"),
            num_channels=self._knob_or_auto(FleetAlgoKnobNumChannelsPerRank, "n"),
        )
        # HT mode requires an explicit per-rank recv budget (> 0 and
        # >= max_dispatch_tokens_per_rank); LL auto-derives it from 0. Match
        # contrib/nccl_ep/ep_test.py (max_recv_tokens_per_rank = num_tokens *
        # n_ranks): the per-rank recv-slot budget is max_dispatch_tokens_per_rank
        # * world_size. The library sizes its internal HT staging buffers to this
        # value, so NcclEpHandle._dispatch_ht MUST size the dispatch output buffer
        # to exactly the same row count (max_tokens_per_rank * world_size) — see
        # the note there. (This is an upper bound on the uniform recv estimate
        # max_tokens_per_rank * top_k for top_k <= world.)
        if p.algorithm == EpAlgorithm.HIGH_THROUGHPUT:
            world = self._bootstrap.world_size
            kwargs["max_recv_tokens_per_rank"] = p.max_tokens_per_rank * world
        return self._nccl_ep.GroupConfig(**kwargs)

    # @flashinfer_api  # disabled per PR #3453 review
    def create_handle(
        self,
        params,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> "Handle":
        from .handle import NcclEpHandle  # local import to break cycle

        return NcclEpHandle(self, params, algo_knobs)

    # @flashinfer_api  # disabled per PR #3453 review
    def update_topology(
        self,
        bootstrap: "BootstrapConfig",
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        """Re-create the EP group over a fresh communicator (grow/shrink)."""
        if not self._destroyed:
            self._group.destroy()
        self._fleet_knobs = (
            _index_knobs(algo_knobs) if algo_knobs else self._fleet_knobs
        )
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._comm = _resolve_comm(bootstrap)
        self._group = self._nccl_ep.Group.create(self._comm, self._build_group_config())
        self._destroyed = False

    # @flashinfer_api  # disabled per PR #3453 review
    def destroy(self) -> None:
        if self._destroyed:
            return
        self._group.destroy()
        self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()

    @property
    def group(self):
        """The live :class:`nccl.ep.Group`."""
        return self._group

    @property
    def nccl_ep(self):
        """The imported ``nccl.ep`` module (Tensor / config dataclasses)."""
        return self._nccl_ep

    @property
    def stream(self) -> int:
        return self._stream

    @property
    def params(self) -> FleetParams:
        return self._params

    @property
    def bootstrap(self) -> "BootstrapConfig":
        return self._bootstrap


# Module-load side effect: register the backend.
_BACKEND_REGISTRY["nccl_ep"] = NcclEpFleet
