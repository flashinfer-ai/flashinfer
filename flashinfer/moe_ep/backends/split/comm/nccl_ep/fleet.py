"""NcclEpFleet — NCCL-EP-backed Fleet implementation.

Uses the ``nccl-ep-v0.1.0`` Pythonic API (``nccl.ep``): builds an
``nccl.core.Communicator`` and an ``nccl.ep.Group``, and hands the live
``Group`` to each :class:`NcclEpHandle`.
"""

from __future__ import annotations

import contextlib
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
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobNumQpsPerRank,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobRdmaBufferSize,
    _index_knobs,
)
from .....config import EpAlgorithm, FleetParams
from .....core.comm.fleet import Fleet, _BACKEND_REGISTRY

if TYPE_CHECKING:
    from .....config import BootstrapConfig
    from .....core.comm.handle import Handle


NCCL_EP_AUTO = 0


def _import_nccl_ep():
    """Import the ``nccl.ep`` package or raise an actionable build error."""
    try:
        import nccl.ep as nccl_ep  # type: ignore[import-not-found]

        return nccl_ep
    except ImportError as e:  # pragma: no cover
        raise MoEEpNotBuiltError(
            "nccl.ep (nccl-ep-v0.1.0) python package unavailable. Rebuild with "
            "BUILD_NCCL_EP=1 (which builds the nccl4py bindings), or install the "
            "nccl4py wheel that ships nccl.ep."
        ) from e


def _resolve_comm(bootstrap: "BootstrapConfig"):
    """Return an ``nccl.core.Communicator`` for the EP process group."""
    from nccl.ep.interop.torch import (  # type: ignore[import-not-found]
        get_nccl_comm_from_group,
    )

    from .....core.bootstrap_utils import bootstrap_comm_group

    return get_nccl_comm_from_group(group=bootstrap_comm_group(bootstrap))


def _map_algorithm(algo: EpAlgorithm):
    from nccl.ep import Algorithm  # type: ignore[import-not-found]

    return {
        EpAlgorithm.LOW_LATENCY: Algorithm.LOW_LATENCY,
        EpAlgorithm.HIGH_THROUGHPUT: Algorithm.HIGH_THROUGHPUT,
    }[algo]


class NcclEpFleet(Fleet):
    """Owns the ``nccl.ep.Group`` lifecycle for one process."""

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: FleetParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        _require_built("nccl_ep")
        validate_arch_for_backend("nccl_ep")
        validate_bootstrap_world_size(bootstrap)

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
        self._comm = _resolve_comm(bootstrap)

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
        if p.algorithm == EpAlgorithm.HIGH_THROUGHPUT:
            world = self._bootstrap.world_size
            kwargs["max_recv_tokens_per_rank"] = p.max_tokens_per_rank * world
        return self._nccl_ep.GroupConfig(**kwargs)

    def create_handle(
        self,
        params,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> "Handle":
        from .handle import NcclEpHandle

        return NcclEpHandle(self, params, algo_knobs)

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


_BACKEND_REGISTRY["nccl_ep"] = NcclEpFleet
