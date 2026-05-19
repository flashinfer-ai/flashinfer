"""NcclEpFleet — NCCL-EP-backed Fleet implementation."""

from __future__ import annotations

import contextlib
import ctypes
from typing import TYPE_CHECKING, Sequence

from .. import MoEEpNotBuiltError, _require_built
from ..algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobNumQpsPerRank,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobRdmaBufferSize,
    _index_knobs,
)
from ..config import EpAlgorithm, FleetParams, QuantType
from ..fleet import Fleet, _BACKEND_REGISTRY
from .ndtensor import get_nccl_lib

if TYPE_CHECKING:
    from ..config import BootstrapConfig
    from ..handle import Handle


# Matches NCCL_EP_AUTO from contrib/nccl_ep/include/nccl_ep.h.
NCCL_EP_AUTO = 0


def _resolve_nccl_comm(bootstrap: "BootstrapConfig") -> int:
    """Return an int ncclComm_t handle.

    Prefers an explicit `bootstrap.nccl_comm` int. Falls back to
    ``nccl_ep.get_nccl_comm_from_group(None)`` which creates a new NCCL
    communicator over the default torch.distributed PG via
    ncclGetUniqueId + ncclCommInitRank — robust across torch versions.
    """
    if bootstrap.nccl_comm is not None:
        return int(bootstrap.nccl_comm)
    try:
        from nccl_ep import get_nccl_comm_from_group  # type: ignore[import-not-found]
    except ImportError as e:
        raise MoEEpNotBuiltError(
            "nccl_ep python bindings unavailable; install via "
            "pip install -e 3rdparty/nccl/contrib/nccl_ep/python"
        ) from e
    comm = get_nccl_comm_from_group(group=None)
    # comm is a ctypes pointer; coerce to int.
    return int(ctypes.cast(comm, ctypes.c_void_p).value or 0)


def _map_algorithm(algo: EpAlgorithm) -> int:
    from nccl_ep import (  # type: ignore[import-not-found]
        NCCL_EP_ALGO_HIGH_THROUGHPUT,
        NCCL_EP_ALGO_LOW_LATENCY,
    )

    return {
        EpAlgorithm.LOW_LATENCY: NCCL_EP_ALGO_LOW_LATENCY,
        EpAlgorithm.HIGH_THROUGHPUT: NCCL_EP_ALGO_HIGH_THROUGHPUT,
    }[algo]


class NcclEpFleet(Fleet):
    """Owns the ncclEpGroup_t lifecycle for one process."""

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: FleetParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        _require_built("nccl_ep")

        self._params = params
        self._fleet_knobs = _index_knobs(algo_knobs)
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._comm = _resolve_nccl_comm(bootstrap)

        lib = get_nccl_lib()
        from nccl_ep import ncclEpGroupConfig_t  # type: ignore[import-not-found]

        cfg = ncclEpGroupConfig_t(
            version=0x10000,
            algorithm=_map_algorithm(params.algorithm),
            num_experts=params.num_experts,
            max_tokens_per_rank=params.max_tokens_per_rank,
            token_size_bytes=params.token_hidden_size * params.dtype_bytes,
            rdma_buffer_size=self._knob_or_auto(FleetAlgoKnobRdmaBufferSize, "bytes_"),
            num_qp_per_rank=self._knob_or_auto(FleetAlgoKnobNumQpsPerRank, "n"),
            num_channels=self._knob_or_auto(FleetAlgoKnobNumChannelsPerRank, "n"),
        )
        self._cfg = cfg
        self._group = lib.ncclEpCreateGroup(
            ctypes.c_void_p(self._comm),
            cfg,
            ctypes.c_void_p(self._stream),
        )
        self._destroyed = False

    def _knob_or_auto(self, knob_cls: type, field: str) -> int:
        k = self._fleet_knobs.get(knob_cls)
        return int(getattr(k, field)) if k is not None else NCCL_EP_AUTO

    @property
    def use_fp8(self) -> bool:
        q = self._fleet_knobs.get(FleetAlgoKnobQuantization)
        return bool(q and QuantType.FP8E4M3 in q.quants)  # type: ignore[attr-defined]

    @property
    def use_ue8m0(self) -> bool:
        q = self._fleet_knobs.get(FleetAlgoKnobQuantization)
        return bool(q and QuantType.UE8M0 in q.quants)  # type: ignore[attr-defined]

    def create_handle(
        self,
        params,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> "Handle":
        from .handle import NcclEpHandle  # local import to break cycle

        return NcclEpHandle(self, params, algo_knobs)

    def update_topology(
        self,
        bootstrap: "BootstrapConfig",
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        """Re-create the EP group over a new ncclComm.

        For grow/shrink, the caller is expected to provide a new bootstrap
        whose ``nccl_comm`` points at the post-ncclCommSplit communicator.
        We blow away the old group and rebuild over the new comm.
        """
        lib = get_nccl_lib()
        if not self._destroyed:
            lib.ncclEpGroupDestroy(self._group, ctypes.c_void_p(self._stream))
        self._fleet_knobs = (
            _index_knobs(algo_knobs) if algo_knobs else self._fleet_knobs
        )
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._comm = _resolve_nccl_comm(bootstrap)
        # Re-issue group create with the same cfg struct.
        self._group = lib.ncclEpCreateGroup(
            ctypes.c_void_p(self._comm),
            self._cfg,
            ctypes.c_void_p(self._stream),
        )
        self._destroyed = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        lib = get_nccl_lib()
        lib.ncclEpGroupDestroy(self._group, ctypes.c_void_p(self._stream))
        self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()

    @property
    def group(self) -> ctypes.c_void_p:
        return self._group

    @property
    def stream(self) -> int:
        return self._stream

    @property
    def params(self) -> FleetParams:
        return self._params


# Module-load side effect: register the backend.
_BACKEND_REGISTRY["nccl_ep"] = NcclEpFleet
