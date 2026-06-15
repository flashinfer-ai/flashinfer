"""NcclEpFleet — NCCL-EP-backed Fleet implementation."""

from __future__ import annotations

import contextlib
import ctypes
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
from ..config import EpAlgorithm, FleetParams, QuantType
from ..fleet import Fleet, _BACKEND_REGISTRY

# from ...api_logging import flashinfer_api  # disabled per PR #3453 review
from .ndtensor import get_nccl_lib

if TYPE_CHECKING:
    from ..config import BootstrapConfig
    from ..handle import Handle


# Matches NCCL_EP_AUTO from contrib/nccl_ep/include/nccl_ep.h.
NCCL_EP_AUTO = 0


def _resolve_nccl_comm(bootstrap: "BootstrapConfig") -> int:
    """Return an int ncclComm_t handle.

    Prefers an explicit `bootstrap.nccl_comm` int. Falls back to
    ``nccl_ep.get_nccl_comm_from_group(None, nccl_lib=...)`` which creates a
    new NCCL communicator over the default torch.distributed PG via
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
    # upstream get_nccl_comm_from_group() requires the NCCLLibrary instance
    # so it can call ncclGetUniqueId + ncclCommInitRank.
    comm = get_nccl_comm_from_group(group=None, nccl_lib=get_nccl_lib())
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
        # Cache the NCCL library handle so later calls (+ __del__) don't
        # re-resolve it; avoids interpreter-shutdown lookups.
        self._lib = get_nccl_lib()
        self._comm = _resolve_nccl_comm(bootstrap)

        self._cfg = self._build_group_config()
        self._group = self._lib.ncclEpCreateGroup(
            ctypes.c_void_p(self._comm),
            self._cfg,
            ctypes.c_void_p(self._stream),
        )
        self._destroyed = False

    def _knob_or_auto(self, knob_cls: type, field: str) -> int:
        k = self._fleet_knobs.get(knob_cls)
        return int(getattr(k, field)) if k is not None else NCCL_EP_AUTO

    def _build_group_config(self):
        """Build ncclEpGroupConfig_t from ``self._params`` + ``self._fleet_knobs``.

        Used by both __init__ and update_topology so that config-affecting
        knobs (rdma_buffer_size / num_qp_per_rank / num_channels) picked up
        from a refreshed knob set actually take effect on the rebuilt group.
        """
        from nccl_ep import ncclEpGroupConfig_t  # type: ignore[import-not-found]

        p = self._params
        return ncclEpGroupConfig_t(
            version=1,  # ncclEpCreateGroup asserts version == 1
            algorithm=_map_algorithm(p.algorithm),
            num_experts=p.num_experts,
            max_tokens_per_rank=p.max_tokens_per_rank,
            token_size_bytes=p.token_hidden_size * p.dtype_bytes,
            rdma_buffer_size=self._knob_or_auto(FleetAlgoKnobRdmaBufferSize, "bytes_"),
            num_qp_per_rank=self._knob_or_auto(FleetAlgoKnobNumQpsPerRank, "n"),
            num_channels=self._knob_or_auto(FleetAlgoKnobNumChannelsPerRank, "n"),
        )

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
        """Re-create the EP group over a new ncclComm.

        For grow/shrink, the caller is expected to provide a new bootstrap
        whose ``nccl_comm`` points at the post-ncclCommSplit communicator.
        We blow away the old group and rebuild over the new comm.
        """
        if not self._destroyed:
            self._lib.ncclEpGroupDestroy(self._group, ctypes.c_void_p(self._stream))
        self._fleet_knobs = (
            _index_knobs(algo_knobs) if algo_knobs else self._fleet_knobs
        )
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._comm = _resolve_nccl_comm(bootstrap)
        # Rebuild cfg from the (possibly refreshed) knobs so config-affecting
        # knobs (rdma_buffer_size / num_qp_per_rank / num_channels) take effect
        # rather than silently reusing the stale struct from __init__.
        self._cfg = self._build_group_config()
        self._group = self._lib.ncclEpCreateGroup(
            ctypes.c_void_p(self._comm),
            self._cfg,
            ctypes.c_void_p(self._stream),
        )
        self._destroyed = False

    # @flashinfer_api  # disabled per PR #3453 review
    def destroy(self) -> None:
        if self._destroyed:
            return
        self._lib.ncclEpGroupDestroy(self._group, ctypes.c_void_p(self._stream))
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

    @property
    def bootstrap(self) -> "BootstrapConfig":
        return self._bootstrap


# Module-load side effect: register the backend.
_BACKEND_REGISTRY["nccl_ep"] = NcclEpFleet
