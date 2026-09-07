"""NcclEpFleet — NCCL-EP-backed Fleet implementation.

Uses the ``nccl-ep-v0.1.0`` Pythonic API (``nccl.ep``): builds an
``nccl.core.Communicator`` and an ``nccl.ep.Group``, and hands the live
``Group`` to each :class:`NcclEpHandle`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import warnings
from typing import TYPE_CHECKING, Sequence

from ..... import _require_built
from .....errors import MoEEpFaultToleranceUnsupportedError, MoEEpNotBuiltError
from .....core.validation.common import (
    validate_arch_for_backend,
    validate_ll_hidden_size,
    validate_bootstrap_world_size,
    validate_fleet_params,
)
from .....algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobAllocator,
    FleetAlgoKnobFaultTolerance,
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobNumQpsPerRank,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobRdmaBufferSize,
    _index_knobs,
)
from .....config import EpAlgorithm, EpLayout, FleetParams
from .....core.bootstrap_utils import resolve_rendezvous_store
from .....core.comm.fleet import Fleet, _BACKEND_REGISTRY
from .....core.comm.fault_tolerance import (
    ACTIVE,
    MASKED,
    FaultToleranceMixin,
    reject_graph_capture as _reject_graph_capture_impl,
)
from ._mask_ffi import mask_ffi

if TYPE_CHECKING:
    from .....config import BootstrapConfig
    from .....core.comm.handle import Handle


logger = logging.getLogger(__name__)

NCCL_EP_AUTO = 0

# nccl_ep HT hard limit: ``ncclEpCreateGroup`` *asserts* (SIGABRT, nccl_ep.cc:1253)
# when a HIGH_THROUGHPUT group is created with
# ``max_dispatch_tokens_per_rank > MAX_SUPPORTED_TOKENS_PER_RANK``. The constant is
# a build-time template bound in the wheel
# (``nccl/ep/include/nccl_ep/common.hpp``: ``#define MAX_SUPPORTED_TOKENS_PER_RANK
# 8192``). We mirror it here to *clamp* the HT dispatch budget (graceful) rather
# than let a large caller value (e.g. vLLM ``max_num_batched_tokens``) hit the C++
# assert. LL has no such cap. Kept in sync with the nccl4py wheel.
_HT_MAX_SUPPORTED_TOKENS_PER_RANK = 8192


def _clamp_ht_max_tokens(params: FleetParams) -> FleetParams:
    """Clamp a HT fleet's ``max_tokens_per_rank`` to the nccl_ep build-time cap.

    HT's ``ncclEpCreateGroup`` aborts when ``max_dispatch_tokens_per_rank`` exceeds
    ``MAX_SUPPORTED_TOKENS_PER_RANK`` (8192). We return a clamped copy so group
    creation succeeds; a single forward that actually dispatches more than the cap
    per rank is caught with a clear error at dispatch (see ``NcclEpHandle._dispatch_ht``)
    rather than silently truncated. No-op for LL (unbounded) or when already within cap.
    """
    if (
        params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
        and params.max_tokens_per_rank > _HT_MAX_SUPPORTED_TOKENS_PER_RANK
    ):
        logger.warning(
            "nccl_ep HT caps max_dispatch_tokens_per_rank at %d "
            "(MAX_SUPPORTED_TOKENS_PER_RANK); requested %d — clamping to avoid the "
            "ncclEpCreateGroup abort. Ensure the per-forward token count per rank "
            "stays <= %d (e.g. vLLM --max-num-batched-tokens); a larger dispatch "
            "will raise at forward time.",
            _HT_MAX_SUPPORTED_TOKENS_PER_RANK,
            params.max_tokens_per_rank,
            _HT_MAX_SUPPORTED_TOKENS_PER_RANK,
        )
        return dataclasses.replace(
            params, max_tokens_per_rank=_HT_MAX_SUPPORTED_TOKENS_PER_RANK
        )
    return params


def _import_nccl_ep():
    """Import the ``nccl.ep`` package or raise an actionable build error."""
    try:
        import nccl.ep as nccl_ep  # type: ignore[import-not-found]

        return nccl_ep
    except ImportError as e:  # pragma: no cover
        raise MoEEpNotBuiltError(
            "nccl.ep (nccl-ep-v0.1.0) python package unavailable. It ships in "
            "the nccl4py wheel, a base dependency of flashinfer-python — "
            "install with `pip install 'nccl4py>=0.3.1'`."
        ) from e


def _resolve_comm(bootstrap: "BootstrapConfig"):
    """Return an ``nccl.core.Communicator`` for the EP Fleet.

    Resolution order (see :class:`BootstrapConfig`):

    1. ``bootstrap.nccl_comm`` set — *adopt* that existing ``ncclComm_t``.
       We wrap it with ``nccl.core.Communicator(ptr=...)``, which does NOT
       register a finalizer, and the Fleet only ever destroys the *group*
       (never the comm), so the adopted communicator's lifetime stays with
       its real owner (e.g. vLLM's process group). This lets a host share
       the exact communicator it already owns instead of paying for a second
       NCCL bootstrap.
    2. otherwise — mirror the torch group from ``bootstrap_comm_group``
       (``bootstrap.process_group`` when set, else the default WORLD group)
       by creating a fresh communicator over its membership.

    ``get_nccl_comm_from_group`` always *creates* a fresh communicator
    (vLLM's robust-across-torch-versions pattern); adoption (case 1) is the
    only path that reuses an existing one.
    """
    if bootstrap.nccl_comm is not None:
        from nccl.core import Communicator  # type: ignore[import-not-found]

        # Wrap-without-own: Communicator(ptr) has no __del__, and the Fleet
        # never calls .destroy()/.abort() on the comm — only on the group.
        return Communicator(ptr=int(bootstrap.nccl_comm))

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


class NcclEpFleet(FaultToleranceMixin, Fleet):
    """Owns the ``nccl.ep.Group`` lifecycle for one process."""

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: FleetParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        _require_built("nccl_ep")
        validate_arch_for_backend("nccl_ep")
        validate_ll_hidden_size(params, "nccl_ep")
        validate_bootstrap_world_size(bootstrap)

        # HT: clamp the per-rank dispatch budget to the library's build-time cap so
        # ncclEpCreateGroup doesn't abort; must clamp the stored params (not just the
        # GroupConfig) so the handle's recv-buffer sizing agrees.
        params = _clamp_ht_max_tokens(params)
        self._params = params
        self._fleet_knobs = _index_knobs(algo_knobs)
        ft = self._fleet_knobs.get(FleetAlgoKnobFaultTolerance)
        self._ft = ft if (ft is not None and ft.enabled) else None  # type: ignore[attr-defined]
        validate_fleet_params(
            params,
            backend="nccl_ep",
            world_size=bootstrap.world_size,
            quant=self._fleet_knobs.get(FleetAlgoKnobQuantization),  # type: ignore[arg-type]
            fault_tolerance=self._ft,  # type: ignore[arg-type]
        )
        self._bootstrap = bootstrap
        self._stream = bootstrap.stream
        self._nccl_ep = _import_nccl_ep()
        self._comm = _resolve_comm(bootstrap)
        self._ft_epoch = 0
        self._ft_store_obj = None
        self._any_handle_created = False
        if self._ft is not None:
            self._check_ft_supported()

        # Cross-handle host-path cache (recv buffers, counter tensors, FFI
        # descriptor memos), populated and consumed by NcclEpHandle. Anchored on
        # the Fleet because callers (e.g. vLLM) create a fresh Handle every
        # forward while the Fleet persists — per-handle caches never hit.
        self._hot_cache: dict = {}

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
        alloc = self._build_alloc_config()
        if alloc is not None:
            kwargs["alloc"] = alloc
        if self._ft is not None:
            # Allocates the per-rank mask buffer + the pinned async-error flag
            # inside ncclEpCreateGroup. Without it a peer timeout trap()s.
            kwargs["enable_mask"] = True
            if self._ft.timeout_ms:  # type: ignore[attr-defined]
                kwargs["timeout_ns"] = int(self._ft.timeout_ms) * 1_000_000  # type: ignore[attr-defined]
        return self._nccl_ep.GroupConfig(**kwargs)

    def _build_alloc_config(self):
        """Build an ``nccl.ep.AllocConfig`` from FleetAlgoKnobAllocator, or None.

        None → leave ``GroupConfig.alloc`` at its default (cudaMalloc/cudaFree).
        """
        knob = self._fleet_knobs.get(FleetAlgoKnobAllocator)
        if knob is None:
            return None
        if knob.torch_caching:  # type: ignore[attr-defined]
            alloc_addr, free_addr = self._install_torch_allocator()
            return self._nccl_ep.AllocConfig(alloc_fn=alloc_addr, free_fn=free_addr)
        # Explicit caller-owned addresses (may be 0 → default path).
        return self._nccl_ep.AllocConfig(
            alloc_fn=int(knob.alloc_fn),  # type: ignore[attr-defined]
            free_fn=int(knob.free_fn),  # type: ignore[attr-defined]
            context=int(knob.context),  # type: ignore[attr-defined]
        )

    def _install_torch_allocator(self):
        """Install alloc/free trampolines backed by torch's CUDA caching
        allocator and return their (alloc_addr, free_addr) C addresses.

        The trampolines are anchored on ``self`` (``_alloc_trampolines``) so
        they outlive the Group per :mod:`nccl.ep.allocator`'s lifetime rule —
        if GC'd while NCCL-EP still holds the pointer, the next C-side call
        lands in freed memory. NcclEpFleet outlives its Group, so this anchor
        is sufficient.
        """
        import ctypes

        import torch
        from nccl.ep import AllocFn, FreeFn  # type: ignore[import-not-found]

        _CUDA_SUCCESS = 0
        _CUDA_ERROR_MEMORY_ALLOCATION = 2

        @AllocFn
        def _alloc(out_ptr, size, context):  # cudaError_t (void**, size_t, void*)
            try:
                ptr = torch.cuda.caching_allocator_alloc(int(size))
                out_ptr[0] = ctypes.c_void_p(int(ptr))
                return _CUDA_SUCCESS
            except Exception:
                return _CUDA_ERROR_MEMORY_ALLOCATION

        @FreeFn
        def _free(ptr, context):  # cudaError_t (void*, void*)
            try:
                if ptr:
                    torch.cuda.caching_allocator_delete(int(ptr))
                return _CUDA_SUCCESS
            except Exception:
                return _CUDA_ERROR_MEMORY_ALLOCATION

        self._alloc_trampolines = (_alloc, _free)  # keepalive (lifetime rule)
        alloc_addr = ctypes.cast(_alloc, ctypes.c_void_p).value
        free_addr = ctypes.cast(_free, ctypes.c_void_p).value
        return alloc_addr, free_addr

    def create_handle(
        self,
        params,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> "Handle":
        from .handle import NcclEpHandle

        self._any_handle_created = True
        return NcclEpHandle(self, params, algo_knobs)

    # ------------------------------------------------------- fault tolerance

    def _check_ft_supported(self) -> None:
        """Fail at construction, not at first fault, when FT can't work here."""
        try:
            names = {f.name for f in dataclasses.fields(self._nccl_ep.GroupConfig)}
        except TypeError:
            # Not introspectable (e.g. the test fake); nothing to assert, and
            # an unknown kwarg would surface as a plain TypeError anyway.
            names = None
        if names is not None and "enable_mask" not in names:
            raise MoEEpFaultToleranceUnsupportedError(
                "nccl.ep.GroupConfig has no `enable_mask` field: the installed "
                "nccl4py predates EP fault tolerance. Upgrade the nccl4py wheel, "
                "or drop FleetAlgoKnobFaultTolerance."
            )
        ffi = mask_ffi()
        if not ffi.available:
            raise MoEEpFaultToleranceUnsupportedError(
                "the loaded libnccl_ep does not export "
                f"{', '.join(ffi.missing)}; fault tolerance needs an NCCL-EP "
                "build carrying the ncclEpMask* API. Check "
                "flashinfer.moe_ep.supports_fault_tolerance('nccl_ep') before "
                "passing FleetAlgoKnobFaultTolerance."
            )

    def _require_ft(self, op: str):
        if self._ft is None:
            self._no_fault_tolerance(op)
        return self._ft

    @staticmethod
    def _reject_graph_capture(op: str) -> None:
        _reject_graph_capture_impl(op)

    def _current_stream(self) -> int:
        if self._stream:
            return self._stream
        import torch

        return torch.cuda.current_stream().cuda_stream

    def _ft_bufs(self):
        """(device Query dst, pinned host Update src), Fleet-owned.

        ncclEpMaskQuery writes to DEVICE memory while ncclEpMaskUpdate reads
        from HOST memory, so the two directions need different staging. The
        host side is PINNED because Update is stream-ordered: if the library
        defers the H2D onto the stream, a pageable buffer we mutate on the
        next call is a use-after-write race. Anchored on _hot_cache, which
        update_topology already clears, so the buffers resize with the world.
        """
        import torch

        n = self._bootstrap.world_size
        bufs = self._hot_cache.get("ft_bufs")
        if bufs is None or bufs[0].numel() != n:
            bufs = (
                torch.empty(n, dtype=torch.int32, device="cuda"),
                torch.empty(n, dtype=torch.int32, pin_memory=True),
            )
            self._hot_cache["ft_bufs"] = bufs
        return bufs

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
                f"rank {self._bootstrap.rank} cannot mask itself (mask[rank] must be 1)"
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
        self._require_ft("query_active_mask")
        self._reject_graph_capture("query_active_mask")
        dev, _ = self._ft_bufs()
        # NCCL's polarity is already ours: 1 = active.
        mask_ffi().mask_query(self._group, dev.data_ptr(), self._current_stream())
        return dev if out is None else out.copy_(dev)

    def query_fault(self) -> bool:
        self._require_ft("query_fault")
        # Pinned host flag: no stream, no sync, free to poll every step.
        #
        # Guarded against capture precisely BECAUSE it is a host read: it is
        # not stream work, so it cannot be captured at all. Called inside a
        # capture region it would return the capture-time value, and the
        # branch taken on it would be baked into the graph's structure
        # forever -- a graph that ignores faults because none existed when it
        # was recorded. That is a quieter failure than the stream-ordered
        # calls below, which is exactly why it needs the guard.
        self._reject_graph_capture("query_fault")
        return mask_ffi().get_async_error(self._group)

    def set_active_mask(self, mask) -> None:
        self._require_ft("set_active_mask")
        self._reject_graph_capture("set_active_mask")
        _, host = self._ft_bufs()
        import torch

        host.copy_(torch.tensor(self._normalize_mask(mask), dtype=torch.int32))
        mask_ffi().mask_update(self._group, host.data_ptr(), self._current_stream())
        self._ft_epoch += 1

    def clear_faults(self, *, readmit: bool = False) -> None:
        self._require_ft("clear_faults")
        if readmit:
            self._reject_graph_capture("clear_faults")
            if not self._any_handle_created:
                # ncclEpMaskClean asserts rdma_buffer != nullptr, which is only
                # allocated by the first handle; calling it earlier aborts the
                # process from C. Fail in Python instead.
                raise RuntimeError(
                    "clear_faults(readmit=True) requires at least one handle to "
                    "have been created on this Fleet (ncclEpMaskClean asserts on "
                    "the LL staging buffer, which the first create_handle "
                    "allocates). Run a forward first, or use readmit=False."
                )
            if self._params.layout is EpLayout.EXPERT_MAJOR:
                warnings.warn(
                    "nccl_ep: ncclEpMaskClean computes its buffer-reset offsets "
                    "assuming the RANK_MAJOR layout, but this Fleet is "
                    "EXPERT_MAJOR, so re-admission may leave stale bytes in the "
                    "LL staging buffer. Prefer degraded serving "
                    "(readmit=False), or rebuild the group via "
                    "update_topology().",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # COLLECTIVE over survivors; internally cudaStreamSynchronize's.
            mask_ffi().mask_clean(self._group, self._current_stream())
            self._ft_epoch += 1
        # MaskClean deliberately does NOT clear the async flag, so always pair
        # them: that is why readmit is a flag rather than a separate method a
        # caller could mis-sequence.
        mask_ffi().error_clear(self._group)

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
        # Topology (world size) changed — drop the cross-handle host caches so
        # recv buffers / counters / FFI descriptors are rebuilt at the new sizes.
        self._hot_cache.clear()
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
