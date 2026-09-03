"""
Copyright (c) 2025-2026 by FlashInfer team.

The raw all-to-all entry points (merged from the former ulysses_a2a.py) wrap
a CUDA kernel adapted from ThunderKittens' NVLink all-to-all:
https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
import ctypes
import functools
import re
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..api_logging import flashinfer_api
from ..jit.comm import gen_ulysses_a2a_module, gen_ulysses_pcie_module
from ..trace.templates.comm import (
    ulysses_exchange_chunks_trace,
    ulysses_gather_heads_trace,
    ulysses_scatter_heads_trace,
)
from ..utils import register_custom_op
from .ulysses_topology import (
    PCIE_AUTO_RDMA_WORLD_SIZES,
    SUPPORTED_WORLD_SIZES,
    UlyssesBackendDecision,
    resolve_ulysses_backend,
)

_INT32_MAX = 2**31 - 1
# Loosest sound bound on the declared capacity. The kernels index elements with
# int32, which bounds an operand's element count, not its byte count -- and the
# byte count is what a communicator declares, because the element type is now a
# property of the call. Four is the widest supported element, so this is the
# largest capacity that could still be spent by a legal operand; the binding
# check is per operand in _validate.
_MAX_CAPACITY_BYTES = _INT32_MAX * 4
_PCIE_MLX5_MAX_INTERLEAVED_STRIDE = 65_535
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The PCIe transport moves bytes (copy engines and RDMA writes, no arithmetic),
# so it takes any 1-, 2- or 4-byte element type torch can hand through DLPack.
_PCIE_SUPPORTED_DTYPES = _SUPPORTED_DTYPES + (
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)
_PCIE_RDMA_TRANSPORTS = ("hybrid", "rdma")

# Communicator lifecycle states. BROKEN permits only a coordinated close: a
# PCIe transport error may leave QPs, CQs and registered buffers half-mutated.
# CLOSED is reached only after a fully successful teardown, so a failed
# close() can be retried.
_OPEN, _BROKEN, _CLOSING, _CLOSED = "open", "broken", "closing", "closed"


class UlyssesCommunicator:
    r"""Ulysses context-parallelism all-to-all communicator.

    Provides the two layout transforms of Ulysses attention over the 4-D
    layout ``[B, S, H, D]`` (a typical attention layer makes four collective
    calls: q/k/v through :meth:`scatter_heads`, the output through
    :meth:`gather_heads`):

    - :meth:`scatter_heads`: ``[B, S_local, H, D] -> [B, S_global, H_local, D]``
      (each rank keeps a head slice of the *full* sequence)
    - :meth:`gather_heads`:  ``[B, S_global, H_local, D] -> [B, S_local, H, D]``
      (each rank gets all heads of its *local* sequence shard back)

    where ``H`` is the global head count, ``H_local = H // world_size`` and
    ``S_global = S_local * world_size``. All backends produce bit-identical
    results.

    Backend selection happens in the constructor, strictly before any IPC
    allocation or JIT compilation (see
    :func:`~flashinfer.comm.resolve_ulysses_backend`):

    - ``backend="auto"``: the fused-transpose NVLink-P2P kernel when the group
      is a verified single-node all-pairs NVLink mesh with a supported world
      size (2/4/6/8); NCCL otherwise — including when NVLink runtime
      initialization fails after a positive topology decision. Inspect
      :attr:`backend` and :attr:`fallback_reason` for the outcome.
    - ``backend="nvlink"``: force the fused kernel; raises on every rank
      (before any IPC/JIT for topology failures) when it cannot be used.
    - ``backend="pcie"``: explicitly enable the experimental single-node PCIe
      transport at world size 1/2/4/8. One rank is an identity path, two
      ranks use CUDA P2P, and four/eight ranks prefer an all-RDMA route
      (every peer's payload over the rank's mlx5 QP) with all-P2P fallback;
      ``FLASHINFER_ULYSSES_PCIE_ROUTE`` (``auto``/``p2p``/``rdma``/``hybrid``)
      forces all-P2P, all-RDMA at any multi-rank world size, or the
      eight-rank 4+4 NUMA hybrid (same-NUMA CUDA P2P plus cross-NUMA mlx5).
      It uses explicitly allocated outputs registered at ``max_bytes``, and
      is never selected by ``"auto"``.
    - ``backend="nccl"``: force the ``dist.all_to_all_single`` path; skips
      the topology/NVML probe and all IPC/JIT entirely (the constructor
      still resolves and guards the CUDA device and performs CUDA-backed
      metadata collectives over ``group``). Supports any world size.

    All ranks must request the same ``backend``. The NCCL path with
    ``world_size > 1`` requires ``group`` to support CUDA all-to-all (an
    NCCL process group); this is checked at construction.
    ``world_size == 1`` is a passthrough: both collectives return the input
    tensor unchanged (no copy).

    Constraints
    -----------
    - The constructor is always collective: every rank of ``group`` must
      call it together. :meth:`close` is collective when the NVLink or PCIe
      backend was armed (their resources are peer-shared); for the pure NCCL
      backend, ``world_size == 1``, or an auto fallback whose NVLink cleanup
      already completed, ``close`` is local and idempotent. Rank-local
      failures inside the constructor's NVLink initialization or inside a
      collective ``close`` are exchanged as group outcomes, so all ranks
      jointly clean up and raise (or fall back) instead of deadlocking; a
      failed ``close`` may be retried by all ranks.
    - Collectives run on the *current* CUDA stream of this rank; every rank
      must issue the same sequence of calls with consistently-shaped operands
      (a shape or call-order mismatch across ranks is a collective failure:
      expect hangs or garbage, exactly as with any collective library). At
      most one collective may be in flight per communicator at a time (the
      NVLink and PCIe signal protocols assume serialized calls); do not call one
      communicator concurrently from multiple streams or threads.
      The experimental PCIe P2P route enqueues asynchronously and is CUDA
      Graph capturable when every output comes from :meth:`allocate_output`
      (which is itself collective and not capturable); the hybrid and
      all-RDMA routes block on the host and refuse capture. PCIe collectives
      are bound to the stream of their first call.
    - Operand tensors must be contiguous 4-D CUDA tensors of the construction
      ``dtype`` (float16 / bfloat16 / float32) on the construction device,
      with every dim positive and ``nbytes`` at most ``max_bytes``;
      :meth:`scatter_heads` additionally requires ``H % world_size == 0`` and
      :meth:`gather_heads` requires ``S_global % world_size == 0``.
      PCIe additionally accepts int8/uint8 and the float8 storage types
      (any 1-, 2- or 4-byte element); its mlx5 routes (hybrid and all-RDMA)
      additionally require batch size 1 and ``H * D * element_size <=
      65_535`` bytes, while the all-P2P routes take any batch size.
    - Multi-rank PCIe calls require an explicit ``out`` returned by
      :meth:`allocate_output`. This keeps registration lifetime and overwrite
      points explicit; pre-register one output per live result and geometry.
    - A PCIe transport failure is not recoverable: registered buffers, queue
      pairs and the GPU epoch counters may already be half-mutated, so the
      communicator enters a BROKEN state that rejects further collectives and
      permits only a collective :meth:`close`.
    - Each rank may use a different CUDA device (e.g. ``cuda:rank``); ranks
      must agree on ``max_bytes``, ``dtype`` and ``backend``.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        Process group of the Ulysses ranks. Defaults to ``dist.group.WORLD``.
    max_bytes : int
        Capacity: the size in bytes of the largest single all-to-all operand
        (input and output have equal ``nbytes``, so this is
        ``B*S_local*H*D*element_size`` for the largest call). Bytes rather than
        elements because ``backend="pcie"`` lets each call name its own element
        type, so a fixed capacity has to be denominated in something they share.
        Sizes the NVLink staging buffer once at construction.
    dtype : torch.dtype
        Default element type of operands (float16 / bfloat16 / float32; PCIe
        additionally int8 / uint8 / float8_e4m3fn / float8_e5m2); enforced on
        every call that does not name its own.
    backend : str
        ``"auto"`` | ``"nvlink"`` | ``"pcie"`` | ``"nccl"`` (see above).
    device : torch.device or str or int, optional
        CUDA device of this rank; normalized to an explicit index (bare
        ``"cuda"`` means the current device, an int is a CUDA ordinal).
        Defaults to the current CUDA device.

    Examples
    --------
    >>> with UlyssesCommunicator(group, max_bytes=q.nbytes, dtype=torch.bfloat16) as comm:
    ...     q_ = comm.scatter_heads(q)   # [B,S_local,H,D] -> [B,S_global,H_local,D]
    ...     ...
    ...     o = comm.gather_heads(o_)    # [B,S_global,H_local,D] -> [B,S_local,H,D]
    """

    @flashinfer_api
    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        *,
        max_bytes: int,
        dtype: torch.dtype,
        backend: str = "auto",
        device: Optional[Union[torch.device, str, int]] = None,
    ):
        r"""Construct a Ulysses communicator.

        Parameters
        ----------
        group : Optional[ProcessGroup], optional
            Process group spanning the participating ranks. ``None`` uses
            ``torch.distributed.group.WORLD``.
        max_bytes : int
            Per-rank upper bound on the size in bytes of a single collective
            operand. Used to size the backend workspace.
        dtype : torch.dtype
            Default element dtype for collective operands. Must be one of
            ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``;
            ``backend="pcie"`` additionally accepts ``torch.int8``,
            ``torch.uint8``, ``torch.float8_e4m3fn`` and
            ``torch.float8_e5m2``.
        backend : str, default = "auto"
            Backend selection policy. ``"auto"`` probes topology and prefers
            NVLink when supported, otherwise falls back to NCCL. ``"nvlink"``
            forces the NVLink backend and raises if unavailable. ``"pcie"``
            explicitly enables the experimental single-node 1/2/4/8-rank
            PCIe transport. ``"nccl"`` forces the NCCL path.
        device : Optional[Union[torch.device, str, int]], optional
            CUDA device bound to this rank. ``None`` uses the current CUDA
            device. Strings and integers are normalized to an explicit CUDA
            ordinal.
        """
        self._state = _CLOSED  # flipped to OPEN only when construction succeeds
        self._nvlink_armed = False  # joint property: set on all ranks or none
        # opaque NVLink-backend handle (a C++ UlyssesA2A* as an int) from
        # init_ulysses_a2a; None until armed and after teardown
        self._fa: Optional[int] = None
        self._out_ptrs: Optional[List[int]] = None
        self._sig_ptrs: Optional[List[int]] = None
        # rank-local resource tracking for staged init/teardown
        self._exports: List[int] = []  # device ptrs this rank cudaMalloc'ed
        self._imports: List[int] = []  # peer ptrs this rank IpcOpen'ed
        self._pcie: Optional[int] = None
        self._pcie_armed = False
        # keyed by output device pointer: committed registrations and the
        # provisional ledger that owns a partially registered allocation
        # registered output device pointer -> native mode (0 scatter, 1 gather)
        self._pcie_outputs: Dict[int, int] = {}
        # the caller stream the first PCIe collective bound this communicator to
        self._pcie_stream: Optional[torch.cuda.Stream] = None
        # Sticky: a Python failure before the native enqueue can leave peers
        # spinning in an unbounded barrier, so close() must not synchronize.
        self._pcie_python_teardown_safe = True
        self._broken_reason: Optional[str] = None

        if group is None:
            group = dist.group.WORLD
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        # ---- collective-safe config validation ------------------------------
        # The bound device must be resolved BEFORE the first collective: NCCL
        # object collectives stage through a tensor on the *current* device,
        # so an explicit device="cuda:rank" without a prior set_device(rank)
        # would otherwise land every rank's metadata collective on GPU 0.
        # _resolve_device never raises; an unparsable input yields the current
        # device as a safe gather guard and the joint config validation right
        # after rejects it on every rank together.
        self.device = self._resolve_device(device)
        # Encode the local config with zero user code (exact type checks and
        # interpreter/torch-provided names only), gather, then validate the
        # identical list jointly so an invalid single-rank config raises the
        # same error on every rank instead of hanging peers in a later gather.
        # Devices are validated per rank but may legitimately differ across
        # ranks (cuda:rank); only max_bytes and dtype must match.
        config = self._encode_config(max_bytes, dtype, device, backend)
        configs = self._gather(config)
        self._validate_configs_jointly(configs)

        self.max_bytes = max_bytes
        self.dtype = dtype

        # ---- backend selection: strictly before any IPC/JIT -----------------
        # topology_decision is what the probe concluded; decision is the
        # *effective* backend after runtime initialization (they differ only
        # when NVLink init failed at runtime and auto fell back to NCCL).
        self.topology_decision: UlyssesBackendDecision = resolve_ulysses_backend(
            backend, group=group, device=self.device
        )
        self.decision: UlyssesBackendDecision = self.topology_decision
        self.backend = self.decision.backend
        self.fallback_reason = (
            self.decision.reason
            if self.backend == "nccl" and backend != "nccl"
            else None
        )
        self.transport = None

        if self.backend == "nvlink":
            err = self._nvlink_init_transaction()
            if err is not None:
                # all ranks cleaned up (verified group-wide by the staged
                # cleanup) and hold the same joint error
                if backend == "nvlink":
                    raise RuntimeError(f"NVLink backend initialization failed: {err}")
                self.backend = "nccl"
                self.fallback_reason = f"nvlink init failed: {err}"
                self.decision = UlyssesBackendDecision("nccl", self.fallback_reason)

        if self.backend == "pcie":
            plan = self.decision.pcie_plan
            self.transport = plan.transport
            fell_back = (
                self.transport == "p2p"
                and self.world_size > 1
                and (
                    plan.requested_route in ("rdma", "hybrid")
                    or (
                        plan.requested_route == "auto"
                        and self.world_size in PCIE_AUTO_RDMA_WORLD_SIZES
                    )
                )
            )
            if fell_back:
                # The all-P2P route is a correctness fallback whose performance
                # depends on the host PCIe topology. Surface it so deployments
                # do not mistake a functional fallback for the intended route.
                warnings.warn(
                    "the PCIe Ulysses backend fell back to the all-P2P route; "
                    "benchmark it against NCCL before deployment: "
                    f"{self.decision.reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # A one-rank layout transform is an identity operation.  Preserve
            # the explicitly requested backend/transport in introspection, but
            # do not compile or arm a transport that can never communicate.
            if self.world_size > 1:
                err = self._pcie_init_transaction()
                if err is not None:
                    raise RuntimeError(f"PCIe backend initialization failed: {err}")

        # NCCL fallback needs a group that can move CUDA tensors; deterministic
        # in the (identical) group object, so a plain raise is group-uniform.
        if self.backend == "nccl" and self.world_size > 1:
            supported, observed = self._group_supports_cuda_alltoall()
            if not supported:
                raise ValueError(
                    "the Ulysses NCCL backend requires a process group "
                    f"supporting CUDA all-to-all (nccl), got '{observed}'"
                )

        self._state = _OPEN

    # ---- collective helpers ---------------------------------------------------

    def _group_supports_cuda_alltoall(self) -> Tuple[bool, str]:
        """A plain get_backend substring check would reject legitimate
        multi-backend groups (init_process_group(backend=None) reports
        "undefined" while its CUDA backend is ProcessGroupNCCL); check the
        backend actually bound to the CUDA device in that case. Deterministic
        in the group, so the resulting raise is group-uniform."""
        try:
            observed = str(dist.get_backend(self.group))
        except Exception as e:  # noqa: BLE001
            observed = f"<error: {type(e).__name__}: {e}>"
        if "nccl" in observed.lower():
            return True, observed
        try:
            cuda_backend = self.group._get_backend(torch.device("cuda"))
            if (
                cuda_backend is not None
                and "nccl" in type(cuda_backend).__name__.lower()
            ):
                return True, f"{observed} (cuda: {type(cuda_backend).__name__})"
        except Exception:  # noqa: BLE001 — no CUDA backend bound
            pass
        return False, observed

    def _gather(self, payload: Any) -> List[Any]:
        out: List[Any] = [None] * self.world_size
        # once the communicator device is resolved, metadata collectives must
        # not run on whatever the caller's current device happens to be
        device = getattr(self, "device", None)
        if device is not None:
            with torch.cuda.device(device):
                dist.all_gather_object(out, payload, group=self.group)
        else:
            dist.all_gather_object(out, payload, group=self.group)
        return out

    @staticmethod
    def _parse_cuda_ordinal(device) -> Tuple[Optional[int], Optional[str]]:
        """Strictly parse a device spec into a CUDA ordinal.

        Returns ``(index_or_None_for_current, error_or_None)``. torch.device
        wraps ordinals into a signed byte (``cuda:256`` silently becomes
        ``cuda:0``), so raw str/int ordinals are validated BEFORE any torch
        normalization; pre-built torch.device objects can only be checked for
        the surviving (possibly wrapped) index range.
        """
        count = torch.cuda.device_count()
        if device is None:
            return None, None
        if isinstance(device, bool):
            return None, f"invalid type: {type(device).__name__}"
        if isinstance(device, int):
            if 0 <= device < count:
                return device, None
            return None, f"ordinal {device} outside visible device count {count}"
        if isinstance(device, str):
            m = re.fullmatch(r"\s*cuda(?::(\d+))?\s*", device)
            if m is None:
                try:
                    parsed = torch.device(device)
                except (RuntimeError, ValueError, TypeError) as e:
                    return None, f"unparsable device: {e}"
                if parsed.type != "cuda":
                    return None, f"device must be a CUDA device, got {parsed}"
                return parsed.index, None
            if m.group(1) is None:
                return None, None  # bare "cuda" == current device
            idx = int(m.group(1))
            if 0 <= idx < count:
                return idx, None
            return None, f"ordinal {idx} outside visible device count {count}"
        if isinstance(device, torch.device):
            if device.type != "cuda":
                return None, f"device must be a CUDA device, got {device}"
            if device.index is None:
                return None, None
            if 0 <= device.index < count:
                return device.index, None
            return None, f"index {device.index} outside visible device count {count}"
        return None, f"invalid type: {type(device).__name__}"

    @classmethod
    def _resolve_device(cls, device) -> torch.device:
        """Never raises: yields the bound device for valid input and a safe
        gather-guard device (the current one) otherwise — the joint config
        validation rejects the invalid input right after."""
        try:
            index, err = cls._parse_cuda_ordinal(device)
            if err is not None:
                index = None
            if index is None:
                index = torch.cuda.current_device()
            return torch.device("cuda", index)
        except Exception:  # noqa: BLE001
            return torch.device("cuda", 0)

    @classmethod
    def _encode_config(cls, max_bytes, dtype, device, backend) -> Tuple[str, ...]:
        if type(max_bytes) is not int:  # bool is an int subclass: reject it too
            nbytes = f"<invalid type: {type(max_bytes).__name__}>"
        else:
            nbytes = str(max_bytes)
        if isinstance(dtype, torch.dtype):
            dt = str(dtype)
        else:
            dt = f"<invalid type: {type(dtype).__name__}>"
        bk = (
            backend
            if isinstance(backend, str)
            else f"<invalid type: {type(backend).__name__}>"
        )
        try:
            index, err = cls._parse_cuda_ordinal(device)
        except Exception as e:  # noqa: BLE001
            index, err = None, f"{type(e).__name__}: {e}"
        if err is not None:
            dev = f"<invalid device: {err}>"
        elif index is None:
            dev = "cuda"
        else:
            dev = f"cuda:{index}"
        return (nbytes, dt, dev, bk)

    def _validate_configs_jointly(self, configs) -> None:
        problems = {}
        for r, (nbytes, dt, dev, bk) in enumerate(configs):
            # The whitelist comes from the GATHERED backend, never from a
            # local argument: every rank must compute the same verdict from
            # the same list, or one rank raises while its peers hang in the
            # next collective.
            dtypes = _PCIE_SUPPORTED_DTYPES if bk == "pcie" else _SUPPORTED_DTYPES
            supported = tuple(str(d) for d in dtypes)
            errs = []
            if not nbytes.isdigit() or int(nbytes) <= 0:
                errs.append(f"max_bytes must be a positive int, got {nbytes}")
            elif int(nbytes) > _MAX_CAPACITY_BYTES:
                errs.append(
                    f"max_bytes must be at most {_MAX_CAPACITY_BYTES} (int32 "
                    f"kernel index range at the widest supported element), got "
                    f"{nbytes}"
                )
            if dt not in supported:
                errs.append(f"dtype must be one of {supported}, got {dt}")
            if not dev.startswith("cuda"):
                errs.append(f"device must be a CUDA device, got {dev}")
            if errs:
                problems[r] = "; ".join(errs)
        if problems:
            raise ValueError(f"invalid UlyssesCommunicator config by rank: {problems}")
        shared = {(nbytes, dt) for (nbytes, dt, _dev, _bk) in configs}
        if len(shared) > 1:
            raise ValueError(
                f"inconsistent UlyssesCommunicator configs across ranks: "
                f"(max_bytes, dtype) = {sorted(shared)}; all ranks must agree"
            )

    # ---- experimental PCIe/mlx5 backend ------------------------------------

    def _pcie_init_transaction(self) -> Optional[str]:
        plan = self.decision.pcie_plan
        uses_rdma = plan.transport in _PCIE_RDMA_TRANSPORTS
        gid_index = plan.gid_indices[self.rank] if uses_rdma else -1
        # Native code routes a peer over RDMA when its group id differs. The
        # hybrid route groups by physical NUMA node; the all-RDMA route makes
        # every rank its own group.
        groups = (
            list(range(self.world_size))
            if plan.transport == "rdma"
            else list(plan.numa_nodes)
        )

        try:
            module = get_ulysses_pcie_module()
            outcome: Tuple[str, ...] = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} PCIe JIT: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return err

        try:
            self._pcie, info = module.init(
                self.rank,
                self.world_size,
                self.device.index,
                groups,
                plan.nic_names[self.rank],
                1 if uses_rdma else 0,
                gid_index,
            )
            info = list(info)
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            info = None
            outcome = ("err", f"rank {self.rank} PCIe init: {type(e).__name__}: {e}")
        gathered = self._gather((outcome, info))
        err = self._first_error([item[0] for item in gathered])
        if err is not None:
            return self._pcie_init_cleanup(err)

        try:
            flat = [byte for _status, record in gathered for byte in record]
            module.connect(self._pcie, flat)
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} PCIe connect: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._pcie_init_cleanup(err)

        self._pcie_armed = True
        return None

    def _pcie_init_cleanup(self, original: str) -> str:
        detail = None
        if self._pcie is not None:
            try:
                get_ulysses_pcie_module().dispose(self._pcie)
                self._pcie = None
            except Exception as e:  # noqa: BLE001
                if self._pcie is not None:
                    detail = f"rank {self.rank}: {type(e).__name__}: {e}"
        failures = [item for item in self._gather(detail) if item is not None]
        if failures:
            raise RuntimeError(
                f"PCIe initialization failed ({original}) and cleanup failed: {failures}"
            )
        return original

    @flashinfer_api
    def allocate_output(
        self, x: torch.Tensor, op: str, *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        r"""Allocate a registered output for one Ulysses layout transform.

        For ``backend="pcie"`` this is a collective cold-path operation: the
        fixed output uses native CUDA allocation, is registered with the
        selected transport, and remains registered until :meth:`close`. This
        gives the caller a handle it owns for the communicator's lifetime.
        Multi-rank PCIe calls require such an output; allocate every geometry
        during setup and pass it through ``out=`` on the exchange.

        Parameters
        ----------
        x : torch.Tensor
            Operand the output is sized for: a contiguous 4-D CUDA tensor with
            the shape ``op`` consumes.
        op : str
            ``"scatter_heads"``, ``"gather_heads"`` or ``"exchange_chunks"``.
            Outputs are registered per transform and are not interchangeable
            between them.
        dtype : torch.dtype, optional
            Element type for the calls this output serves, overriding the
            communicator dtype. pcie backend only. The allocation keeps the
            communicator's byte budget, so a narrower dtype holds
            proportionally more elements.

        Returns
        -------
        torch.Tensor
            Tensor with the shape ``op`` produces from ``x``, on the same
            device and dtype as ``x``. On multi-rank pcie it stays registered
            with the transport until :meth:`close`.
        """
        if op not in ("scatter_heads", "gather_heads", "exchange_chunks"):
            raise ValueError(
                "op must be 'scatter_heads', 'gather_heads' or 'exchange_chunks'"
            )
        self._validate(x, op, dtype)
        shape, mode = self._output_geometry(x, op)
        if self.backend != "pcie" or self.world_size == 1:
            return torch.empty(shape, dtype=x.dtype, device=x.device)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "allocate_output cannot run inside a CUDA graph capture: it "
                    "is collective and reads the group's agreement back on the "
                    "host. Call it for every geometry before capturing."
                )
        module = get_ulysses_pcie_module()
        # Three joint steps, each with a gathered outcome, so a rank-local
        # failure poisons the group instead of deadlocking it. Cleanup is
        # close()'s job — it walks every registered pointer.
        pointer = None
        try:
            # Flat at capacity; callers view it per call. The base pointer is
            # the registration key, so record it before anything can fail.
            # native sizes the allocation as capacity_elements * itemsize(x),
            # so the byte budget has to be converted into x's own elements.
            # Without this a narrower x would allocate proportionally fewer
            # bytes than _validate just admitted, and the ICHECK_GE inside
            # allocate_output would fire mid-collective.
            capacity_elems = self.max_bytes // x.dtype.itemsize
            tensor, info = module.allocate_output(self._pcie, x, mode, capacity_elems)
            pointer = tensor.data_ptr()
            self._pcie_outputs[pointer] = mode
            info = list(info)
            outcome: Tuple[str, Any] = ("ok", (op, tuple(x.shape), str(x.dtype)))
        except Exception as e:  # noqa: BLE001
            info = None
            outcome = (
                "err",
                f"rank {self.rank} PCIe {op} output registration: "
                f"{type(e).__name__}: {e}",
            )
        gathered = self._pcie_gather_or_break((outcome, info), "output registration")
        statuses = [item[0] for item in gathered]
        err = self._first_error(statuses)
        if err is not None:
            self._raise_pcie_broken(err)
        if any(status[1] != statuses[0][1] for status in statuses):
            self._raise_pcie_broken(
                f"rank-inconsistent explicit PCIe output geometry: {statuses}"
            )

        try:
            flat_info = [byte for _status, record in gathered for byte in record]
            module.connect_output(self._pcie, tensor, flat_info)
            outcome2: Tuple[str, ...] = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome2 = (
                "err",
                f"rank {self.rank} PCIe output connect: {type(e).__name__}: {e}",
            )
        err = self._first_error(self._pcie_gather_or_break(outcome2, "output connect"))
        if err is not None:
            self._raise_pcie_broken(err)

        try:
            torch.cuda.synchronize(self.device)
            outcome2 = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome2 = (
                "err",
                f"rank {self.rank} PCIe output ready: {type(e).__name__}: {e}",
            )
        err = self._first_error(self._pcie_gather_or_break(outcome2, "output ready"))
        if err is not None:
            self._raise_pcie_broken(err)

        # Native storage is sized at max_bytes; hand back the view this call
        # requested.
        return tensor.narrow(0, 0, int(torch.Size(shape).numel())).view(shape)

    @flashinfer_api
    def input_buffer(self, out: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        r"""The transport's own input staging buffer behind a registered output.

        The RDMA routes never let the NIC read caller memory: every operand is
        copied into a landing buffer registered once alongside ``out``. Building
        the operand directly in the buffer returned here removes that copy --
        for a layer-scale scatter it is the largest device-to-device move on the
        path -- and the caller's own source allocation with it.

        The result must reach :meth:`scatter_heads` or :meth:`gather_heads`
        unmodified: the fast path is exact pointer identity, and a slice or a
        re-view that moves the base pointer is rejected rather than silently
        staged, because on the scatter path the NIC reads the landing buffer
        whatever the caller passes.

        Parameters
        ----------
        out : torch.Tensor
            An output returned by :meth:`allocate_output`.
        shape : Sequence[int]
            The ``[B, S, H, D]`` operand shape to view the buffer as.

        Returns
        -------
        torch.Tensor
            A view of transport-owned memory in the dtype ``out`` was registered
            under. It stays alive as long as ``out`` does, and every exchange on
            ``out`` overwrites it -- it is where the next operand is built, not
            where one is kept.
        """
        self._require_open("input_buffer")
        if self.backend != "pcie" or self.world_size == 1:
            raise ValueError(
                "input_buffer exists only on the multi-rank pcie backend; other "
                "backends and the world-size-one identity read the operand in place"
            )
        if self.transport not in _PCIE_RDMA_TRANSPORTS:
            raise ValueError(
                f"the {self.transport} PCIe route has no landing buffer: it reads "
                "the caller's operand in place, so there is no copy to remove"
            )
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"input_buffer expects a torch.Tensor, got {type(out).__name__}"
            )
        if out.data_ptr() not in self._pcie_outputs:
            raise ValueError(
                "input_buffer expects an output returned by allocate_output; the "
                "landing buffer is registered per output, not per communicator"
            )
        shape = tuple(int(s) for s in shape)
        if len(shape) != 4:
            raise ValueError(
                f"input_buffer expects a 4-D [B, S, H, D] shape, got {shape}"
            )
        if any(s <= 0 for s in shape):
            raise ValueError(
                f"input_buffer shape dims must all be positive, got {shape}"
            )
        module = get_ulysses_pcie_module()
        landing = module.input_landing(self._pcie, out)
        numel = int(torch.Size(shape).numel())
        if numel > landing.numel():
            raise ValueError(
                f"input_buffer shape {shape} needs {numel} elements of "
                f"{landing.element_size()} bytes, over the {landing.numel()} this "
                f"slot was registered for"
            )
        return landing.narrow(0, 0, numel).view(shape)

    def _output_geometry(self, x: torch.Tensor, op: str) -> Tuple[Tuple[int, ...], int]:
        """Output shape and native mode for one layout transform.

        The single source of truth for both: the public collectives and the
        PCIe output allocators all route through here, so a divisibility rule
        and its error message cannot drift between them.
        """
        B, S, H, D = x.shape
        if op == "exchange_chunks":
            # Equal-length chunk all-to-all: geometry in == geometry out. The
            # payload is already packed destination-major, so there is nothing
            # to scatter or gather -- only chunk r to deliver to peer r.
            if B != 1 or S != 1:
                raise ValueError(
                    f"exchange_chunks expects [1, 1, world_size, chunk], got "
                    f"shape {tuple(x.shape)}"
                )
            if self.world_size != H:
                raise ValueError(
                    f"exchange_chunks requires one chunk per peer (dim 2 == "
                    f"world size {self.world_size}), got shape {tuple(x.shape)}"
                )
            return (B, S, H, D), 2
        if op == "scatter_heads":
            if H % self.world_size != 0:
                raise ValueError(
                    f"scatter_heads requires the global head count (dim 2) to be "
                    f"divisible by world size {self.world_size}, got shape "
                    f"{tuple(x.shape)}"
                )
            return (B, S * self.world_size, H // self.world_size, D), 0
        if S % self.world_size != 0:
            raise ValueError(
                f"gather_heads requires the global sequence length (dim 1) to "
                f"be divisible by world size {self.world_size}, got shape "
                f"{tuple(x.shape)}"
            )
        return (B, S // self.world_size, H * self.world_size, D), 1

    def _validate_out(
        self, out, shape, op: str, dtype: Optional[torch.dtype] = None
    ) -> None:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"{op} out must be a torch.Tensor")
        if out.device != self.device:
            raise ValueError(
                f"{op} out is on {out.device}, but this communicator is bound "
                f"to {self.device}"
            )
        expected_dtype = self.dtype if dtype is None else dtype
        if out.dtype != expected_dtype:
            raise ValueError(
                f"{op} out dtype {out.dtype} does not match the expected "
                f"dtype {expected_dtype}"
            )
        if not out.is_contiguous() or tuple(out.shape) != tuple(shape):
            raise ValueError(f"{op} out must be contiguous with shape {tuple(shape)}")

    @staticmethod
    def _validate_no_overlap(x: torch.Tensor, out: torch.Tensor, op: str) -> None:
        if x.device != out.device:
            return
        x_begin = x.data_ptr()
        x_end = x_begin + x.numel() * x.element_size()
        out_begin = out.data_ptr()
        out_end = out_begin + out.numel() * out.element_size()
        if x_begin < out_end and out_begin < x_end:
            raise ValueError(f"{op} out must not overlap input storage")

    def _pcie_exchange(self, x, out, mode: int) -> torch.Tensor:
        registered_mode = self._pcie_outputs.get(out.data_ptr())
        if registered_mode is None:
            raise ValueError("PCIe out must come from allocate_output()")
        if registered_mode != mode:
            raise ValueError("PCIe out was registered for another operation")
        try:
            get_ulysses_pcie_module().exchange(self._pcie, x, out, mode, *x.shape)
        except Exception as e:  # noqa: BLE001
            reason = f"rank {self.rank} PCIe exchange: {type(e).__name__}: {e}"
            if self.transport == "p2p":
                self._poison_pcie_p2p(reason)
            self._raise_pcie_broken(reason)
        return out

    def _pcie_collective(
        self, x, out, op: str, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Run one multi-rank PCIe collective under its failure envelope.

        The all-P2P barrier has no bounded abort protocol. A peer may enqueue
        before this rank fails validation, capture detection, stream handoff,
        or wrapper dispatch. Such a local failure therefore poisons teardown
        before it enters BROKEN, even when this rank never reached native code.
        """
        was_open = self._state == _OPEN
        try:
            self._validate(x, op, dtype)
            shape, mode = self._output_geometry(x, op)
            if out is None:
                raise ValueError(
                    f"multi-rank PCIe {op} requires out= from allocate_output()"
                )
            with torch.cuda.device(self.device):
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing and self.transport in _PCIE_RDMA_TRANSPORTS:
                raise RuntimeError(
                    f"the {self.transport} PCIe route cannot be captured into "
                    "a CUDA graph: each exchange posts mlx5 work requests and "
                    "polls its completion queue from the host. "
                    "FLASHINFER_ULYSSES_PCIE_ROUTE=p2p on every rank gives a "
                    "capturable all-P2P route."
                )
            if not capturing:
                # Like PcieIpcAllReduceWorkspace, the communicator is bound to
                # the stream of its first collective; peers order their copies
                # against this rank's caller stream, so silently accepting a
                # second stream could let a consumer race the next exchange.
                current = torch.cuda.current_stream(self.device)
                if self._pcie_stream is None:
                    self._pcie_stream = current
                elif current != self._pcie_stream:
                    raise RuntimeError(
                        "PCIe Ulysses collectives are bound to the stream of "
                        "their first call; use one stream per communicator"
                    )
            self._validate_out(out, shape, op, dtype)
            self._validate_no_overlap(x, out, op)
            return self._pcie_exchange(x, out, mode)
        except Exception as e:  # noqa: BLE001
            if was_open and self.transport == "p2p" and self._pcie_python_teardown_safe:
                self._poison_pcie_p2p(
                    f"rank {self.rank} PCIe {op} before/during enqueue: "
                    f"{type(e).__name__}: {e}"
                )
            raise

    def _poison_pcie_p2p(self, reason: str) -> None:
        """Fail-stop an all-P2P communicator whose peer barrier is uncertain."""
        self._pcie_python_teardown_safe = False
        self._raise_pcie_broken(reason)

    def _raise_pcie_broken(self, reason: str) -> None:
        self._broken_reason = reason
        if self._state not in (_CLOSING, _CLOSED):
            self._state = _BROKEN
        raise RuntimeError(
            f"PCIe Ulysses communicator entered BROKEN state: {reason}; "
            "only close() is permitted"
        )

    def _pcie_gather_or_break(self, payload, phase: str):
        try:
            return self._gather(payload)
        except Exception as e:  # noqa: BLE001
            self._raise_pcie_broken(
                f"rank {self.rank} PCIe {phase} gather failed: {type(e).__name__}: {e}"
            )

    # ---- staged NVLink initialization (collective-safe transaction) -----------
    #
    # Every stage ends with an outcome all-gather, so a rank-local failure at
    # any point (JIT compile, cudaMalloc, IPC get-handle, IPC open, kernel
    # init) is seen by all ranks together; they then run the same staged
    # cleanup (close imports -> gather -> free exports -> gather) and return
    # the same joint error. No bare barrier is ever reached by only a subset
    # of ranks.

    def _nvlink_init_transaction(self) -> Optional[str]:
        # stage J: JIT compile / load both modules and read the signal size.
        # Every import (including cudart below) lives inside a stage envelope:
        # an import failing on one rank must become a gathered outcome, not an
        # exception escaping before a gather.
        try:
            from .vllm_ar import meta_size

            with torch.cuda.device(self.device):
                get_ulysses_a2a_module()
                sig_bytes = int(meta_size())
            outcome: Tuple[str, ...] = ("ok", str(sig_bytes))
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} JIT/meta: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return err  # nothing allocated anywhere yet

        # stage A: allocate this rank's export buffers and IPC handles
        out_bytes = self.max_bytes
        handles: Optional[Tuple[Any, Any]] = None
        try:
            from .cuda_ipc import cudart

            with torch.cuda.device(self.device):
                out_ptr = cudart.cudaMalloc(out_bytes)
                self._exports.append(out_ptr.value)
                out_handle = cudart.cudaIpcGetMemHandle(out_ptr)
                sig_ptr = cudart.cudaMalloc(sig_bytes)
                self._exports.append(sig_ptr.value)
                sig_handle = cudart.cudaIpcGetMemHandle(sig_ptr)
            handles = (out_handle, sig_handle)
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} alloc: {type(e).__name__}: {e}")
        gathered = self._gather((outcome, handles))
        err = self._first_error([o for (o, _h) in gathered])
        if err is not None:
            return self._staged_cleanup(err)

        # stage B: open every peer's handles
        all_handles = [h for (_o, h) in gathered]
        out_ptrs: List[int] = [0] * self.world_size
        sig_ptrs: List[int] = [0] * self.world_size
        try:
            from .cuda_ipc import cudart

            with torch.cuda.device(self.device):
                for i, pair in enumerate(all_handles):
                    if i == self.rank:
                        out_ptrs[i] = self._exports[0]
                        sig_ptrs[i] = self._exports[1]
                        continue
                    p = cudart.cudaIpcOpenMemHandle(pair[0])
                    self._imports.append(p.value)
                    out_ptrs[i] = p.value
                    p = cudart.cudaIpcOpenMemHandle(pair[1])
                    self._imports.append(p.value)
                    sig_ptrs[i] = p.value
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} IPC open: {type(e).__name__}: {e}")
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._staged_cleanup(err)

        # stage C: create the kernel handle (zeroes this rank's signal buffer)
        # and synchronize the bound device before reporting success — the
        # zeroing uses cudaMemset, which is asynchronous with respect to the
        # host, so neither the API returning nor the following host-side
        # gather is a CUDA completion fence on its own.
        try:
            with torch.cuda.device(self.device):
                self._fa = init_ulysses_a2a(
                    out_ptrs, sig_ptrs, self.rank, self.world_size, True
                )
                torch.cuda.synchronize()
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} init: {type(e).__name__}: {e}")
        # once every rank passes this gather, every rank's signal buffer is
        # both zeroed on-device (explicit synchronize above) and visible.
        err = self._first_error(self._gather(outcome))
        if err is not None:
            return self._staged_cleanup(err)

        self._out_ptrs = out_ptrs
        self._sig_ptrs = sig_ptrs
        self._nvlink_armed = True
        return None

    @staticmethod
    def _first_error(outcomes: List[Tuple[str, ...]]) -> Optional[str]:
        errs = [o[1] for o in outcomes if o and o[0] == "err"]
        return "; ".join(errs) if errs else None

    def _staged_cleanup(self, err: str) -> str:
        """Joint init-failure cleanup: all ranks arrive here together (they
        all saw the same failed outcome gather) and run the full teardown
        protocol. Cleanup completion is *verified* group-wide; if it cannot
        be completed the constructor fails jointly on every rank (auto is not
        allowed to fall back to NCCL while NVLink resources may linger)."""
        cleanup_err = self._teardown_protocol(sync_first=True)
        if cleanup_err is not None:
            raise RuntimeError(
                f"NVLink backend initialization failed ({err}) and cleanup "
                f"could not be completed: {cleanup_err}"
            )
        return err

    # ---- staged teardown protocol ---------------------------------------------
    #
    # Fixed stage sequence executed by EVERY rank whenever it runs, regardless
    # of how many resources the rank still holds locally (a rank with nothing
    # left still participates in every gather — otherwise a retry after a
    # partial failure deadlocks the ranks that do have work left). Each stage
    # drains with bounded retries; the retry/stop decision is taken from the
    # gathered remaining-counts, so every rank takes the same branch.

    _TEARDOWN_ATTEMPTS = 3

    def _teardown_protocol(self, *, sync_first: bool, stages=None) -> Optional[str]:
        prologue = []
        if sync_first:
            # collectives/memsets are async enqueues: never unmap while the
            # bound device may still be executing one
            prologue.append(("synchronize device", self._try_sync))
        if stages is None:
            stages = [
                ("dispose kernel handle", self._try_dispose),
                ("close peer mappings", self._try_close_imports),
                # exports are freed only after the gathered remaining-import
                # count is zero on EVERY rank: freeing a buffer a peer still has
                # mapped is undefined behavior
                ("free exports", self._try_free_exports),
            ]
        stages = prologue + list(stages)

        for name, step in stages:
            for attempt in range(1, self._TEARDOWN_ATTEMPTS + 1):
                # broad envelope around the WHOLE step: a helper that raises
                # (module import, device-guard enter/exit, anything) must
                # become a nonzero remaining-count, never skip the gather and
                # strand the peers
                try:
                    remaining, detail = step()
                except Exception as e:  # noqa: BLE001
                    remaining = 1
                    detail = (
                        f"rank {self.rank} stage '{name}' raised: "
                        f"{type(e).__name__}: {e}"
                    )
                outcomes = self._gather((remaining, detail))
                if all(r == 0 for (r, _d) in outcomes):
                    break  # stage complete on every rank
                if attempt == self._TEARDOWN_ATTEMPTS:
                    per_rank = {r: d for r, (n, d) in enumerate(outcomes) if n > 0}
                    return f"stage '{name}' incomplete after {attempt} attempts: {per_rank}"
        return None

    def _try_sync(self) -> Tuple[int, Optional[str]]:
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
            return (0, None)
        except Exception as e:  # noqa: BLE001
            return (1, f"rank {self.rank} synchronize: {type(e).__name__}: {e}")

    def _try_dispose(self) -> Tuple[int, Optional[str]]:
        if self._fa is None:
            return (0, None)
        try:
            with torch.cuda.device(self.device):
                dispose_ulysses_a2a(self._fa)
                # ledger update inside the guard: a __exit__ raise after a
                # successful dispose must not lead to a double-delete on retry
                self._fa = None
            return (0, None)
        except Exception as e:  # noqa: BLE001
            return (
                0 if self._fa is None else 1,
                f"rank {self.rank} dispose: {type(e).__name__}: {e}",
            )

    # The release helpers update the resource ledger immediately after each
    # successful release (inside the per-pointer try, device guard included),
    # so a later failure — even a device-guard __exit__ raising — can never
    # lead to a double-close/double-free on the next attempt.

    def _try_close_imports(self) -> Tuple[int, Optional[str]]:
        from .cuda_ipc import cudart

        last = None
        for ptr in list(self._imports):
            try:
                with torch.cuda.device(self.device):
                    cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                    # ledger update inside the guard: even a __exit__ raise
                    # after a successful close cannot cause a double-close
                    self._imports.remove(ptr)
            except Exception as e:  # noqa: BLE001 — keep for retry
                last = f"rank {self.rank} close import: {type(e).__name__}: {e}"
        return (len(self._imports), last)

    def _try_free_exports(self) -> Tuple[int, Optional[str]]:
        from .cuda_ipc import cudart

        last = None
        for ptr in list(self._exports):
            try:
                with torch.cuda.device(self.device):
                    cudart.cudaFree(ctypes.c_void_p(ptr))
                    self._exports.remove(ptr)
            except Exception as e:  # noqa: BLE001 — keep for retry
                last = f"rank {self.rank} free export: {type(e).__name__}: {e}"
        return (len(self._exports), last)

    def _pcie_close(self) -> Optional[str]:
        """Tear the PCIe transport down through the shared staged runner.

        Same shape as the NVLink path: a fixed stage sequence, each stage
        retried a bounded number of times, and every rank taking the same
        branch because the decision is made from the gathered remaining-count
        rather than from local state. Both native calls are idempotent, so a
        retried stage repeats no completed work.
        """
        return self._teardown_protocol(
            sync_first=False,
            stages=[
                # Hybrid failure recovery is bounded in native code. Every rank
                # must prove that bound before any rank enters the otherwise
                # unbounded device synchronize below.
                ("verify native teardown safety", self._try_pcie_teardown_safe),
                ("synchronize device", self._try_sync),
                # Peer imports first: no rank may free an export while another
                # can still reference it.
                ("close peer imports", self._try_pcie_disconnect),
                ("dispose output registrations", self._try_pcie_dispose_outputs),
                ("dispose transport", self._try_pcie_dispose_transport),
            ],
        )

    def _try_pcie_teardown_safe(self) -> Tuple[int, Optional[str]]:
        if not self._pcie_python_teardown_safe:
            return (
                1,
                f"rank {self.rank} all-P2P peer barrier state is uncertain; "
                "process termination required",
            )
        if self._pcie is None:
            return 0, None
        try:
            safe = bool(get_ulysses_pcie_module().teardown_safe(self._pcie))
        except Exception as e:  # noqa: BLE001
            return (
                1,
                f"rank {self.rank} native teardown-safety query: "
                f"{type(e).__name__}: {e}; process termination required",
            )
        if safe:
            return 0, None
        return (
            1,
            f"rank {self.rank} native GPU work could not be bounded; "
            "process termination required",
        )

    def _try_pcie_disconnect(self) -> Tuple[int, Optional[str]]:
        module = get_ulysses_pcie_module()
        errors = []
        for pointer in list(self._pcie_outputs):
            try:
                module.disconnect_output_ptr(self._pcie, pointer)
            except Exception as e:  # noqa: BLE001
                errors.append(f"pointer {pointer}: {type(e).__name__}: {e}")
        return len(errors), "; ".join(errors) if errors else None

    def _try_pcie_dispose_outputs(self) -> Tuple[int, Optional[str]]:
        # The registry is the ledger: a pointer is retried until its dispose
        # succeeds and is then dropped, so remaining is just what is left.
        module = get_ulysses_pcie_module()
        errors = []
        for pointer in list(self._pcie_outputs):
            try:
                module.dispose_output_ptr(self._pcie, pointer)
                self._pcie_outputs.pop(pointer, None)
            except Exception as e:  # noqa: BLE001
                errors.append(f"pointer {pointer}: {type(e).__name__}: {e}")
        return len(self._pcie_outputs), "; ".join(errors) if errors else None

    def _try_pcie_dispose_transport(self) -> Tuple[int, Optional[str]]:
        if self._pcie is None:
            return 0, None
        try:
            get_ulysses_pcie_module().dispose(self._pcie)
            self._pcie = None
        except Exception as e:  # noqa: BLE001
            return (
                1,
                f"rank {self.rank} PCIe transport teardown: {type(e).__name__}: {e}",
            )
        return 0, None

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        r"""Release the communicator. Idempotent once fully closed.

        Collective when the NVLink or PCIe backend was armed: every rank must
        call ``close`` together. PCIe first requires every rank to report that
        native GPU/RDMA work is bounded; if any rank cannot prove that, no rank
        enters device synchronization or releases a registration, and process
        termination is required. A safe PCIe close then synchronizes, closes
        every output's peer imports, disposes its MR/MKey registration, and
        finally disposes the transport. NVLink similarly synchronizes before
        releasing its kernel handle, peer mappings, and exports. Each stage has
        bounded group-coordinated retries. If a retryable teardown stage still
        cannot complete, every rank raises the same error and remains CLOSING;
        every rank may retry ``close()``. CLOSED is reached only after complete
        group-wide teardown. Pure NCCL holds no resources and closes locally.
        """
        if self._state == _CLOSED:
            return
        self._state = _CLOSING

        if getattr(self, "_pcie_armed", False):
            err = self._pcie_close()
            if err is not None:
                advice = (
                    "process termination required"
                    if "process termination required" in err
                    else "retry close() on all ranks"
                )
                raise RuntimeError(
                    f"UlyssesCommunicator.close failed ({advice}): {err}"
                )
            # Disarmed only after group-wide teardown succeeded: on a failed
            # close every rank must re-enter _pcie_close() when the group
            # retries, or its peers hang in the stage gathers.
            self._pcie_armed = False
            self._state = _CLOSED
            return

        if not getattr(self, "_nvlink_armed", False):
            # never held NVLink resources on ANY rank (armed is a joint
            # property: the init transaction either succeeds or cleans up on
            # every rank together), so closing locally cannot desync peers
            self._state = _CLOSED
            return

        err = self._teardown_protocol(sync_first=True)
        if err is not None:
            raise RuntimeError(
                f"UlyssesCommunicator.close failed (retry close() on all ranks): {err}"
            )
        self._out_ptrs = None
        self._sig_ptrs = None
        self._nvlink_armed = False
        self._state = _CLOSED

    def __enter__(self) -> "UlyssesCommunicator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- collectives -----------------------------------------------------------

    @flashinfer_api(trace=ulysses_scatter_heads_trace)
    def scatter_heads(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""``[B, S_local, H, D] -> [B, S_global, H_local, D]``.

        Scatter the global heads across ranks and gather the full sequence:
        afterwards this rank holds head slice
        ``[rank * H_local, (rank+1) * H_local)`` of every token. Runs on the
        current CUDA stream. Returns the input unchanged when
        ``world_size == 1``.

        Parameters
        ----------
        x : torch.Tensor
            Contiguous 4-D CUDA tensor with shape ``[B, S_local, H, D]``.
        out : torch.Tensor, optional
            Preallocated output. It must not overlap ``x``. Multi-rank PCIe
            requires an output returned by :meth:`allocate_output`.
        dtype : torch.dtype, optional
            Element type for this call, overriding the communicator dtype.
            pcie backend only. ``max_bytes`` is denominated in bytes, so a
            narrower dtype here simply fits more elements in the same
            capacity.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_global, H_local, D]`` on the same device
            and dtype as ``x``.
        """
        if self.backend == "pcie" and self.world_size > 1:
            return self._pcie_collective(x, out, "scatter_heads", dtype)
        self._validate(x, "scatter_heads", dtype)
        shape, _mode = self._output_geometry(x, "scatter_heads")
        # ulysses_a2a is parameterized by the [B, S_local, H, D] layout, which
        # is this operand's own shape for scatter_heads.
        B, S_local, H, D = x.shape
        if self.world_size == 1:
            if out is None:
                return x
            self._validate_out(out, shape, "scatter_heads", dtype)
            self._validate_no_overlap(x, out, "scatter_heads")
            out.copy_(x)
            return out
        if self.backend == "nccl" and out is None:
            return self._nccl_scatter_heads(x)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        else:
            self._validate_out(out, shape, "scatter_heads")
            self._validate_no_overlap(x, out, "scatter_heads")
        if self.backend == "nvlink":
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 0)
            return out
        out.copy_(self._nccl_scatter_heads(x))
        return out

    @flashinfer_api(trace=ulysses_exchange_chunks_trace)
    def exchange_chunks(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""``[1, 1, W, C] -> [1, 1, W, C]``: equal-length chunk all-to-all.

        Chunk ``r`` of this rank's input goes to peer ``r``, and lands in slot
        ``rank`` of that peer's output -- the semantics of
        ``torch.distributed.all_to_all_single`` on an already-packed payload.

        This is the transform for a payload that was produced destination-major
        by a preceding pack (a quantizer, say), so its per-peer bytes are
        already contiguous. :meth:`scatter_heads` and :meth:`gather_heads` do
        the head-axis interleave instead, which forces a head-major layout on
        the producer; this entry point removes that constraint. On the RDMA
        routes it is also the cheaper descriptor -- a single-row MKey, exempt
        from the 65535-byte interleaved-stride limit that bounds ``H*D*
        element_size`` for the other two.

        Parameters
        ----------
        x : torch.Tensor
            Contiguous 4-D CUDA tensor shaped ``[1, 1, world_size, chunk]``.
            ``chunk`` is in elements of ``x``'s dtype; a packed uint8 payload
            passes ``dtype=torch.uint8`` and ``chunk`` in bytes.
        out : torch.Tensor, optional
            Preallocated output of the same shape, not overlapping ``x``.
            Multi-rank PCIe requires an output from :meth:`allocate_output`.
        dtype : torch.dtype, optional
            Element type for this call, overriding the communicator dtype.
            pcie backend only.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape, device and dtype as ``x``.
        """
        if self.backend == "pcie" and self.world_size > 1:
            return self._pcie_collective(x, out, "exchange_chunks", dtype)
        self._validate(x, "exchange_chunks", dtype)
        shape, _mode = self._output_geometry(x, "exchange_chunks")
        if self.world_size == 1:
            if out is None:
                return x
            self._validate_out(out, shape, "exchange_chunks", dtype)
            self._validate_no_overlap(x, out, "exchange_chunks")
            out.copy_(x)
            return out
        # nvlink's fused kernel is parameterized by the head-axis interleave,
        # so a chunk exchange has no fused form; NCCL's all_to_all_single is
        # exactly this operation and is what both remaining backends use.
        if out is None:
            return self._nccl_exchange_chunks(x)
        self._validate_out(out, shape, "exchange_chunks", dtype)
        self._validate_no_overlap(x, out, "exchange_chunks")
        dist.all_to_all_single(out.view(-1), x.reshape(-1), group=self.group)
        return out

    def _nccl_exchange_chunks(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        dist.all_to_all_single(out.view(-1), x.reshape(-1), group=self.group)
        return out

    @flashinfer_api(trace=ulysses_gather_heads_trace)
    def gather_heads(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""``[B, S_global, H_local, D] -> [B, S_local, H, D]``.

        Inverse of :meth:`scatter_heads`: gather all head slices for this
        rank's local sequence shard. Runs on the current CUDA stream. Returns
        the input unchanged when ``world_size == 1``.

        Parameters
        ----------
        x : torch.Tensor
            Contiguous 4-D CUDA tensor with shape ``[B, S_global, H_local, D]``.
        out : torch.Tensor, optional
            Preallocated output. It must not overlap ``x``. Multi-rank PCIe
            requires an output returned by :meth:`allocate_output`.
        dtype : torch.dtype, optional
            Element type for this call, overriding the communicator dtype.
            pcie backend only. ``max_bytes`` is denominated in bytes, so a
            narrower dtype here simply fits more elements in the same
            capacity.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_local, H, D]`` on the same device and
            dtype as ``x``.
        """
        if self.backend == "pcie" and self.world_size > 1:
            return self._pcie_collective(x, out, "gather_heads", dtype)
        self._validate(x, "gather_heads", dtype)
        shape, _mode = self._output_geometry(x, "gather_heads")
        # ulysses_a2a is parameterized by the [B, S_local, H, D] layout, which
        # for gather_heads is the *output* shape.
        B, S_local, H, D = shape
        if self.world_size == 1:
            if out is None:
                return x
            self._validate_out(out, shape, "gather_heads", dtype)
            self._validate_no_overlap(x, out, "gather_heads")
            out.copy_(x)
            return out
        if self.backend == "nccl" and out is None:
            return self._nccl_gather_heads(x)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        else:
            self._validate_out(out, shape, "gather_heads")
            self._validate_no_overlap(x, out, "gather_heads")
        if self.backend == "nvlink":
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 1)
            return out
        out.copy_(self._nccl_gather_heads(x))
        return out

    # ---- NCCL fallback ---------------------------------------------------------
    # The conventional all_to_all_single path with explicit permute/contiguous
    # glue before and after (exactly the data movement the fused NVLink kernel
    # folds into its cross-GPU writes). Bit-identical to the NVLink backend.

    def _nccl_scatter_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S_local, H, D = x.shape
        W = self.world_size
        H_local = H // W
        xt = x.reshape(B, S_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk j == rank j's contribution to sequence block j
        return recv.permute(1, 0, 2, 3, 4).reshape(B, W * S_local, H_local, D)

    def _nccl_gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S_global, H_local, D = x.shape
        W = self.world_size
        S_local = S_global // W
        xt = x.reshape(B, W, S_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk p == this rank's sequence block, head slice p
        return (
            recv.permute(1, 2, 0, 3, 4).reshape(B, S_local, W * H_local, D).contiguous()
        )

    # ---- validation ------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._state == _BROKEN:
            raise RuntimeError(
                f"{op} called on a BROKEN UlyssesCommunicator "
                f"({self._broken_reason}); only close() is permitted"
            )
        if self._state != _OPEN:
            raise RuntimeError(
                f"{op} called on a {self._state} UlyssesCommunicator (use-after-close)"
            )

    def _validate(self, x, op: str, dtype: Optional[torch.dtype] = None) -> None:
        self._require_open(op)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{op} expects a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(
                f"{op} expects a 4-D [B, S, H, D] tensor, got {x.dim()}-D shape "
                f"{tuple(x.shape)}"
            )
        if (
            self.backend == "pcie"
            and self.transport in _PCIE_RDMA_TRANSPORTS
            and x.shape[0] != 1
        ):
            raise ValueError(
                f"the {self.transport} PCIe route supports batch=1 only; "
                "FLASHINFER_ULYSSES_PCIE_ROUTE=p2p gives an all-P2P route "
                "that takes any batch size"
            )
        if x.device != self.device:
            raise ValueError(
                f"{op} tensor is on {x.device}, but this communicator is bound "
                f"to {self.device}"
            )
        expected_dtype = self.dtype if dtype is None else dtype
        if x.dtype != expected_dtype:
            raise ValueError(
                f"{op} tensor dtype {x.dtype} does not match the expected "
                f"dtype {expected_dtype}"
            )
        if dtype is not None:
            # A per-call dtype bypasses the joint check the constructor runs
            # across ranks, so re-check locally what that check would have
            # caught. Local only, and only on rank-invariant values: adding a
            # collective here would tear the group apart on a bad argument.
            # Staying local costs no group-wide agreement on the element width:
            # allocate_output all-gathers (op, shape, dtype), and _validate_out
            # below plus native's buffer->dtype check hold every later call to
            # what that registration agreed.
            if self.backend != "pcie":
                raise ValueError(
                    f"{op} per-call dtype is only supported on the pcie "
                    f"backend, not {self.backend}"
                )
            allowed = _PCIE_SUPPORTED_DTYPES
            if dtype not in allowed:
                raise ValueError(
                    f"{op} per-call dtype {dtype} is not one of {sorted(allowed, key=str)}"
                )
        if not x.is_contiguous():
            raise ValueError(f"{op} tensor must be contiguous")
        if any(s <= 0 for s in x.shape):
            raise ValueError(
                f"{op} tensor dims must all be positive, got shape {tuple(x.shape)}"
            )
        if x.nbytes > self.max_bytes:
            raise ValueError(
                f"{op} tensor is {x.nbytes} bytes ({x.numel()} elements of "
                f"{x.element_size()}), exceeding the communicator capacity "
                f"max_bytes={self.max_bytes}"
            )
        if x.numel() > _INT32_MAX:
            # The byte budget above can admit more elements than the kernels
            # can index once the per-call dtype is narrower than the
            # communicator dtype.
            raise ValueError(
                f"{op} tensor has {x.numel()} elements, over the int32 index "
                f"range {_INT32_MAX}"
            )
        if (
            self.backend == "pcie"
            and self.transport in _PCIE_RDMA_TRANSPORTS
            and op != "exchange_chunks"
        ):
            # exchange_chunks is exempt by construction: its descriptor has a
            # single row, so there is no stride to fit in mlx5's 16-bit field.
            global_heads = (
                x.shape[2] if op == "scatter_heads" else x.shape[2] * self.world_size
            )
            head_row_bytes = global_heads * x.shape[3] * x.element_size()
            if head_row_bytes > _PCIE_MLX5_MAX_INTERLEAVED_STRIDE:
                raise ValueError(
                    f"the {self.transport} PCIe route requires H*D*element_size <= "
                    f"{_PCIE_MLX5_MAX_INTERLEAVED_STRIDE} bytes, got {head_row_bytes}"
                )


# =============================================================================
# Raw (advanced) kernel entry points
# =============================================================================
# Merged from the former flashinfer/comm/ulysses_a2a.py submodule: the
# function `ulysses_a2a` exported from flashinfer.comm used to shadow that
# submodule of the same name, breaking attribute-based module access. Custom
# op names, lazy JIT timing, the post-init memset fence and handle ownership
# are unchanged. The underlying CUDA kernel is adapted from ThunderKittens'
# NVLink all-to-all:
# https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu


# Build inputs the PCIe transport needs beyond a normal FlashInfer JIT module.
# The CUDA P2P and mlx5 RDMA routes share one translation unit, so these are
# required even when topology selects all-P2P.
def missing_ulysses_pcie_dependencies() -> List[str]:
    """Names of the rdma-core libraries this machine does not provide.

    Cheap and side-effect free (compiles nothing), so an environment guard can
    distinguish an unsupported host from a real build failure.
    """
    import ctypes.util

    return [
        f"lib{name}"
        for name in ("ibverbs", "mlx5")
        if ctypes.util.find_library(name) is None
    ]


@functools.cache
def get_ulysses_pcie_module():
    try:
        module = gen_ulysses_pcie_module().build_and_load()
    except Exception as e:  # noqa: BLE001
        # The transport compiles CUDA P2P and mlx5 RDMA in one translation unit,
        # so even an all-P2P route needs the verbs/mlx5 toolchain at link time.
        # A raw ninja link error is unreadable; name the missing dependency.
        missing = missing_ulysses_pcie_dependencies()
        detail = (
            f"this machine is missing {', '.join(missing)}"
            if missing
            else "the verbs/mlx5 toolchain is present, so this is a build failure"
        )
        raise RuntimeError(
            "the experimental PCIe Ulysses backend requires the libibverbs and "
            "libmlx5 development headers and libraries at JIT compile/link time; "
            f"{detail} (building the ulysses_pcie module failed: "
            f"{type(e).__name__}: {e})"
        ) from e

    @register_custom_op("flashinfer::init_ulysses_pcie", mutates_args=[])
    def init(
        rank: int,
        world_size: int,
        device: int,
        numa: List[int],
        nic: str,
        use_rdma: int,
        gid_index: int,
    ) -> Tuple[int, List[int]]:
        return module.init_ulysses_pcie(
            rank, world_size, device, numa, nic, use_rdma, gid_index
        )

    @register_custom_op("flashinfer::connect_ulysses_pcie", mutates_args=[])
    def connect(handle: int, metadata: List[int]) -> None:
        module.connect_ulysses_pcie(handle, metadata)

    @register_custom_op("flashinfer::allocate_ulysses_pcie_output", mutates_args=[])
    def allocate_output(
        handle: int, input: torch.Tensor, mode: int, capacity_elements: int
    ) -> Tuple[torch.Tensor, List[int]]:
        return module.allocate_ulysses_pcie_output(
            handle, input, mode, capacity_elements
        )

    @register_custom_op("flashinfer::connect_ulysses_pcie_output", mutates_args=[])
    def connect_output(handle: int, output: torch.Tensor, metadata: List[int]) -> None:
        module.connect_ulysses_pcie_output(handle, output, metadata)

    @register_custom_op("flashinfer::ulysses_pcie_input_landing", mutates_args=[])
    def input_landing(handle: int, output: torch.Tensor) -> torch.Tensor:
        return module.ulysses_pcie_input_landing(handle, output)

    @register_custom_op("flashinfer::ulysses_pcie_exchange", mutates_args=["output"])
    def exchange(
        handle: int,
        input: torch.Tensor,
        output: torch.Tensor,
        mode: int,
        batch: int,
        seq: int,
        heads: int,
        dim: int,
    ) -> None:
        module.ulysses_pcie_exchange(
            handle, input, output, mode, batch, seq, heads, dim
        )

    @register_custom_op("flashinfer::ulysses_pcie_teardown_safe", mutates_args=[])
    def teardown_safe(handle: int) -> int:
        return module.ulysses_pcie_teardown_safe(handle)

    @register_custom_op(
        "flashinfer::disconnect_ulysses_pcie_output_ptr", mutates_args=[]
    )
    def disconnect_output_ptr(handle: int, pointer: int) -> None:
        module.disconnect_ulysses_pcie_output_ptr(handle, pointer)

    @register_custom_op("flashinfer::dispose_ulysses_pcie_output_ptr", mutates_args=[])
    def dispose_output_ptr(handle: int, pointer: int) -> None:
        module.dispose_ulysses_pcie_output_ptr(handle, pointer)

    @register_custom_op("flashinfer::dispose_ulysses_pcie", mutates_args=[])
    def dispose(handle: int) -> None:
        module.dispose_ulysses_pcie(handle)

    return SimpleNamespace(
        init=init,
        connect=connect,
        allocate_output=allocate_output,
        connect_output=connect_output,
        input_landing=input_landing,
        exchange=exchange,
        teardown_safe=teardown_safe,
        disconnect_output_ptr=disconnect_output_ptr,
        dispose_output_ptr=dispose_output_ptr,
        dispose=dispose,
    )


@functools.cache
def get_ulysses_a2a_module():
    module = gen_ulysses_a2a_module().build_and_load()

    @register_custom_op(
        "flashinfer::init_ulysses_a2a",
        mutates_args=[],
    )
    def init_ulysses_a2a(
        out_ipc_ptrs: List[int],
        signal_ipc_ptrs: List[int],
        rank: int,
        world_size: int,
        full_nvlink: bool,
    ) -> int:
        return module.init_ulysses_a2a(
            out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
        )

    @register_custom_op("flashinfer::dispose_ulysses_a2a", mutates_args=[])
    def dispose_ulysses_a2a(fa: int) -> None:
        module.dispose_ulysses_a2a(fa)

    @register_custom_op("flashinfer::ulysses_a2a", mutates_args=["out"])
    def ulysses_a2a(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        B: int,
        S_local: int,
        H: int,
        D: int,
        mode: int,
    ) -> None:
        module.ulysses_a2a(fa, inp, out, B, S_local, H, D, mode)

    return SimpleNamespace(
        init_ulysses_a2a=init_ulysses_a2a,
        dispose_ulysses_a2a=dispose_ulysses_a2a,
        ulysses_a2a=ulysses_a2a,
    )


@flashinfer_api
def init_ulysses_a2a(
    out_ipc_ptrs: List[int],
    signal_ipc_ptrs: List[int],
    rank: int,
    world_size: int,
    full_nvlink: bool,
) -> int:
    r"""Initialize the fused-transpose Ulysses NVLink-P2P all-to-all backend.

    .. note::
        Advanced / internal API. Prefer
        :class:`~flashinfer.comm.UlyssesCommunicator`, which selects the
        backend from the actual GPU topology before any IPC allocation or JIT
        compilation, owns the IPC workspace lifecycle, and validates operands.
        This raw entry point assumes the caller has already verified all-pairs
        NVLink P2P.

    The kernel is a *push* model: each rank writes the head/sequence blocks
    destined for its peers directly into the peers' IPC-shared output staging
    buffers over NVLink, with the Ulysses layout permutation folded into the
    write addresses. Only the output staging buffers and the signal buffers must
    be IPC-shared (allocate them with
    :func:`flashinfer.comm.create_shared_buffer`); the input tensor is read
    locally and needs no registration.

    Parameters
    ----------
    out_ipc_ptrs : list[int]
        Per-rank device pointers (opened via CUDA IPC) to the output staging
        buffers, ordered by rank. Each must be at least as large as the
        all-to-all output for this group.
    signal_ipc_ptrs : list[int]
        Per-rank device pointers to the signal buffers used for the inter-GPU
        barrier. Each buffer must be :func:`flashinfer.comm.vllm_meta_size`
        bytes (same ``Signal`` layout as the vLLM custom all-reduce).
    rank : int
        Current rank within the Ulysses group.
    world_size : int
        Ulysses group size; must be one of ``(2, 4, 6, 8)``.
    full_nvlink : bool
        ``True`` when every pair of ranks is connected via NVLink. The push
        kernel requires all-pairs P2P access; callers must gate on this.

    Returns
    -------
    int
        Opaque handle (``fa``) to pass to subsequent ``ulysses_a2a`` calls.
        Free it with :func:`dispose_ulysses_a2a`.

    Note
    ----
    ``init`` zeroes this rank's own signal buffer with a ``cudaMemset``, which
    is asynchronous with respect to the host. This wrapper therefore
    synchronizes the *current* CUDA device before returning (call it with the
    target device current). Callers still must issue a process-group barrier
    (e.g. ``torch.distributed.barrier``) after all ranks return from init and
    before the first all-to-all call — the barrier alone is not a CUDA
    completion fence, and the device sync alone is not group-wide.
    """
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"ulysses a2a only supports world size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if not full_nvlink:
        raise ValueError(
            "full_nvlink=False is not supported: the fused kernel pushes over "
            "all-pairs NVLink P2P and has no non-P2P path. Use "
            "UlyssesCommunicator(backend='auto') for topology-aware NCCL "
            "fallback instead."
        )
    module = get_ulysses_a2a_module()
    fa = module.init_ulysses_a2a(
        out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
    )
    # make the signal zeroing a real completion fence on this device
    try:
        torch.cuda.synchronize()
    except Exception:
        # the caller never receives fa on this path and could not dispose it:
        # ownership stays here, so release the handle before re-raising
        with contextlib.suppress(Exception):  # surface the sync error, not this
            module.dispose_ulysses_a2a(fa)
        raise
    return fa


@flashinfer_api
def dispose_ulysses_a2a(fa: int) -> None:
    r"""Release a handle returned by :func:`init_ulysses_a2a`.

    Parameters
    ----------
    fa : int
        The opaque backend handle previously returned by
        :func:`init_ulysses_a2a`. It is a C++ ``UlyssesA2A*`` reinterpreted as
        an integer (``fptr_t``), not a device pointer or a Python object, so it
        is only meaningful to this module. After this call the handle is
        dangling and must not be passed to :func:`ulysses_a2a` again.
    """
    get_ulysses_a2a_module().dispose_ulysses_a2a(fa)


@flashinfer_api
def ulysses_a2a(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    B: int,
    S_local: int,
    H: int,
    D: int,
    mode: int,
) -> None:
    r"""Fused-transpose Ulysses all-to-all.

    .. note::
        Advanced / internal API. Prefer
        :meth:`UlyssesCommunicator.scatter_heads` (``mode == 0``) and
        :meth:`UlyssesCommunicator.gather_heads` (``mode == 1``), which derive
        the geometry from the tensor shapes and validate operands.

    ``fa`` is the opaque backend handle returned by :func:`init_ulysses_a2a`
    (a C++ ``UlyssesA2A*`` reinterpreted as an integer ``fptr_t``); it selects
    the all-to-all context to run on and is not a device pointer.

    The result for this rank is written into ``out`` (bit-identical to the
    equivalent NCCL all-to-all followed by the layout permutation).

    ``mode == 0`` (input a2a): ``inp [B, S_local, H, D] -> out [B, S_global, H_local, D]``

    ``mode == 1`` (output a2a): ``inp [B, S_global, H_local, D] -> out [B, S_local, H, D]``

    where ``H`` is the *global* head count, ``H_local = H // world_size`` and
    ``S_global = S_local * world_size``. Both tensors must be contiguous CUDA
    tensors of the same dtype (float32/float16/bfloat16). All ranks must call
    with consistent geometry in the same order; a mismatch is a collective
    failure (hang or corruption), as with any collective.

    Parameters
    ----------
    fa : int
        Opaque backend handle returned by :func:`init_ulysses_a2a`.
    inp : torch.Tensor
        Contiguous 4-D CUDA input tensor.
    out : torch.Tensor
        Contiguous 4-D CUDA output tensor written in place.
    B : int
        Batch size.
    S_local : int
        Local sequence length per rank.
    H : int
        Global head count.
    D : int
        Head dimension.
    mode : int
        ``0`` for scatter-heads input all-to-all, ``1`` for gather-heads
        output all-to-all.
    """
    if type(fa) is not int or fa == 0:
        raise ValueError(
            f"fa must be a nonzero handle returned by init_ulysses_a2a, got {fa!r}"
        )
    for v, vname in (
        (B, "B"),
        (S_local, "S_local"),
        (H, "H"),
        (D, "D"),
        (mode, "mode"),
    ):
        if type(v) is not int:  # bool is an int subclass: reject it too
            raise ValueError(f"{vname} must be an int, got {type(v).__name__}")
    for name, t in (("inp", inp), ("out", out)):
        if not (isinstance(t, torch.Tensor) and t.is_cuda):
            raise ValueError(f"{name} must be a CUDA tensor")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D, got shape {tuple(t.shape)}")
    if inp.device != out.device:
        raise ValueError(f"inp is on {inp.device} but out is on {out.device}")
    if inp.dtype != out.dtype:
        raise ValueError(f"inp dtype {inp.dtype} != out dtype {out.dtype}")
    if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"dtype must be float16/bfloat16/float32, got {inp.dtype}")
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")
    if min(B, S_local, H, D) <= 0:
        raise ValueError(f"B/S_local/H/D must be positive, got {(B, S_local, H, D)}")
    # exact-shape checks: the [B, S_local, H, D]-layout operand of each mode is
    # fully determined by the geometry args; the other operand's split of
    # (S_global, H_local) depends on world_size (unknown here), so check its
    # batch/D dims and total size
    local_shape = (B, S_local, H, D)
    checked, other = (inp, out) if mode == 0 else (out, inp)
    if tuple(checked.shape) != local_shape:
        raise ValueError(
            f"{'inp' if mode == 0 else 'out'} shape {tuple(checked.shape)} does "
            f"not match [B, S_local, H, D] = {local_shape} for mode {mode}"
        )
    if other.shape[0] != B or other.shape[3] != D or other.numel() != checked.numel():
        raise ValueError(
            f"{'out' if mode == 0 else 'inp'} shape {tuple(other.shape)} is "
            f"inconsistent with [B, S_local, H, D] = {local_shape} "
            f"(batch/D dims and total size must match)"
        )
    for name, t in (("inp", inp), ("out", out)):
        if t.numel() > _INT32_MAX:
            raise ValueError(
                f"{name} has {t.numel()} elements, exceeding the int32 index "
                f"range {_INT32_MAX} supported by the ulysses_a2a kernel"
            )
    get_ulysses_a2a_module().ulysses_a2a(fa, inp, out, B, S_local, H, D, mode)
