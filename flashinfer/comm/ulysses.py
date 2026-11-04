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
import dataclasses
import functools
import math
import os
import re
import sys
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .. import cake_ulysses_backend
from ..api_logging import flashinfer_api
from ..jit.comm import gen_ulysses_a2a_module, gen_ulysses_pcie_module
from ..trace.templates.comm import (
    ulysses_exchange_chunks_trace,
    ulysses_gather_heads_trace,
    ulysses_scatter_heads_trace,
    ulysses_scatter_qkv_trace_dispatch,
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
_CHUNK_DTYPES = (*_SUPPORTED_DTYPES, torch.uint8)
# The PCIe transport moves bytes (copy engines and RDMA writes, no arithmetic),
# so it takes any 1-, 2- or 4-byte element type torch can hand through DLPack.
_PCIE_SUPPORTED_DTYPES = _SUPPORTED_DTYPES + (
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)
_PCIE_RDMA_TRANSPORTS = ("hybrid", "rdma")
_ULYSSES_SIGNAL_SLOTS = 4
_ULYSSES_MAX_BLOCKS = 144

# Communicator lifecycle states. BROKEN permits only a coordinated close: a
# PCIe transport error may leave QPs, CQs and registered buffers half-mutated.
# CLOSED is reached only after a fully successful teardown, so a failed
# close() can be retried.
_OPEN, _BROKEN, _CLOSING, _CLOSED = "open", "broken", "closing", "closed"


# ---- the NVLink kernel route -------------------------------------------------
#
# The NVLink backend has two kernels behind one protocol: the exported Cake
# programs of ``flashinfer.cake_ulysses_backend`` on the devices and element
# types that package delivers (SM100 / SM103), and FlashInfer's portable kernel
# (``csrc/ulysses_all_to_all.cu``) everywhere else.  Both run on the same IPC
# staging and signal allocations of the init transaction below.  The route is
# decided once per communicator, inside the transaction, so every rank agrees
# on it; these module-level hooks are the only package entry points the
# communicator calls (tests re-target them).


def _cake_nvlink_route(
    device_index: int, dtype: torch.dtype, world_size: int
) -> Optional[str]:
    """The exported Cake architecture when the package route owns this
    communicator's collectives, else ``None`` (portable kernel)."""
    return cake_ulysses_backend.cake_backend.route_arch(device_index, dtype, world_size)


def _cake_nvlink_warm(arch: str, dtype: torch.dtype, world_size: int) -> None:
    """Load (JIT-building when needed) the scatter and gather libraries of the route."""
    cake_ulysses_backend.cake_backend.warm(arch, dtype, world_size)


def _cake_nvlink_direct(
    arch: str, device_index: int, dtype: torch.dtype, world_size: int
) -> bool:
    """Whether the package also delivers the direct route (outputs registered
    with ``allocate_output``) for the architecture of the staged route."""
    return (
        cake_ulysses_backend.cake_backend.route_arch(
            device_index, dtype, world_size, direct=True
        )
        == arch
    )


def _cake_nvlink_handle(
    *,
    staging_ptrs: List[int],
    signal_ptrs: List[int],
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device_index: int,
) -> "cake_ulysses_backend.CakeUlyssesA2A":
    """Bind the package's launches to this rank's IPC allocations."""
    return cake_ulysses_backend.cake_backend.CakeUlyssesA2A(
        staging_ptrs=staging_ptrs,
        signal_ptrs=signal_ptrs,
        rank=rank,
        world_size=world_size,
        dtype=dtype,
        device_index=device_index,
    )


def _storage_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively test whether two positive-strided tensors share bytes."""
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False

    def storage_end(tensor: torch.Tensor) -> int:
        max_element_offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 0
        )
        return tensor.data_ptr() + (max_element_offset + 1) * tensor.element_size()

    return left.data_ptr() < storage_end(right) and right.data_ptr() < storage_end(left)


def _qkv_byte_range(tensor: torch.Tensor) -> Tuple[int, int]:
    """Byte extent of a validated tensor on the QKV communicator's device."""
    numel = tensor.numel()
    if numel == 0:
        return (0, 0)
    start = tensor.data_ptr()
    if tensor.is_contiguous():
        return start, start + numel * tensor.element_size()
    max_element_offset = sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return start, start + (max_element_offset + 1) * tensor.element_size()


class UlyssesWorkspace:
    r"""Reusable staging storage for NCCL Ulysses collectives.

    A workspace owns two flat CUDA buffers: one for the packed all-to-all
    input and one for the receive result. Passing it to
    :meth:`UlyssesCommunicator.scatter_heads` or
    :meth:`UlyssesCommunicator.gather_heads` removes the per-call NCCL staging
    allocations. The NVLink backend already owns IPC staging storage, so it
    validates but otherwise does not use this object.

    Workspaces are local allocations: they own no process group and create no
    streams. A workspace may be shared by serialized scatter/gather calls, but
    must not be used by concurrent collectives.

    Parameters
    ----------
    max_elems : int
        Capacity of each staging buffer in elements.
    dtype : torch.dtype
        Element dtype; float16, bfloat16, or float32.
    device : torch.device or str or int, optional
        CUDA device. ``None`` uses the current CUDA device.
    """

    @flashinfer_api
    def __init__(
        self,
        *,
        max_elems: int,
        dtype: torch.dtype,
        device: Optional[Union[torch.device, str, int]] = None,
    ):
        r"""Initialize reusable NCCL send and receive staging buffers.

        Parameters
        ----------
        max_elems : int
            Capacity of each staging buffer in elements.
        dtype : torch.dtype
            Element dtype; float16, bfloat16, or float32.
        device : torch.device or str or int, optional
            CUDA device. ``None`` uses the current CUDA device.
        """
        if type(max_elems) is not int or max_elems <= 0:
            raise ValueError(f"max_elems must be a positive int, got {max_elems!r}")
        if max_elems > _INT32_MAX:
            raise ValueError(
                f"max_elems must be at most {_INT32_MAX} (int32 index range), "
                f"got {max_elems}"
            )
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {_SUPPORTED_DTYPES}, got {dtype!r}")
        ordinal, error = UlyssesCommunicator._parse_cuda_ordinal(device)
        if error is not None:
            raise ValueError(f"invalid workspace device: {error}")
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.max_elems = max_elems
        self.dtype = dtype
        self.device = torch.device("cuda", ordinal)
        with torch.cuda.device(self.device):
            self._send_buffer = torch.empty(max_elems, dtype=dtype, device=self.device)
            self._recv_buffer = torch.empty(max_elems, dtype=dtype, device=self.device)

    @property
    def send_buffer(self) -> torch.Tensor:
        """Flat send staging buffer (advanced/debugging use)."""
        return self._send_buffer

    @property
    def recv_buffer(self) -> torch.Tensor:
        """Flat receive staging buffer (advanced/debugging use)."""
        return self._recv_buffer


class UlyssesQKV(NamedTuple):
    """Pre-quantized SageAttention2 operands returned by ``scatter_qkv``.

    ``q`` and ``k`` are contiguous INT8 ``[B, S, H/P, D]`` tensors. ``v``
    is FP8 E4M3 ``[B, D, H/P, S_pad]`` in the consumer's permuted layout.
    Q/K scales are FP32 ``[B, H/P, width]``; V scales are FP32
    ``[B, H/P, D]``. Scale widths use ``used_sequence``, whereas Q/K/V
    storage covers ``logical_sequence``. ``layout`` identifies the Sage
    consumer: ``"sage2_sm90"`` or ``"sage2_sm89_sm120"``.

    The record is immutable, but its six tensors may be overwritten by an
    explicit ``out=`` call. Attention and its softmax scale remain the
    caller's responsibility; ``input_dtype`` is the floating output dtype.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    layout: str
    logical_sequence: int
    used_sequence: int
    input_dtype: torch.dtype


@dataclasses.dataclass(frozen=True, init=False, repr=False, eq=False)
class UlyssesQKVWorkspace:
    """Prepared Lowp geometry and reusable communication storage.

    Created collectively by :meth:`UlyssesCommunicator.prepare_qkv`; direct
    construction is not supported. Metadata is read-only. The workspace is
    bound to its owner communicator and cannot be used after that owner is
    closed, on another communicator, or concurrently across streams.

    ``qkv_shape``, ``world_size``, ``dtype``, ``device``, ``used_sequence``,
    ``quantization`` and ``layout`` describe the prepared operation.
    ``transport`` is always ``"nccl"``, independently of the ordinary
    scatter/gather backend. Storage follows PyTorch tensor lifetimes and
    owns neither an extra process group nor a separate close protocol.
    """

    qkv_shape: Tuple[int, int, int, int]
    world_size: int
    dtype: torch.dtype
    device: torch.device
    used_sequence: int
    quantization: str
    layout: str
    transport: str
    _owner: "UlyssesCommunicator"
    _layout_impl: Any
    _module: Any
    _enable_pdl: bool
    _q_scale_offset: int
    _wire_shape: Tuple[int, int]
    _stats_numel: int
    _output_specs: Tuple[Tuple[Tuple[int, ...], torch.dtype], ...]
    _send_buffer: torch.Tensor
    _recv_buffer: torch.Tensor
    _stats_gather: torch.Tensor

    def __init__(self):
        raise TypeError("use UlyssesCommunicator.prepare_qkv() to create a workspace")

    @property
    def logical_sequence(self) -> int:
        """Full sequence length, including any zero-padded suffix."""
        return self.qkv_shape[1] * self.world_size

    @classmethod
    def _allocate(cls, owner, qkv_shape, used_sequence, layout, layout_impl):
        from . import _ulysses_lowp as lowp

        batch, local_sequence, heads, head_dim = qkv_shape
        world = owner.world_size
        local_heads = heads // world
        spec = layout_impl.payload_spec(
            batch_size=batch,
            local_sequence=local_sequence,
            num_heads=heads,
            world_size=world,
        )
        q_width, k_width = layout_impl.scale_widths(used_sequence)
        wire_shape = (world, int(spec["chunk_bytes"]))
        sequence = world * local_sequence
        # K sum + V amax + two Q descriptors + two raw-K min/max slices.
        stats_numel = batch * heads * (6 * head_dim + 2)
        q_shape = (batch, sequence, local_heads, head_dim)
        output_specs = (
            (q_shape, torch.int8),
            (q_shape, torch.int8),
            (
                (batch, head_dim, local_heads, int(spec["padded_sequence"])),
                torch.float8_e4m3fn,
            ),
            ((batch, local_heads, q_width), torch.float32),
            ((batch, local_heads, k_width), torch.float32),
            ((batch, local_heads, head_dim), torch.float32),
        )
        workspace = object.__new__(cls)
        values = dict(
            qkv_shape=qkv_shape,
            world_size=world,
            dtype=owner.dtype,
            device=owner.device,
            used_sequence=used_sequence,
            quantization="sage2",
            layout=layout,
            transport="nccl",
            _owner=owner,
            _layout_impl=layout_impl,
            _q_scale_offset=int(spec["q_scale_offset"]),
            _wire_shape=wire_shape,
            _stats_numel=stats_numel,
            _output_specs=output_specs,
        )
        with torch.cuda.device(owner.device):
            values["_module"] = (
                lowp.get_ulysses_lowp_sm90_module()
                if layout == "sage2_sm90"
                else lowp.get_ulysses_lowp_module()
            )
            values["_enable_pdl"] = bool(lowp.device_support_pdl(owner.device))
            values["_send_buffer"] = torch.empty(
                wire_shape, dtype=torch.uint8, device=owner.device
            )
            recv = owner.allocate_output(
                values["_send_buffer"].view(1, 1, *wire_shape),
                "exchange_chunks",
                dtype=torch.uint8,
            )
            values["_recv_buffer"] = recv.view(wire_shape)
            values["_stats_gather"] = torch.empty(
                world * stats_numel, dtype=torch.float32, device=owner.device
            )
        for name, value in values.items():
            object.__setattr__(workspace, name, value)
        return workspace


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

    :meth:`exchange_chunks` exchanges already-packed destination-major
    chunks. The NCCL and NVLink backends use NCCL for this operation.
    :meth:`prepare_qkv` and :meth:`scatter_qkv` additionally provide Sage2
    INT8/FP8 QKV preparation, using this chunk exchange and a statistics
    AllGather on the same NCCL group.

    Backend selection for ordinary scatter/gather happens in the constructor,
    strictly before any IPC
    allocation or JIT compilation (see
    :func:`~flashinfer.comm.resolve_ulysses_backend`):

    - ``backend="auto"``: the fused-transpose NVLink-P2P kernel when the group
      is a verified single-node all-pairs NVLink mesh with a supported world
      size (2/4/6/8); where NVLink is unavailable and
      ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`` is set on every rank,
      the experimental PCIe transport below (it warns once); NCCL otherwise —
      including when NVLink runtime initialization fails after a positive
      topology decision. Inspect :attr:`backend` and :attr:`fallback_reason`
      for the outcome.
    - ``backend="nvlink"``: force the fused kernel; raises on every rank
      (before any IPC/JIT for topology failures) when it cannot be used.
    - ``backend="pcie"``: explicitly enable the experimental single-node PCIe
      transport at world size 1/2/4/8. One rank is an identity path, two
      ranks use CUDA P2P, and four/eight ranks prefer an all-RDMA route
      (every peer's payload over the rank's mlx5 QP) with all-P2P fallback;
      ``FLASHINFER_ULYSSES_PCIE_ROUTE`` (``auto``/``p2p``/``rdma``/``hybrid``)
      forces all-P2P, all-RDMA at any multi-rank world size, or the
      eight-rank 4+4 NUMA hybrid (same-NUMA CUDA P2P plus cross-NUMA mlx5).
      Multi-rank outputs are registered up front with :meth:`allocate_output`
      (there are no staging workspaces); ``"auto"`` reaches this backend only
      behind the opt-in above.
    - ``backend="nccl"``: force the ``dist.all_to_all_single`` path; skips
      the topology/NVML probe and all IPC/JIT entirely (the constructor
      still resolves and guards the CUDA device and performs CUDA-backed
      metadata collectives over ``group``). Supports any world size.

    ``nvlink_kernel="portable"`` forces the portable CUDA kernel on NVLink;
    ``"cake"`` requires an available Cake export. All ranks must request the
    same backend and NVLink kernel policy. With ``world_size > 1`` the
    NCCL and NVLink backends require ``group`` to support CUDA all-to-all (an
    NCCL process group); this is checked at construction before any resource
    is armed.
    ``world_size == 1`` is a passthrough: both collectives return the input
    tensor unchanged (no copy).

    The NVLink backend launches one of two kernels on the same IPC staging
    and signal allocations, decided once per communicator inside the NVLink
    initialization (every rank takes the same route; :attr:`nvlink_kernel`
    reports it): the exported Cake programs of
    ``flashinfer.cake_ulysses_backend`` when the device's compute capability
    is one that package delivers for this dtype and world size (SM100 /
    SM103), else FlashInfer's portable NVLink kernel. On the Cake route, a
    collective whose ``out`` was registered with :meth:`allocate_output`
    writes every rank's blocks straight into the peers' registered outputs
    instead of passing them through the staging buffer; the two Cake kernels
    use disjoint words of one signal allocation and may alternate call by
    call. The raw entry points (``init_ulysses_a2a`` / ``ulysses_a2a`` /
    ``dispose_ulysses_a2a``) always drive the portable kernel; a
    communicator's own collectives never mix the portable and the Cake
    kernels on its signal words.

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
      with every dim positive, at most ``2**31 - 1`` elements and at most
      ``max_bytes`` bytes. :meth:`exchange_chunks` also accepts packed uint8
      with a per-call ``dtype`` override;
      :meth:`scatter_heads` additionally requires ``H % world_size == 0`` and
      :meth:`gather_heads` requires ``S_global % world_size == 0``.
      PCIe additionally accepts int8/uint8 and the float8 storage types
      (any 1-, 2- or 4-byte element); its mlx5 routes (hybrid and all-RDMA)
      additionally require batch size 1 and ``H * D * element_size <=
      65_535`` bytes, while the all-P2P routes take any batch size.
    - Multi-rank PCIe calls require an explicit ``out`` returned by
      :meth:`allocate_output`. This keeps registration lifetime and overwrite
      points explicit; pre-register one output per live result and geometry.
    - A rejection before native code runs (a bad operand, a refused capture,
      a second stream) is decided from the call's own arguments, so under the
      SPMD contract every rank rejects it: the RDMA routes raise the ordinary
      error and stay usable, while the all-P2P route, whose barrier has no
      abort protocol, poisons teardown and enters BROKEN. A failure inside
      the transport is fail-stop on every route: registered buffers, queue
      pairs and the GPU epoch counters may already be half-mutated, so the
      communicator enters BROKEN and permits only a collective :meth:`close`.
      On the RDMA routes it publishes a sticky abort that releases the peers
      and ``close()`` completes; on all-P2P ``close()`` refuses to synchronize
      and reports that the process must exit.
    - Each rank may use a different CUDA device (e.g. ``cuda:rank``); ranks
      must agree on ``max_bytes``, ``dtype`` and ``backend``.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        Process group of the Ulysses ranks. Defaults to ``dist.group.WORLD``.
    max_bytes : int, optional
        Per-rank byte capacity of one communication operand, shared across
        dtypes and transforms. Sizes the NVLink staging buffer once at
        construction. For quantized QKV this must cover the complete packed
        Q/K/V payload, including scales and alignment, rather than one
        floating input. Statistics and other workspace allocations are
        separate; this is not a total memory budget. The maximum is
        ``(2**31 - 1) * 4`` bytes, and each call separately enforces the
        int32 element-index limit.
    max_elems : int, optional
        Legacy element capacity. Converted to bytes using ``dtype.itemsize``.
        Provide either ``max_elems`` or ``max_bytes``.
    dtype : torch.dtype
        Default element type of operands (float16 / bfloat16 / float32; PCIe
        additionally int8 / uint8 / float8_e4m3fn / float8_e5m2); enforced on
        every call that does not name its own.
    backend : str
        ``"auto"`` | ``"nvlink"`` | ``"pcie"`` | ``"nccl"`` (see above).
    nvlink_kernel : str
        ``"auto"`` | ``"cake"`` | ``"portable"``; relevant when NVLink is used.
        ``"portable"`` names the in-tree CUDA kernel's architecture coverage,
        not a slow path: it uses the optimized vector mover when eligible.
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
        max_bytes: Optional[int] = None,
        dtype: torch.dtype,
        backend: str = "auto",
        nvlink_kernel: str = "auto",
        device: Optional[Union[torch.device, str, int]] = None,
        max_elems: Optional[int] = None,
    ):
        r"""Construct a Ulysses communicator.

        Parameters
        ----------
        group : Optional[ProcessGroup], optional
            Process group spanning the participating ranks. ``None`` uses
            ``torch.distributed.group.WORLD``.
        max_bytes : int, optional
            Per-rank upper bound in bytes on one communication operand.
            Quantized QKV requires the complete packed payload to fit.
            Used to size the backend workspace.
        max_elems : int, optional
            Legacy upper bound in elements, converted with ``dtype.itemsize``.
            Mutually exclusive with ``max_bytes``.
        dtype : torch.dtype
            Default element dtype for collective operands. Must be one of
            ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``;
            ``backend="pcie"`` additionally accepts ``torch.int8``,
            ``torch.uint8``, ``torch.float8_e4m3fn`` and
            ``torch.float8_e5m2``.
        backend : str, default = "auto"
            Backend selection policy. ``"auto"`` probes topology and prefers
            NVLink when supported, then the experimental PCIe transport where
            NVLink is unavailable and
            ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`` is set on every
            rank, otherwise falls back to NCCL. ``"nvlink"``
            forces the NVLink backend and raises if unavailable. ``"pcie"``
            explicitly enables the experimental single-node 1/2/4/8-rank
            PCIe transport. ``"nccl"`` forces the NCCL path.
        nvlink_kernel : str, default = "auto"
            ``"auto"`` uses Cake when an export exists, otherwise portable.
            ``"portable"`` forces the portable NVLink kernel; ``"cake"``
            requires a Cake export. Portable includes the optimized vector
            mover when eligible. All ranks must select the same value.
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
        # the Cake package handle of the NVLink path (None: portable kernel or
        # not armed) and the exported architecture it was bound for
        self._cake: Optional["cake_ulysses_backend.CakeUlyssesA2A"] = None
        self._cake_arch: Optional[str] = None
        self._portable_max_blocks: Optional[int] = None
        # whether the Cake route also writes registered outputs directly, and
        # the registered outputs: this rank's base device pointer -> the CPU
        # int64 peer table of every rank's allocation of the same registration
        self._cake_direct = False
        self._nvlink_outputs: Dict[int, torch.Tensor] = {}
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
        capacity_bytes, capacity_error = self._normalize_capacity(
            max_bytes, max_elems, dtype
        )
        config = self._encode_config(
            capacity_bytes, dtype, device, backend, nvlink_kernel, capacity_error
        )
        configs = self._gather(config)
        self._validate_configs_jointly(configs)

        self.max_bytes = int(capacity_bytes)
        self.dtype = dtype
        self._nvlink_kernel_policy = nvlink_kernel

        # ---- backend selection: strictly before any IPC/JIT -----------------
        # topology_decision is what the probe concluded; decision is the
        # *effective* backend after runtime initialization (they differ only
        # when NVLink init failed at runtime and auto fell back to NCCL).
        self.topology_decision: UlyssesBackendDecision = resolve_ulysses_backend(
            backend, group=group, device=self.device
        )
        self.decision: UlyssesBackendDecision = self.topology_decision
        self.backend = self.decision.backend
        if nvlink_kernel == "cake" and self.backend != "nvlink":
            raise ValueError(
                "nvlink_kernel='cake' requires an available NVLink backend"
            )
        self.fallback_reason = (
            self.decision.reason
            if self.backend == "nccl" and backend != "nccl"
            else None
        )
        self.transport = None

        # The NCCL backend and NVLink's exchange_chunks both run
        # dist.all_to_all_single on the group, so it must move CUDA tensors.
        # Checked before anything is armed (a raise here leaks nothing) and
        # deterministic in the (identical) group object, so group-uniform.
        if self.backend != "pcie" and self.world_size > 1:
            supported, observed = self._group_supports_cuda_alltoall()
            if not supported:
                raise ValueError(
                    f"the Ulysses {self.backend.upper()} backend requires a process "
                    f"group supporting CUDA all-to-all (nccl), got '{observed}'"
                )

        if self.backend == "nvlink":
            err = self._nvlink_init_transaction()
            if err is not None:
                # all ranks cleaned up (verified group-wide by the staged
                # cleanup) and hold the same joint error
                if backend == "nvlink" or nvlink_kernel == "cake":
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

        self._state = _OPEN

    @property
    def max_elems(self) -> int:
        """Legacy capacity in elements of the communicator's construction dtype."""
        return self.max_bytes // self.dtype.itemsize

    @staticmethod
    def _normalize_capacity(max_bytes, max_elems, dtype):
        """Normalize both public capacity forms without raising before the rank gather."""
        if max_elems is None:
            if max_bytes is None:
                return None, "provide either max_bytes or max_elems"
            return max_bytes, ""
        if max_bytes is not None:
            return max_bytes, "provide only one of max_bytes or max_elems"
        if type(max_elems) is not int:
            return (
                None,
                f"max_elems must be a positive int, got {type(max_elems).__name__}",
            )
        if max_elems <= 0:
            return None, f"max_elems must be a positive int, got {max_elems}"
        if max_elems > _INT32_MAX:
            return None, f"max_elems must be at most {_INT32_MAX}, got {max_elems}"
        if not isinstance(dtype, torch.dtype):
            return None, "max_elems requires a valid dtype"
        return max_elems * dtype.itemsize, ""

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
    def _encode_config(
        cls, max_bytes, dtype, device, backend, nvlink_kernel="auto", capacity_error=""
    ) -> Tuple[str, ...]:
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
        policy = (
            nvlink_kernel
            if type(nvlink_kernel) is str
            else f"<invalid type: {type(nvlink_kernel).__name__}>"
        )
        return (nbytes, dt, dev, bk, policy, capacity_error)

    def _validate_configs_jointly(self, configs) -> None:
        problems = {}
        for r, (nbytes, dt, dev, bk, policy, capacity_error) in enumerate(configs):
            # The whitelist comes from the GATHERED backend, never from a
            # local argument: every rank must compute the same verdict from
            # the same list, or one rank raises while its peers hang in the
            # next collective.
            dtypes = _PCIE_SUPPORTED_DTYPES if bk == "pcie" else _SUPPORTED_DTYPES
            supported = tuple(str(d) for d in dtypes)
            errs = []
            if capacity_error:
                errs.append(capacity_error)
            else:
                if not nbytes.isdigit() or int(nbytes) <= 0:
                    errs.append(f"max_bytes must be a positive int, got {nbytes}")
                elif int(nbytes) > _MAX_CAPACITY_BYTES:
                    errs.append(
                        f"max_bytes must be at most {_MAX_CAPACITY_BYTES} (int32 "
                        f"kernel index range at the widest supported element), "
                        f"got {nbytes}"
                    )
            if dt not in supported:
                errs.append(f"dtype must be one of {supported}, got {dt}")
            if not dev.startswith("cuda"):
                errs.append(f"device must be a CUDA device, got {dev}")
            if policy not in ("auto", "cake", "portable"):
                errs.append(
                    f"nvlink_kernel must be auto, cake, or portable, got {policy}"
                )
            if errs:
                problems[r] = "; ".join(errs)
        if problems:
            raise ValueError(f"invalid UlyssesCommunicator config by rank: {problems}")
        shared = {
            (nbytes, dt, policy) for (nbytes, dt, _dev, _bk, policy, _error) in configs
        }
        if len(shared) > 1:
            raise ValueError(
                f"inconsistent UlyssesCommunicator configs across ranks: "
                f"(max_bytes, dtype, nvlink_kernel) = {sorted(shared)}; all ranks must agree"
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

        For ``backend="nvlink"`` on the exported Cake route (SM100 / SM103,
        :attr:`nvlink_kernel` ``"cake"``) the layout transforms register the
        output the same way: every rank allocates ``max_bytes`` bytes with
        native CUDA allocation and maps every peer's allocation over CUDA IPC.
        A :meth:`scatter_heads` or :meth:`gather_heads` whose ``out=`` is such
        an output then writes every rank's blocks straight into the
        destination ranks' outputs, with no pass through the staging buffer.
        Every rank must pass its output of the same registration (the same
        position in the sequence of ``allocate_output`` calls). The memory
        belongs to the communicator and is freed by :meth:`close`; the
        returned tensor must not be used afterwards. Any other output, and
        every output of the portable NVLink kernel, keeps the staged path.

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
            communicator dtype. For ``exchange_chunks``, NCCL/NVLink also
            accept float16, bfloat16, float32 and uint8; other overrides
            require PCIe. The allocation keeps the communicator's byte budget,
            so a narrower dtype holds
            proportionally more elements.

        Returns
        -------
        torch.Tensor
            Tensor with the shape ``op`` produces from ``x``, on the same
            device and dtype as ``x``. On multi-rank pcie, and on the
            multi-rank NVLink Cake route for ``scatter_heads`` and
            ``gather_heads``, it stays registered until :meth:`close`.
        """
        if op not in ("scatter_heads", "gather_heads", "exchange_chunks"):
            raise ValueError(
                "op must be 'scatter_heads', 'gather_heads' or 'exchange_chunks'"
            )
        self._validate(x, op, dtype)
        shape, mode = self._output_geometry(x, op)
        if (
            self.backend == "nvlink"
            and self.world_size > 1
            and self._cake is not None
            and self._cake_direct
            and op != "exchange_chunks"
        ):
            return self._nvlink_allocate_output(x, op, shape)
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

    def _nvlink_allocate_output(
        self, x: torch.Tensor, op: str, shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Register one output of the NVLink Cake direct route (collective).

        Two joint stages, each with a gathered outcome, so a rank-local failure
        raises the same error on every rank instead of deadlocking it. Nothing
        is launched and no registration is committed before both succeed, so a
        failed call leaves the communicator usable; the allocation and the peer
        mappings it made are released by :meth:`close` with the others.
        """
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "allocate_output cannot run inside a CUDA graph capture: it "
                    "is collective and reads the group's agreement back on the "
                    "host. Call it for every geometry before capturing."
                )
        capacity_elems = self.max_bytes // x.dtype.itemsize
        # stage 1: bind the direct programs (JIT-building them when needed),
        # allocate this rank's output at capacity and export its IPC handle
        pointer = None
        handle = None
        try:
            from .cuda_ipc import cudart

            with torch.cuda.device(self.device):
                self._cake.bind_direct()
                allocation = cudart.cudaMalloc(self.max_bytes)
                pointer = allocation.value
                self._exports.append(pointer)
                handle = cudart.cudaIpcGetMemHandle(allocation)
            outcome: Tuple[str, Any] = ("ok", (op, tuple(x.shape), str(x.dtype)))
        except Exception as e:  # noqa: BLE001
            outcome = (
                "err",
                f"rank {self.rank} NVLink {op} output allocation: "
                f"{type(e).__name__}: {e}",
            )
        gathered = self._gather((outcome, handle))
        statuses = [item[0] for item in gathered]
        err = self._first_error(statuses)
        if err is not None:
            raise RuntimeError(f"NVLink output registration failed: {err}")
        if any(status[1] != statuses[0][1] for status in statuses):
            raise ValueError(
                f"rank-inconsistent NVLink output geometry: {[s[1] for s in statuses]}"
            )

        # stage 2: map every peer's allocation and build the output's peer table
        try:
            from .cuda_ipc import cudart

            peers: List[int] = [0] * self.world_size
            with torch.cuda.device(self.device):
                for i, (_status, peer_handle) in enumerate(gathered):
                    if i == self.rank:
                        peers[i] = pointer
                        continue
                    mapped = cudart.cudaIpcOpenMemHandle(peer_handle)
                    self._imports.append(mapped.value)
                    peers[i] = mapped.value
                out_peers = cake_ulysses_backend.cake_backend.peer_table(peers)
                view = self._cake.output_view(pointer, capacity_elems)
            outcome2: Tuple[str, ...] = ("ok",)
        except Exception as e:  # noqa: BLE001
            outcome2 = (
                "err",
                f"rank {self.rank} NVLink {op} output IPC open: {type(e).__name__}: {e}",
            )
        err = self._first_error(self._gather(outcome2))
        if err is not None:
            raise RuntimeError(f"NVLink output registration failed: {err}")

        self._nvlink_outputs[pointer] = out_peers
        # the allocation holds max_bytes; hand back the view this call requested
        return view.narrow(0, 0, int(torch.Size(shape).numel())).view(shape)

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
        if _storage_ranges_overlap(x, out):
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
                # The all-P2P barrier has no abort protocol: a peer may spin on
                # a barrier this rank never completes, so teardown must not
                # synchronize or unmap. The RDMA routes published the sticky
                # abort natively and keep close() usable.
                self._pcie_python_teardown_safe = False
            self._raise_pcie_broken(reason)
        return out

    def _pcie_collective(
        self,
        x,
        out,
        op: str,
        dtype: Optional[torch.dtype] = None,
        workspace: Optional[UlyssesWorkspace] = None,
    ) -> torch.Tensor:
        """Run one multi-rank PCIe collective under its failure envelope.

        A failure before native code runs (validation, capture detection,
        stream binding, wrapper dispatch) is decided from this call's own
        arguments, so under the SPMD contract every rank rejects the same call
        and no peer has entered the barrier: the RDMA routes raise the ordinary
        error and stay OPEN. The all-P2P barrier has no abort protocol at all,
        so a peer that did enqueue could spin forever; that route poisons
        teardown and enters BROKEN instead. A failure inside native code has
        already been handled by _pcie_exchange (the RDMA routes publish the
        sticky abort there and keep close() usable), so it passes through
        untouched.
        """
        try:
            if workspace is not None:
                raise ValueError(
                    f"multi-rank PCIe {op} takes no workspace=: the transport has "
                    "no staging buffers, outputs come from allocate_output()"
                )
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
            self._validate_out(out, shape, op, dtype)
            self._validate_no_overlap(x, out, op)
            if not capturing:
                # Like PcieIpcAllReduceWorkspace, the communicator is bound to
                # the stream of its first collective; peers order their copies
                # against this rank's caller stream, so silently accepting a
                # second stream could let a consumer race the next exchange.
                # Bound only once the call is known to be valid, so a rejected
                # call does not pin the stream.
                current = torch.cuda.current_stream(self.device)
                if self._pcie_stream is None:
                    self._pcie_stream = current
                elif current != self._pcie_stream:
                    raise RuntimeError(
                        "PCIe Ulysses collectives are bound to the stream of "
                        "their first call; use one stream per communicator"
                    )
            return self._pcie_exchange(x, out, mode)
        except Exception as e:  # noqa: BLE001
            # Still OPEN means the failure happened before native code ran;
            # _pcie_exchange has already moved a native failure to BROKEN.
            if self._state == _OPEN and self.transport == "p2p":
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
        # The Cake package route is decided here too: its libraries are
        # loaded (JIT-built when needed) in this stage so a build failure is a
        # gathered outcome, and the signal allocation below is sized for both
        # kernels (they share it; each uses the words from zero).
        try:
            from .vllm_ar import meta_size

            with torch.cuda.device(self.device):
                module = get_ulysses_a2a_module()
                sig_bytes = int(meta_size()) * _ULYSSES_SIGNAL_SLOTS
                dtype_slot = {
                    torch.float32: 0,
                    torch.float16: 1,
                    torch.bfloat16: 2,
                }[self.dtype]
                local_cap = min(
                    _ULYSSES_MAX_BLOCKS,
                    int(module.ulysses_a2a_safe_blocks(self.world_size, dtype_slot)),
                )
                cake_arch = (
                    None
                    if self._nvlink_kernel_policy == "portable"
                    else _cake_nvlink_route(
                        self.device.index, self.dtype, self.world_size
                    )
                )
                if self._nvlink_kernel_policy == "cake" and cake_arch is None:
                    raise RuntimeError(
                        "Cake Ulysses kernel is unavailable on this device"
                    )
                cake_direct = False
                if cake_arch is not None:
                    _cake_nvlink_warm(cake_arch, self.dtype, self.world_size)
                    sig_bytes = max(sig_bytes, cake_ulysses_backend.SIGNAL_BYTES)
                    cake_direct = _cake_nvlink_direct(
                        cake_arch, self.device.index, self.dtype, self.world_size
                    )
            outcome: Tuple[str, ...] = (
                "ok",
                str(sig_bytes),
                cake_arch or "",
                "direct" if cake_direct else "",
                str(local_cap),
            )
        except Exception as e:  # noqa: BLE001
            outcome = ("err", f"rank {self.rank} JIT/meta: {type(e).__name__}: {e}")
        gathered_j = self._gather(outcome)
        err = self._first_error(gathered_j)
        if err is not None:
            return err  # nothing allocated anywhere yet
        routes = sorted({(o[2], o[3]) for o in gathered_j})
        if len(routes) != 1:
            # one kernel per group: a mixed route would pair two protocols on
            # the shared signal words
            return (
                "ranks disagree on the NVLink kernel route "
                f"(exported architectures per rank: "
                f"{[(o[2] or 'portable') + (' + direct' if o[3] else '') for o in gathered_j]})"
            )
        self._cake_arch = cake_arch
        # Every rank launches the same number of barrier-participating CTAs,
        # including when the group contains GPUs with different capacities.
        self._portable_max_blocks = min(int(o[4]) for o in gathered_j)
        self._cake_direct = cake_direct

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
                self._fa = _init_ulysses_a2a_configured(
                    out_ptrs,
                    sig_ptrs,
                    self.rank,
                    self.world_size,
                    True,
                    signal_slots=_ULYSSES_SIGNAL_SLOTS,
                    max_blocks=self._portable_max_blocks,
                )
                if self._cake_arch is not None:
                    # the package route: the same staging and signal
                    # allocations, bound once as the kernels' peer tables and
                    # this rank's signal view; its signal words are zeroed on
                    # the current stream before the synchronize below
                    self._cake = _cake_nvlink_handle(
                        staging_ptrs=out_ptrs,
                        signal_ptrs=sig_ptrs,
                        rank=self.rank,
                        world_size=self.world_size,
                        dtype=self.dtype,
                        device_index=self.device.index,
                    )
                    self._cake.zero_signals()
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
        if self._fa is None and self._cake is None:
            return (0, None)
        try:
            with torch.cuda.device(self.device):
                if self._cake is not None:
                    # the package handle holds bound arguments only (no device
                    # memory of its own); drop them before the allocations go
                    self._cake.dispose()
                    self._cake = None
                if self._fa is not None:
                    dispose_ulysses_a2a(self._fa)
                    # ledger update inside the guard: a __exit__ raise after a
                    # successful dispose must not lead to a double-delete on retry
                    self._fa = None
            return (0, None)
        except Exception as e:  # noqa: BLE001
            return (
                0 if (self._fa is None and self._cake is None) else 1,
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
        self._cake_arch = None
        self._portable_max_blocks = None
        self._cake_direct = False
        self._nvlink_outputs = {}
        self._nvlink_armed = False
        self._state = _CLOSED

    @property
    def nvlink_kernel(self) -> Optional[str]:
        r"""Which kernel the NVLink backend launches: ``"cake"`` (the exported
        Cake programs of ``flashinfer.cake_ulysses_backend``), ``"portable"``
        (FlashInfer's portable NVLink kernel) or ``None`` when the NVLink
        backend is not armed (NCCL backend, ``world_size == 1``, closed)."""
        if not self._nvlink_armed:
            return None
        return "cake" if self._cake is not None else "portable"

    def _nvlink_a2a(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        B: int,
        S_local: int,
        H: int,
        D: int,
        mode: int,
    ) -> None:
        """One NVLink collective into ``out`` on the current stream, through the
        route the init transaction armed (the Cake package writes ``out``
        completely; the portable kernel stages and copies out). An ``out`` that
        :meth:`allocate_output` registered takes the Cake direct route: every
        rank writes its blocks straight into the peers' outputs."""
        if self._cake is not None:
            out_peers = self._nvlink_outputs.get(out.data_ptr())
            if out_peers is not None:
                self._cake.launch_direct(inp, out, out_peers, B, S_local, H, D, mode)
            else:
                self._cake.launch(inp, out, B, S_local, H, D, mode)
            return
        ulysses_a2a(self._fa, inp, out, B, S_local, H, D, mode)

    def __enter__(self) -> "UlyssesCommunicator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Warn only: teardown is collective and cannot run from a finalizer
        # (unlike workspace_base.AllReduceFusionWorkspace, which destroys
        # here). Unarmed communicators (pure NCCL, world size 1) hold nothing
        # peer-shared and stay silent. getattr defaults cover objects whose
        # __init__ never ran.
        if sys.is_finalizing():
            return
        state = getattr(self, "_state", _CLOSED)
        armed = getattr(self, "_nvlink_armed", False) or getattr(
            self, "_pcie_armed", False
        )
        if state != _CLOSED and armed:
            warnings.warn(
                f"UlyssesCommunicator ({getattr(self, 'backend', '?')}, {state}) was "
                "garbage-collected without close(); its peer-shared resources leak "
                "until process exit. Call close() on every rank or use the context "
                "manager.",
                ResourceWarning,
                stacklevel=2,
            )

    @flashinfer_api
    def create_workspace(self, *, max_elems: Optional[int] = None) -> UlyssesWorkspace:
        r"""Allocate reusable NCCL send/receive staging buffers.

        This operation is rank-local and non-collective. ``max_elems``
        defaults to the communicator capacity and may be smaller when a
        caller knows the maximum chunk size it will communicate. The
        multi-rank pcie backend has no staging buffers and rejects this call;
        its outputs are registered with :meth:`allocate_output` instead.

        Parameters
        ----------
        max_elems : int, optional
            Capacity of each send and receive staging buffer in elements.
            ``None`` fills the communicator's ``max_bytes`` capacity at the
            communicator dtype.

        Returns
        -------
        UlyssesWorkspace
            A workspace on the communicator's device with its dtype.
        """
        if self._state != _OPEN:
            raise RuntimeError(
                "create_workspace called on a "
                f"{self._state} UlyssesCommunicator (use-after-close)"
            )
        if self.backend == "pcie" and self.world_size > 1:
            raise NotImplementedError(
                "create_workspace is not available on the multi-rank pcie backend: "
                "the PCIe transport has no send/receive staging buffers; register "
                "outputs with allocate_output() instead"
            )
        capacity_elems = self.max_bytes // self.dtype.itemsize
        if max_elems is None:
            max_elems = capacity_elems
        if type(max_elems) is not int or max_elems <= 0:
            raise ValueError(f"max_elems must be a positive int, got {max_elems!r}")
        if max_elems > capacity_elems:
            raise ValueError(
                f"workspace max_elems={max_elems} exceeds the communicator "
                f"capacity max_bytes={self.max_bytes} ({capacity_elems} elements "
                f"of {self.dtype.itemsize})"
            )
        return UlyssesWorkspace(
            max_elems=max_elems, dtype=self.dtype, device=self.device
        )

    @flashinfer_api
    def prepare_qkv(
        self,
        qkv_shape: Sequence[int],
        *,
        quantization: str = "sage2",
        used_sequence: Optional[int] = None,
    ) -> UlyssesQKVWorkspace:
        r"""Collectively prepare Sage2 QKV communication and reusable storage.

        Every rank must call this method in the same order, outside CUDA
        graph capture. Shape/configuration, layout/JIT availability, and
        allocation outcomes are agreed in three separate stages. A
        recoverable preparation failure releases only this call's storage;
        it does not close the communicator or invalidate existing workspaces.
        Missing participants and CUDA/process-group failures are not recoverable
        through this protocol.

        This method does not change ``backend``. Quantized QKV always uses
        the group's CUDA NCCL backend, even when ordinary scatter/gather
        uses NVLink. The experimental PCIe backend is not supported by this
        prepared API. Supports homogeneous SM90 or SM120 groups of 2/4/8 ranks.

        Parameters
        ----------
        qkv_shape : Sequence[int]
            Common local ``[B, L, H, D]`` shape, with positive dimensions,
            ``D`` equal to 64 or 128, and ``H`` divisible by the group size.
            Each Q/K/V and the combined byte payload must fit the int32
            element-index limit. The complete packed QKV payload, including
            scales and alignment, must fit ``max_bytes``. Statistics are
            allocated separately.
        quantization : str, default = "sage2"
            QKV encoding policy. Only ``"sage2"`` is supported: grouped INT8
            Q/K and channel-scaled FP8 E4M3 V, using the device's Sage2 layout.
            The communicator dtype must be FP16 or BF16.
        used_sequence : int, optional
            Global live prefix length ``U`` in ``(0, world_size * L]``.
            ``None`` uses the whole sequence. Callers must zero input rows
            outside this prefix; this method does not initialize inputs.

        Returns
        -------
        UlyssesQKVWorkspace
            Owner-bound, read-only geometry with reusable send/receive and
            statistics-gather buffers. Shape or live-length changes require
            another collective preparation call.
        """
        # Validate inside the outcome envelope: a bad argument on one rank
        # must not skip a collective that every other rank is entering.
        outcome: Tuple[Any, ...]
        try:
            self._require_open("prepare_qkv")
            if self.backend == "pcie":
                raise ValueError(
                    "Sage2 QKV requires the nccl or nvlink communicator backend"
                )
            if type(qkv_shape) not in (tuple, list, torch.Size):
                raise TypeError(
                    "qkv_shape must be a tuple/list/torch.Size of four ints"
                )
            shape = tuple(qkv_shape)
            if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("qkv_shape must contain four positive integers")
            batch, local_sequence, heads, head_dim = shape
            if type(quantization) is not str or quantization != "sage2":
                raise ValueError("quantization must be 'sage2'")
            if self.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    "Sage2 QKV requires a float16 or bfloat16 communicator"
                )
            if self.world_size not in (2, 4, 8):
                raise ValueError("Sage2 QKV requires world_size in {2, 4, 8}")
            if head_dim not in (64, 128) or heads % self.world_size:
                raise ValueError(
                    "Sage2 QKV requires D in {64, 128} and H % world_size == 0"
                )
            if math.prod(shape) > _INT32_MAX:
                raise ValueError("each Q/K/V must fit the int32 element-index range")
            used = (
                self.world_size * local_sequence
                if used_sequence is None
                else used_sequence
            )
            if (
                type(used) is not int
                or not 0 < used <= self.world_size * local_sequence
            ):
                raise ValueError(
                    "used_sequence must be an integer in (0, world_size * L]"
                )
            supported, observed = self._group_supports_cuda_alltoall()
            if not supported:
                raise ValueError(
                    f"Sage2 QKV requires a CUDA NCCL process group, got {observed}"
                )
            outcome = (
                "ok",
                (
                    shape,
                    quantization,
                    used,
                    str(self.dtype),
                    self.world_size,
                    self.max_bytes,
                ),
            )
        except Exception as e:  # noqa: BLE001 — all ranks must vote before raising
            outcome = (
                "err",
                f"rank {self.rank} QKV configuration: {type(e).__name__}: {e}",
            )
        outcomes = self._gather(outcome)
        err = self._first_error(outcomes)
        if err is not None:
            raise ValueError(f"prepare_qkv failed: {err}")
        if any(item != outcomes[0] for item in outcomes[1:]):
            raise ValueError(
                f"inconsistent prepare_qkv configuration across ranks: {outcomes}"
            )

        try:
            from . import _ulysses_lowp as lowp

            with torch.cuda.device(self.device):
                cap = lowp.capability(self.device)
                if (
                    cap["device_capability"] not in ((9, 0), (12, 0))
                    or not cap["supported"]
                ):
                    raise RuntimeError(
                        f"Sage2 QKV kernels unavailable on SM90/SM120: {cap}"
                    )
                if head_dim not in cap["supported_head_dims"]:
                    raise RuntimeError(
                        f"Sage2 QKV module does not support D={head_dim}"
                    )
                layout_impl = getattr(lowp, cap["layout_class"])(head_dim=head_dim)
                spec = layout_impl.payload_spec(
                    batch_size=batch,
                    local_sequence=local_sequence,
                    num_heads=heads,
                    world_size=self.world_size,
                )
                payload_bytes = self.world_size * int(spec["chunk_bytes"])
                self._validate_capacity(
                    payload_bytes, "prepare_qkv packed payload", dtype=torch.uint8
                )
            layout = (
                "sage2_sm90"
                if cap["device_capability"] == (9, 0)
                else "sage2_sm89_sm120"
            )
            outcome = (
                "ok",
                (
                    cap["device_capability"],
                    layout,
                    layout_impl.Q_GROUP,
                    layout_impl.K_GROUP,
                ),
            )
        except Exception as e:  # noqa: BLE001 — JIT failure is a group outcome
            outcome = (
                "err",
                f"rank {self.rank} QKV layout/JIT: {type(e).__name__}: {e}",
            )
        outcomes = self._gather(outcome)
        err = self._first_error(outcomes)
        if err is not None:
            raise RuntimeError(f"prepare_qkv failed: {err}")
        if any(item != outcomes[0] for item in outcomes[1:]):
            raise ValueError(
                f"inconsistent QKV hardware/layout across ranks: {outcomes}"
            )

        workspace = None
        try:
            workspace = UlyssesQKVWorkspace._allocate(
                self, shape, used, layout, layout_impl
            )
            outcome = ("ok",)
        except Exception as e:  # noqa: BLE001 — release only this attempted allocation
            outcome = (
                "err",
                f"rank {self.rank} QKV allocation: {type(e).__name__}: {e}",
            )
        err = self._first_error(self._gather(outcome))
        if err is not None:
            workspace = None
            raise RuntimeError(f"prepare_qkv failed: {err}")
        return workspace

    @flashinfer_api(trace=ulysses_scatter_qkv_trace_dispatch)
    def scatter_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        workspace: UlyssesQKVWorkspace,
        out: Optional[UlyssesQKV] = None,
    ) -> UlyssesQKV:
        r"""Quantize and exchange QKV for the prepared SageAttention2 layout.

        Runs one FP32 statistics AllGather and one jointly packed uint8
        AllToAll on ``group``, using NCCL independently of ``backend``.
        Execution uses the current CUDA stream; all ranks must issue the
        same sequence of calls with valid inputs. There is no per-call
        metadata collective, implicit preparation, or precision fallback.
        Local statistics still allocate temporary tensors.

        Parameters
        ----------
        q, k, v : torch.Tensor
            FP16/BF16 CUDA tensors matching the prepared ``[B,L,H,D]`` shape,
            dtype and device. D must be contiguous; the base pointer and
            each outer stride must preserve 16-byte row alignment. Projection
            views satisfying those constraints do not require copies. Input
            rows outside the prepared global live prefix must already be zero.
        workspace : UlyssesQKVWorkspace
            Workspace returned by this communicator's ``prepare_qkv``.
            Must not be shared by concurrent operations or another communicator.
        out : UlyssesQKV, optional
            Reuse these six output tensors. Shapes, dtypes, device, layout,
            logical/live lengths and input dtype must match the preparation.
            Outputs must be contiguous and cannot overlap each other, the
            inputs, or workspace storage. ``None`` allocates independent
            final storage that remains valid after later workspace reuse.

        Returns
        -------
        UlyssesQKV
            Pre-quantized Q/K/V and three FP32 scales, with consumer metadata.
            Returns ``out`` itself when supplied. Attention uses the live
            prefix; the caller restores a full-length floating output before
            invoking ``gather_heads``.
        """
        from . import _ulysses_lowp as lowp

        self._require_open("scatter_qkv")
        if (
            not isinstance(workspace, UlyssesQKVWorkspace)
            or workspace._owner is not self
        ):
            raise ValueError(
                "scatter_qkv workspace must be prepared by this communicator"
            )
        lowp._validate_qkv(q, k, v)
        if (
            tuple(q.shape) != workspace.qkv_shape
            or q.dtype != workspace.dtype
            or q.device != workspace.device
        ):
            raise ValueError(
                "Q/K/V shape, dtype and device must match the prepared workspace"
            )
        # Re-read each tensor's extent once per call, then compare integers.
        # Repeated pairwise metadata queries are costly on this hot path;
        # a cross-call cache would miss resized or rebound tensor storage.
        checked_ranges = [_qkv_byte_range(x) for x in (q, k, v)]
        staging = (
            (workspace._send_buffer, workspace._wire_shape, torch.uint8),
            (workspace._recv_buffer, workspace._wire_shape, torch.uint8),
            (
                workspace._stats_gather,
                (workspace.world_size * workspace._stats_numel,),
                torch.float32,
            ),
        )
        for buffer, shape, dtype in staging:
            lowp._validate_tensor_spec(
                "QKV workspace buffer", buffer, shape, dtype, self.device
            )
            start, end = _qkv_byte_range(buffer)
            if any(
                start < other_end and other_start < end
                for other_start, other_end in checked_ranges
            ):
                raise ValueError(
                    "QKV workspace buffers must not overlap inputs or one another"
                )
            checked_ranges.append((start, end))
        result = self._prepare_qkv_out(workspace, out, checked_ranges, lowp)
        layout = workspace._layout_impl
        _batch, local_sequence, heads, _head_dim = workspace.qkv_shape
        local_heads = heads // self.world_size
        with torch.cuda.device(self.device):
            with torch.cuda.nvtx.range("lowp_local_stats"):
                send, ctx = lowp._local_stats_impl(
                    q,
                    k,
                    v,
                    self.rank,
                    self.world_size,
                    workspace.used_sequence,
                    layout.Q_GROUP,
                    layout.K_GROUP,
                    workspace._enable_pdl,
                    workspace._module,
                )
            with torch.cuda.nvtx.range("lowp_stats_allgather"):
                dist.all_gather_into_tensor(
                    workspace._stats_gather, send, group=self.group
                )
            with torch.cuda.nvtx.range("lowp_finalize_stats"):
                stats = lowp._finalize_stats_impl(
                    workspace._stats_gather,
                    ctx,
                    k,
                    workspace._enable_pdl,
                    workspace._module,
                )
            with torch.cuda.nvtx.range("lowp_quant_pack"):
                lowp._quant_and_pack_impl(
                    q,
                    k,
                    v,
                    stats,
                    workspace._send_buffer,
                    workspace._q_scale_offset,
                    workspace._enable_pdl,
                    workspace._module,
                )
            with torch.cuda.nvtx.range("lowp_input_a2a"):
                self._exchange_qkv_payload(
                    workspace._send_buffer, workspace._recv_buffer
                )
            with torch.cuda.nvtx.range("lowp_unpack"):
                lowp._unpack_for_sage_impl(
                    workspace._recv_buffer,
                    result[:5],
                    local_sequence,
                    self.world_size,
                    workspace.used_sequence,
                    workspace._enable_pdl,
                    workspace._module,
                )
            with torch.cuda.nvtx.range("lowp_v_scale"):
                head_start = self.rank * local_heads
                result.v_scale.copy_(
                    stats.v_scale_global[:, head_start : head_start + local_heads]
                )
        return result

    def _exchange_qkv_payload(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        # The two-dimensional kernel payload is already destination-major.
        # These views preserve storage; exchange_chunks supplies the shared
        # transport contract and validates the current byte budget.
        self.exchange_chunks(
            send.view(1, 1, *send.shape),
            out=recv.view(1, 1, *recv.shape),
            dtype=torch.uint8,
        )

    def _prepare_qkv_out(self, workspace, out, checked_ranges, lowp) -> UlyssesQKV:
        metadata = (
            workspace.layout,
            workspace.logical_sequence,
            workspace.used_sequence,
            workspace.dtype,
        )
        if out is None:
            tensors = tuple(
                torch.empty(shape, dtype=dtype, device=self.device)
                for shape, dtype in workspace._output_specs
            )
            return UlyssesQKV(
                tensors[0],
                tensors[1],
                tensors[2],
                tensors[3],
                tensors[4],
                tensors[5],
                workspace.layout,
                workspace.logical_sequence,
                workspace.used_sequence,
                workspace.dtype,
            )
        if not isinstance(out, UlyssesQKV):
            raise TypeError("out must be a UlyssesQKV result")
        if out[6:] != metadata:
            raise ValueError(
                "out layout, logical_sequence, used_sequence and input_dtype must match"
            )
        for index, (tensor, (shape, dtype)) in enumerate(
            zip(out[:6], workspace._output_specs, strict=True)
        ):
            lowp._validate_tensor_spec(
                f"out.{UlyssesQKV._fields[index]}", tensor, shape, dtype, self.device
            )
            start, end = _qkv_byte_range(tensor)
            if any(
                start < other_end and other_start < end
                for other_start, other_end in checked_ranges
            ):
                raise ValueError(
                    "out tensors must not overlap inputs, workspace storage or each other"
                )
            checked_ranges.append((start, end))
        return out

    # ---- collectives -----------------------------------------------------------

    @flashinfer_api(trace=ulysses_scatter_heads_trace)
    def scatter_heads(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        workspace: Optional[UlyssesWorkspace] = None,
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
            Preallocated contiguous output with shape
            ``[B, S_local * world_size, H // world_size, D]``. Supplying it
            removes the public output allocation. It must not alias ``x``
            when ``world_size > 1``. Multi-rank PCIe requires an output
            returned by :meth:`allocate_output`; on the NVLink Cake route
            such an output takes the direct path.
        dtype : torch.dtype, optional
            Element type for this call, overriding the communicator dtype.
            pcie backend only. ``max_bytes`` is denominated in bytes, so a
            narrower dtype here simply fits more elements in the same
            capacity.
        workspace : UlyssesWorkspace, optional
            Reusable NCCL pack/receive storage. Supplying it removes the two
            NCCL staging allocations. It is validated but unused by the
            NVLink backend, which owns IPC staging internally, and rejected
            by the multi-rank pcie backend, which has no staging buffers.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_global, H_local, D]`` on the same device
            and dtype as ``x``.
        """
        if self.backend == "pcie" and self.world_size > 1:
            return self._pcie_collective(
                x, out, "scatter_heads", dtype, workspace=workspace
            )
        self._validate(x, "scatter_heads", dtype)
        output_shape, _mode = self._output_geometry(x, "scatter_heads")
        # ulysses_a2a is parameterized by the [B, S_local, H, D] layout, which
        # is this operand's own shape for scatter_heads.
        B, S_local, H, D = x.shape
        out = self._prepare_out(x, out, output_shape, "scatter_heads")
        self._validate_workspace(workspace, x.numel(), "scatter_heads")
        self._validate_workspace_out_alias(out, workspace, "scatter_heads")
        if self.world_size == 1:
            if out is None or out is x:
                return x
            out.copy_(x)
            return out
        if self.backend == "nvlink":
            if out is None:
                out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            self._nvlink_a2a(x, out, B, S_local, H, D, 0)
            return out
        return self._nccl_scatter_heads(x, out=out, workspace=workspace)

    @flashinfer_api
    def scatter_qkv_head_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        head_offset: int,
        head_count: int,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[UlyssesWorkspace] = None,
    ) -> torch.Tensor:
        r"""Pack and scatter one Q/K/V head band with one collective.

        ``query``, ``key`` and ``value`` have shape
        ``[B, S_local, H, D]`` and may be independent positive-strided views.
        The output is a fused ``[B, S_global, head_count, 3 * D]`` payload;
        slicing its final dimension produces the three attention operands.

        The method is a transport primitive, not a pipeline scheduler. A
        caller that overlaps input and output communication must use separate
        communicators and workspaces. Runs on the current CUDA stream.

        When the NCCL backend, ``B == 1``, ``out is None`` and a workspace is
        supplied, the returned tensor aliases the workspace receive buffer and
        remains valid only until that workspace is reused. Pass ``out`` for an
        explicit lifetime.

        Parameters
        ----------
        query : torch.Tensor
            Positive-strided CUDA tensor with shape ``[B, S_local, H, D]``.
        key : torch.Tensor
            Positive-strided CUDA tensor with the same shape, dtype, and
            device as ``query``.
        value : torch.Tensor
            Positive-strided CUDA tensor with the same shape, dtype, and
            device as ``query``.
        head_offset : int
            Start of the selected band within each destination rank's
            ``H // world_size`` local heads.
        head_count : int
            Number of consecutive local heads in the selected band.
        out : torch.Tensor, optional
            Preallocated contiguous result with shape
            ``[B, S_local * world_size, head_count, 3 * D]``. It must not
            alias ``query``, ``key``, or ``value``.
        workspace : UlyssesWorkspace, optional
            Reusable staging storage large enough for the fused Q/K/V
            payload. A workspace cannot service concurrent collectives.

        Returns
        -------
        torch.Tensor
            Fused Q/K/V payload with shape
            ``[B, S_global, head_count, 3 * D]``.
        """
        from .ulysses_head_chunk import (
            _launch_pack_qkv,
            _validate_qkv_geometry,
        )

        self._require_open("scatter_qkv_head_chunk")
        if self.backend == "pcie" and self.world_size > 1:
            raise NotImplementedError(
                "scatter_qkv_head_chunk is not available on the multi-rank pcie "
                "backend: the PCIe transport moves whole registered outputs; pack "
                "the band and use scatter_heads with out= from allocate_output()"
            )
        B, S_local, local_heads, D, payload_elems = _validate_qkv_geometry(
            query,
            key,
            value,
            world_size=self.world_size,
            head_offset=head_offset,
            head_count=head_count,
        )
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.device != self.device:
                raise ValueError(
                    f"scatter_qkv_head_chunk {name} is on {tensor.device}, but "
                    f"this communicator is bound to {self.device}"
                )
            if tensor.dtype != self.dtype:
                raise ValueError(
                    f"scatter_qkv_head_chunk {name} dtype {tensor.dtype} does "
                    f"not match communicator dtype {self.dtype}"
                )
        self._validate_capacity(payload_elems, "scatter_qkv_head_chunk")
        self._validate_workspace(workspace, payload_elems, "scatter_qkv_head_chunk")
        output_shape = (
            B,
            S_local * self.world_size,
            head_count,
            3 * D,
        )
        out = self._prepare_out(query, out, output_shape, "scatter_qkv_head_chunk")
        self._validate_workspace_out_alias(
            out,
            workspace,
            "scatter_qkv_head_chunk",
            allow_recv_alias=(self.backend == "nccl" and B == 1),
        )
        if out is not None and any(
            _storage_ranges_overlap(out, tensor) for tensor in (query, key, value)
        ):
            raise ValueError(
                "scatter_qkv_head_chunk out must not alias query, key, or value"
            )

        if self.world_size == 1:
            if out is None:
                out = torch.empty(output_shape, dtype=query.dtype, device=query.device)
            _launch_pack_qkv(
                out,
                query,
                key,
                value,
                world_size=1,
                local_heads=local_heads,
                head_offset=head_offset,
                head_count=head_count,
                nccl_layout=False,
            )
            return out

        if workspace is None:
            send_storage = torch.empty(
                payload_elems, dtype=self.dtype, device=self.device
            )
            recv_storage = None
        else:
            send_storage = workspace._send_buffer[:payload_elems]
            recv_storage = workspace._recv_buffer[:payload_elems]

        if self.backend == "nvlink":
            packed = send_storage.view(B, S_local, self.world_size * head_count, 3 * D)
            _launch_pack_qkv(
                packed,
                query,
                key,
                value,
                world_size=self.world_size,
                local_heads=local_heads,
                head_offset=head_offset,
                head_count=head_count,
                nccl_layout=False,
            )
            if out is None:
                out = torch.empty(output_shape, dtype=self.dtype, device=self.device)
            self._nvlink_a2a(
                packed,
                out,
                B,
                S_local,
                self.world_size * head_count,
                3 * D,
                0,
            )
            return out

        send = send_storage.view(self.world_size, B, S_local, head_count, 3 * D)
        if recv_storage is None:
            recv = torch.empty_like(send)
        else:
            recv = recv_storage.view_as(send)
        _launch_pack_qkv(
            send,
            query,
            key,
            value,
            world_size=self.world_size,
            local_heads=local_heads,
            head_offset=head_offset,
            head_count=head_count,
            nccl_layout=True,
        )
        dist.all_to_all_single(recv, send, group=self.group)
        recv_as_output = recv.view(output_shape) if B == 1 else None
        if out is None and recv_as_output is not None:
            return recv_as_output
        if out is None:
            out = torch.empty(output_shape, dtype=self.dtype, device=self.device)
        if recv_as_output is not None and out.data_ptr() == recv.data_ptr():
            return out
        out.view(B, self.world_size, S_local, head_count, 3 * D).copy_(
            recv.permute(1, 0, 2, 3, 4)
        )
        return out

    @flashinfer_api(trace=ulysses_exchange_chunks_trace)
    def exchange_chunks(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
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
            PCIe accepts its supported storage types; NCCL/NVLink accept
            float16, bfloat16, float32 and uint8. No conversion is performed.

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
        *,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        workspace: Optional[UlyssesWorkspace] = None,
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
            Preallocated contiguous output with shape
            ``[B, S_global // world_size, H_local * world_size, D]``.
            Supplying it removes the public output allocation. It must not
            alias ``x`` when ``world_size > 1``. Multi-rank PCIe requires an
            output returned by :meth:`allocate_output`; on the NVLink Cake
            route such an output takes the direct path.
        dtype : torch.dtype, optional
            Element type for this call, overriding the communicator dtype.
            pcie backend only. ``max_bytes`` is denominated in bytes, so a
            narrower dtype here simply fits more elements in the same
            capacity.
        workspace : UlyssesWorkspace, optional
            Reusable NCCL pack/receive storage. Supplying it removes the two
            NCCL staging allocations. It is validated but unused by the
            NVLink backend, and rejected by the multi-rank pcie backend,
            which has no staging buffers.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_local, H, D]`` on the same device and
            dtype as ``x``.
        """
        if self.backend == "pcie" and self.world_size > 1:
            return self._pcie_collective(
                x, out, "gather_heads", dtype, workspace=workspace
            )
        self._validate(x, "gather_heads", dtype)
        output_shape, _mode = self._output_geometry(x, "gather_heads")
        # ulysses_a2a is parameterized by the [B, S_local, H, D] layout, which
        # is the result's shape for gather_heads.
        B, S_local, H, D = output_shape
        out = self._prepare_out(x, out, output_shape, "gather_heads")
        self._validate_workspace(workspace, x.numel(), "gather_heads")
        self._validate_workspace_out_alias(out, workspace, "gather_heads")
        if self.world_size == 1:
            if out is None or out is x:
                return x
            out.copy_(x)
            return out
        if self.backend == "nvlink":
            if out is None:
                out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            self._nvlink_a2a(x, out, B, S_local, H, D, 1)
            return out
        return self._nccl_gather_heads(x, out=out, workspace=workspace)

    @flashinfer_api
    def gather_output_head_chunk(
        self,
        x: torch.Tensor,
        *,
        local_heads: int,
        head_offset: int,
        out: torch.Tensor,
        workspace: Optional[UlyssesWorkspace] = None,
    ) -> torch.Tensor:
        r"""Gather one attention-output head band into a full destination.

        ``x`` has shape ``[B, S_global, head_count, D]``. ``out`` is the
        preallocated full output ``[B, S_local, world_size * local_heads, D]``.
        Only the selected head band is written; calling this method for every
        non-overlapping band in a complete schedule reconstructs ordinary
        whole-head Ulysses output communication.

        The NCCL path packs the sequence split directly into reusable staging
        and merges the received head band directly into ``out``. The NVLink
        path uses its fused-transpose collective followed by the same merge
        primitive. Runs on the current CUDA stream.

        Parameters
        ----------
        x : torch.Tensor
            Positive-strided CUDA tensor with shape
            ``[B, S_global, head_count, D]``.
        local_heads : int
            Total number of attention-output heads owned by this rank before
            the output all-to-all.
        head_offset : int
            Start of this band within the rank's ``local_heads``.
        out : torch.Tensor
            Preallocated contiguous destination with shape
            ``[B, S_global // world_size, world_size * local_heads, D]``.
            Only this head band is modified.
        workspace : UlyssesWorkspace, optional
            Reusable staging storage large enough for ``x``. A workspace
            cannot service concurrent collectives.

        Returns
        -------
        torch.Tensor
            ``out`` after merging the gathered head band.
        """
        from .ulysses_head_chunk import (
            _launch_merge_rank_major,
            _launch_pack_output_sequence,
            _positive_int,
            _nonnegative_int,
            _validate_cuda_tensor,
        )

        self._require_open("gather_output_head_chunk")
        if self.backend == "pcie" and self.world_size > 1:
            raise NotImplementedError(
                "gather_output_head_chunk is not available on the multi-rank pcie "
                "backend: the PCIe transport moves whole registered outputs; use "
                "gather_heads with out= from allocate_output()"
            )
        local_heads = _positive_int(local_heads, "local_heads")
        head_offset = _nonnegative_int(head_offset, "head_offset")
        x = _validate_cuda_tensor(
            x, "gather_output_head_chunk input", ndim=4, contiguous=False
        )
        B, S_global, head_count, D = x.shape
        if x.device != self.device or x.dtype != self.dtype:
            raise ValueError(
                f"gather_output_head_chunk input device/dtype "
                f"({x.device}, {x.dtype}) must match communicator "
                f"({self.device}, {self.dtype})"
            )
        if S_global % self.world_size != 0:
            raise ValueError(
                f"global sequence length {S_global} must be divisible by "
                f"world size {self.world_size}"
            )
        if head_offset + head_count > local_heads:
            raise ValueError(
                f"head band [{head_offset}, {head_offset + head_count}) "
                f"exceeds local_heads={local_heads}"
            )
        S_local = S_global // self.world_size
        output_shape = (B, S_local, self.world_size * local_heads, D)
        out = self._prepare_out(x, out, output_shape, "gather_output_head_chunk")
        if out is None:  # ``out`` is required by the public signature.
            raise TypeError("gather_output_head_chunk requires out")
        if _storage_ranges_overlap(out, x):
            raise ValueError("gather_output_head_chunk out must not alias the input")
        payload_elems = x.numel()
        self._validate_capacity(payload_elems, "gather_output_head_chunk")
        self._validate_workspace(workspace, payload_elems, "gather_output_head_chunk")
        self._validate_workspace_out_alias(out, workspace, "gather_output_head_chunk")

        if self.world_size == 1:
            _launch_merge_rank_major(
                x.unsqueeze(0),
                out,
                world_size=1,
                local_heads=local_heads,
                head_offset=head_offset,
            )
            return out

        if workspace is None:
            send_storage = torch.empty(
                payload_elems, dtype=self.dtype, device=self.device
            )
            recv_storage = torch.empty_like(send_storage)
        else:
            send_storage = workspace._send_buffer[:payload_elems]
            recv_storage = workspace._recv_buffer[:payload_elems]

        if self.backend == "nvlink":
            if x.is_contiguous():
                contiguous_input = x
            else:
                contiguous_input = send_storage.view_as(x)
                contiguous_input.copy_(x)
            compact = recv_storage.view(B, S_local, self.world_size * head_count, D)
            self._nvlink_a2a(
                contiguous_input,
                compact,
                B,
                S_local,
                self.world_size * head_count,
                D,
                1,
            )
            rank_major = compact.view(
                B, S_local, self.world_size, head_count, D
            ).permute(2, 0, 1, 3, 4)
        else:
            send = send_storage.view(self.world_size, B, S_local, head_count, D)
            recv = recv_storage.view_as(send)
            _launch_pack_output_sequence(
                send,
                x,
                world_size=self.world_size,
            )
            dist.all_to_all_single(recv, send, group=self.group)
            rank_major = recv

        _launch_merge_rank_major(
            rank_major,
            out,
            world_size=self.world_size,
            local_heads=local_heads,
            head_offset=head_offset,
        )
        return out

    # ---- NCCL fallback ---------------------------------------------------------
    # The conventional all_to_all_single path with explicit permute/contiguous
    # glue before and after (exactly the data movement the fused NVLink kernel
    # folds into its cross-GPU writes). Bit-identical to the NVLink backend.

    def _nccl_scatter_heads(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor],
        workspace: Optional[UlyssesWorkspace],
    ) -> torch.Tensor:
        B, S_local, H, D = x.shape
        W = self.world_size
        H_local = H // W
        # Preserve the pre-existing allocation/layout path byte-for-byte when
        # destination passing is not requested. This keeps the default API's
        # behavior and performance independent of the new opt-in feature.
        if out is None and workspace is None:
            xt = (
                x.reshape(B, S_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
            )
            recv = torch.empty_like(xt)
            dist.all_to_all_single(recv, xt, group=self.group)
            # chunk j == rank j's contribution to sequence block j
            return recv.permute(1, 0, 2, 3, 4).reshape(B, W * S_local, H_local, D)

        if out is None:
            out = torch.empty(
                B,
                W * S_local,
                H_local,
                D,
                dtype=x.dtype,
                device=x.device,
            )
        if workspace is None:
            send = torch.empty(
                W, B, S_local, H_local, D, dtype=x.dtype, device=x.device
            )
            recv = torch.empty_like(send)
        else:
            send = workspace._send_buffer[: x.numel()].view(W, B, S_local, H_local, D)
            recv = workspace._recv_buffer[: x.numel()].view(W, B, S_local, H_local, D)
        send.copy_(x.reshape(B, S_local, W, H_local, D).permute(2, 0, 1, 3, 4))
        dist.all_to_all_single(recv, send, group=self.group)
        out.view(B, W, S_local, H_local, D).copy_(recv.permute(1, 0, 2, 3, 4))
        return out

    def _nccl_gather_heads(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor],
        workspace: Optional[UlyssesWorkspace],
    ) -> torch.Tensor:
        B, S_global, H_local, D = x.shape
        W = self.world_size
        S_local = S_global // W
        if out is None and workspace is None:
            xt = (
                x.reshape(B, W, S_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
            )
            recv = torch.empty_like(xt)
            dist.all_to_all_single(recv, xt, group=self.group)
            # chunk p == this rank's sequence block, head slice p
            return (
                recv.permute(1, 2, 0, 3, 4)
                .reshape(B, S_local, W * H_local, D)
                .contiguous()
            )

        if out is None:
            out = torch.empty(
                B,
                S_local,
                W * H_local,
                D,
                dtype=x.dtype,
                device=x.device,
            )
        if workspace is None:
            send = torch.empty(
                W, B, S_local, H_local, D, dtype=x.dtype, device=x.device
            )
            recv = torch.empty_like(send)
        else:
            send = workspace._send_buffer[: x.numel()].view(W, B, S_local, H_local, D)
            recv = workspace._recv_buffer[: x.numel()].view(W, B, S_local, H_local, D)
        send.copy_(x.reshape(B, W, S_local, H_local, D).permute(1, 0, 2, 3, 4))
        dist.all_to_all_single(recv, send, group=self.group)
        out.view(B, S_local, W, H_local, D).copy_(recv.permute(1, 2, 0, 3, 4))
        return out

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

    def _validate_capacity(
        self,
        required_elems: int,
        op: str,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if required_elems > _INT32_MAX:
            raise ValueError(
                f"{op} payload has {required_elems} elements, over the int32 "
                f"index range {_INT32_MAX}"
            )
        element_dtype = self.dtype if dtype is None else dtype
        required_bytes = required_elems * element_dtype.itemsize
        if required_bytes > self.max_bytes:
            raise ValueError(
                f"{op} payload is {required_bytes} bytes ({required_elems} "
                f"elements of {element_dtype.itemsize}), exceeding the "
                f"communicator capacity max_bytes={self.max_bytes}"
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
            # PCIe output registration agrees on (op, shape, dtype); NCCL
            # chunk callers must supply matching shapes and dtypes on all ranks.
            allowed: Tuple[torch.dtype, ...]
            if self.backend == "pcie":
                allowed = _PCIE_SUPPORTED_DTYPES
            elif op == "exchange_chunks":
                allowed = _CHUNK_DTYPES
            else:
                raise ValueError(
                    f"{op} per-call dtype is only supported on the pcie "
                    f"backend, not {self.backend}; only exchange_chunks "
                    "accepts a dtype override on this backend"
                )
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

    def _prepare_out(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor],
        expected_shape: Tuple[int, ...],
        op: str,
    ) -> Optional[torch.Tensor]:
        if out is None:
            return None
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"{op} out expects a torch.Tensor, got {type(out).__name__}"
            )
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"{op} out has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.device != x.device:
            raise ValueError(f"{op} out is on {out.device}, but input is on {x.device}")
        if out.dtype != x.dtype:
            raise ValueError(
                f"{op} out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if not out.is_contiguous():
            raise ValueError(f"{op} out must be contiguous")
        if _storage_ranges_overlap(out, x):
            if self.world_size > 1 or out.data_ptr() != x.data_ptr():
                raise ValueError(f"{op} out must not alias the input")
        return out

    def _validate_workspace(
        self,
        workspace: Optional[UlyssesWorkspace],
        required_elems: int,
        op: str,
    ) -> None:
        if workspace is None:
            return
        if not isinstance(workspace, UlyssesWorkspace):
            raise TypeError(
                f"{op} workspace expects UlyssesWorkspace, got "
                f"{type(workspace).__name__}"
            )
        if workspace.device != self.device:
            raise ValueError(
                f"{op} workspace is on {workspace.device}, but communicator "
                f"is bound to {self.device}"
            )
        if workspace.dtype != self.dtype:
            raise ValueError(
                f"{op} workspace dtype {workspace.dtype} does not match "
                f"communicator dtype {self.dtype}"
            )
        if workspace.max_elems < required_elems:
            raise ValueError(
                f"{op} requires {required_elems} workspace elements, but "
                f"workspace capacity is {workspace.max_elems}"
            )
        for name, buffer in (
            ("send", workspace._send_buffer),
            ("recv", workspace._recv_buffer),
        ):
            if (
                buffer.device != self.device
                or buffer.dtype != self.dtype
                or not buffer.is_contiguous()
                or buffer.numel() < workspace.max_elems
            ):
                raise ValueError(f"{op} workspace {name} buffer was modified")

    @staticmethod
    def _validate_workspace_out_alias(
        out: Optional[torch.Tensor],
        workspace: Optional[UlyssesWorkspace],
        op: str,
        *,
        allow_recv_alias: bool = False,
    ) -> None:
        if out is None or workspace is None:
            return
        if _storage_ranges_overlap(out, workspace._send_buffer):
            raise ValueError(f"{op} out must not alias the workspace send buffer")
        if _storage_ranges_overlap(out, workspace._recv_buffer):
            exact_recv_alias = out.data_ptr() == workspace._recv_buffer.data_ptr()
            if not (allow_recv_alias and exact_recv_alias):
                raise ValueError(
                    f"{op} out must not alias the workspace receive buffer"
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
    """Names of rdma-core build inputs this machine does not provide.

    Cheap and side-effect free (compiles nothing), so an environment guard can
    distinguish an unsupported host from a real build failure. It probes the
    shared libraries and headers required by the shared P2P/RDMA translation
    unit. An older rdma-core may still fail at JIT compile time.

    Returns
    -------
    List[str]
        Missing library and header names; empty when all are found.
    """
    import ctypes.util

    missing = [
        f"lib{name}"
        for name in ("ibverbs", "mlx5")
        if ctypes.util.find_library(name) is None
    ]
    include_roots = [
        "/usr/include",
        "/usr/local/include",
        "/usr/local/cuda/include",
        f"{sys.prefix}/include",
    ]
    for variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        include_roots.extend(
            filter(None, os.environ.get(variable, "").split(os.pathsep))
        )
    for header in ("infiniband/verbs.h", "infiniband/mlx5dv.h"):
        if not any(
            os.path.isfile(os.path.join(root, header)) for root in include_roots
        ):
            missing.append(header)
    return missing


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

    @register_custom_op(
        "flashinfer::init_ulysses_a2a_with_signal_slots", mutates_args=[]
    )
    def init_ulysses_a2a_with_signal_slots(
        out_ipc_ptrs: List[int],
        signal_ipc_ptrs: List[int],
        rank: int,
        world_size: int,
        full_nvlink: bool,
        signal_slots: int,
        max_blocks: int,
    ) -> int:
        return module.init_ulysses_a2a_with_signal_slots(
            out_ipc_ptrs,
            signal_ipc_ptrs,
            rank,
            world_size,
            full_nvlink,
            signal_slots,
            max_blocks,
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
        init_ulysses_a2a_with_signal_slots=init_ulysses_a2a_with_signal_slots,
        ulysses_a2a_safe_blocks=module.ulysses_a2a_safe_blocks,
        dispose_ulysses_a2a=dispose_ulysses_a2a,
        ulysses_a2a=ulysses_a2a,
    )


def _init_ulysses_a2a_configured(
    out_ipc_ptrs: List[int],
    signal_ipc_ptrs: List[int],
    rank: int,
    world_size: int,
    full_nvlink: bool,
    *,
    signal_slots: int,
    max_blocks: Optional[int],
) -> int:
    """Internal geometry-aware initializer used by the communicator."""
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
    if signal_slots not in (1, _ULYSSES_SIGNAL_SLOTS):
        raise ValueError("signal_slots must be 1 or 4")
    if signal_slots == 1 and max_blocks is not None:
        raise ValueError("max_blocks requires signal_slots=4")
    module = get_ulysses_a2a_module()
    if signal_slots == 1:
        fa = module.init_ulysses_a2a(
            out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
        )
    else:
        fa = module.init_ulysses_a2a_with_signal_slots(
            out_ipc_ptrs,
            signal_ipc_ptrs,
            rank,
            world_size,
            full_nvlink,
            signal_slots,
            36 if max_blocks is None else max_blocks,
        )
    # The signal memset is asynchronous. Complete it before the caller's
    # group-wide barrier, and release ownership if synchronization fails.
    try:
        torch.cuda.synchronize()
    except Exception:
        with contextlib.suppress(Exception):
            module.dispose_ulysses_a2a(fa)
        raise
    return fa


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
    return _init_ulysses_a2a_configured(
        out_ipc_ptrs,
        signal_ipc_ptrs,
        rank,
        world_size,
        full_nvlink,
        signal_slots=1,
        max_blocks=None,
    )


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
