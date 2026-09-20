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
import re
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..api_logging import flashinfer_api
from ..jit.comm import gen_ulysses_a2a_module
from ..trace.templates.comm import (
    ulysses_exchange_chunks_trace,
    ulysses_scatter_qkv_trace_dispatch,
)
from ..utils import register_custom_op
from .ulysses_topology import (
    SUPPORTED_WORLD_SIZES,
    UlyssesBackendDecision,
    resolve_ulysses_backend,
)

_INT32_MAX = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CHUNK_DTYPES = (*_SUPPORTED_DTYPES, torch.uint8)
_MAX_CAPACITY_BYTES = _INT32_MAX * max(dtype.itemsize for dtype in _SUPPORTED_DTYPES)

# communicator lifecycle states; CLOSED is only reached after a fully
# successful teardown so a failed close() can be retried
_OPEN, _CLOSING, _CLOSED = "open", "closing", "closed"


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
    ``S_global = S_local * world_size``. Both backends produce bit-identical
    results.

    :meth:`exchange_chunks` exchanges already-packed destination-major
    chunks. Both current backends use NCCL for this operation.
    :meth:`prepare_qkv` and :meth:`scatter_qkv` additionally provide Sage2
    INT8/FP8 QKV preparation, using this chunk exchange and a statistics
    AllGather on the same NCCL group.

    Backend selection for ordinary scatter/gather happens in the constructor,
    strictly before any IPC
    allocation or JIT compilation (see
    :func:`~flashinfer.comm.resolve_ulysses_backend`):

    - ``backend="auto"``: the fused-transpose NVLink-P2P kernel when the group
      is a verified single-node all-pairs NVLink mesh with a supported world
      size (2/4/6/8); NCCL otherwise — including when NVLink runtime
      initialization fails after a positive topology decision. Inspect
      :attr:`backend` and :attr:`fallback_reason` for the outcome.
    - ``backend="nvlink"``: force the fused kernel; raises on every rank
      (before any IPC/JIT for topology failures) when it cannot be used.
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
      call it together. :meth:`close` is collective only when the NVLink
      backend was armed (its resources are IPC-shared); for the pure NCCL
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
      NVLink signal buffers assume serialized calls); do not call one
      communicator concurrently from multiple streams or threads.
    - Operand tensors must be contiguous 4-D CUDA tensors of the construction
      ``dtype`` (float16 / bfloat16 / float32) on the construction device,
      with every dim positive, at most ``2**31 - 1`` elements and at most
      ``max_bytes`` bytes. :meth:`exchange_chunks` also accepts packed uint8
      with a per-call ``dtype`` override;
      :meth:`scatter_heads` additionally requires ``H % world_size == 0`` and
      :meth:`gather_heads` requires ``S_global % world_size == 0``.
    - Each rank may use a different CUDA device (e.g. ``cuda:rank``); ranks
      must agree on ``max_bytes``, ``dtype`` and ``backend``.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        Process group of the Ulysses ranks. Defaults to ``dist.group.WORLD``.
    max_bytes : int
        Per-rank byte capacity of one communication operand, shared across
        dtypes and transforms. Sizes the NVLink staging buffer once at
        construction. For quantized QKV this must cover the complete packed
        Q/K/V payload, including scales and alignment, rather than one
        floating input. Statistics and other workspace allocations are
        separate; this is not a total memory budget. The maximum is
        ``(2**31 - 1) * 4`` bytes, and each call separately enforces the
        int32 element-index limit.
    dtype : torch.dtype
        Element type of all operands (float16 / bfloat16 / float32); enforced
        on every call.
    backend : str
        ``"auto"`` | ``"nvlink"`` | ``"nccl"`` (see above).
    device : torch.device or str or int, optional
        CUDA device of this rank; normalized to an explicit index (bare
        ``"cuda"`` means the current device, an int is a CUDA ordinal).
        Defaults to the current CUDA device.

    Examples
    --------
    >>> with UlyssesCommunicator(group, max_bytes=B*S*H*D*2, dtype=torch.bfloat16) as comm:
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
            Per-rank upper bound in bytes on one communication operand.
            Quantized QKV requires the complete packed payload to fit.
            Used to size the backend workspace.
        dtype : torch.dtype
            Element dtype for collective operands. Must be one of
            ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
        backend : str, default = "auto"
            Backend selection policy. ``"auto"`` probes topology and prefers
            NVLink when supported, otherwise falls back to NCCL. ``"nvlink"``
            forces the NVLink backend and raises if unavailable. ``"nccl"``
            forces the NCCL path.
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
        config = self._encode_config(max_bytes, dtype, device)
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
    def _encode_config(cls, max_bytes, dtype, device) -> Tuple[str, str, str]:
        if type(max_bytes) is not int:  # bool is an int subclass: reject it too
            nbytes = f"<invalid type: {type(max_bytes).__name__}>"
        else:
            nbytes = str(max_bytes)
        if isinstance(dtype, torch.dtype):
            dt = str(dtype)
        else:
            dt = f"<invalid type: {type(dtype).__name__}>"
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
        return (nbytes, dt, dev)

    def _validate_configs_jointly(self, configs) -> None:
        supported = tuple(str(d) for d in _SUPPORTED_DTYPES)
        problems = {}
        for r, (nbytes, dt, dev) in enumerate(configs):
            errs = []
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
            if errs:
                problems[r] = "; ".join(errs)
        if problems:
            raise ValueError(f"invalid UlyssesCommunicator config by rank: {problems}")
        shared = {(nbytes, dt) for (nbytes, dt, _dev) in configs}
        if len(shared) > 1:
            raise ValueError(
                f"inconsistent UlyssesCommunicator configs across ranks: "
                f"(max_bytes, dtype) = {sorted(shared)}; all ranks must agree"
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

    def _teardown_protocol(self, *, sync_first: bool) -> Optional[str]:
        stages = []
        if sync_first:
            # collectives/memsets are async enqueues: never unmap while the
            # bound device may still be executing one
            stages.append(("synchronize device", self._try_sync))
        stages.append(("dispose kernel handle", self._try_dispose))
        stages.append(("close peer mappings", self._try_close_imports))
        # exports are freed only after the gathered remaining-import count is
        # zero on EVERY rank: freeing a buffer a peer still has mapped is
        # undefined behavior
        stages.append(("free exports", self._try_free_exports))

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

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        r"""Release the communicator. Idempotent once fully closed.

        Collective when the NVLink backend was armed: every rank must call
        ``close`` together, and every rank runs the same fixed teardown stage
        sequence even if it holds no resources locally — synchronize the
        bound device (collectives are asynchronous kernel launches; unmapping
        a peer buffer still in use would be undefined behavior), dispose the
        kernel handle, close peer mappings, and only after the group confirms
        all mappings are closed, free the exports. Each stage drains with
        bounded group-coordinated retries. If teardown still cannot complete,
        the call raises the same error on **all** ranks and the state stays
        CLOSING; every rank may retry ``close()``. The state becomes CLOSED
        only after a fully successful group-wide teardown. The pure-NCCL
        backend holds no resources and closes locally.
        """
        if self._state == _CLOSED:
            return
        self._state = _CLOSING

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

    @flashinfer_api
    def create_workspace(self, *, max_elems: Optional[int] = None) -> UlyssesWorkspace:
        r"""Allocate reusable NCCL send/receive staging buffers.

        This operation is rank-local and non-collective. ``max_elems``
        defaults to the communicator capacity and may be smaller when a
        caller knows the maximum chunk size it will communicate.

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
        capacity_elems = self.max_bytes // self.dtype.itemsize
        if max_elems is None:
            max_elems = capacity_elems
        if type(max_elems) is not int or max_elems <= 0:
            raise ValueError(f"max_elems must be a positive int, got {max_elems!r}")
        if max_elems > capacity_elems:
            raise ValueError(
                f"workspace max_elems={max_elems} exceeds communicator "
                f"capacity max_bytes={self.max_bytes} ({capacity_elems} elements "
                f"of {self.dtype.itemsize} bytes)"
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
        uses NVLink. Supports homogeneous SM90 or SM120 groups of 2/4/8 ranks.

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

    @flashinfer_api
    def allocate_output(
        self, x: torch.Tensor, op: str, *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        r"""Allocate output storage for one Ulysses transform.

        This is a rank-local allocation for the currently supported backends.
        Call it during setup and pass the result as ``out`` to reuse storage.

        Parameters
        ----------
        x : torch.Tensor
            Contiguous 4-D CUDA operand with the shape consumed by ``op``.
        op : str
            ``"scatter_heads"``, ``"gather_heads"`` or ``"exchange_chunks"``.
        dtype : torch.dtype, optional
            Per-call dtype override for ``exchange_chunks`` only. The input
            must have this dtype; no conversion is performed. In particular,
            packed byte payloads pass ``torch.uint8``.

        Returns
        -------
        torch.Tensor
            Output-shaped storage on the same device and with the same dtype
            as ``x``. The communicator does not own this allocation.
        """
        if op not in ("scatter_heads", "gather_heads", "exchange_chunks"):
            raise ValueError(
                "op must be 'scatter_heads', 'gather_heads' or 'exchange_chunks'"
            )
        self._validate(x, op, dtype)
        shape, _mode = self._output_geometry(x, op)
        return torch.empty(shape, dtype=x.dtype, device=x.device)

    @flashinfer_api(trace=ulysses_exchange_chunks_trace)
    def exchange_chunks(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""``[1, 1, W, C] -> [1, 1, W, C]``: equal-length chunk all-to-all.

        Input chunk ``r`` goes to peer ``r`` and lands in slot ``rank`` of
        that peer's output. The payload is already packed destination-major;
        no layout conversion or quantization is performed. Both current
        backends use NCCL because the fused NVLink kernel implements only
        head-axis layout transforms. All ranks must issue matching shapes,
        dtypes and call order, on their current CUDA streams.

        Parameters
        ----------
        x : torch.Tensor
            Contiguous CUDA tensor shaped ``[1, 1, world_size, chunk]``.
            ``chunk`` counts elements; it counts bytes for a uint8 payload.
        out : torch.Tensor, optional
            Contiguous output of the same shape, dtype and device, with no
            overlap with ``x`` when ``world_size > 1``. May be created using
            :meth:`allocate_output`.
        dtype : torch.dtype, optional
            Override the communicator dtype for this operation: float16,
            bfloat16, float32 or uint8. The input must already have this dtype.
            Omit to use the communicator dtype. The byte budget is unchanged.

        Returns
        -------
        torch.Tensor
            Source-major chunks of the same shape, device and dtype as ``x``.
            With one rank, returns ``x`` when no output is supplied; otherwise
            copies into and returns ``out``.
        """
        self._validate(x, "exchange_chunks", dtype)
        shape, _mode = self._output_geometry(x, "exchange_chunks")
        out = self._prepare_out(x, out, shape, "exchange_chunks")
        if self.world_size == 1:
            if out is None or out is x:
                return x
            out.copy_(x)
            return out
        if self.backend == "nvlink":
            supported, observed = self._group_supports_cuda_alltoall()
            if not supported:
                raise ValueError(
                    "exchange_chunks requires a CUDA NCCL process group, "
                    f"got {observed}"
                )
        if out is None:
            out = torch.empty_like(x)
        dist.all_to_all_single(out.view(-1), x.view(-1), group=self.group)
        return out

    def _output_geometry(self, x: torch.Tensor, op: str) -> Tuple[Tuple[int, ...], int]:
        """Common output shape and native mode for allocation and exchange."""
        batch, sequence, heads, head_dim = x.shape
        if op == "exchange_chunks":
            if batch != 1 or sequence != 1:
                raise ValueError(
                    "exchange_chunks expects [1, 1, world_size, chunk], "
                    f"got shape {tuple(x.shape)}"
                )
            if heads != self.world_size:
                raise ValueError(
                    "exchange_chunks requires one chunk per peer (dim 2 == "
                    f"world size {self.world_size}), got shape {tuple(x.shape)}"
                )
            return (batch, sequence, heads, head_dim), 2
        if op == "scatter_heads":
            if heads % self.world_size:
                raise ValueError(
                    "scatter_heads requires the global head count (dim 2) to be "
                    f"divisible by world size {self.world_size}, got shape "
                    f"{tuple(x.shape)}"
                )
            return (
                batch,
                sequence * self.world_size,
                heads // self.world_size,
                head_dim,
            ), 0
        if sequence % self.world_size:
            raise ValueError(
                "gather_heads requires the global sequence length (dim 1) to "
                f"be divisible by world size {self.world_size}, got shape "
                f"{tuple(x.shape)}"
            )
        return (
            batch,
            sequence // self.world_size,
            heads * self.world_size,
            head_dim,
        ), 1

    @flashinfer_api
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
            when ``world_size > 1``.
        dtype : torch.dtype, optional
            Reserved per-call dtype override. Must be ``None`` for ordinary
            layout transforms on the currently supported backends.
        workspace : UlyssesWorkspace, optional
            Reusable NCCL pack/receive storage. Supplying it removes the two
            NCCL staging allocations. It is validated but unused by the
            NVLink backend, which owns IPC staging internally.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_global, H_local, D]`` on the same device
            and dtype as ``x``.
        """
        self._validate(x, "scatter_heads", dtype)
        B, S_local, H, D = x.shape
        output_shape, _mode = self._output_geometry(x, "scatter_heads")
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
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 0)
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
            ulysses_a2a(
                self._fa,
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

    @flashinfer_api
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
            alias ``x`` when ``world_size > 1``.
        dtype : torch.dtype, optional
            Reserved per-call dtype override. Must be ``None`` for ordinary
            layout transforms on the currently supported backends.
        workspace : UlyssesWorkspace, optional
            Reusable NCCL pack/receive storage. Supplying it removes the two
            NCCL staging allocations. It is validated but unused by the
            NVLink backend.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B, S_local, H, D]`` on the same device and
            dtype as ``x``.
        """
        self._validate(x, "gather_heads", dtype)
        B, S_global, H_local, D = x.shape
        S_local = S_global // self.world_size
        H = H_local * self.world_size
        output_shape, _mode = self._output_geometry(x, "gather_heads")
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
            ulysses_a2a(self._fa, x, out, B, S_local, H, D, 1)
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
            ulysses_a2a(
                self._fa,
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
        if x.device != self.device:
            raise ValueError(
                f"{op} tensor is on {x.device}, but this communicator is bound "
                f"to {self.device}"
            )
        if dtype is not None:
            if op != "exchange_chunks":
                raise ValueError(
                    f"{op} per-call dtype is not supported on the {self.backend} "
                    "backend; only exchange_chunks accepts a dtype override"
                )
            if dtype not in _CHUNK_DTYPES:
                raise ValueError(
                    f"{op} per-call dtype {dtype} is not one of {_CHUNK_DTYPES}"
                )
        expected_dtype = self.dtype if dtype is None else dtype
        if x.dtype != expected_dtype:
            raise ValueError(
                f"{op} tensor dtype {x.dtype} does not match the expected "
                f"dtype {expected_dtype}"
            )
        if not x.is_contiguous():
            raise ValueError(f"{op} tensor must be contiguous")
        if any(s <= 0 for s in x.shape):
            raise ValueError(
                f"{op} tensor dims must all be positive, got shape {tuple(x.shape)}"
            )
        self._validate_capacity(x.numel(), op, expected_dtype)

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
