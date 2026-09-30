"""RoCEnante: one-shot RoCE all-reduce and all-gather runtime for multi-node tensor parallelism.

Designed for DGX Spark clusters, whose integrated GPU can read pinned host
memory in place and whose ConnectX-7 can RDMA-write into the same memory.
Each rank owns one pinned region (see ``_roce_proxy.c``); every all-reduce is
one kernel launch that stages the input, rings the proxy thread, waits for
every peer's RDMA-written payload, and reduces in fixed rank order.

Runtime constraints:

* one all-reduce in flight per runtime (single channel, one stream);
* all-reduce messages up to ``max_size`` bytes and all-gather shards up to
  ``max_gather_bytes``, both multiples of 16 bytes;
* every rank of the exchange group must construct the runtime collectively;
* every peer payload is striped across all configured HCAs, with a completion
  flag per stripe so the GPU consumes the slot only after every HCA delivered.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from . import _allgather_cute
from ._oneshot_cute import PACK_BYTES, get_launcher
from ._proxy import Layout, Proxy, load as _load_proxy_library

logger = logging.getLogger(__name__)

# Bumped when the adapter-visible surface (constructor keywords, methods, their
# semantics) changes incompatibly; integrations pin the value they were built for.
API_VERSION = 1
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_WORLD_SIZES = tuple(range(2, 17))
DEFAULT_MAX_SIZE = 2 * 1024 * 1024
DEFAULT_MAX_GATHER_BYTES = 16 * 1024 * 1024
DEFAULT_THREADS = 512
DEFAULT_BLOCKS = 8
DEFAULT_GID_INDEX = 3
# Polls of a peer flag before the kernel gives up (each poll is a system-scope
# load of host memory, roughly a microsecond): about 20 s.
DEFAULT_SPIN_LIMIT = 20_000_000
_SLOT_ALIGNMENT = 4096
_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _env_list(*names: str) -> tuple[str, ...]:
    """First non-empty comma-separated list among the environment variables ``names``."""
    for name in names:
        raw = os.getenv(name)
        if raw:
            items = []
            for item in raw.split(","):
                item = item.strip().lstrip("=^")
                if item:
                    items.append(item.split(":")[0])
            if items:
                return tuple(items)
    return ()


def _env_int(*names: str, default: int) -> int:
    """First integer-valued environment variable among ``names``, else ``default``."""
    for name in names:
        raw = os.getenv(name)
        if raw:
            return int(raw)
    return default


def default_gid_index() -> int:
    """``B12X_ROCE_GID_INDEX``, else NCCL's ``NCCL_IB_GID_INDEX``, else 3."""

    return _env_int(
        "B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX", default=DEFAULT_GID_INDEX
    )


def discover_hcas(gid_index: Optional[int] = None) -> tuple[str, ...]:
    """Return the RDMA devices to use, at most two.

    ``B12X_ROCE_HCA`` (or NCCL's ``NCCL_IB_HCA``) selects explicitly; otherwise
    every active device with a populated GID at ``gid_index`` is used.
    """

    explicit = _env_list("B12X_ROCE_HCA", "NCCL_IB_HCA")
    if explicit:
        return explicit[:2]
    gid_index = default_gid_index() if gid_index is None else int(gid_index)
    found = []
    root = Path("/sys/class/infiniband")
    for dev in sorted(root.glob("*")):
        state = dev / "ports" / "1" / "state"
        gid = dev / "ports" / "1" / "gids" / str(gid_index)
        try:
            if "ACTIVE" not in state.read_text():
                continue
            if gid.read_text().strip().replace(":", "").strip("0") == "":
                continue
        except OSError:
            continue
        found.append(dev.name)
    return tuple(found[:2])


def is_supported(device: torch.device | int | str | None = None) -> bool:
    """True on an integrated GPU with at least one active RDMA device.

    The kernel reads pinned host memory in place, which needs an integrated
    (unified-memory) GPU such as the DGX Spark GB10.
    """

    if not torch.cuda.is_available():
        return False
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    props = torch.cuda.get_device_properties(index if index is not None else 0)
    if not getattr(props, "is_integrated", False):
        return False
    return len(discover_hcas()) > 0


def _normalize_device(device: torch.device | int | str) -> torch.device:
    """Coerce an int, string, or device into a CUDA ``torch.device`` with an index."""
    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("RoCE all-reduce requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _capture_id(stream: torch.cuda.Stream) -> int:
    """CUDA's id of the capture ``stream`` is in, 0 when it is not capturing.

    Two captures are distinct even on the same stream, and one capture can be
    joined by several streams; the id identifies the capture itself.
    """

    from cuda.bindings import runtime as cudart

    info = cudart.cudaStreamGetCaptureInfo(stream.cuda_stream)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        return 0
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0
    return int(info[2])


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    return (int(value) + alignment - 1) // alignment * alignment


def _grid_blocks(size_packs: int, threads: int, max_blocks: int) -> int:
    """Choose a power-of-two grid with about two 16-byte packs per thread."""
    packs_per_block = 2 * int(threads)
    required = max(1, (int(size_packs) + packs_per_block - 1) // packs_per_block)
    return min(1 << (required - 1).bit_length(), int(max_blocks))


def _exchange(local: object, group: ProcessGroup) -> list[object]:
    """All-gather one picklable object from every rank of ``group``."""
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return gathered


class RoceOneshotAllReduce:
    """One-shot RDMA all-reduce over the DGX Spark 200 GbE fabric."""

    algorithm = "rocenante"

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        gid_index: Optional[int] = None,
        threads: int = DEFAULT_THREADS,
        blocks: int = DEFAULT_BLOCKS,
    ) -> None:
        """Allocate the pinned region, build and connect the proxy, and start it."""
        self.device = _normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self._lock = threading.Lock()
        self._proxy: Optional[Proxy] = None
        self._gather_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._align_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_event = torch.cuda.Event()
        self._last_stream: Optional[torch.cuda.Stream] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._capture_id = 0
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"unsupported RoCE all-reduce world size {self.world_size}; "
                f"supported sizes are {SUPPORTED_WORLD_SIZES}"
            )
        if int(max_size) < PACK_BYTES:
            raise ValueError("max_size must hold at least one 16-byte pack")
        if int(threads) % 32 != 0 or int(threads) < 32 or int(threads) > 1024:
            raise ValueError("threads must be a multiple of 32 between 32 and 1024")
        if int(blocks) < 1 or int(blocks) & (int(blocks) - 1) != 0:
            # The kernels find the last arriving block with a modulo of a
            # free-running 32-bit counter, which survives the counter wrap only
            # for a power-of-two block count.
            raise ValueError("blocks must be a positive power of two")
        self.max_size = int(max_size)
        self.max_gather_bytes = int(max_gather_bytes)
        self._threads = int(threads)
        self._blocks = int(blocks)
        self._counter_classes = self._blocks.bit_length()
        self.gid_index = default_gid_index() if gid_index is None else int(gid_index)
        self.spin_limit = _env_int("B12X_ROCE_SPIN_LIMIT", default=DEFAULT_SPIN_LIMIT)
        names = tuple(hca_names) if hca_names else discover_hcas(self.gid_index)
        if not names:
            raise RuntimeError("no active RDMA device found for the RoCE all-reduce")
        self.hca_names = names[:2]

        slot_bytes = _align_up(
            max(self.max_size, self.max_gather_bytes), _SLOT_ALIGNMENT
        )
        self._layout = Layout(self.world_size, slot_bytes)
        self._slot_bytes = slot_bytes

        with torch.cuda.device(self.device):
            # Pinned, zero-initialised: flags and the control record start at 0
            # and the first sequence number is 1.
            self._region = torch.zeros(
                self._layout.total_bytes, dtype=torch.uint8, pin_memory=True
            )
            # Epoch and poison are global. Stage and tail arrivals are separate
            # for every power-of-two grid size because their free-running
            # counters use the launch grid as their modulus.
            self._counters = torch.zeros(
                2 + 2 * self._counter_classes,
                dtype=torch.int32,
                device=self.device,
            )
        host_ptr = self._region.data_ptr()
        device_ptr = self._device_pointer(host_ptr)
        if device_ptr != host_ptr:
            raise RuntimeError(
                "RoCE all-reduce needs host pointers that are directly device "
                "accessible (integrated GPU with unified addressing)"
            )
        self._recv_base = host_ptr + self._layout.recv_off
        self._flag_base = host_ptr + self._layout.flag_off
        self._send_base = host_ptr + self._layout.send_off
        self._ctrl_base = host_ptr + self._layout.ctrl_off
        # ctrl record (kernel-written): seq, nbytes, error seq, missing peer,
        # nbytes per slot (the proxy uses these when it has to catch up), and
        # the missing HCA index for timeout diagnostics.
        self._ctrl_words = self._region[
            self._layout.ctrl_off : self._layout.ctrl_off + 28
        ].view(torch.int32)
        self._error_word = self._ctrl_words[2:3]
        # numpy view of the control words: reading it costs nanoseconds, so the
        # health check before and after every launch stays off the profile.
        self._ctrl_np = self._ctrl_words.numpy()
        self._epoch_address = self._counters.data_ptr()
        self._poison_address = self._epoch_address + 4 * (1 + 2 * self._counter_classes)

        error: Optional[str] = None
        try:
            self._proxy = Proxy(
                world_size=self.world_size,
                rank=self.rank,
                hca_names=self.hca_names,
                gid_index=self.gid_index,
                region_ptr=host_ptr,
                region_bytes=self._layout.total_bytes,
                slot_bytes=slot_bytes,
            )
            blob = self._proxy.local_blob()
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = str(exc)
            blob = b""
        # Every rank publishes the configuration the protocol depends on; the
        # ranks must agree exactly, and all of them see the same verdict.
        config = {
            "api_version": API_VERSION,
            "proxy_abi": _load_proxy_library().roce_abi_version()
            if error is None
            else None,
            "world_size": self.world_size,
            "hca_count": len(self.hca_names),
            "traffic_class": self._proxy.traffic_class if error is None else None,
            "slot_bytes": slot_bytes,
            "slots": self._layout.slots,
            "flag_stride": self._layout.flag_stride,
            "max_size": self.max_size,
            "max_gather_bytes": self.max_gather_bytes,
            "spin_limit": self.spin_limit,
            "threads": self._threads,
            "blocks": self._blocks,
        }
        statuses = _exchange((error, blob, config), exchange_group)
        failures = [
            f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None
        ]
        if not failures:
            reference = statuses[0][2]
            for i, s in enumerate(statuses):
                differing = {
                    k: (reference[k], s[2].get(k))
                    for k in reference
                    if s[2].get(k) != reference[k]
                }
                if differing:
                    failures.append(
                        f"rank {i} configuration differs from rank 0: {differing}"
                    )
        if failures:
            self.close()
            raise RuntimeError("RoCE all-reduce setup failed: " + "; ".join(failures))
        try:
            self._proxy.connect([s[1] for s in statuses])
            self._proxy.start()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        verdicts = _exchange(error, exchange_group)
        failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
        if failures:
            self.close()
            raise RuntimeError("RoCE all-reduce connect failed: " + "; ".join(failures))
        if self.rank == 0:
            logger.info(
                "RoCEnante ready: world=%d hcas=%s gid_index=%d max_size=%d traffic_class=%d",
                self.world_size,
                ",".join(self.hca_names),
                self.gid_index,
                self.max_size,
                self._proxy.traffic_class,
            )

    @staticmethod
    def _device_pointer(host_ptr: int) -> int:
        """Device address of pinned host memory (``cudaHostGetDevicePointer``)."""
        from cuda.bindings import runtime as cudart

        err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        return int(ptr)

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
        **_ignored: Any,
    ) -> "RoceOneshotAllReduce":
        """Mirror ``comm.pcie.AllReduce.from_exchange_group``; PCIe-only knobs are ignored."""

        capacity = max(int(max_size), int(eager_buffer_bytes or 0))
        return cls(
            exchange_group=exchange_group,
            device=device,
            max_size=capacity,
            max_gather_bytes=max_gather_bytes,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        max_input_bytes: Optional[int] = None,
        **_ignored: Any,
    ) -> "RoceOneshotAllReduce":
        """Build a runtime from a process group used for both rank identity and the setup exchange."""
        capacity = max(int(max_size), int(max_input_bytes or 0))
        return cls(exchange_group=process_group, device=device, max_size=capacity)

    # -- policy -----------------------------------------------------------------

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        """False: peer auxiliary inputs (fused residual paths) are not implemented."""
        return False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Eligible: contiguous CUDA tensor of a supported dtype, 16-byte total size within ``max_size``.

        The decision depends only on dtype, shape, contiguity and size, which
        tensor-parallel ranks share, never on the pointer value; a closed runtime
        raises rather than declining, so no rank can silently fall back alone.
        """

        if self._closed or self._proxy is None:
            raise RuntimeError("RoCE runtime is closed")
        if inp.dtype not in SUPPORTED_DTYPES or not inp.is_cuda:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_size and nbytes % PACK_BYTES == 0

    # -- channels / streams (single channel runtime) ---------------------------

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """No-op: the runtime has a single channel."""
        return None

    def for_stream(self, stream: object = None, *, channel_id: Optional[str] = None):
        """Return this runtime; the collectives take ``stream`` per call."""
        return self

    # -- compilation ------------------------------------------------------------

    def _launcher_key(self, dtype: torch.dtype) -> tuple[object, ...]:
        """Cache key of the all-reduce launcher for ``dtype``."""
        return (
            _DTYPE_NAMES[dtype],
            self.world_size,
            self.rank,
            self._threads,
            self._layout.slots,
            self._layout.flag_stride,
            len(self.hca_names),
            self.device.index,
        )

    def _gather_launcher_key(self) -> tuple[object, ...]:
        """Cache key of the all-gather launcher."""
        return (
            self.world_size,
            self.rank,
            self._threads,
            self._layout.slots,
            self._layout.flag_stride,
            len(self.hca_names),
            self.device.index,
        )

    def _counter_addresses(self, blocks: int) -> tuple[int, int]:
        """Stage and tail counter addresses for one power-of-two grid size."""
        counter_class = int(blocks).bit_length() - 1
        stage = self._epoch_address + 4 * (1 + counter_class)
        tail = self._epoch_address + 4 * (1 + self._counter_classes + counter_class)
        return stage, tail

    def _prepare_resources(
        self,
        dtypes: Sequence[torch.dtype] = (torch.bfloat16,),
        *,
        padded_gather: bool = False,
    ) -> None:
        """Compile the launchers for ``dtypes`` and allocate scratch ahead of capture.

        Always allocates the two ``max_size`` alignment buffers.  The padded
        all-gather scratch (``max_gather_bytes`` plus ``world_size`` times that)
        is allocated only with ``padded_gather=True`` or on first eager use, so
        a workload whose shards have 16-byte rows never pays for it.
        Compilation and allocation are refused inside a capture.
        """

        with torch.cuda.device(self.device):
            for dtype in dtypes:
                get_launcher(*self._launcher_key(dtype))
            _allgather_cute.get_launcher(*self._gather_launcher_key())
            self._aligned_scratch(0, self._region[:PACK_BYTES])
            if padded_gather:
                self._gather_scratch(PACK_BYTES)


    # -- execution ----------------------------------------------------------------

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        plan,
    ) -> torch.Tensor:
        """Run a published prepared RoCE all-reduce."""
        del channel_id, peer_input_ptrs
        from b12x.preparation.types import require_prepared

        prepared = require_prepared(plan, "comm.roce", self.device)
        if prepared.runtime is not self:
            raise ValueError("RoCE plan belongs to another runtime")
        return self._run_prepared_all_reduce(inp, prepared=prepared, out=out, stream=stream)

    def _run_prepared_all_reduce(
        self,
        inp: torch.Tensor,
        *,
        prepared: object,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
    ) -> torch.Tensor:
        """Run a materialized launcher during priming or once the plan is published."""
        with self._lock:
            self.check_health()
            if not self.should_allreduce(inp):
                raise ValueError(
                    "input is not eligible for the RoCE one-shot all-reduce"
                )
            if out is not None and (
                out.shape != inp.shape
                or out.dtype != inp.dtype
                or out.device != inp.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "out must be a contiguous tensor on the input's device matching the input"
                )
            self._launcher_key(inp.dtype)
            nbytes = inp.numel() * inp.element_size()
            context = (
                torch.cuda.stream(stream) if stream is not None else _nullcontext()
            )
            with torch.cuda.device(self.device), context:
                # Capture state belongs to the target stream, so query it here.
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing and inp.dtype not in prepared.reduce_launchers:
                    raise RuntimeError(
                        "RoCE all-reduce dtype was not prepared for CUDA graph capture"
                    )
                try:
                    launcher = prepared.reduce_launchers[inp.dtype]
                except KeyError as exc:
                    raise ValueError("RoCE all-reduce dtype differs from preparation") from exc
                if out is None:
                    out = torch.empty_like(inp)
                src = inp
                if inp.data_ptr() % PACK_BYTES != 0:
                    src = self._aligned_scratch(0, inp)
                    src.copy_(inp)
                dst = (
                    out
                    if out.data_ptr() % PACK_BYTES == 0
                    else self._aligned_scratch(1, out)
                )
                grid_blocks = _grid_blocks(
                    nbytes // PACK_BYTES, self._threads, self._blocks
                )
                stage_counter, tail_counter = self._counter_addresses(grid_blocks)
                self._order_stream(capturing)
                launcher(
                    src.data_ptr(),
                    dst.data_ptr(),
                    nbytes // PACK_BYTES,
                    nbytes,
                    self._recv_base,
                    self._flag_base,
                    self._send_base,
                    self._ctrl_base,
                    self._slot_bytes,
                    self._epoch_address,
                    stage_counter,
                    tail_counter,
                    self._poison_address,
                    self.spin_limit,
                    grid_blocks,
                )
                if dst is not out:
                    out.copy_(dst)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()
        return out

    def _aligned_scratch(self, which: int, like: torch.Tensor) -> torch.Tensor:
        """16-byte-aligned device scratch shaped like ``like`` (input 0, output 1).

        Allocated by ``prepare`` (two ``max_size`` buffers); a first use under
        CUDA graph capture is refused.
        """

        if self._align_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "RoCE all-reduce alignment scratch must be allocated before CUDA graph "
                    "capture; call prepare() first"
                )
            self._align_buffers = (
                torch.empty(self.max_size, dtype=torch.uint8, device=self.device),
                torch.empty(self.max_size, dtype=torch.uint8, device=self.device),
            )
        nbytes = like.numel() * like.element_size()
        return self._align_buffers[which][:nbytes].view(like.dtype).view(like.shape)

    def _order_stream(self, capturing: bool) -> None:
        """Make the current stream wait for the previous collective's stream.

        The runtime has one epoch, one doorbell and two transport slots, so
        collectives must execute in launch order even across streams.  Under
        capture no event is used: every collective must be on the capture
        stream, which keeps the graph's own ordering; captures are told apart
        by CUDA's capture id, so consecutive captures may use different streams.
        """

        current = torch.cuda.current_stream(self.device)
        if capturing:
            capture_id = _capture_id(current)
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_stream = current
            elif current != self._capture_stream:
                raise RuntimeError(
                    "RoCE collectives must all be captured on one stream"
                )
            return
        if self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._stream_event)

    def _mark_stream(self, capturing: bool) -> None:
        """Record the ordering event on the current stream after a launch."""

        if capturing:
            return
        current = torch.cuda.current_stream(self.device)
        self._stream_event.record(current)
        self._last_stream = current

    def check_health(self) -> None:
        """Raise if the proxy thread died or a kernel wait timed out.

        Cheap (two host memory reads); call it after graph replays, which
        cannot check inline.
        """

        if self._proxy is not None and self._proxy.failed():
            raise RuntimeError(f"RoCE proxy failed: {self._proxy.error()}")
        failed_seq = int(self._ctrl_np[2])
        if failed_seq != 0:
            peer = int(self._ctrl_np[3])
            hca = int(self._ctrl_np[6])
            raise RuntimeError(
                f"RoCE collective on rank {self.rank} timed out waiting for rank "
                f"{peer}, HCA {hca}, at sequence {failed_seq}; the runtime is "
                f"poisoned (its epoch stopped at {failed_seq - 1}, later launches "
                "do nothing) and rank data is no longer trustworthy"
            )

    @property
    def poisoned(self) -> bool:
        """True once a wait timed out or the proxy failed; the runtime cannot be reused."""

        return (self._proxy is not None and self._proxy.failed()) or int(
            self._ctrl_np[2]
        ) != 0

    # -- all-gather ---------------------------------------------------------------

    @staticmethod
    def _normalize_dim(inp: torch.Tensor, dim: int) -> int:
        """Map a negative ``dim`` to its positive index."""
        if dim < 0:
            dim += inp.dim()
        return dim

    def should_all_gather(self, inp: torch.Tensor, dim: int = -1) -> bool:
        """Eligible: contiguous CUDA tensor of any dtype, concat along dim 0 or the last dim.

        16-byte-aligned rows take the direct-layout kernel; anything else goes
        through a padded contiguous gather plus a torch reshape, so shape never
        forces a fallback to another backend.
        """

        if self._closed or self._proxy is None:
            raise RuntimeError("RoCE runtime is closed")
        if not inp.is_cuda or inp.dim() == 0:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        if inp.is_complex() or inp.is_sparse or inp.dtype == torch.bool:
            return False
        dim = self._normalize_dim(inp, dim)
        if dim not in (0, inp.dim() - 1):
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_gather_bytes

    def _direct_gather_layout(self, inp: torch.Tensor, dim: int) -> bool:
        """True when ``inp`` can be gathered straight into the concatenated layout (16-byte rows and pointer)."""
        nbytes = inp.numel() * inp.element_size()
        if nbytes % PACK_BYTES != 0 or inp.data_ptr() % PACK_BYTES != 0:
            return False
        return dim == 0 or (inp.shape[-1] * inp.element_size()) % PACK_BYTES == 0

    def all_gather(
        self,
        inp: torch.Tensor,
        *,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        plan,
    ) -> torch.Tensor:
        """Run a published prepared RoCE all-gather."""
        from b12x.preparation.types import require_prepared

        prepared = require_prepared(plan, "comm.roce", self.device)
        if prepared.runtime is not self:
            raise ValueError("RoCE plan belongs to another runtime")
        return self._run_prepared_all_gather(
            inp, prepared=prepared, dim=dim, out=out, stream=stream
        )

    def _run_prepared_all_gather(
        self,
        inp: torch.Tensor,
        *,
        prepared: object,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
    ) -> torch.Tensor:
        """Run a materialized gather launcher during priming or once the plan is published."""
        with self._lock:
            self.check_health()
            if not self.should_all_gather(inp, dim):
                raise ValueError("input is not eligible for the RoCE all-gather")
            dim = self._normalize_dim(inp, dim)
            shape = list(inp.shape)
            shape[dim] *= self.world_size
            if out is not None and (
                list(out.shape) != shape
                or out.dtype != inp.dtype
                or out.device != inp.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "out must be a contiguous tensor on the input's device of the gathered shape"
                )
            context = (
                torch.cuda.stream(stream) if stream is not None else _nullcontext()
            )
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                # Both paths stage the shard contiguously and only the reader is
                # strided, so ranks may take different paths for one collective.
                direct = self._direct_gather_layout(inp, dim) and (
                    out is None or out.data_ptr() % PACK_BYTES == 0
                )
                if direct:
                    if out is None:
                        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                    nbytes = inp.numel() * inp.element_size()
                    row_packs = (
                        nbytes // PACK_BYTES
                        if dim == 0
                        else (inp.shape[-1] * inp.element_size()) // PACK_BYTES
                    )
                    self._order_stream(capturing)
                    self._launch_gather(
                        prepared.gather_launcher, inp.data_ptr(), out.data_ptr(),
                        nbytes, row_packs,
                    )
                    self._mark_stream(capturing)
                    return out
                # Padded path: pad each shard to a whole number of packs, gather
                # contiguously into fixed scratch, then let torch produce the
                # requested layout.
                nbytes = inp.numel() * inp.element_size()
                padded = _align_up(nbytes, PACK_BYTES)
                staged, gathered = self._gather_scratch(padded)
                staged[:nbytes].copy_(inp.reshape(-1).view(torch.uint8))
                self._order_stream(capturing)
                self._launch_gather(
                    prepared.gather_launcher, staged.data_ptr(), gathered.data_ptr(),
                    padded, padded // PACK_BYTES,
                )
                self._mark_stream(capturing)
                stacked = (
                    gathered.view(self.world_size, padded)[:, :nbytes]
                    .reshape(-1)
                    .view(inp.dtype)
                    .reshape(self.world_size, *inp.shape)
                )
                result = stacked.movedim(0, dim).reshape(shape)
                if out is None:
                    return result.contiguous()
                out.copy_(result)
                return out

    def _gather_scratch(self, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed device scratch for the padded all-gather path, allocated once.

        ``prepare(padded_gather=True)`` or the first eager padded gather
        allocates it; a first use under CUDA graph capture is refused so a
        captured graph never owns the scratch.
        """

        if self._gather_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "RoCE all-gather scratch must be allocated before CUDA graph capture; "
                    "call prepare(padded_gather=True) first"
                )
            capacity = _align_up(self.max_gather_bytes, PACK_BYTES)
            self._gather_buffers = (
                torch.empty(capacity, dtype=torch.uint8, device=self.device),
                torch.empty(
                    self.world_size * capacity, dtype=torch.uint8, device=self.device
                ),
            )
        staged, gathered = self._gather_buffers
        return staged[:padded], gathered[: self.world_size * padded]

    def _launch_gather(
        self, launcher, input_address: int, output_address: int, nbytes: int,
        row_packs: int,
    ) -> None:
        """Launch an already prepared all-gather kernel."""
        # Same size-aware geometry as the all-reduce: a small shard (top-k
        # values and ids, MTP logits) launches a few blocks instead of the
        # full grid, and each power-of-two grid has its own arrival counters
        # so gathers and reductions of any size may interleave in one graph.
        grid_blocks = _grid_blocks(nbytes // PACK_BYTES, self._threads, self._blocks)
        stage_counter, tail_counter = self._counter_addresses(grid_blocks)
        launcher(
            input_address,
            output_address,
            nbytes // PACK_BYTES,
            nbytes,
            row_packs,
            self._recv_base,
            self._flag_base,
            self._send_base,
            self._ctrl_base,
            self._slot_bytes,
            self._epoch_address,
            stage_counter,
            tail_counter,
            self._poison_address,
            self.spin_limit,
            grid_blocks,
        )
        if not torch.cuda.is_current_stream_capturing():
            self.check_health()

    @contextmanager
    def capture(self, stream: object = None, *, channel_id: Optional[str] = None):
        """Context manager for CUDA graph capture; the launchers must already be prepared.

        The cross-stream ordering event is dropped for the capture: an event
        recorded outside a capture must not be waited on inside one, and
        ``torch.cuda.graph`` has already synchronized the device before the
        capture began.  Inside, every collective must use one stream.
        """

        self._last_stream = None
        self._capture_stream = None
        self._capture_id = 0
        try:
            yield self
        finally:
            self._capture_stream = None
            self._capture_id = 0
            self._last_stream = None

    # -- diagnostics / lifecycle --------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Runtime, control-record, and proxy counters for diagnostics."""
        info: dict[str, Any] = {
            "world_size": self.world_size,
            "rank": self.rank,
            "hcas": list(self.hca_names),
            "max_size": self.max_size,
            "max_gather_bytes": self.max_gather_bytes,
            "slot_bytes": self._slot_bytes,
            "epoch": int(self._counters[0].item()),
            "error_seq": int(self._error_word.item()),
            "error_peer": int(self._ctrl_words[3].item()),
            "error_hca": int(self._ctrl_words[6].item()),
            "ctrl_seq": int(self._ctrl_words[0].item()),
            "spin_limit": self.spin_limit,
            "stripe_hcas": list(range(len(self.hca_names))),
        }
        if self._proxy is not None:
            info.update(self._proxy.stats())
        return info

    def close(self) -> None:
        """Stop the proxy and release the transport.

        Waits for the device first so no in-flight kernel still reads the
        pinned region or rings the doorbell after the proxy is gone.  Peers
        that write to this rank afterwards see a remote-access completion
        error and raise on their side, which is the intended shutdown signal.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            if self._proxy is not None:
                self._proxy.close()
                self._proxy = None

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        """Release the transport if ``close`` was never called."""
        with contextlib.suppress(Exception):
            self.close()


class _nullcontext:
    def __enter__(self):
        """No-op context entry."""
        return None

    def __exit__(self, *exc):
        """No-op context exit."""
        return False


__all__ = [
    "DEFAULT_MAX_GATHER_BYTES",
    "DEFAULT_MAX_SIZE",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "RoceOneshotAllReduce",
    "default_gid_index",
    "discover_hcas",
    "is_supported",
]
