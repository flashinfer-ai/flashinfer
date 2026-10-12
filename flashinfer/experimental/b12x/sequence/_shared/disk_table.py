"""Bounded disk row staging shared by PLE and Engram.

The native reader owns bounded request planning and I/O. This module owns
storage, source registration and the host/GPU transaction boundary; hashing
and numerical decoding belong to each embedding operation.
"""

from __future__ import annotations

import math
import operator
import os
import queue
import sys
import threading
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart

# Poison pill handed to the prefetch worker's queue by ``close`` to unblock and
# stop its daemon thread. Distinct by identity from any id list a job may hold.
_PREFETCH_SENTINEL = object()
# Pool size applied when the cache is enabled through the environment but no
# size is given: 1 GiB of pool bytes, rounded down to whole rows by
# ``_resolve_row_cache_size`` (11,930,464 rows at the shipped 90 B/row PLE
# geometry). The row-cache knobs are the only way to enable the cache from a
# shell or manifest; the shipped implementation is one host-tier C S3-FIFO
# (``row_cache.S3FifoRowCache``; SIEVE and W-TinyLFU were retired as branch-
# local experiments, rationale in ``docs/DESIGN.md`` §1 "Policy selection").
# ``B12X_ROW_CACHE_TIER`` accepts only ``host`` (or unset); else it raises.
_ROW_CACHE_DEFAULT_BYTES = 1 << 30


# Suffixes accepted by ``B12X_ROW_CACHE_BYTES``, as multipliers of the decimal
# amount. A bare number is plain bytes.
_ROW_CACHE_SIZE_SUFFIXES = {
    "": 1,
    "K": 1 << 10,
    "M": 1 << 20,
    "G": 1 << 30,
    "T": 1 << 40,
}


def _parse_row_cache_size(text: str, row_bytes: int) -> int:
    """Return ``text`` (e.g. ``"8.6MiB"``, ``"1GiB"``, ``"900000"``) as a row count.

    ``B12X_ROW_CACHE_BYTES`` lets an operator size the pool in the unit that
    appears in the ``row_cache_bytes`` stat and the design docs, instead of
    counting rows. A bare number is plain bytes; a decimal amount takes an
    optional ``KiB``/``MiB``/``GiB``/``TiB`` suffix (the ``i`` and ``B`` are
    optional, case-insensitive). The value rounds *down* to whole rows, so a
    request below one row, and malformed, non-finite or negative input, are
    construction errors rather than a silently disabled cache.
    """
    import re

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d*)?)\s*(?:([KMGT])i?B?|B)?\s*", text, flags=re.IGNORECASE
    )
    if match is None:
        raise ValueError(
            "B12X_ROW_CACHE_BYTES must be a byte count, optionally with a "
            "KiB/MiB/GiB/TiB suffix (e.g. '8.6MiB'), "
            f"got {text!r}"
        )
    suffix = (match.group(2) or "").upper().rstrip("B").rstrip("I")
    if suffix not in _ROW_CACHE_SIZE_SUFFIXES:
        raise ValueError(f"B12X_ROW_CACHE_BYTES has an unknown multiplier in {text!r}")
    size = float(match.group(1)) * _ROW_CACHE_SIZE_SUFFIXES[suffix]
    if not math.isfinite(size) or size < row_bytes:
        raise ValueError(
            f"B12X_ROW_CACHE_BYTES={text!r} rounds to zero rows at {row_bytes} B/row"
        )
    return int(size) // row_bytes


def _resolve_row_cache_size(
    size_env: str | None, rows_env: str | None, tier_env: str | None, row_bytes: int
) -> tuple[int, bool]:
    """Return ``(cache_rows, size_defaulted)`` for the environment size knobs.

    Precedence: ``B12X_ROW_CACHE_BYTES`` > ``B12X_ROW_CACHE_ROWS`` > the 1 GiB
    default > 0 (off). The cache counts as *enabled* when any of
    ``B12X_ROW_CACHE_BYTES``, ``B12X_ROW_CACHE_ROWS`` or
    ``B12X_ROW_CACHE_TIER`` is set to a non-empty value; enabled without a
    size selects ``_ROW_CACHE_DEFAULT_BYTES`` of pool bytes, rounded *down*
    to whole rows, so a 1 GiB budget costs ``(1 << 30) // row_bytes *
    row_bytes`` bytes. ``B12X_ROW_CACHE_ROWS=0`` is an explicit size and keeps
    the historical off-by-default behaviour. A default that would round to
    zero rows (a row larger than the budget) is a construction error, never a
    silently disabled cache.
    """
    # An empty value is unset (matches the docstring's "non-empty" and the
    # TIER normaliser): a manifest that exports B12X_ROW_CACHE_BYTES= must not
    # raise or collide with a set ROWS.
    size_env = size_env if size_env else None
    rows_env = rows_env if rows_env else None
    if size_env is not None and rows_env is not None:
        raise ValueError("set only one of B12X_ROW_CACHE_ROWS / B12X_ROW_CACHE_BYTES")
    if size_env is not None:
        return _parse_row_cache_size(size_env, row_bytes), False
    if rows_env is not None:
        return int(rows_env), False
    enabled = any(
        value is not None and value != "" for value in (size_env, rows_env, tier_env)
    )
    if not enabled:
        return 0, False
    rows = _ROW_CACHE_DEFAULT_BYTES // row_bytes
    if rows <= 0:
        raise ValueError(
            f"1 GiB row-cache default rounds to zero rows at {row_bytes} "
            "B/row; set B12X_ROW_CACHE_BYTES explicitly"
        )
    return rows, True


def _validate_row_cache_tier(cache_tier: str | None) -> str:
    """Normalise the cache tier.

    The row cache has a single ``host`` tier; ``B12X_ROW_CACHE_TIER`` accepts
    only ``host`` (or unset), and any other value is a construction error.
    """
    if cache_tier is None or cache_tier.strip() == "":
        return "host"
    if cache_tier.strip().lower() == "host":
        return "host"
    raise ValueError(f"B12X_ROW_CACHE_TIER={cache_tier!r} is invalid: use 'host'")


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    from cuda.bindings import runtime as cudart

    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host storage requires torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, 0, shape, _contiguous_strides(shape)
    )


class MappedHostAllocation:
    """Own a mapped page-locked allocation and its CPU/CUDA tensor aliases."""

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> None:
        from cuda.bindings import runtime as cudart

        self._host_pointer = 0
        self._closed = True
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host storage requires an indexed CUDA device, got {device}"
            )
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )
        self.device = device
        self.nbytes = nbytes
        with torch.cuda.device(device):
            error, pointer = cudart.cudaHostAlloc(
                nbytes, cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(pointer)
            self._closed = False
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        _check_cuda(cudart.cudaFreeHost(self._host_pointer)[0], "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if (
            self._closed
            or self._host_pointer == 0
            or sys is None
            or sys.is_finalizing()
        ):
            return
        with suppress(Exception):
            self.close()


class DiskRowCache:
    """Read immutable row planes into a reusable, batch-sized staging buffer.

    Source rows have one weight byte plane and an optional independent scale
    byte plane. Hold ``transaction`` across ID production, ``read_rows`` and
    GPU decoding. Downstream graphs consume the decoder's fixed device output,
    not disk I/O.

    Rows come from checkpoint shards via io_uring/GDS, or from ``host_rows``:
    CPU tensors holding this shard's rows ``[shard_start, shard_end)`` in id
    order. The optional row cache (``cache_rows`` > 0) ships a single
    implementation, the host-tier C S3-FIFO
    (``row_cache.S3FifoRowCache``): recently read rows stay in host RAM so
    repeated n-grams skip the source, and a cached batch performs no
    per-request Python work (the plan, counters, ghost bookkeeping and row
    copies all run inside the extension; the io_uring fetch and the CUDA
    transaction/fence discipline are unchanged around it).

    Configuration (read at construction; the ``cache_rows`` argument wins
    over the environment):

    * ``B12X_ROW_CACHE_ROWS`` — pool capacity in rows. ``0`` keeps the cache
      off, the historical default when nothing else is set.
    * ``B12X_ROW_CACHE_BYTES`` — the same knob in pool bytes, rounded down to
      whole rows: ``8.6MiB`` == 100000 rows at 90 B/row. Mutually exclusive
      with ``B12X_ROW_CACHE_ROWS``.
    * Enabled without a size — any of ``B12X_ROW_CACHE_BYTES``,
      ``B12X_ROW_CACHE_ROWS`` or ``B12X_ROW_CACHE_TIER`` set to a non-empty
      value, none naming a size — the pool defaults to 1 GiB:
      ``(1 << 30) // row_bytes`` rows (11,930,464 at the shipped 90 B/row PLE
      geometry; the row count rounds down to whole rows, so the pool costs
      at most 1 GiB).
    * ``B12X_ROW_CACHE_TIER`` — only ``host`` (or unset); the cache has a
      single tier, so any other value raises and names the knob.
    * The cache has a single implementation: the host-tier C S3-FIFO
      (``row_cache.S3FifoRowCache``). Construction raises if the extension
      fails to build — there is no silent fallback. The A/B arm for "cache
      off" is ``B12X_ROW_CACHE_ROWS=0``; the pre-port Python-path cost model
      is in ``docs/DESIGN.md`` §1.
    * ``B12X_PLE_PREFETCH`` — nonzero turns the ``prefetch`` worker on. Under
      vLLM the runner hook is gated separately by ``VLLM_PLE_PREFETCH``; both
      must be set.

    The GDS backend stages on the device itself and is not cached.

    ``prefetch(ids)`` warms the cache ahead of the read path from a background
    worker, driven by ``B12X_PLE_PREFETCH`` (off by default). Reserved rows sit
    in a loading slot that a lookup never hits and eviction never reclaims;
    only a confirmed row becomes a ready hit. The synchronous read path treats
    an in-flight prefetch as a plain miss and takes its own fetch, so the same
    row can be written by both the worker and the batch, always with identical
    bytes (last write wins). Both the disk and ``host_rows`` sources are
    prefetchable through the same ``_fetch`` primitive.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_lookups: int,
        table_rows: int,
        shard_start: int,
        shard_end: int,
        shard_rows: int,
        weight_row_bytes: int,
        scale_row_bytes: int = 0,
        queue_depth: int = 64,
        cache_rows: int | None = None,
        cache_tier: str | None = None,
        host_rows: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> None:
        from b12x.loader._native import load

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("disk row staging requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_lookups = operator.index(max_lookups)
        self.table_rows = operator.index(table_rows)
        self.shard_start = operator.index(shard_start)
        self.shard_end = operator.index(shard_end)
        self.shard_rows = operator.index(shard_rows)
        self.weight_row_bytes = operator.index(weight_row_bytes)
        self.scale_row_bytes = operator.index(scale_row_bytes)
        queue_depth = operator.index(queue_depth)
        if not 0 < self.shard_rows <= (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        if host_rows is not None:
            self._backend = "host"
        else:
            self._backend = os.environ.get("B12X_DISK_BACKEND", "io_uring")
            if self._backend not in ("io_uring", "gds"):
                raise ValueError("B12X_DISK_BACKEND must be io_uring or gds")
        row_bytes = self.weight_row_bytes + self.scale_row_bytes
        tier_env = os.environ.get("B12X_ROW_CACHE_TIER")
        if cache_rows is None:
            # ``B12X_ROW_CACHE_BYTES`` is the same knob in pool bytes (the
            # unit ``stats()['row_cache_bytes']`` reports), because an
            # operator reasons about a host-RAM budget in MiB, not in rows of
            # 90 B. With no size given at all, an enabled cache takes the
            # 1 GiB default; nothing set keeps the historical off-by-default.
            size_env = os.environ.get("B12X_ROW_CACHE_BYTES")
            rows_env = os.environ.get("B12X_ROW_CACHE_ROWS")
            cache_rows = _resolve_row_cache_size(
                size_env, rows_env, tier_env, row_bytes
            )[0]
        cache_rows = operator.index(cache_rows)
        if cache_rows < 0:
            raise ValueError("cache_rows must be nonnegative")
        cache_tier = _validate_row_cache_tier(
            tier_env if cache_tier is None else cache_tier
        )
        if cache_rows and self._backend == "gds":
            raise ValueError("the GDS backend does not support a row cache")
        self.cache_rows = cache_rows
        self.cache_tier = cache_tier if cache_rows else None
        self._gds = None
        self._native = None
        self._reader = None
        self._host_weight = self._host_scale = None
        if self._backend == "host":
            self._host_weight, self._host_scale = self._check_host_rows(host_rows)
        elif self._backend == "io_uring":
            self._native = load()
            self._reader = self._native.ple_reader(
                self.shard_rows,
                self.table_rows,
                self.shard_start,
                self.shard_end,
                self.weight_row_bytes,
                self.scale_row_bytes,
                self.max_lookups,
                queue_depth,
            )
        self.shard_count = (self.table_rows + self.shard_rows - 1) // self.shard_rows
        self.ids_host = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._ids_buffer = memoryview(self.ids_host.numpy())
        self._weight_allocation = self._scale_allocation = None
        self.weight_host = self.scale_host = None
        self._weight_buffer = self._scale_buffer = None
        self.scale = None
        if self._backend == "gds":
            from ._gds import GdsRows

            self._gds = GdsRows(self, queue_depth)
            self.weight, self.scale = self._gds.weight, self._gds.scale
        else:
            # Host staging through the mapped, page-locked buffer is the only
            # non-GDS layout: the cache pool lives in host RAM next to the
            # staging buffer, so on integrated parts it is the only copy and a
            # cached batch skips both the source read and the H2D hop.
            self._weight_allocation = MappedHostAllocation(
                (self.max_lookups, self.weight_row_bytes), torch.uint8, device
            )
            self.weight = self._weight_allocation.device_view
            self.weight_host = self._weight_allocation.host_view
            self._weight_buffer = memoryview(self.weight_host.numpy())
            if self.scale_row_bytes:
                self._scale_allocation = MappedHostAllocation(
                    (self.max_lookups, self.scale_row_bytes), torch.uint8, device
                )
                self.scale = self._scale_allocation.device_view
                self.scale_host = self._scale_allocation.host_view
                self._scale_buffer = memoryview(self.scale_host.numpy())
        self._index = None
        if cache_rows:
            self._init_row_cache()
        self._sources: set[tuple[bool, int]] = set()
        self._frozen = False
        self._lock = threading.RLock()
        self._transaction_thread: int | None = None
        self._transaction_stream: torch.cuda.Stream | None = None
        self._ids_ready = torch.cuda.Event()
        self._cache_done = torch.cuda.Event()
        self._cache_used = False
        self._closed = False
        # Speculative PLE row prefetch. The worker thread and its bounded queue
        # are created lazily on the first enabled ``prefetch`` so a disabled
        # cache never spawns a thread; both stay ``None`` until then.
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_stop = False

    def _check_host_rows(self, host_rows):
        weight, scale = host_rows
        rows = self.shard_end - self.shard_start
        planes = [(weight, self.weight_row_bytes, "weight")]
        if self.scale_row_bytes:
            planes.append((scale, self.scale_row_bytes, "scale"))
        elif scale is not None:
            raise ValueError("host_rows has a scale plane but scale_row_bytes is 0")
        for tensor, width, name in planes:
            if (
                tensor is None
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or tuple(tensor.shape) != (rows, width)
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"host_rows {name} must be a contiguous CPU uint8 tensor of "
                    f"shape ({rows}, {width})"
                )
        return weight.numpy(), (scale.numpy() if self.scale_row_bytes else None)

    def _init_row_cache(self) -> None:
        from .row_cache import S3FifoRowCache

        rows = self.cache_rows
        widths = [self.weight_row_bytes] + (
            [self.scale_row_bytes] if self.scale_row_bytes else []
        )
        # vLLM constructs models under an ambient torch.device context (meta at
        # initialize_model time), so every CPU allocation here names the device
        # explicitly; a bare torch.empty(pin_memory=True) would build a meta
        # tensor and fail "Only dense CPU tensors can be pinned".
        # Miss rows land here first; the CPU reads them back, so they are
        # plain pinned memory rather than write-combined.
        self._miss_ids = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._miss_planes = [
            torch.empty(
                (self.max_lookups, width),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            for width in widths
        ]
        # Host tier: the pool is plain host RAM and the stage planes are views
        # of the mapped-host weight/scale buffers. The C index binds all of
        # them (zero-copy) for the index's lifetime, so they are allocated
        # before the index and kept referenced by this object.
        self._pool = [
            torch.empty((rows, width), dtype=torch.uint8, device="cpu").numpy()
            for width in widths
        ]
        self._stage_host = [self.weight_host.numpy()] + (
            [self.scale_host.numpy()] if self.scale_row_bytes else []
        )
        for pinned in (self.ids_host, self._miss_ids, *self._miss_planes):
            if not pinned.is_pinned():
                raise ValueError(
                    "row cache id and miss buffers must be page-locked before "
                    "binding them to the C index"
                )
        self._miss_ids_view = memoryview(self._miss_ids.numpy())
        self._miss_weight_view = memoryview(self._miss_planes[0].numpy())
        self._miss_scale_view = (
            memoryview(self._miss_planes[1].numpy()) if self.scale_row_bytes else None
        )
        # Raises RuntimeError if the C extension cannot be built; the cache
        # never silently degrades to a Python path (there is none left).
        self._index = S3FifoRowCache(
            rows,
            self.max_lookups,
            self._ids_buffer,
            self._miss_ids_view,
            [
                memoryview(plane)
                for plane in (self._miss_weight_view, self._miss_scale_view)
                if plane is not None
            ],
            [memoryview(array) for array in self._stage_host],
            [memoryview(array) for array in self._pool],
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("disk row cache is closed")

    def __del__(self):
        if not getattr(self, "_closed", True):
            with suppress(Exception):
                self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._transaction_thread is not None:
                raise RuntimeError("cannot close a disk row cache during a transaction")
            self._prefetch_stop = True
            worker = self._prefetch_thread
            jobs = self._prefetch_queue
            if self._cache_used:
                with torch.cuda.device(self.device):
                    self._cache_done.synchronize()
            if self._gds is not None:
                self._gds.close()
            # Drop the capsule (and the buffers it holds) before the mapped
            # host allocations are freed underneath it.
            if self._index is not None:
                self._index.close()
                self._index = None
            for allocation in (self._scale_allocation, self._weight_allocation):
                if allocation is not None:
                    allocation.close()
            self._reader = self._native = None
            self._closed = True
        # Drain the worker outside the lock: a running job reacquires it in its
        # reserve/confirm sections, so joining while holding it would deadlock.
        if worker is not None and jobs is not None:
            with suppress(queue.Full):
                jobs.put_nowait(_PREFETCH_SENTINEL)
            worker.join(timeout=2)

    def prefetch(self, ids) -> bool:
        """Speculatively fetch ``ids`` into the row cache without blocking.

        Advisory only: it never touches the staging buffer, never raises, and
        hands the work to a single lazily-started daemon worker over a bounded
        queue, so callers are never stalled behind disk I/O. Returns ``False``
        when the row cache is disabled (``cache_rows == 0``) or the
        ``B12X_PLE_PREFETCH`` environment variable is unset or ``"0"``;
        otherwise ``True``. ``ids`` may be a list of ints, a CPU int64 tensor,
        or a numpy array; duplicate, out-of-shard, and already-handled ids are
        no-ops. When the queue is saturated the newest request is dropped.
        """
        if not self._prefetch_ok():
            return False
        job = self._normalize_prefetch_ids(ids)
        if not job:
            return True
        self._ensure_prefetch_worker()
        if self._prefetch_queue is None:
            return True
        try:
            self._prefetch_queue.put_nowait(job)
        except queue.Full:
            pass
        return True

    def _prefetch_ok(self) -> bool:
        """Gate speculative prefetch: cache enabled, disk/host backend, opt-in env.

        ``b12x`` has no ``vllm`` env module, so the flag is read straight from
        ``os.environ``. It defaults to off; only a set, nonzero value turns
        prefetch on. Returns ``False`` for a disabled cache (``cache_rows == 0``)
        or a closed one so no worker is ever spawned for a cache that cannot
        hold rows.
        """
        if self._closed or not self.cache_rows or self._index is None:
            return False
        if self._backend == "gds":
            return False
        flag = os.environ.get("B12X_PLE_PREFETCH")
        return flag is not None and flag != "0"

    def _normalize_prefetch_ids(self, ids) -> list[int]:
        """Coerce ``ids`` to the distinct in-shard, not-yet-cached row ids.

        Accepts a list of ints, a CPU int64 tensor, or a numpy array; every
        element is converted with ``int``. Duplicates collapse to first
        occurrence, out-of-shard ids are dropped (nothing to prefetch), and
        ids already committed are dropped as no-ops. Rows merely loading are
        left in: their ``reserve`` returns ``None`` in the worker, so they cost
        a cheap no-op rather than a second fetch.
        """
        import numpy as np

        if isinstance(ids, torch.Tensor):
            if ids.numel() == 0:
                return []
            ids = ids.detach().cpu()
            if ids.dtype != torch.int64:
                ids = ids.to(torch.int64)
            raw = ids.reshape(-1).tolist()
        elif isinstance(ids, np.ndarray):
            if ids.size == 0:
                return []
            raw = ids.reshape(-1).tolist()
        else:
            raw = list(ids)
        low, high = self.shard_start, self.shard_end
        index = self._index
        seen: set[int] = set()
        result: list[int] = []
        with self._lock:
            for value in raw:
                key = int(value)
                if key in seen:
                    continue
                seen.add(key)
                if not low <= key < high:
                    continue
                if index.is_ready(key):
                    continue
                result.append(key)
        return result

    def _ensure_prefetch_worker(self) -> None:
        with self._lock:
            if self._prefetch_thread is not None or self._closed:
                return
            self._prefetch_queue = queue.Queue(maxsize=64)
            self._prefetch_stop = False
            worker = threading.Thread(
                target=self._prefetch_loop,
                name="b12x-ple-prefetch",
                daemon=True,
            )
            self._prefetch_thread = worker
        worker.start()

    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop:
            try:
                job = self._prefetch_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is _PREFETCH_SENTINEL:
                break
            try:
                self._run_prefetch_job(job)
            except Exception:
                # Prefetch is advisory: a source/transport error must never
                # surface to, or destabilize, the caller or the worker.
                pass

    def _run_prefetch_job(self, ids) -> None:
        import numpy as np

        index = self._index
        reserved: list[tuple[int, int]] = []
        with self._lock:
            if self._closed:
                return
            for key in ids:
                slot = index.reserve(key)
                if slot is not None:
                    reserved.append((key, slot))
        if not reserved:
            return
        count = len(reserved)
        weight_np = np.zeros((count, self.weight_row_bytes), dtype=np.uint8)
        scale_np = (
            np.zeros((count, self.scale_row_bytes), dtype=np.uint8)
            if self.scale_row_bytes
            else None
        )
        ids_np = np.fromiter((key for key, _ in reserved), dtype=np.int64, count=count)
        try:
            self._fetch(
                memoryview(ids_np),
                memoryview(weight_np),
                memoryview(scale_np) if scale_np is not None else None,
                count,
            )
            fetched = True
        except Exception:
            fetched = False
        planes = [weight_np] + ([scale_np] if scale_np is not None else [])
        with self._lock:
            if self._closed:
                for _, slot in reserved:
                    index.cancel(slot)
                return
            if not fetched:
                for _, slot in reserved:
                    index.cancel(slot)
                return
            # Write the fetched bytes into the reserved
            # slots (skipping any the synchronous path already adopted or
            # cancelled) and publish them, all under _lock. confirm() is
            # idempotent, so adopting a slot the batch already committed is a
            # no-op, exactly as with the Python index.
            for row, (_, slot) in enumerate(reserved):
                if not index.is_loading(slot):
                    continue
                for plane, pool in zip(planes, self._pool):
                    pool[slot] = plane[row]
            for _, slot in reserved:
                index.confirm(slot)

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        self._require_open()
        if self._backend == "host":
            raise RuntimeError("host-sourced rows have no checkpoint shards")
        with self._lock:
            if self._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            shard_index = operator.index(shard_index)
            offset = operator.index(offset)
            if not 0 <= shard_index < self.shard_count:
                raise ValueError("checkpoint shard index is out of range")
            if offset < 0:
                raise ValueError("checkpoint file offset must be nonnegative")
            if scale and not self.scale_row_bytes:
                raise ValueError("disk table has no row scale plane")
            key = (scale, shard_index)
            if key in self._sources:
                raise ValueError("checkpoint shard is already registered")
            start = shard_index * self.shard_rows
            end = min(start + self.shard_rows, self.table_rows)
            if end <= self.shard_start or start >= self.shard_end:
                return
            if self._gds is not None:
                self._gds.native.add(
                    self._gds.reader, shard_index, os.fspath(path), offset, scale
                )
            else:
                self._native.ple_reader_add(
                    self._reader, shard_index, os.fspath(path), offset, scale
                )
            self._sources.add(key)

    def require_complete(self) -> None:
        self._require_open()
        with self._lock:
            if self._backend == "host":
                return
            first = self.shard_start // self.shard_rows
            last = (self.shard_end + self.shard_rows - 1) // self.shard_rows
            for shard in range(first, last):
                if (False, shard) not in self._sources:
                    raise ValueError(f"missing disk weight shard {shard}")
                if self.scale_row_bytes and (True, shard) not in self._sources:
                    raise ValueError(f"missing disk scale shard {shard}")

    def freeze(self) -> None:
        with self._lock:
            self.require_complete()
            self._frozen = True

    @contextmanager
    def transaction(self) -> Iterator[DiskRowCache]:
        self._require_open()
        if torch.compiler.is_compiling():
            raise RuntimeError("disk table preparation cannot run under torch.compile")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk table preparation must run outside CUDA graph capture"
                )
            with self._lock:
                if self._transaction_thread is not None:
                    raise RuntimeError("disk row transactions cannot be nested")
                stream = torch.cuda.current_stream(self.device)
                if self._cache_used:
                    stream.wait_event(self._cache_done)
                self._transaction_thread = threading.get_ident()
                self._transaction_stream = stream
                try:
                    yield self
                finally:
                    self._cache_done.record(stream)
                    self._cache_used = True
                    self._transaction_stream = None
                    self._transaction_thread = None

    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)

    def _stage_ids(self, ids: torch.Tensor, count: int) -> None:
        if self._transaction_thread != threading.get_ident():
            raise RuntimeError("read_rows requires an active disk row transaction")
        count = operator.index(count)
        if not 0 <= count <= self.max_lookups:
            raise ValueError("disk lookup count exceeds batch capacity")
        if (
            ids.device != self.device
            or ids.dtype != torch.int64
            or not ids.is_contiguous()
        ):
            raise ValueError(
                "disk row IDs must be contiguous CUDA int64 on the cache device"
            )
        if ids.numel() < count:
            raise ValueError("disk row IDs do not cover the requested count")
        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        self._ids_ready.record(self._transaction_stream)

    def _read_staged(self, count: int) -> None:
        # Completes ID production and all prior cache readers before host writes.
        with torch.cuda.device(self.device):
            self._ids_ready.synchronize()
            if self._gds is not None:
                self._gds.read(self._ids_buffer, count, self._transaction_stream)
                return
            if self._index is None:
                self._fetch(
                    self._ids_buffer, self._weight_buffer, self._scale_buffer, count
                )
            elif count:
                self._read_cached(count)

    def _fetch(self, ids, weights, scales, count: int) -> None:
        """Write source rows for ``ids[:count]`` into C-contiguous row buffers."""
        if self._backend == "io_uring":
            self._native.ple_reader_run(self._reader, ids, weights, scales, count)
            return
        import numpy as np

        ids = np.frombuffer(ids, dtype=np.int64, count=count)
        valid = np.flatnonzero((ids >= self.shard_start) & (ids < self.shard_end))
        rows = ids[valid] - self.shard_start
        planes = [(weights, self._host_weight)]
        if self.scale_row_bytes:
            planes.append((scales, self._host_scale))
        for out, table in planes:
            out = np.frombuffer(out, dtype=np.uint8).reshape(-1, table.shape[1])
            out[:count] = 0
            out[valid] = table[rows]

    def _read_cached(self, count: int) -> None:
        # One C call resolves the batch: lookups, counter/ghost/queue
        # mutation, admission evictions, and the hit copies from pool into the
        # bound staging planes. Miss ids land in the bound pinned buffer; the
        # io_uring fetch (unchanged, still through _fetch) reads the distinct
        # miss wave into the bound miss planes; store() then stages misses and
        # writes pool stores. Hits are staged before any store of the same
        # batch, so an eviction made for this batch never clobbers a row the
        # batch still reads (RowPlan; docs/DESIGN.md §3), and the transaction's
        # _cache_done fence still gates pool reuse across the decode.
        _, misses = self._index.read(count)
        if misses:
            self._fetch(
                self._miss_ids_view,
                self._miss_weight_view,
                self._miss_scale_view,
                misses,
            )
        self._index.store()

    def stats(self) -> dict[str, int | float]:
        self._require_open()
        with self._lock:
            if self._gds is not None:
                result = dict(self._gds.native.stats(self._gds.reader))
            elif self._native is not None:
                result = dict(self._native.ple_reader_stats(self._reader))
            else:
                result = {"staging_bytes": 0, "metadata_bytes": 0}
            result["ids_host_bytes"] = (
                self.ids_host.numel() * self.ids_host.element_size()
            )
            result["weight_cache_bytes"] = (
                self.weight.numel() * self.weight.element_size()
            )
            result["scale_cache_bytes"] = (
                self.scale.numel() * self.scale.element_size()
                if self.scale is not None
                else 0
            )
            result["cache_bytes"] = (
                result["weight_cache_bytes"] + result["scale_cache_bytes"]
            )
            result["owned_staging_bytes"] = (
                result["staging_bytes"]
                + result["ids_host_bytes"]
                + result["cache_bytes"]
            )
            descriptors = result.get("descriptor_bytes", 0)
            result["gds_enabled"] = int(self._gds is not None)
            device_staged = self._gds is not None
            result["device_staging_bytes"] = (
                (result["staging_bytes"] + result["cache_bytes"] + descriptors)
                if device_staged
                else 0
            )
            result["owned_host_bytes"] = (
                result["ids_host_bytes"] + descriptors + result["metadata_bytes"]
            )
            if not device_staged:
                result["owned_host_bytes"] += (
                    result["staging_bytes"] + result["cache_bytes"]
                )
            result["owned_staging_bytes"] += 2 * descriptors
            if self._index is not None:
                pool_bytes = self.cache_rows * (
                    self.weight_row_bytes + self.scale_row_bytes
                )
                index_stats = self._index.stats()
                lookups = index_stats["hits"] + index_stats["misses"]
                result.update(
                    row_cache_tier=self.cache_tier,
                    row_cache_rows=self.cache_rows,
                    row_cache_resident=index_stats["resident"],
                    row_cache_bytes=pool_bytes,
                    row_cache_hits=index_stats["hits"],
                    row_cache_misses=index_stats["misses"],
                    row_cache_hit_rate=(
                        index_stats["hits"] / lookups if lookups else 0.0
                    ),
                    row_cache_small=index_stats["s"],
                    row_cache_main=index_stats["m"],
                    row_cache_ghost=index_stats["g"],
                )
            return result
