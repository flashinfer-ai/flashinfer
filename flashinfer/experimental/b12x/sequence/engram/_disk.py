"""Bounded immutable-file row staging for prepared Engram lookup."""
from __future__ import annotations

import operator
import os
import torch

def _require_disk_eager(device: torch.device) -> None:
    if torch.compiler.is_compiling():
        raise RuntimeError("disk Engram preparation cannot run under torch.compile")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "disk Engram preparation must run outside CUDA graph capture"
            )



class DiskTable:
    """Caller-owned immutable weight/scale files and a batch-bounded row cache.

    Register global FP8E4M3 rows of 256 bytes and separate raw E8M0 rows of
    eight bytes. By default each plane is one whole-table source. Offsets
    locate each plane's first byte, including inside a checkpoint container.
    Files must remain immutable for this owner's lifetime. Preparation is
    eager-only; downstream graphs consume the binding's stable BF16 output.

    resident_scales retains the owned original E8M0 bytes in mapped host RAM
    and removes scale-plane disk reads. It is opt-in; callers must budget
    scale RAM. Disk lookup is synchronous. The prefetch argument and methods
    are accepted as no-ops for callers that still pass them.
    """

    def __init__(
        self,
        state: _State,
        shard_rows: int | None = None,
        queue_depth: int = 128,
        *,
        resident_scales: bool = False,
        prefetch: bool = False,
    ) -> None:
        from .._shared.disk_table import DiskRowCache, MappedHostAllocation

        from ._impl import _State
        if not isinstance(state, _State):
            raise TypeError("state must be prepared Engram state")
        if state.caps.device.type != "cuda":
            raise ValueError("disk Engram requires CUDA")
        _require_disk_eager(state.caps.device)
        if not isinstance(resident_scales, bool) or not isinstance(prefetch, bool):
            raise TypeError("resident_scales and prefetch must be bool")
        self.state = state
        self._closed = False
        if not state.compact_rows or state.resident_scales != resident_scales:
            raise ValueError("disk row and scale layouts differ from Engram preparation")
        self.resident_scales = resident_scales
        self._scale_sources: set[int] = set()
        self._scale_owner = None
        self._cache = DiskRowCache(
            device=state.caps.device,
            max_lookups=state.caps.max_tokens * 24,
            table_rows=state.table_rows,
            shard_start=min(state.shard_start, state.table_rows),
            shard_end=min(state.shard_end, state.table_rows),
            shard_rows=state.table_rows if shard_rows is None else shard_rows,
            weight_row_bytes=256,
            scale_row_bytes=0 if resident_scales else 8,
            queue_depth=queue_depth,
        )
        self.weight = self._cache.weight.view(torch.float8_e4m3fn)
        self.scale_bytes = self._cache.scale
        if resident_scales:
            # Retain the original E8M0 bytes, not decoded/requantized scales.
            # The caller must budget this allocation: ceil(rows/TP) * 8 bytes.
            self._scale_owner = MappedHostAllocation(
                state.scale_shape, torch.uint8, state.caps.device
            )
            self._scale_owner.host_view.zero_()
            self.scale_bytes = self._scale_owner.device_view

    @property
    def prefetch_pending(self) -> bool:
        """No disk reads remain pending between synchronous lookups."""
        return False

    def prefetch(self, binding: LookupBinding, token_count: int) -> None:
        """No-op; run_lookup reads and consumes the requested rows."""

    def abort_prefetch(self) -> None:
        """No-op; synchronous lookup owns its entire transaction."""

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._cache.close()
        if self._scale_owner is not None:
            self._scale_owner.close()
        self._closed = True

    def _require_open(self):
        if self._closed:
            raise RuntimeError("disk Engram table is closed")

    def freeze(self):
        self.require_complete()
        self._cache.freeze()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def add_shard(
        self, index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        """Register an immutable global source shard before binding."""
        self._require_open()
        if not (scale and self.resident_scales):
            self._cache.add_shard(index, path, offset, scale=scale)
            return
        with self._cache._lock:
            if self._cache._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            index, offset = operator.index(index), operator.index(offset)
            if not 0 <= index < self._cache.shard_count or offset < 0:
                raise ValueError("invalid scale shard index or offset")
            if index in self._scale_sources:
                raise ValueError("checkpoint scale shard is already registered")
            start = index * self._cache.shard_rows
            end = min(start + self._cache.shard_rows, self.state.table_rows)
            first, last = max(start, self.state.shard_start), min(end, self.state.shard_end)
            if first >= last:
                return
            view = self._scale_owner.host_view[
                first - self.state.shard_start : last - self.state.shard_start
            ]
            data = memoryview(view.numpy()).cast("B")
            with open(os.fspath(path), "rb", buffering=0) as source:
                if offset + (end - start) * 8 > os.fstat(source.fileno()).st_size:
                    raise ValueError("scale plane exceeds checkpoint file bounds")
                source.seek(offset + (first - start) * 8)
                done = 0
                while done < len(data):
                    count = source.readinto(data[done : done + (16 << 20)])
                    if not count:
                        raise ValueError("short resident Engram scale read")
                    done += count
            self._scale_sources.add(index)

    def require_complete(self) -> None:
        self._require_open()
        self._cache.require_complete()
        if self.resident_scales:
            first = self._cache.shard_start // self._cache.shard_rows
            last = (self._cache.shard_end + self._cache.shard_rows - 1) // self._cache.shard_rows
            for shard in range(first, last):
                if shard not in self._scale_sources:
                    raise ValueError(f"missing resident scale shard {shard}")

    def stats(self) -> dict[str, int | float]:
        """Return shared reader counters and batch-bounded staging sizes."""
        self._require_open()
        result = self._cache.stats()
        result["resident_scale_bytes"] = self._scale_owner.nbytes if self._scale_owner else 0
        result["owned_host_bytes"] = result["owned_host_bytes"] + result["resident_scale_bytes"]
        return result
