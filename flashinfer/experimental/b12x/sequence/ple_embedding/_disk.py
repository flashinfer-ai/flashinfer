"""PLE hashing/decoding over shared batch-bounded io_uring row storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._shared.disk_table import DiskRowCache

if TYPE_CHECKING:
    from ._contracts import Binding, TableLayout


class DiskTable:
    """Own PLE file sources and decode a compact batch cache outside graphs."""

    def __init__(self, layout: TableLayout, shard_rows: int, *, queue_depth: int = 64) -> None:
        from ._contracts import TableLayout

        if not isinstance(layout, TableLayout):
            raise TypeError("layout must be TableLayout")
        if layout.caps.table_memory != "io_uring":
            raise ValueError("DiskTable requires table_memory='io_uring'")
        self.layout = layout
        self._cache = DiskRowCache(
            device=layout.caps.device,
            max_lookups=layout.caps.max_tokens * layout.head_count,
            table_rows=layout.padded_vocab_size,
            shard_start=layout.shard_start,
            shard_end=layout.shard_end,
            shard_rows=shard_rows,
            weight_row_bytes=layout.weight_shape[1] * layout.weight_dtype.itemsize,
            scale_row_bytes=(
                layout.head_dim // 16 if layout.caps.quant_mode == "nvfp4_group16" else 0
            ),
            queue_depth=queue_depth,
        )
        self.weight = self._cache.weight.view(layout.weight_dtype)
        self.weight_host = self._cache.weight_host.view(layout.weight_dtype) if self._cache.weight_host is not None else None
        self.weight_scale = (
            self._cache.scale.view(torch.float8_e4m3fn)
            if self._cache.scale is not None
            else None
        )
        self.weight_scale_host = (
            self._cache.scale_host.view(torch.float8_e4m3fn)
            if self._cache.scale_host is not None
            else None
        )

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        self._cache.add_shard(shard_index, path, offset, scale=scale)

    def _require_complete(self) -> None:
        self._cache.require_complete()

    def _freeze(self) -> None:
        self._cache.freeze()

    def stats(self) -> dict[str, int | float]:
        return self._cache.stats()

    def _run(self, binding: Binding, *, state, token_count: int) -> None:
        layout = self.layout
        with self._cache.transaction():
            state.hash_state.run(binding._hash_binding, token_count=token_count)
            self._cache.read_rows(binding._ids, token_count * layout.head_count)
            if token_count:
                weight_scale = self.weight_scale if self._cache.scale_row_bytes else binding.weight_scale
                state.run_lookup(
                    self.weight, weight_scale, binding.weight_scale_2,
                    binding._ids, binding.num_tokens, binding.out, token_count=token_count,
                )
