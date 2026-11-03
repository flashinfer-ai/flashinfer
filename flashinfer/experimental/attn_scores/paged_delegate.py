# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Paged cache staging for the continuous TopK public API."""

import inspect

import torch

from ...api_logging import flashinfer_api
from ...attn_scores.selective_logits import (
    FP8MQATopKWrapper,
    FP8PagedMQATopKWrapper as _PublicWrapper,
)
from .selective_capacity import (
    _CapacitySelectiveTopKPlan,
    _CapacityStepInputs,
    _SelectiveCapacity,
)
from .selective_logits import _byte_range, _overlap, _public_wrapper_method
from .trace import fp8_paged_mqa_topk_trace_dispatch


class FP8PagedMQATopKWrapper(_PublicWrapper):
    def _initialize(self, *, page_size, kv_layout, num_pages, device, **options):
        if any(
            options[name] is None
            for name in ("max_context_length", "max_chunk_size", "max_batch_size")
        ):
            raise ValueError(
                "both strategies require max_context_length, max_chunk_size and max_batch_size"
            )
        fp4 = options.get("qkv_dtype", "fp8") == "mxfp4"
        width = options["head_dim"] // (2 if fp4 else 1)
        capacity = _SelectiveCapacity(
            options["num_heads"],
            width,
            page_size,
            options["compression_ratio"],
            options["max_context_length"],
            options["max_chunk_size"],
            options["max_batch_size"],
        )
        self._layout, device, pages = _CapacitySelectiveTopKPlan._resolve_layout(
            capacity, kv_layout, num_pages, device
        )
        count = max(
            1,
            capacity.max_batch_size
            * (capacity.max_context_length // capacity.compression_ratio),
        )
        if fp4:
            count = (count + 255) // 256 * 256
        self._delegate = object.__new__(FP8MQATopKWrapper)
        # The outer public constructor owns the experimental warning/log entry.
        inspect.unwrap(FP8MQATopKWrapper.__init__)(
            self._delegate, device=device, num_keys=count, **options
        )
        self._capacity_plan = self._delegate._capacity_plan
        self._paged_state = _CapacityStepInputs(
            capacity, device=device, num_pages=pages
        )
        self._keys = torch.zeros(
            (count, width),
            dtype=torch.uint8 if fp4 else torch.float8_e4m3fn,
            device=device,
        )
        self._scales = torch.zeros(
            (count, 4) if fp4 else (count,),
            dtype=torch.uint8 if fp4 else torch.float32,
            device=device,
        )
        self._key_words = self._keys.view(torch.int32)
        self._scale_words = self._scales.view(torch.int32).view(-1)
        self._positions = torch.arange(count, dtype=torch.int64, device=device)
        self._requests = torch.empty_like(self._positions)
        self._local = torch.empty_like(self._positions)
        self._within = torch.empty_like(self._positions)
        self._slots = torch.empty_like(self._positions)
        self._key_indices = torch.empty_like(self._positions)
        self._scale_indices = torch.empty_like(self._positions)
        self._page_ids = torch.empty(count, dtype=torch.int32, device=device)
        self._prefix = torch.zeros(
            capacity.max_batch_size + 1, dtype=torch.int64, device=device
        )
        self._padding = torch.empty(count, dtype=torch.bool, device=device)
        self._invalid = torch.zeros(1, dtype=torch.bool, device=device)
        self._error = torch.zeros(1, dtype=torch.int32, device=device)
        self._pools = {}
        self._planned = False
        self._strategy, self._backend = self._delegate.strategy, options["backend"]
        self._math_mode, self._compression_ratio = (
            self._delegate.math_mode,
            capacity.compression_ratio,
        )
        self._page_size, self._width = page_size, width
        self._columns = self._paged_state.block_table.shape[1]
        self._has_keys = bool(pages and self._columns)
        self._table = self._paged_state.block_table.view(-1)
        self._ranges = tuple(
            _byte_range(value)
            for value in (
                *vars(self).values(),
                *vars(self._paged_state).values(),
                *vars(self._paged_state._host_hint_validation).values(),
            )
            if isinstance(value, torch.Tensor) and value.numel()
        )

    def _refresh_indices(self):
        state = self._paged_state
        state.refresh()
        if not self._has_keys:
            return
        self._prefix.copy_(state.physical_kv_indptr)
        batches = (
            0 if state._written_step_shape is None else state._written_step_shape[1]
        )
        if batches < state.capacity.max_batch_size:
            self._prefix[batches + 1 :].copy_(
                self._prefix[batches].expand(state.capacity.max_batch_size - batches)
            )
        torch.searchsorted(
            self._prefix[1:], self._positions, right=True, out=self._requests
        )
        self._requests.clamp_(max=state.capacity.max_batch_size - 1)
        torch.ge(self._positions, self._prefix[-1], out=self._padding)
        self._requests.masked_fill_(self._padding, 0)
        torch.index_select(self._prefix, 0, self._requests, out=self._local)
        torch.sub(self._positions, self._local, out=self._local)
        self._local.masked_fill_(self._padding, 0)
        torch.div(self._local, self._page_size, rounding_mode="floor", out=self._slots)
        self._slots.add_(self._requests, alpha=self._columns)
        torch.index_select(self._table, 0, self._slots, out=self._page_ids)
        self._page_ids.clamp_(0, state.num_pages - 1)
        torch.remainder(self._local, self._page_size, out=self._within)
        self._key_indices.copy_(self._page_ids)
        self._key_indices.mul_(self._page_size * (self._width + 4) // self._width).add_(
            self._within
        )
        self._scale_indices.copy_(self._page_ids)
        self._scale_indices.mul_(self._page_size * (self._width + 4) // 4)
        self._scale_indices.add_(self._page_size * self._width // 4).add_(self._within)

    def _status(self):
        torch.ne(self._paged_state.error, 0, out=self._invalid)
        torch.where(
            self._invalid,
            self._paged_state.error,
            self._delegate.error,
            out=self._error,
        )

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        block_table,
        *,
        num_query_rows=None,
        next_n=None,
        host_length_hints=None,
    ):
        from .host_length_hints import _validate_host_length_hints

        self._planned = False
        state = self._paged_state
        rows = (
            state.capacity.max_chunk_size if num_query_rows is None else num_query_rows
        )
        _validate_host_length_hints(
            host_length_hints, qo_indptr.numel() - 1, state.capacity, rows
        )
        if torch.cuda.is_current_stream_capturing() and not state.is_prepared:
            raise RuntimeError(
                "prepare_pool or one eager run must precede capturing plan"
            )
        state._host_hint_validation.stage(host_length_hints)
        state.stage_step(
            qo_indptr, kv_indptr, block_table, num_query_rows=rows, next_n=next_n
        )
        if state.is_prepared:
            self._refresh_indices()
        self._delegate.plan(
            qo_indptr,
            kv_indptr,
            num_query_rows=rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )
        self._planned = True
        self._status()

    def _pool_key(self, pool):
        if (
            _CapacitySelectiveTopKPlan._layout(pool) != self._layout
            or pool.data_ptr() % 16
        ):
            raise ValueError("pool must match the constructed CUDA paged KV layout")
        return pool.data_ptr()

    def _gather(self, views):
        if self._has_keys:
            torch.index_select(views[0], 0, self._key_indices, out=self._key_words)
            torch.index_select(views[1], 0, self._scale_indices, out=self._scale_words)

    def prepare_pool(self, kv_fused):
        key = self._pool_key(kv_fused)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_pool must run outside CUDA graph capture")
        if not self._paged_state.is_prepared:
            self._paged_state.prepare()
        self._refresh_indices()
        words = kv_fused.view(torch.int32).reshape(-1)
        views = (words.view(-1, self._width // 4), words)
        self._gather(views)
        self._delegate.prepare_pool(self._keys, k_scale=self._scales)
        self._pools[key] = (kv_fused, views)
        self._status()

    @flashinfer_api(trace=fp8_paged_mqa_topk_trace_dispatch)
    @_public_wrapper_method
    def run(self, q, weights, kv_fused, *, out=None, q_scale=None):
        if not self._planned:
            raise RuntimeError("plan must be called before run")
        key = self._pool_key(kv_fused)
        if out is not None and isinstance(out, torch.Tensor) and out.numel():
            bounds = _byte_range(out)
            pool_ranges = (_byte_range(pool) for pool, _ in self._pools.values())
            if any(
                _overlap(bounds, other)
                for other in (*self._ranges, *pool_ranges, _byte_range(kv_fused))
            ):
                raise ValueError("out must not overlap inputs or plan-owned storage")
        if key not in self._pools:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prepare_pool or one eager run must precede caller capture"
                )
            self.prepare_pool(kv_fused)
        self._gather(self._pools[key][1])
        selected = self._delegate.run(
            q, weights, self._keys, k_scale=self._scales, out=out, q_scale=q_scale
        )
        self._status()
        selected.masked_fill_(self._invalid, -1)
        return selected

    def clear_pools(self):
        self._delegate.clear_pools()
        self._pools.clear()

    def input_buffers(self):
        buffers = self._delegate.input_buffers()
        buffers.update(
            qo_indptr=self._paged_state.qo_indptr,
            kv_indptr=self._paged_state.kv_indptr,
            block_table=self._paged_state.block_table,
        )
        return buffers

    @property
    def error(self):
        return self._error

    @property
    def repair_error(self):
        return self._delegate.repair_error

    @property
    def active_shape(self):
        return self._paged_state.active_shape
