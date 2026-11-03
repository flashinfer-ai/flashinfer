# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capacity-planned full logits with a bounded, serialized score arena."""

import torch

from .selective_capacity import (
    _CapacitySelectiveTopKPlan,
    _CapacityStepInputs,
    _SelectiveCapacity,
    _bind_metadata_module_lifetime,
)

_FULL_SCORE_BUDGET_BYTES = 512 * 1024 * 1024


class _CapacityFullTopKPlan:
    """Device-refreshed metadata and one score stripe shared by all pools."""

    def __init__(
        self,
        *,
        max_context_length,
        max_chunk_size,
        max_batch_size,
        kv_layout,
        num_pages,
        device,
        num_heads,
        head_dim,
        page_size,
        top_k,
        math_mode,
        max_workspace_size=None,
    ):
        from cutlass.cute.runtime import from_dlpack
        from ...attn_scores import attn_scores as scorer
        from ...topk_varlen.kernels.gvr2_topk_host import workspace_bytes
        from .kernels.selective_logits_metadata import (
            FullCapacityPairsKernel,
            FullCapacityPackKernel,
            FullCapacityScatterKernel,
            FullCapacityUniformKernel,
        )
        from .selective_logits import _byte_range, _compile_for_device, _current_stream

        capacity = _SelectiveCapacity(
            num_heads,
            head_dim,
            page_size,
            1,
            max_context_length,
            max_chunk_size,
            max_batch_size,
        )
        self._layout, device, pages = _CapacitySelectiveTopKPlan._resolve_layout(
            capacity, kv_layout, num_pages, device
        )
        self._columns = max(scorer.padded_seq_len(max_context_length), 8192, top_k)
        self._score_budget_bytes = (
            _FULL_SCORE_BUDGET_BYTES
            if max_workspace_size is None
            else max_workspace_size
        )
        minimum_bytes = 2 * self._columns * 4
        if self._score_budget_bytes < minimum_bytes:
            raise ValueError(
                f"max_workspace_size must be at least {minimum_bytes} bytes "
                "for one full-score query pair at capacity"
            )
        self._state = _CapacityStepInputs(capacity, device=device, num_pages=pages)
        self._device = self._state.device
        if not scorer._cached_dsl_targets_device(self._device.index):
            raise RuntimeError("capacity full scoring requires the GPU schedule kernel")
        packed_capacity = 2 * (
            (max_chunk_size + min(max_batch_size, max_chunk_size)) // 2
        )
        self._stripe_rows = min(
            packed_capacity, (self._score_budget_bytes // (4 * self._columns) // 2) * 2
        )
        self._padded_rows = (
            (packed_capacity + self._stripe_rows - 1) // self._stripe_rows
        ) * self._stripe_rows
        self._top_k, self._math_mode = top_k, math_mode
        self._q = torch.zeros(
            (self._padded_rows, num_heads, head_dim),
            dtype=torch.float8_e4m3fn,
            device=self._device,
        )
        self._weights = torch.zeros(
            (self._padded_rows, num_heads),
            dtype=torch.float32,
            device=self._device,
        )
        self._scores = torch.empty(
            (self._stripe_rows, self._columns),
            dtype=torch.float32,
            device=self._device,
        )
        self._lengths = torch.zeros(
            self._padded_rows, dtype=torch.int32, device=self._device
        )
        self._tables = torch.zeros(
            (self._padded_rows, self._state.block_table.shape[1]),
            dtype=torch.int32,
            device=self._device,
        )
        self._hints = torch.zeros(
            (self._padded_rows, top_k),
            dtype=torch.int32,
            device=self._device,
        )
        self._indices = torch.full_like(self._hints, -1)
        self._output = torch.full(
            (max_chunk_size, top_k), -1, dtype=torch.int32, device=self._device
        )
        self._pair_ends = torch.zeros(
            self._padded_rows, dtype=torch.int32, device=self._device
        )
        self._source_rows = torch.full(
            (self._padded_rows,), -1, dtype=torch.int32, device=self._device
        )
        self._inverse_rows = torch.zeros(
            max_chunk_size, dtype=torch.int32, device=self._device
        )
        self._gvr_workspace = torch.zeros(
            workspace_bytes() // 4,
            dtype=torch.int32,
            device=self._device,
        )
        self._workspace = {"gvr2_workspace": self._gvr_workspace}
        self.repair_error = torch.zeros(1, dtype=torch.int32, device=self._device)
        num_sms = torch.cuda.get_device_properties(self._device).multi_processor_count
        self._stripes = []
        for begin in range(0, self._padded_rows, self._stripe_rows):
            end = begin + self._stripe_rows
            schedule = torch.empty(
                (num_sms + 1, 2), dtype=torch.int32, device=self._device
            )
            self._stripes.append(
                (
                    self._q[begin:end].view(
                        self._stripe_rows // 2, 2, num_heads, head_dim
                    ),
                    self._weights[begin:end],
                    self._tables[begin // 2 : end // 2],
                    self._pair_ends[begin // 2 : end // 2],
                    self._lengths[begin:end],
                    self._hints[begin:end],
                    self._indices[begin:end],
                    schedule,
                )
            )
        args = tuple(
            from_dlpack(t, assumed_align=4)
            for t in (
                self._state.safe_qo_indptr,
                self._state.row_ends,
                self._state.block_table,
                self._pair_ends,
                self._tables,
                self._lengths,
                self._hints,
                self._source_rows,
                self._inverse_rows,
            )
        )
        self._metadata_args = args
        self._metadata = _bind_metadata_module_lifetime(
            _compile_for_device(
                self._device,
                FullCapacityPairsKernel(top_k),
                *args,
                self._padded_rows // 2,
                _current_stream(self._device),
            )
        )
        self._uniform_args = tuple(
            from_dlpack(t, assumed_align=4)
            for t in (
                self._state.row_ends,
                self._state.row_requests,
                self._state.block_table,
                self._pair_ends,
                self._tables,
                self._lengths,
                self._hints,
            )
        )
        self._uniform_metadata = {
            n: _bind_metadata_module_lifetime(
                _compile_for_device(
                    self._device,
                    FullCapacityUniformKernel(n, top_k),
                    *self._uniform_args,
                    self._padded_rows // n,
                    _current_stream(self._device),
                )
            )
            for n in (1, 2)
        }
        self._pack_args = tuple(
            from_dlpack(t, assumed_align=4)
            for t in (
                self._state.q.view(torch.uint8).reshape(max_chunk_size, -1),
                self._state.weights,
                self._source_rows,
                self._q.view(torch.uint8).reshape(self._padded_rows, -1),
                self._weights,
            )
        )
        self._pack = _bind_metadata_module_lifetime(
            _compile_for_device(
                self._device,
                FullCapacityPackKernel(),
                *self._pack_args,
                self._padded_rows,
                _current_stream(self._device),
            )
        )
        self._scatter_args = tuple(
            from_dlpack(t, assumed_align=4)
            for t in (
                self._state.row_requests,
                self._inverse_rows,
                self._indices,
                self._output,
            )
        )
        self._scatter = _bind_metadata_module_lifetime(
            _compile_for_device(
                self._device,
                FullCapacityScatterKernel(),
                *self._scatter_args,
                max_chunk_size,
                _current_stream(self._device),
            )
        )
        self._schedules = [stripe[-1] for stripe in self._stripes]
        from ...attn_scores.kernels.schedule_kernel import _compile_schedule_kernel

        with torch.cuda.device(self._device):
            self._schedule_kernel = _compile_schedule_kernel(
                scorer._schedule_bucket(self._stripe_rows),
                scorer._SPLIT_KV,
                num_sms,
                scorer._arch_for_launch(
                    self._device.index, "compute_paged_mqa_logits_schedule"
                ),
            )
        self._layouts = {}
        self._next_n = None
        self._configure(max_chunk_size, max_batch_size, None)
        self._state.prepare()
        self._refresh()
        self._planned = False
        self._supplied_rows = max_chunk_size
        self._selected = self._output
        self._pools = {}
        self._staging_ranges = tuple(
            (tensor.data_ptr(), _byte_range(tensor))
            for tensor in self._state.input_buffers().values()
        )
        from .full_host_hints import _HostHintedFull

        self._host_hints = _HostHintedFull(self)
        self._using_host_hints = False

    def _configure(self, rows, batches, next_n):
        """Select stable views over constructed storage; never allocate CUDA memory."""
        key = (rows, batches, next_n)
        cached = self._layouts.get(key)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prepare_pool must precede capturing a new full layout"
                )
            n = 1 if next_n == 1 else 2
            # A valid uniform prefix contains whole groups. Any incomplete
            # supplied-storage tail is inactive, not a reason to reject plan.
            count = (
                rows // n * n
                if next_n is not None
                else 2 * ((rows + min(batches, rows)) // 2)
            )
            q = self._state.q if next_n is not None else self._q
            weights = self._state.weights if next_n is not None else self._weights
            output = self._output if next_n is not None else self._indices
            stripes = []
            for i, begin in enumerate(range(0, count, self._stripe_rows)):
                end = min(begin + self._stripe_rows, count)
                stripes.append(
                    (
                        q[begin:end].view((end - begin) // n, n, *q.shape[1:]),
                        weights[begin:end],
                        self._tables[begin // n : end // n],
                        self._pair_ends[begin // n : end // n],
                        self._lengths[begin:end],
                        self._hints[begin:end],
                        output[begin:end],
                        self._schedules[i],
                    )
                )
            cached = (
                tuple(stripes),
                count,
                n,
                self._output[:rows],
                self._output[min(count, rows) : rows],
            )
            self._layouts[key] = cached
        (
            self._stripes,
            self._work_rows,
            self._group_rows,
            self._selected,
            self._inactive_tail,
        ) = cached
        self._layout_key, self._next_n = key, next_n

    def _refresh(self):
        from .selective_logits import _current_stream

        self._state.refresh()
        if self._work_rows == 0:
            return
        if self._next_n is None:
            self._metadata(
                *self._metadata_args,
                self._work_rows // 2,
                _current_stream(self._device),
            )
        else:
            self._uniform_metadata[self._group_rows](
                *self._uniform_args,
                self._work_rows // self._group_rows,
                _current_stream(self._device),
            )
        for _, _, _, lengths, _, _, _, schedule in self._stripes:
            self._schedule_kernel(lengths, schedule, lengths.shape[0])

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
        # Host validation may reject after layout/hint staging has begun.
        # Never let run consume a partially published step.
        self._planned = False
        rows = (
            self._state.capacity.max_chunk_size
            if num_query_rows is None
            else num_query_rows
        )
        if (
            type(rows) is not int
            or not 0 <= rows <= self._state.capacity.max_chunk_size
        ):
            raise ValueError("num_query_rows must fit max_chunk_size")
        if next_n is not None and (type(next_n) is not int or next_n not in (1, 2, 4)):
            raise ValueError("next_n must be None, 1, 2 or 4")
        from .host_length_hints import _validate_host_length_hints

        hints = _validate_host_length_hints(
            host_length_hints, qo_indptr.numel() - 1, self._state.capacity, rows
        )
        # H never replaces the capacity layout or its launch bindings. A
        # run-only caller graph retains its own static views and schedules.
        if hints is None:
            self._configure(rows, qo_indptr.numel() - 1, next_n)
        self._state._host_hint_validation.stage(host_length_hints)
        self._state.stage_step(
            qo_indptr, kv_indptr, block_table, num_query_rows=rows, next_n=next_n
        )
        self._refresh()
        self._inactive_tail.fill_(-1)
        self.repair_error.zero_()
        self._supplied_rows = rows
        self._using_host_hints = hints is not None
        if hints is not None:
            self._host_hints.configure(self, *hints, rows)
        self._planned = True

    def _pool_key(self, pool):
        if (
            _CapacitySelectiveTopKPlan._layout(pool) != self._layout
            or not pool.is_contiguous()
        ):
            raise ValueError("KV pool layout must match the constructed layout")
        return pool.data_ptr()

    def _execute(self, pool, bindings=None):
        from ...attn_scores.attn_scores import fp8_paged_mqa_logits
        from ...topk_varlen.topk_varlen import top_k_varlen
        from .selective_logits import _current_stream

        stream = _current_stream(self._device)
        if self._next_n is None and self._work_rows:
            self._pack(*self._pack_args, self._work_rows, stream)
        if bindings is None:
            # Public calls validate and materialize the exact specializations
            # outside capture. The prepared path below uses those same kernels.
            for (
                q,
                weights,
                table,
                pair_ends,
                lengths,
                hints,
                indices,
                schedule,
            ) in self._stripes:
                fp8_paged_mqa_logits(
                    q,
                    pool,
                    weights,
                    table,
                    pair_ends,
                    self._state.capacity.max_context_length,
                    math_mode=self._math_mode,
                    schedule_meta=schedule,
                    out=self._scores[: weights.shape[0]],
                )
                top_k_varlen(
                    self._scores[: weights.shape[0]],
                    lengths,
                    self._top_k,
                    pre_idx=hints,
                    out_indices=indices,
                    workspace=self._workspace,
                    backend="gvr_2",
                )
        else:
            for function, arguments in bindings:
                function(*arguments)
        if self._next_n is None and self._supplied_rows:
            self._scatter(*self._scatter_args, self._supplied_rows, stream)
        return self._selected

    def _bind_launches(
        self, pool, *, stripes=None, group_rows=None, columns=None, scores=None
    ):
        """Bind the warmed public scorer/GVR kernels without per-layer dispatch."""
        import cutlass
        from ...attn_scores import attn_scores as scorer
        from ...topk_varlen.kernels import gvr2_topk_host as gvr

        capacity = self._state.capacity
        stripes = self._stripes if stripes is None else stripes
        group_rows = self._group_rows if group_rows is None else group_rows
        columns = self._columns if columns is None else columns
        scores = self._scores if scores is None else scores
        num_sms = torch.cuda.get_device_properties(self._device).multi_processor_count
        epi, registers = scorer._fp8_epilogue_policy(
            capacity.num_heads, group_rows, cutlass.Float32
        )
        compiled = scorer._cached_compile_fp8_kernel(
            capacity.page_size,
            capacity.num_heads,
            capacity.head_dim,
            group_rows,
            num_sms,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            epi,
            registers,
            scorer._arch_for_launch(self._device.index, "fp8_paged_mqa_logits"),
            scorer._fp8_output_index_bits(scores.shape[0], columns, scores, 4),
            self._math_mode,
            128,
            False,
            None,
            None,
        )
        pool_flat = pool.flatten(1)
        launches = []
        for (
            q,
            weights,
            table,
            pair_ends,
            lengths,
            hints,
            indices,
            schedule,
        ) in stripes:
            rows = weights.shape[0]
            groups = rows // group_rows
            score_rows = scores[:rows]
            logits = score_rows[:, : min(capacity.max_context_length, columns)]
            q_ffi = (
                q.reshape(groups, group_rows * capacity.num_heads, capacity.head_dim)
                .permute(1, 2, 0)
                .view(torch.uint8)
            )
            weights_ffi = weights.reshape(groups, group_rows * capacity.num_heads).t()
            launches.append(
                (
                    compiled,
                    (
                        pool_flat,
                        q_ffi,
                        weights_ffi,
                        logits,
                        table,
                        pair_ends,
                        schedule,
                        pool.shape[0],
                        groups,
                        None,
                    ),
                )
            )
            launcher = gvr._varlen_launcher(rows, columns, self._top_k, columns, 1, 1)
            if launcher[0] == "reg_clus":
                arguments = (score_rows, hints, lengths, indices, launcher[2])
            elif launcher[0] in ("reg", "clus"):
                arguments = (score_rows, hints, lengths, indices, *launcher[2])
            else:
                _, _, prefix, tail = launcher
                arguments = (
                    score_rows,
                    hints,
                    indices,
                    self._gvr_workspace,
                    *prefix,
                    lengths,
                    *tail,
                )
            launches.append((launcher[1], arguments))
        return tuple(launches)

    def prepare_pool(self, pool):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_pool must run outside CUDA graph capture")
        key = (self._pool_key(pool), self._layout_key)
        if key not in self._pools:
            self._execute(pool)
            self._pools[key] = (pool, self._bind_launches(pool))
        self._host_hints.prepare_pool(self, pool)

    def run(self, q, weights, pool, *, out=None):
        from .selective_logits import _byte_range, _overlap

        if not self._planned:
            raise RuntimeError("plan must be called before run")
        if self._using_host_hints and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "host_length_hints plans are eager-only; use an unhinted plan before capturing run"
            )
        copies = self._state._payload_copies(q, weights)
        if q.shape[0] != self._supplied_rows:
            raise ValueError("Q rows must match plan num_query_rows")
        binding = self._pools.get((self._pool_key(pool), self._layout_key))
        if binding is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prepare_pool or one eager run must precede caller capture"
                )
            self.prepare_pool(pool)
            binding = self._pools[(self._pool_key(pool), self._layout_key)]
        if out is not None and (
            out.device != self._device
            or out.dtype != torch.int32
            or out.shape != (q.shape[0], self._top_k)
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous CUDA int32 [Q, top_k]")
        # Staging owners never change; cache their ranges rather than building
        # tensors/dictionaries and recomputing each range for every source.
        for destination, source in copies:
            source_range = _byte_range(source)
            for owner_pointer, owner_range in self._staging_ranges:
                if _overlap(source_range, owner_range) and not (
                    owner_pointer == destination.data_ptr() == source.data_ptr()
                ):
                    raise ValueError("step inputs must not cross-alias staging buffers")
        for destination, source in copies:
            if destination.data_ptr() != source.data_ptr():
                destination.copy_(source)
        from ...attn_scores.attn_scores import _on_device

        with _on_device(self._device.index):
            selected = (
                self._host_hints.run(self, pool)
                if self._using_host_hints
                else self._execute(pool, binding[1])
            )
        if out is not None:
            out.copy_(selected)
            return out
        return selected

    def clear_pools(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("clear_pools must run outside CUDA graph capture")
        torch.cuda.synchronize(self._device)
        self._pools.clear()
        self._host_hints.prepared.clear()
        self._host_hints.bindings.clear()
