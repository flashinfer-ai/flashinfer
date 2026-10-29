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
"""Eager full scoring over CPU-described query views and prepared pitches."""

import torch


class _HostHintedFull:
    def __init__(self, full):
        from cutlass.cute.runtime import from_dlpack
        from ...attn_scores import attn_scores as scorer
        from ...attn_scores.kernels.schedule_kernel import _compile_schedule_kernel

        self.ends = torch.zeros_like(full._pair_ends)
        self.tables = torch.zeros_like(full._tables)
        self.lengths = torch.zeros_like(full._lengths)
        self.hints = torch.zeros_like(full._hints)
        self.args = tuple(
            from_dlpack(t, assumed_align=4)
            for t in (
                full._state.row_ends,
                full._state.row_requests,
                full._state.block_table,
                self.ends,
                self.tables,
                self.lengths,
                self.hints,
            )
        )
        self.buckets = []
        pitch = 8192
        while pitch < full._columns:
            self.buckets.append(pitch)
            pitch *= 2
        self.buckets.append(full._columns)
        self.schedules = {}
        self.kernels = {}
        self.arenas = {}
        self.heights = {}
        sms = torch.cuda.get_device_properties(full._device).multi_processor_count
        for pitch in self.buckets:
            height = min(
                full._state.capacity.max_chunk_size, full._scores.numel() // pitch
            )
            height = height // 2 * 2 if height > 1 else height
            if height < 1:
                raise ValueError("host-hinted score bucket must hold a query row")
            self.heights[pitch] = height
            self.arenas[pitch] = full._scores.view(-1)[: height * pitch].view(
                height, pitch
            )
            # An odd-request split can introduce at most one extra launch per
            # request; each schedule has stable, bucket-private storage.
            count = (full._state.capacity.max_chunk_size + height - 1) // height
            count += 2 * full._state.capacity.max_batch_size
            self.schedules[pitch] = tuple(
                torch.empty((sms + 1, 2), dtype=torch.int32, device=full._device)
                for _ in range(count)
            )
            self.kernels[pitch] = _compile_schedule_kernel(
                scorer._schedule_bucket(height),
                scorer._SPLIT_KV,
                sms,
                scorer._arch_for_launch(
                    full._device.index, "compute_paged_mqa_logits_schedule"
                ),
            )
        self.prepared = {}
        self.group_size = min(2, 128 // full._state.capacity.num_heads)
        self.bindings = {}
        self.stripes = ()
        self.pitch = self.buckets[-1]
        self.selected = full._output

    def _stripe(self, full, begin, end, n, schedule):
        return (
            full._state.q[begin:end].view(
                (end - begin) // n, n, *full._state.q.shape[1:]
            ),
            full._state.weights[begin:end],
            self.tables[begin + n - 1 : end : n],
            self.ends[begin + n - 1 : end : n],
            self.lengths[begin:end],
            self.hints[begin:end],
            full._output[begin:end],
            schedule,
        )

    def prepare_pool(self, full, pool):
        from ...topk_varlen.kernels import gvr2_topk_host as gvr

        key = pool.data_ptr()
        if key in self.prepared:
            return
        # Warm all route bands AND populate exact-row CPU launchers. Engine
        # compilation is banded, not one compilation per possible row count.
        for pitch in self.buckets:
            height = self.heights[pitch]
            gvr.warmup_varlen(
                full._top_k, pitch, num_rows_list=range(1, height + 1), row_stride=pitch
            )
            for n in (1, 2):
                if height < n or n > self.group_size:
                    continue
                stripe = self._stripe(full, 0, height, n, self.schedules[pitch][0])
                # Bucket warmup must not publish rows into the active step's
                # output, including its -1 inactive suffix.
                stripe = (*stripe[:6], full._indices[:height], stripe[-1])
                launches = full._bind_launches(
                    pool,
                    stripes=(stripe,),
                    group_rows=n,
                    columns=pitch,
                    scores=self.arenas[pitch],
                )
                self.kernels[pitch](stripe[3], stripe[-1], stripe[3].shape[0])
                for fn, arguments in launches:
                    fn(*arguments)
        self.prepared[key] = pool
        for _, stripe in self.stripes:
            self.kernels[self.pitch](stripe[3], stripe[-1], stripe[3].shape[0])
        if hasattr(self, "key"):
            self._bind(full, pool)

    def configure(self, full, prefix, bounds, rows):
        from .selective_logits import _current_stream

        maximum = max(bounds, default=0)
        self.pitch = next(p for p in self.buckets if p >= maximum)
        chunks = []
        # Merge only adjacent chunks of equal size, exactly as the ordinary
        # full route does; no pair crosses a request boundary.
        for start, stop in zip(prefix, prefix[1:], strict=False):
            for begin in range(start, stop, self.group_size):
                end = min(begin + self.group_size, stop)
                n = end - begin
                if chunks and chunks[-1][1] == begin and chunks[-1][2] == n:
                    chunks[-1] = (chunks[-1][0], end, n)
                else:
                    chunks.append((begin, end, n))
        stripes = []
        for start, stop, n in chunks:
            height = self.heights[self.pitch] // n * n
            for begin in range(start, stop, height):
                end = min(begin + height, stop)
                schedule = self.schedules[self.pitch][len(stripes)]
                stripes.append((n, self._stripe(full, begin, end, n, schedule)))
        self.stripes = tuple(stripes)
        self.selected = full._output[:rows]
        full._output[prefix[-1] : rows].fill_(-1)
        full._uniform_metadata[1](
            *self.args, full._padded_rows, _current_stream(full._device)
        )
        for _, stripe in self.stripes:
            self.kernels[self.pitch](stripe[3], stripe[-1], stripe[3].shape[0])
        self.key = (
            self.pitch,
            tuple((s[1][0].data_ptr(), s[1][1].shape[0], s[0]) for s in self.stripes),
        )
        for pool in self.prepared.values():
            self._bind(full, pool)

    def _bind(self, full, pool):
        pool_key = pool.data_ptr()
        key = (pool_key, self.key)
        if key not in self.bindings:
            self.bindings[key] = tuple(
                call
                for n, stripe in self.stripes
                for call in full._bind_launches(
                    pool,
                    stripes=(stripe,),
                    group_rows=n,
                    columns=self.pitch,
                    scores=self.arenas[self.pitch],
                )
            )

    def run(self, full, pool):
        launches = self.bindings.get((pool.data_ptr(), self.key))
        if launches is None:
            raise RuntimeError("prepare_pool must precede host-hinted full run")
        for fn, arguments in launches:
            fn(*arguments)
        return self.selected
