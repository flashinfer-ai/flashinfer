# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build GPU sampling, candidate and persistent-schedule metadata."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.utils.smem_allocator import SmemAllocator

from .._selective_contract import _SAMPLE_HISTOGRAM_BINS

_KV_PAIR_TOKENS = 256
# Request-count bound for the shared tile prefix of the explicit schedule.
_OWNER_PREFIX_CAP = 4096


_REFRESH_THREADS = 512
# Pages one check work unit covers (four per thread); a request spans
# ceil(columns / _REFRESH_PAGE_CHUNK) units so long requests spread over CTAs.
_REFRESH_PAGE_CHUNK = 2048
_REFRESH_ROWS_PER_CTA = 2048


class FullCapacityPairsKernel:
    """Front-pad requests into pairs without changing real causal bounds."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        qo,
        row_ends,
        table,
        pair_ends,
        row_tables,
        lengths,
        hints,
        source_rows,
        inverse_rows,
        groups: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            qo,
            row_ends,
            table,
            pair_ends,
            row_tables,
            lengths,
            hints,
            source_rows,
            inverse_rows,
        ).launch(grid=(groups, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        qo,
        row_ends,
        table,
        pair_ends,
        row_tables,
        lengths,
        hints,
        source_rows,
        inverse_rows,
    ):
        pair, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        remaining = cutlass.Int32(pair)
        request = cutlass.Int32(0)
        owner = cutlass.Int32(-1)
        begin = cutlass.Int32(0)
        count = cutlass.Int32(0)
        while (request < qo.shape[0] - 1) & (owner < 0):
            first = qo[request]
            size = qo[request + 1] - first
            pairs = (size + 1) // 2
            if remaining < pairs:
                owner = request
                begin = first
                count = size
            else:
                remaining -= pairs
            request += 1
        final_end = cutlass.Int32(0)
        if owner >= 0:
            last = begin + remaining * 2 + 1 - (count % 2)
            final_end = row_ends[last]
        if thread == 0:
            pair_ends[pair] = final_end
        column = thread
        while column < row_tables.shape[1]:
            page = cutlass.Int32(0)
            if owner >= 0:
                page = table[(owner, column)]
            row_tables[(pair, column)] = page
            column += 256
        for slot in cutlass.range_constexpr(2):
            packed_row = pair * 2 + slot
            source = cutlass.Int32(-1)
            end = cutlass.Int32(0)
            if owner >= 0:
                candidate = begin + remaining * 2 + slot - (count % 2)
                if candidate >= begin:
                    source = candidate
                    end = row_ends[source]
            if thread == 0:
                source_rows[packed_row] = source
                lengths[packed_row] = end
                if source >= 0:
                    inverse_rows[source] = packed_row
            rank = thread
            while rank < self.top_k:
                hints[(packed_row, rank)] = cutlass.Int32(
                    cutlass.Int64(end) * cutlass.Int64(rank) // self.top_k
                )
                rank += 256


class FullCapacityPackKernel:
    """Copy real payload rows and zero the request-local padding slots."""

    @cute.jit
    def __call__(
        self,
        q,
        weights,
        source_rows,
        packed_q,
        packed_weights,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(q, weights, source_rows, packed_q, packed_weights).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, q, weights, source_rows, packed_q, packed_weights):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        source = source_rows[row]
        column = thread
        while column < q.shape[1]:
            value = cutlass.Uint8(0)
            if source >= 0:
                value = q[(source, column)]
            packed_q[(row, column)] = value
            column += 256
        if thread < weights.shape[1]:
            value = cutlass.Float32(0)
            if source >= 0:
                value = weights[(source, thread)]
            packed_weights[(row, thread)] = value


class FullCapacityScatterKernel:
    """Restore compact output order and clear inactive rows on every layer."""

    @cute.jit
    def __call__(
        self,
        row_requests,
        inverse_rows,
        indices,
        output,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(row_requests, inverse_rows, indices, output).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, row_requests, inverse_rows, indices, output):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        rank = thread
        while rank < output.shape[1]:
            value = cutlass.Int32(-1)
            if row_requests[row] >= 0:
                source = inverse_rows[row]
                value = indices[(source, rank)]
            output[(row, rank)] = value
            rank += 256


class FullCapacityUniformKernel:
    """Publish direct uniform groups with no payload packing or padding."""

    def __init__(self, group_rows: int, top_k: int):
        self.group_rows = group_rows
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        row_ends,
        row_requests,
        table,
        group_ends,
        row_tables,
        lengths,
        hints,
        groups: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            row_ends, row_requests, table, group_ends, row_tables, lengths, hints
        ).launch(grid=(groups, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, row_ends, row_requests, table, group_ends, row_tables, lengths, hints
    ):
        group, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        last = group * self.group_rows + self.group_rows - 1
        owner = cutlass.Int32(-1)
        end = cutlass.Int32(0)
        if last < row_ends.shape[0]:
            owner = row_requests[last]
            end = row_ends[last]
        if thread == 0:
            group_ends[group] = end
        column = thread
        while column < row_tables.shape[1]:
            page = cutlass.Int32(0)
            if owner >= 0:
                page = table[(owner, column)]
            row_tables[(group, column)] = page
            column += 256
        for slot in cutlass.range_constexpr(self.group_rows):
            row = group * self.group_rows + slot
            length = cutlass.Int32(0)
            if row < row_ends.shape[0]:
                length = row_ends[row]
            if thread == 0:
                lengths[row] = length
            rank = thread
            while rank < self.top_k:
                hints[(row, rank)] = cutlass.Int32(
                    cutlass.Int64(length) * rank // self.top_k
                )
                rank += 256


class SelectiveCapacityRefreshKernel:
    """Validate a staged step before any dependent map or scorer access.

    Reads int32 query/KV prefixes [B+1], staged extents [2] and a physical-page
    table [B,P]; publishes safe per-request prefixes, active extents, int32 row
    ends/owners across the fixed row capacity and one error word. KV prefixes
    and the context bound use original tokens; outputs use per-request physical
    floors.

    Two capturable launches with static grids:

    - ``check`` (``check_ctas`` CTAs): every CTA re-derives the extent/prefix/
      capacity bits with a block reduction, then, only when they are clear,
      checks the live entries of the page table. Work units are
      ``(request, chunk of _REFRESH_PAGE_CHUNK pages)``, so live pages of one
      long request spread over CTAs and padding columns are never read. Each
      CTA overwrites its own ``page_flags`` slot (8 or 0); no reset is needed.
    - ``publish`` (``ceil(max_rows / _REFRESH_ROWS_PER_CTA)`` CTAs): re-derives
      the same bits, ORs ``page_flags`` when they are clear, and publishes.
      CTA 0 writes the error word, active extents and both prefixes with a
      block scan over ``max_batch``; every CTA writes its slice of row ends
      and owners, finding each row's owner by binary search over ``qo``.

    The staged row extent bounds rather than equals the final query prefix.
    Invalid extents suppress prefix reads; invalid prefixes/capacities suppress
    all map reads. Errors reset active counts and prefixes to 0, row ends to 0
    and owners to -1. Error bits 1/2/4/8 mean extents/prefixes/capacity/page IDs;
    only prefix and capacity bits may coexist. Every launch rewrites all outputs.
    Consumers must run later on the same stream and use the sanitized outputs.
    """

    def __init__(
        self, max_context, max_rows, max_batch, page_size, ratio, num_pages, check_ctas
    ):
        self.max_context = max_context
        self.max_rows = max_rows
        self.max_batch = max_batch
        self.page_size = page_size
        self.ratio = ratio
        self.num_pages = num_pages
        self.check_ctas = check_ctas
        self.publish_ctas = max(
            1, (max_rows + _REFRESH_ROWS_PER_CTA - 1) // _REFRESH_ROWS_PER_CTA
        )

    @cute.jit
    def __call__(
        self,
        qo: cute.Tensor,
        kv: cute.Tensor,
        step: cute.Tensor,
        table: cute.Tensor,
        safe_qo: cute.Tensor,
        physical_kv: cute.Tensor,
        active: cute.Tensor,
        row_ends: cute.Tensor,
        row_requests: cute.Tensor,
        error: cute.Tensor,
        page_flags: cute.Tensor,
        next_n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.check_kernel(qo, kv, step, table, page_flags, next_n).launch(
            grid=(self.check_ctas, 1, 1),
            block=(_REFRESH_THREADS, 1, 1),
            stream=stream,
        )
        self.publish_kernel(
            qo,
            kv,
            step,
            table,
            page_flags,
            safe_qo,
            physical_kv,
            active,
            row_ends,
            row_requests,
            error,
            next_n,
        ).launch(
            grid=(self.publish_ctas, 1, 1),
            block=(_REFRESH_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _block_or(self, value, shared, tidx):
        """OR one Int32 across the CTA; ``shared`` is a one-word scratch."""
        cute.arch.sync_threads()
        if tidx == 0:
            shared[0] = 0
        cute.arch.sync_threads()
        if value != 0:
            cute.arch.atomic_or(shared.iterator, value, sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        return shared[0]

    @cute.jit
    def _prefix_code(self, qo, kv, step, table, shared, tidx, next_n):
        """Extent bit 1, or the OR of prefix bit 2 and capacity bit 4."""
        supplied_rows = step[0]
        batches = step[1]
        code = cutlass.Int32(0)
        if (
            (supplied_rows < 0)
            | (supplied_rows > self.max_rows)
            | (batches < 0)
            | (batches > self.max_batch)
        ):
            code = cutlass.Int32(1)
        local = cutlass.Int32(0)
        if code == 0:
            if tidx == 0:
                if (qo[0] != 0) | (kv[0] != 0):
                    local = local | 2
            request = tidx
            while request < batches:
                qb, qe = cutlass.Int64(qo[request]), cutlass.Int64(qo[request + 1])
                kb, ke = cutlass.Int64(kv[request]), cutlass.Int64(kv[request + 1])
                if (qb < 0) | (qe < qb) | (kb < 0) | (ke < kb) | (ke - kb < qe - qb):
                    local = local | 2
                if (next_n != 0) & (qe - qb != next_n):
                    local = local | 2
                if (qe > supplied_rows) | (ke - kb > self.max_context):
                    local = local | 4
                physical = (ke - kb) // self.ratio
                if cute.ceil_div(physical, self.page_size) > table.shape[1]:
                    local = local | 4
                request = request + _REFRESH_THREADS
        # Bit 1 is CTA-uniform, so every thread reaches the same barriers.
        return code | self._block_or(local, shared, tidx)

    @cute.kernel
    def check_kernel(
        self,
        qo: cute.Tensor,
        kv: cute.Tensor,
        step: cute.Tensor,
        table: cute.Tensor,
        page_flags: cute.Tensor,
        next_n: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        shared = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=16,
        )
        code = self._prefix_code(qo, kv, step, table, shared, tidx, next_n)
        local = cutlass.Int32(0)
        if cutlass.const_expr(table.shape[1] > 0):
            # Invalid prefixes suppress every map read. Padding columns and
            # empty requests do not belong to the input domain.
            if code == 0:
                chunks = cutlass.const_expr(
                    (table.shape[1] + _REFRESH_PAGE_CHUNK - 1) // _REFRESH_PAGE_CHUNK
                )
                unit = cutlass.Int32(cta)
                while unit < step[1] * chunks:
                    request = unit // chunks
                    first = (unit % chunks) * _REFRESH_PAGE_CHUNK
                    length = (
                        cutlass.Int64(kv[request + 1]) - cutlass.Int64(kv[request])
                    ) // self.ratio
                    pages = min(
                        cutlass.Int32(cute.ceil_div(length, self.page_size)),
                        first + _REFRESH_PAGE_CHUNK,
                    )
                    page = first + tidx
                    while page < pages:
                        index = table[request, page]
                        if (index < 0) | (index >= self.num_pages):
                            local = cutlass.Int32(8)
                        page = page + _REFRESH_THREADS
                    unit = unit + self.check_ctas
        flags = self._block_or(local, shared, tidx)
        if tidx == 0:
            page_flags[cta] = flags

    @cute.jit
    def _block_inclusive_sum(self, value, warp_totals, tidx):
        """Inclusive CTA sum of one Int32 per thread, plus the CTA total."""
        lane = tidx % 32
        warp = tidx // 32
        inclusive = value
        for step in cutlass.range_constexpr(5):
            prior = cute.arch.shuffle_sync_up(
                inclusive, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= 1 << step:
                inclusive = inclusive + prior
        cute.arch.sync_threads()
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()
        before = cutlass.Int32(0)
        total = cutlass.Int32(0)
        for other in cutlass.range_constexpr(_REFRESH_THREADS // 32):
            if other < warp:
                before = before + warp_totals[other]
            total = total + warp_totals[other]
        return before + inclusive, total

    @cute.kernel
    def publish_kernel(
        self,
        qo: cute.Tensor,
        kv: cute.Tensor,
        step: cute.Tensor,
        table: cute.Tensor,
        page_flags: cute.Tensor,
        safe_qo: cute.Tensor,
        physical_kv: cute.Tensor,
        active: cute.Tensor,
        row_ends: cute.Tensor,
        row_requests: cute.Tensor,
        error: cute.Tensor,
        next_n: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        shared = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=16,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((_REFRESH_THREADS // 32,), order=(0,)),
            byte_alignment=16,
        )
        code = self._prefix_code(qo, kv, step, table, shared, tidx, next_n)
        # Page bit 8 counts only when the earlier checks pass.
        local = cutlass.Int32(0)
        if code == 0:
            slot = tidx
            while slot < self.check_ctas:
                local = local | page_flags[slot]
                slot = slot + _REFRESH_THREADS
        code = code | self._block_or(local, shared, tidx)

        batches = step[1]
        active_rows = cutlass.Int32(0)
        active_batches = cutlass.Int32(0)
        if code == 0:
            active_rows = qo[batches]
            active_batches = batches

        if cta == 0:
            if tidx == 0:
                error[0] = code
                active[0] = active_rows
                active[1] = active_batches
                safe_qo[0] = cutlass.Int32(0)
                physical_kv[0] = cutlass.Int32(0)
            carry = cutlass.Int32(0)
            base = cutlass.Int32(0)
            while base < self.max_batch:
                request = base + tidx
                physical = cutlass.Int32(0)
                query_end = active_rows
                if request < active_batches:
                    query_end = qo[request + 1]
                    physical = cutlass.Int32(
                        (cutlass.Int64(kv[request + 1]) - cutlass.Int64(kv[request]))
                        // self.ratio
                    )
                inclusive, total = self._block_inclusive_sum(
                    physical, warp_totals, tidx
                )
                if request < self.max_batch:
                    safe_qo[request + 1] = query_end
                    # Valid original int32 prefixes are nonnegative and
                    # monotonic, so the physical prefix fits int32.
                    physical_kv[request + 1] = carry + inclusive
                carry = carry + total
                base = base + _REFRESH_THREADS

        row = cta * _REFRESH_ROWS_PER_CTA + tidx
        row_limit = min((cta + 1) * _REFRESH_ROWS_PER_CTA, self.max_rows)
        while row < row_limit:
            row_end = cutlass.Int32(0)
            owner = cutlass.Int32(-1)
            if row < active_rows:
                # Largest request whose query prefix starts at or before row;
                # prefixes are validated monotonic, so empty requests are skipped.
                low = cutlass.Int32(0)
                high = active_batches - 1
                while low < high:
                    middle = (low + high + 1) // 2
                    if qo[middle] <= row:
                        low = middle
                    else:
                        high = middle - 1
                owner = low
                q_length = qo[owner + 1] - qo[owner]
                kv_length = kv[owner + 1] - kv[owner]
                row_end = cutlass.Int32(
                    (cutlass.Int64(kv_length) - q_length + row - qo[owner] + 1)
                    // self.ratio
                )
            row_ends[row] = row_end
            row_requests[row] = owner
            row = row + _REFRESH_THREADS


class SelectiveCapacitySampleMetadataKernel:
    """Derive direct-page sample coordinates and ranks from sanitized metadata.

    Inputs are int32 physical request prefixes [B+1], causal row ends [R] and
    request owners [R] (owner -1 marks inactive rows). Outputs are int64 logical
    page columns [B,P] plus int32 valid sample counts and 1-based ranks [R].
    P is sample_capacity/page_size. Selected pages are uniformly spaced,
    including first/last; unused slots are 0 and consumers use valid counts.
    Empty/inactive rows get count/rank 0. All outputs are rewritten each launch.

    With group_rows > 0, request-local groups share the first causal end and
    score min(end, sample_capacity) uniformly spaced keys. Every row uses that
    same count and rank; a short group's keys are enumerated exactly. Page
    coordinates are consumed only with group_rows == 0 by whole-page SAMPLE;
    grouped sparse SAMPLE computes individual key positions directly.

    The fixed grid has ceil(max(B*P,R)/256) CTAs. Call after capacity refresh
    on the same stream; no raw caller prefix or map is read here. Coordinates
    index columns of the original page table, not physical KV page IDs.
    """

    def __init__(self, page_size, target_candidates, group_rows=0):
        self.page_size = page_size
        self.target_candidates = target_candidates
        self.group_rows = group_rows

    @cute.jit
    def __call__(
        self,
        physical_kv: cute.Tensor,
        qo_indptr: cute.Tensor,
        row_ends: cute.Tensor,
        row_requests: cute.Tensor,
        page_indices: cute.Tensor,
        sample_ends: cute.Tensor,
        sample_ranks: cute.Tensor,
        stream: cuda.CUstream,
    ):
        items = max(page_indices.shape[0] * page_indices.shape[1], row_ends.shape[0])
        self.kernel(
            physical_kv,
            qo_indptr,
            row_ends,
            row_requests,
            page_indices,
            sample_ends,
            sample_ranks,
        ).launch(
            grid=(cute.ceil_div(items, 256), 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        physical_kv: cute.Tensor,
        qo_indptr: cute.Tensor,
        row_ends: cute.Tensor,
        row_requests: cute.Tensor,
        page_indices: cute.Tensor,
        sample_ends: cute.Tensor,
        sample_ranks: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        item = block * 256 + tidx
        if item < page_indices.shape[0] * page_indices.shape[1]:
            request = item // page_indices.shape[1]
            slot = item % page_indices.shape[1]
            length = physical_kv[request + 1] - physical_kv[request]
            pages = cutlass.Int32(cute.ceil_div(cutlass.Int64(length), self.page_size))
            selected = min(pages, page_indices.shape[1])
            index = cutlass.Int64(0)
            if slot < selected:
                index = cutlass.Int64(slot) * (pages - 1) // max(selected - 1, 1)
            page_indices[request, slot] = index

        if item < row_ends.shape[0]:
            count = cutlass.Int32(0)
            rank = cutlass.Int32(0)
            owner = row_requests[item]
            if owner >= 0:
                end = cutlass.Int32(0)
                if cutlass.const_expr(self.group_rows > 0):
                    first = (
                        qo_indptr[owner]
                        + ((item - qo_indptr[owner]) // self.group_rows)
                        * self.group_rows
                    )
                    # Shared positions cover the first row's causal range.
                    # Every query in the group must score the same positions,
                    # including exact enumeration when that range is short.
                    end = row_ends[first]
                    count = min(end, page_indices.shape[1] * self.page_size)
                else:
                    length = physical_kv[owner + 1] - physical_kv[owner]
                    pages = cutlass.Int32(
                        cute.ceil_div(cutlass.Int64(length), self.page_size)
                    )
                    selected = min(pages, page_indices.shape[1])
                    end = row_ends[item]
                    end_page = end // self.page_size
                    # Count sampled pages before this row's exclusive causal end.
                    low = cutlass.Int32(0)
                    high = selected
                    while low < high:
                        middle = (low + high) // 2
                        page = (
                            cutlass.Int64(middle) * (pages - 1) // max(selected - 1, 1)
                        )
                        if page < end_page:
                            low = middle + 1
                        else:
                            high = middle
                    count = low * self.page_size
                    if low < selected:
                        page = cutlass.Int64(low) * (pages - 1) // max(selected - 1, 1)
                        if page == end_page:
                            count = count + end % self.page_size
                if end > 0:
                    rank = max(
                        1,
                        min(count, cute.ceil_div(self.target_candidates * count, end)),
                    )
            sample_ends[item] = count
            sample_ranks[item] = rank


class SelectiveRepairThresholdKernel:
    """Prepare bounded first-repair ranges and conservative thresholds."""

    def __init__(self, top_k: int, underflow_bin_step: int = 8):
        self.top_k = top_k
        self.underflow_bin_step = underflow_bin_step

    @cute.jit
    def __call__(
        self,
        candidate_counts: cute.Tensor,
        repair_flags: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        row_ends: cute.Tensor,
        repair_thresholds: cute.Tensor,
        repair_ends: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            candidate_counts,
            repair_flags,
            thresholds,
            selected_bins,
            row_ends,
            repair_thresholds,
            repair_ends,
            rows,
        ).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        candidate_counts: cute.Tensor,
        repair_flags: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        row_ends: cute.Tensor,
        repair_thresholds: cute.Tensor,
        repair_ends: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * 256 + tidx
        self.prepare_row(
            candidate_counts,
            repair_flags,
            thresholds,
            selected_bins,
            row_ends,
            repair_thresholds,
            repair_ends,
            row,
            rows,
        )

    @cute.jit
    def prepare_row(
        self,
        candidate_counts: cute.Tensor,
        repair_flags: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        row_ends: cute.Tensor,
        repair_thresholds: cute.Tensor,
        repair_ends: cute.Tensor,
        row: cutlass.Int32,
        rows: cutlass.Int32,
    ):
        if row < rows:
            # Candidate selection returns all keys and pads short rows. The exact count
            # certifies completeness; a segment overflow uses capacity + 1.
            short_complete = (row_ends[row] <= self.top_k) & (
                candidate_counts[(row, 0)] == row_ends[row]
            )
            if short_complete:
                repair_flags[row] = cutlass.Int32(0)
            flagged = repair_flags[row] != 0
            threshold = thresholds[row]
            if flagged:
                if row_ends[row] <= self.top_k:
                    # An underfilled causal row needs every valid key. Make
                    # the bounded repair pass exact instead of refining a sample
                    # threshold that can legitimately omit low scores.
                    threshold = cutlass.Float32(float("-inf"))
                elif candidate_counts[(row, 0)] < self.top_k:
                    refined = min(
                        cutlass.Int32(_SAMPLE_HISTOGRAM_BINS - 1),
                        selected_bins[row] + self.underflow_bin_step,
                    )
                    # Refinement beyond the negative finite range means -inf,
                    # not one of the FP16 NaN encodings above it.
                    repair_key = cutlass.Uint16(min(0xFC00, (refined << 5) + 31))
                    repair_bits = repair_key
                    if not (repair_key & cutlass.Uint16(0x8000)):
                        repair_bits = (
                            repair_key ^ cutlass.Uint16(0xFFFF)
                        ) & cutlass.Uint16(0x7FFF)
                    repair_half = llvm.bitcast(
                        cutlass.Float16.mlir_type, repair_bits.ir_value()
                    )
                    threshold = cutlass.Float32(repair_half)
            repair_thresholds[row] = threshold
            repair_ends[row] = row_ends[row] if flagged else cutlass.Int32(0)


class SelectivePackedTopKKernel:
    """Select exact TopK indices directly from int64 packed candidate records.

    One 512-thread CTA per row writes int32 selected indices, with -1 suffixes
    on short rows. The ordinary entry reads a compacted prefix bounded by lengths.
    The segmented entry reads each bank's count, keeps records in their existing
    slices, and writes the total retained count to lengths; that count is not a
    compacted-prefix certificate. Overflow and underfill below
    ``min(top_k, row_end)`` flag repair instead of certifying a partial result.
    Both entries support graph capture.
    """

    _THREADS = 512
    _COARSE_BINS = 1024

    def __init__(self, top_k: int, capacity: int, num_segments: int = 0):
        self.top_k = top_k
        self.capacity = capacity
        self.num_segments = num_segments
        assert num_segments >= 0
        assert num_segments == 0 or capacity % num_segments == 0

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(packed_values, lengths, selected, rows).launch(
            grid=(rows, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def segmented(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        counts: cute.Tensor,
        flags: cute.Tensor,
        row_ends: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_values, lengths, selected, rows, counts, flags, row_ends
        ).launch(
            grid=(rows, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _ordered_f32(self, packed):
        bits = cutlass.Uint32(cutlass.Uint64(packed) >> 32)
        ordered = bits
        if not (bits & cutlass.Uint32(0x80000000)):
            ordered = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
        return ordered

    @cute.jit
    def _coarse_bin(self, packed):
        bits = cutlass.Int32(cutlass.Uint64(packed) >> 32)
        value = llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        key = half_bits
        if not (half_bits & cutlass.Uint16(0x8000)):
            key = (half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
        return cutlass.Int32(key >> 6)

    @cute.jit
    def _index(self, packed):
        return cutlass.Int32(cutlass.Uint64(packed) & cutlass.Uint64(0xFFFFFFFF))

    @cute.jit
    def _valid_item(self, item, segment_meta):
        valid = cutlass.Boolean(True)
        if cutlass.const_expr(self.num_segments > 0):
            segment_capacity = self.capacity // self.num_segments
            valid = item % segment_capacity < segment_meta[item // segment_capacity]
        return valid

    @cute.jit
    def _scan_span(self, tidx, group, length, segment_meta):
        start = tidx
        end = length
        stride = cutlass.Int32(self._THREADS)
        if cutlass.const_expr(self.num_segments > 0):
            warps_per_segment = cutlass.const_expr(
                max(1, (self._THREADS // 32) // self.num_segments)
            )
            segment = group * (self._THREADS // 32) + tidx // (32 * warps_per_segment)
            start = segment * (self.capacity // self.num_segments) + tidx % (
                32 * warps_per_segment
            )
            end = start
            if segment < self.num_segments:
                end = (
                    segment * (self.capacity // self.num_segments)
                    + segment_meta[segment]
                )
            stride = cutlass.Int32(32 * warps_per_segment)
        return start, end, stride

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        counts: cute.Tensor = None,
        flags: cute.Tensor = None,
        row_ends: cute.Tensor = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        cute.arch.griddepcontrol_wait()

        smem = SmemAllocator()
        histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self._COARSE_BINS,), order=(0,)),
            byte_alignment=128,
        )
        filtered_keys = smem.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_ordered_layout((self.capacity,), order=(0,)),
            byte_alignment=128,
        )
        filtered_indices = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.capacity,), order=(0,)),
            byte_alignment=128,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((16,), order=(0,)),
            byte_alignment=64,
        )
        warp_bases = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((16,), order=(0,)),
            byte_alignment=64,
        )
        scalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=32,
        )
        segment_meta = None
        if cutlass.const_expr(self.num_segments > 0):
            segment_meta = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_layout((2 * self.num_segments + 1,)),
                byte_alignment=16,
            )
            segment_count = cutlass.Int32(0)
            if tidx < self.num_segments:
                segment_count = counts[(row, tidx + 1)]
            keep = min(segment_count, self.capacity // self.num_segments)
            if tidx < self.num_segments:
                segment_meta[tidx] = keep
            warp_keep = cute.arch.warp_reduction_sum(keep)
            warp_count = cute.arch.warp_reduction_sum(segment_count)
            warp_overflow = cute.arch.warp_reduction_sum(
                cutlass.Int32(segment_count > self.capacity // self.num_segments)
            )
            if lane == 31:
                warp_totals[warp] = warp_keep
                warp_bases[warp] = warp_count
                histogram[warp] = warp_overflow
            cute.arch.sync_threads()
            if tidx == 0:
                total = cutlass.Int32(0)
                retained = cutlass.Int32(0)
                overflow = cutlass.Boolean(False)
                for group in cutlass.range_constexpr((self.num_segments + 31) // 32):
                    group_keep = warp_totals[group]
                    warp_totals[group] = retained
                    total = total + warp_bases[group]
                    retained = retained + group_keep
                    overflow = overflow | (histogram[group] != 0)
                segment_meta[2 * self.num_segments] = retained
                counts[(row, 0)] = self.capacity + 1 if overflow else total
                lengths[row] = retained
                # A row needs repair only below min(TopK, row_end): a causal
                # row shorter than TopK is complete once every key is retained.
                required = min(cutlass.Int32(self.top_k), row_ends[row])
                flags[row] = cutlass.Int32(
                    (total < required) | (total > self.capacity) | overflow
                )
            cute.arch.sync_threads()
            length = segment_meta[2 * self.num_segments]
            scan_length = cutlass.Int32(self.capacity)
        else:
            length = min(lengths[row], cutlass.Int32(self.capacity))
            scan_length = length
        if length <= self.top_k:
            rank = tidx
            while rank < self.top_k:
                # Short rows keep every key and pad the fixed-width suffix with -1.
                index = cutlass.Int32(-1)
                if cutlass.const_expr(self.num_segments == 0):
                    if rank < length:
                        index = self._index(packed_values[(row, rank)])
                selected[(row, rank)] = index
                rank = rank + self._THREADS
            if cutlass.const_expr(self.num_segments > 0):
                segment_inclusive = keep
                for segment_step in cutlass.range_constexpr(5):
                    segment_offset = 1 << segment_step
                    segment_prior = cute.arch.shuffle_sync_up(
                        segment_inclusive,
                        segment_offset,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=0,
                    )
                    if lane >= segment_offset:
                        segment_inclusive = segment_inclusive + segment_prior
                if tidx < self.num_segments:
                    segment_meta[self.num_segments + tidx] = (
                        warp_totals[warp] + segment_inclusive - keep
                    )
                cute.arch.sync_threads()
                item = tidx
                while item < scan_length:
                    if self._valid_item(item, segment_meta):
                        segment_capacity = self.capacity // self.num_segments
                        rank = (
                            segment_meta[self.num_segments + item // segment_capacity]
                            + item % segment_capacity
                        )
                        selected[(row, rank)] = self._index(packed_values[(row, item)])
                    item = item + self._THREADS
        else:
            for item in cutlass.range_constexpr(self._COARSE_BINS // self._THREADS):
                histogram[tidx + item * self._THREADS] = cutlass.Int32(0)
            cute.arch.sync_threads()

            for scan_group in cutlass.range_constexpr(
                max(1, (self.num_segments + 15) // 16)
            ):
                item, end, stride = self._scan_span(
                    tidx, scan_group, length, segment_meta
                )
                while item < end:
                    packed = packed_values[(row, item)]
                    coarse_bin = self._coarse_bin(packed)
                    cute.arch.atomic_add(
                        histogram.iterator + coarse_bin,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    item = item + stride
            cute.arch.sync_threads()

            interval_sum = cutlass.Int32(0)
            first_bin = tidx * (self._COARSE_BINS // self._THREADS)
            for offset in cutlass.range_constexpr(self._COARSE_BINS // self._THREADS):
                interval_sum = interval_sum + histogram[first_bin + offset]
            inclusive = interval_sum
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                prior = cute.arch.shuffle_sync_up(
                    inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane >= offset:
                    inclusive = inclusive + prior
            if lane == 31:
                warp_totals[warp] = inclusive
            cute.arch.sync_threads()
            if warp == 0:
                own = warp_totals[lane] if lane < 16 else cutlass.Int32(0)
                prefix_sum = own
                for step in cutlass.range_constexpr(5):
                    offset = 1 << step
                    prior = cute.arch.shuffle_sync_up(
                        prefix_sum, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= offset:
                        prefix_sum = prefix_sum + prior
                if lane < 16:
                    warp_bases[lane] = prefix_sum - own
            cute.arch.sync_threads()

            before = warp_bases[warp] + inclusive - interval_sum
            if (before < self.top_k) & (before + interval_sum >= self.top_k):
                accumulated = before
                selected_bin = first_bin
                selected_before = before
                found = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(
                    self._COARSE_BINS // self._THREADS
                ):
                    count = histogram[first_bin + offset]
                    if (not found) & (accumulated + count >= self.top_k):
                        selected_bin = first_bin + offset
                        selected_before = accumulated
                        found = cutlass.Boolean(True)
                    accumulated = accumulated + count
                scalars[0] = selected_bin
                scalars[1] = selected_before
            if tidx < 3:
                scalars[tidx + 2] = cutlass.Int32(0)
            cute.arch.sync_threads()

            selected_coarse = scalars[0]
            for scan_group in cutlass.range_constexpr(
                max(1, (self.num_segments + 15) // 16)
            ):
                item, end, stride = self._scan_span(
                    tidx, scan_group, length, segment_meta
                )
                while item < end:
                    packed = packed_values[(row, item)]
                    coarse_bin = self._coarse_bin(packed)
                    if coarse_bin < selected_coarse:
                        slot = cute.arch.atomic_add(
                            scalars.iterator + 4,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        if slot < self.top_k:
                            selected[(row, slot)] = self._index(packed)
                    elif coarse_bin == selected_coarse:
                        slot = cute.arch.atomic_add(
                            scalars.iterator + 2,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        if slot < self.capacity:
                            filtered_keys[slot] = self._ordered_f32(packed)
                            filtered_indices[slot] = self._index(packed)
                    item = item + stride
            cute.arch.sync_threads()

            filtered_count = scalars[2]
            remaining = cutlass.Int32(self.top_k) - scalars[4]
            prefix = cutlass.Uint32(0)
            for radix_round in cutlass.range_constexpr(4):
                shift = 24 - radix_round * 8
                if tidx < 256:
                    histogram[tidx] = cutlass.Int32(0)
                cute.arch.sync_threads()
                item = tidx
                while item < filtered_count:
                    key = filtered_keys[item]
                    matches = cutlass.Boolean(True)
                    if cutlass.const_expr(radix_round > 0):
                        matches = (key >> (shift + 8)) == (prefix >> (shift + 8))
                    if matches:
                        radix_bin = cutlass.Int32((key >> shift) & 0xFF)
                        cute.arch.atomic_add(
                            histogram.iterator + radix_bin,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    item = item + self._THREADS
                cute.arch.sync_threads()

                own = histogram[tidx] if tidx < 256 else cutlass.Int32(0)
                inclusive = own
                for step in cutlass.range_constexpr(5):
                    offset = 1 << step
                    prior = cute.arch.shuffle_sync_up(
                        inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= offset:
                        inclusive = inclusive + prior
                if lane == 31:
                    warp_totals[warp] = inclusive
                cute.arch.sync_threads()
                if warp == 0:
                    warp_own = warp_totals[lane] if lane < 16 else cutlass.Int32(0)
                    warp_prefix = warp_own
                    for step in cutlass.range_constexpr(5):
                        offset = 1 << step
                        prior = cute.arch.shuffle_sync_up(
                            warp_prefix, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                        )
                        if lane >= offset:
                            warp_prefix = warp_prefix + prior
                    if lane < 16:
                        warp_bases[lane] = warp_prefix - warp_own
                cute.arch.sync_threads()
                bin_before = warp_bases[warp] + inclusive - own
                if (bin_before < remaining) & (bin_before + own >= remaining):
                    scalars[5] = tidx
                    scalars[6] = remaining - bin_before
                cute.arch.sync_threads()
                selected_bin = scalars[5]
                remaining = scalars[6]
                prefix = prefix | (cutlass.Uint32(selected_bin) << shift)

            if tidx == 0:
                scalars[3] = scalars[4]
            cute.arch.sync_threads()
            item = tidx
            while item < filtered_count:
                if filtered_keys[item] < prefix:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 3,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.top_k:
                        selected[(row, slot)] = filtered_indices[item]
                item = item + self._THREADS
            cute.arch.sync_threads()
            item = tidx
            while item < filtered_count:
                if filtered_keys[item] == prefix:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 3,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.top_k:
                        selected[(row, slot)] = filtered_indices[item]
                item = item + self._THREADS


class SelectiveRepairResolveKernel:
    """Resolve first repair and prepare the remaining exact-repair rows."""

    def __init__(self, top_k: int, capacity: int, threads: int = 256):
        self.top_k = top_k
        self.capacity = capacity
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        repair_flags: cute.Tensor,
        second_repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        selected: cute.Tensor,
        repair_ends: cute.Tensor,
        ranks: cute.Tensor,
        prefixes: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_values,
            repair_lengths,
            positions,
            repair_flags,
            second_repair_flags,
            row_ends,
            selected,
            repair_ends,
            ranks,
            prefixes,
            exact_error,
            rows,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        repair_flags: cute.Tensor,
        second_repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        selected: cute.Tensor,
        repair_ends: cute.Tensor,
        ranks: cute.Tensor,
        prefixes: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        first_repair = repair_flags[row] != 0
        exact_repair = first_repair & (second_repair_flags[row] != 0)
        successful_repair = first_repair & (not exact_repair)

        for item in cutlass.range_constexpr(cute.ceil_div(self.top_k, self.threads)):
            rank = item * self.threads + tidx
            if successful_repair & (rank < self.top_k):
                index = cutlass.Int32(-1)
                if rank < repair_lengths[row]:
                    position = positions[(row, rank)]
                    if (position >= 0) & (position < self.capacity):
                        packed = cutlass.Uint64(packed_values[(row, position)])
                        index = cutlass.Int32(packed & cutlass.Uint64(0xFFFFFFFF))
                selected[(row, rank)] = index

        if tidx == 0:
            second_repair_flags[row] = cutlass.Int32(exact_repair)
            repair_ends[row] = row_ends[row] if exact_repair else cutlass.Int32(0)
            ranks[row] = (
                min(row_ends[row], cutlass.Int32(self.top_k))
                if exact_repair
                else cutlass.Int32(0)
            )
            prefixes[row] = cutlass.Int32(0)
            if row == 0:
                exact_error[0] = cutlass.Int32(0)


class SelectiveTopKRadixSelectKernel:
    """Advance one exact FP32 radix-selection byte for every query row."""

    def __init__(self, threads: int = 256, clear_histogram: bool = False):
        self.threads = threads
        self.clear_histogram = clear_histogram

    @cute.jit
    def __call__(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.clear_histogram):
            self.warp_kernel(
                histograms, prefixes, ranks, threshold_bits, rows, shift, None
            ).launch(
                grid=(cute.ceil_div(rows, self.threads // 32), 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )
        else:
            self.kernel(
                histograms, prefixes, ranks, threshold_bits, rows, shift
            ).launch(
                grid=(cute.ceil_div(rows, self.threads), 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )

    @cute.jit
    def compact(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.warp_kernel(
            histograms, prefixes, ranks, threshold_bits, rows, shift, active_count
        ).launch(
            grid=(cute.ceil_div(rows, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def warp_kernel(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
        active_count: cute.Tensor | None,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        blocks, _, _ = cute.arch.grid_dim()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(active_count is not None):
            rows = min(rows, active_count[0])
        for row in cutlass.range(
            block_idx * (self.threads // 32) + warp,
            rows,
            blocks * (self.threads // 32),
        ):
            rank = ranks[row]
            if rank > 0:
                total = cutlass.Int32(0)
                selected_bin = cutlass.Int32(0)
                greater = cutlass.Int32(0)
                for chunk in cutlass.range_constexpr(8):
                    radix_bin = 255 - chunk * 32 - lane
                    count = histograms[(row, radix_bin)]
                    histograms[(row, radix_bin)] = cutlass.Int32(0)
                    inclusive = count
                    for step in cutlass.range_constexpr(5):
                        delta = 1 << step
                        previous = cute.arch.shuffle_sync_up(
                            inclusive, delta, mask_and_clamp=0
                        )
                        if lane >= delta:
                            inclusive = inclusive + previous
                    before = total + inclusive - count
                    if (before < rank) & (rank <= before + count):
                        selected_bin = radix_bin
                        greater = before
                    total = total + cute.arch.shuffle_sync(inclusive, 31)
                selected_bin = cute.arch.warp_redux_sync(selected_bin, "max")
                greater = cute.arch.warp_redux_sync(greater, "max")
                if lane == 0:
                    prefix = cutlass.Uint32(selected_bin)
                    if shift != 24:
                        prefix = (cutlass.Uint32(prefixes[row]) << 8) | prefix
                    prefixes[row] = cutlass.Int32(prefix)
                    ranks[row] = rank - greater
                    if shift == 0:
                        mask = cutlass.Uint32(
                            ~(cutlass.Int32(prefix) >> 31)
                        ) | cutlass.Uint32(0x80000000)
                        threshold_bits[row] = cutlass.Int32(prefix ^ mask)

    @cute.kernel
    def kernel(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * self.threads + tidx
        if row < rows:
            rank = ranks[row]
            if rank > 0:
                greater = cutlass.Int32(0)
                selected_bin = cutlass.Int32(0)
                found = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(256):
                    radix_bin = cutlass.Int32(255 - offset)
                    count = histograms[(row, radix_bin)]
                    if (not found) & (greater + count >= rank):
                        selected_bin = radix_bin
                        found = cutlass.Boolean(True)
                    if not found:
                        greater = greater + count

                # Compact repair reuses row slots across graph replays. The
                # first byte starts a new key, independent of the prior row.
                prefix = cutlass.Uint32(selected_bin)
                if shift != 24:
                    prefix = (cutlass.Uint32(prefixes[row]) << 8) | prefix
                prefixes[row] = cutlass.Int32(prefix)
                ranks[row] = rank - greater
                if shift == 0:
                    # Invert the monotonic key used by the score kernel without
                    # changing the selected FP32 payload.
                    mask = cutlass.Uint32(
                        ~(cutlass.Int32(prefix) >> 31)
                    ) | cutlass.Uint32(0x80000000)
                    threshold_bits[row] = cutlass.Int32(prefix ^ mask)


class SelectiveTopKCombineExactKernel:
    """Replace flagged candidate rows with exact greater/equal TopK."""

    def __init__(self, top_k: int):
        assert 0 < top_k <= 2048
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            greater_indices,
            greater_counts,
            equal_indices,
            equal_counts,
            ranks,
            repair_flags,
            row_ends,
            output,
            error_count,
            rows,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if repair_flags[row] != 0:
            greater = greater_counts[(row, 0)]
            needed_equal = ranks[row]
            required = min(row_ends[row], cutlass.Int32(self.top_k))
            valid = (greater == required - needed_equal) & (
                equal_counts[(row, 0)] >= needed_equal
            )
            for item_group in cutlass.range_constexpr((self.top_k + 255) // 256):
                column = tidx + item_group * 256
                if column < self.top_k:
                    if valid:
                        if column < required:
                            if column < greater:
                                output[(row, column)] = greater_indices[(row, column)]
                            else:
                                output[(row, column)] = equal_indices[
                                    (row, column - greater)
                                ]
                        else:
                            # Fixed-width suffix for rows shorter than TopK.
                            output[(row, column)] = cutlass.Int32(-1)
                    else:
                        output[(row, column)] = cutlass.Int32(-1)
            if (tidx == 0) & (not valid):
                cute.arch.atomic_add(
                    error_count.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )


class SelectiveLogitsFinalizeKernel:
    """Convert logical candidate counts into packed length/repair flags.

    With ``row_ends``, a row is underfilled only below ``min(top_k, row_end)``:
    a complete causal row shorter than TopK needs no repair. Without it
    (repair-stage counts), every total below TopK is flagged.
    """

    def __init__(
        self,
        capacity: int,
        num_segments: int,
        top_k: int = 512,
        threads: int = 256,
    ):
        assert capacity > 0 and num_segments >= 0
        assert 0 < top_k <= 2048
        self.capacity = capacity
        self.num_segments = num_segments
        self.top_k = top_k
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
        row_ends: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.num_segments == 0):
            self.kernel(
                counts, values, lengths, flags, rows, capacity, row_ends
            ).launch(
                grid=(cute.ceil_div(rows, self.threads), 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        elif cutlass.const_expr(self.num_segments == 64):
            self.lane_segmented_kernel(
                counts, values, lengths, flags, rows, capacity, row_ends
            ).launch(
                grid=(rows, 1, 1),
                block=(512, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            self.segmented_kernel(
                counts, values, lengths, flags, rows, capacity, row_ends
            ).launch(
                grid=(rows, 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def _required(self, row, row_ends):
        required = cutlass.Int32(self.top_k)
        if cutlass.const_expr(row_ends is not None):
            required = min(required, row_ends[row])
        return required

    @cute.kernel
    def kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
        row_ends: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * self.threads + tidx
        # PDL secondary: the scorer's counts/values are visible only after wait.
        cute.arch.griddepcontrol_wait()
        if row < rows:
            total = counts[(row, 0)]
            lengths[row] = min(total, capacity)
            flags[row] = cutlass.Int32(
                (total < self._required(row, row_ends)) | (total > capacity)
            )
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def segmented_kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
        row_ends: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        segment_capacity = cutlass.const_expr(self.capacity // self.num_segments)

        smem = SmemAllocator()
        staged_values = smem.allocate_tensor(
            element_type=cutlass.Int64,
            layout=cute.make_ordered_layout((segment_capacity,), order=(0,)),
            byte_alignment=16,
        )
        segment_meta = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2 * self.num_segments + 3,), order=(0,)),
            byte_alignment=16,
        )

        if tidx == 0:
            total = cutlass.Int32(0)
            retained = cutlass.Int32(0)
            overflow = cutlass.Boolean(False)
            for segment in cutlass.range_constexpr(self.num_segments):
                count = counts[(row, segment + 1)]
                keep = min(count, segment_capacity)
                segment_meta[2 * segment] = retained
                segment_meta[2 * segment + 1] = keep
                retained = retained + keep
                total = total + count
                overflow = overflow | (count > segment_capacity)
            segment_meta[2 * self.num_segments] = total
            segment_meta[2 * self.num_segments + 1] = retained
            segment_meta[2 * self.num_segments + 2] = cutlass.Int32(overflow)
        cute.arch.sync_threads()

        # Stage one segment at a time so an earlier destination range cannot
        # overwrite a later source element in the same in-place move.
        for segment in cutlass.range_constexpr(self.num_segments):
            retained = segment_meta[2 * segment]
            keep = segment_meta[2 * segment + 1]
            begin = cutlass.Int32(segment * segment_capacity)
            item = tidx
            while item < keep:
                staged_values[item] = values[(row, begin + item)]
                item = item + self.threads
            cute.arch.sync_threads()
            item = tidx
            while item < keep:
                values[(row, retained + item)] = staged_values[item]
                item = item + self.threads
            cute.arch.sync_threads()

        if tidx == 0:
            total = segment_meta[2 * self.num_segments]
            retained = segment_meta[2 * self.num_segments + 1]
            overflow = cutlass.Boolean(segment_meta[2 * self.num_segments + 2])
            # Preserve the packed consumer ABI: segment overflow is
            # represented by capacity + 1, while an in-capacity row keeps its
            # exact logical count.
            published = total
            if overflow:
                published = cutlass.Int32(self.capacity + 1)
            counts[(row, 0)] = published
            lengths[row] = retained
            flags[row] = cutlass.Int32(
                (total < self._required(row, row_ends)) | (total > capacity) | overflow
            )
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def lane_segmented_kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
        row_ends: cute.Tensor,
    ):
        """Compact fixed store-lane vector slices through shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        segment_capacity = cutlass.const_expr(self.capacity // self.num_segments)
        finalize_threads = cutlass.const_expr(512)
        num_warps = cutlass.const_expr(finalize_threads // 32)
        threads_per_segment = cutlass.const_expr(finalize_threads // self.num_segments)

        smem = SmemAllocator()
        warp_meta = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4 * num_warps + 4,), order=(0,)),
            byte_alignment=16,
        )

        segment = tidx // threads_per_segment
        segment_thread = tidx % threads_per_segment
        count = counts[(row, segment + 1)]
        keep = min(count, segment_capacity)
        begin = segment * segment_capacity
        items_per_thread = cutlass.const_expr(
            (segment_capacity + threads_per_segment - 1) // threads_per_segment
        )
        staged_values = cute.make_rmem_tensor((items_per_thread,), cutlass.Int64)
        for item_group in cutlass.range_constexpr(items_per_thread):
            item = threads_per_segment * item_group + segment_thread
            staged_values[item_group] = cutlass.Int64(0)
            if item < keep:
                staged_values[item_group] = values[(row, begin + item)]

        lane = tidx % 32
        warp = tidx // 32
        inclusive_keep = cutlass.Int32(0)
        inclusive_count = cutlass.Int32(0)
        if segment_thread == 0:
            inclusive_keep = keep
            inclusive_count = count
        for step in cutlass.range_constexpr(5):
            offset = 1 << step
            shuffled_keep = cute.arch.shuffle_sync_up(
                inclusive_keep, offset, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            shuffled_count = cute.arch.shuffle_sync_up(
                inclusive_count, offset, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= offset:
                inclusive_keep = inclusive_keep + shuffled_keep
                inclusive_count = inclusive_count + shuffled_count
        overflow_mask = cute.arch.vote_ballot_sync(
            (segment_thread == 0) & (count > segment_capacity)
        )
        if lane == 31:
            warp_meta[warp] = inclusive_keep
            warp_meta[num_warps + warp] = inclusive_count
            warp_meta[2 * num_warps + warp] = cutlass.Int32(overflow_mask != 0)
        cute.arch.sync_threads()

        if tidx == 0:
            total = cutlass.Int32(0)
            retained_total = cutlass.Int32(0)
            overflow = cutlass.Boolean(False)
            for warp_idx in cutlass.range_constexpr(num_warps):
                warp_meta[3 * num_warps + warp_idx] = retained_total
                retained_total = retained_total + warp_meta[warp_idx]
                total = total + warp_meta[num_warps + warp_idx]
                overflow = overflow | cutlass.Boolean(
                    warp_meta[2 * num_warps + warp_idx]
                )
            warp_meta[4 * num_warps] = total
            warp_meta[4 * num_warps + 1] = retained_total
            warp_meta[4 * num_warps + 2] = cutlass.Int32(overflow)
        cute.arch.sync_threads()

        retained = warp_meta[3 * num_warps + warp] + inclusive_keep - keep
        for item_group in cutlass.range_constexpr(items_per_thread):
            item = threads_per_segment * item_group + segment_thread
            if item < keep:
                values[(row, retained + item)] = staged_values[item_group]

        if tidx == 0:
            total = warp_meta[4 * num_warps]
            retained_total = warp_meta[4 * num_warps + 1]
            overflow = cutlass.Boolean(warp_meta[4 * num_warps + 2])
            published = total
            if overflow:
                published = cutlass.Int32(self.capacity + 1)
            counts[(row, 0)] = published
            lengths[row] = retained_total
            flags[row] = cutlass.Int32(
                (total < self._required(row, row_ends)) | (total > capacity) | overflow
            )
        cute.arch.griddepcontrol_launch_dependents()


class SelectiveLogitsMetadataScheduleKernel:
    """Build request-owned causal tiles and the split-balanced CTA schedule."""

    def __init__(
        self,
        num_sms: int,
        query_tile: int = 16,
        kv_chunk_size: int = 0,
        split_kv: int = 1,
        paged: bool = False,
    ):
        self.num_sms = num_sms
        self.query_tile = query_tile
        self.kv_chunk_size = kv_chunk_size
        self.split_kv = split_kv
        self.paged = paged

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cu_q,
            cu_kv,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_tiles = context_lens.shape[0]

        # All CTA threads build causal tile records in parallel.
        tile_idx = tidx
        while tile_idx < num_tiles:
            context_len = cutlass.Int32(0)
            if tile_idx < num_tiles:
                owner = cutlass.Int32(0)
                owner_tile_begin = cutlass.Int32(0)
                q_begin = cu_q[0]
                q_end = cu_q[1]
                q_len = q_end - q_begin
                owner_tile_end = cute.ceil_div(q_len, query_tile)

                # Walk monotonically to the request owning this fixed tile slot.
                while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                    owner_tile_begin = owner_tile_end
                    owner = owner + 1
                    q_begin = cu_q[owner]
                    q_end = cu_q[owner + 1]
                    q_len = q_end - q_begin
                    owner_tile_end = owner_tile_begin + cute.ceil_div(q_len, query_tile)

                active = tile_idx < owner_tile_end
                q_start = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                q_valid = cutlass.Int32(0)
                causal_base = cutlass.Int32(0)
                if active:
                    local_tile = tile_idx - owner_tile_begin
                    q_start = q_begin + local_tile * query_tile
                    q_valid = min(q_end - q_start, query_tile)
                    kv_start = cu_kv[owner]
                    kv_len = cu_kv[owner + 1] - kv_start
                    causal_base = kv_len - q_len + local_tile * query_tile
                    # Dense scale TMA needs the aligned prefix; native paged
                    # coordinates begin at logical request column zero.
                    context_len = causal_base + q_valid
                    if cutlass.const_expr(not self.paged):
                        context_len = context_len + kv_start % 4

                # Materialize exact per-row bounds in the same launch.  The
                # prepared route can therefore refresh graph-stable metadata
                # without allocating an offsets tensor or launching PyTorch
                # pointwise kernels between the metadata and score kernels.
                for row in cutlass.range_constexpr(query_tile):
                    row_start = cutlass.Int32(0)
                    row_end = cutlass.Int32(0)
                    if row < q_valid:
                        row_end = causal_base + row + 1
                    tile_meta[tile_idx, 4 + row] = row_start
                    tile_meta[tile_idx, 4 + query_tile + row] = row_end

                context_lens[tile_idx] = context_len
                tile_meta[tile_idx, 0] = q_start
                # Dense uses the packed KV origin; paged ingress needs the
                # owning page-table row for this request-owned query tile.
                tile_meta[tile_idx, 1] = kv_start
                if cutlass.const_expr(self.paged):
                    tile_meta[tile_idx, 1] = owner
                tile_meta[tile_idx, 2] = q_valid
                tile_meta[tile_idx, 3] = causal_base
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()

        # Warp 0 consumes the completed metadata and emits CTA boundaries.
        if warp_idx == 0:
            self._build_schedule(
                context_lens, schedule_meta, schedule_prefix, num_tiles
            )

    @cute.jit
    def _build_schedule(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        prefix_sum: cute.Tensor,
        num_tiles: cutlass.Int32,
    ):
        lane_idx = cute.arch.lane_idx()

        # Inclusive scan one work item per exclusive query tile. Atomic modes may
        # divide it into fixed-size kv_chunk_size work items; candidate mode may
        # instead divide it among split_kv pair-aligned owners with disjoint
        # publisher-segment banks.
        carry = cutlass.Int32(0)
        chunk = cutlass.Int32(0)
        while chunk * 32 < num_tiles:
            tile_idx = cutlass.Int32(chunk * 32) + lane_idx
            context_len = cutlass.Int32(0)
            value = cutlass.Int32(0)
            if tile_idx < num_tiles:
                context_len = context_lens[tile_idx]
            if context_len > 0:
                if cutlass.const_expr(self.kv_chunk_size > 0):
                    value = cute.ceil_div(context_len, self.kv_chunk_size)
                elif cutlass.const_expr(self.split_kv > 1):
                    value = min(
                        cute.ceil_div(context_len, _KV_PAIR_TOKENS),
                        cutlass.Int32(self.split_kv),
                    )
                else:
                    value = cutlass.Int32(1)
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                shuffled = cute.arch.shuffle_sync_up(
                    value, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane_idx >= offset:
                    value = value + shuffled
            value = value + carry
            if tile_idx < num_tiles:
                prefix_sum[tile_idx] = value
            carry = cute.arch.shuffle_sync(value, 31)
            chunk = chunk + 1

        self._partition_schedule(
            context_lens, schedule_meta, prefix_sum, num_tiles, carry
        )

    @cute.jit
    def _work_items(self, context_len):
        """Scheduled work items for one query tile of ``context_len`` keys."""
        value = cutlass.Int32(0)
        if context_len > 0:
            if cutlass.const_expr(self.kv_chunk_size > 0):
                value = cute.ceil_div(context_len, self.kv_chunk_size)
            elif cutlass.const_expr(self.split_kv > 1):
                value = min(
                    cute.ceil_div(context_len, _KV_PAIR_TOKENS),
                    cutlass.Int32(self.split_kv),
                )
            else:
                value = cutlass.Int32(1)
        return value

    @cute.jit
    def _build_schedule_block(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        prefix_sum: cute.Tensor,
        num_tiles: cutlass.Int32,
        tidx,
        warp_totals: cute.Tensor,
    ):
        """Whole-CTA (256-thread) equivalent of ``_build_schedule``.

        Each thread scans a contiguous tile segment, one block scan offsets the
        segments, then warp 0 partitions the identical inclusive prefix.
        """
        lane = tidx % 32
        warp = tidx // 32
        per_thread = cute.ceil_div(num_tiles, 256)
        begin = tidx * per_thread
        end = min(begin + per_thread, num_tiles)
        local = cutlass.Int32(0)
        tile = begin
        while tile < end:
            local = local + self._work_items(context_lens[tile])
            tile = tile + 1
        inclusive = local
        for step in cutlass.range_constexpr(5):
            prior = cute.arch.shuffle_sync_up(
                inclusive, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= 1 << step:
                inclusive = inclusive + prior
        cute.arch.sync_threads()
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()
        running = inclusive - local
        total = cutlass.Int32(0)
        for other in cutlass.range_constexpr(256 // 32):
            if other < warp:
                running = running + warp_totals[other]
            total = total + warp_totals[other]
        tile = begin
        while tile < end:
            running = running + self._work_items(context_lens[tile])
            prefix_sum[tile] = running
            tile = tile + 1
        cute.arch.sync_threads()
        if tidx < 32:
            self._partition_schedule(
                context_lens, schedule_meta, prefix_sum, num_tiles, total
            )

    @cute.jit
    def _partition_schedule(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        prefix_sum: cute.Tensor,
        num_tiles: cutlass.Int32,
        total,
    ):
        """Warp 0: split the inclusive work prefix across persistent CTAs."""
        num_sms = cutlass.const_expr(self.num_sms)
        sm_chunks = cutlass.const_expr((num_sms + 32) // 32)
        lane_idx = cute.arch.lane_idx()
        # Boundaries at or past the total work end just after the last tile
        # with work, not at num_tiles: otherwise the CTA owning the final work
        # item walks every trailing empty (capacity-padding) tile serially.
        # The first tile whose inclusive prefix reaches the total is the last
        # tile with work.
        tail_end = cutlass.Int32(0)
        if total > 0:
            low = cutlass.Int32(0)
            high = cutlass.Int32(num_tiles)
            while low < high:
                middle = (low + high) // 2
                if prefix_sum[middle] < total:
                    low = middle + 1
                else:
                    high = middle
            tail_end = low + 1
        # Partition the tile prefix by quotient/remainder across persistent CTAs.
        quotient = total // num_sms
        remainder = total % num_sms
        for chunk in cutlass.range_constexpr(sm_chunks):
            sm_idx = cutlass.Int32(chunk * 32) + lane_idx
            if sm_idx <= num_sms:
                segment_start = sm_idx * quotient
                if sm_idx <= remainder:
                    segment_start = segment_start + sm_idx
                else:
                    segment_start = segment_start + remainder

                lo = cutlass.Int32(0)
                count = cutlass.Int32(num_tiles)
                while count > 0:
                    half = count // 2
                    mid = lo + half
                    take = cutlass.Boolean(False)
                    if count > 0:
                        if prefix_sum[mid] <= segment_start:
                            take = cutlass.Boolean(True)
                    if take:
                        lo = mid + 1
                        count = count - half - 1
                    else:
                        count = half
                q_idx = min(lo, tail_end)
                kv_split = segment_start
                if q_idx > 0:
                    kv_split = segment_start - prefix_sum[q_idx - 1]
                if cutlass.const_expr(self.split_kv > 1):
                    # Convert the split ordinal into its pair-aligned KV
                    # boundary. The scorer already consumes this coordinate
                    # as a KV-pair boundary, so its pipeline traversal and
                    # drain lifecycle remain unchanged.
                    if kv_split > 0:
                        num_pairs = cute.ceil_div(context_lens[q_idx], _KV_PAIR_TOKENS)
                        active_splits = min(num_pairs, cutlass.Int32(self.split_kv))
                        kv_split = cute.ceil_div(kv_split * num_pairs, active_splits)
                schedule_meta[sm_idx, 0] = q_idx
                schedule_meta[sm_idx, 1] = kv_split


class SelectiveLogitsExplicitMetadataScheduleKernel(
    SelectiveLogitsMetadataScheduleKernel
):
    """Build request-tile metadata from caller-owned exact local row bounds."""

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_end_base: cutlass.Int32,
        active_rows: cute.Tensor | None,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel_explicit(
            cu_q,
            cu_kv,
            row_starts,
            row_ends,
            row_end_base,
            active_rows,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def _tile_prefix(self, cu_q, num_batches, prefix, warp_totals, tidx):
        """Exclusive per-request tile prefix in shared memory: prefix[r] tiles
        precede request r and prefix[num_batches] is the total tile count."""
        query_tile = cutlass.const_expr(self.query_tile)
        lane = tidx % 32
        warp = tidx // 32
        if tidx == 0:
            prefix[0] = cutlass.Int32(0)
        carry = cutlass.Int32(0)
        base = cutlass.Int32(0)
        while base < num_batches:
            request = base + tidx
            count = cutlass.Int32(0)
            if request < num_batches:
                count = cute.ceil_div(cu_q[request + 1] - cu_q[request], query_tile)
            inclusive = count
            for step in cutlass.range_constexpr(5):
                prior = cute.arch.shuffle_sync_up(
                    inclusive, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane >= 1 << step:
                    inclusive = inclusive + prior
            cute.arch.sync_threads()
            if lane == 31:
                warp_totals[warp] = inclusive
            cute.arch.sync_threads()
            before = cutlass.Int32(0)
            total = cutlass.Int32(0)
            for other in cutlass.range_constexpr(256 // 32):
                if other < warp:
                    before = before + warp_totals[other]
                total = total + warp_totals[other]
            if request < num_batches:
                prefix[request + 1] = carry + before + inclusive
            carry = carry + total
            base = base + 256
        cute.arch.sync_threads()

    @cute.kernel
    def kernel_explicit(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_end_base: cutlass.Int32,
        active_rows: cute.Tensor | None,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        num_tiles = context_lens.shape[0]
        active_count = cu_q[1]
        if cutlass.const_expr(active_rows is not None):
            active_count = min(max(active_rows[0], 0), cu_q[1])
            num_tiles = cute.ceil_div(active_count, query_tile)

        # Without an active-row clamp, find each tile's owner by binary search
        # over a shared-memory tile prefix instead of walking every preceding
        # request (O(batch) per tile). Larger batches keep the walk.
        smem = SmemAllocator()
        tile_prefix = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((_OWNER_PREFIX_CAP + 1,), order=(0,)),
            byte_alignment=16,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16,
        )
        search = cutlass.Boolean(False)
        if cutlass.const_expr(active_rows is None):
            search = num_batches <= _OWNER_PREFIX_CAP
            if search:
                self._tile_prefix(cu_q, num_batches, tile_prefix, warp_totals, tidx)

        tile_idx = tidx
        while tile_idx < num_tiles:
            context_len = cutlass.Int32(0)
            if tile_idx < num_tiles:
                owner = cutlass.Int32(0)
                owner_tile_begin = cutlass.Int32(0)
                q_begin = cu_q[0]
                q_end = cu_q[1]
                if cutlass.const_expr(active_rows is not None):
                    q_end = active_count
                q_len = q_end - q_begin
                owner_tile_end = cute.ceil_div(q_len, query_tile)
                if search:
                    # Smallest request whose tiles end after tile_idx; a tile
                    # past every request keeps the walk's last owner.
                    low = cutlass.Int32(0)
                    high = num_batches - 1
                    while low < high:
                        middle = (low + high) // 2
                        if tile_prefix[middle + 1] > tile_idx:
                            high = middle
                        else:
                            low = middle + 1
                    owner = low
                    owner_tile_begin = tile_prefix[owner]
                    owner_tile_end = tile_prefix[owner + 1]
                    q_begin = cu_q[owner]
                    q_end = cu_q[owner + 1]
                    q_len = q_end - q_begin
                else:
                    while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                        owner_tile_begin = owner_tile_end
                        owner = owner + 1
                        q_begin = cu_q[owner]
                        q_end = cu_q[owner + 1]
                        q_len = q_end - q_begin
                        owner_tile_end = owner_tile_begin + cute.ceil_div(
                            q_len, query_tile
                        )

                active = tile_idx < owner_tile_end
                q_start = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                q_valid = cutlass.Int32(0)
                if active:
                    local_tile = tile_idx - owner_tile_begin
                    q_start = q_begin + local_tile * query_tile
                    q_valid = min(q_end - q_start, query_tile)
                    kv_start = cu_kv[owner]
                    max_end = cutlass.Int32(0)
                    for row in cutlass.range_constexpr(query_tile):
                        start = cutlass.Int32(0)
                        end = cutlass.Int32(0)
                        if row < q_valid:
                            start = row_starts[q_start + row]
                            end = row_ends[q_start + row]
                            if end != 0:
                                end = end + row_end_base
                            max_end = max(max_end, end)
                        tile_meta[tile_idx, 4 + row] = start
                        tile_meta[tile_idx, 4 + query_tile + row] = end
                    context_len = max_end
                    if cutlass.const_expr(not self.paged):
                        # Contiguous scale TMA rounds the packed origin down to
                        # a four-float boundary; include that prefix in sizing.
                        context_len = context_len + kv_start % 4

                context_lens[tile_idx] = context_len
                tile_meta[tile_idx, 0] = q_start
                tile_meta[tile_idx, 1] = kv_start
                if cutlass.const_expr(self.paged):
                    tile_meta[tile_idx, 1] = owner
                tile_meta[tile_idx, 2] = q_valid
                tile_meta[tile_idx, 3] = cutlass.Int32(0)
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()
        self._build_schedule_block(
            context_lens, schedule_meta, schedule_prefix, num_tiles, tidx, warp_totals
        )


class SelectiveLogitsCompactMetadataScheduleKernel(
    SelectiveLogitsMetadataScheduleKernel
):
    """Compact query tiles containing flagged rows into a scorer schedule."""

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_flags: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel_compact(
            cu_q,
            row_starts,
            row_ends,
            row_flags,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
            rows,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel_compact(
        self,
        cu_q: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_flags: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        rows: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_tiles = context_lens.shape[0]

        # Clear the fixed-capacity schedule first. Thread zero then emits the
        # active rows in global-Q order, which also makes request ownership a
        # monotonic scan instead of one search per repair row.
        tile_idx = tidx
        while tile_idx < num_tiles:
            context_lens[tile_idx] = cutlass.Int32(0)
            tile_idx = tile_idx + 256
        if tidx == 0:
            schedule_prefix[0] = cutlass.Int32(0)
        cute.arch.sync_threads()

        tile_idx = tidx
        while tile_idx < num_tiles:
            owner = cutlass.Int32(0)
            owner_tile_begin = cutlass.Int32(0)
            q_begin = cu_q[0]
            q_end = cu_q[1]
            owner_tile_end = cute.ceil_div(q_end - q_begin, query_tile)
            while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                owner_tile_begin = owner_tile_end
                owner = owner + 1
                q_begin = cu_q[owner]
                q_end = cu_q[owner + 1]
                owner_tile_end = owner_tile_begin + cute.ceil_div(
                    q_end - q_begin, query_tile
                )

            if tile_idx < owner_tile_end:
                q_start = q_begin + (tile_idx - owner_tile_begin) * query_tile
                q_valid = min(q_end - q_start, query_tile)
                context_len = cutlass.Int32(0)
                for tile_row in cutlass.range_constexpr(query_tile):
                    # A partial request tile has no flag beyond q_valid. A
                    # bitwise conjunction would still evaluate that load.
                    if tile_row < q_valid:
                        if row_flags[q_start + tile_row] != 0:
                            context_len = max(context_len, row_ends[q_start + tile_row])
                if context_len != 0:
                    compact_idx = cute.arch.atomic_add(
                        schedule_prefix.iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    context_lens[compact_idx] = context_len
                    tile_meta[(compact_idx, 0)] = q_start
                    tile_meta[(compact_idx, 1)] = owner
                    tile_meta[(compact_idx, 2)] = q_valid
                    tile_meta[(compact_idx, 3)] = cutlass.Int32(0)
                    for tile_row in cutlass.range_constexpr(query_tile):
                        start = cutlass.Int32(0)
                        end = cutlass.Int32(0)
                        if tile_row < q_valid:
                            if row_flags[q_start + tile_row] != 0:
                                start = row_starts[q_start + tile_row]
                                end = row_ends[q_start + tile_row]
                        tile_meta[(compact_idx, 4 + tile_row)] = start
                        tile_meta[(compact_idx, 4 + query_tile + tile_row)] = end
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()

        if warp_idx == 0:
            self._build_schedule(
                context_lens, schedule_meta, schedule_prefix, num_tiles
            )


class SelectiveRepairRowCompactKernel:
    """Append flagged rows and their first-repair policy to compact storage."""

    def __init__(self, top_k: int = 512, num_heads: int = 0, head_dim: int = 0):
        self.repair_policy = SelectiveRepairThresholdKernel(top_k)
        self.num_heads = num_heads
        self.head_dim = head_dim

    @cute.jit
    def with_pack(
        self,
        inputs: tuple,
        pack_inputs: tuple,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.pack_kernel(*inputs, pack_inputs, rows).launch(
            grid=(cute.ceil_div(rows, 8), 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def with_policy_pack(
        self,
        inputs: tuple,
        pack_inputs: tuple,
        rows: cutlass.Int32,
        policy: tuple,
        stream: cuda.CUstream,
    ):
        self.pack_kernel(*inputs, pack_inputs, rows, policy).launch(
            grid=(cute.ceil_div(rows, 8), 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def pack_kernel(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        pack_inputs: tuple,
        rows: cutlass.Int32,
        policy: tuple = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 8 + tidx // 32
        lane = tidx % 32
        if row < rows:
            if cutlass.const_expr(policy is not None):
                if lane == 0:
                    self.repair_policy.prepare_row(
                        policy[0],
                        flags,
                        policy[1],
                        policy[2],
                        row_ends,
                        thresholds,
                        policy[3],
                        row,
                        rows,
                    )
                # All lanes own the same row and must observe its updated flag.
                cute.arch.sync_warp()
            if flags[row] != 0:
                slot = cutlass.Int32(-1)
                if lane == 0:
                    slot = cute.arch.atomic_add(
                        active_count.iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    row_ids[slot] = row
                    packed_thresholds[slot] = thresholds[row]
                    packed_ends[slot] = row_ends[row]
                slot = cute.arch.shuffle_sync(slot, 0)
                q_words = cutlass.const_expr(self.num_heads * self.head_dim // 4)
                for group in cutlass.range_constexpr(cute.ceil_div(q_words, 32)):
                    item = group * 32 + lane
                    if item < q_words:
                        pack_inputs[2][slot * q_words + item] = pack_inputs[0][
                            row * q_words + item
                        ]
                for group in cutlass.range_constexpr(cute.ceil_div(self.num_heads, 32)):
                    head = group * 32 + lane
                    if head < self.num_heads:
                        pack_inputs[3][slot * self.num_heads + head] = pack_inputs[1][
                            row * self.num_heads + head
                        ]

    @cute.jit
    def __call__(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            flags,
            thresholds,
            row_ends,
            row_ids,
            packed_thresholds,
            packed_ends,
            active_count,
            rows,
        ).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.jit
    def with_policy(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
        policy: tuple,
        stream: cuda.CUstream,
    ):
        self.kernel(
            flags,
            thresholds,
            row_ends,
            row_ids,
            packed_thresholds,
            packed_ends,
            active_count,
            rows,
            policy,
        ).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
        policy: tuple = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 256 + tidx
        if row < rows:
            if cutlass.const_expr(policy is not None):
                self.repair_policy.prepare_row(
                    policy[0],
                    flags,
                    policy[1],
                    policy[2],
                    row_ends,
                    thresholds,
                    policy[3],
                    row,
                    rows,
                )
            if flags[row] != 0:
                slot = cute.arch.atomic_add(
                    active_count.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                if slot < rows:
                    row_ids[slot] = row
                    packed_thresholds[slot] = thresholds[row]
                    packed_ends[slot] = row_ends[row]


class SelectiveRepairRowPackKernel:
    """Gather compact repair Q rows and weights without host synchronization."""

    def __init__(self, num_heads: int, head_dim: int, blocks: int = 148):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.blocks = blocks

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        weights: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        packed_q: cute.Tensor,
        packed_weights: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            weights,
            row_ids,
            active_count,
            packed_q,
            packed_weights,
            rows,
        ).launch(grid=(self.blocks, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        weights: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        packed_q: cute.Tensor,
        packed_weights: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()
        linear = bidx * 256 + tidx
        stride = grid_size * 256
        active = min(active_count[0], rows)
        q_elements = cutlass.const_expr(self.num_heads * self.head_dim)
        q_work = active * q_elements
        while linear < q_work:
            packed_row = linear // q_elements
            element = linear - packed_row * q_elements
            source_row = row_ids[packed_row]
            packed_q[linear] = q[source_row * q_elements + element]
            linear = linear + stride

        linear = bidx * 256 + tidx
        weight_work = active * self.num_heads
        while linear < weight_work:
            packed_row = linear // self.num_heads
            head = linear - packed_row * self.num_heads
            source_row = row_ids[packed_row]
            packed_weights[linear] = weights[source_row * self.num_heads + head]
            linear = linear + stride


class SelectiveCompactRepairResolveKernel:
    """Scatter successful bounded repair rows and compact rows requiring exact repair."""

    def __init__(
        self,
        top_k: int,
        capacity: int,
        threads: int = 256,
        num_heads: int = 0,
        head_dim: int = 0,
    ):
        self.top_k = top_k
        self.capacity = capacity
        self.threads = threads
        self.num_heads = num_heads
        self.head_dim = head_dim

    @cute.jit
    def with_pack(
        self,
        inputs: tuple,
        pack_inputs: tuple,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(*inputs, rows, pack_inputs).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        first_row_ids: cute.Tensor,
        first_count: cute.Tensor,
        repair_flags: cute.Tensor,
        global_row_ends: cute.Tensor,
        selected: cute.Tensor,
        second_row_ids: cute.Tensor,
        second_ends: cute.Tensor,
        second_ranks: cute.Tensor,
        second_flags: cute.Tensor,
        second_count: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_values,
            repair_lengths,
            positions,
            first_row_ids,
            first_count,
            repair_flags,
            global_row_ends,
            selected,
            second_row_ids,
            second_ends,
            second_ranks,
            second_flags,
            second_count,
            exact_error,
            rows,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        first_row_ids: cute.Tensor,
        first_count: cute.Tensor,
        repair_flags: cute.Tensor,
        global_row_ends: cute.Tensor,
        selected: cute.Tensor,
        second_row_ids: cute.Tensor,
        second_ends: cute.Tensor,
        second_ranks: cute.Tensor,
        second_flags: cute.Tensor,
        second_count: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
        pack_inputs: tuple = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        packed_row, _, _ = cute.arch.block_idx()
        active = packed_row < min(first_count[0], rows)
        source_row = cutlass.Int32(0)
        short_complete = cutlass.Boolean(False)
        if active:
            source_row = first_row_ids[packed_row]
            row_end = global_row_ends[source_row]
            # The finalizer flags every result shorter than TopK. For a short
            # causal row, bounded repair is exact when it published every key, so
            # scatter that complete result instead of sending it through exact repair.
            short_complete = (row_end < self.top_k) & (
                repair_lengths[packed_row] == row_end
            )
        exact_repair = active & (repair_flags[packed_row] != 0) & (not short_complete)
        successful_repair = active & (not exact_repair)

        for item in cutlass.range_constexpr(cute.ceil_div(self.top_k, self.threads)):
            rank = item * self.threads + tidx
            if successful_repair & (rank < self.top_k):
                index = cutlass.Int32(-1)
                if rank < repair_lengths[packed_row]:
                    position = positions[(packed_row, rank)]
                    if (position >= 0) & (position < self.capacity):
                        packed = cutlass.Uint64(packed_values[(packed_row, position)])
                        index = cutlass.Int32(packed & cutlass.Uint64(0xFFFFFFFF))
                selected[(source_row, rank)] = index

        slot = cutlass.Int32(-1)
        if (tidx == 0) & exact_repair:
            slot = cute.arch.atomic_add(
                second_count.iterator,
                cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            if slot < rows:
                end = global_row_ends[source_row]
                second_row_ids[slot] = source_row
                second_ends[slot] = end
                second_ranks[slot] = min(end, cutlass.Int32(self.top_k))
                second_flags[source_row] = cutlass.Int32(1)
            else:
                cute.arch.atomic_add(
                    exact_error.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
        if cutlass.const_expr(pack_inputs is not None):
            if exact_repair & (tidx < 32):
                slot = cute.arch.shuffle_sync(slot, 0)
                if (slot >= 0) & (slot < rows):
                    q_words = self.num_heads * self.head_dim // 4
                    for vector in cutlass.range_constexpr(cute.ceil_div(q_words, 32)):
                        word = vector * 32 + tidx
                        if word < q_words:
                            pack_inputs[2][slot * q_words + word] = pack_inputs[0][
                                source_row * q_words + word
                            ]
                    for group in cutlass.range_constexpr(
                        cute.ceil_div(self.num_heads, 32)
                    ):
                        head = group * 32 + tidx
                        if head < self.num_heads:
                            pack_inputs[3][slot * self.num_heads + head] = pack_inputs[
                                1
                            ][source_row * self.num_heads + head]


class SelectiveCompactTopKCombineExactKernel:
    """Combine compact exact repair output into global-Q result rows."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            greater_indices,
            greater_counts,
            equal_indices,
            equal_counts,
            ranks,
            row_ends,
            row_ids,
            active_count,
            output,
            error_count,
            rows,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        packed_row, _, _ = cute.arch.block_idx()
        if packed_row < min(active_count[0], rows):
            output_row = row_ids[packed_row]
            greater = greater_counts[(packed_row, 0)]
            needed_equal = ranks[packed_row]
            required = min(row_ends[packed_row], cutlass.Int32(self.top_k))
            valid = (greater == required - needed_equal) & (
                equal_counts[(packed_row, 0)] >= needed_equal
            )
            for item_group in cutlass.range_constexpr((self.top_k + 255) // 256):
                column = tidx + item_group * 256
                if column < self.top_k:
                    if valid:
                        if column < required:
                            if column < greater:
                                output[(output_row, column)] = greater_indices[
                                    (packed_row, column)
                                ]
                            else:
                                output[(output_row, column)] = equal_indices[
                                    (packed_row, column - greater)
                                ]
                        else:
                            output[(output_row, column)] = cutlass.Int32(-1)
                    else:
                        output[(output_row, column)] = cutlass.Int32(-1)
            if (tidx == 0) & (not valid):
                cute.arch.atomic_add(
                    error_count.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
