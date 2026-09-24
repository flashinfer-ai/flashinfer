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

"""Fused MXFP4 MoE decode preprocessing; minimum architecture SM100 (B300: SM103).

For T=1..16, unpack global expert IDs, convert BF16 router weights to FP32,
optionally sort assignments into compact expert tiles, and clear the combined
BF16 output in one launch. Planning compiles and binds caller-owned buffers
outside capture; execution uses the supplied CUDA stream.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda
import torch

from ...cute_dsl.utils import make_ptr
from .common.kernel_utils import griddepcontrol_launch_dependents
from .moe_utils import get_max_num_tiles


_route_preprocess_kernel_cache: dict = {}

_LIST_BUFFER_NAMES = (
    "wide_list",
    "wide_count",
    "narrow_list",
    "narrow_count",
    "all_list",
    "all_count",
)

_SORT_BUFFER_NAMES = (
    "out_tile_idx_to_expert_idx",
    "out_tile_idx_to_mn_limit",
    "out_expanded_idx_to_permuted_idx",
    "out_permuted_idx_to_expanded_idx",
    "out_total_num_padded_tokens",
    "out_num_non_exiting_tiles",
)


class _RoutePreprocess:
    def __init__(self, mode, threads, clear=True):
        self.packed = mode == "packed"
        self.convert_weights = mode != "separate_fp32"
        self.threads = threads
        # ``clear=False``: the finalize kernel overwrites every output row, so
        # only the route conversion runs (grid sized by routes).
        self.clear = clear

    @cute.jit
    def __call__(
        self,
        ids_src_ptr: cute.Pointer,
        weights_src_ptr: cute.Pointer,
        ids_dst_ptr: cute.Pointer,
        weights_dst_ptr: cute.Pointer,
        output_words_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden_size: cutlass.Int32,
        ids_row_stride: cutlass.Int32,
        ids_col_stride: cutlass.Int32,
        weights_row_stride: cutlass.Int32,
        weights_col_stride: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        ids_src = cute.make_tensor(
            ids_src_ptr,
            cute.make_layout((tokens, top_k), stride=(ids_row_stride, ids_col_stride)),
        )
        weights_src = cute.make_tensor(
            weights_src_ptr,
            cute.make_layout(
                (tokens, top_k), stride=(weights_row_stride, weights_col_stride)
            ),
        )
        ids_dst = cute.make_tensor(ids_dst_ptr, cute.make_layout((tokens * top_k,)))
        weights_dst = cute.make_tensor(
            weights_dst_ptr, cute.make_layout((tokens * top_k,))
        )
        output_words = cute.make_tensor(
            output_words_ptr, cute.make_layout((tokens * (hidden_size // 2),))
        )
        # Size the grid for output clearing (one 16-byte store per thread).
        # The separate route grid-stride loop remains correct even when its
        # domain is larger than this one.
        tasks = tokens * (hidden_size // 8)
        if cutlass.const_expr(not self.clear):
            tasks = tokens * top_k
        self.kernel(ids_src, weights_src, ids_dst, weights_dst, output_words).launch(
            grid=(cute.ceil_div(tasks, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids_src: cute.Tensor,
        weights_src: cute.Tensor,
        ids_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        output_words: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        first = bidx * self.threads + tidx
        task_stride = grid_x * self.threads

        if cutlass.const_expr(self.convert_weights):
            # Route and clear loops have separate limits. Do not use the
            # route-count bound/stride to cover the much larger token output.
            for route in cutlass.range(first, cute.size(weights_dst), task_stride):
                token = route // ids_src.shape[1]
                slot = route % ids_src.shape[1]
                if cutlass.const_expr(self.packed):
                    # Signed int32 shift matches torch.bitwise_right_shift.
                    ids_dst[route] = ids_src[(token, slot)] >> 16
                # Packed mode binds a BF16 view of each int32's low half, with
                # doubled element strides. This is a bit-preserving load, not
                # a numeric conversion of the packed integer itself.
                weights_dst[route] = weights_src[(token, slot)].to(cutlass.Float32)

        if cutlass.const_expr(self.clear):
            # One aligned 16-byte store writes eight BF16 +0 values (the
            # hidden size is a multiple of 8, so the word count divides by 4).
            zeros = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for i in cutlass.range_constexpr(4):
                zeros[i] = cutlass.Uint32(0)
            num_vec = cute.size(output_words) // 4
            for vec in cutlass.range(first, num_vec, task_stride):
                base = cute.assume(vec * 4, divby=4)
                g_out = cute.make_tensor(
                    output_words.iterator + base, layout=cute.make_layout((4,))
                )
                cute.autovec_copy(zeros, g_out)


@dsl_user_op
def _routing_shared_add(
    address: cutlass.Int32, value: cutlass.Int32, *, loc=None, ip=None
):
    # Same narrow primitive used by FlashInfer's existing shared histograms.
    # The address is already in the 32-bit shared-memory address space.
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(address).ir_value(loc=loc, ip=ip),
                cutlass.Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.s32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _routing_warp_inclusive(value: cutlass.Int32, lane: cutlass.Int32):
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        other = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            value += other
    return value


@cute.jit
def _routing_block_exclusive(
    value: cutlass.Int32,
    lane: cutlass.Int32,
    warp: cutlass.Int32,
    warps: cutlass.Constexpr[int],
    buf: cute.Tensor,
):
    """Exclusive prefix and total of ``value`` over the CTA (all threads
    participate; ``buf`` holds ``warps`` int32 words and is reusable after
    the call returns)."""
    inclusive = _routing_warp_inclusive(value, lane)
    if lane == 31:
        buf[warp] = inclusive
    cute.arch.sync_threads()
    if warp == 0:
        subtotal = cutlass.Int32(0)
        if lane < warps:
            subtotal = buf[lane]
        subtotal = _routing_warp_inclusive(subtotal, lane)
        if lane < warps:
            buf[lane] = subtotal
    cute.arch.sync_threads()
    preceding = cutlass.Int32(0)
    if warp > 0:
        preceding = buf[warp - 1]
    total = buf[warps - 1]
    exclusive = preceding + inclusive - value
    cute.arch.sync_threads()
    return exclusive, total


# Routes (token, slot) one fused sorting CTA handles: T*top_k up to 4096, i.e.
# T <= 256 for Kimi's top-16. Beyond that the generic conversion kernel and
# ``moe_sort`` run.
FUSED_ROUTE_MAX_ROUTES = 4096


class _FusedRoutePreprocess:
    def __init__(
        self,
        mode,
        threads,
        single_tile_per_expert=False,
        max_routes=FUSED_ROUTE_MAX_ROUTES,
        clear=True,
        dispatch_lists=False,
    ):
        self.single_tile_per_expert = single_tile_per_expert
        # ``dispatch_lists``: also emit the swap-AB work lists over the sort
        # groups (wide = dense GEMM1 tiles, narrow = swap GEMM1 sub-tiles,
        # all = every occupied sub-tile for the swap GEMM2), replacing the
        # separate ``swapab_dispatch`` launch of the mixed form.
        self.dispatch_lists = dispatch_lists
        self.packed = mode == "packed"
        self.convert_weights = mode != "separate_fp32"
        self.threads = threads
        self.warps = threads // 32
        # Per-route rank / local-expert scratch for T*top_k > threads.
        self.max_routes = max_routes
        # ``clear=False``: only CTA 0 (the sort) runs; the two-stage finalize
        # overwrites every output row itself.
        self.clear = clear

    @cute.jit
    def __call__(
        self,
        ids_src_ptr: cute.Pointer,
        weights_src_ptr: cute.Pointer,
        ids_dst_ptr: cute.Pointer,
        weights_dst_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        expanded_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        padded_total_ptr: cute.Pointer,
        active_total_ptr: cute.Pointer,
        wide_list_ptr: cute.Pointer,
        wide_count_ptr: cute.Pointer,
        narrow_list_ptr: cute.Pointer,
        narrow_count_ptr: cute.Pointer,
        all_list_ptr: cute.Pointer,
        all_count_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden: cutlass.Int32,
        num_experts: cutlass.Int32,
        local_experts: cutlass.Int32,
        local_offset: cutlass.Int32,
        tile_size: cutlass.Int32,
        tile_capacity: cutlass.Int32,
        narrow_tile: cutlass.Int32,
        wide_min_rows: cutlass.Int32,
        wide_min_permille: cutlass.Int32,
        ids_row_stride: cutlass.Int32,
        ids_col_stride: cutlass.Int32,
        weights_row_stride: cutlass.Int32,
        weights_col_stride: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        ids_src = cute.make_tensor(
            ids_src_ptr,
            cute.make_layout((tokens, top_k), stride=(ids_row_stride, ids_col_stride)),
        )
        weights_src = cute.make_tensor(
            weights_src_ptr,
            cute.make_layout(
                (tokens, top_k), stride=(weights_row_stride, weights_col_stride)
            ),
        )
        ids_dst = cute.make_tensor(ids_dst_ptr, cute.make_layout((tokens * top_k,)))
        weights_dst = cute.make_tensor(
            weights_dst_ptr, cute.make_layout((tokens * top_k,))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_layout((tokens * (hidden // 2),))
        )
        tile_expert = cute.make_tensor(
            tile_expert_ptr, cute.make_layout((tile_capacity,))
        )
        tile_limit = cute.make_tensor(
            tile_limit_ptr, cute.make_layout((tile_capacity,))
        )
        expanded = cute.make_tensor(expanded_ptr, cute.make_layout((tokens * top_k,)))
        permuted = cute.make_tensor(
            permuted_ptr, cute.make_layout((tile_capacity * tile_size,))
        )
        padded_total = cute.make_tensor(padded_total_ptr, cute.make_layout((1,)))
        active_total = cute.make_tensor(active_total_ptr, cute.make_layout((1,)))
        sub_tiles = tile_capacity * (tile_size // narrow_tile)
        wide_list = cute.make_tensor(wide_list_ptr, cute.make_layout((tile_capacity,)))
        wide_count = cute.make_tensor(wide_count_ptr, cute.make_layout((1,)))
        narrow_list = cute.make_tensor(narrow_list_ptr, cute.make_layout((sub_tiles,)))
        narrow_count = cute.make_tensor(narrow_count_ptr, cute.make_layout((1,)))
        all_list = cute.make_tensor(all_list_ptr, cute.make_layout((sub_tiles,)))
        all_count = cute.make_tensor(all_count_ptr, cute.make_layout((1,)))
        self.kernel(
            ids_src,
            weights_src,
            ids_dst,
            weights_dst,
            output,
            tile_expert,
            tile_limit,
            expanded,
            permuted,
            padded_total,
            active_total,
            wide_list,
            wide_count,
            narrow_list,
            narrow_count,
            all_list,
            all_count,
            num_experts,
            local_experts,
            local_offset,
            tile_size,
            narrow_tile,
            wide_min_rows,
            wide_min_permille,
        ).launch(
            grid=(
                cute.ceil_div(cute.size(output), self.threads)
                if cutlass.const_expr(self.clear)
                else 1,
                1,
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids_src: cute.Tensor,
        weights_src: cute.Tensor,
        ids_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        output: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        expanded: cute.Tensor,
        permuted: cute.Tensor,
        padded_total: cute.Tensor,
        active_total: cute.Tensor,
        wide_list: cute.Tensor,
        wide_count: cute.Tensor,
        narrow_list: cute.Tensor,
        narrow_count: cute.Tensor,
        all_list: cute.Tensor,
        all_count: cute.Tensor,
        num_experts: cutlass.Int32,
        local_experts: cutlass.Int32,
        local_offset: cutlass.Int32,
        tile_size: cutlass.Int32,
        narrow_tile: cutlass.Int32,
        wide_min_rows: cutlass.Int32,
        wide_min_permille: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        # Let a PDL-launched successor (swap-AB GEMM1) become resident now; it
        # waits (griddepcontrol.wait) before reading any output of this kernel.
        griddepcontrol_launch_dependents()
        smem = utils.SmemAllocator()
        counts = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.threads,)))
        bases = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.threads,)))
        warp_sums = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.warps,)))
        scan_buf = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.warps,)))
        rank_buf = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.max_routes,))
        )
        local_buf = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.max_routes,))
        )
        num_routes = cute.size(expanded)

        # The branch is uniform across the entire CTA. Each barrier below has
        # all threads of CTA0 participating; no named or cluster barriers.
        if block == 0:
            counts[tid] = cutlass.Int32(0)
            cute.arch.sync_threads()
            # Histogram: every route of the CTA (grid-stride over the threads)
            # converts its ID/weight and takes a rank within its local expert.
            for r in cutlass.range(tid, num_routes, self.threads):
                token = r // ids_src.shape[1]
                slot = r % ids_src.shape[1]
                expert = ids_src[(token, slot)]
                if cutlass.const_expr(self.packed):
                    expert = expert >> 16
                    ids_dst[r] = expert
                if cutlass.const_expr(self.convert_weights):
                    weights_dst[r] = weights_src[(token, slot)].to(cutlass.Float32)
                local = expert - local_offset
                expanded[r] = cutlass.Int32(-1)
                rank = cutlass.Int32(-1)
                if (
                    (expert >= 0)
                    & (expert < num_experts)
                    & (local >= 0)
                    & (local < local_experts)
                ):
                    address = (counts.iterator + local).toint().to(cutlass.Int32)
                    rank = _routing_shared_add(address, cutlass.Int32(1))
                rank_buf[r] = rank
                local_buf[r] = local
            cute.arch.sync_threads()

            count = counts[tid]
            if cutlass.const_expr(self.single_tile_per_expert):
                # Unique IDs give at most T rows per expert, and planning
                # requires tile_size >= T. Every expert has zero or one tile.
                ntiles = (count > 0).to(cutlass.Int32)
                active_mask = cute.arch.vote_ballot_sync(count > 0)
                inclusive_mask = cutlass.Uint32(0xFFFFFFFF) >> (31 - lane)
                within_warp = cute.arch.popc(active_mask & inclusive_mask)
            else:
                ntiles = (count + tile_size - 1) // tile_size
                within_warp = _routing_warp_inclusive(ntiles, lane)
            if lane == 31:
                warp_sums[warp] = within_warp
            cute.arch.sync_threads()
            if warp == 0:
                subtotal = cutlass.Int32(0)
                if lane < self.warps:
                    subtotal = warp_sums[lane]
                subtotal = _routing_warp_inclusive(subtotal, lane)
                if lane < self.warps:
                    warp_sums[lane] = subtotal
            cute.arch.sync_threads()
            preceding_warps = cutlass.Int32(0)
            if warp > 0:
                preceding_warps = warp_sums[warp - 1]
            tile_base = preceding_warps + within_warp - ntiles
            row_base = tile_base * tile_size
            bases[tid] = row_base
            if cutlass.const_expr(self.dispatch_lists):
                # Swap-AB work lists over this CTA's sort groups (the same
                # lists ``swapab_dispatch`` builds from tile_limit): ``full``
                # complete groups and one ``last`` partial group per expert.
                nonempty = (count > 0).to(cutlass.Int32)
                full = ntiles - nonempty
                last = count - full * tile_size
                sub = tile_size // narrow_tile
                last_sub = (last + narrow_tile - 1) // narrow_tile
                # Global rule: dense tiles only when the groups above
                # ``wide_min_rows`` hold ``wide_min_permille`` of the rows.
                wide_rows = cutlass.Int32(0)
                if tile_size > wide_min_rows:
                    wide_rows = full * tile_size
                if last > wide_min_rows:
                    wide_rows = wide_rows + last
                _, total_rows = _routing_block_exclusive(
                    count, lane, warp, self.warps, scan_buf
                )
                _, total_wide = _routing_block_exclusive(
                    wide_rows, lane, warp, self.warps, scan_buf
                )
                effective_min = wide_min_rows
                if wide_min_permille > 0:
                    if total_wide * 1000 < total_rows * wide_min_permille:
                        effective_min = tile_size
                full_wide = (tile_size > effective_min).to(cutlass.Int32)
                last_wide = (last > effective_min).to(cutlass.Int32)
                n_wide = full * full_wide + last_wide
                n_all = full * sub + last_sub
                n_narrow = full * sub * (1 - full_wide) + last_sub * (1 - last_wide)
                wide_base, wide_total = _routing_block_exclusive(
                    n_wide, lane, warp, self.warps, scan_buf
                )
                narrow_base, narrow_total = _routing_block_exclusive(
                    n_narrow, lane, warp, self.warps, scan_buf
                )
                all_base, all_total = _routing_block_exclusive(
                    n_all, lane, warp, self.warps, scan_buf
                )
                if tid < local_experts:
                    for j in cutlass.range(full):
                        tile = tile_base + j
                        if full_wide == 1:
                            wide_list[wide_base + j] = tile
                        else:
                            for s in cutlass.range(sub):
                                narrow_list[narrow_base + j * sub + s] = tile * sub + s
                        for s in cutlass.range(sub):
                            all_list[all_base + j * sub + s] = tile * sub + s
                    if last > 0:
                        tile = tile_base + full
                        if last_wide == 1:
                            wide_list[wide_base + full * full_wide] = tile
                        else:
                            narrow_off = narrow_base + full * sub * (1 - full_wide)
                            for s in cutlass.range(last_sub):
                                narrow_list[narrow_off + s] = tile * sub + s
                        for s in cutlass.range(last_sub):
                            all_list[all_base + full * sub + s] = tile * sub + s
                if tid == self.threads - 1:
                    wide_count[0] = wide_total
                    narrow_count[0] = narrow_total
                    all_count[0] = all_total
            if tid < local_experts:
                for j in cutlass.range(ntiles):
                    tile = tile_base + j
                    limit = (tile + 1) * tile_size
                    if limit > row_base + count:
                        limit = row_base + count
                    tile_expert[tile] = tid
                    tile_limit[tile] = limit
            if tid == self.threads - 1:
                active = preceding_warps + within_warp
                active_total[0] = active
                padded_total[0] = active * tile_size
            cute.arch.sync_threads()
            for r in cutlass.range(tid, num_routes, self.threads):
                rank = rank_buf[r]
                if rank >= 0:
                    row = bases[local_buf[r]] + rank
                    expanded[r] = row
                    permuted[row] = r

        if cutlass.const_expr(self.clear):
            # Each word is covered once, independently of the assignment count.
            word = block * self.threads + tid
            if word < cute.size(output):
                output[word] = cutlass.Uint32(0)


class _RoutePreprocessPlan:
    """Fixed-address launcher. One plan per concurrent mutable buffer set."""

    def __init__(
        self,
        compiled,
        arguments,
        owners,
        route_ids,
        route_weights,
        output,
        mode,
        sorts_tokens=False,
    ):
        self._compiled = compiled
        self._arguments = arguments
        self._owners = owners
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.output = output
        self.device = output.device
        self.mode = mode
        self.sorts_tokens = sorts_tokens

    def run(self, stream):
        """Enqueue one kernel; no buffer allocation, JIT or host synchronization."""
        self._compiled(*self._arguments, stream=stream)
        return self.output


def _check_tensor(name, tensor, shape, dtype, device, *, contiguous=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must have nonnegative strides")


def _interval(tensor):
    span = 1 + sum(
        (dim - 1) * stride
        for dim, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()


def _disjoint(left, right):
    a, b = _interval(left), _interval(right)
    return max(a[0], b[0]) >= min(a[1], b[1])


def _plan_route_preprocess(
    topk_ids,
    topk_weights,
    *,
    route_ids=None,
    route_weights=None,
    output,
    threads=256,
    moe_sort_buffers=None,
    num_experts=None,
    num_local_experts=None,
    local_expert_offset=0,
    tile_size=128,
    _single_tile_per_expert=False,
    clear_output=True,
    dispatch_lists=None,
):
    """Compile, bind and enqueue one warmup, outside CUDA Graph capture.

    ``topk_weights=None`` means packed int32 IDs/high16 + BF16 weights/low16.
    Packed input requires caller-owned contiguous int32 ``route_ids`` and FP32
    ``route_weights`` outputs. Separate BF16 weights require ``route_weights``;
    separate IDs are bound directly. Separate FP32 weights are bound directly,
    and require no conversion. Do not pass different destination buffers for
    either direct-bind input. All tensors must be on one device.

    With ``moe_sort_buffers``, CTA0 additionally builds a local expert histogram
    and writes the six existing ``moe_sort`` outputs. Empty experts have no
    tiles; padding inside nonempty expert tiles is unchanged. Expert IDs must
    be unique within each token, as in standard top-k routing. This mode is
    used only when PDL is disabled; later GEMMs wait for the complete kernel.

    Inputs may have independent nonnegative 2-D strides. Output and conversion
    buffers are contiguous. The conversion + clear kernel serves any token
    count; the fused sorting mode supports tokens from 1..16. Top-k is 1..32
    and the hidden size a positive even number. No input data are read on
    the host. Tensor contents must be valid when planning because warmup runs.
    ``dispatch_lists`` (fused sorting only) is a dict with the int32 buffers
    ``wide_list`` (tile_capacity), ``wide_count`` (1), ``narrow_list`` and
    ``all_list`` (tile_capacity * tile_size // narrow_tile), ``narrow_count``
    and ``all_count`` (1) plus the ints ``narrow_tile`` (divides tile_size),
    ``wide_min_rows`` and ``wide_min_permille``: CTA0 then also emits the
    swap-AB work lists ``swapab_dispatch`` would build from the sort groups
    (dense-tile groups above ``wide_min_rows``, the narrow sub-tiles of the
    others, and every occupied sub-tile), so the mixed form needs no extra
    launch.

    Shapes/strides are dynamic kernel arguments; beta values and tactic choices
    do not enter compilation. Cache entries are per input/sort mode, thread
    count, device and single-tile prefix mode. Sorting uses 256..1024 threads,
    selected during planning. The private single-tile option relies on the
    unique-ID contract and falls back when tile_size < tokens.
    """
    if (
        not isinstance(output, torch.Tensor)
        or output.device.type != "cuda"
        or output.ndim != 2
    ):
        raise ValueError("output must be a 2-D CUDA tensor")
    tokens, hidden_size = output.shape
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a 2-D tensor")
    top_k = topk_ids.shape[1]
    if not 1 <= top_k <= 32 or hidden_size <= 0 or hidden_size % 2 or tokens < 1:
        raise ValueError("require T>=1, top_k=1..32 and positive even hidden_size")
    if moe_sort_buffers is None and hidden_size % 8:
        raise ValueError(
            "route preprocessing clears the output with 16-byte stores; the "
            "hidden size must be a multiple of 8"
        )
    if moe_sort_buffers is not None and tokens * top_k > FUSED_ROUTE_MAX_ROUTES:
        # The fused sort is a single-CTA kernel bounded by its per-route smem
        # scratch; the conversion + output clear kernel is grid-strided and
        # serves any token count (moe_sort then groups the rows).
        raise ValueError(
            f"fused route sorting handles at most {FUSED_ROUTE_MAX_ROUTES} routes"
        )
    if threads not in (128, 256):
        raise ValueError("route preprocessing supports 128 or 256 threads per block")
    device = output.device
    shape = (tokens, top_k)
    _check_tensor(
        "output", output, (tokens, hidden_size), torch.bfloat16, device, contiguous=True
    )
    if output.data_ptr() % (4 if moe_sort_buffers is not None else 16):
        raise ValueError(
            "output must be aligned to 4 bytes (fused decode sort) or 16 bytes "
            "(generic conversion + clear) for the BF16 zero stores"
        )
    _check_tensor("topk_ids", topk_ids, shape, torch.int32, device)
    writable = [("output", output)]
    inputs = [("topk_ids", topk_ids)]
    sorts_tokens = moe_sort_buffers is not None
    if sorts_tokens:
        if not (
            isinstance(num_experts, int)
            and isinstance(num_local_experts, int)
            and top_k <= num_experts <= 1024
            and num_local_experts > 0
            and local_expert_offset >= 0
            and local_expert_offset + num_local_experts <= num_experts
            and tile_size > 0
        ):
            raise ValueError("invalid fused decode sorting geometry")
        tile_capacity = get_max_num_tiles(tokens, top_k, num_local_experts, tile_size)
        shapes = (
            (tile_capacity,),
            (tile_capacity,),
            shape,
            (tile_capacity * tile_size,),
            (1,),
            (1,),
        )
        for name, expected in zip(_SORT_BUFFER_NAMES, shapes, strict=True):
            tensor = moe_sort_buffers[name]
            _check_tensor(name, tensor, expected, torch.int32, device, contiguous=True)
            writable.append((name, tensor))
        # 1024 threads: the histogram/scatter loops stride over T * top_k
        # routes (4096 at T=256), and a 256-thread CTA took 12 us there.
        required = max(1024, num_local_experts)
        threads = 1 << (required - 1).bit_length()
        if dispatch_lists is not None:
            narrow_tile = int(dispatch_lists["narrow_tile"])
            if narrow_tile <= 0 or tile_size % narrow_tile:
                raise ValueError("dispatch_lists.narrow_tile must divide tile_size")
            sub_tiles = tile_capacity * (tile_size // narrow_tile)
            for name, expected in (
                ("wide_list", (tile_capacity,)),
                ("wide_count", (1,)),
                ("narrow_list", (sub_tiles,)),
                ("narrow_count", (1,)),
                ("all_list", (sub_tiles,)),
                ("all_count", (1,)),
            ):
                tensor = dispatch_lists[name]
                _check_tensor(
                    "dispatch_lists." + name,
                    tensor,
                    expected,
                    torch.int32,
                    device,
                    contiguous=True,
                )
                writable.append(("dispatch_lists." + name, tensor))
            list_ints = (
                narrow_tile,
                int(dispatch_lists["wide_min_rows"]),
                int(dispatch_lists["wide_min_permille"]),
            )
        else:
            list_ints = (tile_size, tile_size, 0)
    elif dispatch_lists is not None:
        raise ValueError("dispatch_lists requires moe_sort_buffers (fused sorting)")

    if topk_weights is None:
        mode = "packed"
        _check_tensor(
            "route_ids", route_ids, shape, torch.int32, device, contiguous=True
        )
        _check_tensor(
            "route_weights",
            route_weights,
            shape,
            torch.float32,
            device,
            contiguous=True,
        )
        writable.extend((("route_ids", route_ids), ("route_weights", route_weights)))
        weight_source = topk_ids
        weights_dtype = cutlass.BFloat16
        weight_strides = tuple(2 * value for value in topk_ids.stride())
    else:
        if not isinstance(topk_weights, torch.Tensor) or topk_weights.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "topk_weights must be BF16 or FP32, or None for packed input"
            )
        _check_tensor("topk_weights", topk_weights, shape, topk_weights.dtype, device)
        inputs.append(("topk_weights", topk_weights))
        if route_ids is not None and route_ids is not topk_ids:
            raise ValueError(
                "separate IDs are bound directly; pass route_ids=None or topk_ids"
            )
        route_ids = topk_ids
        weight_source = topk_weights
        weight_strides = topk_weights.stride()
        if topk_weights.dtype == torch.bfloat16:
            mode = "separate_bf16"
            weights_dtype = cutlass.BFloat16
            _check_tensor(
                "route_weights",
                route_weights,
                shape,
                torch.float32,
                device,
                contiguous=True,
            )
            writable.append(("route_weights", route_weights))
        else:
            mode = "separate_fp32"
            weights_dtype = cutlass.Float32
            if route_weights is not None and route_weights is not topk_weights:
                raise ValueError(
                    "FP32 weights are bound directly; pass route_weights=None or topk_weights"
                )
            route_weights = topk_weights

    for index, (name, tensor) in enumerate(writable):
        for other_name, other in inputs + writable[:index]:
            if not _disjoint(tensor, other):
                raise ValueError(f"{name} must not overlap {other_name}")

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "route preprocessing must be planned outside graph capture"
            )
        arch = torch.cuda.get_device_capability(device)
        if arch not in ((10, 0), (10, 3)):
            raise RuntimeError("route preprocessing requires SM100/SM103")
        pointers = (
            make_ptr(
                cutlass.Int32,
                topk_ids.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                weights_dtype,
                weight_source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=2 if weights_dtype == cutlass.BFloat16 else 4,
            ),
            make_ptr(
                cutlass.Int32,
                route_ids.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float32,
                route_weights.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint32,
                output.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
        )
        if sorts_tokens:
            pointers += tuple(
                make_ptr(
                    cutlass.Int32,
                    moe_sort_buffers[name].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for name in _SORT_BUFFER_NAMES
            )
            # Work-list pointers; without lists the (never written) slots
            # alias the padded-total word so the signature stays fixed.
            list_source = (
                dispatch_lists
                if dispatch_lists is not None
                else {
                    name: moe_sort_buffers["out_total_num_padded_tokens"]
                    for name in _LIST_BUFFER_NAMES
                }
            )
            pointers += tuple(
                make_ptr(
                    cutlass.Int32,
                    list_source[name].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for name in _LIST_BUFFER_NAMES
            )
            arguments = pointers + (
                tokens,
                top_k,
                hidden_size,
                num_experts,
                num_local_experts,
                local_expert_offset,
                tile_size,
                tile_capacity,
                *list_ints,
                *topk_ids.stride(),
                *weight_strides,
            )
        else:
            arguments = pointers + (
                tokens,
                top_k,
                hidden_size,
                *topk_ids.stride(),
                *weight_strides,
            )
        single_tile_per_expert = (
            _single_tile_per_expert and sorts_tokens and tile_size >= tokens
        )
        cache_key = (
            mode,
            threads,
            device.index,
            arch,
            sorts_tokens,
            single_tile_per_expert,
            bool(clear_output),
            dispatch_lists is not None,
        )
        compiled = _route_preprocess_kernel_cache.get(cache_key)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        if compiled is None:
            kernel = (
                _FusedRoutePreprocess(
                    mode,
                    threads,
                    single_tile_per_expert,
                    max_routes=FUSED_ROUTE_MAX_ROUTES,
                    clear=bool(clear_output),
                    dispatch_lists=dispatch_lists is not None,
                )
                if sorts_tokens
                else _RoutePreprocess(mode, threads, clear=bool(clear_output))
            )
            compiled = cute.compile(kernel, *arguments, stream=stream)
            _route_preprocess_kernel_cache[cache_key] = compiled
        bound = _RoutePreprocessPlan(
            compiled,
            arguments,
            (
                topk_ids,
                topk_weights,
                route_ids,
                route_weights,
                output,
                moe_sort_buffers,
                dispatch_lists,
            ),
            route_ids,
            route_weights,
            output,
            mode,
            sorts_tokens,
        )
        bound.run(stream)
        return bound
