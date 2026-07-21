# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inspect caller-owned canonical BSR metadata in one pass.

A BSR Q-block row is one ``block_indptr`` row keyed by
``(batch, kv_head, q_block)``. An execution Q tile is a Q64 or Q128 tile that
reuses that row. A retained KV route is one KV128 unit (two KV64 fragments)
left after physical-tail trimming. A token guard check is a full-mask test
that can skip 32 per-bit predicate iterations.

Four warps validate four BSR Q-block rows per CTA and publish seven plan-wide
facts in one Int64 summary. No per-route payload is constructed.
"""

import functools
from collections.abc import Callable
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_drv

from .fmha_decode_resources.helpers_block_sparse import (
    _block_sparse_row_retained_route_count,
    resolve_block_sparse_route_origins,
)


_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"

# ``summary`` is a zero-initialized Int64[7] shared with the host wrapper.
# Fields 0..3 are plan-wide maxima/flags; fields 4..6 are plan-wide sums
# weighted by the number of execution Q tiles that reuse each semantic row.
_SUMMARY_ERROR_CODE = 0
_SUMMARY_MAX_SELECTED_KV_BLOCK_COUNT = 1
_SUMMARY_MAX_RETAINED_KV_ROUTE_COUNT = 2
_SUMMARY_SELECTED_TOKEN_HOLE = 3
_SUMMARY_RUNTIME_TOKEN_GUARD_SKIP_COUNT = 4
_SUMMARY_RUNTIME_TOKEN_GUARD_CHECK_COUNT = 5
_SUMMARY_RUNTIME_TOKEN_MASK_FULL_KV_ROUTE_COUNT = 6
_SUMMARY_FIELDS = 7

_BSR_ERROR_NONE = 0
_BSR_ERROR_NOT_STRICTLY_INCREASING = 1
_BSR_ERROR_INDEX_OUT_OF_RANGE = 2
_BSR_ERROR_INVALID_INDPTR = 3


@cute.jit
def _load_runtime_token_word(
    kv_valid_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    fragment_origin: cutlass.Int32,
    fragment_valid: cutlass.Boolean,
    word_in_fragment: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
) -> tuple[cutlass.Uint32, cutlass.Boolean]:
    """Mirror one runtime word load and report holes only in real tokens.

    Padding bits beyond ``seq_len_kv`` are excluded from the hole predicate.
    The returned word remains unmodified because the runtime full-word guard
    requires the complete stored word to equal ``0xffffffff``.
    """

    mask_word = cutlass.Uint32(0)
    word_has_selected_token_hole = cutlass.Boolean(False)
    word_begin = fragment_origin + cutlass.Int32(word_in_fragment * 32)
    if fragment_valid and word_begin < cutlass.Int32(seq_len_kv):
        word_idx = word_begin >> cutlass.Int32(5)
        mask_word = cutlass.Uint32(kv_valid_bits[batch_idx, word_idx])
        remaining_tokens = cutlass.Int32(seq_len_kv) - word_begin
        expected = cutlass.Uint32(0xFFFFFFFF)
        if remaining_tokens < cutlass.Int32(32):
            expected = (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
        word_has_selected_token_hole = (
            word_has_selected_token_hole or (mask_word & expected) != expected
        )
    return mask_word, word_has_selected_token_hole


@cute.jit
def _inspect_kv_route_token_mask(
    block_indices: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    kv_route_idx: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
    q_tile_size: cutlass.Constexpr[int],
    count_token_mask_full_routes: cutlass.Constexpr[bool],
) -> tuple[cutlass.Boolean, cutlass.Int32, cutlass.Int32]:
    """Inspect one retained KV route as its Q64/Q128 token guard sees it.

    Returns ``(route_has_selected_token_hole, route_token_guard_skip_count,
    token_mask_full_kv_route)``. The final value remains an Int32 0/1 and is
    collected only for Q128 inspection.
    """

    origin0, valid0, origin1, valid1 = resolve_block_sparse_route_origins(
        block_indices.iterator,
        bsr_row_begin,
        bsr_row_end,
        kv_route_idx,
        kv_block_size,
        cutlass.Int32(seq_len_kv),
    )
    word0, hole0 = _load_runtime_token_word(
        kv_valid_bits, batch_idx, origin0, valid0, 0, seq_len_kv
    )
    word1, hole1 = _load_runtime_token_word(
        kv_valid_bits, batch_idx, origin0, valid0, 1, seq_len_kv
    )
    word2, hole2 = _load_runtime_token_word(
        kv_valid_bits, batch_idx, origin1, valid1, 0, seq_len_kv
    )
    word3, hole3 = _load_runtime_token_word(
        kv_valid_bits, batch_idx, origin1, valid1, 1, seq_len_kv
    )
    route_token_guard_skip_count = cutlass.Int32(0)
    token_mask_full_kv_route = cutlass.Int32(0)
    full_word = cutlass.Uint32(0xFFFFFFFF)
    if cutlass.const_expr(q_tile_size == 64):
        # Q64's paired lane groups can skip only when both words are full.
        route_token_guard_skip_count = cutlass.Int32(
            word0 == full_word and word2 == full_word
        )
        route_token_guard_skip_count += cutlass.Int32(
            word1 == full_word and word3 == full_word
        )
    else:
        # Q128 lanes share each per-word branch, so each word is one check.
        word0_is_full = word0 == full_word
        word1_is_full = word1 == full_word
        word2_is_full = word2 == full_word
        word3_is_full = word3 == full_word
        route_token_guard_skip_count = cutlass.Int32(word0_is_full)
        route_token_guard_skip_count += cutlass.Int32(word1_is_full)
        route_token_guard_skip_count += cutlass.Int32(word2_is_full)
        route_token_guard_skip_count += cutlass.Int32(word3_is_full)
        if cutlass.const_expr(count_token_mask_full_routes):
            token_mask_full_kv_route = cutlass.Int32(
                word0_is_full and word1_is_full and word2_is_full and word3_is_full
            )
    route_has_selected_token_hole = hole0 or hole1 or hole2 or hole3
    return (
        route_has_selected_token_hole,
        route_token_guard_skip_count,
        token_mask_full_kv_route,
    )


@cute.jit
def _validate_bsr_row_lane(
    block_indices: cute.Tensor,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    num_kv_blocks: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Validate one lane stripe after proving the whole range is bounded."""

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    selected_kv_block_count = bsr_row_end - bsr_row_begin
    bsr_entry_offset = lane_idx
    while bsr_entry_offset < selected_kv_block_count:
        entry_position = bsr_row_begin + bsr_entry_offset
        block_id = cutlass.Int32(block_indices[entry_position])
        in_range = block_id >= 0 and block_id < num_kv_blocks
        if not in_range:
            error_code = cutlass.Int32(_BSR_ERROR_INDEX_OUT_OF_RANGE)
        elif entry_position > bsr_row_begin:
            previous_block_id = cutlass.Int32(block_indices[entry_position - 1])
            if (
                block_id <= previous_block_id
                and error_code < _BSR_ERROR_NOT_STRICTLY_INCREASING
            ):
                error_code = cutlass.Int32(_BSR_ERROR_NOT_STRICTLY_INCREASING)
        remaining_entries = selected_kv_block_count - bsr_entry_offset
        bsr_entry_offset = (
            bsr_entry_offset + _WARP_SIZE
            if remaining_entries > _WARP_SIZE
            else selected_kv_block_count
        )
    return error_code


class _InspectBlockSparseBsr:
    """Validate canonical BSR and reduce the facts needed by ``plan()``.

    The runtime inputs are compact tensors with the following ABI:

    * ``block_indptr``: Int32 ``[B, Hkv, num_q_block_rows + 1]`` offsets;
    * ``block_indices``: Int32 ``[nnz]`` semantic KV-block IDs;
    * ``kv_valid_bits``: optional Uint32 ``[B, ceil(Skv / 32)]`` token bits;
    * ``summary``: zero-initialized Int64 ``[7]`` output.

    One 128-thread CTA contains four independent warps, and each warp handles
    one flattened ``(batch, kv_head, q_block_row)``. Lanes first validate the
    row's index stripe, then scan its retained KV routes when a
    token mask is present. The output fields are:

    * ``[0]`` highest validation error code (zero means canonical BSR);
    * ``[1]`` maximum selected KV-block count in any row;
    * ``[2]`` maximum retained KV-route count;
    * ``[3]`` whether any selected real token has a zero validity bit;
    * ``[4]`` runtime token guard checks that can skip per-bit masking;
    * ``[5]`` runtime token guard checks, including dummy route slots;
    * ``[6]`` runtime token-mask-full retained KV routes (Q128 only).

    Fields 4--6 model the actual schedule and are weighted by the number of
    execution Q tiles consuming the semantic row. Field 5 includes
    minimum-schedule and odd-route dummy slots; fields 4 and 6 count real full
    work only. All three are zero without a token mask. No per-route metadata
    is produced; the attention kernel continues to consume the caller's BSR.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        has_token_bits: bool,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.inspection_q_tile_size = 128 if q_block_size % 128 == 0 else 64
        self.inspect_token_mask = has_token_bits
        self.count_token_mask_full_kv_routes = (
            self.inspect_token_mask and self.inspection_q_tile_size == 128
        )
        self.num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        self.num_kv_blocks = (seq_len_kv + kv_block_size - 1) // kv_block_size
        self.total_bsr_row_count = batch_size * num_kv_heads * self.num_q_block_rows

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: Optional[cute.Tensor],
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        """Launch four BSR-row inspectors per CTA on ``stream``."""

        self.kernel(
            block_indptr,
            block_indices,
            kv_valid_bits,
            summary,
        ).launch(
            grid=[
                (self.total_bsr_row_count + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA,
                1,
                1,
            ],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: Optional[cute.Tensor],
        summary: cute.Tensor,
    ) -> None:
        """Validate rows and atomically reduce their plan-time statistics."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE

        # Q-block row is the fastest-moving dimension, then head and batch.
        linear_bsr_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        bsr_row_is_valid = linear_bsr_row_idx < self.total_bsr_row_count
        safe_linear_bsr_row_idx = (
            linear_bsr_row_idx if bsr_row_is_valid else cutlass.Int32(0)
        )
        q_block_row_idx = safe_linear_bsr_row_idx % self.num_q_block_rows
        linear_batch_head_idx = safe_linear_bsr_row_idx // self.num_q_block_rows
        kv_head_idx = linear_batch_head_idx % self.num_kv_heads
        batch_idx = linear_batch_head_idx // self.num_kv_heads

        bsr_row_begin = cutlass.Int32(0)
        bsr_row_end = cutlass.Int32(0)
        bsr_row_range_is_valid = cutlass.Boolean(False)
        error_code = cutlass.Int32(_BSR_ERROR_NONE)
        lane_has_selected_token_hole = cutlass.Boolean(False)

        if bsr_row_is_valid:
            bsr_row_begin = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx]
            )
            bsr_row_end = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
            )
            num_indices = cutlass.Int32(cute.size(block_indices))
            bsr_row_range_is_valid = cutlass.Boolean(
                bsr_row_begin >= 0
                and bsr_row_begin <= bsr_row_end
                and bsr_row_end <= num_indices
            )
            if not bsr_row_range_is_valid:
                error_code = cutlass.Int32(_BSR_ERROR_INVALID_INDPTR)
            else:
                error_code = _validate_bsr_row_lane(
                    block_indices,
                    bsr_row_begin,
                    bsr_row_end,
                    lane_idx,
                    self.num_kv_blocks,
                )

        # Numeric error codes encode reporting priority. Both this warp
        # reduction and summary[0]'s atomic max retain the strongest error.
        bsr_row_error_code = cutlass.Int32(cute.arch.warp_redux_sync(error_code, "max"))

        retained_kv_route_count = cutlass.Int32(0)
        if lane_idx == 0 and bsr_row_is_valid and bsr_row_error_code == _BSR_ERROR_NONE:
            retained_kv_route_count = _block_sparse_row_retained_route_count(
                block_indices.iterator,
                bsr_row_begin,
                bsr_row_end,
                self.kv_block_size,
                cutlass.Int32(self.seq_len_kv),
            )
        if cutlass.const_expr(self.inspect_token_mask):
            retained_kv_route_count = cutlass.Int32(
                cute.arch.shuffle_sync(retained_kv_route_count, cutlass.Int32(0))
            )

        lane_token_guard_skip_count = cutlass.Int32(0)
        lane_token_mask_full_kv_route_count = cutlass.Int32(0)
        if cutlass.const_expr(self.inspect_token_mask):
            assert kv_valid_bits is not None
            if bsr_row_is_valid and bsr_row_error_code == _BSR_ERROR_NONE:
                kv_route_idx = lane_idx
                while kv_route_idx < retained_kv_route_count:
                    (
                        route_has_selected_token_hole,
                        route_token_guard_skip_count,
                        token_mask_full_kv_route,
                    ) = _inspect_kv_route_token_mask(
                        block_indices,
                        kv_valid_bits,
                        batch_idx,
                        bsr_row_begin,
                        bsr_row_end,
                        kv_route_idx,
                        self.kv_block_size,
                        self.seq_len_kv,
                        self.inspection_q_tile_size,
                        self.count_token_mask_full_kv_routes,
                    )
                    lane_has_selected_token_hole = (
                        lane_has_selected_token_hole or route_has_selected_token_hole
                    )
                    lane_token_guard_skip_count += route_token_guard_skip_count
                    if cutlass.const_expr(self.count_token_mask_full_kv_routes):
                        lane_token_mask_full_kv_route_count += token_mask_full_kv_route
                    remaining_kv_routes = retained_kv_route_count - kv_route_idx
                    kv_route_idx = (
                        kv_route_idx + _WARP_SIZE
                        if remaining_kv_routes > _WARP_SIZE
                        else retained_kv_route_count
                    )
        bsr_row_has_selected_token_hole = cute.arch.vote_any_sync(
            lane_has_selected_token_hole
        )
        bsr_row_token_guard_skip_count = cutlass.Int32(0)
        if cutlass.const_expr(self.inspect_token_mask):
            bsr_row_token_guard_skip_count = cutlass.Int32(
                cute.arch.warp_redux_sync(lane_token_guard_skip_count, "add")
            )
        bsr_row_token_mask_full_kv_route_count = cutlass.Int32(0)
        if cutlass.const_expr(self.count_token_mask_full_kv_routes):
            bsr_row_token_mask_full_kv_route_count = cutlass.Int32(
                cute.arch.warp_redux_sync(lane_token_mask_full_kv_route_count, "add")
            )

        runtime_bsr_row_token_guard_skip_count = cutlass.Int64(0)
        runtime_bsr_row_token_guard_check_count = cutlass.Int64(0)
        runtime_bsr_row_token_mask_full_kv_route_count = cutlass.Int64(0)
        if lane_idx == 0 and bsr_row_is_valid:
            if bsr_row_error_code != _BSR_ERROR_NONE:
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_ERROR_CODE,
                    cutlass.Int64(bsr_row_error_code),
                    sem="relaxed",
                    scope="gpu",
                )
            else:
                selected_kv_block_count = bsr_row_end - bsr_row_begin
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_MAX_SELECTED_KV_BLOCK_COUNT,
                    cutlass.Int64(selected_kv_block_count),
                    sem="relaxed",
                    scope="gpu",
                )
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_MAX_RETAINED_KV_ROUTE_COUNT,
                    cutlass.Int64(retained_kv_route_count),
                    sem="relaxed",
                    scope="gpu",
                )
                if bsr_row_has_selected_token_hole:
                    cute.arch.atomic_max(
                        summary.iterator + _SUMMARY_SELECTED_TOKEN_HOLE,
                        cutlass.Int64(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                if cutlass.const_expr(self.inspect_token_mask):
                    q_token_begin = q_block_row_idx * cutlass.Int32(self.q_block_size)
                    q_tokens_in_bsr_row = cutlass.Int32(self.seq_len_q) - q_token_begin
                    if q_tokens_in_bsr_row > cutlass.Int32(self.q_block_size):
                        q_tokens_in_bsr_row = cutlass.Int32(self.q_block_size)
                    q_tiles_per_bsr_row = (
                        q_tokens_in_bsr_row
                        + cutlass.Int32(self.inspection_q_tile_size - 1)
                    ) // cutlass.Int32(self.inspection_q_tile_size)
                    runtime_bsr_row_token_guard_skip_count = cutlass.Int64(
                        bsr_row_token_guard_skip_count
                    ) * cutlass.Int64(q_tiles_per_bsr_row)
                    if cutlass.const_expr(self.count_token_mask_full_kv_routes):
                        runtime_bsr_row_token_mask_full_kv_route_count = cutlass.Int64(
                            bsr_row_token_mask_full_kv_route_count
                        ) * cutlass.Int64(q_tiles_per_bsr_row)
                    # padded_kv_route_slot_count = max(2,
                    #     round_up(retained_kv_route_count, 2)).
                    # Two KV instances schedule KV routes in pairs.
                    padded_kv_route_slot_count = cute.math.max(
                        (retained_kv_route_count + cutlass.Int32(1))
                        & cutlass.Int32(-2),
                        cutlass.Int32(2),
                    )
                    token_guard_checks_per_route = (
                        2 if self.inspection_q_tile_size == 64 else 4
                    )
                    # Runtime checks = Q-tile reuse * padded slots *
                    # (2 if Q64 else 4). Dummy slots enter only this denominator.
                    runtime_bsr_row_token_guard_check_count = (
                        cutlass.Int64(padded_kv_route_slot_count)
                        * cutlass.Int64(token_guard_checks_per_route)
                        * cutlass.Int64(q_tiles_per_bsr_row)
                    )

        if cutlass.const_expr(self.inspect_token_mask):
            # Reduce four warp contributions in SMEM before contended atomics.
            # Token-mask-free variants omit the block; Q64 does not allocate
            # the SMEM table's third column (token-mask-full route count).
            smem = cutlass.utils.SmemAllocator()
            runtime_count_field_count = 3 if self.count_token_mask_full_kv_routes else 2
            warp_runtime_counts = smem.allocate_tensor(
                cutlass.Int64,
                cute.make_layout((_WARPS_PER_CTA, runtime_count_field_count)),
                byte_alignment=8,
            )
            if lane_idx == 0:
                warp_runtime_counts[warp_idx, 0] = (
                    runtime_bsr_row_token_guard_skip_count
                )
                warp_runtime_counts[warp_idx, 1] = (
                    runtime_bsr_row_token_guard_check_count
                )
                if cutlass.const_expr(self.count_token_mask_full_kv_routes):
                    warp_runtime_counts[warp_idx, 2] = (
                        runtime_bsr_row_token_mask_full_kv_route_count
                    )
            cute.arch.sync_threads()
            if thread_idx == 0:
                cta_runtime_token_guard_skip_count = cutlass.Int64(0)
                cta_runtime_token_guard_check_count = cutlass.Int64(0)
                cta_runtime_token_mask_full_kv_route_count = cutlass.Int64(0)
                for reduce_warp_idx in cutlass.range_constexpr(_WARPS_PER_CTA):
                    cta_runtime_token_guard_skip_count += warp_runtime_counts[
                        reduce_warp_idx, 0
                    ]
                    cta_runtime_token_guard_check_count += warp_runtime_counts[
                        reduce_warp_idx, 1
                    ]
                    if cutlass.const_expr(self.count_token_mask_full_kv_routes):
                        cta_runtime_token_mask_full_kv_route_count += (
                            warp_runtime_counts[reduce_warp_idx, 2]
                        )
                if cta_runtime_token_guard_skip_count > cutlass.Int64(0):
                    cute.arch.atomic_add(
                        summary.iterator + _SUMMARY_RUNTIME_TOKEN_GUARD_SKIP_COUNT,
                        cta_runtime_token_guard_skip_count,
                        sem="relaxed",
                        scope="gpu",
                    )
                if cta_runtime_token_guard_check_count > cutlass.Int64(0):
                    cute.arch.atomic_add(
                        summary.iterator + _SUMMARY_RUNTIME_TOKEN_GUARD_CHECK_COUNT,
                        cta_runtime_token_guard_check_count,
                        sem="relaxed",
                        scope="gpu",
                    )
                if cutlass.const_expr(self.count_token_mask_full_kv_routes):
                    if cta_runtime_token_mask_full_kv_route_count > cutlass.Int64(0):
                        cute.arch.atomic_add(
                            summary.iterator
                            + _SUMMARY_RUNTIME_TOKEN_MASK_FULL_KV_ROUTE_COUNT,
                            cta_runtime_token_mask_full_kv_route_count,
                            sem="relaxed",
                            scope="gpu",
                        )


def _fake_compact(
    dtype: type,
    shape: tuple[object, ...],
    *,
    alignment: int,
) -> cute.Tensor:
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=alignment,
    )


@functools.cache
def compile_block_sparse_inspection(
    *,
    device_index: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
    has_token_bits: bool,
) -> Callable[..., None]:
    """Compile one geometry specialization while keeping ``indices[nnz]`` dynamic.

    Tensor ranks, dtypes, compact strides, and all attention geometry are part
    of the specialization. Only the flat ``block_indices`` extent is symbolic;
    tensor contents may of course vary between calls to the cached function.
    """

    num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
    logical_nnz = cute.sym_int()
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _InspectBlockSparseBsr(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        has_token_bits=has_token_bits,
    )
    with torch.cuda.device(device_index):
        return cute.compile(
            kernel,
            _fake_compact(
                cutlass.Int32,
                (batch_size, num_kv_heads, num_q_block_rows + 1),
                alignment=4,
            ),
            _fake_compact(cutlass.Int32, (logical_nnz,), alignment=4),
            (
                _fake_compact(
                    cutlass.Uint32,
                    (batch_size, (seq_len_kv + 31) // 32),
                    alignment=4,
                )
                if has_token_bits
                else None
            ),
            _fake_compact(cutlass.Int64, (_SUMMARY_FIELDS,), alignment=8),
            stream,
            options=_COMPILE_OPTIONS,
        )


__all__ = ["compile_block_sparse_inspection"]
