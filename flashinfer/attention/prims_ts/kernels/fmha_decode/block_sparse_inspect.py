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
``(batch, kv_head, q_block)``. The inspector reports whether selected real
tokens contain mask holes, but deliberately does not summarize their
morphology: the attention kernel derives route-full state from current token
words on every run. A retained KV route is one KV128 unit left after
physical-tail trimming.

Four warps validate four BSR Q-block rows per CTA and publish four plan-wide
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
)


_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"

# ``summary`` is a zero-initialized Int64[4] shared with the host wrapper.
_SUMMARY_ERROR_CODE = 0
_SUMMARY_MAX_SELECTED_KV_BLOCK_COUNT = 1
_SUMMARY_MAX_RETAINED_KV_ROUTE_COUNT = 2
_SUMMARY_SELECTED_TOKEN_HOLE = 3
_SUMMARY_FIELDS = 4

_BSR_ERROR_NONE = 0
_BSR_ERROR_NOT_STRICTLY_INCREASING = 1
_BSR_ERROR_INDEX_OUT_OF_RANGE = 2
_BSR_ERROR_INVALID_INDPTR = 3


@cute.jit
def _inspect_bsr_row_token_mask_lane(
    block_indices: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Scan one lane stripe of selected token-mask atoms.

    The inspection atom is at most 32 tokens and divides a stored mask word,
    so one source-word load is sufficient. Physical-tail padding is excluded
    from ``expected`` and therefore cannot create a false hole. The scan is
    independent of the execution tile and computes no morphology counters.
    """

    mask_atom_size = min(kv_block_size, 32)
    mask_atoms_per_block = kv_block_size // mask_atom_size
    selected_kv_block_count = bsr_row_end - bsr_row_begin
    selected_mask_atom_count = selected_kv_block_count * cutlass.Int32(
        mask_atoms_per_block
    )
    atom_offset = lane_idx
    lane_has_selected_token_hole = cutlass.Boolean(False)
    while atom_offset < selected_mask_atom_count:
        bsr_entry_offset = atom_offset // cutlass.Int32(mask_atoms_per_block)
        mask_atom_in_block = atom_offset % cutlass.Int32(mask_atoms_per_block)
        block_id = cutlass.Int32(block_indices[bsr_row_begin + bsr_entry_offset])
        atom_origin = block_id * cutlass.Int32(kv_block_size)
        atom_origin += mask_atom_in_block * cutlass.Int32(mask_atom_size)
        if atom_origin < cutlass.Int32(seq_len_kv):
            source_word = cutlass.Uint32(
                kv_valid_bits[batch_idx, atom_origin >> cutlass.Int32(5)]
            )
            bit_offset = atom_origin & cutlass.Int32(31)
            selected_bits = source_word >> bit_offset
            expected = cutlass.Uint32(0xFFFFFFFF)
            if cutlass.const_expr(mask_atom_size < 32):
                expected = cutlass.Uint32((1 << mask_atom_size) - 1)
            remaining_tokens = cutlass.Int32(seq_len_kv) - atom_origin
            if remaining_tokens < cutlass.Int32(mask_atom_size):
                # This branch implies 0 < remaining_tokens < mask_atom_size <= 32,
                # so the dynamic shift is never 32.
                expected = (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
            lane_has_selected_token_hole = cutlass.Boolean(
                lane_has_selected_token_hole or (selected_bits & expected) != expected
            )
        remaining_mask_atoms = selected_mask_atom_count - atom_offset
        atom_offset = (
            atom_offset + _WARP_SIZE
            if remaining_mask_atoms > _WARP_SIZE
            else selected_mask_atom_count
        )
    return lane_has_selected_token_hole


@cute.jit
def _validate_bsr_row_lane(
    block_indices: cute.Tensor,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    num_kv_blocks: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Validate one lane stripe of a canonical ordered BSR row."""

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    selected_kv_block_count = bsr_row_end - bsr_row_begin
    bsr_entry_offset = lane_idx
    while bsr_entry_offset < selected_kv_block_count:
        entry_position = bsr_row_begin + bsr_entry_offset
        block_id = cutlass.Int32(block_indices[entry_position])
        in_range = block_id >= 0 and block_id < num_kv_blocks
        if not in_range:
            error_code = cutlass.Int32(_BSR_ERROR_INDEX_OUT_OF_RANGE)
        else:
            if entry_position > bsr_row_begin:
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
    * ``summary``: zero-initialized Int64 ``[4]`` output.

    One 128-thread CTA contains four independent warps, and each warp handles
    one flattened ``(batch, kv_head, q_block_row)``. Lanes first validate the
    row's index stripe, then inspect its selected real tokens when a token mask
    is present. The output fields are:

    * ``[0]`` highest validation error code (zero means canonical BSR);
    * ``[1]`` maximum selected KV-block count in any row;
    * ``[2]`` maximum retained KV-route count;
    * ``[3]`` whether any selected real token has a zero validity bit;

    No per-route metadata is produced; the attention kernel continues to
    consume the caller's BSR and current token mask directly.
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
        self.inspect_token_mask = has_token_bits
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
        num_indices = cutlass.Int32(cute.size(block_indices))
        error_code = cutlass.Int32(_BSR_ERROR_NONE)
        lane_has_selected_token_hole = cutlass.Boolean(False)

        if bsr_row_is_valid:
            bsr_row_begin = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx]
            )
            bsr_row_end = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
            )
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
            assert kv_valid_bits is not None
            if bsr_row_is_valid and bsr_row_error_code == _BSR_ERROR_NONE:
                lane_has_selected_token_hole = _inspect_bsr_row_token_mask_lane(
                    block_indices,
                    kv_valid_bits,
                    batch_idx,
                    bsr_row_begin,
                    bsr_row_end,
                    lane_idx,
                    self.kv_block_size,
                    self.seq_len_kv,
                )
        bsr_row_has_selected_token_hole = cute.arch.vote_any_sync(
            lane_has_selected_token_hole
        )

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
