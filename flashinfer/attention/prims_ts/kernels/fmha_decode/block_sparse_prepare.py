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

"""Prepare live canonical BSR rows for the PrimTS FMHA route consumer.

The kernel converts caller-owned semantic KV blocks into fixed-stride route
metadata on every run. The metadata keeps logical KV order: physical placement
for K and V remains the attention kernel's responsibility. One warp handles
one BSR row and iterates only that row's live routes, while four warps share a
CTA.

``row_route_offsets`` is a separate plan-owned immutable Int32 tensor.
``route_workspace`` contains only mutable row counts and route metadata
described by ``_PreparedBlockSparseLayout``. Payload outside each live row
count is intentionally stale. ``max_blocks_per_row`` is a run-time Int32
scalar: ``-1`` disables the semantic bound, while a non-negative value keeps a
declared BSR-block limit distinct from packed-route capacity.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass.experimental import primitives as prims

from ..._block_sparse.prepared import (
    _PREPARED_ROUTE_IS_FULL_FLAG,
    _PreparedBlockSparseLayout,
)
from .fmha_decode_resources.helpers_common import _warp_broadcast_i32


_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE


@cute.jit
def _retained_atom_count(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Count selected atoms whose physical origin precedes ``seq_len_kv``."""

    row_nnz = row_end - row_begin
    retained_atoms = cutlass.Int32(0)
    if row_nnz > cutlass.Int32(0):
        atoms_per_block = kv_block_size // atom_size
        retained_atoms = (row_nnz - cutlass.Int32(1)) * cutlass.Int32(atoms_per_block)
        last_block_idx = cutlass.Int32(block_indices[row_end - cutlass.Int32(1)])
        num_kv_blocks = (seq_len_kv + kv_block_size - 1) // kv_block_size
        retained_last_atoms = cutlass.Int32(atoms_per_block)
        last_block_is_in_range = cutlass.Boolean(
            last_block_idx >= cutlass.Int32(0)
            and last_block_idx < cutlass.Int32(num_kv_blocks)
        )
        if last_block_is_in_range:
            last_block_origin = last_block_idx * cutlass.Int32(kv_block_size)
            remaining_tokens = cutlass.Int32(seq_len_kv) - last_block_origin
            retained_last_atoms = cutlass.Int32(0)
            if remaining_tokens > cutlass.Int32(0):
                retained_last_atoms = (
                    remaining_tokens - cutlass.Int32(1)
                ) // cutlass.Int32(atom_size) + cutlass.Int32(1)
                if retained_last_atoms > cutlass.Int32(atoms_per_block):
                    retained_last_atoms = cutlass.Int32(atoms_per_block)
        retained_atoms = retained_atoms + retained_last_atoms
    return retained_atoms


@cute.jit
def _resolve_route_atom(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    route_idx: cutlass.Int32,
    atom_in_route: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    origins_per_route: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Boolean]:
    """Resolve one logical atom to its physical token origin."""

    atoms_per_block = kv_block_size // atom_size
    flat_atom_idx = route_idx * cutlass.Int32(origins_per_route) + atom_in_route
    bsr_entry_offset = flat_atom_idx // cutlass.Int32(atoms_per_block)
    atom_in_block = flat_atom_idx % cutlass.Int32(atoms_per_block)
    valid = cutlass.Boolean(bsr_entry_offset < row_end - row_begin)
    origin = cutlass.Int32(-1)
    if valid:
        block_idx = cutlass.Int32(block_indices[row_begin + bsr_entry_offset])
        num_kv_blocks = (seq_len_kv + kv_block_size - 1) // kv_block_size
        valid = cutlass.Boolean(
            block_idx >= cutlass.Int32(0)
            and block_idx < cutlass.Int32(num_kv_blocks)
        )
        if valid:
            block_origin = block_idx * cutlass.Int32(kv_block_size)
            atom_offset = atom_in_block * cutlass.Int32(atom_size)
            valid = cutlass.Boolean(
                atom_offset < cutlass.Int32(seq_len_kv) - block_origin
            )
            if valid:
                origin = block_origin + atom_offset
    return origin, valid


@cute.jit
def _load_coarse_token_word(
    block_indices: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    route_idx: cutlass.Int32,
    logical_word_idx: cutlass.Int32,
    batch_idx: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    origins_per_route: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
    num_kv_valid_words: cutlass.Constexpr[int],
) -> cutlass.Uint32:
    """Load one logical K32 word from a coarse atom larger than K32."""

    logical_word = cutlass.Uint32(0)
    words_per_atom = atom_size // 32
    atom_in_route = logical_word_idx // cutlass.Int32(words_per_atom)
    word_in_atom = logical_word_idx % cutlass.Int32(words_per_atom)
    origin, valid = _resolve_route_atom(
        block_indices,
        row_begin,
        row_end,
        route_idx,
        atom_in_route,
        kv_block_size,
        atom_size,
        origins_per_route,
        seq_len_kv,
    )
    physical_word_origin = origin + word_in_atom * cutlass.Int32(32)
    if (
        valid
        and physical_word_origin >= cutlass.Int32(0)
        and physical_word_origin < cutlass.Int32(seq_len_kv)
    ):
        physical_word_idx = physical_word_origin >> cutlass.Int32(5)
        if (
            physical_word_idx >= cutlass.Int32(0)
            and physical_word_idx < cutlass.Int32(num_kv_valid_words)
        ):
            logical_word = cutlass.Uint32(
                kv_valid_bits[batch_idx, physical_word_idx]
            )
            remaining_tokens = cutlass.Int32(seq_len_kv) - physical_word_origin
            if remaining_tokens < cutlass.Int32(32):
                logical_word = logical_word & (
                    (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
                )
    return logical_word


@cute.jit
def _load_atom_token_chunk(
    kv_valid_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    origin: cutlass.Int32,
    origin_is_valid: cutlass.Boolean,
    atom_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Constexpr[int],
    num_kv_valid_words: cutlass.Constexpr[int],
) -> cutlass.Uint32:
    """Load the <=K32 mask chunk owned by one resolved-origin lane."""

    token_chunk = cutlass.Uint32(0)
    if origin_is_valid and origin >= cutlass.Int32(0):
        physical_word_idx = origin >> cutlass.Int32(5)
        if (
            physical_word_idx >= cutlass.Int32(0)
            and physical_word_idx < cutlass.Int32(num_kv_valid_words)
        ):
            source_word = cutlass.Uint32(
                kv_valid_bits[batch_idx, physical_word_idx]
            )
            token_chunk = source_word >> (origin & cutlass.Int32(31))
            token_chunk = token_chunk & cutlass.Uint32((1 << atom_size) - 1)
            remaining_tokens = cutlass.Int32(seq_len_kv) - origin
            if remaining_tokens < cutlass.Int32(atom_size):
                token_chunk = token_chunk & (
                    (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
                )
    return token_chunk


class _PrepareBlockSparseRoutes:
    """CuTe DSL launcher for one static prepared-route metadata geometry."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        kv_route_size: int,
        has_token_bits: bool,
    ) -> None:
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.seq_len_kv = seq_len_kv
        self.kv_block_size = kv_block_size
        self.has_token_bits = has_token_bits
        self.num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        self.num_rows = batch_size * num_kv_heads * self.num_q_block_rows
        layout = _PreparedBlockSparseLayout.create(
            kv_route_size=kv_route_size,
            kv_block_size=kv_block_size,
            has_token_bits=has_token_bits,
            route_metadata_capacity=0,
            num_rows=self.num_rows,
        )
        self.atom_size = layout.atom_size
        self.origins_per_route = layout.origins_per_route
        self.token_words_per_route = layout.token_words_per_route
        self.atom_valid_mask_word_offset = layout.atom_valid_mask_word_offset
        self.route_flags_word_offset = layout.route_flags_word_offset
        self.token_words_word_offset = layout.token_words_word_offset
        self.route_metadata_stride_words = layout.route_metadata_stride_words
        self.route_metadata_base_word_offset = (
            layout.route_metadata_base_word_offset
        )
        self.num_kv_valid_words = (seq_len_kv + 31) // 32
        capacity_gcd = math.gcd(kv_route_size, kv_block_size)
        # floor(route_capacity * route_size / block_size), reduced first so
        # the Int32 lane-zero multiplication stays within the validated
        # workspace-address envelope.
        self.route_capacity_block_scale_num = kv_route_size // capacity_gcd
        self.route_capacity_block_scale_den = kv_block_size // capacity_gcd

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: Optional[cute.Tensor],
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        stream: cuda_drv.CUstream,
    ) -> None:
        """Launch four independent row preparers per CTA on ``stream``."""

        self.kernel(
            block_indptr,
            block_indices,
            kv_valid_bits,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
        ).launch(
            grid=[(self.num_rows + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA, 1, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: Optional[cute.Tensor],
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
    ) -> None:
        """Rewrite live counts and route metadata while preserving capacities."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE
        linear_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        row_is_valid = linear_row_idx < self.num_rows

        row_begin = cutlass.Int32(0)
        row_end = cutlass.Int32(0)
        batch_idx = cutlass.Int32(0)
        row_range_is_valid = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0) and row_is_valid:
            q_block_row_idx = linear_row_idx % self.num_q_block_rows
            linear_batch_head_idx = linear_row_idx // self.num_q_block_rows
            kv_head_idx = linear_batch_head_idx % self.num_kv_heads
            batch_idx = linear_batch_head_idx // self.num_kv_heads
            row_begin = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx]
            )
            row_end = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
            )
            num_indices = cutlass.Int32(cute.size(block_indices))
            row_range_is_valid = cutlass.Int32(
                row_begin >= cutlass.Int32(0)
                and row_begin <= row_end
                and row_end <= num_indices
            )
        row_begin = _warp_broadcast_i32(row_begin, 0)
        row_end = _warp_broadcast_i32(row_end, 0)
        batch_idx = _warp_broadcast_i32(batch_idx, 0)
        row_range_is_valid = _warp_broadcast_i32(row_range_is_valid, 0)

        route_count = cutlass.Int32(0)
        row_route_begin = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0) and row_is_valid:
            row_route_begin = cutlass.Int32(row_route_offsets[linear_row_idx])
            row_route_end = cutlass.Int32(
                row_route_offsets[linear_row_idx + cutlass.Int32(1)]
            )
            row_route_capacity = row_route_end - row_route_begin
            stored_route_count = cutlass.Int32(-1)
            if row_range_is_valid != cutlass.Int32(0):
                selected_block_count = row_end - row_begin
                route_block_capacity = (
                    row_route_capacity
                    * cutlass.Int32(self.route_capacity_block_scale_num)
                ) // cutlass.Int32(self.route_capacity_block_scale_den)
                block_count_fits = cutlass.Boolean(
                    selected_block_count <= route_block_capacity
                    and (
                        max_blocks_per_row < cutlass.Int32(0)
                        or selected_block_count <= max_blocks_per_row
                    )
                )
                if block_count_fits:
                    retained_atom_count = _retained_atom_count(
                        block_indices,
                        row_begin,
                        row_end,
                        self.kv_block_size,
                        self.atom_size,
                        self.seq_len_kv,
                    )
                    required_route_count = (
                        retained_atom_count
                        + cutlass.Int32(self.origins_per_route - 1)
                    ) // cutlass.Int32(self.origins_per_route)
                    if required_route_count <= row_route_capacity:
                        route_count = required_route_count
                        stored_route_count = required_route_count
                    else:
                        # Negative row headers make a capacity violation visible
                        # after synchronization while the attention consumer
                        # clamps it to an empty row and never reads out of bounds.
                        stored_route_count = -required_route_count
                else:
                    # Preserve the public semantic-block bound even when
                    # multiple small blocks pack into one prepared route.
                    stored_route_count = -selected_block_count
            route_workspace[linear_row_idx] = stored_route_count
        route_count = _warp_broadcast_i32(route_count, 0)
        row_route_begin = _warp_broadcast_i32(row_route_begin, 0)

        route_idx = cutlass.Int32(0)
        while route_idx < route_count:
            route_ordinal = row_route_begin + route_idx
            route_metadata_word_index = cutlass.Int32(
                self.route_metadata_base_word_offset
            ) + route_ordinal * cutlass.Int32(self.route_metadata_stride_words)
            origin = cutlass.Int32(-1)
            origin_is_valid = cutlass.Boolean(False)
            atom_is_full = cutlass.Boolean(False)
            if lane_idx < cutlass.Int32(self.origins_per_route):
                origin, origin_is_valid = _resolve_route_atom(
                    block_indices,
                    row_begin,
                    row_end,
                    route_idx,
                    lane_idx,
                    self.kv_block_size,
                    self.atom_size,
                    self.origins_per_route,
                    self.seq_len_kv,
                )
                if origin_is_valid:
                    atom_is_full = cutlass.Boolean(
                        origin
                        <= cutlass.Int32(self.seq_len_kv)
                        - cutlass.Int32(self.atom_size)
                    )
                route_workspace[route_metadata_word_index + lane_idx] = (
                    origin
                )

            origin_valid_mask = cutlass.Int32(
                cute.arch.vote_ballot_sync(origin_is_valid)
            )
            structural_route_is_full = cute.arch.vote_all_sync(
                lane_idx >= cutlass.Int32(self.origins_per_route) or atom_is_full
            )
            route_is_full = structural_route_is_full
            if cutlass.const_expr(self.has_token_bits):
                assert kv_valid_bits is not None
                assert self.token_words_word_offset is not None
                token_word = cutlass.Uint32(0)
                if cutlass.const_expr(self.atom_size <= 32):
                    # Reuse each atom lane's origin instead of resolving and
                    # loading the same BSR entry again from the word lanes.
                    token_chunk = _load_atom_token_chunk(
                        kv_valid_bits,
                        batch_idx,
                        origin,
                        origin_is_valid,
                        self.atom_size,
                        self.seq_len_kv,
                        self.num_kv_valid_words,
                    )
                    atoms_per_word = 32 // self.atom_size
                    if lane_idx < cutlass.Int32(self.origins_per_route):
                        atom_in_word = lane_idx % cutlass.Int32(atoms_per_word)
                        token_word = token_chunk << (
                            atom_in_word * cutlass.Int32(self.atom_size)
                        )
                        # Active lanes form an aligned power-of-two prefix, so
                        # butterfly peers stay inside their logical K32 group.
                        active_origin_lanes = (1 << self.origins_per_route) - 1
                        for shuffle_step in cutlass.range_constexpr(
                            int(math.log2(atoms_per_word))
                        ):
                            peer_word = cutlass.Uint32(
                                prims.shfl_sync(
                                    thread_mask=active_origin_lanes,
                                    val=token_word,
                                    offset=1 << shuffle_step,
                                    mask_and_clamp=0x1F,
                                    kind=prims.Shfl.BFLY,
                                )
                            )
                            token_word = token_word | peer_word
                        if atom_in_word == cutlass.Int32(0):
                            logical_word_idx = lane_idx // cutlass.Int32(
                                atoms_per_word
                            )
                            route_workspace[
                                route_metadata_word_index
                                + cutlass.Int32(self.token_words_word_offset)
                                + logical_word_idx
                            ] = cutlass.Int32(token_word)
                    full_atom_mask = cutlass.Uint32((1 << self.atom_size) - 1)
                    token_route_is_full = cute.arch.vote_all_sync(
                        lane_idx >= cutlass.Int32(self.origins_per_route)
                        or token_chunk == full_atom_mask
                    )
                else:
                    if lane_idx < cutlass.Int32(self.token_words_per_route):
                        token_word = _load_coarse_token_word(
                            block_indices,
                            kv_valid_bits,
                            row_begin,
                            row_end,
                            route_idx,
                            lane_idx,
                            batch_idx,
                            self.kv_block_size,
                            self.atom_size,
                            self.origins_per_route,
                            self.seq_len_kv,
                            self.num_kv_valid_words,
                        )
                        route_workspace[
                            route_metadata_word_index
                            + cutlass.Int32(self.token_words_word_offset)
                            + lane_idx
                        ] = cutlass.Int32(token_word)
                    token_route_is_full = cute.arch.vote_all_sync(
                        lane_idx >= cutlass.Int32(self.token_words_per_route)
                        or token_word == cutlass.Uint32(0xFFFFFFFF)
                    )
                route_is_full = cutlass.Boolean(
                    structural_route_is_full and token_route_is_full
                )

            if lane_idx == cutlass.Int32(0):
                route_workspace[
                    route_metadata_word_index
                    + cutlass.Int32(self.atom_valid_mask_word_offset)
                ] = origin_valid_mask
                route_workspace[
                    route_metadata_word_index
                    + cutlass.Int32(self.route_flags_word_offset)
                ] = (
                    cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
                    if route_is_full
                    else cutlass.Int32(0)
                )
            route_idx = route_idx + cutlass.Int32(1)
