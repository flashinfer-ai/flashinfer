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

"""Map BSR blocks onto the KV128 routes consumed by FMHA decode.

BSR indices identify caller-visible blocks of ``kv_block_size`` tokens. Each
decode route consumes two KV64 fragments. When ``kv_block_size / 64`` is odd,
those two fragments may come from adjacent BSR entries; returned origins always
refer to the original, uncompressed KV token coordinates.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32


_KV_FRAGMENT_SIZE = 64
_KV64_FRAGMENTS_PER_ROUTE = 2


@dataclass(frozen=True)
class _BlockSparseFragmentHostResult:
    """Resolved host-side metadata for one 64-token KV fragment."""

    bsr_entry_offset: int
    fragment_in_block: int
    physical_token_offset: int
    valid: bool


def _validate_block_sparse_host_row(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> int:
    """Validate host-only exact-route inputs and return the exclusive row end."""
    if row_begin < 0:
        raise ValueError("row_begin must be non-negative")
    if row_nnz < 0:
        raise ValueError("row_nnz must be non-negative")
    if kv_block_size <= 0 or kv_block_size % _KV_FRAGMENT_SIZE != 0:
        raise ValueError("kv_block_size must be a positive multiple of 64")
    if seq_len_kv <= 0:
        raise ValueError("seq_len_kv must be positive")

    row_end = row_begin + row_nnz
    num_indices = len(block_indices)
    if row_begin > num_indices or row_end > num_indices:
        raise ValueError(
            f"row range [{row_begin}, {row_end}) exceeds block_indices length "
            f"{num_indices}"
        )
    return row_end


def _block_sparse_row_retained_route_count_host(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> int:
    """Return the exact number of KV128 routes retained by the physical tail.

    Canonical BSR rows are sorted and contain only in-range block IDs. Thus all
    entries before the final selected block contribute complete 64-token
    fragments; only the final block can intersect ``seq_len_kv``. The empty-row
    branch deliberately precedes the ``row_end - 1`` index calculation.
    """
    row_end = _validate_block_sparse_host_row(
        block_indices, row_begin, row_nnz, kv_block_size, seq_len_kv
    )
    if row_nnz == 0:
        return 0

    fragments_per_entry = kv_block_size // _KV_FRAGMENT_SIZE
    retained_fragments = (row_nnz - 1) * fragments_per_entry
    last_block_idx = block_indices[row_end - 1]
    last_block_origin = last_block_idx * kv_block_size
    remaining_tokens = seq_len_kv - last_block_origin
    retained_last_fragments = 0
    if remaining_tokens > 0:
        retained_last_fragments = min(
            fragments_per_entry,
            (remaining_tokens - 1) // _KV_FRAGMENT_SIZE + 1,
        )
    retained_fragments += retained_last_fragments
    return (
        retained_fragments + _KV64_FRAGMENTS_PER_ROUTE - 1
    ) // _KV64_FRAGMENTS_PER_ROUTE


@cute.jit
def _block_sparse_row_retained_route_count(
    block_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: Int32,
) -> Int32:
    """Device equivalent of the host exact retained-route count.

    The caller supplies a canonical sorted/in-range BSR row and a positive
    Int32 ``seq_len_kv``. Therefore the final selected block origin and its
    distance to ``seq_len_kv`` are representable as Int32. Empty rows return
    before forming ``row_end - 1``, so they never dereference ``block_indices``.
    """
    row_nnz = row_end - row_begin
    retained_fragments = Int32(0)
    if row_nnz > Int32(0):
        fragments_per_entry = Int32(kv_block_size // _KV_FRAGMENT_SIZE)
        retained_fragments = (row_nnz - Int32(1)) * fragments_per_entry
        last_block_idx = Int32(block_indices[row_end - Int32(1)])
        last_block_origin = last_block_idx * Int32(kv_block_size)
        remaining_tokens = seq_len_kv - last_block_origin
        retained_last_fragments = Int32(0)
        if remaining_tokens > Int32(0):
            retained_last_fragments = (remaining_tokens - Int32(1)) // Int32(
                _KV_FRAGMENT_SIZE
            ) + Int32(1)
            if retained_last_fragments > fragments_per_entry:
                retained_last_fragments = fragments_per_entry
        retained_fragments = retained_fragments + retained_last_fragments
    return (retained_fragments + Int32(_KV64_FRAGMENTS_PER_ROUTE - 1)) // Int32(
        _KV64_FRAGMENTS_PER_ROUTE
    )


def _sparse_keeps_can_skip_structural_mask_host(
    *,
    q_row_is_valid: bool,
    origin0: int,
    origin1: int,
    valid0: bool,
    valid1: bool,
    seq_len_kv: int,
    causal_end: int | None,
) -> bool:
    """Return whether a complete KV128 route needs no structural masking."""
    last_complete_kv_origin = seq_len_kv - _KV_FRAGMENT_SIZE
    can_skip_structural_mask = (
        q_row_is_valid
        and valid0
        and valid1
        and origin0 <= last_complete_kv_origin
        and origin1 <= last_complete_kv_origin
    )
    if causal_end is not None:
        last_causal_origin = causal_end - _KV_FRAGMENT_SIZE
        can_skip_structural_mask = (
            can_skip_structural_mask
            and origin0 <= last_causal_origin
            and origin1 <= last_causal_origin
        )
    return can_skip_structural_mask


@cute.jit
def sparse_keeps_can_skip_structural_mask(
    q_row_is_valid: Boolean,
    origin0: Int32,
    origin1: Int32,
    valid0: Int32,
    valid1: Int32,
    seq_len_kv: Int32,
    causal_end: Int32,
    *,
    apply_causal_mask: cutlass.Constexpr[bool],
) -> Boolean:
    """Return whether one Keeps row can skip structural score masking.

    Both physical 64-token fragments must be complete. Causal and token-bit
    masking are deliberately independent: this helper checks only Q-row,
    fragment, sequence-tail, and optional causal bounds. Limits are positive
    Int32 values, so subtracting the fragment size before comparing avoids
    overflowing when an origin is near the Int32 upper bound.
    """
    fragment_size = Int32(_KV_FRAGMENT_SIZE)
    last_complete_kv_origin = seq_len_kv - fragment_size
    can_skip_structural_mask = Boolean(
        q_row_is_valid
        and valid0 != Int32(0)
        and valid1 != Int32(0)
        and origin0 <= last_complete_kv_origin
        and origin1 <= last_complete_kv_origin
    )
    if cutlass.const_expr(apply_causal_mask):
        last_causal_origin = causal_end - fragment_size
        can_skip_structural_mask = Boolean(
            can_skip_structural_mask
            and origin0 <= last_causal_origin
            and origin1 <= last_causal_origin
        )
    return can_skip_structural_mask


def _resolve_block_sparse_route_host(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    route_idx: int,
    kv_block_size: int,
) -> tuple[_BlockSparseFragmentHostResult, _BlockSparseFragmentHostResult]:
    """Resolve both KV64 fragments in one route without compressed origins.

    ``block_indices`` contains algorithm-block IDs, not route IDs.
    Each valid result therefore converts its selected block back to a physical
    token origin in the original KV sequence. Invalid tail fragments use origin
    zero and never read ``block_indices``.
    """
    if row_begin < 0:
        raise ValueError("row_begin must be non-negative")
    if row_nnz < 0:
        raise ValueError("row_nnz must be non-negative")
    if route_idx < 0:
        raise ValueError("route_idx must be non-negative")
    if kv_block_size <= 0 or kv_block_size % _KV_FRAGMENT_SIZE != 0:
        raise ValueError("kv_block_size must be a positive multiple of 64")

    row_end = row_begin + row_nnz
    num_indices = len(block_indices)
    if row_begin > num_indices or row_end > num_indices:
        raise ValueError(
            f"row range [{row_begin}, {row_end}) exceeds block_indices length "
            f"{num_indices}"
        )

    fragments_per_entry = kv_block_size // _KV_FRAGMENT_SIZE
    results = []
    for fragment_in_route in range(_KV64_FRAGMENTS_PER_ROUTE):
        fragment_idx = route_idx * _KV64_FRAGMENTS_PER_ROUTE + fragment_in_route
        bsr_entry_offset, fragment_in_block = divmod(fragment_idx, fragments_per_entry)
        valid = bsr_entry_offset < row_nnz
        physical_token_offset = 0
        if valid:
            block_idx = block_indices[row_begin + bsr_entry_offset]
            physical_token_offset = (
                block_idx * kv_block_size + fragment_in_block * _KV_FRAGMENT_SIZE
            )
        results.append(
            _BlockSparseFragmentHostResult(
                bsr_entry_offset=bsr_entry_offset,
                fragment_in_block=fragment_in_block,
                physical_token_offset=physical_token_offset,
                valid=valid,
            )
        )

    return results[0], results[1]


def _resolve_block_sparse_route_origins_host(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    route_idx: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> tuple[_BlockSparseFragmentHostResult, _BlockSparseFragmentHostResult]:
    """Resolve one KV128 route and remove fragments beyond the physical tail."""
    if route_idx < 0:
        raise ValueError("route_idx must be non-negative")
    _validate_block_sparse_host_row(
        block_indices, row_begin, row_nnz, kv_block_size, seq_len_kv
    )
    fragments = _resolve_block_sparse_route_host(
        block_indices,
        row_begin,
        row_nnz,
        route_idx,
        kv_block_size,
    )

    def retain_physical_fragment(
        fragment: _BlockSparseFragmentHostResult,
    ) -> _BlockSparseFragmentHostResult:
        valid = fragment.valid and fragment.physical_token_offset < seq_len_kv
        return _BlockSparseFragmentHostResult(
            bsr_entry_offset=fragment.bsr_entry_offset,
            fragment_in_block=fragment.fragment_in_block,
            physical_token_offset=fragment.physical_token_offset if valid else 0,
            valid=valid,
        )

    return (
        retain_physical_fragment(fragments[0]),
        retain_physical_fragment(fragments[1]),
    )


@cute.jit
def resolve_block_sparse_route_origins(
    block_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    route_idx: Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: Int32,
) -> tuple[Int32, Boolean, Int32, Boolean]:
    """Resolve both 64-token fragments of one KV128 route together.

    A same-entry route performs one BSR index load and reuses that block ID.
    The second load is guarded by both structural validity and a cross-entry
    comparison. Invalid structural or physical-tail fragments return origin 0.

    The caller guarantees canonical in-range block IDs and a positive Int32
    ``seq_len_kv``. Hence ``block_idx * kv_block_size`` is representable. The
    fragment offset is compared with the remaining physical extent before the
    final addition, which keeps a partial final block from overflowing Int32.
    """
    fragments_per_entry = Int32(kv_block_size // _KV_FRAGMENT_SIZE)
    fragment_idx0 = route_idx * Int32(_KV64_FRAGMENTS_PER_ROUTE)
    fragment_idx1 = fragment_idx0 + Int32(1)
    bsr_entry_offset0 = fragment_idx0 // fragments_per_entry
    bsr_entry_offset1 = fragment_idx1 // fragments_per_entry
    fragment_in_block0 = fragment_idx0 % fragments_per_entry
    fragment_in_block1 = fragment_idx1 % fragments_per_entry
    row_nnz = row_end - row_begin

    valid0 = Boolean(bsr_entry_offset0 < row_nnz)
    valid1 = Boolean(bsr_entry_offset1 < row_nnz)
    block_idx0 = Int32(0)
    if valid0:
        block_idx0 = Int32(block_indices[row_begin + bsr_entry_offset0])

    block_idx1 = block_idx0
    if valid1 and bsr_entry_offset1 != bsr_entry_offset0:
        block_idx1 = Int32(block_indices[row_begin + bsr_entry_offset1])

    origin0 = Int32(0)
    if valid0:
        block_origin0 = block_idx0 * Int32(kv_block_size)
        fragment_offset0 = fragment_in_block0 * Int32(_KV_FRAGMENT_SIZE)
        remaining_tokens0 = seq_len_kv - block_origin0
        valid0 = Boolean(fragment_offset0 < remaining_tokens0)
        if valid0:
            origin0 = block_origin0 + fragment_offset0

    origin1 = Int32(0)
    if valid1:
        block_origin1 = block_idx1 * Int32(kv_block_size)
        fragment_offset1 = fragment_in_block1 * Int32(_KV_FRAGMENT_SIZE)
        remaining_tokens1 = seq_len_kv - block_origin1
        valid1 = Boolean(fragment_offset1 < remaining_tokens1)
        if valid1:
            origin1 = block_origin1 + fragment_offset1
    return origin0, valid0, origin1, valid1
