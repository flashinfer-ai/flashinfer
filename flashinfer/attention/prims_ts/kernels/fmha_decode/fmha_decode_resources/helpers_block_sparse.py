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

BSR indices identify caller-visible blocks of ``kv_block_size`` tokens. A
route is assembled from 8/16/32-token atoms for fine blocks and KV64 atoms for
coarse blocks. Returned origins always refer to the original, uncompressed KV
token coordinates.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32

from ...._block_sparse.common import (
    _KV_ROUTE_SIZE,
    _MAX_KV_ATOM_SIZE,
    _block_sparse_kv_atom_size,
    _block_sparse_kv_routes_are_block_aligned,
)


_KEEPS_ATOM_SIZE = _MAX_KV_ATOM_SIZE
_KEEPS_ATOMS_PER_ROUTE = _KV_ROUTE_SIZE // _KEEPS_ATOM_SIZE


@dataclass(frozen=True)
class _BlockSparseAtomHostResult:
    """Resolved host-side metadata for one KV atom in an execution route."""

    bsr_entry_offset: int
    atom_in_block: int
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
    _block_sparse_kv_atom_size(kv_block_size)
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
    entries before the final selected block contribute complete atoms; only
    the final block can intersect ``seq_len_kv``. The empty-row
    branch deliberately precedes the ``row_end - 1`` index calculation.
    """
    row_end = _validate_block_sparse_host_row(
        block_indices, row_begin, row_nnz, kv_block_size, seq_len_kv
    )
    if row_nnz == 0:
        return 0

    last_block_idx = block_indices[row_end - 1]
    last_block_origin = last_block_idx * kv_block_size
    remaining_tokens = seq_len_kv - last_block_origin
    if _block_sparse_kv_routes_are_block_aligned(kv_block_size):
        # B128/B256/... map each semantic block directly to whole KV128
        # execution tiles. A final partial tile stays one KV128 TMA and relies
        # on hardware OOB fill, so KV64 metadata fragments do not enter this
        # scheduler count.
        routes_per_entry = kv_block_size // _KV_ROUTE_SIZE
        retained_routes = (row_nnz - 1) * routes_per_entry
        if remaining_tokens > 0:
            retained_routes += min(
                routes_per_entry,
                (remaining_tokens + _KV_ROUTE_SIZE - 1) // _KV_ROUTE_SIZE,
            )
        return retained_routes

    atom_size = _block_sparse_kv_atom_size(kv_block_size)
    atoms_per_entry = kv_block_size // atom_size
    atoms_per_route = _KV_ROUTE_SIZE // atom_size
    retained_atoms = (row_nnz - 1) * atoms_per_entry
    retained_last_atoms = 0
    if remaining_tokens > 0:
        retained_last_atoms = min(
            atoms_per_entry,
            (remaining_tokens - 1) // atom_size + 1,
        )
    retained_atoms += retained_last_atoms
    return (retained_atoms + atoms_per_route - 1) // atoms_per_route


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
    retained_routes = Int32(0)
    if row_nnz > Int32(0):
        last_block_idx = Int32(block_indices[row_end - Int32(1)])
        last_block_origin = last_block_idx * Int32(kv_block_size)
        remaining_tokens = seq_len_kv - last_block_origin
        if cutlass.const_expr(_block_sparse_kv_routes_are_block_aligned(kv_block_size)):
            routes_per_entry = Int32(kv_block_size // _KV_ROUTE_SIZE)
            retained_routes = (row_nnz - Int32(1)) * routes_per_entry
            if remaining_tokens > Int32(0):
                retained_last_routes = (remaining_tokens - Int32(1)) // Int32(
                    _KV_ROUTE_SIZE
                ) + Int32(1)
                if retained_last_routes > routes_per_entry:
                    retained_last_routes = routes_per_entry
                retained_routes = retained_routes + retained_last_routes
        else:
            atom_size = _block_sparse_kv_atom_size(kv_block_size)
            atoms_per_entry = Int32(kv_block_size // atom_size)
            atoms_per_route = Int32(_KV_ROUTE_SIZE // atom_size)
            retained_atoms = (row_nnz - Int32(1)) * atoms_per_entry
            if remaining_tokens > Int32(0):
                retained_last_atoms = (remaining_tokens - Int32(1)) // Int32(
                    atom_size
                ) + Int32(1)
                if retained_last_atoms > atoms_per_entry:
                    retained_last_atoms = atoms_per_entry
                retained_atoms = retained_atoms + retained_last_atoms
            retained_routes = (
                retained_atoms + atoms_per_route - Int32(1)
            ) // atoms_per_route
    return retained_routes


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
    last_complete_kv_origin = seq_len_kv - _KEEPS_ATOM_SIZE
    can_skip_structural_mask = (
        q_row_is_valid
        and valid0
        and valid1
        and origin0 <= last_complete_kv_origin
        and origin1 <= last_complete_kv_origin
    )
    if causal_end is not None:
        last_causal_origin = causal_end - _KEEPS_ATOM_SIZE
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
    fragment_size = Int32(_KEEPS_ATOM_SIZE)
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


def _resolve_block_sparse_route_origins_host(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    route_idx: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> tuple[_BlockSparseAtomHostResult, ...]:
    """Resolve every atom in one KV128 route in physical token coordinates."""
    if route_idx < 0:
        raise ValueError("route_idx must be non-negative")
    _validate_block_sparse_host_row(
        block_indices, row_begin, row_nnz, kv_block_size, seq_len_kv
    )
    atom_size = _block_sparse_kv_atom_size(kv_block_size)
    atoms_per_block = kv_block_size // atom_size
    atoms_per_route = _KV_ROUTE_SIZE // atom_size
    results = []
    for atom_in_route in range(atoms_per_route):
        flat_atom = route_idx * atoms_per_route + atom_in_route
        bsr_entry_offset, atom_in_block = divmod(flat_atom, atoms_per_block)
        valid = bsr_entry_offset < row_nnz
        physical_token_offset = 0
        if valid:
            block_idx = block_indices[row_begin + bsr_entry_offset]
            candidate = block_idx * kv_block_size + atom_in_block * atom_size
            valid = candidate < seq_len_kv
            if valid:
                physical_token_offset = candidate
        results.append(
            _BlockSparseAtomHostResult(
                bsr_entry_offset=bsr_entry_offset,
                atom_in_block=atom_in_block,
                physical_token_offset=physical_token_offset,
                valid=valid,
            )
        )
    return tuple(results)


def _resolve_block_sparse_aligned_route_origin_host(
    block_indices: Sequence[int],
    row_begin: int,
    row_nnz: int,
    route_idx: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> tuple[int, bool]:
    """Resolve one block-aligned KV128 route from canonical BSR."""

    if route_idx < 0:
        raise ValueError("route_idx must be non-negative")
    _validate_block_sparse_host_row(
        block_indices, row_begin, row_nnz, kv_block_size, seq_len_kv
    )
    if not _block_sparse_kv_routes_are_block_aligned(kv_block_size):
        raise ValueError("aligned route resolution requires kv_block_size % 128 == 0")

    routes_per_entry = kv_block_size // _KV_ROUTE_SIZE
    bsr_entry_offset, route_in_block = divmod(route_idx, routes_per_entry)
    if bsr_entry_offset >= row_nnz:
        return 0, False

    block_idx = block_indices[row_begin + bsr_entry_offset]
    block_origin = block_idx * kv_block_size
    route_offset = route_in_block * _KV_ROUTE_SIZE
    if route_offset >= seq_len_kv - block_origin:
        return 0, False
    return block_origin + route_offset, True


@cute.jit
def resolve_block_sparse_aligned_route_origin(
    block_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    route_idx: Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: Int32,
) -> tuple[Int32, Boolean]:
    """Resolve one KV128 route that cannot cross a semantic BSR block."""

    routes_per_entry = Int32(kv_block_size // _KV_ROUTE_SIZE)
    bsr_entry_offset = route_idx // routes_per_entry
    route_in_block = route_idx % routes_per_entry
    valid = Boolean(bsr_entry_offset < row_end - row_begin)
    origin = Int32(0)
    if valid:
        block_idx = Int32(block_indices[row_begin + bsr_entry_offset])
        block_origin = block_idx * Int32(kv_block_size)
        route_offset = route_in_block * Int32(_KV_ROUTE_SIZE)
        valid = Boolean(route_offset < seq_len_kv - block_origin)
        if valid:
            origin = block_origin + route_offset
    return origin, valid


@cute.jit
def resolve_block_sparse_route_atom_origin(
    block_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    route_idx: Int32,
    atom_in_route: Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: Int32,
) -> tuple[Int32, Boolean]:
    """Resolve one runtime atom slot with compile-time route geometry.

    ``atom_in_route`` is lane-derived and must be in
    ``[0, KV128 / atom_size)``. Keeping it runtime lets the metadata producer
    distribute B8's 16 atoms across lanes without unrolling 16 resolver calls.
    """
    atom_size = _block_sparse_kv_atom_size(kv_block_size)
    atoms_per_block = kv_block_size // atom_size
    atoms_per_route = _KV_ROUTE_SIZE // atom_size
    flat_atom = route_idx * Int32(atoms_per_route) + atom_in_route
    bsr_entry_offset = flat_atom // Int32(atoms_per_block)
    atom_in_block = flat_atom % Int32(atoms_per_block)
    valid = Boolean(bsr_entry_offset < row_end - row_begin)
    origin = Int32(0)
    if valid:
        block_idx = Int32(block_indices[row_begin + bsr_entry_offset])
        block_origin = block_idx * Int32(kv_block_size)
        atom_offset = atom_in_block * Int32(atom_size)
        valid = Boolean(atom_offset < seq_len_kv - block_origin)
        if valid:
            origin = block_origin + atom_offset
    return origin, valid


@cute.jit
def resolve_block_sparse_coarse_route_fragments(
    block_indices: cute.Pointer,
    row_begin: Int32,
    row_end: Int32,
    route_idx: Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: Int32,
) -> tuple[Int32, Boolean, Int32, Boolean]:
    """Resolve both KV64 fragments of one non-aligned coarse route.

    B64/B192/... routes may cross BSR entries. A same-entry route performs one
    index load and reuses that block ID; the second load is guarded by both
    structural validity and a cross-entry comparison. Block-aligned
    B128/B256/... routes use ``resolve_block_sparse_aligned_route_origin``
    instead. Invalid structural or physical-tail fragments return origin 0.

    The caller guarantees canonical in-range block IDs and a positive Int32
    ``seq_len_kv``. Hence ``block_idx * kv_block_size`` is representable. The
    atom offset is compared with the remaining physical extent before the
    final addition, which keeps a partial final block from overflowing Int32.
    """
    # Keeps profiles always use coarse blocks, so the canonical atom is KV64.
    # Keep resolving the pair together to reuse one BSR load for same-entry
    # routes instead of materializing the generic per-atom representation.
    atoms_per_entry = Int32(kv_block_size // _KEEPS_ATOM_SIZE)
    atom_idx0 = route_idx * Int32(_KEEPS_ATOMS_PER_ROUTE)
    atom_idx1 = atom_idx0 + Int32(1)
    bsr_entry_offset0 = atom_idx0 // atoms_per_entry
    bsr_entry_offset1 = atom_idx1 // atoms_per_entry
    atom_in_block0 = atom_idx0 % atoms_per_entry
    atom_in_block1 = atom_idx1 % atoms_per_entry
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
        atom_offset0 = atom_in_block0 * Int32(_KEEPS_ATOM_SIZE)
        remaining_tokens0 = seq_len_kv - block_origin0
        valid0 = Boolean(atom_offset0 < remaining_tokens0)
        if valid0:
            origin0 = block_origin0 + atom_offset0

    origin1 = Int32(0)
    if valid1:
        block_origin1 = block_idx1 * Int32(kv_block_size)
        atom_offset1 = atom_in_block1 * Int32(_KEEPS_ATOM_SIZE)
        remaining_tokens1 = seq_len_kv - block_origin1
        valid1 = Boolean(atom_offset1 < remaining_tokens1)
        if valid1:
            origin1 = block_origin1 + atom_offset1
    return origin0, valid0, origin1, valid1
