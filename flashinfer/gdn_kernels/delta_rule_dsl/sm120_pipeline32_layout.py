# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer team


def workspace_shapes(
    total_tokens: int, num_seqs: int, heads: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return ``(matrix_shape, log_shape)`` for the SM120 pipeline32 workspace.

    Slot indexing uses ``start_slot(s) = cu[s] // 32 + s`` (strictly increasing,
    including empty sequences). Capacity is bounded by
    ``total_tokens // 32 + num_seqs`` when ``cu[num_seqs] == total_tokens``.
    """
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    if num_seqs < 0:
        raise ValueError(f"num_seqs must be non-negative, got {num_seqs}")
    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    capacity = total_tokens // 32 + num_seqs
    if capacity == 0:
        capacity = 1
    return (capacity, heads, 32, 32), (capacity, heads, 32)
