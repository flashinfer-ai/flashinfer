"""Lossless encoder for X4T (MXFP4-CSF) compressed UE8M0 scale planes.

The planes produced here are the canonical components consumed by
``make_x4t_scale_batch`` and ``CsfScalePlanes``: per 16-row slab, one base byte
per row followed by one selector bit per scale, plus a sorted uint32 stream of
the scales that are neither the base nor base + 1.

This module is torch-only so that it imports and runs without CUTLASS or CUDA.
"""

from __future__ import annotations

import torch

X4T_TILE_ROWS = 16
X4T_POSITION_MASK = (1 << 24) - 1

# Upper bound for the temporary memory of one row slab (see _slab_rows). It
# covers the per-row histograms and the per-scale temporaries, not the input
# or the returned planes.
_SLAB_BUDGET_BYTES = 256 << 20
# Per row: int32 ``counts`` (256 bins) and ``window`` (255 bins) are alive
# together while the base is chosen.
_HISTOGRAM_BYTES_PER_ROW = (256 + 255) * 4
# Worst-case temporary bytes per scale: int64 histogram index (8), or at the
# exception stage int64 nonzero pairs (16) + gathered values (1) + int64 words
# (8) + int64 value copy (8) + outlier mask (1), rounded up.
_TEMP_BYTES_PER_SCALE = 40


def _slab_rows(rows: int, cols: int) -> int:
    """Rows per slab so that temporaries stay within ``_SLAB_BUDGET_BYTES``.

    The result is a multiple of 16 (at least 16), so slabs of the flattened
    ``[experts * rows, cols]`` row space hold whole 16-row tiles.
    """

    per_row = _HISTOGRAM_BYTES_PER_ROW + cols * _TEMP_BYTES_PER_SCALE
    slab = _SLAB_BUDGET_BYTES // per_row
    return max(X4T_TILE_ROWS, slab // X4T_TILE_ROWS * X4T_TILE_ROWS)


def _encode_slab(
    flat: torch.Tensor, first_row: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    """Encode ``uint8 [n, cols]`` global rows ``first_row ..``.

    ``n`` is a multiple of 16 and ``first_row`` is a multiple of 16. Returns the
    CPU fixed tiles ``[n // 16, tile_bytes]``, the CPU int64 exception words,
    per-expert exception counts, and the first expert covered by the slab.
    """

    n, cols = flat.shape
    device = flat.device
    selector_bytes = (cols + 7) // 8

    counts = torch.zeros((n, 256), dtype=torch.int32, device=device)
    index = flat.to(torch.int64)
    counts.scatter_add_(
        1,
        index,
        torch.ones((1, 1), dtype=torch.int32, device=device).expand(n, cols),
    )
    del index
    window = counts[:, :255] + counts[:, 1:]
    # torch.argmax returns the first maximal index, i.e. the lowest base wins ties.
    bases = window.argmax(dim=1, keepdim=True)
    del counts, window

    offsets = flat.to(torch.int16) - bases.to(torch.int16)
    selected = offsets == 1
    outliers = (offsets != 0) & ~selected
    del offsets

    padded = torch.zeros((n, selector_bytes * 8), dtype=torch.uint8, device=device)
    padded[:, :cols] = selected
    del selected
    shifts = torch.arange(8, dtype=torch.uint8, device=device)
    bits = (padded.view(-1, selector_bytes, 8) << shifts).sum(dim=2, dtype=torch.uint8)
    del padded

    tiles = n // X4T_TILE_ROWS
    fixed = torch.cat(
        (
            bases.to(torch.uint8).view(tiles, X4T_TILE_ROWS),
            bits.view(tiles, X4T_TILE_ROWS * selector_bytes),
        ),
        dim=1,
    )

    hits = outliers.nonzero()  # row-major: sorted by global row, then column
    del outliers
    local_rows, hit_cols = hits[:, 0], hits[:, 1]
    values = flat[local_rows, hit_cols].to(torch.int64)
    global_rows = local_rows + first_row
    experts = global_rows // rows
    positions = (global_rows - experts * rows) * cols + hit_cols
    words = (values << 24) | positions
    first_expert = first_row // rows
    last_expert = (first_row + n - 1) // rows
    per_expert = torch.bincount(
        experts - first_expert, minlength=last_expert - first_expert + 1
    )
    # One device-to-host copy: per-expert counts followed by the exception words.
    packed = torch.cat((per_expert, words)).cpu()
    span = last_expert - first_expert + 1
    return fixed.cpu(), packed[span:], packed[:span].tolist(), first_expert


def encode_x4t_scales(
    scales: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Encode E8M0 scale matrices ``[experts, rows, cols]`` into X4T planes.

    ``scales`` is any one-byte dtype (for example ``torch.float8_e8m0fnu`` or
    ``torch.uint8``) whose bytes are the raw scale values. ``rows`` must be a
    multiple of 16, ``1 <= cols <= 255`` and ``rows * cols <= 2**24``. The work
    runs on ``scales.device`` in row slabs that bound its temporary memory, also
    for a single large expert. It is a load-time operation and cannot run under
    CUDA graph capture.

    Each row stores a base byte, the lowest ``b`` in ``0..254`` that maximizes
    ``count(b) + count(b + 1)``. Values equal to ``base + 1`` set a selector bit,
    values equal to ``base`` leave it clear, and all other values become
    exceptions ``(byte << 24) | (row * cols + col)``, sorted by position.

    Returns per-expert CPU ``(fixed, exceptions)`` tuples: ``fixed`` is uint8
    ``[rows // 16, 16 * (1 + ceil(cols / 8))]`` and ``exceptions`` is a
    one-dimensional uint32 tensor. Each plane owns its storage. They match what
    ``make_x4t_scale_batch`` and ``CsfScalePlanes`` expect, and the decoded
    planes equal ``scales`` exactly.
    """

    if not isinstance(scales, torch.Tensor):
        raise TypeError("X4T scales must be a torch.Tensor")
    if scales.dtype.itemsize != 1:
        raise TypeError(f"X4T scales must be a one-byte dtype, got {scales.dtype}")
    if scales.ndim != 3:
        raise ValueError("X4T scales must have shape [experts, rows, cols]")
    experts, rows, cols = (int(dim) for dim in scales.shape)
    if experts <= 0:
        raise ValueError("X4T scales must contain at least one expert")
    if rows <= 0 or rows % X4T_TILE_ROWS:
        raise ValueError("X4T rows must be a positive multiple of 16")
    if not 1 <= cols <= 255:
        raise ValueError("X4T columns must lie in 1..255")
    if rows * cols > X4T_POSITION_MASK + 1:
        raise ValueError("X4T rows * columns must not exceed 2**24")
    if scales.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("encode_x4t_scales cannot run during CUDA graph capture")

    flat = scales.contiguous().view(torch.uint8).reshape(experts * rows, cols)
    slab = _slab_rows(rows, cols)
    fixed_slabs: list[torch.Tensor] = []
    word_slabs: list[torch.Tensor] = []
    exception_counts = [0] * experts
    for start in range(0, experts * rows, slab):
        fixed, words, counts, first_expert = _encode_slab(
            flat[start : start + slab], start, rows
        )
        fixed_slabs.append(fixed)
        word_slabs.append((words & 0xFFFFFFFF).to(torch.uint32))
        for offset, count in enumerate(counts):
            exception_counts[first_expert + offset] += count
    fixed_all = fixed_slabs[0] if len(fixed_slabs) == 1 else torch.cat(fixed_slabs)
    words_all = word_slabs[0] if len(word_slabs) == 1 else torch.cat(word_slabs)
    del fixed_slabs, word_slabs
    tiles = rows // X4T_TILE_ROWS
    # clone(): each plane owns its storage (serializable, independent lifetimes).
    fixed_planes = tuple(
        fixed_all[e * tiles : (e + 1) * tiles].clone() for e in range(experts)
    )
    exception_planes = tuple(w.clone() for w in words_all.split(exception_counts))
    return fixed_planes, exception_planes


__all__ = ["encode_x4t_scales"]
