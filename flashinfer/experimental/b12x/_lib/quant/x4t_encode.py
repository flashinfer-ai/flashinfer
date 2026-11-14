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

# Upper bound for the temporary memory of one expert chunk (see _chunk_experts).
_CHUNK_BUDGET_BYTES = 256 << 20
# Worst-case temporary bytes per scale: int64 histogram index (8), or at the
# exception stage int64 nonzero pairs (16) + gathered values (1) + int64 words
# (8) + int64 value copy (8) + outlier mask (1), rounded up.
_TEMP_BYTES_PER_SCALE = 40


def _chunk_experts(rows: int, cols: int) -> int:
    """Experts per chunk so that temporaries stay within ``_CHUNK_BUDGET_BYTES``."""

    per_expert = rows * (256 * 4 + cols * _TEMP_BYTES_PER_SCALE)
    return max(1, _CHUNK_BUDGET_BYTES // per_expert)


def _encode_chunk(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Encode ``uint8 [E, rows, cols]``; return CPU fixed, words, per-expert counts."""

    experts, rows, cols = values.shape
    device = values.device
    selector_bytes = (cols + 7) // 8
    flat = values.reshape(experts * rows, cols)

    counts = torch.zeros((experts * rows, 256), dtype=torch.int32, device=device)
    index = flat.to(torch.int64)
    counts.scatter_add_(
        1,
        index,
        torch.ones((1, 1), dtype=torch.int32, device=device).expand(
            experts * rows, cols
        ),
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

    padded = torch.zeros(
        (experts * rows, selector_bytes * 8), dtype=torch.uint8, device=device
    )
    padded[:, :cols] = selected
    del selected
    shifts = torch.arange(8, dtype=torch.uint8, device=device)
    bits = (padded.view(-1, selector_bytes, 8) << shifts).sum(dim=2, dtype=torch.uint8)
    del padded

    tiles = rows // X4T_TILE_ROWS
    fixed = torch.cat(
        (
            bases.to(torch.uint8).view(experts, tiles, X4T_TILE_ROWS),
            bits.view(experts, tiles, X4T_TILE_ROWS * selector_bytes),
        ),
        dim=2,
    )

    outlier_flat = outliers.view(experts, rows * cols)
    hits = outlier_flat.nonzero()  # row-major: sorted by expert, then position
    positions = hits[:, 1]
    expert_values = flat.view(experts, rows * cols)[hits[:, 0], positions]
    words = (expert_values.to(torch.int64) << 24) | positions
    per_expert = outlier_flat.sum(dim=1)
    # One device-to-host copy: per-expert counts followed by the exception words.
    packed = torch.cat((per_expert, words)).cpu()
    split = packed[:experts].tolist()
    return fixed.cpu(), packed[experts:], split


def encode_x4t_scales(
    scales: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Encode E8M0 scale matrices ``[experts, rows, cols]`` into X4T planes.

    ``scales`` is any one-byte dtype (for example ``torch.float8_e8m0fnu`` or
    ``torch.uint8``) whose bytes are the raw scale values. ``rows`` must be a
    multiple of 16, ``1 <= cols <= 255`` and ``rows * cols <= 2**24``. The work
    runs on ``scales.device`` in expert chunks that bound its temporary memory;
    it is a load-time operation and cannot run under CUDA graph capture.

    Each row stores a base byte, the lowest ``b`` in ``0..254`` that maximizes
    ``count(b) + count(b + 1)``. Values equal to ``base + 1`` set a selector bit,
    values equal to ``base`` leave it clear, and all other values become
    exceptions ``(byte << 24) | (row * cols + col)``, sorted by position.

    Returns per-expert CPU ``(fixed, exceptions)`` tuples: ``fixed`` is uint8
    ``[rows // 16, 16 * (1 + ceil(cols / 8))]`` and ``exceptions`` is a
    one-dimensional uint32 tensor; each owns its storage. They match what ``make_x4t_scale_batch`` and
    ``CsfScalePlanes`` expect, and the decoded planes equal ``scales`` exactly.
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

    values = scales.contiguous().view(torch.uint8)
    chunk = _chunk_experts(rows, cols)
    fixed_planes: list[torch.Tensor] = []
    exception_planes: list[torch.Tensor] = []
    for start in range(0, experts, chunk):
        fixed, words, counts = _encode_chunk(values[start : start + chunk])
        # clone(): each plane owns its storage (serializable, frees the chunk).
        fixed_planes.extend(plane.clone() for plane in fixed.unbind(0))
        words = (words & 0xFFFFFFFF).to(torch.uint32)
        exception_planes.extend(w.clone() for w in words.split(counts))
    return tuple(fixed_planes), tuple(exception_planes)


__all__ = ["encode_x4t_scales"]
