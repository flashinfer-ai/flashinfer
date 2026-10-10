"""CPU partitioning of checkpoint NVFP4 byte-window scale planes."""

from __future__ import annotations

import torch


def slice_nvfp4_csf_plane(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    *,
    rows: int,
    columns: int,
    row_range: tuple[int, int] | None = None,
    column_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice compressed scales without decoding their values.

    The checkpoint codec stores one base byte per row and packed four-bit
    offsets in 16-row slabs. Exception words contain an eight-bit value and a
    24-bit row-major position. Row boundaries must align to 16 and column
    boundaries to two. This helper does not accept the MXFP4 scale codecs or
    runtime MMA-packed planes.
    """
    if (
        type(rows) is not int
        or type(columns) is not int
        or rows <= 0
        or rows % 16
        or columns <= 0
        or columns % 2
        or rows * columns > 1 << 24
    ):
        raise ValueError("NVFP4-CSF requires 16-row and two-column alignment")
    if (
        fixed.device.type != "cpu"
        or fixed.dtype != torch.uint8
        or tuple(fixed.shape) != (rows // 16, 16 * (1 + columns // 2))
    ):
        raise ValueError("NVFP4-CSF fixed plane shape, dtype or device mismatch")
    if (
        exceptions.device.type != "cpu"
        or exceptions.dtype != torch.uint32
        or exceptions.ndim != 1
    ):
        raise ValueError("NVFP4-CSF exceptions must be a CPU uint32 vector")

    ranges = []
    for selected, extent, alignment in (
        (row_range, rows, 16),
        (column_range, columns, 2),
    ):
        start, end = (0, extent) if selected is None else selected
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= extent
            or start % alignment
            or end % alignment
        ):
            raise ValueError("NVFP4-CSF slice must be aligned, nonempty and in range")
        ranges.append((start, end))
    (r0, r1), (c0, c1) = ranges
    if bool((fixed[:, :16] > 240).any()):
        raise ValueError("NVFP4-CSF row bases must not exceed 240")

    words = exceptions.to(torch.int64)
    positions = words & 0xFFFFFF
    if positions.numel() and (
        bool(positions[-1] >= rows * columns)
        or bool((positions[1:] <= positions[:-1]).any())
    ):
        raise ValueError("NVFP4-CSF exceptions must be unique, sorted and in range")
    if (r0, r1, c0, c1) == (0, rows, 0, columns):
        return fixed.clone(), exceptions.clone()
    erows, ecols = positions // columns, positions % columns
    keep = (erows >= r0) & (erows < r1) & (ecols >= c0) & (ecols < c1)
    rebased = (erows[keep] - r0) * (c1 - c0) + ecols[keep] - c0
    selected_exceptions = ((words[keep] & 0xFF000000) | rebased).to(torch.uint32)

    bases = fixed[r0 // 16 : r1 // 16, :16]
    codes = fixed[:, 16:].reshape(rows, columns // 2)[r0:r1, c0 // 2 : c1 // 2]
    selected_fixed = torch.cat((bases, codes.reshape((r1 - r0) // 16, -1)), dim=1)
    return selected_fixed, selected_exceptions
