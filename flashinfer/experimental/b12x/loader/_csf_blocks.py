"""Lossless FP4 scale blocks grouped by native 128-by-four scale tiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BLOCK_ROWS = 32
BLOCK_COLUMNS = 2
PAGE_GROUPS = 32
Quantization = Literal["nvfp4", "mxfp4"]


def _offset_bits(quantization: Quantization) -> int:
    if quantization == "nvfp4":
        return 4
    if quantization == "mxfp4":
        return 1
    raise ValueError("CSF quantization must be nvfp4 or mxfp4")


def _block_grid(rows: int, columns: int) -> np.ndarray:
    if rows <= 0 or rows % BLOCK_ROWS or columns <= 0:
        raise ValueError("CSF requires positive 32-row extents and positive columns")
    column_blocks = (columns + BLOCK_COLUMNS - 1) // BLOCK_COLUMNS
    return np.arange(rows // BLOCK_ROWS * column_blocks).reshape(
        rows // BLOCK_ROWS, column_blocks
    )


def block_order(rows: int, columns: int) -> np.ndarray:
    """Visit column pairs within clipped 128-row by four-column scale tiles."""
    grid = _block_grid(rows, columns)
    return np.concatenate(
        [
            grid[r : r + 4, c : c + 2].T.reshape(-1)
            for r in range(0, grid.shape[0], 4)
            for c in range(0, grid.shape[1], 2)
        ]
    )


def group_block_offsets(rows: int, columns: int) -> np.ndarray:
    """Block prefixes for clipped native tiles, derived entirely from geometry."""
    grid = _block_grid(rows, columns)
    counts = [
        min(4, grid.shape[0] - r) * min(2, grid.shape[1] - c)
        for r in range(0, grid.shape[0], 4)
        for c in range(0, grid.shape[1], 2)
    ]
    return np.concatenate(([0], np.cumsum(counts)))


def _page_block_offsets(rows: int, columns: int) -> np.ndarray:
    groups = group_block_offsets(rows, columns)
    return np.concatenate((groups[:-1:PAGE_GROUPS], groups[-1:]))


def packed_scale_row_order(columns: int) -> np.ndarray:
    """Native MMA scale row permutation, including a single K scale group."""
    if columns == 1:
        single = (
            2 * np.arange(4)[:, None] + np.array([0, 1, 8, 9, 16, 17, 24, 25])
        ).ravel()
        order = np.concatenate((single, single + 32))
    else:
        order = np.arange(64).reshape(8, 8).T.ravel()
    return order.reshape(-1, 4)[:, [0, 2, 1, 3]].ravel()


def codes_in_native_order(plane: CsfBlockPlane, destination=0) -> np.ndarray:
    """Reorder packed fields for a native scale destination without payload padding."""
    bits = _offset_bits(plane.quantization)
    if (
        destination == 0
        and bits == 4
        and plane.rows % 128 == 0
        and plane.columns % 4 == 0
    ):
        return plane.codes.reshape(-1, 2, 4, 32).transpose(0, 3, 2, 1).copy().ravel()
    output = np.empty_like(plane.codes)
    cursor = 0
    for r in range(0, plane.rows, 128):
        nr = min(128, plane.rows - r)
        for c in range(0, plane.columns, 4):
            nc = min(4, plane.columns - c)
            count = nr * nc * bits // 8
            packed = plane.codes[cursor : cursor + count]
            fields = np.unpackbits(packed, bitorder="little").reshape(-1, bits)
            values = (fields * (1 << np.arange(bits))).sum(axis=1).astype(np.uint8)
            tile = np.empty((nr, nc), dtype=np.uint8)
            start = 0
            for pair in range(0, nc, 2):
                width = min(2, nc - pair)
                tile[:, pair : pair + width] = values[
                    start : start + nr * width
                ].reshape(nr, width)
                start += nr * width
            if destination == 1:
                perm = packed_scale_row_order(plane.columns)
                ordered = (
                    tile.reshape(nr // 64, 64, nc)[:, perm].reshape(nr, nc).T.ravel()
                )
            elif destination in (2, 3):
                ordered = tile.ravel()
            else:
                ordered = tile.reshape(nr // 32, 32, nc).transpose(1, 0, 2).ravel()
            fields = ((ordered[:, None] >> np.arange(bits)) & 1).ravel()
            output[cursor : cursor + count] = np.packbits(fields, bitorder="little")
            cursor += count
    return output


@dataclass(frozen=True)
class CsfBlockPlane:
    """CPU export arrays for a rectangular logical scale partition.

    Offset width is fixed by quantization: four bits for NVFP4, one for MXFP4.
    Each block packs row-major offsets least-significant field first; a final
    single-column block stores only that column. Exceptions store an exact
    eight-bit scale value above the local position ``2 * row + column``.

    Pages contain 32 whole native tile groups, including clipped tails. Prefix
    entries count records relative to a page; page bases are byte offsets into
    the exception array. Every page has its own terminal prefix entry. Group
    sizes and code offsets follow geometry and need no stored directories.
    """

    rows: int
    columns: int
    quantization: Quantization
    bases: np.ndarray
    codes: np.ndarray
    page_bases: np.ndarray
    offsets: np.ndarray
    exceptions: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.bases,
                self.codes,
                self.page_bases,
                self.offsets,
                self.exceptions,
            )
        )

    def validate(self) -> None:
        order = block_order(self.rows, self.columns)
        blocks = len(order)
        bits = _offset_bits(self.quantization)
        page_blocks = _page_block_offsets(self.rows, self.columns)
        pages = len(page_blocks) - 1
        widths = np.minimum(2, self.columns - 2 * (order % ((self.columns + 1) // 2)))
        for name, dtype, shape in (
            ("bases", np.dtype("u1"), (self.rows,)),
            ("codes", np.dtype("u1"), (self.rows * self.columns * bits // 8,)),
            ("page_bases", np.dtype("<u4"), (pages,)),
            ("offsets", np.dtype("<u2"), (blocks + pages,)),
            ("exceptions", np.dtype("<u2"), (self.exceptions.size,)),
        ):
            value = getattr(self, name)
            if (
                value.dtype != dtype
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"Invalid CSF block {name} dtype, shape, or strides")
        if np.any(self.bases > 256 - (1 << bits)):
            raise ValueError("CSF row palette exceeds the scale byte range")
        page_numbers = np.repeat(np.arange(pages), np.diff(page_blocks))
        indexes = np.arange(blocks) + page_numbers
        sizes = self.offsets[indexes + 1].astype(np.int64) - self.offsets[indexes]
        if np.any(sizes < 0) or np.any(sizes > 32 * widths):
            raise ValueError("CSF block exception count exceeds its extent")
        prefix = np.concatenate(([0], np.cumsum(sizes)))
        if (
            np.any(self.offsets[page_blocks[:-1] + np.arange(pages)] != 0)
            or np.any(self.page_bases != 2 * prefix[page_blocks[:-1]])
            or prefix[-1] != self.exceptions.size
        ):
            raise ValueError("CSF exception pages must cover one consecutive arena")
        record_blocks = np.repeat(np.arange(blocks), sizes)
        positions = self.exceptions & 255
        if (
            np.any(positions >= 64)
            or np.any(positions % 2 >= widths[record_blocks])
            or np.any(
                (record_blocks[1:] == record_blocks[:-1])
                & (positions[1:] <= positions[:-1])
            )
        ):
            raise ValueError(
                "CSF block positions must be sorted, unique, and in bounds"
            )


def export_csf_blocks(
    fixed,
    exceptions,
    *,
    rows: int,
    columns: int,
    quantization: Quantization,
    row_range: tuple[int, int] | None = None,
    column_range: tuple[int, int] | None = None,
) -> CsfBlockPlane:
    """Index a checkpoint CSF rectangle without decoding or requantizing scales."""
    bits = _offset_bits(quantization)
    if rows <= 0 or rows % 16 or columns <= 0:
        raise ValueError("Checkpoint CSF requires 16-row extents and positive columns")
    if rows * columns > 1 << 24:
        raise ValueError("Checkpoint CSF exceeds its 24-bit position field")
    fixed = np.asarray(fixed)
    exceptions = np.asarray(exceptions)
    code_bytes = (columns * bits + 7) // 8
    if fixed.dtype != np.uint8 or fixed.shape != (rows // 16, 16 * (1 + code_bytes)):
        raise ValueError("Invalid checkpoint CSF fixed stream")
    if exceptions.dtype != np.dtype("<u4") or exceptions.ndim != 1:
        raise ValueError(
            "Checkpoint CSF exceptions require a little-endian uint32 vector"
        )
    if np.any(fixed[:, :16] > 256 - (1 << bits)):
        raise ValueError("CSF row palette exceeds the scale byte range")
    packed = fixed[:, 16:].reshape(rows, code_bytes)
    tail_bits = columns * bits % 8
    if tail_bits and np.any(packed[:, -1] >> tail_bits):
        raise ValueError("Checkpoint CSF unused code bits must be zero")
    r0, r1 = (0, rows) if row_range is None else row_range
    c0, c1 = (0, columns) if column_range is None else column_range
    if not (0 <= r0 < r1 <= rows and r0 % 32 == r1 % 32 == 0):
        raise ValueError("CSF block row partitions require complete 32-row groups")
    if not 0 <= c0 < c1 <= columns:
        raise ValueError("CSF block column partition lies outside the scale plane")
    position = exceptions & np.uint32(0xFFFFFF)
    if len(position) and (
        position[-1] >= rows * columns or np.any(position[1:] <= position[:-1])
    ):
        raise ValueError(
            "Checkpoint CSF positions must be sorted, unique, and in range"
        )
    nr, nc = r1 - r0, c1 - c0
    order = block_order(nr, nc)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    bases = fixed[:, :16].reshape(rows)[r0:r1].copy()
    code_columns = np.arange(c0, c1) * bits
    values = (packed[r0:r1, code_columns // 8] >> (code_columns % 8)) & (
        (1 << bits) - 1
    )
    if nc % 2:
        values = np.pad(values, ((0, 0), (0, 1)))
    column_blocks = (nc + 1) // 2
    values = values.reshape(nr // 32, 32, column_blocks, 2)
    values = values.transpose(0, 2, 1, 3).reshape(-1, 64)[order]
    widths = np.minimum(2, nc - 2 * (order % column_blocks))
    values = values[np.arange(64) % 2 < widths[:, None]]
    codes = np.bitwise_or.reduce(
        values.reshape(-1, 8 // bits) << (np.arange(8 // bits) * bits), axis=1
    ).astype(np.uint8)
    rr, cc = position // columns, position % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    rr, cc = rr[keep] - r0, cc[keep] - c0
    block = inverse[(rr // 32) * column_blocks + cc // 2]
    local = (rr % 32) * 2 + cc % 2
    permutation = np.argsort(block * 64 + local, kind="stable")
    records = (local | ((exceptions[keep] >> 24) << 8))[permutation].astype("<u2")
    prefix = np.concatenate(([0], np.cumsum(np.bincount(block, minlength=len(order)))))
    page_blocks = _page_block_offsets(nr, nc)
    if records.nbytes >= 1 << 32:
        raise ValueError("CSF plane exception arena must be smaller than 4 GiB")
    page_bases = (prefix[page_blocks[:-1]] * 2).astype("<u4")
    offsets = np.concatenate(
        [
            prefix[start : stop + 1] - prefix[start]
            for start, stop in zip(page_blocks[:-1], page_blocks[1:], strict=True)
        ]
    ).astype("<u2")
    return CsfBlockPlane(
        nr, nc, quantization, bases, codes, page_bases, offsets, records
    )


def logical_csf_components(plane: CsfBlockPlane):
    """Return row-major offset fields and sparse logical byte replacements."""
    bits = _offset_bits(plane.quantization)
    order = block_order(plane.rows, plane.columns)
    columns = (plane.columns + 1) // 2
    widths = np.minimum(2, plane.columns - 2 * (order % columns))
    fields = np.unpackbits(plane.codes, bitorder="little").reshape(-1, bits)
    fields = (fields * (1 << np.arange(bits))).sum(axis=1).astype(np.uint8)
    blocks = np.zeros((len(order), 64), dtype=np.uint8)
    blocks[np.arange(64) % 2 < widths[:, None]] = fields
    inverse = np.argsort(order)
    logical = (
        blocks[inverse]
        .reshape(plane.rows // 32, columns, 32, 2)
        .transpose(0, 2, 1, 3)
        .reshape(plane.rows, columns * 2)[:, : plane.columns]
        .copy()
    )
    groups = group_block_offsets(plane.rows, plane.columns)
    pages = np.repeat(np.arange(len(groups) - 1) // 32, np.diff(groups))
    index = np.arange(len(order)) + pages
    counts = plane.offsets[index + 1].astype(np.int64) - plane.offsets[index]
    record_blocks = np.repeat(order, counts)
    local = plane.exceptions & 255
    positions = (
        (record_blocks // columns * 32 + local // 2) * plane.columns
        + record_blocks % columns * 2
        + local % 2
    )
    permutation = np.argsort(positions)
    return (
        plane.bases.copy(),
        logical,
        positions[permutation],
        (plane.exceptions[permutation] >> 8).astype(np.uint8),
    )


def _from_logical_components(bases, offsets, positions, values, quantization):
    rows, columns = offsets.shape
    bits = _offset_bits(quantization)
    order = block_order(rows, columns)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    column_blocks = (columns + 1) // 2
    fields = np.pad(offsets, ((0, 0), (0, columns % 2)))
    fields = (
        fields.reshape(rows // 32, 32, column_blocks, 2)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 64)[order]
    )
    widths = np.minimum(2, columns - 2 * (order % column_blocks))
    fields = fields[np.arange(64) % 2 < widths[:, None]]
    codes = np.bitwise_or.reduce(
        fields.reshape(-1, 8 // bits) << (np.arange(8 // bits) * bits), axis=1
    ).astype(np.uint8)
    rr, cc = positions // columns, positions % columns
    block = inverse[(rr // 32) * column_blocks + cc // 2]
    local = rr % 32 * 2 + cc % 2
    permutation = np.argsort(block * 64 + local, kind="stable")
    records = (local | (values.astype(np.uint16) << 8))[permutation].astype("<u2")
    if records.nbytes >= 1 << 32:
        raise ValueError("CSF plane exception arena must be smaller than 4 GiB")
    prefix = np.concatenate(([0], np.cumsum(np.bincount(block, minlength=len(order)))))
    page_blocks = _page_block_offsets(rows, columns)
    page_bases = (prefix[page_blocks[:-1]] * 2).astype("<u4")
    indexes = np.concatenate(
        [
            prefix[start : stop + 1] - prefix[start]
            for start, stop in zip(page_blocks[:-1], page_blocks[1:], strict=True)
        ]
    ).astype("<u2")
    return CsfBlockPlane(
        rows, columns, quantization, bases.copy(), codes, page_bases, indexes, records
    )


def slice_csf_blocks(plane, *, row_range=None, column_range=None):
    """Select a TP rectangle and regroup its code fields and exception indexes."""
    plane.validate()
    r0, r1 = row_range or (0, plane.rows)
    c0, c1 = column_range or (0, plane.columns)
    if not (
        0 <= r0 < r1 <= plane.rows
        and r0 % 32 == r1 % 32 == 0
        and 0 <= c0 < c1 <= plane.columns
    ):
        raise ValueError("CSF slices require complete 32-row groups and valid columns")
    bases, fields, positions, values = logical_csf_components(plane)
    rr, cc = positions // plane.columns, positions % plane.columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
    return _from_logical_components(
        bases[r0:r1], fields[r0:r1, c0:c1], positions, values[keep], plane.quantization
    )


def concatenate_csf_blocks(planes):
    """Fuse projection rows, including groups crossing a gate/up boundary."""
    planes = tuple(planes)
    if not planes or any(
        (p.columns, p.quantization) != (planes[0].columns, planes[0].quantization)
        for p in planes
    ):
        raise ValueError("CSF concatenation requires matching columns and quantization")
    bases, fields, positions, values = [], [], [], []
    cursor = 0
    for plane in planes:
        plane.validate()
        b, f, p, v = logical_csf_components(plane)
        bases.append(b)
        fields.append(f)
        positions.append(p + cursor)
        values.append(v)
        cursor += plane.rows * plane.columns
    return _from_logical_components(
        np.concatenate(bases),
        np.concatenate(fields),
        np.concatenate(positions),
        np.concatenate(values),
        planes[0].quantization,
    )


def pad_csf_blocks(plane, *, rows, columns, row_base):
    """Extend scale geometry for zero-filled TP weight tails during loading."""
    if rows < plane.rows or rows % 32 or columns < plane.columns:
        raise ValueError("CSF padding must extend whole row blocks and logical columns")
    if not 0 <= row_base <= 256 - (1 << _offset_bits(plane.quantization)):
        raise ValueError("CSF padding row base exceeds its scale byte palette")
    if (rows, columns) == (plane.rows, plane.columns):
        return plane
    plane.validate()
    bases, fields, positions, values = logical_csf_components(plane)
    positions = positions // plane.columns * columns + positions % plane.columns
    return _from_logical_components(
        np.pad(bases, (0, rows - plane.rows), constant_values=row_base),
        np.pad(fields, ((0, rows - plane.rows), (0, columns - plane.columns))),
        positions,
        values,
        plane.quantization,
    )


def transform_csf_blocks(plane, value_table):
    """Apply the native scale byte conversion while retaining compact row palettes."""
    table = np.asarray(value_table)
    if table.dtype != np.uint8 or table.shape != (256,):
        raise ValueError("Scale conversion requires one uint8 result per source byte")
    bases, fields, positions, values = logical_csf_components(plane)
    logical = bases[:, None] + fields
    logical.ravel()[positions] = values
    logical = table[logical]
    limit = 1 << _offset_bits(plane.quantization)
    bases = np.minimum(table[bases], 256 - limit)
    offsets = logical.astype(np.int16) - bases[:, None]
    outside = (offsets < 0) | (offsets >= limit)
    positions = np.flatnonzero(outside)
    values = logical.ravel()[positions]
    offsets[outside] = 0
    return _from_logical_components(
        bases, offsets.astype(np.uint8), positions, values, plane.quantization
    )
