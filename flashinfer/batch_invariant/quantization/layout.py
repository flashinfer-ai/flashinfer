"""Scale-factor layouts of the block-scaled quantizer: pure integer logic, parameterised by the block size (16 elements per e4m3 scale for NVFP4, 32 per UE8M0 scale
for MX), imported by the kernels at trace time, by the CPU references and mirrored by the Lean model.

Two layouts of the same scale bytes:

* **128x4 swizzled** (FlashInfer's ``SfLayout.layout_128x4``, what the block-scaled GEMMs consume): tiles of 128 rows x 4
  scale columns are 512 contiguous bytes laid out as ``[row % 32][(row % 128) // 32][col % 4]``; column tiles are
  consecutive within a row tile; row tiles are ``128 * padded_cols`` bytes apart; rows padded to a multiple of 128 and
  columns to a multiple of 4, padding bytes zero.
* **8x4 swizzled** (``SfLayout.layout_8x4``): tiles of 8 rows x 4 scale columns are 32 contiguous bytes laid out as
  ``[row % 8][col % 4]``; column tiles consecutive within a row tile, row tiles ``8 * padded_cols`` bytes apart; rows
  padded to a multiple of 8, columns to a multiple of 4, padding bytes zero (FlashInfer's ``get_sf_out_offset_8x4``).
* **linear** (``SfLayout.layout_linear``): row-major ``(rows, K / block)`` bytes, no padding.

The layout is a placement of the store, never of the numerics: ``unswizzle(swizzled) == linear`` bit for bit for both
swizzles is the R4 statement the GPU suites check. ``LAYOUTS`` names the three; the API maps FlashInfer's ``SfLayout``
onto them.
"""

from __future__ import annotations

SF_ROW_TILE = 128
SF_COL_TILE = 4
SF_ROW_TILE_8X4 = 8
LAYOUTS = ("128x4", "8x4", "linear")


def pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def num_blocks(K: int, block: int) -> int:
    if K <= 0 or K % block:
        raise ValueError(f"K={K} must be a positive multiple of {block}")
    return K // block


def padded_blocks(K: int, block: int) -> int:
    """The padded block count a kernel is compiled for (one kernel per bucket)."""
    return pow2ceil(num_blocks(K, block))


def layout_name(layout) -> str:
    """``True`` / ``"128x4"`` -> ``"128x4"``, ``False`` / ``"linear"`` -> ``"linear"``, ``"8x4"`` -> ``"8x4"``."""
    if layout is True:
        return "128x4"
    if layout is False:
        return "linear"
    if layout in LAYOUTS:
        return layout
    raise ValueError(f"unknown scale-factor layout {layout!r}")


def row_tile(layout) -> int:
    """Rows per tile of the swizzled layouts (the row padding); 1 for the linear layout."""
    return {"128x4": SF_ROW_TILE, "8x4": SF_ROW_TILE_8X4, "linear": 1}[
        layout_name(layout)
    ]


def padded_sf_rows(M: int, layout="128x4") -> int:
    rt = row_tile(layout)
    return -(-M // rt) * rt


def padded_sf_cols(K: int, block: int, layout="128x4") -> int:
    if layout_name(layout) == "linear":
        return num_blocks(K, block)
    return -(-num_blocks(K, block) // SF_COL_TILE) * SF_COL_TILE


def sf_numel(M: int, K: int, block: int, layout="128x4") -> int:
    """Bytes of the scale-factor buffer: the padded 128x4 or 8x4 grid, or ``M * K / block`` for the linear layout."""
    return padded_sf_rows(M, layout) * padded_sf_cols(K, block, layout)


def sf_offset_128x4(row: int, col: int, padded_cols: int) -> int:
    """Byte offset of the scale of block ``col`` of row ``row`` in the 128x4 swizzled layout
    (FlashInfer's ``compute_sf_index_swizzled_128x4_gpu`` / ``get_sf_out_offset_128x4``)."""
    return (
        (col % SF_COL_TILE)
        + (col // SF_COL_TILE) * (SF_ROW_TILE * SF_COL_TILE)
        + (row % 32) * 16
        + ((row % SF_ROW_TILE) // 32) * 4
        + (row // SF_ROW_TILE) * (SF_ROW_TILE * padded_cols)
    )


def sf_offset_8x4(row: int, col: int, padded_cols: int) -> int:
    """Byte offset of the scale of block ``col`` of row ``row`` in FlashInfer's 8x4 swizzled layout
    (``get_sf_out_offset_8x4`` / ``compute_sf_index_swizzled_8x4_gpu``): tiles of 8 rows x 4 columns = 32 bytes as
    ``[row % 8][col % 4]``, column tiles consecutive, row tiles ``8 * padded_cols`` apart."""
    return (
        (col % SF_COL_TILE)
        + (row % SF_ROW_TILE_8X4) * SF_COL_TILE
        + (col // SF_COL_TILE) * (SF_ROW_TILE_8X4 * SF_COL_TILE)
        + (row // SF_ROW_TILE_8X4) * (SF_ROW_TILE_8X4 * padded_cols)
    )


def sf_offset_linear(row: int, col: int, cols: int) -> int:
    """Byte offset in the linear layout: row-major ``(rows, cols)``."""
    return row * cols + col


def sf_offset(layout, row: int, col: int, padded_cols: int) -> int:
    """The offset function of ``layout`` (``padded_cols`` = the layout's column count)."""
    name = layout_name(layout)
    if name == "128x4":
        return sf_offset_128x4(row, col, padded_cols)
    if name == "8x4":
        return sf_offset_8x4(row, col, padded_cols)
    return sf_offset_linear(row, col, padded_cols)


def _offsets(M: int, K: int, block: int, layout, device):
    import torch

    nb, pc = num_blocks(K, block), padded_sf_cols(K, block, layout)
    rows = torch.arange(M, device=device).view(M, 1)
    cols = torch.arange(nb, device=device).view(1, nb)
    name = layout_name(layout)
    if name == "128x4":
        return (
            (cols % SF_COL_TILE)
            + (cols // SF_COL_TILE) * (SF_ROW_TILE * SF_COL_TILE)
            + (rows % 32) * 16
            + ((rows % SF_ROW_TILE) // 32) * 4
            + (rows // SF_ROW_TILE) * (SF_ROW_TILE * pc)
        )
    if name == "8x4":
        return (
            (cols % SF_COL_TILE)
            + (rows % SF_ROW_TILE_8X4) * SF_COL_TILE
            + (cols // SF_COL_TILE) * (SF_ROW_TILE_8X4 * SF_COL_TILE)
            + (rows // SF_ROW_TILE_8X4) * (SF_ROW_TILE_8X4 * pc)
        )
    return rows * nb + cols


def swizzle_sf(sf_rows, M: int, K: int, block: int, layout="128x4"):
    """``(M, K / block)`` scale bytes -> the flat ``sf_numel`` uint8 tensor of ``layout``, padding zero."""
    import torch

    nb = num_blocks(K, block)
    assert tuple(sf_rows.shape) == (M, nb), (tuple(sf_rows.shape), M, nb)
    out = torch.zeros(
        sf_numel(M, K, block, layout), dtype=torch.uint8, device=sf_rows.device
    )
    idx = _offsets(M, K, block, layout, sf_rows.device)
    pc = padded_sf_cols(K, block, layout)
    if M > 0:  # an empty input has no cell to check
        assert int(idx[0, 0]) == sf_offset(layout, 0, 0, pc) and int(
            idx[M - 1, nb - 1]
        ) == sf_offset(layout, M - 1, nb - 1, pc)
    out[idx.reshape(-1)] = sf_rows.reshape(-1)
    return out


def unswizzle_sf(sf_flat, M: int, K: int, block: int, layout="128x4"):
    """Inverse of :func:`swizzle_sf`: the ``(M, K / block)`` logical scale bytes."""
    idx = _offsets(M, K, block, layout, sf_flat.device)
    return sf_flat.reshape(-1)[idx.reshape(-1)].reshape(M, num_blocks(K, block))


def swizzle_sf_128x4(sf_rows, M: int, K: int, block: int):
    return swizzle_sf(sf_rows, M, K, block, "128x4")


def unswizzle_sf_128x4(sf_flat, M: int, K: int, block: int):
    return unswizzle_sf(sf_flat, M, K, block, "128x4")


def sanitize_symbol_name(name: str) -> str:
    """A kernel cache name from a description: letters, digits and underscores only."""
    import re

    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
