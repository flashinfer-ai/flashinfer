# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time layouts of the expert weights read by the MoE kernels.

A weight bank is one expert (bank 0 is the shared expert). Its gate, up and
down projections are FP8 E4M3 codes with FP32 scales for 128 x 128 blocks.
Packing only moves bytes: codes are handled as ``uint8`` and scales are copied
bit for bit. Codes are stored as the 16 x 16 fragment tiles documented in
``_attention_packing``; the kernels feed them to ``m16n8k16`` MMAs after an
in-register FP8 to FP16 conversion, or assemble FP8 ``m16n8k32`` operands from
two consecutive k16 fragments.

Gate/up pages: uint8 ``[banks, rows / 16, 6, 0x8080]``
    Built by :func:`pack_gate_up_pages`. One page covers a 16-row group of both
    the gate and the up projection over 1024 hidden columns. Byte offsets
    inside a page:

    * 0x0000: gate codes, 64 tiles in column order (warp ``w`` reads tiles
      ``8 * w .. 8 * w + 7``, i.e. 128-column block ``w``);
    * 0x4000: up codes, same order;
    * 0x8000: eight gate scales, one per 128-column block;
    * 0x8020: eight up scales;
    * 0x8040: zero padding to the 0x8080 page stride. The kernels copy only the
      first 0x8040 bytes of a page.

Down slices: uint8 ``[banks, 6144, 256]`` storage, FP32 scales ``[banks, 1024, 2]``
    Built by :func:`pack_down_slices`. The 6144 output rows of a bank form 128
    slices of 48 rows, stored back to back (0x3000 bytes each). A slice has
    three 16-row tiles of 0x1000 bytes. A tile has eight 32-column chunks of
    0x200 bytes, chunk ``w`` being read by warp ``w``. In a chunk, each lane
    owns 16 bytes: its fragment of columns 0..15 followed by its fragment of
    columns 16..31, so one 16-byte load fetches the operands of two k16 steps.

    Each slice also owns eight consecutive scale rows. Row ``z < 6`` holds the
    scales of slice rows ``8 * z .. 8 * z + 7``, one per 128-column block; rows
    6 and 7 are zero.
"""

from __future__ import annotations

import torch

from ._attention_packing import fragment_offsets, gather_index, tile_offsets

_SCALE_BLOCK = 128
_GROUP_ROWS = 16
_PAGE_COLUMNS = 1024
_PAGE_BYTES = 0x8080
_HALF_BYTES = _GROUP_ROWS * _PAGE_COLUMNS  # codes of one projection per page
_SCALE_OFFSET = 2 * _HALF_BYTES
_SCALES_PER_PAGE = _PAGE_COLUMNS // _SCALE_BLOCK

_DOWN_SLICES = 128
_DOWN_SLICE_ROWS = 48
_DOWN_COLUMNS = 256
_DOWN_BAND_ROWS = 8
_DOWN_BANDS = _DOWN_SLICE_ROWS // _DOWN_BAND_ROWS
_DOWN_SCALE_ROWS = 8  # scale rows per slice, the last two are zero


def _down_slice_offsets() -> torch.Tensor:
    """Return the packed position of each element of a ``[48, 256]`` down slice."""
    row = torch.arange(_DOWN_SLICE_ROWS).view(-1, 1)
    column = torch.arange(_DOWN_COLUMNS).view(1, -1)
    k = column % 32
    return (
        row // _GROUP_ROWS * (_GROUP_ROWS * _DOWN_COLUMNS)
        + column // 32 * (_GROUP_ROWS * 32)
        + fragment_offsets(row % _GROUP_ROWS, k % 16, lane_stride=16)
        + 8 * (k // 16)
    )


def pack_down_slices(
    down: torch.Tensor, down_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack down projections into 48-row slices.

    ``down`` is FP8 ``[banks, 6144, 256]`` and ``down_scale`` FP32
    ``[banks, 48, 2]``. Returns uint8 codes with the shape of ``down`` and
    FP32 scales ``[banks, 1024, 2]``, both laid out as in the module docstring.
    """
    codes = down.view(torch.uint8)
    banks = codes.shape[0]
    with torch.inference_mode():
        index = gather_index(_down_slice_offsets()).to(codes.device)
        slices = codes.reshape(banks, _DOWN_SLICES, -1).index_select(2, index)

        band = torch.arange(_DOWN_SLICES * _DOWN_BANDS, device=down_scale.device)
        band = band.view(_DOWN_SLICES, _DOWN_BANDS)
        block_row = band * _DOWN_BAND_ROWS // _SCALE_BLOCK
        source_scales = down_scale.to(torch.float32).reshape(banks, -1, 2)
        scales = source_scales.new_zeros(banks, _DOWN_SLICES, _DOWN_SCALE_ROWS, 2)
        scales[:, :, :_DOWN_BANDS] = source_scales[:, block_row]
        return (
            slices.view(codes.shape),
            scales.view(banks, _DOWN_SLICES * _DOWN_SCALE_ROWS, 2),
        )


def pack_gate_up_pages(
    gate: torch.Tensor,
    gate_scale: torch.Tensor,
    up: torch.Tensor,
    up_scale: torch.Tensor,
) -> torch.Tensor:
    """Pack the gate and up projections into gate/up pages.

    ``gate`` and ``up`` are FP8 ``[banks, rows, 6144]`` with ``rows`` a multiple
    of 128, and ``gate_scale`` and ``up_scale`` FP32 ``[banks, rows / 128, 48]``.
    Returns the uint8 pages ``[banks, rows / 16, 6, 0x8080]`` described in the
    module docstring.
    """
    gate_codes = gate.view(torch.uint8)
    up_codes = up.view(torch.uint8)
    banks, rows, hidden = gate_codes.shape
    groups = rows // _GROUP_ROWS
    pages = hidden // _PAGE_COLUMNS
    with torch.inference_mode():
        index = gather_index(tile_offsets(_GROUP_ROWS, _PAGE_COLUMNS, pages))
        index = index.to(gate.device)
        first_row = _GROUP_ROWS * torch.arange(groups, device=gate_scale.device)
        packed = torch.zeros(
            banks, groups, pages, _PAGE_BYTES, dtype=torch.uint8, device=gate.device
        )
        projections = ((gate_codes, gate_scale), (up_codes, up_scale))
        for half, (codes, scales) in enumerate(projections):
            codes = codes.reshape(banks, groups, _GROUP_ROWS * hidden)
            codes = codes.index_select(2, index).view(banks, groups, pages, -1)
            packed[..., half * _HALF_BYTES : (half + 1) * _HALF_BYTES] = codes

            scales = scales.to(torch.float32).reshape(
                banks, -1, pages, _SCALES_PER_PAGE
            )
            scale_bytes = scales[:, first_row // _SCALE_BLOCK].view(torch.uint8)
            offset = _SCALE_OFFSET + half * scale_bytes.shape[-1]
            packed[..., offset : offset + scale_bytes.shape[-1]] = scale_bytes
        return packed
