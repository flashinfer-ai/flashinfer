# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/scratch_layout.py @ ad997e28 (2026-06-07) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Shared byte-layout helpers for caller-owned scratch buffers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

SCRATCH_ALIGN_BYTES = 1024


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def materialize_scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = align_up(offset_bytes, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
    nbytes = shape_numel(shape) * dtype_nbytes(dtype)
    view_bytes = scratch.narrow(0, offset_bytes, nbytes)
    typed_view = view_bytes.view(dtype).view(shape)
    return typed_view, offset_bytes + nbytes


def materialize_scratch_strided_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = align_up(offset_bytes, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
    nbytes = shape_numel(shape) * dtype_nbytes(dtype)
    view_bytes = scratch.narrow(0, offset_bytes, nbytes)
    typed_storage = view_bytes.view(dtype)
    return typed_storage.as_strided(shape, stride), offset_bytes + nbytes


def ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


# ---------------------------------------------------------------------------
# WO-projection arena layout.
#
# Carved from b12x/attention/workspace.py @ 906d63d0: the WO projections share
# a serving arena with attention, so the byte layout lived there upstream.
# Both gemm.wo_projection and the attention workspace need it, which makes
# _lib its home under the import-topology rule (op dirs and group _shared
# trees may not import across groups).  The arena alignment (1024) matches
# SCRATCH_ALIGN_BYTES, so the materialize helpers above are drop-ins for the
# upstream _materialize_arena_view/_materialize_arena_strided_view.
# ---------------------------------------------------------------------------

WO_MXFP8_SCALE_VEC_SIZE = 32
WO_MXFP8_SCALE_ROW_TILE = 128
WO_MXFP8_SCALE_K_TILE = 4


@dataclass(frozen=True, kw_only=True)
class WOProjectionArenaLayout:
    nbytes: int = 0
    x_q_values_offset_bytes: int = 0
    x_q_scale_rows_offset_bytes: int = 0
    x_q_scale_mma_offset_bytes: int = 0
    tmp_offset_bytes: int = 0
    tmp_q_values_offset_bytes: int = 0
    tmp_q_scale_rows_offset_bytes: int = 0
    tmp_q_scale_mma_offset_bytes: int = 0
    output_offset_bytes: int = 0


def check_wo_mxfp8_k(k: int) -> None:
    if int(k) <= 0 or int(k) % 128 != 0:
        raise ValueError(
            f"WO MXFP8 dense-GEMM K must be a positive multiple of 128, got {k}"
        )


def wo_mxfp8_scale_physical_shape(
    *,
    m: int,
    k: int,
    num_groups: int,
) -> tuple[int, int, int, int, int, int]:
    sf_k = int(k) // WO_MXFP8_SCALE_VEC_SIZE
    return (
        int(num_groups),
        ceil_div(int(m), WO_MXFP8_SCALE_ROW_TILE),
        ceil_div(sf_k, WO_MXFP8_SCALE_K_TILE),
        32,
        4,
        4,
    )


def layout_wo_projection(
    *,
    offset_bytes: int,
    tokens: int,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
) -> WOProjectionArenaLayout:
    tokens = max(int(tokens), 1)
    groups = int(groups)
    group_width = int(group_width)
    rank = int(rank)
    hidden = int(hidden)
    if groups <= 0 or group_width <= 0 or rank <= 0 or hidden <= 0:
        raise ValueError(
            "WO projection arena requires positive groups, group_width, rank, and hidden"
        )
    check_wo_mxfp8_k(group_width)
    check_wo_mxfp8_k(rank * groups)

    start = int(offset_bytes)
    cursor = align_up(start, SCRATCH_ALIGN_BYTES)

    x_q_values_offset_bytes = cursor
    cursor += tokens * group_width * groups * dtype_nbytes(torch.float8_e4m3fn)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    x_q_scale_rows_offset_bytes = cursor
    cursor += (
        groups
        * tokens
        * (group_width // WO_MXFP8_SCALE_VEC_SIZE)
        * dtype_nbytes(torch.float8_e8m0fnu)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    x_q_scale_mma_offset_bytes = cursor
    cursor += shape_numel(
        wo_mxfp8_scale_physical_shape(
            m=tokens,
            k=group_width,
            num_groups=groups,
        )
    ) * dtype_nbytes(torch.uint8)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tmp_offset_bytes = cursor
    cursor += tokens * rank * groups * dtype_nbytes(torch.bfloat16)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tmp_q_width = rank * groups
    tmp_q_values_offset_bytes = cursor
    cursor += tokens * tmp_q_width * dtype_nbytes(torch.float8_e4m3fn)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tmp_q_scale_rows_offset_bytes = cursor
    cursor += (
        tokens
        * (tmp_q_width // WO_MXFP8_SCALE_VEC_SIZE)
        * dtype_nbytes(torch.float8_e8m0fnu)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tmp_q_scale_mma_offset_bytes = cursor
    cursor += shape_numel(
        wo_mxfp8_scale_physical_shape(
            m=tokens,
            k=tmp_q_width,
            num_groups=1,
        )
    ) * dtype_nbytes(torch.uint8)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    output_offset_bytes = cursor
    cursor += tokens * hidden * dtype_nbytes(torch.bfloat16)

    return WOProjectionArenaLayout(
        nbytes=max(0, int(cursor) - start),
        x_q_values_offset_bytes=x_q_values_offset_bytes,
        x_q_scale_rows_offset_bytes=x_q_scale_rows_offset_bytes,
        x_q_scale_mma_offset_bytes=x_q_scale_mma_offset_bytes,
        tmp_offset_bytes=tmp_offset_bytes,
        tmp_q_values_offset_bytes=tmp_q_values_offset_bytes,
        tmp_q_scale_rows_offset_bytes=tmp_q_scale_rows_offset_bytes,
        tmp_q_scale_mma_offset_bytes=tmp_q_scale_mma_offset_bytes,
        output_offset_bytes=output_offset_bytes,
    )
