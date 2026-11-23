# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time page layouts of the attention and indexer projection weights.

Every projection except ``indexer_weights_proj`` is a GEMV over FP8 E4M3
weights with FP32 block scales; ``indexer_weights_proj`` holds BF16 weights
without scales. The rows of each projection are split into fixed CTA row
blocks, and each CTA streams its block into shared memory one page at a time.
A page holds the block's weight codes for a range of input columns, arranged
as 16 x 16 fragment tiles, followed by the FP32 scales of that range (none for
``indexer_weights_proj``); the rest of the page is zero.

Packing only moves bytes. FP8 inputs are reinterpreted as ``uint8`` (which also
avoids FP8 copy kernels that some CPU builds lack) and scales are copied bit for
bit, so no value is converted or re-quantized.

Fragment tiles
    A 16 x 16 tile occupies 256 consecutive elements in the register order of
    the ``mma.sync.m16n8k16`` A operand. Element ``(r, k)`` belongs to lane
    ``4 * (r % 8) + (k % 8) // 2``, register ``2 * (k // 8) + r // 8`` of that
    lane, half ``k % 2`` of the register. Each lane stores its four registers as
    eight consecutive elements and loads them with a single access (8 bytes
    for FP8 pages, 16 bytes for the BF16 ``indexer_weights_proj`` pages); FP8
    codes are widened to the MMA input type in registers. See
    :func:`fragment_offsets`.

Page formats (rows of a CTA block x columns per page; offsets in bytes)
    * ``qkv_a_proj`` and ``q_a_indexer_wk``: 32 x 1024, six pages per block.
      Two 16-row groups of 64 tiles at 0x0000 and 0x4000 in warp-interleaved
      order; eight scales at 0x8000, one per 128-column block; 0x8080 bytes
      per page.
    * ``q_b_proj`` and ``indexer_wq_b``: 32 x 1024, two pages per block. As
      above, but tiles in column order.
    * ``w_uk``: 64 x 192, one page. Four 16-row groups of 12 tiles, 0x0C00
      apart; three scales at 0x3000, one per 64-column block; 0x3080 bytes.
    * ``w_uv``: 32 x 512, one page. Two 16-row groups of 32 tiles, 0x2000
      apart; four scales at 0x4000; 0x4080 bytes.
    * ``o_proj``: 48 x 512, five pages per block. Three 16-row groups of 32
      tiles, 0x2000 apart; twelve scales at 0x6000, where scale ``4 * g + j``
      belongs to row group ``g`` and 128-column block ``j``; 0x6080 bytes.
    * ``indexer_weights_proj``: BF16 16 x 1024, six pages per 16-row block, 64
      tiles in column order and no scales; 0x8000 bytes per page.

    In column order, tile ``t`` of a row group holds page columns
    ``16 * t .. 16 * t + 15``. In warp-interleaved order, warp ``w`` of the CTA
    owns the 128-column block ``w`` of the page, and the tile of its k16 step
    ``s`` is stored at position ``8 * s + w``, so that the eight warps read one
    contiguous 2 KiB span at every step.
"""

from __future__ import annotations

import torch

TILE = 16
TILE_ELEMENTS = TILE * TILE

_HEADS = 10
_NOPE_ROWS = 192  # per-head q_b_proj rows without RoPE, followed by 64 RoPE rows
_ROPE_ROWS = 64
_V_ROWS = 256  # per-head W_UV rows of kv_b_proj, after the 192 W_UK rows
_SCALE_BLOCK = 128


def fragment_offsets(row, col, lane_stride: int = 8):
    """Return the position of element ``(row, col)`` inside its fragment tile.

    Evaluates elementwise on integer tensors (or Python integers) with
    ``0 <= row, col < 16``. ``lane_stride`` is the number of elements each lane
    owns; layouts that store the fragments of several tiles back to back per
    lane pass a multiple of eight and add the tile's offset within the lane.
    """
    lane = 4 * (row % 8) + (col % 8) // 2
    register = 2 * (col // 8) + row // 8
    return lane_stride * lane + 2 * register + col % 2


def tile_offsets(
    rows: int, page_columns: int, pages: int, *, warp_interleaved: bool = False
) -> torch.Tensor:
    """Return the code-area position of each element of a CTA row block.

    The result has shape ``[rows, pages * page_columns]``. Entry ``(r, k)`` is
    the position of that element in a code area laid out as
    ``[pages][rows // 16 row groups][page_columns // 16 tiles][256]``.
    ``warp_interleaved`` selects the tile order of 1024-column pages described
    in the module docstring.
    """
    row = torch.arange(rows).view(rows, 1)
    column = torch.arange(pages * page_columns).view(1, -1)
    page, k = column // page_columns, column % page_columns
    if warp_interleaved:
        tile = 8 * (k % 128 // TILE) + k // 128
    else:
        tile = k // TILE
    return (
        page * (rows * page_columns)
        + row // TILE * (TILE * page_columns)
        + tile * TILE_ELEMENTS
        + fragment_offsets(row % TILE, k % TILE)
    )


def gather_index(offsets: torch.Tensor) -> torch.Tensor:
    """Turn destination positions into a gather index.

    ``offsets`` gives, for each source element in row-major order, its position
    in the packed layout and must be a permutation. The returned index
    satisfies ``packed[j] = source.flatten()[index[j]]``, so one
    ``index_select`` over the flattened source builds the packed layout.
    """
    flat = offsets.reshape(-1)
    index = torch.empty_like(flat)
    index[flat] = torch.arange(flat.numel())
    return index


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().view(torch.uint8)


def _pages(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    page_columns: int,
    page_bytes: int,
    *,
    warp_interleaved: bool = False,
) -> torch.Tensor:
    """Pack CTA row blocks into pages.

    ``blocks`` is uint8 ``[ctas, rows, pages * page_columns]`` and ``scales`` is
    FP32 ``[ctas, pages, n]``. Returns uint8 ``[ctas, pages, page_bytes]``
    holding the codes, then the ``n`` scales, then zeros.
    """
    ctas, rows, columns = blocks.shape
    pages = columns // page_columns
    code_bytes = rows * page_columns
    index = gather_index(
        tile_offsets(rows, page_columns, pages, warp_interleaved=warp_interleaved)
    ).to(blocks.device)
    codes = blocks.reshape(ctas, rows * columns).index_select(1, index)
    scale_bytes = scales.detach().contiguous().view(torch.uint8)
    packed = blocks.new_zeros(ctas, pages, page_bytes)
    packed[..., :code_bytes] = codes.view(ctas, pages, code_bytes)
    packed[..., code_bytes : code_bytes + scale_bytes.shape[-1]] = scale_bytes
    return packed


def pack_fragment_pages(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack ``[32 * ctas, 6144]`` FP8 codes into warp-interleaved 0x8080-byte pages.

    ``scales`` is FP32 ``[rows / 128, 48]`` for 128 x 128 blocks; 32-row block
    ``c`` uses scale row ``c // 4``. Returns the flat uint8 buffer of shape
    ``[ctas, 6, 0x8080]``.
    """
    ctas = weights.shape[0] // 32
    blocks = _bytes(weights).reshape(ctas, 32, weights.shape[1])
    first_row = 32 * torch.arange(ctas, device=scales.device)
    block_scales = scales[first_row // 128].reshape(ctas, -1, 8)
    return _pages(blocks, block_scales, 1024, 0x8080, warp_interleaved=True).reshape(-1)


def pack_qkv_a_proj(
    q_a_weight: torch.Tensor,
    q_a_scale: torch.Tensor,
    kv_a_weight: torch.Tensor,
    kv_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Pack the MLA input projections into 82 blocks of six 0x8080-byte pages.

    ``q_a_weight`` is ``q_a_proj`` FP8 ``[2048, 6144]`` with FP32 ``[16, 48]``
    scales (blocks 0..63). ``kv_a_weight`` is the whole ``kv_a_proj_with_mqa``
    FP8 ``[576, 6144]`` (512 latent rows, then 64 RoPE key rows) with FP32
    ``[5, 48]`` scales, packed as one run of 18 blocks (blocks 64..81).
    """
    return torch.cat(
        [
            pack_fragment_pages(q_a_weight, q_a_scale),
            pack_fragment_pages(kv_a_weight, kv_a_scale),
        ]
    )


def pack_q_a_indexer_wk(
    q_a_weight: torch.Tensor,
    q_a_scale: torch.Tensor,
    wk_weight: torch.Tensor,
    wk_scale: torch.Tensor,
) -> torch.Tensor:
    """Pack the indexer-rank input projections into 68 blocks of six pages.

    ``q_a_proj`` FP8 ``[2048, 6144]`` with FP32 ``[16, 48]`` scales fills
    blocks 0..63; ``indexer.wk`` FP8 ``[128, 6144]`` with FP32 ``[1, 48]``
    scales fills blocks 64..67.
    """
    return torch.cat(
        [
            pack_fragment_pages(q_a_weight, q_a_scale),
            pack_fragment_pages(wk_weight, wk_scale),
        ]
    )


def pack_q_b_proj(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack the rank-local ``q_b_proj`` ``[10 * 256, 2048]`` into two pages per block.

    Each head has 192 rows without RoPE followed by 64 RoPE rows, that is six
    and two 32-row blocks. CTA blocks list every head's non-RoPE blocks first
    (head-major), then every head's RoPE blocks. ``scale`` is FP32
    ``[20, 16]`` for 128 x 128 blocks. Returns uint8 ``[80, 2, 0x8080]``.
    """
    per_head = (_NOPE_ROWS + _ROPE_ROWS) // 32
    nope = _NOPE_ROWS // 32
    block = torch.arange(_HEADS * per_head, device=weight.device)
    block = block.view(_HEADS, per_head)
    order = torch.cat([block[:, :nope].reshape(-1), block[:, nope:].reshape(-1)])
    blocks = _bytes(weight).reshape(-1, 32, weight.shape[1]).index_select(0, order)
    first_row = 32 * order.to(scale.device)
    block_scales = scale.reshape(scale.shape[0], -1, 8)[first_row // 128]
    return _pages(blocks, block_scales, 1024, 0x8080)


def split_kv_b_proj(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the rank-local ``kv_b_proj`` into its per-head W_UK and W_UV parts.

    ``weight`` is FP8 ``[10 * 448, 512]`` and ``scale`` FP32 ``[35, 4]`` for
    its 128 x 128 blocks. Rows ``448 * h .. 448 * h + 191`` of head ``h`` are
    W_UK (key without RoPE) and the next 256 rows are W_UV (value). Every
    64-row chunk starts at a multiple of 64 and so lies inside one 128-row
    scale block.

    Returns new contiguous tensors ``(w_uk, w_uk_scale, w_uv, w_uv_scale)``:

    * ``w_uk``: FP8 ``[10, 512, 192]``, W_UK transposed, which absorbs the
      query into the latent space. ``w_uk_scale[h, i, j]`` is FP32
      ``[10, 4, 3]``, the scale of latent block ``i`` and 64-row chunk ``j``.
    * ``w_uv``: FP8 ``[10, 256, 512]``. ``w_uv_scale[h, j, i]`` is FP32
      ``[10, 4, 4]``, the scale of 64-row chunk ``j`` and latent block ``i``.
    """
    head_rows = _NOPE_ROWS + _V_ROWS
    codes = _bytes(weight).reshape(_HEADS, head_rows, weight.shape[1])
    w_uk = codes[:, :_NOPE_ROWS].transpose(1, 2).contiguous().view(weight.dtype)
    w_uv = codes[:, _NOPE_ROWS:].contiguous().view(weight.dtype)
    first_row = head_rows * torch.arange(_HEADS, device=scale.device).view(-1, 1)
    chunk = torch.arange(0, head_rows, 64, device=scale.device).view(1, -1)
    chunk_scales = scale[(first_row + chunk) // _SCALE_BLOCK]  # [10, 7, 4]
    nope_chunks = _NOPE_ROWS // 64
    w_uk_scale = chunk_scales[:, :nope_chunks].transpose(1, 2).contiguous()
    w_uv_scale = chunk_scales[:, nope_chunks:].contiguous()
    return w_uk, w_uk_scale, w_uv, w_uv_scale


def pack_w_uk(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack W_UK transposed, ``[10, 512, 192]``, into one page per block.

    Each head has eight 64-row blocks; ``scale`` is FP32 ``[10, 4, 3]`` for
    128-row x 64-column blocks (see :func:`split_kv_b_proj`). Returns uint8
    ``[80, 0x3080]``.
    """
    blocks = _bytes(weight).reshape(-1, 64, weight.shape[-1])
    first_row = 64 * torch.arange(blocks.shape[0], device=scale.device)
    block_scales = scale.reshape(-1, 1, scale.shape[-1])[first_row // 128]
    return _pages(blocks, block_scales, blocks.shape[-1], 0x3080).view(
        blocks.shape[0], 0x3080
    )


def pack_w_uv(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack W_UV, ``[10, 256, 512]``, into one page per block.

    Each head has eight 32-row blocks; ``scale`` is FP32 ``[10, 4, 4]`` for
    64-row x 128-column blocks (see :func:`split_kv_b_proj`). Returns uint8
    ``[80, 0x4080]``.
    """
    blocks = _bytes(weight).reshape(-1, 32, weight.shape[-1])
    first_row = 32 * torch.arange(blocks.shape[0], device=scale.device)
    block_scales = scale.reshape(-1, 1, scale.shape[-1])[first_row // 64]
    return _pages(blocks, block_scales, blocks.shape[-1], 0x4080).view(
        blocks.shape[0], 0x4080
    )


def pack_o_proj(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack the rank-local ``o_proj`` ``[6144, 2560]`` into five pages per 48-row block.

    ``scale`` is FP32 ``[48, 20]`` for 128 x 128 blocks. A page stores the
    scale of each (16-row group, 128-column block) pair it covers. Returns
    uint8 ``[128, 5, 0x6080]``.
    """
    ctas, pages = weight.shape[0] // 48, weight.shape[1] // 512
    blocks = _bytes(weight).reshape(ctas, 48, weight.shape[1])
    device = scale.device
    cta = torch.arange(ctas, device=device).view(-1, 1, 1, 1)
    page = torch.arange(pages, device=device).view(1, -1, 1, 1)
    group = torch.arange(3, device=device).view(1, 1, -1, 1)
    column_block = torch.arange(4, device=device).view(1, 1, 1, -1)
    first_row = 48 * cta + 16 * group
    block_scales = scale[first_row // 128, 4 * page + column_block]
    return _pages(blocks, block_scales.reshape(ctas, pages, 12), 512, 0x6080)


def pack_indexer_wq_b(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack ``indexer.wq_b`` ``[4096, 2048]`` into two pages per block.

    ``scale`` is FP32 ``[32, 16]`` for 128 x 128 blocks. Returns uint8
    ``[128, 2, 0x8080]``.
    """
    blocks = _bytes(weight).reshape(-1, 32, weight.shape[1])
    first_row = 32 * torch.arange(blocks.shape[0], device=scale.device)
    block_scales = scale.reshape(scale.shape[0], -1, 8)[first_row // 128]
    return _pages(blocks, block_scales, 1024, 0x8080)


def pack_indexer_weights_proj(weight: torch.Tensor) -> torch.Tensor:
    """Pack the BF16 ``indexer.weights_proj`` ``[32, 6144]`` into fragment-tile pages.

    Returns BF16 ``[2, 6 * 16384]``: per 16-row block, six 1024-column pages of
    64 tiles in column order.
    """
    groups = weight.shape[0] // TILE
    index = gather_index(tile_offsets(TILE, 1024, weight.shape[1] // 1024))
    source = weight.detach().reshape(groups, TILE * weight.shape[1])
    return source.index_select(1, index.to(weight.device))
