"""Read compact W4A8 MXFP4 scale words from inline MXFP4-CSF storage.

Compact (N64) W4A8 kernels stage one native scale tile per pipeline stage:
128 rows (64 in a row group's tail) of four K32 columns (two in a K tail),
one 32-bit word per row. A thread of MMA warp ``w`` and quad ``q`` consumes the
words of rows ``32 w + 8 nt + q`` for ``nt = 0..3``. Slot ``s = 32 w + 4 q + nt``
orders a tile's rows by consumer thread. This storage keeps every tile
compressed; each thread rebuilds exactly its four words in registers after the
stage barrier, so no scale expansion runs before the MoE.

Storage of one projection, in bytes, for ``E`` experts, ``B`` row blocks (the
128-row tiles of every row group in native order) and ``K`` K128 tiles per row
block (``T = B K``):

- ``E x T`` tile blocks of ``TILE_BYTES``, expert, row block and K tile major:
  64 bytes of selector nibbles in slot order (slot ``s`` in byte ``s // 2``, low
  nibble first; bit ``c`` selects ``base + 1`` over ``base`` in column ``c``),
  then a header word and ``RECORDS`` exception records. The header holds the
  number of exceptions (0 to ``RECORDS``), or ``HEAVY`` plus the index of the
  tile's raw copy. A record holds the slot (bits 0-6), the column (bits 8-9)
  and the native byte (bits 16-23).
- ``E x B`` base blocks of 128 bytes: the base of each slot's row; padding
  slots hold zero.
- raw tiles of 512 bytes, one per tile with more than ``RECORDS`` exceptions:
  the native word of every slot.

Words of padding rows are zero, as the native stages write them; kernels clear
the two padding columns of a K tail tile. Bases are chosen per row from the
native bytes, independently of the checkpoint's own (unsliced) row bases.
Preparation builds the storage from the native compact plane and verifies that
it decodes to that plane exactly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
import triton as tr
import triton.language as tl
from triton import cdiv as triton_cdiv
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    ld_shared_u32,
)

TILE_BYTES = 96
SELECTOR_BYTES = 64
RECORDS = 7
BASE_BYTES = 128
RAW_TILE_BYTES = 512
HEAVY = 0x80000000
_EXPERT_CHUNK = 32


@dataclass(frozen=True)
class Mxfp4CsfInlinePlane:
    """Inline-readable compact W4A8 scales of one projection."""

    storage: torch.Tensor
    num_experts: int
    rows: int
    columns: int
    group_rows: int
    heavy_tiles: int

    @property
    def geometry(self):
        return inline_geometry(self.rows, self.columns, self.group_rows)

    @property
    def row_blocks(self) -> int:
        return self.geometry[0]

    @property
    def k_tiles(self) -> int:
        return self.geometry[1]

    @property
    def tiles_bytes(self) -> int:
        return self.num_experts * self.row_blocks * self.k_tiles * TILE_BYTES

    @property
    def bases_bytes(self) -> int:
        return self.num_experts * self.row_blocks * BASE_BYTES


def inline_geometry(rows: int, columns: int, group_rows: int) -> tuple[int, int]:
    """Row blocks and K128 tiles per expert of a compact native scale plane."""
    rows, columns, group_rows = int(rows), int(columns), int(group_rows)
    if group_rows <= 0 or rows % group_rows or group_rows % 64 or columns % 2:
        raise ValueError("inline W4A8 scales require compact N64/K64 row groups")
    return rows // group_rows * ((group_rows + 127) // 128), (columns + 3) // 4


@lru_cache(maxsize=16)
def _slot_offsets(rows: int, columns: int, group_rows: int) -> np.ndarray:
    """Native byte offset of ``[block, k_tile, slot, column]``, -1 for padding."""
    blocks, k_tiles = inline_geometry(rows, columns, group_rows)
    tiles_per_group = (group_rows + 127) // 128
    block = np.arange(blocks)[:, None, None, None]
    k_tile = np.arange(k_tiles)[None, :, None, None]
    slot = np.arange(128)[None, None, :, None]
    column = np.arange(4)[None, None, None, :]
    group, tile = block // tiles_per_group, block % tiles_per_group
    tile_rows = np.minimum(128, group_rows - tile * 128)
    tile_columns = np.minimum(4, columns - k_tile * 4)
    row = (slot >> 5) * 32 + (slot & 3) * 8 + ((slot >> 2) & 7)
    offset = (
        group * group_rows * columns
        + tile * 128 * columns
        + k_tile * tile_rows * 4
        + row * tile_columns
        + column
    )
    valid = (row < tile_rows) & (column < tile_columns)
    return np.where(valid, offset, -1).astype(np.int64)


def _as_int32_bits(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values >= 1 << 31, values - (1 << 32), values).to(torch.int32)


@contextmanager
def _allocations_from(pool):
    if pool is None:
        yield
    else:
        with torch.cuda.use_mem_pool(pool):
            yield


def build_mxfp4_csf_inline(
    native: torch.Tensor, *, rows: int, columns: int, group_rows: int
) -> Mxfp4CsfInlinePlane:
    """Compress native compact W4A8 scale bytes ``[E, rows * columns]``.

    The storage is verified against ``native`` before it is returned.
    """
    if native.dtype != torch.uint8 or native.dim() != 2:
        raise TypeError("inline W4A8 scales require uint8 [experts, bytes] storage")
    if native.shape[1] != rows * columns or not native.is_contiguous():
        raise ValueError("inline W4A8 scales require one contiguous native plane")
    # Encoding and verification allocate many times the storage they produce.
    # They run in a private pool, and only the storage is copied out: in the
    # caller's pool it would otherwise sit between their freed blocks, which an
    # allocator that does not split large blocks (vLLM loads weights with
    # max_split_size_mb:20) can neither reuse nor release, about 20 MiB per plane.
    pool = torch.cuda.MemPool() if native.is_cuda else None
    with _allocations_from(pool):
        staged, heavy_tiles = _encode_inline_storage(
            native, rows=rows, columns=columns, group_rows=group_rows
        )
    storage = staged.clone()
    del staged
    plane = Mxfp4CsfInlinePlane(
        storage,
        native.shape[0],
        int(rows),
        int(columns),
        int(group_rows),
        heavy_tiles,
    )
    with _allocations_from(pool):
        exact = all(
            torch.equal(
                decode_mxfp4_csf_inline(
                    plane, e0, min(e0 + _EXPERT_CHUNK, plane.num_experts)
                ),
                native[e0 : e0 + _EXPERT_CHUNK],
            )
            for e0 in range(0, plane.num_experts, _EXPERT_CHUNK)
        )
    del pool
    if not exact:
        raise RuntimeError(
            "inline MXFP4-CSF scales do not reproduce the native scale plane"
        )
    return plane


def _encode_inline_storage(
    native: torch.Tensor, *, rows: int, columns: int, group_rows: int
) -> tuple[torch.Tensor, int]:
    blocks, k_tiles = inline_geometry(rows, columns, group_rows)
    experts, tiles = native.shape[0], blocks * k_tiles
    device = native.device
    offsets = torch.from_numpy(_slot_offsets(rows, columns, group_rows)).to(device)
    valid = (offsets >= 0).view(tiles, 128, 4)
    index = offsets.clamp_min(0).view(-1)
    row_valid = (
        valid.view(blocks, k_tiles, 128, 4)
        .permute(0, 2, 1, 3)
        .reshape(blocks * 128, k_tiles * 4)
        .to(torch.int32)
    )
    tile_blocks = torch.zeros(
        experts, tiles, TILE_BYTES, dtype=torch.uint8, device=device
    )
    bases = torch.zeros(experts, blocks, BASE_BYTES, dtype=torch.uint8, device=device)
    counts = torch.zeros(experts, tiles, dtype=torch.int64, device=device)
    raw, records = [], []
    column_bits = torch.arange(4, device=device, dtype=torch.uint8)
    # Bound the per-row histograms to about 64 MiB per expert chunk.
    chunk = max(1, min(_EXPERT_CHUNK, (1 << 16) // (blocks * 128)))
    for e0 in range(0, experts, chunk):
        e1 = min(experts, e0 + chunk)
        values = native[e0:e1, index].view(e1 - e0, tiles, 128, 4)
        values = torch.where(valid, values, 0)
        # Each row's base covers the most of its bytes with base or base + 1.
        row_values = (
            values.view(e1 - e0, blocks, k_tiles, 128, 4)
            .permute(0, 1, 3, 2, 4)
            .reshape(e1 - e0, blocks * 128, k_tiles * 4)
        )
        histogram = torch.zeros(
            e1 - e0, blocks * 128, 256, dtype=torch.int32, device=device
        )
        histogram.scatter_add_(
            2, row_values.long(), row_valid.expand(e1 - e0, -1, -1).contiguous()
        )
        base = (histogram[..., :255] + histogram[..., 1:]).argmax(-1)
        base = base.to(torch.uint8).view(e1 - e0, blocks, 128)
        bases[e0:e1] = base
        tile_base = (
            base.unsqueeze(2)
            .expand(e1 - e0, blocks, k_tiles, 128)
            .reshape(e1 - e0, tiles, 128, 1)
        )
        selected = valid & (values == tile_base + 1)
        exception = valid & (values != tile_base) & ~selected
        count = exception.sum((2, 3))
        counts[e0:e1] = count
        heavy = count > RECORDS
        nibbles = (selected.to(torch.uint8) << column_bits).sum(-1, dtype=torch.uint8)
        nibbles = torch.where(heavy[..., None], 0, nibbles)
        tile_blocks[e0:e1, :, :SELECTOR_BYTES] = nibbles[..., 0::2] | (
            nibbles[..., 1::2] << 4
        )
        if bool(heavy.any()):
            raw.append(values[heavy])
        position = (exception & ~heavy[..., None, None]).nonzero()
        if position.numel():
            # nonzero() lists a tile's exceptions consecutively in slot and
            # column order; each record's rank is its offset in that run.
            expert, tile, slot, column = position.unbind(1)
            light = torch.where(heavy, 0, count).view(-1)
            first = torch.cumsum(light, 0) - light
            rank = torch.arange(position.shape[0], device=device)
            rank = rank - first[expert * tiles + tile]
            word = (
                slot | (column << 8) | (values[expert, tile, slot, column].long() << 16)
            )
            records.append((expert + e0, tile, rank, word))
    heavy = counts > RECORDS
    heavy_tiles = int(heavy.sum())
    header = torch.where(
        heavy, HEAVY + torch.cumsum(heavy.view(-1), 0).view(experts, tiles) - 1, counts
    )
    words = tile_blocks.view(torch.int32)
    words[..., SELECTOR_BYTES // 4] = _as_int32_bits(header)
    for expert, tile, rank, word in records:
        words[expert, tile, SELECTOR_BYTES // 4 + 1 + rank] = _as_int32_bits(word)
    tiles_bytes = experts * tiles * TILE_BYTES
    bases_end = tiles_bytes + experts * blocks * BASE_BYTES
    storage = torch.empty(
        bases_end + heavy_tiles * RAW_TILE_BYTES, dtype=torch.uint8, device=device
    )
    storage[:tiles_bytes] = tile_blocks.view(-1)
    storage[tiles_bytes:bases_end] = bases.view(-1)
    if heavy_tiles:
        storage[bases_end:] = torch.cat(raw).view(-1)
    return storage, heavy_tiles


def decode_mxfp4_csf_inline(
    plane: Mxfp4CsfInlinePlane, first: int = 0, last: int | None = None
) -> torch.Tensor:
    """Native compact scale bytes ``[last - first, rows * columns]`` of a plane."""
    blocks, k_tiles = plane.geometry
    last = plane.num_experts if last is None else int(last)
    experts, tiles = last - first, blocks * k_tiles
    storage = plane.storage
    device = storage.device
    offsets = torch.from_numpy(
        _slot_offsets(plane.rows, plane.columns, plane.group_rows)
    ).to(device)
    tile_blocks = storage[: plane.tiles_bytes].view(
        plane.num_experts, tiles, TILE_BYTES
    )
    tile_blocks = tile_blocks[first:last]
    bases = storage[plane.tiles_bytes : plane.tiles_bytes + plane.bases_bytes]
    bases = bases.view(plane.num_experts, blocks, 1, 128)[first:last]
    bases = bases.expand(experts, blocks, k_tiles, 128).reshape(experts, tiles, 128, 1)
    nibbles = tile_blocks[..., :SELECTOR_BYTES]
    nibbles = torch.stack((nibbles & 15, nibbles >> 4), -1).view(experts, tiles, 128)
    column_bits = torch.arange(4, device=device, dtype=torch.uint8)
    values = bases + ((nibbles.unsqueeze(-1) >> column_bits) & 1)
    words = tile_blocks.reshape(-1).view(torch.int32).view(experts, tiles, -1)
    words = words[..., SELECTOR_BYTES // 4 :].long() & 0xFFFFFFFF
    header = words[..., 0]
    heavy = header >= HEAVY
    if bool(heavy.any()):
        raw = storage[plane.tiles_bytes + plane.bases_bytes :].view(-1, 128, 4)
        values[heavy] = raw[header[heavy] - HEAVY]
    for record in range(RECORDS):
        live = ~heavy & (header > record)
        if not bool(live.any()):
            continue
        expert, tile = live.nonzero().unbind(1)
        word = words[expert, tile, 1 + record]
        values[expert, tile, word & 127, (word >> 8) & 3] = ((word >> 16) & 255).to(
            torch.uint8
        )
    target = offsets.view(-1)
    keep = target >= 0
    native = torch.zeros(
        experts, plane.rows * plane.columns, dtype=torch.uint8, device=device
    )
    native[:, target[keep]] = values.view(experts, -1)[:, keep]
    return native


@tr.jit
def _expand_tiles(
    Storage,
    Words,
    NativeWords,
    NativeHalves,
    total,
    TILES_PER_EXPERT: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCKS: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    PLANE_BYTES: tl.constexpr,
    TILES_BYTES: tl.constexpr,
    RAW_WORDS: tl.constexpr,
    TILES: tl.constexpr,
):
    """Native words of ``TILES`` tile blocks: rows are tiles, columns are slots."""
    index = tl.program_id(0).to(tl.int64) * TILES + tl.arange(0, TILES)
    live = index < total
    expert, tile = index // TILES_PER_EXPERT, index % TILES_PER_EXPERT
    block, k_tile = tile // K_TILES, tile % K_TILES
    group, group_tile = block // TILES_PER_GROUP, block % TILES_PER_GROUP
    tile_rows = tl.minimum(128, GROUP_ROWS - group_tile * 128)
    tile_columns = tl.minimum(4, COLUMNS - k_tile * 4)
    first = (
        expert * PLANE_BYTES
        + group * (GROUP_ROWS * COLUMNS)
        + group_tile * (128 * COLUMNS)
        + k_tile * tile_rows * 4
    )
    header = tl.load(Words + index * 24 + 16, live, 0)
    heavy = header < 0
    count = tl.where(heavy, 0, header)
    slot = tl.arange(0, 128)[None, :]
    block_words = (index * 24)[:, None]
    present = live[:, None]
    selectors = tl.load(Words + block_words + slot // 8, present, 0).to(tl.uint32)
    selectors = (selectors >> ((slot % 8) * 4).to(tl.uint32)) & 15
    bases = TILES_BYTES + (expert * BLOCKS + block) * 128
    base = tl.load(Storage + bases[:, None] + slot, present, 0)
    word = base.to(tl.uint32) * 0x01010101 + ((selectors * 0x00204081) & 0x01010101)
    for record in tl.static_range(7):
        active = live & (record < count)
        patch = tl.load(Words + index * 24 + 17 + record, active, 0)[:, None]
        shift = (((patch >> 8) & 3) * 8).to(tl.uint32)
        value = ((patch >> 16) & 255).to(tl.uint32) << shift
        hit = active[:, None] & (slot == (patch & 127))
        word = tl.where(
            hit, (word & ~(tl.full((), 255, tl.uint32) << shift)) | value, word
        )
    raw_tile = (RAW_WORDS + (header.to(tl.int64) & 0x7FFFFFFF) * 128)[:, None]
    raw = tl.load(Words + raw_tile + slot, (live & heavy)[:, None], 0)
    word = tl.where(heavy[:, None], raw.to(tl.uint32), word)
    row = (slot // 32) * 32 + (slot % 4) * 8 + (slot // 4) % 8
    offset = first[:, None] + row * tile_columns[:, None]
    keep = present & (row < tile_rows[:, None])
    full = (tile_columns == 4)[:, None]
    tl.store(NativeWords + offset // 4, word.to(tl.int32, bitcast=True), keep & full)
    # K tail tiles hold two columns: the low half of each word.
    half = (word & 0xFFFF).to(tl.uint16).to(tl.int16, bitcast=True)
    tl.store(NativeHalves + offset // 2, half, keep & ~full)


def expansion_payload(plane: Mxfp4CsfInlinePlane) -> dict[str, int]:
    """Program identity of a plane's expansion: geometry only, never its contents."""
    blocks, k_tiles = plane.geometry
    return {
        "total": plane.num_experts * blocks * k_tiles,
        "TILES_PER_EXPERT": blocks * k_tiles,
        "K_TILES": k_tiles,
        "BLOCKS": blocks,
        "TILES_PER_GROUP": (plane.group_rows + 127) // 128,
        "GROUP_ROWS": plane.group_rows,
        "COLUMNS": plane.columns,
        "PLANE_BYTES": plane.rows * plane.columns,
        "TILES_BYTES": plane.tiles_bytes,
        "RAW_WORDS": (plane.tiles_bytes + plane.bases_bytes) // 4,
    }


_TILES_PER_PROGRAM = 8


def compile_expansion(payload: dict[str, int]):
    """Compile one plane geometry's expansion ahead of serving."""
    from triton.runtime.jit import MockTensor

    payload = dict(payload)
    total = int(payload.pop("total"))
    storage_bytes = payload["RAW_WORDS"] * 4 + RAW_TILE_BYTES
    native_bytes = total // payload["TILES_PER_EXPERT"] * payload["PLANE_BYTES"]
    return _expand_tiles.warmup(
        MockTensor(torch.uint8, (storage_bytes,)),
        MockTensor(torch.int32, (storage_bytes // 4,)),
        MockTensor(torch.int32, (native_bytes // 4,)),
        MockTensor(torch.int16, (native_bytes // 2,)),
        total,
        **payload,
        TILES=_TILES_PER_PROGRAM,
        num_warps=4,
        grid=(triton_cdiv(total, _TILES_PER_PROGRAM),),
    )


def expand_mxfp4_csf_inline(plane: Mxfp4CsfInlinePlane, native: torch.Tensor) -> None:
    """Write every expert's native compact scale bytes into ``native``.

    ``native`` is the ``[E, rows * columns]`` uint8 view of the compact scale
    plane the storage was built from. Calls above the inline token limit run
    the native kernels over it.
    """
    if native.dtype != torch.uint8 or not native.is_contiguous():
        raise TypeError("inline W4A8 expansion requires contiguous uint8 storage")
    if native.shape != (plane.num_experts, plane.rows * plane.columns):
        raise ValueError("inline W4A8 expansion target does not match the plane")
    if native.device != plane.storage.device:
        raise ValueError("inline W4A8 expansion storage must share one device")
    from b12x._lib.compile_plan import launch_triton

    payload = expansion_payload(plane)
    total = payload.pop("total")
    # Fused-MoE preparation compiles this program (compile_expansion); frozen
    # serving launches it without lowering.
    launch_triton(
        _expand_tiles,
        (triton_cdiv(total, _TILES_PER_PROGRAM),),
        plane.storage,
        plane.storage.view(torch.int32),
        native.view(torch.int32),
        native.view(torch.int16),
        total,
        **payload,
        TILES=_TILES_PER_PROGRAM,
        num_warps=4,
    )


@cute.jit
def stage_inline_tile(
    storage: Int64,
    tile: Int64,
    dst: Int32,
    tid: Int32,
    first: cutlass.Constexpr = 0,
):
    """Copy one tile block to shared memory with six 16-byte cp.async."""
    lane = tid - Int32(first)
    if lane >= Int32(0) and lane < Int32(TILE_BYTES // 16):
        cp_async4_shared_global(
            dst + lane * Int32(16),
            storage + tile * Int64(TILE_BYTES) + Int64(lane * Int32(16)),
        )


@cute.jit
def inline_row_bases(
    storage: Int64, tiles_bytes: Int64, block: Int64, slot0: Int32
) -> Uint32:
    """The four row bases of slots ``slot0 .. slot0 + 3`` of a row block."""
    return ld_global_nc_u32(
        storage + tiles_bytes + block * Int64(BASE_BYTES) + Int64(slot0)
    )


@cute.jit
def inline_scale_words(tile: Int32, bases: Uint32, slot0: Int32, raw: Int64):
    """Native words of slots ``slot0 .. slot0 + 3`` from a staged tile block.

    ``tile`` is the shared address of the block, ``bases`` the slots' row bases
    and ``raw`` the global address of the plane's raw tiles. Records patch the
    words in registers; a heavy tile reads its raw words from global memory.
    Returns four ``Uint32`` values, one per ``nt``.
    """
    shift = Uint32(slot0 & Int32(4)) << Uint32(2)
    selectors = ld_shared_u32(tile + ((slot0 >> Int32(3)) << Int32(2))) >> shift
    spread = Uint32(0x00204081)
    ones = Uint32(0x01010101)
    w0 = (bases & Uint32(0xFF)) * ones + ((selectors & Uint32(0xF)) * spread & ones)
    w1 = ((bases >> Uint32(8)) & Uint32(0xFF)) * ones + (
        ((selectors >> Uint32(4)) & Uint32(0xF)) * spread & ones
    )
    w2 = ((bases >> Uint32(16)) & Uint32(0xFF)) * ones + (
        ((selectors >> Uint32(8)) & Uint32(0xF)) * spread & ones
    )
    w3 = (bases >> Uint32(24)) * ones + (
        ((selectors >> Uint32(12)) & Uint32(0xF)) * spread & ones
    )
    header = ld_shared_u32(tile + Int32(SELECTOR_BYTES))
    if header != Uint32(0):
        if (header & Uint32(HEAVY)) != Uint32(0):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(
                raw
                + Int64(header & Uint32(HEAVY - 1)) * Int64(RAW_TILE_BYTES)
                + Int64(slot0 * Int32(4))
            )
        else:
            for index in cutlass.range_constexpr(RECORDS):
                if Uint32(index) < header:
                    record = ld_shared_u32(tile + Int32(SELECTOR_BYTES + 4 + 4 * index))
                    slot = Int32(record & Uint32(127)) - slot0
                    byte_shift = ((record >> Uint32(8)) & Uint32(3)) << Uint32(3)
                    value = ((record >> Uint32(16)) & Uint32(0xFF)) << byte_shift
                    keep = (Uint32(0xFF) << byte_shift) ^ Uint32(0xFFFFFFFF)
                    w0 = Uint32(
                        cutlass.select_(slot == Int32(0), (w0 & keep) | value, w0)
                    )
                    w1 = Uint32(
                        cutlass.select_(slot == Int32(1), (w1 & keep) | value, w1)
                    )
                    w2 = Uint32(
                        cutlass.select_(slot == Int32(2), (w2 & keep) | value, w2)
                    )
                    w3 = Uint32(
                        cutlass.select_(slot == Int32(3), (w3 & keep) | value, w3)
                    )
    return w0, w1, w2, w3


__all__ = [
    "HEAVY",
    "Mxfp4CsfInlinePlane",
    "RECORDS",
    "TILE_BYTES",
    "build_mxfp4_csf_inline",
    "decode_mxfp4_csf_inline",
    "compile_expansion",
    "expand_mxfp4_csf_inline",
    "expansion_payload",
    "inline_geometry",
    "inline_row_bases",
    "inline_scale_words",
    "stage_inline_tile",
]
