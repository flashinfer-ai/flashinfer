"""Native-order NVFP4 expansion with a tile-local exception-word index.

The prepared buffer contains native-order row bases and four-bit offsets,
followed by bitmaps that identify four-byte scale words containing exceptions.
A rank lookup retrieves each complete replacement word without scanning an
exception list. Checkpoint tensors and their lossless codec are unchanged.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32, T, dsl_user_op
from cutlass._mlir.dialects import llvm

_HEADER_BYTES = 1024


@dsl_user_op
def _global_scale_vector(
    codes, bases, metadata, records, lane_word, *, loc=None, ip=None
):
    """Decode four adjacent words with shared address and exception metadata."""
    unpack = "\n".join(
        f"""
        shr.u32 packed, code{index // 2}, {16 * (index % 2)};
        shr.u32 shifted, packed, 4;
        prmt.b32 packed, packed, shifted, 0x5140;
        and.b32 packed, packed, 0x0f0f0f0f;
        prmt.b32 base, base_word, base_word, 0x{index}{index}{index}{index};
        add.u32 ${index}, packed, base;
        """
        for index in range(4)
    )
    exceptions = "\n".join(
        f"""
        and.b32 test, selected, {1 << index};
        setp.ne.u32 present, test, 0;
        @present ld.global.u32 ${index}, [address];
        @present add.u64 address, address, 4;
        """
        for index in range(4)
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            codes.ir_value(),
            bases.ir_value(),
            metadata.ir_value(),
            records.ir_value(),
            lane_word.ir_value(),
        ],
        """{
        .reg .b32 code0, code1, base_word, packed, shifted, base;
        .reg .b32 group, bit, mask, selected, flag, first, prefixes, rank, test;
        .reg .b64 address, offset;
        .reg .pred present;
        ld.global.v2.u32 {code0, code1}, [$4];
        ld.global.u32 base_word, [$5];
        """
        + unpack
        + """
        shr.u32 group, $8, 5;
        mad.wide.u32 address, group, 4, $6;
        ld.global.u32 mask, [address+8];
        and.b32 bit, $8, 31;
        shr.u32 selected, mask, bit;
        and.b32 selected, selected, 15;
        setp.eq.u32 present, selected, 0;
        @present bra CSF_VECTOR_DONE;
        ld.global.v2.u32 {first, prefixes}, [$6];
        shl.b32 group, group, 3;
        shr.u32 prefixes, prefixes, group;
        and.b32 prefixes, prefixes, 255;
        mov.u32 flag, 1;
        shl.b32 flag, flag, bit;
        sub.u32 flag, flag, 1;
        and.b32 mask, mask, flag;
        popc.b32 rank, mask;
        add.u32 first, first, prefixes;
        add.u32 first, first, rank;
        cvt.u64.u32 offset, first;
        shl.b64 offset, offset, 2;
        add.u64 address, $7, offset;
        """
        + exceptions
        + """
        CSF_VECTOR_DONE:
        }""",
        "=&r,=&r,=&r,=&r,l,l,l,l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _store_scale_vector(address, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [x.ir_value() for x in (address, a, b, c, d)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dataclass(frozen=True)
class IndexedNvfp4Scales:
    storage: torch.Tensor
    rows: int
    columns: int
    num_experts: int


def prepare_indexed_scales(batch):
    """Partition exceptions by the scale tile consumed by one K64 MMA slice."""
    batch.validate()
    if batch.codec != 0 or batch.layout != 0:
        raise ValueError(
            "Indexed NVFP4 expansion requires native-order byte-window scales"
        )
    e, r, c = batch.num_experts, batch.rows, batch.columns
    tiles = e * r * c // 512
    records = batch.exceptions.cpu().numpy().view("<u4")
    expert_bounds = batch.task_offsets.cpu().numpy()[:, [0, -1]]
    fixed = batch.fixed.cpu().numpy().reshape(-1)
    fixed16 = fixed.view("<u2")
    lengths = expert_bounds[:, 1] - expert_bounds[:, 0]
    experts = np.repeat(np.arange(e, dtype=np.int64), lengths)
    words = np.concatenate([records[first:last] for first, last in expert_bounds])
    position = words & 0xFFFFFF
    row, col = position // c, position % c
    native = (
        ((row // 128 * (c // 4) + col // 4) * 32 + row % 32) * 16
        + (row % 128 // 32) * 4
        + col % 4
    )
    # Global word indices keep expert payloads and tile partitions disjoint.
    affected, inverse = np.unique(
        experts * (r * c // 4) + native // 4, return_inverse=True
    )
    slab_offset = (affected // (c // 4 * 128)) * (128 * (1 + c // 2))
    column = (affected // 128) % (c // 4)
    lane = affected % 128
    packed = fixed16[(slab_offset + 128) // 2 + column * 128 + lane].astype(np.uint32)
    values = (packed | (packed << 8)) & 0x00FF00FF
    values = (values | (values << 4)) & 0x0F0F0F0F
    values += fixed[slab_offset + lane].astype(np.uint32) * 0x01010101
    values = values.astype("<u4")
    values.view(np.uint8)[inverse * 4 + native % 4] = (words >> 24).astype(np.uint8)
    tile = affected // 128
    group = (affected % 128) // 32
    masks = np.zeros((tiles, 4), dtype=np.uint32)
    np.bitwise_or.at(
        masks, (tile, group), np.uint32(1) << (affected % 32).astype(np.uint32)
    )
    counts = np.bincount(tile * 4 + group, minlength=tiles * 4).reshape(tiles, 4)
    prefix = np.cumsum(counts, axis=1, dtype=np.uint32) - counts.astype(np.uint32)
    prefixes = np.sum(
        prefix << (np.arange(4, dtype=np.uint32) * 8), axis=1, dtype=np.uint32
    )
    sizes = counts.sum(axis=1)
    ends = np.cumsum(sizes, dtype=np.uint64)
    offsets = ends - sizes
    metadata = np.column_stack(
        (
            offsets.astype(np.uint32),
            prefixes,
            masks,
            ends.astype(np.uint32),
            np.zeros(tiles, dtype=np.uint32),
        )
    ).astype("<u4")
    payload_count = len(values)
    if payload_count >= 1 << 32:
        raise ValueError("NVFP4 indexed exception count exceeds uint32 indexing")
    offsets = metadata.reshape(-1).view(np.uint8)
    payload = values.view(np.uint8)
    payload = np.pad(payload, (0, (-len(payload)) % 16))
    # A zero operand and invalid-tile marker support native TMA padding without
    # reading beyond a projection's final 128-row or 64-column scale atom.
    header = np.zeros(_HEADER_BYTES, dtype=np.uint8)
    header.view("<u4")[167] = 1
    storage = torch.from_numpy(np.concatenate((header, fixed, offsets, payload))).to(
        batch.fixed.device
    )
    return IndexedNvfp4Scales(storage, r, c, e)


class IndexedNvfp4Plane:
    """Expand equal-size native spans using the prepared exception-word index."""

    def __init__(self, geometry):
        rows, columns, codec, layout = geometry
        if codec != 0 or layout != 0:
            raise ValueError(
                "Indexed expansion requires native-order byte-window scales"
            )
        self.rows, self.columns = int(rows), int(columns)
        self.fixed_bytes = self.rows * (1 + self.columns // 2)
        self.tiles = self.rows * self.columns // 512
        self.tasks = (self.rows * self.columns + 4095) // 4096

    @cute.jit
    def tensors(
        self, fixed, exceptions, offsets, output, lut, experts, exception_bytes
    ):
        return fixed, cute.recast_ptr(output, dtype=cutlass.Uint32), experts

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        storage, output, experts = tensors
        word = task * Int32(1024) + tid * Int32(4)
        if word < Int32(self.rows * self.columns // 4):
            row = word // Int32(128 * (self.columns // 4))
            column = (word // Int32(128)) % Int32(self.columns // 4)
            lane_word = word % Int32(128)
            origin = storage.toint() + Int64(_HEADER_BYTES)
            slab = Int64(expert) * Int64(self.rows // 128) + Int64(row)
            fixed = slab * Int64(128 * (1 + self.columns // 2))
            tile = slab * Int64(self.columns // 4) + Int64(column)
            partitions = Int64(experts) * Int64(self.fixed_bytes)
            codes = (
                origin
                + fixed
                + Int64(128)
                + (Int64(column) * Int64(128) + Int64(lane_word)) * Int64(2)
            )
            bases = origin + fixed + Int64(lane_word)
            metadata = origin + partitions + tile * Int64(32)
            records = origin + partitions + Int64(experts) * Int64(self.tiles * 32)
            values = _global_scale_vector(codes, bases, metadata, records, lane_word)
            destination = Int64(expert) * Int64(self.rows * self.columns) + Int64(
                word
            ) * Int64(4)
            _store_scale_vector(output.toint() + destination, *values)
