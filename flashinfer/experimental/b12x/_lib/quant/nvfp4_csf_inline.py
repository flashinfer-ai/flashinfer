"""Read losslessly compressed NVFP4 scales into registers.

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
from b12x._lib.intrinsics import cp_async_bulk_g2s_mbar

_HEADER_BYTES = 1024


@dsl_user_op
def _expect_payload(barrier, byte_count, *, loc=None, ip=None):
    """Add payload bytes before issuing the scale transactions for this stage."""
    llvm.inline_asm(
        None,
        [barrier.toint().ir_value(), byte_count.ir_value()],
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _shared_scale_word(stage_address, word, records_address, mask_partial=False, *, loc=None, ip=None):
    """Keep bitmap lookup temporaries local to one scale-register definition."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [stage_address.ir_value(), word.ir_value(), records_address.ir_value()],
            """{
        .reg .b32 address, packed, shifted, base, tile, group, bit;
        .reg .b32 metadata, mask, flag, test, first, prefixes, rank, origin;
        .reg .b64 offset, replacement;
        .reg .pred absent;
        mad.lo.u32 address, $2, 2, $1;
        ld.shared.u16 packed, [address];
        shr.u32 shifted, packed, 4;
        prmt.b32 packed, packed, shifted, 0x5140;
        and.b32 packed, packed, 0x0f0f0f0f;
        and.b32 address, $2, 127;
        add.u32 address, address, 512;
        add.u32 address, address, $1;
        ld.shared.u8 base, [address];
        prmt.b32 base, base, base, 0;
        add.u32 $0, packed, base;
        shr.u32 tile, $2, 7;
        mad.lo.u32 metadata, tile, 32, $1;
        add.u32 metadata, metadata, 640;
        shr.u32 group, $2, 5;
        and.b32 group, group, 3;
        mad.lo.u32 address, group, 4, metadata;
        ld.shared.u32 mask, [address+8];
        and.b32 bit, $2, 31;
        mov.u32 flag, 1;
        shl.b32 flag, flag, bit;
        and.b32 test, mask, flag;
        setp.eq.u32 absent, test, 0;
        @absent bra CSF_WORD_DONE;
        ld.shared.u32 first, [metadata];
        ld.shared.u32 prefixes, [metadata+4];
        shl.b32 group, group, 3;
        shr.u32 prefixes, prefixes, group;
        and.b32 prefixes, prefixes, 255;
        sub.u32 flag, flag, 1;
        and.b32 mask, mask, flag;
        popc.b32 rank, mask;
        add.u32 rank, rank, prefixes;
        add.u32 first, first, rank;
        ld.shared.u32 origin, [$1+640];
        and.b32 origin, origin, 0xfffffffc;
        sub.u32 rank, first, origin;
        setp.ge.u32 absent, rank, 80;
        @absent bra CSF_WORD_GLOBAL;
        mad.lo.u32 address, rank, 4, $1;
        ld.shared.u32 $0, [address+704];
        bra CSF_WORD_DONE;
        CSF_WORD_GLOBAL:
        cvt.u64.u32 offset, first;
        shl.b64 offset, offset, 2;
        add.u64 replacement, $3, offset;
        ld.global.u32 $0, [replacement];
        CSF_WORD_DONE:
        """ + ("""
        ld.shared.u32 test, [metadata+28];
        setp.eq.u32 absent, test, 0;
        selp.b32 $0, $0, 0, absent;
        """ if mask_partial else "") + "}",
            "=&r,r,r,l,~{memory}",
            # Operand slots are reused after pipeline barriers. This read must
            # not be eliminated or hoisted when its pointer repeats.
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _load_at(pointer, offset: Int64):
    return cute.make_tensor(pointer + offset, cute.make_layout(1))[0]


@dsl_user_op
def _global_scale_vector(codes, bases, metadata, records, lane_word, *, loc=None, ip=None):
    """Decode four adjacent words with shared address and exception metadata."""
    unpack = "\n".join(
        f"""
        shr.u32 packed, code{index // 2}, {16 * (index % 2)};
        shr.u32 shifted, packed, 4;
        prmt.b32 packed, packed, shifted, 0x5140;
        and.b32 packed, packed, 0x0f0f0f0f;
        prmt.b32 base, base_word, base_word, 0x{index}{index}{index}{index};
        add.u32 ${index}, packed, base;
        """ for index in range(4)
    )
    exceptions = "\n".join(
        f"""
        and.b32 test, selected, {1 << index};
        setp.ne.u32 present, test, 0;
        @present ld.global.u32 ${index}, [address];
        @present add.u64 address, address, 4;
        """ for index in range(4)
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [codes.ir_value(), bases.ir_value(), metadata.ir_value(), records.ir_value(),
         lane_word.ir_value()],
        """{
        .reg .b32 code0, code1, base_word, packed, shifted, base;
        .reg .b32 group, bit, mask, selected, flag, first, prefixes, rank, test;
        .reg .b64 address, offset;
        .reg .pred present;
        ld.global.v2.u32 {code0, code1}, [$4];
        ld.global.u32 base_word, [$5];
        """ + unpack + """
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
        """ + exceptions + """
        CSF_VECTOR_DONE:
        }""",
        "=&r,=&r,=&r,=&r,l,l,l,l,r,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def _store_scale_vector(address, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [x.ir_value() for x in (address, a, b, c, d)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dataclass(frozen=True)
class InlineNvfp4Scales:
    storage: torch.Tensor
    rows: int
    columns: int
    num_experts: int


def prepare_inline_scales(batch):
    """Partition exceptions by the scale tile consumed by one K64 MMA slice."""
    batch.validate()
    if batch.codec != 0 or batch.layout != 0:
        raise ValueError("Inline NVFP4 requires native-order byte-window scales")
    e, r, c = batch.num_experts, batch.rows, batch.columns
    tiles = r * c // 512
    records = batch.exceptions.cpu().numpy().view("<u4")
    expert_bounds = batch.task_offsets.cpu().numpy()[:, [0, -1]]
    fixed = batch.fixed.cpu().numpy().reshape(-1)
    fixed16 = fixed.view("<u2")
    metadata, payloads = [], []
    payload_count = 0
    for expert, (first, last) in enumerate(expert_bounds):
        words = records[first:last]
        position = words & 0xFFFFFF
        row, col = position // c, position % c
        native = (
            ((row // 128 * (c // 4) + col // 4) * 32 + row % 32) * 16
            + (row % 128 // 32) * 4
            + col % 4
        )
        affected, inverse = np.unique(native // 4, return_inverse=True)
        slab = expert * (r // 128) + affected // (c // 4 * 128)
        slab_offset = slab * (128 * (1 + c // 2))
        column = (affected // 128) % (c // 4)
        lane = affected % 128
        packed = fixed16[(slab_offset + 128) // 2 + column * 128 + lane].astype(
            np.uint32
        )
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
        offsets = np.cumsum(counts.sum(axis=1), dtype=np.uint64)
        offsets = offsets - counts.sum(axis=1) + payload_count
        ends = offsets + counts.sum(axis=1)
        metadata.append(
            np.column_stack(
                (
                    offsets.astype(np.uint32),
                    prefixes,
                    masks,
                    ends.astype(np.uint32),
                    np.zeros(tiles, dtype=np.uint32),
                )
            ).astype("<u4")
        )
        payloads.append(values)
        payload_count += len(values)
    if payload_count >= 1 << 32:
        raise ValueError("NVFP4 inline exception count exceeds uint32 indexing")
    offsets = np.concatenate(metadata).reshape(-1).view(np.uint8)
    payload = np.concatenate(payloads).view(np.uint8)
    payload = np.pad(payload, (0, (-len(payload)) % 16))
    # A zero operand and invalid-tile marker support native TMA padding without
    # reading beyond a projection's final 128-row or 64-column scale atom.
    header = np.zeros(_HEADER_BYTES, dtype=np.uint8)
    header.view("<u4")[167] = 1
    storage = torch.from_numpy(np.concatenate((header, fixed, offsets, payload))).to(
        batch.fixed.device
    )
    return InlineNvfp4Scales(storage, r, c, e)


class InlineNvfp4Reader:
    def __init__(self, rows, columns):
        self.rows, self.columns = int(rows), int(columns)
        self.fixed_bytes = self.rows * (1 + self.columns // 2)
        self.tiles = self.rows * self.columns // 512

    @cute.jit
    def word(
        self,
        storage,
        experts: Int32,
        expert: Int32,
        row_tile: Int32,
        column_tile: Int32,
        lane_word: Int32,
    ):
        """Return four exact scale bytes in native F8_128x4 order."""
        return self.word_pointer(
            storage.iterator, experts, expert, row_tile, column_tile, lane_word
        )

    @cute.jit
    def word_pointer(
        self,
        storage,
        experts: Int32,
        expert: Int32,
        row_tile: Int32,
        column_tile: Int32,
        lane_word: Int32,
    ):
        storage = cute.recast_ptr(storage, dtype=cutlass.Uint8) + Int64(_HEADER_BYTES)
        words = cute.recast_ptr(storage, dtype=cutlass.Uint32)
        halves = cute.recast_ptr(storage, dtype=cutlass.Uint16)
        slab = Int64(expert) * Int64(self.rows // 128) + Int64(row_tile)
        fixed = slab * Int64(128 * (1 + self.columns // 2))
        code_word = Int64(column_tile) * Int64(128) + Int64(lane_word)
        packed = _load_at(halves, (fixed + Int64(128)) // Int64(2) + code_word).to(
            Uint32
        )
        value = (packed | (packed << Uint32(8))) & Uint32(0x00FF00FF)
        value = (value | (value << Uint32(4))) & Uint32(0x0F0F0F0F)
        value += _load_at(storage, fixed + Int64(lane_word)).to(Uint32) * Uint32(
            0x01010101
        )
        tile = slab * Int64(self.columns // 4) + Int64(column_tile)
        partitions = Int64(experts) * Int64(self.fixed_bytes // 4)
        records = partitions + Int64(experts) * Int64(self.tiles * 8)
        meta = partitions + tile * Int64(8)
        group, bit = lane_word // Int32(32), lane_word % Int32(32)
        mask = _load_at(words, meta + Int64(2) + Int64(group))
        flag = Uint32(1) << bit.to(Uint32)
        if (mask & flag) != Uint32(0):
            start = _load_at(words, meta).to(Int64)
            prefixes = _load_at(words, meta + Int64(1))
            rank = ((prefixes >> (group.to(Uint32) * Uint32(8))) & Uint32(255)).to(
                Int64
            )
            rank += cute.arch.popc(mask & (flag - Uint32(1))).to(Int64)
            value = _load_at(words, records + start + rank).to(Uint32)
        return value

    @cute.jit
    def prefetch_bounds(
        self,
        storage,
        experts: Int32,
        expert: Int32,
        outer: Int32,
        *,
        columns: cutlass.Constexpr,
    ):
        """Distribute a complete operand sweep's payload bounds across a warp."""
        extent = (self.columns + 7) // 8 if columns else self.rows // 128
        if cutlass.const_expr(extent <= 32):
            lane = Int32(cute.arch.lane_idx())
            first, end = Uint32(0), Uint32(0)
            row = outer if cutlass.const_expr(columns) else lane
            col = lane if cutlass.const_expr(columns) else outer
            if lane < Int32(extent) and row < Int32(self.rows // 128):
                tile = (Int64(expert) * Int64(self.rows // 128) + Int64(row)) * Int64(
                    self.columns // 4
                ) + Int64(col) * Int64(2)
                address = Int64(experts) * Int64(self.fixed_bytes // 4) + tile * Int64(
                    8
                )
                words = cute.recast_ptr(storage, dtype=cutlass.Uint32) + Int64(_HEADER_BYTES // 4)
                first = _load_at(words, address) & Uint32(0xFFFFFFFC)
                end_word = Int64(14)
                if cutlass.const_expr(self.columns % 8):
                    if col == Int32(self.columns // 8):
                        end_word = Int64(6)
                end = (_load_at(words, address + end_word) + Uint32(3)) & Uint32(0xFFFFFFFC)
            return (first, end)
        else:
            return None

    @cute.jit
    def stage(
        self,
        storage,
        experts: Int32,
        expert: Int32,
        row_tile: Int32,
        k_tile: Int32,
        shared,
        stage: Int32,
        barrier,
        bounds=None,
        bound_index: Int32 = 0,
    ):
        """Stage packed scales, bitmaps and up to 320 bytes of replacement words.

        The native 1024-byte shared slot retains its stride. Its contents
        follow the compressed storage contract until the consumer constructs
        scale registers. The operand barrier accounts for 704 fixed bytes plus
        the aligned payload extent. Dense exception tails use global reads.
        """
        lane = Int32(cute.arch.lane_idx())
        target = cute.recast_ptr(shared, dtype=cutlass.Uint8) + stage * Int32(1024)
        zero = cute.recast_ptr(storage, dtype=cutlass.Uint8)
        origin = zero + Int64(_HEADER_BYTES)
        if cutlass.const_expr(bounds is not None):
            cached_first = cute.arch.shuffle_sync(bounds[0], bound_index)
            cached_end = cute.arch.shuffle_sync(bounds[1], bound_index)
        if lane == Int32(0):
            if row_tile >= Int32(self.rows // 128):
                cp_async_bulk_g2s_mbar(target.toint(), zero.toint(), Int32(704), barrier.toint())
            else:
                slab = Int64(expert) * Int64(self.rows // 128) + Int64(row_tile)
                fixed = slab * Int64(128 * (1 + self.columns // 2))
                tile = slab * Int64(self.columns // 4) + Int64(k_tile) * Int64(2)
                meta = Int64(experts) * Int64(self.fixed_bytes) + tile * Int64(32)
                metadata = cute.recast_ptr(origin + meta, dtype=cutlass.Uint32)
                if cutlass.const_expr(bounds is not None):
                    first, end = cached_first, cached_end
                else:
                    first = _load_at(metadata, Int64(0)) & Uint32(0xFFFFFFFC)
                    end_word = Int64(14)
                    if cutlass.const_expr(self.columns % 8):
                        if k_tile == Int32(self.columns // 8):
                            end_word = Int64(6)
                    end = (_load_at(metadata, end_word) + Uint32(3)) & Uint32(0xFFFFFFFC)
                count = cutlass.min((end - first).to(Int32), Int32(80)) * Int32(4)
                records = Int64(experts) * Int64(self.fixed_bytes + self.tiles * 32)
                if count > Int32(0):
                    _expect_payload(barrier, count)
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(704)).toint(),
                        (origin + records + first.to(Int64) * Int64(4)).toint(),
                        count,
                        barrier.toint(),
                    )
                if cutlass.const_expr(self.columns % 8):
                    second_atom = k_tile < Int32(self.columns // 8)
                    cp_async_bulk_g2s_mbar(
                        target.toint(),
                        (origin + fixed + Int64(128) + Int64(k_tile) * Int64(512)).toint(),
                        Int32(256), barrier.toint(),
                    )
                    second_codes = zero.toint()
                    second_metadata = (zero + Int64(640)).toint()
                    if second_atom:
                        second_codes = (origin + fixed + Int64(384) + Int64(k_tile) * Int64(512)).toint()
                        second_metadata = (origin + meta + Int64(32)).toint()
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(256)).toint(), second_codes,
                        Int32(256), barrier.toint(),
                    )
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(512)).toint(), (origin + fixed).toint(),
                        Int32(128), barrier.toint(),
                    )
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(640)).toint(), (origin + meta).toint(),
                        Int32(32), barrier.toint(),
                    )
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(672)).toint(), second_metadata,
                        Int32(32), barrier.toint(),
                    )
                else:
                    cp_async_bulk_g2s_mbar(
                        target.toint(),
                        (origin + fixed + Int64(128) + Int64(k_tile) * Int64(512)).toint(),
                        Int32(512),
                        barrier.toint(),
                    )
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(512)).toint(),
                        (origin + fixed).toint(),
                        Int32(128),
                        barrier.toint(),
                    )
                    cp_async_bulk_g2s_mbar(
                        (target + Int32(640)).toint(),
                        (origin + meta).toint(),
                        Int32(64),
                        barrier.toint(),
                    )

    @cute.jit
    def shared_word(self, stage_address, word: Int32, records_address):
        return _shared_scale_word(
            stage_address, word, records_address, mask_partial=bool(self.columns % 8)
        )

    @cute.jit
    def _operand_word(self, ordinal: Int32, row_half: cutlass.Constexpr):
        if cutlass.const_expr(row_half < 0):
            return ordinal
        # F8_128x4 interleaves four groups of 32 rows. A 64-row half
        # occupies two adjacent words in each four-word row group.
        return (
            (ordinal // Int32(64)) * Int32(128)
            + (ordinal % Int32(64) // Int32(2)) * Int32(4)
            + ordinal % Int32(2) + Int32(row_half * 2)
        )

    @cute.jit
    def _read_shared_operand(
        self, target, records_address, thread: Int32,
        threads: cutlass.Constexpr, row_half: cutlass.Constexpr,
    ):
        count = 256 if row_half < 0 else 128
        iterations = (count + threads - 1) // threads
        values = cute.make_rmem_tensor(cute.make_layout(iterations), cutlass.Uint32)
        for index in cutlass.range_constexpr(iterations):
            ordinal = thread + Int32(index * threads)
            if ordinal < Int32(count):
                values[index] = self.shared_word(
                    target.toint(), self._operand_word(ordinal, row_half), records_address
                )
        return values

    @cute.jit
    def _store_shared_operand(
        self, target, values, thread: Int32,
        threads: cutlass.Constexpr, row_half: cutlass.Constexpr,
    ):
        count = 256 if row_half < 0 else 128
        output = cute.make_tensor(
            cute.recast_ptr(target, dtype=cutlass.Uint32), cute.make_layout(256)
        )
        for index in cutlass.range_constexpr(cute.size(values)):
            ordinal = thread + Int32(index * threads)
            if ordinal < Int32(count):
                output[self._operand_word(ordinal, row_half)] = values[index]

    @cute.jit
    def expand_shared(
        self, storage, experts: Int32, shared, stage: Int32,
        threads: cutlass.Constexpr, barrier, second_shared=None, second_stage: Int32 = 0,
        row_half: cutlass.Constexpr = -1, second_row_half: cutlass.Constexpr = -1,
    ):
        """Reconstruct one or two ready operands using two consumer barriers.

        All lanes retain every required input before either slot is overwritten.
        A split gate operand needs the upper 64 rows of its first atom and the
        lower 64 rows of its second atom. Other rows remain compressed and must
        not be consumed by that MMA. Every consumer still reaches both barriers.
        """
        thread, _, _ = cute.arch.thread_idx()
        target = cute.recast_ptr(shared, dtype=cutlass.Uint8) + stage * Int32(1024)
        records_address = storage.toint() + Int64(_HEADER_BYTES) + Int64(experts) * Int64(
            self.fixed_bytes + self.tiles * 32
        )
        values = self._read_shared_operand(
            target, records_address, Int32(thread), threads, row_half
        )
        if cutlass.const_expr(second_shared is not None):
            second_target = cute.recast_ptr(second_shared, dtype=cutlass.Uint8) + second_stage * Int32(1024)
            second_values = self._read_shared_operand(
                second_target, records_address, Int32(thread), threads, second_row_half
            )
        barrier.arrive_and_wait()
        self._store_shared_operand(target, values, Int32(thread), threads, row_half)
        if cutlass.const_expr(second_shared is not None):
            self._store_shared_operand(
                second_target, second_values, Int32(thread), threads, second_row_half
            )
        barrier.arrive_and_wait()


class IndexedNvfp4Plane(InlineNvfp4Reader):
    """Expand equal-size native spans using the prepared exception-word index."""

    def __init__(self, geometry):
        rows, columns, codec, layout = geometry
        if codec != 0 or layout != 0:
            raise ValueError("Indexed expansion requires native-order byte-window scales")
        super().__init__(rows, columns)
        self.tasks = (self.rows * self.columns + 4095) // 4096

    @cute.jit
    def tensors(self, fixed, exceptions, offsets, output, lut, experts, exception_bytes):
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
            codes = origin + fixed + Int64(128) + (
                Int64(column) * Int64(128) + Int64(lane_word)
            ) * Int64(2)
            bases = origin + fixed + Int64(lane_word)
            metadata = origin + partitions + tile * Int64(32)
            records = origin + partitions + Int64(experts) * Int64(self.tiles * 32)
            values = _global_scale_vector(codes, bases, metadata, records, lane_word)
            destination = (Int64(expert) * Int64(self.rows * self.columns)
                           + Int64(word) * Int64(4))
            _store_scale_vector(output.toint() + destination, *values)
