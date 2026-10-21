"""Pipeline-stage readers for the resident compact CSF scale payload."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics.packed import byte_perm
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    st_shared_v4_u32,
)
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.quant.csf_blocks import CompactBlockStoragePlane


class CsfInlineReader:
    def __init__(self, geometry):
        if len(geometry) != 11 or geometry[3] != 9:
            raise ValueError(
                "Inline CSF requires native codes and compact group records"
            )
        self.plane = CompactBlockStoragePlane(geometry)
        self.plane.shared_exceptions = True
        self.expert_bytes = self.plane.rows * self.plane.columns

    @cute.jit
    def stage_native(self, storage, offset, target, lane):
        assert self.plane.destination == 2
        if lane < Int32(32):
            expert = (offset // Int64(self.expert_bytes)).to(Int32)
            local = (offset % Int64(self.expert_bytes)).to(Int32)
            tile, half = local // Int32(1024), local % Int32(1024) // Int32(512)
            columns = Int32(self.plane.storage.column_groups)
            row_group = tile // columns * Int32(2) + half
            group = row_group * columns + tile % columns
            self.stage_compressed(storage, expert, group, target, lane)

    @cute.jit
    def expand_slots(
        self,
        storage,
        address,
        tid,
        threads: cutlass.Constexpr,
        slots: cutlass.Constexpr,
    ):
        warp, lane = tid // Int32(32), tid % Int32(32)
        for chunk in cutlass.range_constexpr(
            (slots + threads // 32 - 1) // (threads // 32)
        ):
            slot = warp + Int32(chunk * (threads // 32))
            if slot < Int32(slots):
                target = cute.make_ptr(
                    cutlass.Uint8,
                    address + slot * Int32(512),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                group = ld_shared_u32(target.toint().to(Int32) + Int32(500)).to(Int32)
                self.expand_compressed(storage, group, target, lane)
                self.pad_rows(group, target, lane)

    @cute.jit
    def stage_compressed(self, storage, expert, group, target, lane, bounds=None):
        """Stage codes, bases and the first 32 exceptions in a 512-byte slot.

        Callers publish the asynchronous copies before reconstructing the slot.
        Dense exception tails remain in the resident payload.
        """
        storage = cute.recast_ptr(storage, dtype=cutlass.Uint8)
        source_expert = expert * Int32(self.plane.sections) + group // Int32(
            self.plane.storage.groups
        )
        source_group = group % Int32(self.plane.storage.groups)
        row_group = source_group // Int32(self.plane.storage.column_groups)
        column_group = source_group % Int32(self.plane.storage.column_groups)
        nr = cutlass.min(
            Int32(128), Int32(self.plane.storage.rows) - row_group * Int32(128)
        )
        nc = cutlass.min(Int32(4), Int32(self.plane.columns) - column_group * Int32(4))
        valid = group < Int32(self.plane.storage.groups * self.plane.sections)
        first, end = Uint32(0), Uint32(0)
        if valid:
            if cutlass.const_expr(bounds is not None):
                first, end = bounds
            else:
                first, end = self.plane.group_record_bounds(
                    storage, source_expert, source_group
                )
        address = target.toint().to(Int32)
        value_start = self.plane.group_value_offset(storage) + first.to(Int64)
        if lane == Int32(0):
            st_shared_u32(address + Int32(384), first)
            st_shared_u32(address + Int32(388), end)
            st_shared_u32(address + Int32(496), (value_start & Int64(3)).to(Uint32))
            st_shared_u32(address + Int32(500), group.to(Uint32))
        codes = (
            Int64(self.plane.storage.codes)
            + source_expert.to(Int64) * Int64(self.plane.storage.code_bytes)
            + row_group.to(Int64)
            * Int64(128 * self.plane.columns * self.plane.storage.bits // 8)
            + (column_group * nr).to(Int64) * Int64(self.plane.storage.bits) // Int64(2)
        )
        bases = Int64(self.plane.storage.bases) + (
            source_expert * Int32(self.plane.storage.rows) + row_group * Int32(128)
        ).to(Int64)
        if lane < Int32(16):
            if valid and lane * Int32(16) < nr * nc * Int32(
                self.plane.storage.bits
            ) // Int32(8):
                cp_async4_shared_global(
                    address + lane * Int32(16),
                    (storage + codes + lane.to(Int64) * Int64(16)).toint(),
                )
            else:
                st_shared_v4_u32(
                    address + lane * Int32(16),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
        if lane < Int32(8):
            if valid and lane * Int32(16) < nr:
                cp_async4_shared_global(
                    address + Int32(256) + lane * Int32(16),
                    (storage + bases + lane.to(Int64) * Int64(16)).toint(),
                )
            else:
                st_shared_v4_u32(
                    address + Int32(256) + lane * Int32(16),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
        count = cutlass.min(end - first, Uint32(32))
        if lane < Int32(17) and lane.to(Uint32) * Uint32(2) < count + (
            first & Uint32(1)
        ):
            cp_async_u32_shared_global(
                address + Int32(392) + lane * Int32(4),
                (
                    storage
                    + Int64(self.plane.storage.source_records)
                    + (first & ~Uint32(1)).to(Int64) * Int64(2)
                    + lane.to(Int64) * Int64(4)
                ).toint(),
            )
        if lane < Int32(9) and lane.to(Uint32) * Uint32(4) < count + (
            value_start & Int64(3)
        ):
            cp_async_u32_shared_global(
                address + Int32(460) + lane * Int32(4),
                (
                    storage + (value_start & ~Int64(3)) + lane.to(Int64) * Int64(4)
                ).toint(),
            )

    @cute.jit
    def expand_compressed(self, storage, group, target, lane):
        """Reconstruct a ready compact slot and synchronize the decoding warp."""
        storage = cute.recast_ptr(storage, dtype=cutlass.Uint8)
        address = target.toint().to(Int32)
        first = ld_shared_u32(address + Int32(384))
        end = ld_shared_u32(address + Int32(388))
        position, replacement = Int32(0), Uint32(0)
        if first + lane.to(Uint32) < end:
            p = (first & Uint32(1)).to(Int32) + lane
            position = (
                (
                    ld_shared_u32(address + Int32(392) + p // Int32(2) * Int32(4))
                    >> ((p % Int32(2)) * Int32(16)).to(Uint32)
                )
                & Uint32(65535)
            ).to(Int32)
            v = ld_shared_u32(address + Int32(496)).to(Int32) + lane
            replacement = (
                ld_shared_u32(address + Int32(460) + v // Int32(4) * Int32(4))
                >> ((v % Int32(4)) * Int32(8)).to(Uint32)
            ) & Uint32(255)
        source_group = group % Int32(self.plane.storage.groups)
        nr = cutlass.min(
            Int32(128),
            Int32(self.plane.storage.rows)
            - source_group // Int32(self.plane.storage.column_groups) * Int32(128),
        )
        nc = cutlass.min(
            Int32(4),
            Int32(self.plane.columns)
            - source_group % Int32(self.plane.storage.column_groups) * Int32(4),
        )
        values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        values.fill(Uint32(0))
        for quarter in cutlass.range_constexpr(4):
            index = lane * Int32(16) + Int32(quarter * 4)
            code_index = index
            if cutlass.const_expr(self.plane.destination == 0):
                code_index = (lane * (nr // Int32(32)) + Int32(quarter)) * nc
                valid = Int32(quarter) < nr // Int32(32)
            else:
                valid = index < nr * nc
            if valid:
                field = code_index * Int32(self.plane.storage.bits)
                packed = ld_shared_u32(address + field // Int32(32) * Int32(4)) >> (
                    field % Int32(32)
                ).to(Uint32)
                if cutlass.const_expr(self.plane.storage.bits == 4):
                    offsets = byte_perm(
                        packed, packed >> Uint32(4), Int32(0x5140)
                    ) & Uint32(0x0F0F0F0F)
                else:
                    offsets = packed & Uint32(15)
                    offsets = (offsets | (offsets << Uint32(14))) & Uint32(0x00030003)
                    offsets = (offsets | (offsets << Uint32(7))) & Uint32(0x01010101)
                base = Uint32(0)
                if cutlass.const_expr(self.plane.destination == 1):
                    base = ld_shared_u32(address + Int32(256) + index % nr)
                else:
                    row = code_index // nc
                    base_word = ld_shared_u32(
                        address + Int32(256) + row // Int32(4) * Int32(4)
                    )
                    if cutlass.const_expr(self.plane.destination == 0):
                        base = byte_perm(
                            base_word, base_word, (row % Int32(4)) * Int32(0x1111)
                        )
                        if nc == Int32(2):
                            offsets &= Uint32(0x00000F0F)
                    elif nc == Int32(4):
                        base = byte_perm(
                            base_word, base_word, (row % Int32(4)) * Int32(0x1111)
                        )
                    else:
                        pair = base_word >> ((row % Int32(4)) * Int32(8)).to(Uint32)
                        base = (
                            (pair & Uint32(255))
                            | ((pair & Uint32(0xFF00)) << Uint32(8))
                        ) * Uint32(0x101)
                values[quarter] = base + offsets
        cute.arch.sync_warp()
        st_shared_v4_u32(
            address + lane * Int32(16), values[0], values[1], values[2], values[3]
        )
        cute.arch.sync_warp()
        if first + lane.to(Uint32) < end:
            self.plane.write_group_record(
                source_group, target, position, replacement.to(cutlass.Uint8)
            )
        self.plane.patch_group_records(
            storage,
            source_group,
            lane,
            target,
            cutlass.min(first + Uint32(32), end),
            end,
        )
        cute.arch.sync_warp()

    @cute.jit
    def pad_rows(self, group, target, lane):
        if cutlass.const_expr(self.plane.destination == 3):
            column = group % Int32(self.plane.storage.column_groups)
            nc = cutlass.min(Int32(4), Int32(self.plane.columns) - column * Int32(4))
            if nc == Int32(2):
                address = target.toint().to(Int32)
                low = ld_shared_u32(address + lane * Int32(8))
                high = ld_shared_u32(address + lane * Int32(8) + Int32(4))
                cute.arch.sync_warp()
                st_shared_v4_u32(
                    address + lane * Int32(16),
                    low & Uint32(65535),
                    low >> Uint32(16),
                    high & Uint32(65535),
                    high >> Uint32(16),
                )
                cute.arch.sync_warp()


class Nvfp4CsfStageReader(CsfInlineReader):
    def __init__(self, rows, columns, experts):
        super().__init__((rows, columns, 4, 9, rows, columns, experts, 8, 1, 0, 1))

    @cute.jit
    def prefetch_bounds(
        self, storage, experts, expert, outer, *, columns: cutlass.Constexpr
    ):
        storage = cute.recast_ptr(storage, dtype=cutlass.Uint8)
        extent = (
            (self.plane.columns + 7) // 8 if columns else (self.plane.rows + 127) // 128
        )
        bounds = cute.make_rmem_tensor(
            cute.make_layout(((extent + 31) // 32, 4)), cutlass.Uint32
        )
        bounds.fill(Uint32(0))
        lane = Int32(cute.arch.lane_idx())
        for chunk in cutlass.range_constexpr((extent + 31) // 32):
            tile = lane + Int32(chunk * 32)
            row = outer if cutlass.const_expr(columns) else tile
            column = tile if cutlass.const_expr(columns) else outer
            for half in cutlass.range_constexpr(2):
                group = (
                    row * Int32(self.plane.storage.column_groups)
                    + column * Int32(2)
                    + Int32(half)
                )
                if tile < Int32(extent) and column * Int32(2) + Int32(half) < Int32(
                    self.plane.storage.column_groups
                ):
                    first, end = self.plane.group_record_bounds(storage, expert, group)
                    bounds[chunk, half * 2] = first
                    bounds[chunk, half * 2 + 1] = end
        return bounds

    @cute.jit
    def stage(
        self,
        storage,
        experts,
        expert,
        row_tile,
        k_tile,
        shared,
        stage,
        barrier,
        bounds=None,
        bound_index: Int32 = 0,
    ):
        """Publish both scale groups through the operand transaction barrier."""
        target = cute.recast_ptr(shared, dtype=cutlass.Uint8) + stage * Int32(1024)
        lane = Int32(cute.arch.lane_idx())
        for half in cutlass.range_constexpr(2):
            group = (
                row_tile * Int32(self.plane.storage.column_groups)
                + k_tile * Int32(2)
                + Int32(half)
            )
            if k_tile * Int32(2) + Int32(half) >= Int32(
                self.plane.storage.column_groups
            ):
                group = Int32(self.plane.storage.groups)
            slot = target + Int32(half * 512)
            cached = None
            if cutlass.const_expr(bounds is not None):
                first, end = Uint32(0), Uint32(0)
                for chunk in cutlass.range_constexpr(cute.size(bounds, mode=[0])):
                    selected_first = cute.arch.shuffle_sync(
                        bounds[chunk, half * 2], bound_index % Int32(32)
                    )
                    selected_end = cute.arch.shuffle_sync(
                        bounds[chunk, half * 2 + 1], bound_index % Int32(32)
                    )
                    if bound_index // Int32(32) == Int32(chunk):
                        first, end = selected_first, selected_end
                cached = (first, end)
            self.stage_compressed(storage, expert, group, slot, lane, cached)
        cute.arch.cp_async_mbarrier_arrive_noinc(barrier)
        cute.arch.mbarrier_arrive(barrier)

    @cute.jit
    def expand_shared(
        self, storage, shared, stage, threads: cutlass.Constexpr, barrier
    ):
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        for chunk in cutlass.range_constexpr(
            (2 + threads // 32 - 1) // (threads // 32)
        ):
            group_slot = warp + Int32(chunk * (threads // 32))
            if group_slot < Int32(2):
                target = (
                    cute.recast_ptr(shared, dtype=cutlass.Uint8)
                    + stage * Int32(1024)
                    + group_slot * Int32(512)
                )
                group = ld_shared_u32(target.toint().to(Int32) + Int32(500)).to(Int32)
                self.expand_compressed(storage, group, target, lane)
        barrier.arrive_and_wait()
