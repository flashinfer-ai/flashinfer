"""Routed lossless NVFP4 scale expansion into native F8_128x4 storage.

One grid expands gate/up and down scales. Each CTA owns a 128-row output
region, writes its fixed stream, and applies its prepartitioned exceptions
after a barrier. Output scratch is caller-owned and may be shared only by
serialized layer execution on one CUDA stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


@dataclass(frozen=True)
class Nvfp4CsfBatch:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    task_offsets: torch.Tensor
    rows: int
    columns: int
    codec: int = 0
    layout: int = 0
    value_lut: torch.Tensor | None = None

    @property
    def num_experts(self):
        return int(self.fixed.shape[0])

    @property
    def geometry(self):
        return self.rows, self.columns, self.codec, self.layout

    def validate(self):
        alignment = 4 if self.codec == 0 else 8
        if (
            self.rows <= 0
            or self.rows % 128
            or self.columns <= 0
            or self.columns % alignment
        ):
            raise ValueError("NVFP4-CSF requires 128-row and codec-column alignment")
        if self.rows * self.columns > (1 << (19 if self.codec == 2 else 24)):
            raise ValueError("NVFP4-CSF position field is too small for the matrix")
        if self.codec not in (0, 1, 2):
            raise ValueError("Unknown NVFP4-CSF scale codec")
        if self.layout not in (0, 1) or (self.layout and self.codec):
            raise ValueError("MMA-packed NVFP4-CSF requires the byte-window codec")
        if self.layout and (
            self.value_lut is None
            or self.value_lut.dtype != torch.uint8
            or self.value_lut.shape != (256,)
            or self.value_lut.device != self.fixed.device
            or not self.value_lut.is_contiguous()
        ):
            raise ValueError(
                "MMA-packed NVFP4-CSF requires a resident 256-byte conversion table"
            )
        slab = 128 if self.codec == 0 else 16
        expected = (self.num_experts, self.rows // slab, slab * (1 + self.columns // 2))
        if self.fixed.dtype != torch.uint8 or tuple(self.fixed.shape) != expected:
            raise ValueError("NVFP4-CSF fixed stream geometry or dtype mismatch")
        width = 3 if self.codec == 2 else 4
        if (
            self.exceptions.dtype != torch.uint8
            or self.exceptions.ndim != 1
            or self.exceptions.numel() % width
        ):
            raise ValueError("NVFP4-CSF exception stream extent or dtype mismatch")
        if self.task_offsets.dtype != torch.int64 or tuple(self.task_offsets.shape) != (
            self.num_experts,
            self.rows // 128 + 1,
        ):
            raise ValueError(
                "NVFP4-CSF requires expert-major 128-row exception partitions"
            )
        tensors = (self.fixed, self.exceptions, self.task_offsets)
        if self.fixed.device.type != "cuda" or any(
            t.device != self.fixed.device or not t.is_contiguous() for t in tensors
        ):
            raise ValueError(
                "NVFP4-CSF runtime tensors must be contiguous on one CUDA device"
            )


def make_nvfp4_csf_batch(
    fixed_planes,
    exception_planes,
    *,
    rows,
    columns,
    device,
    codec=0,
    layout=0,
    row_rotation=0,
    value_lut=None,
):
    """Upload compressed CPU planes and build immutable exception partitions."""
    if len(fixed_planes) == 0 or len(fixed_planes) != len(exception_planes):
        raise ValueError("NVFP4-CSF component lists must be nonempty and equally sized")
    alignment = 4 if codec == 0 else 8
    if (
        rows <= 0
        or rows % 128
        or columns <= 0
        or columns % alignment
        or codec not in (0, 1, 2)
    ):
        raise ValueError("Unsupported NVFP4-CSF geometry or codec")
    position_bits = 19 if codec == 2 else 24
    if rows * columns > 1 << position_bits:
        raise ValueError("NVFP4-CSF position field is too small for the matrix")
    fixed_list, exception_list, partitions = [], [], []
    cursor = 0
    for fixed, exceptions in zip(fixed_planes, exception_planes, strict=True):
        f = np.asarray(fixed, dtype=np.uint8).reshape(
            rows // 16, 16 * (1 + columns // 2)
        )
        e = np.asarray(exceptions).view(np.uint8).reshape(-1)
        width = 3 if codec == 2 else 4
        if len(e) % width:
            raise ValueError("Invalid NVFP4-CSF exception record length")
        if width == 4:
            words = e.copy().view("<u4")
        else:
            b = e.reshape(-1, 3).astype(np.uint32)
            words = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        positions = words & ((1 << position_bits) - 1)
        if len(words) and (
            positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError("NVFP4-CSF exceptions must be unique, sorted and in range")
        if row_rotation:
            if codec or not 0 < row_rotation < rows:
                raise ValueError(
                    "Row rotation requires the byte-window codec and a valid split"
                )
            bases = np.roll(f[:, :16].reshape(rows), -row_rotation)
            codes = np.roll(
                f[:, 16:].reshape(rows, columns // 2), -row_rotation, axis=0
            )
            f = np.concatenate(
                (bases.reshape(rows // 16, 16), codes.reshape(rows // 16, -1)), 1
            )
            positions = (
                (positions // columns + rows - row_rotation) % rows
            ) * columns + positions % columns
            words = (words & np.uint32(0xFF000000)) | positions
            order = np.argsort(positions)
            words, positions = words[order], positions[order]
            e = words.astype("<u4").view(np.uint8)
        bounds = np.arange(rows // 128 + 1, dtype=np.int64) * (128 * columns)
        partitions.append(np.searchsorted(positions, bounds).astype(np.int64) + cursor)
        cursor += len(words)
        if codec == 0:
            # Store nibbles in native scale-byte order. This changes only the
            # resident layout; checkpoint slabs remain TP-independent.
            bases = f[:, :16].reshape(rows)
            packed = f[:, 16:].reshape(rows, columns // 2)
            logical = np.empty((rows, columns), dtype=np.uint8)
            logical[:, ::2], logical[:, 1::2] = packed & 15, packed >> 4
            if layout == 0:
                bases = bases.reshape(rows // 128, 4, 32).transpose(0, 2, 1)
                native = logical.reshape(rows // 128, 4, 32, columns // 4, 4)
                native = native.transpose(0, 3, 2, 1, 4).reshape(rows // 128, -1)
            else:
                # Match W4A16's 64-row scale transpose and adjacent pair swap.
                perm = (
                    np.arange(64)
                    .reshape(8, 8)
                    .T.reshape(-1)
                    .reshape(-1, 4)[:, [0, 2, 1, 3]]
                    .reshape(-1)
                )
                bases = bases.reshape(rows // 128, 2, 64)[:, :, perm]
                native = logical.reshape(rows // 128, 2, 64, columns)[:, :, perm, :]
                native = (
                    native.reshape(rows // 128, 128, columns)
                    .transpose(0, 2, 1)
                    .reshape(rows // 128, -1)
                )
            nibbles = native[:, ::2] | (native[:, 1::2] << 4)
            f = np.concatenate((bases.reshape(rows // 128, 128), nibbles), axis=1)
        fixed_list.append(f)
        exception_list.append(e)
    result = Nvfp4CsfBatch(
        torch.from_numpy(np.stack(fixed_list)).to(device),
        torch.from_numpy(np.concatenate(exception_list)).to(device),
        torch.from_numpy(np.stack(partitions)).to(device),
        int(rows),
        int(columns),
        int(codec),
        int(layout),
        value_lut,
    )
    result.validate()
    return result


def repack_nvfp4_csf_batch(batch, *, row_rotation, value_lut):
    """Permute resident compressed slabs for the standard W4A16 scale layout."""
    if batch.codec or batch.layout:
        raise ValueError("MMA repacking requires native byte-window slabs")
    e, r, c = batch.num_experts, batch.rows, batch.columns
    native = batch.fixed.cpu().numpy()
    bases = (
        native[:, :, :128]
        .reshape(e, r // 128, 32, 4)
        .transpose(0, 1, 3, 2)
        .reshape(e, r // 16, 16)
    )
    packed = native[:, :, 128:]
    offsets = np.empty((e, r // 128, 128 * c), dtype=np.uint8)
    offsets[:, :, ::2], offsets[:, :, 1::2] = packed & 15, packed >> 4
    logical = (
        offsets.reshape(e, r // 128, c // 4, 32, 4, 4)
        .transpose(0, 1, 4, 3, 2, 5)
        .reshape(e, r, c)
    )
    fixed = np.concatenate(
        (
            bases,
            (logical[:, :, ::2] | (logical[:, :, 1::2] << 4)).reshape(e, r // 16, -1),
        ),
        2,
    )
    words = batch.exceptions.cpu().numpy().view("<u4")
    bounds = batch.task_offsets.cpu().numpy()
    exceptions = [words[start:end] for start, end in bounds[:, [0, -1]]]
    return make_nvfp4_csf_batch(
        fixed,
        exceptions,
        rows=r,
        columns=c,
        device=batch.fixed.device,
        layout=1,
        row_rotation=row_rotation,
        value_lut=value_lut,
    )


class _Plane:
    def __init__(self, geometry):
        self.rows, self.columns, self.codec, self.layout = map(int, geometry)
        self.tasks = self.rows // 128
        self.tile_bytes = 16 * (1 + self.columns // 2)

    @cute.jit
    def tensors(
        self, fixed, exceptions, offsets, output, lut, experts, exception_bytes
    ):
        return (
            cute.make_tensor(
                fixed,
                cute.make_layout(
                    (Int64(experts) * Int64(self.rows // 16 * self.tile_bytes),)
                ),
            ),
            cute.make_tensor(exceptions, cute.make_layout((exception_bytes,))),
            cute.make_tensor(
                offsets, cute.make_layout((Int64(experts) * Int64(self.tasks + 1),))
            ),
            cute.make_tensor(
                output,
                cute.make_layout((Int64(experts) * Int64(self.rows * self.columns),)),
            ),
            cute.make_tensor(lut, cute.make_layout((256,))),
        )

    @cute.jit
    def output_offset(self, row, column):
        if cutlass.const_expr(self.layout == 1):
            within = (row % Int32(8)) * Int32(8) + (row % Int32(64)) // Int32(8)
            within = (
                (within & Int32(-4))
                | ((within & Int32(1)) << Int32(1))
                | ((within & Int32(2)) >> Int32(1))
            )
            return column * Int32(self.rows) + row // Int32(64) * Int32(64) + within
        return (
            (
                ((row // Int32(128)) * Int32(self.columns // 4) + column // Int32(4))
                * Int32(32)
                + row % Int32(32)
            )
            * Int32(16)
            + ((row % Int32(128)) // Int32(32)) * Int32(4)
            + column % Int32(4)
        )

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        fixed, exceptions, offsets, output, lut = tensors
        fixed16 = cute.recast_tensor(fixed, cutlass.Uint16)
        output32 = cute.recast_tensor(output, cutlass.Uint32)
        if cutlass.const_expr(self.layout == 1):
            tile = (Int64(expert) * Int64(self.tasks) + Int64(task)) * Int64(
                128 * (1 + self.columns // 2)
            )
            fixed32 = cute.recast_tensor(fixed, cutlass.Uint32)
            fixed64 = cute.recast_tensor(fixed, cutlass.Uint64)
            output64 = cute.recast_tensor(output, cutlass.Uint64)
            bases64 = fixed64[(tile >> Int64(3)) + Int64(tid % Int32(16))]
            pair = tid
            while pair < Int32(128 * self.columns // 8):
                spread = fixed32[((tile + Int64(128)) >> Int64(2)) + Int64(pair)].to(
                    Uint64
                )
                spread = (spread | (spread << Uint64(16))) & Uint64(0x0000FFFF0000FFFF)
                spread = (spread | (spread << Uint64(8))) & Uint64(0x00FF00FF00FF00FF)
                spread = (spread | (spread << Uint64(4))) & Uint64(0x0F0F0F0F0F0F0F0F)
                source = spread + bases64
                converted = Uint64(0)
                for lane in cutlass.range_constexpr(8):
                    byte = (source >> Uint64(8 * lane)) & Uint64(255)
                    converted |= lut[Int64(byte)].to(Uint64) << Uint64(8 * lane)
                destination = (
                    Int64(expert) * Int64(self.rows * self.columns)
                    + Int64(pair // Int32(16)) * Int64(self.rows)
                    + Int64(task * 128 + (pair % Int32(16)) * 8)
                )
                output64[destination >> Int64(3)] = converted
                pair += Int32(256)
        elif cutlass.const_expr(self.codec == 0):
            tile = (Int64(expert) * Int64(self.tasks) + Int64(task)) * Int64(
                128 * (1 + self.columns // 2)
            )
            fixed32 = cute.recast_tensor(fixed, cutlass.Uint32)
            output64 = cute.recast_tensor(output, cutlass.Uint64)
            # Each thread writes two consecutive words. Their row bases repeat
            # every 128 output words, independent of the column-group iteration.
            bases = fixed16[(tile + Int64((tid * 2) % 128)) >> Int64(1)].to(Uint32)
            base0 = (bases & Uint32(255)) * Uint32(0x01010101)
            base1 = (bases >> Uint32(8)) * Uint32(0x01010101)
            output_start = (
                Int64(expert) * Int64(self.rows) + Int64(task) * Int64(128)
            ) * Int64(self.columns // 4)
            pair = tid
            while pair < Int32(128 * self.columns // 8):
                packed = fixed32[((tile + Int64(128)) >> Int64(2)) + Int64(pair)]
                low, high = packed & Uint32(65535), packed >> Uint32(16)
                low = (low | (low << Uint32(8))) & Uint32(0x00FF00FF)
                high = (high | (high << Uint32(8))) & Uint32(0x00FF00FF)
                low = (low | (low << Uint32(4))) & Uint32(0x0F0F0F0F)
                high = (high | (high << Uint32(4))) & Uint32(0x0F0F0F0F)
                value64 = (low + base0).to(Uint64) | (
                    (high + base1).to(Uint64) << Uint64(32)
                )
                output64[(output_start >> Int64(1)) + Int64(pair)] = value64
                pair += Int32(256)
        else:
            local = tid % Int32(128)
            row = task * Int32(128) + (local % Int32(4)) * Int32(32) + local // Int32(4)
            tile = (
                Int64(expert) * Int64(self.rows // 16) + Int64(row // Int32(16))
            ) * Int64(self.tile_bytes)
            rr = row % Int32(16)
            base = fixed[tile + Int64(rr)].to(Uint32)
            output_start = (
                Int64(expert) * Int64(self.rows) + Int64(task * 128)
            ) * Int64(self.columns // 4)
            word = tid
            while word < Int32(128 * self.columns // 4):
                column = (word // Int32(128)) * Int32(4)
                values = Uint32(0)
                if cutlass.const_expr(self.codec == 0):
                    address = tile + Int64(
                        16 + rr * Int32(self.columns // 2) + column // Int32(2)
                    )
                    packed = fixed16[address >> Int64(1)].to(Uint32)
                    # Spread four nibbles to bytes before adding the row base.
                    # base <= 240 prevents carries between the four byte lanes.
                    values = (packed | (packed << Uint32(8))) & Uint32(0x00FF00FF)
                    values = (values | (values << Uint32(4))) & Uint32(0x0F0F0F0F)
                    values += base * Uint32(0x01010101)
                else:
                    selector = fixed[
                        tile
                        + Int64(16 + rr * Int32(self.columns // 8) + column // Int32(8))
                    ].to(Uint32)
                    mantissa_address = tile + Int64(
                        16
                        + 16 * (self.columns // 8)
                        + rr * Int32(3 * self.columns // 8)
                        + (column // Int32(8)) * Int32(3)
                    )
                    mantissa = (
                        fixed[mantissa_address].to(Uint32)
                        | (fixed[mantissa_address + Int64(1)].to(Uint32) << Uint32(8))
                        | (fixed[mantissa_address + Int64(2)].to(Uint32) << Uint32(16))
                    )
                    for lane in cutlass.range_constexpr(4):
                        cc = Uint32(column % Int32(8) + Int32(lane))
                        high = base + ((selector >> cc) & Uint32(1))
                        value = (high << Uint32(3)) | (
                            (mantissa >> (cc * Uint32(3))) & Uint32(7)
                        )
                        values |= value << Uint32(8 * lane)
                output32[output_start + Int64(word)] = values
                word += Int32(256)
        cute.arch.sync_threads()
        partition = Int64(expert) * Int64(self.tasks + 1) + Int64(task)
        entry = offsets[partition] + Int64(tid)
        end = offsets[partition + Int64(1)]
        while entry < end:
            if cutlass.const_expr(self.codec == 2):
                exception_address = entry * Int64(3)
                exception_packed = (
                    exceptions[exception_address].to(Uint32)
                    | (exceptions[exception_address + Int64(1)].to(Uint32) << Uint32(8))
                    | (
                        exceptions[exception_address + Int64(2)].to(Uint32)
                        << Uint32(16)
                    )
                )
                position = exception_packed & Uint32((1 << 19) - 1)
                exception_value = exception_packed >> Uint32(19)
            else:
                words = cute.recast_tensor(exceptions, cutlass.Uint32)
                exception_packed = words[entry]
                position = exception_packed & Uint32(0xFFFFFF)
                exception_value = exception_packed >> Uint32(24)
            exception_row = Int32(position // Uint32(self.columns))
            exception_column = Int32(position % Uint32(self.columns))
            exception_destination = Int64(expert) * Int64(
                self.rows * self.columns
            ) + Int64(self.output_offset(exception_row, exception_column))
            if cutlass.const_expr(self.codec != 0):
                exception_value = (exception_value << Uint32(3)) | (
                    output[exception_destination].to(Uint32) & Uint32(7)
                )
            if cutlass.const_expr(self.layout == 1):
                output[exception_destination] = lut[exception_value.to(Int64)]
            else:
                output[exception_destination] = exception_value.to(cutlass.Uint8)
            entry += Int64(256)


class _Pair:
    def __init__(self, first, second, indexed=False):
        plane = _Plane
        if indexed:
            from b12x._lib.quant.nvfp4_csf_inline import IndexedNvfp4Plane

            plane = IndexedNvfp4Plane
        from b12x._lib.quant.nvfp4_csf_packed import PackedStoragePlane

        self.first = (PackedStoragePlane if first[3] == 2 else plane)(first)
        self.second = (PackedStoragePlane if second[3] == 2 else plane)(second)

    @cute.jit
    def __call__(
        self,
        f13: cute.Pointer,
        e13: cute.Pointer,
        p13: cute.Pointer,
        o13: cute.Pointer,
        lut13: cute.Pointer,
        f2: cute.Pointer,
        e2: cute.Pointer,
        p2: cute.Pointer,
        o2: cute.Pointer,
        lut2: cute.Pointer,
        ids_ptr: cute.Pointer,
        experts: Int32,
        bytes13: Int64,
        bytes2: Int64,
        capacity: Int32,
        mode: Int32,
        barrier_count_ptr: cute.Pointer,
        barrier_epoch_ptr: cute.Pointer,
        barrier_slots: Int32,
        stream: cuda.CUstream,
    ):
        first = self.first.tensors(f13, e13, p13, o13, lut13, experts, bytes13)
        second = self.second.tensors(f2, e2, p2, o2, lut2, experts, bytes2)
        ids = cute.make_tensor(ids_ptr, cute.make_layout((capacity,)))
        barrier_count = cute.make_tensor(
            barrier_count_ptr, cute.make_layout((barrier_slots,))
        )
        barrier_epoch = cute.make_tensor(
            barrier_epoch_ptr, cute.make_layout((barrier_slots,))
        )
        self.kernel(
            first,
            second,
            ids,
            experts,
            capacity,
            mode,
            barrier_count,
            barrier_epoch,
            barrier_slots,
        ).launch(
            grid=(capacity * Int32(self.first.tasks + self.second.tasks), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        first,
        second,
        ids,
        experts: Int32,
        capacity: Int32,
        mode: Int32,
        barrier_count,
        barrier_epoch,
        barrier_slots: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        # The following MoE launch observes both resets through stream order.
        # This folds its two fill launches into the scale expansion.
        if block == 0:
            slot_to_reset = Int32(tid)
            while slot_to_reset < barrier_slots:
                barrier_count[slot_to_reset] = Int32(0)
                barrier_epoch[slot_to_reset] = Int32(0)
                slot_to_reset += Int32(256)
        slot = Int32(block) // Int32(self.first.tasks + self.second.tasks)
        task = Int32(block) % Int32(self.first.tasks + self.second.tasks)
        expert = slot
        # mode 0: potentially repeated routes; mode 1: unique IDs;
        # mode 2: expert counts; mode 3: every expert, no route read.
        if mode != Int32(3):
            raw = ids[slot].to(Int64)
            if mode == Int32(2):
                if raw <= Int64(0):
                    expert = Int32(-1)
            else:
                expert = Int32(-1)
                if raw >= Int64(0) and raw < Int64(experts):
                    expert = raw.to(Int32)
                if mode == Int32(0):
                    # Every warp checks the same prefix in parallel. The
                    # resulting predicate is uniform across the CTA, so only
                    # the first route may write an expert's scale region.
                    duplicate = cutlass.Boolean(False)
                    prior = Int32(tid) % Int32(32)
                    while prior < slot:
                        duplicate = duplicate | (ids[prior].to(Int64) == raw)
                        prior += Int32(32)
                    if cute.arch.vote_any_sync(duplicate):
                        expert = Int32(-1)
        if expert >= Int32(0) and expert < experts:
            if task < Int32(self.first.tasks):
                self.first.decode(first, expert, task, Int32(tid))
            else:
                self.second.decode(
                    second, expert, task - Int32(self.first.tasks), Int32(tid)
                )


@program_cache
def compile_nvfp4_csf_pair(first, second, ids64=False, indexed=False):
    launch = _Pair(first, second, indexed)
    key = (first, second, bool(ids64), bool(indexed))
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    plane = (
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
    )
    return b12x_compile(
        launch,
        *plane,
        *plane,
        make_ptr(
            cutlass.Int64 if ids64 else cutlass.Int32,
            8 if ids64 else 4,
            cute.AddressSpace.gmem,
            assumed_align=8 if ids64 else 4,
        ),
        1,
        1,
        1,
        1,
        0,
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.nvfp4_csf_pair", 7, key),
    )


def decode_nvfp4_csf_pair(
    first, second, ids, out13, out2, *, mode=0, program=None, barriers=None,
    indexed_scales=None,
):
    """Launch the precompiled pair into caller-owned native NVFP4 scale grids."""
    if first.num_experts != second.num_experts or mode not in (0, 1, 2, 3):
        raise ValueError("NVFP4-CSF expert counts or routing mode disagree")
    if (
        ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.device != first.fixed.device
    ):
        raise ValueError("NVFP4-CSF routes must be contiguous CUDA int32/int64")
    for batch, out in ((first, out13), (second, out2)):
        if (
            out.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or tuple(out.shape)
            != (
                (batch.num_experts, batch.columns, batch.rows)
                if batch.layout
                else (batch.num_experts, batch.rows, batch.columns)
            )
            or not out.is_contiguous()
            or out.device != batch.fixed.device
        ):
            raise ValueError("NVFP4-CSF output must match the native scale geometry")
    if first.fixed.device != second.fixed.device:
        raise ValueError("NVFP4-CSF projections must be on one CUDA device")
    if (
        out13.data_ptr() < out2.data_ptr() + out2.numel() * out2.element_size()
        and out2.data_ptr() < out13.data_ptr() + out13.numel() * out13.element_size()
    ):
        raise ValueError("NVFP4-CSF output buffers must not overlap")
    capacity = first.num_experts if mode == 3 else ids.numel()
    if mode == 2 and capacity != first.num_experts:
        raise ValueError("NVFP4-CSF count routing requires one count per expert")
    if not capacity:
        if barriers is not None:
            raise ValueError("Fused barrier reset requires a nonempty decoder grid")
        return
    if barriers is not None and (
        len(barriers) != 2
        or barriers[0].numel() != barriers[1].numel()
        or any(
            t.dtype != torch.int32
            or t.numel() == 0
            or t.device != first.fixed.device
            or not t.is_contiguous()
            for t in barriers
        )
    ):
        raise ValueError(
            "Fused barrier reset requires two equally sized CUDA int32 arrays"
        )
    if program is None:
        program = compile_nvfp4_csf_pair(
            first.geometry, second.geometry, ids.dtype == torch.int64,
            indexed_scales is not None,
        )

    def device_ptr(t, dtype, align):
        return make_ptr(
            dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    args = []
    for index, (batch, out) in enumerate(((first, out13), (second, out2))):
        args.extend(
            (
                device_ptr(
                    batch.fixed if indexed_scales is None else indexed_scales[index].storage,
                    cutlass.Uint8, 16,
                ),
                device_ptr(batch.exceptions, cutlass.Uint8, 16),
                device_ptr(batch.task_offsets, cutlass.Int64, 8),
                device_ptr(out, cutlass.Uint8, 16),
                device_ptr(
                    batch.fixed if batch.value_lut is None else batch.value_lut,
                    cutlass.Uint8,
                    16,
                ),
            )
        )
    program(
        *args,
        device_ptr(
            ids,
            cutlass.Int64 if ids.dtype == torch.int64 else cutlass.Int32,
            8 if ids.dtype == torch.int64 else 4,
        ),
        first.num_experts,
        first.exceptions.numel(),
        second.exceptions.numel(),
        capacity,
        mode,
        device_ptr(
            first.task_offsets if barriers is None else barriers[0], cutlass.Int32, 4
        ),
        device_ptr(
            first.task_offsets if barriers is None else barriers[1], cutlass.Int32, 4
        ),
        0 if barriers is None else barriers[0].numel(),
        current_cuda_stream(),
    )


@dataclass(frozen=True)
class Nvfp4CsfDecoder:
    """Compressed expert planes and retained int32/int64 routing programs."""

    first: Nvfp4CsfBatch
    second: Nvfp4CsfBatch
    programs: tuple
    routing_programs: tuple
    active: torch.Tensor
    inline_scales: tuple | None = None
    indexed_programs: tuple | None = None

    @classmethod
    def prepare(cls, first, second, out13, out2, *, inline_scales=None):
        from b12x._lib.quant.csf_routing import compile_csf_active_experts

        for plane, output in ((first, out13), (second, out2)):
            plane.validate()
            expected = (
                (plane.num_experts, plane.columns, plane.rows)
                if plane.layout
                else (plane.num_experts, plane.rows, plane.columns)
            )
            if (
                tuple(output.shape) != expected
                or output.dtype != torch.float8_e4m3fn
                or output.device != plane.fixed.device
                or not output.is_contiguous()
            ):
                raise ValueError(
                    "NVFP4-CSF scratch must match native E4M3 scale storage"
                )
        if first.num_experts != second.num_experts:
            raise ValueError("NVFP4-CSF projections must have equal expert counts")
        if first.fixed.device != second.fixed.device:
            raise ValueError("NVFP4-CSF projections must be on one CUDA device")
        if (
            out13.data_ptr() < out2.data_ptr() + out2.numel() * out2.element_size()
            and out2.data_ptr() < out13.data_ptr() + out13.numel() * out13.element_size()
        ):
            raise ValueError("NVFP4-CSF output buffers must not overlap")
        programs = tuple(
            compile_nvfp4_csf_pair(first.geometry, second.geometry, ids64)
            for ids64 in (False, True)
        )
        return cls(
            first,
            second,
            programs,
            tuple(compile_csf_active_experts(ids64) for ids64 in (False, True)),
            torch.empty(
                first.num_experts, dtype=torch.int32, device=first.fixed.device
            ),
            inline_scales=inline_scales,
            indexed_programs=(
                tuple(
                    compile_nvfp4_csf_pair(first.geometry, second.geometry, ids64, True)
                    for ids64 in (False, True)
                ) if inline_scales is not None else None
            ),
        )

    def decode_all(self, out13, out2):
        """Expand every expert on the current stream, as long prefills do."""
        decode_nvfp4_csf_pair(
            self.first, self.second, self.active, out13, out2, mode=3,
            program=self.programs[0],
        )

    def decode(self, ids, out13, out2, *, barriers=None):
        from b12x._lib.quant.csf_routing import mark_active_experts

        # Build presence once for batch decode. Repeated token routes do not
        # imply that every expert is active. Long prefills bound the scan work
        # by expanding the complete, statically sized expert set instead.
        mode = 0
        if ids.numel() >= 16 * self.first.num_experts:
            mode = 3
        elif ids.numel() >= 64:
            mark_active_experts(
                ids, self.active, self.routing_programs[int(ids.dtype == torch.int64)]
            )
            ids, mode = self.active, 2
        # Short route lists retain direct duplicate checks. Equal-size spans
        # avoid assigning a large gate/up slab and a small down slab the same
        # CTA budget. Presence and full-expert expansion retain their slab grid.
        indexed = self.indexed_programs is not None and mode == 0
        programs = self.indexed_programs if indexed else self.programs
        decode_nvfp4_csf_pair(
            self.first,
            self.second,
            ids,
            out13,
            out2,
            mode=mode,
            program=programs[int(ids.dtype == torch.int64)],
            barriers=barriers,
            indexed_scales=self.inline_scales if indexed else None,
        )
