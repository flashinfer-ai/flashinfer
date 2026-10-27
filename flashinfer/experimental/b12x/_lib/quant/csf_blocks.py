"""Shared block storage for FP4 scales and native NVFP4 scale expansion."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cutlass_dsl import Boolean, Int32, Int64, Uint32, Uint64

from b12x._lib.intrinsics.packed import byte_perm
from b12x._lib.intrinsics.memory import ld_shared_v4_u32, st_shared_v4_u32
from b12x.loader._csf_blocks import (
    CsfBlockPlane,
    Quantization,
    _offset_bits,
    codes_in_native_order,
    block_order,
    group_block_offsets,
    packed_scale_row_order,
    slice_csf_blocks,
)

HEADER_BYTES = 1024
BLOCK_STORAGE_LAYOUT = 3


def _align16(value):
    return (value + 15) // 16 * 16


class BlockStorageLayout:
    """Byte offsets of the shared payload, determined by tensor geometry."""

    def __init__(
        self,
        rows,
        columns,
        experts,
        quantization,
        group_records=False,
    ):
        if rows <= 0 or rows % 32 or columns <= 0 or experts <= 0:
            raise ValueError("CSF blocks require positive geometry with 32-row extents")
        self.bits = _offset_bits(quantization)
        self.rows, self.columns, self.experts = rows, columns, experts
        self.column_blocks = (columns + 1) // 2
        self.column_groups = (columns + 3) // 4
        self.blocks = rows // 32 * self.column_blocks
        self.groups = (rows + 127) // 128 * self.column_groups
        self.pages = (self.groups + 31) // 32
        self.prefixes = (self.groups if group_records else self.blocks) + self.pages
        self.code_bytes = rows * columns * self.bits // 8
        self.bases = HEADER_BYTES
        self.codes = _align16(self.bases + experts * rows)
        self.page_bases = _align16(self.codes + experts * self.code_bytes)
        self.offsets = _align16(self.page_bases + experts * self.pages * 4)
        self.source_records = _align16(self.offsets + experts * self.prefixes * 2)
        self.metadata = self.page_bases
        self.records = self.metadata + experts * self.groups * 32


@dataclass(frozen=True)
class CsfBlockScales:
    storage: torch.Tensor
    rows: int
    columns: int
    num_experts: int
    quantization: Quantization

    @property
    def geometry(self):
        return BlockStorageLayout(
            self.rows, self.columns, self.num_experts, self.quantization
        )

    def validate(self):
        layout = self.geometry
        if (
            self.storage.dtype != torch.uint8
            or self.storage.ndim != 1
            or not self.storage.is_contiguous()
            or self.storage.device.type != "cuda"
            or self.storage.numel() < layout.records
        ):
            raise ValueError(
                "CSF block storage must be contiguous CUDA bytes covering its indexes"
            )


def upload_csf_blocks(planes: tuple[CsfBlockPlane, ...], device) -> CsfBlockScales:
    """Upload block arrays and prepare the aligned group index on the GPU."""
    payload = upload_csf_block_payload(planes, device)
    layout = payload.geometry
    record_bytes = sum(plane.exceptions.nbytes for plane in planes)
    source = payload.storage
    storage = torch.zeros(
        _align16(layout.records + record_bytes * 2), dtype=torch.uint8, device=device
    )
    storage[layout.bases : layout.metadata].copy_(
        source[layout.bases : layout.metadata]
    )
    scales = CsfBlockScales(
        storage,
        payload.rows,
        payload.columns,
        payload.num_experts,
        payload.quantization,
    )
    prepare_csf_block_metadata(scales, source)
    return scales


@dataclass(frozen=True)
class CsfBlockPayload(CsfBlockScales):
    """Block codes and compact exception records without a prepared bitmap index."""

    native_codes: bool = False
    group_record_bytes: int = 0
    destination: int = 0
    sections: int = 1

    @property
    def geometry(self):
        return BlockStorageLayout(
            self.rows,
            self.columns,
            self.num_experts,
            self.quantization,
            self.group_record_bytes != 0,
        )

    def validate(self):
        if (
            self.storage.dtype != torch.uint8
            or self.storage.ndim != 1
            or not self.storage.is_contiguous()
            or self.storage.device.type != "cuda"
            or self.storage.numel() < self.geometry.source_records
        ):
            raise ValueError("CSF payload must cover its compact indexes")


def upload_csf_block_payload(
    planes: tuple[CsfBlockPlane, ...],
    device,
    *,
    native_codes=False,
    group_record_bytes=0,
    destination=0,
    group_rows=None,
) -> CsfBlockPayload:
    """Upload lossless block arrays without constructing an exception-word bitmap."""
    if not planes:
        raise ValueError("CSF block batches must contain experts")
    if group_record_bytes not in (0, 3, 4):
        raise ValueError("Group exception records require three or four bytes")
    if destination not in (0, 1, 2, 3) or (
        destination and (not native_codes or not group_record_bytes)
    ):
        raise ValueError("Packed destinations require native codes and group records")
    first = planes[0]
    sections = 1
    if group_rows is not None and group_rows != first.rows:
        if destination not in (2, 3) or group_rows <= 0 or first.rows % group_rows:
            raise ValueError("Native row sections must partition each scale plane")
        sections = first.rows // group_rows
        planes = tuple(
            slice_csf_blocks(plane, row_range=(start, start + group_rows))
            for plane in planes
            for start in range(0, first.rows, group_rows)
        )
        first = planes[0]
    if destination == 1 and first.rows % 64:
        raise ValueError("MMA-packed scales require whole 64-row groups")
    layout = BlockStorageLayout(
        first.rows,
        first.columns,
        len(planes),
        first.quantization,
        group_record_bytes != 0,
    )
    record_count = sum(plane.exceptions.size for plane in planes)
    record_bytes = record_count * (group_record_bytes or 2)
    if record_bytes >= 1 << 32:
        raise ValueError("CSF block batch exception arena must be smaller than 4 GiB")
    buffer = np.zeros(_align16(layout.source_records + record_bytes), dtype=np.uint8)
    if group_record_bytes == 3:
        buffer[:8] = np.array(
            [layout.source_records + 2 * record_count], dtype="<u8"
        ).view(np.uint8)
    byte_cursor = 0
    for expert, plane in enumerate(planes):
        plane.validate()
        if (plane.rows, plane.columns, plane.quantization) != (
            first.rows,
            first.columns,
            first.quantization,
        ):
            raise ValueError("CSF block batch geometry and quantization must agree")
        if group_record_bytes:
            order = block_order(plane.rows, plane.columns)
            groups = group_block_offsets(plane.rows, plane.columns)
            group_of_block = np.repeat(np.arange(layout.groups), np.diff(groups))
            block_prefix = np.arange(layout.blocks) + group_of_block // 32
            counts = (
                plane.offsets[block_prefix + 1].astype(np.int64)
                - plane.offsets[block_prefix]
            )
            record_blocks = np.repeat(order, counts)
            local = plane.exceptions & 255
            positions = (
                (record_blocks // layout.column_blocks % 4 * 32 + local // 2) * 4
                + record_blocks % layout.column_blocks % 2 * 2
                + local % 2
            ).astype("<u2")
            values = (plane.exceptions >> 8).astype(np.uint8)
            if destination == 1:
                row, column = positions // 4, positions % 4
                inverse = np.argsort(packed_scale_row_order(plane.columns))
                packed_row = row // 64 * 64 + inverse[row % 64]
                nr = np.minimum(
                    128, plane.rows - record_blocks // layout.column_blocks // 4 * 128
                )
                positions = (column * nr + packed_row).astype("<u2")
            elif destination == 3:
                nc = np.minimum(
                    4, plane.columns - record_blocks % layout.column_blocks // 2 * 4
                )
                positions = (positions // 4 * nc + positions % 4).astype("<u2")
            prefixes = np.concatenate(([0], np.cumsum(counts)))[groups]
            pages = (prefixes[:-1:32] + byte_cursor).astype("<u4")
            offsets = np.concatenate(
                [
                    prefixes[start : min(start + 32, layout.groups) + 1]
                    - prefixes[start]
                    for start in range(0, layout.groups, 32)
                ]
            ).astype("<u2")
            if group_record_bytes == 4:
                records = positions.astype("<u4") | (values.astype("<u4") << 16)
                record_parts = ((layout.source_records + byte_cursor * 4, records),)
            else:
                record_parts = (
                    (layout.source_records + byte_cursor * 2, positions),
                    (layout.source_records + record_count * 2 + byte_cursor, values),
                )
        else:
            pages = (plane.page_bases.astype(np.uint64) + byte_cursor).astype("<u4")
            offsets = plane.offsets
            record_parts = ((layout.source_records + byte_cursor, plane.exceptions),)
        bases = plane.bases
        if native_codes and destination == 0:
            bases = np.concatenate(
                [
                    bases[start : start + 128].reshape(-1, 32).T.reshape(-1)
                    for start in range(0, plane.rows, 128)
                ]
            )
        if destination == 1:
            perm = packed_scale_row_order(plane.columns)
            bases = bases.reshape(-1, 64)[:, perm].copy().ravel()
        for origin, data in (
            (layout.bases + expert * first.rows, bases),
            (
                layout.codes + expert * layout.code_bytes,
                codes_in_native_order(plane, destination)
                if native_codes
                else plane.codes,
            ),
            (layout.page_bases + expert * layout.pages * 4, pages),
            (layout.offsets + expert * layout.prefixes * 2, offsets),
            *record_parts,
        ):
            buffer[origin : origin + data.nbytes] = data.view(np.uint8)
        byte_cursor += (
            plane.exceptions.size if group_record_bytes else plane.exceptions.nbytes
        )
    source = torch.from_numpy(buffer).to(device)
    return CsfBlockPayload(
        source,
        first.rows,
        first.columns,
        len(planes),
        first.quantization,
        native_codes,
        group_record_bytes,
        destination,
        sections,
    )


class _GroupMetadata:
    def __init__(self, layout):
        self.layout = layout

    @cute.jit
    def __call__(self, source: cute.Pointer, storage: cute.Pointer, stream):
        self.kernel(source, storage).launch(
            grid=(self.layout.experts * self.layout.groups, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, storage):
        tid, _, _ = cute.arch.thread_idx()
        group_index, _, _ = cute.arch.block_idx()
        row = Int32(tid)
        expert = Int32(group_index) // Int32(self.layout.groups)
        group = Int32(group_index) % Int32(self.layout.groups)
        row_group = group // Int32(self.layout.column_groups)
        column_group = group % Int32(self.layout.column_groups)
        nr = cutlass.min(Int32(128), Int32(self.layout.rows) - row_group * Int32(128))
        nrb = nr // Int32(32)
        page = group // Int32(32)
        block = row_group * Int32(
            4 * self.layout.column_blocks
        ) + nrb * column_group * Int32(2)
        prefixes = cute.recast_ptr(
            source + Int64(self.layout.offsets), dtype=cutlass.Uint16
        )
        input_words = cute.recast_ptr(source, dtype=cutlass.Uint32)
        record_origin = _load(
            input_words,
            Int64(self.layout.page_bases // 4)
            + Int64(expert) * Int64(self.layout.pages)
            + Int64(page),
        ).to(Uint32) >> Uint32(1)
        prefix = Int64(expert) * Int64(self.layout.blocks + self.layout.pages) + Int64(
            block + page
        )
        first = record_origin + _load(prefixes, prefix).to(Uint32)
        values = Uint32(0)
        present = Boolean(False)
        if row < nr:
            base = _load(
                source,
                Int64(self.layout.bases)
                + Int64(expert) * Int64(self.layout.rows)
                + Int64(row_group * Int32(128) + row),
            ).to(Uint32)
            codes = Int64(self.layout.codes) + Int64(expert) * Int64(
                self.layout.code_bytes
            )
            codes += Int64(row_group) * Int64(
                128 * self.layout.columns * self.layout.bits // 8
            )
            codes += Int64(column_group * nr) * Int64(4 * self.layout.bits) // Int64(8)
            for column in cutlass.range_constexpr(4):
                offset = Uint32(0)
                if column_group * Int32(4) + Int32(column) < Int32(self.layout.columns):
                    width = cutlass.min(
                        Int32(2),
                        Int32(self.layout.columns)
                        - column_group * Int32(4)
                        - Int32(column // 2 * 2),
                    )
                    field = (
                        Int32(column // 2) * nr * Int32(2)
                        + row * width
                        + Int32(column % 2)
                    ) * Int32(self.layout.bits)
                    offset = (
                        _load(source, codes + Int64(field // Int32(8))).to(Uint32)
                        >> (field % Int32(8)).to(Uint32)
                    ) & Uint32((1 << self.layout.bits) - 1)
                values |= (base + offset) << Uint32(column * 8)
            for pair in cutlass.range_constexpr(2):
                if column_group * Int32(4) + Int32(pair * 2) < Int32(
                    self.layout.columns
                ):
                    block_prefix = prefix + Int64(Int32(pair) * nrb + row // Int32(32))
                    entry = record_origin + _load(prefixes, block_prefix).to(Uint32)
                    end = record_origin + _load(prefixes, block_prefix + Int64(1)).to(
                        Uint32
                    )
                    lower = (row % Int32(32) * Int32(2)).to(Uint32)
                    while entry < end:
                        records = cute.recast_ptr(
                            source + Int64(self.layout.source_records),
                            dtype=cutlass.Uint16,
                        )
                        record = _load(records, entry.to(Int64)).to(Uint32)
                        position = record & Uint32(255)
                        if position > lower + Uint32(1):
                            entry = end
                        else:
                            if position >= lower:
                                shift = (
                                    position % Uint32(2) + Uint32(pair * 2)
                                ) * Uint32(8)
                                values = (values & ~(Uint32(255) << shift)) | (
                                    (record >> Uint32(8)) << shift
                                )
                                present = Boolean(True)
                            entry += Uint32(1)
        mask = cute.arch.vote_ballot_sync(present).to(Uint32)
        warp, lane = row // Int32(32), row % Int32(32)
        smem = cutlass.utils.SmemAllocator()
        counts = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(4), byte_alignment=16
        )
        if lane == Int32(0):
            counts[warp] = cute.arch.popc(mask).to(Uint32)
        cute.arch.sync_threads()
        before = Uint32(0)
        packed_prefixes = Uint32(0)
        total = Uint32(0)
        for index in cutlass.range_constexpr(4):
            packed_prefixes |= total << Uint32(index * 8)
            if warp > Int32(index):
                before += counts[index]
            total += counts[index]
        target = cute.make_tensor(
            cute.recast_ptr(storage + Int64(self.layout.metadata), dtype=cutlass.Uint32)
            + Int64(group_index) * Int64(8),
            cute.make_layout(8),
        )
        if lane == Int32(0):
            target[warp + Int32(2)] = mask
        if row == Int32(0):
            target[0] = first
            target[1] = packed_prefixes
            target[6] = first + total
            target[7] = Uint32(1)
        if present:
            rank = before + cute.arch.popc(
                mask & ((Uint32(1) << lane.to(Uint32)) - Uint32(1))
            ).to(Uint32)
            records = cute.recast_ptr(
                storage + Int64(self.layout.records), dtype=cutlass.Uint32
            )
            cute.make_tensor(records + (first + rank).to(Int64), cute.make_layout(1))[
                0
            ] = values


def prepare_csf_block_metadata(scales, source):
    """Build a native-group bitmap index with complete exception words on the GPU.

    The code blocks and row bases retain their checkpoint values and order.
    Each group indexes the logical rows with exceptions using four bitmaps and
    population-count prefixes. Complete replacement words remove record scans
    from the consumer. Each group reserves at most one word per input record,
    so allocation requires no device-to-host count transfer. Compact input
    indexes and records are released after preparation.
    """
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream
    from b12x._lib.utils.pointers import make_ptr

    with torch.cuda.device(scales.storage.device):
        pointer = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
        program = b12x_compile(
            _GroupMetadata(scales.geometry),
            pointer,
            pointer,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "quant.csf_block_group_metadata",
                2,
                (scales.rows, scales.columns, scales.num_experts, scales.quantization),
            ),
        )
        program(
            make_ptr(
                cutlass.Uint8,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scales.storage.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            current_cuda_stream(),
        )


@dataclass(frozen=True)
class Nvfp4BlockPlane:
    """Block storage bound to the existing routed NVFP4 expansion launch."""

    scales: CsfBlockScales
    codec: int = 0
    layout: int = 0
    value_lut: None = None

    @property
    def fixed(self):
        return self.scales.storage

    @property
    def exceptions(self):
        return self.fixed[:0]

    @property
    def task_offsets(self):
        return self.fixed[:16].view(torch.int64)

    @property
    def rows(self):
        return (self.scales.rows + 127) // 128 * 128

    @property
    def columns(self):
        return (self.scales.columns + 3) // 4 * 4

    @property
    def num_experts(self):
        return self.scales.num_experts

    @property
    def geometry(self):
        return (
            self.rows,
            self.columns,
            self.codec,
            BLOCK_STORAGE_LAYOUT,
            self.scales.rows,
            self.scales.columns,
            self.num_experts,
        )

    def validate(self):
        self.scales.validate()
        if self.scales.quantization != "nvfp4":
            raise ValueError("Native NVFP4 scale expansion requires NVFP4 blocks")


@cute.jit
def _load(pointer, offset):
    return cute.make_tensor(pointer + offset, cute.make_layout(1))[0]


class Nvfp4BlockStoragePlane:
    """One CTA expands a native 128-row slab from compact indexed blocks."""

    def __init__(self, geometry):
        self.rows, self.columns, _, _, rows, columns, experts = map(int, geometry)
        self.logical_rows, self.logical_columns = rows, columns
        self.storage = BlockStorageLayout(rows, columns, experts, "nvfp4")
        self.tasks = self.rows // 128

    @cute.jit
    def tensors(
        self, fixed, exceptions, offsets, output, lut, experts, exception_bytes
    ):
        return fixed, output

    @cute.jit
    def output_offset(self, row, column):
        return (
            (
                (row // Int32(128) * Int32(self.columns // 4) + column // Int32(4))
                * Int32(32)
                + row % Int32(32)
            )
            * Int32(16)
            + row % Int32(128) // Int32(32) * Int32(4)
            + column % Int32(4)
        )

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        storage, output = tensors
        words = cute.recast_ptr(output, dtype=cutlass.Uint32)
        index_words = cute.recast_ptr(storage, dtype=cutlass.Uint32)
        nr = cutlass.min(Int32(128), Int32(self.logical_rows) - task * Int32(128))
        nrb = nr // Int32(32)
        bases = Int64(self.storage.bases) + Int64(expert) * Int64(self.logical_rows)
        codes = (
            Int64(self.storage.codes)
            + Int64(expert) * Int64(self.storage.code_bytes)
            + Int64(task) * Int64(128 * self.logical_columns // 2)
        )
        word = tid
        while word < Int32(128 * self.columns // 4):
            col_group = word // Int32(128)
            lane = word % Int32(128)
            rr = lane % Int32(4) * Int32(32) + lane // Int32(4)
            value = Uint32(0)
            if rr < nr:
                base = _load(storage, bases + Int64(task * 128 + rr)).to(Uint32)
                for c in cutlass.range_constexpr(4):
                    offset = Uint32(0)
                    if col_group * Int32(4) + Int32(c) < Int32(self.logical_columns):
                        width = cutlass.min(
                            Int32(2),
                            Int32(self.logical_columns)
                            - col_group * Int32(4)
                            - Int32(c // 2 * 2),
                        )
                        field = (
                            Int32(c // 2) * nr * Int32(2) + rr * width + Int32(c % 2)
                        )
                        address = codes + Int64(col_group * nr * 2 + field // Int32(2))
                        offset = (
                            _load(storage, address).to(Uint32)
                            >> ((field % Int32(2)).to(Uint32) * Uint32(4))
                        ) & Uint32(15)
                    value |= (base + offset) << Uint32(c * 8)
                metadata = Int64(self.storage.metadata // 4) + (
                    Int64(expert) * Int64(self.storage.groups)
                    + Int64(task * Int32(self.storage.column_groups) + col_group)
                ) * Int64(8)
                group, bit = rr // Int32(32), rr % Int32(32)
                mask = _load(index_words, metadata + Int64(2) + Int64(group)).to(Uint32)
                flag = Uint32(1) << bit.to(Uint32)
                if (mask & flag) != Uint32(0):
                    first = _load(index_words, metadata).to(Uint32)
                    prefix = _load(index_words, metadata + Int64(1)).to(Uint32)
                    rank = (prefix >> (group.to(Uint32) * Uint32(8))) & Uint32(255)
                    rank += cute.arch.popc(mask & (flag - Uint32(1))).to(Uint32)
                    value = _load(
                        index_words,
                        Int64(self.storage.records // 4) + (first + rank).to(Int64),
                    ).to(Uint32)
            destination = (
                Int64(expert) * Int64(self.rows) + Int64(task * 128)
            ) * Int64(self.columns // 4) + Int64(word)
            cute.make_tensor(words + destination, cute.make_layout(1))[0] = value
            word += Int32(256)


@dataclass(frozen=True)
class Nvfp4CompactBlockPlane(Nvfp4BlockPlane):
    """Compact block records consumed by the cooperative native expander."""

    groups_per_task: int = 8
    shared_exceptions: bool = True

    @property
    def rows(self):
        if self.scales.destination in (1, 3):
            return self.scales.rows * self.scales.sections
        rows = super().rows * self.scales.sections
        return (rows + 255) // 256 * 256 if self.scales.destination == 2 else rows

    @property
    def columns(self):
        return (
            self.scales.columns
            if self.scales.destination in (1, 3)
            else super().columns
        )

    @property
    def num_experts(self):
        return self.scales.num_experts // self.scales.sections

    @property
    def geometry(self):
        return (
            self.rows,
            self.columns,
            self.scales.geometry.bits,
            (
                8
                if self.scales.group_record_bytes == 3
                else 6
                if self.scales.group_record_bytes == 4
                else 4
            )
            + int(self.scales.native_codes),
            self.scales.rows,
            self.scales.columns,
            self.num_experts,
            self.groups_per_task,
            int(self.shared_exceptions),
            self.scales.destination,
            self.scales.sections,
        )

    def validate(self):
        if not isinstance(self.scales, CsfBlockPayload):
            raise TypeError("Compact expansion requires compact block records")
        if self.groups_per_task not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Expansion tasks require 1, 2, 4, 8, 16, 32, or 64 groups")
        self.scales.validate()
        if self.layout != int(self.scales.destination == 1):
            raise ValueError("Scale output orientation differs from its native layout")


class CompactBlockStoragePlane:
    """Vector expansion with cooperative compact exception application."""

    def __init__(self, geometry):
        (
            self.rows,
            self.columns,
            bits,
            layout,
            rows,
            columns,
            experts,
            groups,
            shared,
            destination,
            sections,
        ) = map(int, geometry)
        self.destination = destination
        self.sections = sections
        self.native_codes = layout % 2 == 1
        self.group_record_bytes = 3 if layout >= 8 else 4 if layout >= 6 else 0
        self.storage = BlockStorageLayout(
            rows,
            columns,
            experts * sections,
            "nvfp4" if bits == 4 else "mxfp4",
            self.group_record_bytes != 0,
        )
        self.groups_per_task = groups
        self.warps = min(groups, 8)
        self.threads = self.warps * 32
        self.shared_exceptions = bool(shared)
        self.groups = (
            self.rows // 128 * self.storage.column_groups
            if destination == 2
            else self.storage.groups * sections
        )
        self.tasks = (self.groups + groups - 1) // groups

    @cute.jit
    def tensors(
        self, fixed, exceptions, offsets, output, lut, experts, exception_bytes
    ):
        return fixed, output

    @cute.jit
    def aligned_f8_words(self, storage, expert, group, lane):
        codes = (
            Int64(self.storage.codes)
            + Int64(expert) * Int64(self.storage.code_bytes)
            + Int64(group) * Int64(64 * self.storage.bits)
        )
        bases = (
            Int64(self.storage.bases)
            + Int64(expert) * Int64(self.storage.rows)
            + Int64(group // Int32(self.storage.column_groups)) * Int64(128)
        )
        base_word = _load(
            cute.recast_ptr(storage + bases, dtype=cutlass.Uint32), lane.to(Int64)
        ).to(Uint32)
        if cutlass.const_expr(self.storage.bits == 4):
            packed = _load(
                cute.recast_ptr(storage + codes, dtype=cutlass.Uint64), lane.to(Int64)
            ).to(Uint64)
        else:
            packed = _load(
                cute.recast_ptr(storage + codes, dtype=cutlass.Uint16), lane.to(Int64)
            ).to(Uint64)
        values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        for quarter in cutlass.range_constexpr(4):
            offsets = (packed >> Uint64(quarter * 4 * self.storage.bits)).to(Uint32)
            if cutlass.const_expr(self.storage.bits == 4):
                offsets = byte_perm(offsets, offsets >> Uint32(4), Int32(0x5140))
                offsets &= Uint32(0x0F0F0F0F)
            else:
                offsets &= Uint32(15)
                offsets = (offsets | (offsets << Uint32(14))) & Uint32(0x00030003)
                offsets = (offsets | (offsets << Uint32(7))) & Uint32(0x01010101)
            base = byte_perm(base_word, base_word, Int32(quarter * 0x1111))
            values[quarter] = base + offsets
        return values

    @cute.jit
    def nvfp4_f8_tile_words(
        self,
        storage,
        expert,
        group,
        lane,
        values,
        nr: cutlass.Constexpr,
        nc: cutlass.Constexpr,
    ):
        row_group = group // Int32(self.storage.column_groups)
        column_group = group % Int32(self.storage.column_groups)
        codes = (
            Int64(self.storage.codes)
            + Int64(expert) * Int64(self.storage.code_bytes)
            + Int64(row_group) * Int64(64 * self.storage.columns)
            + Int64(column_group) * Int64(nr * 2)
        )
        if cutlass.const_expr(nr * nc == 512):
            packed = _load(
                cute.recast_ptr(storage + codes, dtype=cutlass.Uint64), lane.to(Int64)
            ).to(Uint64)
        elif cutlass.const_expr(nr * nc == 256):
            packed = _load(
                cute.recast_ptr(storage + codes, dtype=cutlass.Uint32), lane.to(Int64)
            ).to(Uint64)
        else:
            packed = _load(
                cute.recast_ptr(storage + codes, dtype=cutlass.Uint16), lane.to(Int64)
            ).to(Uint64)
        bases = (
            Int64(self.storage.bases)
            + Int64(expert) * Int64(self.storage.rows)
            + Int64(row_group) * Int64(128)
        )
        base_word = _load(
            cute.recast_ptr(
                storage + bases,
                dtype=cutlass.Uint32 if nr == 128 else cutlass.Uint16,
            ),
            lane.to(Int64),
        ).to(Uint32)
        for quarter in cutlass.range_constexpr(4):
            values[quarter] = Uint32(0)
            if cutlass.const_expr(quarter < nr // 32):
                offsets = (packed >> Uint64(quarter * nc * 4)).to(Uint32)
                offsets &= Uint32((1 << (nc * 4)) - 1)
                offsets = byte_perm(offsets, offsets >> Uint32(4), Int32(0x5140))
                offsets &= Uint32(0x0F0F0F0F)
                base = byte_perm(base_word, base_word, Int32(quarter * 0x1111))
                values[quarter] = base + offsets

    @cute.jit
    def group_words(self, storage, expert, group, lane):
        if cutlass.const_expr(self.destination != 0):
            return self.destination_words(storage, expert, group, lane)
        if cutlass.const_expr(
            self.native_codes
            and self.storage.rows % 128 == 0
            and self.storage.columns % 4 == 0
        ):
            return self.aligned_f8_words(storage, expert, group, lane)
        row_group = group // Int32(self.storage.column_groups)
        column_group = group % Int32(self.storage.column_groups)
        if cutlass.const_expr(
            self.native_codes
            and self.storage.bits == 4
            and self.storage.rows % 64 == 0
            and self.storage.columns % 2 == 0
        ):
            values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
            if row_group < Int32(self.storage.rows // 128):
                if column_group < Int32(self.storage.columns // 4):
                    self.nvfp4_f8_tile_words(
                        storage, expert, group, lane, values, 128, 4
                    )
                else:
                    self.nvfp4_f8_tile_words(
                        storage, expert, group, lane, values, 128, 2
                    )
            else:
                if column_group < Int32(self.storage.columns // 4):
                    self.nvfp4_f8_tile_words(
                        storage, expert, group, lane, values, 64, 4
                    )
                else:
                    self.nvfp4_f8_tile_words(
                        storage, expert, group, lane, values, 64, 2
                    )
            return values
        nr = cutlass.min(Int32(128), Int32(self.storage.rows) - row_group * Int32(128))
        codes = (
            Int64(self.storage.codes)
            + Int64(expert) * Int64(self.storage.code_bytes)
            + Int64(row_group)
            * Int64(128 * self.storage.columns * self.storage.bits // 8)
            + Int64(column_group * nr) * Int64(self.storage.bits) // Int64(2)
        )
        values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        packed = Uint64(0)
        if cutlass.const_expr(self.native_codes):
            if nr == Int32(128) and column_group * Int32(4) + Int32(4) <= Int32(
                self.storage.columns
            ):
                if cutlass.const_expr(self.storage.bits == 4):
                    packed = _load(
                        cute.recast_ptr(storage + codes, dtype=cutlass.Uint64),
                        lane.to(Int64),
                    ).to(Uint64)
                else:
                    packed = _load(
                        cute.recast_ptr(storage + codes, dtype=cutlass.Uint16),
                        lane.to(Int64),
                    ).to(Uint64)
        for quarter in cutlass.range_constexpr(4):
            row = lane + Int32(quarter * 32)
            value = Uint32(0)
            if row < nr:
                base = _load(
                    storage,
                    Int64(self.storage.bases)
                    + Int64(expert) * Int64(self.storage.rows)
                    + Int64(row_group * Int32(128))
                    + Int64(
                        row % Int32(32) * (nr // Int32(32)) + row // Int32(32)
                        if cutlass.const_expr(self.native_codes)
                        else row
                    ),
                ).to(Uint32)
                full_native = Boolean(False)
                if cutlass.const_expr(self.native_codes):
                    full_native = nr == Int32(128) and column_group * Int32(4) + Int32(
                        4
                    ) <= Int32(self.storage.columns)
                if full_native:
                    offsets = (packed >> Uint64(quarter * 4 * self.storage.bits)).to(
                        Uint32
                    )
                    if cutlass.const_expr(self.storage.bits == 4):
                        offsets &= Uint32(0xFFFF)
                        offsets = (offsets | (offsets << Uint32(8))) & Uint32(
                            0x00FF00FF
                        )
                        offsets = (offsets | (offsets << Uint32(4))) & Uint32(
                            0x0F0F0F0F
                        )
                    else:
                        offsets &= Uint32(15)
                        offsets = (offsets | (offsets << Uint32(14))) & Uint32(
                            0x00030003
                        )
                        offsets = (offsets | (offsets << Uint32(7))) & Uint32(
                            0x01010101
                        )
                    value = base * Uint32(0x01010101) + offsets
                else:
                    for column in cutlass.range_constexpr(4):
                        offset = Uint32(0)
                        if column_group * Int32(4) + Int32(column) < Int32(
                            self.storage.columns
                        ):
                            width = cutlass.min(
                                Int32(2),
                                Int32(self.storage.columns)
                                - column_group * Int32(4)
                                - Int32(column // 2 * 2),
                            )
                            if cutlass.const_expr(self.native_codes):
                                if nr == Int32(128) and column_group * Int32(4) + Int32(
                                    4
                                ) <= Int32(self.storage.columns):
                                    offset = (
                                        packed
                                        >> Uint64(
                                            (quarter * 4 + column) * self.storage.bits
                                        )
                                    ).to(Uint32) & Uint32((1 << self.storage.bits) - 1)
                                else:
                                    nc = cutlass.min(
                                        Int32(4),
                                        Int32(self.storage.columns)
                                        - column_group * Int32(4),
                                    )
                                    field = (
                                        (lane * (nr // Int32(32)) + Int32(quarter)) * nc
                                        + Int32(column)
                                    ) * Int32(self.storage.bits)
                                    offset = (
                                        _load(
                                            storage, codes + Int64(field // Int32(8))
                                        ).to(Uint32)
                                        >> (field % Int32(8)).to(Uint32)
                                    ) & Uint32((1 << self.storage.bits) - 1)
                            else:
                                field = (
                                    Int32(column // 2) * nr * Int32(2)
                                    + row * width
                                    + Int32(column % 2)
                                ) * Int32(self.storage.bits)
                                offset = (
                                    _load(storage, codes + Int64(field // Int32(8))).to(
                                        Uint32
                                    )
                                    >> (field % Int32(8)).to(Uint32)
                                ) & Uint32((1 << self.storage.bits) - 1)
                        value |= (base + offset) << Uint32(column * 8)
            values[quarter] = value
        return values

    @cute.jit
    def destination_words(self, storage, expert, group, lane):
        row_group = group // Int32(self.storage.column_groups)
        column_group = group % Int32(self.storage.column_groups)
        nr = cutlass.min(Int32(128), Int32(self.storage.rows) - row_group * Int32(128))
        nc = cutlass.min(
            Int32(4), Int32(self.storage.columns) - column_group * Int32(4)
        )
        codes = (
            Int64(self.storage.codes)
            + Int64(expert) * Int64(self.storage.code_bytes)
            + Int64(row_group)
            * Int64(128 * self.storage.columns * self.storage.bits // 8)
            + Int64(column_group * nr) * Int64(self.storage.bits) // Int64(2)
        )
        bases = Int64(self.storage.bases) + Int64(
            expert * self.storage.rows + row_group * 128
        )
        values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        for quarter in cutlass.range_constexpr(4):
            values[quarter] = Uint32(0)
        vectorized = Boolean(True)
        if cutlass.const_expr(self.destination == 2):
            vectorized = nc == Int32(4)
        elif cutlass.const_expr(self.destination == 3):
            vectorized = nc != Int32(3)
        if vectorized:
            if lane * Int32(16) < nr * nc:
                if cutlass.const_expr(self.storage.bits == 4):
                    packed = _load(
                        cute.recast_ptr(storage + codes, dtype=cutlass.Uint64),
                        lane.to(Int64),
                    ).to(Uint64)
                else:
                    packed = _load(
                        cute.recast_ptr(storage + codes, dtype=cutlass.Uint16),
                        lane.to(Int64),
                    ).to(Uint64)
                for quarter in cutlass.range_constexpr(4):
                    offset = (packed >> Uint64(quarter * 4 * self.storage.bits)).to(
                        Uint32
                    )
                    if cutlass.const_expr(self.storage.bits == 4):
                        offset &= Uint32(0xFFFF)
                        offset = (offset | (offset << Uint32(8))) & Uint32(0x00FF00FF)
                        offset = (offset | (offset << Uint32(4))) & Uint32(0x0F0F0F0F)
                    else:
                        offset &= Uint32(15)
                        offset = (offset | (offset << Uint32(14))) & Uint32(0x00030003)
                        offset = (offset | (offset << Uint32(7))) & Uint32(0x01010101)
                    index = lane * Int32(16) + Int32(quarter * 4)
                    base = Uint32(0)
                    if cutlass.const_expr(self.destination == 1):
                        row = index % nr
                        base = _load(
                            cute.recast_ptr(storage + bases, dtype=cutlass.Uint32),
                            (row // Int32(4)).to(Int64),
                        ).to(Uint32)
                    else:
                        row = index // nc
                        if nc == Int32(4):
                            base = _load(storage, bases + row.to(Int64)).to(
                                Uint32
                            ) * Uint32(0x01010101)
                        elif nc == Int32(2):
                            pair = _load(
                                cute.recast_ptr(storage + bases, dtype=cutlass.Uint16),
                                (row // Int32(2)).to(Int64),
                            ).to(Uint32)
                            base = (
                                (pair & Uint32(255))
                                | ((pair & Uint32(0xFF00)) << Uint32(8))
                            ) * Uint32(0x101)
                        else:
                            base = _load(
                                cute.recast_ptr(storage + bases, dtype=cutlass.Uint32),
                                (row // Int32(4)).to(Int64),
                            ).to(Uint32)
                    values[quarter] = base + offset
        else:
            for quarter in cutlass.range_constexpr(4):
                value = Uint32(0)
                for byte in cutlass.range_constexpr(4):
                    index = lane * Int32(16) + Int32(quarter * 4 + byte)
                    if cutlass.const_expr(self.destination == 2):
                        row, column = index // Int32(4), index % Int32(4)
                    else:
                        row, column = index // nc, index % nc
                    if row < nr and column < nc:
                        field = (row * nc + column) * Int32(self.storage.bits)
                        offset = (
                            _load(storage, codes + (field // Int32(8)).to(Int64)).to(
                                Uint32
                            )
                            >> (field % Int32(8)).to(Uint32)
                        ) & Uint32((1 << self.storage.bits) - 1)
                        base = _load(storage, bases + row.to(Int64)).to(Uint32)
                        value |= (base + offset) << Uint32(byte * 8)
                values[quarter] = value
        return values

    @cute.jit
    def group_output_offset(self, group):
        row_group = group // Int32(self.storage.column_groups)
        column_group = group % Int32(self.storage.column_groups)
        if cutlass.const_expr(self.destination == 1):
            return Int64(column_group * 4) * Int64(self.rows) + Int64(row_group * 128)
        elif cutlass.const_expr(self.destination == 2):
            return Int64(
                (
                    row_group // Int32(2) * Int32(self.storage.column_groups)
                    + column_group
                )
                * Int32(1024)
                + row_group % Int32(2) * Int32(512)
            )
        elif cutlass.const_expr(self.destination == 3):
            section = group // Int32(self.storage.groups)
            local_row = row_group % Int32((self.storage.rows + 127) // 128) * Int32(128)
            nr = cutlass.min(Int32(128), Int32(self.storage.rows) - local_row)
            return Int64(section * self.storage.rows + local_row) * Int64(
                self.columns
            ) + Int64(column_group * nr * 4)
        else:
            return Int64(group) * Int64(512)

    @cute.jit
    def store_group(self, output, group, lane, v0, v1, v2, v3):
        from b12x._lib.quant.nvfp4_csf_indexed import _store_scale_vector

        offset = lane * Int32(16)
        active = Boolean(True)
        if cutlass.const_expr(self.destination in (1, 3)):
            row_group = (
                group % Int32(self.storage.groups) // Int32(self.storage.column_groups)
            )
            column_group = group % Int32(self.storage.column_groups)
            nr = cutlass.min(
                Int32(128), Int32(self.storage.rows) - row_group * Int32(128)
            )
            nc = cutlass.min(
                Int32(4), Int32(self.storage.columns) - column_group * Int32(4)
            )
            active = offset < nr * nc
            if cutlass.const_expr(self.destination == 1):
                offset = offset // nr * Int32(self.rows) + offset % nr
        if active:
            address = (
                output.toint() + self.group_output_offset(group) + offset.to(Int64)
            )
            _store_scale_vector(address, v0, v1, v2, v3)

    @cute.jit
    def patch_blocks(self, storage, expert, group, lane, target):
        row_group = group // Int32(self.storage.column_groups)
        column_group = group % Int32(self.storage.column_groups)
        nrb = cutlass.min(
            Int32(4), Int32(self.storage.rows // 32) - row_group * Int32(4)
        )
        row_block = lane // Int32(4) % Int32(4)
        pair = lane // Int32(16)
        if row_block < nrb and column_group * Int32(4) + pair * Int32(2) < Int32(
            self.storage.columns
        ):
            page = group // Int32(32)
            block = (
                row_group * Int32(4 * self.storage.column_blocks)
                + nrb * column_group * Int32(2)
                + pair * nrb
                + row_block
            )
            words = cute.recast_ptr(storage, dtype=cutlass.Uint32)
            origin = _load(
                words,
                Int64(self.storage.page_bases // 4)
                + Int64(expert) * Int64(self.storage.pages)
                + Int64(page),
            ).to(Uint32) >> Uint32(1)
            prefixes = cute.recast_ptr(
                storage + Int64(self.storage.offsets), dtype=cutlass.Uint16
            )
            prefix = Int64(expert) * Int64(
                self.storage.blocks + self.storage.pages
            ) + Int64(block + page)
            entry = (
                origin
                + _load(prefixes, prefix).to(Uint32)
                + (lane % Int32(4)).to(Uint32)
            )
            end = origin + _load(prefixes, prefix + Int64(1)).to(Uint32)
            records = cute.recast_ptr(
                storage + Int64(self.storage.source_records), dtype=cutlass.Uint16
            )
            while entry < end:
                record = _load(records, entry.to(Int64)).to(Uint32)
                position = (record & Uint32(255)).to(Int32)
                destination = (
                    position // Int32(2) * Int32(16)
                    + row_block * Int32(4)
                    + pair * Int32(2)
                    + position % Int32(2)
                )
                cute.make_tensor(target + destination, cute.make_layout(1))[0] = (
                    record >> Uint32(8)
                ).to(cutlass.Uint8)
                entry += Uint32(4)

    @cute.jit
    def group_record_bounds(self, storage, expert, group):
        page = group // Int32(32)
        words = cute.recast_ptr(storage, dtype=cutlass.Uint32)
        origin = _load(
            words,
            Int64(self.storage.page_bases // 4)
            + Int64(expert * self.storage.pages + page),
        ).to(Uint32)
        prefixes = cute.recast_ptr(
            storage + Int64(self.storage.offsets), dtype=cutlass.Uint16
        )
        index = Int64(expert * self.storage.prefixes + group + page)
        first = origin + _load(prefixes, index).to(Uint32)
        end = origin + _load(prefixes, index + Int64(1)).to(Uint32)
        return first, end

    @cute.jit
    def group_value_offset(self, storage):
        if cutlass.const_expr(self.group_record_bytes == 3):
            return _load(cute.recast_ptr(storage, dtype=cutlass.Uint64), Int64(0)).to(
                Int64
            )
        else:
            return Int64(0)

    @cute.jit
    def load_group_record(self, storage, entry, value_offset):
        if cutlass.const_expr(self.group_record_bytes == 3):
            positions = cute.recast_ptr(
                storage + Int64(self.storage.source_records), dtype=cutlass.Uint16
            )
            position = _load(positions, entry.to(Int64)).to(Int32)
            value = _load(storage, value_offset + entry.to(Int64)).to(cutlass.Uint8)
        else:
            records = cute.recast_ptr(
                storage + Int64(self.storage.source_records), dtype=cutlass.Uint32
            )
            record = _load(records, entry.to(Int64)).to(Uint32)
            position = (record & Uint32(511)).to(Int32)
            value = (record >> Uint32(16)).to(cutlass.Uint8)
        return position, value

    @cute.jit
    def write_group_record(self, group, target, position, value):
        destination = position
        if cutlass.const_expr(self.destination == 0):
            row = position // Int32(4)
            destination = (
                row % Int32(32) * Int32(16)
                + row // Int32(32) * Int32(4)
                + position % Int32(4)
            )
        elif cutlass.const_expr(self.destination == 1 and not self.shared_exceptions):
            nr = cutlass.min(
                Int32(128),
                Int32(self.storage.rows)
                - group // Int32(self.storage.column_groups) * Int32(128),
            )
            destination = position // nr * Int32(self.rows) + position % nr
        cute.make_tensor(target + destination, cute.make_layout(1))[0] = value

    @cute.jit
    def patch_group_records(self, storage, group, lane, target, first, end):
        value_offset = self.group_value_offset(storage)
        entry = first + lane.to(Uint32)
        while entry < end:
            position, value = self.load_group_record(storage, entry, value_offset)
            self.write_group_record(group, target, position, value)
            entry += Uint32(32)

    @cute.jit
    def patch_group(self, storage, expert, group, lane, target):
        if cutlass.const_expr(self.group_record_bytes == 0):
            self.patch_blocks(storage, expert, group, lane, target)
        else:
            first, end = self.group_record_bounds(storage, expert, group)
            self.patch_group_records(storage, group, lane, target, first, end)

    @cute.jit
    def decode_prefetched_group(self, storage, output, expert, task, tid):
        smem = cutlass.utils.SmemAllocator()
        shared = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(self.warps * 128), byte_alignment=16
        )
        warp, lane = tid // Int32(32), tid % Int32(32)
        group = task * Int32(self.groups_per_task) + warp
        if warp < Int32(self.warps) and group < Int32(self.groups):
            first, end = Uint32(0), Uint32(0)
            source_group = Int32(0)
            position, value = Int32(0), cutlass.Uint8(0)
            values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
            values.fill(Uint32(0))
            if group < Int32(self.storage.groups * self.sections):
                source_expert = expert * Int32(self.sections) + group // Int32(
                    self.storage.groups
                )
                source_group = group % Int32(self.storage.groups)
                value_offset = self.group_value_offset(storage)
                first, end = self.group_record_bounds(
                    storage, source_expert, source_group
                )
                entry = first + lane.to(Uint32)
                if entry < end:
                    position, value = self.load_group_record(
                        storage, entry, value_offset
                    )
                values = self.group_words(storage, source_expert, source_group, lane)
            target = cute.recast_ptr(
                shared.iterator, dtype=cutlass.Uint8
            ) + warp * Int32(512)
            address = target.toint().to(Int32) + lane * Int32(16)
            st_shared_v4_u32(address, values[0], values[1], values[2], values[3])
            cute.arch.sync_warp()
            if first + lane.to(Uint32) < end:
                self.write_group_record(source_group, target, position, value)
            self.patch_group_records(
                storage, source_group, lane, target, first + Uint32(32), end
            )
            cute.arch.sync_warp()
            v0, v1, v2, v3 = ld_shared_v4_u32(address)
            self.store_group(
                output + Int64(expert) * Int64(self.rows * self.columns),
                group,
                lane,
                v0,
                v1,
                v2,
                v3,
            )

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        if cutlass.const_expr(
            self.shared_exceptions
            and self.group_record_bytes != 0
            and self.groups_per_task <= 8
            and (
                self.destination != 0
                or self.storage.rows % 128
                or self.storage.columns % 4
            )
        ):
            self.decode_prefetched_group(*tensors, expert, task, tid)
        else:
            self.decode_staged(tensors, expert, task, tid)

    @cute.jit
    def decode_staged(self, tensors, expert, task, tid):
        storage, output = tensors
        if cutlass.const_expr(self.shared_exceptions):
            smem = cutlass.utils.SmemAllocator()
            shared = smem.allocate_tensor(
                cutlass.Uint32,
                cute.make_layout(self.groups_per_task * 128),
                byte_alignment=16,
            )
        warp, lane = tid // Int32(32), tid % Int32(32)
        expert_output = output + Int64(expert) * Int64(self.rows * self.columns)
        for chunk in cutlass.range_constexpr(self.groups_per_task // self.warps):
            local_group = Int32(chunk * self.warps) + warp
            group = task * Int32(self.groups_per_task) + local_group
            if warp < Int32(self.warps) and group < Int32(self.groups):
                values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
                for quarter in cutlass.range_constexpr(4):
                    values[quarter] = Uint32(0)
                if group < Int32(self.storage.groups * self.sections):
                    source_expert = expert * Int32(self.sections) + group // Int32(
                        self.storage.groups
                    )
                    source_group = group % Int32(self.storage.groups)
                    values = self.group_words(
                        storage, source_expert, source_group, lane
                    )
                if cutlass.const_expr(self.shared_exceptions):
                    address = shared.iterator.toint().to(Int32) + (
                        local_group * Int32(512) + lane * Int32(16)
                    )
                    st_shared_v4_u32(
                        address, values[0], values[1], values[2], values[3]
                    )
                else:
                    self.store_group(
                        expert_output,
                        group,
                        lane,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                    )
        cute.arch.sync_warp()
        for chunk in cutlass.range_constexpr(self.groups_per_task // self.warps):
            local_group = Int32(chunk * self.warps) + warp
            group = task * Int32(self.groups_per_task) + local_group
            if warp < Int32(self.warps) and group < Int32(self.groups):
                if cutlass.const_expr(self.shared_exceptions):
                    target = cute.recast_ptr(
                        shared.iterator, dtype=cutlass.Uint8
                    ) + local_group * Int32(512)
                else:
                    target = expert_output + self.group_output_offset(group)
                if group < Int32(self.storage.groups * self.sections):
                    source_expert = expert * Int32(self.sections) + group // Int32(
                        self.storage.groups
                    )
                    source_group = group % Int32(self.storage.groups)
                    self.patch_group(storage, source_expert, source_group, lane, target)
        if cutlass.const_expr(self.shared_exceptions):
            cute.arch.sync_warp()
            for chunk in cutlass.range_constexpr(self.groups_per_task // self.warps):
                local_group = Int32(chunk * self.warps) + warp
                group = task * Int32(self.groups_per_task) + local_group
                if warp < Int32(self.warps) and group < Int32(self.groups):
                    address = shared.iterator.toint().to(Int32) + (
                        local_group * Int32(512) + lane * Int32(16)
                    )
                    v0, v1, v2, v3 = ld_shared_v4_u32(address)
                    self.store_group(
                        expert_output,
                        group,
                        lane,
                        v0,
                        v1,
                        v2,
                        v3,
                    )
