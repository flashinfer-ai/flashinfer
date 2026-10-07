"""Expand MXFP4-CSF scales directly into the standard W4A8 scale layout.

Preparation permutes selector bits and exception positions into the native
scale order. Row bases remain one byte per row. Runtime expansion writes only
caller-owned scale buffers; one CTA owns each contiguous 8 KiB region, including
its exceptions. Compact N64 tails and padded N256 tiles follow native W4A8
weight preparation, including its E8M0 clamp to 247.
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
class Mxfp4CsfBatch:
    bases: torch.Tensor
    selectors: torch.Tensor
    exceptions: torch.Tensor
    task_offsets: torch.Tensor
    rows: int
    columns: int
    group_rows: int
    compact: bool
    logical_columns: int

    @property
    def num_experts(self):
        return self.bases.shape[0]

    @property
    def geometry(self):
        return (
            self.rows,
            self.columns,
            self.group_rows,
            self.compact,
            self.logical_columns,
        )


def repack_mxfp4_csf_batch(batch, *, compact, group_rows, row_rotation=0):
    """Reorder compressed bytes at load time without retaining expanded scales."""
    batch.validate()
    e, r, c = batch.num_experts, batch.rows, batch.columns
    if group_rows <= 0 or r % group_rows:
        raise ValueError("W4A8 CSF requires whole row groups")
    if compact and (group_rows % 64 or c % 2):
        raise ValueError("Compact W4A8 CSF requires N64/K64 geometry")
    if not 0 <= row_rotation < r:
        raise ValueError("CSF row rotation must lie within the logical row extent")
    # Map every native output byte to a logical source position. This is
    # immutable model geometry; it is never retained or built during replay.
    if compact:
        out_rows, out_cols = r, c
        row_map = (np.arange(r) + row_rotation) % r
        chunks = []
        for group in range(0, r, group_rows):
            for tile in range(0, group_rows, 128):
                rr = row_map[group + tile : group + min(tile + 128, group_rows)]
                for col in range(0, c, 4):
                    chunks.append(
                        (rr[:, None] * c + np.arange(col, min(col + 4, c))).ravel()
                    )
        positions = np.concatenate(chunks)
    else:
        out_rows, out_cols = (r + 255) // 256 * 256, (c + 3) // 4 * 4
        row_map = np.arange(out_rows)
        # Native preparation pads gated halves independently when needed.
        if r == 2 * group_rows and out_rows != r:
            half_pad = (group_rows + 127) // 128 * 128
            half = row_map // half_pad
            within = row_map % half_pad
            valid_rows = (half < 2) & (within < group_rows)
            row_map = (half * group_rows + within + row_rotation) % r
        else:
            valid_rows = row_map < r
            row_map = (row_map + row_rotation) % r
        row_map = np.where(valid_rows, row_map, -1)
        logical = row_map[:, None] * c + np.arange(out_cols)
        logical[(row_map < 0), :] = -1
        logical[:, c:] = -1
        positions = (
            logical.reshape(out_rows // 256, 256, out_cols // 4, 4)
            .transpose(0, 2, 1, 3)
            .ravel()
        )
    size = len(positions)
    if size >= 1 << 24:
        raise ValueError("Native CSF scale positions exceed the 24-bit record field")
    inverse = np.empty(r * c, dtype=np.uint32)
    valid = positions >= 0
    inverse[positions[valid]] = np.flatnonzero(valid)
    fixed = batch.fixed.cpu().numpy()
    records = batch.exceptions.cpu().numpy()
    bounds = batch.exception_offsets.cpu().numpy()
    bases, selectors, exceptions, partitions = [], [], [], []
    cursor = 0
    tasks = (size + 8191) // 8192
    for expert in range(e):
        base = fixed[expert, :, :16].reshape(r)
        bits = np.unpackbits(
            fixed[expert, :, 16:].reshape(r, -1), axis=1, bitorder="little"
        )[:, :c]
        # Native scale preparation clamps E8M0 bytes above 247. Clamp the
        # palette and clear increments that would cross the same boundary.
        bits[base >= 247] = 0
        base = np.minimum(base, 247)
        native_base = base[np.maximum(row_map, 0)].copy()
        native_base[row_map < 0] = 0
        native_bits = bits.ravel()[np.maximum(positions, 0)].copy()
        native_bits[~valid] = 0
        words = records[bounds[expert] : bounds[expert + 1]]
        loc = inverse[words & 0xFFFFFF]
        values = np.minimum(words >> 24, 247)
        order = np.argsort(loc)
        loc = loc[order]
        packed = loc | (values[order] << 24)
        partitions.append(np.searchsorted(loc, np.arange(tasks + 1) * 8192) + cursor)
        cursor += len(packed)
        bases.append(native_base)
        selectors.append(np.packbits(native_bits, bitorder="little"))
        exceptions.append(packed)
    device = batch.fixed.device
    return Mxfp4CsfBatch(
        torch.from_numpy(np.stack(bases)).to(device),
        torch.from_numpy(np.stack(selectors)).to(device),
        torch.from_numpy(np.concatenate(exceptions).astype(np.uint32)).to(device),
        torch.from_numpy(np.stack(partitions).astype(np.int64)).to(device),
        out_rows,
        out_cols,
        group_rows,
        bool(compact),
        c,
    )


class _Plane:
    def __init__(self, geometry):
        self.rows, self.columns, self.group_rows, self.compact, self.logical_columns = (
            geometry
        )
        self.size = self.rows * self.columns
        self.tasks = (self.size + 8191) // 8192

    @cute.jit
    def tensors(self, base, bits, exceptions, offsets, out, experts, entries):
        return (
            cute.make_tensor(base, cute.make_layout((Int64(experts) * self.rows,))),
            cute.make_tensor(
                bits, cute.make_layout((Int64(experts) * (self.size // 8),))
            ),
            cute.make_tensor(exceptions, cute.make_layout((entries,))),
            cute.make_tensor(
                offsets, cute.make_layout((Int64(experts) * (self.tasks + 1),))
            ),
            cute.make_tensor(out, cute.make_layout((Int64(experts) * self.size,))),
        )

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        bases, bits, exceptions, offsets, out = tensors
        base16 = cute.recast_tensor(bases, cutlass.Uint16)
        base32 = cute.recast_tensor(bases, cutlass.Uint32)
        out64 = cute.recast_tensor(out, cutlass.Uint64)
        local = task * Int32(8192) + tid * Int32(8)
        end = cutlass.min((task + Int32(1)) * Int32(8192), Int32(self.size))
        while local < end:
            cols = Int32(4)
            if cutlass.const_expr(self.compact):
                group = local // Int32(self.group_rows * self.columns)
                within = local % Int32(self.group_rows * self.columns)
                nt = within // Int32(128 * self.columns)
                nr = cutlass.min(Int32(128), Int32(self.group_rows) - nt * Int32(128))
                within -= nt * Int32(128 * self.columns)
                kt = within // (nr * Int32(4))
                cols = cutlass.min(Int32(4), Int32(self.columns) - kt * Int32(4))
                row = (
                    group * Int32(self.group_rows)
                    + nt * Int32(128)
                    + (within - kt * nr * Int32(4)) // cols
                )
            else:
                row = local // Int32(256 * self.columns) * Int32(256) + (
                    local % Int32(1024)
                ) // Int32(4)
            address = Int64(expert) * Int64(self.rows) + Int64(row)
            packed_base = Uint64(0)
            if cols == Int32(4):
                b = base16[address >> Int64(1)].to(Uint32)
                packed_base = ((b & Uint32(255)) * Uint32(0x01010101)).to(Uint64)
                packed_base |= ((b >> Uint32(8)) * Uint32(0x01010101)).to(
                    Uint64
                ) << Uint64(32)
            else:
                b = base32[address >> Int64(2)].to(Uint64)
                b = (b | (b << Uint64(16))) & Uint64(0x0000FFFF0000FFFF)
                b = (b | (b << Uint64(8))) & Uint64(0x00FF00FF00FF00FF)
                packed_base = b * Uint64(257)
            if cutlass.const_expr(not self.compact and self.logical_columns % 4):
                column = (local // Int32(1024)) % Int32(self.columns // 4) * Int32(4)
                if column == Int32(self.logical_columns // 4 * 4):
                    mask = (1 << (8 * (self.logical_columns % 4))) - 1
                    packed_base &= Uint64(mask | (mask << 32))
            index = Int64(expert) * Int64(self.size // 8) + Int64(local // Int32(8))
            spread = bits[index].to(Uint64)
            spread = (spread | (spread << Uint64(28))) & Uint64(0x0000000F0000000F)
            spread = (spread | (spread << Uint64(14))) & Uint64(0x0003000300030003)
            spread = (spread | (spread << Uint64(7))) & Uint64(0x0101010101010101)
            out64[index] = packed_base + spread
            local += Int32(2048)
        cute.arch.sync_threads()
        part = Int64(expert) * Int64(self.tasks + 1) + Int64(task)
        entry = offsets[part] + Int64(tid)
        while entry < offsets[part + Int64(1)]:
            word = exceptions[entry]
            destination = Int64(expert) * Int64(self.size) + (
                word & Uint32(0xFFFFFF)
            ).to(Int64)
            out[destination] = (word >> Uint32(24)).to(cutlass.Uint8)
            entry += Int64(256)


class _Pair:
    def __init__(self, first, second):
        self.first, self.second = _Plane(first), _Plane(second)

    @cute.jit
    def __call__(
        self,
        b13: cute.Pointer,
        s13: cute.Pointer,
        e13: cute.Pointer,
        p13: cute.Pointer,
        o13: cute.Pointer,
        b2: cute.Pointer,
        s2: cute.Pointer,
        e2: cute.Pointer,
        p2: cute.Pointer,
        o2: cute.Pointer,
        ids_ptr: cute.Pointer,
        experts: Int32,
        entries13: Int64,
        entries2: Int64,
        capacity: Int32,
        all_experts: Int32,
        stream: cuda.CUstream,
    ):
        first = self.first.tensors(b13, s13, e13, p13, o13, experts, entries13)
        second = self.second.tensors(b2, s2, e2, p2, o2, experts, entries2)
        ids = cute.make_tensor(ids_ptr, cute.make_layout((capacity,)))
        self.kernel(first, second, ids, experts, all_experts).launch(
            grid=(capacity * Int32(self.first.tasks + self.second.tasks), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, first, second, ids, experts: Int32, all_experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        slot = Int32(block) // Int32(self.first.tasks + self.second.tasks)
        task = Int32(block) % Int32(self.first.tasks + self.second.tasks)
        expert = slot
        if all_experts == Int32(2):
            if ids[slot] <= 0:
                expert = Int32(-1)
        elif all_experts == Int32(0):
            raw = ids[slot].to(Int64)
            expert = Int32(-1)
            if raw >= Int64(0) and raw < Int64(experts):
                expert = raw.to(Int32)
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
def compile_mxfp4_csf_pair(first, second, ids64=False):
    kernel = _Pair(first, second)
    key = first, second, bool(ids64)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    types = (cutlass.Uint8, cutlass.Uint8, cutlass.Uint32, cutlass.Int64, cutlass.Uint8)
    plane = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16) for t in types
    )
    return b12x_compile(
        kernel,
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
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.mxfp4_csf_pair", 2, key),
    )


@dataclass(frozen=True)
class Mxfp4CsfDecoder:
    first: Mxfp4CsfBatch
    second: Mxfp4CsfBatch
    programs: tuple
    routing_programs: tuple
    active: torch.Tensor

    @classmethod
    def prepare(cls, first, second, out13, out2):
        from b12x._lib.quant.csf_routing import compile_csf_active_experts

        if first.num_experts != second.num_experts:
            raise ValueError("MXFP4-CSF projections must have equal expert counts")
        if first.bases.device != second.bases.device:
            raise ValueError("MXFP4-CSF projections must be on one CUDA device")
        if (
            out13.data_ptr() < out2.data_ptr() + out2.numel() * out2.element_size()
            and out2.data_ptr()
            < out13.data_ptr() + out13.numel() * out13.element_size()
        ):
            raise ValueError("MXFP4-CSF output buffers must not overlap")
        for plane, output in ((first, out13), (second, out2)):
            if (
                output.element_size() * output.numel()
                != plane.num_experts * plane.rows * plane.columns
                or not output.is_contiguous()
                or output.device != plane.bases.device
            ):
                raise ValueError(
                    "MXFP4-CSF output must match native W4A8 scale storage"
                )
        return cls(
            first,
            second,
            tuple(
                compile_mxfp4_csf_pair(first.geometry, second.geometry, ids64)
                for ids64 in (False, True)
            ),
            tuple(compile_csf_active_experts(ids64) for ids64 in (False, True)),
            torch.empty(
                first.num_experts, dtype=torch.int32, device=first.bases.device
            ),
        )

    def decode(self, ids, out13, out2):
        from b12x._lib.quant.csf_routing import mark_active_experts

        if (
            ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or ids.device != self.first.bases.device
        ):
            raise ValueError("MXFP4-CSF routes must be contiguous int32/int64")
        all_experts = 0
        if ids.numel() >= 16 * self.first.num_experts:
            all_experts = 1
        elif ids.numel() >= 64:
            mark_active_experts(
                ids, self.active, self.routing_programs[int(ids.dtype == torch.int64)]
            )
            ids, all_experts = self.active, 2
        capacity = self.first.num_experts if all_experts else ids.numel()
        if not capacity:
            return

        def ptr(t, dtype, align=16):
            return make_ptr(
                dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
            )

        args = []
        for plane, output in ((self.first, out13), (self.second, out2)):
            args.extend(
                (
                    ptr(plane.bases, cutlass.Uint8),
                    ptr(plane.selectors, cutlass.Uint8),
                    ptr(plane.exceptions, cutlass.Uint32),
                    ptr(plane.task_offsets, cutlass.Int64),
                    ptr(output, cutlass.Uint8),
                )
            )
        self.programs[int(ids.dtype == torch.int64)](
            *args,
            ptr(
                ids,
                cutlass.Int64 if ids.dtype == torch.int64 else cutlass.Int32,
                8 if ids.dtype == torch.int64 else 4,
            ),
            self.first.num_experts,
            self.first.exceptions.numel(),
            self.second.exceptions.numel(),
            capacity,
            all_experts,
            current_cuda_stream(),
        )
