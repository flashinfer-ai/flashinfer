"""Graph-stable GPU expansion of fixed-stride X4T UE8M0 scale planes.

X4T leaves the MXFP4 E2M1 nibble plane untouched.  Each independently
addressable 16-row scale slab contains 16 adjacent-palette bases followed by
one selector bit per scale.  A sorted uint32 side stream supplies the sparse
non-palette values.  The device expansion is therefore two ordered kernels:

1. a fixed-stride base/selector decode;
2. a sparse exception scatter.

Both kernels consume a fixed-capacity list of active expert IDs and write a
reusable expert-indexed scratch tensor.  No allocation, host readback, prefix
scan, or variable-length tile parsing occurs during graph replay.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


X4T_TILE_ROWS = 16
X4T_POSITION_MASK = (1 << 24) - 1
_GRID_CTAS_PER_SM = 4

# Kimi-K3 TP12 scale geometry.  The specialized W4A16 decoder below owns one
# disjoint packed-row region per CTA, so it can apply sparse overrides after a
# CTA barrier without a second launch.
_TP12_W13_ROWS = 512
_TP12_W13_COLUMNS = 112
_TP12_W13_ROW_ROTATION = 256
_TP12_W13_ROWS_PER_TASK = 128
_TP12_W13_TASKS = _TP12_W13_ROWS // _TP12_W13_ROWS_PER_TASK
_TP12_W2_ROWS = 3584
_TP12_W2_COLUMNS = 8
_TP12_W2_ROWS_PER_TASK = 896
_TP12_W2_TASKS = _TP12_W2_ROWS // _TP12_W2_ROWS_PER_TASK
_TP12_TASKS_PER_EXPERT = _TP12_W13_TASKS + _TP12_W2_TASKS
_TP12_THREADS = 256


@dataclass(frozen=True)
class X4TScaleBatch:
    """Persistent GPU representation of same-shaped X4T scale planes."""

    fixed: torch.Tensor
    exceptions: torch.Tensor
    exception_offsets: torch.Tensor
    rows: int
    columns: int
    task_exception_offsets: torch.Tensor | None = None
    exception_task_rows: int = 0
    exception_row_rotation: int = 0

    @property
    def num_experts(self) -> int:
        return int(self.fixed.shape[0])

    @property
    def tile_count(self) -> int:
        return int(self.rows) // X4T_TILE_ROWS

    @property
    def selector_bytes(self) -> int:
        return (int(self.columns) + 7) // 8

    @property
    def tile_bytes(self) -> int:
        return X4T_TILE_ROWS * (1 + self.selector_bytes)

    def validate(self) -> None:
        rows = int(self.rows)
        columns = int(self.columns)
        if rows <= 0 or rows % X4T_TILE_ROWS:
            raise ValueError("X4T rows must be a positive multiple of 16")
        if not 1 <= columns <= 255:
            raise ValueError("X4T columns must lie in 1..255")
        if self.fixed.dtype != torch.uint8 or self.fixed.ndim != 3:
            raise TypeError("X4T fixed stream must be uint8 [E,tiles,tile_bytes]")
        if tuple(self.fixed.shape[1:]) != (self.tile_count, self.tile_bytes):
            raise ValueError(
                "X4T fixed stream shape disagrees with rows/columns: "
                f"got {tuple(self.fixed.shape)}, expected "
                f"[E,{self.tile_count},{self.tile_bytes}]"
            )
        if self.exceptions.dtype != torch.uint32 or self.exceptions.ndim != 1:
            raise TypeError("X4T exceptions must be a one-dimensional uint32 tensor")
        if self.exception_offsets.dtype != torch.int64 or self.exception_offsets.ndim != 1:
            raise TypeError("X4T exception offsets must be a one-dimensional int64 tensor")
        if int(self.exception_offsets.numel()) != self.num_experts + 1:
            raise ValueError("X4T exception offsets must contain E+1 entries")
        tensors = (self.fixed, self.exceptions, self.exception_offsets)
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("X4T tensors must be contiguous")
        if any(tensor.device != self.fixed.device for tensor in tensors):
            raise ValueError("X4T tensors must share one device")
        if self.fixed.device.type != "cuda":
            raise ValueError("X4T runtime tensors must reside on CUDA")
        task_offsets = self.task_exception_offsets
        if task_offsets is not None:
            task_rows = int(self.exception_task_rows)
            rotation = int(self.exception_row_rotation)
            if task_rows <= 0 or rows % task_rows:
                raise ValueError("X4T exception task rows must divide the row axis")
            if rotation < 0 or rotation >= rows or rotation % task_rows:
                raise ValueError(
                    "X4T exception row rotation must align to an exception task"
                )
            task_count = rows // task_rows
            if (
                task_offsets.dtype != torch.int64
                or tuple(task_offsets.shape) != (self.num_experts, task_count + 1)
                or not task_offsets.is_contiguous()
                or task_offsets.device != self.fixed.device
            ):
                raise ValueError(
                    "X4T task exception offsets must be contiguous int64 "
                    "[E,tasks+1] on the batch device"
                )
        elif int(self.exception_task_rows) or int(self.exception_row_rotation):
            raise ValueError("X4T exception task geometry requires task offsets")


def make_x4t_scale_batch(
    fixed_planes: Sequence[torch.Tensor],
    exception_planes: Sequence[torch.Tensor],
    *,
    rows: int,
    columns: int,
    device: torch.device | str,
    exception_task_rows: int | None = None,
    exception_row_rotation: int = 0,
) -> X4TScaleBatch:
    """Upload canonical per-expert X4T components into one runtime batch.

    This helper is a load/warmup operation.  The returned tensors are the
    persistent compressed representation; full scale planes are never built.
    """

    if len(fixed_planes) != len(exception_planes) or not fixed_planes:
        raise ValueError("X4T component lists must be nonempty and equally sized")
    rows = int(rows)
    columns = int(columns)
    if rows <= 0 or rows % X4T_TILE_ROWS:
        raise ValueError("X4T rows must be a positive multiple of 16")
    if not 1 <= columns <= 255:
        raise ValueError("X4T columns must lie in 1..255")
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS * (1 + (columns + 7) // 8)
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    source_device = fixed_planes[0].device
    if source_device.type != "cpu" and source_device != resolved:
        raise ValueError("X4T components must reside on CPU or the target device")
    normalized_fixed: list[torch.Tensor] = []
    normalized_exceptions: list[torch.Tensor] = []
    offsets = [0]
    task_rows = 0 if exception_task_rows is None else int(exception_task_rows)
    rotation = int(exception_row_rotation)
    if task_rows:
        if rows % task_rows or rotation < 0 or rotation >= rows or rotation % task_rows:
            raise ValueError(
                "X4T exception task geometry must be aligned within the row axis"
            )
        task_count = rows // task_rows
    else:
        if rotation:
            raise ValueError("X4T exception row rotation requires task rows")
        task_count = 0
    task_offsets: list[list[int]] = []
    for fixed, exceptions in zip(fixed_planes, exception_planes, strict=True):
        if fixed.device != source_device or fixed.dtype != torch.uint8:
            raise TypeError(
                "X4T fixed plane components must be same-device uint8 tensors"
            )
        if int(fixed.numel()) != tile_count * tile_bytes:
            raise ValueError("X4T fixed plane component has the wrong byte length")
        if exceptions.device != source_device or exceptions.dtype != torch.uint32:
            raise TypeError(
                "X4T exception plane components must be same-device uint32 tensors"
            )
        exception_words = exceptions.contiguous().view(-1).to(torch.int64)
        if int(exception_words.numel()):
            positions = exception_words & X4T_POSITION_MASK
            if int(positions[-1]) >= rows * columns or bool(
                (positions[1:] <= positions[:-1]).any()
            ):
                raise ValueError(
                    "X4T exception positions must be in range and strictly increasing"
                )
        normalized_fixed.append(fixed.contiguous().view(tile_count, tile_bytes))
        if task_rows:
            positions = exception_words & X4T_POSITION_MASK
            source_rows = positions // columns
            rotated_rows = (source_rows - rotation) % rows
            task_ids = rotated_rows // task_rows
            order = torch.argsort(task_ids, stable=True)
            ordered = exception_words[order].to(torch.uint32)
            counts = torch.bincount(task_ids, minlength=task_count)
            task_begin = offsets[-1]
            per_task = [task_begin]
            for count in counts.tolist():
                per_task.append(per_task[-1] + int(count))
            task_offsets.append(per_task)
            normalized_exceptions.append(ordered)
        else:
            normalized_exceptions.append(exceptions.contiguous().view(-1))
        offsets.append(offsets[-1] + int(exceptions.numel()))

    fixed_gpu = torch.stack(normalized_fixed).to(device=resolved, non_blocking=False)
    if offsets[-1]:
        exceptions_cpu = torch.cat(normalized_exceptions)
    else:
        exceptions_cpu = torch.empty((0,), dtype=torch.uint32)
    batch = X4TScaleBatch(
        fixed=fixed_gpu,
        exceptions=exceptions_cpu.to(device=resolved, non_blocking=False),
        exception_offsets=torch.tensor(offsets, dtype=torch.int64, device=resolved),
        rows=rows,
        columns=columns,
        task_exception_offsets=(
            torch.tensor(task_offsets, dtype=torch.int64, device=resolved)
            if task_rows
            else None
        ),
        exception_task_rows=task_rows,
        exception_row_rotation=rotation,
    )
    batch.validate()
    return batch


class _X4TBaseDecodeLaunch:
    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        packed_w4a16: bool,
        row_rotation: int,
        clamp_e8m0_bf16: bool,
        threads: int,
    ) -> None:
        self.rows = int(rows)
        self.columns = int(columns)
        self.tile_count = self.rows // X4T_TILE_ROWS
        self.selector_bytes = (self.columns + 7) // 8
        self.tile_bytes = X4T_TILE_ROWS * (1 + self.selector_bytes)
        self.packed_w4a16 = bool(packed_w4a16)
        self.row_rotation = int(row_rotation)
        self.clamp_e8m0_bf16 = bool(clamp_e8m0_bf16)
        self.threads = int(threads)

    @cute.jit
    def __call__(
        self,
        fixed_ptr: cute.Pointer,
        expert_ids_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        num_experts: Int32,
        active_capacity: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        fixed = cute.make_tensor(
            fixed_ptr,
            cute.make_layout((num_experts * self.tile_count * self.tile_bytes,)),
        )
        expert_ids = cute.make_tensor(
            expert_ids_ptr, cute.make_layout((active_capacity,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((num_experts * self.rows * self.columns,)),
        )
        self.kernel(fixed, expert_ids, output, num_experts, active_capacity).launch(
            grid=(grid_x, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _packed_row(self, row: Int32) -> Int32:
        rotated = row
        if cutlass.const_expr(self.row_rotation > 0):
            if row >= Int32(self.row_rotation):
                rotated = row - Int32(self.row_rotation)
            else:
                rotated = row + Int32(self.rows - self.row_rotation)
        block = rotated & ~Int32(63)
        low = rotated & Int32(7)
        high = (rotated >> Int32(3)) & Int32(7)
        swapped_high = (
            (high & Int32(4))
            | ((high & Int32(1)) << Int32(1))
            | ((high >> Int32(1)) & Int32(1))
        )
        return block | (low << Int32(3)) | swapped_high

    @cute.jit
    def _store(
        self,
        output: cute.Tensor,
        expert: Int32,
        row: Int32,
        column: Int32,
        value: Uint32,
    ) -> None:
        if cutlass.const_expr(self.clamp_e8m0_bf16):
            if value > Uint32(247):
                value = Uint32(247)
        if cutlass.const_expr(self.packed_w4a16):
            packed_row = self._packed_row(row)
            offset = (
                (expert * Int32(self.columns) + column) * Int32(self.rows)
                + packed_row
            )
        else:
            offset = (
                (expert * Int32(self.rows) + row) * Int32(self.columns) + column
            )
        output[offset] = Uint8(value)

    @cute.kernel
    def kernel(
        self,
        fixed: cute.Tensor,
        expert_ids: cute.Tensor,
        output: cute.Tensor,
        num_experts: Int32,
        active_capacity: Int32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        task = Int32(block)
        total_tasks = active_capacity * Int32(self.tile_count)
        while task < total_tasks:
            slot = task // Int32(self.tile_count)
            tile = task - slot * Int32(self.tile_count)
            expert = expert_ids[slot].to(Int32)
            if expert >= Int32(0) and expert < num_experts:
                tile_base = (
                    (expert * Int32(self.tile_count) + tile)
                    * Int32(self.tile_bytes)
                )
                element = Int32(tid)
                tile_values = Int32(X4T_TILE_ROWS * self.columns)
                while element < tile_values:
                    local_row = element // Int32(self.columns)
                    column = element - local_row * Int32(self.columns)
                    base = fixed[tile_base + local_row].to(Uint32)
                    selector_offset = (
                        tile_base
                        + Int32(X4T_TILE_ROWS)
                        + local_row * Int32(self.selector_bytes)
                        + column // Int32(8)
                    )
                    selector = fixed[selector_offset].to(Uint32)
                    bit = (selector >> Uint32(column & Int32(7))) & Uint32(1)
                    row = tile * Int32(X4T_TILE_ROWS) + local_row
                    self._store(output, expert, row, column, base + bit)
                    element += Int32(self.threads)
            task += Int32(grid)


class _X4TExceptionScatterLaunch:
    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        packed_w4a16: bool,
        row_rotation: int,
        clamp_e8m0_bf16: bool,
        threads: int,
    ) -> None:
        self.rows = int(rows)
        self.columns = int(columns)
        self.packed_w4a16 = bool(packed_w4a16)
        self.row_rotation = int(row_rotation)
        self.clamp_e8m0_bf16 = bool(clamp_e8m0_bf16)
        self.threads = int(threads)

    @cute.jit
    def __call__(
        self,
        exceptions_ptr: cute.Pointer,
        offsets_ptr: cute.Pointer,
        expert_ids_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        num_experts: Int32,
        exception_count: Int64,
        active_capacity: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        exceptions = cute.make_tensor(
            exceptions_ptr, cute.make_layout((exception_count,))
        )
        offsets = cute.make_tensor(
            offsets_ptr, cute.make_layout((num_experts + Int32(1),))
        )
        expert_ids = cute.make_tensor(
            expert_ids_ptr, cute.make_layout((active_capacity,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((num_experts * self.rows * self.columns,)),
        )
        self.kernel(exceptions, offsets, expert_ids, output, num_experts, active_capacity).launch(
            grid=(grid_x, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _packed_row(self, row: Int32) -> Int32:
        rotated = row
        if cutlass.const_expr(self.row_rotation > 0):
            if row >= Int32(self.row_rotation):
                rotated = row - Int32(self.row_rotation)
            else:
                rotated = row + Int32(self.rows - self.row_rotation)
        block = rotated & ~Int32(63)
        low = rotated & Int32(7)
        high = (rotated >> Int32(3)) & Int32(7)
        swapped_high = (
            (high & Int32(4))
            | ((high & Int32(1)) << Int32(1))
            | ((high >> Int32(1)) & Int32(1))
        )
        return block | (low << Int32(3)) | swapped_high

    @cute.kernel
    def kernel(
        self,
        exceptions: cute.Tensor,
        offsets: cute.Tensor,
        expert_ids: cute.Tensor,
        output: cute.Tensor,
        num_experts: Int32,
        active_capacity: Int32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        slot = Int32(block)
        while slot < active_capacity:
            expert = expert_ids[slot].to(Int32)
            if expert >= Int32(0) and expert < num_experts:
                begin = offsets[expert].to(Int64)
                end = offsets[expert + Int32(1)].to(Int64)
                cursor = begin + Int64(tid)
                while cursor < end:
                    entry = exceptions[cursor].to(Uint32)
                    position = entry & Uint32(X4T_POSITION_MASK)
                    value = entry >> Uint32(24)
                    if cutlass.const_expr(self.clamp_e8m0_bf16):
                        if value > Uint32(247):
                            value = Uint32(247)
                    row = Int32(position // Uint32(self.columns))
                    column = Int32(position - Uint32(row * Int32(self.columns)))
                    if cutlass.const_expr(self.packed_w4a16):
                        packed_row = self._packed_row(row)
                        output_offset = (
                            (expert * Int32(self.columns) + column)
                            * Int32(self.rows)
                            + packed_row
                        )
                    else:
                        output_offset = (
                            (expert * Int32(self.rows) + row)
                            * Int32(self.columns)
                            + column
                        )
                    output[output_offset] = Uint8(value)
                    cursor += Int64(self.threads)
            slot += Int32(grid)


class _X4TTP12W4A16DecodeLaunch:
    """One-launch Kimi-K3 TP12 decoder for both packed W4A16 scale planes.

    Work is scheduled in contiguous packed-row regions instead of logical X4T
    tiles.  Every warp therefore writes contiguous bytes.  Each CTA is the
    sole owner of its output region, which makes it safe to apply that region's
    sparse exceptions after a block barrier in the same launch.
    """

    def __init__(
        self,
        *,
        w13_row_rotation: int,
        use_expert_map: bool,
        expert_ids_unique: bool,
    ) -> None:
        self.w13_row_rotation = int(w13_row_rotation)
        self.use_expert_map = bool(use_expert_map)
        self.expert_ids_unique = bool(expert_ids_unique)

    @cute.jit
    def __call__(
        self,
        w13_fixed_ptr: cute.Pointer,
        w13_exceptions_ptr: cute.Pointer,
        w13_offsets_ptr: cute.Pointer,
        w13_task_offsets_ptr: cute.Pointer,
        w2_fixed_ptr: cute.Pointer,
        w2_exceptions_ptr: cute.Pointer,
        w2_offsets_ptr: cute.Pointer,
        w2_task_offsets_ptr: cute.Pointer,
        expert_ids_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        w13_output_ptr: cute.Pointer,
        w2_output_ptr: cute.Pointer,
        num_experts: Int32,
        w13_exception_count: Int64,
        w2_exception_count: Int64,
        active_capacity: Int32,
        route_num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        w13_fixed = cute.make_tensor(
            w13_fixed_ptr,
            cute.make_layout(
                (
                    num_experts
                    * (_TP12_W13_ROWS // X4T_TILE_ROWS)
                    * (X4T_TILE_ROWS * (1 + _TP12_W13_COLUMNS // 8)),
                )
            ),
        )
        w13_exceptions = cute.make_tensor(
            w13_exceptions_ptr, cute.make_layout((w13_exception_count,))
        )
        w13_offsets = cute.make_tensor(
            w13_offsets_ptr, cute.make_layout((num_experts + Int32(1),))
        )
        w13_task_offsets = cute.make_tensor(
            w13_task_offsets_ptr,
            cute.make_layout((num_experts * Int32(_TP12_W13_TASKS + 1),)),
        )
        w2_fixed = cute.make_tensor(
            w2_fixed_ptr,
            cute.make_layout(
                (
                    num_experts
                    * (_TP12_W2_ROWS // X4T_TILE_ROWS)
                    * (X4T_TILE_ROWS * (1 + _TP12_W2_COLUMNS // 8)),
                )
            ),
        )
        w2_exceptions = cute.make_tensor(
            w2_exceptions_ptr, cute.make_layout((w2_exception_count,))
        )
        w2_offsets = cute.make_tensor(
            w2_offsets_ptr, cute.make_layout((num_experts + Int32(1),))
        )
        w2_task_offsets = cute.make_tensor(
            w2_task_offsets_ptr,
            cute.make_layout((num_experts * Int32(_TP12_W2_TASKS + 1),)),
        )
        expert_ids = cute.make_tensor(
            expert_ids_ptr, cute.make_layout((active_capacity,))
        )
        expert_map = cute.make_tensor(
            expert_map_ptr, cute.make_layout((route_num_experts,))
        )
        w13_output = cute.make_tensor(
            w13_output_ptr,
            cute.make_layout(
                (num_experts * _TP12_W13_COLUMNS * _TP12_W13_ROWS,)
            ),
        )
        w2_output = cute.make_tensor(
            w2_output_ptr,
            cute.make_layout((num_experts * _TP12_W2_COLUMNS * _TP12_W2_ROWS,)),
        )
        self.kernel(
            w13_fixed,
            w13_exceptions,
            w13_offsets,
            w13_task_offsets,
            w2_fixed,
            w2_exceptions,
            w2_offsets,
            w2_task_offsets,
            expert_ids,
            expert_map,
            w13_output,
            w2_output,
            num_experts,
            active_capacity,
            route_num_experts,
        ).launch(
            grid=(active_capacity * Int32(_TP12_TASKS_PER_EXPERT), 1, 1),
            block=(_TP12_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _inverse_packed_row_in_64(self, packed: Int32) -> Int32:
        low = packed >> Int32(3)
        swapped_high = packed & Int32(7)
        high = (
            (swapped_high & Int32(4))
            | ((swapped_high & Int32(1)) << Int32(1))
            | ((swapped_high >> Int32(1)) & Int32(1))
        )
        return (high << Int32(3)) | low

    @cute.jit
    def _packed_row(self, row: Int32) -> Int32:
        block = row & ~Int32(63)
        low = row & Int32(7)
        high = (row >> Int32(3)) & Int32(7)
        swapped_high = (
            (high & Int32(4))
            | ((high & Int32(1)) << Int32(1))
            | ((high >> Int32(1)) & Int32(1))
        )
        return block | (low << Int32(3)) | swapped_high

    @cute.jit
    def _clamp_scale(self, value: Uint32) -> Uint32:
        result = value
        if value > Uint32(247):
            result = Uint32(247)
        return result

    @cute.jit
    def _resolve_expert(
        self,
        expert_ids: cute.Tensor,
        expert_map: cute.Tensor,
        slot: Int32,
        num_experts: Int32,
        route_num_experts: Int32,
    ) -> Int32:
        raw_expert = expert_ids[slot].to(Int32)
        expert = raw_expert
        if cutlass.const_expr(self.use_expert_map):
            expert = Int32(-1)
            if raw_expert >= Int32(0) and raw_expert < route_num_experts:
                expert = expert_map[raw_expert].to(Int32)
        if expert < Int32(0) or expert >= num_experts:
            expert = Int32(-1)
        return expert

    @cute.jit
    def _decode_w13(
        self,
        fixed: cute.Tensor,
        exceptions: cute.Tensor,
        offsets: cute.Tensor,
        task_offsets: cute.Tensor,
        output: cute.Tensor,
        expert: Int32,
        task: Int32,
        tid: Int32,
    ) -> None:
        tile_count = Int32(_TP12_W13_ROWS // X4T_TILE_ROWS)
        selector_bytes = Int32(_TP12_W13_COLUMNS // 8)
        tile_bytes = Int32(X4T_TILE_ROWS * (1 + _TP12_W13_COLUMNS // 8))
        packed_row_base = task * Int32(_TP12_W13_ROWS_PER_TASK)
        output_u32 = cute.recast_tensor(output, cutlass.Uint32)
        word = tid
        word_count = Int32(
            _TP12_W13_ROWS_PER_TASK * _TP12_W13_COLUMNS // 4
        )
        while word < word_count:
            packed_element = word << Int32(2)
            column = packed_element // Int32(_TP12_W13_ROWS_PER_TASK)
            packed_in_task = (
                packed_element - column * Int32(_TP12_W13_ROWS_PER_TASK)
            )
            packed_values = Uint32(0)
            for lane in cutlass.range_constexpr(4):
                packed_row = packed_in_task + Int32(lane)
                packed_in_64 = packed_row & Int32(63)
                rotated_row = (
                    packed_row_base
                    + (packed_row & ~Int32(63))
                    + self._inverse_packed_row_in_64(packed_in_64)
                )
                source_row = rotated_row
                if cutlass.const_expr(self.w13_row_rotation > 0):
                    source_row += Int32(self.w13_row_rotation)
                    if source_row >= Int32(_TP12_W13_ROWS):
                        source_row -= Int32(_TP12_W13_ROWS)
                tile = source_row >> Int32(4)
                local_row = source_row & Int32(15)
                tile_base = (expert * tile_count + tile) * tile_bytes
                base = fixed[tile_base + local_row].to(Uint32)
                selector = fixed[
                    tile_base
                    + Int32(X4T_TILE_ROWS)
                    + local_row * selector_bytes
                    + (column >> Int32(3))
                ].to(Uint32)
                value = base + (
                    (selector >> Uint32(column & Int32(7))) & Uint32(1)
                )
                packed_values |= self._clamp_scale(value) << Uint32(8 * lane)
            output_byte_offset = (
                (expert * Int32(_TP12_W13_COLUMNS) + column)
                * Int32(_TP12_W13_ROWS)
                + packed_row_base
                + packed_in_task
            )
            output_u32[output_byte_offset >> Int32(2)] = packed_values
            word += Int32(_TP12_THREADS)

        cute.arch.sync_threads()
        task_offset = expert * Int32(_TP12_W13_TASKS + 1) + task
        cursor = task_offsets[task_offset].to(Int64) + Int64(tid)
        end = task_offsets[task_offset + Int32(1)].to(Int64)
        while cursor < end:
            entry = exceptions[cursor].to(Uint32)
            position = entry & Uint32(X4T_POSITION_MASK)
            source_row = Int32(position // Uint32(_TP12_W13_COLUMNS))
            column = Int32(
                position - Uint32(source_row * Int32(_TP12_W13_COLUMNS))
            )
            rotated_row = source_row
            if cutlass.const_expr(self.w13_row_rotation > 0):
                rotated_row -= Int32(self.w13_row_rotation)
                if rotated_row < Int32(0):
                    rotated_row += Int32(_TP12_W13_ROWS)
            output[
                (expert * Int32(_TP12_W13_COLUMNS) + column)
                * Int32(_TP12_W13_ROWS)
                + self._packed_row(rotated_row)
            ] = Uint8(self._clamp_scale(entry >> Uint32(24)))
            cursor += Int64(_TP12_THREADS)

    @cute.jit
    def _decode_w2(
        self,
        fixed: cute.Tensor,
        exceptions: cute.Tensor,
        offsets: cute.Tensor,
        task_offsets: cute.Tensor,
        output: cute.Tensor,
        expert: Int32,
        task: Int32,
        tid: Int32,
    ) -> None:
        tile_count = Int32(_TP12_W2_ROWS // X4T_TILE_ROWS)
        tile_bytes = Int32(X4T_TILE_ROWS * (1 + _TP12_W2_COLUMNS // 8))
        packed_row_base = task * Int32(_TP12_W2_ROWS_PER_TASK)
        output_u32 = cute.recast_tensor(output, cutlass.Uint32)
        word = tid
        word_count = Int32(_TP12_W2_ROWS_PER_TASK * _TP12_W2_COLUMNS // 4)
        while word < word_count:
            packed_element = word << Int32(2)
            column = packed_element // Int32(_TP12_W2_ROWS_PER_TASK)
            packed_in_task = (
                packed_element - column * Int32(_TP12_W2_ROWS_PER_TASK)
            )
            packed_values = Uint32(0)
            for lane in cutlass.range_constexpr(4):
                packed_row = packed_in_task + Int32(lane)
                packed_in_64 = packed_row & Int32(63)
                source_row = (
                    packed_row_base
                    + (packed_row & ~Int32(63))
                    + self._inverse_packed_row_in_64(packed_in_64)
                )
                tile = source_row >> Int32(4)
                local_row = source_row & Int32(15)
                tile_base = (expert * tile_count + tile) * tile_bytes
                base = fixed[tile_base + local_row].to(Uint32)
                selector = fixed[
                    tile_base + Int32(X4T_TILE_ROWS) + local_row
                ].to(Uint32)
                value = base + (
                    (selector >> Uint32(column & Int32(7))) & Uint32(1)
                )
                packed_values |= self._clamp_scale(value) << Uint32(8 * lane)
            output_byte_offset = (
                (expert * Int32(_TP12_W2_COLUMNS) + column)
                * Int32(_TP12_W2_ROWS)
                + packed_row_base
                + packed_in_task
            )
            output_u32[output_byte_offset >> Int32(2)] = packed_values
            word += Int32(_TP12_THREADS)

        cute.arch.sync_threads()
        task_offset = expert * Int32(_TP12_W2_TASKS + 1) + task
        cursor = task_offsets[task_offset].to(Int64) + Int64(tid)
        end = task_offsets[task_offset + Int32(1)].to(Int64)
        while cursor < end:
            entry = exceptions[cursor].to(Uint32)
            position = entry & Uint32(X4T_POSITION_MASK)
            source_row = Int32(position >> Uint32(3))
            column = Int32(position & Uint32(7))
            output[
                (expert * Int32(_TP12_W2_COLUMNS) + column)
                * Int32(_TP12_W2_ROWS)
                + self._packed_row(source_row)
            ] = Uint8(self._clamp_scale(entry >> Uint32(24)))
            cursor += Int64(_TP12_THREADS)

    @cute.kernel
    def kernel(
        self,
        w13_fixed: cute.Tensor,
        w13_exceptions: cute.Tensor,
        w13_offsets: cute.Tensor,
        w13_task_offsets: cute.Tensor,
        w2_fixed: cute.Tensor,
        w2_exceptions: cute.Tensor,
        w2_offsets: cute.Tensor,
        w2_task_offsets: cute.Tensor,
        expert_ids: cute.Tensor,
        expert_map: cute.Tensor,
        w13_output: cute.Tensor,
        w2_output: cute.Tensor,
        num_experts: Int32,
        active_capacity: Int32,
        route_num_experts: Int32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        slot = Int32(block) // Int32(_TP12_TASKS_PER_EXPERT)
        local_task = Int32(block) - slot * Int32(_TP12_TASKS_PER_EXPERT)

        expert = Int32(-1)
        if cutlass.const_expr(self.expert_ids_unique):
            if slot < active_capacity:
                expert = self._resolve_expert(
                    expert_ids,
                    expert_map,
                    slot,
                    num_experts,
                    route_num_experts,
                )
        else:
            # A routed expert can own several packed M blocks (or appear in
            # several tokens' top-k lists). Resolve and deduplicate once per
            # CTA so repeated slots cannot concurrently write scale scratch.
            smem = cutlass.utils.SmemAllocator()

            @cute.struct
            class Storage:
                expert: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int32, 1], 4
                ]

            storage = smem.allocate(Storage)
            shared_expert = storage.expert.get_tensor(cute.make_layout(1))
            if tid == Int32(0):
                if slot < active_capacity:
                    expert = self._resolve_expert(
                        expert_ids,
                        expert_map,
                        slot,
                        num_experts,
                        route_num_experts,
                    )
                    prior = Int32(0)
                    while prior < slot and expert >= Int32(0):
                        prior_expert = self._resolve_expert(
                            expert_ids,
                            expert_map,
                            prior,
                            num_experts,
                            route_num_experts,
                        )
                        if prior_expert == expert:
                            expert = Int32(-1)
                        prior += Int32(1)
                shared_expert[Int32(0)] = expert
            cute.arch.sync_threads()
            expert = shared_expert[Int32(0)].to(Int32)

        if slot < active_capacity:
            if expert >= Int32(0) and expert < num_experts:
                if local_task < Int32(_TP12_W13_TASKS):
                    self._decode_w13(
                        w13_fixed,
                        w13_exceptions,
                        w13_offsets,
                        w13_task_offsets,
                        w13_output,
                        expert,
                        local_task,
                        Int32(tid),
                    )
                else:
                    self._decode_w2(
                        w2_fixed,
                        w2_exceptions,
                        w2_offsets,
                        w2_task_offsets,
                        w2_output,
                        expert,
                        local_task - Int32(_TP12_W13_TASKS),
                        Int32(tid),
                    )


@functools.cache
def _compiled_x4t_base(
    rows: int,
    columns: int,
    packed_w4a16: bool,
    row_rotation: int,
    clamp_e8m0_bf16: bool,
    threads: int,
) -> Callable:
    key = (
        int(rows),
        int(columns),
        bool(packed_w4a16),
        int(row_rotation),
        bool(clamp_e8m0_bf16),
        int(threads),
    )
    launch = _X4TBaseDecodeLaunch(
        rows,
        columns,
        packed_w4a16=packed_w4a16,
        row_rotation=row_rotation,
        clamp_e8m0_bf16=clamp_e8m0_bf16,
        threads=threads,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.x4t_base", 1, key),
    )


@functools.cache
def _compiled_x4t_scatter(
    rows: int,
    columns: int,
    packed_w4a16: bool,
    row_rotation: int,
    clamp_e8m0_bf16: bool,
    threads: int,
) -> Callable:
    key = (
        int(rows),
        int(columns),
        bool(packed_w4a16),
        int(row_rotation),
        bool(clamp_e8m0_bf16),
        int(threads),
    )
    launch = _X4TExceptionScatterLaunch(
        rows,
        columns,
        packed_w4a16=packed_w4a16,
        row_rotation=row_rotation,
        clamp_e8m0_bf16=clamp_e8m0_bf16,
        threads=threads,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.x4t_scatter", 1, key),
    )


@functools.cache
def _compiled_x4t_tp12_w4a16(
    w13_row_rotation: int,
    use_expert_map: bool,
    expert_ids_unique: bool,
) -> Callable:
    launch = _X4TTP12W4A16DecodeLaunch(
        w13_row_rotation=w13_row_rotation,
        use_expert_map=use_expert_map,
        expert_ids_unique=expert_ids_unique,
    )
    key = (
        _TP12_W13_ROWS,
        _TP12_W13_COLUMNS,
        _TP12_W2_ROWS,
        _TP12_W2_COLUMNS,
        _TP12_THREADS,
        int(w13_row_rotation),
        bool(use_expert_map),
        bool(expert_ids_unique),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.x4t_tp12_w4a16", 1, key),
    )


def decode_x4t_tp12_w4a16_scales(
    w13: X4TScaleBatch,
    w2: X4TScaleBatch,
    expert_ids: torch.Tensor,
    w13_output: torch.Tensor,
    w2_output: torch.Tensor,
    *,
    expert_map: torch.Tensor | None = None,
    w13_row_rotation: int = _TP12_W13_ROW_ROTATION,
    expert_ids_unique: bool = False,
    stream: cuda.CUstream | None = None,
) -> None:
    """Decode both Kimi-K3 TP12 X4T planes into native W4A16 layouts.

    This is the serving path: one graph-safe launch, no temporary allocation,
    exact sparse overrides, FC1 half rotation folded into ``w13_output``, and
    the BF16 W4A16 E8M0 maximum-byte clamp folded into both outputs.
    """

    w13.validate()
    w2.validate()
    if (w13.rows, w13.columns) != (_TP12_W13_ROWS, _TP12_W13_COLUMNS):
        raise ValueError(
            "TP12 X4T w13 scales must have logical shape [E,512,112]"
        )
    if (w2.rows, w2.columns) != (_TP12_W2_ROWS, _TP12_W2_COLUMNS):
        raise ValueError("TP12 X4T w2 scales must have logical shape [E,3584,8]")
    if w13.num_experts != w2.num_experts:
        raise ValueError("TP12 X4T w13 and w2 expert counts must match")
    w13_row_rotation = int(w13_row_rotation)
    if w13_row_rotation not in (0, _TP12_W13_ROW_ROTATION):
        raise ValueError("TP12 X4T w13 row rotation must be 0 or 256")
    if (
        w13.task_exception_offsets is None
        or w13.exception_task_rows != _TP12_W13_ROWS_PER_TASK
        or w13.exception_row_rotation != w13_row_rotation
    ):
        raise ValueError(
            "TP12 X4T w13 requires exception ranges for the selected row rotation"
        )
    if (
        w2.task_exception_offsets is None
        or w2.exception_task_rows != _TP12_W2_ROWS_PER_TASK
        or w2.exception_row_rotation != 0
    ):
        raise ValueError("TP12 X4T w2 requires 512-row exception ranges")
    device = w13.fixed.device
    if w2.fixed.device != device:
        raise ValueError("TP12 X4T scale planes must share one CUDA device")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1:
        raise TypeError("TP12 X4T expert_ids must be a one-dimensional int32 tensor")
    if not expert_ids.is_contiguous() or expert_ids.device != device:
        raise ValueError("TP12 X4T expert_ids must be contiguous on the batch device")
    if expert_map is not None:
        if expert_map.dtype != torch.int32 or expert_map.ndim != 1:
            raise TypeError("TP12 X4T expert_map must be a one-dimensional int32 tensor")
        if not expert_map.is_contiguous() or expert_map.device != device:
            raise ValueError("TP12 X4T expert_map must be contiguous on the batch device")
    expected_w13 = (w13.num_experts, _TP12_W13_COLUMNS, _TP12_W13_ROWS)
    expected_w2 = (w13.num_experts, _TP12_W2_COLUMNS, _TP12_W2_ROWS)
    for name, tensor, expected in (
        ("w13_output", w13_output, expected_w13),
        ("w2_output", w2_output, expected_w2),
    ):
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected:
            raise ValueError(f"TP12 X4T {name} must be uint8 with shape {expected}")
        if not tensor.is_contiguous() or tensor.device != device:
            raise ValueError(f"TP12 X4T {name} must be contiguous on the batch device")

    use_expert_map = expert_map is not None
    map_tensor = expert_ids if expert_map is None else expert_map
    # The pointer is compile-time dead when no map is selected, but retain a
    # one-element tensor extent so the runtime layout never has a zero mode.
    route_num_experts = 1 if expert_map is None else int(expert_map.numel())
    compiled = _compiled_x4t_tp12_w4a16(
        w13_row_rotation,
        use_expert_map,
        bool(expert_ids_unique),
    )
    stream = current_cuda_stream() if stream is None else stream
    compiled(
        make_ptr(cutlass.Uint8, w13.fixed.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, w13.exceptions.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, w13.exception_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int64, w13.task_exception_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Uint8, w2.fixed.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, w2.exceptions.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, w2.exception_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int64, w2.task_exception_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int32, expert_ids.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, map_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, w13_output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, w2_output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        w13.num_experts,
        int(w13.exceptions.numel()),
        int(w2.exceptions.numel()),
        int(expert_ids.numel()),
        route_num_experts,
        stream,
    )


def decode_x4t_scales(
    batch: X4TScaleBatch,
    expert_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    layout: str = "logical",
    row_rotation: int = 0,
    clamp_e8m0_bf16: bool = False,
) -> None:
    """Expand selected experts into a preallocated expert-indexed tensor.

    ``expert_ids`` is a fixed-capacity int32 vector.  Negative entries are
    inactive, which makes the call CUDA-graph safe without a host-visible
    active count.  ``layout='w4a16_packed'`` writes ``[E,K/32,N]`` in the exact
    byte permutation consumed by the native W4A16 kernels; ``logical`` writes
    ``[E,N,K/32]``.
    """

    batch.validate()
    if layout not in {"logical", "w4a16_packed"}:
        raise ValueError("X4T output layout must be 'logical' or 'w4a16_packed'")
    packed = layout == "w4a16_packed"
    row_rotation = int(row_rotation)
    if row_rotation < 0 or row_rotation >= int(batch.rows):
        if row_rotation != 0:
            raise ValueError("X4T row_rotation must be zero or lie within the row axis")
    if packed and int(batch.rows) % 64:
        raise ValueError("W4A16 packed X4T output requires rows divisible by 64")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1:
        raise TypeError("X4T expert_ids must be a one-dimensional int32 tensor")
    if not expert_ids.is_contiguous() or expert_ids.device != batch.fixed.device:
        raise ValueError("X4T expert_ids must be contiguous on the batch device")
    expected = (
        (batch.num_experts, batch.columns, batch.rows)
        if packed
        else (batch.num_experts, batch.rows, batch.columns)
    )
    if output.dtype != torch.uint8 or tuple(output.shape) != expected:
        raise ValueError(f"X4T output must be uint8 with shape {expected}")
    if not output.is_contiguous() or output.device != batch.fixed.device:
        raise ValueError("X4T output must be contiguous on the batch device")

    threads = 128 if int(batch.columns) <= 8 else 256
    active_capacity = int(expert_ids.numel())
    total_tiles = active_capacity * batch.tile_count
    sm_count = torch.cuda.get_device_properties(batch.fixed.device).multi_processor_count
    base_grid = max(1, min(total_tiles, sm_count * _GRID_CTAS_PER_SM))
    base = _compiled_x4t_base(
        batch.rows,
        batch.columns,
        packed,
        row_rotation,
        bool(clamp_e8m0_bf16),
        threads,
    )
    base(
        make_ptr(cutlass.Uint8, batch.fixed.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, expert_ids.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        batch.num_experts,
        active_capacity,
        base_grid,
        current_cuda_stream(),
    )

    scatter_grid = max(1, min(active_capacity, sm_count * _GRID_CTAS_PER_SM))
    scatter = _compiled_x4t_scatter(
        batch.rows,
        batch.columns,
        packed,
        row_rotation,
        bool(clamp_e8m0_bf16),
        threads,
    )
    scatter(
        make_ptr(cutlass.Uint32, batch.exceptions.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int64, batch.exception_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Int32, expert_ids.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        batch.num_experts,
        int(batch.exceptions.numel()),
        active_capacity,
        scatter_grid,
        current_cuda_stream(),
    )
