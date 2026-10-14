"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# NVFP4-transport tensor-parallel all-reduce for ``[rows, 6144]`` bfloat16
# activations.  Every rank quantises its rows to NVFP4 records (E2M1 values,
# E4M3 block scales and one float32 row scale), pushes them into Lamport
# receive windows of its peers and decodes the gathered records in rank order,
# so the result is the rank-order sum of the quantised partials and bitwise
# identical on every rank.  World size 2 runs a one-shot exchange, world size
# 4 a two-shot exchange (owner ``row % 4`` reduces, re-encodes and all-gathers;
# three phase-B variants are chosen per call from the rows each CTA owns).
# The quantisation follows the codec of the NVFP4 all-reduce reference by
# Fan Shi (https://github.com/fans-nv/fp4-all-reduce, commit 2878bf5).

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from ..jit.fp4_allreduce import (
    CAPACITY,
    HIDDEN,
    RECORD_BYTES,
    WORLD_SIZES,
    get_fp4_allreduce_module,
    kernel_name,
    phase_b_variant,
    program_table,
    target_arch,
)
from .cuda_ipc import create_shared_buffer, free_shared_buffer
from .dlpack_utils import pack_strided_memory

PAGE_BYTES = 4096
VALUE_BYTES = 3072  # 6144 E2M1 values of one row
SCALE_BYTES = 384  # 384 E4M3 block scales of one row
ROW_SCALE_BYTES = 16  # one float32 row scale, padded to 16 bytes in a receive slot
SLOT_ROW_BYTES = VALUE_BYTES + SCALE_BYTES + ROW_SCALE_BYTES  # 3472
ROTATION = 3  # receive buffers per window: three-slot rotation with in-kernel reset
SENTINEL_VALUE_WORD = 0x88888888
SENTINEL_SCALE_WORD = 0xFFFFFFFF
SENTINEL_ROW_SCALE_WORD = 0x80000000
FAMILIES = ("allreduce",)  # delivered program families of the workspace layout


def _page_align(size: int) -> int:
    return (size + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES


@dataclass(frozen=True)
class FillPlan:
    """A 32-bit sentinel fill of ``words`` words at word offset ``offset`` of a region.

    ``period == 0`` writes ``value_low`` to every word; otherwise word ``i`` gets
    ``value_low`` when ``i % period < split`` and ``value_high`` elsewhere.
    """

    offset: int
    words: int
    value_low: int
    value_high: int = 0
    period: int = 0
    split: int = 0


@dataclass(frozen=True)
class RegionPlan:
    """One shared receive window of a rank: its byte size and sentinel fills."""

    name: str
    bytes: int
    fills: tuple[FillPlan, ...]


@dataclass(frozen=True)
class WindowPlan:
    """Receive-window layout of one rank (see :func:`window_plan`)."""

    world_size: int
    capacity: int
    ctrl_words: int
    row_multiple: int
    regions: tuple[RegionPlan, ...]

    @property
    def table_entries(self) -> int:
        return len(self.regions) * self.world_size

    @property
    def total_bytes(self) -> int:
        return sum(region.bytes for region in self.regions)


def _slot_array_fills(offset: int, rows: int) -> tuple[FillPlan, ...]:
    """Value, scale and row-scale planes of one slot array holding ``rows`` slot rows."""
    values = rows * VALUE_BYTES // 4
    scales = rows * SCALE_BYTES // 4
    row_scales = rows * ROW_SCALE_BYTES // 4
    return (
        FillPlan(offset, values, SENTINEL_VALUE_WORD),
        FillPlan(offset + values, scales, SENTINEL_SCALE_WORD),
        FillPlan(offset + values + scales, row_scales, SENTINEL_ROW_SCALE_WORD),
    )


def window_plan(
    world_size: int, capacity: int = CAPACITY, family: str = "allreduce"
) -> WindowPlan:
    """Receive-window layout and sentinel fills of one rank.

    World size 2 (one-shot) owns two windows: the one-shot window (three rotation
    buffers of ``2 * capacity`` slot rows of 3472 bytes laid out as a value plane,
    a scale plane and a row-scale plane) and the NVFP4-input window (three buffers
    of ``2 * capacity`` records of 3456 bytes, armed with the value sentinel over
    the value bytes and the scale sentinel over the scale bytes of every record).
    World size 4 (two-shot) owns one window of three buffers, each a
    reduce-scatter and an all-gather slot array of ``4 * ceil(capacity / 4)``
    slot rows.  Every buffer is page aligned.
    """
    if family not in FAMILIES:
        raise ValueError(
            f"unknown fp4_allreduce family {family!r}; delivered families: {FAMILIES}"
        )
    if world_size not in WORLD_SIZES:
        raise ValueError(
            f"fp4_allreduce supports world sizes {WORLD_SIZES}, got {world_size}"
        )
    if capacity != CAPACITY:
        raise ValueError(
            "the delivered fp4_allreduce programs are specialised for a capacity "
            f"of {CAPACITY} rows, got {capacity}"
        )
    table = program_table(world_size)
    if world_size == 2:
        slot_rows = world_size * capacity
        oneshot_buffer = _page_align(slot_rows * SLOT_ROW_BYTES)
        record_buffer = _page_align(slot_rows * RECORD_BYTES)
        oneshot_fills = tuple(
            fill
            for buffer in range(ROTATION)
            for fill in _slot_array_fills(buffer * oneshot_buffer // 4, slot_rows)
        )
        record_fills = tuple(
            FillPlan(
                buffer * record_buffer // 4,
                slot_rows * RECORD_BYTES // 4,
                SENTINEL_VALUE_WORD,
                SENTINEL_SCALE_WORD,
                RECORD_BYTES // 4,
                VALUE_BYTES // 4,
            )
            for buffer in range(ROTATION)
        )
        regions = (
            RegionPlan("oneshot", ROTATION * oneshot_buffer, oneshot_fills),
            RegionPlan("nvfp4", ROTATION * record_buffer, record_fills),
        )
    else:
        owner_rows = -(-capacity // world_size)
        slot_rows = world_size * owner_rows
        slot_array = _page_align(slot_rows * SLOT_ROW_BYTES)
        buffer_bytes = 2 * slot_array
        fills = tuple(
            fill
            for buffer in range(ROTATION)
            for array in range(2)
            for fill in _slot_array_fills(
                (buffer * buffer_bytes + array * slot_array) // 4, slot_rows
            )
        )
        regions = (RegionPlan("twoshot", ROTATION * buffer_bytes, fills),)
    plan = WindowPlan(
        world_size,
        capacity,
        int(table["ctrl_words"]),
        int(table["row_multiple"]),
        regions,
    )
    if plan.table_entries != int(table["table_entries"]):
        raise RuntimeError(
            f"the window plan of world size {world_size} yields {plan.table_entries} "
            f"table entries, the binding expects {table['table_entries']}"
        )
    return plan


def _as_int32(word: int) -> int:
    return word - (1 << 32) if word >= (1 << 31) else word


def _arm(words: torch.Tensor, fills: tuple[FillPlan, ...]) -> None:
    """Write the sentinel fills into ``words`` (an int32 view of one region)."""
    patterns: dict[tuple[int, int, int, int, int], torch.Tensor] = {}
    for fill in fills:
        view = words[fill.offset : fill.offset + fill.words]
        if fill.period == 0:
            view.fill_(_as_int32(fill.value_low))
            continue
        key = (fill.words, fill.period, fill.split, fill.value_low, fill.value_high)
        pattern = patterns.get(key)
        if pattern is None:
            positions = torch.arange(fill.words, device=words.device) % fill.period
            pattern = torch.where(
                positions < fill.split,
                torch.full_like(positions, _as_int32(fill.value_low)),
                torch.full_like(positions, _as_int32(fill.value_high)),
            ).to(torch.int32)
            patterns[key] = pattern
        view.copy_(pattern)


class Fp4AllReduceWorkspace:
    """Per-rank state of the NVFP4-transport all-reduce, created collectively.

    Every rank of ``group`` (world size 2 or 4) constructs one instance with the
    same ``capacity``: the receive windows are shared through CUDA IPC, armed
    with their Lamport sentinels and published in the peer window table; the
    call-state words start at zero.  The CTA count is the SM count of the device
    and must agree on every rank.  ``destroy()`` releases the shared windows and
    is collective as well.  Construction loads (and on first use compiles) the
    JIT module of the device's architecture.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        *,
        capacity: int = CAPACITY,
        device: Optional[torch.device] = None,
        family: str = "allreduce",
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "fp4_allreduce requires an initialized torch.distributed process group"
            )
        self.group = dist.group.WORLD if group is None else group
        self.world_size = int(dist.get_world_size(self.group))
        self.rank = int(dist.get_rank(self.group))
        if self.world_size not in WORLD_SIZES:
            raise ValueError(
                f"fp4_allreduce supports world sizes {WORLD_SIZES}, got a group of {self.world_size}"
            )
        device = torch.device(torch.cuda.current_device() if device is None else device)
        if device.type != "cuda":
            raise ValueError(f"fp4_allreduce needs a CUDA device, got {device}")
        index = torch.cuda.current_device() if device.index is None else device.index
        self.device = torch.device("cuda", index)
        self.family = family
        self.plan = window_plan(self.world_size, capacity, family)
        self.capacity = int(capacity)
        self.arch = target_arch(index)
        self.module = get_fp4_allreduce_module(self.world_size, self.arch)
        grid = int(torch.cuda.get_device_properties(index).multi_processor_count)
        gathered = [
            torch.zeros(1, dtype=torch.int64, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            gathered,
            torch.tensor([grid], dtype=torch.int64, device=self.device),
            group=self.group,
        )
        grids = [int(value.item()) for value in gathered]
        if any(value != grid for value in grids):
            raise RuntimeError(
                "fp4_allreduce needs the same CTA count (SM count) on every rank, "
                f"got {grids}"
            )
        self.grid = grid
        self._shared: list[list[int]] = []
        pointers: list[int] = []
        for region in self.plan.regions:
            shared = create_shared_buffer(region.bytes, group=self.group)
            self._shared.append(shared)
            words = pack_strided_memory(
                shared[self.rank], region.bytes, region.bytes, 1, torch.int32, index
            ).reshape(-1)
            _arm(words, region.fills)
            pointers.extend(int(pointer) for pointer in shared)
        self.ctrl = torch.zeros(
            self.plan.ctrl_words, dtype=torch.int32, device=self.device
        )
        self.table = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        # No rank may push before every rank has finished arming its windows.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    @property
    def closed(self) -> bool:
        return not self._shared

    def destroy(self) -> None:
        """Release the shared receive windows (collective; a second call is a no-op)."""
        shared, self._shared = self._shared, []
        if not shared:
            return
        torch.cuda.synchronize(self.device)
        for pointers in shared:
            free_shared_buffer(pointers, group=self.group)

    def __enter__(self) -> "Fp4AllReduceWorkspace":
        return self

    def __exit__(self, *exc_info) -> None:
        self.destroy()

    def variant(self, rows: int) -> int:
        """The phase-B variant a call with ``rows`` rows runs on this workspace's grid."""
        return phase_b_variant(int(rows), self.grid, self.world_size)

    def kernel_name(self, input_format: str, rows: int) -> str:
        """Kernel symbol this rank launches for ``rows`` rows of ``input_format`` (``"bf16"`` or ``"nvfp4"``)."""
        return kernel_name(self.world_size, input_format, self.rank, self.variant(rows))


def _check_workspace(workspace: Fp4AllReduceWorkspace) -> None:
    if not isinstance(workspace, Fp4AllReduceWorkspace):
        raise TypeError("workspace must be a Fp4AllReduceWorkspace")
    if workspace.closed:
        raise RuntimeError("the fp4_allreduce workspace has been destroyed")


def _check_rows(rows: int, workspace: Fp4AllReduceWorkspace) -> None:
    if rows <= 0 or rows > workspace.capacity:
        raise ValueError(f"rows must be in (0, {workspace.capacity}], got {rows}")
    if rows % workspace.plan.row_multiple:
        raise ValueError(
            f"world size {workspace.world_size} needs rows to be a multiple of "
            f"{workspace.plan.row_multiple}, got {rows}"
        )


def _check_activation(
    x: torch.Tensor, workspace: Fp4AllReduceWorkspace, name: str
) -> None:
    if (
        x.dim() != 2
        or x.shape[1] != HIDDEN
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be a contiguous bfloat16 [rows, {HIDDEN}] tensor, "
            f"got {tuple(x.shape)} {x.dtype}"
        )
    if x.device != workspace.device:
        raise ValueError(f"{name} must live on {workspace.device}, got {x.device}")


def _output(
    out: Optional[torch.Tensor], rows: int, workspace: Fp4AllReduceWorkspace
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (rows, HIDDEN), dtype=torch.bfloat16, device=workspace.device
        )
    _check_activation(out, workspace, "out")
    if out.shape[0] != rows:
        raise ValueError(f"out must have {rows} rows, got {out.shape[0]}")
    return out


def fp4_allreduce(
    x: torch.Tensor,
    workspace: Fp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """Sum ``x`` over the workspace's group through the NVFP4 transport.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` partial of this rank; ``rows`` is at
        most the workspace capacity, a multiple of 4 for world size 4, and the
        same on every rank of the call.
    workspace : Fp4AllReduceWorkspace
        The collective workspace of the group.
    out : torch.Tensor, optional
        Contiguous bfloat16 ``[rows, 6144]`` output; allocated when omitted.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute (the
        kernel waits for its predecessor before touching the windows).

    Returns
    -------
    torch.Tensor
        ``out``: the rank-order sum of the NVFP4-quantised partials, bitwise
        identical on every rank.  The call allocates nothing when ``out`` is
        given and performs no host synchronisation, so it can be captured into
        a CUDA graph.
    """
    _check_workspace(workspace)
    _check_activation(x, workspace, "x")
    rows = int(x.shape[0])
    _check_rows(rows, workspace)
    out = _output(out, rows, workspace)
    workspace.module.fp4_allreduce(
        x,
        None,
        out,
        workspace.ctrl,
        workspace.table,
        workspace.rank,
        0,
        workspace.grid,
        bool(launch_with_pdl),
    )
    return out


def fp4_allreduce_nvfp4(
    records: torch.Tensor,
    global_scales: torch.Tensor,
    workspace: Fp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """Sum pre-quantised NVFP4 records over the workspace's group.

    Parameters
    ----------
    records : torch.Tensor
        Contiguous uint8 ``[rows, 3456]`` records of this rank: 3072 packed E2M1
        value bytes (element ``2i`` in the low nibble) followed by 384 E4M3 block
        scales, encoded with the static per-tensor scale ``global_scales[rank]``
        (``1 / global_scale`` is the row scale of every row).
    global_scales : torch.Tensor
        Contiguous float32 ``[world_size]`` tensor holding the static scale of
        every rank's records; identical on every rank.
    workspace : Fp4AllReduceWorkspace
        The collective workspace of the group.
    out : torch.Tensor, optional
        Contiguous bfloat16 ``[rows, 6144]`` output; allocated when omitted.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute.

    Returns
    -------
    torch.Tensor
        ``out``: the rank-order sum of the decoded records (``record_r`` decoded
        with ``global_scales[r]``), bitwise identical on every rank.
    """
    _check_workspace(workspace)
    if (
        records.dim() != 2
        or records.shape[1] != RECORD_BYTES
        or records.dtype != torch.uint8
        or not records.is_contiguous()
    ):
        raise ValueError(
            f"records must be a contiguous uint8 [rows, {RECORD_BYTES}] tensor, "
            f"got {tuple(records.shape)} {records.dtype}"
        )
    if records.device != workspace.device:
        raise ValueError(
            f"records must live on {workspace.device}, got {records.device}"
        )
    if (
        global_scales.dim() != 1
        or global_scales.numel() < workspace.world_size
        or global_scales.dtype != torch.float32
        or not global_scales.is_contiguous()
        or global_scales.device != workspace.device
    ):
        raise ValueError(
            f"global_scales must be a contiguous float32 [{workspace.world_size}] tensor on "
            f"{workspace.device}"
        )
    rows = int(records.shape[0])
    _check_rows(rows, workspace)
    out = _output(out, rows, workspace)
    workspace.module.fp4_allreduce(
        records,
        global_scales,
        out,
        workspace.ctrl,
        workspace.table,
        workspace.rank,
        1,
        workspace.grid,
        bool(launch_with_pdl),
    )
    return out


class Fp4AllReduceFusedNormOutput(enum.Enum):
    """Output precision of the fused all-reduce + residual + RMSNorm epilogue."""

    BF16 = "bf16"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


@dataclass(frozen=True)
class Fp4AllReduceFusedNormSpec:
    """Request of the fused epilogue: residual add, then RMSNorm scaled by ``1 + weight``.

    ``weight`` is a ``[6144]`` tensor, ``eps`` the RMSNorm epsilon and ``output``
    the precision of the normalised result (NVFP4 and MXFP8 outputs carry their
    block scales next to the packed values).
    """

    weight: torch.Tensor
    eps: float
    output: Fp4AllReduceFusedNormOutput = Fp4AllReduceFusedNormOutput.BF16


def delivered_families() -> tuple[str, ...]:
    """Program families the installed sources deliver (the fused epilogue is not among them yet)."""
    return FAMILIES


def fp4_allreduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    spec: Fp4AllReduceFusedNormSpec,
    workspace: Fp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """All-reduce ``x`` through the NVFP4 transport, add ``residual`` and apply the RMSNorm of ``spec``.

    The fused programs are not part of the delivered sources: the call raises
    ``NotImplementedError`` naming the delivered families.  The signature is the
    extension point of that delivery (``residual_out`` receives the summed
    residual; ``out`` the normalised result in the precision of ``spec.output``).
    """
    _check_workspace(workspace)
    raise NotImplementedError(
        f"fp4_allreduce_rmsnorm with a {spec.output.value} output is not delivered by these "
        f"sources; delivered families: {delivered_families()}"
    )


__all__ = [
    "CAPACITY",
    "FAMILIES",
    "FillPlan",
    "Fp4AllReduceFusedNormOutput",
    "Fp4AllReduceFusedNormSpec",
    "Fp4AllReduceWorkspace",
    "RegionPlan",
    "WindowPlan",
    "delivered_families",
    "fp4_allreduce",
    "fp4_allreduce_nvfp4",
    "fp4_allreduce_rmsnorm",
    "window_plan",
]
