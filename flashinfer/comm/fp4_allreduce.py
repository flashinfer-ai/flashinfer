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
# The ``rmsnorm`` family fuses the one-shot exchange (world size 2) with the
# residual add and the Gemma RMSNorm: ``z = sum + residual`` is kept in float32,
# ``residual_out = bf16(z)``, ``y = z * rsqrt(mean(z^2) + eps) * (1 + weight)``
# is written as bfloat16 or quantised to NVFP4 / MXFP8 in the layouts of
# ``fp4_quantize`` / ``mxfp8_quantize``.  The quantisation follows the codec of
# the NVFP4 all-reduce reference by Fan Shi
# (https://github.com/fans-nv/fp4-all-reduce, commit 2878bf5).

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from ..jit.fp4_allreduce import (
    CAPACITY,
    FAMILIES,
    HIDDEN,
    RECORD_BYTES,
    RMSNORM_EPS,
    get_module,
    kernel_name,
    phase_b_variant,
    program_table,
    target_arch,
    world_sizes,
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
SCALE_ROW_TILE = (
    128  # swizzled output scale tensors cover rows padded to a multiple of 128
)
NVFP4_OUTPUT_VALUE_BYTES = 3072  # 6144 E2M1 values per row
NVFP4_OUTPUT_SCALES_PER_ROW = 384  # one E4M3 scale per 16 values
MXFP8_OUTPUT_VALUE_BYTES = 6144  # one E4M3 byte per value
MXFP8_OUTPUT_SCALES_PER_ROW = 192  # one UE8M0 scale per 32 values


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

    family: str
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


def _oneshot_region(name: str, world_size: int, capacity: int) -> RegionPlan:
    """Three rotation buffers of ``world_size * capacity`` planar slot rows (value / scale / row-scale planes)."""
    slot_rows = world_size * capacity
    buffer = _page_align(slot_rows * SLOT_ROW_BYTES)
    fills = tuple(
        fill
        for index in range(ROTATION)
        for fill in _slot_array_fills(index * buffer // 4, slot_rows)
    )
    return RegionPlan(name, ROTATION * buffer, fills)


def window_plan(
    world_size: int, capacity: int = CAPACITY, family: str = "allreduce"
) -> WindowPlan:
    """Receive-window layout and sentinel fills of one rank.

    Family ``allreduce``, world size 2 (one-shot) owns two windows: the one-shot
    window (three rotation buffers of ``2 * capacity`` slot rows of 3472 bytes
    laid out as a value plane, a scale plane and a row-scale plane) and the
    NVFP4-input window (three buffers of ``2 * capacity`` records of 3456 bytes,
    armed with the value sentinel over the value bytes and the scale sentinel
    over the scale bytes of every record).  World size 4 (two-shot) owns one
    window of three buffers, each a reduce-scatter and an all-gather slot array
    of ``4 * ceil(capacity / 4)`` slot rows.  Family ``rmsnorm`` (world size 2)
    owns one one-shot window of the same planar layout; its NVFP4 input travels
    on that window with the static row scale in the row-scale plane, so no
    record window exists.  Every buffer is page aligned.
    """
    if family not in FAMILIES:
        raise ValueError(
            f"unknown fp4_allreduce family {family!r}; delivered families: {FAMILIES}"
        )
    if world_size not in world_sizes(family):
        raise ValueError(
            f"fp4_allreduce family {family!r} supports world sizes {world_sizes(family)}, "
            f"got {world_size}"
        )
    if capacity != CAPACITY:
        raise ValueError(
            "the delivered fp4_allreduce programs are specialised for a capacity "
            f"of {CAPACITY} rows, got {capacity}"
        )
    table = program_table(world_size, family)
    regions: tuple[RegionPlan, ...]
    if family == "rmsnorm":
        regions = (_oneshot_region("rmsnorm", world_size, capacity),)
    elif world_size == 2:
        slot_rows = world_size * capacity
        record_buffer = _page_align(slot_rows * RECORD_BYTES)
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
            _oneshot_region("oneshot", world_size, capacity),
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
        family,
        world_size,
        capacity,
        int(table["ctrl_words"]),
        int(table["row_multiple"]),
        regions,
    )
    if plan.table_entries != int(table["table_entries"]):
        raise RuntimeError(
            f"the window plan of family {family!r} world size {world_size} yields "
            f"{plan.table_entries} table entries, the binding expects {table['table_entries']}"
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

    Every rank of ``group`` constructs one instance with the same ``capacity``
    and ``family`` (``"allreduce"``: world size 2 or 4, the bfloat16 all-reduce;
    ``"rmsnorm"``: world size 2, the all-reduce fused with the residual add and
    the RMSNorm).  The receive windows are shared through CUDA IPC, armed with
    their Lamport sentinels and published in the peer window table; the
    call-state words start at zero.  Each family keeps its own windows and
    call state, so one workspace serves the calls of its family only.  The CTA
    count is the SM count of the device and must agree on every rank.
    ``destroy()`` releases the shared windows and is collective as well.
    Construction loads (and on first use compiles) the JIT module of the
    device's architecture.
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
        if family not in FAMILIES:
            raise ValueError(
                f"unknown fp4_allreduce family {family!r}; delivered families: {FAMILIES}"
            )
        self.group = dist.group.WORLD if group is None else group
        self.world_size = int(dist.get_world_size(self.group))
        self.rank = int(dist.get_rank(self.group))
        if self.world_size not in world_sizes(family):
            raise ValueError(
                f"fp4_allreduce family {family!r} supports world sizes {world_sizes(family)}, "
                f"got a group of {self.world_size}"
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
        self.module = get_module(self.world_size, self.arch, family)
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

    def variant(self, rows: int, output_format: str = "bf16") -> int:
        """The program variant a call with ``rows`` rows (and, for the fused-norm family, ``output_format``) runs on this workspace's grid."""
        return phase_b_variant(
            int(rows), self.grid, self.world_size, self.family, output_format
        )

    def kernel_name(
        self, input_format: str, rows: int, output_format: str = "bf16"
    ) -> str:
        """Kernel symbol this rank launches for ``rows`` rows of ``input_format`` (and, for the fused-norm family, ``output_format``)."""
        return kernel_name(
            self.world_size,
            input_format,
            self.rank,
            self.variant(rows, output_format),
            self.family,
            output_format,
        )


def _check_workspace(workspace: Fp4AllReduceWorkspace, family: str) -> None:
    if not isinstance(workspace, Fp4AllReduceWorkspace):
        raise TypeError("workspace must be a Fp4AllReduceWorkspace")
    if workspace.closed:
        raise RuntimeError("the fp4_allreduce workspace has been destroyed")
    if workspace.family != family:
        raise ValueError(
            f"this call needs a workspace of family {family!r}, got one of family "
            f"{workspace.family!r}"
        )


def _check_rows(rows: int, workspace: Fp4AllReduceWorkspace) -> None:
    if rows <= 0 or rows > workspace.capacity:
        raise ValueError(f"rows must be in (0, {workspace.capacity}], got {rows}")
    if rows % workspace.plan.row_multiple:
        raise ValueError(
            f"family {workspace.family!r} with world size {workspace.world_size} needs rows "
            f"to be a multiple of {workspace.plan.row_multiple}, got {rows}"
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


def _check_records(
    records: torch.Tensor,
    global_scales: torch.Tensor,
    workspace: Fp4AllReduceWorkspace,
) -> None:
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


def _output(
    out: Optional[torch.Tensor],
    rows: int,
    workspace: Fp4AllReduceWorkspace,
    name: str = "out",
) -> torch.Tensor:
    if out is None:
        return torch.empty(
            (rows, HIDDEN), dtype=torch.bfloat16, device=workspace.device
        )
    _check_activation(out, workspace, name)
    if out.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {out.shape[0]}")
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
        The collective workspace of the group (family ``"allreduce"``).
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
    _check_workspace(workspace, "allreduce")
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
        The collective workspace of the group (family ``"allreduce"``).
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
    _check_workspace(workspace, "allreduce")
    _check_records(records, global_scales, workspace)
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


_OUTPUT_CODES = {
    Fp4AllReduceFusedNormOutput.BF16: 0,
    Fp4AllReduceFusedNormOutput.NVFP4: 1,
    Fp4AllReduceFusedNormOutput.MXFP8: 2,
}


@dataclass(frozen=True)
class Fp4AllReduceFusedNormSpec:
    """Request of the fused epilogue: residual add, then RMSNorm scaled by ``1 + weight``.

    ``weight`` is a contiguous bfloat16 ``[6144]`` tensor on the workspace device,
    ``eps`` the RMSNorm epsilon (the delivered programs are compiled for
    ``1e-6``; another value is rejected) and ``output`` the precision of the
    normalised result.  ``global_scale`` is required for the NVFP4 output: a
    float32 ``[1]`` tensor on the device holding the quantiser's input global
    scale (``448 * 6 / amax(y)`` in the convention of ``fp4_quantize``); it is read
    by the kernel, never by the host, so the call stays graph capturable.
    """

    weight: torch.Tensor
    eps: float = RMSNORM_EPS
    output: Fp4AllReduceFusedNormOutput = Fp4AllReduceFusedNormOutput.BF16
    global_scale: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Fp4AllReduceFusedNormResult:
    """Outputs of one fused call.

    ``residual_out`` is ``bf16(sum + residual)`` (``[rows, 6144]``).  With the
    bfloat16 output ``out`` holds ``y``; with the NVFP4 output ``values`` holds
    the packed E2M1 pairs (uint8 ``[rows, 3072]``, element ``2i`` in the low
    nibble) and ``scales`` the 128x4-swizzled E4M3 block scales (uint8,
    ``ceil(rows / 128) * 128 * 384`` bytes); with the MXFP8 output ``values``
    holds E4M3 bytes (``[rows, 6144]``, ``torch.float8_e4m3fn``) and ``scales``
    the swizzled UE8M0 scales (uint8, ``ceil(rows / 128) * 128 * 192`` bytes) --
    the layouts ``fp4_quantize`` / ``mxfp8_quantize`` produce for ``y``.  Scale
    bytes of padding rows are left untouched (zero when allocated here).
    """

    output: Fp4AllReduceFusedNormOutput
    residual_out: torch.Tensor
    out: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None


def delivered_families() -> tuple[str, ...]:
    """Program families the installed sources deliver."""
    return FAMILIES


def padded_scale_rows(rows: int) -> int:
    """Rows covered by a swizzled output scale tensor (``rows`` rounded up to a multiple of 128)."""
    return (int(rows) + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE


def fused_norm_scale_bytes(rows: int, output: Fp4AllReduceFusedNormOutput) -> int:
    """Bytes of the swizzled scale tensor of a quantised fused-norm output for ``rows`` rows."""
    if output == Fp4AllReduceFusedNormOutput.NVFP4:
        return padded_scale_rows(rows) * NVFP4_OUTPUT_SCALES_PER_ROW
    if output == Fp4AllReduceFusedNormOutput.MXFP8:
        return padded_scale_rows(rows) * MXFP8_OUTPUT_SCALES_PER_ROW
    raise ValueError("the bfloat16 output has no scale tensor")


def _check_spec(
    spec: Fp4AllReduceFusedNormSpec, workspace: Fp4AllReduceWorkspace
) -> None:
    if not isinstance(spec, Fp4AllReduceFusedNormSpec):
        raise TypeError("spec must be a Fp4AllReduceFusedNormSpec")
    if not isinstance(spec.output, Fp4AllReduceFusedNormOutput):
        raise TypeError("spec.output must be a Fp4AllReduceFusedNormOutput")
    if float(spec.eps) != RMSNORM_EPS:
        raise ValueError(
            f"the delivered fused-norm programs are compiled for eps {RMSNORM_EPS!r}, got {spec.eps!r}"
        )
    weight = spec.weight
    if (
        weight.dim() != 1
        or weight.numel() != HIDDEN
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"spec.weight must be a contiguous bfloat16 [{HIDDEN}] tensor, got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
    if weight.device != workspace.device:
        raise ValueError(
            f"spec.weight must live on {workspace.device}, got {weight.device}"
        )
    if spec.output == Fp4AllReduceFusedNormOutput.NVFP4:
        scale = spec.global_scale
        if scale is None:
            raise ValueError("the NVFP4 output needs spec.global_scale")
        if (
            scale.dtype != torch.float32
            or scale.numel() < 1
            or not scale.is_contiguous()
            or scale.device != workspace.device
        ):
            raise ValueError(
                f"spec.global_scale must be a contiguous float32 tensor on {workspace.device}"
            )
    elif spec.global_scale is not None:
        raise ValueError("spec.global_scale belongs to the NVFP4 output")


def _residual_out(
    residual: torch.Tensor,
    residual_out: Optional[torch.Tensor],
    rows: int,
    workspace: Fp4AllReduceWorkspace,
) -> torch.Tensor:
    residual_out = _output(residual_out, rows, workspace, "residual_out")
    if residual_out.data_ptr() == residual.data_ptr():
        raise ValueError("residual_out must not alias residual")
    return residual_out


def _quantised_outputs(
    spec: Fp4AllReduceFusedNormSpec,
    rows: int,
    workspace: Fp4AllReduceWorkspace,
    values: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.output == Fp4AllReduceFusedNormOutput.NVFP4:
        shape, dtype = (rows, NVFP4_OUTPUT_VALUE_BYTES), torch.uint8
    else:
        shape, dtype = (rows, MXFP8_OUTPUT_VALUE_BYTES), torch.float8_e4m3fn
    if values is None:
        values = torch.empty(shape, dtype=dtype, device=workspace.device)
    elif (
        tuple(values.shape) != shape
        or values.dtype not in (dtype, torch.uint8)
        or not values.is_contiguous()
        or values.device != workspace.device
    ):
        raise ValueError(
            f"values must be a contiguous {dtype} {shape} tensor on {workspace.device}, "
            f"got {tuple(values.shape)} {values.dtype}"
        )
    need = fused_norm_scale_bytes(rows, spec.output)
    if scales is None:
        scales = torch.zeros((need,), dtype=torch.uint8, device=workspace.device)
    elif (
        scales.dtype != torch.uint8
        or scales.numel() < need
        or not scales.is_contiguous()
        or scales.device != workspace.device
    ):
        raise ValueError(
            f"scales must be a contiguous uint8 tensor of at least {need} bytes on "
            f"{workspace.device}"
        )
    return values, scales


def _launch_fused(
    workspace: Fp4AllReduceWorkspace,
    *,
    source: torch.Tensor,
    input_scales: Optional[torch.Tensor],
    input_format: int,
    residual: torch.Tensor,
    spec: Fp4AllReduceFusedNormSpec,
    out: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    launch_with_pdl: bool,
) -> Fp4AllReduceFusedNormResult:
    rows = int(source.shape[0])
    _check_rows(rows, workspace)
    _check_activation(residual, workspace, "residual")
    if residual.shape[0] != rows:
        raise ValueError(f"residual must have {rows} rows, got {residual.shape[0]}")
    _check_spec(spec, workspace)
    residual_out = _residual_out(residual, residual_out, rows, workspace)
    if spec.output == Fp4AllReduceFusedNormOutput.BF16:
        if values is not None or scales is not None:
            raise ValueError("values / scales belong to the quantised outputs")
        out = _output(out, rows, workspace)
        values = scales = None
    else:
        if out is not None:
            raise ValueError("out belongs to the bfloat16 output")
        values, scales = _quantised_outputs(spec, rows, workspace, values, scales)
    workspace.module.fp4_allreduce_rmsnorm(
        source,
        input_scales,
        residual,
        spec.weight,
        out,
        values,
        scales,
        spec.global_scale,
        residual_out,
        workspace.ctrl,
        workspace.table,
        workspace.rank,
        input_format,
        _OUTPUT_CODES[spec.output],
        workspace.grid,
        bool(launch_with_pdl),
    )
    return Fp4AllReduceFusedNormResult(
        output=spec.output,
        residual_out=residual_out,
        out=out,
        values=values,
        scales=scales,
    )


def fp4_allreduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    spec: Fp4AllReduceFusedNormSpec,
    workspace: Fp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> Fp4AllReduceFusedNormResult:
    """All-reduce ``x`` through the NVFP4 transport, add ``residual`` and apply the RMSNorm of ``spec``.

    The sum is the rank-order sum of the NVFP4-quantised partials (as
    :func:`fp4_allreduce`), kept in float32: ``z = sum + residual``,
    ``residual_out = bf16(z)``, ``y = z * rsqrt(mean(z^2) + eps) * (1 + weight)``.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` partial of this rank; ``rows`` is a
        multiple of 4, at most the workspace capacity and the same on every rank.
    residual : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` residual stream of this rank (the
        same tensor on every rank gives rank-identical outputs).
    spec : Fp4AllReduceFusedNormSpec
        Weight, eps, output precision and (NVFP4) global scale.
    workspace : Fp4AllReduceWorkspace
        The collective workspace of the group (family ``"rmsnorm"``).
    out : torch.Tensor, optional
        Bfloat16 output buffer ``[rows, 6144]`` (bfloat16 output only).
    values, scales : torch.Tensor, optional
        Buffers of a quantised output (see :class:`Fp4AllReduceFusedNormResult`);
        ``scales`` may be larger than needed.
    residual_out : torch.Tensor, optional
        Bfloat16 ``[rows, 6144]`` buffer for the summed residual; must not alias
        ``residual``.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute.

    Returns
    -------
    Fp4AllReduceFusedNormResult
        The output tensors (the given buffers, or freshly allocated ones).  The
        call allocates nothing when every buffer is given and performs no host
        synchronisation, so it can be captured into a CUDA graph.
    """
    _check_workspace(workspace, "rmsnorm")
    _check_activation(x, workspace, "x")
    return _launch_fused(
        workspace,
        source=x,
        input_scales=None,
        input_format=0,
        residual=residual,
        spec=spec,
        out=out,
        values=values,
        scales=scales,
        residual_out=residual_out,
        launch_with_pdl=launch_with_pdl,
    )


def fp4_allreduce_rmsnorm_nvfp4(
    records: torch.Tensor,
    global_scales: torch.Tensor,
    residual: torch.Tensor,
    spec: Fp4AllReduceFusedNormSpec,
    workspace: Fp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> Fp4AllReduceFusedNormResult:
    """Sum pre-quantised NVFP4 records, add ``residual`` and apply the RMSNorm of ``spec`` (bfloat16 output).

    ``records`` and ``global_scales`` are as in :func:`fp4_allreduce_nvfp4`; the
    delivered programs produce the bfloat16 output for this input, so
    ``spec.output`` must be ``Fp4AllReduceFusedNormOutput.BF16``.  The other
    parameters and the result are those of :func:`fp4_allreduce_rmsnorm`.
    """
    _check_workspace(workspace, "rmsnorm")
    _check_records(records, global_scales, workspace)
    if (
        isinstance(spec, Fp4AllReduceFusedNormSpec)
        and spec.output != Fp4AllReduceFusedNormOutput.BF16
    ):
        raise ValueError(
            "the fused-norm programs deliver the quantised outputs for bfloat16 input only; "
            "NVFP4 records produce the bfloat16 output"
        )
    return _launch_fused(
        workspace,
        source=records,
        input_scales=global_scales,
        input_format=1,
        residual=residual,
        spec=spec,
        out=out,
        values=None,
        scales=None,
        residual_out=residual_out,
        launch_with_pdl=launch_with_pdl,
    )


__all__ = [
    "CAPACITY",
    "FAMILIES",
    "FillPlan",
    "Fp4AllReduceFusedNormOutput",
    "Fp4AllReduceFusedNormResult",
    "Fp4AllReduceFusedNormSpec",
    "Fp4AllReduceWorkspace",
    "RegionPlan",
    "WindowPlan",
    "delivered_families",
    "fp4_allreduce",
    "fp4_allreduce_nvfp4",
    "fp4_allreduce_rmsnorm",
    "fp4_allreduce_rmsnorm_nvfp4",
    "fused_norm_scale_bytes",
    "padded_scale_rows",
    "window_plan",
]
