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

Cake NVFP4-transport tensor-parallel all-reduce (SM100 / SM103 / SM107).

Every rank quantises its ``[rows, 6144]`` bfloat16 partial to NVFP4 records
(E2M1 values, one E4M3 scale per 16 values, one float32 row scale), pushes
them into Lamport receive windows of its peers and decodes the gathered
records in rank order with float32 accumulation, so the result is the
rank-order sum of the quantised partials and bitwise identical on every rank.
World size 2 runs a one-shot exchange; world size 4 a two-shot exchange whose
owner ranks (``row % 4``) reduce, re-encode and all-gather, in one of three
phase-B schedules chosen per call from the rows each CTA owns.  The
``rmsnorm`` family (world size 2) fuses the one-shot exchange with the
residual add and a Gemma RMSNorm: ``z = sum + residual`` in float32,
``residual_out = bf16(z)``, ``rinv = rsqrt(mean(z^2) + eps)`` from the float32
sum of squares, ``y = bf16(z) * rinv * (weight_bias + weight)`` written as
bfloat16 or quantised to NVFP4 / MXFP8 in the layouts of ``fp4_quantize`` /
``mxfp8_quantize``.  Pre-quantised NVFP4 inputs carry one static scale per
rank.

Usage protocol::

    # 1. Every rank creates the workspace collectively (one per family and input format)
    workspace = CakeFp4AllReduceWorkspace(group, family="allreduce", input_format="bf16")

    # 2. Launch (every rank, same row count, same launch order)
    out = cake_fp4_allreduce(x, workspace)

    # 3. Destroy collectively after the last launch completed
    workspace.destroy()

Requirements: two or four ranks (two for ``rmsnorm``) on one node with CUDA
IPC peer access, SM100 (B200), SM103 (B300) or SM107 devices with the same SM count
that are build targets of this process (``FLASHINFER_CUDA_ARCH_LIST`` when
set, else the visible devices), hidden size 6144, contiguous tensors on the
workspace device and ``rows <= capacity`` (a multiple of 4 for world size 4).
A call allocates nothing when every output buffer is given and performs no
host synchronisation, so it can be captured into a CUDA graph.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from ..api_logging import flashinfer_api
from ..jit.cake_fp4_allreduce import (
    FAMILIES,
    HIDDEN,
    INPUT_FORMATS,
    LIMITS,
    RECORD_BYTES,
    SCALAR_PARAMS,
    device_facts,
    evaluate_linear_form,
    get_cake_fp4_allreduce_module,
    grid_max,
    program_for,
    program_record,
    target_arch,
    window_contract,
    workspace_programs,
    world_sizes,
)
from .cuda_ipc import create_shared_buffer, free_shared_buffer
from .dlpack_utils import pack_strided_memory

DEFAULT_CAPACITY = 16384
RMSNORM_EPS = 1e-6
RMSNORM_WEIGHT_BIAS = 1.0  # Gemma: y = bf16(z) * rinv * (weight_bias + weight)
SCALE_ROW_TILE = 128  # swizzled output scales cover rows padded to a multiple of 128
NVFP4_OUTPUT_VALUE_BYTES = HIDDEN // 2  # 6144 E2M1 values per row, two per byte
NVFP4_OUTPUT_SCALES_PER_ROW = HIDDEN // 16  # one E4M3 scale per 16 values
MXFP8_OUTPUT_VALUE_BYTES = HIDDEN  # one E4M3 byte per value
MXFP8_OUTPUT_SCALES_PER_ROW = HIDDEN // 32  # one UE8M0 scale per 32 values


@dataclass(frozen=True)
class CakeFp4AllReduceRegionPlan:
    """One receive window of a rank: its parameter name, byte size and sentinel fill.

    The fill repeats every ``period_bytes``; within a period the runs of
    ``fills`` (``(word, bytes)``) are written back to back.
    """

    name: str
    param: str
    bytes: int
    period_bytes: int
    fills: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CakeFp4AllReduceWindowPlan:
    """Receive-window layout of one rank for one (family, world size, input format, capacity)."""

    family: str
    world_size: int
    input_format: str
    capacity: int
    ctrl_words: int
    regions: tuple[CakeFp4AllReduceRegionPlan, ...]
    table: tuple[str, ...]

    @property
    def table_entries(self) -> int:
        return len(self.table) * self.world_size

    @property
    def total_bytes(self) -> int:
        return sum(region.bytes for region in self.regions)

    def region(self, name: str) -> CakeFp4AllReduceRegionPlan:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"no receive window named {name!r}")


def cake_fp4_allreduce_window_plan(
    family: str, world_size: int, input_format: str, capacity: int = DEFAULT_CAPACITY
) -> CakeFp4AllReduceWindowPlan:
    """Byte sizes and sentinel fills of one rank's receive windows, evaluated from the registry contract.

    Every window is an exact number of fill periods; every fill run is a whole
    number of 32-bit words.
    """

    _check_family_format(family, world_size, input_format)
    capacity = _check_capacity(capacity, family, world_size)
    contract = window_contract(family, world_size, input_format)
    regions = []
    for buffer in contract["buffers"]:
        nbytes = evaluate_linear_form(buffer["bytes"], capacity)
        period = (
            nbytes
            if buffer["period"] is None
            else evaluate_linear_form(buffer["period"], capacity)
        )
        if nbytes <= 0 or period <= 0 or nbytes % period:
            raise RuntimeError(
                f"receive window {buffer['name']!r} has {nbytes} bytes, not a whole "
                f"number of {period}-byte periods"
            )
        runs: list[tuple[int, int]] = []
        rest = None
        for word, run in buffer["fills"]:
            if run == "rest":
                if rest is not None:
                    raise RuntimeError(
                        f"receive window {buffer['name']!r} names two 'rest' fills"
                    )
                rest = len(runs)
                runs.append((int(word), 0))
            else:
                runs.append((int(word), evaluate_linear_form(run, capacity)))
        fixed = sum(run for _, run in runs)
        if rest is not None:
            runs[rest] = (runs[rest][0], period - fixed)
        elif fixed != period:
            raise RuntimeError(
                f"receive window {buffer['name']!r} fills {fixed} of {period} bytes per period"
            )
        for word, run in runs:
            if run < 0 or run % 4:
                raise RuntimeError(
                    f"receive window {buffer['name']!r} fills {run} bytes with "
                    f"{word:#010x}, not whole words"
                )
        regions.append(
            CakeFp4AllReduceRegionPlan(
                name=str(buffer["name"]),
                param=str(buffer["param"]),
                bytes=nbytes,
                period_bytes=period,
                fills=tuple(runs),
            )
        )
    names = {region.name for region in regions}
    if set(contract["table"]) - names:
        raise RuntimeError(
            f"the peer table names windows {contract['table']} but the contract "
            f"declares {sorted(names)}"
        )
    return CakeFp4AllReduceWindowPlan(
        family=family,
        world_size=int(world_size),
        input_format=input_format,
        capacity=capacity,
        ctrl_words=int(contract["ctrl_words"]),
        regions=tuple(regions),
        table=tuple(contract["table"]),
    )


def _check_family_format(family: str, world_size: int, input_format: str) -> None:
    if family not in FAMILIES:
        raise ValueError(
            f"unknown Cake NVFP4-transport all-reduce family {family!r}; families: {FAMILIES}"
        )
    if input_format not in INPUT_FORMATS:
        raise ValueError(
            f"unknown input format {input_format!r}; input formats: {INPUT_FORMATS}"
        )
    sizes = world_sizes(family)
    if int(world_size) not in sizes:
        raise ValueError(
            f"family {family!r} supports world sizes {sizes}, got {world_size}"
        )


def _check_capacity(capacity: int, family: str, world_size: int) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
    if int(world_size) == 4 and capacity > int(LIMITS["tp4_cap_max"]):
        raise ValueError(
            f"world size 4 supports a capacity of at most {LIMITS['tp4_cap_max']} rows "
            f"(32-bit in-window offsets), got {capacity}"
        )
    return capacity


def _row_multiple(world_size: int) -> int:
    return int(LIMITS["tp4_row_multiple"]) if int(world_size) == 4 else 1


def _as_int32(word: int) -> int:
    return word - (1 << 32) if word >= (1 << 31) else word


def _arm(words: torch.Tensor, region: CakeFp4AllReduceRegionPlan) -> None:
    """Write the periodic sentinel fill of ``region`` into ``words`` (an int32 view of this rank's copy)."""

    periods = words.view(-1, region.period_bytes // 4)
    offset = 0
    for word, run in region.fills:
        run_words = run // 4
        if run_words:
            periods[:, offset : offset + run_words].fill_(_as_int32(word))
        offset += run_words


class CakeFp4AllReduceWorkspace:
    """Per-rank state of the NVFP4-transport all-reduce, created collectively.

    Every rank of ``group`` constructs one instance with the same ``family``,
    ``input_format`` and ``capacity``.  ``family`` is ``"allreduce"`` (world
    size 2 or 4; the sum of the partials) or ``"rmsnorm"`` (world size 2; the
    sum fused with the residual add and the RMSNorm, any output format per
    call); ``input_format`` is ``"bf16"`` (activations) or ``"nvfp4"``
    (pre-quantised records).  ``capacity`` is the largest row count a call may
    carry (at most 103000 for world size 4).  The receive windows are shared
    through CUDA IPC, armed with their Lamport sentinels and published in the
    peer window table; the control window (the registry's control words: one
    four-word call-state slot per CTA) starts at zero.  The CTA count of every
    launch is the SM count of the device, at most the grid bound of the
    workspace's programs (the CTAs their control window provisions for), and
    must agree on every rank.
    Construction loads (and on first use compiles) every program this
    workspace can launch for the device's exact target.  ``destroy()``
    releases the shared windows and is collective as well.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        *,
        family: str = "allreduce",
        input_format: str = "bf16",
        capacity: int = DEFAULT_CAPACITY,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "the Cake NVFP4-transport all-reduce requires an initialized "
                "torch.distributed process group"
            )
        self.group = dist.group.WORLD if group is None else group
        self.world_size = int(dist.get_world_size(self.group))
        self.rank = int(dist.get_rank(self.group))
        _check_family_format(family, self.world_size, input_format)
        self.family = family
        self.input_format = input_format
        self.capacity = _check_capacity(capacity, family, self.world_size)
        self.row_multiple = _row_multiple(self.world_size)
        device = torch.device(torch.cuda.current_device() if device is None else device)
        if device.type != "cuda":
            raise ValueError(f"the workspace needs a CUDA device, got {device}")
        index = torch.cuda.current_device() if device.index is None else device.index
        self.device = torch.device("cuda", index)
        self.target = target_arch(index)
        self.plan = cake_fp4_allreduce_window_plan(
            family, self.world_size, input_format, self.capacity
        )
        self.programs = workspace_programs(
            family, self.world_size, input_format, self.target
        )
        # Every CTA owns a four-word call-state slot of the control window, so
        # the programs bound the CTA count (``grid_max``, the same on every
        # path: one-shot, two-shot and fused norm); the launch uses the
        # device's SM count up to that bound.
        self.grid = min(int(device_facts(index).sm_count), grid_max(self.programs))
        self._modules = {
            program: get_cake_fp4_allreduce_module(program, self.target)
            for program in self.programs
        }
        gathered = [
            torch.zeros(1, dtype=torch.int64, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            gathered,
            torch.tensor([self.grid], dtype=torch.int64, device=self.device),
            group=self.group,
        )
        grids = [int(value.item()) for value in gathered]
        if any(value != self.grid for value in grids):
            raise RuntimeError(
                "the Cake NVFP4-transport all-reduce needs the same CTA count "
                f"(the SM count within the programs' grid bound) on every rank, "
                f"got {grids}"
            )
        self._shared: dict[str, list[int]] = {}
        self._views: list[torch.Tensor] = []
        self.windows: dict[str, torch.Tensor] = {}
        for region in self.plan.regions:
            shared = create_shared_buffer(region.bytes, group=self.group)
            self._shared[region.name] = shared
            words = pack_strided_memory(
                shared[self.rank], region.bytes, region.bytes, 1, torch.int32, index
            )
            _arm(words.view(-1), region)
            window = pack_strided_memory(
                shared[self.rank], region.bytes, region.bytes, 1, torch.uint8, index
            )
            self._views.append(window)
            self.windows[region.param] = window.view(-1)
        pointers = [
            int(pointer) for name in self.plan.table for pointer in self._shared[name]
        ]
        self.ctrl = torch.zeros(
            self.plan.ctrl_words, dtype=torch.int32, device=self.device
        )
        self.table = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        # The MXFP8 programs take the NVFP4 global-scale pointer too; they never read it.
        self._gs_out_unused = torch.ones(1, dtype=torch.float32, device=self.device)
        # No rank may push before every rank has finished arming its windows.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    @property
    def closed(self) -> bool:
        return not self._shared

    def destroy(self) -> None:
        """Release the shared receive windows (collective; a second call is a no-op)."""

        shared, self._shared = self._shared, {}
        if not shared:
            return
        self.windows = {}
        self._views = []
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for region in self.plan.regions:
            free_shared_buffer(shared[region.name], group=self.group)

    def __enter__(self) -> "CakeFp4AllReduceWorkspace":
        return self

    def __exit__(self, *exc_info) -> None:
        self.destroy()

    def program(self, rows: int, output_format: Optional[str] = None) -> str:
        """The program a call with ``rows`` rows (and, for ``rmsnorm``, ``output_format``) runs on this rank."""

        return program_for(
            self.family,
            self.world_size,
            self.input_format,
            output_format,
            int(rows),
            self.grid,
            self.target,
        )

    def module(self, program: str):
        """The built module of ``program`` on this workspace's target."""

        try:
            return self._modules[program]
        except KeyError:
            raise ValueError(
                f"program {program!r} does not belong to this workspace "
                f"(family {self.family!r}, world size {self.world_size}, input format "
                f"{self.input_format!r}, target {self.target!r})"
            ) from None


def _check_workspace(
    workspace: CakeFp4AllReduceWorkspace, family: str, input_format: str
) -> None:
    if not isinstance(workspace, CakeFp4AllReduceWorkspace):
        raise TypeError("workspace must be a CakeFp4AllReduceWorkspace")
    if workspace.closed:
        raise RuntimeError("the workspace has been destroyed")
    if workspace.family != family or workspace.input_format != input_format:
        raise ValueError(
            f"this call needs a workspace of family {family!r} and input format "
            f"{input_format!r}, got family {workspace.family!r} and input format "
            f"{workspace.input_format!r}"
        )


def _check_rows(rows: int, workspace: CakeFp4AllReduceWorkspace) -> None:
    if rows < 1 or rows > workspace.capacity:
        raise ValueError(f"rows must be in [1, {workspace.capacity}], got {rows}")
    if rows % workspace.row_multiple:
        raise ValueError(
            f"world size {workspace.world_size} needs rows to be a multiple of "
            f"{workspace.row_multiple}, got {rows}"
        )


def _check_activation(
    x: torch.Tensor, workspace: CakeFp4AllReduceWorkspace, name: str
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
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
    workspace: CakeFp4AllReduceWorkspace,
) -> None:
    if not isinstance(records, torch.Tensor) or not isinstance(
        global_scales, torch.Tensor
    ):
        raise TypeError("records and global_scales must be torch.Tensors")
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
            "global_scales must be a contiguous float32 tensor of at least "
            f"{workspace.world_size} elements on {workspace.device}"
        )


def _output(
    out: Optional[torch.Tensor],
    rows: int,
    workspace: CakeFp4AllReduceWorkspace,
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


def _launch(
    workspace: CakeFp4AllReduceWorkspace,
    program: str,
    operands: dict[str, torch.Tensor],
    rows: int,
    *,
    eps: Optional[float] = None,
    weight_bias: Optional[float] = None,
    launch_with_pdl: bool,
) -> None:
    """Call the program's binding: the pointer parameters in kernel order, then the scalars, the grid and the PDL flag."""

    record = program_record(program)
    params = tuple(record["params"])
    pointers = tuple(name for name in params if name not in SCALAR_PARAMS)
    scalars = tuple(name for name in params if name in SCALAR_PARAMS)
    expected = ("rank", "rows", "cap") + (("eps", "wbias") if "eps" in scalars else ())
    if params != pointers + expected:
        raise RuntimeError(
            f"program {program!r} declares parameters {params}, the binding "
            f"expects the pointers followed by {expected}"
        )
    args: list = []
    for name in pointers:
        if name in operands:
            args.append(operands[name])
        elif name in workspace.windows:
            args.append(workspace.windows[name])
        elif name == "peer_ptrs":
            args.append(workspace.table)
        elif name == "ctrl":
            args.append(workspace.ctrl)
        else:
            raise RuntimeError(
                f"program {program!r} parameter {name!r} has no operand or window"
            )
    args += [workspace.rank, int(rows), workspace.capacity]
    if "eps" in scalars:
        args += [float(eps), float(weight_bias)]
    args += [workspace.grid, bool(launch_with_pdl)]
    getattr(workspace.module(program), record["entry"])(*args)


@flashinfer_api
def cake_fp4_allreduce(
    x: torch.Tensor,
    workspace: CakeFp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """Sum ``x`` over the workspace's group through the NVFP4 transport.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` partial of this rank; ``rows`` is
        at most ``workspace.capacity``, a multiple of 4 for world size 4, and
        the same on every rank of the call.
    workspace : CakeFp4AllReduceWorkspace
        The collective workspace of the group (family ``"allreduce"``, input
        format ``"bf16"``).
    out : torch.Tensor, optional
        Contiguous bfloat16 ``[rows, 6144]`` output; allocated when omitted.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute (the
        kernel waits for its predecessor before touching the windows).

    Returns
    -------
    torch.Tensor
        ``out``: the rank-order sum of the NVFP4-quantised partials, accumulated
        in float32 and rounded to bfloat16, bitwise identical on every rank.
        The call allocates nothing when ``out`` is given and performs no host
        synchronisation, so it can be captured into a CUDA graph.
    """

    _check_workspace(workspace, "allreduce", "bf16")
    _check_activation(x, workspace, "x")
    rows = int(x.shape[0])
    _check_rows(rows, workspace)
    out = _output(out, rows, workspace)
    _launch(
        workspace,
        workspace.program(rows),
        {"x": x, "out": out},
        rows,
        launch_with_pdl=launch_with_pdl,
    )
    return out


@flashinfer_api
def cake_fp4_allreduce_nvfp4(
    records: torch.Tensor,
    global_scales: torch.Tensor,
    workspace: CakeFp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """Sum pre-quantised NVFP4 records over the workspace's group.

    Parameters
    ----------
    records : torch.Tensor
        Contiguous uint8 ``[rows, 3456]`` records of this rank: 3072 packed
        E2M1 value bytes (element ``2i`` in the low nibble) followed by 384
        E4M3 block scales (one per 16 values).  Record ``r`` decodes as
        ``e2m1(value) * e4m3(scale) * global_scales[r]``.
    global_scales : torch.Tensor
        Contiguous float32 tensor of at least ``world_size`` elements holding
        the static scale of every rank's records; identical on every rank.
    workspace : CakeFp4AllReduceWorkspace
        The collective workspace of the group (family ``"allreduce"``, input
        format ``"nvfp4"``).
    out : torch.Tensor, optional
        Contiguous bfloat16 ``[rows, 6144]`` output; allocated when omitted.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute.

    Returns
    -------
    torch.Tensor
        ``out``: the rank-order float32 sum of the decoded records rounded to
        bfloat16, bitwise identical on every rank.
    """

    _check_workspace(workspace, "allreduce", "nvfp4")
    _check_records(records, global_scales, workspace)
    rows = int(records.shape[0])
    _check_rows(rows, workspace)
    out = _output(out, rows, workspace)
    _launch(
        workspace,
        workspace.program(rows),
        {"xq": records, "gs": global_scales, "out": out},
        rows,
        launch_with_pdl=launch_with_pdl,
    )
    return out


class CakeFp4AllReduceFusedNormOutput(enum.Enum):
    """Output precision of the fused all-reduce + residual + RMSNorm epilogue."""

    BF16 = "bf16"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


@dataclass(frozen=True)
class CakeFp4AllReduceFusedNormSpec:
    """Request of the fused epilogue: residual add, then RMSNorm scaled by ``weight_bias + weight``.

    ``weight`` is a contiguous bfloat16 ``[6144]`` tensor on the workspace
    device, ``eps`` the RMSNorm epsilon and ``weight_bias`` the constant added
    to the weight (Gemma: ``1.0``); both are runtime scalars.  ``output`` is
    the precision of the normalised result.  ``global_scale`` is required for
    the NVFP4 output: a float32 tensor of at least one element on the device
    holding the quantiser's input global scale (``448 * 6 / amax(y)`` in the
    convention of ``fp4_quantize``); the kernel reads it, the host never does,
    so the call stays graph capturable.
    """

    weight: torch.Tensor
    eps: float = RMSNORM_EPS
    weight_bias: float = RMSNORM_WEIGHT_BIAS
    output: CakeFp4AllReduceFusedNormOutput = CakeFp4AllReduceFusedNormOutput.BF16
    global_scale: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class CakeFp4AllReduceFusedNormResult:
    """Outputs of one fused call.

    ``residual_out`` is ``bf16(sum + residual)`` (``[rows, 6144]``).  With the
    bfloat16 output ``out`` holds ``y``; with the NVFP4 output ``values`` holds
    the packed E2M1 pairs (uint8 ``[rows, 3072]``, element ``2i`` in the low
    nibble) and ``scales`` the 128x4-swizzled E4M3 block scales (uint8,
    ``ceil(rows / 128) * 128 * 384`` bytes); with the MXFP8 output ``values``
    holds E4M3 bytes (``[rows, 6144]``, ``torch.float8_e4m3fn``) and
    ``scales`` the swizzled UE8M0 scales (uint8, ``ceil(rows / 128) * 128 *
    192`` bytes) -- the layouts ``fp4_quantize`` / ``mxfp8_quantize`` produce
    for ``y``.  Scale bytes of padding rows are left untouched (zero when
    allocated here).
    """

    output: CakeFp4AllReduceFusedNormOutput
    residual_out: torch.Tensor
    out: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None


def padded_scale_rows(rows: int) -> int:
    """Rows covered by a swizzled output scale tensor (``rows`` rounded up to a multiple of 128)."""

    return (int(rows) + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE


def fused_norm_scale_bytes(rows: int, output: CakeFp4AllReduceFusedNormOutput) -> int:
    """Bytes of the swizzled scale tensor of a quantised fused-norm output for ``rows`` rows."""

    if output == CakeFp4AllReduceFusedNormOutput.NVFP4:
        return padded_scale_rows(rows) * NVFP4_OUTPUT_SCALES_PER_ROW
    if output == CakeFp4AllReduceFusedNormOutput.MXFP8:
        return padded_scale_rows(rows) * MXFP8_OUTPUT_SCALES_PER_ROW
    raise ValueError("the bfloat16 output has no scale tensor")


def _check_spec(
    spec: CakeFp4AllReduceFusedNormSpec, workspace: CakeFp4AllReduceWorkspace
) -> None:
    if not isinstance(spec, CakeFp4AllReduceFusedNormSpec):
        raise TypeError("spec must be a CakeFp4AllReduceFusedNormSpec")
    if not isinstance(spec.output, CakeFp4AllReduceFusedNormOutput):
        raise TypeError("spec.output must be a CakeFp4AllReduceFusedNormOutput")
    for name in ("eps", "weight_bias"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"spec.{name} must be a float, got {value!r}")
    weight = spec.weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError("spec.weight must be a torch.Tensor")
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
    if spec.output == CakeFp4AllReduceFusedNormOutput.NVFP4:
        scale = spec.global_scale
        if scale is None:
            raise ValueError("the NVFP4 output needs spec.global_scale")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float32
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
    workspace: CakeFp4AllReduceWorkspace,
) -> torch.Tensor:
    residual_out = _output(residual_out, rows, workspace, "residual_out")
    if residual_out.data_ptr() == residual.data_ptr():
        raise ValueError("residual_out must not alias residual")
    return residual_out


def _quantised_outputs(
    spec: CakeFp4AllReduceFusedNormSpec,
    rows: int,
    workspace: CakeFp4AllReduceWorkspace,
    values: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.output == CakeFp4AllReduceFusedNormOutput.NVFP4:
        shape, dtype = (rows, NVFP4_OUTPUT_VALUE_BYTES), torch.uint8
    else:
        shape, dtype = (rows, MXFP8_OUTPUT_VALUE_BYTES), torch.float8_e4m3fn
    if values is None:
        values = torch.empty(shape, dtype=dtype, device=workspace.device)
    elif (
        not isinstance(values, torch.Tensor)
        or tuple(values.shape) != shape
        or values.dtype not in (dtype, torch.uint8)
        or not values.is_contiguous()
        or values.device != workspace.device
    ):
        raise ValueError(
            f"values must be a contiguous {dtype} {shape} tensor on {workspace.device}"
        )
    need = fused_norm_scale_bytes(rows, spec.output)
    if scales is None:
        scales = torch.zeros((need,), dtype=torch.uint8, device=workspace.device)
    elif (
        not isinstance(scales, torch.Tensor)
        or scales.dtype != torch.uint8
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
    workspace: CakeFp4AllReduceWorkspace,
    *,
    operands: dict[str, torch.Tensor],
    rows: int,
    residual: torch.Tensor,
    spec: CakeFp4AllReduceFusedNormSpec,
    out: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    launch_with_pdl: bool,
) -> CakeFp4AllReduceFusedNormResult:
    _check_rows(rows, workspace)
    _check_activation(residual, workspace, "residual")
    if residual.shape[0] != rows:
        raise ValueError(f"residual must have {rows} rows, got {residual.shape[0]}")
    _check_spec(spec, workspace)
    residual_out = _residual_out(residual, residual_out, rows, workspace)
    operands = dict(operands, res_in=residual, w=spec.weight, res_out=residual_out)
    if spec.output == CakeFp4AllReduceFusedNormOutput.BF16:
        if values is not None or scales is not None:
            raise ValueError("values / scales belong to the quantised outputs")
        out = _output(out, rows, workspace)
        values = scales = None
        operands["out"] = out
    else:
        if out is not None:
            raise ValueError("out belongs to the bfloat16 output")
        values, scales = _quantised_outputs(spec, rows, workspace, values, scales)
        operands["outq"] = (
            values if values.dtype == torch.uint8 else values.view(torch.uint8)
        )
        operands["out_sf"] = scales
        operands["gs_out"] = (
            spec.global_scale
            if spec.output == CakeFp4AllReduceFusedNormOutput.NVFP4
            else workspace._gs_out_unused
        )
    _launch(
        workspace,
        workspace.program(rows, spec.output.value),
        operands,
        rows,
        eps=spec.eps,
        weight_bias=spec.weight_bias,
        launch_with_pdl=launch_with_pdl,
    )
    return CakeFp4AllReduceFusedNormResult(
        output=spec.output,
        residual_out=residual_out,
        out=out,
        values=values,
        scales=scales,
    )


@flashinfer_api
def cake_fp4_allreduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    spec: CakeFp4AllReduceFusedNormSpec,
    workspace: CakeFp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> CakeFp4AllReduceFusedNormResult:
    """All-reduce ``x`` through the NVFP4 transport, add ``residual`` and apply the RMSNorm of ``spec``.

    The sum is the rank-order float32 sum of the NVFP4-quantised partials (as
    :func:`cake_fp4_allreduce`): ``z = sum + residual`` in float32,
    ``residual_out = bf16(z)``, ``rinv = rsqrt(mean(z^2) + eps)`` from the
    float32 sum of squares, ``y = bf16(z) * rinv * (weight_bias + weight)``
    (the bfloat16-rounded ``z`` enters the product).  ``residual_out`` and the
    quantised outputs are bitwise identical on every rank.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` partial of this rank; ``rows`` is
        at most ``workspace.capacity`` and the same on every rank.
    residual : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` residual stream of this rank (the
        same tensor on every rank gives rank-identical outputs).
    spec : CakeFp4AllReduceFusedNormSpec
        Weight, eps, weight bias, output precision and (NVFP4) global scale.
    workspace : CakeFp4AllReduceWorkspace
        The collective workspace of the group (family ``"rmsnorm"``, input
        format ``"bf16"``).
    out : torch.Tensor, optional
        Bfloat16 output buffer ``[rows, 6144]`` (bfloat16 output only).
    values, scales : torch.Tensor, optional
        Buffers of a quantised output (see
        :class:`CakeFp4AllReduceFusedNormResult`); ``scales`` may be larger
        than needed.
    residual_out : torch.Tensor, optional
        Bfloat16 ``[rows, 6144]`` buffer for the summed residual; must not
        alias ``residual``.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute.

    Returns
    -------
    CakeFp4AllReduceFusedNormResult
        The output tensors (the given buffers, or freshly allocated ones).  The
        call allocates nothing when every buffer is given and performs no host
        synchronisation, so it can be captured into a CUDA graph.
    """

    _check_workspace(workspace, "rmsnorm", "bf16")
    _check_activation(x, workspace, "x")
    return _launch_fused(
        workspace,
        operands={"x": x},
        rows=int(x.shape[0]),
        residual=residual,
        spec=spec,
        out=out,
        values=values,
        scales=scales,
        residual_out=residual_out,
        launch_with_pdl=launch_with_pdl,
    )


@flashinfer_api
def cake_fp4_allreduce_rmsnorm_nvfp4(
    records: torch.Tensor,
    global_scales: torch.Tensor,
    residual: torch.Tensor,
    spec: CakeFp4AllReduceFusedNormSpec,
    workspace: CakeFp4AllReduceWorkspace,
    *,
    out: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> CakeFp4AllReduceFusedNormResult:
    """Sum pre-quantised NVFP4 records, add ``residual`` and apply the RMSNorm of ``spec``.

    The sum is the rank-order float32 sum of the decoded records (as
    :func:`cake_fp4_allreduce_nvfp4`); the epilogue is that of
    :func:`cake_fp4_allreduce_rmsnorm`, with every output precision available.

    Parameters
    ----------
    records : torch.Tensor
        Contiguous uint8 ``[rows, 3456]`` records of this rank, encoded as for
        :func:`cake_fp4_allreduce_nvfp4`; ``rows`` is at most
        ``workspace.capacity`` and the same on every rank.
    global_scales : torch.Tensor
        Contiguous float32 tensor of at least ``world_size`` elements holding
        the static scale of every rank's records; identical on every rank.
    residual : torch.Tensor
        Contiguous bfloat16 ``[rows, 6144]`` residual stream of this rank.
    spec : CakeFp4AllReduceFusedNormSpec
        Weight, eps, weight bias, output precision and (NVFP4) global scale.
    workspace : CakeFp4AllReduceWorkspace
        The collective workspace of the group (family ``"rmsnorm"``, input
        format ``"nvfp4"``).
    out, values, scales, residual_out : torch.Tensor, optional
        Output buffers as for :func:`cake_fp4_allreduce_rmsnorm`.
    launch_with_pdl : bool
        Launch with the programmatic-stream-serialization attribute.

    Returns
    -------
    CakeFp4AllReduceFusedNormResult
        The output tensors (the given buffers, or freshly allocated ones).
    """

    _check_workspace(workspace, "rmsnorm", "nvfp4")
    _check_records(records, global_scales, workspace)
    return _launch_fused(
        workspace,
        operands={"xq": records, "gs": global_scales},
        rows=int(records.shape[0]),
        residual=residual,
        spec=spec,
        out=out,
        values=values,
        scales=scales,
        residual_out=residual_out,
        launch_with_pdl=launch_with_pdl,
    )


__all__ = [
    "DEFAULT_CAPACITY",
    "HIDDEN",
    "MXFP8_OUTPUT_SCALES_PER_ROW",
    "MXFP8_OUTPUT_VALUE_BYTES",
    "NVFP4_OUTPUT_SCALES_PER_ROW",
    "NVFP4_OUTPUT_VALUE_BYTES",
    "RECORD_BYTES",
    "RMSNORM_EPS",
    "RMSNORM_WEIGHT_BIAS",
    "SCALE_ROW_TILE",
    "CakeFp4AllReduceFusedNormOutput",
    "CakeFp4AllReduceFusedNormResult",
    "CakeFp4AllReduceFusedNormSpec",
    "CakeFp4AllReduceRegionPlan",
    "CakeFp4AllReduceWindowPlan",
    "CakeFp4AllReduceWorkspace",
    "cake_fp4_allreduce",
    "cake_fp4_allreduce_nvfp4",
    "cake_fp4_allreduce_rmsnorm",
    "cake_fp4_allreduce_rmsnorm_nvfp4",
    "cake_fp4_allreduce_window_plan",
    "fused_norm_scale_bytes",
    "padded_scale_rows",
]
