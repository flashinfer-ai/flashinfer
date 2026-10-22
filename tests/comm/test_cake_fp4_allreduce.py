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

Distributed correctness of the Cake NVFP4-transport tensor-parallel all-reduce.

One test per (family, input format, world size): the ``allreduce`` family on two
ranks (one-shot) and four ranks (two-shot) with bfloat16 partials or
pre-quantised NVFP4 records, and the ``rmsnorm`` family (the two-rank one-shot
exchange fused with the residual add and the Gemma RMSNorm; bfloat16, NVFP4 and
MXFP8 outputs).  Each test spawns one process per rank on CUDA devices
``0..world_size-1`` and checks, against the exact float32 reference of
``tests/comm/cake_fp4_allreduce_reference.py`` (its sum of squares in the order
of the fused-norm program the launch runs, from the registry): the workspace
plan, two launches
of every row case on one workspace (the second finds the rotated receive
buffer and the advanced call state), an odd row count under a larger capacity,
back-to-back and alternating-row calls, CUDA-graph capture and replay, negative
controls, the runtime ``eps`` / ``weight_bias`` scalars, the argument
validation and the destroy semantics.  All-reduce outputs and ``residual_out``
are bitwise; the bfloat16 ``y`` of the fused norm is within ``atol = rtol =
1e-2``; the quantised outputs equal the reference quantisation outside the E2M1
rounding ties with at most 1e-4 of the elements allowed to differ.

Skipped without enough CUDA devices or when their capability is not a build
target of the export (``FLASHINFER_CUDA_ARCH_LIST`` or the visible devices).
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm.cake_fp4_allreduce import (
    CakeFp4AllReduceFusedNormOutput,
    CakeFp4AllReduceFusedNormResult,
    CakeFp4AllReduceFusedNormSpec,
    CakeFp4AllReduceWorkspace,
    cake_fp4_allreduce,
    cake_fp4_allreduce_nvfp4,
    cake_fp4_allreduce_rmsnorm,
    cake_fp4_allreduce_rmsnorm_nvfp4,
)
from flashinfer.jit import cake_fp4_allreduce as loader
from flashinfer.jit.cake_fp4_allreduce_programs import LIMITS, PROGRAMS, WINDOWS
from tests.comm import cake_fp4_allreduce_reference as ref

HIDDEN = ref.HIDDEN
RECORD_BYTES = ref.RECORD_BYTES
DEFAULT_CAPACITY = 16384
LARGER_CAPACITY = 4096  # the odd-row cases run under this capacity
# Fewer rows than teams / warps, one row per team plus some, the smallest
# production tier and the largest tier; the four-rank cases are multiples of 4,
# and bv0_rows() adds the grid-derived tier that selects the bv-0 two-shot
# schedule (no fixed row count does so on every shipped grid). The fused norm
# adds the 4096-row tier, which runs its residual-prefetch programs (18 to 28
# rows per CTA on 148 to 212 SMs).
ROWS = {
    ("allreduce", 2): (3, 6, 1024, 16384),
    ("allreduce", 4): (4, 8, 44, 1024, 16384),
    ("rmsnorm", 2): (3, 11, 1024, 4096, 16384),
}
ODD_ROWS = {("allreduce", 2): 1031, ("allreduce", 4): 1028, ("rmsnorm", 2): 1031}
SMALL_ROWS = {("allreduce", 2): 3, ("allreduce", 4): 4, ("rmsnorm", 2): 3}
MID_ROWS = 1024


def bv0_rows(grid: int) -> int:
    """The four-rank tier that selects the bv-0 two-shot schedule on ``grid`` CTAs.

    The bv-0 window is 40 < rows // grid <= 64, so the tier follows the grid: 52
    rows per CTA, rounded down to a multiple of 4 (7696 rows on 148 SMs, 7904 on
    152, 11024 on 212), under DEFAULT_CAPACITY.
    """
    rows = 4 * ((52 * grid) // 4)
    assert loader.select_bv(rows, grid) == 0 and rows <= DEFAULT_CAPACITY, (rows, grid)
    return rows


LAUNCHES_PER_CASE = 2
BACK_TO_BACK_CALLS = 4
GRAPH_CALLS = 3
GRAPH_REPLAYS = 10
SEEDS = {"allreduce": 1000, "rmsnorm": 2000}
JOIN_TIMEOUT_S = 1800.0  # the first run compiles the programs of the family
CLEANUP_TIMEOUT_S = 30.0
CASES = [
    ("allreduce", "bf16", 2),
    ("allreduce", "nvfp4", 2),
    ("allreduce", "bf16", 4),
    ("allreduce", "nvfp4", 4),
    ("rmsnorm", "bf16", 2),
    ("rmsnorm", "nvfp4", 2),
]
OUTPUTS = (
    CakeFp4AllReduceFusedNormOutput.BF16,
    CakeFp4AllReduceFusedNormOutput.NVFP4,
    CakeFp4AllReduceFusedNormOutput.MXFP8,
)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, tensor.contiguous(), group=group)
    return peers


def _check_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {tuple(actual.shape)} {actual.dtype}, expected "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    differing = ref.count_mismatches(actual, expected)
    if differing:
        raise AssertionError(
            f"{label}: {differing} of {actual.numel()} elements differ from the reference"
        )


def _check_mismatch_fraction(
    actual: torch.Tensor, expected: torch.Tensor, allowance: float, label: str
) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {tuple(actual.shape)} {actual.dtype}, expected "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    fraction = ref.mismatch_fraction(actual, expected)
    if fraction > allowance:
        raise AssertionError(
            f"{label}: {fraction:.3e} of {actual.numel()} elements differ from the "
            f"reference (allowance {allowance:.0e})"
        )


def _check_y(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """The bfloat16 ``y`` inside ``atol = rtol = 1e-2`` of the reference."""
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"{label}: got {tuple(actual.shape)} {actual.dtype}, expected "
            f"{tuple(expected.shape)} {torch.bfloat16}"
        )
    violations = ref.tolerance_violations(actual, expected, ref.Y_ATOL, ref.Y_RTOL)
    if violations:
        raise AssertionError(
            f"{label}: {violations} of {actual.numel()} elements outside atol = rtol = "
            f"{ref.Y_ATOL} (largest bfloat16 distance {ref.max_bf16_ulp(actual, expected)})"
        )


def _check_identical_across_ranks(
    tensor: torch.Tensor, group: dist.ProcessGroup, label: str
) -> None:
    for peer_rank, peer in enumerate(_gather(tensor, group)):
        if ref.count_mismatches(peer, tensor):
            raise AssertionError(
                f"{label}: rank {dist.get_rank(group)} and rank {peer_rank} disagree"
            )


def _expect(exc_type: type, fn: Callable[[], object], label: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - the test reports the wrong type
        raise AssertionError(
            f"{label}: raised {type(exc).__name__} instead of {exc_type.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: no {exc_type.__name__} was raised")


def _sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _check_workspace(
    workspace: CakeFp4AllReduceWorkspace,
    family: str,
    input_format: str,
    world_size: int,
    rank: int,
    device: torch.device,
    capacity: int,
) -> None:
    """World size, rank, grid == SM count, table entries, control words, capacity and target of a fresh workspace."""
    window = WINDOWS[f"{family}/{world_size}/{input_format}"]
    assert workspace.family == family
    assert workspace.input_format == input_format
    assert workspace.world_size == world_size
    assert workspace.rank == rank
    assert workspace.capacity == capacity
    assert workspace.device == device
    assert workspace.grid == _sm_count(device)
    assert workspace.table.dtype == torch.int64
    assert workspace.table.numel() == len(window["table"]) * world_size
    assert all(int(pointer) != 0 for pointer in workspace.table.tolist())
    assert workspace.ctrl.dtype in (torch.int32, torch.uint32)
    assert workspace.ctrl.numel() == window["ctrl_words"]
    assert int(torch.count_nonzero(workspace.ctrl).item()) == 0
    assert workspace.target == loader.target_arch(device.index)
    assert workspace.target in ("sm_100a", "sm_103a", "sm_107a")
    capability = torch.cuda.get_device_capability(device)
    assert loader.supported_capability(capability) == tuple(capability)
    if family == "allreduce" and world_size == 2:
        assert workspace.grid <= LIMITS["tp2_oneshot_grid_max"]
    if world_size == 4:
        assert capacity <= LIMITS["tp4_cap_max"]
        for rows in (*ROWS[("allreduce", 4)], bv0_rows(workspace.grid)):
            bv = loader.select_bv(rows, workspace.grid)
            assert bv in (0, 1, 2)
            program = loader.program_for(
                family="allreduce",
                world_size=4,
                input_format=input_format,
                output_format=None,
                rows=rows,
                grid=workspace.grid,
                target=workspace.target,
            )
            record = PROGRAMS[program]
            assert record["bv"] == bv and workspace.target in record["targets"]
            assert program.endswith("_rst") == (record["reset"] == "st")


# ---------------------------------------------------------------------------
# all-reduce family
# ---------------------------------------------------------------------------


@dataclass
class AllReduceCase:
    """Every rank's input of one launch (bfloat16 partials or NVFP4 records) and the expected bfloat16 sum."""

    rows: int
    inputs: list[torch.Tensor]
    global_scales: Optional[torch.Tensor]
    expected: torch.Tensor

    def launch(self, workspace: CakeFp4AllReduceWorkspace, rank: int, **kwargs):
        if self.global_scales is None:
            return cake_fp4_allreduce(self.inputs[rank], workspace, **kwargs)
        return cake_fp4_allreduce_nvfp4(
            self.inputs[rank], self.global_scales, workspace, **kwargs
        )


def _allreduce_case(
    input_format: str, rows: int, seed: int, world_size: int, device: torch.device
) -> AllReduceCase:
    xs = ref.partials(rows, seed, world_size, device)
    if input_format == "bf16":
        return AllReduceCase(rows, xs, None, ref.reference_allreduce(xs))
    records, global_scales = ref.records_of(xs)
    return AllReduceCase(
        rows,
        records,
        global_scales,
        ref.reference_allreduce_nvfp4(records, global_scales),
    )


def _allreduce_cases(
    workspace, rows_list, input_format, world_size, rank, device, group, seed
) -> None:
    """Two launches per row count on one workspace, bitwise against the reference and identical across the ranks."""
    for rows in rows_list:
        for launch in range(LAUNCHES_PER_CASE):
            case = _allreduce_case(
                input_format,
                rows,
                ref.launch_seed(seed + 17 * rows, launch),
                world_size,
                device,
            )
            given = (
                torch.full(
                    (rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
                )
                if launch % 2
                else None
            )
            out = case.launch(
                workspace, rank, out=given, launch_with_pdl=launch % 2 == 0
            )
            torch.cuda.synchronize(device)
            if given is not None:
                assert out.data_ptr() == given.data_ptr()
            label = (
                f"{input_format} rows={rows} cap={workspace.capacity} launch={launch}"
            )
            _check_equal(out, case.expected, label)
            _check_identical_across_ranks(out, group, label)


def _back_to_back(workspace, input_format, world_size, rank, device, group, seed):
    """Four calls without host synchronisation into caller-provided buffers: the three-buffer rotation wraps around."""
    rows = MID_ROWS
    cases = [
        _allreduce_case(input_format, rows, seed + 7777 * call, world_size, device)
        for call in range(BACK_TO_BACK_CALLS)
    ]
    outputs = [
        torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device) for _ in cases
    ]
    for case, out in zip(cases, outputs, strict=True):
        case.launch(workspace, rank, out=out)
    torch.cuda.synchronize(device)
    for call, (case, out) in enumerate(zip(cases, outputs, strict=True)):
        _check_equal(out, case.expected, f"{input_format} back-to-back call {call}")


def _alternating_rows(workspace, input_format, world_size, rank, device, group, seed):
    """Changing row counts between calls: the in-kernel reset must cover the previous call's ranges."""
    key = ("allreduce", world_size)
    sequence = (ROWS[key][-1], SMALL_ROWS[key], ROWS[key][-1], MID_ROWS)
    for call, rows in enumerate(sequence):
        case = _allreduce_case(input_format, rows, seed + 31 * call, world_size, device)
        out = case.launch(workspace, rank)
        torch.cuda.synchronize(device)
        _check_equal(
            out,
            case.expected,
            f"{input_format} alternating rows call {call} rows={rows}",
        )


def _graph_replay(workspace, input_format, world_size, rank, device, group, seed):
    """Three captured calls replayed ten times continue the rotation; an eager call after the replays as well."""
    rows = MID_ROWS
    case = _allreduce_case(input_format, rows, seed + 4242, world_size, device)
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        case.launch(workspace, rank, out=out)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    _check_equal(out, case.expected, f"{input_format} warm-up call before capture")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(GRAPH_CALLS):
            case.launch(workspace, rank, out=out)
    for replay in range(GRAPH_REPLAYS):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        _check_equal(out, case.expected, f"{input_format} graph replay {replay}")
    eager = case.launch(workspace, rank)
    torch.cuda.synchronize(device)
    _check_equal(eager, case.expected, f"{input_format} eager call after the replays")
    del graph


def _negative_control(workspace, input_format, world_size, rank, device, group, seed):
    """A stale expectation must fail: a perturbed partial on the last rank, or a wrong static scale of its records."""
    rows = MID_ROWS
    case = _allreduce_case(input_format, rows, seed + 99, world_size, device)
    last = world_size - 1
    if input_format == "bf16":
        perturbed = [x.clone() for x in case.inputs]
        perturbed[last][rows // 2, HIDDEN // 2] += 1000.0
        wrong = AllReduceCase(rows, perturbed, None, ref.reference_allreduce(perturbed))
    else:
        scales = case.global_scales.clone()
        scales[last] *= 2.0
        wrong = AllReduceCase(
            rows,
            case.inputs,
            scales,
            ref.reference_allreduce_nvfp4(case.inputs, scales),
        )
    out = wrong.launch(workspace, rank)
    torch.cuda.synchronize(device)
    if ref.count_mismatches(out, case.expected) == 0:
        raise AssertionError(
            f"{input_format} negative control: the perturbed input did not change the result"
        )
    _check_equal(
        out, wrong.expected, f"{input_format} reference of the perturbed input"
    )


def _validation(workspace, input_format, world_size, device):
    """Argument errors are reported before any launch."""
    bf16 = dict(dtype=torch.bfloat16, device=device)
    u8 = dict(dtype=torch.uint8, device=device)
    rows = 64
    capacity = workspace.capacity
    x = torch.zeros((rows, HIDDEN), **bf16)
    records = torch.zeros((rows, RECORD_BYTES), **u8)
    scales = torch.ones((world_size,), dtype=torch.float32, device=device)
    if input_format == "bf16":

        def call(tensor, **kwargs):
            return cake_fp4_allreduce(tensor, workspace, **kwargs)

        _expect(
            ValueError,
            lambda: call(
                torch.zeros((rows, HIDDEN), dtype=torch.float32, device=device)
            ),
            "float32 input",
        )
        _expect(
            ValueError,
            lambda: call(torch.zeros((rows, HIDDEN - 1), **bf16)),
            "hidden size",
        )
        _expect(
            ValueError,
            lambda: call(torch.zeros((capacity + 1, HIDDEN), **bf16)),
            "rows above the capacity",
        )
        _expect(
            ValueError,
            lambda: call(torch.zeros((rows, 2 * HIDDEN), **bf16)[:, ::2]),
            "non-contiguous input",
        )
        _expect(ValueError, lambda: call(torch.zeros((0, HIDDEN), **bf16)), "zero rows")
        _expect(
            ValueError,
            lambda: call(
                x, out=torch.zeros((rows, HIDDEN), dtype=torch.float32, device=device)
            ),
            "float32 out",
        )
        _expect(
            ValueError,
            lambda: call(x, out=torch.zeros((rows + 1, HIDDEN), **bf16)),
            "out rows",
        )
        if world_size == 4:
            _expect(
                ValueError,
                lambda: call(torch.zeros((6, HIDDEN), **bf16)),
                "rows not a multiple of 4",
            )
        _expect(
            ValueError,
            lambda: cake_fp4_allreduce_nvfp4(records, scales, workspace),
            "NVFP4 records on a bfloat16-input workspace",
        )
    else:

        def call(tensor, scale_tensor=scales, **kwargs):
            return cake_fp4_allreduce_nvfp4(tensor, scale_tensor, workspace, **kwargs)

        _expect(
            ValueError,
            lambda: call(torch.zeros((rows, RECORD_BYTES - 1), **u8)),
            "record width",
        )
        _expect(
            ValueError,
            lambda: call(
                torch.zeros((rows, RECORD_BYTES), dtype=torch.int8, device=device)
            ),
            "record dtype",
        )
        _expect(ValueError, lambda: call(records, scales[:-1]), "scale count")
        _expect(
            ValueError,
            lambda: call(records, scales.to(torch.float64)),
            "scale dtype",
        )
        _expect(
            ValueError,
            lambda: call(torch.zeros((capacity + 1, RECORD_BYTES), **u8)),
            "rows above the capacity",
        )
        _expect(
            ValueError, lambda: call(torch.zeros((0, RECORD_BYTES), **u8)), "zero rows"
        )
        _expect(
            ValueError,
            lambda: call(records, out=torch.zeros((rows + 1, HIDDEN), **bf16)),
            "out rows",
        )
        if world_size == 4:
            _expect(
                ValueError,
                lambda: call(torch.zeros((6, RECORD_BYTES), **u8)),
                "rows not a multiple of 4",
            )
        _expect(
            ValueError,
            lambda: cake_fp4_allreduce(x, workspace),
            "bfloat16 partial on an NVFP4-input workspace",
        )
    _expect(TypeError, lambda: cake_fp4_allreduce(x, object()), "workspace type")
    spec = CakeFp4AllReduceFusedNormSpec(weight=torch.ones((HIDDEN,), **bf16))
    _expect(
        ValueError,
        lambda: cake_fp4_allreduce_rmsnorm(x, x.clone(), spec, workspace),
        "fused-norm call on an all-reduce workspace",
    )


def _allreduce_suite(input_format, world_size, rank, device, group):
    key = ("allreduce", world_size)
    seed = SEEDS["allreduce"]
    workspace = CakeFp4AllReduceWorkspace(
        group, family="allreduce", input_format=input_format, device=device
    )
    try:
        _check_workspace(
            workspace,
            "allreduce",
            input_format,
            world_size,
            rank,
            device,
            DEFAULT_CAPACITY,
        )
        tiers = ROWS[key] if world_size == 2 else (*ROWS[key], bv0_rows(workspace.grid))
        _allreduce_cases(
            workspace, tiers, input_format, world_size, rank, device, group, seed
        )
        _back_to_back(workspace, input_format, world_size, rank, device, group, seed)
        _alternating_rows(
            workspace, input_format, world_size, rank, device, group, seed
        )
        _graph_replay(workspace, input_format, world_size, rank, device, group, seed)
        _negative_control(
            workspace, input_format, world_size, rank, device, group, seed
        )
        _validation(workspace, input_format, world_size, device)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
        workspace.destroy()  # a second destroy is a no-op
    after = _allreduce_case(input_format, 8, seed, world_size, device)
    _expect(RuntimeError, lambda: after.launch(workspace, rank), "call after destroy")
    # The odd row count under a larger capacity: the window is sized for 4096 rows, the call uses part of it.
    with CakeFp4AllReduceWorkspace(
        group,
        family="allreduce",
        input_format=input_format,
        capacity=LARGER_CAPACITY,
        device=device,
    ) as larger:
        _check_workspace(
            larger, "allreduce", input_format, world_size, rank, device, LARGER_CAPACITY
        )
        _allreduce_cases(
            larger,
            (ODD_ROWS[key],),
            input_format,
            world_size,
            rank,
            device,
            group,
            seed,
        )
        over = _allreduce_case(
            input_format, LARGER_CAPACITY + 4, seed, world_size, device
        )
        _expect(
            ValueError, lambda: over.launch(larger, rank), "rows above the capacity"
        )
        dist.barrier(group=group)


# ---------------------------------------------------------------------------
# fused residual + Gemma RMSNorm family (world size 2)
# ---------------------------------------------------------------------------


@dataclass
class NormCase:
    """Every rank's input of one fused launch, the rank-identical residual and weight, and the reference."""

    rows: int
    inputs: list[torch.Tensor]
    global_scales: Optional[torch.Tensor]
    residual: torch.Tensor
    weight: torch.Tensor
    reference: ref.FusedNormReference

    def rebuild(self, *, eps: float, weight_bias: float) -> ref.FusedNormReference:
        if self.global_scales is None:
            return ref.reference_fused_rmsnorm(
                self.inputs,
                self.residual,
                self.weight,
                eps=eps,
                weight_bias=weight_bias,
                ss_order=self.reference.ss_order,
            )
        return ref.reference_fused_rmsnorm(
            residual_bf16=self.residual,
            weight_bf16=self.weight,
            records=self.inputs,
            global_scales=self.global_scales,
            eps=eps,
            weight_bias=weight_bias,
            ss_order=self.reference.ss_order,
        )

    def launch(
        self, workspace, rank, spec, **kwargs
    ) -> CakeFp4AllReduceFusedNormResult:
        if self.global_scales is None:
            return cake_fp4_allreduce_rmsnorm(
                self.inputs[rank], self.residual, spec, workspace, **kwargs
            )
        return cake_fp4_allreduce_rmsnorm_nvfp4(
            self.inputs[rank],
            self.global_scales,
            self.residual,
            spec,
            workspace,
            **kwargs,
        )


def _ss_order(workspace, rows: int, output: CakeFp4AllReduceFusedNormOutput) -> str:
    """The sum-of-squares order of the fused-norm program a launch with ``rows`` rows and ``output`` runs on
    this workspace's target (its registry record): the reference must sum in that order."""
    return PROGRAMS[workspace.program(rows, output.value)]["ss_order"]


def _norm_case(
    input_format: str,
    rows: int,
    seed: int,
    device: torch.device,
    *,
    ss_order: str,
    eps: float = ref.EPS,
    weight_bias: float = ref.WEIGHT_BIAS,
) -> NormCase:
    """The inputs of one fused launch and its reference, summing the squares in ``ss_order`` (the order of the
    program the launch runs on this target, :func:`_ss_order`)."""
    xs = ref.partials(rows, seed, 2, device)
    residual = ref.residual(rows, seed, device)
    weight = ref.weight(seed, device)
    if input_format == "bf16":
        reference = ref.reference_fused_rmsnorm(
            xs, residual, weight, eps=eps, weight_bias=weight_bias, ss_order=ss_order
        )
        return NormCase(rows, xs, None, residual, weight, reference)
    records, global_scales = ref.records_of(xs)
    reference = ref.reference_fused_rmsnorm(
        residual_bf16=residual,
        weight_bf16=weight,
        records=records,
        global_scales=global_scales,
        eps=eps,
        weight_bias=weight_bias,
        ss_order=ss_order,
    )
    return NormCase(rows, records, global_scales, residual, weight, reference)


def _spec_and_expected(case: NormCase, output: CakeFp4AllReduceFusedNormOutput):
    """The call's spec and the expected quantised outputs (``None`` for the bfloat16 output)."""
    y = case.reference.y
    if output == CakeFp4AllReduceFusedNormOutput.BF16:
        return CakeFp4AllReduceFusedNormSpec(weight=case.weight), None
    if output == CakeFp4AllReduceFusedNormOutput.NVFP4:
        global_scale = ref.global_scale_out(y)
        spec = CakeFp4AllReduceFusedNormSpec(
            weight=case.weight, output=output, global_scale=global_scale
        )
        values, scales = ref.quant_nvfp4(y, global_scale)
        return spec, (values, scales, ref.nvfp4_tie_mask(y, global_scale))
    spec = CakeFp4AllReduceFusedNormSpec(weight=case.weight, output=output)
    values, scales = ref.quant_mxfp8(y)
    return spec, (values, scales, None)


def _check_norm_result(
    result: CakeFp4AllReduceFusedNormResult,
    case: NormCase,
    output: CakeFp4AllReduceFusedNormOutput,
    expected,
    label: str,
    group: dist.ProcessGroup,
) -> None:
    assert isinstance(result, CakeFp4AllReduceFusedNormResult)
    assert result.output == output
    _check_equal(
        result.residual_out, case.reference.residual_out, f"{label}: residual_out"
    )
    _check_identical_across_ranks(result.residual_out, group, f"{label}: residual_out")
    if output == CakeFp4AllReduceFusedNormOutput.BF16:
        assert result.values is None and result.scales is None
        _check_y(result.out, case.reference.y, f"{label}: y")
        _check_identical_across_ranks(result.out, group, f"{label}: y")
        return
    assert result.out is None
    exp_values, exp_scales, tie_mask = expected
    per_row = (
        ref.NVFP4_SCALES_PER_ROW
        if output == CakeFp4AllReduceFusedNormOutput.NVFP4
        else ref.MXFP8_SCALES_PER_ROW
    )
    values = result.values.view(torch.uint8)
    assert tuple(values.shape) == tuple(exp_values.shape), (
        tuple(values.shape),
        tuple(exp_values.shape),
    )
    assert result.scales.dtype == torch.uint8
    assert result.scales.numel() == ref.scale_bytes(case.rows, per_row)
    scales = result.scales.reshape(-1)
    if tie_mask is not None:
        values = ref.canonical(values, tie_mask)
        exp_values = ref.canonical(exp_values, tie_mask)
    _check_mismatch_fraction(
        values, exp_values, ref.QUANTISED_MISMATCH_ALLOWANCE, f"{label}: values"
    )
    _check_mismatch_fraction(
        scales, exp_scales, ref.QUANTISED_MISMATCH_ALLOWANCE, f"{label}: scales"
    )
    # The consumer's block-order view of the scales agrees as well.
    _check_mismatch_fraction(
        ref.unswizzle_scales(scales, case.rows, per_row),
        ref.unswizzle_scales(exp_scales, case.rows, per_row),
        ref.QUANTISED_MISMATCH_ALLOWANCE,
        f"{label}: scales in block order",
    )
    _check_identical_across_ranks(
        result.values.view(torch.uint8), group, f"{label}: values"
    )
    _check_identical_across_ranks(scales, group, f"{label}: scales")


def _given_buffers(previous: CakeFp4AllReduceFusedNormResult) -> dict:
    """Caller-provided output buffers shaped like the outputs of a previous call (the scales zeroed: padding rows stay untouched)."""
    buffers = {"residual_out": torch.empty_like(previous.residual_out)}
    if previous.out is not None:
        buffers["out"] = torch.empty_like(previous.out)
    if previous.values is not None:
        buffers["values"] = torch.empty_like(previous.values)
    if previous.scales is not None:
        buffers["scales"] = torch.zeros_like(previous.scales)
    return buffers


def _rmsnorm_cases(
    workspace, output, rows_list, input_format, rank, device, group, seed
):
    """Two launches per row count and output format on one workspace; the second into caller-provided buffers."""
    for rows in rows_list:
        program = workspace.program(rows, output.value)
        assert program.endswith("_pf") == loader.select_prefetch(
            rows, workspace.grid
        ), (
            program,
            rows,
            workspace.grid,
        )
        ss_order = PROGRAMS[program]["ss_order"]
        assert ("_sop" in program) == (ss_order == "perm"), program
        previous = None
        for launch in range(LAUNCHES_PER_CASE):
            case = _norm_case(
                input_format,
                rows,
                ref.launch_seed(seed + 17 * rows + 1000 * len(output.value), launch),
                device,
                ss_order=ss_order,
            )
            spec, expected = _spec_and_expected(case, output)
            buffers = _given_buffers(previous) if previous is not None else {}
            result = case.launch(
                workspace, rank, spec, launch_with_pdl=launch % 2 == 0, **buffers
            )
            torch.cuda.synchronize(device)
            for name, buffer in buffers.items():
                assert getattr(result, name).data_ptr() == buffer.data_ptr(), name
            label = (
                f"rmsnorm {input_format}->{output.value} rows={rows} "
                f"cap={workspace.capacity} launch={launch}"
            )
            _check_norm_result(result, case, output, expected, label, group)
            previous = result


def _rmsnorm_runtime_scalars(workspace, input_format, rank, device, group, seed):
    """``eps`` and ``weight_bias`` are runtime scalars: a visible value changes ``y`` as the reference predicts."""
    rows = MID_ROWS
    # mean(z^2) is about 80 for these inputs (the outliers): eps = 1e-5 is invisible in bfloat16 and
    # checks the plumbing, eps = 100 and weight_bias = 0 move y far outside the tolerance.
    ss_order = _ss_order(workspace, rows, CakeFp4AllReduceFusedNormOutput.BF16)
    for eps, weight_bias in ((1e-5, 1.0), (100.0, 1.0), (1e-6, 0.0)):
        case = _norm_case(
            input_format,
            rows,
            seed + 555,
            device,
            ss_order=ss_order,
            eps=eps,
            weight_bias=weight_bias,
        )
        spec = CakeFp4AllReduceFusedNormSpec(
            weight=case.weight, eps=eps, weight_bias=weight_bias
        )
        result = case.launch(workspace, rank, spec)
        torch.cuda.synchronize(device)
        label = f"rmsnorm {input_format} eps={eps} weight_bias={weight_bias}"
        _check_equal(
            result.residual_out, case.reference.residual_out, f"{label}: residual_out"
        )
        _check_y(result.out, case.reference.y, f"{label}: y")
        if (eps, weight_bias) in ((100.0, 1.0), (1e-6, 0.0)):
            default = case.rebuild(eps=ref.EPS, weight_bias=ref.WEIGHT_BIAS)
            if ref.tolerance_violations(result.out, default.y) == 0:
                raise AssertionError(
                    f"{label}: y stayed inside the tolerance of the default-scalar reference"
                )


def _rmsnorm_negative_controls(workspace, input_format, rank, device, group, seed):
    """Stale expectations must fail: a reference with weight bias 1 + 2^-7, or (NVFP4 input) a wrong static scale."""
    rows = MID_ROWS
    ss_order = _ss_order(workspace, rows, CakeFp4AllReduceFusedNormOutput.BF16)
    case = _norm_case(input_format, rows, seed + 99, device, ss_order=ss_order)
    spec = CakeFp4AllReduceFusedNormSpec(weight=case.weight)
    result = case.launch(workspace, rank, spec)
    torch.cuda.synchronize(device)
    _check_equal(
        result.residual_out,
        case.reference.residual_out,
        "negative control: residual_out",
    )
    _check_y(
        result.out, case.reference.y, "negative control: y against the right reference"
    )
    biased = case.rebuild(eps=ref.EPS, weight_bias=ref.NEGATIVE_CONTROL_WEIGHT_BIAS)
    _check_equal(
        result.residual_out,
        biased.residual_out,
        "negative control: biased residual_out",
    )
    if ref.tolerance_violations(result.out, biased.y) == 0:
        raise AssertionError(
            f"a reference with weight bias {ref.NEGATIVE_CONTROL_WEIGHT_BIAS} stayed inside the y tolerance"
        )
    if input_format == "nvfp4":
        scales = case.global_scales.clone()
        scales[1] *= 2.0
        wrong = NormCase(
            rows,
            case.inputs,
            scales,
            case.residual,
            case.weight,
            ref.reference_fused_rmsnorm(
                residual_bf16=case.residual,
                weight_bf16=case.weight,
                records=case.inputs,
                global_scales=scales,
                ss_order=ss_order,
            ),
        )
        result = wrong.launch(workspace, rank, spec)
        torch.cuda.synchronize(device)
        if ref.count_mismatches(result.residual_out, case.reference.residual_out) == 0:
            raise AssertionError("a wrong static scale did not change residual_out")
        if ref.tolerance_violations(result.out, case.reference.y) == 0:
            raise AssertionError("a wrong static scale kept y inside the tolerance")
        _check_equal(
            result.residual_out,
            wrong.reference.residual_out,
            "wrong scale: residual_out",
        )
        _check_y(
            result.out, wrong.reference.y, "wrong scale: y against its own reference"
        )


def _rmsnorm_back_to_back(workspace, input_format, rank, device, group, seed):
    """Four fused calls without host synchronisation into caller-provided buffers (rotation wrap-around)."""
    rows = MID_ROWS
    ss_order = _ss_order(workspace, rows, CakeFp4AllReduceFusedNormOutput.BF16)
    calls = []
    for call in range(BACK_TO_BACK_CALLS):
        case = _norm_case(
            input_format, rows, seed + 7777 * call, device, ss_order=ss_order
        )
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        residual_out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        calls.append((case, out, residual_out))
    for case, out, residual_out in calls:
        case.launch(
            workspace,
            rank,
            CakeFp4AllReduceFusedNormSpec(weight=case.weight),
            out=out,
            residual_out=residual_out,
        )
    torch.cuda.synchronize(device)
    for call, (case, out, residual_out) in enumerate(calls):
        _check_equal(
            residual_out,
            case.reference.residual_out,
            f"rmsnorm back-to-back call {call}: residual_out",
        )
        _check_y(out, case.reference.y, f"rmsnorm back-to-back call {call}: y")


def _rmsnorm_alternating_rows(workspace, input_format, rank, device, group, seed):
    key = ("rmsnorm", 2)
    sequence = (ROWS[key][-1], SMALL_ROWS[key], ROWS[key][-1], MID_ROWS)
    for call, rows in enumerate(sequence):
        case = _norm_case(
            input_format,
            rows,
            seed + 31 * call,
            device,
            ss_order=_ss_order(workspace, rows, CakeFp4AllReduceFusedNormOutput.BF16),
        )
        result = case.launch(
            workspace, rank, CakeFp4AllReduceFusedNormSpec(weight=case.weight)
        )
        torch.cuda.synchronize(device)
        _check_norm_result(
            result,
            case,
            CakeFp4AllReduceFusedNormOutput.BF16,
            None,
            f"rmsnorm alternating rows call {call} rows={rows}",
            group,
        )


def _rmsnorm_graph_replay(workspace, input_format, rank, device, group, seed):
    rows = MID_ROWS
    case = _norm_case(
        input_format,
        rows,
        seed + 4242,
        device,
        ss_order=_ss_order(workspace, rows, CakeFp4AllReduceFusedNormOutput.BF16),
    )
    spec = CakeFp4AllReduceFusedNormSpec(weight=case.weight)
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    residual_out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        case.launch(workspace, rank, spec, out=out, residual_out=residual_out)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    _check_y(out, case.reference.y, "rmsnorm warm-up call before capture: y")
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(GRAPH_CALLS):
            case.launch(workspace, rank, spec, out=out, residual_out=residual_out)
    for replay in range(GRAPH_REPLAYS):
        out.zero_()
        residual_out.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        _check_equal(
            residual_out,
            case.reference.residual_out,
            f"rmsnorm graph replay {replay}: residual_out",
        )
        _check_equal(out, first, f"rmsnorm graph replay {replay}: y (deterministic)")
    result = case.launch(workspace, rank, spec)
    torch.cuda.synchronize(device)
    _check_equal(result.out, first, "rmsnorm eager call after the replays: y")
    del graph


def _rmsnorm_validation(workspace, input_format, device):
    bf16 = dict(dtype=torch.bfloat16, device=device)
    rows = 64
    capacity = workspace.capacity
    x = torch.zeros((rows, HIDDEN), **bf16)
    records = torch.zeros((rows, RECORD_BYTES), dtype=torch.uint8, device=device)
    scales = torch.ones((2,), dtype=torch.float32, device=device)
    residual = torch.zeros((rows, HIDDEN), **bf16)
    weight = torch.zeros((HIDDEN,), **bf16)
    spec = CakeFp4AllReduceFusedNormSpec(weight=weight)
    nvfp4 = CakeFp4AllReduceFusedNormOutput.NVFP4
    mxfp8 = CakeFp4AllReduceFusedNormOutput.MXFP8

    if input_format == "bf16":

        def call(source=x, residual_in=residual, norm_spec=spec, **kwargs):
            return cake_fp4_allreduce_rmsnorm(
                source, residual_in, norm_spec, workspace, **kwargs
            )

        _expect(
            ValueError,
            lambda: call(
                torch.zeros((rows, HIDDEN), dtype=torch.float32, device=device)
            ),
            "float32 input",
        )
        _expect(
            ValueError,
            lambda: call(torch.zeros((rows, HIDDEN - 1), **bf16)),
            "hidden size",
        )
        _expect(
            ValueError,
            lambda: call(
                torch.zeros((capacity + 1, HIDDEN), **bf16),
                torch.zeros((capacity + 1, HIDDEN), **bf16),
            ),
            "rows above the capacity",
        )
        _expect(
            ValueError,
            lambda: cake_fp4_allreduce_rmsnorm_nvfp4(
                records, scales, residual, spec, workspace
            ),
            "NVFP4 records on a bfloat16-input workspace",
        )
    else:

        def call(
            source=records,
            residual_in=residual,
            norm_spec=spec,
            scale_tensor=scales,
            **kwargs,
        ):
            return cake_fp4_allreduce_rmsnorm_nvfp4(
                source, scale_tensor, residual_in, norm_spec, workspace, **kwargs
            )

        _expect(
            ValueError,
            lambda: call(
                torch.zeros((rows, RECORD_BYTES - 1), dtype=torch.uint8, device=device)
            ),
            "record width",
        )
        _expect(ValueError, lambda: call(scale_tensor=scales[:-1]), "scale count")
        _expect(
            ValueError,
            lambda: call(
                torch.zeros(
                    (capacity + 1, RECORD_BYTES), dtype=torch.uint8, device=device
                ),
                torch.zeros((capacity + 1, HIDDEN), **bf16),
            ),
            "rows above the capacity",
        )
        _expect(
            ValueError,
            lambda: cake_fp4_allreduce_rmsnorm(x, residual, spec, workspace),
            "bfloat16 partial on an NVFP4-input workspace",
        )
    _expect(
        ValueError,
        lambda: call(residual_in=torch.zeros((rows + 1, HIDDEN), **bf16)),
        "residual rows",
    )
    _expect(
        ValueError,
        lambda: call(
            residual_in=torch.zeros((rows, HIDDEN), dtype=torch.float32, device=device)
        ),
        "float32 residual",
    )
    _expect(
        ValueError,
        lambda: call(residual_out=residual),
        "residual_out aliasing residual",
    )
    _expect(
        ValueError,
        lambda: call(residual_out=torch.zeros((rows + 1, HIDDEN), **bf16)),
        "residual_out rows",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(
                weight=torch.zeros((HIDDEN,), dtype=torch.float32, device=device)
            )
        ),
        "float32 weight",
    )
    _expect(
        ValueError,
        lambda: call(norm_spec=CakeFp4AllReduceFusedNormSpec(weight=weight[:-1])),
        "weight length",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(weight=weight, output=nvfp4)
        ),
        "NVFP4 output without a global scale",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(
                weight=weight,
                output=nvfp4,
                global_scale=torch.ones((1,), dtype=torch.float64, device=device),
            )
        ),
        "float64 global scale",
    )
    _expect(
        ValueError,
        lambda: call(
            values=torch.zeros(
                (rows, ref.NVFP4_VALUE_BYTES), dtype=torch.uint8, device=device
            )
        ),
        "values buffer with the bfloat16 output",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(weight=weight, output=mxfp8),
            values=torch.zeros((rows, HIDDEN - 1), dtype=torch.uint8, device=device),
        ),
        "values buffer shape",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(weight=weight, output=mxfp8),
            scales=torch.zeros((16,), dtype=torch.uint8, device=device),
        ),
        "scales buffer too small",
    )
    _expect(
        ValueError,
        lambda: call(
            norm_spec=CakeFp4AllReduceFusedNormSpec(weight=weight, output=mxfp8),
            out=torch.zeros((rows, HIDDEN), **bf16),
        ),
        "out buffer with a quantised output",
    )
    _expect(TypeError, lambda: call(norm_spec=object()), "spec type")
    _expect(
        TypeError,
        lambda: cake_fp4_allreduce_rmsnorm(x, residual, spec, object()),
        "workspace type",
    )
    _expect(
        ValueError,
        lambda: cake_fp4_allreduce(x, workspace),
        "all-reduce call on a fused-norm workspace",
    )


def _rmsnorm_suite(input_format, rank, device, group):
    key = ("rmsnorm", 2)
    seed = SEEDS["rmsnorm"]
    workspace = CakeFp4AllReduceWorkspace(
        group, family="rmsnorm", input_format=input_format, device=device
    )
    try:
        _check_workspace(
            workspace, "rmsnorm", input_format, 2, rank, device, DEFAULT_CAPACITY
        )
        for output in OUTPUTS:
            _rmsnorm_cases(
                workspace, output, ROWS[key], input_format, rank, device, group, seed
            )
        _rmsnorm_runtime_scalars(workspace, input_format, rank, device, group, seed)
        _rmsnorm_negative_controls(workspace, input_format, rank, device, group, seed)
        _rmsnorm_back_to_back(workspace, input_format, rank, device, group, seed)
        _rmsnorm_alternating_rows(workspace, input_format, rank, device, group, seed)
        _rmsnorm_graph_replay(workspace, input_format, rank, device, group, seed)
        _rmsnorm_validation(workspace, input_format, device)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
        workspace.destroy()  # a second destroy is a no-op
    after = _norm_case(
        input_format,
        8,
        seed,
        device,
        ss_order=_ss_order(workspace, 8, CakeFp4AllReduceFusedNormOutput.BF16),
    )
    _expect(
        RuntimeError,
        lambda: after.launch(
            workspace, rank, CakeFp4AllReduceFusedNormSpec(weight=after.weight)
        ),
        "call after destroy",
    )
    with CakeFp4AllReduceWorkspace(
        group,
        family="rmsnorm",
        input_format=input_format,
        capacity=LARGER_CAPACITY,
        device=device,
    ) as larger:
        _check_workspace(
            larger, "rmsnorm", input_format, 2, rank, device, LARGER_CAPACITY
        )
        for output in OUTPUTS[:2]:
            _rmsnorm_cases(
                larger,
                output,
                (ODD_ROWS[key],),
                input_format,
                rank,
                device,
                group,
                seed,
            )
        dist.barrier(group=group)


# ---------------------------------------------------------------------------
# process harness
# ---------------------------------------------------------------------------


def _worker(
    family: str, input_format: str, world_size: int, rank: int, port: int
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD
        if family == "allreduce":
            _allreduce_suite(input_format, world_size, rank, device, group)
        else:
            _rmsnorm_suite(input_format, rank, device, group)
    finally:
        dist.destroy_process_group()


def _skip_unless_supported(world_size: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    capabilities = {
        tuple(torch.cuda.get_device_capability(index)) for index in range(world_size)
    }
    if len(capabilities) != 1:
        pytest.skip("test requires homogeneous CUDA devices")
    (capability,) = capabilities
    if loader.supported_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not a build target of the Cake NVFP4-transport "
            f"all-reduce (targets: {loader.target_capabilities()})"
        )


def _run_workers(family: str, input_format: str, world_size: int) -> None:
    _skip_unless_supported(world_size)
    port = _open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker, args=(family, input_format, world_size, rank, port)
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    try:
        for rank, process in enumerate(processes):
            process.join(timeout=max(1.0, deadline - time.monotonic()))
            if process.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {JOIN_TIMEOUT_S}s"
                )
            assert process.exitcode == 0, f"rank {rank} exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=CLEANUP_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=CLEANUP_TIMEOUT_S)


@pytest.mark.parametrize(
    ("family", "input_format", "world_size"),
    CASES,
    ids=[f"{family}-{fmt}-tp{world}" for family, fmt, world in CASES],
)
def test_cake_fp4_allreduce(family: str, input_format: str, world_size: int) -> None:
    _run_workers(family, input_format, world_size)
