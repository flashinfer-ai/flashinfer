"""Benchmark the NVFP4-transport all-reduce (and its fused residual + RMSNorm variant) against NCCL.

Launch with ``torchrun --nproc-per-node {2,4} benchmarks/comm/bench_fp4_allreduce.py``
(``--family rmsnorm`` needs two ranks).

Two arms, interleaved in the order of ``--order`` (default ``fp4,nccl,nccl,fp4`` so
same-session drift is visible):

* ``fp4``: ``fp4_allreduce`` (``fp4_allreduce_nvfp4`` with ``--input-format nvfp4``) or,
  with ``--family rmsnorm``, ``fp4_allreduce_rmsnorm`` / ``fp4_allreduce_rmsnorm_nvfp4``
  with the output of ``--output``; a CUDA graph of ``--graph-calls`` back-to-back
  calls (default 10: the three-slot rotation wraps around);
* ``nccl``: ``torch.distributed.all_reduce`` of the same bfloat16 tensor, followed for
  ``--family rmsnorm`` by ``gemma_fused_add_rmsnorm`` and, for the quantised outputs,
  by ``fp4_quantize`` / ``mxfp8_quantize``; the same number of calls per graph
  (captured when the NCCL build supports graph capture, eager otherwise; the mode
  is recorded).

Every timed iteration replays one graph.  CUPTI records the kernel spans of the
iteration on every rank; the per-iteration time is the maximum over ranks of
``last kernel end - first kernel start`` and the row value is the median over
iterations divided by ``--graph-calls``.  Every leg is bracketed by a
correctness check: the ``fp4`` arm must match the float32 codec reference
(bitwise for the all-reduce, bitwise residual and the 1e-2 tolerance on ``y`` for
the fused norm, bitwise quantised bytes against the quantisers of the kernel's
own ``y``), the ``nccl`` arm must be close to the exact float32 result.  Published
numbers of another implementation can be passed as ``--published`` (JSON
``{"tp2/bf16/4096": microseconds_per_call, ...}``) and are printed next to the
measured rows; none are embedded here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from flashinfer.comm.fp4_allreduce import (
    Fp4AllReduceFusedNormOutput,
    Fp4AllReduceFusedNormSpec,
    Fp4AllReduceWorkspace,
    fp4_allreduce,
    fp4_allreduce_nvfp4,
    fp4_allreduce_rmsnorm,
    fp4_allreduce_rmsnorm_nvfp4,
)
from flashinfer.jit.fp4_allreduce import HIDDEN
from flashinfer.testing.utils import bench_gpu_time
from tests.comm import fp4_allreduce_reference as codec

_BOOL = {"false": False, "true": True}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", default="1024,2048,4096,8192,16384", help="comma-separated row counts"
    )
    parser.add_argument(
        "--family", choices=("allreduce", "rmsnorm"), default="allreduce"
    )
    parser.add_argument("--input-format", choices=("bf16", "nvfp4"), default="bf16")
    parser.add_argument(
        "--output",
        choices=("bf16", "nvfp4", "mxfp8"),
        default="bf16",
        help="output of the fused norm (--family rmsnorm; bfloat16 input only for the quantised outputs)",
    )
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=200)
    parser.add_argument(
        "--order", default="fp4,nccl,nccl,fp4", help="comma-separated legs"
    )
    parser.add_argument("--launch-with-pdl", choices=tuple(_BOOL), default="true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--published",
        type=Path,
        default=None,
        help="JSON of published per-call microseconds",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write the JSON report here (rank 0)",
    )
    return parser.parse_args()


def _gather(tensor: torch.Tensor, group) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, tensor.contiguous(), group=group)
    return peers


def _exact_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        total = codec.f32_add_rn(total, partial.to(torch.float32))
    return total


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.to(torch.float64) - b.to(torch.float64)).norm()
    return float(diff / max(float(b.to(torch.float64).norm()), 1e-30))


def _time(
    calls: Callable[[], None],
    *,
    graph: bool,
    dry_run_iters: int,
    repeat_iters: int,
    require_graph: bool,
):
    """Per-iteration spans (ms), already reduced to the maximum over ranks, and the execution mode."""
    runner = calls
    mode = "eager"
    if graph:
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                calls()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=side):
                calls()
            runner = captured.replay
            mode = "graph"
        except Exception as exc:  # noqa: BLE001 - NCCL builds without graph capture fall back to eager
            if require_graph:
                raise
            torch.cuda.synchronize()
            mode = f"eager (capture failed: {type(exc).__name__})"
    samples = bench_gpu_time(
        runner,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=False,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        aggregate_op=max,
    )
    return [float(sample) for sample in samples], mode


def _summary(samples_ms: list[float], graph_calls: int) -> dict:
    per_call_us = sorted(1000.0 * sample / graph_calls for sample in samples_ms)
    n = len(per_call_us)
    return {
        "median_us_per_call": statistics.median(per_call_us),
        "p10_us_per_call": per_call_us[int(0.10 * (n - 1))],
        "p90_us_per_call": per_call_us[int(0.90 * (n - 1))],
        "iterations": n,
    }


class _AllReduceRow:
    """One row of the all-reduce family: the fp4 call, the NCCL call and their correctness brackets."""

    def __init__(self, args, rows, rank, group, device, workspace, launch_with_pdl):
        self.rows = rows
        self.x = codec.generate_rank_input(rows, rank, args.seed + 17 * rows, device)
        partials = _gather(self.x, group)
        self.exact = _exact_sum(partials)
        self.out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        self.nccl_buffer = torch.empty_like(self.x)
        self.group = group
        self.device = device
        self.workspace = workspace
        self.input_format = args.input_format
        if args.input_format == "bf16":
            self.expected = codec.reference_allreduce(partials)

            def fp4_call():
                fp4_allreduce(
                    self.x, workspace, out=self.out, launch_with_pdl=launch_with_pdl
                )

        else:
            scale = codec.tensor_global_scale(self.x)
            records = codec.encode_nvfp4_records(self.x, scale)
            scales = [
                float(value.item())
                for value in _gather(
                    torch.tensor([scale], dtype=torch.float32, device=device), group
                )
            ]
            scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
            self.expected = codec.reference_allreduce_nvfp4(
                _gather(records, group), scales
            )

            def fp4_call():
                fp4_allreduce_nvfp4(
                    records,
                    scale_tensor,
                    workspace,
                    out=self.out,
                    launch_with_pdl=launch_with_pdl,
                )

        self.fp4_call = fp4_call
        self.kernel = workspace.kernel_name(args.input_format, rows)
        self.reference_arm = "torch.distributed.all_reduce"

    def nccl_call(self):
        self.nccl_buffer.copy_(self.x)
        dist.all_reduce(self.nccl_buffer, group=self.group)

    def check(self, leg: str, phase: str) -> None:
        torch.cuda.synchronize(self.device)
        if leg == "fp4":
            self.out.zero_()
            self.fp4_call()
            torch.cuda.synchronize(self.device)
            if not torch.equal(self.out, self.expected):
                differing = int((self.out != self.expected).sum().item())
                raise AssertionError(
                    f"rows={self.rows} {leg}/{phase}: {differing} elements differ from the codec reference"
                )
        else:
            self.nccl_call()
            torch.cuda.synchronize(self.device)
            if not torch.allclose(
                self.nccl_buffer.float(), self.exact, rtol=1 / 64, atol=0.5
            ):
                raise AssertionError(
                    f"rows={self.rows} {leg}/{phase}: NCCL result is not close to the exact sum"
                )


class _FusedNormRow:
    """One row of the fused-norm family: the fp4 call, the NCCL + RMSNorm (+ quantiser) call and their brackets."""

    def __init__(self, args, rows, rank, group, device, workspace, launch_with_pdl):
        from flashinfer.norm import gemma_fused_add_rmsnorm

        self.rows = rows
        self.group = group
        self.device = device
        self.workspace = workspace
        self.output = Fp4AllReduceFusedNormOutput(args.output)
        seed = args.seed + 17 * rows
        self.x = codec.generate_rank_input(rows, rank, seed, device)
        self.residual = codec.generate_residual(rows, seed, device)
        self.weight = codec.generate_weight(seed, device)
        partials = _gather(self.x, group)
        self.exact_z = codec.f32_add_rn(
            _exact_sum(partials), self.residual.to(torch.float32)
        )
        self.residual_out = torch.empty(
            (rows, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        self.values = self.scales = None
        self.nccl_buffer = torch.empty_like(self.x)
        self.nccl_residual = torch.empty_like(self.residual)
        self.quantise = None
        if args.input_format == "bf16":
            self.ref = codec.reference_fused_rmsnorm(
                partials, self.residual, self.weight
            )
        else:
            if self.output != Fp4AllReduceFusedNormOutput.BF16:
                raise SystemExit(
                    "the quantised outputs are delivered for bfloat16 input only"
                )
            scale = codec.tensor_global_scale(self.x)
            records = codec.encode_nvfp4_records(self.x, scale)
            scales = [
                float(value.item())
                for value in _gather(
                    torch.tensor([scale], dtype=torch.float32, device=device), group
                )
            ]
            scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
            self.ref = codec.reference_fused_rmsnorm(
                residual_bf16=self.residual,
                weight_bf16=self.weight,
                records_by_rank=_gather(records, group),
                global_scales=scales,
            )
        spec_bf16 = Fp4AllReduceFusedNormSpec(weight=self.weight)
        if self.output == Fp4AllReduceFusedNormOutput.BF16:
            self.spec = spec_bf16
        else:
            # The quantised programs quantise the same bf16(y) the bfloat16 program stores: probe it once.
            probe = fp4_allreduce_rmsnorm(self.x, self.residual, spec_bf16, workspace)
            torch.cuda.synchronize(device)
            self.probe_y = probe.out.clone()
            if self.output == Fp4AllReduceFusedNormOutput.NVFP4:
                from flashinfer import fp4_quantize

                self.global_scale = codec.output_global_scale(self.probe_y)
                self.spec = Fp4AllReduceFusedNormSpec(
                    weight=self.weight,
                    output=self.output,
                    global_scale=self.global_scale,
                )
                self.expected_values, self.expected_scales = (
                    codec.nvfp4_output_reference(self.probe_y, self.global_scale)
                )

                def quantise(y):
                    fp4_quantize(
                        y,
                        global_scale=self.global_scale,
                        sf_vec_size=16,
                        sf_use_ue8m0=False,
                        is_sf_swizzled_layout=True,
                        is_sf_8x4_layout=False,
                        is_global_scale_inversed=False,
                        backend="cuda",
                        nvfp4_4over6=None,
                    )

            else:
                from flashinfer import mxfp8_quantize

                self.spec = Fp4AllReduceFusedNormSpec(
                    weight=self.weight, output=self.output
                )
                self.expected_values, self.expected_scales = (
                    codec.mxfp8_output_reference(self.probe_y)
                )

                def quantise(y):
                    mxfp8_quantize(
                        y, is_sf_swizzled_layout=True, alignment=32, backend="cuda"
                    )

            self.quantise = quantise
            first = fp4_allreduce_rmsnorm(self.x, self.residual, self.spec, workspace)
            torch.cuda.synchronize(device)
            self.values, self.scales = first.values, first.scales
        if args.input_format == "bf16":

            def fp4_call():
                fp4_allreduce_rmsnorm(
                    self.x,
                    self.residual,
                    self.spec,
                    workspace,
                    out=self.out
                    if self.output == Fp4AllReduceFusedNormOutput.BF16
                    else None,
                    values=self.values,
                    scales=self.scales,
                    residual_out=self.residual_out,
                    launch_with_pdl=launch_with_pdl,
                )

        else:

            def fp4_call():
                fp4_allreduce_rmsnorm_nvfp4(
                    records,
                    scale_tensor,
                    self.residual,
                    self.spec,
                    workspace,
                    out=self.out,
                    residual_out=self.residual_out,
                    launch_with_pdl=launch_with_pdl,
                )

        self.fp4_call = fp4_call
        self.gemma = gemma_fused_add_rmsnorm
        self.kernel = workspace.kernel_name(args.input_format, rows, self.output.value)
        self.reference_arm = (
            "torch.distributed.all_reduce + gemma_fused_add_rmsnorm"
            + ("" if self.quantise is None else f" + {args.output} quantiser")
        )

    def nccl_call(self):
        self.nccl_buffer.copy_(self.x)
        self.nccl_residual.copy_(self.residual)
        dist.all_reduce(self.nccl_buffer, group=self.group)
        self.gemma(self.nccl_buffer, self.nccl_residual, self.weight, codec.NORM_EPS)
        if self.quantise is not None:
            self.quantise(self.nccl_buffer)

    def check(self, leg: str, phase: str) -> None:
        torch.cuda.synchronize(self.device)
        label = f"rows={self.rows} {leg}/{phase}"
        if leg == "fp4":
            self.residual_out.zero_()
            self.out.zero_()
            if self.values is not None:
                self.values.view(torch.uint8).zero_()
                self.scales.zero_()
            self.fp4_call()
            torch.cuda.synchronize(self.device)
            if not torch.equal(self.residual_out, self.ref.residual_out):
                raise AssertionError(
                    f"{label}: residual_out differs from the reference"
                )
            if self.output == Fp4AllReduceFusedNormOutput.BF16:
                report = codec.check_normalised_output(self.out, self.ref)
                if not report["passed"]:
                    raise AssertionError(
                        f"{label}: y leaves the 1e-2 tolerance on {report['violations']} elements"
                    )
            else:
                if not torch.equal(self.values.view(torch.uint8), self.expected_values):
                    raise AssertionError(
                        f"{label}: quantised values differ from the quantiser of y"
                    )
                if not torch.equal(self.scales, self.expected_scales):
                    raise AssertionError(
                        f"{label}: quantised scales differ from the quantiser of y"
                    )
        else:
            self.nccl_call()
            torch.cuda.synchronize(self.device)
            if not torch.allclose(
                self.nccl_residual.float(), self.exact_z, rtol=1 / 64, atol=0.5
            ):
                raise AssertionError(
                    f"{label}: NCCL residual is not close to the exact sum"
                )
            # The NCCL arm normalises the bfloat16 NCCL sum; it is only compared loosely to the codec reference.
            if _rel_l2(self.nccl_buffer, self.ref.y_bf16) > 1 / 8:
                raise AssertionError(
                    f"{label}: the NCCL + RMSNorm result is far from the reference y"
                )


def main() -> None:
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    launch_with_pdl = _BOOL[args.launch_with_pdl]
    legs = [leg.strip() for leg in args.order.split(",") if leg.strip()]
    if any(leg not in ("fp4", "nccl") for leg in legs):
        raise SystemExit(f"--order may only name fp4 and nccl legs, got {legs}")
    published = (
        json.loads(args.published.read_text()) if args.published is not None else {}
    )
    started = time.monotonic()
    workspace = Fp4AllReduceWorkspace(group, device=device, family=args.family)
    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    row_class = _FusedNormRow if args.family == "rmsnorm" else _AllReduceRow
    report_rows = []
    try:
        for rows in rows_list:
            case = row_class(
                args, rows, rank, group, device, workspace, launch_with_pdl
            )

            def fp4_calls(case=case):
                for _ in range(args.graph_calls):
                    case.fp4_call()

            def nccl_calls(case=case):
                for _ in range(args.graph_calls):
                    case.nccl_call()

            row = {
                "family": args.family,
                "world_size": world_size,
                "rows": rows,
                "input_format": args.input_format,
                "output": args.output if args.family == "rmsnorm" else "bf16",
                "launch_with_pdl": launch_with_pdl,
                "kernel": case.kernel,
                "variant": workspace.variant(rows),
                "reference_arm": case.reference_arm,
                "legs": [],
            }
            for leg in legs:
                dist.barrier(group=group)
                case.check(leg, "pre")
                dist.barrier(group=group)
                samples, mode = _time(
                    fp4_calls if leg == "fp4" else nccl_calls,
                    graph=True,
                    dry_run_iters=args.dry_run_iters,
                    repeat_iters=args.repeat_iters,
                    require_graph=leg == "fp4",
                )
                case.check(leg, "post")
                row["legs"].append(
                    {"arm": leg, "mode": mode, **_summary(samples, args.graph_calls)}
                )
            medians = {}
            for arm in ("fp4", "nccl"):
                values = [
                    leg["median_us_per_call"]
                    for leg in row["legs"]
                    if leg["arm"] == arm
                ]
                if values:
                    medians[arm] = statistics.fmean(values)
            row["median_us_per_call"] = medians
            if "fp4" in medians and "nccl" in medians:
                row["nccl_over_fp4"] = medians["nccl"] / medians["fp4"]
            key = f"tp{world_size}/{args.input_format}/{rows}"
            if args.family == "rmsnorm":
                key = f"tp{world_size}/rmsnorm/{args.input_format}/{args.output}/{rows}"
            if key in published:
                row["published_us_per_call"] = float(published[key])
                row["published_over_fp4"] = float(published[key]) / medians["fp4"]
            report_rows.append(row)
            if rank == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
    if rank == 0:
        report = {
            "family": args.family,
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(device),
            "arch": workspace.arch,
            "grid": workspace.grid,
            "torch": torch.__version__,
            "timing": {
                "method": "bench_gpu_time(enable_cupti=True) over graph replays",
                "graph_calls": args.graph_calls,
                "dry_run_iters": args.dry_run_iters,
                "repeat_iters": args.repeat_iters,
                "order": legs,
                "rollup": "per-iteration span, maximum over ranks, median over iterations, divided by graph_calls",
            },
            "rows": report_rows,
            "physical_runtime_seconds": time.monotonic() - started,
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
