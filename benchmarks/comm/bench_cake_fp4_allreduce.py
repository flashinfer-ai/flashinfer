"""Benchmark the Cake NVFP4-transport all-reduce (and its fused residual + RMSNorm form) against NCCL.

Launch with ``torchrun --nproc-per-node {2,4} benchmarks/comm/bench_cake_fp4_allreduce.py``
(``--family rmsnorm`` needs two ranks).

Two arms, interleaved in the order of ``--order`` (default ``cake,nccl,nccl,cake`` so
same-session drift is visible):

* ``cake``: ``cake_fp4_allreduce`` (``cake_fp4_allreduce_nvfp4`` with ``--input-format
  nvfp4``) or, with ``--family rmsnorm``, ``cake_fp4_allreduce_rmsnorm`` /
  ``cake_fp4_allreduce_rmsnorm_nvfp4`` with the output of ``--output``; a CUDA graph of
  ``--graph-calls`` back-to-back calls (default 10: the three-slot window rotation wraps
  around);
* ``nccl``: ``torch.distributed.all_reduce`` of the same bfloat16 tensor, followed for
  ``--family rmsnorm`` by ``gemma_fused_add_rmsnorm`` and, for the quantised outputs, by
  ``fp4_quantize`` / ``mxfp8_quantize``; the same number of calls per graph (captured when
  the NCCL build supports graph capture, eager otherwise; the mode is recorded).

Every timed iteration replays one graph between two CUDA events on every rank; the
per-iteration time is the maximum over ranks and the row value is the median over
iterations divided by ``--graph-calls``.  Every leg is bracketed by a sanity check: the
``cake`` arm must produce bitwise identical outputs on every rank and stay close to the
exact float32 result (the NVFP4 transport quantises, so the bound is loose); the ``nccl``
arm must be close to the exact float32 result.
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

from flashinfer.comm.cake_fp4_allreduce import (
    HIDDEN,
    RECORD_BYTES,
    CakeFp4AllReduceFusedNormOutput,
    CakeFp4AllReduceFusedNormSpec,
    CakeFp4AllReduceWorkspace,
    cake_fp4_allreduce,
    cake_fp4_allreduce_nvfp4,
    cake_fp4_allreduce_rmsnorm,
    cake_fp4_allreduce_rmsnorm_nvfp4,
)

_BOOL = {"false": False, "true": True}
# NVFP4 record scale: the row amax maps onto the largest E4M3 x E2M1 product (448 x 6).
_ROW_AMAX_TARGET = 2688.0
# Relative L2 distance the quantised transport may put between a result and the exact sum.
_CAKE_REL_L2_MAX = 0.25
_NCCL_REL_L2_MAX = 0.05


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
        help="output of the fused norm (--family rmsnorm)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="workspace row capacity (default: the largest --rows entry)",
    )
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=200)
    parser.add_argument(
        "--order", default="cake,nccl,nccl,cake", help="comma-separated legs"
    )
    parser.add_argument("--launch-with-pdl", choices=tuple(_BOOL), default="true")
    parser.add_argument("--seed", type=int, default=1000)
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


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.to(torch.float64) - b.to(torch.float64)).norm()
    return float(diff / max(float(b.to(torch.float64).norm()), 1e-30))


def _rank_identical(tensor: torch.Tensor, group) -> bool:
    """Whether every rank holds the same bytes in ``tensor``."""

    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    peers = _gather(flat, group)
    return all(torch.equal(flat, peer) for peer in peers)


def _generate(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (rows, HIDDEN), dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)


def _encode_records(
    x: torch.Tensor, group, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 records of ``x`` under a static per-tensor scale, and every rank's decode scale.

    ``fp4_quantize`` with the linear scale layout gives the 3072 packed value
    bytes and the 384 E4M3 scales of every row; the kernel decodes record
    ``r`` as ``e2m1(value) * e4m3(scale) * global_scales[r]`` with
    ``global_scales[r] = 1 / input_global_scale_r``.
    """

    from flashinfer import fp4_quantize

    amax = x.to(torch.float32).abs().amax().clamp_min(1e-30)
    input_global_scale = (_ROW_AMAX_TARGET / amax).reshape(1).to(torch.float32)
    values, scales = fp4_quantize(
        x,
        global_scale=input_global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    rows = int(x.shape[0])
    records = torch.cat(
        (
            values.view(torch.uint8).reshape(rows, HIDDEN // 2),
            scales.view(torch.uint8).reshape(rows, HIDDEN // 16),
        ),
        dim=1,
    ).contiguous()
    assert records.shape[1] == RECORD_BYTES
    decode_scale = (1.0 / input_global_scale).to(torch.float32)
    global_scales = torch.cat(_gather(decode_scale, group)).contiguous()
    return records, global_scales


def _time(
    calls: Callable[[], None],
    *,
    group,
    device: torch.device,
    warmup_iters: int,
    repeat_iters: int,
    require_graph: bool,
) -> tuple[list[float], str]:
    """Per-iteration spans (ms), reduced to the maximum over ranks, and the execution mode."""

    runner = calls
    mode = "eager"
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            calls()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize(device)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=side):
            calls()
        runner = captured.replay
        mode = "graph"
    except Exception as exc:  # noqa: BLE001 - NCCL builds without graph capture fall back to eager
        if require_graph:
            raise
        torch.cuda.synchronize(device)
        mode = f"eager (capture failed: {type(exc).__name__})"
    for _ in range(warmup_iters):
        runner()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    samples = torch.zeros(repeat_iters, dtype=torch.float64, device=device)
    for index in range(repeat_iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        stop.record()
        stop.synchronize()
        samples[index] = start.elapsed_time(stop)
    dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=group)
    return [float(sample) for sample in samples.tolist()], mode


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
    """One row of the all-reduce family: the cake call, the NCCL call and their sanity brackets."""

    def __init__(self, args, rows, rank, group, device, workspace, launch_with_pdl):
        self.rows = rows
        self.group = group
        self.device = device
        self.x = _generate(rows, args.seed + 17 * rows + 1000003 * rank, device)
        partials = _gather(self.x, group)
        self.exact = torch.zeros_like(self.x, dtype=torch.float32)
        for partial in partials:
            self.exact += partial.to(torch.float32)
        self.out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        self.nccl_buffer = torch.empty_like(self.x)
        if args.input_format == "bf16":

            def cake_call():
                cake_fp4_allreduce(
                    self.x, workspace, out=self.out, launch_with_pdl=launch_with_pdl
                )

        else:
            records, global_scales = _encode_records(self.x, group, device)

            def cake_call():
                cake_fp4_allreduce_nvfp4(
                    records,
                    global_scales,
                    workspace,
                    out=self.out,
                    launch_with_pdl=launch_with_pdl,
                )

        self.cake_call = cake_call
        self.program = workspace.program(rows)
        self.reference_arm = "torch.distributed.all_reduce"

    def nccl_call(self):
        self.nccl_buffer.copy_(self.x)
        dist.all_reduce(self.nccl_buffer, group=self.group)

    def check(self, leg: str, phase: str) -> None:
        torch.cuda.synchronize(self.device)
        label = f"rows={self.rows} {leg}/{phase}"
        if leg == "cake":
            self.out.zero_()
            self.cake_call()
            torch.cuda.synchronize(self.device)
            if not _rank_identical(self.out, self.group):
                raise AssertionError(f"{label}: the output differs between ranks")
            distance = _rel_l2(self.out, self.exact)
            if distance > _CAKE_REL_L2_MAX:
                raise AssertionError(
                    f"{label}: relative L2 distance {distance:.4f} to the exact sum"
                )
        else:
            self.nccl_call()
            torch.cuda.synchronize(self.device)
            distance = _rel_l2(self.nccl_buffer, self.exact)
            if distance > _NCCL_REL_L2_MAX:
                raise AssertionError(
                    f"{label}: NCCL relative L2 distance {distance:.4f} to the exact sum"
                )


class _FusedNormRow:
    """One row of the fused-norm family: the cake call, the NCCL + RMSNorm (+ quantiser) call and their brackets."""

    def __init__(self, args, rows, rank, group, device, workspace, launch_with_pdl):
        from flashinfer import gemma_fused_add_rmsnorm

        self.rows = rows
        self.group = group
        self.device = device
        self.output = CakeFp4AllReduceFusedNormOutput(args.output)
        seed = args.seed + 17 * rows
        self.x = _generate(rows, seed + 1000003 * rank, device)
        self.residual = _generate(rows, seed + 7, device)
        self.weight = (
            0.1 * torch.randn(HIDDEN, dtype=torch.float32, device=device)
        ).to(torch.bfloat16)
        partials = _gather(self.x, group)
        self.exact_z = self.residual.to(torch.float32)
        for partial in partials:
            self.exact_z += partial.to(torch.float32)
        self.residual_out = torch.empty(
            (rows, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        self.values = self.scales = None
        self.nccl_buffer = torch.empty_like(self.x)
        self.nccl_residual = torch.empty_like(self.residual)
        self.gemma = gemma_fused_add_rmsnorm
        self.eps = 1e-6
        # Exact float32 form of the normalised row, for the loose brackets.
        rinv = torch.rsqrt(self.exact_z.square().mean(dim=1, keepdim=True) + self.eps)
        self.exact_y = self.exact_z * rinv * (1.0 + self.weight.to(torch.float32))
        self.quantise = None
        records = global_scales = None
        if args.input_format == "nvfp4":
            records, global_scales = _encode_records(self.x, group, device)
        if self.output == CakeFp4AllReduceFusedNormOutput.BF16:
            self.spec = CakeFp4AllReduceFusedNormSpec(weight=self.weight, eps=self.eps)
        elif self.output == CakeFp4AllReduceFusedNormOutput.NVFP4:
            from flashinfer import fp4_quantize

            amax = self.exact_y.abs().amax().clamp_min(1e-30)
            self.global_scale = (_ROW_AMAX_TARGET / amax).reshape(1).to(torch.float32)
            self.spec = CakeFp4AllReduceFusedNormSpec(
                weight=self.weight,
                eps=self.eps,
                output=self.output,
                global_scale=self.global_scale,
            )

            def quantise(y):
                fp4_quantize(
                    y,
                    global_scale=self.global_scale,
                    sf_vec_size=16,
                    sf_use_ue8m0=False,
                    is_sf_swizzled_layout=True,
                )

            self.quantise = quantise
        else:
            from flashinfer import mxfp8_quantize

            self.spec = CakeFp4AllReduceFusedNormSpec(
                weight=self.weight, eps=self.eps, output=self.output
            )

            def quantise(y):
                mxfp8_quantize(y, is_sf_swizzled_layout=True, alignment=32)

            self.quantise = quantise
        if self.output != CakeFp4AllReduceFusedNormOutput.BF16:
            # Allocate the quantised output buffers once, through the API.
            if args.input_format == "bf16":
                first = cake_fp4_allreduce_rmsnorm(
                    self.x, self.residual, self.spec, workspace
                )
            else:
                first = cake_fp4_allreduce_rmsnorm_nvfp4(
                    records, global_scales, self.residual, self.spec, workspace
                )
            torch.cuda.synchronize(device)
            self.values, self.scales = first.values, first.scales
        if args.input_format == "bf16":

            def cake_call():
                cake_fp4_allreduce_rmsnorm(
                    self.x,
                    self.residual,
                    self.spec,
                    workspace,
                    out=self.out
                    if self.output == CakeFp4AllReduceFusedNormOutput.BF16
                    else None,
                    values=self.values,
                    scales=self.scales,
                    residual_out=self.residual_out,
                    launch_with_pdl=launch_with_pdl,
                )

        else:

            def cake_call():
                cake_fp4_allreduce_rmsnorm_nvfp4(
                    records,
                    global_scales,
                    self.residual,
                    self.spec,
                    workspace,
                    out=self.out
                    if self.output == CakeFp4AllReduceFusedNormOutput.BF16
                    else None,
                    values=self.values,
                    scales=self.scales,
                    residual_out=self.residual_out,
                    launch_with_pdl=launch_with_pdl,
                )

        self.cake_call = cake_call
        self.program = workspace.program(rows, self.output.value)
        self.reference_arm = (
            "torch.distributed.all_reduce + gemma_fused_add_rmsnorm"
            + ("" if self.quantise is None else f" + {args.output} quantiser")
        )

    def nccl_call(self):
        self.nccl_buffer.copy_(self.x)
        self.nccl_residual.copy_(self.residual)
        dist.all_reduce(self.nccl_buffer, group=self.group)
        self.gemma(self.nccl_buffer, self.nccl_residual, self.weight, self.eps)
        if self.quantise is not None:
            self.quantise(self.nccl_buffer)

    def check(self, leg: str, phase: str) -> None:
        torch.cuda.synchronize(self.device)
        label = f"rows={self.rows} {leg}/{phase}"
        if leg == "cake":
            self.residual_out.zero_()
            self.out.zero_()
            if self.values is not None:
                self.values.view(torch.uint8).zero_()
                self.scales.zero_()
            self.cake_call()
            torch.cuda.synchronize(self.device)
            if not _rank_identical(self.residual_out, self.group):
                raise AssertionError(f"{label}: residual_out differs between ranks")
            distance = _rel_l2(self.residual_out, self.exact_z)
            if distance > _CAKE_REL_L2_MAX:
                raise AssertionError(
                    f"{label}: residual_out relative L2 distance {distance:.4f} to the exact sum"
                )
            if self.output == CakeFp4AllReduceFusedNormOutput.BF16:
                if not _rank_identical(self.out, self.group):
                    raise AssertionError(f"{label}: y differs between ranks")
                distance = _rel_l2(self.out, self.exact_y)
                if distance > _CAKE_REL_L2_MAX:
                    raise AssertionError(
                        f"{label}: y relative L2 distance {distance:.4f} to the exact result"
                    )
            else:
                if not _rank_identical(self.values, self.group):
                    raise AssertionError(
                        f"{label}: quantised values differ between ranks"
                    )
                if not _rank_identical(self.scales, self.group):
                    raise AssertionError(
                        f"{label}: quantised scales differ between ranks"
                    )
        else:
            self.nccl_call()
            torch.cuda.synchronize(self.device)
            distance = _rel_l2(self.nccl_residual, self.exact_z)
            if distance > _NCCL_REL_L2_MAX:
                raise AssertionError(
                    f"{label}: NCCL residual relative L2 distance {distance:.4f} to the exact sum"
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
    if any(leg not in ("cake", "nccl") for leg in legs):
        raise SystemExit(f"--order may only name cake and nccl legs, got {legs}")
    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    capacity = max(rows_list) if args.capacity is None else int(args.capacity)
    started = time.monotonic()
    workspace = CakeFp4AllReduceWorkspace(
        group,
        family=args.family,
        input_format=args.input_format,
        capacity=capacity,
        device=device,
    )
    row_class = _FusedNormRow if args.family == "rmsnorm" else _AllReduceRow
    report_rows = []
    try:
        for rows in rows_list:
            case = row_class(
                args, rows, rank, group, device, workspace, launch_with_pdl
            )

            def cake_calls(case=case):
                for _ in range(args.graph_calls):
                    case.cake_call()

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
                "program": case.program,
                "reference_arm": case.reference_arm,
                "legs": [],
            }
            for leg in legs:
                dist.barrier(group=group)
                case.check(leg, "pre")
                dist.barrier(group=group)
                samples, mode = _time(
                    cake_calls if leg == "cake" else nccl_calls,
                    group=group,
                    device=device,
                    warmup_iters=args.warmup_iters,
                    repeat_iters=args.repeat_iters,
                    require_graph=leg == "cake",
                )
                case.check(leg, "post")
                row["legs"].append(
                    {"arm": leg, "mode": mode, **_summary(samples, args.graph_calls)}
                )
            medians = {}
            for arm in ("cake", "nccl"):
                values = [
                    leg["median_us_per_call"]
                    for leg in row["legs"]
                    if leg["arm"] == arm
                ]
                if values:
                    medians[arm] = statistics.fmean(values)
            row["median_us_per_call"] = medians
            if "cake" in medians and "nccl" in medians:
                row["nccl_over_cake"] = medians["nccl"] / medians["cake"]
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
            "target": workspace.target,
            "grid": workspace.grid,
            "capacity": workspace.capacity,
            "torch": torch.__version__,
            "timing": {
                "method": "CUDA events around graph replays",
                "graph_calls": args.graph_calls,
                "warmup_iters": args.warmup_iters,
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
