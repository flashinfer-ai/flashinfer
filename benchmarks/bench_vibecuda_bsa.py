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

Benchmark VibeCUDA VSA against the refreshed CAKE VSA baseline.

The matrix is the union frozen by FlashInfer PR #4593 and refreshed by PR
#4804: 16 canonical rows plus two FastWan direct-metadata rows. Planning,
metadata construction, and caller-owned output allocation are outside timing.

The current VibeCUDA API consumes a dense boolean block mask. The two FastWan
rows require direct q2k metadata and non-uniform block lengths, so they are
reported as unsupported instead of silently changing their math.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from flashinfer.cake_vsa import plan_cake_vsa, run_cake_vsa
from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing.utils import bench_gpu_time


@dataclass(frozen=True)
class Workload:
    name: str
    cohort: str
    block_size: int
    sequence: int
    selected: int
    heads: int = 8
    head_dim: int = 128
    real_sequence: int | None = None
    dit_shape: tuple[int, int, int] | None = None


WORKLOADS = (
    Workload("canonical_blk128_s16384_d50", "canonical", 128, 16384, 64),
    Workload("canonical_blk128_s16384_d25", "canonical", 128, 16384, 32),
    Workload("canonical_blk128_s16384_d10", "canonical", 128, 16384, 13),
    Workload("canonical_blk128_s32768_d25", "canonical", 128, 32768, 64),
    Workload("canonical_blk128_s32768_d10", "canonical", 128, 32768, 26),
    Workload("canonical_blk128_s80000_d25", "canonical", 128, 80000, 156),
    Workload("canonical_blk128_s80000_d10", "canonical", 128, 80000, 62),
    Workload("canonical_blk64_s1024_d25", "canonical", 64, 1024, 4),
    Workload("canonical_blk64_s1024_d50", "canonical", 64, 1024, 8),
    Workload("canonical_blk64_s1024_d75", "canonical", 64, 1024, 12),
    Workload("canonical_blk64_s2048_d25", "canonical", 64, 2048, 8),
    Workload("canonical_blk64_s2048_d50", "canonical", 64, 2048, 16),
    Workload("canonical_blk64_s2048_d75", "canonical", 64, 2048, 24),
    Workload("canonical_blk64_s4096_d25", "canonical", 64, 4096, 16),
    Workload("canonical_blk64_s4096_d50", "canonical", 64, 4096, 32),
    Workload("canonical_blk64_s4096_d75", "canonical", 64, 4096, 48),
    Workload(
        "fastwan_61x448x832_s23296",
        "fastwan",
        64,
        23296,
        math.ceil(0.20 * (23296 // 64)),
        heads=12,
        real_sequence=23296,
        dit_shape=(16, 28, 52),
    ),
    Workload(
        "fastwan_61x480x832_s24960_p26624",
        "fastwan",
        64,
        26624,
        math.ceil(0.20 * (26624 // 64)),
        heads=12,
        real_sequence=24960,
        dit_shape=(16, 30, 52),
    ),
)


def _variable_block_lens(workload: Workload) -> torch.Tensor:
    blocks = workload.sequence // workload.block_size
    if workload.dit_shape is None:
        return torch.full(
            (blocks,), workload.block_size, dtype=torch.int32, device="cuda"
        )
    t, h, w = workload.dit_shape
    tile = 4
    sizes = [
        min(tile, t - t0) * min(tile, h - h0) * min(tile, w - w0)
        for t0 in range(0, t, tile)
        for h0 in range(0, h, tile)
        for w0 in range(0, w, tile)
    ]
    if len(sizes) != blocks or sum(sizes) != workload.real_sequence:
        raise AssertionError(f"invalid FastWan tiling for {workload.name}")
    return torch.tensor(sizes, dtype=torch.int32, device="cuda")


def _sparse_metadata(workload: Workload):
    blocks = workload.sequence // workload.block_size
    generator = torch.Generator().manual_seed(459300000 + blocks + workload.selected)
    head_count = workload.heads if workload.cohort == "fastwan" else 1
    rows = []
    for head in range(head_count):
        per_head = []
        for query_block in range(blocks):
            chosen = torch.randperm(blocks, generator=generator)[: workload.selected]
            if workload.cohort == "fastwan":
                chosen = (chosen + head * 5 + query_block * 3).remainder(blocks)
            per_head.append(chosen.sort().values.to(torch.int32))
        rows.append(torch.stack(per_head))
    q2k = torch.stack(rows)
    if q2k.shape[0] == 1:
        q2k = q2k.expand(workload.heads, -1, -1).clone()
    q2k = q2k.cuda().contiguous()
    q2k_num = torch.full(
        (workload.heads, blocks),
        workload.selected,
        dtype=torch.int32,
        device="cuda",
    )
    if workload.cohort == "fastwan":
        return None, q2k, q2k_num
    block_mask = torch.zeros(
        (workload.heads, blocks, blocks), dtype=torch.bool, device="cuda"
    )
    block_mask.scatter_(2, q2k.long(), True)
    return block_mask, None, None


def _inputs(workload: Workload, index: int):
    generator = torch.Generator(device="cuda").manual_seed(459300000 + index)
    shape = (workload.sequence, workload.heads, workload.head_dim)
    q = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    block_mask, q2k, q2k_num = _sparse_metadata(workload)
    return q, k, v, block_mask, q2k, q2k_num, _variable_block_lens(workload)


def _cake_callable(workload: Workload, inputs):
    q, k, v, block_mask, q2k, q2k_num, kv_block_lens = inputs
    plan = plan_cake_vsa(
        None,
        None,
        block_mask,
        kv_block_lens if workload.block_size == 64 else None,
        q2k,
        q2k_num,
        M=workload.sequence,
        N=workload.sequence,
        R=workload.block_size,
        C=workload.block_size,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads,
        head_dim=workload.head_dim,
        q_data_type=q.dtype,
        sm_scale=None,
        device=q.device,
    )
    output = torch.empty_like(q)

    def invoke():
        return run_cake_vsa(
            plan, q, k, v, out=output, lse=None, return_lse=False, backend="cake"
        )

    return invoke, output


def _vibecuda_callable(workload: Workload, inputs):
    q, k, v, block_mask, _, _, kv_block_lens = inputs
    if block_mask is None or bool(torch.any(kv_block_lens != workload.block_size)):
        raise NotImplementedError(
            "VibeCUDA lacks direct q2k metadata with non-uniform kv_block_lens"
        )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vibecuda")
    wrapper.plan(
        None,
        None,
        workload.sequence,
        workload.sequence,
        workload.block_size,
        workload.block_size,
        workload.heads,
        workload.heads,
        workload.head_dim,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        block_mask=block_mask,
    )
    output = torch.empty_like(q)

    def invoke():
        return wrapper.run(q, k, v, out=output, return_lse=False)

    return invoke, output


def _measure(invoke, dry_run_iters: int, repeat_iters: int):
    samples = [
        float(value)
        for value in bench_gpu_time(
            invoke,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
    ]
    return {
        "median_ms": float(np.median(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
        "timing": "cupti",
        "cold_l2_cache": True,
        "plan": "outside_timed_region",
        "output_allocation": "outside_timed_region",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["cake", "vibecuda"])
    parser.add_argument(
        "--cohort", choices=("all", "canonical", "fastwan"), default="all"
    )
    parser.add_argument(
        "--row-index",
        type=int,
        help="Run one zero-based row in a fresh process",
    )
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("VSA benchmark requires SM100 or SM103")
    selected = [
        workload
        for index, workload in enumerate(WORKLOADS)
        if (args.cohort == "all" or workload.cohort == args.cohort)
        and (args.row_index is None or index == args.row_index)
    ]
    if args.row_index is not None and not selected:
        raise ValueError(f"row {args.row_index} is outside the selected cohort")
    results = []
    for index, workload in enumerate(WORKLOADS):
        if workload not in selected:
            continue
        print(
            f"running row {index}: {workload.name} with {','.join(args.backends)}",
            flush=True,
        )
        inputs = _inputs(workload, index)
        row = {"workload": asdict(workload), "backends": {}}
        callables = {}
        for backend in args.backends:
            try:
                if backend == "cake":
                    callables[backend] = _cake_callable(workload, inputs)
                elif backend == "vibecuda":
                    callables[backend] = _vibecuda_callable(workload, inputs)
                else:
                    raise ValueError(f"unknown backend: {backend}")
            except NotImplementedError as exc:
                row["backends"][backend] = {
                    "status": "unsupported",
                    "reason": str(exc),
                }
            except (RuntimeError, ValueError) as exc:
                status = (
                    "unsupported"
                    if backend == "vibecuda" and "currently supports" in str(exc)
                    else "error"
                )
                row["backends"][backend] = {
                    "status": status,
                    "reason": str(exc),
                }

        precision_passed = False
        if "cake" in callables and "vibecuda" in callables:
            cake_invoke, cake_output = callables["cake"]
            vibecuda_invoke, vibecuda_output = callables["vibecuda"]
            try:
                cake_invoke()
                torch.cuda.synchronize()
                vibecuda_invoke()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    vibecuda_output,
                    cake_output,
                    atol=1e-2,
                    rtol=1e-2,
                    msg=lambda msg: f"{workload.name}: {msg}",
                )
                precision_passed = True
                row["precision"] = {
                    "status": "passed",
                    "reference": "refreshed CAKE output",
                    "atol": 1e-2,
                    "rtol": 1e-2,
                }
            except (AssertionError, RuntimeError, ValueError) as exc:
                row["comparison_error"] = str(exc)

        for backend, (invoke, _) in callables.items():
            if backend in row["backends"]:
                continue
            try:
                row["backends"][backend] = {
                    "status": "passed",
                    **_measure(invoke, args.dry_run_iters, args.repeat_iters),
                }
            except (RuntimeError, ValueError) as exc:
                row["backends"][backend] = {
                    "status": "error",
                    "reason": str(exc),
                }
        if (
            all(
                row["backends"].get(name, {}).get("status") == "passed"
                for name in ("cake", "vibecuda")
            )
            and precision_passed
        ):
            row["speedup"] = (
                row["backends"]["cake"]["median_ms"]
                / row["backends"]["vibecuda"]["median_ms"]
            )
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    speedups = [row["speedup"] for row in results if "speedup" in row]
    complete = len(speedups) == len(results)
    partial_arithmetic_mean = sum(speedups) / len(speedups) if speedups else None
    partial_geometric_mean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    summary = {
        "requested_rows": len(results),
        "passing_comparison_rows": len(speedups),
        "complete": complete,
        "arithmetic_mean": partial_arithmetic_mean if complete else None,
        "geometric_mean": partial_geometric_mean if complete else None,
        "minimum": min(speedups) if complete and speedups else None,
        "maximum": max(speedups) if complete and speedups else None,
        "partial_comparison_arithmetic_mean": partial_arithmetic_mean,
        "partial_comparison_geometric_mean": partial_geometric_mean,
        "baseline": "FlashInfer PR #4804 CAKE VSA",
        "baseline_commit": "8fcf0fa6d44d5dc5ee7ab2ea9664ec96ea723c4e",
        "baseline_merged_commit": "adc49a85302ef16259aad0cf7c323049a5072851",
        "results": results,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
