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

Kimi-K3 TP12 fused LatentMoE tail: fused Cake backend versus the stock chain.

Twelve ranks on one multi-node NVLink domain (three GB200 / GB300 NVL72
compute trays), one process per GPU::

    torchrun --nnodes 3 --nproc-per-node 4 --node-rank <n> --master-addr <host> \
        benchmarks/bench_cake_kimi_k3_tp12_tail.py --M 1 8 32 64 128 256 512 1024 2048 4096

Stock chain (the fastest stock route of every stage on GB200 / GB300):
NCCL ``all_reduce(routed_partial)`` -> ``flashinfer.norm.rmsnorm`` ->
cuBLAS ``torch.mm`` (the full replicated up-projection) ->
NCCL ``all_reduce(shared_partial)`` -> ``torch.add``, captured in one CUDA graph.
Fused: ``flashinfer.kimi_k3_tp12_tail`` (two generated launches per rank plus
cuBLAS for ``M > 4``), captured in one CUDA graph.  Timing is CUPTI kernel
activity per rank (cold L2); ``bench_gpu_time_with_cupti`` gathers every
iteration's span from all ranks, the row time is the median over iterations of
the maximum over ranks (a collective finishes when its slowest rank does), and
every rank's own median is reported as the rank spread.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

import flashinfer
from flashinfer.kimi_k3_tp12_tail import (
    create_kimi_k3_tp12_tail_workspace,
    prepare_kimi_k3_tp12_tail,
)
from flashinfer.testing.utils import bench_gpu_time_with_cupti

HIDDEN, LATENT, EPS = 7168, 3584, 1.0e-5
DEFAULT_ROWS = (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def make_inputs(M: int, rank: int, device: torch.device, seed: int = 620) -> dict:
    g = torch.Generator(device=device).manual_seed(seed * 1000 + M)
    norm_w = (1.0 + 0.1 * torch.randn((LATENT,), generator=g, device=device)).to(
        torch.bfloat16
    )
    up_w = (0.01 * torch.randn((HIDDEN, LATENT), generator=g, device=device)).to(
        torch.bfloat16
    )
    gr = torch.Generator(device=device).manual_seed(seed * 1000 + M + 17 * (rank + 1))
    routed = (0.3 * torch.randn((M, LATENT), generator=gr, device=device)).to(
        torch.bfloat16
    )
    shared = (0.1 * torch.randn((M, HIDDEN), generator=gr, device=device)).to(
        torch.bfloat16
    )
    return dict(
        norm_w=norm_w,
        up_w=up_w,
        routed=routed,
        shared=shared,
        out=torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device),
        y=torch.empty((M, LATENT), dtype=torch.bfloat16, device=device),
        gemm=torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device),
    )


def stock_chain(inp: dict, group_name: str):
    def all_reduce(x: torch.Tensor) -> torch.Tensor:
        return fc.wait_tensor(fc.all_reduce(x, "sum", group_name))

    def run() -> torch.Tensor:
        routed_sum = all_reduce(inp["routed"])
        flashinfer.norm.rmsnorm(routed_sum, inp["norm_w"], eps=EPS, out=inp["y"])
        shared_sum = all_reduce(inp["shared"])
        torch.mm(inp["y"], inp["up_w"].t(), out=inp["gemm"])
        torch.add(inp["gemm"], shared_sum, out=inp["out"])
        return inp["out"]

    return run


def graph_capture(run):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.synchronize()
    return graph.replay, graph


def _per_rank(spans) -> tuple[float, ...]:
    """``aggregate_op`` for ``bench_gpu_time_with_cupti``: keep every rank's span of the iteration."""
    return tuple(float(v) for v in spans)


def measure_us(fn, *, warmup: int, iters: int, world: int) -> tuple[float, list[float]]:
    """Time ``fn`` with CUPTI on every rank: ``(row time, every rank's own median)`` in microseconds.

    ``bench_gpu_time_with_cupti`` aggregates the per-iteration spans across the
    initialised process group (elementwise ``max`` by default, which makes every
    rank report the same numbers); with ``_per_rank`` each iteration comes back
    as the tuple of all ranks' spans.  The row time is the median over
    iterations of the maximum over ranks; the per-rank medians give the spread.
    """
    times = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        aggregate_op=_per_rank,
    )
    if not times or any(not isinstance(t, tuple) or len(t) != world for t in times):
        raise RuntimeError(
            "bench_gpu_time_with_cupti did not gather the spans of all ranks; "
            "the benchmark needs an initialised twelve-rank process group"
        )
    row_us = statistics.median(max(spans) for spans in times) * 1e3
    per_rank_us = [
        statistics.median(spans[r] for spans in times) * 1e3 for r in range(world)
    ]
    return row_us, per_rank_us


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--M", type=int, nargs="+", default=list(DEFAULT_ROWS))
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--groups", type=int, default=3, help="counterbalanced A B / B A groups"
    )
    ap.add_argument("--json", default=None, help="rank 0 writes the table here")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 12:
        raise SystemExit(f"the Kimi-K3 TP12 tail needs twelve ranks, got {world}")
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    group_name = dist.group.WORLD.group_name
    max_tokens = args.max_tokens or max(args.M)
    workspace = create_kimi_k3_tp12_tail_workspace(rank=rank, max_tokens=max_tokens)
    rows = []
    try:
        for M in args.M:
            inp = make_inputs(M, rank, device)
            stock_replay, stock_graph = graph_capture(stock_chain(inp, group_name))
            stock_out = inp["out"].clone()
            runner = prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
            fused_replay, fused_graph = graph_capture(runner)
            fused_out = inp["out"].clone()
            max_abs = float((fused_out.float() - stock_out.float()).abs().max().item())
            arms = {"stock_graph": stock_replay, "fused_graph": fused_replay}
            samples: dict[str, list[float]] = {name: [] for name in arms}
            per_rank_samples: dict[str, list[list[float]]] = {name: [] for name in arms}
            for g in range(args.groups):
                order = list(arms) if g % 2 == 0 else list(reversed(arms))
                for name in order:
                    dist.barrier()
                    row_us, per_rank = measure_us(
                        arms[name], warmup=args.warmup, iters=args.iters, world=world
                    )
                    samples[name].append(row_us)
                    per_rank_samples[name].append(per_rank)
            stock_us = statistics.median(samples["stock_graph"])
            fused_us = statistics.median(samples["fused_graph"])
            # every rank's own median (over iterations, then over the groups) and its spread
            per_rank_us = {
                name: [
                    statistics.median(group[r] for group in groups)
                    for r in range(world)
                ]
                for name, groups in per_rank_samples.items()
            }
            rank_spread_us = {
                name: max(values) - min(values) for name, values in per_rank_us.items()
            }
            row = dict(
                M=M,
                stock_us=stock_us,
                fused_us=fused_us,
                speedup=stock_us / fused_us,
                max_abs_diff_vs_stock=max_abs,
                kernels=list(runner.kernel_keys),
                per_rank_us=per_rank_us,
                rank_spread_us=rank_spread_us,
            )
            rows.append(row)
            if rank == 0:
                print(
                    f"M={M:5d}  stock {stock_us:8.1f} us  fused {fused_us:8.1f} us  "
                    f"speedup {row['speedup']:.2f}x  max|diff| {max_abs:.4f}  "
                    f"rank spread {rank_spread_us['stock_graph']:.1f} / "
                    f"{rank_spread_us['fused_graph']:.1f} us  {runner.kernel_keys}",
                    flush=True,
                )
            del arms, stock_replay, fused_replay, stock_graph, fused_graph
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        workspace.destroy()
    if rank == 0 and args.json:
        props = torch.cuda.get_device_properties(device)
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    gpu=props.name,
                    world_size=world,
                    torch=torch.__version__,
                    warmup=args.warmup,
                    iters=args.iters,
                    groups=args.groups,
                    rows=rows,
                ),
                f,
                indent=1,
            )
    dist.barrier()
    torch.cuda.synchronize()
    # The NCCL collectives captured into the stock-chain CUDA graphs leave their work objects
    # pending in the process group, and ``destroy_process_group()`` then waits for them without
    # end (torch 2.13 / CUDA 13.3 on GB200 and GB300).  Every rank is past the final barrier and
    # the table is written, so abort the communicators instead of waiting on them.
    abort = getattr(dist.distributed_c10d, "_abort_process_group", None)
    if abort is not None:
        abort()
    else:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
