"""Latency benchmark of the Cake SM120 MXFP4 x MXFP8 MegaMoE backend (``MoEEpLayer``).

Reports, per (shape, variant, tokens per rank), the maximum over ranks of the per-rank
median of two spans (CUDA events, cold L2 unless ``--no-l2-flush``):

* ``kernel_us``: the persistent mega kernel alone (dispatch, both expert GEMMs, SwiGLU,
  requant, return and weighted combine in one cooperative launch);
* ``forward_us``: ``MoEEpLayer.forward`` (input quantization + staging, the per-call
  routing all-gather, the kernel launch);
* ``bucket_us``: the same call with a rank-uniform token bucket
  (``stage_inputs(t, compile_tokens_per_rank=tokens)`` + ``compute_staged``): no routing
  all-gather, no host synchronization;
* ``graph_us``: replay of a CUDA graph that captured that bucketed call.

Weights are random MXFP4 (FP8 e4m3 shared expert), activations Gaussian, routing softmax
top-k over all experts -- the benchmark measures latency; correctness is covered by
``tests/moe_ep/test_moe_ep_sm120_w4a8_cake_*.py``.  Every timed configuration must finish
with the kernel's protocol flag clear.

Examples (one process per GPU; EP world size = torchrun world size):

    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/moe_ep/backends/mega/kernel/sm120/bench_moe_ep_sm120_w4a8_cake_mega.py

    # two hosts (EP16), DeepSeek-V3 routed geometry, both variants:
    torchrun --nnodes 2 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $HOST:29500 \\
        benchmarks/moe_ep/backends/mega/kernel/sm120/bench_moe_ep_sm120_w4a8_cake_mega.py \\
        --shapes dsv3 --variants noshared fp8shared

Pin NCCL's topology on multi-GPU PCIe hosts (``NCCL_TOPO_FILE``); see
``flashinfer/moe_ep/kernel_src/sm120/cake_mxfp4_megamoe/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

SHAPES = {
    # name: (hidden, intermediate, experts, top_k, swiglu_limit)
    "v4flash": (4096, 2048, 256, 6, 10),
    "dsv3": (7168, 2048, 256, 8, 0),
    "qwen3_235b": (4096, 1536, 128, 8, 0),
    "qwen3_30b": (2048, 768, 128, 8, 0),
}
L2_FLUSH_BYTES = 256 << 20


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes", nargs="+", default=["v4flash"], choices=sorted(SHAPES))
    p.add_argument(
        "--variants",
        nargs="+",
        default=["noshared", "fp8shared"],
        choices=("noshared", "fp8shared"),
    )
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="live tokens per rank (the workspace capacity is the largest)",
    )
    p.add_argument(
        "--table",
        default="all_gin",
        help="transport_table: all_gin | all_lsa | lsa_where_reachable | lsa_pairs | tuned | auto",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-l2-flush", action="store_true")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--jsonl", default=None, help="append one JSON record per row")
    return p.parse_args()


def _table_spec(name: str, world: int):
    if name != "lsa_pairs":
        return name
    return tuple(
        0 if a == b else (1 if a // 2 == b // 2 else 2)
        for a in range(world)
        for b in range(world)
    )


class _KernelTimer:
    """Records CUDA events around the backend's cooperative launch (same stream)."""

    def __init__(self, module) -> None:
        self._module = module
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def __getattr__(self, name):
        return getattr(self._module, name)

    def launch_cooperative(self, *args):
        if torch.cuda.is_current_stream_capturing():
            # Timing events cannot be recorded into a graph; graph_us times the replay.
            self._module.launch_cooperative(*args)
            return
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._module.launch_cooperative(*args)
        stop.record()
        self.spans.append((start, stop))


def _max_rank(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _time_forward(run, iters: int, flush) -> list[float]:
    spans = []
    for _ in range(iters):
        dist.barrier()
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        stop.synchronize()
        spans.append(start.elapsed_time(stop) * 1e3)
    return spans


def main() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_synthetic import (
        Geometry,
        make_layer,
        random_inputs,
        random_weights,
    )

    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.cuda.current_device())
    rank, world = dist.get_rank(), dist.get_world_size()
    flush = (
        None
        if args.no_l2_flush
        else torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    )
    capacity = max(args.tokens)
    rows = []
    if rank == 0:
        print(
            f"# Cake SM120 MegaMoE, EP{world}, table={args.table}, "
            f"{torch.cuda.get_device_name()}, cold L2={flush is not None}"
        )
        print(
            f"{'shape':11} {'variant':9} {'tokens':>6} {'kernel_us':>10} {'forward_us':>11} "
            f"{'bucket_us':>9} {'graph_us':>9}  instance"
        )
    for shape in args.shapes:
        hidden, inter, experts, top_k, limit = SHAPES[shape]
        geometry = Geometry(hidden, inter, experts, top_k)
        for variant in args.variants:
            shared = variant == "fp8shared"
            weights = random_weights(
                rank, world, geometry, shared=shared, seed=args.seed
            )
            layer = make_layer(
                rank,
                world,
                weights,
                geometry,
                max_tokens_per_rank=capacity,
                shared=shared,
                transport_table=_table_spec(args.table, world),
                swiglu_limit=limit,
            )
            layer.warmup(random_inputs(256, geometry, seed=args.seed + rank))
            ws = layer._workspace
            timer = _KernelTimer(ws.module)
            ws.module = timer
            for tokens in args.tokens:
                inputs = random_inputs(
                    tokens, geometry, seed=args.seed + 7 * rank + tokens
                )
                for _ in range(args.warmup):
                    layer(inputs)
                torch.cuda.synchronize()
                timer.spans.clear()
                forward = _time_forward(lambda: layer(inputs), args.iters, flush)
                kernel = [a.elapsed_time(b) * 1e3 for a, b in timer.spans]

                def bucketed(inputs=inputs, tokens=tokens):
                    layer.stage_inputs(inputs, compile_tokens_per_rank=tokens)
                    return layer.compute_staged(output=None)

                for _ in range(args.warmup):
                    bucketed()
                bucket_fwd = _time_forward(bucketed, args.iters, flush)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                dist.barrier()
                with torch.cuda.graph(graph):
                    bucketed()
                torch.cuda.synchronize()
                dist.barrier()
                for _ in range(args.warmup):
                    graph.replay()
                graph_fwd = _time_forward(graph.replay, args.iters, flush)
                del graph
                error = layer._kernel.protocol_error(ws)
                if _max_rank(float(error)) != 0:
                    raise RuntimeError(f"protocol_error at {shape} {variant} {tokens}")
                row = {
                    "shape": shape,
                    "variant": variant,
                    "ep": world,
                    "tokens_per_rank": tokens,
                    "table": ws.transport_table.name,
                    "instance": ws.last_launch.get("instance"),
                    "kernel_us": _max_rank(statistics.median(kernel)),
                    "forward_us": _max_rank(statistics.median(forward)),
                    "bucket_us": _max_rank(statistics.median(bucket_fwd)),
                    "graph_us": _max_rank(statistics.median(graph_fwd)),
                    "iters": args.iters,
                    "cold_l2": flush is not None,
                    "gpu": torch.cuda.get_device_name(),
                }
                rows.append(row)
                if rank == 0:
                    print(
                        f"{shape:11} {variant:9} {tokens:6d} {row['kernel_us']:10.1f} "
                        f"{row['forward_us']:11.1f} {row['bucket_us']:9.1f} {row['graph_us']:9.1f}  "
                        f"{row['instance']} ({row['table']})",
                        flush=True,
                    )
            ws.module = timer._module
            layer.destroy()
            dist.barrier()
    if rank == 0 and args.jsonl:
        with open(args.jsonl, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
