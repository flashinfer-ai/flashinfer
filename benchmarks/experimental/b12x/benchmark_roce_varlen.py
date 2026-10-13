"""RoCEnante variable-length all-gather across DGX Sparks: latency and receive bandwidth.

Launch one rank per host with torchrun, for example on two Sparks::

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        benchmarks/experimental/b12x/benchmark_roce_varlen.py --output varlen.json

Every rank contributes the same byte count; each size is checked for exact
bytes on every rank, then timed as an ``--ops``-gather CUDA graph (median replay
per rank, maximum over ranks).  Bandwidth is what one rank receives:
``(world - 1) * bytes`` per gather.  ``--chunk-kib`` and ``--blocks`` take
comma-separated lists to sweep the runtime's chunk size and grid.
"""

import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--sizes-mib", default="0.0625,1,8,64,256")
ap.add_argument("--chunk-kib", default="256")
ap.add_argument("--blocks", default="4")
ap.add_argument("--ops", type=int, default=10)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--output", default="")
args = ap.parse_args()

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

from b12x.comm import roce
from b12x.comm.roce import _preparation
from b12x.preparation import PreparationSession, PreparedCall


def log(msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


sizes = [int(float(s) * (1 << 20)) // 16 * 16 for s in args.sizes_mib.split(",")]
capacity = max(sizes)
rows = []
for chunk_kib in (int(c) for c in args.chunk_kib.split(",")):
    for blocks in (int(b) for b in args.blocks.split(",")):
        runtime = roce.AllReduce(
            exchange_group=dist.group.WORLD,
            device=device,
            max_size=1 << 20,
            max_gather_bytes=1 << 20,
            max_varlen_bytes=capacity,
            varlen_chunk_bytes=chunk_kib << 10,
            varlen_blocks=blocks,
        )
        query = roce.query_from_runtime(
            runtime,
            surface="AllReduce.all_gather_varlen",
            call={"dtypes": ("bfloat16",)},
            topology="roce_rdma",
            peer_hosts=tuple(f"rank-{r}" for r in range(world)),
        )
        plan = roce.plan(query, runtime=runtime)
        seed_source = torch.zeros(4096, dtype=torch.uint8, device=device)
        seed_counts = torch.full((world,), 16, dtype=torch.int64, device=device)

        def prepare(state):
            return _preparation.prepared_varlen_call(
                state, source=seed_source, counts=seed_counts
            )

        session = PreparationSession(device=device, autotune=False, compile_workers=2)
        session.prepare((plan.request(name="roce", prepare_call=prepare),))
        chunk_bytes = runtime.stats()["varlen_chunk_bytes"]
        def contribution(peer):
            """Rank ``peer``'s bytes, which every rank regenerates to check its rows."""
            gen = torch.Generator(device=device).manual_seed(1000 + peer)
            return torch.randint(
                0, 256, (capacity,), dtype=torch.uint8, device=device, generator=gen
            )

        source = contribution(rank)
        out = torch.empty((world, capacity), dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream()
        for nbytes in sizes:
            counts = torch.full((world,), nbytes, dtype=torch.int64, device=device)

            def op():
                runtime.all_gather_varlen(source, counts, out=out, plan=plan)

            op()
            torch.cuda.synchronize()
            for peer in range(world):
                if not torch.equal(out[peer, :nbytes], contribution(peer)[:nbytes]):
                    raise SystemExit(f"rank {rank}: row {peer} wrong at {nbytes} bytes")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                op()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
                for _ in range(args.ops):
                    op()
            graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.reps):
                dist.barrier()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - begin) / args.ops * 1e6)
            times = [None] * world
            dist.all_gather_object(times, statistics.median(samples))
            runtime.check_health()
            us = max(times)
            row = {
                "bytes": nbytes,
                "chunk_bytes": chunk_bytes,
                "blocks": blocks,
                "us": round(us, 1),
                "recv_GBps": round((world - 1) * nbytes / us / 1e3, 2),
            }
            rows.append(row)
            log(
                f"{nbytes / (1 << 20):9.4f} MiB  chunk {chunk_bytes >> 10:5d} KiB  "
                f"blocks {blocks}  {us:10.1f} us  recv {row['recv_GBps']:6.2f} GB/s"
            )
            del graph
        runtime.close()
        dist.barrier()

if rank == 0 and args.output:
    with open(args.output, "w") as f:
        json.dump({"world": world, "rows": rows}, f, indent=1)
dist.destroy_process_group()
