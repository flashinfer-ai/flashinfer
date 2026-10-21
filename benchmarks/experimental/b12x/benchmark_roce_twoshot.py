"""RoCEnante one-shot vs pipelined two-shot vs NCCL all-reduce across DGX Sparks.

Launch one rank per host with torchrun, for example on four Sparks::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        benchmarks/experimental/b12x/benchmark_roce_twoshot.py --output roce.json

BF16 messages; each path is first checked against an FP32 reference and for
bitwise agreement across ranks (and one-shot against two-shot), then timed as
20-op CUDA graphs: the median replay per rank, maximum over ranks.
"""

import argparse
import hashlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--sizes-kib", default="1024,2048,4096,8192,16384,32768,65536")
ap.add_argument("--paths", default="oneshot,twoshot,nccl")
ap.add_argument("--max-mib", type=int, default=32)
ap.add_argument("--pipe-mib", type=int, default=64)
ap.add_argument("--pipe-blocks", type=int, default=None)
ap.add_argument("--chunk-kib", type=int, default=None)
ap.add_argument("--ops", type=int, default=20)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--output", default="")
ap.add_argument(
    "--auto", action="store_true", help="also time the runtime's own size routing"
)
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


runtime = roce.AllReduce(
    exchange_group=dist.group.WORLD,
    device=device,
    max_size=args.max_mib << 20,
    max_gather_bytes=1 << 20,
    pipe_bytes=args.pipe_mib << 20,
    **({} if args.pipe_blocks is None else {"pipe_blocks": args.pipe_blocks}),
    **({} if args.chunk_kib is None else {"pipe_chunk_bytes": args.chunk_kib << 10}),
)
query = roce.query_from_runtime(
    runtime,
    surface="AllReduce.all_reduce",
    call={"dtypes": ("bfloat16",)},
    topology="roce_rdma",
    peer_hosts=tuple(f"rank-{r}" for r in range(world)),
)
plan = roce.plan(query, runtime=runtime)
seed = torch.zeros(8, dtype=torch.bfloat16, device=device)


def prepare(state):
    call = _preparation.prepared_call(state, inp=seed)
    return PreparedCall(run=call.run)


session = PreparationSession(device=device, autotune=False, compile_workers=2)
session.prepare((plan.request(name="roce", prepare_call=prepare),))
log(
    f"runtime ready: {json.dumps({k: v for k, v in runtime.stats().items() if 'pipe' in k or k in ('rails', 'max_size')})}"
)

paths = args.paths.split(",") + (["auto"] if args.auto else [])
default_threshold = runtime.pipe_threshold
nccl_group = dist.new_group(backend="nccl") if "nccl" in paths else None
if nccl_group is not None:
    warm = torch.ones(1024, dtype=torch.bfloat16, device=device)
    dist.all_reduce(warm, group=nccl_group)
    torch.cuda.synchronize()


def make_op(path, inp, out):
    if path == "nccl":

        def op():
            dist.all_reduce(out, group=nccl_group)

        return op
    runtime.pipe_threshold = {"twoshot": 1, "oneshot": 1 << 62}.get(
        path, default_threshold
    )

    def op():
        runtime.all_reduce(inp, out=out, plan=plan)

    return op


def digest(t):
    return hashlib.sha1(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]


rows = []
stream = torch.cuda.Stream()
for kib in (int(s) for s in args.sizes_kib.split(",")):
    nbytes = kib << 10
    numel = nbytes // 2
    gen = torch.Generator(device=device).manual_seed(rank + 17)
    inp = torch.randn(numel, dtype=torch.bfloat16, device=device, generator=gen)
    expected = torch.zeros(numel, dtype=torch.float32, device=device)
    for peer in range(world):
        g = torch.Generator(device=device).manual_seed(peer + 17)
        expected.add_(
            torch.randn(numel, dtype=torch.bfloat16, device=device, generator=g).float()
        )
    expected = expected.to(torch.bfloat16)
    row = {"bytes": nbytes}
    digests = {}
    for path in paths:
        if path == "oneshot" and nbytes > runtime.max_size:
            continue
        if path in ("twoshot", "auto") and nbytes > runtime.pipe_bytes:
            continue
        out = torch.empty_like(inp)
        if path == "nccl":
            out.copy_(inp)
        op = make_op(path, inp, out)
        op()
        torch.cuda.synchronize()
        err = (out.float() - expected.float()).abs().max().item()
        same = [None] * world
        dist.all_gather_object(same, digest(out))
        if path != "nccl":
            digests[path] = same[0]
            if len(set(same)) != 1 or err > 0.07:
                raise SystemExit(
                    f"rank {rank} {path} {nbytes}: err {err} digests {same}"
                )
        # graph: in-place NCCL keeps growing values; correctness was checked eagerly
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op()
            op()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.ops):
                op()
        graph.replay()
        torch.cuda.synchronize()
        if path != "nccl":
            gerr = (out.float() - expected.float()).abs().max().item()
            if gerr > 0.07:
                raise SystemExit(f"rank {rank} {path} {nbytes}: graph err {gerr}")
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
        row[path] = round(max(times), 1)
        row[path + "_err"] = round(err, 4)
        runtime.check_health()
        del graph
    if "oneshot" in digests and "twoshot" in digests:
        row["bitwise_equal"] = digests["oneshot"] == digests["twoshot"]
    rows.append(row)
    if rank == 0:
        busbw = {
            p: round(2 * (world - 1) / world * nbytes / row[p] / 1e3, 1)
            for p in paths
            if p in row
        }
        log(
            f"{kib:6d} KiB  "
            + "  ".join(f"{p} {row[p]:8.1f} us" for p in paths if p in row)
            + f"  busbw GB/s {busbw}"
            + (
                f"  bitwise1v2 {row.get('bitwise_equal')}"
                if "bitwise_equal" in row
                else ""
            )
        )
if rank == 0 and args.output:
    json.dump(
        {
            "world": world,
            "args": vars(args),
            "rows": rows,
            "stats": {
                k: v
                for k, v in runtime.stats().items()
                if isinstance(v, (int, float, str, list))
            },
        },
        open(args.output, "w"),
        indent=1,
    )
runtime.close()
dist.barrier()
dist.destroy_process_group()
