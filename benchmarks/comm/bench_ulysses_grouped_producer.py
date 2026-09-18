# SPDX-License-Identifier: Apache-2.0
"""Projection/head pipeline reference; framework scheduling, not a public runtime.

torchrun --standalone --nproc_per_node=2 benchmarks/comm/bench_ulysses_grouped_producer.py --schedule 14,14

Compares whole GEMM + identical head pipeline with coarse producer overlap.
Times X -> local all-head O; excludes norm, RoPE and output projection. This
is NOT the original 0915 full-sublayer benchmark and inherits no speedup claim.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flashinfer.comm import UlyssesCommunicator
from flashinfer.comm.ulysses_experimental import prepare_ulysses_producer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq", type=int, default=2048, help="global physical sequence"
    )
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--schedule", default="14,14")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    if args.seq <= 0 or args.seq % world or args.heads % world or args.iterations <= 0:
        raise ValueError("sequence/heads must be positive and divisible by world")
    schedule = tuple(int(x) for x in args.schedule.split(","))
    if not schedule or min(schedule) <= 0 or sum(schedule) != args.heads // world:
        raise ValueError("schedule must partition local heads")
    configs = [None] * world
    dist.all_gather_object(configs, vars(args))
    if any(c != configs[0] for c in configs):
        raise ValueError("benchmark arguments differ across ranks")
    rows, local_heads = args.seq // world, args.heads // world
    dtype, device = torch.bfloat16, torch.device("cuda", torch.cuda.current_device())
    gin, gout = dist.new_group(), dist.new_group()
    capacity = rows * world * max(schedule) * args.dim
    cin = UlyssesCommunicator(
        group=gin, backend="nccl", max_elems=3 * capacity, dtype=dtype, device=device
    )
    cout = UlyssesCommunicator(
        group=gout, backend="nccl", max_elems=capacity, dtype=dtype, device=device
    )
    # QKV projection remains on caller; compute has its own stream so it can
    # overlap subsequent GEMMs. Streams/process groups belong to this example.
    ins, comp, outs = [torch.cuda.Stream() for _ in range(3)]
    produced, incoming, computed = [
        [torch.cuda.Event() for _ in schedule] for _ in range(3)
    ]
    torch.manual_seed(177)
    weight = (
        torch.randn(3 * args.heads * args.dim, args.hidden, device=device, dtype=dtype)
        * 0.02
    )
    torch.manual_seed(178 + rank)
    x = torch.randn(rows, args.hidden, device=device, dtype=dtype)
    producer = prepare_ulysses_producer(
        weight,
        world_size=world,
        heads=args.heads,
        head_dim=args.dim,
        local_seq=rows,
        schedule=schedule,
    )
    raw = torch.empty(rows, 3 * args.heads * args.dim, device=device, dtype=dtype)
    received = [
        torch.empty(1, args.seq, c, 3 * args.dim, device=device, dtype=dtype)
        for c in schedule
    ]
    output = torch.empty(1, rows, args.heads, args.dim, device=device, dtype=dtype)
    wi = cin.create_workspace(max_elems=3 * capacity)
    wo = cout.create_workspace(max_elems=capacity)
    attention_outputs = [None] * len(schedule)

    def run(coarse):
        caller = torch.cuda.current_stream()
        ins.wait_stream(caller)
        comp.wait_stream(caller)
        outs.wait_stream(caller)
        if not coarse:
            torch.mm(x, weight.t(), out=raw)
            whole = raw.view(1, rows, 3, args.heads, args.dim).unbind(2)
        offset = 0
        for i, count in enumerate(schedule):
            q, k, v = producer.produce(x, i) if coarse else whole
            produced[i].record(caller)
            with torch.cuda.stream(ins):
                ins.wait_event(produced[i])
                cin.scatter_qkv_head_chunk(
                    q,
                    k,
                    v,
                    head_offset=0 if coarse else offset,
                    head_count=count,
                    out=received[i],
                    workspace=wi,
                )
                incoming[i].record()
            with torch.cuda.stream(comp):
                comp.wait_event(incoming[i])
                a, b, c = received[i].split(args.dim, -1)
                y = F.scaled_dot_product_attention(
                    a.transpose(1, 2), b.transpose(1, 2), c.transpose(1, 2)
                )
                attention_outputs[i] = y.transpose(1, 2).contiguous()
                computed[i].record()
            with torch.cuda.stream(outs):
                outs.wait_event(computed[i])
                cout.gather_output_head_chunk(
                    attention_outputs[i],
                    local_heads=local_heads,
                    head_offset=offset,
                    out=output,
                    workspace=wo,
                )
                attention_outputs[i].record_stream(outs)
            offset += count
        caller.wait_stream(outs)
        return output

    expected = run(False).clone()
    actual = run(True).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    for _ in range(5):
        run(False)
        run(True)
        torch.cuda.synchronize()
    samples = {"whole_projection": [], "coarse_projection": []}
    for i in range(args.iterations):
        order = (("whole_projection", False), ("coarse_projection", True))
        for name, mode in order if i % 2 == 0 else reversed(order):
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            run(mode)
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[name].append(elapsed.item())
    if rank == 0:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "world": world,
                    "shape": vars(args),
                    "scope": "projection + dense SDPA pipeline, no norm/RoPE/output projection",
                    "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                    "samples_ms": samples,
                },
                indent=2,
            )
        )
    torch.cuda.synchronize()
    cin.close()
    cout.close()
    dist.destroy_process_group(gin)
    dist.destroy_process_group(gout)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
