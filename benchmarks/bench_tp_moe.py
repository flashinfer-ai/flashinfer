# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark :func:`flashinfer.fused_moe.tp_moe` (one fused TP MoE layer per call) on tp_size GPUs.

Protocol, per M: one call captured in a CUDA graph; every replay is preceded by a 512 MiB L2 flush, an NCCL all-reduce that lines
the ranks up and a short GPU sleep, and timed with events around the call inside the graph.  Reported: the median over the
rounds on each rank, then the max over ranks (the layer finishes when its slowest rank does).

  python benchmarks/bench_tp_moe.py --model dsv4_flash --tp-size 8
  python benchmarks/bench_tp_moe.py --model dsv41_flash --tp-size 4 --ms 8,32,64 --entry ids
"""

import argparse
import json
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _block_fp8(w, blk):
    n, k = w.shape
    v = w.view(n // blk, blk, k // blk, blk)
    s = torch.exp2(
        torch.ceil(torch.log2(v.abs().amax(dim=(1, 3)).clamp_min(1e-30) / 448.0))
    )
    return (v / s[:, None, :, None]).to(torch.float8_e4m3fn).view(
        n, k
    ).contiguous(), s.contiguous()


def _checkpoint(shape, model, dev):
    """synthetic checkpoint tensors: MXFP4 routed experts (UE8M0 scales 2^-8 .. 2^-3), block-FP8 shared expert"""
    H, I, E, Is = (
        shape.hidden_size,
        shape.intermediate_size,
        shape.num_experts,
        shape.shared_intermediate_size,
    )
    g = torch.Generator(device=dev).manual_seed(1)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g
    )
    s13 = torch.randint(
        119, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g
    )
    s2 = torch.randint(
        119, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g
    )
    blk = 128 if model == "dsv4_flash" else 32
    sh13 = _block_fp8(torch.randn(2 * Is, H, device=dev, generator=g) * 0.02, blk)
    sh2 = _block_fp8(torch.randn(H, Is, device=dev, generator=g) * 0.02, blk)
    return (w13, s13, w2, s2) + sh13 + sh2


def _worker(rank, args, port, results):
    from flashinfer.fused_moe import tp_moe as T

    world = args.tp_size
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        device_id=dev,
    )
    cfg = T.TpMoeConfig(args.model, world)
    shp = cfg.shape
    T.tp_moe_warmup(cfg, args.ms, device=dev)
    ws = T.TpMoeWorkspace(cfg, dist.group.WORLD, dev)
    weights = T.tp_moe_prepare_weights(
        cfg, rank, *_checkpoint(shp, args.model, dev), device=dev
    )
    g = torch.Generator(device=dev).manual_seed(2)
    bias = torch.randn(shp.num_experts, device=dev, generator=g) * 0.05
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    sync = torch.zeros(1, dtype=torch.int32, device=dev)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    torch.cuda._sleep(1_000_000)
    b.record()
    b.synchronize()
    sleep_cycles = int(args.sleep_us * 1_000_000 / (a.elapsed_time(b) * 1e3))
    rows = []
    for m in args.ms:
        x = (torch.randn(m, shp.hidden_size, device=dev, generator=g) * 0.5).bfloat16()
        logits = torch.randn(m, shp.num_experts, device=dev, generator=g)
        out = torch.empty_like(x)
        kw = dict(router_logits=logits, gate_bias=bias)
        if args.model == "dsv41_flash":
            kw.update(
                gate_bias_vision=bias,
                input_ids=torch.zeros(m, dtype=torch.int32, device=dev),
                image_token_id=-1,
            )
        if args.entry == "ids":
            scores = torch.sqrt(torch.nn.functional.softplus(logits))
            ids = torch.topk(scores + bias, shp.top_k, dim=-1).indices
            w = scores.gather(1, ids)
            kw = dict(
                topk_ids=ids.to(torch.int32),
                topk_weights=(w / w.sum(-1, keepdim=True)).float(),
            )
        for _ in range(3):
            T.tp_moe(x, weights, ws, out=out, **kw)
        ev0, ev1 = (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph):
            # an arch module that clears per-token flags once per graph (one memset node per graph, not per layer) does it here,
            # outside the timed region
            reset = getattr(ws._impl, "reset_token_flags", None)
            if reset is not None:
                reset(ws.state)
            ev0.record()
            T.tp_moe(x, weights, ws, out=out, **kw)
            ev1.record()
        times = []
        for _ in range(args.rounds):
            flush.zero_()
            dist.all_reduce(sync)
            torch.cuda._sleep(sleep_cycles)
            graph.replay()
            torch.cuda.synchronize()
            times.append(ev0.elapsed_time(ev1) * 1e3)
        med = [None] * world
        dist.all_gather_object(med, statistics.median(times))
        rows.append(dict(m=m, us=max(med), per_rank=med))
        del graph
    if rank == 0:
        results.extend(rows)
    ws.destroy()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", choices=["dsv4_flash", "dsv41_flash"], default="dsv4_flash"
    )
    ap.add_argument("--tp-size", type=int, choices=[4, 8], default=8)
    ap.add_argument(
        "--ms",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[1, 8, 16, 24, 32, 48, 64],
    )
    ap.add_argument("--entry", choices=["gate", "ids"], default="gate")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--sleep-us", type=float, default=200.0)
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args()
    with mp.Manager() as mgr:
        results = mgr.list()
        mp.spawn(
            _worker, args=(args, _free_port(), results), nprocs=args.tp_size, join=True
        )
        rows = list(results)
    print(
        f"tp_moe {args.model} TP{args.tp_size} ({args.entry} entry): us per call, max over ranks of the per-rank median"
    )
    print("     M        us")
    for r in rows:
        print(f"{r['m']:6d} {r['us']:9.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
