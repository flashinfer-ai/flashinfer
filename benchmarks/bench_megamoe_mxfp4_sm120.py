"""SM120 MXFP4 x MXFP8 MegaMoE timing (CUDA graph): the moe_ep SM120 backends (nosync / cutedsl / cake) through
MoEEpLayer, on the same weights, inputs and routes.

Every backend takes BF16 activations; MoEEpLayer.stage_inputs quantizes them to MXFP8 once, outside the timed
region, and the timed call is compute_staged: the standalone MoE kernel(s), including the dispatch and combine
communication. Same method as bench_megamoe_fp4_sm90.py: each graph queues an unmeasured NCCL all-reduce gate,
then the timing events around `repeats` back-to-back calls; per round the time is the maximum over ranks, and the
median over rounds is reported. --cold-l2 writes 512 MB before every round (with --repeats 1), so the weights come from DRAM as
in a decode step that walks every layer. Routing: random top-k (--routing random) or balanced (--routing balanced): top-k slot s of global token g goes to
rank s * (EP / top-k) + g % (EP / top-k) when top-k divides EP, else rank (g * top-k + s) % EP, local expert
(g + 7 s) % E_local, so every rank receives the same number of routes (± 1) and a token's copies go to different ranks.
--route-calls replays recorded routes instead: rank<r>.pt holds a list of (layer, rows of each rank, expert ids
[rows, top-k]) per MoE call (e.g. a serving engine's decode steps), one timed call each; --rows n keeps each call's
first n tokens per rank (the same routing at a smaller batch). Groups across hosts: run the command on every host with
NODE_RANK (0 on the host of MASTER_ADDR) and --ep = hosts x GPUs per host.

    python benchmarks/bench_megamoe_mxfp4_sm120.py --ep 4 --shape flash --cold-l2 --repeats 1 --tokens 1 8 32 128
"""

import argparse
import datetime
import json
import os
import statistics

import torch
import torch.distributed as dist

SHAPES = {  # hidden, intermediate, experts, top-k
    "flash": (4096, 2048, 256, 6),  # DeepSeek-V4-Flash routed experts
    "flash41": (5120, 2304, 384, 6),  # DeepSeek-V4.1-Flash routed experts
    "glm53flash": (4096, 2048, 288, 8),  # GLM-5.3-Flash routed experts
}


def _capture(run, repeats):
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        dist.all_reduce(gate)
        start.record()
        for _ in range(repeats):
            run()
        end.record()
    return graph, start, end


def _routes(args, rank, m, experts, topk, generator):
    if args.load_routing:
        r = torch.load(os.path.join(args.load_routing, f"routing_T{m}_r{rank}.pt"))
        return r["ids"].cuda().long(), r["w"].cuda().float()
    epr = experts // args.ep
    if args.routing == "balanced":
        g = rank * m + torch.arange(m, device="cuda")[:, None]
        s = torch.arange(topk, device="cuda")[None, :]
        if args.ep % topk == 0:  # copy s of token g: rank s * stride + g % stride
            stride = args.ep // topk
            dest = s * stride + g % stride
        else:
            dest = (g * topk + s) % args.ep
        idx = dest * epr + (g + 7 * s) % epr
    else:
        idx = (
            torch.randn(m, experts, device="cuda", generator=generator)
            .topk(topk, dim=-1)
            .indices
        )
    return idx, torch.softmax(
        torch.randn(m, topk, device="cuda", generator=generator), -1
    )


def _moe_ep(args, h, ih, experts, topk, generator, raw):
    """The moe_ep SM120 MXFP4 x MXFP8 MegaMoE backends through MoEEpLayer: BF16 inputs staged (and quantized) once
    by stage_inputs, the timed call is compute_staged."""
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        PrequantizedMoEWeights,
    )

    p1, s1 = raw(2 * ih, h)
    p2, s2 = raw(h, ih)
    weights = PrequantizedMoEWeights(
        w13=p1.view(torch.float4_e2m1fn_x2),
        w2=p2.view(torch.float4_e2m1fn_x2),
        w13_scale=s1.view(torch.float8_e8m0fnu),
        w2_scale=s2.view(torch.float8_e8m0fnu),
    )

    def fallback_config():
        if args.backend == "cutedsl" or (
            args.backend == "nosync" and args.fallback == "cutedsl"
        ):
            from flashinfer.moe_ep import (
                Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig as Config,
            )

            return Config(
                intermediate_size=ih,
                top_k=topk,
                gate_up_clamp=10.0,
                knobs=json.loads(args.knobs) if args.knobs else None,
            )
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig as Config,
        )

        return Config(
            intermediate_size=ih,
            top_k=topk,
            swiglu_limit=10,
            transport_table=args.transport,
        )

    cfg = fallback_config()
    nosync = None
    if args.backend == "nosync":  # the configured backend plus the small-batch option
        from flashinfer.moe_ep import NosyncOptions

        nosync = NosyncOptions(
            max_tokens_per_rank=args.capacity,
            transport=args.nosync_transport,
            tokens_per_item=args.tokens_per_item,
            max_tokens_per_expert=args.max_tokens_per_expert,
        )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=args.ep,
            rank=dist.get_rank(),
            device=torch.cuda.current_device(),
            process_group=dist.group.WORLD,
        ),
        fleet_params=FleetParams(
            num_experts=experts, max_tokens_per_rank=args.capacity, token_hidden_size=h
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=cfg, quantize_input=True, preprocess_weights=True
        ),
        nosync=nosync,
    )

    def setup(m, x_bf16, idx, weights_):
        t = MoEEpTensors(hidden_states=x_bf16, topk_ids=idx, topk_weights=weights_)
        layer.stage_inputs(t, compile_tokens_per_rank=m)
        return lambda: layer.compute_staged(output=layer.output_buffer)

    return setup, layer.destroy


def _recorded_calls(args, rank, topk):
    """(tokens per rank to compile for, expert ids) of each recorded call this rank replays."""
    first, last = (int(v) for v in args.calls.split(":"))
    calls = torch.load(os.path.join(args.route_calls, f"rank{rank}.pt"))[first:last]
    items = []
    for _, rows, ids in calls:
        ids = ids.long().reshape(-1, topk)
        m = max(rows)  # every rank compiles for the largest rank's rows
        if args.rows:
            ids, m = ids[: args.rows], min(m, args.rows)
        items.append((m, ids.cuda()))
    return items


def _worker(local, args):
    per_host = (
        torch.cuda.device_count() if args.ep > torch.cuda.device_count() else args.ep
    )
    rank = int(os.environ.get("NODE_RANK", "0")) * per_host + local
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=args.ep,
        device_id=torch.device("cuda", local),
        timeout=datetime.timedelta(minutes=15),
    )
    h, ih, experts, topk = SHAPES[args.shape]
    epr = experts // args.ep
    generator = torch.Generator(device="cuda").manual_seed(7919 + rank)

    def raw(n, k):  # MXFP4: packed E2M1 bytes, UE8M0 scales per 32
        packed = torch.randint(
            0,
            256,
            (epr, n, k // 2),
            device="cuda",
            generator=generator,
            dtype=torch.uint8,
        )
        sf = torch.randint(
            122, 125, (epr, n, k // 32), device="cuda", generator=generator
        ).to(torch.uint8)
        return packed, sf

    setup, destroy = _moe_ep(args, h, ih, experts, topk, generator, raw)
    flush = (
        torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        if args.cold_l2
        else None
    )
    results = []
    if args.route_calls:
        items = _recorded_calls(args, rank, topk)
    else:
        items = [(m, None) for m in args.tokens]
    for m, recorded in items:
        if recorded is None:
            x = torch.randn(
                m, h, device="cuda", generator=generator, dtype=torch.bfloat16
            )
            idx, weights = _routes(args, rank, m, experts, topk, generator)
        else:
            idx = recorded
            weights = torch.full(idx.shape, 1.0 / topk, device="cuda")
            x = torch.randn(
                idx.shape[0],
                h,
                device="cuda",
                generator=generator,
                dtype=torch.bfloat16,
            )
        if args.save_routing:  # for another backend's run on identical routes
            os.makedirs(args.save_routing, exist_ok=True)
            torch.save(
                {"ids": idx.int().cpu(), "w": weights.cpu()},
                os.path.join(args.save_routing, f"routing_T{m}_r{rank}.pt"),
            )
        run = setup(m, x, idx, weights)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph, start, end = _capture(run, args.repeats)
        times = []
        for _ in range(args.rounds):
            if flush is not None:
                flush.fill_(1)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            t = torch.tensor(
                [start.elapsed_time(end) * 1e3 / args.repeats], device="cuda"
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            times.append(t.item())
        us = statistics.median(times)
        results.append({"tokens_per_rank": m, "us": us})
        routing = (
            "recorded"
            if args.route_calls
            else "saved"
            if args.load_routing
            else args.routing
        )
        if rank == 0:
            print(
                f"{args.backend} {args.shape} EP{args.ep} M={m} routing={routing} "
                f"{'cold' if args.cold_l2 else 'warm'} L2: {us:.1f} us",
                flush=True,
            )
        del graph
    if rank == 0 and args.route_calls:
        us = [r["us"] for r in results]
        print(
            f"{args.backend} {args.shape} EP{args.ep} recorded calls {args.calls}: mean "
            f"{statistics.mean(us):.1f} median {statistics.median(us):.1f} min {min(us):.1f} "
            f"max {max(us):.1f} us",
            flush=True,
        )
    if rank == 0 and args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "args": vars(args),
                    "results": results,
                },
                f,
                indent=2,
            )
    destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=[2, 4, 8, 16, 24, 32], default=4)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="flash")
    parser.add_argument(
        "--route-calls",
        type=str,
        default="",
        help="directory of rank<r>.pt recorded routes (see the module docstring)",
    )
    parser.add_argument(
        "--calls", type=str, default="0:40", help="first:last recorded call to replay"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="recorded calls: keep each call's first n tokens per rank (0: all)",
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 8, 16, 20, 32, 64]
    )
    parser.add_argument("--capacity", type=int, default=224)
    parser.add_argument("--routing", choices=["random", "balanced"], default="random")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--save-routing",
        type=str,
        default="",
        help="directory for each rank's routes per M",
    )
    parser.add_argument(
        "--load-routing", type=str, default="", help="replay routes another run saved"
    )
    parser.add_argument(
        "--backend",
        choices=["nosync", "cutedsl", "cake"],
        default="nosync",
        help="cutedsl / cake: that moe_ep backend; nosync: the --fallback backend with "
        "MoEEpLayer(nosync=NosyncOptions()), so calls up to 224 tokens per rank run "
        "sm120_mxfp4_mxfp8_bf16_nosync",
    )
    parser.add_argument(
        "--fallback",
        choices=["cake", "cutedsl"],
        default="cake",
        help="nosync: the configured backend (its options: --transport for cake, --knobs for "
        "cutedsl)",
    )
    parser.add_argument("--knobs", type=str, default="", help="cutedsl: JSON knobs")
    parser.add_argument(
        "--transport",
        type=str,
        default="all_lsa",
        help="cake (also as the nosync fallback): transport table",
    )
    parser.add_argument(
        "--tokens-per-item",
        type=int,
        choices=[8, 16, 24, 32, 64],
        default=None,
        help="nosync: force the work-item width (default: by routed rows per local expert)",
    )
    parser.add_argument(
        "--max-tokens-per-expert",
        type=int,
        default=128,
        help="nosync: NosyncOptions.max_tokens_per_expert (larger calls run the --fallback backend)",
    )
    parser.add_argument(
        "--nosync-transport",
        choices=["p2p", "nic", "auto"],
        default=None,
        help="nosync: transport (default: the cake table, see --transport)",
    )
    args = parser.parse_args()
    if args.cold_l2 and args.repeats != 1:
        parser.error("cold L2 timing requires --repeats 1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    torch.multiprocessing.spawn(
        _worker, args=(args,), nprocs=min(args.ep, torch.cuda.device_count())
    )
