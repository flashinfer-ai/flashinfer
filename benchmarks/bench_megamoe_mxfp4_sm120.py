"""SM120 MXFP4 x MXFP8 MegaMoE timing (CUDA graph): the moe_ep SM120 backends (nosync / cutedsl / cake) through
MoEEpLayer, on the same weights, inputs and routes.

Every backend takes BF16 activations; MoEEpLayer.stage_inputs quantizes them to MXFP8 once, outside the timed
region, and the timed call is compute_staged: the standalone MoE kernel(s), including the dispatch and combine
communication. Same method as bench_megamoe_fp4_sm90.py: each graph queues an unmeasured NCCL all-reduce gate,
then the timing events around `repeats` back-to-back calls; per round the time is the maximum over ranks, and the
median over rounds is reported. --cold-l2 writes 512 MB before every round (with --repeats 1), so the weights come from DRAM as
in a decode step that walks every layer. Routing: random top-k (--routing random) or balanced (--routing balanced): top-k slot s of global token g goes to
rank s % EP, local expert (g + 7 s) % E_local, so every rank gets floor or ceil of top-k / EP routes per token.

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
    "h3072": (3072, 1280, 384, 8),  # the shape of the SM90 / SM100 MegaMoE tables
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
        idx = (s % args.ep) * epr + (g + 7 * s) % epr
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
    if args.backend == "nosync":
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig as Config,
        )

        cfg = Config(
            intermediate_size=ih,
            top_k=topk,
            transport=args.nosync_transport,
            tokens_per_item=args.tokens_per_item,
        )
    elif args.backend == "cutedsl":
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig as Config,
        )

        cfg = Config(
            intermediate_size=ih,
            top_k=topk,
            gate_up_clamp=10.0,
            knobs=json.loads(args.knobs) if args.knobs else None,
        )
    else:
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig as Config,
        )

        cfg = Config(
            intermediate_size=ih,
            top_k=topk,
            swiglu_limit=10,
            transport_table=args.transport,
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
    )

    def setup(m, x_bf16, idx, weights_):
        t = MoEEpTensors(hidden_states=x_bf16, topk_ids=idx, topk_weights=weights_)
        layer.stage_inputs(t, compile_tokens_per_rank=m)
        return lambda: layer.compute_staged(output=layer.output_buffer)

    return setup, layer.destroy


def _worker(rank, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=args.ep,
        device_id=torch.device("cuda", rank),
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
    for m in args.tokens:
        x = torch.randn(m, h, device="cuda", generator=generator, dtype=torch.bfloat16)
        idx, weights = _routes(args, rank, m, experts, topk, generator)
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
        if rank == 0:
            print(
                f"{args.backend} {args.shape} EP{args.ep} M={m} routing={'saved' if args.load_routing else args.routing} "
                f"{'cold' if args.cold_l2 else 'warm'} L2: {us:.1f} us",
                flush=True,
            )
        del graph
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
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="flash")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 8, 16, 20, 32, 64]
    )
    parser.add_argument("--capacity", type=int, default=128)
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
        help="moe_ep backend sm120_mxfp4_mxfp8_bf16_nosync / _cutedsl / _cake",
    )
    parser.add_argument("--knobs", type=str, default="", help="cutedsl: JSON knobs")
    parser.add_argument(
        "--transport", type=str, default="all_lsa", help="cake: transport table"
    )
    parser.add_argument(
        "--tokens-per-item",
        type=int,
        choices=[8, 16, 24, 32],
        default=None,
        help="nosync: force the work-item width (default: by routed rows per local expert)",
    )
    parser.add_argument(
        "--nosync-transport",
        choices=["p2p", "nic", "auto"],
        default="p2p",
        help="nosync: transport",
    )
    args = parser.parse_args()
    if args.cold_l2 and args.repeats != 1:
        parser.error("cold L2 timing requires --repeats 1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)
