"""Benchmark the SM90 FP8 pull MegaMoE kernel across nodes (NCCL GIN) vs one node.

Modes (one process per GPU under torchrun):

* ``--mode internode``: ``ranks_per_node = LOCAL_WORLD_SIZE`` and the NCCL GIN
  rail-proxy path (several nodes, e.g. 2 x 4 GPUs).  Needs nccl4py with the
  CuTe GIN device API and a matching NCCL (``NCCL_HOME``).
* ``--mode intranode``: the same EP world inside one NVLink domain (NVSHMEM
  symmetric heap), the all-NVLink reference for the same per-rank problem.

Each point stages fresh routed inputs, runs warmup launches, then times
``iters`` launches of ``hopper_fp8_mega_moe`` (dispatch + FC1 + FC2 + combine +
TopkReduce) with a barrier and a device sync around every sample.  The table
reports, per token count, the median over iterations of the slowest rank.

    torchrun --nnodes 2 --node_rank $NODE --nproc_per_node 4 --master_addr $M \\
        benchmarks/bench_moe_ep_sm90_mega_internode.py --mode internode
    torchrun --nproc_per_node 8 benchmarks/bench_moe_ep_sm90_mega_internode.py --mode intranode
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def _args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["internode", "intranode"], required=True)
    p.add_argument("--tokens", default="8,64,256,1024,4096")
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=3072)
    p.add_argument(
        "--scale-mode", choices=["per_tensor", "blockwise"], default="per_tensor"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output-csv", default=None)
    p.add_argument(
        "--stamps",
        action="store_true",
        help="internode only: record a per-phase %%globaltimer timeline of one "
        "extra launch (developer stamps compiled into the kernel)",
    )
    return p.parse_args()


def _weights(rank, experts, hidden, intermediate, scale_mode):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )

    e4m3_max = 448.0
    act_scale = 8.0 / (0.95 * e4m3_max)
    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.empty(
        experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    for e in range(experts):  # per-expert fill keeps the bf16 temporaries small
        w13[e].normal_(generator=g).mul_(hidden**-0.5)
        w2[e].normal_(generator=g).mul_(intermediate**-0.5)
    out = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        kind="fp8_e4m3",
        fp8_scale_mode=scale_mode,
        fc1_activation_dequant_scale=act_scale,
        fc2_activation_dequant_scale=act_scale,
    )
    del w13, w2
    torch.cuda.empty_cache()
    return out, act_scale


def _inputs(rank, tokens, hidden, experts_total, topk):
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    hs = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g)
    scores = torch.randn(
        tokens, experts_total, dtype=torch.float32, device="cuda", generator=g
    )
    w, ids = torch.topk(scores, topk, dim=-1)
    return hs, torch.softmax(w, dim=-1).float(), ids.to(torch.int64)


_PHASES = {
    1: "own prep",
    2: "D2 sync",
    3: "D3 puts (CTA0)",
    4: "D5 META (CTA0)",
    5: "D6 sync",
    6: "D7 proxy prep",
    7: "D8 sync",
    8: "D10 node barrier",
    9: "pull pass0 done",
    10: "PAY wait (CTA0)",
    11: "payload flags",
    12: "pull done",
    20: "tail entry (GEMM done)",
    21: "B1",
    22: "node reduce",
    23: "reduce sync",
    25: "RET wait (CTA0)",
    27: "tail done",
}


def _print_timeline(stamps, rank, tokens):
    """Per phase: CTA0 time and the min / max over CTAs, us from the first stamp."""
    raw = stamps.view(256, 64)[:200].to(torch.int64)
    hit = raw != 0
    t0 = int(raw[hit].min())
    rel = ((raw - t0) % (1 << 31)).double() / 1e3
    parts = []
    for p, name in sorted(_PHASES.items()):
        col = hit[:, p]
        if not bool(col.any()):
            continue
        vals = rel[col, p]
        cta0 = f"{rel[0, p]:.1f}" if bool(hit[0, p]) else "-"
        parts.append(f"{name}: cta0 {cta0} min {vals.min():.1f} max {vals.max():.1f}")
    print(f"[timeline rank {rank} T={tokens}] " + " | ".join(parts), flush=True)


def main():
    args = _args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    torch.cuda.set_device(local)

    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_fp8_mega_moe,
        hopper_fp8_mega_moe,
    )

    stamps = None
    if args.mode == "internode":
        if world <= per_node:
            raise SystemExit("--mode internode needs a multi-node torchrun job")
        dist.init_process_group("gloo")
        gin_kwargs = dict(ranks_per_node=per_node, enable_nccl_gin=True)
        if args.stamps:
            stamps = torch.zeros(256 * 64, dtype=torch.int32, pin_memory=True)
            os.environ["MEGA_GIN_DEBUG_ADDR"] = str(stamps.data_ptr())
    else:
        if world != per_node:
            raise SystemExit("--mode intranode runs inside one node")
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
            init_dist,
        )

        init_dist()
        gin_kwargs = {}

    experts_per_rank = args.num_experts // world
    (l1, l2), act_scale = _weights(
        rank, experts_per_rank, args.hidden, args.intermediate, args.scale_mode
    )
    rows = []
    for tokens in [int(t) for t in args.tokens.split(",")]:
        symm = get_symm_buffer_for_hopper_fp8_mega_moe(
            args.num_experts,
            tokens,
            args.top_k,
            args.hidden,
            args.intermediate,
            rank,
            world,
            kind="fp8_e4m3",
            fp8_scale_mode=args.scale_mode,
            gate_up_clamp=10.0,
            **gin_kwargs,
        )
        cfg = symm._frontend.config
        hs, tw, ti = _inputs(rank, tokens, args.hidden, args.num_experts, args.top_k)
        stage_mega_moe_inputs(
            hs,
            tw,
            ti,
            symm.x,
            symm.x_sf,
            symm.topk_idx,
            symm.topk_weights,
            kind="fp8_e4m3",
            fp8_scale_mode=args.scale_mode,
            fc1_activation_dequant_scale=act_scale,
        )
        y = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")

        def call():
            hopper_fp8_mega_moe(
                y,
                l1,
                l2,
                symm,
                num_tokens=tokens,
                gate_up_clamp=10.0,
                fast_math=True,
            )

        align_buf = torch.zeros(1, dtype=torch.float32, device="cuda")

        def align():
            # Device-side rendezvous right before the timed launch, so no rank
            # spends its sample waiting inside the kernel for a late peer (a
            # host barrier leaves launch skew of the order of the kernel).
            if args.mode == "internode":
                from nccl.core import NcclRedOp

                symm._gin_arena.comm.allreduce(
                    align_buf,
                    align_buf,
                    NcclRedOp.SUM,
                    stream=torch.cuda.current_stream().cuda_stream,
                )
            else:
                dist.all_reduce(align_buf)

        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            align()
            start.record()
            call()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) * 1e3)
        med = torch.tensor([statistics.median(samples)], dtype=torch.float64)
        gathered = [torch.zeros_like(med) for _ in range(world)]
        dist.all_gather(gathered, med)
        rank_meds = [float(g.item()) for g in gathered]
        row = dict(
            mode=args.mode,
            world=world,
            per_node=per_node,
            tokens=tokens,
            scale_mode=args.scale_mode,
            swap_ab=cfg.swap_ab,
            tile=cfg.mma_tiler_mnk,
            token_back=cfg.resolved_token_back_mode,
            max_rank_us=max(rank_meds),
            min_rank_us=min(rank_meds),
        )
        rows.append(row)
        if rank == 0:
            print(
                f"BENCH {args.mode} W={world} R={per_node} T={tokens} {args.scale_mode} "
                f"swap_ab={cfg.swap_ab} tile={cfg.mma_tiler_mnk} tb={cfg.resolved_token_back_mode}: "
                f"max-rank median {max(rank_meds):.1f} us (min-rank {min(rank_meds):.1f} us)",
                flush=True,
            )
        if stamps is not None:
            dist.barrier()
            torch.cuda.synchronize()
            align()
            stamps.zero_()
            call()
            torch.cuda.synchronize()
            _print_timeline(stamps, rank, tokens)
        torch.cuda.synchronize()
        dist.barrier()
        symm.destroy()
        torch.cuda.empty_cache()
    if rank == 0 and args.output_csv:
        import csv

        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    dist.barrier()


if __name__ == "__main__":
    main()
