"""MoE Expert-Parallel benchmark: dispatch → compute → combine.

Canonical cases mirror the NCCL-EP ``ep_bench`` reference
(``contrib/nccl_ep/README.md`` in the NCCL repo): **128 tokens/rank, hidden 7168,
top-k 8, 256 experts, BF16**, swept over **8/16/32/64 GPUs** and over the two EP
**algorithms — Low-Latency (LL) and High-Throughput (HT)** (one table each).
Select that geometry with ``--reference`` and the algorithm with
``--algorithm {ll,ht}``; the GPU count comes from the torchrun world size.

Launch (one process per GPU; 4 GPU/node on GB200 → N nodes = GPUs/4):

    # LL, EXPERT_MAJOR (default), reference geometry, 8 GPUs (2 GB200 nodes):
    torchrun --nnodes 2 --nproc_per_node 4 --rdzv_backend c10d \
        --rdzv_endpoint $HOST:$PORT benchmarks/bench_moe_ep.py \
        --reference --algorithm ll --backend nccl_ep --quant bf16

    # LL, RANK_MAJOR variant: add --layout rank_major
    # HT variant: --algorithm ht
    # Ad-hoc (non-reference) sizing still works via --tokens / --tokens-per-rank.

``--tokens-per-rank`` (the reference uses 128) fixes per-rank work as GPUs scale
and takes precedence over the global ``--tokens``. Rank 0 prints one CSV row
(prefixed with the algorithm + layout). ``--baseline`` times the comm-only
identity path. Weights are random but correctly shaped/typed — this measures
latency, not accuracy (correctness is covered by tests/moe_ep/).

Note: NCCL-EP HT uses the FLAT receive layout; the LL path supports two layouts,
selected with ``--layout {expert_major,rank_major}``:
  * EXPERT_MAJOR — recv [num_local_experts, per_rank*world, hidden]; every padded
    row is pre-assigned to one expert (the inner compute is a top_k=1 batch).
  * RANK_MAJOR — recv [world, per_rank, hidden]; tokens grouped by source rank.
    The inner compute is driven by the library's received per-token routing at the
    model's real top_k (do_finalize pre-reduce), with non-local picks masked to
    weight 0 — far less padded work than EXPERT_MAJOR, which is the point of
    measuring it.
``--algorithm ht`` exercises HT FLAT (``--layout`` ignored). For the HT
``nccl_ep_b200_ib`` config, add ``--ep-test-geometry`` (rank-derived
top_k=min(8,world), num_experts=min(256, top_k*world), per contrib/nccl_ep/
ep_test.py) and sweep ``--tokens-per-rank {4096,8192}``.

The CSV reports BOTH per-stage latency (µs) and dispatch/combine bandwidth (GB/s,
ep_bench send-side convention: unique (token,node) selections x hidden x dtype /
stage time; ``*_rdma_*`` = remote-node only).
"""

from __future__ import annotations

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


# Canonical benchmark cases, mirroring the NCCL-EP `ep_bench` reference
# (contrib/nccl_ep/README.md): BF16 dispatch+combine, LL mode, 128 tokens/rank,
# hidden 7168, top-k 8, 256 experts, swept over 8/16/32/64 GPUs (1/2/4/8 nodes @
# 8 GPU/node; on GB200 that's 2/4/8/16 nodes @ 4 GPU/node). `--reference` selects
# this geometry; the GPU count comes from the launcher (torchrun world size).
#
# Algorithm: NCCL-EP supports two algorithms — Low-Latency (LL) and
# High-Throughput (HT). The reference table is LL-only. `--algorithm {ll,ht}`
# selects which to benchmark; run once per algorithm to produce the LL and HT
# tables. (HT requires the backend's FLAT-layout handle path.)
_REFERENCE = dict(
    num_experts=256, top_k=8, hidden=7168, intermediate=2048, tokens_per_rank=128
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["nccl_ep", "nixl_ep"], default="nccl_ep")
    p.add_argument(
        "--algorithm",
        choices=["ll", "ht"],
        default="ll",
        help="EP algorithm: ll = Low-Latency, ht = High-Throughput",
    )
    p.add_argument("--quant", choices=["nvfp4", "bf16"], default="bf16")
    p.add_argument(
        "--layout",
        choices=["expert_major", "rank_major"],
        default="expert_major",
        help="LL receive layout (ll only; ht always uses FLAT). expert_major: "
        "recv [num_local_experts, per_rank*world, hidden]; rank_major: recv "
        "[world, per_rank, hidden] (far less padded compute)",
    )
    p.add_argument(
        "--reference",
        action="store_true",
        help="use the ep_bench reference geometry (hidden=7168, experts=256, "
        "top_k=8, 128 tokens/rank)",
    )
    p.add_argument(
        "--ep-test-geometry",
        action="store_true",
        help="derive top_k / num_experts from world size like contrib/nccl_ep/"
        "ep_test.py: top_k=min(8,world), num_experts=min(256, top_k*world). This is "
        "the geometry the nccl_ep_b200_ib HT benchmark uses (resolved in main() once "
        "world size is known); overrides --num-experts / --top-k.",
    )
    # Sizing. --tokens-per-rank (preferred, matches the reference) wins over the
    # global --tokens; the reference uses 128 tokens/rank.
    p.add_argument("--tokens-per-rank", type=int, default=None)
    p.add_argument("--tokens", type=int, default=8192, help="global token count")
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--baseline",
        action="store_true",
        help="time the comm-only identity path (no compute_config)",
    )
    args = p.parse_args()
    if args.reference:
        args.num_experts = _REFERENCE["num_experts"]
        args.top_k = _REFERENCE["top_k"]
        args.hidden = _REFERENCE["hidden"]
        args.intermediate = _REFERENCE["intermediate"]
        if args.tokens_per_rank is None:
            args.tokens_per_rank = _REFERENCE["tokens_per_rank"]
    return args


def _build_compute(args, *, local_num_experts, local_expert_offset, max_tokens, device):
    """Build (MoEConfig, MoEWeightPack) for the requested quant, or (None, None)."""
    import torch

    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoEWeightPack,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    routing = RoutingConfig(num_experts=args.num_experts, top_k=args.top_k)
    experts = ExpertConfig(
        intermediate_size=args.intermediate,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    execution = ExecutionConfig(tune_max_num_tokens=max_tokens)

    w1 = torch.randn(
        local_num_experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randn(
        local_num_experts,
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    wp = MoEWeightPack()

    if args.quant == "nvfp4":
        cfg = MoEConfig(
            routing=routing,
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=experts,
            backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
            execution=execution,
        )
        wp.prepare_for(
            "trtllm_fp4_routed",
            TrtllmFp4Config.prepare_weights(
                w1,
                w2,
                num_local_experts=local_num_experts,
                hidden_size=args.hidden,
                intermediate_size=args.intermediate,
                device=device,
            ),
        )
        wp.prepare_for(
            "cute_dsl_nvfp4",
            CuteDslConfig.prepare_weights(
                w1,
                w2,
                num_local_experts=local_num_experts,
                hidden_size=args.hidden,
                intermediate_size=args.intermediate,
                device=device,
            ),
        )
    else:  # bf16
        from flashinfer import shuffle_matrix_a
        from flashinfer.fused_moe.core import convert_to_block_layout

        # BlockMajorK shuffled weights for the trtllm bf16 routed runner. Recipe
        # validated against tests/moe/test_trtllm_gen_routed_fused_moe.py
        # (epilogue_tile_m=64, block_k=128). Note: epilogue_tile_m only permutes
        # weight rows (same FLOPs), so it doesn't affect timing — but 64 is the
        # numerically-correct value (128 produces wrong output; see
        # tests/moe_ep/test_moe_ep_compute_correctness.py).
        def _block_major_k(w):
            epilogue_tile_m = 64
            block_k = 128
            shuffled = []
            for i in range(w.shape[0]):
                s = shuffle_matrix_a(w[i].view(torch.uint8), epilogue_tile_m)
                s = convert_to_block_layout(s, block_k)
                shuffled.append(s)
            return torch.stack(shuffled).view(torch.bfloat16)

        cfg = MoEConfig(
            routing=routing,
            quant=QuantConfig(variant=QuantVariant.BF16),
            experts=experts,
            backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
            execution=execution,
        )
        wp.prepare_for(
            "trtllm_bf16_routed",
            {
                "gemm1_weights": _block_major_k(w1),
                "gemm2_weights": _block_major_k(w2),
            },
        )
    return cfg, wp


def main() -> int:
    args = _parse_args()

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
    )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --tokens-per-rank (the reference uses 128/rank) takes precedence over the
    # global --tokens, so the per-rank transport work is fixed as GPUs scale.
    if args.tokens_per_rank is not None:
        per_rank = args.tokens_per_rank
        args.tokens = per_rank * world_size
    else:
        per_rank = args.tokens // world_size
    # ep_test.py geometry (the nccl_ep_b200_ib HT config): rank-derived experts/top_k.
    if args.ep_test_geometry:
        args.top_k = min(8, world_size)
        args.num_experts = min(256, args.top_k * world_size)
    local_num_experts = args.num_experts // world_size
    local_expert_offset = rank * local_num_experts
    ep_algorithm = (
        EpAlgorithm.HIGH_THROUGHPUT
        if args.algorithm == "ht"
        else EpAlgorithm.LOW_LATENCY
    )
    ep_layout = (
        EpLayout.RANK_MAJOR if args.layout == "rank_major" else EpLayout.EXPERT_MAJOR
    )
    # EpLayout exposes only the two LL receive layouts. HT always uses the library's
    # FLAT layout internally, so FleetParams.layout is ignored for HT — set the inert
    # EXPERT_MAJOR placeholder (RANK_MAJOR is LL-only).
    if ep_algorithm is EpAlgorithm.HIGH_THROUGHPUT:
        ep_layout = EpLayout.EXPERT_MAJOR

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(
        per_rank, args.hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    # DISTINCT top-k experts per token (via topk over random scores), matching
    # contrib/nccl_ep/ep_test.py. Plain randint draws WITH replacement, so a token
    # can pick the same expert twice — which the HT cross-node path mishandles
    # (illegal access at nccl_ep.cc:2884); distinct picks avoid it and are realistic.
    _scores = torch.randn(per_rank, args.num_experts, device=device, generator=g)
    topk_ids = _scores.topk(args.top_k, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(per_rank, args.top_k, device=device, generator=g), dim=-1
    )

    # NIXL-EP needs a TCPStore for its Buffer rendezvous (NCCL-EP ignores it).
    # Open it on a sibling port to torch.distributed's MASTER_PORT, mirroring
    # tests/moe_ep/smoke_nixl_ep.py.
    tcp_store = None
    if args.backend == "nixl_ep":
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        tcp_store = dist.TCPStore(
            host_name=master_addr,
            port=master_port + 1,
            world_size=world_size,
            is_master=(rank == 0),
        )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        tcp_store=tcp_store,
    )
    fleet_params = FleetParams(
        num_experts=args.num_experts,
        max_tokens_per_rank=per_rank,
        token_hidden_size=args.hidden,
        dtype_bytes=2,
        algorithm=ep_algorithm,
        layout=ep_layout,
    )

    # Compute batch size differs by layout: EXPERT_MAJOR pads to
    # num_local_experts * (per_rank * world); RANK_MAJOR processes per_rank * world
    # received tokens (round-robin to one local expert each).
    if ep_layout is EpLayout.RANK_MAJOR:
        compute_max_tokens = per_rank * world_size
    else:
        compute_max_tokens = local_num_experts * per_rank * world_size

    if args.baseline:
        compute_config, weights = None, None
    else:
        compute_config, weights = _build_compute(
            args,
            local_num_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            max_tokens=compute_max_tokens,
            device=device,
        )

    layer = MoEEpLayer(
        bootstrap,
        fleet_params,
        backend=args.backend,
        compute_config=compute_config,
        weights=weights,
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)

    # The EP forward is effectively synchronous (dispatch host-syncs to read the
    # per-expert recv counts), which defeats the async-stream assumptions of
    # bench_gpu_time. Time it with a synchronized wall-clock loop instead, and
    # collect per-stage GPU times from the layer's opt-in CUDA-event timing.
    from statistics import median
    from time import perf_counter

    layer.enable_timing = True

    # Warmup (includes the cross-backend autotune pass + JIT).
    for _ in range(args.warmup):
        layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    e2e_us_samples: list[float] = []
    disp_us: list[float] = []
    comp_us: list[float] = []
    comb_us: list[float] = []
    for _ in range(args.repeat):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = perf_counter()
        layer.forward(t)
        torch.cuda.synchronize()
        e2e_us_samples.append((perf_counter() - t0) * 1e6)
        tm = layer.last_timings_ms
        disp_us.append(tm.get("dispatch", 0.0) * 1e3)
        comp_us.append(tm.get("compute", 0.0) * 1e3)
        comb_us.append(tm.get("combine", 0.0) * 1e3)

    e2e_us = median(e2e_us_samples)
    d_us, cp_us, cb_us = median(disp_us), median(comp_us), median(comb_us)
    # tok/s on the global batch (all ranks dispatch `tokens` total).
    tok_s = (args.tokens / (e2e_us * 1e-6)) if e2e_us > 0 else float("nan")

    # Dispatch/combine bandwidth (GB/s), ep_bench send-side convention: bytes moved
    # = unique (token, node) selections * hidden * dtype_bytes; rdma = remote-node
    # only. GB are decimal (bytes / 1e9), matching contrib/nccl_ep/ep_bench.cu. This
    # is a per-rank (rank 0) figure. (combine returns the same data volume.)
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", min(world_size, 8)))
    num_nodes = max(1, world_size // ranks_per_node)
    experts_per_node = max(1, args.num_experts // num_nodes)
    my_node = rank // ranks_per_node
    node_of_expert = (topk_ids // experts_per_node).clamp_(0, num_nodes - 1)
    onehot = torch.zeros(per_rank, num_nodes, dtype=torch.bool, device=device)
    onehot.scatter_(1, node_of_expert, True)
    send_tokens = int(onehot.sum().item())
    remote = onehot.clone()
    remote[:, my_node] = False
    rdma_tokens = int(remote.sum().item())
    send_bytes = send_tokens * args.hidden * 2
    rdma_bytes = rdma_tokens * args.hidden * 2
    disp_s, comb_s = d_us * 1e-6, cb_us * 1e-6

    def _gbps(b, s):
        return (b / 1e9) / s if s > 0 else 0.0

    disp_gbps, disp_rdma = _gbps(send_bytes, disp_s), _gbps(rdma_bytes, disp_s)
    comb_gbps, comb_rdma = _gbps(send_bytes, comb_s), _gbps(rdma_bytes, comb_s)

    dist.barrier()
    if rank == 0:
        mode = "identity" if args.baseline else args.quant
        layout_name = "ht_flat" if args.algorithm == "ht" else args.layout
        print(
            "BENCH_CSV,algo,layout,tokens,gpus,backend,quant,dispatch_us,compute_us,combine_us,e2e_us,tok_s,disp_gbps,disp_rdma_gbps,comb_gbps,comb_rdma_gbps\n"
            f"BENCH_CSV,{args.algorithm},{layout_name},{args.tokens},{world_size},{args.backend},{mode},"
            f"{d_us:.1f},{cp_us:.1f},{cb_us:.1f},{e2e_us:.1f},{tok_s:.1f},"
            f"{disp_gbps:.1f},{disp_rdma:.1f},{comb_gbps:.1f},{comb_rdma:.1f}"
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
