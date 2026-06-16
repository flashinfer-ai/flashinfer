"""MoE Expert-Parallel benchmark: dispatch → compute → combine.

Sweeps the matrix in ``docs/design_docs/MoE_EP_verif.md``:
tokens ∈ {8192, 16384} × world-size ∈ {8, 16} × backend ∈ {nccl_ep, nixl_ep}
× quant ∈ {nvfp4, bf16}, on GB200 (SM100) / GB300 (SM103).

Launch (one process per GPU):

    torchrun --nproc_per_node=8 benchmarks/bench_moe_ep.py \
        --tokens 8192 --backend nccl_ep --quant nvfp4

    # 16 GPUs across 2 nodes:
    torchrun --nnodes 2 --nproc_per_node=8 --rdzv_backend c10d \
        --rdzv_endpoint $HOST:$PORT benchmarks/bench_moe_ep.py \
        --tokens 16384 --backend nixl_ep --quant bf16

``--tokens`` is the **global** batch; per-rank tokens (= ``max_tokens_per_rank``)
is ``tokens // world_size``.  Rank 0 prints one CSV row.  Use ``--baseline`` to
time the comm-only identity path (the "vs identity-stub" column).

Weights are random but correctly shaped/typed — this measures latency, not
accuracy (correctness is covered by tests/moe_ep/). NVFP4 weight views are built
with the canonical prepare helpers; bf16 weights use the trtllm BlockMajorK
shuffle.
"""

from __future__ import annotations

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=8192, help="global token count")
    p.add_argument("--backend", choices=["nccl_ep", "nixl_ep"], default="nccl_ep")
    p.add_argument("--quant", choices=["nvfp4", "bf16"], default="nvfp4")
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
    return p.parse_args()


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

        # BlockMajorK shuffled weights for the trtllm bf16 routed runner — mirrors
        # the per-expert recipe in tests/moe/test_dpsk_fused_moe_fp8.py. Validate
        # the exact layout on-cluster; timing is layout-shape-sensitive only.
        def _block_major_k(w):
            epilogue_tile_m = 128
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

    per_rank = args.tokens // world_size
    local_num_experts = args.num_experts // world_size
    local_expert_offset = rank * local_num_experts

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(
        per_rank, args.hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    topk_ids = torch.randint(
        0,
        args.num_experts,
        (per_rank, args.top_k),
        device=device,
        dtype=torch.int64,
        generator=g,
    )
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
        algorithm=EpAlgorithm.LOW_LATENCY,
    )

    if args.baseline:
        compute_config, weights = None, None
    else:
        compute_config, weights = _build_compute(
            args,
            local_num_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            max_tokens=local_num_experts * per_rank * world_size,
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

    dist.barrier()
    if rank == 0:
        mode = "identity" if args.baseline else args.quant
        print(
            "BENCH_CSV,tokens,gpus,backend,quant,dispatch_us,compute_us,combine_us,e2e_us,tok_s\n"
            f"BENCH_CSV,{args.tokens},{world_size},{args.backend},{mode},"
            f"{d_us:.1f},{cp_us:.1f},{cb_us:.1f},{e2e_us:.1f},{tok_s:.1f}"
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
