"""Benchmark the fused BF16 CuTeDSL MegaMoE backend on Blackwell.

Launch one process per EP rank, for example:
``torchrun --nproc_per_node=4 benchmarks/bench_bf16_cutedsl_megamoe.py``.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        Bf16CutedslMegaMoeConfig,
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
    )
    from flashinfer.testing import bench_gpu_time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert args.experts % world_size == 0
    local_experts = args.experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    scores = torch.randn(args.tokens, args.experts, device="cuda", generator=generator)
    topk_weights, topk_ids = scores.topk(args.top_k, dim=-1)
    tensors = MoEEpTensors(
        hidden_states=torch.randn(
            args.tokens,
            args.hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ),
        topk_ids=topk_ids,
        topk_weights=topk_weights.float(),
    )
    weights = MoEWeightPack(
        w13=torch.randn(
            local_experts,
            2 * args.intermediate,
            args.hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ),
        w2=torch.randn(
            local_experts,
            args.hidden,
            args.intermediate,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ),
    )
    layer = MoEEpLayer(
        BootstrapConfig(world_size=world_size, rank=rank),
        FleetParams(
            num_experts=args.experts,
            max_tokens_per_rank=args.tokens,
            token_hidden_size=args.hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Bf16CutedslMegaMoeConfig(
                intermediate_size=args.intermediate, top_k=args.top_k
            )
        ),
    )
    layer.forward(tensors)
    times_ms = bench_gpu_time(
        layer.forward,
        input_args=(tensors,),
        repeat_iters=args.iters,
        enable_cupti=True,
    )
    if rank == 0:
        print(f"bf16_cutedsl,{np.median(times_ms):.4f} ms,{np.std(times_ms):.4f} ms")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
