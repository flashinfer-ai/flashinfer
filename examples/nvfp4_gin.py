# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run the fixed N2R2 legacy NVFP4 GIN correctness example.

On each of two independent NVLink domains, with NCCL 2.30.7/nccl4py 0.3.1
and the matching CuTe DSL environment, launch (NODE_RANK is 0 or 1):

    torchrun --nnodes=2 --nproc-per-node=2 --node-rank=$NODE_RANK \
      --master-addr=$MASTER_ADDR --master-port=29500 examples/nvfp4_gin.py

The runtime verifies actual railed GIN and LSA-domain capabilities. This is
an experimental fixed-shape correctness example, with no performance timing.
"""

import os

import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError("This GIN example requires N2R2: two hosts, two ranks each.")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(101 + rank)
    weights = MoEWeightPack(
        w13=torch.randn(
            (12, 3072, 7168), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01,
        w2=torch.randn(
            (12, 7168, 1536), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01,
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=4, rank=rank, auto_bootstrap=False, device=device
        ),
        fleet_params=FleetParams(
            num_experts=48, max_tokens_per_rank=32, token_hidden_size=7168
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=1536,
                top_k=4,
                internode=True,
                ranks_per_node=2,
            ),
            preprocess_weights=True,
            quantize_input=True,
        ),
    )
    # On a fatal device/communication error, preserve registered resources and
    # let torchrun terminate the entire job. Close only on the successful path.
    for iteration in range(12):
        x = torch.randn(
            (32, 7168), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        scores = torch.rand((32, 48), device="cuda", generator=generator)
        topk_weights, topk_ids = scores.topk(4, dim=-1)
        y = layer.forward(
            MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
        )
        torch.cuda.synchronize()
        if not torch.isfinite(y).all().item():
            raise AssertionError(
                f"rank {rank}: nonfinite output at iteration {iteration}"
            )
        dist.barrier()
    layer.destroy()
    print(f"rank {rank}: 12 GIN calls passed finite/shape smoke", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
