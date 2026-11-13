"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Experimental MOSCA MegaMoE backend: one DeepSeek-V4-Pro-sized MoE layer at EP8
# (384 experts, hidden 7168, intermediate 3072, top-6), run in both token layouts.
#
#   torchrun --nproc-per-node 8 examples/experimental/mosca_megamoe_ep.py
#
# "partitioned": every rank passes its own tokens and gets its own rows back
#   (the contract of the stable sm100_fp8_fp4_bf16_deepgemm backend).
# "replicated": every rank passes the same batch (attention TP) and gets the
#   whole batch back; the all-gather after the MoE happens inside the kernel.
#
# Importing the config from flashinfer.experimental is the opt-in; constructing
# the layer emits one ExperimentalWarning.

import os

import torch
import torch.distributed as dist

from flashinfer.experimental.mosca.megamoe import (
    Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
)
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
)

HIDDEN, INTER, EXPERTS, TOPK = 7168, 3072, 384, 6


def make_layer(rank, world, dev, token_layout, max_tokens_per_rank):
    # This rank's experts as BF16 (quantized to FP4 at load). A checkpoint
    # would pass packed FP4 bytes plus per-32 scales instead.
    local = EXPERTS // world
    torch.manual_seed(rank)
    weights = MoEWeightPack(
        w13=torch.randn(local, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16)
        * 0.02,
        w2=torch.randn(local, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02,
    )
    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world, rank=rank),
        fleet_params=FleetParams(
            num_experts=EXPERTS,
            # Replicated: this bounds the whole (replicated) batch.
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=HIDDEN,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(
                intermediate_size=INTER,
                top_k=TOPK,
                activation_clamp=10.0,
                token_layout=token_layout,
            )
        ),
    )


def routed_batch(tokens, dev, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.randn(tokens, EXPERTS, generator=gen, device=dev)
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), TOPK, dim=-1)
    return MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", torch.cuda.current_device())

    # Partitioned: each rank routes its own 5 + rank tokens.
    layer = make_layer(rank, world, dev, "partitioned", max_tokens_per_rank=64)
    layer.warmup()
    batch = routed_batch(5 + rank, dev, seed=rank)
    y = layer.forward(batch)
    torch.cuda.synchronize()
    print(
        f"[rank {rank}] partitioned: input {tuple(batch.hidden_states.shape)} "
        f"-> output {tuple(y.shape)}"
    )
    layer.destroy()

    # Replicated: the same 37 tokens on every rank (as after a TP attention block).
    layer = make_layer(rank, world, dev, "replicated", max_tokens_per_rank=64 * world)
    layer.warmup()
    batch = routed_batch(37, dev, seed=0)
    y = layer.forward(batch)
    torch.cuda.synchronize()
    print(
        f"[rank {rank}] replicated: input {tuple(batch.hidden_states.shape)} "
        f"-> output {tuple(y.shape)} (every rank gets the whole batch)"
    )
    layer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
