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

# Run the experimental Cake W8A8 MXFP8 MegaMoE EP16 backend through MoEEpMegaLayer.
#
# 16 ranks on SM103 GPUs in one NVLink domain (four 4-GPU nodes):
#
#   torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d \
#       --rdzv_endpoint $HOST:29500 examples/experimental/cake_w8a8_megamoe_ep16.py --tokens 16

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from flashinfer.experimental.cake_w8a8_megamoe_ep16 import routing_overflow
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpMegaLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig,
)

WORLD = 16
LOCAL_EXPERTS = 32
EXPERTS = WORLD * LOCAL_EXPERTS
HIDDEN = 3072
INTERMEDIATE = 5120
TOP_K = 8


def _select_device_index() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("each rank needs a visible CUDA device")
    # A launcher may give each process a private one-device visibility mask.
    return 0 if visible == 1 else local_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the experimental Cake W8A8 MXFP8 MegaMoE EP16 backend through "
            "MoEEpMegaLayer (torchrun, 16 ranks on SM103 GPUs in one NVLink domain)."
        )
    )
    parser.add_argument(
        "--tokens", type=int, default=16, help="live tokens per rank (1..64)"
    )
    parser.add_argument(
        "--graph", action="store_true", help="also capture and replay a CUDA graph"
    )
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        raise RuntimeError("this example requires torchrun with WORLD_SIZE=16")

    device_index = _select_device_index()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("this example requires SM103 GPUs")
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    try:
        rank = dist.get_rank()
        generator = torch.Generator(device=device).manual_seed(1000 + rank)
        w13 = torch.randn(
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w13.mul_(HIDDEN**-0.5)
        w2 = torch.randn(
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w2.mul_(INTERMEDIATE**-0.5)

        layer = MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=WORLD, rank=rank, device=device_index),
            fleet_params=FleetParams(
                num_experts=EXPERTS, max_tokens_per_rank=64, token_hidden_size=HIDDEN
            ),
            weights=MoEWeightPack(w13=w13, w2=w2),
            backend=MegaConfig(megakernel=Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()),
        )
        del w13, w2
        layer.warmup()  # collective: JIT load, workspace allocation, one launch

        token = torch.arange(args.tokens, dtype=torch.int64, device=device).unsqueeze(1)
        slot = torch.arange(TOP_K, dtype=torch.int64, device=device).unsqueeze(0)
        # Every expert receives the same global load (<= 32 routes) for any token count.
        topk_ids = (token * TOP_K + slot + rank * TOP_K) % EXPERTS
        inputs = MoEEpTensors(
            hidden_states=torch.randn(
                (args.tokens, HIDDEN),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ),
            topk_ids=topk_ids,
            topk_weights=torch.full(
                (args.tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device
            ),
        )
        output = layer.forward(inputs)
        torch.cuda.synchronize(device)
        overflow = int(routing_overflow(layer).item())
        if rank == 0:
            print(
                f"eager output shape={tuple(output.shape)} dtype={output.dtype} overflow_flag={overflow}"
            )

        if args.graph:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                layer.stage_inputs(inputs, compile_tokens_per_rank=args.tokens)
                view = layer.compute_staged(output=None)
            torch.cuda.synchronize(device)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            if rank == 0:
                print(f"graph replay matches eager: {bool(torch.equal(view, output))}")
        layer.destroy()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
