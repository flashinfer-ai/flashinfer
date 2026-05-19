"""NCCL-EP smoke entry point.

Usage:
    torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tests.smoke_nccl_ep

Constructs an :class:`MoEEpLayer` with ``backend="nccl_ep"`` and runs one
dispatch → identity → combine → complete pass. Asserts the output has the
same shape as the input. With identity inner compute on softmax-normalized
topk_weights, the output approximates the input within bf16 tolerance.

Designed for the Phase 4 on-cluster validation step. On the dev box this
also exits 0 with ``--nproc_per_node=1`` provided the EP backends were
built (``BUILD_NCCL_EP=1``).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    import torch
    import torch.distributed as dist

    # Initialize the process group; srun/torchrun sets RANK/WORLD_SIZE.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        # PyTorch's NCCL backend ties device to LOCAL_RANK if set, else rank.
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
    )

    num_tokens = 64
    num_experts = 8
    hidden = 4096
    topk = 4

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", generator=g), dim=-1
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        nccl_comm=None,  # use upstream get_nccl_comm_from_group()
        tcp_store=None,
    )
    layer = MoEEpLayer(
        bootstrap,
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_tokens,
            token_hidden_size=hidden,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
        ),
        backend="nccl_ep",
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()

    assert y.shape == x.shape, f"shape mismatch: y={y.shape} vs x={x.shape}"
    y_mean = float(y.float().mean().item())
    print(f"rank {rank}: nccl_ep smoke OK, y.mean={y_mean:.4f}")

    dist.barrier()
    if rank == 0:
        print("SMOKE_RESULT: nccl_ep OK")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
