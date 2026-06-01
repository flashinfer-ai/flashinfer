"""NIXL-EP smoke entry point.

Usage:
    torchrun --nproc_per_node=8 tests/moe_ep/smoke_nixl_ep.py

Same shape as the NCCL-EP smoke, but constructs a torch.distributed.TCPStore
for the NIXL Buffer rendezvous (NIXL doesn't share NCCL's communicator).
"""

from __future__ import annotations

import os
import sys

# See smoke_nccl_ep.py: drop this script's dir from sys.path so the installed
# `nccl_ep` / `nixl_ep` ctypes modules aren't shadowed by the test subpackages
# under tests/moe_ep/.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


def main() -> int:
    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)

    # NIXL Buffer rendezvous needs a TCPStore visible to all ranks.
    # srun/torchrun sets MASTER_ADDR/MASTER_PORT for the NCCL backend;
    # we open a NEW TCPStore on a sibling port for NIXL — using the
    # same port as torch.distributed would clash.
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    nixl_port = master_port + 1
    tcp_store = dist.TCPStore(
        host_name=master_addr,
        port=nixl_port,
        world_size=world_size,
        is_master=(rank == 0),
    )

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
        nccl_comm=None,
        tcp_store=tcp_store,
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
        backend="nixl_ep",
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()

    assert y.shape == x.shape, f"shape mismatch: y={y.shape} vs x={x.shape}"
    y_mean = float(y.float().mean().item())
    print(f"rank {rank}: nixl_ep smoke OK, y.mean={y_mean:.4f}")

    dist.barrier()
    if rank == 0:
        print("SMOKE_RESULT: nixl_ep OK")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
