"""NCCL-EP smoke entry point.

Usage:
    torchrun --nproc_per_node=4 tests/moe_ep/smoke_nccl_ep.py

Constructs an :class:`MoEEpLayer` with ``backend="nccl_ep"`` (default
``IdentityConfig`` inner kernel) and runs one dispatch → identity → combine
pass. Asserts the output has the same shape as the input. With
softmax-normalized topk_weights, the output approximates the input within
bf16 tolerance.

Requires ``nccl.ep``, which is available by default: it ships in the
``nccl4py`` wheel, a base dependency of flashinfer-python (a plain
``pip install -e .`` is enough).
"""

from __future__ import annotations

import os
from datetime import timedelta
import sys

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)

# When launched as `torchrun tests/moe_ep/smoke_nccl_ep.py`, Python inserts
# this script's directory (tests/moe_ep/) at sys.path[0]. That dir holds the
# `nccl_ep/` and `nixl_ep/` *test* subpackages, which would shadow the
# installed `nccl_ep` / `nixl_ep` ctypes modules the EP backends import. Drop
# the script dir so the real packages resolve.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


def main() -> int:
    import torch
    import torch.distributed as dist

    # Initialize the process group; srun/torchrun sets RANK/WORLD_SIZE.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=_PG_TIMEOUT)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        # PyTorch's NCCL backend ties device to LOCAL_RANK if set, else rank.
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)

    from flashinfer.moe_ep import (
        dummy_moe_weights,
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
        weights=dummy_moe_weights(
            num_local_experts=num_experts // world_size,
            hidden=hidden,
        ),
        backend="nccl_ep",
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()

    assert y.shape == x.shape, f"shape mismatch: y={y.shape} vs x={x.shape}"
    y_mean = float(y.float().mean().item())
    print(f"rank {rank}: nccl_ep smoke OK, y.mean={y_mean:.4f}")

    layer.destroy()
    dist.barrier()
    if rank == 0:
        print("SMOKE_RESULT: nccl_ep OK")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
