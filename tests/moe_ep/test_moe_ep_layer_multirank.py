"""Phase 2 / C3 — multi-rank roundtrip on 4+ GPUs via MoEEpLayer + Identity kernel.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_layer_multirank.py -v -m "nvep and gpu_4" --backend=nccl_ep      # or nixl_ep

The identity inner compute makes ``dispatch → identity → combine`` a
roundtrip of the per-rank ``hidden_states`` weighted by ``sum(topk_weights)``.
With softmax-normalized topk_weights that sum to 1 per token, output ≈ input
within bf16 tolerance.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)


def pytest_generate_tests(metafunc):
    """Generate `comm_backend` param values from --backend CLI."""
    if "comm_backend" not in metafunc.fixturenames:
        return
    cli = metafunc.config.getoption("--backend", default=None)
    if cli == "both" or cli is None:
        metafunc.parametrize("comm_backend", ["nccl_ep", "nixl_ep"])
    else:
        metafunc.parametrize("comm_backend", [cli])


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_moe_ep_roundtrip_ll_bf16_h4096(comm_backend):
    """MoEEpLayer + IdentityConfig roundtrips hidden_states on 4+ GPUs.

    With softmax-normalized topk_weights, sum(topk_weights, dim=-1)==1, so
    after combine reweights by them the output equals the input within bf16
    tolerance.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        dummy_moe_weights,
        IdentityConfig,
        MoEEpLayer,
        MoEEpTensors,
        NCCLEPConfig,
        NvepConfig,
        SplitConfig,
    )

    backend_name = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend_name,
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

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

    if comm_backend == "nixl_ep":
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        tcp_store = dist.TCPStore(
            host_name=master_addr,
            port=master_port + 1,
            world_size=world_size,
            is_master=(rank == 0),
        )
    else:
        tcp_store = None

    comm = NvepConfig() if comm_backend == "nixl_ep" else NCCLEPConfig()
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
            tcp_store=tcp_store,
        ),
        fleet_params=FleetParams(
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
        backend=SplitConfig(comm=comm, kernel=IdentityConfig()),
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    assert y.shape == x.shape
    # With softmax-normalized topk_weights that sum to 1, dispatch+combine
    # is effectively a roundtrip: y ≈ x within bf16 tolerance.
    torch.testing.assert_close(y, x, atol=5e-2, rtol=5e-2)
    layer.destroy()
    print(f"rank {rank}: {comm_backend} identity roundtrip OK")
