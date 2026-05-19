"""Phase 2 / C3 — multi-rank roundtrip on 8 GPUs.

Launched via torchrun:
    torchrun --nproc_per_node=8 -m pytest \\
        tests/moe_ep/test_moe_ep_layer_multirank.py \\
        -v -m "nvep and gpu_8" \\
        --backend=nccl_ep      # or nixl_ep

The identity inner compute makes ``dispatch → identity → combine`` a
roundtrip of the per-rank ``hidden_states`` weighted by ``sum(topk_weights)``.
With softmax-normalized topk_weights that sum to 1 per token, output ≈ input
within bf16 tolerance.
"""

from __future__ import annotations

import os

import pytest


def pytest_generate_tests(metafunc):
    """Generate `backend` param values from --backend CLI."""
    if "backend" not in metafunc.fixturenames:
        return
    cli = metafunc.config.getoption("--backend")
    if cli == "both" or cli is None:
        metafunc.parametrize("backend", ["nccl_ep", "nixl_ep"])
    else:
        metafunc.parametrize("backend", [cli])


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_moe_ep_roundtrip_ll_bf16_h4096(backend):
    """Identity inner compute → dispatch+combine roundtrips hidden_states.

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
        MoEEpLayer,
        MoEEpTensors,
    )

    backend_name = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend_name)
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

    if backend == "nixl_ep":
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
        backend=backend,
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()

    assert y.shape == x.shape
    # With softmax-normalized topk_weights that sum to 1, dispatch+combine
    # is effectively a roundtrip: y ≈ x within bf16 tolerance.
    torch.testing.assert_close(y, x, atol=5e-2, rtol=5e-2)
    print(f"rank {rank}: {backend} roundtrip OK")
