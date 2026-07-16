"""Multi-GPU functional correctness for the HT (High-Throughput) FLAT path (bf16).

Mirrors ``test_moe_ep_compute_correctness.py`` but for
``EpAlgorithm.HIGH_THROUGHPUT`` / ``nccl.ep.Layout.FLAT``.

Asserts ``EP == non-EP kernel``; HT shares the ``trtllm_bf16_routed`` compute
kernel with LL, so its ``non-EP kernel == torch oracle`` anchor is the
single-GPU ``test_split_fused_moe_kernel_vs_reference.py`` bf16 test.

Launch (4 GPU):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_ht_correctness.py -v -s -m "nvep and gpu_4"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)

HIDDEN = 7168
INTERMEDIATE = 2048
RTOL = 3e-2
ATOL = 3e-2
PER_RANK_SIZES = [4096, 8192]


def _geometry(world_size):
    top_k = min(8, world_size)
    num_experts = min(256, top_k * world_size)
    num_local_experts = num_experts // world_size
    return top_k, num_experts, num_local_experts


def _build_bf16_moe_config(*, num_experts, top_k, offset, local_n, max_tokens):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=offset,
            local_num_experts=local_n,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )


def _build_fused_moe_weights(w1, w2, *, num_experts, top_k):
    # Same weight prep as the EP layer (gated-act reorder + shuffle + BlockMajorK).
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    cfg = _build_bf16_moe_config(
        num_experts=num_experts,
        top_k=top_k,
        offset=0,
        local_n=w1.shape[0],
        max_tokens=1,
    )
    return materialize_fused_moe_weights(MoEWeightPack(w13=w1, w2=w2), cfg)


def _kernel_full_moe_reference(
    x, w1_full, w2_full, topk_ids, topk_weights, num_experts, top_k
):
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer

    cfg = _build_bf16_moe_config(
        num_experts=num_experts,
        top_k=top_k,
        offset=0,
        local_n=num_experts,
        max_tokens=x.shape[0],
    )
    wp = _build_fused_moe_weights(
        w1_full, w2_full, num_experts=num_experts, top_k=top_k
    )
    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=x.device),
        selected_experts=topk_ids.to(torch.int32),
        final_scales=topk_weights.to(torch.float32),
    )
    return MoELayer(cfg)(act, wp)


def _run_ht(per_rank):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
    )

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    top_k, num_experts, local_n = _geometry(world)
    offset = rank * local_n

    gw = torch.Generator(device="cuda").manual_seed(2024)
    w1 = (
        torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(num_experts, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(per_rank, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    scores = torch.randn(per_rank, num_experts, device="cuda", generator=g)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(per_rank, top_k, device="cuda", generator=g), dim=-1
    )

    moe_config = _build_bf16_moe_config(
        num_experts=num_experts,
        top_k=top_k,
        offset=offset,
        local_n=local_n,
        max_tokens=per_rank * local_n,
    )
    canonical_weights = MoEWeightPack(
        w13=w1[offset : offset + local_n].contiguous(),
        w2=w2[offset : offset + local_n].contiguous(),
    )

    layer = MoEEpLayer(
        BootstrapConfig(
            world_size=world,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
        ),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=per_rank,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
            algorithm=EpAlgorithm.HIGH_THROUGHPUT,
        ),
        weights=canonical_weights,
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )
    y = layer.forward(
        MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    )
    torch.cuda.synchronize()
    assert y.shape == x.shape

    y_kernel = _kernel_full_moe_reference(
        x, w1, w2, topk_ids, topk_weights, num_experts, top_k
    )
    yf, kf = y.float(), y_kernel.float()
    rel = (yf - kf).abs().amax().item() / kf.abs().amax().clamp_min(1e-6).item()
    if rank == 0:
        print(
            f"[HT t/rank={per_rank} world={world} experts={num_experts} top_k={top_k}] "
            f"EP-vs-kernel rel-err={rel:.4f}"
        )
    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    layer.destroy()
    return rank, rel


def pytest_generate_tests(metafunc):
    if "per_rank" in metafunc.fixturenames:
        metafunc.parametrize("per_rank", PER_RANK_SIZES)


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_ht_matches_dense_reference(per_rank):
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    rank, rel = _run_ht(per_rank)
    dist.barrier()
    print(f"rank {rank}: HT t/rank={per_rank} OK (rel-err={rel:.4f})")


def _main():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    for per_rank in PER_RANK_SIZES:
        r, rel = _run_ht(per_rank)
        dist.barrier()
        if r == 0:
            print(f"[OK] HT t/rank={per_rank}: EP matches kernel (rel-err={rel:.4f})")
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
