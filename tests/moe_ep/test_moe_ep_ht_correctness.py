"""Multi-GPU functional correctness for the HT (High-Throughput) FLAT path (bf16).

Mirrors tests/moe_ep/test_moe_ep_compute_correctness.py (the LL test) but for
``EpAlgorithm.HIGH_THROUGHPUT`` / ``nccl.ep.Layout.FLAT``. Each rank runs
``dispatch -> compute -> combine`` over its own tokens and compares the combined
output against the SAME unified ``MoELayer`` kernel run non-EP (all experts local,
real top_k) — immune to kernel-convention/weight-shuffle quirks, so it isolates EP
dispatch/compute/combine correctness.

Geometry follows the upstream ``contrib/nccl_ep/ep_test.py`` (the config the
nccl_ep_b200_ib HT benchmark uses): rank-derived
  top_k = min(8, world);  num_experts = min(256, top_k*world);
  num_local_experts = num_experts // world;  hidden = 7168;  bf16.
``-t`` is PER-RANK tokens; we test the benchmark's 4096 and 8192 per-rank sizes.
intermediate_size = 2048 (DeepSeek-class; ep_test.py does no FFN so it is our choice).

Launch (Pre-Nyx B200; HT multi-rank works on the current nccl4py wheel):
    torchrun --nproc_per_node=8 -m pytest \\
        tests/moe_ep/test_moe_ep_ht_correctness.py -v -s -m "nvep and gpu_8"

Or directly:
    torchrun --nproc_per_node=8 tests/moe_ep/test_moe_ep_ht_correctness.py
"""

from __future__ import annotations

import os

import pytest

HIDDEN = 7168
INTERMEDIATE = 2048
# bf16 tolerance (weights ~1/sqrt(fan_in) -> activations O(1), precision-bound).
RTOL = 3e-2
ATOL = 3e-2
PER_RANK_SIZES = [4096, 8192]


def _geometry(world_size):
    """Rank-derived HT geometry, matching contrib/nccl_ep/ep_test.py."""
    top_k = min(8, world_size)
    num_experts = min(256, top_k * world_size)
    num_local_experts = num_experts // world_size
    return top_k, num_experts, num_local_experts


def _block_major_k(w):
    """BlockMajorK shuffle for the trtllm bf16 routed runner (epilogue_tile_m=64)."""
    import torch

    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe.core import convert_to_block_layout

    out = []
    for i in range(w.shape[0]):
        s = shuffle_matrix_a(w[i].view(torch.uint8), 64)
        out.append(convert_to_block_layout(s, 128))
    return torch.stack(out).view(torch.bfloat16)


def _build_bf16_compute(w1, w2, *, num_experts, top_k, offset, local_n, max_tokens):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoEWeightPack,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    cfg = MoEConfig(
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
    wp = MoEWeightPack()
    wp.prepare_for(
        "trtllm_bf16_routed",
        {"gemm1_weights": _block_major_k(w1), "gemm2_weights": _block_major_k(w2)},
    )
    return cfg, wp


def _kernel_full_moe_reference(
    x, w1_full, w2_full, topk_ids, topk_weights, num_experts, top_k
):
    """Full MoE via the SAME kernel, all experts local (offset 0), real top_k."""
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer

    cfg, wp = _build_bf16_compute(
        w1_full,
        w2_full,
        num_experts=num_experts,
        top_k=top_k,
        offset=0,
        local_n=num_experts,
        max_tokens=x.shape[0],
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
        MoEEpLayer,
        MoEEpTensors,
    )

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    top_k, num_experts, local_n = _geometry(world)
    offset = rank * local_n

    # Full weights, identical on every rank (constant seed), ~1/sqrt(fan_in).
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

    cfg, wp = _build_bf16_compute(
        w1[offset : offset + local_n].contiguous(),
        w2[offset : offset + local_n].contiguous(),
        num_experts=num_experts,
        top_k=top_k,
        offset=offset,
        local_n=local_n,
        max_tokens=per_rank * local_n,
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
        backend="nccl_ep",
        compute_config=cfg,
        weights=wp,
    )
    y = layer.forward(
        MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    )
    torch.cuda.synchronize()
    assert y.shape == x.shape, (
        f"HT t={per_rank}: shape {tuple(y.shape)} != {tuple(x.shape)}"
    )

    y_kernel = _kernel_full_moe_reference(
        x, w1, w2, topk_ids, topk_weights, num_experts, top_k
    )
    yf, kf = y.float(), y_kernel.float()
    rel = (yf - kf).abs().amax().item() / kf.abs().amax().clamp_min(1e-6).item()
    if rank == 0:
        import contextlib

        msg = (
            f"[HT t/rank={per_rank} world={world} experts={num_experts} top_k={top_k} "
            f"local={local_n}] EP-vs-kernel rel-err={rel:.4f}  EP mean|.|={yf.abs().mean():.4f} "
            f"kernel mean|.|={kf.abs().mean():.4f}\n"
        )
        print(msg)
        with (
            contextlib.suppress(OSError),
            open(f"/host/logs/relerr_ht_t{per_rank}_w{world}.txt", "w") as fh,
        ):
            fh.write(msg)
    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    layer.destroy()
    return rank, rel


def pytest_generate_tests(metafunc):
    if "per_rank" in metafunc.fixturenames:
        metafunc.parametrize("per_rank", PER_RANK_SIZES)


@pytest.mark.nvep
@pytest.mark.gpu_8
@pytest.mark.arch_blackwell
def test_moe_ep_ht_matches_dense_reference(per_rank):
    """High-throughput (HT, FLAT) bf16 dispatch->compute->combine equals a non-EP MoE reference."""
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank, rel = _run_ht(per_rank)
    dist.barrier()
    print(f"rank {rank}: HT t/rank={per_rank} OK (rel-err={rel:.4f})")


def _main():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    for per_rank in PER_RANK_SIZES:
        r, rel = _run_ht(per_rank)
        dist.barrier()
        if r == 0:
            print(f"[OK] HT t/rank={per_rank}: EP matches kernel (rel-err={rel:.4f})")
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
