"""Multi-GPU correctness for fused-gemm2-combine (NVFP4 LL EXPERT_MAJOR).

Clone of ``test_moe_ep_compute_correctness_nvfp4.py`` for the overlap path:
CuteDSL GEMM2 ``tile_ready`` + dense permuted-C ship + dest-side combine
(``skip_combine=True``). Isolates that path against split-moe on the same
CuteDSL compute family (plain ``CuteDslConfig()``, NCCL combine).

Launch (4 GPU, SM100+):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_fused_gemm2_combine_correctness.py -v -s \\
        -m "nvep and gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of CuteDSL kernels can exceed torch's 10-min default
# watchdog while other ranks wait in a collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)

NUM_EXPERTS = 16
TOP_K = 8
TOKENS_PER_RANK = 128
HIDDEN = 8192
INTERMEDIATE = 2048
RTOL = 5e-2
ATOL = 5e-2


def _build_nvfp4_cute_moe_config(
    *, offset, local_num_experts, max_tokens, tile_signal: bool
):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
    )

    cute = CuteDslConfig(
        enable_tile_signal=tile_signal,
        store_permuted_c=tile_signal,
    )
    return MoEConfig(
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(cute,)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )


def _local_weight_pack(w1_full, w2_full, offset, local_num_experts):
    from flashinfer.moe_ep import MoEWeightPack

    return MoEWeightPack(
        w13=w1_full[offset : offset + local_num_experts].contiguous().clone(),
        w2=w2_full[offset : offset + local_num_experts].contiguous().clone(),
    )


def _forward_ep(
    *,
    x,
    topk_ids,
    topk_weights,
    weights,
    moe_config,
    fleet_params,
    bootstrap,
    skip_combine: bool,
    expect_overlap_stats: bool = False,
):
    import torch

    from flashinfer.moe_ep import (
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        NcclEpConfig,
        SplitConfig,
    )

    layer = MoEEpLayer(
        bootstrap,
        fleet_params,
        weights=weights,
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
            skip_combine=skip_combine,
        ),
    )
    t = MoEEpTensors(
        hidden_states=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape

    stats = None
    if expect_overlap_stats:
        kernel = getattr(layer, "_kernel", None)
        stats = (
            getattr(kernel, "last_overlap_stats", None) if kernel is not None else None
        )
        if not stats:
            raise AssertionError(
                "fused-gemm2-combine did not populate last_overlap_stats"
            )
        live = int(stats.get("live_rows", -1))
        expected = int(stats.get("expected_rows", -2))
        assert live == expected, (
            f"overlap live_rows={live} != expected_rows={expected} (stats={stats})"
        )
    layer.destroy()
    return y, stats


def _run_fused_gemm2_combine_vs_split_moe():
    import torch
    import torch.distributed as dist

    from flashinfer.fused_moe.api import CuteDslConfig
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if not CuteDslConfig.supported(arch):
        pytest.skip(
            f"CuteDSL NVFP4 fused-gemm2-combine requires SM100/103/107, got sm{arch}"
        )

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts
    max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    gw = torch.Generator(device="cuda").manual_seed(2024)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2_full = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(TOKENS_PER_RANK, HIDDEN, device="cuda", generator=g).to(
        torch.bfloat16
    )
    scores = torch.randn(TOKENS_PER_RANK, NUM_EXPERTS, device="cuda", generator=g)
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(TOKENS_PER_RANK, TOP_K, device="cuda", generator=g), dim=-1
    )

    fleet_params = FleetParams(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=TOKENS_PER_RANK,
        token_hidden_size=HIDDEN,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.EXPERT_MAJOR,
    )

    def _bootstrap():
        return BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
        )

    y_overlap, stats = _forward_ep(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        weights=_local_weight_pack(w1_full, w2_full, offset, local_num_experts),
        moe_config=_build_nvfp4_cute_moe_config(
            offset=offset,
            local_num_experts=local_num_experts,
            max_tokens=max_tokens,
            tile_signal=True,
        ),
        fleet_params=fleet_params,
        bootstrap=_bootstrap(),
        skip_combine=True,
        expect_overlap_stats=True,
    )

    expected_rows = int(stats["expected_rows"])
    totals = torch.tensor([expected_rows], device="cuda", dtype=torch.int64)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    assert int(totals.item()) > 0, (
        "overlap expected_rows summed to 0 across ranks; routing never hit a "
        "local expert, so an empty inbox could pass assert_close"
    )

    dist.barrier()

    y_split, _ = _forward_ep(
        x=x.clone(),
        topk_ids=topk_ids.clone(),
        topk_weights=topk_weights.clone(),
        weights=_local_weight_pack(w1_full, w2_full, offset, local_num_experts),
        moe_config=_build_nvfp4_cute_moe_config(
            offset=offset,
            local_num_experts=local_num_experts,
            max_tokens=max_tokens,
            tile_signal=False,
        ),
        fleet_params=fleet_params,
        bootstrap=_bootstrap(),
        skip_combine=False,
    )

    yf, rf = y_overlap.float(), y_split.float()

    def _rel(a, b):
        return (a - b).abs().amax().item() / b.abs().amax().clamp_min(1e-6).item()

    overlap_vs_split = _rel(yf, rf)
    if rank == 0:
        print(
            f"[nvfp4 fused-gemm2-combine] overlap-vs-split-moe rel-err="
            f"{overlap_vs_split:.4f} live_rows={stats['live_rows']} "
            f"expected_rows={stats['expected_rows']}"
        )

    torch.testing.assert_close(yf, rf, rtol=RTOL, atol=ATOL)
    return rank, overlap_vs_split, stats


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_fused_gemm2_combine_matches_split_moe():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 fused-gemm2-combine requires SM100+")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        pytest.skip(
            f"num_experts={NUM_EXPERTS} not divisible by world_size={world_size}"
        )

    rank, rel_err, stats = _run_fused_gemm2_combine_vs_split_moe()
    dist.barrier()
    print(
        f"rank {rank}: fused-gemm2-combine==split-moe OK "
        f"(rel-err={rel_err:.4f}, live_rows={stats['live_rows']}, "
        f"expected_rows={stats['expected_rows']})"
    )
