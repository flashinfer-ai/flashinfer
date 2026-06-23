"""Multi-GPU functional correctness for LL EXPERT_MAJOR + RANK_MAJOR (bf16).

Exercises dispatch → fused_moe compute → combine and compares against a
single-process dense MoE reference via the same ``MoELayer`` kernel.

Launch (4 GPU):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_compute_correctness.py -v -s -m "nvep and gpu_4"
"""

from __future__ import annotations

import os

import pytest

NUM_EXPERTS = 16
TOP_K = 8
TOKENS_PER_RANK = 128
HIDDEN = 8192
INTERMEDIATE = 2048
RTOL = 3e-2
ATOL = 3e-2


def _block_major_k(w):
    import torch

    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe.core import convert_to_block_layout

    epilogue_tile_m = 64
    block_k = 128
    shuffled = []
    for i in range(w.shape[0]):
        s = shuffle_matrix_a(w[i].view(torch.uint8), epilogue_tile_m)
        s = convert_to_block_layout(s, block_k)
        shuffled.append(s)
    return torch.stack(shuffled).view(torch.bfloat16)


def _build_bf16_moe_config(*, offset, local_num_experts, max_tokens):
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
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )


def _build_fused_moe_weights(w1_local, w2_local):
    from flashinfer.fused_moe.api import MoEWeightPack

    wp = MoEWeightPack()
    wp.prepare_for(
        "trtllm_bf16_routed",
        {
            "gemm1_weights": _block_major_k(w1_local),
            "gemm2_weights": _block_major_k(w2_local),
        },
    )
    return wp


def _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights):
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer

    num_tokens = x.shape[0]
    cfg = _build_bf16_moe_config(
        offset=0, local_num_experts=NUM_EXPERTS, max_tokens=num_tokens
    )
    wp = _build_fused_moe_weights(w1_full, w2_full)
    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=x.device),
        selected_experts=topk_ids.to(torch.int32),
        final_scales=topk_weights.to(torch.float32),
    )
    return MoELayer(cfg)(act, wp)


def _run_one_layout(layout_str):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts

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

    layout = (
        EpLayout.RANK_MAJOR if layout_str == "rank_major" else EpLayout.EXPERT_MAJOR
    )
    if layout is EpLayout.RANK_MAJOR:
        max_tokens = TOKENS_PER_RANK * world_size
    else:
        max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    moe_config = _build_bf16_moe_config(
        offset=offset,
        local_num_experts=local_num_experts,
        max_tokens=max_tokens,
    )
    canonical_weights = MoEWeightPack(
        w13=w1_full[offset : offset + local_num_experts].contiguous(),
        w2=w2_full[offset : offset + local_num_experts].contiguous(),
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        nccl_comm=None,
    )
    layer = MoEEpLayer(
        bootstrap,
        FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=layout,
            weights=canonical_weights,
        ),
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape

    y_kernel = _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights)
    yf, kf = y.float(), y_kernel.float()

    def _rel(a, b):
        return (a - b).abs().amax().item() / b.abs().amax().clamp_min(1e-6).item()

    ep_vs_kernel = _rel(yf, kf)
    if rank == 0:
        print(f"[{layout_str}] EP-vs-kernel rel-err={ep_vs_kernel:.4f}")

    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    layer.destroy()
    return rank, ep_vs_kernel


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        metafunc.parametrize("layout", ["expert_major", "rank_major"])


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_compute_matches_dense_reference(layout):
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        pytest.skip(
            f"num_experts={NUM_EXPERTS} not divisible by world_size={world_size}"
        )

    rank, ep_vs_kernel = _run_one_layout(layout)
    dist.barrier()
    print(f"rank {rank}: {layout} EP==kernel OK (EP-vs-kernel={ep_vs_kernel:.4f})")


def _main():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        if rank == 0:
            print(
                f"SKIP: num_experts={NUM_EXPERTS} not divisible by world={world_size}"
            )
        return
    for layout in ("expert_major", "rank_major"):
        r, ep_vs_kernel = _run_one_layout(layout)
        dist.barrier()
        if r == 0:
            print(f"[OK] {layout}: EP==kernel (rel-err={ep_vs_kernel:.4f})")
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
