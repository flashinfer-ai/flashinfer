"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (DeepGEMM backend).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and the ``deep_gemm`` package with
``fp8_fp4_mega_moe`` support.

Weights: loaded fp4 ``int8`` weights plus raw fp32 block-32 scales are wrapped
in ``MoEWeightPack`` with no external ``transform_sf_into_required_layout``;
FlashInfer preprocesses them when ``preprocess_weights=True``.
"""

from __future__ import annotations

import os

import pytest


def _launcher_ranks() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _make_inputs(
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    return (
        hidden_states,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int64),
    )


def _make_moe_weight_pack(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    """Loaded fp4 weights + fp32 block scales (no SF layout transform)."""
    import torch
    from deep_gemm.utils import per_token_cast_to_fp4

    from flashinfer.moe_ep import MoEWeightPack

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13_bf16 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2_bf16 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )

    # Loaded checkpoint layout: fp4-packed int8 weights [E, N, K//2].
    w13 = torch.empty(
        num_local_experts,
        2 * intermediate,
        hidden // 2,
        dtype=torch.int8,
        device="cuda",
    )
    w2 = torch.empty(
        num_local_experts,
        hidden,
        intermediate // 2,
        dtype=torch.int8,
        device="cuda",
    )
    # Raw fp32 block-32 scales — same role as w13_weight_scale_inv / w2_weight_scale_inv.
    w13_sf_fp32 = torch.empty(
        num_local_experts,
        2 * intermediate,
        hidden // 32,
        dtype=torch.float32,
        device="cuda",
    )
    w2_sf_fp32 = torch.empty(
        num_local_experts,
        hidden,
        intermediate // 32,
        dtype=torch.float32,
        device="cuda",
    )
    for expert in range(num_local_experts):
        w13_q, w13_sf = per_token_cast_to_fp4(
            w13_bf16[expert], use_ue8m0=True, gran_k=32
        )
        w2_q, w2_sf = per_token_cast_to_fp4(w2_bf16[expert], use_ue8m0=True, gran_k=32)
        w13[expert].copy_(w13_q)
        w2[expert].copy_(w2_q)
        w13_sf_fp32[expert].copy_(w13_sf)
        w2_sf_fp32[expert].copy_(w2_sf)

    return MoEWeightPack(
        w13=w13,
        w2=w2,
        w13_scale=w13_sf_fp32,
        w2_scale=w2_sf_fp32,
    )


def _mega_problem(rank: int, world_size: int):
    hidden = 4096
    intermediate = 2048
    num_tokens = 64
    max_tokens = 64
    num_experts = 8
    topk = 4
    activation_clamp = 10.0
    fast_math = True

    assert hidden % 128 == 0
    assert intermediate % 128 == 0
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
    )
    weights = _make_moe_weight_pack(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=weights,
    )


def _reference_mega_moe(group, problem: dict, *, destroy_buffer: bool = True):
    """Reference deep_gemm mega-MoE path for correctness checks."""
    import deep_gemm
    import torch

    from flashinfer.moe_ep import DeepGemmMegaMoeConfig, preprocess_mega_weights
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging import (
        stage_mega_moe_inputs,
    )

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
    )
    num_tokens = problem["num_tokens"]
    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        symm_buffer.x[:num_tokens],
        symm_buffer.x_sf[:num_tokens],
        symm_buffer.topk_idx[:num_tokens],
        symm_buffer.topk_weights[:num_tokens],
    )

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        problem["weights"],
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    kernel = DeepGemmMegaKernelBackend(
        DeepGemmMegaMoeConfig(
            intermediate_size=problem["intermediate"],
            top_k=problem["topk"],
            activation_clamp=problem["activation_clamp"],
            fast_math=problem["fast_math"],
        )
    )
    kernel.compute(
        symm_buffer,
        (transformed_l1, transformed_l2),
        output=y,
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _run_mega_layer(rank, world_size):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = _mega_problem(rank, world_size)
    weights = problem["weights"]

    # Pass loaded fp4 + fp32 scales directly; transform_sf runs in preprocess.
    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=problem["num_experts"],
            max_tokens_per_rank=problem["max_tokens"],
            token_hidden_size=problem["hidden"],
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=DeepGemmMegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                activation_clamp=problem["activation_clamp"],
                fast_math=problem["fast_math"],
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)

    t = MoEEpTensors(
        hidden_states=problem["hidden_states"],
        topk_ids=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
    )
    y_layer = mega.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    y_ref = _reference_mega_moe(dist.group.WORLD, problem, destroy_buffer=True)
    dist.barrier()

    assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
    assert y_layer.dtype == torch.bfloat16
    assert torch.isfinite(y_layer).all()
    torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
    mega.destroy()
    return rank


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mega_layer_matches_deep_gemm_reference():
    """MoEEpMegaLayer matches the deep_gemm mega-MoE reference."""
    pytest.importorskip("deep_gemm")
    pytest.importorskip("triton")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size)
    print(f"rank {rank}: mega layer matches deep_gemm reference")
