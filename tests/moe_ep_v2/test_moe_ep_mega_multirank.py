"""Multi-rank smoke + correctness tests for MoEEpMegaLayer.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep_v2/test_moe_ep_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and the ``deep_gemm`` package with
``fp8_fp4_mega_moe`` support.
"""

from __future__ import annotations

import os

import pytest

deep_gemm = pytest.importorskip("deep_gemm")
pytest.importorskip("triton")


def _init_dist():
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return dist


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


def _make_bf16_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    return w13, w2


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
    w13, w2 = _make_bf16_weights(
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
        w13=w13,
        w2=w2,
    )


def _reference_mega_moe_staged(group, problem: dict, *, destroy_buffer: bool = True):
    """Reference with bf16 activations staged inside the symm buffer."""
    import torch

    from flashinfer.moe_ep_v2.mega.staging import stage_mega_moe_inputs
    from flashinfer.moe_ep_v2.mega.weights import MoEWeightPack, preprocess_mega_weights

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

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    y = torch.empty(
        num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
    )
    deep_gemm.fp8_fp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        activation_clamp=problem["activation_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _reference_mega_moe_prestaged(
    group, problem: dict, x_fp8, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied fp8 activations + packed scales."""
    import torch

    from flashinfer.moe_ep_v2.mega.weights import MoEWeightPack, preprocess_mega_weights

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
    )
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].copy_(x_fp8)
    symm_buffer.x_sf[:num_tokens].copy_(x_sf)
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    y = torch.empty(
        num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
    )
    deep_gemm.fp8_fp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        activation_clamp=problem["activation_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _run_mega_layer(dist_mod, rank, world_size, *, stage_inputs: bool):
    import torch

    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
    )

    problem = _mega_problem(rank, world_size)
    if stage_inputs:
        t_hidden = problem["hidden_states"]
        t_scales = None
    else:
        from deep_gemm.utils import per_token_cast_to_fp8

        x_fp8, x_sf = per_token_cast_to_fp8(
            problem["hidden_states"],
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        t_hidden = x_fp8
        t_scales = x_sf

    mega = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=problem["num_experts"],
            max_tokens_per_rank=problem["max_tokens"],
            token_hidden_size=problem["hidden"],
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        ),
        backend=MegaConfig(
            kernel=DeepGemmMegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                activation_clamp=problem["activation_clamp"],
                fast_math=problem["fast_math"],
            ),
            stage_inputs=stage_inputs,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)

    t = MoEEpTensors(
        hidden_states=t_hidden,
        topk_ids=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
        scales=t_scales,
    )
    y_layer = mega.forward(t)
    torch.cuda.synchronize()
    dist_mod.barrier()

    if stage_inputs:
        y_ref = _reference_mega_moe_staged(
            dist_mod.group.WORLD, problem, destroy_buffer=True
        )
    else:
        y_ref = _reference_mega_moe_prestaged(
            dist_mod.group.WORLD, problem, t_hidden, t_scales, destroy_buffer=True
        )
    dist_mod.barrier()

    assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
    assert y_layer.dtype == torch.bfloat16
    assert torch.isfinite(y_layer).all()
    torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
    mega.destroy()
    return rank


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mega_layer_matches_deep_gemm_reference():
    """MoEEpMegaLayer with on-the-fly bf16→fp8 staging."""
    dist_mod = _init_dist()
    assert dist_mod.get_world_size() >= 4
    rank = _run_mega_layer(
        dist_mod, dist_mod.get_rank(), dist_mod.get_world_size(), stage_inputs=True
    )
    print(f"rank {rank}: mega layer (staged inputs) matches deep_gemm reference")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer with pre-staged fp8 activations (stage_inputs=False)."""
    dist_mod = _init_dist()
    assert dist_mod.get_world_size() >= 4
    rank = _run_mega_layer(
        dist_mod, dist_mod.get_rank(), dist_mod.get_world_size(), stage_inputs=False
    )
    print(f"rank {rank}: mega layer (prestaged inputs) matches deep_gemm reference")
