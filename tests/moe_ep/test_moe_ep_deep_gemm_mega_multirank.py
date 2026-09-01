"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (DeepGEMM backend).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and the ``deep_gemm`` package with
``fp8_fp4_mega_moe`` support.

Weights: loaded fp4 ``int8`` weights plus raw fp32 block-32 scales are wrapped
in ``MoEWeightPack`` with no external ``transform_sf_into_required_layout``;
FlashInfer preprocesses them when ``preprocess_weights=True``.

Torch-oracle anchor: parity alone cannot catch a kernel that is wrong but
self-consistent at ``world_size > 1`` (peer-pull addressing, expert→rank
ownership, cross-rank combine), because both sides run the same CUDA kernel.
``test_moe_ep_deep_gemm_mega_multirank_torch_oracle`` closes that gap with the
sm90_fp8_fp8_bf16_pull_cutedsl twin's methodology: each rank all-gathers the actual loaded fp4
weight legs and checks its real-EP kernel output slice against the pure-torch
oracle run on its tokens + the global expert set.
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

    from flashinfer.moe_ep import (
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.staging import (
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
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
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
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
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
            megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
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


def _all_gather_stack(t):
    """all_gather a per-rank tensor and stack it on a new leading rank dim."""
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    gathered = [torch.empty_like(tc) for _ in range(world_size)]
    dist.all_gather(gathered, tc)
    return torch.stack(gathered)


def _torch_dg_mega_reference_global(
    *,
    hidden_states,  # (T, hidden) bf16, this rank's tokens
    topk_ids,  # (T, topk) int64, global expert ids
    topk_weights,  # (T, topk) fp32
    w13,  # (E, 2I, hidden//2) packed e2m1 int8, GLOBAL expert set
    w2,  # (E, hidden, I//2) packed e2m1 int8
    w13_sf,  # (E, 2I, hidden//32) fp32
    w2_sf,  # (E, hidden, I//32) fp32
    hidden,
    intermediate,
    clamp,
):
    """Pure-torch fp8_fp4 DeepGEMM mega-MoE oracle over a global expert set.

    Parameterized twin of the single-GPU ``_torch_dg_mega_reference``: same
    conventions (fc1 acc → bf16 round, ``gate=min(gate,c)`` /
    ``up=clamp(up,±c)``, topk weight folded before the fc1-out per-32
    e4m3/UE8M0 round-trip, plain-sum combine), but takes explicit dims and raw
    weight tensors so the caller can pass the all-gathered global expert set.
    """
    import torch
    from deep_gemm.utils import per_token_cast_to_fp8

    from .test_deep_gemm_mega_kernel_vs_reference import (
        GRAN_K,
        _dequant_fp4_gran32,
        _fp8_gran32_roundtrip,
    )

    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]

    # Same activation quant recipe as the FI dg staging (packed-vs-plain SF
    # layout differs, the values do not).
    x_q, x_sf = per_token_cast_to_fp8(hidden_states, use_ue8m0=True, gran_k=GRAN_K)
    x_deq = x_q.float() * x_sf.repeat_interleave(GRAN_K, dim=-1)

    out = torch.zeros(
        num_tokens, hidden, dtype=torch.float32, device=hidden_states.device
    )
    for expert in range(num_experts):
        routing_mask = topk_ids == expert
        if not routing_mask.any():
            continue
        routed = routing_mask.nonzero(as_tuple=False)
        tokens, slots = routed[:, 0], routed[:, 1]

        w13_d = _dequant_fp4_gran32(w13[expert], w13_sf[expert])
        w2_d = _dequant_fp4_gran32(w2[expert], w2_sf[expert])

        fc1 = x_deq[tokens] @ w13_d.transpose(0, 1)  # (R, 2I) fp32
        # The kernel rounds the fc1 accumulator to bf16 before clamp + SiLU.
        fc1 = fc1.to(torch.bfloat16).float()
        gate = fc1[:, :intermediate]
        up = fc1[:, intermediate:]
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
        swiglu = gate * torch.sigmoid(gate) * up
        # topk weight folded before the fc1-out fp8 round-trip.
        swiglu = swiglu * topk_weights[tokens, slots].unsqueeze(-1)

        swiglu_rt = _fp8_gran32_roundtrip(swiglu)
        out.index_put_((tokens,), swiglu_rt @ w2_d.transpose(0, 1), accumulate=True)

    return out.to(torch.bfloat16)


def _oracle_problem(rank: int, world_size: int):
    """Multirank problem conditioned for the torch oracle's tight band.

    Like the single-GPU dg oracle: dim^-0.5 scaled weights + softmaxed topk
    weights keep |y| ~ O(1) so the atol=0.15 band is meaningful.  Token 0 is
    forced to route one expert per EP rank (contiguous block ownership: rank
    r owns [r*L, (r+1)*L)) to guarantee cross-rank traffic by construction.
    """
    import torch
    from deep_gemm.utils import per_token_cast_to_fp4

    from flashinfer.moe_ep import MoEWeightPack

    problem = _mega_problem(rank, world_size)
    problem["topk_weights"] = torch.softmax(problem["topk_weights"], dim=-1)

    num_local = problem["num_experts"] // world_size
    forced = (
        torch.arange(min(problem["topk"], world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    problem["topk_ids"][0, : forced.numel()] = forced

    hidden = problem["hidden"]
    intermediate = problem["intermediate"]
    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13_bf16 = torch.randn(
        num_local,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (hidden**-0.5)
    w2_bf16 = torch.randn(
        num_local,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (intermediate**-0.5)

    GRAN_K = 32
    w13 = torch.empty(
        num_local, 2 * intermediate, hidden // 2, dtype=torch.int8, device="cuda"
    )
    w2 = torch.empty(
        num_local, hidden, intermediate // 2, dtype=torch.int8, device="cuda"
    )
    w13_sf = torch.empty(
        num_local,
        2 * intermediate,
        hidden // GRAN_K,
        dtype=torch.float32,
        device="cuda",
    )
    w2_sf = torch.empty(
        num_local, hidden, intermediate // GRAN_K, dtype=torch.float32, device="cuda"
    )
    for e in range(num_local):
        w13[e], w13_sf[e] = per_token_cast_to_fp4(
            w13_bf16[e], use_ue8m0=True, gran_k=GRAN_K
        )
        w2[e], w2_sf[e] = per_token_cast_to_fp4(
            w2_bf16[e], use_ue8m0=True, gran_k=GRAN_K
        )

    problem["weights"] = MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_sf, w2_scale=w2_sf)
    return problem


def _run_mega_torch_oracle(rank, world_size):
    """Real-EP kernel launch vs the pure-torch oracle on the GLOBAL expert set.

    Parity alone cannot catch a kernel that is wrong but self-consistent at
    ``world_size > 1`` (peer-pull addressing, expert→rank ownership,
    cross-rank combine), because both sides run the same CUDA kernel; this
    closes that gap (sm90_fp8_fp8_bf16_pull_cutedsl twin methodology).  Every rank stages its
    own shard, runs the fused kernel with real cross-rank traffic, all-gathers
    the ACTUAL loaded fp4 weight legs of all ranks (no reliance on cross-rank
    RNG determinism), and checks its own output slice against the pure-torch
    reference on its tokens + the global expert set (y[t] is per-token math).
    """
    import deep_gemm
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.staging import (
        stage_mega_moe_inputs,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    problem = _oracle_problem(rank, world_size)
    n = problem["num_tokens"]
    wp = problem["weights"]

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x[:n],
            symm_buffer.x_sf[:n],
            symm_buffer.topk_idx[:n],
            symm_buffer.topk_weights[:n],
        )

        transformed = preprocess_mega_weights(
            wp,
            intermediate_size=problem["intermediate"],
            hidden_size=problem["hidden"],
        )

        y_kernel = torch.empty(
            n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        kernel = DeepGemmMegaKernelBackend(
            Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                activation_clamp=problem["activation_clamp"],
                # fast_math uses __expf/fast-rcp SiLU; disable for a tighter
                # oracle band (the layer test covers fast_math=True).
                fast_math=False,
            )
        )
        kernel.compute(symm_buffer, transformed, output=y_kernel)
        torch.cuda.synchronize()
        dist.barrier()

        # (R, E_local, ...) → (E, ...) rank-major matches global expert ids.
        w13_g = _all_gather_stack(wp.w13).flatten(0, 1)
        w2_g = _all_gather_stack(wp.w2).flatten(0, 1)
        w13_sf_g = _all_gather_stack(wp.w13_scale).flatten(0, 1)
        w2_sf_g = _all_gather_stack(wp.w2_scale).flatten(0, 1)

        y_ref = _torch_dg_mega_reference_global(
            hidden_states=problem["hidden_states"],
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            w13=w13_g,
            w2=w2_g,
            w13_sf=w13_sf_g,
            w2_sf=w2_sf_g,
            hidden=problem["hidden"],
            intermediate=problem["intermediate"],
            clamp=problem["activation_clamp"],
        )

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        print(
            f"[dg multirank oracle rank {rank}] rel_l2={rel_l2.item():.4g} "
            f"max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Single-GPU oracle tolerances: both sides consume identical quantized
        # operands; the residual is fp8 RTNE flips at fc1-out plus GEMM
        # accumulation-order noise on |y|~O(1).
        torch.testing.assert_close(yk, yr, atol=0.15, rtol=0.05)
        assert rel_l2.item() < 0.02
        return rank
    finally:
        symm_buffer.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_deep_gemm_mega_multirank_torch_oracle():
    """Real cross-rank EP kernel vs pure-torch global math (see helper doc)."""
    pytest.importorskip("deep_gemm")
    pytest.importorskip("triton")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(rank, world_size)
    print(f"rank {rank}: deep_gemm mega kernel matches the multi-rank torch oracle")
