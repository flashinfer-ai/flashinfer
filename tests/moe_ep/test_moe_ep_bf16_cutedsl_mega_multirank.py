"""Multi-rank fused-launch tests for MoEEpMegaLayer (sm100_bf16_bf16_bf16_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_bf16_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("BF16 MegaMoE requires sm_100a or sm_103a")


def _launcher_ranks() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size


def _make_inputs(
    rank: int, *, num_tokens: int, hidden: int, num_experts: int, topk: int
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
    return hidden_states, topk_weights.to(torch.float32), topk_ids.to(torch.int64)


def _make_bf16_weights(
    rank: int, *, num_local_experts: int, hidden: int, intermediate: int
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


def _mega_problem(
    rank: int, world_size: int, *, num_tokens: int = 64, max_tokens: int = 64
):
    hidden, intermediate, num_experts, topk = 2048, 1024, 8, 4
    assert num_experts % world_size == 0
    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
    )
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_experts // world_size,
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
        gate_up_clamp=10.0,
        fast_math=True,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
    )


def _assert_ikr_close(y, y_ref, *, topk):
    import torch

    a = y.float()
    b = y_ref.float()
    diff = (a - b).abs()
    row_scale = torch.maximum(a.abs(), b.abs()).amax(dim=1, keepdim=True)
    tol = 5e-2 + (topk * 2.0**-8 * 8.0) * row_scale
    worst = (diff - tol).max().item()
    assert worst <= 0.0, (
        f"ikr output outside the bf16 K-term accumulation band "
        f"(worst overshoot {worst:.4f}, max diff {diff.max().item():.4f})"
    )


def _all_gather_stack(t):
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    gathered = [torch.empty_like(tc) for _ in range(world_size)]
    dist.all_gather(gathered, tc)
    return torch.stack(gathered)


def _megakernel_config(
    problem: dict,
    *,
    in_kernel_fc2_reduce: bool = False,
    knobs: dict | None = None,
):
    from flashinfer.moe_ep import Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig

    return Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
        enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        knobs=knobs,
    )


def _reference_bf16_mega_moe(
    problem: dict,
    *,
    in_kernel_fc2_reduce: bool = False,
    knobs: dict | None = None,
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        bf16_mega_moe,
        get_symm_buffer_for_bf16_mega_moe,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_bf16_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        knobs=knobs,
    )
    num_tokens = problem["num_tokens"]
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        transformed_l1, transformed_l2 = preprocess_mega_weights(
            MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            intermediate_size=problem["intermediate"],
            hidden_size=problem["hidden"],
        )
        y = torch.empty(
            num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        bf16_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=problem["gate_up_clamp"],
            fast_math=problem["fast_math"],
        )
        torch.cuda.synchronize()
        return y
    finally:
        symm_buffer.destroy()


def _run_mega_layer(
    rank,
    world_size,
    *,
    in_kernel_fc2_reduce: bool = False,
    knobs: dict | None = None,
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)
    kernel = create_mega_kernel(
        _megakernel_config(
            problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce, knobs=knobs
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    try:
        mega = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size, rank=rank, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=problem["num_experts"],
                max_tokens_per_rank=problem["max_tokens"],
                token_hidden_size=problem["hidden"],
            ),
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            backend=MegaConfig(
                megakernel=_megakernel_config(
                    problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce, knobs=knobs
                ),
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)
        t = MoEEpTensors(
            hidden_states=problem["hidden_states"],
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
        )
        y_layer = mega.forward(t).clone()
        y_layer2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        y_ref = _reference_bf16_mega_moe(
            problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce, knobs=knobs
        )
        dist.barrier()
        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        if in_kernel_fc2_reduce:
            _assert_ikr_close(y_layer, y_ref, topk=problem["topk"])
            _assert_ikr_close(y_layer2, y_ref, topk=problem["topk"])
        else:
            torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
            torch.testing.assert_close(y_layer2, y_ref, atol=0.0, rtol=0.0)
        mega.destroy()
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


def _run_mega_torch_oracle(rank, world_size, *, in_kernel_fc2_reduce: bool = False):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        bf16_mega_moe,
        compute_megamoe_reference_bf16,
        get_symm_buffer_for_bf16_mega_moe,
    )

    from .test_mxfp8_cutedsl_preprocess_vs_reference import (
        _assert_mega_oracle_term_band_close,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)
    num_local = problem["num_experts"] // world_size
    forced = (
        torch.arange(min(problem["topk"], world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    problem["topk_ids"][0, : forced.numel()] = forced
    kernel = create_mega_kernel(
        _megakernel_config(problem, in_kernel_fc2_reduce=in_kernel_fc2_reduce)
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    try:
        n = problem["num_tokens"]
        symm_buffer = get_symm_buffer_for_bf16_mega_moe(
            problem["num_experts"],
            problem["max_tokens"],
            problem["topk"],
            problem["hidden"],
            problem["intermediate"],
            rank,
            world_size,
            gate_up_clamp=problem["gate_up_clamp"],
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        )
        try:
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                symm_buffer.x,
                symm_buffer.topk_idx,
                symm_buffer.topk_weights,
            )
            x_local = symm_buffer.x[:n].clone()
            idx_local = symm_buffer.topk_idx[:n].clone()
            w_local = symm_buffer.topk_weights[:n].clone()
            transformed_l1, transformed_l2 = preprocess_mega_weights(
                MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
                intermediate_size=problem["intermediate"],
                hidden_size=problem["hidden"],
            )
            y_kernel = torch.empty(
                n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
            )
            bf16_mega_moe(
                y_kernel,
                transformed_l1,
                transformed_l2,
                symm_buffer,
                num_tokens=n,
                gate_up_clamp=problem["gate_up_clamp"],
                fast_math=problem["fast_math"],
            )
            torch.cuda.synchronize()
            dist.barrier()
            combine_ref = compute_megamoe_reference_bf16(
                input_activation=_all_gather_stack(x_local),
                input_topk_idx=_all_gather_stack(idx_local),
                input_topk_weights=_all_gather_stack(w_local),
                fc1_weight=_all_gather_stack(transformed_l1[0]),
                fc2_weight=_all_gather_stack(transformed_l2[0]),
                ref_compute_graph="deepgemm",
                gate_up_clamp=problem["gate_up_clamp"],
                apply_topk_in_fc1=True,
            )
            yk = y_kernel.to(torch.float32)
            y_ref = combine_ref[rank].to(torch.float32).sum(dim=1)
            rel_l2 = (yk - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
            print(
                f"[bf16 multirank oracle rank {rank} ikr={in_kernel_fc2_reduce}] "
                f"rel_l2={rel_l2.item():.4g}"
            )
            _assert_mega_oracle_term_band_close(
                yk, combine_ref[rank], ikr=in_kernel_fc2_reduce, label=f"rank{rank}"
            )
            assert rel_l2.item() < (0.03 if in_kernel_fc2_reduce else 0.02)
            return rank
        finally:
            symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ("static", "atomic_counter"))
@pytest.mark.parametrize(
    "token_back_mode",
    ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
)
def test_bf16_multirank_modes_are_constructible(load_balance_mode, token_back_mode):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Config

    config = MegaMoEBf16Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=256,
        num_topk=8,
        num_total_experts=256,
        hidden=7168,
        intermediate=2048,
        load_balance_mode=load_balance_mode,  # type: ignore[arg-type]
        token_back_mode=token_back_mode,  # type: ignore[arg-type]
    )
    assert config.num_experts_per_rank == 64


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_bf16_cutedsl_mega_layer_matches_reference():
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size)
    print(f"rank {rank}: sm100_bf16_bf16_bf16_cutedsl mega layer matches reference")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_bf16_cutedsl_mega_layer_in_kernel_fc2_reduce():
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, in_kernel_fc2_reduce=True)
    print(
        f"rank {rank}: sm100_bf16_bf16_bf16_cutedsl mega layer "
        "(in_kernel_fc2_reduce) matches reference within tolerance"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("in_kernel_fc2_reduce", [False, True])
def test_moe_ep_bf16_cutedsl_mega_multirank_torch_oracle(in_kernel_fc2_reduce):
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(
        rank, world_size, in_kernel_fc2_reduce=in_kernel_fc2_reduce
    )
    print(
        f"rank {rank}: sm100_bf16_bf16_bf16_cutedsl mega kernel "
        f"(ikr={in_kernel_fc2_reduce}) matches the multi-rank torch oracle"
    )
