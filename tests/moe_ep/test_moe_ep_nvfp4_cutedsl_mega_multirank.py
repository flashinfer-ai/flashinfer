"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (nvfp4_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and CuTeDSL runtime deps
(``nvidia-cutlass-dsl[cu13]``, ``nvshmem4py-cu13``).  Kernels ship in-tree under
``flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels``.

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

Weights: the CuTeDSL kernel consumes NVFP4 expert weights in kernel-ready
(swizzled scale-factor) layout. These tests pass canonical bf16
:class:`~flashinfer.moe_ep.MoEWeightPack`; the layer quantizes them at init via
``preprocess_weights=True`` (see ``preprocess_mega_weights``). To supply
pre-quantized NVFP4 weights instead, pass ``w13``/``w2`` plus ``w13_scale``/``w2_scale``.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels")


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _launcher_ranks() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size


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


def _make_epilogue_params(rank: int, num_local_experts: int):
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        make_dummy_epilogue_params,
    )

    g = torch.Generator(device="cuda").manual_seed(19 + rank)
    return make_dummy_epilogue_params(num_local_experts, generator=g)


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


def _make_packed_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(31 + rank)
    w13 = torch.randint(
        0,
        256,
        (num_local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        0,
        256,
        (num_local_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w13_scale = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden // 16,
        dtype=torch.float32,
        device="cuda",
        generator=g,
    ).to(torch.float8_e4m3fn)
    w2_scale = torch.randn(
        num_local_experts,
        hidden,
        intermediate // 16,
        dtype=torch.float32,
        device="cuda",
        generator=g,
    ).to(torch.float8_e4m3fn)
    return w13, w2, w13_scale, w2_scale


def _mega_problem(rank: int, world_size: int):
    hidden = 2048
    intermediate = 1024
    num_tokens = 64
    max_tokens = 64
    num_experts = 8
    topk = 4
    gate_up_clamp = 10.0
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
    fc1_alpha, fc2_alpha, fc1_norm_const = _make_epilogue_params(
        rank, num_local_experts
    )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        fast_math=fast_math,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )


def _reference_nvfp4_mega_moe_staged(problem: dict, *, destroy_buffer: bool = True):
    """Reference with bf16 activations staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=problem["fc1_alpha"],
        fc2_alpha=problem["fc2_alpha"],
        fc1_norm_const=problem["fc1_norm_const"],
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
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    nvfp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _reference_nvfp4_mega_moe_prestaged(
    problem: dict, x_nvfp4, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied NVFP4 activations + fp8 block scales."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=problem["fc1_alpha"],
        fc2_alpha=problem["fc2_alpha"],
        fc1_norm_const=problem["fc1_norm_const"],
    )
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].copy_(x_nvfp4)
    symm_buffer.x_sf[:num_tokens].copy_(x_sf)
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    nvfp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _megakernel_config(problem: dict, *, epilogue_via_config: bool):
    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig

    kwargs = dict(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    if epilogue_via_config:
        kwargs.update(
            fc1_alpha=problem["fc1_alpha"],
            fc2_alpha=problem["fc2_alpha"],
            fc1_norm_const=problem["fc1_norm_const"],
        )
    return Nvfp4CutedslMegaMoeConfig(**kwargs)


def _run_mega_layer(rank, world_size, *, quantize_input: bool):
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
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = _mega_problem(rank, world_size)
    kernel = create_mega_kernel(
        _megakernel_config(problem, epilogue_via_config=quantize_input)
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )

    try:
        if quantize_input:
            t_hidden = problem["hidden_states"]
            t_scales = None
        else:
            from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
                get_symm_buffer_for_mega_moe,
            )

            staging_buffer = get_symm_buffer_for_mega_moe(
                problem["num_experts"],
                problem["max_tokens"],
                problem["topk"],
                problem["hidden"],
                2 * problem["intermediate"],
                rank,
                world_size,
                gate_up_clamp=problem["gate_up_clamp"],
            )
            num_tokens = problem["num_tokens"]
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                staging_buffer.x[:num_tokens],
                staging_buffer.x_sf[:num_tokens],
                staging_buffer.topk_idx[:num_tokens],
                staging_buffer.topk_weights[:num_tokens],
            )
            t_hidden = staging_buffer.x[:num_tokens].clone()
            t_scales = staging_buffer.x_sf[:num_tokens].clone()
            staging_buffer.destroy()

        mega = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
            ),
            fleet_params=FleetParams(
                num_experts=problem["num_experts"],
                max_tokens_per_rank=problem["max_tokens"],
                token_hidden_size=problem["hidden"],
            ),
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            backend=MegaConfig(
                megakernel=_megakernel_config(
                    problem, epilogue_via_config=quantize_input
                ),
                quantize_input=quantize_input,
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        tensor_kwargs = {}
        if not quantize_input:
            tensor_kwargs = dict(
                fc1_alpha=problem["fc1_alpha"],
                fc2_alpha=problem["fc2_alpha"],
                fc1_norm_const=problem["fc1_norm_const"],
            )
        t = MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            scales=t_scales,
            **tensor_kwargs,
        )
        y_layer = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()

        if quantize_input:
            y_ref = _reference_nvfp4_mega_moe_staged(problem, destroy_buffer=True)
        else:
            y_ref = _reference_nvfp4_mega_moe_prestaged(
                problem, t_hidden, t_scales, destroy_buffer=True
            )
        dist.barrier()

        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
        mega.destroy()
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_matches_reference():
    """MoEEpMegaLayer (nvfp4_cutedsl) with on-the-fly bf16→NVFP4 staging.

    Per-expert ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` are supplied
    via :class:`Nvfp4CutedslMegaMoeConfig` (workspace allocation).
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True)
    print(f"rank {rank}: nvfp4_cutedsl mega layer (staged inputs) matches reference")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer (nvfp4_cutedsl) with pre-staged NVFP4 activations.

    Per-expert epilogue scalars are supplied via :class:`MoEEpTensors` and copied
    into the symm workspace when ``quantize_input=False``.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=False)
    print(f"rank {rank}: nvfp4_cutedsl mega layer (prestaged inputs) matches reference")


@pytest.mark.arch_blackwell
def test_nvfp4_cutedsl_preprocess_accepts_sglang_packed_weights():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_local_experts = problem["num_experts"] // world_size
    w13, w2, w13_scale, w2_scale = _make_packed_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=problem["hidden"],
        intermediate=problem["intermediate"],
    )

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    fc1_weight, fc1_sf = transformed_l1
    fc2_weight, fc2_sf = transformed_l2
    assert fc1_weight.shape == (
        num_local_experts,
        problem["hidden"] // 2,
        2 * problem["intermediate"],
    )
    assert fc2_weight.shape == (
        num_local_experts,
        problem["intermediate"] // 2,
        problem["hidden"],
    )
    assert fc1_weight.dtype == torch.float4_e2m1fn_x2
    assert fc2_weight.dtype == torch.float4_e2m1fn_x2
    assert fc1_sf.shape[0] == num_local_experts
    assert fc2_sf.shape[0] == num_local_experts


@pytest.mark.arch_blackwell
def test_nvfp4_cutedsl_staging_uses_input_norm_const():
    _require_cuda()

    import torch

    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_tokens = problem["num_tokens"]
    hidden = problem["hidden"]
    topk = problem["topk"]
    sf_cols = hidden // 16

    x_nvfp4_a = torch.empty(
        num_tokens, hidden // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"
    )
    x_nvfp4_b = torch.empty_like(x_nvfp4_a)
    x_sf_a = torch.empty(num_tokens, sf_cols, dtype=torch.float8_e4m3fn, device="cuda")
    x_sf_b = torch.empty_like(x_sf_a)
    topk_idx_a = torch.empty(num_tokens, topk, dtype=torch.int64, device="cuda")
    topk_idx_b = torch.empty_like(topk_idx_a)
    topk_weights_a = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights_b = torch.empty_like(topk_weights_a)

    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        x_nvfp4_a,
        x_sf_a,
        topk_idx_a,
        topk_weights_a,
        norm_const=1.0,
    )
    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        x_nvfp4_b,
        x_sf_b,
        topk_idx_b,
        topk_weights_b,
        norm_const=2.0,
    )

    assert not torch.equal(x_sf_a.view(torch.uint8), x_sf_b.view(torch.uint8))
    torch.testing.assert_close(topk_idx_a, topk_idx_b, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_a, topk_weights_b, atol=0, rtol=0)


def test_nvfp4_cutedsl_mega_kernel_is_registered():
    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Nvfp4CutedslMegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "nvfp4_cutedsl"
