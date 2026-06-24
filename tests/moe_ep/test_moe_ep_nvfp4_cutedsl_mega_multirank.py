"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (nvfp4_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and the ``cutedsl_nvfp4_mega_moe_front_end``
package (``pip install -e cutedsl_megamoe/front_end``).

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

pytest.importorskip("cutedsl_nvfp4_mega_moe_front_end")


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

    from cutedsl_nvfp4_mega_moe_front_end import make_dummy_epilogue_params

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

    from cutedsl_nvfp4_mega_moe_front_end import get_symm_buffer_for_mega_moe, nvfp4_mega_moe
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

    y = torch.empty(
        num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
    )
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

    from cutedsl_nvfp4_mega_moe_front_end import get_symm_buffer_for_mega_moe, nvfp4_mega_moe
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

    y = torch.empty(
        num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
    )
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


def _run_mega_layer(rank, world_size, *, stage_inputs: bool):
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
        _megakernel_config(problem, epilogue_via_config=stage_inputs)
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )

    try:
        if stage_inputs:
            t_hidden = problem["hidden_states"]
            t_scales = None
        else:
            from cutedsl_nvfp4_mega_moe_front_end import get_symm_buffer_for_mega_moe

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
                weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            ),
            backend=MegaConfig(
                megakernel=_megakernel_config(
                    problem, epilogue_via_config=stage_inputs
                ),
                stage_inputs=stage_inputs,
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        tensor_kwargs = {}
        if not stage_inputs:
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

        if stage_inputs:
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
    rank = _run_mega_layer(rank, world_size, stage_inputs=True)
    print(f"rank {rank}: nvfp4_cutedsl mega layer (staged inputs) matches reference")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer (nvfp4_cutedsl) with pre-staged NVFP4 activations.

    Per-expert epilogue scalars are supplied via :class:`MoEEpTensors` and copied
    into the symm workspace during ``stage_inputs``.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, stage_inputs=False)
    print(
        f"rank {rank}: nvfp4_cutedsl mega layer (prestaged inputs) matches reference"
    )


def test_nvfp4_cutedsl_mega_kernel_is_registered():
    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Nvfp4CutedslMegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "nvfp4_cutedsl"
