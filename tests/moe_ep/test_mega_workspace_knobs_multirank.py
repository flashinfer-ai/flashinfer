"""Fixed workspace knobs and graph replay through the MegaMoE layer API."""

from __future__ import annotations

import dataclasses

import pytest

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import (
    _launcher_ranks,
    _mega_problem,
    _require_cuda,
)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_explicit_workspace_knobs_and_graph_replay(monkeypatch, tmp_path):
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("needs >=2 ranks")

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)
    knobs, alternative = autotune.nvfp4_candidates()[:2]
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "absent.json"))
    fleet = FleetParams(
        num_experts=problem["num_experts"],
        max_tokens_per_rank=problem["max_tokens"],
        token_hidden_size=problem["hidden"],
    )
    config = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        gate_up_clamp=problem["gate_up_clamp"],
        knobs=knobs,
    )
    weights = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    tensors = MoEEpTensors(
        hidden_states=problem["hidden_states"],
        topk_ids=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
    )
    reference = MoEEpMegaLayer(bootstrap, fleet, weights, MegaConfig(megakernel=config))
    fixed = None
    graphs = []
    try:
        reference.warmup(tensors)
        expected = reference.forward(tensors).clone()
        torch.cuda.synchronize()
        dist.barrier()

        fixed = MoEEpMegaLayer(
            bootstrap,
            fleet,
            weights,
            MegaConfig(
                megakernel=dataclasses.replace(config, knobs=None),
                preprocess_weights=False,
                transformed_weights=reference._get_transformed_weights(),
            ),
        )
        selected = fixed.create_workspace(fleet.max_tokens_per_rank, knobs=knobs)
        larger = fixed.create_workspace(
            2 * fleet.max_tokens_per_rank, knobs=alternative
        )
        profiles = [selected, larger]
        assert selected._backend_workspace is not larger._backend_workspace
        for profile in profiles:
            fixed.warmup(tensors, workspace=profile)
            actual = fixed.forward(tensors, workspace=profile)
            torch.cuda.synchronize()
            dist.barrier()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fixed.forward(tensors, workspace=profile)
            graphs.append((graph, captured, profile))
            dist.barrier()

        tensors.hidden_states.mul_(0.5)
        for graph, captured, profile in graphs:
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            replayed = captured.clone()
            actual = fixed.forward(tensors, workspace=profile)
            torch.cuda.synchronize()
            dist.barrier()
            torch.testing.assert_close(replayed, actual, rtol=0, atol=0)
            assert not torch.equal(actual, expected)
        assert fixed._preprocessing_count == 0
    finally:
        torch.cuda.synchronize()
        for graph, _, _ in graphs:
            graph.reset()
        graphs.clear()
        if fixed is not None:
            fixed.destroy()
        reference.destroy()
