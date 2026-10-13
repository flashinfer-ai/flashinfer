"""Collective tuning result reuse through the MegaMoE layer API."""

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
def test_tuned_knobs_create_fixed_layer(monkeypatch, tmp_path):
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
    candidates = autotune.nvfp4_candidates()[:2]
    monkeypatch.setattr(autotune, "nvfp4_candidates", lambda **kwargs: candidates)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "tuning.json"))
    fleet = FleetParams(
        num_experts=problem["num_experts"],
        max_tokens_per_rank=problem["max_tokens"],
        token_hidden_size=problem["hidden"],
    )
    config = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        gate_up_clamp=problem["gate_up_clamp"],
        knobs="auto",
    )
    weights = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    tensors = MoEEpTensors(
        hidden_states=problem["hidden_states"],
        topk_ids=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
    )
    auto = MoEEpMegaLayer(bootstrap, fleet, weights, MegaConfig(megakernel=config))
    fixed = None
    try:
        with pytest.raises(RuntimeError, match="no completed automatic tuning result"):
            auto.get_tuned_knobs()
        assert auto.warmup(tensors) is None
        knobs = auto.get_tuned_knobs()
        assert knobs in candidates
        expected = auto.forward(tensors).clone()
        torch.cuda.synchronize()
        dist.barrier()

        # Fixed-layer creation must use the returned dictionary without a file.
        monkeypatch.setenv(
            "FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "absent.json")
        )
        fixed = MoEEpMegaLayer(
            bootstrap,
            fleet,
            weights,
            MegaConfig(
                megakernel=dataclasses.replace(config, knobs=knobs),
                preprocess_weights=False,
                transformed_weights=auto._get_transformed_weights(),
            ),
        )
        fixed.warmup(tensors)
        actual = fixed.forward(tensors)
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert fixed._preprocessing_count == 0
    finally:
        torch.cuda.synchronize()
        if fixed is not None:
            fixed.destroy()
        auto.destroy()
