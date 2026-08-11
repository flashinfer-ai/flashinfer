"""Single-rank BF16 CuTeDSL MegaMoE integration tests."""

from __future__ import annotations

import pytest


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("BF16 MegaMoE requires sm_100a or sm_103a")


@pytest.mark.arch_blackwell
def test_bf16_megamoe_public_reference_is_lazy():
    """Keep the CPU import boundary free of the CuTeDSL reference dependency."""
    import flashinfer.moe_ep.kernel_src.cutedsl_megamoe as megamoe

    assert callable(megamoe.get_symm_buffer_for_bf16_mega_moe)
    assert callable(megamoe.bf16_mega_moe)
    # Resolving the raw reference happens only on a GPU test host with CuTeDSL.
    assert "compute_megamoe_reference_bf16" in megamoe.__all__


@pytest.mark.arch_blackwell
def test_bf16_kernel_matches_mega_reference(monkeypatch):
    """The public BF16 shim launch matches the independent MegaMoE reference."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        bf16_mega_moe,
        compute_megamoe_reference_bf16,
        get_symm_buffer_for_bf16_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, intermediate = 1024, 1024
    num_tokens, max_tokens, num_experts, topk = 32, 64, 4, 2
    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=generator
    )
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1, sorted=False)
    w13 = torch.randn(
        num_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randn(
        num_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    symm_buffer = get_symm_buffer_for_bf16_mega_moe(
        num_experts,
        max_tokens,
        topk,
        hidden,
        intermediate,
        rank=0,
        world_size=1,
    )
    try:
        stage_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        combine_ref = compute_megamoe_reference_bf16(
            input_activation=symm_buffer.x[:num_tokens].unsqueeze(0),
            input_topk_idx=symm_buffer.topk_idx[:num_tokens].unsqueeze(0),
            input_topk_weights=symm_buffer.topk_weights[:num_tokens].unsqueeze(0),
            fc1_weight=transformed_l1[0].unsqueeze(0),
            fc2_weight=transformed_l2[0].unsqueeze(0),
            ref_compute_graph="deepgemm",
            apply_topk_in_fc1=True,
        )
        y_ref = combine_ref[0].to(torch.float32).sum(dim=1)
        y_kernel = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        bf16_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            sync=True,
        )
        torch.testing.assert_close(
            y_kernel.to(torch.float32), y_ref, atol=8.0, rtol=0.05
        )
    finally:
        symm_buffer.destroy()
