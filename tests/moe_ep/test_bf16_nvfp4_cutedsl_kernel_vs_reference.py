"""Single-rank mixed NVFP4-weight/BF16-activation MegaMoE integration tests."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep.core.validation.common import is_bf16_nvfp4_cutedsl_supported

cuda_13_required = pytest.mark.skipif(
    not is_bf16_nvfp4_cutedsl_supported(),
    reason="bf16_nvfp4 requires CUDA 13+",
)


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("mixed MegaMoE requires sm_100a or sm_103a")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch lacks NVFP4 support required by the mixed reference")


def test_bf16_nvfp4_rejects_legacy_logical_prequantized_shapes():
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    experts, hidden, intermediate = 2, 128, 128
    weights = MoEWeightPack(
        w13=torch.zeros(experts, 2 * intermediate, hidden, dtype=torch.uint8),
        w2=torch.zeros(experts, hidden, intermediate, dtype=torch.uint8),
        w13_scale=torch.zeros(experts, 2 * intermediate, hidden // 16),
        w2_scale=torch.zeros(experts, hidden, intermediate // 16),
    )
    with pytest.raises(ValueError, match="must have packed shapes"):
        preprocess_mega_weights(
            weights,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )


@cuda_13_required
@pytest.mark.arch_blackwell
def test_bf16_nvfp4_prequantized_uint8_scales_and_stride_validation():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    experts, hidden, intermediate = 2, 128, 128
    weights = MoEWeightPack(
        w13=torch.randint(
            0,
            256,
            (experts, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
            device="cuda",
        ),
        w2=torch.randint(
            0,
            256,
            (experts, hidden, intermediate // 2),
            dtype=torch.uint8,
            device="cuda",
        ),
        w13_scale=torch.randint(
            0,
            128,
            (experts, 2 * intermediate, hidden // 16),
            dtype=torch.uint8,
            device="cuda",
        ),
        w2_scale=torch.randint(
            0,
            128,
            (experts, hidden, intermediate // 16),
            dtype=torch.uint8,
            device="cuda",
        ),
    )
    transformed = preprocess_mega_weights(
        weights,
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    assert transformed[0][1].dtype == torch.float8_e4m3fn
    assert transformed[1][1].dtype == torch.float8_e4m3fn
    validate_transformed_mega_weights(
        transformed,
        intermediate_size=intermediate,
        hidden_size=hidden,
        world_size=1,
        num_experts=experts,
    )

    n_major = (
        (transformed[0][0].contiguous(), transformed[0][1]),
        transformed[1],
    )
    with pytest.raises(MoEEpConfigError, match="must be K-major"):
        validate_transformed_mega_weights(
            n_major,
            intermediate_size=intermediate,
            hidden_size=hidden,
            world_size=1,
            num_experts=experts,
        )


@cuda_13_required
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("hidden", "intermediate", "num_experts", "topk"),
    [
        pytest.param(1024, 1024, 4, 2, id="square-e4"),
        pytest.param(1056, 1088, 1, 1, id="nonsquare-singleton-e1"),
    ],
)
def test_bf16_nvfp4_kernel_matches_mega_reference(
    monkeypatch, hidden, intermediate, num_experts, topk
):
    """The public mixed shim launch matches its BF16-domain torch reference."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        bf16_nvfp4_mega_launch_thunk,
        compute_megamoe_reference_bf16_nvfp4,
        get_symm_buffer_for_bf16_nvfp4_mega_moe,
        bf16_nvfp4_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    num_tokens, max_tokens = 32, 64
    generator = torch.Generator(device="cuda").manual_seed(29)
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
    assert transformed_l1[0].dtype == torch.float4_e2m1fn_x2
    assert transformed_l2[0].dtype == torch.float4_e2m1fn_x2
    fc1_alpha = torch.linspace(0.75, 1.25, num_experts, device="cuda")
    fc2_alpha = torch.linspace(1.25, 0.75, num_experts, device="cuda")
    symm_buffer = get_symm_buffer_for_bf16_nvfp4_mega_moe(
        num_experts,
        max_tokens,
        topk,
        hidden,
        intermediate,
        rank=0,
        world_size=1,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
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
        combine_ref = compute_megamoe_reference_bf16_nvfp4(
            input_activation=symm_buffer.x[:num_tokens].unsqueeze(0),
            input_topk_idx=symm_buffer.topk_idx[:num_tokens].unsqueeze(0),
            input_topk_weights=symm_buffer.topk_weights[:num_tokens].unsqueeze(0),
            fc1_weight=transformed_l1[0].unsqueeze(0),
            fc1_weight_sf=transformed_l1[1].view(torch.float8_e4m3fn).unsqueeze(0),
            fc2_weight=transformed_l2[0].unsqueeze(0),
            fc2_weight_sf=transformed_l2[1].view(torch.float8_e4m3fn).unsqueeze(0),
            ref_compute_graph="deepgemm",
            apply_topk_in_fc1=True,
            fc1_alpha=symm_buffer.fc1_alpha.unsqueeze(0),
            fc2_alpha=symm_buffer.fc2_alpha.unsqueeze(0),
        )
        y_ref = combine_ref[0].to(torch.float32).sum(dim=1)
        y_kernel = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        bf16_nvfp4_mega_moe(
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

        # Form-A must clear skipped routes on every launch. Otherwise a route
        # masked after a prior live launch leaks its stale partial into sum().
        # Padding is intentionally untouched: clearing it can write hundreds
        # of megabytes for a small live batch in a large-capacity workspace.
        symm_buffer.topk_idx[0, 0] = -1
        symm_buffer.combine_output[0, 0].fill_(123)
        symm_buffer.combine_output[num_tokens:].fill_(77)
        bf16_nvfp4_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            sync=True,
        )
        assert torch.count_nonzero(symm_buffer.combine_output[0, 0]) == 0
        assert torch.all(symm_buffer.combine_output[num_tokens:] == 77)

        symm_buffer.combine_output[num_tokens:].fill_(66)
        launch = bf16_nvfp4_mega_launch_thunk(
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
        )
        launch()
        torch.cuda.synchronize()
        assert torch.all(symm_buffer.combine_output[num_tokens:] == 66)

        empty = torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
        bf16_nvfp4_mega_moe(
            empty,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=0,
            sync=True,
        )
    finally:
        symm_buffer.destroy()
