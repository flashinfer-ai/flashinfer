"""Host-side coverage for mixed MXFP8-weight/BF16-activation MegaMoE."""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep import Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_config import (
    Sm100_Bf16_Cutedsl_MegaMoeConfigBase,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    MegaMoEBf16Mxfp8Config,
)


def _config(**kwargs):
    return MegaMoEBf16Mxfp8Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=256,
        num_topk=4,
        num_total_experts=32,
        hidden=1024,
        intermediate=1024,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("mma_tiler_mnk", "transform_buffer", "accumulator_overlap", "transform_k_tile"),
    [
        ((256, 128, 128), "tmem", False, 128),
        ((256, 256, 128), "smem", False, 128),
        ((256, 256, 128), "tmem", True, 64),
    ],
)
def test_mixed_config_accepts_supported_implementation(
    mma_tiler_mnk, transform_buffer, accumulator_overlap, transform_k_tile
):
    config = _config(
        mma_tiler_mnk=mma_tiler_mnk,
        transform_buffer=transform_buffer,
        accumulator_overlap=accumulator_overlap,
        transform_k_tile=transform_k_tile,
    )
    assert config.num_experts_per_rank == 8


def test_mixed_config_rejects_unsupported_implementation_and_token_back():
    with pytest.raises(ValueError, match="implementation tuple"):
        _config(mma_tiler_mnk=(256, 256, 128))
    with pytest.raises(ValueError, match="standalone"):
        _config(token_back_mode="standalone_warps")  # type: ignore[arg-type]


def test_mixed_backend_is_registered():
    backend = create_mega_kernel(
        Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
        )
    )
    assert backend.kernel_name() == "sm100_bf16_mxfp8_bf16_cutedsl"


def test_mixed_config_inherits_bf16_options():
    config = Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        gate_up_clamp=1.5,
        in_kernel_fc2_reduce=True,
    )
    assert isinstance(config, Sm100_Bf16_Cutedsl_MegaMoeConfigBase)
    assert config.gate_up_clamp == 1.5
    assert config.in_kernel_fc2_reduce


def test_mixed_autotune_candidate_matches_default_config():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import bf16_mxfp8_candidates
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16_mxfp8 import (
        MegaMoEBf16Mxfp8Frontend,
    )

    config = _config()
    frontend = MegaMoEBf16Mxfp8Frontend(config)
    assert bf16_mxfp8_candidates() == [
        {
            "mma_tiler_mnk": (256, 128, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
            "cluster_shape_mnk": (2, 1, 1),
            "flag_batch": 1,
            "epi_flag_batch": (1, 1),
            "token_back_mode": "epi_warps",
            "load_balance_mode": "static",
        }
    ]
    frontend.apply_knobs(bf16_mxfp8_candidates()[0])
    assert frontend.config == config


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("kind", "weight_dtype"),
    [
        ("bf16_mxfp8_e4m3", torch.float8_e4m3fn),
        ("bf16_mxfp8_e5m2", torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize(
    ("hidden", "intermediate"),
    [(1024, 1024), (2048, 768)],
)
def test_bf16_mxfp8_kernel_matches_mega_reference(
    monkeypatch, kind, weight_dtype, hidden, intermediate
):
    """The public mixed shim launch matches its BF16-domain torch reference."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("mixed MegaMoE requires sm_100a or sm_103a")
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch lacks E8M0 support required by the mixed reference")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        Mxfp8ScaleDtype,
        autotune_bf16_mxfp8_mega_moe,
        compute_megamoe_reference_bf16_mxfp8,
        get_symm_buffer_for_bf16_mxfp8_mega_moe,
        bf16_mxfp8_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    num_tokens, max_tokens, num_experts, topk = 32, 64, 4, 2
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
        kind=kind,
    )
    assert transformed_l1[0].dtype == weight_dtype
    assert transformed_l2[0].dtype == weight_dtype
    symm_buffer = get_symm_buffer_for_bf16_mxfp8_mega_moe(
        num_experts,
        max_tokens,
        topk,
        hidden,
        intermediate,
        rank=0,
        world_size=1,
        kind=kind,
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
        autotune_bf16_mxfp8_mega_moe(
            torch.empty_like(hidden_states),
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            warmup_iters=0,
            timed_iters=1,
        )
        combine_ref = compute_megamoe_reference_bf16_mxfp8(
            input_activation=symm_buffer.x[:num_tokens].unsqueeze(0),
            input_topk_idx=symm_buffer.topk_idx[:num_tokens].unsqueeze(0),
            input_topk_weights=symm_buffer.topk_weights[:num_tokens].unsqueeze(0),
            fc1_weight=transformed_l1[0].unsqueeze(0),
            fc1_weight_sf=transformed_l1[1].view(Mxfp8ScaleDtype).unsqueeze(0),
            fc2_weight=transformed_l2[0].unsqueeze(0),
            fc2_weight_sf=transformed_l2[1].view(Mxfp8ScaleDtype).unsqueeze(0),
            ref_compute_graph="deepgemm",
            apply_topk_in_fc1=True,
        )
        y_ref = combine_ref[0].to(torch.float32).sum(dim=1)
        y_kernel = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        bf16_mxfp8_mega_moe(
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


@pytest.mark.arch_blackwell
def test_bf16_mxfp8_preprocesses_canonical_prequantized_weights():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        Mxfp8ScaleDtype,
        mxfp8_quantize_per_block_32,
    )

    experts, hidden, intermediate = 2, 128, 128
    generator = torch.Generator(device="cuda").manual_seed(31)
    w13 = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randn(
        experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    def quantize_canonical(weights):
        quantized, scales = zip(
            *(
                mxfp8_quantize_per_block_32(
                    weight.to(torch.float32), torch.float8_e4m3fn
                )
                for weight in weights
            ),
            strict=True,
        )
        return torch.stack(quantized), torch.stack(scales)

    w13_q, w13_scale = quantize_canonical(w13)
    w2_q, w2_scale = quantize_canonical(w2)
    transformed_from_bf16 = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    transformed_from_mxfp8 = preprocess_mega_weights(
        MoEWeightPack(
            w13=w13_q,
            w2=w2_q,
            w13_scale=w13_scale.view(torch.uint8),
            w2_scale=w2_scale.view(torch.uint8),
        ),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )

    for (expected_weight, expected_scale), (actual_weight, actual_scale) in zip(
        transformed_from_bf16, transformed_from_mxfp8, strict=True
    ):
        assert actual_weight.stride(1) == 1
        assert actual_scale.dtype == Mxfp8ScaleDtype
        torch.testing.assert_close(
            actual_weight.view(torch.uint8), expected_weight.view(torch.uint8)
        )
        torch.testing.assert_close(
            actual_scale.view(torch.uint8), expected_scale.view(torch.uint8)
        )
