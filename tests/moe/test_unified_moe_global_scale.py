from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe.api import (
    BackendOptions,
    CakeWarpDecodeConfig,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import MoELayer
from flashinfer.fused_moe.prepare import _resolve_cute_dsl_intermediate_scales
from flashinfer.fused_moe.runners import _fold_trtllm_nvfp4_activation_scale
from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

from .utils import check_accuracy


def _sm100_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _run_nvfp4_calibrated_global_scale_case(
    backend_config,
    backend_key,
    *,
    num_tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    top_k,
    per_token_scale=False,
    routing_input_mode=None,
):
    """Non-unit activation, weight, and intermediate scales reach the kernel."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    x = (torch.randn(num_tokens, hidden_size, device=device) * 0.5).to(torch.bfloat16)
    w1 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        * 0.05
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device) * 0.05
    ).to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device=device)
    topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    if backend_key == "cake":
        topk_weights = topk_weights.to(torch.bfloat16)
    topk_ids = topk_ids.to(torch.int32)

    fp4_range = 448.0 * 6.0
    a1_gs = (fp4_range / x.float().abs().max()).reshape(1)
    w1_gs = fp4_range / w1.float().abs().amax(dim=(1, 2))
    w2_gs = fp4_range / w2.float().abs().amax(dim=(1, 2))
    gemm1 = torch.einsum("th,eoh->teo", x.float(), w1.float())
    intermediate = (
        torch.nn.functional.silu(gemm1[..., intermediate_size:])
        * gemm1[..., :intermediate_size]
    )
    a2_gs = (fp4_range / intermediate.abs().max()).reshape(1)

    quant = QuantConfig(
        weight=QuantFormat.NVFP4,
        activation=QuantFormat.NVFP4,
        per_token_scale=per_token_scale,
    )
    x_q, x_sf = TrtllmFp4Config.prepare_activations(
        x, quant=quant, hidden_states_scale_global=a1_gs
    )
    activation_kwargs = {}
    if routing_input_mode is not None:
        activation_kwargs["routing_input_mode"] = routing_input_mode
    activations = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        per_token_scale=(
            torch.ones(num_tokens, device=device, dtype=torch.float32)
            if per_token_scale
            else None
        ),
        hidden_states_scale_global=a1_gs,
        **activation_kwargs,
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        backend_key,
        backend_config.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=SwiGLU(),
            device=device,
            gemm1_scales_global=w1_gs,
            gemm2_scales_global=w2_gs,
            intermediate_scale_global=a2_gs,
        ),
    )
    config_kwargs = {}
    if backend_key == "cake":
        config_kwargs["execution"] = ExecutionConfig(enable_pdl=True)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(backend_config,)),
        **config_kwargs,
    )
    runner = MoELayer(config).runners[0]
    output = runner.forward(runner.pack_inputs(activations, weights), tactic=-1)

    reference = torch.zeros(num_tokens, hidden_size, device=device)
    for token in range(num_tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            fc1 = torch.mv(w1[expert].float(), x[token].float())
            activated = (
                torch.nn.functional.silu(fc1[intermediate_size:])
                * fc1[:intermediate_size]
            )
            reference[token] += topk_weights[token, slot] * torch.mv(
                w2[expert].float(), activated
            )

    assert torch.isfinite(output).all()
    ok, match_ratio, atol = check_accuracy(
        output.float(), reference, percent_threshold=0.92
    )
    assert ok, (
        f"only {match_ratio:.2%} of FP4 outputs matched "
        f"the magnitude-scaled tolerance (atol={atol:.4g})"
    )


@pytest.mark.skipif(not _sm100_available(), reason="requires SM100")
@pytest.mark.parametrize(
    "backend_config,backend_key,per_token_scale",
    [
        (CuteDslConfig(), "cute_dsl", False),
        (CuteDslConfig(), "cute_dsl", True),
        (TrtllmFp4Config(), "trtllm_fp4_routed", False),
    ],
)
def test_nvfp4_calibrated_global_scales_match_reference(
    backend_config, backend_key, per_token_scale
):
    """Validate calibrated scales, including CuteDSL per-token scaling."""
    _run_nvfp4_calibrated_global_scale_case(
        backend_config,
        backend_key,
        num_tokens=32,
        hidden_size=256,
        intermediate_size=256,
        num_experts=8,
        top_k=2,
        per_token_scale=per_token_scale,
    )


@pytest.mark.skipif(not _sm100_available(), reason="requires SM100")
def test_cake_nvfp4_calibrated_global_scales_match_reference():
    """Validate Cake scale folding on one of its supported SM100 geometries."""
    _run_nvfp4_calibrated_global_scale_case(
        CakeWarpDecodeConfig(backend="cake"),
        "cake",
        num_tokens=4,
        hidden_size=2048,
        intermediate_size=1536,
        num_experts=60,
        top_k=4,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


@pytest.mark.parametrize(
    "scale,match",
    [
        (torch.ones(1, dtype=torch.float64), "torch.float32"),
        (torch.ones(2, dtype=torch.float32), "exactly one element"),
        (torch.tensor([float("nan")]), "finite"),
        (torch.tensor([0.0]), "positive"),
    ],
)
def test_activation_global_scale_is_revalidated_before_folding(scale, match):
    """Reject a global scale mutated after activation-pack construction."""
    act = SimpleNamespace(
        hidden_states_q=torch.empty(1), hidden_states_scale_global=scale
    )
    view = {
        "output1_scale_scalar": torch.ones(2),
        "output1_scale_gate_scalar": torch.ones(2),
    }
    with pytest.raises(ValueError, match=match):
        _fold_trtllm_nvfp4_activation_scale(act, view)

    act.hidden_states_q = torch.empty(1, device="meta")
    act.hidden_states_scale_global = torch.ones(1)
    with pytest.raises(ValueError, match="same device"):
        _fold_trtllm_nvfp4_activation_scale(act, view)


def test_activation_global_scale_is_reshaped_before_folding():
    """Accept any singleton shape without broadcasting the expert scale vector."""
    act = SimpleNamespace(
        hidden_states_q=torch.empty(1),
        hidden_states_scale_global=torch.tensor([[2.0]], dtype=torch.float32),
    )
    view = {
        "output1_scale_scalar": torch.ones(3),
        "output1_scale_gate_scalar": torch.ones(3),
    }
    output1, output1_gate = _fold_trtllm_nvfp4_activation_scale(act, view)
    torch.testing.assert_close(output1, torch.full((3,), 0.5))
    torch.testing.assert_close(output1_gate, torch.full((3,), 0.5))


@pytest.mark.parametrize(
    "calibrated,recipe,expected_calibrated,expected_input",
    [
        (None, None, 1.0, 1.0),
        (torch.tensor([[2.0]]), None, 2.0, 2.0),
        (None, NVFP44Over6Config(e4m3_max=256), 1.0, 1.0 / (6 * 256)),
        (None, NVFP44Over6Config(e4m3_max=448), 1.0, 1.0 / (6 * 448)),
    ],
)
def test_cute_dsl_intermediate_scale_preserves_calibration_and_recipe(
    calibrated, recipe, expected_calibrated, expected_input
):
    weights = torch.empty(1, dtype=torch.bfloat16)
    dequant_scale, input_scale = _resolve_cute_dsl_intermediate_scales(
        weights, calibrated, recipe
    )
    assert dequant_scale.shape == input_scale.shape == (1,)
    assert dequant_scale.dtype == input_scale.dtype == torch.float32
    torch.testing.assert_close(dequant_scale, torch.full((1,), expected_calibrated))
    torch.testing.assert_close(input_scale, torch.full((1,), expected_input))


@pytest.mark.parametrize("e4m3_max", [256, 448])
def test_cute_dsl_intermediate_scale_rejects_conflicting_recipe(e4m3_max):
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_cute_dsl_intermediate_scales(
            torch.empty(1, dtype=torch.bfloat16),
            torch.ones(1),
            NVFP44Over6Config(e4m3_max=e4m3_max),
        )


@pytest.mark.parametrize(
    "recipe",
    [None, NVFP44Over6Config(e4m3_max=256), NVFP44Over6Config(e4m3_max=448)],
)
def test_cute_dsl_prepare_weights_forwards_scales_and_recipe(monkeypatch, recipe):
    from flashinfer.fused_moe import prepare

    captured = {}
    expected = {}

    def capture_weights(*args, **kwargs):
        captured.update(kwargs)
        return expected

    # Spy only on the API handoff; scale arithmetic is tested above without mocks.
    monkeypatch.setattr(prepare, "prepare_cute_dsl_weights", capture_weights)
    gemm1 = torch.tensor([2.0, 3.0])
    gemm2 = torch.tensor([5.0, 7.0])
    intermediate = torch.tensor([11.0]) if recipe is None else None
    result = CuteDslConfig.prepare_weights(
        torch.empty(2, 128, 64, dtype=torch.bfloat16),
        torch.empty(2, 64, 64, dtype=torch.bfloat16),
        num_local_experts=2,
        hidden_size=64,
        intermediate_size=64,
        gemm1_scales_global=gemm1,
        gemm2_scales_global=gemm2,
        intermediate_scale_global=intermediate,
        nvfp4_4over6=recipe,
    )
    assert result is expected
    assert captured["gemm1_scales_global"] is gemm1
    assert captured["gemm2_scales_global"] is gemm2
    assert captured["intermediate_scale_global"] is intermediate
    assert captured["nvfp4_4over6"] is recipe
