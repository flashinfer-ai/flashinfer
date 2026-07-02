"""Tests for hybrid CUTLASS-prefill / b12x-decode dispatch.

The tests exercise dispatch at the threshold, verify numerical parity between
the two physical FP4 weight layouts, and confirm CUTLASS's BF16-input NVFP4
quantization convention.
"""

import pytest
import torch
from torch.nn import functional as F

from flashinfer import ActivationType, B12xMoEWrapper, cutlass_fused_moe
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize


SF_VEC_SIZE = 16


def _is_sm12x_supported() -> bool:
    """Return whether the active CUDA device supports b12x kernels."""
    if not torch.cuda.is_available():
        return False
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_13_or_newer() -> bool:
    """Return whether FlashInfer is using CUDA 13 or newer."""
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(), reason="Requires SM120/SM121 GPU"
)
cuda_13_required = pytest.mark.skipif(
    not _cuda_13_or_newer(), reason="b12x fused MoE requires CUDA 13 or later"
)


def _round_up(x: int, y: int) -> int:
    """Round ``x`` up to a multiple of ``y``."""
    return (x + y - 1) // y * y


def _make_bf16_weights(*, num_experts, hidden, intermediate, is_gated, device, seed=0):
    """Create one BF16 baseline shared by both FP4 physical layouts."""
    torch.manual_seed(seed)
    w1_rows = (2 if is_gated else 1) * intermediate
    w1_bf16 = (
        torch.randn(num_experts, w1_rows, hidden, dtype=torch.bfloat16, device=device)
        / 10
    ).contiguous()
    w2_bf16 = (
        torch.randn(
            num_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
        )
        / 10
    ).contiguous()
    return w1_bf16, w2_bf16


def _make_matching_fp4_weights(w1_bf16, w2_bf16):
    """Build both backend views from one quantization of each BF16 weight.

    The packed FP4 values and E4M3 scale-factor samples are shared. CUTLASS
    consumes the normalized 2D SF storage directly, while b12x consumes an MMA
    view of that same storage. Non-unit activation global scales make this
    fixture sensitive to accidentally folding an activation scale into the
    b12x weight scales.
    """
    num_experts, w1_rows, hidden = w1_bf16.shape
    intermediate = w2_bf16.shape[2]
    device = w1_bf16.device

    weight_gs = torch.ones((), device=device, dtype=torch.float32)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_bf16.view(num_experts * w1_rows, hidden),
        global_scale=weight_gs,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_bf16.view(num_experts * hidden, intermediate),
        global_scale=weight_gs,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(num_experts, w1_rows, hidden // 2)
    w2_q = w2_q_flat.view(num_experts, hidden, intermediate // 2)
    w1_blockscale = w1_sf_flat.view(
        num_experts,
        _round_up(w1_rows, 128),
        _round_up(hidden // SF_VEC_SIZE, 4),
    )
    w2_blockscale = w2_sf_flat.view(
        num_experts,
        _round_up(hidden, 128),
        _round_up(intermediate // SF_VEC_SIZE, 4),
    )

    # These are deliberately far from one. CUTLASS consumes the large global
    # scales; b12x consumes their reciprocals (the small activation scales).
    # Both kernels must cancel them in their respective quant/dequant paths.
    a1_gs = torch.tensor(16.0, device=device, dtype=torch.float32)
    a2_gs = torch.tensor(8.0, device=device, dtype=torch.float32)
    w1_alpha = (1.0 / a1_gs).expand(num_experts).contiguous()
    w2_alpha = (1.0 / a2_gs).expand(num_experts).contiguous()
    b12x = {
        "w1_weight": w1_q,
        "w1_weight_sf": convert_sf_to_mma_layout(
            w1_sf_flat,
            m=w1_rows,
            k=hidden,
            num_groups=num_experts,
            sf_vec_size=SF_VEC_SIZE,
        ),
        "w1_alpha": w1_alpha,
        "w2_weight": w2_q,
        "w2_weight_sf": convert_sf_to_mma_layout(
            w2_sf_flat,
            m=hidden,
            k=intermediate,
            num_groups=num_experts,
            sf_vec_size=SF_VEC_SIZE,
        ),
        "w2_alpha": w2_alpha,
        "fc2_input_scale": w2_alpha,
    }
    cutlass = {
        "w1_q": w1_q,
        "w2_q": w2_q,
        "quant_scales": [
            a1_gs,
            w1_blockscale.view(torch.int32),
            w1_alpha.clone(),
            a2_gs,
            w2_blockscale.view(torch.int32),
            w2_alpha.clone(),
        ],
        "a1_gs": a1_gs,
    }
    return {
        "b12x": b12x,
        "cutlass": cutlass,
    }


def _routing(num_tokens, num_experts, top_k, device):
    """Create normalized top-k routing weights and int32 expert IDs."""
    logits = torch.randn(num_tokens, num_experts, device=device)
    weights = F.softmax(logits, dim=1, dtype=torch.float)
    weights, ids = torch.topk(weights, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).float()
    return weights, ids.to(torch.int32)


def _run(moe, x, b12x_weights, routing=None):
    """Run a wrapper with shared b12x weights and optionally fixed routing."""
    if routing is None:
        routing = _routing(x.size(0), _CFG["num_experts"], _CFG["top_k"], x.device)
    weights, ids = routing
    return moe.run(
        x=x,
        w1_weight=b12x_weights["w1_weight"],
        w1_weight_sf=b12x_weights["w1_weight_sf"],
        w1_alpha=b12x_weights["w1_alpha"],
        fc2_input_scale=b12x_weights["fc2_input_scale"],
        w2_weight=b12x_weights["w2_weight"],
        w2_weight_sf=b12x_weights["w2_weight_sf"],
        w2_alpha=b12x_weights["w2_alpha"],
        token_selected_experts=ids,
        token_final_scales=weights,
    )


def _run_cutlass(x, cutlass_weights, routing):
    """Run standalone CUTLASS with the exact tensors registered by hybrid."""
    weights, ids = routing
    output = torch.empty_like(x)
    cutlass_fused_moe(
        input=x,
        token_selected_experts=ids,
        token_final_scales=weights,
        fc1_expert_weights=cutlass_weights["w1_q"].contiguous().view(torch.long),
        fc2_expert_weights=cutlass_weights["w2_q"].contiguous().view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=cutlass_weights["quant_scales"],
        input_sf=None,
        output=output,
        activation_type=ActivationType.Relu2,
    )
    return output


def _assert_cross_backend_close(actual, expected):
    """Require scale-sensitive agreement between independent FP4 kernels."""
    actual = actual.float()
    expected = expected.float()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    reference_rms = expected.square().mean().sqrt().clamp_min(1e-6)
    nrmse = ((actual - expected).square().mean().sqrt() / reference_rms).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    assert nrmse < 0.05, (
        f"normalized RMSE {nrmse:.5f} exceeds the FP4 parity limit 0.05"
    )
    assert cosine > 0.999, (
        f"cosine similarity {cosine:.6f} is below the FP4 parity limit 0.999"
    )


def _assert_b12x_repeatability(actual, expected):
    """Bound atomic accumulation-order noise between two B12x launches."""
    actual = actual.float()
    expected = expected.float()
    reference_rms = expected.square().mean().sqrt().clamp_min(1e-6)
    nrmse = ((actual - expected).square().mean().sqrt() / reference_rms).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    assert nrmse < 0.02, f"B12x repeatability NRMSE {nrmse:.6f} exceeds 0.02"
    assert cosine > 0.9998, f"B12x repeatability cosine {cosine:.6f} is below 0.9998"


# Realistic-but-small Nemotron-Super-shaped config: ReLU2 (non-gated), small E.
_CFG = {
    "num_experts": 8,
    "top_k": 2,
    "hidden": 2048,
    "intermediate": 1024,
    "is_gated": False,
}
_ACTIVATION = "relu2"
_UNSET = object()


@pytest.fixture(scope="module")
def setup():
    """Build matching b12x and CUTLASS weights on a supported device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not _is_sm12x_supported():
        pytest.skip("SM120/SM121 required")
    if not _cuda_13_or_newer():
        pytest.skip("CUDA 13+ required")

    device = torch.device("cuda")
    baseline = _make_bf16_weights(
        num_experts=_CFG["num_experts"],
        hidden=_CFG["hidden"],
        intermediate=_CFG["intermediate"],
        is_gated=_CFG["is_gated"],
        device=device,
    )
    packed = _make_matching_fp4_weights(*baseline)

    # The two paths consume identical packed FP4 bytes and identical E4M3 SF
    # samples; only the required scale-factor view and alpha contract differ.
    assert torch.equal(packed["b12x"]["w1_weight"], packed["cutlass"]["w1_q"])
    assert torch.equal(packed["b12x"]["w2_weight"], packed["cutlass"]["w2_q"])
    return {"device": device, **packed}


def _spy_cutlass(monkeypatch):
    """Replace CUTLASS MoE with a counting spy that delegates to the kernel."""
    import flashinfer.fused_moe.core as core

    real = core.cutlass_fused_moe
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("flashinfer.fused_moe.core.cutlass_fused_moe", spy)
    return calls


def _register_cutlass(moe, weights):
    """Register only the three tensors accepted by the public wrapper API."""
    moe.register_cutlass_prefill_weights(
        w1_q=weights["w1_q"],
        w2_q=weights["w2_q"],
        quant_scales=weights["quant_scales"],
    )


def _build_wrapper(
    *,
    threshold=_UNSET,
    activation=_ACTIVATION,
    quant_mode=None,
    swiglu_limit=None,
):
    """Build a wrapper while preserving omitted-threshold semantics."""
    kwargs = {}
    if threshold is not _UNSET:
        kwargs["cutlass_prefill_threshold"] = threshold
    if quant_mode is not None:
        kwargs["quant_mode"] = quant_mode
    return B12xMoEWrapper(
        num_experts=_CFG["num_experts"],
        top_k=_CFG["top_k"],
        hidden_size=_CFG["hidden"],
        intermediate_size=_CFG["intermediate"],
        activation=activation,
        swiglu_limit=swiglu_limit,
        use_cuda_graph=False,
        **kwargs,
    )


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_default_threshold_disables_hybrid(setup, monkeypatch):
    """An omitted threshold defaults to pure b12x behavior."""
    monkeypatch.delenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", raising=False)
    moe = _build_wrapper()
    calls = _spy_cutlass(monkeypatch)
    assert moe.cutlass_prefill_threshold == 0

    for m in (1, 32, 256):
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(moe, x, setup["b12x"])
        assert out.shape == (m, _CFG["hidden"])
        assert out.dtype == torch.bfloat16
    assert calls["n"] == 0


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_decode_below_threshold_uses_b12x(setup, monkeypatch):
    """Token counts below the threshold stay on b12x."""
    moe = _build_wrapper(threshold=64)
    _register_cutlass(moe, setup["cutlass"])
    calls = _spy_cutlass(monkeypatch)

    for m in (1, 4, 32, 63):
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(moe, x, setup["b12x"])
        assert out.shape == (m, _CFG["hidden"])
        assert torch.isfinite(out).all()
    assert calls["n"] == 0


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_prefill_at_or_above_threshold_uses_cutlass(setup, monkeypatch):
    """Token counts at or above the threshold run CUTLASS once per call."""
    moe = _build_wrapper(threshold=64)
    _register_cutlass(moe, setup["cutlass"])
    calls = _spy_cutlass(monkeypatch)

    sizes = (64, 128, 256)
    for m in sizes:
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(moe, x, setup["b12x"])
        assert out.shape == (m, _CFG["hidden"])
        assert torch.isfinite(out).all()
    assert calls["n"] == len(sizes)


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_prefill_without_registered_weights_raises(setup):
    """Hybrid prefill requires separately registered CUTLASS weights."""
    moe = _build_wrapper(threshold=64)
    x = (
        torch.randn(64, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
        / 10
    )
    with pytest.raises(RuntimeError, match="register_cutlass_prefill_weights"):
        _run(moe, x, setup["b12x"])

    x_decode = (
        torch.randn(1, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
        / 10
    )
    assert _run(moe, x_decode, setup["b12x"]).shape == (1, _CFG["hidden"])


@pytest.mark.parametrize("num_tokens", [63, 64, 65])
@cute_dsl_available
@sm120_required
@cuda_13_required
def test_hybrid_branch_matches_corresponding_standalone(setup, num_tokens):
    """Each hybrid branch numerically matches its standalone backend."""
    threshold = 64
    torch.manual_seed(123)
    x = (
        torch.randn(
            num_tokens,
            _CFG["hidden"],
            dtype=torch.bfloat16,
            device=setup["device"],
        )
        / 10
    )
    routing = _routing(num_tokens, _CFG["num_experts"], _CFG["top_k"], setup["device"])

    hybrid = _build_wrapper(threshold=threshold)
    _register_cutlass(hybrid, setup["cutlass"])
    actual = _run(hybrid, x, setup["b12x"], routing)

    if num_tokens < threshold:
        standalone = _build_wrapper(threshold=0)
        expected = _run(standalone, x, setup["b12x"], routing)
    else:
        expected = _run_cutlass(x, setup["cutlass"], routing)

    if num_tokens < threshold:
        # B12x scatters top-k contributions with atomics; independent launches
        # can differ by a small BF16 accumulation-order error.
        _assert_b12x_repeatability(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_cutlass_and_b12x_match_for_identical_quantized_weights(setup):
    """Standalone kernels agree on one FP4 tensor with non-unit scales."""
    num_tokens = 64
    torch.manual_seed(456)
    x = (
        torch.randn(
            num_tokens,
            _CFG["hidden"],
            dtype=torch.bfloat16,
            device=setup["device"],
        )
        / 10
    )
    routing = _routing(num_tokens, _CFG["num_experts"], _CFG["top_k"], setup["device"])

    b12x = _build_wrapper(threshold=0)
    b12x_output = _run(b12x, x, setup["b12x"], routing)
    cutlass_output = _run_cutlass(x, setup["cutlass"], routing)

    _assert_cross_backend_close(cutlass_output, b12x_output)


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_cutlass_bf16_input_matches_explicit_fp4_quantization(setup):
    """BF16 input with no input_sf matches explicit NVFP4 input quantization."""
    num_tokens = 64
    torch.manual_seed(321)
    x_bf16 = (
        torch.randn(
            num_tokens,
            _CFG["hidden"],
            dtype=torch.bfloat16,
            device=setup["device"],
        )
        / 10
    )
    routing_weights, selected_experts = _routing(
        num_tokens, _CFG["num_experts"], _CFG["top_k"], setup["device"]
    )
    x_fp4, x_sf = fp4_quantize(x_bf16, setup["cutlass"]["a1_gs"])
    w1 = setup["cutlass"]["w1_q"].contiguous().view(torch.long)
    w2 = setup["cutlass"]["w2_q"].contiguous().view(torch.long)
    bf16_output = torch.empty_like(x_bf16)
    fp4_output = torch.empty_like(x_bf16)

    common = {
        "token_selected_experts": selected_experts,
        "token_final_scales": routing_weights,
        "fc1_expert_weights": w1,
        "fc2_expert_weights": w2,
        "output_dtype": torch.bfloat16,
        "quant_scales": setup["cutlass"]["quant_scales"],
        "activation_type": ActivationType.Relu2,
    }
    cutlass_fused_moe(
        input=x_bf16,
        input_sf=None,
        output=bf16_output,
        **common,
    )
    cutlass_fused_moe(
        input=x_fp4,
        input_sf=x_sf,
        output=fp4_output,
        **common,
    )
    torch.testing.assert_close(bf16_output, fp4_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("silu", ActivationType.Swiglu),
        ("gelu_tanh", ActivationType.GegluTanh),
        ("swigluoai_uninterleave", ActivationType.Swiglu),
        ("relu2", ActivationType.Relu2),
    ],
)
@cute_dsl_available
@sm120_required
@cuda_13_required
def test_cutlass_activation_mapping(setup, monkeypatch, activation, expected):
    """Every current B12x activation maps explicitly to CUTLASS."""
    import flashinfer.fused_moe.core as core

    captured = {}

    def fake_cutlass(**kwargs):
        captured.update(kwargs)
        return [kwargs["output"]]

    monkeypatch.setattr(core, "cutlass_fused_moe", fake_cutlass)
    limit = 7.0 if activation == "swigluoai_uninterleave" else None
    moe = _build_wrapper(threshold=1, activation=activation, swiglu_limit=limit)
    _register_cutlass(moe, setup["cutlass"])
    routing_weights, selected_experts = _routing(
        1, _CFG["num_experts"], _CFG["top_k"], setup["device"]
    )
    output = torch.empty(
        1, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"]
    )
    result = moe._run_cutlass_prefill(
        x=torch.zeros_like(output),
        token_selected_experts=selected_experts,
        token_final_scales=routing_weights,
        output=output,
    )

    assert result is output
    assert captured["activation_type"] == expected
    if activation == "swigluoai_uninterleave":
        assert torch.all(captured["swiglu_alpha"] == moe.swiglu_alpha)
        assert torch.all(captured["swiglu_beta"] == moe.swiglu_beta)
        assert torch.all(captured["swiglu_limit"] == moe.swiglu_limit)
    else:
        assert captured["swiglu_alpha"] is None
        assert captured["swiglu_beta"] is None
        assert captured["swiglu_limit"] is None


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_unsupported_cutlass_activation_raises(setup):
    """Hybrid dispatch never silently falls through to another activation."""
    moe = _build_wrapper(threshold=1, activation="unsupported")
    _register_cutlass(moe, setup["cutlass"])
    routing_weights, selected_experts = _routing(
        1, _CFG["num_experts"], _CFG["top_k"], setup["device"]
    )
    output = torch.empty(
        1, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"]
    )
    with pytest.raises(ValueError, match="does not support activation"):
        moe._run_cutlass_prefill(
            x=torch.zeros_like(output),
            token_selected_experts=selected_experts,
            token_final_scales=routing_weights,
            output=output,
        )


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_env_var_is_used_only_when_threshold_is_omitted(monkeypatch):
    """The environment supplies the threshold only when the kwarg is omitted."""
    monkeypatch.setenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", "32")
    assert _build_wrapper().cutlass_prefill_threshold == 32
    assert _build_wrapper(threshold=0).cutlass_prefill_threshold == 0
    assert _build_wrapper(threshold=16).cutlass_prefill_threshold == 16


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_invalid_env_var_uses_effective_default(monkeypatch):
    """An invalid environment value leaves the effective default at zero."""
    monkeypatch.setenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", "garbage")
    assert _build_wrapper().cutlass_prefill_threshold == 0
    assert _build_wrapper(threshold=16).cutlass_prefill_threshold == 16


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_hybrid_dispatch_rejects_w4a16(monkeypatch):
    """Hybrid CUTLASS dispatch is restricted to NVFP4 weights."""
    monkeypatch.delenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", raising=False)
    with pytest.raises(NotImplementedError, match="supports only.*nvfp4"):
        _build_wrapper(threshold=1, quant_mode="w4a16")
