"""
Tests for the hybrid CUTLASS-prefill / b12x-decode dispatch in
``B12xMoEWrapper`` (the ``cutlass_prefill_threshold`` knob).

Verifies:
  - Default ``cutlass_prefill_threshold=0`` keeps pure b12x behavior.
  - Threshold > 0: ``num_tokens < threshold`` routes to b12x;
    ``num_tokens >= threshold`` routes to ``cutlass_fused_moe``.
  - Calling without registered cutlass weights raises a clear error.
  - ``FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD`` overrides the kwarg.
  - Both paths produce bf16 output of the expected shape.

The dispatch decision is verified by monkey-patching
``flashinfer.fused_moe.core.cutlass_fused_moe`` with a counting spy — the
underlying b12x kernels still run (or not) for real, so the tests also
cover the wiring end-to-end.
"""

import pytest
import torch
from torch.nn import functional as F

from flashinfer import B12xMoEWrapper
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout


SF_VEC_SIZE = 16
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def _is_sm12x_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_13_or_newer() -> bool:
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
    return (x + y - 1) // y * y


def _make_b12x_weights(
    *, num_local_experts, hidden, intermediate, is_gated, device, seed=0
):
    """B12x-format NVFP4 weights (non-interleaved gate/up, MMA-swizzled SF)."""
    torch.manual_seed(seed)
    w1_rows = (2 if is_gated else 1) * intermediate

    w1_bf16 = (
        torch.randn(
            num_local_experts, w1_rows, hidden, dtype=torch.bfloat16, device=device
        )
        / 10
    )
    w2_bf16 = (
        torch.randn(
            num_local_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
        )
        / 10
    )

    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_bf16.view(num_local_experts * w1_rows, hidden),
        global_scale=w1_gs,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(num_local_experts, w1_rows, hidden // 2)
    w1_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=w1_rows,
        k=hidden,
        num_groups=num_local_experts,
        sf_vec_size=SF_VEC_SIZE,
    )

    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_bf16.view(num_local_experts * hidden, intermediate),
        global_scale=w2_gs,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    w2_q = w2_q_flat.view(num_local_experts, hidden, intermediate // 2)
    w2_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden,
        k=intermediate,
        num_groups=num_local_experts,
        sf_vec_size=SF_VEC_SIZE,
    )

    return {
        "w1_weight": w1_q,
        "w1_weight_sf": w1_sf,
        "w1_alpha": torch.ones(num_local_experts, device=device, dtype=torch.float32),
        "w2_weight": w2_q,
        "w2_weight_sf": w2_sf,
        "w2_alpha": torch.ones(num_local_experts, device=device, dtype=torch.float32),
        "fc2_input_scale": torch.tensor([1.0], device=device, dtype=torch.float32),
    }


def _make_cutlass_weights(
    *, num_experts, hidden, intermediate, is_gated, device, seed=1
):
    """CUTLASS NVFP4-format weights (per-expert global scales + blockscale tensors)."""
    torch.manual_seed(seed)
    w1_n = (2 if is_gated else 1) * intermediate

    w1_bf16 = (
        torch.randn(
            num_experts, w1_n, hidden, dtype=torch.bfloat16, device=device
        )
        / 10
    ).contiguous()
    w2_bf16 = (
        torch.randn(
            num_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
        )
        / 10
    ).contiguous()

    w1_q = torch.empty(
        (num_experts, w1_n, hidden // 2), device=device, dtype=torch.uint8
    )
    w2_q = torch.empty(
        (num_experts, hidden, intermediate // 2), device=device, dtype=torch.uint8
    )
    w1_blockscale = torch.empty(
        (
            num_experts,
            _round_up(w1_n, 128),
            _round_up(hidden // SF_VEC_SIZE, 4),
        ),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2_blockscale = torch.empty(
        (
            num_experts,
            _round_up(hidden, 128),
            _round_up(intermediate // SF_VEC_SIZE, 4),
        ),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w1_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    w2_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)

    for ex in range(num_experts):
        w1_amax = torch.abs(w1_bf16[ex]).max().to(torch.float32)
        w2_amax = torch.abs(w2_bf16[ex]).max().to(torch.float32)
        w1_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_q[ex], w1_blockscale[ex] = fp4_quantize(w1_bf16[ex], w1_gs[ex])
        w2_q[ex], w2_blockscale[ex] = fp4_quantize(w2_bf16[ex], w2_gs[ex])

    a1_gs = torch.ones((), device=device, dtype=torch.float32)
    a2_gs = torch.ones((), device=device, dtype=torch.float32)
    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    return {"w1_q": w1_q, "w2_q": w2_q, "quant_scales": quant_scales}


def _routing(num_tokens, num_experts, top_k, device):
    logits = torch.randn(num_tokens, num_experts, device=device)
    weights = F.softmax(logits, dim=1, dtype=torch.float)
    weights, ids = torch.topk(weights, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).float()
    return weights, ids.to(torch.int32)


def _run(moe, x, b12x_weights, num_experts, top_k, device):
    weights, ids = _routing(x.size(0), num_experts, top_k, device)
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


# Realistic-but-small Nemotron-Super-shaped config: ReLU2 (non-gated), small E.
_CFG = dict(
    num_experts=8,
    num_local_experts=8,
    top_k=2,
    hidden=2048,
    intermediate=1024,
    is_gated=False,
)
_ACTIVATION = "relu2"


@pytest.fixture(scope="module")
def setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not _is_sm12x_supported():
        pytest.skip("SM120/SM121 required")
    if not _cuda_13_or_newer():
        pytest.skip("CUDA 13+ required")

    device = torch.device("cuda")
    return {
        "device": device,
        "b12x": _make_b12x_weights(device=device, **_CFG),
        "cutlass": _make_cutlass_weights(
            num_experts=_CFG["num_experts"],
            hidden=_CFG["hidden"],
            intermediate=_CFG["intermediate"],
            is_gated=_CFG["is_gated"],
            device=device,
        ),
    }


def _spy_cutlass(monkeypatch):
    """Replace cutlass_fused_moe with a counting spy that delegates to the real impl."""
    import flashinfer.fused_moe.core as core

    real = core.cutlass_fused_moe
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("flashinfer.fused_moe.core.cutlass_fused_moe", spy)
    return calls


def _build_wrapper(*, threshold: int) -> B12xMoEWrapper:
    return B12xMoEWrapper(
        num_experts=_CFG["num_experts"],
        top_k=_CFG["top_k"],
        hidden_size=_CFG["hidden"],
        intermediate_size=_CFG["intermediate"],
        num_local_experts=_CFG["num_local_experts"],
        activation=_ACTIVATION,
        use_cuda_graph=False,
        cutlass_prefill_threshold=threshold,
    )


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_threshold_zero_disables_hybrid(setup, monkeypatch):
    """Default threshold=0 → cutlass_fused_moe is never called for any m."""
    moe = _build_wrapper(threshold=0)
    calls = _spy_cutlass(monkeypatch)

    for m in (1, 32, 256):
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(
            moe, x, setup["b12x"], _CFG["num_experts"], _CFG["top_k"], setup["device"]
        )
        assert out.shape == (m, _CFG["hidden"])
        assert out.dtype == torch.bfloat16
    assert calls["n"] == 0


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_decode_below_threshold_uses_b12x(setup, monkeypatch):
    """num_tokens < threshold → b12x kernels run, cutlass_fused_moe is not called."""
    moe = _build_wrapper(threshold=64)
    moe.register_cutlass_prefill_weights(**setup["cutlass"])
    calls = _spy_cutlass(monkeypatch)

    for m in (1, 4, 32, 63):
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(
            moe, x, setup["b12x"], _CFG["num_experts"], _CFG["top_k"], setup["device"]
        )
        assert out.shape == (m, _CFG["hidden"])
        assert torch.isfinite(out).all()
    assert calls["n"] == 0


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_prefill_above_threshold_uses_cutlass(setup, monkeypatch):
    """num_tokens >= threshold → cutlass_fused_moe is invoked once per call."""
    moe = _build_wrapper(threshold=64)
    moe.register_cutlass_prefill_weights(**setup["cutlass"])
    calls = _spy_cutlass(monkeypatch)

    sizes = (64, 128, 256)
    for m in sizes:
        x = (
            torch.randn(m, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
            / 10
        )
        out = _run(
            moe, x, setup["b12x"], _CFG["num_experts"], _CFG["top_k"], setup["device"]
        )
        assert out.shape == (m, _CFG["hidden"])
        assert torch.isfinite(out).all()
    assert calls["n"] == len(sizes)


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_prefill_without_registered_weights_raises(setup):
    """Threshold > 0 + prefill call without registered cutlass weights → RuntimeError."""
    moe = _build_wrapper(threshold=64)
    # Intentionally do NOT call register_cutlass_prefill_weights.

    x = (
        torch.randn(64, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"])
        / 10
    )
    with pytest.raises(RuntimeError, match="register_cutlass_prefill_weights"):
        _run(
            moe, x, setup["b12x"], _CFG["num_experts"], _CFG["top_k"], setup["device"]
        )

    # Decode call must still work even without cutlass weights registered.
    x_dec = (
        torch.randn(1, _CFG["hidden"], dtype=torch.bfloat16, device=setup["device"]) / 10
    )
    out = _run(
        moe, x_dec, setup["b12x"], _CFG["num_experts"], _CFG["top_k"], setup["device"]
    )
    assert out.shape == (1, _CFG["hidden"])


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_env_var_overrides_kwarg(setup, monkeypatch):
    """FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD takes precedence over the kwarg."""
    monkeypatch.setenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", "32")
    moe = _build_wrapper(threshold=0)  # would be disabled without the env override
    assert moe.cutlass_prefill_threshold == 32


@cute_dsl_available
@sm120_required
@cuda_13_required
def test_invalid_env_var_falls_back_to_kwarg(setup, monkeypatch):
    """Non-integer env var is ignored (the kwarg value is kept)."""
    monkeypatch.setenv("FLASHINFER_B12X_CUTLASS_PREFILL_THRESHOLD", "garbage")
    moe = _build_wrapper(threshold=16)
    assert moe.cutlass_prefill_threshold == 16
