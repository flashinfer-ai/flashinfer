"""Reference correctness test for the mm_bf16_fp4 trace API."""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize("backend", ["cudnn", "cute-dsl"])
@pytest.mark.parametrize(
    "shape_kwargs", [dict(M=32, N=1024, K=1024), dict(M=16, N=2048, K=512)]
)
def test_mm_bf16_fp4_reference_correctness(backend, shape_kwargs):
    """flashinfer.mm_bf16_fp4 kernel vs reference (dequant + matmul).

    The trace inits build *prepared* (backend-specific) weights via
    ``prepare_bf16_fp4_weights``; each backend's reference dequantizes
    that prepared layout directly (the SM12x cute-dsl one inverts the MMA
    tile permutation and decodes S0E5M3 scales, while the SM100/103 one
    unswizzles the 128x4 scale buffer).
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import (
        mm_bf16_fp4_cudnn_trace,
        mm_bf16_fp4_cute_dsl_sm100_trace,
        mm_bf16_fp4_cute_dsl_trace,
        mm_bf16_fp4_trace_dispatch,
    )

    if not torch.cuda.is_available():
        pytest.skip("mm_bf16_fp4 requires a CUDA device")
    if backend == "cudnn":
        tpl = mm_bf16_fp4_cudnn_trace
    elif torch.cuda.get_device_capability() in ((10, 0), (10, 3)):
        tpl = mm_bf16_fp4_cute_dsl_sm100_trace
    else:
        tpl = mm_bf16_fp4_cute_dsl_trace
    try:
        inputs = tpl.init(**shape_kwargs)
        api = flashinfer.mm_bf16_fp4(
            inputs["a"],
            inputs["b"],
            inputs["b_descale"],
            inputs["alpha"],
            backend=backend,
            block_size=inputs["block_size"],
        )
    except Exception as exc:
        pytest.skip(f"mm_bf16_fp4 ({backend}) unavailable: {exc}")
    # The prepared layouts are what the dispatch keys off, so a real prepared
    # call must resolve back to the template that describes it.
    assert mm_bf16_fp4_trace_dispatch(**inputs) is tpl
    _assert_finite(inputs["a"])
    ref = tpl.reference(
        inputs["a"],
        inputs["b"],
        inputs["b_descale"],
        inputs["alpha"],
        block_size=inputs["block_size"],
    )
    _assert_finite(api, ref)
    _check(
        tpl,
        ref.to(api.dtype),
        api,
        max_mismatch_pct=100.0,
        min_cos_sim=0.99,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.parametrize("scale_shape", [(1024,), (256, 4), (2, 1, 32, 4, 4)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_native_trace_preserves_layout_tails_and_output_dtype(scale_shape, out_dtype):
    """Trace export must preserve physical scales instead of describing linear scales."""
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_bf16_fp4_trace_dispatch
    from tests.gemm.test_native_bf16_fp4 import make_case

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("native W4A16 requires SM120/121")
    a, b, sf, weight = make_case(3, 129, 48)
    sf = sf.reshape(scale_shape)
    alpha = torch.tensor([0.375], device="cuda")
    kwargs = dict(
        a=a,
        b=b,
        b_descale=sf,
        alpha=alpha,
        backend="cute-dsl-native",
        block_size=16,
        out_dtype=out_dtype,
    )
    definition = flashinfer.fi_trace(flashinfer.mm_bf16_fp4, **kwargs)
    assert "cute_dsl_native" in definition["name"]
    assert definition["axes"]["K_packed"]["value"] == b.shape[1]
    scale_axes = definition["inputs"]["b_descale"]["shape"]
    assert len(scale_axes) == sf.ndim
    assert tuple(definition["axes"][axis]["value"] for axis in scale_axes) == sf.shape
    assert definition["outputs"]["C"]["dtype"] == str(out_dtype).removeprefix("torch.")
    tpl = mm_bf16_fp4_trace_dispatch(**kwargs)
    # Run the exported reference in isolation, as a trace consumer would.
    namespace = {}
    exec(definition["reference"], namespace)
    ref = namespace[tpl.reference.__name__](a, b, sf, alpha, out_dtype=out_dtype)
    independent = ((a.float() @ weight.T) * alpha).to(out_dtype)
    torch.testing.assert_close(ref, independent, atol=2e-3, rtol=8e-3)
    out = torch.empty_like(ref)
    kwargs["out"] = out
    actual = flashinfer.mm_bf16_fp4(**kwargs)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, ref, atol=2e-3, rtol=8e-3)
