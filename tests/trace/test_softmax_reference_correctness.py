"""Reference correctness test for the softmax trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=8, vocab_size=128), dict(batch_size=3, vocab_size=257)],
)
def test_softmax_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import softmax_trace

    inputs = softmax_trace.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.softmax(inputs["logits"], temperature=inputs["temperature"])
    ref_out = softmax_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _check(softmax_trace, ref_out, api_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _class_runs_here() -> bool:
    if not torch.cuda.is_available():
        return False
    import flashinfer.batch_invariant as bi
    from flashinfer.batch_invariant.sampling.api import LSE_ROW_ARCHS

    return bi.is_supported_device(torch.device("cuda"), LSE_ROW_ARCHS)


@pytest.mark.skipif(
    not _class_runs_here(),
    reason="the batch-invariant sampling class runs on SM100 / SM103 / SM107 only",
)
def test_softmax_f32_out_reference_correctness():
    """``softmax(batch_invariant=True, out_dtype=torch.float32)`` traces to the fp32-output template."""
    import flashinfer
    from flashinfer.trace.templates.sampling import (
        softmax_f32_trace,
        softmax_trace_dispatch,
    )

    inputs = softmax_f32_trace.init(batch_size=4, vocab_size=257)
    assert softmax_trace_dispatch(**inputs) is softmax_f32_trace
    assert softmax_trace_dispatch(logits=inputs["logits"]) is not softmax_f32_trace
    api_out = flashinfer.softmax(**inputs)
    assert api_out.dtype == torch.float32
    ref_out = softmax_f32_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _check(softmax_f32_trace, ref_out, api_out, atol=1e-3, rtol=1e-3)
