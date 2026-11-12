"""Reference correctness test for the log_softmax trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


def _class_runs_here() -> bool:
    if not torch.cuda.is_available():
        return False
    import flashinfer.batch_invariant as bi
    from flashinfer.batch_invariant.sampling.api import LSE_ROW_ARCHS

    return bi.is_supported_device(torch.device("cuda"), LSE_ROW_ARCHS)


pytestmark = pytest.mark.skipif(
    not _class_runs_here(),
    reason="the batch-invariant sampling class runs on SM100 / SM103 / SM107 only",
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=8, vocab_size=128), dict(batch_size=3, vocab_size=257)],
)
def test_log_softmax_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import log_softmax_trace

    inputs = log_softmax_trace.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.log_softmax(
        inputs["logits"], temperature=inputs["temperature"]
    )
    ref_out = log_softmax_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _check(log_softmax_trace, ref_out, api_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
