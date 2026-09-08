"""A cuDNN GEMM tactic that cannot be used must be reported as FAILED during
autotuner profiling (so the autotuner disqualifies it), while still falling
back to tactic=-1 outside profiling for serving robustness.

Regression for the silent-fallback-during-profiling issue raised in the
#3707 review: otherwise the fallback's timing is attributed to the requested
tactic and a tactic that never ran can win the autotune.
"""

import warnings

import pytest
import torch

from flashinfer.autotuner.autotuner import _profile_measurement_scope
from flashinfer.utils import get_compute_capability


def _fp8_inputs(m=8, n=4096, k=4096):
    from tests.utils_fp8 import to_float8

    a = torch.randn([1, m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([1, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    a8, a_scale = to_float8(a)
    b8, b_scale = to_float8(b)
    out = torch.empty([1, m, n], device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    return [a8, b8, a_scale, b_scale, out, ws]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cudnn_fp8_tactic_failure_raises_in_profiling_but_falls_back_serving():
    if get_compute_capability(torch.device("cuda"))[0] < 9:
        pytest.skip("cuDNN FP8 GEMM requires SM90+")
    from flashinfer.gemm.gemm_base import (
        _check_cudnn_availability,
        _cudnn_gemm_fp8_runner,
    )

    try:
        _check_cudnn_availability()
    except Exception:
        pytest.skip("cuDNN not available")

    runner = _cudnn_gemm_fp8_runner()
    inputs = _fp8_inputs()
    bad_tactic = 9999  # out-of-range plan index -> deterministic fallback path

    # Serving (not profiling): warn + fall back to -1 + produce a valid result.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        runner.forward(inputs, tactic=bad_tactic)
        assert any("falling back to default tactic=-1" in str(x.message) for x in w)
    torch.cuda.synchronize()
    assert torch.isfinite(inputs[4]).all()

    # Profiling: the failed tactic must raise so the autotuner marks it inf,
    # rather than silently timing the fallback under the requested tactic.
    with (
        pytest.raises(RuntimeError, match="during profiling"),
        _profile_measurement_scope(),
    ):
        runner.forward(inputs, tactic=bad_tactic)
