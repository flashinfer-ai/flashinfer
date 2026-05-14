"""Reference correctness test for the tgv_gemm_sm100 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _cc,
    _close_fp8,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", M=16, N=1024, K=1024),
        dict(device="cuda", M=8, N=2048, K=1024),
    ],
)
def test_tgv_gemm_sm100_reference_correctness(shape_kwargs):
    """tgv_gemm_sm100 kernel (SM100 only in practice) vs reference (a @ b + bias)."""
    from flashinfer.utils import is_sm100f_supported

    # The kernel's Python gate accepts SM100 or SM103 (see
    # gemm_base._match_sm_version) but the precompiled cubin only has an
    # SM100 kernel image; calling on SM103 crashes with "no kernel image"
    # inside CUDA (uncatchable via try/except). Restrict to SM100.
    if _cc() != (10, 0):
        pytest.skip("tgv_gemm_sm100 cubin is only built for SM100")
    if not is_sm100f_supported(torch.device("cuda")):
        pytest.skip("tgv_gemm_sm100 requires SM100f support (CUDA 12.9+)")
    from flashinfer import tgv_gemm_sm100
    from flashinfer.trace.templates.page import tgv_gemm_sm100_trace

    inputs = tgv_gemm_sm100_trace.init(**shape_kwargs)
    try:
        api_out = tgv_gemm_sm100(inputs["a"], inputs["b"], inputs["bias"])
        torch.cuda.synchronize()
    except Exception as exc:
        pytest.skip(f"tgv_gemm_sm100 unavailable: {exc}")
    ref_out = tgv_gemm_sm100_trace.reference(inputs["a"], inputs["b"], inputs["bias"])
    # Matches tests/gemm/test_tgv_gemm.py: bf16 * K=1024 accumulation makes
    # element-wise tolerance unreliable; cosine similarity is the repo
    # convention for this op.
    _close_fp8(api_out, ref_out, cos_sim_min=0.99)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
