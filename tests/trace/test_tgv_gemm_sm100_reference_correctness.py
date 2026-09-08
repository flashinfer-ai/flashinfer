"""Reference correctness test for the tgv_gemm_sm100 trace API."""

import torch
import pytest

from tests.trace.reference_utils import _check


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", M=16, N=1024, K=1024),
        dict(device="cuda", M=8, N=2048, K=1024),
    ],
)
def test_tgv_gemm_sm100_reference_correctness(shape_kwargs):
    """TGV SM100-family kernel vs the a @ b + bias reference."""
    from flashinfer.utils import get_compute_capability, is_sm100f_supported

    if get_compute_capability(torch.device("cuda")) not in [(10, 0), (10, 3)]:
        pytest.skip("tgv_gemm_sm100 requires SM100 or SM103")
    if not is_sm100f_supported(torch.device("cuda")):
        pytest.skip("tgv_gemm_sm100 requires SM100f support (CUDA 12.9+)")
    from flashinfer import tgv_gemm_sm100
    from flashinfer.trace.templates.page import tgv_gemm_sm100_trace

    inputs = tgv_gemm_sm100_trace.init(**shape_kwargs)
    assert inputs["b"].shape == (shape_kwargs["K"], shape_kwargs["N"])
    assert inputs["b"].stride(0) == 1, "tgv_gemm_sm100 expects column-major b"
    api_out = tgv_gemm_sm100(inputs["a"], inputs["b"], inputs["bias"])
    torch.cuda.synchronize()
    ref_out = tgv_gemm_sm100_trace.reference(inputs["a"], inputs["b"], inputs["bias"])
    # Matches tests/gemm/test_tgv_gemm.py: bf16 * K=1024 accumulation makes
    # element-wise tolerance unreliable; cosine similarity is the repo
    # convention for this op.
    _check(
        tgv_gemm_sm100_trace, ref_out, api_out, max_mismatch_pct=100.0, min_cos_sim=0.99
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
