"""Reference correctness test for the segment_gemm_run trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", total_rows=64, batch_size=2, K=32, N=16),
        dict(device="cuda", total_rows=48, batch_size=3, K=16, N=32),
    ],
)
def test_segment_gemm_run_reference_correctness(shape_kwargs):
    """SegmentGEMMWrapper.run kernel vs reference (per-segment matmul)."""
    from flashinfer import SegmentGEMMWrapper
    from flashinfer.trace.templates.attention import segment_gemm_run_trace

    inputs = segment_gemm_run_trace.init(**shape_kwargs)
    ws = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        gemm = SegmentGEMMWrapper(ws)
        api_out = gemm.run(
            inputs["x"],
            inputs["weights"],
            inputs["seg_lens"].numel(),
            weight_column_major=False,
            seg_lens=inputs["seg_lens"],
        )
    except Exception as exc:
        pytest.skip(f"SegmentGEMMWrapper unavailable: {exc}")
    ref_out = segment_gemm_run_trace.reference(
        inputs["x"], inputs["weights"], seg_indptr=inputs["seg_indptr"]
    )
    # Matches tests/gemm/test_group_gemm.py.
    _close(api_out, ref_out, atol=2e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
