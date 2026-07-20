from __future__ import annotations

from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import _should_use_exact_b16_wo


def test_exact_b16_wo_is_spark_only() -> None:
    assert _should_use_exact_b16_wo(tokens=16, sm_count=20)
    assert not _should_use_exact_b16_wo(tokens=16, sm_count=188)
    assert not _should_use_exact_b16_wo(tokens=8, sm_count=20)
