import pytest
import torch

from benchmarks.routines import flashinfer_benchmark_utils as bench
from flashinfer.gemm import mm_mxfp8


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [(9, 0), (10, 7)])
def test_mm_mxfp8_benchmark_matches_cudnn_support(monkeypatch, capability):
    monkeypatch.setattr(bench, "get_compute_capability", lambda device: capability)
    major, minor = capability
    supported = mm_mxfp8.is_backend_supported("cudnn", major * 10 + minor)
    filtered = bench.filter_backends_by_compute_capability(
        ["cudnn"], "mm_mxfp8", torch.device("cuda")
    )
    assert filtered == (["cudnn"] if supported else [])
