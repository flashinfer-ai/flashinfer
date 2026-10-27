"""MoE benchmark accuracy gates preserve direction and relative magnitude."""

import pytest
import torch

from benchmarks.experimental.b12x.benchmark_moe import check_oracle_metrics
from b12x.moe._shared.kernels.reference import compare_to_reference


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_nvfp4_gate_is_scale_invariant(scale):
    reference = torch.tensor([[1.0, -1.0, 2.0, -2.0]]) * scale
    actual = reference + torch.tensor([[0.005, -0.005, -0.005, 0.005]]) * scale
    metrics = compare_to_reference(actual, reference)
    assert not check_oracle_metrics("test", metrics, 1)
    if scale == 100.0:
        assert metrics.max_abs > 0.05


def test_nvfp4_gate_rejects_magnitude_error_despite_high_cosine():
    reference = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    metrics = compare_to_reference(reference * 1.02, reference)
    assert metrics.cos > 0.9999
    failures = check_oracle_metrics("test", metrics, 1)
    assert len(failures) == 1 and "normalized_rmse=" in failures[0]


def test_nvfp4_gate_retains_cosine_threshold():
    reference = torch.tensor([[1.0, 0.0]])
    metrics = compare_to_reference(torch.tensor([[1.0, 0.0145]]), reference)
    assert metrics.normalized_rmse < 0.015
    failures = check_oracle_metrics("test", metrics, 1)
    assert len(failures) == 1 and "cos=" in failures[0]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nvfp4_gate_rejects_nonfinite_output(invalid):
    metrics = compare_to_reference(torch.tensor([[1.0, invalid]]), torch.ones((1, 2)))
    failures = check_oracle_metrics("test", metrics, 1)
    assert len(failures) == 1 and "non-finite metrics" in failures[0]
