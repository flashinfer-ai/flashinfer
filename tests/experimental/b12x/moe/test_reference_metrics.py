from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.reference import compare_to_reference
from b12x.testing.reference.w4a16_reference import compare_to_reference as compare_w4a16


def test_compare_to_reference_treats_matching_zero_rows_as_cosine_one() -> None:
    actual = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0]],
        dtype=torch.float32,
    )

    assert compare_to_reference(actual, reference).cos == 1.0
    assert compare_w4a16(actual, reference).cos == 1.0


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_normalized_rmse_is_relative_l2(scale) -> None:
    reference = torch.tensor([[1.0, -2.0], [3.0, -4.0]]) * scale
    actual = reference + torch.tensor([[0.01, -0.02], [0.0, 0.01]]) * scale
    metrics = compare_to_reference(actual, reference)
    relative_l2 = (actual - reference).norm() / reference.norm()
    assert metrics.normalized_rmse == pytest.approx(relative_l2.item())


def test_normalized_rmse_handles_zero_reference() -> None:
    reference = torch.zeros((2, 3))
    assert compare_to_reference(reference, reference).normalized_rmse == 0.0
    assert (
        compare_to_reference(torch.ones_like(reference), reference).normalized_rmse > 1
    )
