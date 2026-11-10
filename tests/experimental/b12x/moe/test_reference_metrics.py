from __future__ import annotations

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
