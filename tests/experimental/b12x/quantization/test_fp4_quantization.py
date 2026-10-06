from __future__ import annotations

import torch

from b12x._lib.intrinsics import fp4_quantize_values_torch


def test_fp4_quantize_values_tie_breaks_to_larger_magnitude() -> None:
    values = torch.tensor([-3.5, 3.5], dtype=torch.float32)
    expected = torch.tensor([-4.0, 4.0], dtype=torch.float32)

    torch.testing.assert_close(fp4_quantize_values_torch(values), expected, rtol=0, atol=0)
