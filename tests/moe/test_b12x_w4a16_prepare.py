"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _reference_scale_factor(scales: torch.Tensor, a_dtype: torch.dtype) -> float:
    if a_dtype == torch.float16:
        return 1.0
    ws_float = scales.float() * (2**7)
    positive = ws_float[ws_float > 0]
    if positive.numel() == 0:
        return 1.0
    max_scalar = float(positive.max().item())
    if max_scalar < 448 * (2**7):
        return float(2 ** math.floor(math.log2((448 * (2**7)) / max_scalar)))
    return 1.0


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.0], 1.0),
        ([1.0], 256.0),
        ([0.0, 2.0, 224.0], 2.0),
        ([448.0], 1.0),
    ],
)
def test_nvfp4_compute_scale_factor_boundaries(values, expected):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        _nvfp4_compute_scale_factor,
    )

    scales = torch.tensor(values, device="cuda").to(torch.float8_e4m3fn)

    assert _nvfp4_compute_scale_factor(scales, torch.bfloat16) == expected


def test_nvfp4_compute_scale_factor_matches_reference_across_experts():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        _nvfp4_compute_scale_factor,
    )

    values = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5],
            [2.0, 3.0, 4.0, 6.0],
            [8.0, 12.0, 16.0, 24.0],
        ],
        device="cuda",
    ).to(torch.float8_e4m3fn)

    assert _nvfp4_compute_scale_factor(
        values, torch.bfloat16
    ) == _reference_scale_factor(values, torch.bfloat16)

    bf16_values = values.float().to(torch.bfloat16)
    assert _nvfp4_compute_scale_factor(
        bf16_values, torch.bfloat16
    ) == _reference_scale_factor(bf16_values, torch.bfloat16)


def test_nvfp4_compute_scale_factor_empty_and_fp16_bypass():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        _nvfp4_compute_scale_factor,
    )

    empty = torch.empty((0, 4), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.full((2, 4), 448.0, dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )

    assert _nvfp4_compute_scale_factor(empty, torch.bfloat16) == 1.0
    assert _nvfp4_compute_scale_factor(scales, torch.float16) == 1.0
