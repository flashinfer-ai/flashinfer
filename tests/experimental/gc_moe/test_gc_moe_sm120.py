# SPDX-License-Identifier: Apache-2.0
"""Numerics of the SM120 GC MoE path on Blackwell consumer devices."""

import pytest
import torch

from .reference import cosine, run_public_case


@pytest.fixture(autouse=True)
def sm120():
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (
        12,
        0,
    ):
        pytest.skip("requires SM120")


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_matches_reference(gc_group, num_tokens):
    y, expected = run_public_case("sm120", gc_group, num_tokens=num_tokens)
    assert y.dtype == torch.bfloat16
    assert y.shape == expected.shape
    assert cosine(y, expected) > 0.97
