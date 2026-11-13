# SPDX-License-Identifier: Apache-2.0
"""Numerics of the SM90 GC MoE path on Hopper devices."""

import pytest
import torch

from .reference import cosine, run_public_case


@pytest.fixture(autouse=True)
def sm90():
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (
        9,
        0,
    ):
        pytest.skip("requires SM90")


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_matches_reference(gc_group, num_tokens):
    y, expected = run_public_case("sm90", gc_group, num_tokens=num_tokens)
    assert y.dtype == torch.bfloat16
    assert y.shape == expected.shape
    assert cosine(y, expected) > 0.97
