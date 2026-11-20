# SPDX-License-Identifier: Apache-2.0
"""Numerics of the SM100/SM103 GC MoE path on B300-class devices."""

import pytest
import torch

import flashinfer

from .reference import GEOMETRY, cosine, relative_l2, run_public_case


@pytest.fixture(autouse=True)
def sm100():
    if not torch.cuda.is_available() or tuple(
        torch.cuda.get_device_capability()
    ) not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_matches_reference(gc_group, num_tokens):
    y, expected = run_public_case("sm100", gc_group, num_tokens=num_tokens)
    assert y.dtype == torch.bfloat16
    assert y.shape == expected.shape
    assert cosine(y, expected) > 0.97
    assert relative_l2(y, expected) < 0.05


def test_base_shares_the_allocation(gc_group):
    geometry = dict(
        num_experts=GEOMETRY["num_experts"],
        num_max_tokens_per_rank=GEOMETRY["capacity"],
        num_topk=GEOMETRY["num_topk"],
        hidden=GEOMETRY["hidden"],
        intermediate_hidden=GEOMETRY["intermediate_hidden"],
    )
    with (
        flashinfer.GCMoEWorkspace(gc_group, **geometry) as first,
        flashinfer.GCMoEWorkspace(gc_group, base=first, **geometry) as second,
    ):
        assert second.x.data_ptr() == first.x.data_ptr()
        assert second.topk_idx.data_ptr() == first.topk_idx.data_ptr()
