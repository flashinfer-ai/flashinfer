# SPDX-License-Identifier: Apache-2.0
"""Order-insensitive MiniMax trace validation, including exported check code."""

import json
from pathlib import Path

import pytest
import torch

from flashinfer.trace.templates.minimax_m3 import minimax_m3_index_decode_trace


@pytest.fixture(params=["template", "exported"])
def check(request):
    """Exercise the live checker and its standalone serialized form."""
    if request.param == "template":
        return minimax_m3_index_decode_trace.check
    path = (
        Path(__file__).parent
        / "fi_trace_out"
        / ("minimax_m3_index_decode_h1_d128_ps128_bt128_kv1_k16.json")
    )
    namespace = {}
    exec(json.loads(path.read_text())["check"], namespace)
    return namespace["_minimax_m3_index_decode_check"]


@pytest.mark.parametrize("wrapper", ["tensor", "list", "tuple", "dict"])
def test_permuted_prefix_is_accepted(check, wrapper):
    """Equivalent selected multisets pass in every supported output container."""
    reference = torch.tensor([[[0, 2, 5, -1], [1, 3, -1, -1]]], dtype=torch.int32)
    actual = torch.tensor([[[5, 0, 2, -1], [3, 1, -1, -1]]], dtype=torch.int32)

    def wrap(tensor):
        """Use the selected trace-output representation."""
        if wrapper == "list":
            return [tensor]
        if wrapper == "tuple":
            return (tensor,)
        if wrapper == "dict":
            return {"indices": tensor}
        return tensor

    assert check(wrap(reference), wrap(actual))


@pytest.mark.parametrize(
    "actual",
    [
        [[[0, 2, 6, -1], [1, 3, -1, -1]]],  # wrong index
        [[[0, 2, 2, -1], [1, 3, -1, -1]]],  # duplicate replaces an index
        [[[0, -1, 2, 5], [1, 3, -1, -1]]],  # padding moved
        [[[1, 2, 5, -1], [0, 3, -1, -1]]],  # indices moved between rows
    ],
)
def test_invalid_selection_is_rejected(check, actual):
    """Matching global sets cannot hide wrong indices, multiplicities or padding."""
    reference = torch.tensor([[[0, 2, 5, -1], [1, 3, -1, -1]]], dtype=torch.int32)
    assert not check(reference, torch.tensor(actual, dtype=torch.int32))


def test_invalid_output_contract_is_rejected(check):
    """Reject shape, dtype and container mismatches rather than comparing nothing."""
    reference = torch.tensor([[[0, 2, -1, -1]]], dtype=torch.int32)
    for actual in (
        reference.squeeze(0),
        reference[..., :3],
        reference.float(),
        reference.long(),
        [],
        [reference, reference],
        {"other": reference},
        {"indices": reference, "extra": reference},
        None,
    ):
        assert not check(reference, actual)


def test_all_padding_is_accepted(check):
    """Zero-length rows with only padding remain valid."""
    reference = torch.full((1, 2, 16), -1, dtype=torch.int32)
    assert check(reference, reference.clone())
