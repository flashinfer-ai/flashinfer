# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared FP8 weights retain their precision and logical gate/up ordering."""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_weights import (
    preprocess_shared_weights,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.kernel_helpers import (
    from_blocked,
)


def _weights(offset=0):
    shapes = ((256, 128), (128, 128), (256, 4), (128, 4))
    tensors = []
    for index, shape in enumerate(shapes):
        raw = torch.arange(shape[0] * shape[1] + offset, dtype=torch.int32)
        raw = (raw % (119 if index < 2 else 11) + (0 if index < 2 else 121)).to(
            torch.uint8
        )
        tensors.append(
            raw[offset:]
            .reshape(shape)
            .view(torch.float8_e4m3fn if index < 2 else torch.float8_e8m0fnu)
        )
    return tensors


@pytest.mark.parametrize("offset", [0, 1])
def test_shared_weight_layout_preserves_bytes_and_callers(offset):
    weights = _weights(offset)
    original = [tensor.view(torch.uint8).clone() for tensor in weights]
    transformed = preprocess_shared_weights(
        *weights, hidden_size=128, intermediate_size=128
    )
    # The FC1 accumulator pairs 16 gate rows with the matching 16 up rows.
    logical_rows = torch.tensor(
        [
            row
            for start in range(0, 128, 16)
            for part in (0, 128)
            for row in range(start + part, start + part + 16)
        ]
    )
    (fc1, sf1), (fc2, sf2) = transformed
    assert fc1.dtype == fc2.dtype == torch.float8_e4m3fn
    assert sf1.dtype == sf2.dtype == torch.float8_e8m0fnu
    assert tuple(fc1.shape) == (1, 128, 256)
    assert tuple(fc2.shape) == (1, 128, 128)
    assert fc1.data_ptr() % 16 == fc2.data_ptr() % 16 == 0
    assert torch.equal(fc1[0].T.view(torch.uint8), original[0][logical_rows])
    assert torch.equal(fc2[0].T.view(torch.uint8), original[1])
    assert torch.equal(
        from_blocked(sf1[0], 256, 4).view(torch.uint8), original[2][logical_rows]
    )
    assert torch.equal(from_blocked(sf2[0], 128, 4).view(torch.uint8), original[3])
    for tensor, before in zip(weights, original, strict=True):
        assert torch.equal(tensor.view(torch.uint8), before)


@pytest.mark.parametrize("invalid", ["expert_axis", "sharded", "fp4", "scale_dtype"])
def test_shared_weight_preparation_rejects_incompatible_contract(invalid):
    weights = _weights()
    if invalid == "expert_axis":
        weights = [tensor.unsqueeze(0) for tensor in weights]
    elif invalid == "sharded":
        weights[0] = weights[0][:128]
    elif invalid == "fp4":
        weights[0] = weights[0].view(torch.uint8)
    else:
        weights[2] = weights[2].float()
    with pytest.raises(ValueError):
        preprocess_shared_weights(*weights, hidden_size=128, intermediate_size=128)
