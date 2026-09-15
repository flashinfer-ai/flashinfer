# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject contiguous but unaligned views before the vectorized native launch."""

import pytest
import torch

import flashinfer.fused_moe.cute_dsl.moe_utils as utils


@pytest.mark.parametrize("operand", ["input", "output"])
def test_tiled_unpermute_rejects_unaligned_storage_offset(monkeypatch, operand):
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors required")
    tokens, top_k, hidden = 2, 2, 4096
    inputs = torch.empty(tokens * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    target = inputs if operand == "input" else output
    backing = torch.empty(target.numel() + 1, device="cuda", dtype=target.dtype)
    unaligned = backing[1:].view_as(target)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 == 2
    if operand == "input":
        inputs = unaligned
    else:
        output = unaligned
    inverse = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
    scales = torch.ones(tokens, top_k, device="cuda", dtype=torch.float32)
    calls = []

    def launch(*args):
        calls.append(args)
        raise AssertionError("Unaligned pointers reached the native vectorized launch")

    monkeypatch.setattr(
        utils,
        "_get_moe_utils_module",
        lambda: {"flashinfer_moe_unpermute_bf16_float_scale_tiled": launch},
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        utils.moe_unpermute(
            inputs, output, inverse, scales, tokens, top_k, use_wide_tiling=True
        )
    assert calls == []


def test_tiled_unpermute_accepts_aligned_nonzero_storage_offset():
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors required")
    tokens, top_k, hidden = 2, 2, 4096
    storage = torch.ones(
        tokens * top_k * hidden + 8, device="cuda", dtype=torch.bfloat16
    )
    inputs = storage[8:].view(tokens * top_k, hidden)
    out_storage = torch.full(
        (tokens * hidden + 8,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    output = out_storage[8:].view(tokens, hidden)
    assert inputs.storage_offset() == output.storage_offset() == 8
    assert inputs.data_ptr() % 16 == output.data_ptr() % 16 == 0
    inverse = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
    scales = torch.tensor([[1.0, 2.0], [0.5, 1.5]], device="cuda")
    utils.moe_unpermute(
        inputs, output, inverse, scales, tokens, top_k, use_wide_tiling=True
    )
    expected = torch.tensor([3.0, 2.0], device="cuda", dtype=torch.bfloat16)
    assert torch.equal(output, expected[:, None].expand_as(output))
    assert torch.isnan(out_storage[:8]).all()
