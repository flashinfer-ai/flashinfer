# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extended native finalizer ranges, rejection before launch, and captured views."""

import pytest
import torch

import flashinfer.fused_moe.cute_dsl.moe_utils as utils


def buffers(tokens, hidden=4096, top_k=8):
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors required")
    inputs = torch.ones(tokens * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.full(
        (tokens, hidden), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    inverse = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
    scales = torch.ones(tokens, top_k, device="cuda", dtype=torch.float32)
    return inputs, output, inverse, scales


@pytest.mark.parametrize("tokens", [1025, 3072, 8192])
@pytest.mark.parametrize("expanded", [False, True])
def test_extended_tiled_wrapper_launch_contract(monkeypatch, tokens, expanded):
    inputs, output, inverse, scales = buffers(tokens)
    calls = []
    monkeypatch.setattr(
        utils,
        "_get_moe_utils_module",
        lambda: {
            "flashinfer_moe_unpermute_bf16_float_scale_tiled": lambda *args: calls.append(
                args
            )
        },
    )
    utils.moe_unpermute(
        inputs,
        output,
        inverse,
        scales,
        tokens,
        8,
        input_is_expanded=expanded,
        round_scales_to_bf16=True,
        use_wide_tiling=True,
    )
    assert len(calls) == 1
    assert calls[0][:9] == (
        inputs.data_ptr(),
        output.data_ptr(),
        inverse.data_ptr(),
        scales.data_ptr(),
        tokens,
        4096,
        8,
        expanded,
        True,
    )
    assert calls[0][9] == torch.cuda.current_stream().cuda_stream


@pytest.mark.parametrize(
    "tokens,hidden,top_k",
    [
        (0, 4096, 8),
        (8193, 4096, 8),
        (1025, 8192, 8),
        (1025, 4096, 2),
        (1025, 4096, 4),
        (1025, 4096, 16),
        (1025, 2048, 8),
    ],
)
def test_extended_tiled_wrapper_rejects_before_launch(
    monkeypatch, tokens, hidden, top_k
):
    inputs, output, inverse, scales = buffers(tokens, hidden, top_k)
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError(
            "Unsupported token/hidden/top-k combination reached a native launch"
        )

    monkeypatch.setattr(
        utils,
        "_get_moe_utils_module",
        lambda: {"flashinfer_moe_unpermute_bf16_float_scale_tiled": forbidden},
    )
    with pytest.raises(ValueError, match="Wide tiling requires"):
        utils.moe_unpermute(
            inputs, output, inverse, scales, tokens, top_k, use_wide_tiling=True
        )
    assert not calls
    assert torch.isnan(output).all()


@pytest.mark.parametrize("operand", ["input", "output"])
def test_extended_tiled_alignment_guard(monkeypatch, operand):
    inputs, output, inverse, scales = buffers(1025)
    target = inputs if operand == "input" else output
    backing = torch.empty(target.numel() + 1, device="cuda", dtype=target.dtype)
    unaligned = backing[1:].view_as(target)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 == 2
    if operand == "input":
        inputs = unaligned
    else:
        output = unaligned
    calls = []
    monkeypatch.setattr(
        utils,
        "_get_moe_utils_module",
        lambda: {
            "flashinfer_moe_unpermute_bf16_float_scale_tiled": lambda *args: calls.append(
                args
            )
        },
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        utils.moe_unpermute(
            inputs, output, inverse, scales, 1025, 8, use_wide_tiling=True
        )
    assert not calls


@pytest.mark.parametrize(
    "tokens,hidden,top_k",
    [(0, 4096, 8), (8193, 4096, 8), (1025, 8192, 8), (1025, 4096, 4)],
)
def test_extended_tiled_native_entry_rejects_range(tokens, hidden, top_k):
    inputs, output, inverse, scales = buffers(tokens, hidden, top_k)
    native = utils._get_moe_utils_module()[
        "flashinfer_moe_unpermute_bf16_float_scale_tiled"
    ]
    with pytest.raises(Exception, match="Tiled MoE supports 1..1024"):
        native(
            inputs.data_ptr(),
            output.data_ptr(),
            inverse.data_ptr(),
            scales.data_ptr(),
            tokens,
            hidden,
            top_k,
            False,
            True,
            torch.cuda.current_stream().cuda_stream,
        )
    torch.cuda.synchronize()
    assert torch.isnan(output).all()


@pytest.mark.parametrize("tokens", [1025, 3072, 8192])
def test_extended_aligned_offset_capture(tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors required")
    storage = torch.ones(tokens * 8 * 4096 + 8, device="cuda", dtype=torch.bfloat16)
    inputs = storage[8:].view(tokens * 8, 4096)
    out_storage = torch.full(
        (tokens * 4096 + 8,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    output = out_storage[8:].view(tokens, 4096)
    assert inputs.storage_offset() == output.storage_offset() == 8
    assert inputs.data_ptr() % 16 == output.data_ptr() % 16 == 0
    inverse = torch.arange(tokens * 8, device="cuda", dtype=torch.int32)
    inverse[::13] = -1
    scales = (
        torch.arange(1, 9, device="cuda", dtype=torch.float32)[None, :]
        .expand(tokens, -1)
        .contiguous()
        / 16
    )

    def reference(sign):
        return (
            (sign * torch.where(inverse.view(tokens, 8) >= 0, scales, 0.0).sum(dim=1))
            .to(torch.bfloat16)[:, None]
            .expand_as(output)
        )

    def run():
        utils.moe_unpermute(
            inputs,
            output,
            inverse,
            scales,
            tokens,
            8,
            use_wide_tiling=True,
            round_scales_to_bf16=True,
        )

    run()
    expected = reference(1)
    assert torch.equal(output, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs.neg_()
    inverse[1] = -1
    scales.mul_(2)
    changed = reference(-1)
    assert not torch.equal(expected, changed)
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output, changed)
    assert torch.isnan(out_storage[:8]).all()
