# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU contracts for NVFP4 weight decoding."""

import torch

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    nvfp4_scale_from_blocked,
    nvfp4_weight_from_swizzled_to_bf16,
    nvfp4_weight_to_bf16,
    swiglu_fold_interleave,
    to_blocked,
)


def test_nvfp4_k16_block_scale_decode():
    packed = torch.tensor(
        [[0x20, 0x42, 0x64, 0x86, 0xA8, 0xCA, 0xEC, 0x0E]],
        dtype=torch.uint8,
    ).view(torch.float4_e2m1fn_x2)
    block_scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    decoded = nvfp4_weight_to_bf16(packed, block_scale)

    expected = torch.tensor(
        [
            [
                0.0,
                2.0,
                2.0,
                4.0,
                4.0,
                8.0,
                8.0,
                -0.0,
                -0.0,
                -2.0,
                -2.0,
                -4.0,
                -4.0,
                -8.0,
                -8.0,
                0.0,
            ]
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(decoded, expected)


def test_atom_swizzle_round_trip_with_row_and_column_padding():
    rows, cols = 130, 5
    values = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=torch.float32)
    raw = values[torch.arange(rows * cols) % values.numel()].reshape(rows, cols)
    raw = raw.to(torch.float8_e4m3fn)

    swizzled = to_blocked(raw)
    restored = nvfp4_scale_from_blocked(swizzled, rows, cols)

    # Both axes require padding: 130 -> 256 rows and 5 -> 8 columns.
    assert swizzled.numel() == 256 * 8
    assert torch.equal(restored.view(torch.uint8), raw.view(torch.uint8))


def test_k_major_decode_multiple_rows_blocks_and_padded_scale_plane():
    output_features, reduction = 130, 80
    packed_u8 = (
        torch.arange(output_features * (reduction // 2), dtype=torch.int64)
        .mul(37)
        .add(11)
        .remainder(256)
        .to(torch.uint8)
        .reshape(output_features, reduction // 2)
    )
    logical_weight = packed_u8.view(torch.float4_e2m1fn_x2)
    # Runner ABI: physical host shape is (packed-K, N), with packed K stride 1.
    weight_kn = logical_weight.transpose(0, 1)
    assert weight_kn.stride(0) == 1

    scale_values = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=torch.float32)
    raw_scale = scale_values[
        torch.arange(output_features * (reduction // 16)) % scale_values.numel()
    ].reshape(output_features, reduction // 16)
    raw_scale = raw_scale.to(torch.float8_e4m3fn)

    expected = nvfp4_weight_to_bf16(logical_weight, raw_scale)
    actual = nvfp4_weight_from_swizzled_to_bf16(
        weight_kn,
        to_blocked(raw_scale),
    )

    assert torch.equal(actual, expected)


def test_gate_up_fold_uses_16_column_interleave():
    # Four distinguishable 16-column groups represent
    # [gate0, up0, gate1, up1], not a single gate-half/up-half split.
    gate0 = torch.full((1, 16), 0.25)
    up0 = torch.full((1, 16), 2.0)
    gate1 = torch.full((1, 16), -0.5)
    up1 = torch.full((1, 16), 3.0)
    interleaved = torch.cat((gate0, up0, gate1, up1), dim=1)

    actual = swiglu_fold_interleave(interleaved, gate_up_interleave=16)
    expected = torch.cat(
        (
            up0 * (gate0 * torch.sigmoid(gate0)),
            up1 * (gate1 * torch.sigmoid(gate1)),
        ),
        dim=1,
    )

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
