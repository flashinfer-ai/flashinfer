# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact INT4 conversion and metadata layout tests on Hopper."""

import pytest


def _require_hopper():
    import torch
    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("requires SM90 and CUDA 12.3+")


@pytest.mark.arch_hopper
def test_int4_conversion_all_positive_bf16_scales():
    """Cover all scale bits and nibble slots, plus all packed halfwords."""
    _require_hopper()
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.int4_conversion import (
        convert_int4x8_to_fp8x8,
    )

    @cute.kernel
    def convert(src: cute.Tensor, metadata: cute.Tensor, dst: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 256 + thread
        if index < cute.size(src):
            lo, hi = convert_int4x8_to_fp8x8(
                cutlass.Int32(src[index]), metadata[index, 0], metadata[index, 1]
            )
            dst[index, 0] = lo
            dst[index, 1] = hi

    @cute.jit
    def launch(src: cute.Tensor, metadata: cute.Tensor, dst: cute.Tensor):
        convert(src, metadata, dst).launch(
            grid=((cute.size(src) + 255) // 256, 1, 1), block=(256, 1, 1)
        )

    # Include zero to verify signed underflow; live input scales are positive.
    scale_codes = torch.arange(0x7F80, dtype=torch.int32, device="cuda")
    bits = scale_codes.repeat_interleave(16)
    seeds = torch.arange(16, dtype=torch.int32, device="cuda").repeat(
        scale_codes.numel()
    )
    codes = (seeds[:, None] + torch.arange(8, device="cuda")) & 15
    scales = torch.stack((bits, bits.flip(0)), dim=1)
    halfwords = torch.arange(65536, dtype=torch.int32, device="cuda")
    low_codes = (halfwords[:, None] >> (torch.arange(4, device="cuda") * 4)) & 15
    codes = torch.cat((codes, torch.cat((low_codes, 15 - low_codes.flip(1)), dim=1)))
    scales = torch.cat(
        (
            scales,
            torch.stack((halfwords % 0x7F80, (halfwords * 113 + 7) % 0x7F80), dim=1),
        )
    )
    sf = scales.to(torch.int16).view(torch.bfloat16)
    # Construct tables independently of the production weight-preparation code.
    metadata = (
        (-torch.arange(1, 9, device="cuda").bfloat16() * sf[:, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint64)
        .reshape(-1, 2)
    )
    code_map = torch.tensor(
        [7, 0, 1, 2, 3, 4, 5, 6, 15, 14, 13, 12, 11, 10, 9, 8],
        dtype=torch.int64,
        device="cuda",
    )
    shifts = torch.arange(8, dtype=torch.int64, device="cuda") * 4
    packed = (code_map[codes.long()] << shifts).sum(dim=1).to(torch.int32)
    output = torch.empty((packed.numel(), 2), dtype=torch.int32, device="cuda")
    arguments = tuple(from_dlpack(t) for t in (packed, metadata, output))
    cute.compile(launch, *arguments)(*arguments)
    q = torch.where(codes >= 8, codes - 16, codes).to(torch.bfloat16)
    expected = (
        (q * sf.repeat_interleave(4, dim=1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    actual = output.view(torch.uint8)
    mismatch = actual != expected
    if bool(mismatch.any()):
        samples = [
            (
                int(q[row, col]),
                hex(int(scales[row, col // 4])),
                hex(int(expected[row, col])),
                hex(int(actual[row, col])),
            )
            for row, col in mismatch.nonzero()[:16].tolist()
        ]
        pytest.fail(
            f"INT4 mismatches: {int(mismatch.sum())}; (q, scale, expected, actual): {samples}"
        )


@pytest.mark.arch_hopper
@pytest.mark.parametrize("tile_m,tile_k", [(128, 128), (256, 128), (256, 256)])
def test_int4_scale_layout_coordinates(tile_m, tile_k):
    _require_hopper()
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.int4_conversion import (
        make_expanded_offset_view,
        make_expanded_offset_view_k256,
        make_offset_smem_layout,
        make_offset_smem_layout_k256,
    )

    make = make_offset_smem_layout if tile_k == 128 else make_offset_smem_layout_k256
    expand = (
        make_expanded_offset_view if tile_k == 128 else make_expanded_offset_view_k256
    )
    k_blocks = tile_k // 128

    @cute.kernel
    def read_coordinates(raw: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 256 + thread
        staged = cute.make_tensor(raw.iterator, make(tile_m, 3))
        expanded = expand(staged, tile_m)
        if index < tile_m * tile_k * 3:
            m = index // (tile_k * 3)
            k = (index // 3) % tile_k
            stage = index % 3
            output[m, k, stage] = expanded[m, k, stage]

    @cute.jit
    def launch(raw: cute.Tensor, output: cute.Tensor):
        read_coordinates(raw, output).launch(
            grid=((tile_m * tile_k * 3 + 255) // 256, 1, 1), block=(256, 1, 1)
        )

    raw = torch.arange(tile_m * k_blocks * 3, dtype=torch.int32, device="cuda")
    actual = torch.empty((tile_m, tile_k, 3), dtype=torch.int32, device="cuda")
    arguments = tuple(from_dlpack(t) for t in (raw, actual))
    cute.compile(launch, *arguments)(*arguments)
    m = torch.arange(tile_m, device="cuda")[:, None, None]
    k = torch.arange(tile_k, device="cuda")[None, :, None]
    stage = torch.arange(3, device="cuda")[None, None, :]
    expected = (
        stage * (tile_m * k_blocks)
        + (m // 64) * (64 * k_blocks)
        + (k // 128) * 64
        + m % 64
    )
    torch.testing.assert_close(actual, expected.to(torch.int32), rtol=0, atol=0)
