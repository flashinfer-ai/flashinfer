# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive E2M1 software conversion, including signed zero and scale range."""

import torch


def test_fp4_software_decoder_all_words():
    from tests.moe.test_frost_low_latency import _require_gpu

    _require_gpu(allow_sm120=True, allow_sm90=True)
    device = torch.device("cuda")
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.fused_moe.backends.frost_low_latency.fma.fp4_weight import (
        _software_fp4_quad,
        scaled_quad,
    )

    @cute.kernel
    def kernel(words, scales, bits, out, block: cutlass.Constexpr):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = bid * 256 + tid
        if row < cute.size(words):
            raw = _software_fp4_quad(words[row])
            values = scaled_quad(words[row], scales[row], False, block)
            for j in cutlass.range_constexpr(4):
                bits[row, j] = raw[j]
                out[row, j] = values[j].to(cutlass.Float32)

    @cute.jit
    def launch(words, scales, bits, out, block: cutlass.Constexpr, stream):
        kernel(words, scales, bits, out, block).launch(
            grid=((cute.size(words) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    words = torch.arange(65536, device=device, dtype=torch.int32)
    bits = torch.empty((65536, 4), device=device, dtype=torch.float16)
    out = torch.empty((65536, 4), device=device, dtype=torch.float32)
    codes = (words[:, None] >> torch.tensor([0, 4, 8, 12], device=device)) & 15
    lut = torch.tensor(
        [0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600],
        device=device,
        dtype=torch.int32,
    )
    expected_bits = (lut[codes & 7] | ((codes & 8) << 12)).to(torch.int16)
    expected = expected_bits.view(torch.float16).float()
    major, minor = torch.cuda.get_device_capability(device)
    for block, dtype, values in (
        (16, torch.float16, (0.0, 2.0**-9, 1.0, 448.0)),
        (32, torch.float32, (2.0**-120, 2.0**-17, 1.0, 2.0**19, 2.0**120)),
    ):
        scale = torch.ones(65536, device=device, dtype=dtype)
        compiled = cute.compile(
            launch,
            *(from_dlpack(v) for v in (words, scale, bits, out)),
            block,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
        )
        for value in values:
            scale.fill_(value)
            bits.fill_(float("nan"))
            out.fill_(float("nan"))
            compiled(words, scale, bits, out)
            torch.testing.assert_close(
                bits.view(torch.int16), expected_bits, atol=0, rtol=0
            )
            torch.testing.assert_close(
                out, expected * scale[:, None].float(), atol=0, rtol=0
            )
