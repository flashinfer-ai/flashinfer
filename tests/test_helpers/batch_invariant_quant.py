# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Shared inputs and comparisons of the batch-invariant quantizer tests (``tests/quantization/test_batch_invariant_*_quantize.py``)."""

import torch

from flashinfer import SfLayout

FI_LAYOUTS = {
    "128x4": SfLayout.layout_128x4,
    "8x4": SfLayout.layout_8x4,
    "linear": SfLayout.layout_linear,
}


def varied_inputs(M, K, dtype, seed=0):
    """Inputs that exercise a quantizer broadly rather than typically: magnitudes spread over about 40 binades (12 for
    f16) with random signs, 1 % of the elements scaled up as outliers, 1 % exact +0 and 1 % exact -0. Every value is
    finite and normal in binary32, and every block of 16 keeps an amax far above the smallest representable scale (no
    tiny-block corner)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    e = (torch.rand(M, K, generator=g) * 2 - 1) * (
        20.0 if dtype != torch.float16 else 6.0
    )
    x = (
        torch.where(torch.rand(M, K, generator=g) < 0.5, -1.0, 1.0)
        * (torch.rand(M, K, generator=g) + 1.0)
        * torch.exp2(e)
    )
    r = torch.rand(M, K, generator=g)
    x = torch.where(r < 0.01, x * (2.0**12 if dtype != torch.float16 else 2.0**4), x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = x.to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def adversarial_inputs(M, K, dtype, seed=0, special=True, block=16):
    """``varied_inputs`` plus the corners of the recipes: near-cancelling pairs, subnormal elements (bf16 / fp32) and, when
    the batch has at least 8 rows, an all-zero row, a row with one all-zero block, a block of -0, a row of tiny values
    whose scale rounds to the smallest scale byte, infinities, and the dtype's maximum. For the exact CPU oracle (which
    pins what the class computes there)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = varied_inputs(M, K, dtype, seed).float().cpu()
    pairs = torch.rand(M, K // 2, generator=g) < 0.15
    idx = torch.arange(0, K - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, K // 2, generator=g)),
        x[:, idx + 1],
    )
    r = torch.rand(M, K, generator=g)
    x = torch.where(
        (r >= 0.03) & (r < 0.035), x * 2.0**-130, x
    )  # subnormal in binary32 / bf16; zero in f16
    if special and M >= 8:
        x[1] = 0
        x[2, :block] = 0
        x[3, :block] = -0.0
        x[4] = (torch.rand(K, generator=g) * 2 - 1) * (
            2.0**-40 if block == 16 else 2.0**-125
        )
        x[5, 0] = float("inf")
        x[6, min(block + 1, K - 1)] = -float("inf")
        x[7] = torch.finfo(dtype).max
    x = x.to(dtype)
    assert not torch.isnan(x).any()
    return x.cuda()


def same_bits(a, b) -> bool:
    """Elementwise bit equality of two tuples of tensors (fp32 compared as int32)."""

    def view(t):
        return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.uint8)

    return all(
        p.shape == q.shape and torch.equal(view(p.contiguous()), view(q.contiguous()))
        for p, q in zip(a, b, strict=True)
    )
