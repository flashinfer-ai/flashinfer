# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer selected-channel KEXT followed by upstream FlashInfer FP4 GEMM.

Run: python examples/resinfer/kext.py
Weight expansion and quantization are offline preparation. This example is
a correctness smoke, not a performance or model accuracy benchmark.
"""

import torch

from flashinfer import mm_fp4
from flashinfer.quantization import nvfp4_quantize
from flashinfer.resinfer import nvfp4_quantize_kext


def main():
    torch.manual_seed(42)
    m, n, k = 32, 128, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Example: select channels 0 and 7 in every group of eight. Real models
    # supply calibrated masks; ascending bit order also fixes weight order.
    masks = torch.full((k // 8,), 0x81, device="cuda", dtype=torch.uint8)
    selected = (
        torch.arange(k // 8, device="cuda")[:, None] * 8
        + torch.tensor([0, 7], device="cuda")
    ).flatten()
    weight_extended = torch.cat((w, w[:, selected]), dim=1)
    a_scale = torch.tensor([64.0], device="cuda")
    w_scale = torch.tensor([32.0], device="cuda")
    wp, ws = nvfp4_quantize(weight_extended, w_scale, backend="cute-dsl")
    ap, ass = nvfp4_quantize_kext(x, a_scale, masks, residue_per_8=2)
    y = mm_fp4(ap, wp.T, ass, ws, (a_scale * w_scale).reciprocal(), backend="cute-dsl")
    assert y.shape == (m, n) and torch.isfinite(y).all()
    print(f"KEXT output: {tuple(y.shape)}, {y.dtype}; upstream CuTe-DSL GEMM")


if __name__ == "__main__":
    main()
