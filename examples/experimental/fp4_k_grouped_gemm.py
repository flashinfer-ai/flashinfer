"""Grouped packed FP4 GEMM with one empty group and caller-owned FP32 output."""

import torch
from flashinfer.fp4_k_grouped_gemm import prepare_fp4_k_grouped_gemm

m, n, group_ks = 256, 128, (257, 0, 511)
alignment = 768
padded_ks = [(k + alignment - 1) // alignment * alignment for k in group_ks]
k = sum(padded_ks)
a = torch.zeros((m, k // 2), dtype=torch.uint8, device="cuda")
b = torch.zeros((n, k // 2), dtype=torch.uint8, device="cuda")
# Encoded zeros with valid scale1. Real callers prepack quantized values once.
sfa = torch.full((k // 128, m), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
sfb = torch.full((k // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
out = torch.ones((3, m, n), dtype=torch.float32, device="cuda")
plan = prepare_fp4_k_grouped_gemm(
    a,
    b,
    sfa,
    sfb,
    m=m,
    group_ks=group_ks,
    k_alignment=alignment,
    use_psum_layout=False,
    output_dtype="fp32",
    accumulate=True,
    out=out,
)
plan.run()  # out += grouped products; the empty group's values remain exactly1.
torch.testing.assert_close(plan.output, torch.ones_like(plan.output), atol=0, rtol=0)
