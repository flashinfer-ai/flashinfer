"""Prepared mixed FP8×FP4 GEMM on prepacked user tensors; minimum SM103a."""

import torch
from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm

m, n, k = 16, 4608, 5120
a = torch.full((m, k), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
sfa = torch.full((k // 128, m), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
sfb = torch.full((k // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
plan = prepare_fp8_fp4_gemm(a, b, sfa, sfb, m=m)
print(plan.run().shape)
