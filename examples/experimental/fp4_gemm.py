"""Native packed FP4 GEMM on SM100a/SM103a; no quantization occurs in plan.run()."""

import torch
from flashinfer.fp4_gemm import prepare_fp4_gemm

m, n, k = 16, 4608, 5120
# E2M1 code 2 is +1; each byte packs two values. UE8M0 exponent127 is scale1.
a = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
sfa = torch.full((k // 128, m), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
sfb = torch.full((k // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
plan = prepare_fp4_gemm(a, b, sfa, sfb, m=m, alpha=0.5)
output = plan.run()
print(output.shape, output.dtype)
