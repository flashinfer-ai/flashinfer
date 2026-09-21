"""Prepare batched projections and submit on the current PyTorch stream."""
import torch
from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm

values_a = torch.ones((16, 8, 4096), dtype=torch.float8_e4m3fn, device='cuda')
values_b = torch.full((8, 1024, 4096), 1/64, dtype=torch.float8_e4m3fn, device='cuda')
scales_a = torch.ones((16, 8, 32), dtype=torch.float32, device='cuda')
scales_b = torch.ones((8, 8, 32), dtype=torch.float32, device='cuda')
plan = prepare_fp8_batched_gemm((values_a, scales_a), (values_b, scales_b))
values, scales = plan.run()
print(values.shape, scales.shape)
