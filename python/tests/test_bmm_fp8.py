import torch
import numpy as np
from flashinfer import bmm_fp8

input = torch.randn([1, 48, 64], device="cuda", dtype=torch.bfloat16)
input_fp8 = input.to(torch.float8_e4m3fn)
mat2 = torch.randn([1, 64, 80], device="cuda", dtype=torch.bfloat16)
mat2_fp8 = mat2.to(torch.float8_e4m3fn).transpose(-1, -2).contiguous()
mat2_fp8 = mat2_fp8.transpose(-1, -2)

res = torch.empty([1, 48, 80], device="cuda", dtype=torch.bfloat16)
bmm_fp8(input_fp8, mat2_fp8, res)
res_bf16 = input @ mat2

np.testing.assert_allclose(
    res.float().cpu().numpy(), res_bf16.float().cpu().numpy(), rtol=1e-1, atol=1e-1
)
