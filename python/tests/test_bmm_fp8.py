import torch
from flashinfer import bmm_fp8

input = torch.randn([16, 64, 48], device="cuda", dtype=torch.bfloat16)
# transpose, cuBLASLt row major column major
input_fp8 = input.to(torch.float8_e4m3fn).transpose(-1, -2)
mat2 = torch.randn([16, 64, 80], device="cuda", dtype=torch.bfloat16)
mat2_fp8 = mat2.to(torch.float8_e4m3fn)

res = torch.empty([16, 48, 80], device="cuda", dtype=torch.bfloat16)

bmm_fp8(input_fp8, mat2_fp8, res)

print(res)
