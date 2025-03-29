import pytest
import torch

import flashinfer
import flashinfer.triton

def torch_gemm(a, b, c, alpha=1.0, beta=0.0):
    convert = a.dtype == torch.float8_e4m3fn

    if convert:
        a = a.to(torch.float16)
        b = b.to(torch.float16)
        c = c.to(torch.float16)

    # Perform matrix multiplication
    x = torch.matmul(a, b.T)
    # Scale the result by alpha
    c = alpha * x + beta * c

    if convert:
        c = c.to(torch.float8_e4m3fn)

    return c

def torch_addmm(a, b, c, alpha=1.0, beta=0.0):
    # Transpose b to match torch_gemm's matmul(a, b.T)
    C = torch.addmm(c, a, b.T, beta=beta, alpha=alpha)
    return C

@pytest.mark.parametrize("M", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("N", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 8192])
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0]) 
@pytest.mark.parametrize("num_sms", [1, 16, 64, 128, 132, 133])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
# todo: torch.float8_e4m3fn, bf16
# @pytest.mark.parametrize("M", [128])
# @pytest.mark.parametrize("N", [128])
# @pytest.mark.parametrize("K", [128])
# @pytest.mark.parametrize("alpha", [2.0])
# @pytest.mark.parametrize("beta", [2.0]) 
# @pytest.mark.parametrize("num_sms", [1])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    c = torch.randn((M, N), device="cuda", dtype=torch.float16).to(dtype)

    c_torch = torch_gemm(a, b, c, alpha, beta) if dtype == torch.float16 or dtype == torch.float32 else None
    c_triton = flashinfer.triton.sm_constraint_gemm.matmul_persistent(a, b.T, c, alpha, beta, num_sms)

    in_place_triton = c_triton.data_ptr() == c.data_ptr() and torch.allclose(c_triton.to(torch.float16), c.to(torch.float16))
    assert in_place_triton # modified in place

    cmp_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    if c_torch is not None:
        torch_vs_triton = torch.allclose(c_torch.to(cmp_dtype), c_triton.to(cmp_dtype), atol=1.0)
        if torch_vs_triton == False:
            print(f"c_torch: {c_torch}")
            print(f"c_triton: {c_triton}")
            print(f"max diff: {torch.max(torch.abs(c_torch.to(cmp_dtype) - c_triton.to(cmp_dtype)))}")
        assert torch_vs_triton # value is correct
