import pytest
import torch

import flashinfer
import flashinfer.triton

def torch_gemm(a, b, c, alpha=1.0, beta=0.0):
    # Perform matrix multiplication
    x = torch.matmul(a, b.T)
    # Scale the result by alpha
    c = alpha * x + beta * c
    return c

def torch_addmm(a, b, c, alpha=1.0, beta=0.0):
    # Transpose b to match torch_gemm's matmul(a, b.T)
    C = torch.addmm(c, a, b.T, beta=beta, alpha=alpha)
    return C

# @pytest.mark.parametrize("M", [128, 256, 512])
# @pytest.mark.parametrize("N", [128, 256, 512])
# @pytest.mark.parametrize("K", [128, 256, 512])
# @pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
# @pytest.mark.parametrize("beta", [0.0, 0.5, 2.0]) 
# @pytest.mark.parametrize("num_sms", [1, 2, 4, 8, 16, 32, 64, 128, 132])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("beta", [0.0]) 
@pytest.mark.parametrize("num_sms", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    b = b.T.contiguous()
    c = torch.randn((M, N), device="cuda", dtype=dtype)

    print(f"a: {a.shape}, b: {b.shape}, c: {c.shape}")
    c_ref = torch_gemm(a, b, c, alpha, beta)
    c_triton = flashinfer.triton.sm_constraint_gemm.matmul_persistent(a, b.T, c, alpha, beta, num_sms)
    assert torch.allclose(c_ref, c_triton) # value is correct
    assert torch.allclose(c, c_triton) # modified in place
