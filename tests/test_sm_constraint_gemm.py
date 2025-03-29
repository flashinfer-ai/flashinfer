import pytest
import torch

import flashinfer
import flashinfer.triton

def torch_gemm(a, b, c, alpha=1.0, beta=0.0):
    C = torch.addmm(c, a, b, beta=beta, alpha=alpha)
    return C

@pytest.mark.parametrize("M", [128, 256, 512])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("K", [128, 256, 512])
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0]) 
@pytest.mark.parametrize("num_sms", [1, 2, 4, 8, 16, 32, 64, 128])
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms):
    a = torch.randn(M, K)
    b = torch.randn(N, K)
    c = torch.randn(M, N)
    c_ref = torch_gemm(a, b, c, alpha, beta)
    c_triton = flashinfer.triton.sm_constraint_gemm.matmul_persistent(a, b, c, alpha, beta, num_sms)
    assert torch.allclose(c_ref, c_triton)
