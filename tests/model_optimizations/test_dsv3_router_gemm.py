import torch

import flashinfer.model_optimizations.dsv3.routergemm as dsv3_router_gemm


def test_dsv3_router_gemm_op():
    num_tokens = 16
    num_experts = 256
    hidden_dim = 7168
    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda")
    mat_b = torch.randn(num_experts, hidden_dim, device="cuda")
    out = torch.randn(num_tokens, num_experts, device="cuda")
    dsv3_router_gemm.dsv3_router_gemm_op(mat_a, mat_b, out, False)
    assert torch.allclose(out, mat_a @ mat_b.T)


if __name__ == "__main__":
    test_dsv3_router_gemm_op()
