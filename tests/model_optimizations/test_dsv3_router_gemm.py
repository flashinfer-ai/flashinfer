import torch

import flashinfer.model_optimizations.dsv3.routergemm as dsv3_router_gemm
import torch.nn.functional as F


# TODO: Add negative tests
def test_dsv3_router_gemm_op():
    num_tokens = 16
    num_experts = 256
    hidden_dim = 7168
    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(
        num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16
    ).t()  # column major
    out = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    # TODO: We shouldn't need to pass bias here, but just for testing
    dsv3_router_gemm.dsv3_router_gemm_op(mat_a, mat_b, out, False, None)
    ref = mat_a @ mat_b

    cos_sim = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    test_dsv3_router_gemm_op()
