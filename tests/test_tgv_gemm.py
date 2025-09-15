import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    tgv_gemm_sm100,
)


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
@pytest.mark.parametrize("k", [1024, 2048, 3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tgv_gemm_sm100(m, n, k, dtype):
    """Test tgv_gemm_sm100 with autotuner integration."""
    A = torch.randn(m, k, device="cuda", dtype=dtype)
    B = torch.randn(n, k, device="cuda", dtype=dtype).t()  # column major
    bias = torch.randn(n, device="cuda", dtype=dtype)

    print(
        f"Input tensors: A {A.shape}, B {B.shape}, bias {bias.shape}, dtype: {A.dtype}",
        flush=True,
    )

    # Reference computation
    reference = F.linear(A, B.T, bias)

    # Test with TGV runner only
    print("Testing tgv_gemm_sm100 with TGV runner", flush=True)
    result = tgv_gemm_sm100(A, B, bias)

    # Check correctness
    cos_sim = F.cosine_similarity(reference.reshape(-1), result.reshape(-1), dim=0)
    print(f"Cosine similarity: {cos_sim:.6f}", flush=True)
    assert cos_sim > 0.99

    # Test with PDL enabled
    print("Testing tgv_gemm_sm100 with PDL", flush=True)
    result_pdl = tgv_gemm_sm100(A, B, bias, pdl=True)

    # Check correctness for PDL
    cos_sim_pdl = F.cosine_similarity(
        reference.reshape(-1), result_pdl.reshape(-1), dim=0
    )
    print(f"PDL Cosine similarity: {cos_sim_pdl:.6f}", flush=True)
    assert cos_sim_pdl > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
