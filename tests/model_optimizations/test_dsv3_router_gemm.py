import torch
import pytest
import flashinfer.model_optimizations.dsv3.routergemm as dsv3_router_gemm
import torch.nn.functional as F


# Positive tests
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 13, 16])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_dim", [7168])
def test_dsv3_router_gemm_op(num_tokens, num_experts, hidden_dim):
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


# Negative tests - test values just outside valid ranges
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,mat_a_dtype,mat_b_dtype,out_dtype,mat_b_transpose,expected_error",
    [
        # Invalid num_tokens (must be 1-16)
        (
            0,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_tokens",
        ),
        (
            17,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_tokens",
        ),
        # Invalid num_experts (must be 256)
        (
            8,
            255,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
        ),
        (
            8,
            257,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
        ),
        # Invalid hidden_dim (must be 7168)
        (
            8,
            256,
            7167,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
        ),
        (
            8,
            256,
            7169,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
        ),
        # Invalid dtypes
        (8, 256, 7168, torch.float32, torch.bfloat16, torch.float32, True, "bfloat16"),
        (8, 256, 7168, torch.bfloat16, torch.float32, torch.float32, True, "bfloat16"),
        (8, 256, 7168, torch.bfloat16, torch.bfloat16, torch.bfloat16, True, "float32"),
        # Invalid stride (mat_b not transposed = row-major instead of column-major)
        (
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            False,
            "column-major",
        ),
    ],
)
def test_dsv3_router_gemm_op_negative(
    num_tokens,
    num_experts,
    hidden_dim,
    mat_a_dtype,
    mat_b_dtype,
    out_dtype,
    mat_b_transpose,
    expected_error,
):
    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=mat_a_dtype)
    mat_b = torch.randn(num_experts, hidden_dim, device="cuda", dtype=mat_b_dtype)
    if mat_b_transpose:
        mat_b = mat_b.t()  # column major
    out = torch.randn(num_tokens, num_experts, device="cuda", dtype=out_dtype)

    with pytest.raises(ValueError, match=expected_error):
        dsv3_router_gemm.dsv3_router_gemm_op(mat_a, mat_b, out, False, None)
