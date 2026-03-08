import torch
import pytest
from flashinfer.dsv3_ops import mm_M1_16_K7168_N128, mm_M1_16_K7168_N256
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability


# Positive tests
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 13, 16])
@pytest.mark.parametrize(
    "num_experts,output_dtype,fn_to_test",
    (
        [256, torch.float32, mm_M1_16_K7168_N256],
        [128, torch.bfloat16, mm_M1_16_K7168_N128],
    ),
)
@pytest.mark.parametrize("hidden_dim", [7168])
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_dsv3_router_gemm_op(
    num_tokens, num_experts, hidden_dim, launch_with_pdl, output_dtype, fn_to_test
):
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability_number != 100:
        pytest.skip("DSv3 Router GEMM is only supported on SM100")

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(
        num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16
    ).t()  # column major
    out = torch.empty(num_tokens, num_experts, device="cuda", dtype=output_dtype)
    fn_to_test(mat_a, mat_b, out, launch_with_pdl=launch_with_pdl)
    ref = mat_a @ mat_b

    cos_sim = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


# Negative tests - test values just outside valid ranges
@pytest.mark.parametrize(
    "fn_array,num_tokens,num_experts,hidden_dim,mat_a_dtype,mat_b_dtype,out_dtype,mat_b_transpose,expected_error",
    [
        # Invalid num_tokens (must be 1-16)
        pytest.param(
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256],
            0,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_tokens",
            id="all-num_tokens_0",
        ),
        pytest.param(
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256],
            17,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_tokens",
            id="all-num_tokens_17",
        ),
        # Invalid num_experts (must be 128 or 256, depending on the function)
        pytest.param(
            [mm_M1_16_K7168_N128],
            8,
            127,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="N128-num_experts_127",
        ),
        pytest.param(
            [mm_M1_16_K7168_N128],
            8,
            129,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="N128-num_experts_129",
        ),
        pytest.param(
            [mm_M1_16_K7168_N256],
            8,
            255,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="N256-num_experts_255",
        ),
        pytest.param(
            [mm_M1_16_K7168_N256],
            8,
            257,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="N256-num_experts_257",
        ),
        # Invalid hidden_dim (must be 7168)
        pytest.param(
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256],
            8,
            256,
            7167,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
            id="all-hidden_dim_7167",
        ),
        pytest.param(
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256],
            8,
            256,
            7169,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
            id="all-hidden_dim_7169",
        ),
        # Invalid dtypes
        pytest.param(
            [mm_M1_16_K7168_N128],
            8,
            128,
            7168,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            True,
            "bfloat16",
            id="N128-invalid_mat_a_dtype",
        ),
        pytest.param(
            [mm_M1_16_K7168_N128],
            8,
            128,
            7168,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            True,
            "bfloat16",
            id="N128-invalid_mat_b_dtype",
        ),
        pytest.param(
            [mm_M1_16_K7168_N128],
            8,
            128,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "bfloat16",
            id="N128-invalid_out_dtype",
        ),
        pytest.param(
            [mm_M1_16_K7168_N256],
            8,
            256,
            7168,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            True,
            "bfloat16",
            id="N256-invalid_mat_a_dtype",
        ),
        pytest.param(
            [mm_M1_16_K7168_N256],
            8,
            256,
            7168,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            True,
            "bfloat16",
            id="N256-invalid_mat_b_dtype",
        ),
        pytest.param(
            [mm_M1_16_K7168_N256],
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            True,
            "float32",
            id="N256-invalid_out_dtype",
        ),
        # Invalid stride (mat_b not transposed = row-major instead of column-major)
        pytest.param(
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256],
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            False,
            "column-major",
            id="all-invalid_stride",
        ),
    ],
)
def test_dsv3_router_gemm_op_negative(
    fn_array,
    num_tokens,
    num_experts,
    hidden_dim,
    mat_a_dtype,
    mat_b_dtype,
    out_dtype,
    mat_b_transpose,
    expected_error,
):
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability_number != 100:
        pytest.skip("DSv3 Router GEMM is only supported on SM100")

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=mat_a_dtype)
    mat_b = torch.randn(num_experts, hidden_dim, device="cuda", dtype=mat_b_dtype)
    if mat_b_transpose:
        mat_b = mat_b.t()  # column major
    out = torch.randn(num_tokens, num_experts, device="cuda", dtype=out_dtype)

    for fn in fn_array:
        with pytest.raises(ValueError, match=expected_error):
            fn(mat_a, mat_b, out, launch_with_pdl=False)
