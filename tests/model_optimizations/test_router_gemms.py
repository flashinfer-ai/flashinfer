import torch
import pytest
from flashinfer.gemm import (
    mm_M1_16,
    mm_M1_16_K6144_N256,
    mm_M1_16_K7168_N128,
    mm_M1_16_K7168_N256,
)
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability

# Kept in sync with flashinfer.gemm.routergemm._ROUTER_GEMM_SUPPORTED_ARCHS.
SUPPORTED_ARCHS = [90, 100, 103, 107]


def skip_if_unsupported():
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability_number not in SUPPORTED_ARCHS:
        pytest.skip(f"Router GEMM is only supported on SM{SUPPORTED_ARCHS}")
    return compute_capability_number


# Positive tests
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 13, 16])
@pytest.mark.parametrize(
    "num_experts,output_dtype,hidden_dim,fn_to_test",
    (
        [256, torch.float32, 7168, mm_M1_16_K7168_N256],
        [128, torch.bfloat16, 7168, mm_M1_16_K7168_N128],
        [256, torch.float32, 6144, mm_M1_16_K6144_N256],
    ),
)
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_dsv3_router_gemm_op(
    num_tokens, num_experts, hidden_dim, launch_with_pdl, output_dtype, fn_to_test
):
    skip_if_unsupported()

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
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256, mm_M1_16_K6144_N256],
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
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256, mm_M1_16_K6144_N256],
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
            [mm_M1_16_K7168_N128, mm_M1_16_K7168_N256, mm_M1_16_K6144_N256],
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
        # K6144_N256 specific: invalid num_experts (must be 256)
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            255,
            6144,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="K6144_N256-num_experts_255",
        ),
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            257,
            6144,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "num_experts",
            id="K6144_N256-num_experts_257",
        ),
        # K6144_N256 specific: invalid hidden_dim (must be 6144)
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            256,
            6143,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
            id="K6144_N256-hidden_dim_6143",
        ),
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            256,
            6145,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            True,
            "hidden_dim",
            id="K6144_N256-hidden_dim_6145",
        ),
        # K6144_N256 specific: invalid dtypes
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            256,
            6144,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            True,
            "bfloat16",
            id="K6144_N256-invalid_mat_a_dtype",
        ),
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            256,
            6144,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            True,
            "bfloat16",
            id="K6144_N256-invalid_mat_b_dtype",
        ),
        pytest.param(
            [mm_M1_16_K6144_N256],
            8,
            256,
            6144,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            True,
            "float32",
            id="K6144_N256-invalid_out_dtype",
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
    skip_if_unsupported()

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=mat_a_dtype)
    mat_b = torch.randn(num_experts, hidden_dim, device="cuda", dtype=mat_b_dtype)
    if mat_b_transpose:
        mat_b = mat_b.t()  # column major
    out = torch.randn(num_tokens, num_experts, device="cuda", dtype=out_dtype)

    for fn in fn_array:
        with pytest.raises(ValueError, match=expected_error):
            fn(mat_a, mat_b, out, launch_with_pdl=False)


# ---------------------------------------------------------------------------
# Generic entry point: any expert count, any K that is a multiple of 1024,
# float32 or bfloat16 output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 13, 16])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "num_experts,hidden_dim",
    [
        (128, 7168),  # Mistral Large 3
        (256, 7168),  # DeepSeek-V3
        (256, 6144),  # GLM-MoE-DSA
        (384, 7168),  # Kimi-K2
        (896, 7168),  # Kimi-K3
        (32, 1024),  # smallest supported K
        (257, 2048),  # expert count that is not a power of two
    ],
)
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_mm_M1_16(num_tokens, num_experts, hidden_dim, output_dtype, launch_with_pdl):
    skip_if_unsupported()

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(
        num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16
    ).t()  # column major
    out = torch.empty(num_tokens, num_experts, device="cuda", dtype=output_dtype)
    mm_M1_16(mat_a, mat_b, out, launch_with_pdl=launch_with_pdl)
    ref = mat_a @ mat_b

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
@pytest.mark.parametrize(
    "num_experts,hidden_dim,output_dtype,alias",
    (
        [256, 7168, torch.float32, mm_M1_16_K7168_N256],
        [128, 7168, torch.bfloat16, mm_M1_16_K7168_N128],
        [256, 6144, torch.float32, mm_M1_16_K6144_N256],
    ),
)
def test_mm_M1_16_matches_fixed_shape_alias(
    num_tokens, num_experts, hidden_dim, output_dtype, alias
):
    """The fixed-shape names are aliases, so they must select the same kernel."""
    skip_if_unsupported()

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(
        num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16
    ).t()
    out_generic = torch.empty(
        num_tokens, num_experts, device="cuda", dtype=output_dtype
    )
    out_alias = torch.empty(num_tokens, num_experts, device="cuda", dtype=output_dtype)

    mm_M1_16(mat_a, mat_b, out_generic, launch_with_pdl=False)
    alias(mat_a, mat_b, out_alias, launch_with_pdl=False)

    torch.testing.assert_close(out_generic, out_alias, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,mat_a_dtype,mat_b_dtype,out_dtype,mat_b_layout,expected_error",
    [
        # num_tokens outside [1, 16]
        pytest.param(
            0,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "t",
            "num_tokens",
            id="num_tokens_0",
        ),
        pytest.param(
            17,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "t",
            "num_tokens",
            id="num_tokens_17",
        ),
        # hidden_dim must be a multiple of one K iteration (1024)
        pytest.param(
            8,
            256,
            7167,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "t",
            "hidden_dim",
            id="hidden_dim_7167",
        ),
        pytest.param(
            8,
            256,
            1536,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "t",
            "hidden_dim",
            id="hidden_dim_1536",
        ),
        # dtypes
        pytest.param(
            8,
            256,
            7168,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            "t",
            "bfloat16",
            id="mat_a_fp32",
        ),
        pytest.param(
            8,
            256,
            7168,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            "t",
            "bfloat16",
            id="mat_b_fp32",
        ),
        pytest.param(
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float16,
            "t",
            "torch.float32 or torch.bfloat16",
            id="out_fp16",
        ),
        # mat_b layout
        pytest.param(
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "row_major",
            "column-major",
            id="mat_b_row_major",
        ),
        pytest.param(
            8,
            256,
            7168,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            "strided",
            "column-major and contiguous",
            id="mat_b_strided",
        ),
    ],
)
def test_mm_M1_16_negative(
    num_tokens,
    num_experts,
    hidden_dim,
    mat_a_dtype,
    mat_b_dtype,
    out_dtype,
    mat_b_layout,
    expected_error,
):
    skip_if_unsupported()

    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=mat_a_dtype)
    if mat_b_layout == "strided":
        # A column-major *view* of a wider weight matrix: stride(0) == 1 as the
        # kernel wants, but the columns are not densely packed.
        mat_b = torch.randn(
            num_experts, 2 * hidden_dim, device="cuda", dtype=mat_b_dtype
        )[:, :hidden_dim].t()
    else:
        mat_b = torch.randn(num_experts, hidden_dim, device="cuda", dtype=mat_b_dtype)
        if mat_b_layout == "t":
            mat_b = mat_b.t()
    out = torch.randn(num_tokens, num_experts, device="cuda", dtype=out_dtype)

    with pytest.raises(ValueError, match=expected_error):
        mm_M1_16(mat_a, mat_b, out, launch_with_pdl=False)
