import cutlass
import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    tgv_gemm_sm100,
)

from flashinfer.gemm.gemm_base import _match_sm_version


_TGV_GEMM_CASES = [
    (m, n, k, dtype, dtype, None, None)
    for m in [1, 8, 16, 32, 64]
    for n in [1024, 2048, 4096]
    for k in [1024, 2048, 3072]
    for dtype in [torch.bfloat16, torch.float16]
] + [
    (7, 129, 256, a_dtype, b_dtype, sf_dtype, sf_vec_size)
    for a_dtype, b_dtype, sf_dtype, sf_vec_size in [
        (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
        (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
        (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
        (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
    ]
]


def _blockscaled_tgv_case(m, n, k, a_dtype, b_dtype, sf_dtype, sf_vec_size):
    from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
        _decode_ab_to_f32,
        make_blockscaled_tensors,
    )

    kernel_a, kernel_b, sfa, sfb, _, sfa_simple, sfb_simple = make_blockscaled_tensors(
        (n, m, k, 1),
        b_dtype,
        a_dtype,
        sf_dtype,
        sf_vec_size,
        cutlass.BFloat16,
    )
    a = kernel_b[:, :, 0]
    b = kernel_a[:, :, 0].T
    bias = torch.linspace(-0.5, 0.5, n, dtype=torch.bfloat16, device="cuda")

    a_scale = torch.repeat_interleave(
        sfb_simple.cpu().to(torch.float32), sf_vec_size, dim=1
    )[:, :k]
    b_scale = torch.repeat_interleave(
        sfa_simple.cpu().to(torch.float32), sf_vec_size, dim=1
    )[:, :k]
    reference = (
        torch.einsum(
            "mkl,nkl->mnl",
            _decode_ab_to_f32(kernel_b) * a_scale,
            _decode_ab_to_f32(kernel_a) * b_scale,
        )
        + bias.cpu().to(torch.float32)[None, :, None]
    ).to(torch.bfloat16)[:, :, 0]
    return a, b, sfb, sfa, bias, reference


@pytest.mark.parametrize("m,n,k,a_dtype,b_dtype,sf_dtype,sf_vec_size", _TGV_GEMM_CASES)
def test_tgv_gemm_sm100(m, n, k, a_dtype, b_dtype, sf_dtype, sf_vec_size):
    """Test tgv_gemm_sm100 with autotuner integration."""
    device = torch.device("cuda")
    if not _match_sm_version(device, ["100", "103"]):
        pytest.skip("TGV GEMM requires SM100, SM103 architecture")

    if sf_dtype is not None:
        a, b, a_descale, b_descale, bias, reference = _blockscaled_tgv_case(
            m, n, k, a_dtype, b_dtype, sf_dtype, sf_vec_size
        )
        result = tgv_gemm_sm100(
            a,
            b,
            bias,
            a_descale=a_descale,
            b_descale=b_descale,
        )
        torch.testing.assert_close(result.cpu(), reference, atol=1e-1, rtol=1e-3)
        result_pdl = tgv_gemm_sm100(
            a,
            b,
            bias,
            pdl=True,
            a_descale=a_descale,
            b_descale=b_descale,
        )
        torch.testing.assert_close(result_pdl.cpu(), reference, atol=1e-1, rtol=1e-3)
        return

    A = torch.randn(m, k, device=device, dtype=a_dtype)
    B = torch.randn(n, k, device=device, dtype=b_dtype).t()  # column major
    bias = torch.randn(n, device=device, dtype=a_dtype)

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
