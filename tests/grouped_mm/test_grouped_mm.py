"""
Tests for flashinfer.grouped_mm (cuDNN MOE Grouped GEMM).
"""

import pytest
import torch

from flashinfer.grouped_mm import grouped_mm_bf16

try:
    import cudnn

    CUDNN_AVAILABLE = True
    CUDNN_BACKEND_VERSION = cudnn.backend_version()
except (ImportError, OSError):
    CUDNN_AVAILABLE = False
    CUDNN_BACKEND_VERSION = 0

requires_cudnn_moe = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 91800,
    reason="cuDNN MOE requires backend >= 9.18.0",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def ref_grouped_mm(a, b, m_indptr, out_dtype):
    """Loop over experts, matmul with transposed weight (NT layout)."""
    num_experts = b.shape[0]
    n = b.shape[1]
    cum_m = a.shape[0]
    out = torch.zeros(cum_m, n, dtype=out_dtype, device=a.device)
    for e in range(num_experts):
        start = m_indptr[e].item()
        end = m_indptr[e + 1].item()
        if start < end:
            out[start:end] = (a[start:end].float() @ b[e].float().T).to(out_dtype)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroupedMmBf16:
    @requires_cudnn_moe
    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    @pytest.mark.parametrize("tokens_per_expert", [32, 128, 256])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("n", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_uniform_distribution(
        self,
        num_experts,
        tokens_per_expert,
        k,
        n,
        dtype,
    ):
        """Each expert gets the same number of tokens."""
        torch.manual_seed(42)
        cum_m = num_experts * tokens_per_expert

        a = torch.randn(cum_m, k, dtype=dtype, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=dtype, device="cuda")
        m_indptr = (
            torch.arange(num_experts + 1, device="cuda") * tokens_per_expert
        ).to(torch.int32)

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=dtype)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=dtype)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_uniform_distribution(self, dtype):
        """Experts get different numbers of tokens."""
        torch.manual_seed(0)
        num_experts = 4
        seg_lens = [64, 32, 128, 96]
        cum_m = sum(seg_lens)
        k, n = 512, 256

        a = torch.randn(cum_m, k, dtype=dtype, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=dtype, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=dtype)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=dtype)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_empty_experts(self):
        """Some experts receive zero tokens."""
        torch.manual_seed(1)
        num_experts = 4
        seg_lens = [64, 0, 128, 0]
        cum_m = sum(seg_lens)
        k, n = 256, 256

        a = torch.randn(cum_m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=torch.float16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=torch.float16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.float16)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_single_expert(self):
        """Degenerate case: only one expert (equivalent to dense mm)."""
        torch.manual_seed(2)
        cum_m, k, n = 256, 512, 1024

        a = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, cum_m], dtype=torch.int32, device="cuda")

        out = grouped_mm_bf16(a, b, m_indptr)
        ref = (a.float() @ b[0].float().T).to(torch.bfloat16)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_preallocated_output(self):
        """Pass a pre-allocated output tensor."""
        torch.manual_seed(3)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe

        a = torch.randn(cum_m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=torch.float16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        out = torch.empty(cum_m, n, dtype=torch.float16, device="cuda")
        result = grouped_mm_bf16(a, b, m_indptr, out=out)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.float16)

        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_out_dtype_fp32(self):
        """Output in float32 while input is bf16."""
        torch.manual_seed(4)
        num_experts, tpe, k, n = 2, 128, 256, 256
        cum_m = num_experts * tpe

        a = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=torch.float32)
        assert out.dtype == torch.float32

        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.float32)
        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_graph_cache_reuse(self):
        """Calling twice with identical shapes should reuse the cached graph."""
        torch.manual_seed(5)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.bfloat16

        a = torch.randn(cum_m, k, dtype=dtype, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=dtype, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        out1 = grouped_mm_bf16(a, b, m_indptr)
        out2 = grouped_mm_bf16(a, b, m_indptr)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Validation / error tests (do not require cuDNN)
# ---------------------------------------------------------------------------


class TestGroupedMmBf16Validation:
    def test_dtype_mismatch(self):
        a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_bf16(a, b, m_indptr)

    def test_wrong_m_indptr_dtype(self):
        a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_bf16(a, b, m_indptr)

    def test_wrong_m_indptr_length(self):
        a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, 32, 48, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_bf16(a, b, m_indptr)

    def test_k_mismatch(self):
        a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 256, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_bf16(a, b, m_indptr)


if __name__ == "__main__":
    test = TestGroupedMmBf16()
    test.test_uniform_distribution(4, 128, 256, 256, torch.float16)
    test.test_non_uniform_distribution(torch.bfloat16)
    test.test_empty_experts()
    test.test_single_expert()
    test.test_preallocated_output()
    test.test_out_dtype_fp32()
    test.test_graph_cache_reuse()
    print("All tests passed!")
