"""Tests for flashinfer.grouped_mm.grouped_mm_fp8."""

import pytest
import torch

from flashinfer.grouped_mm import grouped_mm_fp8

from .conftest import (
    ref_grouped_mm,
    requires_cudnn_moe,
    requires_grouped_mm_fp8_cc,
)


@requires_grouped_mm_fp8_cc
class TestGroupedMmFp8:
    @staticmethod
    def _make_fp8(shape, dtype, device="cuda"):
        """Generate a random FP8 tensor by casting from a bounded normal distribution."""
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp(-1, 1)
        return t.to(dtype)

    @requires_cudnn_moe
    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    @pytest.mark.parametrize("tokens_per_expert", [32, 128, 256])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("n", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
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

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (
            torch.arange(num_experts + 1, device="cuda") * tokens_per_expert
        ).to(torch.int32)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_non_uniform_distribution(self, dtype):
        """Experts get different numbers of tokens."""
        torch.manual_seed(0)
        num_experts = 4
        seg_lens = [64, 32, 128, 96]
        cum_m = sum(seg_lens)
        k, n = 512, 256

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_empty_experts(self):
        """Some experts receive zero tokens."""
        torch.manual_seed(1)
        num_experts = 4
        seg_lens = [64, 0, 128, 0]
        cum_m = sum(seg_lens)
        k, n = 256, 256
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_single_expert(self):
        """Degenerate case: only one expert (equivalent to dense mm)."""
        torch.manual_seed(2)
        cum_m, k, n = 256, 512, 1024
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((1, n, k), dtype)
        m_indptr = torch.tensor([0, cum_m], dtype=torch.int32, device="cuda")
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = (alpha.float() * (a.float() @ b[0].float().T)).to(torch.bfloat16)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_preallocated_output(self):
        """Pass a pre-allocated output tensor."""
        torch.manual_seed(3)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        result = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out=out)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    @pytest.mark.parametrize(
        "out_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_out_dtype(self, out_dtype):
        """Verify different output dtypes work correctly."""
        torch.manual_seed(4)
        num_experts, tpe, k, n = 2, 128, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=out_dtype)
        assert out.dtype == out_dtype

        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=out_dtype, alpha=alpha)
        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    @pytest.mark.parametrize("alpha_val", [0.5, 1.0, 2.0])
    def test_alpha_scaling(self, alpha_val):
        """Verify the alpha scaling factor is applied correctly."""
        torch.manual_seed(6)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)
        alpha = torch.tensor([alpha_val], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_no_alpha(self):
        """FP8 grouped mm without alpha scaling."""
        torch.manual_seed(7)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        out = grouped_mm_fp8(a, b, m_indptr, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_graph_cache_reuse(self):
        """Calling twice with identical shapes should reuse the cached graph."""
        torch.manual_seed(5)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe
        dtype = torch.float8_e4m3fn

        a = self._make_fp8((cum_m, k), dtype)
        b = self._make_fp8((num_experts, n, k), dtype)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out1 = grouped_mm_fp8(a, b, m_indptr, alpha=alpha)
        out2 = grouped_mm_fp8(a, b, m_indptr, alpha=alpha)
        torch.testing.assert_close(out1, out2)

    @requires_cudnn_moe
    def test_mixed_fp8_dtypes(self):
        """a and b can use different FP8 sub-types."""
        torch.manual_seed(8)
        num_experts, tpe, k, n = 4, 64, 256, 256
        cum_m = num_experts * tpe

        a = self._make_fp8((cum_m, k), torch.float8_e4m3fn)
        b = self._make_fp8((num_experts, n, k), torch.float8_e5m2)
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        out = grouped_mm_fp8(a, b, m_indptr, alpha=alpha, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16, alpha=alpha)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)


@requires_cudnn_moe
@requires_grouped_mm_fp8_cc
class TestGroupedMmFp8Validation:
    def test_wrong_input_dtype(self):
        a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.float16, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr)

    def test_wrong_m_indptr_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr)

    def test_wrong_m_indptr_length(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 48, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr)

    def test_k_mismatch(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 256, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr)

    def test_wrong_alpha_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        alpha = torch.tensor([1.0], dtype=torch.float16, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr, alpha=alpha)

    def test_wrong_alpha_shape(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        alpha = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp8(a, b, m_indptr, alpha=alpha)
