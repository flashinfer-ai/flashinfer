"""Tests for flashinfer.grouped_mm.grouped_mm_bf16."""

import pytest
import torch

from flashinfer.grouped_mm import grouped_mm_bf16

from .conftest import (
    ref_grouped_mm,
    requires_cudnn_moe,
    requires_grouped_mm_bf16_cc,
)


@requires_grouped_mm_bf16_cc
class TestGroupedMmBf16:
    @requires_cudnn_moe
    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    @pytest.mark.parametrize("tokens_per_expert", [32, 128, 256])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("n", [256, 1024])
    def test_uniform_distribution(
        self,
        num_experts,
        tokens_per_expert,
        k,
        n,
    ):
        """Each expert gets the same number of tokens."""
        torch.manual_seed(42)
        cum_m = num_experts * tokens_per_expert
        dtype = torch.bfloat16

        a = torch.randn(cum_m, k, dtype=dtype, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=dtype, device="cuda")
        m_indptr = (
            torch.arange(num_experts + 1, device="cuda") * tokens_per_expert
        ).to(torch.int32)

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=dtype)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=dtype)

        torch.testing.assert_close(out, ref, atol=0.125, rtol=0.125)

    @requires_cudnn_moe
    def test_non_uniform_distribution(self):
        """Experts get different numbers of tokens."""
        torch.manual_seed(0)
        num_experts = 4
        seg_lens = [64, 32, 128, 96]
        cum_m = sum(seg_lens)
        k, n = 512, 256
        dtype = torch.bfloat16

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

        a = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        out = grouped_mm_bf16(a, b, m_indptr, out_dtype=torch.bfloat16)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16)

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

        a = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        result = grouped_mm_bf16(a, b, m_indptr, out=out)
        ref = ref_grouped_mm(a, b, m_indptr, out_dtype=torch.bfloat16)

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


@requires_grouped_mm_bf16_cc
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
