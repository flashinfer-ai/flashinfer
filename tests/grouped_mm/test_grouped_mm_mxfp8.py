"""Tests for flashinfer.grouped_mm.grouped_mm_mxfp8."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.grouped_mm import grouped_mm_mxfp8

from .conftest import (
    ref_grouped_mm,
    requires_cudnn_moe_block_scale,
    requires_grouped_mm_mxfp8_cc,
)


@requires_grouped_mm_mxfp8_cc
class TestGroupedMmMxfp8:
    """Tests for grouped_mm_mxfp8 (block-scaled MXFP8 grouped GEMM)."""

    MIN_COS_SIM = 0.9

    @staticmethod
    def _quantize_a(a_bf16):
        """Quantize 2D token activations (cum_m, k) → mxfp8 + scale."""
        _cum_m, k = a_bf16.shape
        a_mxfp8, a_scale = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=True)
        a_scale = a_scale.reshape(-1, k // 32)
        return a_mxfp8, a_scale

    @staticmethod
    def _quantize_b(b_bf16):
        """Quantize 3D expert weights (num_experts, n, k) → mxfp8 + scale."""
        num_experts, _n, k = b_bf16.shape
        b_mxfp8, b_scale = mxfp8_quantize(b_bf16, is_sf_swizzled_layout=True)
        b_scale = b_scale.reshape(num_experts, -1, k // 32)
        return b_mxfp8, b_scale

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    @pytest.mark.parametrize("tokens_per_expert", [128, 256])
    @pytest.mark.parametrize("k", [128, 256, 512])
    @pytest.mark.parametrize("n", [128, 256])
    def test_uniform_distribution(self, num_experts, tokens_per_expert, k, n):
        """Each expert gets the same number of tokens."""
        torch.manual_seed(42)
        cum_m = num_experts * tokens_per_expert

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (
            torch.arange(num_experts + 1, device="cuda") * tokens_per_expert
        ).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM, (
            f"Cosine similarity {cos_sim:.4f} too low (expected > {self.MIN_COS_SIM})"
        )

    @requires_cudnn_moe_block_scale
    def test_non_uniform_distribution(self):
        """Experts get different numbers of tokens."""
        torch.manual_seed(0)
        num_experts = 4
        seg_lens = [256, 128, 384, 256]
        cum_m = sum(seg_lens)
        k, n = 512, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_empty_experts(self):
        """Some experts receive zero tokens."""
        torch.manual_seed(1)
        num_experts = 4
        seg_lens = [256, 0, 512, 0]
        cum_m = sum(seg_lens)
        k, n = 256, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_single_expert(self):
        """Degenerate case: only one expert (equivalent to dense mm)."""
        torch.manual_seed(2)
        cum_m, k, n = 256, 512, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, cum_m], dtype=torch.int32, device="cuda")

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = (a_bf16.float() @ b_bf16[0].float().T).to(torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_preallocated_output(self):
        """Pass a pre-allocated output tensor."""
        torch.manual_seed(3)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        result = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out=out,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        assert result.data_ptr() == out.data_ptr()
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            result.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize(
        "out_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_out_dtype(self, out_dtype):
        """Verify different output dtypes work correctly."""
        torch.manual_seed(4)
        num_experts, tpe, k, n = 2, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=out_dtype,
        )
        assert out.dtype == out_dtype

        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=out_dtype)
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_graph_cache_reuse(self):
        """Calling twice with identical shapes should reuse the cached graph."""
        torch.manual_seed(5)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out1 = grouped_mm_mxfp8(a_mxfp8, b_mxfp8, a_scale, b_scale, m_indptr)
        out2 = grouped_mm_mxfp8(a_mxfp8, b_mxfp8, a_scale, b_scale, m_indptr)
        torch.testing.assert_close(out1, out2)

    @requires_cudnn_moe_block_scale
    def test_quantize_dtypes(self):
        """Verify quantized data and scale tensor dtypes."""
        torch.manual_seed(9)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        assert a_mxfp8.dtype == torch.float8_e4m3fn
        assert b_mxfp8.dtype == torch.float8_e4m3fn
        assert a_scale.dtype == torch.uint8
        assert b_scale.dtype == torch.uint8
        assert a_scale.ndim == 2
        assert b_scale.ndim == 3


@requires_cudnn_moe_block_scale
@requires_grouped_mm_mxfp8_cc
class TestGroupedMmMxfp8Validation:
    def test_wrong_input_dtype(self):
        a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_descale_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.float32, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_m_indptr_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_m_indptr_length(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 48, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_k_mismatch(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 256, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 8, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)
