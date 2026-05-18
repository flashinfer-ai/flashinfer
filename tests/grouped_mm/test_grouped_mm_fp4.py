"""Tests for flashinfer.grouped_mm.grouped_mm_fp4."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout
from flashinfer.fp4_quantization import nvfp4_quantize
from flashinfer.grouped_mm import grouped_mm_fp4

from .conftest import (
    ref_grouped_mm,
    requires_cudnn_moe_block_scale,
    requires_grouped_mm_fp4_cc,
)


@requires_grouped_mm_fp4_cc
class TestGroupedMmFp4:
    """Tests for grouped_mm_fp4 (NVFP4 grouped GEMM)."""

    MIN_COS_SIM = 0.9

    @staticmethod
    def _quantize_a(a_bf16):
        """Quantize 2D token activations (cum_m, k) → nvfp4 + scale."""
        global_sf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
        a_fp4, a_sf = nvfp4_quantize(
            a_bf16,
            global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        _cum_m, k = a_bf16.shape
        a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
        return a_fp4, a_sf, global_sf

    @staticmethod
    def _quantize_b(b_bf16):
        """Quantize 3D expert weights (num_experts, n, k) → nvfp4 + scale.

        Reshapes to 2D for quantization, then reshapes back.
        """
        num_experts, n, k = b_bf16.shape
        b_2d = b_bf16.reshape(num_experts * n, k)
        global_sf = (448 * 6) / b_2d.float().abs().nan_to_num().max()
        b_fp4, b_sf = nvfp4_quantize(
            b_2d,
            global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        b_fp4 = b_fp4.reshape(num_experts, n, k // 2)
        b_sf = b_sf.view(torch.float8_e4m3fn).reshape(num_experts, -1, k // 16)
        return b_fp4, b_sf, global_sf

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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            block_size=16,
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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            block_size=16,
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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            block_size=16,
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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            block_size=16,
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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        result = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out=out,
            block_size=16,
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
    @pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
    def test_out_dtype(self, out_dtype):
        """Verify different output dtypes work correctly."""
        torch.manual_seed(4)
        num_experts, tpe, k, n = 2, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out_dtype=out_dtype,
            block_size=16,
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

        a_fp4, a_sf, a_gsf = self._quantize_a(a_bf16)
        b_fp4, b_sf, b_gsf = self._quantize_b(b_bf16)
        alpha = torch.tensor(
            [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda"
        )

        out1 = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            block_size=16,
        )
        out2 = grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            block_size=16,
        )
        torch.testing.assert_close(out1, out2)


@requires_cudnn_moe_block_scale
@requires_grouped_mm_fp4_cc
class TestGroupedMmFp4Validation:
    def test_wrong_input_dtype(self):
        a = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 64, dtype=torch.bfloat16, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.float8_e4m3fn, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp4(a, b, a_descale, b_descale, m_indptr, block_size=16)

    def test_wrong_m_indptr_dtype(self):
        a = torch.zeros(64, 64, dtype=torch.uint8, device="cuda")
        b = torch.zeros(2, 64, 64, dtype=torch.uint8, device="cuda")
        a_descale = torch.zeros(64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        b_descale = torch.zeros(2, 64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp4(a, b, a_descale, b_descale, m_indptr, block_size=16)

    def test_wrong_m_indptr_length(self):
        a = torch.zeros(64, 64, dtype=torch.uint8, device="cuda")
        b = torch.zeros(2, 64, 64, dtype=torch.uint8, device="cuda")
        a_descale = torch.zeros(64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        b_descale = torch.zeros(2, 64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 48, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp4(a, b, a_descale, b_descale, m_indptr, block_size=16)

    def test_k_mismatch(self):
        a = torch.zeros(64, 64, dtype=torch.uint8, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.uint8, device="cuda")
        a_descale = torch.zeros(64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        b_descale = torch.zeros(2, 64, 16, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp4(a, b, a_descale, b_descale, m_indptr, block_size=16)

    def test_wrong_block_size(self):
        a = torch.zeros(64, 64, dtype=torch.uint8, device="cuda")
        b = torch.zeros(2, 64, 64, dtype=torch.uint8, device="cuda")
        a_descale = torch.zeros(64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        b_descale = torch.zeros(2, 64, 8, dtype=torch.float8_e4m3fn, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_fp4(a, b, a_descale, b_descale, m_indptr, block_size=64)
