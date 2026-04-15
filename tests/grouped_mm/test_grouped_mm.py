"""
Tests for flashinfer.grouped_mm (cuDNN MOE Grouped GEMM).
"""

import pytest
import torch

import torch.nn.functional as F

from flashinfer.grouped_mm import grouped_mm_bf16, grouped_mm_fp8, grouped_mm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability

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

requires_cudnn_moe_block_scale = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 92100,
    reason="cuDNN MOE block-scale MXFP8 requires backend >= 9.21.0",
)

_SM_MAJOR = None


def _get_sm_major():
    global _SM_MAJOR
    if _SM_MAJOR is None:
        _SM_MAJOR = get_compute_capability(torch.device("cuda"))[0]
    return _SM_MAJOR


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 10,
    reason="MXFP8 grouped GEMM requires SM100+",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def ref_grouped_mm(a, b, m_indptr, out_dtype, alpha=None):
    """Loop over experts, matmul with transposed weight (NT layout)."""
    num_experts = b.shape[0]
    n = b.shape[1]
    cum_m = a.shape[0]
    out = torch.zeros(cum_m, n, dtype=torch.float32, device=a.device)
    for e in range(num_experts):
        start = m_indptr[e].item()
        end = m_indptr[e + 1].item()
        if start < end:
            out[start:end] = a[start:end].float() @ b[e].float().T
    if alpha is not None:
        out = out * alpha.float()
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


class TestGroupedMmMxfp8:
    """Tests for grouped_mm_mxfp8 (block-scaled MXFP8 grouped GEMM)."""

    BLOCK_SIZE = 32
    MIN_COS_SIM = 0.9

    @staticmethod
    def _quantize_a(a_bf16):
        """Quantize 2D token activations (cum_m, k) → mxfp8 + scale."""
        cum_m, k = a_bf16.shape
        a_mxfp8, a_scale = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=True)
        a_scale = a_scale.reshape(-1, k // 32)
        return a_mxfp8, a_scale

    @staticmethod
    def _quantize_b(b_bf16):
        """Quantize 3D expert weights (num_experts, n, k) → mxfp8 + scale."""
        num_experts, n, k = b_bf16.shape
        b_mxfp8, b_scale = mxfp8_quantize(b_bf16, is_sf_swizzled_layout=True)
        b_scale = b_scale.reshape(num_experts, -1, k // 32)
        return b_mxfp8, b_scale

    @requires_cudnn_moe_block_scale
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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
    @requires_sm100
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


# ---------------------------------------------------------------------------
# MXFP8 Validation / error tests (do not require cuDNN)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FP8 Validation / error tests (do not require cuDNN)
# ---------------------------------------------------------------------------


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
    test.test_uniform_distribution(4, 128, 256, 256)
    test.test_non_uniform_distribution()
    test.test_empty_experts()
    test.test_single_expert()
    test.test_preallocated_output()
    test.test_out_dtype_fp32()
    test.test_graph_cache_reuse()
    print("All bf16 tests passed!")

    test_fp8 = TestGroupedMmFp8()
    test_fp8.test_uniform_distribution(4, 128, 256, 256, torch.float8_e4m3fn)
    test_fp8.test_non_uniform_distribution(torch.float8_e4m3fn)
    test_fp8.test_empty_experts()
    test_fp8.test_single_expert()
    test_fp8.test_preallocated_output()
    test_fp8.test_out_dtype(torch.bfloat16)
    test_fp8.test_alpha_scaling(0.5)
    test_fp8.test_no_alpha()
    test_fp8.test_graph_cache_reuse()
    test_fp8.test_mixed_fp8_dtypes()
    print("All fp8 tests passed!")

    test_mxfp8 = TestGroupedMmMxfp8()
    test_mxfp8.test_uniform_distribution(4, 128, 256, 256)
    test_mxfp8.test_non_uniform_distribution()
    test_mxfp8.test_empty_experts()
    test_mxfp8.test_single_expert()
    test_mxfp8.test_preallocated_output()
    test_mxfp8.test_out_dtype(torch.bfloat16)
    test_mxfp8.test_graph_cache_reuse()
    test_mxfp8.test_quantize_dtypes()
    print("All mxfp8 tests passed!")
