# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the fused SVDQuant NVFP4 + LoRA GEMM (Sm100BlockScaledLoRADenseGemmKernel).

Validates the host-glue contract (prepare_svdquant_state + mm_fp4_svdquant) on the three
Qwen-Image linear shapes. The kernel always fuses the LoRA up-projection, so the output is the
full SVDQuant result -- NVFP4 residual + (X@L2^T)@L1^T -- checked against the eager reference
(the NVFP4 residual matches torch.ops.trtllm.nvfp4_gemm; the full op is ~50 dB SQNR vs eager).

The activation is quantized with the SAME quantizer the deployment injects
(torch.ops.trtllm.fp4_quantize), so the test exercises the real injection path. It is skipped
when SM100a or the TRT-LLM reference op are unavailable.
"""
import pytest
import torch

from flashinfer.utils import is_sm100a_supported


def _has_trtllm_fp4():
    try:
        import tensorrt_llm  # noqa: F401
        return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")
    except Exception:
        return False


_SKIP_SM = not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda"))
_SKIP_REF = not _has_trtllm_fp4()


def _sqnr(ref: torch.Tensor, t: torch.Tensor) -> float:
    ref, t = ref.float(), t.float()
    noise = (ref - t).pow(2).mean()
    if noise == 0:
        return 99.0
    return float(10 * torch.log10(ref.pow(2).mean() / noise))


@pytest.mark.skipif(_SKIP_SM, reason="requires SM100a (Blackwell)")
@pytest.mark.skipif(_SKIP_REF, reason="requires torch.ops.trtllm.fp4_quantize/nvfp4_gemm reference")
@pytest.mark.parametrize("N,K", [(3072, 3072), (12288, 3072), (3072, 12288)])
def test_svdquant_lora(N, K):
    from flashinfer.gemm import prepare_svdquant_state, mm_fp4_svdquant

    dev = "cuda"
    M, r = 2048, 32
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    L1 = torch.randn(N, r, device=dev, dtype=torch.bfloat16) * 0.1  # up   [O, r]
    L2 = torch.randn(r, K, device=dev, dtype=torch.bfloat16) * 0.1  # down [r, I]

    gsx = (448 * 6) / x.float().abs().max()
    gsw = (448 * 6) / W.float().abs().max()
    xq, x_sf = torch.ops.trtllm.fp4_quantize(x, gsx, 16, False)
    wq, w_sf = torch.ops.trtllm.fp4_quantize(W, gsw, 16, False)
    alpha = torch.tensor([1.0 / (gsx * gsw)], device=dev, dtype=torch.float32)

    # Reference: NVFP4 residual (matches trtllm.nvfp4_gemm) + the bf16 SVDQuant LoRA correction.
    resid_ref = torch.ops.trtllm.nvfp4_gemm(xq, wq, x_sf, w_sf, alpha, torch.bfloat16).float()
    ref = resid_ref + (x.float() @ L2.float().t()) @ L1.float().t()

    state = prepare_svdquant_state(
        M, wq, w_sf, L1, L2, 1.0 / gsx, 1.0 / gsw, r, pre_quant_scale=None,
    )
    y = mm_fp4_svdquant(x, state, ext_xq=xq, ext_sf=x_sf).float()

    assert torch.isfinite(y).all(), "output has non-finite values"
    sqnr = _sqnr(ref, y)
    assert sqnr > 40.0, f"SQNR {sqnr:.1f} dB below 40 dB (N={N}, K={K})"
