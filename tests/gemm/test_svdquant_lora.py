# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the fused SVDQuant NVFP4 + LoRA GEMM (Sm100BlockScaledLoRADenseGemmKernel).

Validates the host-glue contract (prepare_svdquant_state + mm_fp4_svdquant) on the three
Qwen-Image linear shapes. The kernel always fuses the LoRA up-projection, so the output is the
full SVDQuant result -- NVFP4 residual + (X@L2_merged^T)@L1^T -- checked against the eager reference
(the NVFP4 residual matches torch.ops.trtllm.nvfp4_gemm; the full op is ~50 dB SQNR vs eager).

The activation is quantized by FlashInfer's fused PQS quantizer and checked byte-for-byte against
TRT-LLM quantizing a materialized BF16 ``x * pqs``. The resulting payload and scale factors are
then injected into the fused GEMM. The full-shape test is skipped when SM100a or the TRT-LLM
reference ops are unavailable.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported


def _has_trtllm_fp4():
    try:
        import tensorrt_llm  # noqa: F401

        return (
            hasattr(torch.ops, "trtllm")
            and hasattr(torch.ops.trtllm, "fp4_quantize")
            and hasattr(torch.ops.trtllm, "nvfp4_gemm")
        )
    except Exception:
        return False


_SKIP_SM = not torch.cuda.is_available() or not is_sm100a_supported(
    torch.device("cuda")
)
_SKIP_REF = not _has_trtllm_fp4()


def _sqnr(ref: torch.Tensor, t: torch.Tensor) -> float:
    ref, t = ref.float(), t.float()
    noise = (ref - t).pow(2).mean()
    if noise == 0:
        return 99.0
    return float(10 * torch.log10(ref.pow(2).mean() / noise))


@pytest.mark.skipif(_SKIP_SM, reason="requires SM100a (Blackwell)")
@pytest.mark.skipif(
    _SKIP_REF, reason="requires torch.ops.trtllm.fp4_quantize/nvfp4_gemm reference"
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (2048, 3072, 3072),
        (2048, 12288, 3072),
        (2048, 3072, 12288),
        (257, 3072, 3072),  # exact symbolic M; no framework row padding
    ],
)
def test_svdquant_lora(M, N, K):
    from flashinfer.gemm import prepare_svdquant_state, mm_fp4_svdquant
    from flashinfer.quantization import nvfp4_quantize_cute_dsl

    dev = "cuda"
    r = 32
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    L1 = torch.randn(N, r, device=dev, dtype=torch.bfloat16) * 0.1  # up   [O, r]
    L2 = torch.randn(r, K, device=dev, dtype=torch.bfloat16) * 0.1  # down [r, I]
    pqs = (1.0 + 0.1 * torch.randn(K, device=dev, dtype=torch.bfloat16)).contiguous()
    x_hat = (x * pqs).to(torch.bfloat16)
    L2_merged = (L2 * pqs).to(torch.bfloat16)

    gsx = (448 * 6) / x_hat.float().abs().max()
    gsw = (448 * 6) / W.float().abs().max()
    ref_xq, ref_x_sf = torch.ops.trtllm.fp4_quantize(x_hat, gsx, 16, False)
    xq, x_sf = nvfp4_quantize_cute_dsl(x, gsx, pre_quant_scale=pqs)
    assert torch.equal(xq, ref_xq), "fused pqs changed the FP4 activation payload"
    x_sf_bytes = x_sf.view(torch.uint8).reshape(-1)
    ref_x_sf_bytes = ref_x_sf.view(torch.uint8).reshape(-1)
    assert torch.equal(x_sf_bytes, ref_x_sf_bytes), (
        "fused pqs changed activation scale factors"
    )
    wq, w_sf = torch.ops.trtllm.fp4_quantize(W, gsw, 16, False)
    alpha = torch.tensor([1.0 / (gsx * gsw)], device=dev, dtype=torch.float32)

    # Reference: NVFP4 residual (matches trtllm.nvfp4_gemm) + the bf16 SVDQuant LoRA correction.
    resid_ref = torch.ops.trtllm.nvfp4_gemm(
        xq, wq, x_sf, w_sf, alpha, torch.bfloat16
    ).float()
    ref = resid_ref + (x.float() @ L2_merged.float().t()) @ L1.float().t()

    state = prepare_svdquant_state(
        M,
        wq,
        w_sf,
        L1,
        L2_merged,
        1.0 / gsx,
        1.0 / gsw,
        r,
    )
    y = mm_fp4_svdquant(x, state, ext_xq=xq, ext_sf=x_sf).float()
    down_ref = x @ L2_merged.t()
    paper_down = x_hat @ L2.t()

    assert torch.isfinite(y).all(), "output has non-finite values"
    assert torch.equal(state["D_active"], down_ref), (
        "LoRA down projection must use raw x with caller-merged L2"
    )
    merge_sqnr = _sqnr(paper_down, down_ref)
    assert merge_sqnr > 40.0, (
        f"one-time BF16 L2 merge SQNR {merge_sqnr:.1f} dB below 40 dB"
    )
    sqnr = _sqnr(ref, y)
    assert sqnr > 40.0, f"SQNR {sqnr:.1f} dB below 40 dB (N={N}, K={K})"


@pytest.mark.skipif(_SKIP_SM, reason="requires SM100a (Blackwell)")
@pytest.mark.parametrize(
    "M,K",
    [
        (1024, 3072),  # aligned, non-TMA
        (16384, 3072),  # aligned, TMA
        (16385, 3072),  # misaligned SVDQuant dispatch avoids a padded TMA copy
        (4096, 12288),  # wide SVDQuant dispatch uses the faster non-TMA kernel
    ],
)
def test_nvfp4_quantize_fused_pre_quant_scale(M, K):
    """Fused pqs must reproduce quantizing a materialized BF16 x*pqs bit for bit.

    The shapes cover the swizzled non-TMA and TMA kernels plus both PQS-only
    performance overrides.
    """
    from flashinfer.quantization import nvfp4_quantize_cute_dsl

    torch.manual_seed(1)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    pqs = (1.0 + 0.1 * torch.randn(K, device="cuda", dtype=torch.bfloat16)).contiguous()
    x_hat = (x * pqs).to(torch.bfloat16)
    global_scale = ((448 * 6) / x_hat.float().abs().max()).reshape(1)

    ref_q, ref_sf = nvfp4_quantize_cute_dsl(x_hat, global_scale)
    q, sf = nvfp4_quantize_cute_dsl(
        x,
        global_scale,
        pre_quant_scale=pqs,
    )

    assert torch.equal(q, ref_q), f"FP4 payload differs for M={M}, K={K}"
    assert torch.equal(sf, ref_sf), f"FP8 scale factors differ for M={M}, K={K}"


@pytest.mark.skipif(_SKIP_SM, reason="requires SM100a (Blackwell)")
def test_compiled_cache_recomputes_runtime_sf_m():
    """A symbolic-M compiled kernel must not reuse the first batch's sf_m."""
    from flashinfer.gemm.kernels import svdquant_lora

    N, K, r = 3072, 3072, 32
    tactic = ((128, 128), (1, 1), False)
    key = (torch.cuda.current_device(), N, K, r, torch.bfloat16, tactic, None)
    old_cache = dict(svdquant_lora._COMPILED_KERNEL_CACHE)
    sentinel = object()
    try:
        svdquant_lora._COMPILED_KERNEL_CACHE.clear()
        svdquant_lora._COMPILED_KERNEL_CACHE[key] = (sentinel, 24, 48, 32)
        compiled, sf_m, sf_n, sf_k, lora_k = svdquant_lora._build_compiled(
            N, K, 257, r, tactic=tactic
        )
        assert compiled is sentinel
        assert (sf_m, sf_n, sf_k, lora_k) == (3, 24, 48, 32)
    finally:
        svdquant_lora._COMPILED_KERNEL_CACHE.clear()
        svdquant_lora._COMPILED_KERNEL_CACHE.update(old_cache)
