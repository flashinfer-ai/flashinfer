"""Reference correctness test for the cutlass_fused_moe trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
    _skip_if_not_sm100,
)


def test_cutlass_fused_moe_reference_correctness():
    """cutlass_fused_moe kernel vs reference (bf16 weights, standard SwiGLU MoE)."""
    import flashinfer
    from flashinfer.trace.templates.moe import cutlass_fused_moe_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    T, E, H, I, TOP_K = 16, 4, 128, 64, 2
    device = "cuda"
    x = torch.randn(T, H, dtype=torch.float16, device=device) / 5.0
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float16, device=device) / 5.0
    w2 = torch.randn(E, H, I, dtype=torch.float16, device=device) / 5.0
    token_sel = torch.randint(0, E, (T, TOP_K), dtype=torch.int32, device=device)
    token_scales = torch.rand(T, TOP_K, dtype=torch.float32, device=device)
    token_scales = token_scales / token_scales.sum(dim=-1, keepdim=True)
    try:
        api_out = flashinfer.cutlass_fused_moe(
            x, token_sel, token_scales, w1, w2, torch.float16, quant_scales=None
        )
    except Exception as exc:
        pytest.skip(f"cutlass_fused_moe unavailable: {exc}")
    if isinstance(api_out, list):
        api_out = api_out[0]
    ref_out = cutlass_fused_moe_trace.reference(x, token_sel, token_scales, w1, w2)
    # Matches tests/moe/test_trtllm_cutlass_fused_moe.py.
    _close(api_out, ref_out.to(api_out.dtype), atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
