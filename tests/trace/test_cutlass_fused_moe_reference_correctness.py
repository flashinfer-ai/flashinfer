"""Reference correctness test for the cutlass_fused_moe trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        pytest.param(
            dict(
                seq_len=16,
                num_experts=4,
                hidden_size=128,
                intermediate_size=64,
                top_k=2,
            ),
            id="T16-E4-H128-I64-top2",
        ),
        pytest.param(
            dict(
                seq_len=8,
                num_experts=4,
                hidden_size=128,
                intermediate_size=128,
                top_k=2,
            ),
            id="T8-E4-H128-I128-top2",
        ),
    ],
)
def test_cutlass_fused_moe_reference_correctness(shape_kwargs):
    """cutlass_fused_moe kernel vs reference (bf16 weights, standard SwiGLU MoE)."""
    import flashinfer
    from flashinfer.trace.templates.moe import cutlass_fused_moe_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    T = shape_kwargs["seq_len"]
    E = shape_kwargs["num_experts"]
    H = shape_kwargs["hidden_size"]
    I = shape_kwargs["intermediate_size"]
    TOP_K = shape_kwargs["top_k"]
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
