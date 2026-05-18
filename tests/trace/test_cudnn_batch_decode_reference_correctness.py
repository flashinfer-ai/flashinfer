"""Reference correctness test for the cudnn_batch_decode trace API."""

import math
import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        pytest.param(
            dict(
                batch_size=4,
                num_qo_heads=8,
                num_kv_heads=2,
                head_dim=128,
                page_size=16,
                seq_len=64,
            ),
            id="B4-Hq8-Hk2-D128-PS16-S64",
        ),
        pytest.param(
            dict(
                batch_size=1,
                num_qo_heads=4,
                num_kv_heads=1,
                head_dim=128,
                page_size=16,
                seq_len=32,
            ),
            id="B1-Hq4-Hk1-D128-PS16-S32",
        ),
    ],
)
def test_cudnn_batch_decode_reference_correctness(shape_kwargs):
    """cudnn_batch_decode_with_kv_cache kernel vs reference (page-gather SDPA)."""
    import flashinfer
    from flashinfer.trace.templates.attention import cudnn_batch_decode_trace

    torch.manual_seed(0)
    B = shape_kwargs["batch_size"]
    Hq = shape_kwargs["num_qo_heads"]
    Hk = shape_kwargs["num_kv_heads"]
    D = shape_kwargs["head_dim"]
    PS = shape_kwargs["page_size"]
    s_kv = shape_kwargs["seq_len"]
    nppr = (s_kv + PS - 1) // PS  # num_pages_per_seq
    total_pages = nppr * B
    # cuDNN expects K/V as separate tensors in layout
    #   [num_pages, num_kv_heads, page_size, head_dim]
    kv_cache = torch.randn(
        total_pages, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda"
    )
    k_cache = kv_cache[:, 0, :, :, :].contiguous()
    v_cache = kv_cache[:, 1, :, :, :].contiguous()
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, nppr
    )
    actual_seq_lens_kv = torch.full(
        (B, 1, 1, 1), s_kv, dtype=torch.int32, device="cuda"
    )
    scale = 1.0 / math.sqrt(D)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_sequence_kv=s_kv,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
        )
    except Exception as exc:
        pytest.skip(f"cudnn_batch_decode_with_kv_cache unavailable: {exc}")
    ref_out = cudnn_batch_decode_trace.reference(
        q,
        k_cache,
        v_cache,
        scale,
        workspace,
        s_kv,
        block_tables=block_tables,
        actual_seq_lens_kv=actual_seq_lens_kv.flatten(),
    )
    # Matches tests/attention/test_cudnn_decode.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
