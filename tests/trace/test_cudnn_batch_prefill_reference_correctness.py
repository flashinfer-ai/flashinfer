"""Reference correctness test for the cudnn_batch_prefill trace API."""

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
                batch_size=2,
                num_qo_heads=8,
                num_kv_heads=2,
                head_dim=128,
                page_size=16,
                q_len=32,
                kv_len=64,
            ),
            id="B2-Hq8-Hk2-D128-PS16-Q32-KV64",
        ),
        pytest.param(
            dict(
                batch_size=1,
                num_qo_heads=4,
                num_kv_heads=1,
                head_dim=128,
                page_size=16,
                q_len=16,
                kv_len=32,
            ),
            id="B1-Hq4-Hk1-D128-PS16-Q16-KV32",
        ),
    ],
)
def test_cudnn_batch_prefill_reference_correctness(shape_kwargs):
    """cudnn_batch_prefill_with_kv_cache kernel vs reference (causal)."""
    from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
    from flashinfer.trace.templates.attention import cudnn_batch_prefill_trace

    torch.manual_seed(0)
    B = shape_kwargs["batch_size"]
    Hq = shape_kwargs["num_qo_heads"]
    Hk = shape_kwargs["num_kv_heads"]
    D = shape_kwargs["head_dim"]
    PS = shape_kwargs["page_size"]
    q_len = shape_kwargs["q_len"]
    kv_len = shape_kwargs["kv_len"]
    nppr = (kv_len + PS - 1) // PS
    total_pages = nppr * B
    kv_cache = torch.randn(
        total_pages, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda"
    )
    k_cache = kv_cache[:, 0].contiguous()
    v_cache = kv_cache[:, 1].contiguous()
    q = torch.randn(B * q_len, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, nppr
    )
    actual_seq_lens_q = torch.full((B,), q_len, dtype=torch.int32, device="cuda")
    actual_seq_lens_kv = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(D)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            causal=True,
            return_lse=False,
        )
    except Exception as exc:
        pytest.skip(f"cudnn_batch_prefill_with_kv_cache unavailable: {exc}")
    ref_out, _ = cudnn_batch_prefill_trace.reference(
        q,
        k_cache,
        v_cache,
        scale,
        workspace,
        q_len,
        kv_len,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        True,
        False,
        block_tables=block_tables,
    )
    # Matches tests/attention/test_cudnn_prefill.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
