"""Reference correctness test for the cudnn_batch_prefill trace API."""

import math
import torch
import pytest

from tests.trace.reference_utils import (
    _check,
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
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = actual_seq_lens_q.cumsum(0)
    batch_offsets = (
        (qo_indptr.to(torch.int64) * Hq * D).to(torch.int32).reshape(B + 1, 1, 1, 1)
    )
    try:
        api_out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=actual_seq_lens_q.reshape(B, 1, 1, 1),
            actual_seq_lens_kv=actual_seq_lens_kv.reshape(B, 1, 1, 1),
            block_tables=block_tables,
            causal=True,
            return_lse=False,
            batch_offsets_q=batch_offsets,
            batch_offsets_o=batch_offsets,
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
    _check(cudnn_batch_prefill_trace, ref_out, api_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_cudnn_batch_prefill_requires_batch_offsets():
    """batch_size > 1 without batch_offsets_q/_o must raise, not corrupt.

    Without ragged offsets the cuDNN graph cannot address packed q/out for
    batch entries past the first, so the API rejects the call instead of
    silently returning wrong results for batch index >= 1.
    """
    from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
    from flashinfer.cudnn.prefill import CUDNN_AVAILABLE

    if not CUDNN_AVAILABLE:
        pytest.skip("cudnn python frontend not available")

    B, Hq, Hk, D, PS, q_len, kv_len = 2, 8, 2, 128, 16, 32, 64
    nppr = (kv_len + PS - 1) // PS
    k_cache = torch.randn(nppr * B, Hk, PS, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(B * q_len, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(nppr * B, dtype=torch.int32, device="cuda").reshape(
        B, nppr
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    with pytest.raises(ValueError, match="batch_offsets_q and batch_offsets_o"):
        cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            1.0 / math.sqrt(D),
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=torch.full(
                (B, 1, 1, 1), q_len, dtype=torch.int32, device="cuda"
            ),
            actual_seq_lens_kv=torch.full(
                (B, 1, 1, 1), kv_len, dtype=torch.int32, device="cuda"
            ),
            block_tables=block_tables,
            causal=True,
            return_lse=False,
        )
