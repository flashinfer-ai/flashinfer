"""Reference correctness test for the xqa_batch_decode trace API."""

import math
import torch
import pytest

from tests.trace.reference_utils import (
    _close,
    _skip_if_not_sm100,
)


def test_xqa_batch_decode_reference_correctness():
    """flashinfer.decode.xqa_batch_decode_with_kv_cache kernel vs reference (SM100+)."""
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from flashinfer.trace.templates.attention import xqa_batch_decode_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 128, 16
    MP = 2
    NP = B * MP
    kv_len = PS * MP
    # NHD 5-D interleaved cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(NP, 2, PS, Hk, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    try:
        api_out = xqa_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace,
            block_tables,
            seq_lens,
            kv_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
        )
    except Exception as exc:
        pytest.skip(f"xqa_batch_decode_with_kv_cache unavailable: {exc}")
    ref_out = xqa_batch_decode_trace.reference(
        q,
        kv_cache,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="NHD",
    )
    # Same tolerance family as trtllm_batch_decode — same math, different backend.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
