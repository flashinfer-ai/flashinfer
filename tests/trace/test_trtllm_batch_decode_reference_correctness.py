"""Reference correctness test for the trtllm_batch_decode trace API."""

import math
import torch
import pytest

from tests.trace.reference_utils import (
    _close,
    _skip_if_not_sm100_or_103,
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
                max_pages_per_seq=2,
            ),
            id="B2-Hq8-Hk2-D128-PS16-MP2",
        ),
        pytest.param(
            dict(
                batch_size=1,
                num_qo_heads=8,
                num_kv_heads=2,
                head_dim=128,
                page_size=16,
                max_pages_per_seq=3,
            ),
            id="B1-Hq8-Hk2-D128-PS16-MP3",
        ),
    ],
)
def test_trtllm_batch_decode_reference_correctness(shape_kwargs):
    """trtllm_batch_decode kernel vs reference (paged HND decode, SM100/103)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.trace.templates.attention import trtllm_batch_decode_trace

    # TllmGenFmhaRunner is only instantiated for SM100/SM103; on SM12x the
    # kernel raises "Unsupported architecture" at runtime.
    _skip_if_not_sm100_or_103()
    torch.manual_seed(0)
    B = shape_kwargs["batch_size"]
    Hq = shape_kwargs["num_qo_heads"]
    Hk = shape_kwargs["num_kv_heads"]
    D = shape_kwargs["head_dim"]
    PS = shape_kwargs["page_size"]
    MP = shape_kwargs["max_pages_per_seq"]  # pages per seq
    NP = B * MP
    kv_len = PS * MP
    # HND layout for the kernel: [num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_cache_hnd = torch.randn(NP, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    api_out = trtllm_batch_decode_with_kv_cache(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="HND",
    )
    ref_out = trtllm_batch_decode_trace.reference(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="HND",
    )
    # Matches tests/attention/test_cudnn_decode.py / trtllm_gen bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
