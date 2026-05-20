"""Reference correctness test for the xqa_batch_decode_mla trace API."""

import math
import torch
import pytest

from tests.trace.reference_utils import (
    _cc,
    _close_pass_ratio,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        pytest.param(
            dict(batch_size=2, num_heads=128, page_size=64, seq_len=128),
            id="B2-H128-PS64-S128",
        ),
        pytest.param(
            dict(batch_size=1, num_heads=128, page_size=32, seq_len=64),
            id="B1-H128-PS32-S64",
        ),
    ],
)
def test_xqa_batch_decode_mla_reference_correctness(shape_kwargs):
    """flashinfer.mla.xqa_batch_decode_with_kv_cache_mla kernel vs reference (SM120/121)."""
    from flashinfer.mla import xqa_batch_decode_with_kv_cache_mla
    from flashinfer.trace.templates.attention import xqa_batch_decode_mla_trace

    if _cc()[0] != 12:
        pytest.skip("XQA MLA kernel only supports SM120/121")
    torch.manual_seed(0)
    B = shape_kwargs["batch_size"]
    num_heads = shape_kwargs["num_heads"]
    kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim = 512, 64, 512
    D_qk = kv_lora_rank + qk_rope_head_dim  # 576
    q_len = 1
    page_size = shape_kwargs["page_size"]
    seq_len = shape_kwargs["seq_len"]
    n_pages = (seq_len + page_size - 1) // page_size
    total_pages = n_pages * B
    query_fp32 = (
        torch.randn(B, q_len, num_heads, D_qk, dtype=torch.float32, device="cuda") / 4.0
    )
    kv_fp32 = (
        torch.randn(total_pages, page_size, D_qk, dtype=torch.float32, device="cuda")
        / 4.0
    )
    query_fp8 = query_fp32.to(torch.float8_e4m3fn)
    kv_fp8 = kv_fp32.to(torch.float8_e4m3fn)
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, n_pages
    )
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    bmm1_scale = 1.0 / math.sqrt(D_qk)
    try:
        api_out = xqa_batch_decode_with_kv_cache_mla(
            query=query_fp8,
            kv_cache=kv_fp8,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
        )
    except Exception as exc:
        pytest.skip(f"xqa_batch_decode_with_kv_cache_mla unavailable: {exc}")
    ref_out = xqa_batch_decode_mla_trace.reference(
        query_fp32,
        kv_fp32,
        workspace,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
    )
    # Matches tests/attention/test_xqa.py pass-ratio (>=95% for FP8 MLA).
    _close_pass_ratio(
        api_out.float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.95,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
