"""Reference correctness test for the trtllm_batch_decode_mla trace API."""

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
            dict(batch_size=4, num_heads=128, page_size=64, seq_len=128),
            id="B4-H128-PS64-S128",
        ),
        pytest.param(
            dict(batch_size=1, num_heads=64, page_size=128, seq_len=128),
            id="B1-H64-PS128-S128",
        ),
    ],
)
def test_trtllm_batch_decode_mla_reference_correctness(shape_kwargs):
    """trtllm_batch_decode_with_kv_cache_mla kernel vs reference (SM100/103)."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.trace.templates.attention import trtllm_batch_decode_mla_trace

    # TRT-LLM MLA kernel is only instantiated on SM100/SM103 (trtllm-gen).
    _skip_if_not_sm100_or_103()
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
    query = torch.randn(B, q_len, num_heads, D_qk, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(
        total_pages, page_size, D_qk, dtype=torch.float16, device="cuda"
    )
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, n_pages
    )
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    bmm1_scale = 1.0 / math.sqrt(D_qk)
    try:
        api_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            is_var_seq=False,
        )
    except Exception as exc:
        pytest.skip(f"trtllm_batch_decode_with_kv_cache_mla unavailable: {exc}")
    ref_out = trtllm_batch_decode_mla_trace.reference(
        query,
        kv_cache,
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
    # Matches tests/attention/test_cute_dsl_mla_decode.py element-wise tol.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
