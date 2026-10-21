"""Paged extend (chunked prefill) on MiMo-V2 DiffKV shapes with an E4M3 KV cache."""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.paged.reference import paged_attention_reference
from ..conftest import require_b12x
from b12x.testing.reference.paged_attention_helpers import (
    quantize_paged_kv_cache_e4m3,
)

from .test_attention_paged_forward import _cosine_similarity, _make_workspace
from .test_attention_paged_planner import _make_inputs

# name: (q_heads, kv_heads, window_left, sinks) per TP4 rank; QK 192, V 128.
FAMILIES = {
    "global": (16, 1, -1, False),
    "sliding": (16, 2, 127, True),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@torch.inference_mode()
def test_paged_extend_mimo_diffkv_fp8_matches_reference(family: str) -> None:
    require_b12x()
    q_heads, kv_heads, window_left, sinks = FAMILIES[family]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[37, 64, 5],
        cache_seqlens=[37 + 128, 64 + 640, 5 + 3000],
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim_qk=192,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )
    sink_bias = (
        torch.linspace(-0.2, 0.2, q_heads, dtype=torch.float32, device=q.device)
        if sinks
        else None
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty(q.shape[0], q_heads, 128, dtype=q.dtype, device=q.device),
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=sink_bias,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=window_left,
        attention_sink_bias=sink_bias,
    )
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_base2 * math.log(2.0) - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.9999
