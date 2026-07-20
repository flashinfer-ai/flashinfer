# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""attention.paged: decode and extend parity vs the in-tree reference through
the eager public lifecycle (plan -> bind -> run), BF16 and FP8 KV — the same
call sequence vLLM's integration uses.

Note: b12x's paged CUDA-graph replay tests are stale against the current
decode-graph planner heuristics (they fail upstream too), so they were not
ported; graph coverage for the attention family comes from
test_compressed_mla.py until they are refreshed.
"""

from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.attention import paged
from flashinfer.experimental.sm12x.attention.paged.reference import (
    paged_attention_reference,
)

from .._reference.paged_attention_helpers import (
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)
from ..conftest import require_sm12x


def _run_eager(
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    *,
    mode: str,
    k_descale=None,
    v_descale=None,
):
    plan = paged.plan(
        paged.Caps(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=q.shape[0],
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=1024,
            max_partial_rows=16384,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=False,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = paged.bind(
        plan,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=int(q.shape[0]),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse = paged.run(binding=binding)
    return out, lse


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def test_run_decode_matches_reference() -> None:
    require_sm12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 128, 192, 256],
        page_size=64,
        seed=73,
    )
    out, _ = _run_eager(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="decode"
    )
    ref_out, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    torch.cuda.synchronize()
    assert (out - ref_out).abs().max().item() <= 0.02
    assert _cosine(out, ref_out) >= 0.99999


def test_run_decode_fp8_kv_matches_reference() -> None:
    require_sm12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 192, 256, 448],
        page_size=64,
        seed=91,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )
    out, _ = _run_eager(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref_out, _ = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    torch.cuda.synchronize()
    assert (out - ref_out).abs().max().item() <= 0.06
    assert _cosine(out, ref_out) >= 0.999


def test_run_extend_matches_reference() -> None:
    require_sm12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 16],
        cache_seqlens=[128, 192],
        page_size=64,
        seed=57,
    )
    out, _ = _run_eager(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="extend"
    )
    ref_out, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    torch.cuda.synchronize()
    assert (out - ref_out).abs().max().item() <= 0.02
    assert _cosine(out, ref_out) >= 0.99999
