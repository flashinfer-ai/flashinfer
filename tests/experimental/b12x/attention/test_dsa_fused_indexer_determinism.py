"""Determinism tests for the fused DSA indexer top-k selection.

The fused decode indexer computes q·k scores internally and selects the top-k
through ``_fused_radix_select`` (single-CTA trim and serial last-CTA merge) or
the cooperative grid-barrier merge (ctas_per_group > 1, the 32K canary path).
Exact score ties (common with FP8-quantized indexer logits) previously
resolved through ``_smem_xadd`` atomics in a nondeterministic order, so the
selected read plan jittered across runs on an identical prompt. These tests
probe whether the fused selection is stable across repeated calls on input
with a total tie at the top-k boundary.
"""

import torch
import pytest

torch_device = "cuda"

_PS = 64  # compressed-index page size


def _build_tied_case(rows, heads, seqlen, *, device):
    """All query/K values equal with unit weights/scales -> every token score
    is identical, so the top-k boundary is a total tie.

    The values use the smallest fp8 e4m3 subnormal (2^-9) so the shared score
    is ~2^-5: small enough that the 2^-24 position tie-break perturb remains
    above the fp32 ulp and actually distinguishes tied candidates (large
    scores would round the perturb away, as in the unrealistic q=k=1 case).
    """
    pr = (seqlen + _PS - 1) // _PS
    npages = rows * pr
    v = 2.0**-9
    q_fp8 = (
        torch.full((rows, heads, 128), v, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    weights = torch.ones(rows, heads, dtype=torch.float32, device=device)
    k_fp8 = (
        torch.full((npages, _PS, 128), v, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    k_scales = torch.ones(npages, _PS, dtype=torch.float32, device=device)
    page_table = (
        torch.arange(npages, dtype=torch.int32, device=device)
        .view(rows, pr)
        .contiguous()
    )
    seqlens = torch.full((rows,), seqlen, dtype=torch.int32, device=device)
    return q_fp8, weights, k_fp8, k_scales, page_table, seqlens


def _run_fused_once(
    q_fp8,
    weights,
    k_fp8,
    k_scales,
    page_table,
    seqlens,
    heads,
    topk,
    *,
    ctas_per_group=None,
    merge_threshold=None,
):
    from b12x.attention.dsa_indexer.fused_indexer import run_fused_paged_indexer

    idx, _ = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        ctas_per_group=ctas_per_group,
        merge_threshold=merge_threshold,
    )
    torch.cuda.synchronize(torch_device)
    return idx


def _assert_stable(run, iters=20):
    sets = set()
    for _ in range(iters):
        idx = run()
        sets.add(tuple(sorted(idx[0].cpu().tolist())))
    assert len(sets) == 1, (
        f"fused indexer nondeterministic on ties: {len(sets)} distinct sets, "
        f"examples={list(sets)[:3]}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_indexer_coop_merge_tied_determinism():
    """Cooperative grid-barrier merge (the 32K canary path) must return identical
    index sets across calls on tied input."""
    rows, heads, topk, seqlen = 1, 64, 512, 32768
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_tied_case(
        rows, heads, seqlen, device=torch.device("cuda")
    )
    _assert_stable(
        lambda: _run_fused_once(
            q_fp8, weights, k_fp8, k_scales, page_table, seqlens, heads, topk
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_indexer_single_cta_tied_determinism():
    """Single-CTA _fused_radix_select trim must return identical index sets on ties."""
    rows, heads, topk, seqlen = 1, 16, 512, 8192
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_tied_case(
        rows, heads, seqlen, device=torch.device("cuda")
    )
    _assert_stable(
        lambda: _run_fused_once(
            q_fp8,
            weights,
            k_fp8,
            k_scales,
            page_table,
            seqlens,
            heads,
            topk,
            ctas_per_group=1,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_indexer_serial_merge_tied_determinism():
    """Serial last-CTA merge (_fused_radix_select over the packed slab) must return
    identical index sets on ties despite nondeterministic pack order."""
    rows, heads, topk, seqlen = 1, 16, 512, 8192
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_tied_case(
        rows, heads, seqlen, device=torch.device("cuda")
    )
    _assert_stable(
        lambda: _run_fused_once(
            q_fp8,
            weights,
            k_fp8,
            k_scales,
            page_table,
            seqlens,
            heads,
            topk,
            ctas_per_group=16,
            merge_threshold=1 << 30,  # force the serial last-CTA arm
        )
    )
