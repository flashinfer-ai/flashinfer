"""Parity tests for the unified fused indexer kernel (score + top-k, paged path).

The fused kernel scores q·k via the paged mxfp8 MMA and selects the row-wise
top-k in one launch (no global logits blob). Golden = the exact scorer math
(e4m3 q·k → ReLU·weight → sum heads → ×k_scale) followed by torch.topk, which
is the same contract the production scorer + tiled_topk path satisfies.
"""

from __future__ import annotations

import pytest
import torch
from b12x.preparation import FrozenMapping

from b12x.attention.dsa_indexer.fused_indexer import (
    KV_LAYOUT_CONTIGUOUS_MLA,
    KV_LAYOUT_PAGED,
    DSAFusedIndexerKernel,
    _BATCH_SLACK,
    _COOP_STATE_WORDS,
    _DSV4_BATCH_SLACK,
    _resolve_fused_batch_slack,
    fused_indexer_decode_warmup_rows,
    fused_indexer_scratch_max_rows,
    fused_indexer_scratch_capacity,
    resolve_fused_indexer_path,
    run_fused_paged_indexer,
    run_fused_indexer_mla,
)

_PS = 64  # compressed-index page size


def test_fused_indexer_warmup_rows_cover_glm_policies(monkeypatch):
    props = type("Props", (), {"multi_processor_count": 188})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)

    rows = fused_indexer_decode_warmup_rows(
        topk=2048,
        num_heads=32,
        max_pages=9114,
        device=torch.device("cuda"),
    )

    assert rows == tuple(range(1, 17))


def test_fused_indexer_warmup_rows_deduplicate_policies(monkeypatch):
    props = type("Props", (), {"multi_processor_count": 4})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)

    rows = fused_indexer_decode_warmup_rows(
        topk=2048,
        num_heads=32,
        max_pages=9114,
        device=torch.device("cuda"),
    )

    assert rows == (1, 2, 3)


def test_fused_indexer_warmup_rows_cover_sm120_c4_policies(monkeypatch):
    props = type(
        "Props",
        (),
        {"major": 12, "minor": 0, "multi_processor_count": 188},
    )()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)

    rows = fused_indexer_decode_warmup_rows(
        topk=512,
        num_heads=64,
        max_pages=1024,
        device=torch.device("cuda"),
    )

    assert rows == tuple(range(1, 17))


def test_fused_indexer_warmup_rows_cover_glm53_flash_c4_policies(monkeypatch):
    props = type(
        "Props",
        (),
        {"major": 12, "minor": 0, "multi_processor_count": 188},
    )()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)

    rows = fused_indexer_decode_warmup_rows(
        topk=512,
        num_heads=32,
        max_pages=1024,
        device=torch.device("cuda"),
    )

    assert rows == tuple(range(1, 17))


def test_coop_merge_co_residency_guard():
    """A group that oversubscribes the SMs must drop the coop grid-barrier arm.

    The cooperative merge spins in a grid barrier until all ctas_per_group CTAs of a
    group arrive, but the kernel runs one CTA per SM, so ctas_per_group > num_sms can
    never be fully co-resident and deadlocks. The build must fall back to the serial
    last-CTA arm. This host-side check is the only executable regression for the
    deadlock -- the hang itself wedges the GPU and cannot be exercised directly.
    """

    def build(ctas, num_sms):
        return DSAFusedIndexerKernel(
            num_heads_static=16,
            topk=2048,
            kv_layout=KV_LAYOUT_PAGED,
            ctas_per_group=ctas,
            num_sms=num_sms,
        )

    # ctas_per_group > num_sms -> serial fallback (coop disabled).
    over = build(49, 48)
    assert over.coop_co_resident is False
    assert over.coop_merge_possible is False
    # A full single wave (==) stays cooperative -- the shipped rows=1 decode config.
    edge = build(48, 48)
    assert edge.coop_co_resident is True
    assert edge.coop_merge_possible is True
    # Comfortably under the SM count stays cooperative.
    assert build(24, 48).coop_merge_possible is True
    # num_sms unknown (0) applies no guard: the ctas=1 FLAT path / direct construction.
    assert build(188, 0).coop_co_resident is True


def test_paged_dsv4_always_uses_large_fused_carry():
    assert (
        _resolve_fused_batch_slack(
            kv_layout=KV_LAYOUT_PAGED,
            num_heads=64,
            topk=512,
        )
        == _DSV4_BATCH_SLACK
    )


@pytest.mark.parametrize(
    "kv_layout,num_heads,topk",
    [
        (KV_LAYOUT_CONTIGUOUS_MLA, 64, 512),
        (KV_LAYOUT_PAGED, 32, 2048),
        (KV_LAYOUT_PAGED, 16, 512),
    ],
)
def test_large_fused_carry_is_paged_dsv4_specific(kv_layout, num_heads, topk):
    assert (
        _resolve_fused_batch_slack(
            kv_layout=kv_layout,
            num_heads=num_heads,
            topk=topk,
        )
        == _BATCH_SLACK
    )


@pytest.mark.parametrize("width", [8192, 16384, 32768, 131072])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_fused_indexer_route_covers_glm_decode_buckets(width, rows):
    assert resolve_fused_indexer_path(
        topk=2048,
        num_rows=rows,
        width=width,
        num_heads=32,
    )


@pytest.mark.parametrize("rows", [32, 64])
def test_fused_indexer_route_stops_after_glm_decode_buckets(rows):
    # Rows >= 32 belong to the streamed tiled route: measured wins at every
    # capacity bucket (e.g. r64@131K 1043us tiled vs 1541us fused).
    assert not resolve_fused_indexer_path(
        topk=2048,
        num_rows=rows,
        width=16384,
        num_heads=32,
    )


def test_fused_indexer_route_keeps_generic_head_count_limit():
    assert not resolve_fused_indexer_path(
        topk=2048,
        num_rows=8,
        width=16384,
        num_heads=64,
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64])
def test_fused_indexer_route_requires_c4_blackwell_capability(rows):
    # C4 routing is hardware-specific. An unknown/CPU capability keeps the
    # streamed tiled route instead of assuming either Blackwell policy.
    assert not resolve_fused_indexer_path(
        topk=512,
        num_rows=rows,
        width=4160 * 64,
        num_heads=64,
    )


def test_fused_indexer_route_stops_after_c4_decode_buckets():
    assert not resolve_fused_indexer_path(
        topk=512,
        num_rows=65,
        width=4160 * 64,
        num_heads=64,
    )


@pytest.mark.parametrize("compute_capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_fused_indexer_route_covers_sm12x_c4_decode_buckets(compute_capability, rows):
    assert resolve_fused_indexer_path(
        topk=512,
        num_rows=rows,
        width=1024 * 64,
        num_heads=64,
        compute_capability=compute_capability,
    )


@pytest.mark.parametrize("compute_capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_fused_indexer_route_covers_glm53_flash_c4_decode_buckets(
    compute_capability, rows
):
    assert resolve_fused_indexer_path(
        topk=512,
        num_rows=rows,
        width=1024 * 64,
        num_heads=32,
        compute_capability=compute_capability,
    )


@pytest.mark.parametrize("compute_capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("rows", [17, 32])
def test_fused_indexer_route_stops_after_sm12x_c4_decode_buckets(
    compute_capability, rows
):
    assert not resolve_fused_indexer_path(
        topk=512,
        num_rows=rows,
        width=1024 * 64,
        num_heads=64,
        compute_capability=compute_capability,
    )


@pytest.mark.parametrize("compute_capability", [(12, 0), (12, 1)])
def test_fused_indexer_scratch_rows_cover_sm12x_c4_policy(compute_capability):
    assert (
        fused_indexer_scratch_max_rows(
            topk=512,
            num_heads=64,
            compute_capability=compute_capability,
        )
        == 16
    )


def _build_case(rows, heads, seqlen, topk, *, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pr = (seqlen + _PS - 1) // _PS
    npages = rows * pr
    q_fp8 = (
        (torch.randn((rows, heads, 128), generator=g) / 3)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    weights = torch.randn((rows, heads), generator=g, dtype=torch.float32).to(device)
    k_fp8 = (
        (torch.randn((npages, _PS, 128), generator=g) / 3)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    k_scales = (
        torch.rand((npages, _PS), generator=g, dtype=torch.float32).to(device) + 0.1
    )
    page_table = (
        torch.arange(npages, dtype=torch.int32, device=device)
        .view(rows, pr)
        .contiguous()
    )
    seqlens = torch.full((rows,), seqlen, dtype=torch.int32, device=device)
    return q_fp8, weights, k_fp8, k_scales, page_table, seqlens


def _golden_topk(
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk, return_scores=False
):
    rows, seqlen = int(q_fp8.shape[0]), int(seqlens[0])
    qf, kf = q_fp8.float(), k_fp8.float()
    vals, idxs, scores = [], [], []
    for r in range(rows):
        pages = page_table[r].long()
        kr = kf[pages].reshape(-1, 128)[:seqlen]
        sc = k_scales[pages].reshape(-1)[:seqlen]
        logit = (
            torch.relu(torch.einsum("hd,td->ht", qf[r], kr)) * weights[r].unsqueeze(1)
        ).sum(0) * sc
        tk = torch.topk(logit, topk, largest=True, sorted=True)
        vals.append(tk.values)
        idxs.append(set(tk.indices.tolist()))
        scores.append(logit)
    if return_scores:
        return torch.stack(vals), idxs, torch.stack(scores)
    return torch.stack(vals), idxs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("rows,seqlen", [(1, 4096), (2, 4096), (4, 8192)])
def test_fused_indexer_paged_matches_reference(topk, rows, seqlen, monkeypatch):
    if seqlen <= topk:
        seqlen = topk * 2  # ensure the radix path (not just no-selection copy) engages
    device = torch.device("cuda")
    heads = 16
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=7, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )

    assert idx.shape == (rows, topk)
    assert bool((idx >= 0).all())  # every slot filled (seqlen > topk)
    # value-multiset parity (tie-robust); fp8 e4m3 dot in f32 matches the MMA.
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    # exact selected-index set per row (random fp8 logits have no rank-k ties here)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_paged_emits_native_physical_slots():
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 1024
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=71, device=device
    )
    page_table = torch.flip(page_table, dims=(1,)).contiguous()
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        output_physical_slots=True,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    assert torch.allclose(
        torch.sort(val, dim=1, descending=True).values,
        gold_vals,
        atol=1e-2,
        rtol=0,
    )
    for row in range(rows):
        logical = torch.tensor(
            sorted(gold_idx_sets[row]), dtype=torch.int64, device=device
        )
        page_ids = page_table[row, torch.div(logical, _PS, rounding_mode="floor")]
        physical = page_ids * _PS + torch.remainder(logical, _PS)
        assert set(idx[row].tolist()) == set(physical.tolist())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
@pytest.mark.parametrize("seqlen", [4101, 8255, 5377])
def test_fused_indexer_paged_partial_last_page(seqlen):
    # seqlen % 64 != 0 -> the last page is partial (valid_slots < 64). Exercises
    # the masked-tail path (no partial-logits pre-zero must still be exact).
    device = torch.device("cuda")
    rows, heads, topk = 2, 16, 2048
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=23, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_paged_short_context_no_radix():
    # seqlen <= topk: every token is selected (the no-selection fold path).
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 64
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=11, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    for r in range(rows):
        valid = idx[r][idx[r] >= 0]
        assert valid.numel() == seqlen
        assert set(valid.tolist()) == set(range(seqlen))
        assert int((idx[r] == -1).sum()) == topk - seqlen


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_paged_strided_q_scalar_load_fallback():
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 1024
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=17, device=device
    )
    padded_q = torch.empty((rows, heads, 129), dtype=torch.float8_e4m3fn, device=device)
    padded_q[..., :128].copy_(q_fp8)
    strided_q = padded_q[..., :128]
    assert strided_q.stride(1) == 129

    idx, val = run_fused_paged_indexer(
        q_bytes=strided_q.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        strided_q, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for row in range(rows):
        assert set(idx[row].tolist()) == gold_idx_sets[row]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
@pytest.mark.parametrize("ctas_per_group", [96, 188])
def test_fused_indexer_paged_long_context_merge_exact(ctas_per_group):
    # Long context + many CTAs stresses the in-kernel last-CTA merge: each CTA's
    # local top-k is a cluster of high values, so the merged radix threshold bin
    # is the worst case for the bounded SMEM candidate buffer. Must stay exact.
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 2048, 65536
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=3, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        ctas_per_group=ctas_per_group,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_preinitialized_state_graph_replay_switches_merge_mode():
    """One initialized workspace remains exact across serial/coop graph replays."""
    device = torch.device("cuda")
    rows, heads, topk, max_seqlen = 2, 32, 512, 4097
    ctas_per_group = 8
    merge_threshold = 3000
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, max_seqlen, topk, seed=41, device=device
    )
    pack_capacity, _ = fused_indexer_scratch_capacity(
        rows, topk, torch.cuda.get_device_properties(device).multi_processor_count
    )
    pack_values = torch.empty(pack_capacity, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_capacity, dtype=torch.int32, device=device)
    merge_state = torch.zeros(
        rows * _COOP_STATE_WORDS, dtype=torch.int32, device=device
    )
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)

    kwargs = dict(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        out_indices=out_indices,
        out_values=out_values,
        ctas_per_group=ctas_per_group,
        merge_threshold=merge_threshold,
        pack_values=pack_values,
        pack_indices=pack_indices,
        merge_state=merge_state,
        merge_state_preinitialized=True,
    )
    # Compile and exercise one launch before capture. The kernel must restore
    # its counters; no memset is part of the captured graph below.
    run_fused_paged_indexer(**kwargs)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused_paged_indexer(**kwargs)

    # Alternate across the runtime dispatch boundary and repeat both arms. This
    # catches stale arrival/output/total/cleanup counters on graph replay.
    for live_seqlen in (2048, max_seqlen, max_seqlen, 2048, max_seqlen):
        seqlens.fill_(live_seqlen)
        graph.replay()
        torch.cuda.synchronize(device)
        gold_values, gold_idx_sets = _golden_topk(
            q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
        )
        fused_sorted = torch.sort(out_values, dim=1, descending=True).values
        assert torch.allclose(fused_sorted, gold_values, atol=1e-2, rtol=0)
        for row in range(rows):
            assert set(out_indices[row].tolist()) == gold_idx_sets[row]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_fused_indexer_mla_matches_reference(rows):
    # FLAT/MLA: contiguous K, per-row [k_start, k_end) windows -> absolute indices.
    device = torch.device("cuda")
    heads, topk, krows = 16, 512, 8192
    g = torch.Generator(device="cpu").manual_seed(5)
    q_fp8 = (
        (torch.randn((rows, heads, 128), generator=g) / 3)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    weights = torch.randn((rows, heads), generator=g, dtype=torch.float32).to(device)
    k_fp8 = (
        (torch.randn((krows, 128), generator=g) / 3).to(torch.float8_e4m3fn).to(device)
    )
    k_scales = torch.rand((krows,), generator=g, dtype=torch.float32).to(device) + 0.1
    k_start = torch.zeros((rows,), dtype=torch.int32, device=device)
    k_end = torch.tensor(
        [min(krows, topk + (i + 1) * 512) for i in range(rows)],
        dtype=torch.int32,
        device=device,
    )
    idx, val = run_fused_indexer_mla(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        k_start=k_start,
        k_end=k_end,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    qf, kf = q_fp8.float(), k_fp8.float()
    for r in range(rows):
        a, b = int(k_start[r]), int(k_end[r])
        logit = (
            torch.relu(torch.einsum("hd,td->ht", qf[r], kf[a:b]))
            * weights[r].unsqueeze(1)
        ).sum(0) * k_scales[a:b]
        gset = set((torch.topk(logit, topk).indices + a).tolist())  # absolute index
        assert set(idx[r].tolist()) == gset


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_paged_direct_k_high_page_id_i64_offsets():
    """Direct-K addressing stays exact after the packed-page Int32 boundary."""
    device = torch.device("cuda")
    record_bytes = 1_077_120
    pid_lo, pages_used = 2000, 64
    # Keep this case above the Int32 byte-offset boundary. If either constant
    # is retuned, a green test must not silently stop exercising i64 offsets.
    assert pid_lo * record_bytes >= (1 << 31), (
        f"page base offset {pid_lo * record_bytes} no longer crosses the Int32 boundary"
    )
    pool_pages = pid_lo + pages_used + 4
    need = pool_pages * record_bytes + (1 << 29)
    free, _ = torch.cuda.mem_get_info(device)
    if free < need:
        pytest.skip(f"needs ~{need / 2**30:.1f} GiB free device memory")

    rows, heads, topk = 4, 64, 512
    seqlen = pages_used * _PS
    g = torch.Generator(device="cpu").manual_seed(7)
    pool = torch.zeros(pool_pages * record_bytes, dtype=torch.uint8, device=device)
    k_quant = pool.as_strided((pool_pages, _PS, 128), (record_bytes, 128, 1))
    k_scales = pool.view(torch.float32).as_strided(
        (pool_pages, _PS), (record_bytes // 4, 1), storage_offset=2048
    )
    k_fp8 = (torch.randn((pages_used, _PS, 128), generator=g) / 3).to(
        torch.float8_e4m3fn
    )
    k_quant[pid_lo : pid_lo + pages_used] = k_fp8.view(torch.uint8).to(device)
    k_scales[pid_lo : pid_lo + pages_used] = (
        torch.rand((pages_used, _PS), generator=g) + 0.1
    ).to(device)

    q_fp8 = (
        (torch.randn((rows, heads, 128), generator=g) / 3)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    weights = torch.randn((rows, heads), generator=g, dtype=torch.float32).to(device)
    page_table = (
        torch.arange(pid_lo, pid_lo + pages_used, dtype=torch.int32, device=device)
        .repeat(rows, 1)
        .contiguous()
    )
    seqlens = torch.full((rows,), seqlen, dtype=torch.int32, device=device)

    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_quant,
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        ctas_per_group=8,
    )
    torch.cuda.synchronize(device)

    gold_vals, gold_idx_sets, gold_scores = _golden_topk(
        q_fp8,
        weights,
        k_quant.view(torch.float8_e4m3fn)[pid_lo : pid_lo + pages_used],
        k_scales[pid_lo : pid_lo + pages_used],
        page_table - pid_lo,
        seqlens,
        topk,
        return_scores=True,
    )
    assert torch.allclose(
        torch.sort(val, dim=1, descending=True).values,
        gold_vals,
        atol=1e-2,
        rtol=0,
    )
    for row in range(rows):
        assert set(idx[row].tolist()) == gold_idx_sets[row]
        # Value-to-index pairing. Comparing the sorted value list and the index set
        # separately would still pass if correct values were emitted against the
        # wrong indices, so gather the golden score for each index the kernel
        # actually returned and compare position by position. Gathering by the
        # returned index (rather than asserting an index order) keeps this robust
        # to ties at the top-k boundary.
        paired = gold_scores[row].index_select(0, idx[row].long())
        assert torch.allclose(val[row].float(), paired.float(), atol=1e-2, rtol=0)


# --- fused_merge policy knob -------------------------------------------------


def test_dsa_indexer_config_and_merge_thresholds() -> None:
    from b12x.attention.dsa_indexer._tuning import (
        DsaIndexerConfig,
        FUSED_MERGE_AUTO,
        FUSED_MERGE_COOPERATIVE,
        FUSED_MERGE_SERIAL,
        TUNING,
    )
    from b12x.attention.dsa_indexer.fused_indexer import (
        _FORCE_LAST_CTA,
        _resolve_default_merge_threshold,
        resolve_fused_merge_threshold,
    )

    config = DsaIndexerConfig.from_config(
        FrozenMapping({"backend": "native", "fused_merge": FUSED_MERGE_AUTO})
    )
    assert TUNING.encode_config(config) == config.to_dict()
    common = dict(ctas_per_group=47, num_heads=32, topk=2048)
    assert resolve_fused_merge_threshold(FUSED_MERGE_COOPERATIVE, **common) == 0
    assert resolve_fused_merge_threshold(FUSED_MERGE_SERIAL, **common) == _FORCE_LAST_CTA
    assert resolve_fused_merge_threshold(
        FUSED_MERGE_AUTO, **common
    ) == _resolve_default_merge_threshold(**common)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for indexer planning"
)
def test_plan_indexer_scratch_applies_prepared_fused_merge() -> None:
    from b12x.attention.dsa_indexer._tuning import (
        FUSED_MERGE_AUTO,
        FUSED_MERGE_COOPERATIVE,
        FUSED_MERGE_SERIAL,
    )
    from b12x.attention.dsa_indexer.fused_indexer import _FORCE_LAST_CTA
    from b12x.attention.dsa_indexer.scratch import (
        INDEXER_PAGED_ROUTE_FUSED,
        B12XIndexerScratchCaps,
        plan_indexer_scratch,
    )

    caps = B12XIndexerScratchCaps(
        device=torch.device("cuda"), source_layout="paged", num_q_heads=32,
        max_q_rows=4, topk=2048, max_page_table_width=512,
        route=INDEXER_PAGED_ROUTE_FUSED,
    )
    thresholds = {
        choice: int(plan_indexer_scratch(caps, fused_merge=choice).inner.layout.fused_merge_threshold)
        for choice in (FUSED_MERGE_COOPERATIVE, FUSED_MERGE_SERIAL)
    }
    assert thresholds[FUSED_MERGE_COOPERATIVE] == 0
    assert thresholds[FUSED_MERGE_SERIAL] == _FORCE_LAST_CTA
    assert int(plan_indexer_scratch(caps, fused_merge=FUSED_MERGE_AUTO).inner.layout.fused_merge_threshold) == 0

# --- cooperative merge across repeated launches ------------------------------

_SLOT_SENTINEL = 0x5A5A5A5A


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
@pytest.mark.parametrize(
    "rows,heads,topk,seqlen",
    [(2, 32, 2048, 4096), (3, 64, 512, 8192), (4, 32, 512, 6000)],
)
def test_fused_indexer_cooperative_merge_repeated_launches_cover_every_slot(
    rows, heads, topk, seqlen
):
    """Every output slot is written and the group state is restored per launch."""
    device = torch.device("cuda")
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=77 + rows, device=device
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    pack_capacity, state_words = fused_indexer_scratch_capacity(rows, topk, num_sms)
    pack_values = torch.empty(pack_capacity, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_capacity, dtype=torch.int32, device=device)
    merge_state = torch.zeros(state_words, dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    gold_values, gold_index_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    for _launch in range(4):
        out_indices.fill_(_SLOT_SENTINEL)
        run_fused_paged_indexer(
            q_bytes=q_fp8.view(torch.uint8),
            weights=weights,
            k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
            k_scales=k_scales,
            real_page_table=page_table,
            seqlens=seqlens,
            num_heads=heads,
            topk=topk,
            out_indices=out_indices,
            out_values=out_values,
            merge_threshold=0,
            pack_values=pack_values,
            pack_indices=pack_indices,
            merge_state=merge_state,
            merge_state_preinitialized=True,
        )
        torch.cuda.synchronize(device)
        assert not torch.any(out_indices == _SLOT_SENTINEL)
        fused_sorted = torch.sort(out_values, dim=1, descending=True).values
        assert torch.allclose(fused_sorted, gold_values, atol=1e-2, rtol=0)
        for row in range(rows):
            assert set(out_indices[row].tolist()) == gold_index_sets[row]
        assert int(merge_state.abs().sum()) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused indexer"
)
def test_fused_indexer_cooperative_pack_path_pads_absent_slots_across_launches():
    """Rows with fewer candidates than top-k pack every candidate and pad the rest."""
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 32, 2048, 1500
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=91, device=device
    )
    seqlens[1] = 700
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    pack_capacity, state_words = fused_indexer_scratch_capacity(rows, topk, num_sms)
    pack_values = torch.empty(pack_capacity, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_capacity, dtype=torch.int32, device=device)
    merge_state = torch.zeros(state_words, dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    for _launch in range(4):
        out_indices.fill_(_SLOT_SENTINEL)
        run_fused_paged_indexer(
            q_bytes=q_fp8.view(torch.uint8),
            weights=weights,
            k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
            k_scales=k_scales,
            real_page_table=page_table,
            seqlens=seqlens,
            num_heads=heads,
            topk=topk,
            out_indices=out_indices,
            out_values=out_values,
            merge_threshold=0,
            pack_values=pack_values,
            pack_indices=pack_indices,
            merge_state=merge_state,
            merge_state_preinitialized=True,
        )
        torch.cuda.synchronize(device)
        assert not torch.any(out_indices == _SLOT_SENTINEL)
        for row, live in enumerate(seqlens.tolist()):
            valid = out_indices[row][out_indices[row] >= 0]
            assert int(valid.numel()) == live
            assert set(valid.tolist()) == set(range(live))
            assert int((out_indices[row] == -1).sum()) == topk - live
            assert torch.isinf(out_values[row][out_indices[row] == -1]).all()
        assert int(merge_state.abs().sum()) == 0
