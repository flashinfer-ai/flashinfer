# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The SDSA (Simplified DSA) indexer kernels and quantize_write_index_qk_nvfp4
against exact references over random paged BF16 or NVFP4 keys."""

from functools import partial

import pytest
import torch

from flashinfer import nvfp4_kv_dequantize
from flashinfer.sdsa import indexer
from flashinfer.sdsa import (
    gather_key_sample,
    indexer_scores,
    indexer_threshold_select,
    quantize_write_index_qk_nvfp4,
    thresholds_from_sample,
    top_k_from_candidates,
)

SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
sm10x_only = pytest.mark.skipif(not SM10X, reason="requires SM10x")


def rows_of(nvfp4: bool, rows: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random BF16 rows of small integers, or NVFP4 ones, times scales in
    {1/8, ..., 2}: dot products are exact in FP32 either way."""
    if not nvfp4:
        scales = 2.0 ** torch.randint(-3, 2, (rows, 128), device="cuda")
        values = torch.randint(-8, 9, (rows, 128), device="cuda") * scales
        return values.bfloat16(), None
    scales = 2.0 ** torch.randint(-3, 2, (rows, 8), device="cuda")
    values = torch.randint(256, (rows, 64), device="cuda").byte()
    return values, scales.to(torch.float8_e4m3fn)


def ordered(bits: torch.Tensor) -> torch.Tensor:
    """int32 score bits as int64 values in the order of their floats, as the
    kernels order them (-0 below +0, NaN with the sign bit below -inf, other
    NaN above +inf)."""
    bits = bits.long() & 0xFFFFFFFF
    return torch.where(bits >= 2**31, bits ^ 0xFFFFFFFF, bits | 2**31)


def check_top_keys(
    candidates: torch.Tensor, ranges: torch.Tensor, top_keys: torch.Tensor
) -> None:
    """Each row's keys are distinct keys of the pairs in its range, then -1, and
    their scores are the row's top_k largest (ties at the cutoff in any
    order)."""
    top_k = top_keys.shape[1]
    for row, (first, last) in enumerate(ranges.T.tolist()):
        keys, bits = candidates[row, first:last].long().unbind(1)
        count = min(top_k, last - first)
        assert bool((top_keys[row, count:] == -1).all())
        if count:
            chosen = top_keys[row, :count].long()
            order = keys.argsort()
            where = torch.searchsorted(keys[order], chosen).clamp(max=len(keys) - 1)
            assert torch.equal(keys[order][where], chosen)
            assert chosen.unique().numel() == count
            mine = ordered(bits[order][where]).sort().values
            assert torch.equal(mine, ordered(bits).topk(count).values.sort().values)


def dequantize(values: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
    if scales is None:
        return values.double()
    one = torch.ones(1, device="cuda")
    return nvfp4_kv_dequantize(values, scales.view(torch.uint8), one).double()


@pytest.mark.parametrize(
    "lengths",
    [
        (1, 63, 127, 64, 0, 4096),  # ends on a page, mid key tile; an empty group
        (70000, 3, 128 * 257 + 5),  # runs of several CTAs, one cut short
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("keys", ["bf16", "bf16_triton", "nvfp4", "nvfp4_packed"])
@torch.inference_mode()
def test_indexer_scores_match_reference_and_replay(
    lengths, rows, keys, monkeypatch
) -> None:
    """Keys on shuffled pages; table entries past a length would fault if read,
    and scores past it stay unwritten. A replay sees new lengths and tables.
    nvfp4_packed: keys and scales are views of one cache, each page's 64 keys
    then their scales."""
    if keys.startswith("nvfp4") and not SM10X:
        pytest.skip("Off SM10x, keys are BF16 (scored by Triton)")
    if keys == "bf16_triton":  # the path off SM10x, also on SM10x
        monkeypatch.setattr(indexer, "get_compute_capability", lambda _: (9, 0))
        keys = "bf16"
    torch.manual_seed(rows)
    nvfp4 = keys != "bf16"
    batch, capacity = len(lengths), max(lengths) + 300
    width = -(-capacity // 64)
    pages = batch * width
    values, scales = rows_of(nvfp4, 64 * pages)
    key_cache, key_scales = values.view(pages, 64, -1), None
    if nvfp4:
        key_scales = scales.view(pages, 64, 8)
    if keys == "nvfp4_packed":
        parts = (key_cache.flatten(1), key_scales.view(torch.uint8).flatten(1))
        cache = torch.cat(parts, 1)
        key_cache = cache[:, : 64 * 64].unflatten(1, (64, 64))
        key_scales = cache[:, 64 * 64 :].view(torch.float8_e4m3fn)
        key_scales = key_scales.unflatten(1, (64, 8))
    table = torch.empty(batch, width, dtype=torch.int32, device="cuda")
    context = torch.empty(batch, dtype=torch.int32, device="cuda")
    requests = torch.arange(batch, dtype=torch.int32, device="cuda").flip(0)
    query, query_scales = rows_of(nvfp4, batch * rows)
    scores = torch.empty(batch * rows, capacity, device="cuda")

    def prepare(lengths: list[int]) -> None:
        context.copy_(torch.tensor(lengths, dtype=torch.int32))
        ends = (context[:, None] + 63) // 64  # group j's, for request requests[j]
        shuffled = torch.randperm(pages, device="cuda")[: table.numel()].int()
        past = torch.arange(width, device="cuda") >= ends
        table[requests.long()] = shuffled.view_as(table).masked_fill(past, 2**31 - 1)
        scores.fill_(-7)

    def check(lengths: list[int]) -> None:
        for group, length in enumerate(lengths):
            mine = slice(group * rows, group * rows + rows)
            assert bool((scores[mine, length:] == -7).all())
            if not length:
                continue
            page = table[int(requests[group]), : -(-length // 64)].long()
            key = key_cache[page].flatten(0, 1)[:length]
            key_scale = query_scale = None
            if nvfp4:
                key_scale = key_scales[page].flatten(0, 1)[:length]
                query_scale = query_scales[mine]
            expected = dequantize(query[mine], query_scale)
            expected = expected @ dequantize(key, key_scale).T
            torch.testing.assert_close(
                scores[mine, :length], expected.float(), rtol=0, atol=0
            )

    def run() -> None:
        indexer_scores(
            query,
            key_cache,
            table,
            context,
            requests=requests,
            scores=scores,
            query_scales=query_scales,
            key_scales=key_scales,
        )

    prepare(list(lengths))
    run()  # warm-up compiles
    check(list(lengths))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replayed = [max(n - 61, 0) for n in reversed(lengths)]
    prepare(replayed)
    graph.replay()
    check(replayed)


@sm10x_only
@pytest.mark.parametrize(
    ("query_lens", "context_lens"),
    [
        ((1, 3, 7, 2), (1, 63, 127, 64)),  # keys end on a page, mid key tile
        ((129, 3, 1), (4097, 2051, 2049)),
        # 8 query tiles: runs of several tiles per CTA, some cut short, some empty.
        ((3, 128, 641), (16384, 16000, 12289)),
    ],
)
@pytest.mark.parametrize("keys", ["bf16", "nvfp4"])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("top_k", [64, 512, 2048])
@torch.inference_mode()
def test_indexer_threshold_select_matches_reference_and_replay(
    query_lens, context_lens, top_k, keys, repair
) -> None:
    """Causal 128-row tiles, odd ones reversed; rows 2 mod 8 score 0 (ties past
    the tie slots), rows 5 mod 8 NaN. Counts are exact, and each written slot
    holds a distinct key of its class with its score (less the threshold). The
    repair reselects each row that misses at or below its Top-K cutoff, and
    top_k_from_candidates then writes each row's Top-K keys."""
    torch.manual_seed(0)
    nvfp4, width = keys == "nvfp4", max(context_lens)
    capacity, blocks = max(width // 2, top_k - 1), -(-width // 64)
    tiles = [(r, s) for r, n in enumerate(query_lens) for s in range(0, n, 128)]
    ends = torch.zeros(len(tiles), 128, dtype=torch.int32)
    for tile, (request, start) in enumerate(tiles):
        count = min(query_lens[request] - start, 128)
        first = context_lens[request] - query_lens[request] + start + 1
        ends[tile, :count] = torch.arange(first, first + count)
    ends[1::2] = ends[1::2].flip(1)
    ends, rows = ends.flatten().cuda(), 128 * len(tiles)
    requests = torch.tensor([r for r, _ in tiles], dtype=torch.int32, device="cuda")
    pages = len(query_lens) * blocks
    values, scales = rows_of(nvfp4, 64 * pages)
    key_cache = values.view(pages, 64, -1)
    key_scales = scales.view(pages, 64, 8) if nvfp4 else None
    table = torch.randperm(pages, device="cuda").int().view(-1, blocks)
    query, query_scales = rows_of(nvfp4, rows)
    query[2::8] = 0
    if nvfp4:
        query_scales.view(torch.uint8)[5::8] = 0x7F  # E4M3 NaN
    else:
        query[5::8] = torch.nan
    thresholds, requested = torch.empty(2, rows, device="cuda")
    arrivals = torch.zeros(len(tiles), dtype=torch.int32, device="cuda")
    ranges = torch.empty(2, rows, dtype=torch.int32, device="cuda")
    repairs = dict(arrivals=arrivals, ranges=ranges) if repair else {}
    counters = torch.empty(rows, 2, dtype=torch.int32, device="cuda")
    candidates = torch.empty(rows, top_k + capacity, 2, dtype=torch.int32).cuda()
    top_keys = torch.empty(rows, top_k, dtype=torch.int32, device="cuda")

    def run() -> None:
        thresholds.copy_(requested)
        counters.zero_()
        candidates.fill_(-1)
        indexer_threshold_select(
            *(query, key_cache, ends, requests, table),
            thresholds=thresholds,
            counters=counters,
            candidates=candidates,
            max_context=width,
            top_k=top_k,
            query_scales=query_scales,
            key_scales=key_scales,
            **repairs,
        )
        if repair:
            top_k_from_candidates(candidates, ranges, top_keys=top_keys)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        keys = dequantize(key_cache.flatten(0, 1), None if scales is None else scales)
        queries = dequantize(query, query_scales)
        scores = torch.zeros(rows, width, device="cuda")
        for tile, (request, _) in enumerate(tiles):
            mine = slice(128 * tile, 128 * tile + 128)
            slot = table[request].long()[:, None] * 64 + torch.arange(64).cuda()
            scores[mine] = (queries[mine] @ keys[slot.flatten()[:width]].T).float()
        return scores, torch.arange(width, device="cuda") < ends[:, None]

    def check(scores: torch.Tensor, valid: torch.Tensor) -> None:
        if repair:  # rows that fit keep the threshold, the others go at or below
            need = valid.sum(1).clamp(max=top_k)
            above = (valid & (scores > requested[:, None])).sum(1)
            tied = (valid & ~(scores > requested[:, None])).sum(1)
            tied -= (valid & (scores < requested[:, None])).sum(1)
            fits = (above <= capacity) & (above + tied.clamp(max=top_k) >= need)
            ranked = scores.masked_fill(~valid, -torch.inf).sort(1, descending=True)
            cutoff = ranked[0].gather(1, (need - 1).clamp(min=0)[:, None])[:, 0]
            assert torch.equal(thresholds[fits], requested[fits])
            assert bool((thresholds[~fits] <= cutoff[~fits]).all())
        above = valid & (scores > thresholds[:, None])
        tied = valid & ~(scores > thresholds[:, None]) & ~(scores < thresholds[:, None])
        assert torch.equal(counters, torch.stack((above.sum(1), tied.sum(1)), 1).int())
        fit, tie = above.sum(1).clamp(max=capacity), tied.sum(1).clamp(max=top_k)
        if repair:  # every row fits, its slots in its range
            need = valid.sum(1).clamp(max=top_k)
            assert bool(((fit + tie >= need) & (above.sum(1) <= capacity)).all())
            assert torch.equal(ranges, torch.stack((top_k - tie, top_k + fit)).int())
        for row in range(rows):
            fitted, tied_count = int(fit[row]), int(tie[row])
            written = candidates[row, top_k - tied_count : top_k + fitted]
            if not repair:  # a repaired row's first sweep leaves stale slots
                assert bool((candidates[row, : top_k - tied_count] == -1).all())
                assert bool((candidates[row, top_k + fitted :] == -1).all())
            keys, bits = written.long().unbind(1)
            assert bool((keys >= 0).all()) and keys.unique().numel() == keys.numel()
            assert bool(tied[row, keys[:tied_count]].all())
            assert bool(above[row, keys[tied_count:]].all())
            score, threshold = scores[row, keys], thresholds[row]
            stored = bits.int().view(torch.float32)
            exact = torch.where(score <= 2 * threshold, score - threshold, score)
            torch.testing.assert_close(stored, exact, rtol=0, atol=0, equal_nan=True)
        if repair:  # the Top-K keys, from the candidates and of the scores
            check_top_keys(candidates, ranges, top_keys)
            count = valid.sum(1).clamp(max=top_k)
            for row in range(rows):
                row_scores = scores[row, valid[row]]
                if bool(row_scores.isfinite().all()):
                    chosen = top_keys[row, : int(count[row])].long()
                    best = row_scores.topk(int(count[row])).values.sort().values
                    assert torch.equal(scores[row, chosen].sort().values, best)

    scores, valid = reference()
    median = scores.nan_to_num(0).masked_fill(~valid, 0).median(1).values
    for threshold in (median, scores[:, 0].nan_to_num(0), -torch.inf, torch.inf):
        requested[:] = threshold
        run()  # the first run compiles
        check(scores, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ends.copy_((ends - 9).clamp(min=0))
    scores, valid = reference()
    requested.copy_(scores.nan_to_num(0).masked_fill(~valid, 0).median(1).values)
    graph.replay()
    check(scores, valid)


@sm10x_only
@pytest.mark.parametrize("scores", ["normal", "few", "same", "special"])
@pytest.mark.parametrize("top_k", [512, 2048, 65535])
@torch.inference_mode()
def test_top_k_from_candidates_matches_reference_and_replay(top_k, scores) -> None:
    """Rows of distinct keys with random ranges: empty, shorter than top_k, one
    pair longer, and up to every slot. Scores are normal, from a few values
    (many equal to the top_k-th), all the same, or with zeros of both signs,
    infinities and NaNs. A CUDA graph replay then uses new
    ranges."""
    torch.manual_seed(0)
    rows, slots = 200, 3 * top_k + 2048
    # Distinct keys in each row: an odd stride through 2**20, from a random key.
    offsets = torch.randint(0, 1 << 20, (rows, 1), device="cuda")
    keys = ((torch.arange(slots, device="cuda") * 7919 + offsets) % (1 << 20)).int()
    if scores == "normal":
        values = torch.randn(rows, slots, device="cuda")
    elif scores == "few":
        values = torch.randint(0, 7, (rows, slots), device="cuda") * 0.5
    elif scores == "same":
        values = torch.full((rows, slots), 1.5, device="cuda")
    else:
        special = torch.tensor([0.0, -0.0, torch.inf, -torch.inf, torch.nan, 1.0])
        values = special.cuda()[torch.randint(0, 6, (rows, slots), device="cuda")]
        values[::2] = values[::2].neg()  # NaNs of both signs
    candidates = torch.stack((keys, values.view(torch.int32)), 2).contiguous()
    ranges = torch.empty(2, rows, dtype=torch.int32, device="cuda")
    top_keys = torch.empty(rows, top_k, dtype=torch.int32, device="cuda")

    def new_ranges() -> None:
        first = torch.randint(0, slots // 2, (rows,), device="cuda")
        count = torch.randint(0, slots - slots // 2, (rows,), device="cuda")
        count[:4] = torch.tensor([0, top_k - 1, top_k, top_k + 1])
        first[4], count[4] = 0, slots
        ranges.copy_(torch.stack((first, first + count)))

    new_ranges()
    top_k_from_candidates(candidates, ranges, top_keys=top_keys)
    check_top_keys(candidates, ranges, top_keys)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        top_k_from_candidates(candidates, ranges, top_keys=top_keys)
    new_ranges()
    top_keys.fill_(-7)
    graph.replay()
    check_top_keys(candidates, ranges, top_keys)


@sm10x_only
@torch.inference_mode()
def test_top_k_from_candidates_rejects_bad_shapes() -> None:
    zeros = partial(torch.zeros, dtype=torch.int32, device="cuda")
    candidates, ranges, top_keys = zeros(4, 16, 2), zeros(2, 4), zeros(4, 8)
    for wrong in (
        (candidates[..., :1], ranges, top_keys),
        (candidates, zeros(4, 2), top_keys),
        (candidates, ranges, zeros(5, 8)),
        (candidates, ranges, zeros(4)),
        (candidates, ranges, zeros(4, 8, 1)),
    ):
        with pytest.raises(ValueError, match="must be \\[rows, slots, 2\\]"):
            top_k_from_candidates(wrong[0], wrong[1], top_keys=wrong[2])
    for wrong in (
        (candidates.long(), ranges, top_keys),
        (candidates, ranges, zeros(8, 4).T),
    ):
        with pytest.raises(ValueError, match="contiguous int32"):
            top_k_from_candidates(wrong[0], wrong[1], top_keys=wrong[2])
    with pytest.raises(ValueError, match="below 65536"):
        top_k_from_candidates(candidates, ranges, top_keys=zeros(4, 1 << 16))


@sm10x_only
@pytest.mark.parametrize("far", [False, True])
@pytest.mark.parametrize("keys", ["bf16", "nvfp4"])
@torch.inference_mode()
def test_gather_key_sample_matches_reference(keys, far) -> None:
    """Slots below a request's sample end get the key, and its scales, at their
    offset into their run of pages; the rest stay untouched. `far`: the pages
    sit past the first 2**31 bytes of the key cache."""
    torch.manual_seed(0)
    nvfp4, samples, run_keys, lengths = keys == "nvfp4", 3, 128, [1, 200, 4097, 8193]
    blocks = -(-max(lengths) // 64)
    pages = len(lengths) * blocks
    values, scales = rows_of(nvfp4, 64 * pages)
    key_cache, key_scales = values.view(pages, 64, -1), None
    table = torch.randperm(pages, device="cuda").int().view(-1, blocks)
    if nvfp4:
        key_scales = scales.view(pages, 64, 8)
    if far:
        skip = 2**31 // (key_cache[0].numel() * key_cache.element_size())
        key_cache = torch.cat(
            (key_cache.new_zeros(skip, 64, key_cache.shape[2]), key_cache)
        )
        if nvfp4:
            key_scales = torch.cat((key_scales.new_zeros(skip, 64, 8), key_scales))
        table += skip
    capacity = 128 * -(-max(lengths) * samples // (128 * run_keys))
    offsets = torch.randint(run_keys, (capacity,), dtype=torch.uint8, device="cuda")
    ends = [length // run_keys * samples for length in lengths]
    shape = (len(ends), capacity)
    sample_keys = torch.full((*shape, key_cache.shape[2]), 85, device="cuda")
    sample_keys = sample_keys.to(key_cache.dtype)
    sample_scales = None
    if nvfp4:
        sample_scales = torch.full((*shape, 8), 85, dtype=torch.uint8, device="cuda")
        sample_scales = sample_scales.view(torch.float8_e4m3fn)
    expected = [t if t is None else t.clone() for t in (sample_keys, sample_scales)]
    for request, end in enumerate(ends):
        key = torch.arange(end, device="cuda") // samples * run_keys
        key += offsets[:end].long()
        page = table[request, key // 64].long()
        expected[0][request, :end] = key_cache[page, key % 64]
        if nvfp4:
            expected[1][request, :end] = key_scales[page, key % 64]
    gather_key_sample(
        *(key_cache, table, offsets, torch.tensor(ends, device="cuda").int()),
        sample_keys=sample_keys,
        samples_per_run=samples,
        pages_per_run=run_keys // 64,
        key_scales=key_scales,
        sample_scales=sample_scales,
    )
    assert torch.equal(sample_keys.view(torch.uint8), expected[0].view(torch.uint8))
    if nvfp4:
        got = sample_scales.view(torch.uint8)
        assert torch.equal(got, expected[1].view(torch.uint8))


@sm10x_only
@torch.inference_mode()
def test_thresholds_from_sample_matches_reference() -> None:
    """A row's threshold is the least of its sampled scores at its Top
    positions, a -1 position or a NaN score reading as -inf; its counters are
    zeroed."""
    torch.manual_seed(0)
    rows, samples, rank = 77, 300, 40
    scores = torch.randn(rows, samples, device="cuda")
    scores[3, 5] = torch.nan
    top = torch.rand(rows, samples, device="cuda").argsort(1)[:, :rank].int()
    top[3, 0] = 5  # a NaN score
    top[7, -4:] = -1  # a row with fewer samples
    thresholds = torch.empty(rows, device="cuda")
    counters = torch.full((rows, 2), 7, dtype=torch.int32, device="cuda")
    thresholds_from_sample(scores, top, thresholds=thresholds, counters=counters)
    picked = scores.gather(1, top.clamp(min=0).long()).masked_fill(top < 0, -torch.inf)
    expected = torch.where(picked.isnan(), -torch.inf, picked).amin(1)
    assert torch.equal(thresholds, expected)
    assert bool((counters == 0).all())


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """x rounded to E2M1, to nearest with ties to an even code, saturating."""
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device).double()
    distance = (x.double().abs().clamp(max=6)[..., None] - grid).abs()
    distance += torch.arange(8, device=x.device) % 2 * 1e-12  # ties: even codes
    return (grid[distance.argmin(-1)] * x.sign()).float()


def nvfp4_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x's NVFP4 scales (amax / 6 of each 16 values, in E4M3) and its values
    after quantizing, as float32."""
    blocks = x.float().view(len(x), 8, 16)
    scale = (blocks.abs().amax(-1) / 6).to(torch.float8_e4m3fn).float()
    inverse = torch.where(scale > 0, 1 / scale, 0)
    return scale, (e2m1(blocks * inverse[..., None]) * scale[..., None]).flatten(1)


@sm10x_only
@pytest.mark.parametrize(
    ("tokens", "queries"), [(1, 9), (9, 300), (300, 1), (0, 5), (5, 0)]
)
@torch.inference_mode()
def test_quantize_write_index_qk_nvfp4_matches_reference(
    tokens: int, queries: int
) -> None:
    """Each written key and query has its blocks of 16 scaled by an E4M3 amax / 6
    and rounded to E2M1. Slots and rows are shuffled, negative ones are skipped,
    and nothing else is written."""
    torch.manual_seed(tokens + queries)
    pages, rows = -(-tokens // 64) + 2, queries + 7
    spread = 4 ** torch.randn(tokens + queries, 1, device="cuda")
    x = (torch.randn(tokens + queries, 128, device="cuda") * spread).bfloat16()
    x[0, :16] = 0  # a zero block: scale 0
    key, query = x.split([tokens, queries])
    slots = torch.randperm(64 * pages, device="cuda")[:tokens]
    slots[1::5] = -1
    query_rows = torch.randperm(rows, device="cuda")[:queries]
    query_rows[1::4] = -1
    key_cache = torch.full((pages, 64, 64), 7, dtype=torch.uint8, device="cuda")
    key_scales = torch.full_like(key_cache[..., :8], 7).view(torch.float8_e4m3fn)
    packed_query = torch.full((rows, 64), 7, dtype=torch.uint8, device="cuda")
    query_scales = torch.full_like(packed_query[:, :8], 7).view(torch.float8_e4m3fn)
    quantize_write_index_qk_nvfp4(
        key,
        slots,
        query,
        query_rows,
        key_cache=key_cache,
        key_scales=key_scales,
        packed_query=packed_query,
        query_scales=query_scales,
    )
    for values, at, written, scales in (
        (key, slots, key_cache.view(-1, 64), key_scales.view(-1, 8)),
        (query, query_rows, packed_query, query_scales),
    ):
        live = at >= 0
        if len(values):  # nvfp4_kv_dequantize takes no empty input
            scale, expected = nvfp4_reference(values)
            torch.testing.assert_close(
                scales[at[live]].float(), scale[live], rtol=0, atol=0
            )
            got = dequantize(written[at[live]], scales[at[live]]).float()
            torch.testing.assert_close(got, expected[live], rtol=0, atol=0)
        untouched = torch.ones(len(written), dtype=torch.bool, device="cuda")
        untouched[at[live]] = False
        assert bool((written[untouched] == 7).all())


@sm10x_only
@torch.inference_mode()
def test_quantize_write_index_qk_nvfp4_rejects_bad_inputs(monkeypatch) -> None:
    key = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(1, dtype=torch.int64, device="cuda")
    outputs = dict(
        key_cache=torch.zeros(4, 64, 64, dtype=torch.uint8, device="cuda"),
        key_scales=torch.zeros(4, 64, 8, dtype=torch.float8_e4m3fn, device="cuda"),
        packed_query=torch.zeros(4, 64, dtype=torch.uint8, device="cuda"),
        query_scales=torch.zeros(4, 8, dtype=torch.float8_e4m3fn, device="cuda"),
    )

    def run(key=key, slots=slots, query=key, query_rows=slots, **changed):
        quantize_write_index_qk_nvfp4(
            key, slots, query, query_rows, **(outputs | changed)
        )

    for wrong in (
        dict(key=key[:, :64]),
        dict(query=key[:, :64]),
        dict(slots=slots[:0]),
    ):
        with pytest.raises(ValueError, match="Keys and queries must be"):
            run(**wrong)
    for name in outputs:
        with pytest.raises(ValueError, match="key_cache must be"):
            run(**{name: outputs[name][:3, :8]})
    for wrong in (
        dict(key=key.float()),
        dict(query=key.float()),
        dict(slots=slots.int()),
        dict(query_rows=slots.int()),
        dict(key_cache=outputs["key_cache"].bfloat16()),
        dict(packed_query=outputs["packed_query"].bfloat16()),
        dict(key_scales=outputs["key_scales"].float()),
        dict(query_scales=outputs["query_scales"].float()),
    ):
        with pytest.raises(ValueError, match="key and query must be BF16 or FP16"):
            run(**wrong)
    monkeypatch.setattr(
        "flashinfer.sdsa.indexer.get_compute_capability",
        lambda device: (9, 0),
    )
    with pytest.raises(ValueError, match="requires SM10x"):
        run()


@sm10x_only
@torch.inference_mode()
def test_indexer_scores_rejects_bad_shapes() -> None:
    query = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    table = torch.zeros(2, 2, dtype=torch.int32, device="cuda")
    context = torch.zeros(2, dtype=torch.int32, device="cuda")
    requests = torch.zeros(2, dtype=torch.int32, device="cuda")
    scores = torch.zeros(4, 128, device="cuda")
    run = indexer_scores
    with pytest.raises(ValueError, match="block table must cover"):
        run(query, key_cache, table[:, :1], context, requests=requests, scores=scores)
    with pytest.raises(ValueError, match="both BF16 or both NVFP4"):
        run(query.float(), key_cache, table, context, requests=requests, scores=scores)
    with pytest.raises(ValueError, match="need scales"):
        run(
            query.view(torch.uint8)[:, :64],
            key_cache.view(torch.uint8)[..., :64],
            table,
            context,
            requests=requests,
            scores=scores,
        )
    with pytest.raises(ValueError, match="NVFP4 scales must be"):
        run(
            torch.zeros(4, 64, dtype=torch.uint8, device="cuda"),
            torch.zeros(4, 64, 64, dtype=torch.uint8, device="cuda"),
            table,
            context,
            requests=requests,
            scores=scores,
            query_scales=torch.zeros(4, 8, device="cuda"),
            key_scales=torch.zeros(4, 64, 8, device="cuda"),
        )
    with pytest.raises(ValueError, match="multiple of 16 bytes apart"):
        padded = torch.zeros(4, 64 * 128 + 4, dtype=torch.bfloat16, device="cuda")
        keys = padded[:, : 64 * 128].view(4, 64, 128)
        run(query, keys, table, context, requests=requests, scores=scores)
    with pytest.raises(ValueError, match="multiple of 16 bytes apart"):
        padded = torch.zeros(4, 64 * 8 + 4, dtype=torch.uint8, device="cuda")
        run(
            torch.zeros(4, 64, dtype=torch.uint8, device="cuda"),
            torch.zeros(4, 64, 64, dtype=torch.uint8, device="cuda"),
            table,
            context,
            requests=requests,
            scores=scores,
            query_scales=torch.zeros(4, 8, dtype=torch.uint8, device="cuda"),
            key_scales=padded[:, : 64 * 8].view(4, 64, 8),
        )


@sm10x_only
@torch.inference_mode()
def test_indexer_threshold_select_rejects_bad_shapes() -> None:
    query = torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    ends = torch.ones(128, dtype=torch.int32, device="cuda")
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    outputs = dict(
        thresholds=torch.zeros(128, device="cuda"),
        counters=torch.zeros(128, 2, dtype=torch.int32, device="cuda"),
        candidates=torch.zeros(128, 64, 2, dtype=torch.int32, device="cuda"),
    )
    # Each shape alone is wrong, though together they hold (2, 2) after the rows.
    bad = dict(
        outputs,
        counters=torch.zeros(128, 2, 2, dtype=torch.int32, device="cuda"),
        candidates=torch.zeros(128, 64, dtype=torch.int32, device="cuda"),
    )
    run = indexer_threshold_select
    # The kernel addresses the pairs of a row as int32, so int64 must not pass.
    long_pairs = dict(outputs, candidates=outputs["candidates"].long())
    with pytest.raises(ValueError, match="must be int32"):
        run(
            query,
            key_cache,
            ends,
            requests,
            table,
            **long_pairs,
            max_context=128,
            top_k=8,
        )
    long_ranges = dict(
        outputs,
        arrivals=torch.zeros(1, dtype=torch.int32, device="cuda"),
        ranges=torch.zeros(2, 128, dtype=torch.int64, device="cuda"),
    )
    with pytest.raises(ValueError, match="both int32"):
        run(
            query,
            key_cache,
            ends,
            requests,
            table,
            **long_ranges,
            max_context=128,
            top_k=8,
        )
    for wrong in (bad, dict(outputs, candidates=bad["candidates"])):
        with pytest.raises(ValueError, match="counters must be"):
            run(
                query,
                key_cache,
                ends,
                requests,
                table,
                **wrong,
                max_context=128,
                top_k=8,
            )


@torch.inference_mode()
def test_indexer_scores_off_sm10x_rejects_bad_layouts(
    monkeypatch,
) -> None:
    """The Triton path off SM10x checks the layouts its kernel assumes."""
    monkeypatch.setattr(indexer, "get_compute_capability", lambda _: (9, 0))
    query = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    table = torch.zeros(2, 2, dtype=torch.int32, device="cuda")
    context = torch.zeros(2, dtype=torch.int32, device="cuda")
    requests = torch.zeros(2, dtype=torch.int32, device="cuda")
    scores = torch.zeros(4, 128, device="cuda")
    run = partial(indexer_scores, requests=requests)
    with pytest.raises(ValueError, match="Keys must be"):
        run(query[:, :64], key_cache[..., :64], table, context, scores=scores)
    with pytest.raises(ValueError, match="contiguous rows"):
        run(
            query,
            key_cache,
            table,
            context,
            scores=torch.zeros(128, 4, device="cuda").T,
        )
    with pytest.raises(ValueError, match="contiguous rows"):  # every other length
        run(query, key_cache, table, context.repeat(2)[::2], scores=scores)


@sm10x_only
@torch.inference_mode()
def test_thresholds_from_sample_rejects_bad_counters() -> None:
    scores = torch.zeros(4, 128, device="cuda")
    top = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    thresholds = torch.zeros(4, device="cuda")
    with pytest.raises(ValueError, match="counters must be"):
        thresholds_from_sample(
            scores,
            top,
            thresholds=thresholds,
            counters=torch.zeros(4, 1, dtype=torch.int32, device="cuda"),
        )


@sm10x_only
@torch.inference_mode()
def test_gather_key_sample_rejects_bad_shapes() -> None:
    """Sample scales short of slots or of scales per key, and keys or scales not
    64 to a page, are refused: the kernel would write or read past them. (No
    request has samples, so a call let through writes nothing.)"""
    zeros = partial(torch.zeros, device="cuda")
    args = dict(
        key_cache=zeros(4, 64, 64, dtype=torch.uint8),
        key_scales=zeros(4, 64, 8, dtype=torch.float8_e4m3fn),
        sample_keys=zeros(2, 128, 64, dtype=torch.uint8),
        sample_scales=zeros(2, 128, 8, dtype=torch.float8_e4m3fn),
    )

    def run(**changes) -> None:
        call = dict(args, **changes)
        gather_key_sample(
            call.pop("key_cache"),
            zeros(2, 2, dtype=torch.int32),  # block table
            zeros(128, dtype=torch.uint8),  # offsets
            zeros(2, dtype=torch.int32),  # sample ends
            samples_per_run=1,
            pages_per_run=2,
            **call,
        )

    run()
    for wrong in (
        dict(sample_scales=args["sample_scales"][:, :64]),  # fewer slots
        dict(sample_scales=args["sample_scales"][..., :4]),  # fewer scales a key
        dict(key_scales=args["key_scales"][:, :32]),  # 32 keys a page
    ):
        with pytest.raises(ValueError, match="Scales must be"):
            run(**wrong)
    with pytest.raises(ValueError, match="dtypes"):  # uint8 scale bytes
        run(sample_scales=args["sample_scales"].view(torch.uint8))
    with pytest.raises(ValueError, match="Mismatched"):  # BF16 keys, 32 a page
        run(
            key_cache=zeros(4, 32, 128, dtype=torch.bfloat16),
            sample_keys=zeros(2, 128, 128, dtype=torch.bfloat16),
            key_scales=None,
            sample_scales=None,
        )


@torch.inference_mode()
def test_sample_helpers_need_cute_dsl_4_7(monkeypatch) -> None:
    """Without CuTe DSL 4.7 the sample and Top-K helpers say so, like the other
    CuTe single-head APIs, before they import the CuTe scorer."""
    monkeypatch.setattr(indexer, "is_cute_dsl_experimental_available", lambda: False)
    zeros = partial(torch.zeros, device="cuda")
    with pytest.raises(RuntimeError, match="needs CuTe DSL 4.7"):
        gather_key_sample(
            zeros(4, 64, 128, dtype=torch.bfloat16),
            zeros(2, 2, dtype=torch.int32),
            zeros(128, dtype=torch.uint8),
            zeros(2, dtype=torch.int32),
            sample_keys=zeros(2, 128, 128, dtype=torch.bfloat16),
            samples_per_run=1,
            pages_per_run=2,
        )
    with pytest.raises(RuntimeError, match="needs CuTe DSL 4.7"):
        thresholds_from_sample(
            zeros(4, 128),
            zeros(4, 8, dtype=torch.int32),
            thresholds=zeros(4),
            counters=zeros(4, 2, dtype=torch.int32),
        )
    with pytest.raises(RuntimeError, match="needs CuTe DSL 4.7"):
        top_k_from_candidates(
            zeros(4, 16, 2, dtype=torch.int32),
            zeros(2, 4, dtype=torch.int32),
            top_keys=zeros(4, 8, dtype=torch.int32),
        )
