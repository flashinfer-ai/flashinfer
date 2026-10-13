# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The single-head paged MQA kernels and quantize_write_index_k_nvfp4 against
exact references over random paged BF16 or NVFP4 keys."""

import pytest
import torch

from flashinfer import nvfp4_kv_dequantize
from flashinfer.attn_scores import (
    quantize_write_index_k_nvfp4,
    single_head_paged_mqa_logits,
    single_head_paged_mqa_threshold_select,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="single_head_paged_mqa_logits requires SM10x",
)


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
@pytest.mark.parametrize("keys", ["bf16", "nvfp4", "nvfp4_packed"])
@torch.inference_mode()
def test_single_head_paged_mqa_logits_match_reference_and_replay(
    lengths, rows, keys
) -> None:
    """Keys on shuffled pages; table entries past a length would fault if read,
    and scores past it stay unwritten. A replay sees new lengths and tables.
    nvfp4_packed: keys and scales are views of one cache, each page's 64 keys
    then their scales."""
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
        single_head_paged_mqa_logits(
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
def test_single_head_paged_mqa_threshold_select_matches_reference_and_replay(
    query_lens, context_lens, top_k, keys, repair
) -> None:
    """Causal 128-row tiles, odd ones reversed; rows 2 mod 8 score 0 (ties past
    the tie slots), rows 5 mod 8 NaN. Counts are exact, and each written slot
    holds a distinct key of its class with its score (less the threshold). The
    repair reselects each row that misses at or below its Top-K cutoff."""
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

    def run() -> None:
        thresholds.copy_(requested)
        counters.zero_()
        candidates.fill_(-1)
        single_head_paged_mqa_threshold_select(
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


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """x rounded to E2M1, to nearest with ties to an even code, saturating."""
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device).double()
    distance = (x.double().abs().clamp(max=6)[..., None] - grid).abs()
    distance += torch.arange(8, device=x.device) % 2 * 1e-12  # ties: even codes
    return (grid[distance.argmin(-1)] * x.sign()).float()


@pytest.mark.parametrize("tokens", [1, 9, 300])
@torch.inference_mode()
def test_quantize_write_index_k_nvfp4_matches_reference(tokens: int) -> None:
    """Each written key's blocks of 16 are scaled by an E4M3 amax / 6 and
    rounded to E2M1; slots are shuffled, negative ones skipped, and nothing
    else is written."""
    torch.manual_seed(tokens)
    pages = -(-tokens // 64) + 2
    spread = 4 ** torch.randn(tokens, 1, device="cuda")
    key = (torch.randn(tokens, 128, device="cuda") * spread).bfloat16()
    key[0, :16] = 0  # a zero block: scale 0
    slots = torch.randperm(64 * pages, device="cuda")[:tokens]
    slots[1::5] = -1
    key_cache = torch.full((pages, 64, 64), 7, dtype=torch.uint8, device="cuda")
    key_scales = torch.full_like(key_cache[..., :8], 7).view(torch.float8_e4m3fn)
    quantize_write_index_k_nvfp4(key, slots, key_cache, key_scales)

    blocks = key.float().view(tokens, 8, 16)
    scale = (blocks.abs().amax(-1) / 6).to(torch.float8_e4m3fn).float()
    inverse = torch.where(scale > 0, 1 / scale, 0)
    expected = (e2m1(blocks * inverse[..., None]) * scale[..., None]).flatten(1)
    live = slots >= 0
    rows, scale_rows = key_cache.view(-1, 64), key_scales.view(-1, 8)
    torch.testing.assert_close(
        scale_rows[slots[live]].float(), scale[live], rtol=0, atol=0
    )
    got = dequantize(rows[slots[live]], scale_rows[slots[live]]).float()
    torch.testing.assert_close(got, expected[live], rtol=0, atol=0)
    untouched = torch.ones(64 * pages, dtype=torch.bool, device="cuda")
    untouched[slots[live]] = False
    assert bool((rows[untouched] == 7).all())


@torch.inference_mode()
def test_single_head_paged_mqa_logits_rejects_bad_shapes() -> None:
    query = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    table = torch.zeros(2, 2, dtype=torch.int32, device="cuda")
    context = torch.zeros(2, dtype=torch.int32, device="cuda")
    requests = torch.zeros(2, dtype=torch.int32, device="cuda")
    scores = torch.zeros(4, 128, device="cuda")
    run = single_head_paged_mqa_logits
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
