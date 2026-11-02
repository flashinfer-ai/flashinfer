# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""single_head_paged_mqa_logits and quantize_write_index_qk_nvfp4 against exact
references over random paged BF16 or NVFP4 keys."""

import pytest
import torch

from flashinfer import nvfp4_kv_dequantize
from flashinfer.attn_scores import (
    quantize_write_index_qk_nvfp4,
    single_head_paged_mqa_logits,
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
        "flashinfer.attn_scores.single_head.get_compute_capability",
        lambda device: (9, 0),
    )
    with pytest.raises(ValueError, match="requires SM10x"):
        run()


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
