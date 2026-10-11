# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""single_head_paged_mqa_logits against an exact reference over random paged BF16
or NVFP4 keys."""

import pytest
import torch

from flashinfer import nvfp4_kv_dequantize
from flashinfer.attn_scores import single_head_paged_mqa_logits

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
