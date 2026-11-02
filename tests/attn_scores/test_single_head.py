# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""single_head_paged_mqa_logits against an exact reference over random paged BF16
keys."""

import pytest
import torch

from flashinfer.attn_scores import single_head_paged_mqa_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="single_head_paged_mqa_logits requires SM10x",
)


def bf16(rows: int) -> torch.Tensor:
    """Random BF16 rows of small integers times {1/8, ..., 2}: exact FP32 dots."""
    scales = 2.0 ** torch.randint(-3, 2, (rows, 128), device="cuda")
    return (torch.randint(-8, 9, (rows, 128), device="cuda") * scales).bfloat16()


@pytest.mark.parametrize(
    "lengths",
    [
        (1, 63, 127, 64, 0, 4096),  # ends on a page, mid key tile; an empty group
        (70000, 3, 128 * 257 + 5),  # runs of several CTAs, one cut short
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 8])
@torch.inference_mode()
def test_single_head_paged_mqa_logits_match_reference_and_replay(lengths, rows) -> None:
    """Keys on shuffled pages; table entries past a length would fault if read,
    and scores past it stay unwritten. A replay sees new lengths and tables."""
    torch.manual_seed(rows)
    batch, capacity = len(lengths), max(lengths) + 300
    width = -(-capacity // 64)
    pages = batch * width
    key_cache = bf16(64 * pages).view(pages, 64, 128)
    table = torch.empty(batch, width, dtype=torch.int32, device="cuda")
    context = torch.empty(batch, dtype=torch.int32, device="cuda")
    requests = torch.arange(batch, dtype=torch.int32, device="cuda").flip(0)
    query = torch.empty(batch * rows, 128, dtype=torch.bfloat16, device="cuda")
    scores = torch.empty(batch * rows, capacity, device="cuda")

    def prepare(lengths: list[int]) -> None:
        context.copy_(torch.tensor(lengths, dtype=torch.int32))
        ends = (context[:, None] + 63) // 64  # group j's, for request requests[j]
        shuffled = torch.randperm(pages, device="cuda")[: table.numel()].int()
        past = torch.arange(width, device="cuda") >= ends
        table[requests.long()] = shuffled.view_as(table).masked_fill(past, 2**31 - 1)
        query.copy_(bf16(batch * rows))
        scores.fill_(-7)

    def check(lengths: list[int]) -> None:
        for group, length in enumerate(lengths):
            mine = slice(group * rows, group * rows + rows)
            request = int(requests[group])
            keys = key_cache[table[request, : -(-length // 64)].long()].flatten(0, 1)
            expected = query[mine].double() @ keys[:length].double().T
            torch.testing.assert_close(
                scores[mine, :length], expected.float(), rtol=0, atol=0
            )
            assert bool((scores[mine, length:] == -7).all())

    def run() -> None:
        single_head_paged_mqa_logits(
            query, key_cache, table, context, requests=requests, scores=scores
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
    with pytest.raises(ValueError, match="BF16 of dimension 128"):
        run(query.float(), key_cache, table, context, requests=requests, scores=scores)
