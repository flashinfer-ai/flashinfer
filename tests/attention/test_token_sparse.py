"""Token-sparse attention inputs from key positions: token_sparse_kv_indices's K/V
index lists, group_union_masks's unions and masks."""

from functools import partial

import pytest
import torch

from flashinfer.token_sparse import group_union_masks, token_sparse_kv_indices

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="token-sparse attention requires SM10x",
)
same = partial(torch.testing.assert_close, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    ("tokens", "requests_per_token"), [(1, 1), (3, 2), (77, 1), (5, 4)]
)
@torch.inference_mode()
def test_token_sparse_kv_indices_match_reference_and_replay(
    tokens: int, requests_per_token: int, top_k: int
) -> None:
    used = tokens * requests_per_token
    # Each token's Top-K: `counts` keys of its request, then -1s.
    table = torch.randperm(4 * 2**14, device="cuda").int().view(4, 2**14)
    requests = torch.randint(0, 4, (tokens,), dtype=torch.int32, device="cuda")
    counts = torch.randint(0, top_k + 1, (tokens,), dtype=torch.int32).cuda()
    keys = torch.randint(0, 2**20, (tokens, top_k), dtype=torch.int32).cuda()
    keys[torch.arange(top_k, device="cuda") >= counts[:, None]] = -1
    slots = table[requests.long()[:, None], keys.clamp(min=0) // 64] * 64 + keys % 64
    tables = torch.full((used + 3, 2, top_k), -7, dtype=torch.int32, device="cuda")
    lengths = tables[:, 0, 0].clone()
    expected_tables, expected_lengths = tables.clone(), lengths.clone()
    expected = slots.masked_fill(keys < 0, -1).repeat_interleave(requests_per_token, 0)
    expected_tables[:used] = expected[:, None]
    expected_lengths[:used] = counts.repeat_interleave(requests_per_token)
    run = partial(
        token_sparse_kv_indices,
        keys,
        requests,
        table,
        requests_per_token=requests_per_token,
    )
    run(block_tables=tables, seq_lens=lengths)  # warm-up compiles
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(block_tables=tables, seq_lens=lengths)
    for _ in range(2):  # eager, then replayed
        same(tables, expected_tables)
        same(lengths, expected_lengths)
        tables.fill_(-7)
        lengths.fill_(-7)
        graph.replay()


@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("group", [1, 4, 16])
@torch.inference_mode()
def test_group_union_masks_match_reference_and_replay(group: int, top_k: int) -> None:
    """Two requests of 128 tokens each: request 0 has 5000 keys; request 1 has
    600000, past the 2**19 keys of a narrower block table. Short rows end in
    -1s; padding rows (mask row -1) end each request's tokens, and the last
    group is all padding (union row -1). Mask and union rows run in reverse
    order. Nothing is written past a union or a mask row's words."""
    torch.manual_seed(group)
    rows = 256
    contexts, width = [5000, 600000], 600000 // 64 + 1
    tables = torch.randperm(2 * width, device="cuda").int().view(2, width)
    request = torch.arange(rows, device="cuda") // 128
    lattice = torch.tensor(contexts, device="cuda")[request, None] // 4096
    picked = torch.rand(rows, 4096, device="cuda").argsort(1)[:, :top_k]
    picked = picked * lattice + torch.randint_like(picked, 64) % lattice
    counts = torch.full((rows,), top_k, device="cuda")
    counts[1::3] = torch.randint(1, top_k, (len(counts[1::3]),), device="cuda")
    counts[100:128] = counts[-max(group, 3) :] = 0  # padding rows
    keys = picked.masked_fill(torch.arange(top_k, device="cuda") >= counts[:, None], -1)
    positions = keys.int()
    live = counts > 0
    query_rows = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    query_rows[live] = torch.arange(int(live.sum()), device="cuda").flip(0).int()
    used = live.view(-1, group).any(1)
    unions = int(used.sum())
    group_unions = torch.full_like(used, -1, dtype=torch.int32)
    group_unions[used] = torch.arange(unions, device="cuda").flip(0).int()
    requests = (torch.arange(rows // group, device="cuda") * group // 128).int()
    columns = min(group * top_k, 128 * -(-contexts[1] // 128))
    columns = 256 * -(-columns // 256)
    union = torch.full((unions, columns), -7, dtype=torch.int32, device="cuda")
    sizes = torch.full((unions,), -7, dtype=torch.int32, device="cuda")
    mask = torch.full((int(live.sum()), columns // 32), -7, dtype=torch.int32).cuda()
    outputs = dict(union=union, union_sizes=sizes, mask=mask, num_keys=contexts[1])
    inputs = positions, query_rows, group_unions, requests, tables

    def run() -> None:
        for buffer in union, sizes, mask:
            buffer.fill_(-7)
        group_union_masks(*inputs, **outputs)

    def check() -> None:
        for g in range(rows // group):
            mine = keys[g * group : (g + 1) * group]
            members = mine[mine >= 0].unique()
            size, table = len(members), tables[requests[g]].long()
            if not size:  # an all-padding group
                continue
            row = int(group_unions[g])
            assert sizes[row] == size
            same(union[row, :size], (table[members // 64] * 64 + members % 64).int())
            assert bool((union[row, size:] == -7).all())
            words = 4 * -(-size // 128)
            bits = torch.zeros(group, 32 * words, dtype=torch.long, device="cuda")
            column = torch.searchsorted(members, mine.clamp(min=0))
            bits.scatter_reduce_(1, column, (mine >= 0).long(), "amax")
            weights = 1 << torch.arange(32, device="cuda")
            packed = (bits.view(group, words, 32) * weights).sum(-1).int()
            queries = query_rows[g * group : (g + 1) * group]
            same(mask[queries[queries >= 0], :words], packed[queries >= 0])
            assert bool((mask[queries[queries >= 0], words:] == -7).all())

    run()  # warm-up compiles
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    context = torch.tensor(contexts, device="cuda")[request, None]
    keys = torch.where(keys >= 0, context - 1 - keys, -1)  # new keys
    positions.copy_(keys)
    graph.replay()
    check()
