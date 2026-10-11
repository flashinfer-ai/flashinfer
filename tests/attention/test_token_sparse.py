"""token_sparse_kv_indices: key positions to token-sparse attention's K/V index lists."""

from functools import partial

import pytest
import torch

from flashinfer.token_sparse import token_sparse_kv_indices

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
