"""Static token-sparse MQA/GQA decode through trtllm_batch_decode_with_kv_cache(sparse_top_k=...)."""

from functools import partial

import pytest
import torch

from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.utils import get_compute_capability

HEAD_DIM, TOKENS = 256, 8192


@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("rows", [7, 300])
@pytest.mark.parametrize("heads", [2, 4, 5, 8, 12, 16, 24, 48, 96])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_token_sparse_matches_reference(rows, heads, dtype, top_k):
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen token-sparse cubins are SM100/SM103")
    generator = torch.Generator(device="cuda").manual_seed(rows * heads)
    randn = partial(torch.randn, device="cuda", generator=generator)
    kv, q = randn(TOKENS, 2 * HEAD_DIM), randn(rows, heads, HEAD_DIM).to(dtype)
    # Short counts around the KV tile (128) and the top-k, and a full row.
    lengths = (1, 5, 127, 128, 129, top_k - 1, top_k)
    counts = torch.tensor([lengths[r % len(lengths)] for r in range(rows)]).int().cuda()
    order = torch.rand(rows, TOKENS, device="cuda", generator=generator).argsort(-1)
    slots = order[:, :top_k].int()
    slots.masked_fill_(torch.arange(top_k).cuda() >= counts[:, None], -1)
    kv[~torch.isin(torch.arange(TOKENS).cuda(), slots)] = 300.0  # reading one shows
    kv = kv.to(dtype)
    k, v = kv.view(-1, 1, 1, 2 * HEAD_DIM).split(HEAD_DIM, dim=-1)  # views of one row
    table = torch.stack([slots, slots], dim=1)  # the K list and the V list
    out = torch.empty(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    run = partial(
        trtllm_batch_decode_with_kv_cache, q, (k, v), workspace, table, counts
    )
    run = partial(run, top_k, bmm1_scale=HEAD_DIM**-0.5, out=out, backend="trtllm-gen")
    run = partial(run, sparse_top_k=top_k, multi_ctas_kv_counter_buffer=counter)

    def check() -> None:  # FP8 also rounds the probabilities to E4M3 before P.V
        selected = kv.float()[table[:, 0].long().clamp(min=0)]
        logits = torch.einsum("rhd,rkd->rhk", q.float(), selected[..., :HEAD_DIM])
        logits.masked_fill_(table[:, None, 0] < 0, float("-inf"))
        expected = (logits * HEAD_DIM**-0.5).softmax(-1) @ selected[..., HEAD_DIM:]
        tolerance = 1e-1 if dtype == torch.float8_e4m3fn else 1e-2
        torch.testing.assert_close(
            out.float(), expected, atol=tolerance, rtol=tolerance
        )

    for _ in range(2):  # the second run reuses the split-KV semaphores
        run(uses_shared_paged_kv_idx=False)
        check()
        table.copy_(table.flip(0))  # row r takes the last but r-th selection and count
        counts.copy_(counts.flip(0))
