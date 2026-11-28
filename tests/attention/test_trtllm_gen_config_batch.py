"""trtllm_fmha_set_config_batch: one generation launch over k blocks of n requests
matches k separate launches of n requests bit for bit."""

from functools import partial

import pytest
import torch

from flashinfer.decode import (
    trtllm_batch_decode_with_kv_cache,
    trtllm_fmha_set_config_batch,
)
from flashinfer.utils import get_compute_capability

HEAD_DIM, TOKENS = 256, 8192


def _skip_unless_sm10x() -> None:
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen token-sparse cubins are SM100/SM103")


def _inputs(rows: int, heads: int, dtype: torch.dtype, top_k: int, seed: int):
    """Token-sparse MQA decode inputs: each row attends to its own top_k slots."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    randn = partial(torch.randn, device="cuda", generator=generator)
    kv = randn(TOKENS, 2 * HEAD_DIM).to(dtype)
    q = randn(rows, heads, HEAD_DIM).to(dtype)
    # Short counts around the KV tile (128) and the top-k, and full rows, so that
    # some rows split their KV over several CTAs and others do not.
    lengths = (1, 5, 127, 128, 129, top_k - 1, top_k)
    counts = torch.tensor(
        [lengths[r % len(lengths)] for r in range(rows)], dtype=torch.int32
    ).cuda()
    order = torch.rand(rows, TOKENS, device="cuda", generator=generator).argsort(-1)
    slots = order[:, :top_k].int()
    slots.masked_fill_(torch.arange(top_k, device="cuda") >= counts[:, None], -1)
    k, v = kv.view(-1, 1, 1, 2 * HEAD_DIM).split(HEAD_DIM, dim=-1)
    table = torch.stack([slots, slots], dim=1)  # the K list and the V list
    return q, (k, v), table, counts


def _decode(q, kv, table, counts, top_k, out, workspace, counter) -> None:
    trtllm_batch_decode_with_kv_cache(
        q,
        kv,
        workspace,
        table,
        counts,
        top_k,
        bmm1_scale=HEAD_DIM**-0.5,
        out=out,
        backend="trtllm-gen",
        sparse_top_k=top_k,
        multi_ctas_kv_counter_buffer=counter,
        uses_shared_paged_kv_idx=False,
    )


def _buffers():
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    return workspace, counter


@pytest.mark.parametrize("block,blocks", [(32, 10), (256, 3)])
@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_config_batch_matches_block_launches(block, blocks, heads, dtype, top_k):
    _skip_unless_sm10x()
    rows = block * blocks
    q, kv, table, counts = _inputs(rows, heads, dtype, top_k, seed=rows * heads + top_k)
    workspace, counter = _buffers()
    expected = torch.empty(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, block):
        part = slice(first, first + block)
        _decode(
            q[part],
            kv,
            table[part],
            counts[part],
            top_k,
            expected[part],
            workspace,
            counter,
        )
    out = torch.full_like(expected, float("nan"))
    trtllm_fmha_set_config_batch(block)
    try:
        _decode(q, kv, table, counts, top_k, out, workspace, counter)
    finally:
        trtllm_fmha_set_config_batch(0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("config_batch", [0, 64, 96])
def test_config_batch_off_or_not_smaller_is_a_plain_launch(config_batch):
    """0 turns it off; a config batch not smaller than the launch is ignored."""
    _skip_unless_sm10x()
    rows, heads, top_k = 64, 8, 2048
    q, kv, table, counts = _inputs(rows, heads, torch.bfloat16, top_k, seed=7)
    workspace, counter = _buffers()
    plain = torch.empty(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _decode(q, kv, table, counts, top_k, plain, workspace, counter)
    out = torch.full_like(plain, float("nan"))
    trtllm_fmha_set_config_batch(config_batch)
    try:
        _decode(q, kv, table, counts, top_k, out, workspace, counter)
    finally:
        trtllm_fmha_set_config_batch(0)
    assert torch.equal(out.view(torch.int16), plain.view(torch.int16))


def test_config_batch_inside_cuda_graph():
    """The setting is read when the launch is recorded; replays keep it."""
    _skip_unless_sm10x()
    block, blocks, heads, top_k = 32, 6, 8, 2048
    rows = block * blocks
    q, kv, table, counts = _inputs(rows, heads, torch.bfloat16, top_k, seed=11)
    workspace, counter = _buffers()
    expected = torch.empty(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, block):
        part = slice(first, first + block)
        _decode(
            q[part],
            kv,
            table[part],
            counts[part],
            top_k,
            expected[part],
            workspace,
            counter,
        )
    out = torch.full_like(expected, float("nan"))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    trtllm_fmha_set_config_batch(block)
    try:
        with torch.cuda.stream(
            stream
        ):  # warm up (JIT, cubin loading) outside the capture
            _decode(q, kv, table, counts, top_k, out, workspace, counter)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            _decode(q, kv, table, counts, top_k, out, workspace, counter)
    finally:
        trtllm_fmha_set_config_batch(0)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
