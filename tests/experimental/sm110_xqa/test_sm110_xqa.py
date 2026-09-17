# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Mathematical correctness of the public SM110 XQA interface."""

import math

import pytest
import torch

from flashinfer.experimental.sm110_xqa.jit import get_manifest
from flashinfer.sm110_xqa import attention, prepare


@pytest.fixture(autouse=True)
def exact_sm110():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("requires physical SM110")


def _sample(shape, distribution="uniform", seed=17):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.empty(shape, device="cuda", dtype=torch.float16)
    return (
        values.uniform_(-1, 1, generator=generator)
        if distribution == "uniform"
        else values.normal_(generator=generator)
    )


def _mask(queries, batch, kind):
    rows = []
    for row in range(queries):
        visible = set(range(row + 1)) if kind == "causal" else {row}
        ancestor = row
        while kind != "causal" and ancestor:
            ancestor = (ancestor - 1) // 2
            visible.add(ancestor)
        words = []
        for word in range((queries + 31) // 32):
            bits = sum(1 << (token % 32) for token in visible if token // 32 == word)
            words.append(bits if bits < 2**31 else bits - 2**32)
        rows.append(words)
    return (
        torch.tensor(rows, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch, 1, 1)
    )


def _cache(kv, fp8, paged):
    if fp8:
        k_scale = float(kv[:, 0].abs().max()) / 448.0
        v_scale = float(kv[:, 1].abs().max()) / 448.0
        scales = torch.tensor([k_scale, v_scale], device="cuda").reshape(1, 2, 1, 1, 1)
        stored = (kv.float() / scales).to(torch.float8_e4m3fn)
        dense = stored.float() * scales
    else:
        stored, dense, k_scale, v_scale = kv, kv.float(), 1.0, 1.0
    if not paged:
        return stored, None, dense, k_scale, v_scale
    batch, _, heads, capacity, dim = kv.shape
    pages = capacity // 128
    storage = stored.reshape(batch, 2, heads, pages, 128, dim).permute(0, 1, 3, 4, 2, 5)
    storage = storage.contiguous().reshape(batch * 2 * pages, 128, heads, dim)
    permutation = torch.randperm(storage.shape[0], device="cuda")
    page_table = permutation.argsort().to(torch.int32).reshape(batch, 2, pages)
    return storage[permutation], page_table, dense, k_scale, v_scale


def _oracle(q, dense, lengths, mask=None):
    batch, queries, q_heads, dim = q.shape
    heads = dense.shape[2]
    ratio = q_heads // heads
    result = []
    for request, length in enumerate(lengths):
        key = dense[request, 0, :, :length].repeat_interleave(ratio, dim=0)
        value = dense[request, 1, :, :length].repeat_interleave(ratio, dim=0)
        scores = torch.einsum("qhd,hkd->qhk", q[request].float(), key) / math.sqrt(dim)
        if mask is not None:
            draft = torch.arange(queries, device="cuda")
            allowed = (
                (mask[request, :, draft // 32].to(torch.int64) >> (draft % 32)) & 1
            ).bool()
            scores[:, :, length - queries :] = scores[
                :, :, length - queries :
            ].masked_fill(~allowed[:, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        result.append(torch.einsum("qhk,hkd->qhd", probabilities, value))
    return torch.stack(result).to(torch.float16)


@pytest.mark.parametrize(
    "batch,capacity,ratio,ragged",
    [
        (1, 1024, 8, False),
        (2, 512, 8, False),
        (4, 256, 8, False),
        (1, 1, 8, False),
        (1, 63, 8, False),
        (1, 65, 8, False),
        (1, 255, 8, False),
        (1, 257, 8, False),
        (4, 1024, 8, True),
        (1, 1024, 4, False),
        (2, 513, 16, True),
    ],
)
def test_decode(batch, capacity, ratio, ragged):
    q = _sample((batch, 4 * ratio, 128))
    kv = _sample((batch, 2, 4, capacity, 128), seed=23)
    lengths = (
        [max(1, capacity * (request + 1) // batch) for request in range(batch)]
        if ragged
        else [capacity] * batch
    )
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    originals = tuple(tensor.clone() for tensor in (q, kv, seq))
    expected = _oracle(q[:, None], kv.float(), lengths)[:, 0]
    out = torch.full_like(q, float("nan"))
    plan = prepare(q, kv, seq, out=out)
    assert plan.run() is out
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    for tensor, original in zip((q, kv, seq), originals, strict=True):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "partition_tokens,capacity",
    [
        (64, 513),
        (256, 1024),
        (256, 513),
        (320, 513),
        (1024, 513),
    ],
)
def test_decode_runtime_partitions_and_counter_replay(partition_tokens, capacity):
    q = _sample((3, 32, 128))
    kv = _sample((3, 2, 4, capacity, 128), seed=43)
    lengths = [0, 65, capacity]
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    plan = prepare(q, kv, seq, partition_tokens=partition_tokens)
    manifest = get_manifest()
    producer = manifest["routes"]["decode_fp16_contiguous"]
    partitions = (capacity + partition_tokens - 1) // partition_tokens
    expected_route = (
        producer["single_partition_route"]
        if partitions == 1 and producer["merge_stats_cache"]
        else "decode_fp16_contiguous"
    )
    assert plan.route == expected_route
    assert manifest["routes"][plan.route]["merge_stats_cache"] == (
        producer["merge_stats_cache"] and partitions > 1
    )
    expected = _oracle(q[:, None], kv.float(), lengths)[:, 0]
    partial, statistics, counters = plan.workspace
    assert partial.shape == (96, partitions, 128)
    assert statistics.shape == (*partial.shape[:2], 2)
    for _ in range(3):
        plan.output.fill_(float("nan"))
        plan.run()
        torch.testing.assert_close(plan.output, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(counters, torch.zeros_like(counters), atol=0, rtol=0)


def test_decode_caller_workspace_is_initialized_once_at_prepare():
    q = _sample((1, 32, 128))
    kv = _sample((1, 2, 4, 257, 128), seed=47)
    seq = torch.tensor([257], device="cuda", dtype=torch.int32)
    workspace = (
        torch.empty((32, 2, 128), device="cuda", dtype=torch.float32),
        torch.empty((32, 2, 2), device="cuda", dtype=torch.float32),
        torch.full((1, 4), 17, device="cuda", dtype=torch.int32),
    )
    plan = prepare(q, kv, seq, workspace=workspace, partition_tokens=256)
    assert all(
        actual is expected
        for actual, expected in zip(plan.workspace, workspace, strict=True)
    )
    torch.testing.assert_close(
        workspace[2], torch.zeros_like(workspace[2]), atol=0, rtol=0
    )
    expected = _oracle(q[:, None], kv.float(), [257])[:, 0]
    plan.run()
    torch.testing.assert_close(plan.output, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        workspace[2], torch.zeros_like(workspace[2]), atol=0, rtol=0
    )


def test_decode_stats_cache_rejects_misaligned_caller_statistics():
    if not get_manifest()["routes"]["decode_fp16_contiguous"]["merge_stats_cache"]:
        pytest.skip("requires a frozen stats-cache producer")
    q = _sample((1, 32, 128))
    kv = _sample((1, 2, 4, 257, 128), seed=53)
    seq = torch.tensor([257], device="cuda", dtype=torch.int32)
    statistics = torch.empty(32 * 2 * 2 + 1, device="cuda", dtype=torch.float32)[
        1:
    ].view(32, 2, 2)
    assert statistics.is_contiguous() and statistics.data_ptr() % 8 == 4
    workspace = (
        torch.empty((32, 2, 128), device="cuda", dtype=torch.float32),
        statistics,
        torch.empty((1, 4), device="cuda", dtype=torch.int32),
    )
    out = torch.full_like(q, float("nan"))
    plan = prepare(q, kv, seq, out=out, workspace=workspace, partition_tokens=256)
    # Exercise the native TensorView address check before any device launch.
    with pytest.raises(ValueError, match="statistics.*alignment"):
        plan.run()
    assert torch.isnan(out).all().item()


CONTIGUOUS_CASES = [
    (1, 20, 256, 8, "causal", "uniform"),
    (1, 1, 1, 8, "causal", "normal"),
    (1, 20, 255, 8, "binary_tree", "normal"),
    (1, 31, 257, 8, "binary_tree", "uniform"),
    (1, 32, 256, 8, "causal", "normal"),
    (1, 33, 257, 8, "binary_tree", "normal"),
    (2, 7, 129, 4, "binary_tree", "normal"),
    (1, 5, 65, 16, "causal", "uniform"),
]
CACHE_CASES = [
    (1, 20, 256, 8, None, "uniform"),
    (2, 7, 256, 4, (129, 255), "normal"),
    (1, 33, 384, 8, (257,), "normal"),
    (1, 5, 128, 16, (65,), "uniform"),
    (1, 1, 128, 8, (1,), "normal"),
    (1, 32, 128, 8, (128,), "normal"),
    (1, 31, 384, 8, (257,), "uniform"),
]


def _tree_case(
    batch, queries, capacity, ratio, lengths, distribution, mask_kind, fp8, paged
):
    q = _sample((batch, queries, 4 * ratio, 512), distribution)
    kv = _sample((batch, 2, 4, capacity, 512), distribution, seed=29)
    kv[:, 0].mul_(0.5)
    kv[:, 1].mul_(2.0)
    lengths = [capacity] * batch if lengths is None else list(lengths)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    mask = _mask(queries, batch, mask_kind)
    stored, pages, dense, k_scale, v_scale = _cache(kv, fp8, paged)
    expected = _oracle(q, dense, lengths, mask)
    inputs = tuple(
        tensor for tensor in (q, stored, seq, mask, pages) if tensor is not None
    )
    originals = [tensor.view(torch.uint8).clone() for tensor in inputs]
    out = attention(
        q,
        stored,
        seq,
        mask=mask,
        page_table=pages,
        page_size=128 if paged else 0,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    for tensor, original in zip(inputs, originals, strict=True):
        torch.testing.assert_close(tensor.view(torch.uint8), original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "batch,queries,capacity,ratio,mask_kind,distribution", CONTIGUOUS_CASES
)
def test_tree_contiguous(batch, queries, capacity, ratio, mask_kind, distribution):
    _tree_case(
        batch, queries, capacity, ratio, None, distribution, mask_kind, False, False
    )


@pytest.mark.parametrize("fp8,paged", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize(
    "batch,queries,capacity,ratio,lengths,distribution", CACHE_CASES
)
def test_tree_cache(batch, queries, capacity, ratio, lengths, distribution, fp8, paged):
    _tree_case(
        batch,
        queries,
        capacity,
        ratio,
        lengths,
        distribution,
        "causal" if lengths is None else "binary_tree",
        fp8,
        paged,
    )


@pytest.mark.parametrize(
    "fp8,paged", [(False, False), (False, True), (True, False), (True, True)]
)
def test_packed_tree(fp8, paged):
    counts, lengths = (7, 33), (129, 257)
    uniform_q = _sample((2, 33, 32, 512), "normal")
    kv = _sample((2, 2, 4, 384, 512), "normal", seed=31)
    kv[:, 0].mul_(0.5)
    kv[:, 1].mul_(2.0)
    uniform_mask = _mask(33, 2, "binary_tree")
    q = torch.cat([uniform_q[index, :count] for index, count in enumerate(counts)])
    mask = torch.cat(
        [uniform_mask[index, :count] for index, count in enumerate(counts)]
    )
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 7, 40], device="cuda", dtype=torch.int32)
    stored, pages, dense, k_scale, v_scale = _cache(kv, fp8, paged)
    expected = torch.cat(
        [
            _oracle(
                uniform_q[index : index + 1, :count],
                dense[index : index + 1],
                [lengths[index]],
                uniform_mask[index : index + 1, :count],
            )[0]
            for index, count in enumerate(counts)
        ]
    )
    inputs = tuple(
        tensor
        for tensor in (q, stored, seq, mask, pages, offsets)
        if tensor is not None
    )
    originals = [tensor.view(torch.uint8).clone() for tensor in inputs]
    out = attention(
        q,
        stored,
        seq,
        mask=mask,
        page_table=pages,
        page_size=128 if paged else 0,
        q_cu_seq_lens=offsets,
        max_q_len=33,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    for tensor, original in zip(inputs, originals, strict=True):
        torch.testing.assert_close(tensor.view(torch.uint8), original, atol=0, rtol=0)


@pytest.mark.parametrize("partition_tokens", [64, 1024])
def test_prepared_decode_nondefault_stream_and_graph_replay(partition_tokens):
    q = _sample((2, 32, 128))
    kv = _sample((2, 2, 4, 512, 128), seed=37)
    seq = torch.tensor([512, 257], dtype=torch.int32, device="cuda")
    preparation_stream = torch.cuda.Stream()
    preparation_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(preparation_stream):
        plan = prepare(q, kv, seq, partition_tokens=partition_tokens)
    producer = get_manifest()["routes"]["decode_fp16_contiguous"]
    expected_route = (
        producer["single_partition_route"]
        if partition_tokens >= 512 and producer["merge_stats_cache"]
        else "decode_fp16_contiguous"
    )
    assert plan.route == expected_route
    stream = torch.cuda.Stream()
    stream.wait_stream(preparation_stream)
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        for _ in range(3):
            q.mul_(0.5)
            plan.output.fill_(float("nan"))
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    expected = _oracle(q[:, None], kv.float(), [512, 257])[:, 0]
    torch.testing.assert_close(plan.output, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        plan.workspace[2], torch.zeros_like(plan.workspace[2]), atol=0, rtol=0
    )


def test_rejects_output_alias_before_launch():
    q = _sample((1, 32, 128))
    kv = _sample((1, 2, 4, 256, 128), seed=41)
    seq = torch.tensor([256], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="share storage"):
        prepare(q, kv, seq, out=q)
