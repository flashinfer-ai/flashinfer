# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Public continuous Full and paged delegation correctness and replay."""

from contextlib import contextmanager

import pytest
import torch

from flashinfer import FP8MQATopKWrapper, FP8PagedMQATopKWrapper
from .test_attn_scores import _check_selection, _paged_inputs, _requires_selective_sm100


@contextmanager
def _warmed_capture_guard(monkeypatch):
    from flashinfer.experimental.attn_scores import selective_logits

    def forbidden(*args, **kwargs):
        raise AssertionError("warmed capture must not compile or synchronize")

    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    with monkeypatch.context() as guard:
        guard.setattr(selective_logits, "_compile_for_device", forbidden)
        for method in ("item", "cpu", "tolist"):
            guard.setattr(torch.Tensor, method, forbidden)
        guard.setattr(torch.cuda, "synchronize", forbidden)
        yield
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before


def _continuous_inputs(heads, ratio, *, capacity=4096):
    torch.manual_seed(42)
    q_lengths, kv_lengths = [5, 0, 7], [1029, 0, 2053]
    q = (torch.randn(sum(q_lengths), heads, 128, device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(q.shape[0], heads, device="cuda")
    keys = (torch.randn(3 * (capacity // ratio), 128, device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(keys.shape[0], device="cuda") + 0.5
    qo = torch.tensor([0, 5, 5, 12], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 1029, 1029, 3082], dtype=torch.int32, device="cuda")
    return q, weights, keys, scales, qo, kv, q_lengths, kv_lengths


def _rows(q, weights, keys, scales, q_lengths, kv_lengths, ratio):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        result, qb, kb = [], 0, 0
        for qn, kn in zip(q_lengths, kv_lengths, strict=True):
            for row in range(qn):
                end = (kn - qn + row + 1) // ratio
                dots = q[qb + row].float() @ keys[kb : kb + end].float().T
                result.append(
                    (dots.relu() * weights[qb + row, :, None]).sum(0)
                    * scales[kb : kb + end]
                )
            qb += qn
            kb += kn // ratio
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("minimum_budget", [False, True])
def test_continuous_full_ragged_stripes_rebinding_and_graph(
    heads, ratio, minimum_budget, monkeypatch
):
    q, weights, keys, scale_a, qo, kv, qn, kn = _continuous_inputs(heads, ratio)
    wrapper = FP8MQATopKWrapper(
        strategy="full",
        num_heads=heads,
        top_k=512,
        compression_ratio=ratio,
        max_context_length=4096,
        max_chunk_size=13,
        max_batch_size=3,
        num_keys=keys.shape[0],
        device="cuda",
        max_workspace_size=65536 if minimum_budget else None,
    )
    wrapper.plan(qo, kv, num_query_rows=q.shape[0])
    scale_b = scale_a.flip(0).contiguous()
    for scale in (scale_a, scale_b, scale_a):
        selected = wrapper.run(q, weights, keys, k_scale=scale)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(selected, _rows(q, weights, keys, scale, qn, kn, ratio), 512)
    assert (wrapper._capacity_plan._selected[12:] == -1).all()
    from flashinfer.experimental.attn_scores import selective_logits

    def forbidden(*args, **kwargs):
        raise AssertionError("warmed execution must not compile or read a CUDA scalar")

    with monkeypatch.context() as guard:
        guard.setattr(selective_logits, "_compile_for_device", forbidden)
        guard.setattr(torch.Tensor, "item", forbidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), _warmed_capture_guard(monkeypatch):
            wrapper.plan(qo, kv, num_query_rows=q.shape[0])
            selected = wrapper.run(q, weights, keys, k_scale=scale_a)
        graph.replay()
    _check_selection(selected, _rows(q, weights, keys, scale_a, qn, kn, ratio), 512)
    del graph
    wrapper.clear_pools()
    wrapper.prepare_pool(keys, k_scale=scale_b)
    selected = wrapper.run(q, weights, keys, k_scale=scale_b)
    _check_selection(selected, _rows(q, weights, keys, scale_b, qn, kn, ratio), 512)


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("ratio", [1, 4])
def test_continuous_v1_ragged_binding_invalid_prefix_and_replay(
    heads, ratio, monkeypatch
):
    q, weights, keys, scale_a, qo, kv, qn, kn = _continuous_inputs(heads, ratio)
    op = FP8MQATopKWrapper(
        backend="v1",
        num_heads=heads,
        top_k=512,
        compression_ratio=ratio,
        max_context_length=4096,
        max_chunk_size=13,
        max_batch_size=3,
        num_keys=keys.shape[0],
        device="cuda",
    )
    op.plan(qo, kv, num_query_rows=q.shape[0])
    scale_b = scale_a.flip(0).contiguous()
    for scale in (scale_a, scale_b, scale_a):
        selected = op.run(q, weights, keys, k_scale=scale)
        _check_selection(selected, _rows(q, weights, keys, scale, qn, kn, ratio), 512)
        assert op.error.item() == op.repair_error.item() == 0
    invalid = qo.clone()
    invalid[0] = 1
    op.plan(invalid, kv, num_query_rows=q.shape[0])
    assert bool((op.run(q, weights, keys, k_scale=scale_a) == -1).all())
    assert op.error.item() & 2
    op.plan(qo, kv, num_query_rows=q.shape[0])
    op.run(q, weights, keys, k_scale=scale_a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _warmed_capture_guard(monkeypatch):
        op.plan(qo, kv, num_query_rows=q.shape[0])
        selected = op.run(q, weights, keys, k_scale=scale_a)
    original_a = scale_a.clone()
    for scale in (scale_b, original_a):
        scale_a.copy_(scale)
        graph.replay()
        _check_selection(selected, _rows(q, weights, keys, scale_a, qn, kn, ratio), 512)
        assert op.error.item() == op.repair_error.item() == 0
    assert bool((op._capacity_plan._selected[q.shape[0] :] == -1).all())


def _paged_fixture(ratio, page_size, *, single):
    q, weights, keys, scales, qo, kv, qn, kn = _continuous_inputs(64, ratio)
    if single:
        q, weights = q[5:].contiguous(), weights[5:].contiguous()
        start = kn[0] // ratio
        keys = keys[start : start + 4096 // ratio].clone()
        scales = scales[start : start + 4096 // ratio].clone()
        qo = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
        kv = torch.tensor([0, 2053], dtype=torch.int32, device="cuda")
        qn, kn = [7], [2053]
    physical = [0]
    for length in kn:
        physical.append(physical[-1] + length // ratio)
    prefix = torch.tensor(physical, dtype=torch.int32, device="cuda")
    pool, live_table = _paged_inputs(
        (q, keys[: physical[-1]], scales[: physical[-1]], weights, qo, prefix),
        page_size,
    )
    columns = (4096 // ratio + page_size - 1) // page_size
    table = torch.full((len(qn), columns), -1, dtype=torch.int32, device="cuda")
    table[:, : live_table.shape[1]].copy_(live_table)
    return q, weights, keys, scales, qo, kv, qn, kn, pool, table


def _unpack(pool, table, lengths, ratio):
    page_size = pool.shape[1]
    flat = pool.view(pool.shape[0], -1)
    keys = (
        flat[:, : page_size * 128]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .view(-1, page_size, 128)
    )
    scales = (
        flat[:, page_size * 128 :].contiguous().view(torch.float32).view(-1, page_size)
    )
    packed_keys, packed_scales = [], []
    for request, original in enumerate(lengths):
        length = original // ratio
        if length:
            ids = table[request, : (length + page_size - 1) // page_size].long()
            packed_keys.append(keys[ids].reshape(-1, 128)[:length])
            packed_scales.append(scales[ids].reshape(-1)[:length])
    return torch.cat(packed_keys), torch.cat(packed_scales)


@_requires_selective_sm100
@pytest.mark.parametrize(
    "strategy,backend", [("full", "v1"), ("selective", "v1"), ("selective", "v2")]
)
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("page_size", [32, 64, 128])
def test_paged_delegate_oracle_native_and_mutable_graph(
    strategy, backend, ratio, page_size, monkeypatch
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        FP8PagedMQATopKWrapper as Native,
    )
    from flashinfer.experimental.attn_scores import selective_logits

    q, weights, keys, scales, qo, kv, qn, kn, pool, table = _paged_fixture(
        ratio, page_size, single=backend == "v2"
    )
    options = dict(
        num_heads=64,
        head_dim=128,
        top_k=512,
        compression_ratio=ratio,
        max_context_length=4096,
        max_chunk_size=q.shape[0] + 1,
        max_batch_size=len(qn),
        kv_layout=pool,
        page_size=page_size,
    )
    wrapper = FP8PagedMQATopKWrapper(strategy=strategy, backend=backend, **options)
    native = Native(strategy="full" if ratio == 1 else "selective", **options)
    for op in (wrapper, native):
        op.plan(qo, kv, table, num_query_rows=q.shape[0])
        selected = op.run(q, weights, pool)
        assert op.error.item() == op.repair_error.item() == 0
        _check_selection(selected, _rows(q, weights, keys, scales, qn, kn, ratio), 512)
    pointers = {
        name: tensor.data_ptr() for name, tensor in wrapper.input_buffers().items()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "warmed paged replay must not compile or read a CUDA scalar"
        )

    with monkeypatch.context() as guard:
        guard.setattr(selective_logits, "_compile_for_device", forbidden)
        guard.setattr(torch.Tensor, "item", forbidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), _warmed_capture_guard(monkeypatch):
            wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
            selected = wrapper.run(q, weights, pool)
        graph.replay()
    _check_selection(selected, _rows(q, weights, keys, scales, qn, kn, ratio), 512)
    table[0, :2].copy_(table[0, :2].flip(0))
    weights.mul_(-1)
    words = pool.view(pool.shape[0], -1)
    words[:, : page_size * 128].bitwise_xor_(128)
    words[:, page_size * 128 :].view(torch.float32).mul_(2)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    live_keys, live_scales = _unpack(pool, table, kn, ratio)
    _check_selection(
        selected, _rows(q, weights, live_keys, live_scales, qn, kn, ratio), 512
    )
    native.plan(qo, kv, table, num_query_rows=q.shape[0])
    _check_selection(
        native.run(q, weights, pool),
        _rows(q, weights, live_keys, live_scales, qn, kn, ratio),
        512,
    )
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in wrapper.input_buffers().items()
    }
    table[0, 0] = pool.shape[0]
    graph.replay()
    assert wrapper.error.item() == 8
    assert (selected == -1).all()


@_requires_selective_sm100
def test_paged_delegate_out_rejects_another_prepared_pool():
    q, weights, _, _, qo, kv, _, _, pool, table = _paged_fixture(1, 64, single=True)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=8,
        max_batch_size=1,
        page_size=64,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
    other = pool.clone()
    wrapper.prepare_pool(pool)
    wrapper.prepare_pool(other)
    output = other.view(torch.int32).view(-1)[: q.shape[0] * 512].view(q.shape[0], 512)
    with pytest.raises(ValueError, match="overlap"):
        wrapper.run(q, weights, pool, out=output)


def _fp4_values(packed, scales):
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = table[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    return values * torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=-1)


@_requires_selective_sm100
@pytest.mark.parametrize(
    "strategy,backend", [("full", "v1"), ("selective", "v1"), ("selective", "v2")]
)
def test_paged_delegate_empty_pool_and_next_n(strategy, backend):
    pool = torch.empty((0, 64, 1, 132), dtype=torch.uint8, device="cuda")
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        backend=backend,
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=8,
        max_batch_size=1,
        page_size=64,
        kv_layout=pool,
    )
    prefix = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.empty((0, 64), dtype=torch.int32, device="cuda")
    q = torch.empty((0, 64, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.empty((0, 64), device="cuda")
    wrapper.plan(prefix, prefix, table, num_query_rows=0, next_n=1)
    assert wrapper.run(q, weights, pool).shape == (0, 512)
    assert wrapper.error.item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(prefix, prefix, table, num_query_rows=0, next_n=1)
        result = wrapper.run(q, weights, pool)
    graph.replay()
    assert result.shape == (0, 512)
    assert wrapper.error.item() == 0
    live = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    table = torch.zeros((1, 64), dtype=torch.int32, device="cuda")
    q = torch.zeros((1, 64, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones((1, 64), device="cuda")
    wrapper.plan(live, live, table, num_query_rows=1, next_n=1)
    assert (wrapper.run(q, weights, pool) == -1).all()
    assert wrapper.error.item() & 8


@_requires_selective_sm100
@pytest.mark.parametrize(
    "strategy,backend", [("full", "v1"), ("selective", "v1"), ("selective", "v2")]
)
def test_paged_delegate_forwards_host_hints_and_next_n(strategy, backend):
    q, weights, keys, scales, qo, kv, _, kn, pool, table = _paged_fixture(
        1, 64, single=True
    )
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        backend=backend,
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=8,
        max_batch_size=1,
        page_size=64,
        kv_layout=pool,
    )
    q, weights = q[:4], weights[:4]
    qo = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    hints = (qo.cpu(), torch.tensor(kn, dtype=torch.int32))
    wrapper.plan(qo, kv, table, num_query_rows=4, next_n=4, host_length_hints=hints)
    expected = _rows(q, weights, keys, scales, [4], kn, 1)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    assert wrapper.error.item() == 0
    wrapper.plan(qo, kv, table, num_query_rows=4, next_n=2)
    assert (wrapper.run(q, weights, pool) == -1).all()
    assert wrapper.error.item() & 2
    wrapper.plan(
        qo, kv, table, num_query_rows=4, host_length_hints=(hints[0], hints[1] - 1)
    )
    assert (wrapper.run(q, weights, pool) == -1).all()
    assert wrapper.error.item() & 2
    wrapper.plan(qo, kv, table, num_query_rows=4, next_n=4, host_length_hints=hints)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    assert wrapper.error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("model_config", [None, "DS-V4-Flash"])
def test_paged_delegate_mxfp4_matches_contiguous_and_oracle(
    page_size, model_config, monkeypatch
):
    torch.manual_seed(123)
    rows, heads, count = 7, 64, 1024
    q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
    qs = torch.randint(126, 129, (rows, heads, 4), device="cuda", dtype=torch.uint8)
    keys = torch.randint(0, 256, (count, 64), device="cuda", dtype=torch.uint8)
    scales = torch.randint(126, 129, (count, 4), device="cuda", dtype=torch.uint8)
    weights = torch.rand(rows, heads, device="cuda") + 0.5
    pages = (513 + page_size - 1) // page_size
    pk = keys[: pages * page_size].view(pages, page_size * 64)
    ps = scales[: pages * page_size].view(pages, page_size * 4)
    pool = torch.cat((pk, ps), dim=1).flip(0).contiguous().view(pages, page_size, 1, 68)
    page_table = torch.full(
        (1, count // page_size), -1, dtype=torch.int32, device="cuda"
    )
    page_table[0, :pages] = torch.arange(
        pages - 1, -1, -1, dtype=torch.int32, device="cuda"
    )
    qo = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 2053], dtype=torch.int32, device="cuda")
    options = dict(
        backend="v2",
        qkv_dtype="mxfp4",
        model_config=model_config,
        num_heads=heads,
        top_k=512,
        compression_ratio=4,
        max_context_length=4096,
        max_chunk_size=32,
        max_batch_size=1,
    )
    paged = FP8PagedMQATopKWrapper(
        strategy="selective", page_size=page_size, kv_layout=pool, **options
    )
    continuous = FP8MQATopKWrapper(device="cuda", num_keys=count, **options)
    paged.plan(qo, kv, page_table, num_query_rows=rows)
    continuous.plan(qo, kv, num_query_rows=rows)
    expected = continuous.run(q, weights, keys, k_scale=scales, q_scale=qs).clone()
    actual = paged.run(q, weights, pool, q_scale=qs)
    assert paged.error.item() == paged.repair_error.item() == 0
    decoded_q, decoded_k = _fp4_values(q, qs), _fp4_values(keys, scales)
    reference = _rows(
        decoded_q,
        weights,
        decoded_k,
        torch.ones(count, device="cuda"),
        [rows],
        [2053],
        4,
    )
    _check_selection(expected, reference, 512)
    _check_selection(actual, reference, 512)
    assert torch.equal(paged._keys[:513], keys[:513])
    assert torch.equal(paged._scales[:513], scales[:513])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _warmed_capture_guard(monkeypatch):
        paged.plan(qo, kv, page_table, num_query_rows=rows)
        actual = paged.run(q, weights, pool, q_scale=qs)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    _check_selection(actual, reference, 512)
