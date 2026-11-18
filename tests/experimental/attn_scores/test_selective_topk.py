# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Contract and exactness checks for the selective backend."""

import importlib.util
import math
import subprocess
import sys
import pytest
import torch

from flashinfer.attn_scores import FP8MQATopKWrapper
from tests.test_helpers.parametrize import parametrize_product


def _selective_available():
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("cutlass") is not None
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


def _blackwell():
    if not _selective_available():
        pytest.skip("requires SM100 or SM103 and CuTe DSL")


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


def _check_selection(selected, expected_rows, top_k):
    for row, scores in enumerate(expected_rows):
        count = min(top_k, scores.numel())
        indices = selected[row, :count].long()
        assert ((indices >= 0) & (indices < scores.numel())).all()
        assert indices.unique().numel() == count
        torch.testing.assert_close(
            scores[indices].sort().values,
            scores.topk(count).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )
        assert (selected[row, count:] == -1).all()


_requires_selective_sm100 = pytest.mark.skipif(
    not _selective_available(), reason="requires SM100 or SM103 and CuTe DSL"
)


def test_selective_import_keeps_shared_cutlass_methods():
    # A fresh process catches import-time mutation even after other selective tests.
    program = """
import cutlass.cute as cute
import cutlass.pipeline as pipeline
release = pipeline.PipelineTmaAsync.consumer_release
fma = cute.arch.fma_packed_f32x2
add = cute.arch.add_packed_f32x2
import flashinfer.experimental.attn_scores.kernels.selective_topk
assert pipeline.PipelineTmaAsync.consumer_release is release, 'shared TMA release changed'
assert cute.arch.fma_packed_f32x2 is fma, 'shared packed FMA changed'
assert cute.arch.add_packed_f32x2 is add, 'shared packed add changed'
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("invalid", ["scale", "score_ends"])
def test_selective_raw_rejects_invalid_scale_or_score_end_storage(invalid):
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.api import (
        ThrowawayTopK,
    )

    q = torch.empty((2, 4, 128), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.ones((2, 4), device="cuda")
    bounds = torch.zeros(2, device="cuda", dtype=torch.int32)
    scales = torch.ones(127 if invalid == "scale" else 128, device="cuda")
    score_ends = torch.zeros(1, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="key scales|score ends"):
        ThrowawayTopK(
            q,
            k,
            weights,
            bounds,
            bounds,
            k_scale=scales,
            _score_ends=score_ends if invalid == "score_ends" else None,
        )


def test_selective_aligned_extent_invalid_prefix_fails_closed_on_replay():
    _blackwell()
    rows, keys, heads = 64, 128, 4
    q = torch.randn((rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((keys, 128), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand((rows, heads), device="cuda")
    scales = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        strategy="selective",
        backend="cutedsl",
        num_heads=heads,
        top_k=512,
        max_context_length=256,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
    )
    op.plan(qo, kv, num_query_rows=rows)
    op.prepare_pool(k, k_scale=scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.plan(qo, kv, num_query_rows=rows)
        result = op.run(q, weights, k, k_scale=scales)
    graph.replay()
    torch.cuda.synchronize()
    assert op.error.item() == 0
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        scores = (
            torch.einsum("rhd,kd->rhk", q.float(), k.float()).relu()
            * weights[:, :, None]
        ).sum(1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    for row in range(rows):
        end = keys - rows + row + 1
        selected = result[row, :end].long()
        assert selected.unique().numel() == end
        assert torch.all((selected >= 0) & (selected < end))
        assert torch.all(result[row, end:] == -1)
        torch.testing.assert_close(
            scores[row, selected].sort().values, scores[row, :end].sort().values
        )
    kv[-1] = 256
    graph.replay()
    torch.cuda.synchronize()
    assert op.error.item() & 4
    assert torch.all(result == -1)


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
def test_selective_caller_graph_ownership_and_host_hints(encoding, monkeypatch):
    """Hints, prepared K/scale bindings and replans share caller-owned graphs."""
    _blackwell()
    torch.manual_seed(47)
    rows, keys, heads = 64, 1024, 64
    weights = torch.randn((rows, heads), device="cuda")
    if encoding == "fp8":
        q = torch.randn((rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn((keys, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(keys, device="cuda")
        q_scale = None
    else:
        q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
        k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        scales = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
    pool = k
    other_pool = pool.clone()
    other_scales = scales.clone()
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    hints = (torch.tensor([0, rows]), torch.tensor([keys]))
    graph = torch.cuda.CUDAGraph()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qf = q.float() if encoding == "fp8" else _decode_fp4(q)
        kf = k.float() if encoding == "fp8" else _decode_fp4(k)
        scores = (torch.einsum("rhd,kd->rhk", qf, kf).relu() * weights[:, :, None]).sum(
            1
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
    scores.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
    )

    def assert_exact(value):
        assert torch.all(value >= 0) and torch.all(value < ends[:, None])
        ordered = value.sort(1).values
        assert not torch.any(ordered[:, 1:] == ordered[:, :-1])
        selected = scores.gather(1, value.long()).sort(1, descending=True).values
        torch.testing.assert_close(
            selected, scores.topk(512, dim=1).values, rtol=1e-4, atol=1e-3
        )

    def unexpected_graph(*args, **kwargs):
        raise AssertionError("wrapper must not own a CUDA graph")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "CUDAGraph", unexpected_graph)
        op = FP8MQATopKWrapper(
            backend="cutedsl",
            qkv_dtype=encoding,
            num_heads=heads,
            top_k=512,
            max_context_length=keys,
            max_chunk_size=rows,
            max_batch_size=1,
            device="cuda",
            num_keys=keys,
        )
        op.plan(qo, kv, num_query_rows=rows, next_n=None, host_length_hints=hints)
        expected = op.run(q, weights, pool, k_scale=scales, q_scale=q_scale)
        torch.cuda.synchronize()
        assert op.error.item() == op.repair_error.item() == 0
        assert_exact(expected)
        plan = op._capacity_plan
        original_engine, plan_type = plan._engine, type(plan)
        original_execute = plan_type._prepare_layer_inputs

        def fail_second_pool(self, value, **kwargs):
            if self is plan and value.data_ptr() == other_pool.data_ptr():
                raise RuntimeError("injected pool warmup failure")
            return original_execute(self, value, **kwargs)

        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(plan_type, "_prepare_layer_inputs", fail_second_pool)
            with pytest.raises(RuntimeError, match="injected pool warmup failure"):
                op.prepare_pool(other_pool, k_scale=other_scales)
        assert plan._engine is original_engine and len(plan._pool_bindings) == 1
        assert_exact(op.run(q, weights, pool, k_scale=scales, q_scale=q_scale))
        op.prepare_pool(other_pool, k_scale=other_scales)
        with torch.cuda.graph(graph):
            op.plan(qo, kv, num_query_rows=rows)
            result = op.run(
                q, weights, other_pool, k_scale=other_scales, q_scale=q_scale
            )
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(result)
        assert {binding[0].data_ptr() for binding in plan._pool_bindings.values()} == {
            pool.data_ptr(),
            other_pool.data_ptr(),
        }
        addresses = {
            name: value.data_ptr() for name, value in op.input_buffers().items()
        }
        op.plan(
            qo,
            kv,
            num_query_rows=rows,
            host_length_hints=(hints[0], torch.tensor([keys - 128])),
        )
        invalid = op.run(q, weights, pool, k_scale=scales, q_scale=q_scale)
        torch.cuda.synchronize()
        assert op.error.item() & 2
        assert torch.all(invalid == -1)
        op.plan(qo, kv, num_query_rows=rows, host_length_hints=hints)
        recovered = op.run(q, weights, pool, k_scale=scales, q_scale=q_scale)
        torch.cuda.synchronize()
        assert op.error.item() == op.repair_error.item() == 0
        assert_exact(recovered)
        assert addresses == {
            name: value.data_ptr() for name, value in op.input_buffers().items()
        }
        del graph
        op.clear_pools()
        assert not plan._pool_bindings


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"backend": "bad"}, "backend"),
        ({"backend": "cutedsl", "num_heads": 8}, "H4/H32/H64"),
        ({"backend": "cutedsl", "num_heads": 64, "max_batch_size": 148}, "prefill"),
        ({"backend": "v1"}, "backend"),
        ({"backend": "v2"}, "backend"),
        (
            {
                "backend": "cutedsl",
                "qkv_dtype": "mxfp8",
                "math_mode": "relu_fast",
                "num_heads": 64,
            },
            "MXFP8.*relu_fast",
        ),
    ],
)
def test_selective_unsupported_contract(kwargs, match):
    settings = dict(
        device="cuda",
        max_context_length=8192,
        max_chunk_size=64,
        max_batch_size=1,
        num_heads=64,
    )
    settings.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FP8MQATopKWrapper(**settings)


@pytest.mark.parametrize(
    "name,config",
    [
        ("DS-V3.2", (64, 128, 2048, 1)),
        ("DS-V4-Flash", (64, 128, 512, 4)),
        ("DS-V4-Pro", (64, 128, 1024, 4)),
        ("GLM-5.3-Flash", (32, 128, 512, 4)),
        ("GLM-5.3", (32, 128, 2048, 1)),
        ("Qwen3.8-Flash-Next", (4, 128, 512, 4)),
        (None, (4, 128, 512, 1)),
    ],
)
def test_selective_automatic_model_config(name, config):
    from flashinfer.experimental.attn_scores.indexer_config import resolve_model_config

    assert resolve_model_config(*config) == (
        None if name == "Qwen3.8-Flash-Next" else name
    )
    if name is None:
        return
    assert resolve_model_config(*config, model_config=name) == name
    with pytest.raises(ValueError, match="compression_ratio"):
        resolve_model_config(*config[:3], 2, model_config=name)


def _decode_fp4(packed):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    return torch.stack(
        (table[(packed & 15).long()], table[(packed >> 4).long()]), -1
    ).flatten(-2)


@pytest.mark.parametrize("math_mode", ["relu_strict", "relu_fast"])
@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "heads,top_k,keys,score_dtype",
    [
        (4, 512, 8192, torch.float32),
        (32, 512, 8192, torch.float32),
        (32, 2048, 8192, torch.float32),
        (64, 512, 8192, torch.float32),
        (64, 1024, 8192, torch.float32),
        (64, 2048, 8192, torch.float32),
        (32, 512, 16384, torch.float32),
        (32, 512, 16384, torch.float16),
        (4, 512, 8192, torch.float16),
        (32, 2048, 8192, torch.float16),
        (64, 512, 8192, torch.float16),
        (64, 1024, 8192, torch.float16),
        (64, 2048, 8192, torch.float16),
        (4, 512, 8192, torch.bfloat16),
        (32, 512, 8192, torch.bfloat16),
        (32, 512, 16384, torch.bfloat16),
        (32, 2048, 8192, torch.bfloat16),
        (64, 512, 8192, torch.bfloat16),
        (64, 1024, 8192, torch.bfloat16),
        (64, 2048, 8192, torch.bfloat16),
    ],
)
def test_selective_layer_updates_and_capture(
    encoding, heads, top_k, keys, score_dtype, math_mode
):
    _blackwell()
    torch.manual_seed(13)
    rows = 64
    weights = torch.randn((rows, heads), device="cuda")
    if encoding == "fp8":
        q = torch.randn((rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn((keys, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(keys, device="cuda")
        q_scale = None
    else:
        q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
        k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        scales = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        qkv_dtype=encoding,
        logits_dtype=score_dtype,
        math_mode=math_mode,
        num_heads=heads,
        top_k=top_k,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv)
    result = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    eager_result = result.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    for step in range(3):
        if step == 2:
            weights.neg_()
            if encoding == "fp8":
                q.copy_(torch.randn(q.shape, device="cuda").to(q.dtype))
            else:
                q.bitwise_xor_(0x88)
                q_scale.fill_(128)
        if step:
            graph.replay()
        torch.cuda.synchronize()
        assert op.error.item() == 0 and op.repair_error.item() == 0
        current = eager_result if step == 0 else result
        qf = (
            q.float() if encoding == "fp8" else _decode_fp4(q) * (2 if step == 2 else 1)
        )
        kf = k.float() if encoding == "fp8" else _decode_fp4(k)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            scores = torch.einsum("rhd,kd->rhk", qf, kf).relu()
            scores = (scores * weights[:, :, None]).sum(1).to(score_dtype).float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
        scores.masked_fill_(
            torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
        )
        selected = scores.gather(1, current.long()).sort(1, descending=True).values
        expected = scores.topk(top_k, dim=1).values
        torch.testing.assert_close(selected, expected, rtol=1e-4, atol=1e-3)
        assert not torch.any(
            current.sort(1).values[:, 1:] == current.sort(1).values[:, :-1]
        )


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("preset", [False, True], ids=["generic", "preset"])
def test_selective_empty_initial_tile_and_causal_topk_boundary(encoding, top_k, preset):
    _blackwell()
    rows, heads, keys, ratio = 8192, 64, 2048, 4
    generator = torch.Generator(device="cuda").manual_seed(1)

    def quantize(count):
        values = torch.randn(
            (count, 128), device="cuda", dtype=torch.float32, generator=generator
        )
        if encoding == "fp8":
            return values.to(torch.float8_e4m3fn), None
        from flashinfer.quantization.fp4_quantization import mxfp4_quantize
        from flashinfer.tllm_enums import SfLayout

        packed, scales = mxfp4_quantize(
            values.to(torch.bfloat16), backend="cuda", sfLayout=SfLayout.layout_linear
        )
        return packed.view(count, 64), scales.view(count, 4)

    q, q_scales = quantize(rows * heads)
    q = q.view(rows, heads, -1)
    k, scales = quantize(keys)
    if encoding == "fp8":
        scales = torch.rand(keys, device="cuda", generator=generator) * 0.1 + 0.01
        q_scale, qf, kf = None, q.float(), k.float()
    else:
        q_scales = q_scales.view(rows, heads, 4)
        qf = _decode_fp4(q) * torch.exp2(q_scales.float() - 127).repeat_interleave(
            32, -1
        )
        kf = _decode_fp4(k) * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        q_scale = q_scales.contiguous().view(torch.int32).squeeze(-1)
    weights = (
        torch.rand(
            (rows, heads), device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.01
    )
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32)
    model = ("DS-V4-Flash" if top_k == 512 else "DS-V4-Pro") if preset else None
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        qkv_dtype=encoding,
        logits_dtype=torch.float32,
        model_config=model,
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=keys * ratio,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv, num_query_rows=rows)
    op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    graph.replay()
    torch.cuda.synchronize()
    assert op.error.item() == 0 and op.repair_error.item() == 0
    # The first persistent H64 tile is wholly empty. Later tiles cross K;
    # selecting every key in the early rows alone cannot expose stale scores.
    assert bool((selected[: ratio - 1] == -1).all())
    assert bool((op._capacity_plan._engine.counts[: ratio - 1] == 0).all())
    ends = torch.arange(1, rows + 1, device="cuda") // ratio
    assert bool((ends == top_k + 1).any()) and bool((ends > 1.05 * top_k).any())
    stride = 2 * torch.cuda.get_device_properties(q.device).multi_processor_count
    first_persistent_after_k = (
        (ratio * top_k + ratio - 1 + stride - 1) // stride * stride
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        positions = torch.arange(keys, device="cuda")
        for first in range(0, rows, 32):
            stop = min(first + 32, rows)
            scores = torch.einsum("rhd,kd->rhk", qf[first:stop], kf).relu()
            scores = (scores * weights[first:stop, :, None]).sum(1)
            if encoding == "fp8":
                scores.mul_(scales[None])
            scores.masked_fill_(positions[None] >= ends[first:stop, None], -torch.inf)
            indices = selected[first:stop].long()
            valid = indices >= 0
            assert torch.equal(valid.sum(1), ends[first:stop].clamp(max=top_k))
            assert bool(((indices < ends[first:stop, None]) | ~valid).all())
            ordered = indices.sort(1).values
            assert not bool(
                ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
            )
            actual = scores.gather(1, indices.clamp(min=0)).masked_fill(
                ~valid, -torch.inf
            )
            expected = scores.topk(top_k, dim=1).values
            query_rows = torch.arange(first, stop, device="cuda")
            exact_rows = (ends[first:stop] == top_k + 1) | (
                (query_rows >= first_persistent_after_k)
                & (query_rows < first_persistent_after_k + 2)
            )
            if bool(exact_rows.any()):
                torch.testing.assert_close(
                    actual[exact_rows].sort(1).values,
                    expected[exact_rows].sort(1).values,
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                actual.sort(1, descending=True).values, expected, rtol=1e-4, atol=1e-4
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
def test_selective_fast_contract_changes_fail_closed(encoding):
    _blackwell()
    rows, capacity_rows, keys, heads = 48, 64, 1024, 32
    weights = torch.full((rows, heads), 0.125, device="cuda")
    if encoding == "fp8":
        q = torch.ones((rows, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
        k = torch.ones((keys, 128), device="cuda", dtype=q.dtype)
        scales = torch.ones(keys, device="cuda")
        q_scale = None
    else:
        q = torch.full((rows, heads, 64), 0x22, device="cuda", dtype=torch.uint8)
        k = torch.full((keys, 64), 0x22, device="cuda", dtype=torch.uint8)
        scales = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        qkv_dtype=encoding,
        math_mode="relu_fast",
        num_heads=heads,
        max_context_length=keys,
        max_chunk_size=capacity_rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv, num_query_rows=rows)
    op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)

    # Bounds apply only to active rows; unused capacity may contain old data.
    buffers = op.input_buffers()
    buffers["weights"][rows:].fill_(float("nan"))
    if q_scale is not None:
        buffers["q_scale"][rows:].fill_(-1)

    for value in (0.0, 2**-125, 0.125):
        op.plan(qo, kv, num_query_rows=rows)
        weights.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert op.repair_error.item() == 0
        assert torch.all(result >= 0)
        ordered = result.sort(1).values
        assert not torch.any(ordered[:, 1:] == ordered[:, :-1])

    for value in (2**-130, float("nan"), float("inf"), -float("inf")):
        op.plan(qo, kv, num_query_rows=rows)
        weights.fill_(0.125)
        weights[0, 0] = value
        graph.replay()
        torch.cuda.synchronize()
        assert op.repair_error.item() == 16
        assert torch.all(result == -1)

    if q_scale is not None:
        weights.fill_(0.125)
        for change_q in (True, False):
            op.plan(qo, kv, num_query_rows=rows)
            if change_q:
                q_scale[0, 0, 0] = 160
            else:
                q_scale.fill_(127)
                scales[0, 0] = 160
            graph.replay()
            torch.cuda.synchronize()
            assert op.repair_error.item() == 16
            assert torch.all(result == -1)
        op.plan(qo, kv, num_query_rows=rows)
        q_scale.fill_(159)
        scales.fill_(159)
        graph.replay()
        torch.cuda.synchronize()
        assert op.repair_error.item() == 0
        assert torch.all(result >= 0)


@pytest.mark.parametrize("named", [False, True])
def test_selective_qwen_contract_changes_fail_closed(named):
    _blackwell()
    rows, keys, heads = 64, 8192, 4
    q = torch.ones((rows, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.ones((keys, 128), device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.full((rows, heads), 128**-0.5, device="cuda")
    scales = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * 4], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=heads,
        max_context_length=keys * 4,
        compression_ratio=4,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
        model_config="Qwen3.8-Flash-Next" if named else None,
    )
    op.plan(qo, kv)

    def check_valid(output):
        ends = (keys * 4 - rows + torch.arange(1, rows + 1, device="cuda")) // 4
        indices = output.long()
        assert bool(((indices >= 0) & (indices < ends[:, None])).all())
        ordered = indices.sort(1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
        # All keys are identical, so every selected score equals the row maximum.
        selected_k = k[indices].float()
        selected_scores = (q.float().sum(-1) * weights.float()).sum(-1)[:, None]
        selected_scores = selected_scores * selected_k[:, :, 0] * scales[indices]
        oracle = 128 * weights.float().sum(1) * scales[0]
        torch.testing.assert_close(selected_scores, oracle[:, None].expand(rows, 512))
        assert op.error.item() == op.repair_error.item() == 0

    initial = op.run(q, weights, k, k_scale=scales)
    torch.cuda.synchronize()
    check_valid(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op.run(q, weights, k, k_scale=scales)
    weights[0, 0] = 0
    graph.replay()
    torch.cuda.synchronize()
    if named:
        assert op.repair_error.item() == 16
        assert torch.all(result == -1)
    else:
        check_valid(result)
    weights.fill_(128**-0.5)
    op.plan(qo, kv)
    scales.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    if named:
        assert op.repair_error.item() == 16
        assert torch.all(result == -1)
    else:
        check_valid(result)


@pytest.mark.parametrize(
    "ratio,encoding,score_dtype",
    [
        (1, "fp8", torch.float32),
        (4, "fp8", torch.bfloat16),
        (4, "fp8", torch.float32),
        (4, "mxfp4", torch.float16),
        (1, "mxfp4", torch.bfloat16),
        (4, "mxfp4", torch.float32),
    ],
)
def test_selective_partial_storage_updates_and_short_rows(ratio, encoding, score_dtype):
    _blackwell()
    torch.manual_seed(19)
    rows, keys, heads = 64, 600, 64
    weights = torch.randn((rows, heads), device="cuda")
    if encoding == "fp8":
        q = torch.randn((rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn((keys, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(keys, device="cuda")
        q_scale = None
    else:
        q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
        k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        scales = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
    if encoding == "mxfp4":
        k = torch.cat((k, torch.zeros((168, 64), device="cuda", dtype=k.dtype)))
        scales = torch.cat(
            (scales, torch.full((168, 4), 127, device="cuda", dtype=scales.dtype))
        )
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=heads,
        qkv_dtype=encoding,
        logits_dtype=score_dtype,
        compression_ratio=ratio,
        max_context_length=keys * ratio,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv)
    op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.plan(qo, kv)
        result = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    for step in range(3):
        if step == 1:
            if encoding == "fp8":
                k.copy_(torch.randn(k.shape, device="cuda").to(k.dtype))
            else:
                k.bitwise_xor_(0x88)
        raw_context = keys * ratio if step < 2 else 128
        kv[1] = raw_context
        graph.replay()
        torch.cuda.synchronize()
        assert op.error.item() == 0 and op.repair_error.item() == 0
        qf = q.float() if encoding == "fp8" else _decode_fp4(q)
        kf = k[:keys].float() if encoding == "fp8" else _decode_fp4(k[:keys])
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            scores = torch.einsum("rhd,kd->rhk", qf, kf).relu()
            scores = (scores * weights[:, :, None]).sum(1).to(score_dtype).float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        ends = (raw_context - rows + torch.arange(1, rows + 1, device="cuda")) // ratio
        scores.masked_fill_(
            torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
        )
        valid = torch.arange(512, device="cuda")[None, :] < ends.clamp(max=512)[:, None]
        assert torch.equal(result >= 0, valid)
        assert torch.all(torch.where(valid, result < ends[:, None], result == -1))
        selected = scores.gather(1, result.clamp_min(0).long()).masked_fill(
            ~valid, -torch.inf
        )
        torch.testing.assert_close(
            selected.sort(1, descending=True).values,
            scores.topk(512, dim=1).values,
            rtol=1e-4,
            atol=1e-3,
        )
        ordered = result.sort(1).values
        assert not torch.any(
            (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, :-1] >= 0)
        )


@pytest.mark.parametrize("logits_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("math_mode", ["relu_strict", "relu_fast"])
def test_selective_nan_scores_rank_as_negative_infinity(logits_dtype, math_mode):
    _blackwell()
    torch.manual_seed(17)
    rows, keys, heads = 64, 8192, 64
    q = torch.randn((rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((keys, 128), device="cuda").to(torch.float8_e4m3fn)
    k[0, 0] = float("nan")
    q[0, 0, 0] = float("nan")
    weights = torch.ones((rows, heads), device="cuda")
    scales = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        logits_dtype=logits_dtype,
        math_mode=math_mode,
        num_heads=heads,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv)
    result = op.run(q, weights, k, k_scale=scales)
    torch.cuda.synchronize()
    assert op.error.item() == 0 and op.repair_error.item() == 0
    assert torch.all(result >= 0)
    assert torch.all(result < keys)
    assert not torch.any(result[1:] == 0)
    ordered = result.sort(1).values
    assert not torch.any(ordered[:, 1:] == ordered[:, :-1])


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, -float("inf"), float("nan")])
@pytest.mark.parametrize("warp", [False, True])
def test_selective_sample_proposal_constant_range_with_padding(value, warp):
    _blackwell()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.proposal_parallel import (
        propose_cached,
    )
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.proposal_warp import (
        propose_warp,
    )
    from flashinfer.experimental.attn_scores._topk_runtime import (
        _compile_for_device,
        _current_stream,
    )

    class Proposal:
        def __init__(self, warp):
            self.warp = warp

        @cute.jit
        def __call__(
            self,
            scores: cute.Tensor,
            thresholds: cute.Tensor,
            info: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(scores, thresholds, info).launch(
                grid=(1, 1, 1), block=(32 if self.warp else 128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(
            self, scores: cute.Tensor, thresholds: cute.Tensor, info: cute.Tensor
        ):
            allocator = cutlass.memory.SmemAllocator()
            hist = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout(264), byte_alignment=16
            )
            scalars = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout(2), byte_alignment=16
            )
            tid = cute.arch.thread_idx()[0]
            if cutlass.const_expr(self.warp):
                propose_warp(
                    cutlass.Int32(0),
                    tid,
                    scores,
                    hist,
                    thresholds,
                    cutlass.Int32(512),
                    cutlass.Int32(64),
                    cutlass.Int32(1),
                    info,
                )
            else:
                propose_cached(
                    cutlass.Int32(0),
                    tid,
                    scores,
                    hist,
                    scalars,
                    thresholds,
                    cutlass.Int32(512),
                    cutlass.Int32(64),
                    4,
                    0,
                    cutlass.Int32(1),
                    info,
                )

    scores = torch.full((1, 512), -torch.inf, device="cuda")
    scores[:, :128] = value
    thresholds = torch.empty(1, device="cuda")
    info = torch.empty((1, 288), device="cuda")
    tensors = tuple(from_dlpack(x, assumed_align=4) for x in (scores, thresholds, info))
    stream = _current_stream(scores.device)
    compiled = _compile_for_device(scores.device, Proposal(warp), *tensors, stream)
    compiled(*tensors, stream)
    torch.cuda.synchronize()
    assert thresholds.item() == (-float("inf") if math.isnan(value) else value)


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("preset", [False, True], ids=["generic", "preset"])
def test_selective_persistent_capacity_padding_exact_keys(encoding, top_k, preset):
    _check_selective_capacity_padding(encoding, top_k, preset)


def test_selective_h32_persistent_capacity_padding_exact_keys():
    _check_selective_capacity_padding(
        "fp8", 512, False, heads=32, ratio=1, rows=2372, capacity=2376
    )


@pytest.mark.parametrize(
    "heads,top_k,keys,rows,ratio,logits_dtype",
    [
        (4, 512, 4096, 4096, 4, torch.float16),
        (32, 512, 24576, 4096, 4, torch.float16),
        (32, 2048, 4096, 2048, 1, torch.float32),
    ],
)
def test_selective_fp8_filter_variants_retire_capacity_padding(
    heads, top_k, keys, rows, ratio, logits_dtype
):
    # Exercise packed H4, mid-context H32 and full-tile H32 filters. Empty
    # capacity tiles precede live tiles in the persistent reverse traversal.
    _check_selective_capacity_padding(
        "fp8",
        top_k,
        False,
        heads=heads,
        ratio=ratio,
        rows=rows,
        capacity=8192,
        keys=keys,
        logits_dtype=logits_dtype,
    )


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "heads,top_k,keys,rows,ratio",
    [
        (4, 512, 4096, 4096, 4),
        (32, 2048, 4096, 2048, 1),
        (32, 512, 4096, 4096, 4),
        (32, 512, 32768, 4096, 4),
    ],
)
def test_selective_fp4_filter_variants_retire_capacity_padding(
    heads, top_k, keys, rows, ratio, logits_dtype
):
    _check_selective_capacity_padding(
        "mxfp4",
        top_k,
        False,
        heads=heads,
        ratio=ratio,
        rows=rows,
        capacity=8192,
        keys=keys,
        logits_dtype=logits_dtype,
    )


def _check_selective_capacity_padding(
    encoding,
    top_k,
    preset,
    *,
    heads=64,
    ratio=4,
    rows=4096,
    capacity=8192,
    keys=4096,
    logits_dtype=torch.float32,
):
    _blackwell()
    positions = torch.arange(keys, device="cuda", dtype=torch.int64)
    rotations = torch.arange(rows, device="cuda", dtype=torch.int64) % 3
    # Twelve independent key bits give unique scores. Rotate the query weights
    # across persistent iterations so stale scores cannot preserve key order.
    bits = torch.zeros((keys, 128), device="cuda", dtype=torch.float32)
    query = torch.zeros((rows, 128), device="cuda", dtype=torch.float32)
    for block in range(4):
        for bit in range(3):
            bits[:, block * 32 + bit] = (positions >> (block * 3 + bit)) & 1
            query[:, block * 32 + bit] = torch.exp2(
                (3 * ((block + rotations) % 4) + bit - 3).float()
            )
    if encoding == "fp8":
        q = query[:, None].expand(-1, heads, -1).contiguous().to(torch.float8_e4m3fn)
        k = bits.to(torch.float8_e4m3fn)
        scales, q_scale = torch.ones(keys, device="cuda"), None
        reference_q = query
    else:
        query_nibbles = torch.zeros((rows, 128), device="cuda", dtype=torch.uint8)
        query_scales = torch.empty((rows, 4), device="cuda", dtype=torch.uint8)
        for block in range(4):
            query_nibbles[:, block * 32 : block * 32 + 3] = torch.tensor(
                [2, 4, 6], device="cuda", dtype=torch.uint8
            )
            query_scales[:, block] = 127 + 3 * ((block + rotations) % 4)
        packed_q = query_nibbles[:, 0::2] | (query_nibbles[:, 1::2] << 4)
        q = packed_q[:, None].expand(-1, heads, -1).contiguous()
        key_nibbles = bits.to(torch.uint8) * 2
        k = key_nibbles[:, 0::2] | (key_nibbles[:, 1::2] << 4)
        scales = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale = (
            query_scales[:, None]
            .expand(-1, heads, -1)
            .contiguous()
            .view(torch.int32)
            .squeeze(-1)
        )
        reference_q = query * 8
    weight = 128**-0.5 if (heads, top_k, ratio) == (4, 512, 4) else 1 / heads
    weights = torch.full((rows, heads), weight, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32)
    model = ("DS-V4-Flash" if top_k == 512 else "DS-V4-Pro") if preset else None
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        qkv_dtype=encoding,
        logits_dtype=logits_dtype,
        model_config=model,
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=keys * ratio,
        max_chunk_size=capacity,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
    )
    op.plan(qo, kv, num_query_rows=rows)
    # Nonzero padding must still publish no candidates. The empty range is at
    # the physical KV end, so a consumed dummy also exercises scale bounds.
    buffers = op.input_buffers()
    buffers["q"][rows:].fill_(1)
    buffers["weights"][rows:].fill_(1 / heads)
    if encoding == "mxfp4":
        buffers["q_scale"][rows:].fill_(0x7F7F7F7F)
    op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    graph.replay()
    torch.cuda.synchronize()
    engine = op._capacity_plan._engine
    assert bool((engine.output[rows:] == -1).all())
    assert bool((engine.counts[rows:] == 0).all())
    assert bool((op._capacity_plan._selected[rows:] == -1).all())
    ends = (keys * ratio - rows + torch.arange(1, rows + 1, device="cuda")) // ratio
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Every head has identical Q and a common weight, so the independent
        # FP32 reference is the one-head dot product times heads * weight.
        for first in range(0, rows, 32):
            stop = min(first + 32, rows)
            scores = (reference_q[first:stop] @ bits.T) * (heads * weight)
            scores.masked_fill_(positions[None] >= ends[first:stop, None], -torch.inf)
            indices = selected[first:stop].long()
            assert bool(((indices >= 0) & (indices < ends[first:stop, None])).all())
            ordered = indices.sort(1).values
            assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
            # Ranking precision and repeated key patterns can introduce ties.
            scores = scores.to(logits_dtype)
            expected = scores.topk(top_k, dim=1).values.sort(1).values
            actual = scores.gather(1, indices).sort(1).values
            assert torch.equal(actual, expected), (encoding, top_k, preset, first)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    assert op.error.item() == 0 and op.repair_error.item() == 0


@pytest.mark.parametrize(
    "encoding,math_mode,logits_dtype,weights_dtype,heads,top_k",
    [
        ("mxfp8", "relu_strict", torch.float32, torch.float32, 64, 512),
        ("mxfp8", "relu_strict", torch.bfloat16, torch.bfloat16, 64, 512),
        ("fp8", "relu_strict", torch.float16, torch.bfloat16, 64, 512),
        ("fp8", "native_bf16", torch.bfloat16, torch.bfloat16, 64, 512),
        ("fp8", "native_fp16", torch.float16, torch.bfloat16, 64, 512),
        ("mxfp8", "native_fp16", torch.float16, torch.bfloat16, 4, 1024),
        ("mxfp8", "native_bf16", torch.bfloat16, torch.bfloat16, 32, 2048),
        ("mxfp4", "native_fp16", torch.float16, torch.bfloat16, 4, 1024),
        ("mxfp4", "native_bf16", torch.bfloat16, torch.bfloat16, 32, 2048),
        ("fp8", "relu_strict", torch.float32, torch.bfloat16, 4, 1024),
    ],
)
def test_cutedsl_precision_modes_against_oracle(
    encoding, math_mode, logits_dtype, weights_dtype, heads, top_k, ratio=1
):
    """Exactly representable operands expose block scaling and lowp accumulation."""
    _blackwell()
    rows, keys = 64, 4096
    torch.manual_seed(613)
    q = torch.zeros((rows, heads, 128), device="cuda", dtype=torch.float32)
    q[:, :, 0] = 1
    if encoding == "mxfp8":
        q[:, :, (32, 64, 96)] = 1
    q = q.to(torch.float8_e4m3fn)
    k_float = torch.zeros((keys, 128), device="cuda", dtype=torch.float32)
    k_float[:, 0] = (torch.arange(keys, device="cuda") % 16 + 1) / 16
    if encoding == "mxfp8":
        k_float[:, (32, 64, 96)] = k_float[:, 0:1]
    k = k_float.to(torch.float8_e4m3fn)
    key_values = k[:, 0].float()
    if encoding == "mxfp4":
        q = torch.zeros((rows, heads, 64), device="cuda", dtype=torch.uint8)
        q[:, :, (0, 16, 32, 48)] = 2  # E2M1 value 1.0 in each scale block.
        k = torch.zeros((keys, 64), device="cuda", dtype=torch.uint8)
        codes = (torch.arange(keys, device="cuda") % 6 + 1).to(torch.uint8)
        k[:, (0, 16, 32, 48)] = codes[:, None]
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4], device="cuda")
        key_values = values[codes.long()]
    if math_mode == "relu_strict" and weights_dtype == torch.bfloat16:
        storage = torch.empty(
            (rows, 8 if heads == 4 else heads + 8), device="cuda", dtype=weights_dtype
        )
        weights = storage[:, :heads]
        weights.copy_(torch.randint(1, 9, (rows, heads), device="cuda").float() / 8)
    else:
        weights = torch.ones((rows, heads), device="cuda", dtype=weights_dtype)
    q_factor = torch.ones(rows, device="cuda")
    if encoding in ("mxfp8", "mxfp4"):
        scale_bytes = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
        scale_bytes[::2, :] = 128
        q_exponents = torch.randint(126, 129, (rows,), device="cuda", dtype=torch.uint8)
        q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
        q_scale[:, :, :] = q_exponents[:, None, None]
        q_factor = torch.exp2(q_exponents.float() - 127)
        q_scale = q_scale.contiguous().view(torch.int32).squeeze(-1)
        k_scale = scale_bytes
        scales = torch.exp2(scale_bytes[:, 0].float() - 127)
    else:
        k_scale = torch.ones(keys, device="cuda")
        q_scale = None
        scales = k_scale
    qo = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, keys * ratio], dtype=torch.int32, device="cuda")
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=heads,
        top_k=top_k,
        max_context_length=keys * ratio,
        compression_ratio=ratio,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
        qkv_dtype=encoding,
        math_mode=math_mode,
        logits_dtype=logits_dtype,
        weights_dtype=weights_dtype,
    )
    op.plan(qo, kv)
    eager = op.run(q, weights, k, k_scale=k_scale, q_scale=q_scale).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = op.run(q, weights, k, k_scale=k_scale, q_scale=q_scale)

    def check(indices):
        row_factor = weights.float().sum(1) * q_factor
        if encoding in ("mxfp4", "mxfp8"):
            row_factor = row_factor * 4
        scores = row_factor[:, None] * (key_values * scales)[None, :]
        expected = scores.to(logits_dtype).float()
        ends = (keys * ratio - rows + torch.arange(1, rows + 1, device="cuda")) // ratio
        expected.masked_fill_(
            torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
        )
        idx = indices.long()
        assert bool(((idx >= 0) & (idx < ends[:, None])).all())
        ordered = idx.sort(1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
        actual = expected.gather(1, idx).sort(1, descending=True).values
        torch.testing.assert_close(
            actual, expected.topk(top_k, dim=1).values, rtol=0, atol=0
        )
        assert op.error.item() == op.repair_error.item() == 0

    check(eager)
    graph.replay()
    torch.cuda.synchronize()
    check(replay)


def test_shared_fp8_filter_against_oracle():
    """Shared-KV H64 matches the independent score oracle."""
    _blackwell()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("shared-KV query tiles require SM100")
    rows = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    keys, heads, top_k = 65536, 64, 512
    torch.manual_seed(622)
    q = torch.zeros((rows, heads, 128), device="cuda")
    q[:, :, 0] = torch.randint(8, 13, (rows, heads), device="cuda") / 8
    q = q.to(torch.float8_e4m3fn)
    k = torch.zeros((keys, 128), device="cuda")
    k[:top_k, 0] = torch.randint(8, 13, (top_k,), device="cuda") / 8
    k = k.to(torch.float8_e4m3fn)
    weights = torch.ones((rows, heads), device="cuda")
    scale = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    options = dict(
        backend="cutedsl",
        num_heads=heads,
        top_k=top_k,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
        math_mode="native_fp16",
        logits_dtype=torch.float16,
    )

    def build():
        op = FP8MQATopKWrapper(**options)
        op.plan(qo, kv)
        output = op.run(q, weights, k, k_scale=scale).clone()
        ordered = output.long().sort(1).values
        expected = torch.arange(top_k, device="cuda").expand(rows, -1)
        assert torch.equal(ordered, expected)
        assert op.error.item() == op.repair_error.item() == 0
        return op

    shared = build()
    assert shared._capacity_plan._engine.shared_fp8_queries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = shared.run(q, weights, k, k_scale=scale)
    graph.replay()
    torch.cuda.synchronize()
    reference = q.float()[:, :, 0].sum(1)[:, None] * k.float()[:, 0][None, :]
    selected_scores = reference.gather(1, replay.long()).sort(1).values
    expected_scores = reference.topk(top_k, dim=1).values.sort(1).values
    torch.testing.assert_close(selected_scores, expected_scores, rtol=0, atol=0)


@pytest.mark.parametrize(
    "math_mode,dtype", [("native_bf16", torch.bfloat16), ("native_fp16", torch.float16)]
)
def test_native_head_accumulation_matches_rounded_fma_oracle(math_mode, dtype):
    """Exercise rounding at every low-precision FMA and at the final tree."""
    _blackwell()
    torch.manual_seed(79)
    rows, heads, keys, top_k = 64, 64, 1024, 512
    q = (torch.randn((rows, heads, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k = (torch.randn((keys, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    weights = (torch.randn((rows, heads), device="cuda") * 0.2).to(torch.bfloat16)
    scale = torch.where(torch.arange(keys, device="cuda") % 2 == 0, 0.5, 1.0)
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=heads,
        top_k=top_k,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
        math_mode=math_mode,
        logits_dtype=dtype,
        weights_dtype=torch.bfloat16,
    )
    op.plan(qo, kv)
    eager = op.run(q, weights, k, k_scale=scale).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = op.run(q, weights, k, k_scale=scale)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dots = torch.einsum("rhd,kd->rhk", q.float(), k.float()).relu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    fp32_scores = (dots * weights.float()[:, :, None]).sum(1) * scale
    factor = 512 if math_mode == "native_fp16" else 1
    dots = (dots / factor).to(dtype).double()
    rounded_weights = (weights.float() * factor).to(dtype).double()
    chains = [
        torch.zeros((rows, keys), device="cuda", dtype=torch.float64) for _ in range(4)
    ]
    for h in range(heads):
        i = h % 4
        chains[i] = (
            (dots[:, h, :] * rounded_weights[:, h, None] + chains[i]).to(dtype).double()
        )
    left = (chains[0] + chains[2]).to(dtype).double()
    right = (chains[1] + chains[3]).to(dtype).double()
    scores = (left + right).to(dtype).float() * scale
    scores = scores.to(dtype).float()
    ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
    scores.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
    )
    fp32_scores.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
    )
    native_set = scores.topk(top_k, dim=1).indices.sort(1).values
    fp32_set = fp32_scores.topk(top_k, dim=1).indices.sort(1).values
    assert bool((native_set != fp32_set).any()), "fixture must distinguish native math"

    def check(indices):
        idx = indices.long()
        assert bool(((idx >= 0) & (idx < ends[:, None])).all())
        ordered = idx.sort(1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
        actual = scores.gather(1, idx)
        kth = scores.topk(top_k, dim=1).values[:, -1:]
        # The reference does not model the MMA dot-product accumulation order.
        tolerance = 2e-3 + kth.abs() * 2e-3
        assert bool((actual >= kth - tolerance).all())
        assert op.error.item() == op.repair_error.item() == 0

    check(eager)
    graph.replay()
    torch.cuda.synchronize()
    check(replay)


def test_module_retirement_is_deferred_during_caller_capture(monkeypatch):
    """GC of a cyclic wrapper cannot unload its CUDA library during capture."""
    import gc
    import weakref
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitModule
    from flashinfer.experimental.attn_scores import _topk_capacity as selective_capacity

    _blackwell()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    calls = []
    original = CudaDialectJitModule.unload

    def observe(module):
        if module.cuda_library:
            calls.append((id(module), torch.cuda.is_current_stream_capturing()))
        return original(module)

    monkeypatch.setattr(CudaDialectJitModule, "unload", observe)
    try:
        rows, heads, keys = 64, 64, 1024
        q = torch.ones((rows, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
        k = torch.ones((keys, 128), device="cuda", dtype=torch.float8_e4m3fn)
        weights = torch.ones((rows, heads), device="cuda")
        scale = torch.ones(keys, device="cuda")
        qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)

        def build():
            op = FP8MQATopKWrapper(
                strategy="selective",
                backend="cutedsl",
                num_heads=heads,
                max_context_length=keys,
                max_chunk_size=rows,
                max_batch_size=1,
                num_keys=keys,
                device="cuda",
            )
            op.plan(qo, kv)
            selected = op.run(q, weights, k, k_scale=scale)
            ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
            ids = selected.long()
            assert bool(((ids >= 0) & (ids < ends[:, None])).all())
            ordered = ids.sort(1).values
            assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
            selected_scores = k[ids, 0].float() * weights.sum(1)[:, None] * 128
            torch.testing.assert_close(
                selected_scores, torch.full_like(selected_scores, 8192)
            )
            assert op.error.item() == op.repair_error.item() == 0
            return op

        op, successor = build(), build()
        compiled = op._capacity_plan._state._compiled
        module_id, module_ref = (
            id(compiled.jit_module),
            weakref.ref(compiled.jit_module),
        )
        owner_ref = weakref.ref(op)
        gc.collect()  # Collect unrelated predecessors while the owner remains live.
        cycle = [op]
        cycle.append(cycle)
        del op, compiled
        sentinel = torch.zeros(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            del cycle
            gc.collect()
            sentinel.fill_(1)
        assert owner_ref() is None
        assert module_id in {
            id(module) for module in selective_capacity._deferred_metadata_modules
        }
        assert not any(mid == module_id for mid, _ in calls)
        graph.replay()
        torch.cuda.synchronize()
        assert sentinel.item() == 1
        # A normal eager boundary drains deferred modules without changing scores.
        successor.plan(qo, kv)
        output = successor.run(q, weights, k, k_scale=scale)
        ids = output.long()
        assert bool(
            (
                (ids >= 0)
                & (
                    ids
                    < keys - rows + torch.arange(1, rows + 1, device="cuda")[:, None]
                )
            ).all()
        )
        assert not bool((ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).any())
        selected_scores = k[ids, 0].float() * weights.sum(1)[:, None] * 128
        torch.testing.assert_close(
            selected_scores, torch.full_like(selected_scores, 8192)
        )
        assert (module_id, False) in calls
        assert module_ref() is None or module_ref().is_unloaded()
        assert not any(capturing for _, capturing in calls)
    finally:
        if enabled:
            gc.enable()


def test_cross_thread_gc_defers_unload_until_plan(monkeypatch):
    """Another thread may collect a wrapper while the caller captures."""
    import gc
    import threading
    import weakref
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitModule
    from flashinfer.experimental.attn_scores._topk_capacity import (
        _drain_deferred_metadata_modules,
    )

    _blackwell()
    rows, heads, keys = 64, 64, 1024
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    options = dict(
        strategy="selective",
        num_heads=heads,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
    )
    owner = FP8MQATopKWrapper(**options)
    successor = FP8MQATopKWrapper(**options)
    owner.plan(qo, kv)
    successor.plan(qo, kv)
    q = torch.ones((rows, heads, 128), dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.ones((keys, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones((rows, heads), device="cuda")
    scale = torch.ones(keys, device="cuda")
    result = owner.run(q, weights, k, k_scale=scale)
    ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
    assert bool(((result >= 0) & (result < ends[:, None])).all())
    assert not bool(
        (result.sort(1).values[:, 1:] == result.sort(1).values[:, :-1]).any()
    )
    selected_scores = k[result.long(), 0].float() * weights.sum(1)[:, None] * 128
    torch.testing.assert_close(selected_scores, torch.full_like(selected_scores, 8192))
    torch.cuda.synchronize()
    module = owner._capacity_plan._state._compiled.jit_module
    module_id = id(module)
    owner_ref = weakref.ref(owner)
    original = CudaDialectJitModule.unload
    calls = []

    def observe(module):
        if id(module) == module_id and module.cuda_library:
            calls.append(torch.cuda.is_current_stream_capturing())
        return original(module)

    monkeypatch.setattr(CudaDialectJitModule, "unload", observe)
    gc.collect()
    _drain_deferred_metadata_modules()
    enabled = gc.isenabled()
    gc.disable()
    cycle = [owner]
    cycle.append(cycle)
    holder = [cycle]
    del owner, cycle, module
    ready, start, done = threading.Event(), threading.Event(), threading.Event()

    def worker():
        torch.cuda.set_device(0)
        ready.set()
        start.wait()
        try:
            holder.clear()
            gc.collect()
        finally:
            done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(30)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    sentinel = torch.zeros(1, device="cuda")
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin()
            start.set()
            assert done.wait(30)
            assert owner_ref() is None
            assert calls == []
            sentinel.fill_(1)
            graph.capture_end()
        successor.plan(qo, kv)
        assert calls == [False]
        graph.replay()
        torch.cuda.synchronize()
        assert sentinel.item() == 1
    finally:
        start.set()
        thread.join(timeout=30)
        if enabled:
            gc.enable()


def test_retirement_failure_does_not_block_remaining_modules(monkeypatch, caplog):
    from flashinfer.experimental.attn_scores import _topk_capacity as capacity

    events = []

    class Module:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def unload(self):
            events.append(self.name)
            if self.fail:
                raise RuntimeError("unload failed")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(capacity, "_deferred_metadata_modules", [])
    capacity._retire_metadata_module(Module("bad", fail=True))
    capacity._retire_metadata_module(Module("good"))
    capacity._drain_deferred_metadata_modules()
    assert events == ["bad", "good"]
    assert capacity._deferred_metadata_modules == []
    assert "Failed to unload a retired CUDA metadata module" in caplog.text


def test_shared_mxfp4_filter_graph_matches_oracle():
    """Exercise the shared-KV route with independent eager and replay checks."""
    _blackwell()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("shared-KV query tiles require SM100")
    rows = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    heads, keys, top_k = 64, 65536, 512
    torch.manual_seed(623)
    q = torch.zeros((rows, heads, 64), device="cuda", dtype=torch.uint8)
    q[:, :, 0] = 2
    k = torch.zeros((keys, 64), device="cuda", dtype=torch.uint8)
    codes = torch.randint(2, 5, (top_k,), device="cuda", dtype=torch.uint8)
    k[:top_k, 0] = codes
    q_scale = torch.full((rows, heads, 4), 127, device="cuda", dtype=torch.uint8)
    q_scale = q_scale.view(torch.int32).squeeze(-1)
    k_scale = torch.full((keys, 4), 127, device="cuda", dtype=torch.uint8)
    weights = torch.ones((rows, heads), device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * 4], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        num_heads=heads,
        top_k=top_k,
        compression_ratio=4,
        qkv_dtype="mxfp4",
        logits_dtype=torch.float16,
        math_mode="native_fp16",
        model_config="DS-V4-Flash",
        max_context_length=keys * 4,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
    )
    op.plan(qo, kv)
    eager = op.run(q, weights, k, k_scale=k_scale, q_scale=q_scale).clone()
    assert op._capacity_plan._engine.shared_kv_queries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = op.run(q, weights, k, k_scale=k_scale, q_scale=q_scale)
    graph.replay()
    torch.cuda.synchronize()
    values = torch.tensor([0, 0.5, 1, 1.5, 2], device="cuda")
    reference = values[k[:, 0].long()] * heads
    for result in (eager, replay):
        ids = result.long()
        assert bool(((ids >= 0) & (ids < keys)).all())
        ordered = ids.sort(1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
        actual = reference[ids].sort(1).values
        expected = reference.topk(top_k).values.sort().values.expand(rows, -1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert op.error.item() == op.repair_error.item() == 0


def test_selective_strict_and_fast_instances_alternate_against_oracle():
    _blackwell()
    torch.manual_seed(642)
    rows, heads, keys, top_k = 64, 64, 1024, 512
    q = (torch.randn((rows, heads, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k = (torch.randn((keys, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    weights = torch.rand((rows, heads), device="cuda")
    scales = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    options = dict(
        num_heads=heads,
        top_k=top_k,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
    )
    wrappers = [
        FP8MQATopKWrapper(math_mode=mode, **options)
        for mode in ("relu_strict", "relu_fast")
    ]
    for op in wrappers:
        op.plan(qo, kv)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        scores = (
            torch.einsum("rhd,kd->rhk", q.float(), k.float()).relu()
            * weights[:, :, None]
        ).sum(1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
    scores.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] >= ends[:, None], -torch.inf
    )
    for index in (0, 1, 0, 1):
        result = wrappers[index].run(q, weights, k, k_scale=scales).long()
        assert bool(((result >= 0) & (result < ends[:, None])).all())
        ordered = result.sort(1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
        actual = scores.gather(1, result).sort(1, descending=True).values
        torch.testing.assert_close(
            actual, scores.topk(top_k, dim=1).values, atol=1e-3, rtol=1e-4
        )


@_requires_selective_sm100
def test_continuous_all_partial_request_tails_against_oracle():
    torch.manual_seed(641)
    lengths = list(range(1, 16))
    heads, keys_per_request = 64, 640
    rows = 16
    q = (torch.randn((rows, heads, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device="cuda")
    requests = 1
    keys = (torch.randn((requests * keys_per_request, 128), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand((requests * keys_per_request,), device="cuda") + 0.5
    op = FP8MQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=512,
        max_context_length=keys_per_request,
        max_chunk_size=rows,
        max_batch_size=requests,
        num_keys=keys.shape[0],
        device="cuda",
    )

    def run(length_group, q_live, weights_live):
        qp = [0]
        kp = [0]
        for length in length_group:
            qp.append(qp[-1] + length)
            kp.append(kp[-1] + keys_per_request)
        qo = torch.tensor(qp, device="cuda", dtype=torch.int32)
        kv = torch.tensor(kp, device="cuda", dtype=torch.int32)
        op.plan(qo, kv, num_query_rows=q_live.shape[0])
        selected = op.run(q_live, weights_live, keys, k_scale=scales)
        reference = _rows(
            q_live,
            weights_live,
            keys,
            scales,
            length_group,
            [keys_per_request] * len(length_group),
            1,
        )
        _check_selection(selected, reference, 512)
        return qo, kv, reference

    for length in lengths:
        q_live, w_live = q[:length], weights[:length]
        qo, kv, reference = run([length], q_live, w_live)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.plan(qo, kv, num_query_rows=q_live.shape[0])
        result = op.run(q_live, w_live, keys, k_scale=scales)
    graph.replay()
    _check_selection(result, reference, 512)


def test_continuous_full_rejected_before_initialization():
    with pytest.raises(ValueError, match="selective only"):
        FP8MQATopKWrapper(
            strategy="full",
            num_heads=64,
            max_context_length=1024,
            max_chunk_size=4,
            max_batch_size=1,
            device="cuda",
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    "encoding,expected",
    [("fp8", "relu_fast"), ("mxfp4", "relu_fast"), ("mxfp8", "relu_strict")],
)
@pytest.mark.parametrize("explicit_none", [False, True])
def test_continuous_selective_default_math_mode(encoding, expected, explicit_none):
    options = {"math_mode": None} if explicit_none else {}
    wrapper = FP8MQATopKWrapper(
        num_heads=64,
        qkv_dtype=encoding,
        max_context_length=1024,
        max_chunk_size=4,
        max_batch_size=1,
        device="cuda",
        **options,
    )
    assert wrapper.math_mode == expected


@pytest.mark.parametrize(
    "num_keys,valid", [(1024 * 1024, True), (1024 * 1024 + 1, False)]
)
def test_selective_key_capacity_boundary(num_keys, valid):
    _blackwell()
    options = dict(
        num_heads=64,
        max_context_length=1024,
        max_chunk_size=4,
        max_batch_size=1,
        device="cuda",
        num_keys=num_keys,
    )
    if valid:
        assert FP8MQATopKWrapper(**options).input_buffers()["q"].shape[0] == 4
    else:
        with pytest.raises(ValueError, match="1048576"):
            FP8MQATopKWrapper(**options)


def _typed_model_scores(q, k, weights, mode, dtype, fp8):
    if len(q) > 64:
        return torch.cat(
            [
                _typed_model_scores(
                    q[i : i + 64], k, weights[i : i + 64], mode, dtype, fp8
                )
                for i in range(0, len(q), 64)
            ]
        )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dots = torch.einsum("rhd,kd->rhk", q.float(), k.float()).relu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    native = {"native_bf16": torch.bfloat16, "native_fp16": torch.float16}.get(mode)
    factor = 512 if fp8 and mode == "native_fp16" else 1
    if native is not None:
        dots = (dots / factor).to(native).double()
        weights = (weights.float() * factor).to(native).double()
    else:
        dots, weights = dots.double(), weights.double()
    chains = [torch.zeros_like(dots[:, 0]) for _ in range(4)]
    for h in range(32):
        chains[h % 4] = (
            (dots[:, h] * weights[:, h, None] + chains[h % 4])
            .to(native or torch.float32)
            .double()
        )
    if native is not None:
        scores = (
            (
                (chains[0] + chains[2]).to(native).double()
                + (chains[1] + chains[3]).to(native).double()
            )
            .to(native)
            .float()
        )
    else:
        scores = ((chains[0].float() + chains[1].float()) + chains[2].float()) + chains[
            3
        ].float()
    return scores.to(dtype).float()


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("path", ["kimi", "ds_full", "ds_publish", "ds_consume"])
@pytest.mark.parametrize("math_mode", ["relu_strict", "native_bf16", "native_fp16"])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_model_logits_and_reduction_matrix(
    path,
    math_mode,
    logits_dtype,
    weights_dtype,
    weight_scale=1.0,
    default_weights=False,
    ties=False,
    keys_override=None,
    rows_override=None,
):
    _blackwell()
    torch.manual_seed(341)
    rows, heads = rows_override or 16, 32
    keys = 20736 if path == "ds_publish" else 32768 if path == "ds_consume" else 4096
    if keys_override is not None:
        keys = keys_override
    fp8 = path in ("kimi", "glm")
    top_k = 2048 if fp8 else 512
    ratio = 2 if path == "ds_full" else 1
    if fp8:
        q = (torch.randn(rows, heads, 128, device="cuda") * 0.125).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(keys, 128, device="cuda") * 0.125).to(torch.float8_e4m3fn)
        qr, kr = q.float(), k.float()
        qs, ks = None, torch.ones(keys, device="cuda")
    else:
        q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
        k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        qs = torch.full((rows, heads, 4), 124, device="cuda", dtype=torch.uint8)
        ks = torch.full((keys, 4), 124, device="cuda", dtype=torch.uint8)
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        qr = (
            table[torch.stack((q & 15, q >> 4), -1).long()].reshape(rows, heads, 128)
            * 0.125
        )
        kr = table[torch.stack((k & 15, k >> 4), -1).long()].reshape(keys, 128) * 0.125
    weights = (
        (torch.rand(rows, heads, device="cuda") * 0.25 - 0.0625) * weight_scale
    ).to(weights_dtype)
    if ties:
        weights.zero_()
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32)
    blocks = None
    if path == "ds_consume":
        blocks = torch.full((rows, 2048), 2147483647, device="cuda", dtype=torch.int32)
        blocks[:, :2047] = torch.arange(2047, device="cuda", dtype=torch.int32) * 2
        blocks[:, -1] = (
            keys - rows + torch.arange(rows, device="cuda", dtype=torch.int32)
        ) // 8
    op = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=keys * ratio,
        max_chunk_size=rows,
        max_batch_size=1,
        num_keys=keys,
        device="cuda",
        qkv_dtype="fp8" if fp8 else "mxfp4",
        math_mode=math_mode,
        logits_dtype=logits_dtype,
        **({} if default_weights else {"weights_dtype": weights.dtype}),
        model_config=("GLM-5.3" if path == "glm" else "Kimi-K3.1-Pro")
        if fp8
        else "DS-V4.1-Flash",
        publish_candidate_blocks=path == "ds_publish",
        candidate_capacity=2048 if blocks is not None else 0,
    )
    op.plan(qo, kv)
    kwargs = dict(k_scale=ks, q_scale=qs, candidate_blocks=blocks)
    eager = op.run(q, weights, k, **kwargs).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = op.run(q, weights, k, **kwargs)
    scores = _typed_model_scores(qr, kr, weights, math_mode, logits_dtype, fp8)
    ends = (keys * ratio - rows + torch.arange(1, rows + 1, device="cuda")) // ratio
    scores.masked_fill_(
        torch.arange(keys, device="cuda")[None] >= ends[:, None], -torch.inf
    )
    if blocks is not None:
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        positions = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        allowed.scatter_(1, positions, True)
        scores.masked_fill_(~allowed, -torch.inf)

    def check(output):
        idx = output.long()
        assert ((idx >= 0) & (idx < ends[:, None])).all()
        ordered = idx.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        actual = scores.gather(1, idx).sort(1).values
        expected = scores.topk(top_k, dim=1).values.sort(1).values
        torch.testing.assert_close(
            actual,
            expected,
            rtol=(
                2e-5
                if math_mode == "relu_strict" and logits_dtype == torch.float32
                else 0
            ),
            atol=(
                1e-6
                if math_mode == "relu_strict" and logits_dtype == torch.float32
                else 0
            ),
        )
        assert op.error.item() == op.repair_error.item() == 0

    check(eager)
    engine = op._capacity_plan._engine
    if path == "ds_consume" and not (
        math_mode == "native_bf16" and weights_dtype == torch.bfloat16
    ):
        assert engine.scores.dtype == logits_dtype
        assert engine.samples.dtype == logits_dtype
    if path != "ds_consume":
        engine.stages[0][1].run()
        engine.thresholds.fill_(float("inf"))
        for _, stage in engine.stages[1:]:
            stage.run()
            if stage is engine.stages[1][1]:
                assert (engine.counts == 0).all()
        assert engine.recovery_queue[0].item() > 0
        check(engine.output)
    else:
        from flashinfer.experimental.attn_scores.kernels.selective_topk.api import (
            ThrowawayTopK,
        )

        assert isinstance(engine, ThrowawayTopK)
        for name, stage in engine.stages:
            if name.startswith("K1"):
                stage.run()
        engine.thresholds.fill_(float("inf"))
        for name, stage in engine.stages:
            if name in ("K2", "K3"):
                stage.run()
        assert (engine.counts == 0).all()
        if hasattr(engine, "queue"):
            assert engine.queue[0].item() > 0
        for name, stage in engine.stages:
            if name.startswith("K4"):
                stage.run()
        check(engine.output)
        # Overflow must also enter exact recovery without consuming truncated candidates.
        engine.counts.fill_(4097)
        for name, stage in engine.stages:
            if name.startswith("K4"):
                stage.run()
        check(engine.output)
    graph.replay()
    torch.cuda.synchronize()
    check(replay)
    q.view(torch.uint8).bitwise_xor_(0x80 if fp8 else 0x88)
    if blocks is not None:
        blocks[:, :2047] += 1
    scores = _typed_model_scores(-qr, kr, weights, math_mode, logits_dtype, fp8)
    scores.masked_fill_(
        torch.arange(keys, device="cuda")[None] >= ends[:, None], -torch.inf
    )
    if blocks is not None:
        allowed.zero_()
        positions = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        allowed.scatter_(1, positions, True)
        scores.masked_fill_(~allowed, -torch.inf)
    graph.replay()
    torch.cuda.synchronize()
    check(replay)
    if path == "ds_publish":
        published = op.candidate_blocks.long()
        assert (published[:, 1:] > published[:, :-1]).all()
        latest = (ends - 1) // 8
        assert (published == latest[:, None]).any(1).all()
        block_scores = scores.reshape(rows, -1, 8).amax(-1)
        block_scores.scatter_(1, latest[:, None], torch.inf)
        torch.testing.assert_close(
            block_scores.gather(1, published).sort(1).values,
            block_scores.topk(2048, dim=1).values.sort(1).values,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_typed_block_publication_keeps_latest_on_infinite_ties(dtype):
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.api import DTYPES
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.candidate_blocks import (
        compile_block_selector,
    )

    rows, columns = 8, 4096
    scores = torch.full((rows, columns), torch.inf, device="cuda", dtype=dtype)
    ends = columns * 8 - torch.arange(rows, device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(ends)
    blocks = torch.empty((rows, 2048), device="cuda", dtype=torch.int32)
    lengths = torch.empty_like(ends)
    kernel = compile_block_selector(DTYPES[dtype], columns)
    kernel(scores, ends, blocks, lengths, starts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(scores, ends, blocks, lengths, starts)
    for shrink in (False, True):
        if shrink:
            ends.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) * 137)
        graph.replay()
        latest = (ends - 1) // 8
        count = ((ends + 7) // 8).clamp(max=2048)
        valid = torch.arange(2048, device="cuda")[None] < count[:, None]
        assert ((blocks >= 0) & (blocks <= latest[:, None]) | ~valid).all()
        assert ((blocks == latest[:, None]).any(1) | (count == 0)).all()
        assert (blocks[~valid] == 2147483647).all()
        assert ((blocks[:, 1:] > blocks[:, :-1]) | ~valid[:, 1:]).all()
        torch.testing.assert_close(lengths, count * 8 - ((8 - ends % 8) % 8))


@pytest.mark.parametrize("path", ["ds_full", "ds_consume", "ds_publish"])
@pytest.mark.parametrize("weight_scale", [2.0**18, 2.0**-24])
@pytest.mark.parametrize("logits_dtype", [torch.float16, torch.float32])
def test_native_bf16_monotone_output_conversion(path, weight_scale, logits_dtype):
    test_model_logits_and_reduction_matrix(
        path, "native_bf16", logits_dtype, torch.bfloat16, weight_scale
    )


@pytest.mark.parametrize("weight_scale", [2.0**18, 2.0**-24])
@pytest.mark.parametrize("logits_dtype", [torch.float16, torch.bfloat16])
def test_fp32_monotone_output_conversion(weight_scale, logits_dtype):
    test_model_logits_and_reduction_matrix(
        "kimi", "relu_strict", logits_dtype, torch.bfloat16, weight_scale
    )


def test_ds41_shared_bf16_prefill_replay(keys_override=None, publish=False):
    """Check native head arithmetic across persistent query tiles and replay."""
    _blackwell()
    torch.manual_seed(817)
    rows, heads, keys, top_k = 8192, 32, keys_override or 65536, 512
    ratio = 1 if publish else 2
    q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
    k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
    qs = torch.full((rows, heads, 4), 124, device="cuda", dtype=torch.uint8)
    ks = torch.full((keys, 4), 124, device="cuda", dtype=torch.uint8)
    weights = (torch.rand(rows, heads, device="cuda") * 0.25 - 0.0625).to(
        torch.bfloat16
    )
    op = FP8MQATopKWrapper(
        model_config="DS-V4.1-Flash",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        publish_candidate_blocks=publish,
        qkv_dtype="mxfp4",
        math_mode="native_bf16",
        logits_dtype=torch.bfloat16,
        weights_dtype=torch.bfloat16,
        max_context_length=keys * ratio,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
    )
    op.plan(
        torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32),
    )
    kwargs = dict(q_scale=qs.view(torch.int32).squeeze(-1), k_scale=ks)
    eager = op.run(q, weights, k, **kwargs).clone()
    if not publish:
        assert op._capacity_plan._engine.shared_kv_queries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op.run(q, weights, k, **kwargs)
    check_rows = torch.tensor([0, 1, 7, 8, 63, 4096, 8190, 8191], device="cuda")
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    kr = table[torch.stack((k & 15, k >> 4), -1).long()].reshape(keys, 128) * 0.125
    ends = (keys * ratio - rows + check_rows + 1) // ratio

    def check(actual):
        selected_q = q[check_rows]
        qr = (
            table[torch.stack((selected_q & 15, selected_q >> 4), -1).long()].reshape(
                len(check_rows), heads, 128
            )
            * 0.125
        )
        scores = _typed_model_scores(
            qr, kr, weights[check_rows], "native_bf16", torch.bfloat16, False
        )
        scores.masked_fill_(
            torch.arange(keys, device="cuda")[None] >= ends[:, None], -torch.inf
        )
        indices = actual[check_rows].long()
        assert ((indices >= 0) & (indices < ends[:, None])).all()
        ordered = indices.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        torch.testing.assert_close(
            scores.gather(1, indices).sort(1).values,
            scores.topk(top_k, dim=1).values.sort(1).values,
            rtol=0,
            atol=0,
        )
        if publish:
            block_scores = scores.reshape(len(check_rows), -1, 8).amax(2)
            block_scores.scatter_(1, ((ends - 1) // 8)[:, None], torch.inf)
            blocks = op.candidate_blocks[check_rows].long()
            torch.testing.assert_close(
                block_scores.gather(1, blocks).sort(1).values,
                block_scores.topk(2048, dim=1).values.sort(1).values,
                rtol=0,
                atol=0,
            )
        assert op.error.item() == op.repair_error.item() == 0

    check(eager)
    graph.replay()
    check(output)
    q.bitwise_xor_(0x88)
    graph.replay()
    check(output)


@pytest.mark.parametrize("logits_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sparse_candidate_binding_replay(logits_dtype):
    """Read live candidate tables in eager, graph and recovery paths."""
    _blackwell()
    torch.manual_seed(428)
    rows, heads, keys = 512, 32, 32768
    q = torch.randint(0, 256, (rows, heads, 64), device="cuda", dtype=torch.uint8)
    k = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
    qs = torch.full((rows, heads, 4), 124, device="cuda", dtype=torch.uint8)
    ks = torch.full((keys, 4), 124, device="cuda", dtype=torch.uint8)
    weights = (torch.rand(rows, heads, device="cuda") * 0.25).to(torch.bfloat16)
    blocks = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(rows, 1)
    latest = (keys - rows + torch.arange(rows, device="cuda", dtype=torch.int32)) // 8
    blocks[:, -1] = latest
    op = FP8MQATopKWrapper(
        model_config="DS-V4.1-Flash",
        num_heads=heads,
        top_k=512,
        compression_ratio=1,
        qkv_dtype="mxfp4",
        math_mode="native_bf16",
        logits_dtype=logits_dtype,
        weights_dtype=torch.bfloat16,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        candidate_capacity=2048,
        device="cuda",
    )

    def plan(count):
        op.plan(
            torch.tensor([0, count], device="cuda", dtype=torch.int32),
            torch.tensor([0, keys], device="cuda", dtype=torch.int32),
            num_query_rows=count,
        )

    def run(table, count=rows):
        return op.run(
            q[:count],
            weights[:count],
            k,
            q_scale=qs[:count].view(torch.int32).squeeze(-1),
            k_scale=ks,
            candidate_blocks=table,
        )

    plan(rows)
    eager = run(blocks).clone()
    engine = op._capacity_plan._engine
    assert engine.direct_candidates
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(blocks)
    check_rows = torch.tensor([0, 1, 7, 31, 127], device="cuda")
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    kr = table[torch.stack((k & 15, k >> 4), -1).long()].reshape(keys, 128) * 0.125

    def check(actual, candidates, count=rows):
        selected_q = q[check_rows]
        qr = table[torch.stack((selected_q & 15, selected_q >> 4), -1).long()]
        qr = qr.reshape(len(check_rows), heads, 128) * 0.125
        scores = _typed_model_scores(
            qr, kr, weights[check_rows], "native_bf16", logits_dtype, False
        )
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        positions = (
            candidates[check_rows, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        allowed.scatter_(1, positions, True)
        ends = keys - count + check_rows + 1
        allowed &= torch.arange(keys, device="cuda")[None] < ends[:, None]
        scores.masked_fill_(~allowed, -torch.inf)
        indices = actual[check_rows].long()
        ordered = indices.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        torch.testing.assert_close(
            scores.gather(1, indices).sort(1).values,
            scores.topk(512, dim=1).values.sort(1).values,
            rtol=0,
            atol=0,
        )
        assert op.error.item() == op.repair_error.item() == 0

    check(eager, blocks)
    blocks[:, :-1].add_(256)
    q.bitwise_xor_(0x88)
    graph.replay()
    check(output, blocks)
    other = blocks.clone()
    other[:, :-1].add_(512)
    check(run(other), other)
    unaligned = torch.empty(blocks.numel() + 1, device="cuda", dtype=torch.int32)[1:]
    unaligned = unaligned.view_as(blocks)
    unaligned.copy_(other)
    unaligned[:, :-1].add_(32)
    check(run(unaligned), unaligned)
    torch.testing.assert_close(op._capacity_plan._candidate_input, unaligned)
    graph.replay()
    check(output, blocks)
    engine.stages[0][1].run(other)
    engine.thresholds.fill_(float("inf"))
    for _, stage in engine.stages[1:]:
        stage.run(other)
    check(engine.output, other)
    plan(128)
    other[:128, -1] = (
        keys - 128 + torch.arange(128, device="cuda", dtype=torch.int32)
    ) // 8
    check(run(other[:128], 128), other, 128)
    plan(rows)
    graph.replay()
    check(output, blocks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interior", [False, True])
def test_paired_block_maxima_graph_replay(dtype, interior):
    _blackwell()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.attn_scores._topk_runtime import (
        _compile_for_device,
        _current_stream,
    )
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.candidate_blocks import (
        write_block_pair,
        write_block_pair_interior,
    )

    class Maxima:
        @cute.jit
        def __call__(
            self,
            values: cute.Tensor,
            ends: cute.Tensor,
            out: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(values, ends, out).launch(
                grid=(values.shape[0] // 2, values.shape[1] // 128, 1),
                block=(128, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, values: cute.Tensor, ends: cute.Tensor, out: cute.Tensor):
            row = cute.arch.block_idx()[0] * 2
            key = cute.arch.block_idx()[1] * 128 + cute.arch.thread_idx()[0]
            first, second = values[row, key], values[row + 1, key]
            if cutlass.const_expr(interior):
                write_block_pair_interior(
                    out,
                    row,
                    key,
                    first,
                    second,
                    paired_fp32=out.element_type == cutlass.Float32,
                )
            else:
                write_block_pair(
                    out,
                    row,
                    key,
                    ends[row],
                    ends[row + 1],
                    first,
                    second,
                    paired_fp32=out.element_type == cutlass.Float32,
                )

    torch.manual_seed(4103)
    values = torch.randn((8, 384), device="cuda") * 10
    values[:, 9] = 1e10
    values[:, 21] = -float("inf")
    ends = torch.tensor(
        [0, 1, 7, 8, 129, 257, 383, 384], dtype=torch.int32, device="cuda"
    )
    out = torch.full((8, 48), -torch.inf, device="cuda", dtype=dtype)
    tensors = tuple(from_dlpack(t, assumed_align=4) for t in (values, ends, out))
    stream = _current_stream(values.device)
    compiled = _compile_for_device(values.device, Maxima(), *tensors, stream)
    compiled(*tensors, stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*tensors, _current_stream(values.device))
    for replay in (False, True):
        if replay:
            values.neg_()
            ends.copy_(
                torch.tensor(
                    [9, 47, 128, 261, 320, 384, 1, 127],
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            out.fill_(-torch.inf)
            graph.replay()
        reference = values.to(dtype).float()
        if not interior:
            reference.masked_fill_(
                torch.arange(384, device="cuda")[None] >= ends[:, None], -torch.inf
            )
        reference = reference.reshape(8, 48, 8).amax(-1)
        if not interior:
            rows = (ends > 0).nonzero().flatten()
            reference[rows, (ends[rows] - 1) // 8] = torch.inf
        torch.testing.assert_close(out.float(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("path", ["ds_publish", "ds_consume"])
@pytest.mark.parametrize("math_mode", ["relu_strict", "native_bf16", "native_fp16"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
def test_publisher_consumer_default_weights_gpu_oracle(
    path, math_mode, ties, weights_dtype
):
    """Exercise default FP32 weights, partial blocks, recovery and live graph inputs."""
    test_model_logits_and_reduction_matrix(
        path,
        math_mode,
        torch.bfloat16,
        weights_dtype,
        default_weights=weights_dtype == torch.float32,
        ties=ties,
    )


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "math_mode,weights_dtype",
    [
        ("native_bf16", torch.bfloat16),
        ("native_bf16", torch.float32),
        ("relu_strict", torch.float32),
        ("native_fp16", torch.bfloat16),
    ],
)
def test_throwaway_candidate_input_contract(pdl, math_mode, weights_dtype):
    """The shared entry point selects logical IDs and handles partial or empty rows."""
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.api import (
        ThrowawayTopK,
    )

    rows, keys = 8, 4096
    q = torch.zeros((rows, 32, 64), dtype=torch.uint8, device="cuda")
    k = torch.zeros((keys, 64), dtype=torch.uint8, device="cuda")
    weights = torch.ones((rows, 32), dtype=weights_dtype, device="cuda")
    qs = torch.full((rows, 32, 4), 127, dtype=torch.uint8, device="cuda")
    ks = torch.full((keys, 4), 127, dtype=torch.uint8, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.tensor(
        [0, 1, 7, 8, 511, 512, 513, 2051], dtype=torch.int32, device="cuda"
    )
    blocks = torch.arange(2048, dtype=torch.int32, device="cuda").repeat(rows, 1)
    blocks.masked_fill_(blocks * 8 >= ends[:, None], torch.iinfo(torch.int32).max)
    kwargs = dict(
        q_scale=qs,
        k_scale=ks,
        top_k=512,
        compression_ratio=1,
        math_mode=math_mode,
        pdl=pdl,
        logits_dtype=torch.bfloat16,
        candidate_k_block_ids=blocks,
        candidate_valid_lens=ends,
    )
    for bad in (
        dict(candidate_valid_lens=None),
        dict(candidate_k_block_ids=blocks[:, :1024]),
        dict(compression_ratio=2),
        dict(top_k=2048),
    ):
        with pytest.raises(ValueError):
            ThrowawayTopK(q, k, weights, starts, ends, **(kwargs | bad))
    engine = ThrowawayTopK(q, k, weights, starts, ends, **kwargs)
    assert [name.split(".")[0] for name, _ in engine.stages][0] == "K1"
    assert {name.split(".")[0] for name, _ in engine.stages} == {"K1", "K2", "K3", "K4"}
    engine.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = engine.run(blocks)
    graph.replay()
    torch.cuda.synchronize()
    for row, end in enumerate(ends.tolist()):
        count = min(end, 512)
        valid = out[row, :count]
        assert ((valid >= 0) & (valid < end)).all()
        assert valid.unique().numel() == count
        assert (out[row, count:] == -1).all()


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_candidate_default_mode(publish, explicit_none):
    _blackwell()
    options = dict(
        model_config="DS-V4.1-Flash",
        num_heads=32,
        top_k=512,
        compression_ratio=1,
        qkv_dtype="mxfp4",
        max_context_length=4096,
        max_chunk_size=16,
        max_batch_size=1,
        device="cuda",
        publish_candidate_blocks=publish,
        candidate_capacity=0 if publish else 2048,
    )
    op = FP8MQATopKWrapper(**options, **({"math_mode": None} if explicit_none else {}))
    assert op._capacity_plan.math_mode == "relu_strict"
    with pytest.raises(ValueError, match="require"):
        FP8MQATopKWrapper(**options, math_mode="relu_fast")


@pytest.mark.parametrize(
    "invalid",
    [
        {"publish_candidate_blocks": True},
        {"candidate_capacity": 128},
        {"compression_ratio": 2},
        {"qkv_dtype": "fp8"},
        {"model_config": "Kimi-K3.1-Pro"},
        {"max_batch_size": 2},
    ],
)
def test_candidate_configuration_rejected(invalid):
    _blackwell()
    options = dict(
        model_config="DS-V4.1-Flash",
        num_heads=32,
        top_k=512,
        compression_ratio=1,
        qkv_dtype="mxfp4",
        max_context_length=4096,
        max_chunk_size=16,
        max_batch_size=1,
        candidate_capacity=2048,
        device="cuda",
    )
    with pytest.raises((ValueError, TypeError)):
        FP8MQATopKWrapper(**(options | invalid))


def _short_candidate_input(math_mode, weights_dtype, *, publish=False):
    rows, keys = 16, 4096
    torch.manual_seed(7325)
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda")
    k = torch.randint(0, 256, (keys, 64), dtype=torch.uint8, device="cuda")
    qs = torch.full((rows, 32, 4), 124, dtype=torch.uint8, device="cuda")
    ks = torch.full((keys, 4), 124, dtype=torch.uint8, device="cuda")
    w = torch.rand((rows, 32), dtype=weights_dtype, device="cuda")
    blocks = torch.full((rows, 2048), 2147483647, dtype=torch.int32, device="cuda")
    blocks[:, : keys // 8] = torch.arange(keys // 8, device="cuda", dtype=torch.int32)
    ends = keys - rows + torch.arange(1, rows + 1, device="cuda")
    blocks.masked_fill_(blocks.long() * 8 >= ends[:, None], 2147483647)
    op = FP8MQATopKWrapper(
        model_config="DS-V4.1-Flash",
        num_heads=32,
        top_k=512,
        compression_ratio=1,
        qkv_dtype="mxfp4",
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        math_mode=math_mode,
        weights_dtype=weights_dtype,
        logits_dtype=torch.float32,
        publish_candidate_blocks=publish,
        candidate_capacity=0 if publish else 2048,
    )
    op.plan(
        torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        torch.tensor([0, keys], dtype=torch.int32, device="cuda"),
    )
    return op, q, k, w, qs, ks, blocks, ends


def _check_short_candidate(output, q, k, w, mode, ends, blocks=None):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    qr = (
        table[torch.stack((q & 15, q >> 4), -1).long()].reshape(len(q), 32, 128) * 0.125
    )
    kr = table[torch.stack((k & 15, k >> 4), -1).long()].reshape(len(k), 128) * 0.125
    scores = _typed_model_scores(qr, kr, w, mode, torch.float32, False)
    scores.masked_fill_(
        torch.arange(len(k), device="cuda")[None] >= ends[:, None], -torch.inf
    )
    if blocks is not None:
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        positions = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        allowed.scatter_(1, positions, True)
        scores.masked_fill_(~allowed, -torch.inf)
    actual = scores.gather(1, output.long()).sort(1).values
    expected = scores.topk(512, dim=1).values.sort(1).values
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-5 if mode == "relu_strict" else 0,
        atol=1e-6 if mode == "relu_strict" else 0,
    )
    assert (output.sort(1).values[:, 1:] != output.sort(1).values[:, :-1]).all()


@pytest.mark.parametrize(
    "mode,weight",
    [
        ("native_bf16", torch.bfloat16),
        ("relu_strict", torch.float32),
        ("native_fp16", torch.float32),
    ],
)
def test_candidate_pools_share_scratch_and_replay(mode, weight):
    _blackwell()
    op, q, k, w, qs, ks, blocks, ends = _short_candidate_input(mode, weight)
    pools = [k, k.bitwise_xor(0x88)]
    for pool in pools:
        op.prepare_pool(pool, k_scale=ks)
    engines = list(op._capacity_plan._engines.values())
    assert len(engines) == 2
    for name in ("samples", "thresholds", "candidates", "counts", "scores", "output"):
        assert (
            getattr(engines[0], name).data_ptr() == getattr(engines[1], name).data_ptr()
        )

    def run(pool):
        return op.run(q, w, pool, q_scale=qs, k_scale=ks, candidate_blocks=blocks)

    for pool in pools:
        _check_short_candidate(run(pool), q, pool, w, mode, ends)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a = run(pools[0]).clone()
        b = run(pools[1]).clone()
    graph.replay()
    _check_short_candidate(a, q, pools[0], w, mode, ends)
    _check_short_candidate(b, q, pools[1], w, mode, ends)


@pytest.mark.parametrize(
    "mode,weight",
    [
        ("native_bf16", torch.bfloat16),
        ("relu_strict", torch.float32),
        ("native_fp16", torch.float32),
    ],
)
def test_publisher_to_short_consumer(mode, weight):
    _blackwell()
    publisher, q, k, w, qs, ks, _, ends = _short_candidate_input(
        mode, weight, publish=True
    )
    consumer, *_ = _short_candidate_input(mode, weight)
    publisher.prepare_pool(k, k_scale=ks)
    consumer.prepare_pool(k, k_scale=ks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publisher.run(q, w, k, q_scale=qs, k_scale=ks)
        out = consumer.run(
            q, w, k, q_scale=qs, k_scale=ks, candidate_blocks=publisher.candidate_blocks
        )
    graph.replay()
    torch.testing.assert_close(publisher.candidate_valid_lens, ends.to(torch.int32))
    torch.testing.assert_close(
        consumer._capacity_plan._candidate_lens, ends.to(torch.int32)
    )
    _check_short_candidate(out, q, k, w, mode, ends)


@pytest.mark.parametrize("mode", ["native_bf16", "native_fp16", "relu_strict"])
@pytest.mark.parametrize("weight", [torch.bfloat16, torch.float32])
def test_candidate_value_validation_on_graph_replay(mode, weight):
    _blackwell()
    op, q, k, w, qs, ks, blocks, ends = _short_candidate_input(mode, weight)
    valid = blocks.clone()
    op.run(q, w, k, q_scale=qs, k_scale=ks, candidate_blocks=blocks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.run(q, w, k, q_scale=qs, k_scale=ks, candidate_blocks=blocks)
    graph.replay()
    _check_short_candidate(out, q, k, w, mode, ends)
    for value in (-1, 2147483647, len(k) // 8, 0):
        blocks.copy_(valid)
        blocks[:, 1] = value
        graph.replay()
        assert (out == -1).all()
        assert op.repair_error.item() & 32
    blocks.copy_(valid)
    blocks[:, -1] = 0
    graph.replay()
    assert (out == -1).all()
    blocks.copy_(valid)
    graph.replay()
    _check_short_candidate(out, q, k, w, mode, ends)


@pytest.mark.parametrize("mode", ["native_bf16", "native_fp16", "relu_strict"])
@pytest.mark.parametrize("weight", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pdl", [False, True])
def test_candidate_k1_validation_full_table(mode, weight, pdl):
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.api import (
        ThrowawayTopK,
    )

    _, q, _, w, qs, _, _, _ = _short_candidate_input(mode, weight)
    rows, keys = len(q), 32768
    k = torch.randint(0, 256, (keys, 64), dtype=torch.uint8, device="cuda")
    ks = torch.full((keys, 4), 124, dtype=torch.uint8, device="cuda")
    ends = torch.full((rows,), keys, device="cuda", dtype=torch.int32)
    lengths = torch.full_like(ends, 16384)
    blocks = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(rows, 1)
    blocks[:, -1] = keys // 8 - 1
    valid = blocks.clone()
    op = ThrowawayTopK(
        q,
        k,
        w,
        torch.zeros_like(ends),
        ends,
        q_scale=qs,
        k_scale=ks,
        top_k=512,
        compression_ratio=1,
        model_config="DS-V4.1-Flash",
        math_mode=mode,
        pdl=pdl,
        candidate_k_block_ids=blocks,
        candidate_valid_lens=lengths,
    )
    _check_short_candidate(op.run(), q, k, w, mode, ends, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op.run()
    for column, value in (
        (1301, -1),
        (1301, keys // 8),
        (1301, 1300),
        (1301, 2147483647),
        (2047, 2048),
    ):
        blocks.copy_(valid)
        blocks[0, column] = value
        graph.replay()
        assert (output[0] == -1).all()
        assert op.candidate_error.item() & 32
        _check_short_candidate(output[1:], q[1:], k, w[1:], mode, ends[1:], valid[1:])
    blocks.copy_(valid)
    graph.replay()
    _check_short_candidate(output, q, k, w, mode, ends, valid)
    assert (lengths == 16384).all()


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "device", "strides", "overlap", "no_capacity"]
)
def test_candidate_tensor_contract_rejected(invalid):
    _blackwell()
    op, q, k, w, qs, ks, blocks, _ = _short_candidate_input(
        "relu_strict", torch.float32, publish=invalid == "no_capacity"
    )
    kwargs = {}
    if invalid == "shape":
        blocks = blocks[:, :1024].contiguous()
    elif invalid == "dtype":
        blocks = blocks.long()
    elif invalid == "device":
        blocks = blocks.cpu()
    elif invalid == "strides":
        blocks = blocks.T.contiguous().T
    elif invalid == "overlap":
        kwargs["out"] = blocks.flatten()[: len(q) * 512].view(len(q), 512)
    with pytest.raises((ValueError, TypeError)):
        op.run(q, w, k, q_scale=qs, k_scale=ks, candidate_blocks=blocks, **kwargs)


@pytest.mark.parametrize("mode", ["relu_strict", "native_bf16", "native_fp16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_kimi_long_context_oracle(mode, dtype):
    test_model_logits_and_reduction_matrix(
        "kimi", mode, dtype, torch.float32, keys_override=65536
    )


@pytest.mark.parametrize("mode", ["relu_strict", "native_bf16", "native_fp16"])
@pytest.mark.parametrize("weight", [torch.float32, torch.bfloat16])
def test_candidate_without_triton(monkeypatch, mode, weight):
    import builtins

    _blackwell()
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ModuleNotFoundError("Triton is unavailable", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    op, q, k, w, qs, ks, blocks, ends = _short_candidate_input(mode, weight)
    output = op.run(q, w, k, q_scale=qs, k_scale=ks, candidate_blocks=blocks)
    torch.cuda.synchronize()
    _check_short_candidate(output, q, k, w, mode, ends)


def test_kimi_default_math_and_explicit_policy():
    _blackwell()
    rows, keys = 16, 65536
    torch.manual_seed(7326)
    q = (torch.randn(rows, 32, 128, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    k = (torch.randn(keys, 128, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    w = torch.rand(rows, 32, device="cuda") * 0.125
    scales = torch.ones(keys, device="cuda")
    op = FP8MQATopKWrapper(
        model_config="Kimi-K3.1-Pro",
        num_heads=32,
        top_k=2048,
        compression_ratio=1,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        qkv_dtype="fp8",
    )
    assert op._capacity_plan.math_mode == "relu_fast"
    op.plan(
        torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        torch.tensor([0, keys], device="cuda", dtype=torch.int32),
    )
    actual = op.run(q, w, k, k_scale=scales)
    expected = _rows(q, w, k, scales, [rows], [keys], 1)
    _check_selection(actual, expected, 2048)
    from flashinfer.experimental.attn_scores.indexer_config import resolve_model_config

    assert resolve_model_config(32, 128, 2048, 1) == "GLM-5.3"
    assert resolve_model_config(32, 128, 2048, 1, "Kimi-K3.1-Pro") == "Kimi-K3.1-Pro"


def test_raw_publisher_rejects_nonzero_start():
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.api import (
        ThrowawayTopK,
    )

    _, q, k, w, qs, ks, _, ends = _short_candidate_input("relu_strict", torch.float32)
    starts = torch.ones(len(q), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="zero-based"):
        ThrowawayTopK(
            q,
            k,
            w,
            starts,
            ends.to(torch.int32),
            q_scale=qs,
            k_scale=ks,
            top_k=512,
            model_config="DS-V4.1-Flash",
            compression_ratio=1,
            _filter_config=dict(publish_blocks=True),
        )


@pytest.mark.parametrize("path", ["kimi", "glm"])
def test_strict_fp8_long_context_bf16_weights(path):
    test_model_logits_and_reduction_matrix(
        path, "relu_strict", torch.float32, torch.bfloat16, keys_override=65536
    )


@parametrize_product(
    {
        "encoding": ["fp8", "mxfp4", "mxfp8"],
        "heads": [4, 64],
        "mode": ["native_bf16", "native_fp16"],
        "dtype": [torch.float32, torch.float16, torch.bfloat16],
        "ratio": [1, 2],
    },
    regular=[
        ("fp8", 4, "native_bf16", torch.float32, 1),
        ("fp8", 64, "native_fp16", torch.float16, 2),
        ("mxfp4", 4, "native_fp16", torch.bfloat16, 1),
        ("mxfp4", 64, "native_bf16", torch.float16, 2),
        ("mxfp8", 4, "native_bf16", torch.bfloat16, 2),
        ("mxfp8", 64, "native_fp16", torch.float32, 1),
    ],
)
def test_general_native_cross_dtype_and_ratio(encoding, heads, mode, dtype, ratio):
    test_cutedsl_precision_modes_against_oracle(
        encoding, mode, dtype, torch.bfloat16, heads, 512, ratio=ratio
    )


@pytest.mark.parametrize(
    "mode,weight", [("native_bf16", torch.bfloat16), ("relu_strict", torch.float32)]
)
def test_candidate_prepare_ignores_stale_table(mode, weight):
    _blackwell()
    op, q, k, w, qs, ks, blocks, ends = _short_candidate_input(mode, weight)
    kwargs = dict(q_scale=qs, k_scale=ks, candidate_blocks=blocks)
    result = op.run(q, w, k, **kwargs)
    _check_short_candidate(result, q, k, w, mode, ends)
    assert op.repair_error.item() == 0
    # Live bounds change before the caller supplies the next layer's table.
    ends = ends - 256
    op.plan(
        torch.tensor([0, len(q)], device="cuda", dtype=torch.int32),
        torch.tensor([0, len(k) - 256], device="cuda", dtype=torch.int32),
    )
    k2 = k.bitwise_xor(0x88)
    op.prepare_pool(k2, k_scale=ks)
    assert op.repair_error.item() == 0
    blocks.masked_fill_(blocks.long() * 8 >= ends[:, None], 2147483647)
    result = op.run(q, w, k2, **kwargs)
    _check_short_candidate(result, q, k2, w, mode, ends)
    assert op.repair_error.item() == 0


@pytest.mark.parametrize("columns", [8192, 131072])
def test_publisher_fp16_adjacent_scores(columns):
    _blackwell()
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.candidate_blocks import (
        compile_block_selector,
    )
    from flashinfer.experimental.attn_scores.kernels.selective_topk.api import DTYPES

    torch.manual_seed(6137)
    scores = torch.zeros((8, columns), device="cuda", dtype=torch.float16)
    # Adjacent scores form a crossing bucket with more than 288 entries.
    # Their FP32 sortable keys differ only below bit 16.
    scores[:, :1800] = 64
    scores[:, 1800:2120] = 1 + torch.arange(320, device="cuda") % 2 / 1024
    for row in range(len(scores)):
        scores[row] = scores[row, torch.randperm(columns, device="cuda")]
    ends = torch.full((8,), columns * 8, device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(ends)
    out = torch.empty((8, 2048), device="cuda", dtype=torch.int32)
    lengths = torch.empty_like(ends)
    fn = compile_block_selector(DTYPES[torch.float16], columns)
    scores[:, -1] = torch.inf
    fn(scores, ends, out, lengths, starts)
    torch.testing.assert_close(
        scores.gather(1, out.long()).sort(1).values,
        scores.topk(2048, dim=1).values.sort(1).values,
        rtol=0,
        atol=0,
    )
    # Invalid raw replay bounds must not overflow the publication bitmap.
    ends.fill_(columns * 8 + 8192)
    out.fill_(-1)
    fn(scores, ends, out, lengths, starts)
    assert ((out >= 0) & (out < columns)).all()
    assert (out[:, 1:] > out[:, :-1]).all()
    # Empty and malformed negative raw bounds must not write before the output.
    starts.fill_(-24)
    ends.fill_(-16)
    guarded = torch.full((8 * 2048 + 8,), -7, device="cuda", dtype=torch.int32)
    fn(scores, ends, guarded[4:-4].view(8, 2048), lengths, starts)
    assert (guarded[:4] == -7).all() and (guarded[-4:] == -7).all()
    assert (guarded[4:-4] == 2147483647).all()
    assert (lengths == 0).all()


def test_publisher_rejects_grid_y_overflow():
    _blackwell()
    with pytest.raises(ValueError, match="65535"):
        FP8MQATopKWrapper(
            model_config="DS-V4.1-Flash",
            num_heads=32,
            top_k=512,
            compression_ratio=1,
            qkv_dtype="mxfp4",
            max_context_length=131072,
            max_chunk_size=65536,
            max_batch_size=1,
            device="cuda",
            publish_candidate_blocks=True,
        )


@parametrize_product(
    {
        "keys,publish": [
            (131072, True),
            (262144, True),
            (524288, False),
            (1048576, False),
        ]
    },
    regular=[((131072, True),), ((524288, False),)],
)
def test_ds41_large_query_specializations(keys, publish):
    test_ds41_shared_bf16_prefill_replay(keys_override=keys, publish=publish)


def test_kimi_large_query_specialization():
    test_model_logits_and_reduction_matrix(
        "kimi",
        "relu_strict",
        torch.float32,
        torch.float32,
        keys_override=65536,
        rows_override=8192,
    )


def test_consumer_nonfinite_scores_use_negative_infinity():
    _blackwell()
    op, q, k, w, qs, ks, blocks, ends = _short_candidate_input(
        "native_bf16", torch.bfloat16
    )
    # UE8M0 255 supplies NaN scales; finite keys remain selectable.
    q.zero_()
    k[:1024].zero_()
    ks[:1024].fill_(255)
    w.fill_(1)
    kwargs = dict(q_scale=qs, k_scale=ks, candidate_blocks=blocks)
    output = op.run(q, w, k, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op.run(q, w, k, **kwargs)
    graph.replay()
    # Zero dot products with finite scales must outrank any NaN scores.
    assert ((output >= 1024) & (output < ends[:, None])).all()
    assert op.repair_error.item() == 0
    engine = op._capacity_plan._engine
    engine.stages[0][1].run()
    engine.thresholds.fill_(float("inf"))
    for _, stage in engine.stages[1:]:
        stage.run()
    assert engine.queue[0].item() > 0
    assert ((engine.output >= 1024) & (engine.output < ends[:, None])).all()
    assert op.repair_error.item() == 0


@pytest.mark.parametrize("banks", [0, 4, 20, 32, 64])
def test_sparse_selector_rejects_unsupported_banks(banks):
    pytest.importorskip("cutlass")
    from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.sparse.selection import (
        SparseSelector,
    )

    with pytest.raises(ValueError, match="16 banks"):
        SparseSelector(banks=banks)


def test_recovery_rejects_unsupported_splits_with_optimized_python():
    pytest.importorskip("cutlass")
    program = """
from flashinfer.experimental.attn_scores.kernels.selective_topk.prefill.fp4_queue_recovery import FP4PrefillMQALogitsKernel
if FP4PrefillMQALogitsKernel.__doc__ is None:
    raise RuntimeError("missing recovery class docstring")
for splits in (0, 2, 16):
    try:
        FP4PrefillMQALogitsKernel(mode="recover", splits=splits)
    except ValueError as error:
        if "splits" not in str(error):
            raise
    else:
        raise RuntimeError("invalid split count accepted")
"""
    subprocess.run([sys.executable, "-O", "-c", program], check=True)
