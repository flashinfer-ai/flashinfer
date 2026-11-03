# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Contract and exactness checks for the explicit indexer-throwaway v2 backend."""

import math
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch

from flashinfer.attn_scores import FP8MQATopKWrapper


def test_v2_import_keeps_shared_cutlass_methods():
    # A fresh process catches import-time mutation even after other v2 tests.
    program = """
import cutlass.cute as cute
import cutlass.pipeline as pipeline
release = pipeline.PipelineTmaAsync.consumer_release
fma = cute.arch.fma_packed_f32x2
add = cute.arch.add_packed_f32x2
import flashinfer.experimental.attn_scores.kernels.indexer_topk_v2
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


def test_v2_pipeline_factory_is_local(monkeypatch):
    import cutlass.pipeline as pipeline
    from flashinfer.experimental.attn_scores.kernels.indexer_topk_v2.ordering import (
        PipelineTmaAsync,
    )

    native = pipeline.PipelineTmaAsync
    release = native.consumer_release
    original = object.__new__(native)
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return original

    monkeypatch.setattr(native, "create", staticmethod(create))
    result = PipelineTmaAsync.create("state", num_stages=2)
    assert result is original
    assert type(result) is PipelineTmaAsync
    assert calls == [(("state",), {"num_stages": 2})]
    assert native.consumer_release is release
    assert PipelineTmaAsync.consumer_release is not release


def test_v2_trace_export_rejects_backend():
    from flashinfer.fi_trace import fi_trace

    from flashinfer.experimental.attn_scores import _wrapper_type

    op = object.__new__(_wrapper_type())
    op._backend = "v2"
    with pytest.raises(ValueError, match="trace export.*v2.*not supported"):
        fi_trace(
            op.run,
            q=torch.empty((1, 64, 128), dtype=torch.float8_e4m3fn),
            weights=torch.empty((1, 64)),
            kv_fused=torch.empty((1, 128, 1, 132), dtype=torch.uint8),
        )


def test_v2_vendor_manifest():
    root = (
        Path(__file__).resolve().parents[3]
        / "flashinfer/experimental/attn_scores/kernels/indexer_topk_v2"
    )
    manifest = json.loads((root / "VENDOR_MANIFEST.json").read_text())
    actual = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }
    assert actual == manifest["files"]


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
def test_v2_caller_graph_ownership_and_host_hints(encoding, monkeypatch):
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
            backend="v2",
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
        original_engine, original_execute = plan._engine, plan._execute_layer

        def fail_second_pool(value):
            if value.data_ptr() == other_pool.data_ptr():
                raise RuntimeError("injected pool warmup failure")
            return original_execute(value)

        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(plan, "_execute_layer", fail_second_pool)
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
        ({"backend": "v2", "num_heads": 8}, "H4/H32/H64"),
        ({"backend": "v2", "num_heads": 64, "max_batch_size": 148}, "prefill"),
        ({"backend": "v1", "qkv_dtype": "mxfp4"}, "v1 supports"),
    ],
)
def test_v2_unsupported_contract(kwargs, match):
    settings = dict(
        device="cuda",
        max_context_length=8192,
        max_chunk_size=64,
        max_batch_size=1,
    )
    settings.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FP8MQATopKWrapper(**settings)


def _blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100 or SM103")


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
def test_v2_layer_updates_and_capture(
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
        backend="v2",
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
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op.run(q, weights, k, k_scale=scales, q_scale=q_scale)
    for step in range(2):
        if step:
            weights.neg_()
            if encoding == "fp8":
                q.copy_(torch.randn(q.shape, device="cuda").to(q.dtype))
            else:
                q.bitwise_xor_(0x88)
                q_scale.fill_(128)
        graph.replay()
        torch.cuda.synchronize()
        assert op.error.item() == 0 and op.repair_error.item() == 0
        qf = q.float() if encoding == "fp8" else _decode_fp4(q) * (2 if step else 1)
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
        selected = scores.gather(1, result.long()).sort(1, descending=True).values
        expected = scores.topk(top_k, dim=1).values
        torch.testing.assert_close(selected, expected, rtol=1e-4, atol=1e-3)
        assert not torch.any(
            result.sort(1).values[:, 1:] == result.sort(1).values[:, :-1]
        )


@pytest.mark.parametrize("encoding", ["fp8", "mxfp4"])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("preset", [False, True], ids=["generic", "preset"])
def test_v2_empty_initial_tile_and_causal_topk_boundary(encoding, top_k, preset):
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
        backend="v2",
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
def test_v2_fast_contract_changes_fail_closed(encoding):
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
        backend="v2",
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


def test_v2_qwen_contract_changes_fail_closed():
    _blackwell()
    rows, keys, heads = 64, 8192, 4
    q = torch.ones((rows, heads, 128), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.ones((keys, 128), device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.full((rows, heads), 128**-0.5, device="cuda")
    scales = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    op = FP8MQATopKWrapper(
        backend="v2",
        num_heads=heads,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
        num_keys=k.shape[0],
        model_config="Qwen3.8-Flash-Next",
    )
    op.plan(qo, kv)
    op.run(q, weights, k, k_scale=scales)
    torch.cuda.synchronize()
    assert op.repair_error.item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op.run(q, weights, k, k_scale=scales)
    weights[0, 0] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert op.repair_error.item() == 16
    assert torch.all(result == -1)
    weights.fill_(128**-0.5)
    op.plan(qo, kv)
    scales.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert op.repair_error.item() == 16
    assert torch.all(result == -1)


@pytest.mark.parametrize(
    "page,ratio,encoding,score_dtype",
    [
        (32, 1, "fp8", torch.float32),
        (64, 4, "fp8", torch.bfloat16),
        (128, 4, "fp8", torch.float32),
        (32, 4, "mxfp4", torch.float16),
        (64, 1, "mxfp4", torch.bfloat16),
        (128, 4, "mxfp4", torch.float32),
    ],
)
def test_v2_partial_storage_updates_and_short_rows(page, ratio, encoding, score_dtype):
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
        backend="v2",
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
def test_v2_nan_scores_rank_as_negative_infinity(logits_dtype, math_mode):
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
        backend="v2",
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
def test_v2_sample_proposal_constant_range_with_padding(value, warp):
    _blackwell()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.attn_scores.kernels.indexer_topk_v2.prefill.proposal_parallel import (
        propose_cached,
    )
    from flashinfer.experimental.attn_scores.kernels.indexer_topk_v2.prefill.proposal_warp import (
        propose_warp,
    )
    from flashinfer.experimental.attn_scores.selective_logits import (
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
def test_v2_persistent_capacity_padding_exact_keys(encoding, top_k, preset):
    _check_v2_capacity_padding(encoding, top_k, preset)


def test_v2_h32_persistent_capacity_padding_exact_keys():
    _check_v2_capacity_padding(
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
def test_v2_fp8_filter_variants_retire_capacity_padding(
    heads, top_k, keys, rows, ratio, logits_dtype
):
    # Exercise packed H4, mid-context H32 and full-tile H32 filters. Empty
    # capacity tiles precede live tiles in the persistent reverse traversal.
    _check_v2_capacity_padding(
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
def test_v2_fp4_filter_variants_retire_capacity_padding(
    heads, top_k, keys, rows, ratio, logits_dtype
):
    _check_v2_capacity_padding(
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


def _check_v2_capacity_padding(
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
    weights = torch.full((rows, heads), 1 / heads, device="cuda")
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, keys * ratio], device="cuda", dtype=torch.int32)
    model = ("DS-V4-Flash" if top_k == 512 else "DS-V4-Pro") if preset else None
    op = FP8MQATopKWrapper(
        backend="v2",
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
        # Every head has identical Q and weight 1/H, so the independent FP32
        # weighted ReLU reference equals this one-head dot product exactly.
        for first in range(0, rows, 32):
            stop = min(first + 32, rows)
            scores = reference_q[first:stop] @ bits.T
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
