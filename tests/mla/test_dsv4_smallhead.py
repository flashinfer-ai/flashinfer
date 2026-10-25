# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public DSv4 decode routing, FP64 math, and changing-input CUDA Graph checks."""

import pytest
import torch

from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as decode
from flashinfer.mla import _smallhead
from flashinfer.utils import (
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    get_device_sm_count,
)


@pytest.fixture(autouse=True)
def require_b300():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("small-head automatic dispatch is qualified on SM103")
    reason = _smallhead._dsl_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)


def _fixture(rows=1, heads=16, layout="HND", width=640):
    # Nonzero origins and physical page padding reproduce the framework's
    # cache contract without allocating the complete model working set.
    # Public ordinary decode requires [batch, query_length, heads, dim] for
    # every KV layout. Rank3 below refers only to the cache, not to the query.
    dtype = torch.float8_e4m3fn
    query = torch.empty((rows + 1, 1, heads, 512), device="cuda", dtype=dtype)[1:]
    pools = []
    for page, origin in ((32, 288), (128, 219)):
        storage = torch.empty((origin + 16 * 364, 512), device="cuda", dtype=dtype)
        storage.fill_(float("nan"))
        pools.append(
            storage.as_strided(
                (16, 1, page, 512), (364 * 512, page * 512, 512, 1), origin * 512
            )
        )
    args = dict(
        query=query,
        swa_kv_cache=pools[0],
        compressed_kv_cache=pools[1],
        sparse_indices=torch.empty((rows + 1, width), dtype=torch.int32, device="cuda")[
            1:
        ],
        sparse_topk_lens=torch.empty(rows + 4, dtype=torch.int32, device="cuda")[4:],
        seq_lens=torch.empty(rows + 4, dtype=torch.int32, device="cuda")[4:],
        sinks=torch.empty(heads + 4, dtype=torch.float32, device="cuda")[4:],
        workspace_buffer=torch.empty(
            64 * 1024 * 1024 + 16, dtype=torch.uint8, device="cuda"
        )[16:],
        out=torch.empty((rows + 1, 1, heads, 512), dtype=torch.bfloat16, device="cuda")[
            1:
        ],
        bmm1_scale=512**-0.5 / (64 * 32),
        bmm2_scale=1 / 32,
        max_q_len=1,
        sparse_indices_are_storage_offsets=True,
        kv_layout="HND" if layout == "rank3" else layout,
        enable_pdl=True,
    )
    if layout == "NHD":
        args["swa_kv_cache"] = pools[0].transpose(1, 2)
        args["compressed_kv_cache"] = pools[1].transpose(1, 2)
    elif layout == "rank3":
        args["swa_kv_cache"] = pools[0][:, 0]
        args["compressed_kv_cache"] = pools[1][:, 0]
    return args, pools


def _change(args, pools, generation):
    rows = args["query"].shape[0]
    width = args["sparse_indices"].shape[1]
    torch.manual_seed(713 + generation)
    args["query"].copy_(
        (torch.randn_like(args["query"], dtype=torch.float32) * 32).to(
            args["query"].dtype
        )
    )
    for begin, end, pool in ((0, 128, pools[0]), (128, width, pools[1])):
        pool.copy_((torch.randn_like(pool, dtype=torch.float32) * 8).to(pool.dtype))
        logical = torch.randint(
            0,
            pool.shape[0] * pool.shape[2],
            (rows, end - begin),
            device="cuda",
            dtype=torch.int32,
        )
        args["sparse_indices"][:, begin:end] = (
            logical // pool.shape[2] * 364 + logical % pool.shape[2]
        )
    args["sparse_indices"][:, 3::37] = -1
    args["sparse_topk_lens"].fill_(width if generation != 1 else max(128, width - 33))
    args["seq_lens"].fill_(7 if generation == 1 else 8192)
    args["sinks"].copy_(torch.linspace(-2, 2, args["sinks"].numel(), device="cuda"))
    if generation == 2:
        args["sparse_indices"][0].fill_(-1)


def _reference(args, pools):
    query = args["query"].reshape(-1, args["sinks"].numel(), 512).double()
    ids = args["sparse_indices"].long()
    values, masks = [], []
    for begin, end, pool in ((0, 128, pools[0]), (128, ids.shape[1], pools[1])):
        span = (pool.shape[0] - 1) * 364 + pool.shape[2]
        physical = pool.as_strided((span, 512), (512, 1))
        index = ids[:, begin:end]
        position = torch.arange(begin, end, device="cuda")[None]
        valid = (
            (index >= 0)
            & (index < span)
            & (position < args["sparse_topk_lens"][:, None])
        )
        if begin == 0:
            valid &= position < args["seq_lens"][:, None]
        values.append(
            torch.where(
                valid[..., None], physical[index.clamp(0, span - 1)].double(), 0.0
            )
        )
        masks.append(valid)
    kv, valid = torch.cat(values, 1), torch.cat(masks, 1)
    scores = torch.einsum("bhd,bkd->bhk", query, kv) * args["bmm1_scale"]
    scores.masked_fill_(~valid[:, None], -float("inf"))
    sink = args["sinks"].double()[None, :, None].expand(query.shape[0], -1, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    return torch.einsum("bhk,bkd->bhd", probabilities, kv) * args["bmm2_scale"]


def _check_fp64(out, expected):
    actual = out.reshape(expected.shape).double()
    error = actual - expected
    assert bool(torch.isfinite(actual).all())
    assert float(error.norm() / expected.norm().clamp_min(1e-30)) < 0.005
    assert float(error.abs().max() / expected.abs().max().clamp_min(1e-30)) < 0.01


def _spy(monkeypatch):
    seen = []
    real = _smallhead.try_run

    def traced(**kwargs):
        selected = real(**kwargs)
        seen.append(selected)
        return selected

    monkeypatch.setattr(_smallhead, "try_run", traced)
    return seen


@pytest.mark.parametrize("heads,rows", [(16, 1), (16, 2), (16, 4), (32, 1), (32, 2)])
@pytest.mark.parametrize("layout", ["HND", "NHD", "rank3"])
@torch.inference_mode()
def test_public_auto_math_and_graph_rebinding(monkeypatch, heads, rows, layout):
    args, pools = _fixture(rows, heads, layout)
    _change(args, pools, 0)
    native_out = torch.empty_like(args["out"])
    seen = _spy(monkeypatch)
    decode(**dict(args, backend="trtllm-gen", out=native_out))
    assert seen == []  # Explicit native selection must not be overridden.
    result = decode(**args)
    assert result is args["out"] and seen == [True]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(**args)
        decode(**dict(args, backend="trtllm-gen", out=native_out))
    assert seen == [True, True]
    for generation in range(3):
        _change(args, pools, generation)
        expected = _reference(args, pools)
        args["out"].fill_(float("nan"))
        native_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check_fp64(args["out"], expected)
        torch.testing.assert_close(args["out"], native_out, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "case", ["rows8", "heads64", "width128", "pdl_off", "counter", "no_sinks"]
)
@torch.inference_mode()
def test_public_auto_retains_native_fallback(monkeypatch, case):
    args, pools = _fixture(
        rows=8 if case == "rows8" else 1,
        heads=64 if case == "heads64" else 16,
        width=128 if case == "width128" else 640,
    )
    _change(args, pools, 0)
    if case == "pdl_off":
        args["enable_pdl"] = False
    if case == "no_sinks":
        args["sinks"] = None
    if case == "counter":
        args["multi_ctas_kv_counter_buffer"] = (
            _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                None,
                1,
                16,
                get_device_sm_count(args["query"].device),
                args["query"].device,
            )
        )
    seen = _spy(monkeypatch)
    expected = decode(
        **dict(args, backend="trtllm-gen", out=torch.empty_like(args["out"]))
    )
    result = decode(**args)
    torch.cuda.synchronize()
    assert seen == ([] if case == "counter" else [False])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_unprepared_capture_falls_back_then_warmed_capture_selects_cute(monkeypatch):
    args, pools = _fixture()
    _change(args, pools, 0)
    # First load the native module. Clear only immutable CuTe compiled metadata,
    # then exercise an actual auto call inside capture, without a CuTe warmup.
    decode(**dict(args, backend="trtllm-gen"))
    torch.cuda.synchronize()
    _smallhead._get_kernel.cache_clear()
    seen = _spy(monkeypatch)
    native_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(native_graph):
        decode(**args)
    assert seen == [False]
    decode(**args)
    assert seen == [False, True]
    cute_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cute_graph):
        decode(**args)
    assert seen == [False, True, True]
    for generation in (0, 1):
        _change(args, pools, generation)
        expected = _reference(args, pools)
        for graph in (native_graph, cute_graph):
            args["out"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                args["out"].reshape(expected.shape).double(),
                expected,
                rtol=0.1,
                atol=0.1,
            )
        _check_fp64(args["out"], expected)
