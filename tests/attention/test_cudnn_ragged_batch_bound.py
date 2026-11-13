# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hopper batch cache bounds with real replans and retained CUDA Graphs."""

from itertools import accumulate

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill


def _require_hopper(monkeypatch):
    if not prefill.CUDNN_AVAILABLE or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper and cuDNN")
    if not prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    monkeypatch.setenv("FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE", "1")


def _indptr(lengths):
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)


def _reference(q, k, v, qs, ks, causal, scales):
    outputs, stats = [], []
    q_start = k_start = 0
    for nq, nk in zip(qs, ks, strict=True):
        qq = q[q_start : q_start + nq].double().transpose(0, 1)
        kk = k[k_start : k_start + nk].double().repeat_interleave(4, 1).transpose(0, 1)
        vv = v[k_start : k_start + nk].double().repeat_interleave(4, 1).transpose(0, 1)
        scores = qq @ kk.transpose(-1, -2) * (256**-0.5 * scales[0] * scales[1])
        if causal:
            rows = torch.arange(nq, device=q.device)
            cols = torch.arange(nk, device=q.device)
            scores.masked_fill_(cols[None] > rows[:, None] + nk - nq, -torch.inf)
        outputs.append((scores.softmax(-1) @ vv * scales[2]).transpose(0, 1).float())
        stats.append(scores.logsumexp(-1).transpose(0, 1).float())
        q_start += nq
        k_start += nk
    return torch.cat(outputs), torch.cat(stats)


def _check(o, lse, expected, dtype):
    tolerance = 0.002 if dtype == torch.float16 else 0.02
    torch.testing.assert_close(o.float(), expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, expected[1], atol=0.002, rtol=0.002)


def _plan(w, qs, ks, dtype, causal):
    w.plan(
        _indptr(qs),
        _indptr(ks),
        16,
        4,
        256,
        q_data_type=dtype,
        kv_data_type=dtype,
        causal=causal,
        max_token_per_sequence=64,
        max_sequence_kv=512,
    )
    if w.is_cuda_graph_enabled:
        # Graph mode already declares explicit lengths and exact bounds.
        assert not w._cudnn_plan.metadata._bounded_batch
        assert w._cudnn_plan.override is None
    else:
        assert w._cudnn_plan.metadata._bounded_batch
        assert w._cudnn_plan.override == (1 << (len(qs) - 1).bit_length(), 128, 65536)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_hopper_batch_bucket_replan(monkeypatch, dtype, causal):
    _require_hopper(monkeypatch)
    torch.manual_seed(13561)
    ws = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cudnn")
    graphs = []
    for batch in (2, 3, 4, 5, 3):
        qs = [31 + i for i in range(batch)]
        ks = [257 + 9 * i for i in range(batch)]
        q = torch.randn(sum(qs), 16, 256, device="cuda", dtype=dtype)
        k = torch.randn(sum(ks), 4, 256, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        _plan(w, qs, ks, dtype, causal)
        out, lse = w.run(q, k, v, return_lse=True, lse_base="ln")
        _check(out, lse, _reference(q, k, v, qs, ks, causal, (1, 1, 1)), dtype)
        graphs.append(w._cudnn_plan.bound_graph.graph)
    assert graphs[0] is not graphs[1]
    assert graphs[1] is graphs[2] is graphs[4]
    assert graphs[3] is not graphs[2]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale_kind", ["default", "scalar", "tensor"])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_hopper_retained_batch_bound_graph(
    monkeypatch, dtype, causal, scale_kind, graph_mode
):
    _require_hopper(monkeypatch)
    torch.manual_seed(13577)
    qs, ks = [31, 33, 32], [257, 271, 263]
    q = torch.randn(sum(qs), 16, 256, device="cuda", dtype=dtype)
    k = torch.randn(sum(ks), 4, 256, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    ws = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws,
        backend="cudnn",
        use_cuda_graph=graph_mode,
        qo_indptr_buf=torch.empty(4, device="cuda", dtype=torch.int32),
        kv_indptr_buf=torch.empty(4, device="cuda", dtype=torch.int32),
    )
    _plan(w, qs, ks, dtype, causal)
    scales = (1.0, 1.0, 1.0) if scale_kind == "default" else (0.5, 0.75, 0.25)
    kwargs = (
        {}
        if scale_kind == "default"
        else dict(zip(("q_scale", "k_scale", "v_scale"), scales, strict=True))
    )
    if scale_kind == "tensor":
        kwargs = {
            name: torch.full((1,), value, device="cuda", dtype=torch.float32)
            for name, value in kwargs.items()
        }
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)

    def run():
        w.run(q, k, v, out=out, lse=lse, return_lse=True, lse_base="ln", **kwargs)

    run()
    _check(out, lse, _reference(q, k, v, qs, ks, causal, scales), dtype)
    original_graph = w._cudnn_plan.bound_graph
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            run()
        # Same allocation and fixed batch, with changed Q/KV partitions.
        qs, ks = [28, 35, 33], [249, 278, 264]
        q.mul_(0.75)
        k.neg_()
        v.mul_(0.5)
        _plan(w, qs, ks, dtype, causal)
        if scale_kind == "tensor":
            scales = (0.75, 0.5, 0.625)
            for tensor, value in zip(kwargs.values(), scales, strict=True):
                tensor.fill_(value)
        reference = _reference(q, k, v, qs, ks, causal, scales)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        capture.replay()
        _check(out, lse, reference, dtype)
        run()
        if scale_kind == "default":
            assert w._cudnn_plan.bound_graph.graph is original_graph.graph
        replacement = torch.empty_like(ws)
        w.reset_workspace_buffer(replacement, w._int_workspace_buffer)
        other, other_lse = torch.empty_like(out), torch.empty_like(lse)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            rebound = (q.clone(), k.clone(), v.clone())
            w.run(
                *rebound,
                out=other,
                lse=other_lse,
                return_lse=True,
                lse_base="ln",
                **kwargs,
            )
        torch.cuda.current_stream().wait_stream(stream)
        _check(other, other_lse, reference, dtype)
        # The capture retains its previous workspace and metadata pointers.
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        capture.replay()
        _check(out, lse, reference, dtype)
        torch.cuda.synchronize()
    finally:
        capture.reset()
