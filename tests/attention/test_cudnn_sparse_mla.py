# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

from flashinfer.mla import (
    trtllm_batch_decode_with_kv_cache_mla,
    trtllm_prefill_with_kv_cache_mla,
)
from flashinfer.mla import _cudnn_sparse


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Requires SM100/SM103")


@pytest.fixture
def cudnn_available():
    if _cudnn_sparse._get_sparse_attention_forward() is None:
        pytest.skip("Requires cuDNN Frontend >= 1.29.0 with DSA")


def _inputs(rows=6, dim=576, topk=65, layout="fixed", heads=64):
    torch.manual_seed(42)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    kv = torch.randn(128, 1, 32, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, 4096, (rows, topk), device="cuda", dtype=torch.int32)
    args = dict(
        query=q.view(rows, 1, heads, dim),
        kv_cache=kv,
        workspace_buffer=torch.zeros(32 << 20, device="cuda", dtype=torch.uint8),
        qk_nope_head_dim=192 if dim == 576 else 256,
        kv_lora_rank=512,
        qk_rope_head_dim=dim - 512,
        block_tables=ids.view(rows, 1, topk),
        seq_lens=torch.full((rows,), 4096, device="cuda", dtype=torch.int32),
        max_seq_len=4096,
        sparse_mla_top_k=topk,
        bmm1_scale=1 / 16,
        backend="cudnn",
        enable_pdl=False,
    )
    if dim == 512:
        args["sparse_mla_top_k_lens"] = torch.full(
            (rows,), topk, device="cuda", dtype=torch.int32
        )
    if layout == "compact":
        args.update(
            query=q,
            block_tables=ids,
            cum_seq_lens_q=torch.tensor(
                [0, rows // 2, rows], device="cuda", dtype=torch.int32
            ),
            max_q_len=rows - rows // 2,
            seq_lens=args["seq_lens"][:2],
        )
    elif layout == "prefill":
        args.update(
            query=q.view(1, rows, heads, dim),
            block_tables=ids.view(1, rows, topk),
            seq_lens=args["seq_lens"][:1],
        )
    return args


def _reference(args, row_ids=None):
    h, d = args["query"].shape[-2:]
    q = args["query"].view(-1, h, d)
    kv = args["kv_cache"].view(-1, d).float()
    ids = args["block_tables"].view(len(q), -1)
    lengths = args.get("sparse_mla_top_k_lens")
    outputs, stats = [], []
    for row in range(len(q)) if row_ids is None else row_ids:
        valid = (ids[row] >= 0) & (ids[row] < len(kv))
        if lengths is not None:
            valid &= torch.arange(ids.shape[1], device="cuda") < lengths[row]
        selected = kv[ids[row][valid].long()]
        if not len(selected):
            outputs.append(torch.zeros(h, 512, device="cuda"))
            stats.append(torch.full((h,), float("inf"), device="cuda"))
            continue
        logits = q[row].float() @ selected.T * args["bmm1_scale"]
        outputs.append(logits.softmax(-1) @ selected[:, :512])
        stats.append(logits.logsumexp(-1))
    lse = torch.stack(stats)
    if args.get("return_lse_base") != "basee":
        lse *= math.log2(math.e)
    return torch.stack(outputs), lse


@pytest.mark.parametrize(
    "dim,topk,layout",
    [
        (576, 63, "fixed"),
        (576, 64, "prefill"),
        (576, 65, "compact"),
        (576, 2048, "fixed"),
        (512, 65, "compact"),
        (512, 2048, "prefill"),
        (512, 2051, "fixed"),
    ],
)
def test_reference_and_preallocated_outputs(cudnn_available, dim, topk, layout):
    args = _inputs(dim=dim, topk=topk, layout=layout)
    args["out"] = torch.full(
        args["query"].shape[:-1] + (512,),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    args["return_lse"] = True
    ids = args["block_tables"].view(6, -1)
    # Exercise duplicate slots, invalid physical rows, and an empty selection.
    ids[0, :] = -1
    ids[1, 1] = ids[1, 0]
    ids[2, 0] = 4096
    ids[2, 1] = -7
    if "sparse_mla_top_k_lens" in args:
        args["sparse_mla_top_k_lens"][3] = topk // 2
        args["sparse_mla_top_k_lens"][4] = 0
    expected, expected_lse = _reference(args)
    for api in (
        trtllm_batch_decode_with_kv_cache_mla,
        trtllm_prefill_with_kv_cache_mla,
    ):
        out, lse = api(**args)
        assert out is args["out"]
        torch.testing.assert_close(
            out.view_as(expected).float(), expected, atol=0.003, rtol=0.03
        )
        torch.testing.assert_close(lse, expected_lse, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("base", [None, "basee", "base2"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_lse_buffer_and_units(cudnn_available, base, return_lse):
    args = _inputs(dim=576)
    args.update(
        lse=torch.empty(6, 1, 64, device="cuda"),
        return_lse=return_lse,
        return_lse_base=base,
    )
    # D576 also accepts a per-query selected length on the explicit backend.
    args["sparse_mla_top_k_lens"] = torch.tensor(
        [1, 2, 3, 4, 5, 0], device="cuda", dtype=torch.int32
    )
    expected, expected_lse = _reference(args)
    result = trtllm_batch_decode_with_kv_cache_mla(**args)
    if return_lse:
        out, lse = result
        assert lse is args["lse"]
    else:
        out = result
    torch.testing.assert_close(
        out.view_as(expected).float(), expected, atol=0.003, rtol=0.03
    )
    torch.testing.assert_close(
        args["lse"].view_as(expected_lse), expected_lse, atol=0.003, rtol=0.003
    )


@pytest.mark.parametrize("layout", ["decode", "fixed_multi", "compact"])
@pytest.mark.parametrize("backend", ["cudnn", "auto"])
def test_seq_lens_bound_valid_selected_slots(cudnn_available, layout, backend):
    args = _inputs(rows=128, dim=576, topk=64)
    args["query"].zero_()
    kv = args["kv_cache"].view(-1, 576)
    kv.copy_(torch.arange(len(kv), device="cuda")[:, None])
    args["block_tables"].copy_(torch.arange(64, device="cuda"))
    if layout == "fixed_multi":
        args["query"] = args["query"].view(32, 4, 64, 576)
        args["block_tables"] = args["block_tables"].view(32, 4, 64)
        args["seq_lens"] = args["seq_lens"][:32]
    elif layout == "compact":
        args["query"] = args["query"].view(128, 64, 576)
        args["block_tables"] = args["block_tables"].view(128, 64)
        args["seq_lens"] = args["seq_lens"][:32]
        lengths = torch.tensor([3, 5] * 16, device="cuda", dtype=torch.int32)
        args["cum_seq_lens_q"] = torch.cat(
            [torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0)]
        ).int()
        args["max_q_len"] = 5
    args["seq_lens"].fill_(32)
    args["seq_lens"][-1] = 65  # Also exercise the top-k capacity bound.
    expected = trtllm_batch_decode_with_kv_cache_mla(
        **{**args, "backend": "trtllm-gen"}
    )
    args["backend"] = backend
    actual = trtllm_batch_decode_with_kv_cache_mla(**args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Ignoring seq_lens gives 31.5 instead of these causal-prefix averages.
    first_mean = {"decode": 15.5, "fixed_multi": 14.0, "compact": 14.5}[layout]
    assert actual.flatten()[0].item() == first_mean
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = trtllm_batch_decode_with_kv_cache_mla(**args)
    args["seq_lens"].sub_(7)
    actual.fill_(float("nan"))
    graph.replay()
    expected = trtllm_batch_decode_with_kv_cache_mla(
        **{**args, "backend": "trtllm-gen"}
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rows,layout", [(127, "fixed"), (128, "fixed"), (129, "compact"), (128, "prefill")]
)
def test_auto_threshold(cudnn_available, monkeypatch, rows, layout):
    args = _inputs(rows=rows, topk=128, layout=layout)
    args["backend"] = "auto"
    calls = []
    original = _cudnn_sparse.try_cudnn_sparse_mla

    def spy(**kwargs):
        result = original(**kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(_cudnn_sparse, "try_cudnn_sparse_mla", spy)
    out = trtllm_batch_decode_with_kv_cache_mla(**args)
    assert calls == [rows >= 128]
    selected = [0, rows - 1]
    expected, _ = _reference(args, selected)
    torch.testing.assert_close(
        out.view(rows, 64, 512)[selected].float(), expected, atol=0.003, rtol=0.03
    )


def test_optional_dependency_fallback(monkeypatch):
    args = _inputs(rows=128, topk=128)
    monkeypatch.setattr(_cudnn_sparse, "_get_sparse_attention_forward", lambda: None)
    with pytest.raises(ImportError, match="nvidia-cudnn-frontend"):
        trtllm_batch_decode_with_kv_cache_mla(**args)
    args["backend"] = "auto"
    out = trtllm_batch_decode_with_kv_cache_mla(**args)
    expected, _ = _reference(args, [0, 127])
    torch.testing.assert_close(
        out.view(128, 64, 512)[[0, 127]].float(), expected, atol=0.003, rtol=0.03
    )


@pytest.mark.parametrize("base", [None, "basee", "base2"])
def test_compact_auto_preserves_lse_units(cudnn_available, base):
    args = _inputs(rows=128, topk=128, layout="compact")
    args.update(backend="auto", return_lse=True, return_lse_base=base)
    out, lse = trtllm_prefill_with_kv_cache_mla(**args)
    args["return_lse_base"] = "basee" if base is None else base
    expected, expected_lse = _reference(args, [0, 127])
    torch.testing.assert_close(out[[0, 127]].float(), expected, atol=0.003, rtol=0.03)
    torch.testing.assert_close(lse[[0, 127]], expected_lse, atol=0.003, rtol=0.003)


def test_fp8_falls_back(monkeypatch):
    args = _inputs(rows=128, topk=128)
    args["query"] = args["query"].to(torch.float8_e4m3fn)
    args["kv_cache"] = args["kv_cache"].to(torch.float8_e4m3fn)

    def must_not_import():
        raise AssertionError("FP8 must not import cuDNN DSA")

    monkeypatch.setattr(_cudnn_sparse, "_get_sparse_attention_forward", must_not_import)
    with pytest.raises(ValueError, match="BF16"):
        trtllm_batch_decode_with_kv_cache_mla(**args)
    args["backend"] = "auto"
    out = trtllm_batch_decode_with_kv_cache_mla(**args)
    args["backend"] = "trtllm-gen"
    expected = trtllm_batch_decode_with_kv_cache_mla(**args)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_sparse_trace_uses_physical_tokens(cudnn_available):
    from flashinfer.trace.templates.attention import (
        _trtllm_batch_decode_mla_sparse_reference,
    )

    args = _inputs(dim=576, topk=65)
    args["sparse_mla_top_k_lens"] = torch.tensor(
        [0, 1, 2, 3, 4, 5], device="cuda", dtype=torch.int32
    )
    out = trtllm_batch_decode_with_kv_cache_mla(**args)
    ref = _trtllm_batch_decode_mla_sparse_reference(**args)
    torch.testing.assert_close(out, ref, atol=0.003, rtol=0.03)


@pytest.mark.parametrize("heads", [8, 16, 32])
def test_tensor_parallel_heads_fall_back(monkeypatch, heads):
    args = _inputs(rows=128, heads=heads, topk=128)

    def must_not_import():
        raise AssertionError("Ineligible shape must not import cuDNN")

    monkeypatch.setattr(_cudnn_sparse, "_get_sparse_attention_forward", must_not_import)
    with pytest.raises(ValueError, match="64 query heads"):
        trtllm_batch_decode_with_kv_cache_mla(**args)
    args["backend"] = "auto"
    out = trtllm_batch_decode_with_kv_cache_mla(**args)
    expected, _ = _reference(args, [0, 127])
    torch.testing.assert_close(
        out.view(128, heads, 512)[[0, 127]].float(), expected, atol=0.003, rtol=0.03
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"bmm2_scale": 0.5}, "bmm2_scale=1"),
        ({"enable_pdl": True}, "enable_pdl"),
        ({"sinks": []}, "sinks"),
        ({"uses_shared_paged_kv_idx": False}, "shared_paged"),
        ({"skip_softmax_threshold_scale_factor": 0.5}, "skip_softmax"),
    ],
)
def test_explicit_rejects_unsupported_options(change, match):
    args = _inputs()
    args.update(change)
    with pytest.raises(ValueError, match=match):
        trtllm_batch_decode_with_kv_cache_mla(**args)


def test_workspace_validation(cudnn_available):
    args = _inputs()
    args["workspace_buffer"] = torch.empty(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="workspace"):
        trtllm_batch_decode_with_kv_cache_mla(**args)


def test_shared_workspace_across_auto_backends(cudnn_available):
    large = _inputs(rows=128, topk=128)
    small = _inputs(rows=1, topk=128)
    small["workspace_buffer"] = large["workspace_buffer"]
    large["backend"] = small["backend"] = "auto"
    for args in (large, small, large, small):
        out = trtllm_batch_decode_with_kv_cache_mla(**args)
        expected, _ = _reference(args, [0])
        torch.testing.assert_close(out[0].float(), expected, atol=0.003, rtol=0.03)


def test_malformed_indices_are_rejected(cudnn_available):
    args = _inputs(rows=128, topk=128)
    args["backend"] = "auto"
    args["block_tables"] = args["block_tables"].to(torch.int64)
    with pytest.raises(ValueError, match="dtype"):
        trtllm_batch_decode_with_kv_cache_mla(**args)


def test_graph_replay_updates_inputs_and_lengths(cudnn_available):
    args = _inputs(rows=128, dim=512, topk=2051)
    args["backend"] = "auto"
    args["return_lse"] = True
    trtllm_batch_decode_with_kv_cache_mla(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = trtllm_batch_decode_with_kv_cache_mla(**args)
    args["query"].mul_(0.7)
    args["kv_cache"].add_(0.125)
    args["block_tables"].add_(7).remainder_(4096)
    args["sparse_mla_top_k_lens"][:2] = torch.tensor(
        [0, 17], device="cuda", dtype=torch.int32
    )
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    expected, expected_lse = _reference(args, [0, 1, 127])
    torch.testing.assert_close(
        out.view(128, 64, 512)[[0, 1, 127]].float(), expected, atol=0.003, rtol=0.03
    )
    torch.testing.assert_close(lse[[0, 1, 127]], expected_lse, atol=0.003, rtol=0.003)


def test_cold_capture_requires_warmup(cudnn_available):
    args = _inputs(topk=67)
    _cudnn_sparse._operations.clear()
    with (
        pytest.raises(RuntimeError, match="Warm up"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        trtllm_batch_decode_with_kv_cache_mla(**args)


def test_concurrent_stream_workspaces(cudnn_available):
    args = [_inputs(dim=576), _inputs(dim=576)]
    args[1]["query"].mul_(2)
    expected = [_reference(a)[0] for a in args]
    trtllm_batch_decode_with_kv_cache_mla(**args[0])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream, call in zip(streams, args, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(trtllm_batch_decode_with_kv_cache_mla(**call))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for out, ref in zip(outputs, expected, strict=True):
        torch.testing.assert_close(out.view_as(ref).float(), ref, atol=0.003, rtol=0.03)
