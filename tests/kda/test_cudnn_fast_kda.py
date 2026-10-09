# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native submission must preserve the public cuDNN KDA contract."""

from unittest.mock import patch

import pytest
import torch

from flashinfer import _kda_cudnn_fast as fast
from flashinfer.kda import recurrent_kda
from tests.kda.test_recurrent_kda_cudnn_backend import _gate_kwargs, _make_inputs
from tests.test_helpers.cudnn_linear_attention import requires_cudnn_linear_attention

pytestmark = requires_cudnn_linear_attention


def make_args(length=256):
    inputs = _make_inputs([length], 6, initial_state=True, cu_seqlens_dtype=torch.int32)
    return dict(
        inputs,
        **{k: v for k, v in _gate_kwargs(inputs).items() if k not in inputs},
        output=torch.empty_like(inputs["v"]),
        output_final_state=True,
        backend="cudnn",
    )


@pytest.fixture(autouse=True)
def native_cache(monkeypatch):
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        pytest.skip("native KDA requires PyTorch's DLPack exchange API")
    monkeypatch.setattr(fast, "_enabled", True)
    monkeypatch.setattr(fast, "_native_unavailable", False)
    fast.clear()
    with torch.inference_mode():
        yield
    torch.cuda.synchronize()
    fast.clear()


def compare(args, count=1, hits=None):
    seed = args["initial_state"].clone()
    with patch.object(fast, "_enabled", False):
        for _ in range(count):
            expected, _ = recurrent_kda(**args)
        torch.cuda.synchronize()
        expected = expected.clone()
        expected_state = args["initial_state"].clone()
    args["initial_state"].copy_(seed)
    before = fast.stats()["hits"]
    for _ in range(count):
        actual, state = recurrent_kda(**args)
        assert actual.data_ptr() == args["output"].data_ptr()
        assert state is args["initial_state"]
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    if hits is not None:
        assert fast.stats()["hits"] - before == hits


def test_warmup_rerun_and_new_operands():
    args = make_args()
    compare(args, hits=0)
    assert fast.stats()["entries"] == 1
    compare(args, count=8, hits=8)
    args["q"].add_(0.125)
    compare(args, hits=1)
    compare(make_args(), hits=1)
    old_output = args["output"].detach()
    old_output.fill_(37)
    args["output"].set_(torch.empty_like(args["output"]))
    compare(args, hits=1)
    assert torch.all(old_output == 37)


def test_shape_scalar_and_layout_changes():
    args = make_args()
    compare(args, hits=0)
    compare(make_args(512), hits=0)
    args["scale"] = 0.25
    compare(args, hits=0)
    compare(args, hits=1)
    args["cu_seqlens"] = args["cu_seqlens"].long()
    compare(args, hits=0)


def test_state_version_and_return_flag():
    with torch.inference_mode(False), torch.no_grad():
        args = make_args()
    compare(args, hits=0)
    before = args["initial_state"]._version
    recurrent_kda(**args)
    assert args["initial_state"]._version == before + 1
    args["output_final_state"] = False
    before = fast.stats()["hits"]
    _, state = recurrent_kda(**args)
    assert state is None
    assert fast.stats()["hits"] == before + 1


def test_stream_isolation_and_cache_bound():
    args = make_args()
    streams = [torch.cuda.Stream() for _ in range(fast._max_entries + 1)]
    torch.cuda.synchronize()
    for index, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            compare(args, hits=0)
            compare(args, hits=1)
        assert fast.stats()["entries"] == min(index + 1, fast._max_entries)
    fast.clear()
    assert fast.stats()["entries"] == 0


def test_capture_falls_back_and_replays_after_clear():
    args = make_args()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        compare(args, hits=0)
        compare(args, hits=1)
    torch.cuda.synchronize()
    seed = args["initial_state"].clone()
    graph = torch.cuda.CUDAGraph()
    before = fast.stats()["hits"]
    with torch.cuda.graph(graph, stream=stream):
        recurrent_kda(**args)
    assert fast.stats()["hits"] == before
    fast.clear()
    args["q"].add_(0.125)
    args["initial_state"].copy_(seed)
    with patch.object(fast, "_enabled", False):
        expected, _ = recurrent_kda(**args)
        torch.cuda.synchronize()
        expected = expected.clone()
        expected_state = args["initial_state"].clone()
    args["initial_state"].copy_(seed)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(args["output"], expected, rtol=0, atol=0)
    torch.testing.assert_close(args["initial_state"], expected_state, rtol=0, atol=0)


def test_missing_build_falls_back_once(monkeypatch):
    from flashinfer.jit import cudnn_fast_kda

    monkeypatch.setattr(fast, "cuDNNFastKDA", None)
    with patch.object(
        cudnn_fast_kda,
        "get_cudnn_fast_kda_module",
        side_effect=RuntimeError("no compiler"),
    ) as build:
        args = make_args()
        compare(args, hits=0)
        compare(args, hits=0)
    assert build.call_count == 1
    assert fast.stats()["entries"] == 0


def test_state_pool_bypasses_native():
    import flashinfer.cudnn as cudnn_backend

    args = make_args()
    args["ssm_state_indices"] = torch.zeros(1, device="cuda", dtype=torch.int32)
    with (
        patch.object(fast, "try_execute", side_effect=AssertionError("native pool")),
        patch.object(fast, "prepare", side_effect=AssertionError("native pool")),
        patch.object(
            cudnn_backend,
            "cudnn_recurrent_kda",
            return_value=(args["output"], args["initial_state"]),
        ) as adapter,
    ):
        recurrent_kda(**args)
    assert adapter.call_args.kwargs["state_indices"] is args["ssm_state_indices"]


def test_incompatible_frontend_plan_falls_back(monkeypatch):
    from types import SimpleNamespace

    from flashinfer.cudnn import linear_attention as la

    original_prepare = fast._prepare
    incomplete_plan = SimpleNamespace(
        _fi_la_overwrite=True,
        _fi_la_ordered=True,
        _compiled_plans=[],
        _plan_index=0,
        _normalize_ordered=None,
    )

    def incompatible(inputs, config):
        with patch.object(la, "_build_la_graph", return_value=(incomplete_plan, None)):
            original_prepare(inputs, config)

    monkeypatch.setattr(fast, "_prepare", incompatible)
    args = make_args()
    compare(args, hits=0)
    compare(args, hits=0)
    assert fast._native_unavailable
    assert fast.stats()["entries"] == 0
