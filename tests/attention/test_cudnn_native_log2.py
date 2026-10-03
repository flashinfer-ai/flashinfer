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

"""Native Stats must preserve base, routing and capture across wrapper reuse."""

import math
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.cudnn import decode, prefill, utils


@pytest.mark.parametrize("support", [False, True])
@pytest.mark.parametrize("failure", [None, "missing_argument", "internal_error"])
def test_log2_probe_checks_backend_headers_and_caches(monkeypatch, support, failure):
    class Declined(Exception):
        pass

    class Tensor:
        def set_output(self, value):
            return self

        def set_data_type(self, value):
            return self

    class Graph:
        probes = 0

        def __init__(self, **kwargs):
            pass

        def tensor(self, **kwargs):
            return Tensor()

        def sdpa(self, **kwargs):
            assert kwargs["stats_use_log2"]
            if failure == "missing_argument":
                raise TypeError(
                    "sdpa() got unexpected keyword argument 'stats_use_log2'"
                )
            if failure == "internal_error":
                raise TypeError("malformed stats_use_log2 metadata")
            return Tensor(), Tensor()

        def validate(self):
            pass

        def create_execution_plans(self, modes):
            assert modes == ["A"]

        def backend_plan_entries(self):
            Graph.probes += 1
            return [object()] if support else []

    class Backend:
        pygraph = Graph
        data_type = SimpleNamespace(HALF=1, FLOAT=2)
        heur_mode = SimpleNamespace(A="A")
        cudnnGraphNotSupportedError = Declined

        @staticmethod
        def backend_version():
            return 92700

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(utils, "get_cudnn_attention_handle", lambda *args: None)
    if failure == "internal_error":
        with pytest.raises(TypeError, match="malformed"):
            utils.supports_native_cudnn_log2(Backend, torch.device("cuda", 0))
        return
    for _ in range(2):
        assert utils.supports_native_cudnn_log2(Backend, torch.device("cuda", 0)) == (
            support and failure is None
        )
    assert Graph.probes == (0 if failure else 1)


@pytest.mark.parametrize("missing_api", [False, True])
def test_log2_probe_legacy_fe_or_runtime_never_touches_cuda(monkeypatch, missing_api):
    class Backend:
        pygraph = object if missing_api else SimpleNamespace(backend_plan_entries=None)

        @staticmethod
        def backend_version():
            return 92700 if missing_api else 92600

    def unexpected(*args):
        pytest.fail("unsupported capability must not query CUDA")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    assert not utils.supports_native_cudnn_log2(Backend, torch.device("cuda", 0))


def test_log2_graph_fallback_only_on_capability_decline():
    class Declined(Exception):
        pass

    backend = SimpleNamespace(cudnnGraphNotSupportedError=Declined)
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("stats_use_log2"):
            raise Declined("this layout does not support native log2")
        return "ln graph", []

    assert utils.build_cudnn_graph_with_log2(backend, build, (), {}, True) == (
        "ln graph",
        [],
    )
    assert calls == [{"stats_use_log2": True}, {}]

    def broken(*args, **kwargs):
        raise RuntimeError("compiler failure")

    with pytest.raises(RuntimeError, match="compiler failure"):
        utils.build_cudnn_graph_with_log2(backend, broken, (), {}, True)


def test_log2_actual_graph_must_keep_backend_candidates():
    class Declined(Exception):
        pass

    graph = SimpleNamespace(
        validate=lambda: None,
        build_operation_graph=lambda: None,
        create_execution_plans=lambda modes: None,
        backend_plan_entries=lambda: [],
    )
    with pytest.raises(Declined, match="exclude the cuDNN backend"):
        utils.require_native_cudnn_log2(
            graph,
            SimpleNamespace(
                cudnnGraphNotSupportedError=Declined, heur_mode=SimpleNamespace(A="A")
            ),
        )
    assert not getattr(graph, "_flashinfer_stats_use_log2", False)


@pytest.mark.parametrize(
    "return_lse,lse_base,dtype",
    [
        (False, "log2", torch.bfloat16),
        (True, "ln", torch.bfloat16),
        (True, "log2", torch.float8_e4m3fn),
    ],
)
def test_prefill_ln_no_lse_or_fp8_does_not_probe(
    monkeypatch, return_lse, lse_base, dtype
):
    def unexpected(*args):
        pytest.fail("ln and output-only plans must not probe native log2")

    monkeypatch.setattr(prefill, "supports_native_cudnn_log2", unexpected)
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    monkeypatch.setattr(
        prefill, "_build_prefill_graph", lambda **kwargs: (object(), [])
    )
    q = torch.empty(4, 2, 64, dtype=dtype)
    lengths = torch.tensor([4], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        4, 4, False, return_lse, actual_seq_lens_q=lengths, actual_seq_lens_kv=lengths
    )
    prefill.prepare_cudnn_batch_prefill(
        q, q, q, 0.125, torch.empty(0), metadata=metadata, lse_base=lse_base
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_fp8_decode_is_rejected_before_log2_probe(monkeypatch, prepared):
    def unexpected(*args):
        pytest.fail("unsupported FP8 decode must not probe native log2")

    monkeypatch.setattr(decode, "supports_native_cudnn_log2", unexpected)
    monkeypatch.setattr(decode, "CUDNN_AVAILABLE", True)
    q = torch.empty(1, 2, 64, dtype=torch.float8_e4m3fn)
    k = torch.empty(1, 1, 16, 64, dtype=q.dtype)
    kwargs = dict(
        max_sequence_kv=16,
        actual_seq_lens_kv=torch.tensor([16], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        return_lse=True,
        out=torch.empty_like(q),
        lse=torch.empty(1, 2),
        q_len_per_req=1,
        window_left=-1,
        sinks=None,
    )
    with pytest.raises(
        ValueError, match="only supports torch.float16 and torch.bfloat16"
    ):
        if prepared:
            decode.prepare_cudnn_batch_decode(q, k, k, 0.125, **kwargs)
        else:
            decode.cudnn_batch_decode_with_kv_cache(
                q, k, k, 0.125, torch.empty(0), **kwargs
            )


def test_native_log2_graph_decline_is_cached(monkeypatch):
    calls = []
    fallback = object()

    def build(*args, **kwargs):
        calls.append(kwargs.get("stats_use_log2", False))
        if kwargs.get("stats_use_log2"):
            raise decode.cudnn.cudnnGraphNotSupportedError("unsupported log2 layout")
        return fallback, []

    monkeypatch.setattr(decode, "_make_decode_graph", build)
    q = torch.empty(7, 5, 32, dtype=torch.float16, device="meta")
    k = torch.empty(7, 1, 77, 32, dtype=q.dtype, device=q.device)
    for _ in range(2):
        result, _ = decode._build_decode_graph(
            q, k, k, 0.125, max_sequence_kv=77, return_lse=True, stats_use_log2=True
        )
        assert result is fallback
    assert calls == [True, False]


def _reference(q, k, v, q_lens, kv_lens):
    outputs, stats = [], []
    qstart = 0
    for batch, (nq, nk) in enumerate(zip(q_lens, kv_lens, strict=True)):
        query = q[qstart : qstart + nq].float()
        key = (
            k[2 * batch : 2 * batch + 2]
            .flatten(0, 1)[:nk]
            .repeat_interleave(2, 1)
            .float()
        )
        value = (
            v[2 * batch : 2 * batch + 2]
            .flatten(0, 1)[:nk]
            .repeat_interleave(2, 1)
            .float()
        )
        scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(128)
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        stats.append(scores.logsumexp(-1).T)
        qstart += nq
    return torch.cat(outputs), torch.cat(stats)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN FE")
@pytest.mark.parametrize("kind", ["decode", "ragged", "paged"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("native", [False, True], ids=["fallback", "native"])
def test_cudnn_log2_capture_replan_and_base_switch(monkeypatch, kind, dtype, native):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer")
    device = torch.device("cuda", torch.cuda.current_device())
    if native and not utils.supports_native_cudnn_log2(prefill.cudnn, device):
        pytest.skip("FE headers and runtime do not support native backend log2 Stats")
    torch.manual_seed(812)
    q_lens = [1, 1] if kind == "decode" else [3, 5]
    kv_lens = [19, 25]
    q = torch.randn(sum(q_lens), 4, 128, device=device, dtype=dtype)
    k = torch.randn(4, 16, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    out, lse = torch.empty_like(q), torch.empty(q.shape[:2], device=device)
    ws = torch.empty(128 << 20, device=device, dtype=torch.uint8)
    ip = torch.tensor([0, 2, 4], dtype=torch.int32)
    indices = torch.arange(4, dtype=torch.int32)
    last = torch.tensor([3, 9], dtype=torch.int32)
    wrapper_type = {
        "decode": flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        "paged": flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        "ragged": flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
    }[kind]

    def plan(wrapper):
        qo = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32)
        if kind == "decode":
            wrapper.plan(ip, indices, last, 4, 2, 128, 16, q_data_type=dtype)
        elif kind == "paged":
            wrapper.plan(qo, ip, indices, last, 4, 2, 128, 16, q_data_type=dtype)
        else:
            kv = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32)
            wrapper.plan(qo, kv, 4, 2, 128, q_data_type=dtype)

    # Ragged storage remains live across capture; refill after changing paged V.
    keys = torch.cat([k[:2].flatten(0, 1)[:19], k[2:].flatten(0, 1)[:25]])
    values = torch.cat([v[:2].flatten(0, 1)[:19], v[2:].flatten(0, 1)[:25]])

    def run(wrapper, base="log2"):
        kwargs = dict(out=out, lse=lse, return_lse=True)
        if kind != "decode":
            kwargs["lse_base"] = base
        if kind == "ragged":
            return wrapper.run(q, keys, values, **kwargs)
        return wrapper.run(q, (k, v), **kwargs)

    def graph_of(wrapper):
        prepared = wrapper._cudnn_prepared
        return prepared.graph

    def check(base="log2"):
        expected, expected_lse = _reference(q, k, v, q_lens, kv_lens)
        torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
        if base == "log2":
            expected_lse *= math.log2(math.e)
        torch.testing.assert_close(lse, expected_lse, atol=0.003, rtol=0.003)

    # Check the previous ln + conversion route against the same math reference.
    with monkeypatch.context() as patch:
        patch.setattr(decode, "supports_native_cudnn_log2", lambda *args: False)
        patch.setattr(prefill, "supports_native_cudnn_log2", lambda *args: False)
        baseline = wrapper_type(ws, "NHD", backend="cudnn")
        plan(baseline)
        run(baseline)
        check()
    if not native:
        monkeypatch.setattr(decode, "supports_native_cudnn_log2", lambda *args: False)
        monkeypatch.setattr(prefill, "supports_native_cudnn_log2", lambda *args: False)
    wrapper = wrapper_type(ws, "NHD", backend="cudnn")
    plan(wrapper)
    run(wrapper)
    check()
    built = graph_of(wrapper)
    assert bool(getattr(built, "_flashinfer_stats_use_log2", False)) == native

    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            run(wrapper)
        q.mul_(0.75)
        v.mul_(-0.5)
        values.copy_(torch.cat([v[:2].flatten(0, 1)[:19], v[2:].flatten(0, 1)[:25]]))
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        capture.replay()
        check()
        if kind != "decode":
            run(wrapper, "ln")
            check("ln")
            assert not getattr(graph_of(wrapper), "_flashinfer_stats_use_log2", False)
            q_lens[:] = [4, 4]
        plan(wrapper)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        run(wrapper)
        check()
        assert (
            bool(getattr(graph_of(wrapper), "_flashinfer_stats_use_log2", False))
            == native
        )
    finally:
        capture.reset()
