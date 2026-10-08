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

"""Host-only contracts shared by public and prepared cuDNN execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import decode, prefill
from flashinfer.cudnn.utils import supports_ordered_cudnn_execution


def test_ordered_execution_feature_detection():
    class Legacy:
        def execute(self, tensor_dict, workspace=None, handle=None):
            pass

    class Ordered:
        def execute(self, tensors, workspace=None, handle=None, tensor_uids=None):
            pass

    assert not supports_ordered_cudnn_execution(Legacy)
    assert supports_ordered_cudnn_execution(Ordered)


def test_ordered_execution_errors_are_not_retried(monkeypatch):
    calls = []

    class Graph:
        def execute(self, buffers, **kwargs):
            calls.append((buffers, kwargs))
            raise RuntimeError("invalid binding")

    monkeypatch.setattr(decode, "_create_cudnn_handle", lambda stream: 17)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    tensor = torch.empty(1)
    with pytest.raises(RuntimeError, match="invalid binding"):
        decode._execute_decode(
            Graph(),
            tensor,
            tensor,
            tensor,
            tensor,
            None,
            tensor,
            actual_seq_lens_q=tensor,
            actual_seq_lens_kv=tensor,
            block_tables=tensor,
            return_lse=False,
            sinks=None,
            execution_bindings=(1, 2, 3, 1000, 100, 101, 201, 202),
        )
    assert len(calls) == 1
    assert calls[0][1]["handle"] == 17
    assert calls[0][1]["workspace"] is tensor


@pytest.mark.parametrize("ordered", [False, True])
def test_planned_prefill_rebinds_tensors_and_metadata(monkeypatch, ordered):
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    monkeypatch.setattr(prefill, "_create_cudnn_handle", lambda stream: 23)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    calls = []

    def execute(buffers, *, tensor_uids=None, **kwargs):
        mapping = (
            dict(zip(tensor_uids, buffers, strict=True))
            if tensor_uids is not None
            else buffers
        )
        calls.append((buffers, mapping, kwargs))
        mapping[1000].copy_(mapping[1])
        mapping[1001].fill_(1)

    class Graph:
        def execute(self, buffers, *, tensor_uids=None, **kwargs):
            execute(buffers, tensor_uids=tensor_uids, **kwargs)

    graph = Graph() if ordered else SimpleNamespace(execute=execute)
    prepared = prefill.CudnnPrefillGraph(
        None, graph, override_cache=None, return_lse=True
    )
    previous = None
    sources = []
    for value in (1, 2):
        indptr = torch.tensor([0, 3], dtype=torch.int32)
        metadata = prefill._PrefillMetadata(
            3,
            3,
            False,
            True,
            cu_seq_lens_q=indptr,
            cu_seq_lens_kv=indptr,
            batch_offsets_q=indptr,
            batch_offsets_o=indptr,
            batch_offsets_k=indptr,
            batch_offsets_v=indptr,
            batch_offsets_stats=indptr,
        )
        q = torch.full((3, 4, 8), float(value))
        out, lse, workspace = torch.empty_like(q), torch.empty(3, 4), torch.empty(8)
        plan = prefill._CudnnPrefillPlan.prepare(metadata, q.dtype, q.device, previous)
        prepared.run_planned(q, q, q, out, lse, workspace, plan=plan, lse_base="ln")
        torch.testing.assert_close(out, q)
        torch.testing.assert_close(lse, torch.ones_like(lse))
        assert calls[-1][1][100].data_ptr() == indptr.data_ptr()
        assert calls[-1][1][1] is q
        assert calls[-1][1][1000] is out
        assert calls[-1][2] == dict(workspace=workspace, handle=23)
        sources.append((q, out, indptr))
        previous = plan
    assert calls[0][0] is not calls[1][0]
    assert isinstance(calls[0][0], list if ordered else dict)
    torch.testing.assert_close(sources[0][1], sources[0][0])


@pytest.fixture
def decode_inputs(monkeypatch):
    monkeypatch.setattr(decode, "CUDNN_AVAILABLE", True)
    q = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 16, 8, dtype=q.dtype)
    return (
        q,
        k,
        k.clone(),
        dict(
            actual_seq_lens_kv=torch.full((2, 1, 1, 1), 8, dtype=torch.int32),
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            out=torch.empty_like(q),
            return_lse=True,
            lse=torch.empty(2, 4),
            q_len_per_req=1,
            sinks=torch.arange(4, dtype=torch.float32),
        ),
    )


def test_decode_public_allocation_and_copy_prepared_boundary(
    monkeypatch, decode_inputs
):
    q, k, v, kwargs = decode_inputs
    q = torch.stack((q, -q), dim=-1)[..., 0]
    assert q.stride(-1) == 2
    seen = {}

    def execute(pack, **unused):
        seen.update(pack)
        output = pack[decode.UIDs.O_UID.value]
        output.copy_(pack[decode.UIDs.Q_UID.value].view_as(output))
        pack[decode.UIDs.STATS_UID.value].fill_(1)

    monkeypatch.setattr(
        decode,
        "_build_decode_graph",
        lambda **kw: (SimpleNamespace(execute=execute), []),
        raising=False,
    )
    monkeypatch.setattr(decode, "_create_cudnn_handle", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    out, lse = decode.cudnn_batch_decode_with_kv_cache(
        q,
        k,
        v,
        0.5,
        torch.empty(0),
        max_sequence_kv=8,
        **dict(kwargs, out=None, lse=None),
    )
    torch.testing.assert_close(out, q)
    assert lse.shape == (2, 4)
    assert seen[decode.UIDs.Q_UID.value].stride(-1) == 1
    assert seen[decode.UIDs.SINK_UID.value].shape == (1, 4, 1, 1)
    with pytest.raises(ValueError, match="unit innermost stride"):
        decode.prepare_cudnn_batch_decode(
            q, k, v, 0.5, max_sequence_kv=8, window_left=-1, **kwargs
        )


@pytest.mark.parametrize(
    "buffer,fault",
    [
        (buffer, fault)
        for buffer in ("out", "lse", "sinks")
        for fault in ("shape", "dtype", "device", "stride")
        if (buffer, fault) != ("sinks", "stride")
    ],
)
def test_decode_public_prepared_share_cold_validation(decode_inputs, buffer, fault):
    q, k, v, kwargs = decode_inputs
    value = kwargs[buffer]
    if fault == "shape":
        value = value[:-1]
    elif fault == "dtype":
        value = value.to(torch.float64)
    elif fault == "device":
        value = torch.empty_like(value, device="meta")
    else:
        value = torch.stack((value, value), dim=-1)[..., 0]
    kwargs[buffer] = value
    for prepared in (False, True):
        with pytest.raises(ValueError, match=buffer):
            if prepared:
                decode.prepare_cudnn_batch_decode(
                    q, k, v, 0.5, max_sequence_kv=8, window_left=-1, **kwargs
                )
            else:
                decode.cudnn_batch_decode_with_kv_cache(
                    q, k, v, 0.5, torch.empty(0), max_sequence_kv=8, **kwargs
                )


@pytest.mark.parametrize("name", ["batch_offsets_q", "batch_offsets_o"])
@pytest.mark.parametrize("fault", ["shape", "dtype", "device", "values"])
def test_decode_public_rejects_unsupported_offsets(decode_inputs, name, fault):
    q, k, v, kwargs = decode_inputs
    offsets = torch.tensor([0, 32, 64], dtype=torch.int64)
    error = ValueError
    if fault == "shape":
        offsets = offsets.view(3, 1)
    elif fault == "dtype":
        offsets = offsets.float()
    elif fault == "device":
        offsets = offsets.to("meta")
    else:
        offsets[0] = 1
        error = RuntimeError  # CPU equivalent of the CUDA asynchronous assertion.
    with pytest.raises(error, match=name):
        decode.cudnn_batch_decode_with_kv_cache(
            q, k, v, 0.5, torch.empty(0), max_sequence_kv=8, **kwargs, **{name: offsets}
        )


def test_decode_cubin_preserves_offsets(monkeypatch, decode_inputs):
    q, k, v, kwargs = decode_inputs
    monkeypatch.setattr(decode, "CUDNN_AVAILABLE", False)
    calls = []
    monkeypatch.setattr(
        decode,
        "get_cudnn_fmha_gen_module",
        lambda: SimpleNamespace(decode=lambda *args: calls.append(args)),
    )
    offsets_q = torch.tensor([32, 64, 96], dtype=torch.int64)
    offsets_o = torch.tensor([64, 96, 128], dtype=torch.int64)
    decode.cudnn_batch_decode_with_kv_cache(
        q,
        k,
        v,
        0.5,
        torch.empty(0),
        max_sequence_kv=8,
        **dict(kwargs, return_lse=False, lse=None, sinks=None),
        batch_offsets_q=offsets_q,
        batch_offsets_o=offsets_o,
    )
    assert len(calls) == 1
    assert calls[0][10] is offsets_q
    assert calls[0][11] is offsets_o


@pytest.mark.parametrize("ordered", [False, True])
def test_decode_binding_is_per_call_and_lse_is_base2(
    monkeypatch, decode_inputs, ordered
):
    q, k, v, kwargs = decode_inputs
    packs, raw_packs = [], []

    def execute(pack, tensor_uids=None, **unused):
        raw_packs.append(pack)
        if tensor_uids is not None:
            pack = dict(zip(tensor_uids, pack, strict=True))
        packs.append(pack)
        pack[decode.UIDs.O_UID.value].copy_(pack[decode.UIDs.Q_UID.value])
        pack[decode.UIDs.STATS_UID.value].fill_(torch.log(torch.tensor(2.0)))

    class OrderedGraph:
        def execute(self, buffers, tensor_uids=None, **kwargs):
            execute(buffers, tensor_uids, **kwargs)

    graph = OrderedGraph() if ordered else SimpleNamespace(execute=execute)
    monkeypatch.setattr(
        decode, "_build_decode_graph", lambda *a, **kw: (graph, []), raising=False
    )
    monkeypatch.setattr(decode, "_create_cudnn_handle", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    plan = decode.prepare_cudnn_batch_decode(
        q, k, v, 0.5, max_sequence_kv=8, window_left=-1, **kwargs
    )
    out0, out1 = torch.empty_like(q), torch.empty_like(q)
    lse0, lse1 = torch.empty(2, 4), torch.empty(2, 4)
    for query, output, stats in ((q, out0, lse0), (-q, out1, lse1)):
        plan.run(
            query,
            k,
            v,
            output,
            stats,
            torch.empty(0),
            actual_seq_lens_kv=kwargs["actual_seq_lens_kv"],
            block_tables=kwargs["block_tables"],
        )
        torch.testing.assert_close(output, query)
        torch.testing.assert_close(stats, torch.ones_like(stats))
    assert packs[0] is not packs[1]
    assert raw_packs[0] is not raw_packs[1]
    assert isinstance(raw_packs[0], list if ordered else dict)
    assert packs[0][decode.UIDs.O_UID.value] is out0
    assert packs[0][decode.UIDs.Q_UID.value] is q
    decode.cudnn_batch_decode_with_kv_cache(
        q, k, v, 0.5, torch.empty(0), max_sequence_kv=8, **kwargs
    )
    torch.testing.assert_close(kwargs["lse"], lse0)
    assert packs[-1][decode.UIDs.SINK_UID.value].shape == (1, 4, 1, 1)


def test_prefill_match_uses_complete_build_metadata(monkeypatch):
    monkeypatch.setattr(prefill, "CUDNN_AVAILABLE", True)
    q = torch.empty(3, 4, 8, dtype=torch.bfloat16)
    kv = torch.empty(7, 2, 8, dtype=q.dtype)
    actual_kv = torch.tensor([3, 4], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        2,
        4,
        False,
        False,
        actual_seq_lens_q=torch.tensor([1, 2]),
        actual_seq_lens_kv=actual_kv,
    )

    monkeypatch.setattr(
        prefill,
        "_build_prefill_graph",
        lambda **kw: (SimpleNamespace(get_workspace_size=lambda: 0), []),
        raising=False,
    )
    prepared = prefill.prepare_cudnn_batch_prefill(
        q, kv, kv, 0.5, torch.empty(0), metadata=metadata
    )

    def matches(metadata):
        plan = prefill._CudnnPrefillPlan.prepare(metadata, q.dtype, q.device)
        return prepared.matches_plan(q, kv, kv, 0.5, plan, metadata.return_lse)

    assert matches(metadata)
    assert not matches(
        replace(metadata, actual_seq_lens_kv=torch.empty(4, dtype=torch.int32)[::2])
    )
    assert not matches(replace(metadata, return_lse=True))
    assert not matches(replace(metadata, causal=True))


@pytest.mark.parametrize(
    "lse_base,expected", [("ln", 1.0), ("log2", 1.4426950408889634)]
)
def test_prefill_metadata_rebind_and_lse_base(monkeypatch, lse_base, expected):
    q = torch.empty(3, 4, 8)
    indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        2,
        4,
        False,
        True,
        actual_seq_lens_q=torch.tensor([1, 2]),
        cu_seq_lens_q=indptr,
        batch_offsets_stats=indptr,
    )
    seen = []

    def execute(pack, **kwargs):
        seen.append(pack)
        pack[prefill.UIDs.STATS_UID.value].fill_(1)

    monkeypatch.setattr(prefill, "_create_cudnn_handle", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    plan = prefill.CudnnPrefillGraph(
        None, SimpleNamespace(execute=execute), override_cache=None, return_lse=True
    )
    for ptr in (indptr, indptr.clone()):
        output, stats = torch.empty_like(q), torch.empty(3, 4)
        plan.run(
            q,
            q,
            q,
            output,
            stats,
            torch.empty(0),
            metadata=replace(metadata, cu_seq_lens_q=ptr),
            lse_base=lse_base,
        )
        assert seen[-1][prefill.UIDs.ACTUAL_SEQ_LENS_Q_UID.value] is ptr
        assert seen[-1][prefill.UIDs.O_UID.value] is output
        torch.testing.assert_close(stats, torch.full_like(stats, expected))
    assert seen[0] is not seen[1]
    assert seen[0][prefill.UIDs.ACTUAL_SEQ_LENS_Q_UID.value] is indptr


@pytest.mark.parametrize("mode", ["error", "backend"])
def test_prefill_hn_decline_is_remembered_in_prepared_match(monkeypatch, mode):
    monkeypatch.setattr(prefill, "_CUDNN_NATIVE_HN_SUPPORTED", True)
    monkeypatch.setattr(
        prefill, "_cudnn_supports_shape_override", lambda: mode == "backend"
    )
    q = torch.empty(3, 4, 8, dtype=torch.bfloat16)
    kv = torch.empty(7, 2, 8, dtype=q.dtype)
    cuq = torch.tensor([0, 1, 3], dtype=torch.int32)
    cuk = torch.tensor([0, 3, 7], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        2,
        4,
        False,
        True,
        cu_seq_lens_q=cuq,
        cu_seq_lens_kv=cuk,
        batch_offsets_q=cuq,
        batch_offsets_o=cuq,
        batch_offsets_stats=cuq,
        batch_offsets_k=cuk,
        batch_offsets_v=cuk,
    )
    calls = []

    def build(**kwargs):
        stride = kwargs.get("stats_head_stride", 0)
        calls.append(stride)
        if stride and mode == "error":
            raise prefill.cudnn.cudnnGraphNotSupportedError("HN declined")
        return SimpleNamespace(selected_engine=None, get_workspace_size=lambda: 0), []

    monkeypatch.setattr(prefill, "_build_prefill_graph", build)
    prepared = prefill.prepare_cudnn_batch_prefill(
        q, kv, kv, 0.5, torch.empty(0), metadata=metadata, stats_head_stride=3
    )
    plan = prefill._CudnnPrefillPlan.prepare(metadata, q.dtype, q.device)
    assert calls == [3, 0]
    assert prepared.stats_head_stride == 0
    assert prepared.matches_plan(q, kv, kv, 0.5, plan, True, 3)
    assert not prepared.matches_plan(q, kv, kv, 0.5, plan, True, 0)
    assert prepared.matches_plan(q, kv, kv, 0.5, plan, True, 4) == (mode == "backend")
    assert not prepared.matches_plan(q, kv, kv, 0.5, plan, False, 3)
    # Layout/capacity affect compiled descriptors; runtime output addresses do not.
    keys = [
        prefill._sdpa_prefill_key_fn(
            q, kv, kv, 0.5, stats_head_stride=n, **metadata.graph_kwargs(None)
        )
        for n in (0, 3, 4)
    ]
    assert len(set(keys)) == 3


@pytest.mark.parametrize("supported", [False, True])
def test_prefill_hn_override_key_ignores_runtime_token_count(monkeypatch, supported):
    monkeypatch.setattr(prefill, "_CUDNN_NATIVE_HN_SUPPORTED", supported)
    q = torch.empty(3, 4, 8, dtype=torch.bfloat16)
    kv = torch.empty(7, 2, 8, dtype=q.dtype)
    ptr = torch.tensor([0, 3], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        3,
        7,
        False,
        True,
        cu_seq_lens_q=ptr,
        cu_seq_lens_kv=ptr,
        batch_offsets_q=ptr,
        batch_offsets_o=ptr,
        batch_offsets_stats=ptr,
        batch_offsets_k=ptr,
        batch_offsets_v=ptr,
    )
    for override in (None, (4096, 128, 128)):
        keys = [
            prefill._sdpa_prefill_key_fn(
                q, kv, kv, 0.5, stats_head_stride=n, **metadata.graph_kwargs(override)
            )
            for n in (0, 3, 4)
        ]
        assert keys[0] != keys[1]  # NH/HN remain different specializations.
        assert (keys[1] == keys[2]) == (supported and override is not None)


def test_prefill_hn_same_plan_rebinds_current_output_stride(monkeypatch):
    monkeypatch.setattr(prefill, "_CUDNN_NATIVE_HN_SUPPORTED", True)
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: True)
    monkeypatch.setattr(prefill, "_create_cudnn_handle", lambda stream: 23)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    q = torch.empty(3, 4, 8, dtype=torch.bfloat16)
    kv = torch.empty(7, 2, 8, dtype=q.dtype)
    cuq = torch.tensor([0, 3], dtype=torch.int32)
    cuk = torch.tensor([0, 7], dtype=torch.int32)
    metadata = prefill._PrefillMetadata(
        3,
        7,
        False,
        True,
        cu_seq_lens_q=cuq,
        cu_seq_lens_kv=cuk,
        batch_offsets_q=cuq,
        batch_offsets_o=cuq,
        batch_offsets_stats=cuq,
        batch_offsets_k=cuk,
        batch_offsets_v=cuk,
    )
    calls = []

    class Graph:
        def execute(self, buffers, *, tensor_uids=None, **kwargs):
            calls.append(kwargs)

    plan = prefill._CudnnPrefillPlan.prepare(metadata, q.dtype, q.device)
    key = prefill._sdpa_prefill_key_fn(
        q, kv, kv, 0.5, stats_head_stride=3, **metadata.graph_kwargs(plan.override)
    )
    prepared = prefill.CudnnPrefillGraph(
        key,
        Graph(),
        override_cache=plan.override,
        return_lse=True,
        stats_head_stride=3,
        requested_stats_head_stride=3,
    )
    for tokens in (3, 2, 3):
        assert prepared.matches_plan(q[:tokens], kv, kv, 0.5, plan, True, tokens)
        prepared.run_planned(
            q[:tokens],
            kv,
            kv,
            torch.empty_like(q[:tokens]),
            torch.empty(4, tokens),
            torch.empty(0),
            plan=plan,
            lse_base="ln",
        )
    for call, tokens in zip(calls, (3, 2, 3), strict=True):
        at = call["override_uids"].index(prefill.UIDs.STATS_UID.value)
        assert call["override_strides"][at] == [4 * tokens, tokens, 1, 1]
    assert calls[0]["override_strides"] is not calls[1]["override_strides"]


@pytest.mark.parametrize("supported", [False, True])
def test_prefill_capacity_is_a_plan_descriptor(monkeypatch, supported):
    """A larger lifetime capacity must not reuse a smaller compiled graph."""
    monkeypatch.setattr(prefill, "_cudnn_supports_bounded_ragged", lambda: supported)
    monkeypatch.setattr(prefill, "_cudnn_supports_direct_seqlens", lambda *a, **k: True)
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    monkeypatch.setattr(
        prefill,
        "_build_prefill_graph",
        lambda **kw: (SimpleNamespace(get_workspace_size=lambda: 0), []),
        raising=False,
    )
    indptr = torch.tensor([0, 128, 129, 130], dtype=torch.int32)
    q = torch.empty(130, 16, 128, dtype=torch.bfloat16)
    kv = torch.empty(3 * 2048, 4, 128, dtype=q.dtype)

    def metadata(capacity):
        return prefill._PrefillMetadata(
            128,
            2048,
            False,
            False,
            batch_offsets_q=indptr,
            batch_offsets_k=indptr,
            max_total_num_rows=capacity,
        ).resolve_from_plan(q.dtype, 16, 4, 128, 128)

    first = metadata(130)
    plan = prefill._CudnnPrefillPlan.prepare(first, q.dtype, q.device)
    prepared = prefill.prepare_cudnn_batch_prefill(
        q, kv, kv, 0.5, torch.empty(0), metadata=first
    )
    same = prefill._CudnnPrefillPlan.prepare(metadata(130), q.dtype, q.device, plan)
    larger = prefill._CudnnPrefillPlan.prepare(metadata(384), q.dtype, q.device, plan)
    assert same is plan
    assert prepared.matches_plan(q, kv, kv, 0.5, same, False)
    assert prepared.matches_plan(q, kv, kv, 0.5, larger, False) is not supported
    assert plan.execution_shape == (3, 128, 2048)
    if supported:
        assert first.max_total_num_rows == 130
        assert larger.metadata.max_total_num_rows >= 384
    else:
        assert first.max_total_num_rows is None
