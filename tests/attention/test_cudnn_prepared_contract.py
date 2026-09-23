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


def test_decode_binding_is_per_call_and_lse_is_base2(monkeypatch, decode_inputs):
    q, k, v, kwargs = decode_inputs
    packs = []

    def execute(pack, **unused):
        packs.append(pack)
        pack[decode.UIDs.O_UID.value].copy_(pack[decode.UIDs.Q_UID.value])
        pack[decode.UIDs.STATS_UID.value].fill_(torch.log(torch.tensor(2.0)))

    graph = SimpleNamespace(execute=execute)
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
        prefill, "_build_prefill_graph", lambda **kw: (object(), []), raising=False
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
