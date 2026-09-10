# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unified PrimTS direct APIs: contracts, accuracy, and live graph metadata."""

from dataclasses import replace
import importlib
import inspect

import pytest
import torch


pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import decode, mla_decode


_APIS = (
    (decode, "batch_decode_with_paged_kv_cache", "_batch_decode_with_workspace"),
    (
        mla_decode,
        "batch_mla_decode_with_paged_kv_cache",
        "_batch_mla_decode_with_workspace",
    ),
)
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="PrimTS requires Blackwell",
)


@pytest.mark.parametrize("module,name,helper", _APIS)
def test_explicit_dispatch_preserves_live_tensors(monkeypatch, module, name, helper):
    """Trusted routing does not plan, infer lengths, or copy request metadata."""
    seen = {}
    sentinel = object()

    def launch(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(module, helper, launch)
    api = inspect.unwrap(getattr(module, name))
    query, cache, tables, lengths, workspace, output, offsets = (
        object() for _ in range(7)
    )
    assert (
        api(
            query,
            cache,
            tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=1536,
            qo_indptr=offsets,
            max_seq_len_q=4,
            out=output,
            validate=False,
        )
        is sentinel
    )
    assert seen["args"][:3] == (query, cache, workspace)
    assert seen["args"][-3:] == (tables, lengths, 1536)
    assert seen["kwargs"]["qo_indptr"] is offsets
    assert seen["kwargs"]["out"] is output
    assert seen["kwargs"]["validate"] is False
    assert seen["kwargs"]["validate_values"] is False


@pytest.mark.parametrize("module,name,_helper", _APIS)
def test_explicit_capture_contract_errors(monkeypatch, module, name, _helper):
    api = inspect.unwrap(getattr(module, name))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    args = (None,) * 4
    with pytest.raises(TypeError, match="validate must be a bool"):
        api(*args, validate=0)
    with pytest.raises(ValueError, match="workspace_buffer"):
        api(*args, validate=False)
    with pytest.raises(ValueError, match="max_kv_len"):
        api(*args, workspace_buffer=object(), out=object(), validate=False)
    with pytest.raises(ValueError, match="max_seq_len_q"):
        api(
            *args,
            workspace_buffer=object(),
            max_kv_len=32,
            qo_indptr=object(),
            out=object(),
            validate=False,
        )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for kwargs in (
        {},
        dict(workspace_buffer=object(), max_kv_len=32, out=object()),
        dict(workspace_buffer=object(), max_kv_len=32, validate=False),
    ):
        with pytest.raises(RuntimeError, match="CUDA graph capture requires"):
            api(*args, **kwargs)


@pytest.mark.parametrize("module,name,_helper", _APIS)
def test_explicit_cold_compile_rejects_capture(monkeypatch, module, name, _helper):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    compiler = (
        module._get_compiled_decode
        if module is decode
        else module._get_compiled_mla_decode
    )
    # Exercise the cache-miss body without disturbing any live compilation cache.
    with pytest.raises(RuntimeError, match="warm up"):
        compiler.__wrapped__(None)


def _forbid_host_work(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("trusted direct launch allocated, read back, or planned")

    for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("item", "tolist", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    for module, _, _ in _APIS:
        for name in vars(module):
            if name.startswith("_validate_"):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(decode.BatchDecodePagedTSWrapper, "plan", forbidden)
    monkeypatch.setattr(mla_decode.BatchMLADecodePagedTSWrapper, "plan", forbidden)


def _graph_parity(monkeypatch, run, legacy, output, lengths, tables, offsets):
    """Replay after changing all live metadata, with the same storage/extent."""
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        _forbid_host_work(patch)
        with torch.cuda.graph(graph):
            assert run() is output
    graph.replay()
    torch.testing.assert_close(output, legacy(), rtol=0, atol=0)
    lengths.sub_(1)
    tables[:, 0].copy_(tables[:, 1])
    if offsets is not None and offsets.numel() == 3:
        # Preserve the total extent while redistributing two nonempty requests.
        offsets[1].fill_(2)
    expected = legacy().clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "batch,dtype,dim,packed,cache_form",
    [
        (1, torch.bfloat16, 128, False, "tuple"),
        (64, torch.bfloat16, 128, False, "combined"),
        (2, torch.float16, 64, False, "tuple"),
        (2, torch.float8_e4m3fn, 256, True, "combined"),
    ],
)
def test_unified_fmha_parity_and_capture(
    monkeypatch, batch, dtype, dim, packed, cache_form
):
    helpers = importlib.import_module("tests.attention.test_attention_ts_decode")
    sq = 4 if packed or dtype == torch.float16 else 1
    max_k = 1536 if batch in (1, 64) else 257
    case = helpers._make_decode_case(
        kv_lens=(max_k,) * batch,
        num_qo_heads=28 if dim == 128 else 8,
        num_kv_heads=4 if dim == 128 else 1,
        head_dim=dim,
        seq_len_q=sq,
        page_size=32,
        qkv_dtype=dtype,
        output_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        cache_form=cache_form,
        mask_type="causal",
        window_left=127 if sq > 1 else -1,
        device="cuda",
        seed=5082,
    )
    offsets = None
    if packed:
        case, offsets = helpers._pack_decode_case(case, (1, 4))
    lengths = torch.full((batch,), max_k, dtype=torch.int32, device="cuda")
    tables = torch.full(
        (batch, case.block_tables.shape[1] + 3), -1, dtype=torch.int32, device="cuda"
    )[:, :-3]
    tables.copy_(case.block_tables)
    size = decode.get_prims_ts_batch_decode_workspace_size(
        batch,
        case.q.shape[-2],
        case.k_cache.shape[1],
        dim,
        32,
        max_k,
        seq_len_q=1 if packed else sq,
        qo_indptr=offsets,
        max_seq_len_q=sq,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        device=case.q.device,
    )
    workspace = torch.zeros(size, dtype=torch.uint8, device="cuda")
    old_workspace = torch.zeros_like(workspace)
    output = torch.empty_like(case.q, dtype=case.output_dtype)
    old_output = torch.empty_like(output)
    options = dict(
        seq_len_q=1 if packed else sq,
        qo_indptr=offsets,
        max_seq_len_q=sq,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
    )

    def run(validate=False):
        return decode.batch_decode_with_paged_kv_cache(
            case.q,
            case.paged_kv_cache,
            tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=max_k,
            out=output,
            validate=validate,
            **options,
        )

    def legacy():
        return decode.prims_ts_batch_decode_with_kv_cache(
            case.q,
            case.paged_kv_cache,
            old_workspace,
            tables,
            lengths,
            max_k,
            out=old_output,
            **options,
        )

    result = run(True)
    if dtype != torch.float8_e4m3fn:
        helpers._assert_case_correct(result, case)
    # This extra packed D256 FP8/window case checks API parity below. On the
    # unchanged 9a4324eb baseline, 13/10240 elements already exceed the P448
    # reference tolerance (max absolute error 0.00041962). Do not relax that
    # tolerance or claim new arithmetic coverage; existing FP8 accuracy tests
    # remain unchanged. Baseline differential replay was also checked separately.
    torch.testing.assert_close(run(), legacy(), rtol=0, atol=0)
    with pytest.raises(
        ValueError, match="overlap" if dtype == case.output_dtype else "dtype"
    ):
        decode.batch_decode_with_paged_kv_cache(
            case.q,
            case.paged_kv_cache,
            tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=max_k,
            out=case.q,
            **options,
        )
    lengths.fill_(max_k + 1)
    with pytest.raises(ValueError, match="within"):
        run(True)
    lengths.fill_(max_k)
    _graph_parity(monkeypatch, run, legacy, output, lengths, tables, offsets)


@_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "batch,dtype,packed,compact",
    [
        (1, torch.bfloat16, False, True),
        (64, torch.bfloat16, False, False),
        (2, torch.float8_e4m3fn, True, True),
        (3, torch.bfloat16, True, False),
    ],
)
def test_unified_mla_accuracy_and_capture(monkeypatch, batch, dtype, packed, compact):
    helpers = importlib.import_module("tests.attention.test_attention_ts_mla_decode")
    sq = 4 if packed else 1
    max_k = 1536 if not packed else 257
    case = helpers._make_mla_case(
        batch_size=batch,
        num_qo_heads=32,
        max_seq_len=max_k,
        kv_seq_lens=(max_k,) * batch,
        seq_len_q=sq,
        qkv_dtype=dtype,
        mask_type="causal",
        page_size=32,
        device="cuda",
        seed=5083,
    )
    offsets = None
    if packed:
        case, offsets = helpers._pack_mla_case(
            case, (1, 4) if batch == 2 else (1, 0, 4)
        )
    if compact:
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    wrapper = helpers._plan_case(case, qo_indptr=offsets, max_seq_len_q=sq)
    size = mla_decode.get_prims_ts_batch_mla_decode_workspace_size(
        batch,
        32,
        512,
        64,
        32,
        max_k,
        max_seq_len_q=sq,
        q_dtype=dtype,
        kv_dtype=dtype,
        device=case.query.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    old_workspace = torch.empty_like(workspace)
    shape = (*case.query.shape[:-1], 512)
    output = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    old_output = torch.empty_like(output)
    options = dict(
        qo_indptr=offsets,
        max_seq_len_q=sq,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
    )

    def run(validate=False):
        return mla_decode.batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens,
            workspace_buffer=workspace,
            max_kv_len=max_k,
            out=output,
            validate=validate,
            **options,
        )

    def legacy():
        return mla_decode.prims_ts_batch_mla_decode_with_kv_cache(
            case.query,
            case.kv_cache,
            old_workspace,
            512,
            64,
            case.block_tables,
            case.seq_lens,
            max_k,
            out=old_output,
            **options,
        )

    helpers._assert_case_correct(
        run(True), case, helpers._policy_dict(wrapper), qo_indptr=offsets
    )
    torch.testing.assert_close(run(), legacy(), rtol=0, atol=0)
    case.block_tables[0, 0].fill_(-1)
    with pytest.raises(ValueError, match="invalid page ID"):
        run(True)
    case.block_tables[0, 0].copy_(case.block_tables[0, 1])
    _graph_parity(
        monkeypatch, run, legacy, output, case.seq_lens, case.block_tables, offsets
    )


@_GPU
@pytest.mark.arch_blackwell
def test_unified_mla_all_empty_packed_query():
    query = torch.empty((0, 32, 576), dtype=torch.bfloat16, device="cuda")
    cache = torch.empty((1, 32, 576), dtype=query.dtype, device="cuda")
    tables = torch.zeros((2, 1), dtype=torch.int32, device="cuda")
    lengths = torch.ones(2, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(3, dtype=torch.int32, device="cuda")
    size = mla_decode.get_prims_ts_batch_mla_decode_workspace_size(
        2,
        32,
        512,
        64,
        32,
        1,
        max_seq_len_q=4,
        device=query.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    out = torch.empty((0, 32, 512), dtype=query.dtype, device="cuda")
    for validate in (True, False):
        assert (
            mla_decode.batch_mla_decode_with_paged_kv_cache(
                query,
                cache,
                tables,
                lengths,
                qo_indptr=offsets,
                max_seq_len_q=4,
                max_kv_len=1,
                workspace_buffer=workspace,
                out=out,
                validate=validate,
            )
            is out
        )
