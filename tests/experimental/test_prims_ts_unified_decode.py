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
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import decode, mla_decode


_APIS = (
    (decode, "batch_decode_with_paged_kv_cache", "BatchDecodePagedTSWrapper"),
    (
        mla_decode,
        "batch_mla_decode_with_paged_kv_cache",
        "BatchMLADecodePagedTSWrapper",
    ),
)
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="PrimTS requires Blackwell",
)


@pytest.mark.parametrize("module,name,wrapper_name", _APIS)
@pytest.mark.parametrize("mode", ["owned", "validated_external", "trusted_external"])
def test_explicit_dispatch_preserves_live_tensors(
    monkeypatch, module, name, wrapper_name, mode
):
    """Owned and external scratch both reach the same wrapper plan/run path."""
    seen = {}
    sentinel = object()

    class Wrapper:
        def __init__(self, **kwargs):
            pass

        def plan(self, *args, **kwargs):
            seen["plan_args"], seen["plan_kwargs"] = args, kwargs

        def run(self, *args, **kwargs):
            seen["run_args"], seen["run_kwargs"] = args, kwargs
            return sentinel

    monkeypatch.setattr(module, wrapper_name, Wrapper)
    api = inspect.unwrap(getattr(module, name))
    query = torch.empty((4, 8, 64 if module is decode else 576))
    cache = (
        (torch.empty((2, 1, 32, 64)),) * 2
        if module is decode
        else torch.empty((2, 32, 576))
    )
    tables = torch.zeros((2, 48), dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    workspace = torch.empty(4096, dtype=torch.uint8)
    output = torch.empty((4, 8, 64 if module is decode else 512))
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    owned = mode == "owned"
    validate = mode != "trusted_external"
    size_calls = []

    def workspace_size(*args, **kwargs):
        size_calls.append((args, kwargs))
        return 4096

    monkeypatch.setattr(module, "_validate_qo_indptr", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_validate_out", lambda *a, **kw: None)
    if module is decode:
        monkeypatch.setattr(module, "_validate_q", lambda *a, **kw: None)
        monkeypatch.setattr(
            module, "_validate_block_table_metadata", lambda *a: (query.device, 2, 48)
        )
        monkeypatch.setattr(
            module,
            "_normalize_paged_kv_cache_views",
            lambda *a, **kw: (*cache, 2, 1, 32, 64),
        )
        monkeypatch.setattr(
            module, "get_prims_ts_batch_decode_workspace_size", workspace_size
        )
    else:
        monkeypatch.setattr(module, "_validate_query", lambda *a, **kw: None)
        monkeypatch.setattr(
            module, "_validate_mla_metadata", lambda *a: (query.device, 2, 48)
        )
        monkeypatch.setattr(
            module, "_normalize_mla_kv_cache", lambda *a, **kw: (cache, 2, 32)
        )
        monkeypatch.setattr(
            module, "get_prims_ts_batch_mla_decode_workspace_size", workspace_size
        )
    assert (
        api(
            query,
            cache,
            tables,
            lengths,
            workspace_buffer=None if owned else workspace,
            max_kv_len=1536,
            qo_indptr=offsets,
            max_seq_len_q=4,
            out=output,
            validate=validate,
            **({"page_size": 4} if module is decode else {}),
        )
        is sentinel
    )
    planned_workspace = seen["plan_kwargs"]["workspace_buffer"]
    if owned:
        assert len(size_calls) == 1
        assert planned_workspace.numel() == 4096
        assert planned_workspace.dtype == torch.int8
        assert planned_workspace.device == query.device
        assert planned_workspace is not workspace
    else:
        assert not size_calls
        assert planned_workspace is workspace
    assert seen["plan_kwargs"]["validate"] is validate
    assert seen["plan_args"][-1] == 1536
    assert seen["run_args"][0] is query
    assert seen["run_args"][1] is cache
    assert any(arg is tables for arg in seen["run_args"])
    if module is decode and owned:
        assert seen["plan_kwargs"]["seq_lens"] == (1, 1)
        assert seen["run_args"][2] is None
    else:
        assert any(arg is lengths for arg in seen["run_args"])
    assert seen["run_kwargs"]["qo_indptr"] is offsets
    assert seen["run_kwargs"]["out"] is output
    assert seen["run_kwargs"]["validate"] is validate
    if module is decode:
        assert seen["plan_args"][-2] == 4
        assert seen["plan_kwargs"]["initialize_workspace"] is owned


@pytest.mark.parametrize("module,_name,wrapper_name", _APIS)
def test_trusted_plan_only_binds_existing_scratch(
    monkeypatch, module, _name, wrapper_name
):
    """Exercise real planning with mocked compilation, not a mocked wrapper."""
    workspace = torch.full((4096,), 7, dtype=torch.uint8)
    before = workspace.clone()
    if module is decode:
        spec = SimpleNamespace(
            config=SimpleNamespace(
                use_separate_reduction_kernel=False, use_split_kv=False
            ),
            policy=(),
            scratch_shapes=((1, 1, 1, 1, 1), (1,), (1,)),
        )
        family = "decode"
        geometry = (1, 8, 1, 64, 32, 32)
        compiled = (lambda *a: None, None)
        options = {"initialize_workspace": False}
    else:
        spec = SimpleNamespace(kernel_workspace_bytes=0, policy=(("split_kv", 1),))
        family = "mla_decode"
        geometry = (1, 8, 512, 64, 32, 32)
        compiled = lambda *a: None
        options = {}
    monkeypatch.setattr(module, f"_resolve_{family}_launch_spec", lambda *a: spec)
    monkeypatch.setattr(module, f"_make_{family}_compile_spec", lambda *a, **kw: None)
    monkeypatch.setattr(module, f"_get_compiled_{family}", lambda *a: compiled)
    wrapper = getattr(module, wrapper_name)()
    with monkeypatch.context() as patch:
        _forbid_host_work(patch)
        wrapper.plan(
            "cuda:0",
            *geometry,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            workspace_buffer=workspace,
            validate=False,
            **options,
        )
    assert wrapper._plan_state.workspace_buffer is workspace
    torch.testing.assert_close(workspace, before)


def test_removed_direct_apis_are_not_exported():
    from flashinfer.attention import prims_ts
    import flashinfer.decode as public_decode
    import flashinfer.mla as public_mla

    for module, legacy_name in (
        (decode, "prims_ts_batch_decode_with_kv_cache"),
        (mla_decode, "prims_ts_batch_mla_decode_with_kv_cache"),
        (prims_ts, "prims_ts_batch_decode_with_kv_cache"),
        (prims_ts, "prims_ts_batch_mla_decode_with_kv_cache"),
        (public_decode, "prims_ts_batch_decode_with_kv_cache"),
        (public_mla, "prims_ts_batch_mla_decode_with_kv_cache"),
    ):
        assert not hasattr(module, legacy_name)
        assert legacy_name not in dir(module)


@pytest.mark.parametrize("page_size,storage_page_size", [(4, 16), (4, 32), (16, 16)])
def test_decode_metadata_validates_encoded_locator_capacity(
    page_size, storage_page_size
):
    runtime = SimpleNamespace(
        num_physical_pages=2,
        k_cache=torch.empty((2, 1, storage_page_size, 64)),
        q=torch.empty((1, 8, 64)),
    )
    capacity = 2 * storage_page_size // page_size
    tables = torch.tensor([[capacity - 1]], dtype=torch.int32)
    options = dict(
        seq_lens=torch.tensor([page_size], dtype=torch.int32),
        block_tables=tables,
        qo_indptr=None,
        planned_seq_lens_host=None,
        max_kv_len=page_size,
        page_size=page_size,
        use_packed_q=False,
        seq_len_q=1,
        batch_size=1,
        mask_type="causal",
    )
    decode._validate_decode_run_metadata_values(runtime, **options)
    tables.fill_(capacity)
    with pytest.raises(ValueError, match="invalid page ID"):
        decode._validate_decode_run_metadata_values(runtime, **options)


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
        raise AssertionError(
            "trusted wrapper launch allocated, read back, reset, or validated"
        )

    for name in (
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
        "tensor",
    ):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("item", "tolist", "cpu", "zero_"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    for module, _, _ in _APIS:
        for name in vars(module):
            if name.startswith("_validate_") and name != "_validate_layout":
                monkeypatch.setattr(module, name, forbidden)


def _graph_parity(monkeypatch, run, reference, output, lengths, tables, offsets):
    """Replay after changing all live metadata, with the same storage/extent."""
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        _forbid_host_work(patch)
        with torch.cuda.graph(graph):
            assert run() is output
    graph.replay()
    torch.testing.assert_close(output, reference(), rtol=0, atol=0)
    lengths.sub_(1)
    tables[:, 0].copy_(tables[:, 1])
    if offsets is not None and offsets.numel() == 3:
        # Preserve the total extent while redistributing two nonempty requests.
        offsets[1].fill_(2)
    expected = reference().clone()
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

    reference_wrapper = decode.BatchDecodePagedTSWrapper()
    reference_wrapper.plan(
        case.q.device,
        batch,
        case.q.shape[-2],
        case.k_cache.shape[1],
        dim,
        32,
        max_k,
        max_seq_len_q=sq,
        packed_query=packed,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        workspace_buffer=old_workspace,
    )

    def reference():
        return reference_wrapper.run(
            case.q,
            case.paged_kv_cache,
            lengths,
            tables,
            qo_indptr=offsets,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
            out=old_output,
            validate=False,
        )

    result = run(True)
    if dtype != torch.float8_e4m3fn:
        helpers._assert_case_correct(result, case)
    # This extra packed D256 FP8/window case checks API parity below. On the
    # unchanged 9a4324eb baseline, 13/10240 elements already exceed the P448
    # reference tolerance (max absolute error 0.00041962). Do not relax that
    # tolerance or claim new arithmetic coverage; existing FP8 accuracy tests
    # remain unchanged. Baseline differential replay was also checked separately.
    torch.testing.assert_close(run(), reference(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="dtype"):
        decode.batch_decode_with_paged_kv_cache(
            case.q,
            case.paged_kv_cache,
            tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=max_k,
            out=torch.empty_like(output, dtype=torch.float32),
            **options,
        )
    lengths.fill_(max_k + 1)
    with pytest.raises(ValueError, match="within"):
        run(True)
    lengths.fill_(max_k)
    _graph_parity(monkeypatch, run, reference, output, lengths, tables, offsets)


@_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("physical_layout", ["compact", "kv-packed-nhd"])
def test_unified_fmha_encoded_subpages_and_capture(monkeypatch, physical_layout):
    helpers = importlib.import_module("tests.attention.test_attention_ts_decode")
    case = helpers._make_decode_case(
        kv_lens=(37, 37),
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=1,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=5084,
    )
    cache = helpers._pack_page4_cache_into_storage_pages(
        case.k_cache,
        case.v_cache,
        storage_page_size=16,
        physical_layout=physical_layout,
    )
    tables = case.block_tables
    lengths = torch.full((2,), 37, dtype=torch.int32, device="cuda")
    size = decode.get_prims_ts_batch_decode_workspace_size(
        2,
        12,
        1,
        256,
        4,
        37,
        q_dtype=torch.bfloat16,
        mask_type="causal",
        storage_page_size=16,
        device="cuda",
    )
    workspace = torch.zeros(size, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(case.q)
    expected = torch.empty_like(output)
    plan = decode.prepare_prims_ts_batch_decode_with_kv_cache(
        case.q,
        cache,
        torch.zeros_like(workspace),
        tables,
        lengths,
        37,
        out=expected,
        mask_type="causal",
        page_size=4,
    )

    def run(validate=False):
        return decode.batch_decode_with_paged_kv_cache(
            case.q,
            cache,
            tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=37,
            out=output,
            mask_type="causal",
            page_size=4,
            validate=validate,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )

    def prepared():
        return plan.run(
            case.q, out=expected, bmm1_scale=case.bmm1_scale, bmm2_scale=case.bmm2_scale
        )

    helpers._assert_case_correct(run(True), case)
    _graph_parity(monkeypatch, run, prepared, output, lengths, tables, None)


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

    def reference():
        return wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens,
            qo_indptr=offsets,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
            out=old_output,
            validate=False,
        )

    helpers._assert_case_correct(
        run(True), case, helpers._policy_dict(wrapper), qo_indptr=offsets
    )
    torch.testing.assert_close(run(), reference(), rtol=0, atol=0)
    case.block_tables[0, 0].fill_(-1)
    with pytest.raises(ValueError, match="invalid page ID"):
        run(True)
    case.block_tables[0, 0].copy_(case.block_tables[0, 1])
    _graph_parity(
        monkeypatch, run, reference, output, case.seq_lens, case.block_tables, offsets
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
