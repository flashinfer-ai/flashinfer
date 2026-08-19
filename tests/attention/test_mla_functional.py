"""Canonical numerical coverage for the public one-shot MLA API."""

import inspect
import warnings

import pytest
import torch

import flashinfer
import flashinfer.mla._batch_mla._functional as mla_functional
from tests.test_helpers.mla import (
    MLATestCase,
    assert_mla_close,
    functional_kwargs,
    make_mla_inputs,
    reference_result,
    require_architecture,
    unpack_mla_result,
)
from flashinfer.mla._batch_mla._contracts import _FunctionalMLARequest


_FUNCTIONAL_BACKENDS = {
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
}

_FUNCTIONAL_CASES = (
    MLATestCase("sm80-fa2-decode", (8, 0), "fa2"),
    MLATestCase(
        "sm90-fa2-fp16-page64-base2",
        (9, 0),
        "fa2",
        page_size=64,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        lse_mode="base2",
    ),
    MLATestCase(
        "sm90-fa2-prefill-page128-base2",
        (9, 0),
        "fa2",
        q_len=2,
        page_size=128,
        lse_mode="base2",
    ),
    MLATestCase("sm90-fa3-decode", (9, 0), "fa3", lse_mode="base2"),
    MLATestCase(
        "sm90-fa3-fp16-page64-base2",
        (9, 0),
        "fa3",
        page_size=64,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        lse_mode="base2",
    ),
    MLATestCase(
        "sm90-fa3-prefill-page128",
        (9, 0),
        "fa3",
        q_len=2,
        page_size=128,
    ),
    MLATestCase(
        "sm100-cutlass-decode",
        (10, 0),
        "cutlass",
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-cutlass-page64",
        (10, 0),
        "cutlass",
        page_size=64,
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-cutlass-page128",
        (10, 0),
        "cutlass",
        page_size=128,
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-trtllm-decode",
        (10, 0),
        "trtllm-gen",
        qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-trtllm-base2-scalar-pdl",
        (10, 0),
        "trtllm-gen",
        page_size=64,
        lse_mode="base2",
        scale_mode="bmm-scalar",
        enable_pdl=True,
        qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-trtllm-tensor-skip-softmax",
        (10, 0),
        "trtllm-gen",
        page_size=64,
        scale_mode="bmm-tensor",
        skip_softmax=True,
        enable_pdl=False,
        qk_nope_head_dim=128,
    ),
    MLATestCase("sm100-cute-monolithic", (10, 0), "cute-dsl-monolithic"),
    MLATestCase(
        "sm100-cute-monolithic-fp16-basee",
        (10, 0),
        "cute-dsl-monolithic",
        page_size=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        lse_mode="basee",
        scale_mode="bmm-scalar",
    ),
    MLATestCase(
        "sm100-cute-modular",
        (10, 0),
        "cute-dsl-modular",
    ),
    MLATestCase(
        "sm100-cute-modular-fp16-scalar",
        (10, 0),
        "cute-dsl-modular",
        page_size=64,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        scale_mode="bmm-scalar",
    ),
    MLATestCase(
        "sm120-xqa-decode",
        (12, 0),
        "xqa",
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm120-xqa-page64-scalar-pdl",
        (12, 0),
        "xqa",
        page_size=64,
        scale_mode="bmm-scalar",
        enable_pdl=True,
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm120-xqa-page128-pdl-off",
        (12, 0),
        "xqa",
        page_size=128,
        enable_pdl=False,
        softmax_scale_qk_nope_head_dim=128,
    ),
)

_AUTO_CASES = (
    MLATestCase("sm100-auto", (10, 0), "auto", qk_nope_head_dim=128),
    MLATestCase("sm103-auto", (10, 3), "auto", qk_nope_head_dim=128),
    MLATestCase("sm120-auto", (12, 0), "auto", softmax_scale_qk_nope_head_dim=128),
    MLATestCase("sm121-auto", (12, 1), "auto", softmax_scale_qk_nope_head_dim=128),
)

_PUBLIC_MLA_API = {
    "MLAHeadDimensions",
    "deepseek_mla_dimensions",
    "nope_mla_dimensions",
    "smaller_mla_dimensions",
    "supported_mla_head_dimensions",
    "MLALayerDimensions",
    "supported_mla_layer_dimensions",
    "BatchMLAPagedAttentionWrapper",
    "MLAPlanMetadata",
    "MLAAutoSelectionTrace",
    "batch_mla_paged_attention",
    "trtllm_batch_decode_with_kv_cache_mla",
    "xqa_batch_decode_with_kv_cache_mla",
    "trtllm_batch_decode_sparse_mla_dsv4",
}


def _compare_result(case, result, expected_output, expected_lse):
    actual_output, actual_lse = unpack_mla_result(result, case.lse_mode != "none")
    assert_mla_close(actual_output.reshape_as(expected_output), expected_output)
    if expected_lse is not None:
        assert actual_lse is not None
        assert_mla_close(actual_lse.reshape_as(expected_lse), expected_lse)


@pytest.mark.parametrize("case", _FUNCTIONAL_CASES, ids=lambda case: case.case_id)
def test_explicit_functional_api_matches_reference(case):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, expected_lse = reference_result(case, inputs)
    kwargs = functional_kwargs(case, inputs)
    if case.backend == "trtllm-gen":
        out = torch.empty(
            (2, case.q_len, 128, 512), dtype=case.output_dtype, device="cuda"
        )
        kwargs["out"] = out

    result = flashinfer.mla.batch_mla_paged_attention(**kwargs)

    if case.backend == "trtllm-gen":
        actual = result[0] if isinstance(result, tuple) else result
        assert actual.data_ptr() == out.data_ptr()
    _compare_result(case, result, expected_output, expected_lse)


@pytest.mark.parametrize("case", _AUTO_CASES, ids=lambda case: case.case_id)
def test_functional_auto_matches_reference(case):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, expected_lse = reference_result(case, inputs)

    result = flashinfer.mla.batch_mla_paged_attention(**functional_kwargs(case, inputs))

    _compare_result(case, result, expected_output, expected_lse)


def test_functional_exports_and_tensor_first_ownership_are_stable():
    assert hasattr(flashinfer.mla, "batch_mla_paged_attention")
    assert not hasattr(flashinfer.decode, "batch_mla_paged_attention")
    assert not hasattr(flashinfer, "batch_mla_paged_attention")
    for name in ("MLAQuery", "MLAKVCache"):
        assert name not in flashinfer.mla.__all__
        assert not hasattr(flashinfer.mla, name)
        with pytest.raises(ImportError):
            exec(f"from flashinfer.mla import {name}", {})

    signature = inspect.signature(flashinfer.mla.BatchMLAPagedAttentionWrapper.run)
    for name in (
        "query",
        "q_nope",
        "q_pe",
        "kv_cache",
        "ckv_cache",
        "kpe_cache",
    ):
        assert name in signature.parameters
    assert "query_object" not in signature.parameters
    assert "kv" not in signature.parameters

    namespace = {}
    exec("from flashinfer.mla import *", namespace)
    assert {name for name in namespace if not name.startswith("__")} == _PUBLIC_MLA_API


class _PoisonReference:
    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"ignored reference was inspected through {name}")


def _raw_functional_request(**overrides):
    values = dict(
        query=torch.empty(2, 1, 4, 3),
        q_nope=torch.empty(2, 1, 4, 2),
        q_pe=torch.empty(2, 1, 4, 1),
        kv_cache=torch.empty(4, 1, 3),
        ckv_cache=torch.empty(4, 1, 2),
        kpe_cache=torch.empty(4, 1, 1),
        query_availability="redundant",
        kv_availability="redundant",
        workspace_buffer=torch.empty(16, dtype=torch.int8),
        qk_nope_head_dim=2,
        kv_lora_rank=2,
        qk_rope_head_dim=1,
        block_tables=torch.zeros(2, 1, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        max_seq_len=1,
        sparse_mla_top_k=0,
        out=None,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=None,
        is_var_seq=True,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        cute_dsl_impl="auto",
        kv_scale_format="auto",
        cum_seq_lens_q=None,
        max_q_len=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
        cp_world=1,
        cp_rank=0,
        causal_seqlens_kv_global=None,
    )
    values.update(overrides)
    return _FunctionalMLARequest(**values)


def test_packed_functional_selection_materializes_independent_split_inputs(
    monkeypatch,
):
    q_nope = torch.ones(1, 1, 1, 2)
    q_pe = torch.full((1, 1, 1, 1), 2.0)
    ckv_cache = torch.full((1, 1, 2), 3.0)
    kpe_cache = torch.full((1, 1, 1), 4.0)
    request = _raw_functional_request(
        query=None,
        q_nope=q_nope,
        q_pe=q_pe,
        kv_cache=None,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        query_availability="split",
        kv_availability="split",
    )
    concatenations = []
    real_cat = torch.cat

    def recording_cat(tensors, *, dim):
        concatenations.append((tuple(tensors), dim))
        return real_cat(tuple(tensors), dim=dim)

    class PackedRunner:
        native_query_representation = "packed"
        native_kv_representation = "packed"

    monkeypatch.setattr(torch, "cat", recording_cat)

    with pytest.warns(UserWarning, match="materialization"):
        selected = mla_functional._select_functional_request(request, PackedRunner)

    assert torch.equal(selected.query, torch.tensor([[[[1.0, 1.0, 2.0]]]]))
    assert torch.equal(selected.kv_cache, torch.tensor([[[3.0, 3.0, 4.0]]]))
    assert selected.q_nope is selected.q_pe is None
    assert selected.ckv_cache is selected.kpe_cache is None
    assert [dim for _, dim in concatenations] == [-1, -1]


def test_functional_split_materialization_warns_once_per_process(monkeypatch):
    request = _raw_functional_request(
        query=None,
        q_nope=torch.ones(1, 1, 1, 2),
        q_pe=torch.ones(1, 1, 1, 1),
        kv_cache=None,
        ckv_cache=torch.ones(1, 1, 2),
        kpe_cache=torch.ones(1, 1, 1),
        query_availability="split",
        kv_availability="split",
    )

    class PackedRunner:
        native_query_representation = "packed"
        native_kv_representation = "packed"

    monkeypatch.setattr(
        mla_functional,
        "_functional_split_materialization_warning_emitted",
        False,
        raising=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mla_functional._select_functional_request(request, PackedRunner)
        mla_functional._select_functional_request(request, PackedRunner)

    user_warnings = [item for item in caught if item.category is UserWarning]
    assert len(user_warnings) == 1
    assert "materialization" in str(user_warnings[0].message)
    assert "per-call allocation" in str(user_warnings[0].message)
    assert user_warnings[0].filename.endswith("test_mla_functional.py")
    assert not [item for item in caught if item.category is DeprecationWarning]


@pytest.mark.parametrize(
    "form", ("packed", "adjacent-split", "redundant", "split-native")
)
def test_functional_zero_copy_forms_do_not_materialize_or_warn(monkeypatch, form):
    packed_query = torch.arange(3.0).reshape(1, 1, 1, 3)
    packed_kv = torch.arange(3.0).reshape(1, 1, 3)
    split_query = (packed_query[..., :2], packed_query[..., 2:])
    split_kv = (packed_kv[..., :2], packed_kv[..., 2:])

    if form == "packed":
        request = _raw_functional_request(
            query=packed_query,
            q_nope=None,
            q_pe=None,
            kv_cache=packed_kv,
            ckv_cache=None,
            kpe_cache=None,
            query_availability="packed",
            kv_availability="packed",
        )
        native_representation = "packed"
    elif form == "adjacent-split":
        request = _raw_functional_request(
            query=None,
            q_nope=split_query[0],
            q_pe=split_query[1],
            kv_cache=None,
            ckv_cache=split_kv[0],
            kpe_cache=split_kv[1],
            query_availability="split",
            kv_availability="split",
        )
        native_representation = "packed"
    elif form == "redundant":
        request = _raw_functional_request(
            query=packed_query,
            q_nope=_PoisonReference(),
            q_pe=_PoisonReference(),
            kv_cache=packed_kv,
            ckv_cache=_PoisonReference(),
            kpe_cache=_PoisonReference(),
            query_availability="redundant",
            kv_availability="redundant",
        )
        native_representation = "packed"
    else:
        request = _raw_functional_request(
            query=None,
            q_nope=split_query[0],
            q_pe=split_query[1],
            kv_cache=None,
            ckv_cache=split_kv[0],
            kpe_cache=split_kv[1],
            query_availability="split",
            kv_availability="split",
        )
        native_representation = "split"

    class NativeRunner:
        native_query_representation = native_representation
        native_kv_representation = native_representation

    monkeypatch.setattr(
        mla_functional,
        "_functional_split_materialization_warning_emitted",
        False,
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("zero-copy functional form called torch.cat")
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = mla_functional._select_functional_request(request, NativeRunner)

    assert not [item for item in caught if item.category is UserWarning]
    if native_representation == "packed":
        assert selected.query.data_ptr() == packed_query.data_ptr()
        assert selected.kv_cache.data_ptr() == packed_kv.data_ptr()
    else:
        assert selected.q_nope is split_query[0]
        assert selected.q_pe is split_query[1]
        assert selected.ckv_cache is split_kv[0]
        assert selected.kpe_cache is split_kv[1]


def test_functional_materialization_propagates_allocation_failure(monkeypatch):
    request = _raw_functional_request(
        query=None,
        q_nope=torch.ones(1, 1, 1, 2),
        q_pe=torch.ones(1, 1, 1, 1),
        query_availability="split",
    )

    class PackedRunner:
        native_query_representation = "packed"
        native_kv_representation = "packed"

    monkeypatch.setattr(
        mla_functional,
        "_functional_split_materialization_warning_emitted",
        False,
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("functional compatibility allocation")
        ),
    )

    with (
        pytest.warns(UserWarning, match="materialization"),
        pytest.raises(
            torch.cuda.OutOfMemoryError,
            match="functional compatibility allocation",
        ),
    ):
        mla_functional._select_functional_request(request, PackedRunner)


def test_explicit_functional_runner_receives_only_its_independently_selected_forms(
    monkeypatch,
):
    query_preference = "packed"
    kv_preference = "split"
    packed_query = torch.empty(2, 1, 4, 3)
    split_kv = (torch.empty(4, 1, 2), torch.empty(4, 1, 1))
    request = _raw_functional_request(
        query=packed_query,
        q_nope=_PoisonReference(),
        q_pe=_PoisonReference(),
        kv_cache=_PoisonReference(),
        ckv_cache=split_kv[0],
        kpe_cache=split_kv[1],
    )
    seen = []

    class RecordingRunner:
        native_query_representation = query_preference
        native_kv_representation = kv_preference

        def __init__(self, selected_request):
            seen.append(selected_request)
            self.request = selected_request

        def prepare_for_dispatch(self):
            pass

        @property
        def inputs(self):
            return []

        def __call__(self, *, inputs, tactic, **_kwargs):
            return self.request

    monkeypatch.setitem(mla_functional._FUNCTIONAL_MLA_RUNNERS, "fa2", RecordingRunner)

    selected = mla_functional._run_functional_mla(request, "fa2")

    assert seen == [selected]
    assert selected.query is packed_query
    assert selected.q_nope is selected.q_pe is None
    assert selected.kv_cache is None
    assert (selected.ckv_cache, selected.kpe_cache) == split_kv


def test_supported_functional_api_emits_no_deprecation_warning():
    case = MLATestCase("sm90-fa3-warning", (9, 0), "fa3")
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flashinfer.mla.batch_mla_paged_attention(**functional_kwargs(case, inputs))
    assert not [item for item in caught if item.category is DeprecationWarning]


def test_trtllm_legacy_facade_warns_once_and_preserves_output_identity(monkeypatch):
    monkeypatch.setattr(
        mla_functional,
        "_trtllm_batch_decode_with_kv_cache_mla_warning_emitted",
        False,
        raising=False,
    )
    case = MLATestCase(
        "sm100-trtllm-legacy", (10, 0), "trtllm-gen", qk_nope_head_dim=128
    )
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    kwargs = functional_kwargs(case, inputs)
    out = torch.empty((2, 1, 128, 512), dtype=case.output_dtype, device="cuda")
    kwargs["out"] = out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(**kwargs)
        second_result = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(**kwargs)

    deprecations = [item for item in caught if item.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert "batch_mla_paged_attention" in str(deprecations[0].message)
    assert result.data_ptr() == out.data_ptr()
    assert second_result.data_ptr() == out.data_ptr()
    assert_mla_close(result.reshape_as(expected_output), expected_output)


def test_xqa_legacy_facade_warns_once_and_matches_reference(monkeypatch):
    monkeypatch.setattr(
        mla_functional,
        "_xqa_batch_decode_with_kv_cache_mla_warning_emitted",
        False,
        raising=False,
    )
    case = MLATestCase("sm120-xqa-legacy", (12, 0), "xqa", qk_nope_head_dim=128)
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    kwargs = functional_kwargs(case, inputs)
    for name in (
        "backend",
        "return_lse",
        "cum_seq_lens_q",
        "max_q_len",
        "is_var_seq",
        "uses_shared_paged_kv_idx",
    ):
        kwargs.pop(name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = flashinfer.mla.xqa_batch_decode_with_kv_cache_mla(**kwargs)
        second_result = flashinfer.mla.xqa_batch_decode_with_kv_cache_mla(**kwargs)

    deprecations = [item for item in caught if item.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert "batch_mla_paged_attention" in str(deprecations[0].message)
    assert_mla_close(result.reshape_as(expected_output), expected_output)
    assert_mla_close(second_result.reshape_as(expected_output), expected_output)


def test_xqa_legacy_fi_trace_warns_once(monkeypatch):
    monkeypatch.setattr(
        mla_functional,
        "_xqa_batch_decode_with_kv_cache_mla_warning_emitted",
        False,
        raising=False,
    )
    kwargs = dict(
        query=torch.empty(2, 1, 128, 576, dtype=torch.bfloat16),
        kv_cache=torch.empty(4, 1, 64, 576, dtype=torch.bfloat16),
        workspace_buffer=torch.empty(1024, dtype=torch.int8),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros(2, 1, dtype=torch.int32),
        seq_lens=torch.full((2,), 64, dtype=torch.int32),
        max_seq_len=64,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        definition = flashinfer.mla.xqa_batch_decode_with_kv_cache_mla.fi_trace(
            **kwargs
        )
        flashinfer.mla.xqa_batch_decode_with_kv_cache_mla.fi_trace(**kwargs)
    deprecations = [item for item in caught if item.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert "batch_mla_paged_attention" in str(deprecations[0].message)
    assert definition["name"].startswith("xqa_batch_decode_mla")
