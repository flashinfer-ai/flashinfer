"""Canonical numerical coverage for the public one-shot MLA API."""

import inspect
import warnings

import pytest
import torch

import flashinfer
import flashinfer.mla._core as mla_core
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
    "MLAQuery",
    "MLAKVCache",
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


def test_functional_case_table_covers_every_explicit_backend():
    assert {case.backend for case in _FUNCTIONAL_CASES} == _FUNCTIONAL_BACKENDS


def test_functional_case_table_covers_public_configuration_dimensions():
    assert {case.q_dtype for case in _FUNCTIONAL_CASES} == {
        torch.float16,
        torch.bfloat16,
    }
    assert {case.page_size for case in _FUNCTIONAL_CASES} == {32, 64, 128}
    assert {case.q_len for case in _FUNCTIONAL_CASES} == {1, 2}
    assert {case.lse_mode for case in _FUNCTIONAL_CASES} == {
        "none",
        "base2",
        "basee",
    }
    assert {case.scale_mode for case in _FUNCTIONAL_CASES} == {
        "default",
        "bmm-scalar",
        "bmm-tensor",
    }
    assert {case.skip_softmax for case in _FUNCTIONAL_CASES} == {False, True}
    assert {case.enable_pdl for case in _FUNCTIONAL_CASES} == {None, False, True}


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


def test_functional_exports_and_legacy_signature_are_stable():
    assert hasattr(flashinfer.mla, "batch_mla_paged_attention")
    assert not hasattr(flashinfer.decode, "batch_mla_paged_attention")
    assert not hasattr(flashinfer, "batch_mla_paged_attention")
    assert inspect.signature(
        flashinfer.mla.batch_mla_paged_attention
    ) == inspect.signature(flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla)
    namespace = {}
    exec("from flashinfer.mla import *", namespace)
    assert {name for name in namespace if not name.startswith("__")} == _PUBLIC_MLA_API


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
        mla_core,
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
