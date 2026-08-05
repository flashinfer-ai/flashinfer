"""Canonical numerical coverage for the public stateful MLA wrapper."""

from dataclasses import replace
from contextlib import nullcontext
import warnings

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.mla._batch_mla import _auto_policy
from flashinfer.mla import (
    BatchMLAPagedAttentionWrapper,
    MLAKVCache,
    MLAPlanMetadata,
    MLAQuery,
)
from tests.test_helpers.mla import (
    MLATestCase,
    assert_mla_close,
    make_mla_inputs,
    reference_result,
    require_architecture,
    unpack_mla_result,
    wrapper_plan_kwargs,
    wrapper_run_kwargs,
)


_WRAPPER_BACKENDS = {
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
}

_WRAPPER_CASES = (
    MLATestCase("sm80-fa2-decode", (8, 0), "fa2"),
    MLATestCase(
        "sm90-fa2-fp16-csr-split-basee",
        (9, 0),
        "fa2",
        page_size=64,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        kv_layout="independent-split",
        lse_mode="basee",
        metadata_form="csr",
    ),
    MLATestCase(
        "sm90-fa2-prefill-adjacent-base2",
        (9, 0),
        "fa2",
        q_len=2,
        page_size=128,
        kv_layout="adjacent-split",
        lse_mode="base2",
    ),
    MLATestCase("sm90-fa3-decode", (9, 0), "fa3", lse_mode="base2"),
    MLATestCase(
        "sm90-fa3-fp16-csr-adjacent-basee",
        (9, 0),
        "fa3",
        page_size=64,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        kv_layout="adjacent-split",
        lse_mode="basee",
        metadata_form="csr",
    ),
    MLATestCase(
        "sm90-fa3-fp8-kv-scale",
        (9, 0),
        "fa3",
        page_size=128,
        kv_dtype=torch.float8_e4m3fn,
        kv_layout="independent-split",
        scale_mode="kv-per-tensor",
    ),
    MLATestCase(
        "sm100-cutlass-decode",
        (10, 0),
        "cutlass",
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-cutlass-adjacent-page64",
        (10, 0),
        "cutlass",
        page_size=64,
        kv_layout="adjacent-split",
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm100-cutlass-fp8-output-scale",
        (10, 0),
        "cutlass",
        page_size=128,
        output_dtype=torch.float8_e4m3fn,
        kv_layout="independent-split",
        output_scale="per-tensor",
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
        kv_layout="adjacent-split",
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
        is_var_seq=True,
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
        kv_layout="adjacent-split",
        lse_mode="basee",
        scale_mode="bmm-scalar",
        is_var_seq=True,
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
        kv_layout="adjacent-split",
        scale_mode="bmm-scalar",
        is_var_seq=True,
    ),
    MLATestCase(
        "sm120-xqa-decode",
        (12, 0),
        "xqa",
        softmax_scale_qk_nope_head_dim=128,
    ),
    MLATestCase(
        "sm120-xqa-adjacent-scalar-pdl",
        (12, 0),
        "xqa",
        page_size=64,
        kv_layout="adjacent-split",
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
    MLATestCase("sm90-auto", (9, 0), "auto"),
    MLATestCase("sm100-auto", (10, 0), "auto", qk_nope_head_dim=128),
    MLATestCase("sm103-auto", (10, 3), "auto", qk_nope_head_dim=128),
    MLATestCase("sm120-auto", (12, 0), "auto", softmax_scale_qk_nope_head_dim=128),
    MLATestCase("sm121-auto", (12, 1), "auto", softmax_scale_qk_nope_head_dim=128),
)

_AUTOTUNE_CASES = (
    MLATestCase(
        "sm90-autotune-split-base2",
        (9, 0),
        "auto",
        kv_layout="independent-split",
        lse_mode="base2",
    ),
    MLATestCase("sm90-autotune-fixed-q2", (9, 0), "auto", q_len=2),
    MLATestCase("sm100-autotune", (10, 0), "auto", qk_nope_head_dim=128),
    MLATestCase(
        "sm120-autotune",
        (12, 0),
        "auto",
        softmax_scale_qk_nope_head_dim=128,
    ),
)


def _workspace() -> torch.Tensor:
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _metadata(case, inputs):
    if case.metadata_form == "csr":
        return MLAPlanMetadata.csr(
            inputs.qo_indptr,
            inputs.kv_indptr,
            inputs.kv_indices,
            inputs.kv_len_arr,
        )
    return MLAPlanMetadata.dense(
        inputs.cum_seq_lens_q,
        inputs.block_tables,
        inputs.seq_lens,
        case.q_len,
    )


def _query_and_kv(inputs):
    query = MLAQuery.split(inputs.q_nope, inputs.q_pe)
    if inputs.kv_cache is not None:
        return query, MLAKVCache.packed(inputs.kv_cache)
    return query, MLAKVCache.split(inputs.ckv_cache, inputs.kpe_cache)


def _run_public_wrapper(case, inputs, *, tuning_buckets=None, tune_mode=True):
    wrapper = BatchMLAPagedAttentionWrapper(_workspace(), backend=case.backend)
    plan_kwargs = wrapper_plan_kwargs(case, inputs)
    for name in ("cum_seq_lens_q", "block_tables", "seq_lens", "max_q_len"):
        plan_kwargs.pop(name)
    tuning_context = (
        autotune(tune_mode, tuning_buckets=tuning_buckets)
        if tuning_buckets is not None
        else nullcontext()
    )
    with tuning_context:
        wrapper.plan(metadata=_metadata(case, inputs), **plan_kwargs)
    query, kv = _query_and_kv(inputs)
    run_kwargs = wrapper_run_kwargs(case, inputs)
    for name in ("kv_cache", "ckv_cache", "kpe_cache"):
        run_kwargs.pop(name)
    result = wrapper.run(query=query, kv=kv, **run_kwargs)
    return wrapper, unpack_mla_result(result, case.lse_mode != "none")


def test_wrapper_case_table_covers_every_explicit_backend():
    assert {case.backend for case in _WRAPPER_CASES} == _WRAPPER_BACKENDS


def test_wrapper_rejects_removed_autotune_selector():
    with pytest.raises(ValueError, match="backend must be one of"):
        BatchMLAPagedAttentionWrapper(torch.empty(1), backend="autotune")


def test_wrapper_case_table_covers_public_configuration_dimensions():
    assert {case.q_dtype for case in _WRAPPER_CASES} == {
        torch.float16,
        torch.bfloat16,
    }
    assert {case.page_size for case in _WRAPPER_CASES} == {32, 64, 128}
    assert {case.q_len for case in _WRAPPER_CASES} == {1, 2}
    assert {case.kv_layout for case in _WRAPPER_CASES} == {
        "combined",
        "adjacent-split",
        "independent-split",
    }
    assert {case.lse_mode for case in _WRAPPER_CASES} == {
        "none",
        "base2",
        "basee",
    }
    assert {case.scale_mode for case in _WRAPPER_CASES} == {
        "default",
        "bmm-scalar",
        "bmm-tensor",
        "kv-per-tensor",
    }
    assert {case.output_scale for case in _WRAPPER_CASES} == {
        "none",
        "per-tensor",
    }
    assert {case.skip_softmax for case in _WRAPPER_CASES} == {False, True}
    assert {case.metadata_form for case in _WRAPPER_CASES} == {"dense", "csr"}
    assert {case.enable_pdl for case in _WRAPPER_CASES} == {None, False, True}


@pytest.mark.parametrize("case", _WRAPPER_CASES, ids=lambda case: case.case_id)
def test_explicit_wrapper_matches_reference(case):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, expected_lse = reference_result(case, inputs)

    wrapper, (actual_output, actual_lse) = _run_public_wrapper(case, inputs)

    assert wrapper.resolved_backend == case.backend
    assert_mla_close(
        actual_output,
        expected_output,
        fp8=case.output_dtype == torch.float8_e4m3fn,
    )
    if expected_lse is not None:
        assert_mla_close(actual_lse, expected_lse)


@pytest.mark.parametrize("case", _AUTO_CASES, ids=lambda case: case.case_id)
def test_wrapper_auto_matches_selected_backend_and_reference(case):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, expected_lse = reference_result(case, inputs)

    automatic, automatic_result = _run_public_wrapper(case, inputs)
    assert automatic.resolved_backend in _WRAPPER_BACKENDS
    assert automatic.auto_selection_trace is not None
    assert automatic.auto_selection_trace.mode == "deterministic"
    explicit, explicit_result = _run_public_wrapper(
        replace(case, backend=automatic.resolved_backend), inputs
    )

    assert explicit.resolved_backend == automatic.resolved_backend
    assert_mla_close(automatic_result[0], expected_output)
    assert_mla_close(automatic_result[0], explicit_result[0])
    if expected_lse is not None:
        assert_mla_close(automatic_result[1], expected_lse)
        assert_mla_close(automatic_result[1], explicit_result[1])


def test_failed_auto_replan_preserves_previous_state(monkeypatch):
    class BackendImpl:
        def __init__(self, result):
            self.result = result

        def run_from_wrapper(self, **_kwargs):
            return self.result

    trace = _auto_policy.MLAAutoSelectionTrace(
        candidates=("fa2",),
        rejections=(),
        mode="cache-only",
        bypass_reason=None,
        resolved_backend="fa2",
    )
    responses = iter(
        (
            _auto_policy._MLAAutoPlanResult("fa2", BackendImpl("old-result"), trace),
            RuntimeError("replan failed"),
        )
    )

    def resolve(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(_auto_policy, "plan_auto_backend", resolve)
    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    plan_kwargs = {
        "metadata": MLAPlanMetadata.csr(
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            kv_indices=torch.tensor([0], dtype=torch.int32),
            kv_len_arr=torch.tensor([1], dtype=torch.int32),
        ),
        "num_heads": 1,
        "head_dim_ckv": 2,
        "head_dim_kpe": 1,
        "page_size": 1,
        "causal": False,
        "sm_scale": 1.0,
        "q_data_type": torch.float32,
        "kv_data_type": torch.float32,
        "kv_layout": "independent-split",
    }

    with autotune(False):
        wrapper.plan(**plan_kwargs)
    old_trace = wrapper.auto_selection_trace

    with pytest.raises(RuntimeError, match="replan failed"), autotune(False):
        wrapper.plan(**plan_kwargs)

    assert wrapper.resolved_backend == "fa2"
    assert wrapper.auto_selection_trace is old_trace
    assert (
        wrapper.run(
            query=MLAQuery.split(torch.empty(1, 1, 2), torch.empty(1, 1, 1)),
            kv=MLAKVCache.split(torch.empty(1, 1, 2), torch.empty(1, 1, 1)),
        )
        == "old-result"
    )


@pytest.mark.parametrize("case", _AUTOTUNE_CASES, ids=lambda case: case.case_id)
def test_wrapper_autotune_populates_cache_and_matches_reference(case):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, expected_lse = reference_result(case, inputs)
    AutoTuner.get().clear_cache()

    tuned, tuned_result = _run_public_wrapper(
        case, inputs, tuning_buckets=(1, 2, 4), tune_mode=True
    )
    warm, warm_result = _run_public_wrapper(
        case, inputs, tuning_buckets=(1, 2, 4), tune_mode=False
    )

    assert tuned.resolved_backend in _WRAPPER_BACKENDS
    assert warm.resolved_backend == tuned.resolved_backend
    assert tuned.auto_selection_trace is not None
    assert tuned.auto_selection_trace.mode == "tuning"
    assert warm.auto_selection_trace is not None
    assert warm.auto_selection_trace.mode == "cache-only"
    assert_mla_close(tuned_result[0], expected_output)
    assert_mla_close(warm_result[0], expected_output)
    assert_mla_close(warm_result[0], tuned_result[0])
    if expected_lse is not None:
        assert_mla_close(tuned_result[1], expected_lse)
        assert_mla_close(warm_result[1], expected_lse)


def test_sm100_wrapper_autotune_run_is_cuda_graph_capturable():
    case = MLATestCase(
        "sm100-autotune-cuda-graph", (10, 0), "auto", qk_nope_head_dim=128
    )
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    AutoTuner.get().clear_cache()
    wrapper = BatchMLAPagedAttentionWrapper(
        _workspace(),
        use_cuda_graph=True,
        qo_indptr=torch.empty_like(inputs.qo_indptr),
        kv_indptr=torch.empty_like(inputs.kv_indptr),
        kv_indices=torch.empty_like(inputs.kv_indices),
        kv_len_arr=torch.empty_like(inputs.kv_len_arr),
        backend="auto",
    )
    plan_kwargs = wrapper_plan_kwargs(case, inputs)
    for name in ("cum_seq_lens_q", "block_tables", "seq_lens", "max_q_len"):
        plan_kwargs.pop(name)
    with autotune(True, tuning_buckets=(1, 2, 4)):
        wrapper.plan(metadata=_metadata(case, inputs), **plan_kwargs)

    query, kv = _query_and_kv(inputs)
    graph_output = torch.empty_like(inputs.q_nope)
    wrapper.run(query=query, kv=kv, out=graph_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(query=query, kv=kv, out=graph_output)
    graph.replay()
    torch.cuda.synchronize()

    assert_mla_close(graph_output, expected_output)


def test_sm90_wrapper_autotune_supports_profiler():
    case = MLATestCase("sm90-autotune-profiler", (9, 0), "auto")
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    AutoTuner.get().clear_cache()
    wrapper = BatchMLAPagedAttentionWrapper(_workspace(), backend="auto")
    plan_kwargs = wrapper_plan_kwargs(case, inputs)
    for name in ("cum_seq_lens_q", "block_tables", "seq_lens", "max_q_len"):
        plan_kwargs.pop(name)
    with autotune(True, tuning_buckets=(1, 2, 4)):
        wrapper.plan(metadata=_metadata(case, inputs), use_profiler=True, **plan_kwargs)

    query, kv = _query_and_kv(inputs)
    output = wrapper.run(
        query=query,
        kv=kv,
        profiler_buffer=torch.empty(1 << 20, dtype=torch.uint64, device="cuda"),
    )

    assert wrapper.auto_selection_trace is not None
    assert wrapper.auto_selection_trace.mode == "tuning"
    assert_mla_close(output, expected_output)


def test_legacy_wrapper_split_calls_warn_once_and_match_reference():
    case = MLATestCase("sm90-fa3-legacy", (9, 0), "fa3", kv_layout="independent-split")
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    wrapper = BatchMLAPagedAttentionWrapper(_workspace(), backend="fa3")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper.plan(
            inputs.qo_indptr,
            inputs.kv_indptr,
            inputs.kv_indices,
            inputs.kv_len_arr,
            128,
            512,
            64,
            case.page_size,
            False,
            case.sm_scale,
            case.q_dtype,
            case.kv_dtype,
            kv_layout=case.kv_layout,
        )
        first = wrapper.run(
            inputs.q_nope, inputs.q_pe, inputs.ckv_cache, inputs.kpe_cache
        )
        warning_count = len(caught)
        second = wrapper.run(
            inputs.q_nope, inputs.q_pe, inputs.ckv_cache, inputs.kpe_cache
        )

    assert warning_count == len(caught)
    assert sum("Positional MLA arguments" in str(item.message) for item in caught) == 1
    assert sum("Legacy MLA metadata" in str(item.message) for item in caught) == 1
    assert_mla_close(first, expected_output)
    assert_mla_close(second, expected_output)
