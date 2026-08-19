"""Canonical numerical coverage for the public stateful MLA wrapper."""

from dataclasses import replace
from contextlib import nullcontext
import inspect
import warnings

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.mla._batch_mla import _auto_policy
from flashinfer.mla._batch_mla import _wrapper as wrapper_module
from flashinfer.mla._batch_mla._contracts import (
    _resolve_structural_mla_input,
)
from flashinfer.mla import (
    BatchMLAPagedAttentionWrapper,
    MLAPlanMetadata,
)
from tests.test_helpers.mla import (
    MLATestCase,
    assert_mla_close,
    functional_kwargs,
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
        kv_layout="adjacent-split",
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
    run_kwargs = wrapper_run_kwargs(case, inputs)
    run_kwargs.update(q_nope=inputs.q_nope, q_pe=inputs.q_pe)
    result = wrapper.run(**run_kwargs)
    return wrapper, unpack_mla_result(result, case.lse_mode != "none")


def test_wrapper_rejects_removed_autotune_selector():
    with pytest.raises(ValueError, match="backend must be one of"):
        BatchMLAPagedAttentionWrapper(torch.empty(1), backend="autotune")


def _minimal_mla_plan_metadata():
    return MLAPlanMetadata.csr(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_len_arr=torch.tensor([1], dtype=torch.int32),
    )


def test_metadata_driven_plan_is_warning_free(monkeypatch):
    captured = {}
    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    monkeypatch.setattr(
        wrapper,
        "_plan_backend",
        lambda backend, args: captured.update(backend=backend, args=args),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        wrapper.plan(
            metadata=_minimal_mla_plan_metadata(),
            num_heads=2,
            head_dim_ckv=2,
            head_dim_kpe=1,
            page_size=4,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.float32,
            kv_data_type=torch.float32,
            query_layout="split",
            kv_cache_layout="packed",
        )
    assert captured["args"].query_kind == "independent-split"
    assert captured["args"].kv_kind == "packed"


def test_trusted_dual_structural_query_reaches_split_backend_unchanged():
    recorded = {}
    output = torch.empty(1)

    class PoisonReference:
        def __getattribute__(self, _name):
            raise AssertionError("ignored redundant reference was accessed")

    class BackendImpl:
        def run_from_wrapper(self, **kwargs):
            recorded.update(kwargs)
            resolved_query = _resolve_structural_mla_input(
                kwargs["query"],
                desired="split",
                widths=None,
                name="query",
            )
            resolved_kv = _resolve_structural_mla_input(
                kwargs["kv_cache"],
                desired="split",
                widths=None,
                name="KV cache",
            )
            assert resolved_query[0] is q_nope
            assert resolved_query[1] is q_pe
            assert resolved_kv[0] is ckv_cache
            assert resolved_kv[1] is kpe_cache
            return output

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper._selected_backend = "fa2"
    wrapper._backend_impl = BackendImpl()
    q_nope, q_pe = torch.empty(1, 1, 2), torch.empty(1, 1, 1)
    ckv_cache, kpe_cache = torch.empty(1, 1, 2), torch.empty(1, 1, 1)

    actual = wrapper.run(
        query=(PoisonReference(), (q_nope, q_pe)),
        kv_cache=(PoisonReference(), (ckv_cache, kpe_cache)),
    )

    assert actual is output
    assert recorded["query"][1] == (q_nope, q_pe)
    assert recorded["kv_cache"][1] == (ckv_cache, kpe_cache)
    assert "q_nope" not in recorded
    assert "ckv_cache" not in recorded


def test_trusted_dual_structural_query_reaches_packed_backend_unchanged():
    recorded = {}
    output = torch.empty(1)

    class PoisonReference:
        def __getattribute__(self, _name):
            raise AssertionError("ignored redundant reference was accessed")

    class BackendImpl:
        def run_from_wrapper(self, **kwargs):
            recorded.update(kwargs)
            assert (
                _resolve_structural_mla_input(
                    kwargs["query"],
                    desired="packed",
                    widths=None,
                    name="query",
                )
                is query
            )
            assert (
                _resolve_structural_mla_input(
                    kwargs["kv_cache"],
                    desired="packed",
                    widths=None,
                    name="KV cache",
                )
                is kv_cache
            )
            return output

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")
    wrapper._selected_backend = "cutlass"
    wrapper._backend_impl = BackendImpl()
    query = torch.empty(1, 1, 3)
    kv_cache = torch.empty(1, 1, 3)
    q_nope_poison, q_pe_poison = PoisonReference(), PoisonReference()
    ckv_poison, kpe_poison = PoisonReference(), PoisonReference()

    actual = wrapper.run(
        query=(query, (q_nope_poison, q_pe_poison)),
        kv_cache=(kv_cache, (ckv_poison, kpe_poison)),
    )

    assert actual is output
    assert recorded["query"][0] is query
    assert recorded["kv_cache"][0] is kv_cache
    assert "q_nope" not in recorded
    assert "ckv_cache" not in recorded


def test_planned_tensor_keywords_keep_mixed_raw_references():
    recorded = {}
    output = torch.empty(1)

    class BackendImpl:
        def run_from_wrapper(self, **kwargs):
            recorded.update(kwargs)
            return output

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper._selected_backend = "fa2"
    wrapper._backend_impl = BackendImpl()
    query = torch.empty(1, 1, 3)
    ckv_cache, kpe_cache = torch.empty(1, 1, 2), torch.empty(1, 1, 1)

    actual = wrapper.run(
        query=query,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
    )

    assert actual is output
    assert recorded["query"] is query
    assert recorded["kv_cache"] == (ckv_cache, kpe_cache)
    assert "q_nope" not in recorded
    assert "ckv_cache" not in recorded


def test_structural_split_run_forwards_query_and_kv_cache_unchanged():
    recorded = {}
    output = torch.empty(1)

    class BackendImpl:
        def run_from_wrapper(self, **kwargs):
            recorded.update(kwargs)
            return output

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper._selected_backend = "fa2"
    wrapper._backend_impl = BackendImpl()
    query = (torch.empty(1, 1, 2), torch.empty(1, 1, 1))
    kv_cache = (torch.empty(1, 1, 2), torch.empty(1, 1, 1))

    actual = wrapper.run(query=query, kv_cache=kv_cache)

    assert actual is output
    assert recorded["query"] is query
    assert recorded["kv_cache"] is kv_cache
    assert "q_nope" not in recorded
    assert "ckv_cache" not in recorded


def test_legacy_split_run_warns_once_and_translates_to_structural_split():
    recorded = []
    output = torch.empty(1)

    class BackendImpl:
        def run_from_wrapper(self, **kwargs):
            recorded.append(kwargs)
            return output

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper._selected_backend = "fa2"
    wrapper._backend_impl = BackendImpl()
    q_nope, q_pe = torch.empty(1, 1, 2), torch.empty(1, 1, 1)
    ckv_cache, kpe_cache = torch.empty(1, 1, 2), torch.empty(1, 1, 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper.run(q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv_cache, kpe_cache=kpe_cache)
        wrapper.run(q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv_cache, kpe_cache=kpe_cache)

    assert (
        sum("Legacy MLA tensor arguments" in str(item.message) for item in caught) == 1
    )
    assert recorded[0]["query"] == (q_nope, q_pe)
    assert recorded[0]["kv_cache"] == (ckv_cache, kpe_cache)
    assert "q_nope" not in recorded[0]
    assert "ckv_cache" not in recorded[0]


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


def test_sm90_fa3_cuda_graph_plan_accepts_host_control_metadata():
    case = MLATestCase(
        "sm90-fa3-host-control-metadata",
        (9, 0),
        "fa3",
        metadata_form="csr",
        kv_layout="independent-split",
    )
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    expected_output, _ = reference_result(case, inputs)
    wrapper = BatchMLAPagedAttentionWrapper(
        _workspace(),
        use_cuda_graph=True,
        qo_indptr=torch.empty_like(inputs.qo_indptr),
        kv_indptr=torch.empty_like(inputs.kv_indptr),
        kv_indices=torch.empty_like(inputs.kv_indices),
        kv_len_arr=torch.empty_like(inputs.kv_len_arr),
        backend=case.backend,
    )
    plan_kwargs = wrapper_plan_kwargs(case, inputs)
    for name in ("cum_seq_lens_q", "block_tables", "seq_lens", "max_q_len"):
        plan_kwargs.pop(name)
    plan_kwargs["query_layout"] = "split"
    wrapper.plan(
        metadata=MLAPlanMetadata.csr(
            inputs.qo_indptr.cpu(),
            inputs.kv_indptr.cpu(),
            inputs.kv_indices,
            inputs.kv_len_arr.cpu(),
        ),
        **plan_kwargs,
    )

    actual_output, _ = unpack_mla_result(
        wrapper.run(
            query=(inputs.q_nope, inputs.q_pe),
            kv_cache=(inputs.ckv_cache, inputs.kpe_cache),
        ),
        False,
    )

    assert wrapper.resolved_backend == "fa3"
    assert_mla_close(actual_output, expected_output)


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
        "query_layout": "split",
        "kv_cache_layout": "split",
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
            q_nope=torch.empty(1, 1, 2),
            q_pe=torch.empty(1, 1, 1),
            ckv_cache=torch.empty(1, 1, 2),
            kpe_cache=torch.empty(1, 1, 1),
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

    graph_output = torch.empty_like(inputs.q_nope)
    wrapper.run(
        q_nope=inputs.q_nope,
        q_pe=inputs.q_pe,
        kv_cache=inputs.kv_cache,
        ckv_cache=inputs.ckv_cache,
        kpe_cache=inputs.kpe_cache,
        out=graph_output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q_nope=inputs.q_nope,
            q_pe=inputs.q_pe,
            kv_cache=inputs.kv_cache,
            ckv_cache=inputs.ckv_cache,
            kpe_cache=inputs.kpe_cache,
            out=graph_output,
        )
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

    output = wrapper.run(
        q_nope=inputs.q_nope,
        q_pe=inputs.q_pe,
        kv_cache=inputs.kv_cache,
        ckv_cache=inputs.ckv_cache,
        kpe_cache=inputs.kpe_cache,
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
    assert (
        sum(
            "flat metadata arguments should be replaced with MLAPlanMetadata"
            in str(item.message)
            for item in caught
        )
        == 1
    )
    assert_mla_close(first, expected_output)
    assert_mla_close(second, expected_output)


def test_packed_native_wrapper_rejects_independent_structural_kv_zero_copy():
    class Backend:
        def run_from_wrapper(self, **kwargs):
            _resolve_structural_mla_input(
                kwargs["kv_cache"],
                desired="packed",
                widths=(2, 1),
                name="KV cache",
            )
            return "launched"

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")
    wrapper._selected_backend = "cutlass"
    wrapper._backend_impl = Backend()
    wrapper._input_contract = wrapper_module.MLAInputContract(
        lse_mode="none",
        output_dtype=torch.float32,
        output_scale="none",
        scale_mode="default",
        skip_softmax=False,
        head_dim_ckv=2,
        head_dim_kpe=1,
    )

    with pytest.raises(ValueError, match=r"KV cache.*packed representation zero-copy"):
        wrapper.run(
            query=torch.empty(1, 1, 3),
            kv_cache=(torch.empty(1, 1, 2), torch.empty(1, 1, 1)),
        )


def test_unplanned_cutlass_warning_is_attributed_to_public_caller(monkeypatch):
    recorded = []

    class Backend:
        def run_from_wrapper(self, **kwargs):
            recorded.append(kwargs)
            return kwargs["out"]

    monkeypatch.setattr(
        wrapper_module._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda _cls, _args: Backend()),
    )
    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")
    query = (torch.full((1, 1, 512), 1.0), torch.full((1, 1, 64), 2.0))
    kv_cache = (torch.full((1, 1, 512), 3.0), torch.full((1, 1, 64), 4.0))
    out = torch.empty(1, 1, 512)
    kv_len = torch.tensor([1], dtype=torch.int32)
    page_table = torch.zeros(1, 1, dtype=torch.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_lineno = inspect.currentframe().f_lineno + 1
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            out=out,
            kv_len=kv_len,
            page_table=page_table,
        )
        wrapper.run(query=query, kv_cache=kv_cache, out=out)

    cutlass_warnings = [
        item
        for item in caught
        if "explicitly requested CUTLASS backend" in str(item.message)
    ]
    assert len(cutlass_warnings) == 1
    warning = cutlass_warnings[0]
    assert warning.filename == __file__
    assert warning.lineno == expected_lineno
    assert len(recorded) == 2
    expected_query = torch.cat(query, dim=-1)
    expected_kv_cache = torch.cat(kv_cache, dim=-1)
    for call in recorded:
        torch.testing.assert_close(call["query"], expected_query)
        torch.testing.assert_close(call["kv_cache"], expected_kv_cache)


_TENSOR_FIRST_PRODUCTION_ROWS = (
    (
        MLATestCase("sm80-fa2-split", (8, 0), "fa2", kv_layout="adjacent-split"),
        "split",
        False,
    ),
    (
        MLATestCase(
            "sm90-fa3-split-prefill-lse",
            (9, 0),
            "fa3",
            q_len=2,
            kv_layout="adjacent-split",
            lse_mode="base2",
        ),
        "split",
        False,
    ),
    (
        MLATestCase(
            "sm100-cutlass-packed",
            (10, 0),
            "cutlass",
            softmax_scale_qk_nope_head_dim=128,
        ),
        "packed",
        False,
    ),
    (
        MLATestCase(
            "sm100-trtllm-redundant-graph",
            (10, 0),
            "trtllm-gen",
            qk_nope_head_dim=128,
        ),
        "redundant",
        True,
    ),
    (
        MLATestCase(
            "sm100-cute-monolithic-prefill-lse",
            (10, 0),
            "cute-dsl-monolithic",
            q_len=2,
            lse_mode="basee",
        ),
        "packed",
        False,
    ),
    (
        MLATestCase(
            "sm100-cute-modular-split",
            (10, 0),
            "cute-dsl-modular",
            kv_layout="adjacent-split",
        ),
        "split",
        False,
    ),
    (
        MLATestCase(
            "sm120-xqa-packed-graph",
            (12, 0),
            "xqa",
            softmax_scale_qk_nope_head_dim=128,
        ),
        "packed",
        True,
    ),
)


def _tensor_first_references(case, inputs, form):
    packed_kv = (
        inputs.kv_cache
        if inputs.kv_cache is not None
        else torch.cat((inputs.ckv_cache, inputs.kpe_cache), dim=-1)
    )
    q_nope = inputs.q_nope.reshape(2, case.q_len, 128, 512)
    q_pe = inputs.q_pe.reshape(2, case.q_len, 128, 64)
    ckv_cache, kpe_cache = packed_kv[..., :512], packed_kv[..., 512:]
    if form == "packed":
        return {"query": inputs.query, "kv_cache": packed_kv}
    if form == "split":
        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "ckv_cache": ckv_cache,
            "kpe_cache": kpe_cache,
        }
    return {
        "query": inputs.query,
        "q_nope": q_nope,
        "q_pe": q_pe,
        "kv_cache": packed_kv,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
    }


def _run_tensor_first_functional(case, inputs, references, use_cuda_graph):
    kwargs = functional_kwargs(case, inputs)
    kwargs.pop("query")
    kwargs.pop("kv_cache")
    kwargs.update(references)
    if use_cuda_graph:
        kwargs["out"] = torch.empty(
            (2, case.q_len, 128, 512),
            dtype=case.output_dtype,
            device="cuda",
        )
        flashinfer.mla.batch_mla_paged_attention(**kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = flashinfer.mla.batch_mla_paged_attention(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
    else:
        result = flashinfer.mla.batch_mla_paged_attention(**kwargs)
    return unpack_mla_result(result, case.lse_mode != "none")


def _run_tensor_first_wrapper(case, inputs, references, use_cuda_graph):
    wrapper_kwargs = {"backend": case.backend, "use_cuda_graph": use_cuda_graph}
    if use_cuda_graph:
        wrapper_kwargs.update(
            qo_indptr=torch.empty_like(inputs.qo_indptr),
            kv_indptr=torch.empty_like(inputs.kv_indptr),
            kv_indices=torch.empty_like(inputs.kv_indices),
            kv_len_arr=torch.empty_like(inputs.kv_len_arr),
        )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        **wrapper_kwargs,
    )
    plan_kwargs = wrapper_plan_kwargs(case, inputs)
    for name in ("cum_seq_lens_q", "block_tables", "seq_lens", "max_q_len"):
        plan_kwargs.pop(name)
    wrapper.plan(
        metadata=MLAPlanMetadata.dense(
            inputs.cum_seq_lens_q,
            inputs.block_tables,
            inputs.seq_lens,
            case.q_len,
        ),
        **plan_kwargs,
    )
    run_kwargs = wrapper_run_kwargs(case, inputs)
    for name in ("kv_cache", "ckv_cache", "kpe_cache"):
        run_kwargs.pop(name, None)
    run_kwargs.update(
        {
            name: (
                tensor.flatten(0, 1) if name in ("query", "q_nope", "q_pe") else tensor
            )
            for name, tensor in references.items()
        }
    )
    if use_cuda_graph:
        run_kwargs["out"] = torch.empty_like(inputs.q_nope)
        wrapper.run(**run_kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = wrapper.run(**run_kwargs)
        graph.replay()
        torch.cuda.synchronize()
    else:
        result = wrapper.run(**run_kwargs)
    return unpack_mla_result(result, case.lse_mode != "none")


@pytest.mark.parametrize(
    "case,input_form,use_cuda_graph",
    _TENSOR_FIRST_PRODUCTION_ROWS,
    ids=lambda value: value.case_id if isinstance(value, MLATestCase) else str(value),
)
def test_tensor_first_functional_and_wrapper_production_matrix(
    case, input_form, use_cuda_graph
):
    require_architecture(case.architecture)
    inputs = make_mla_inputs(case)
    references = _tensor_first_references(case, inputs, input_form)
    expected_output, expected_lse = reference_result(
        case,
        inputs,
        causal=case.backend == "cute-dsl-monolithic" and case.q_len > 1,
    )

    functional_output, functional_lse = _run_tensor_first_functional(
        case, inputs, references, use_cuda_graph
    )
    assert_mla_close(functional_output.reshape_as(expected_output), expected_output)
    if expected_lse is not None:
        assert functional_lse is not None
        assert_mla_close(functional_lse.reshape_as(expected_lse), expected_lse)

    wrapper_output, wrapper_lse = _run_tensor_first_wrapper(
        case, inputs, references, use_cuda_graph
    )
    assert_mla_close(wrapper_output.reshape_as(expected_output), expected_output)
    if expected_lse is not None:
        assert wrapper_lse is not None
        assert_mla_close(wrapper_lse.reshape_as(expected_lse), expected_lse)
