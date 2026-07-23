"""Planning and dispatch behavior for the public batch MLA wrapper."""

import dataclasses
import gc
import inspect
import logging
from typing import Tuple, Union

import pytest
import torch

import flashinfer._backend as backend_errors
from flashinfer.mla import _core as mla_core
from flashinfer.mla._batch_mla import _wrapper as batch_mla_core
from flashinfer.mla._batch_mla import _functional as batch_mla_functional
from flashinfer.mla._batch_mla import _planning as batch_mla_planning
from flashinfer.mla._batch_mla import _wrapper as batch_mla_wrapper
from flashinfer.mla._batch_mla._backends import _fa_common
from flashinfer.mla._batch_mla._backends import cutlass_backend
from flashinfer.mla._batch_mla._backends import cute_dsl_backend
from flashinfer.mla._batch_mla._backends import fa2_backend
from flashinfer.mla._batch_mla._backends import fa3_backend
from flashinfer.mla._batch_mla._backends import trtllm_gen_backend
from flashinfer.mla._batch_mla._backends import xqa_backend


class _RecordingBackend:
    def __init__(self, name):
        self._backend = name
        self.device = torch.device("cpu")
        self.plan_kwargs = None
        self.run_kwargs = None
        self.result = torch.tensor([17])

    def plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self.result

    def run_from_wrapper(self, **kwargs):
        self.run_kwargs = kwargs
        return self.result


_RUN_FROM_WRAPPER_PARAMETERS = (
    "self",
    "q_nope",
    "q_pe",
    "ckv_cache",
    "kpe_cache",
    "out",
    "lse",
    "return_lse",
    "profiler_buffer",
    "kv_len",
    "page_table",
    "return_lse_base_on_e",
    "o_scale",
    "ckv_scale",
    "kpe_scale",
    "sinks",
    "skip_softmax_threshold_scale_factor",
    "bmm1_scale",
    "bmm2_scale",
)

_RUN_FROM_WRAPPER_RETURN = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def test_batch_mla_public_surfaces_have_explicit_implementation_modules():
    assert (
        batch_mla_wrapper.BatchMLAPagedAttentionWrapper.__module__
        == "flashinfer.mla._batch_mla._wrapper"
    )
    assert (
        batch_mla_functional.xqa_batch_decode_with_kv_cache_mla.__module__
        == "flashinfer.mla._batch_mla._functional"
    )
    assert (
        mla_core.BatchMLAPagedAttentionWrapper
        is batch_mla_wrapper.BatchMLAPagedAttentionWrapper
    )
    assert (
        mla_core.xqa_batch_decode_with_kv_cache_mla
        is batch_mla_functional.xqa_batch_decode_with_kv_cache_mla
    )


def _run_from_wrapper_kwargs(**overrides):
    kwargs = dict(
        q_nope=object(),
        q_pe=object(),
        ckv_cache=object(),
        kpe_cache=object(),
        out=object(),
        lse=object(),
        return_lse=True,
        profiler_buffer=object(),
        kv_len=None,
        page_table=None,
        return_lse_base_on_e=True,
        o_scale=None,
        ckv_scale=None,
        kpe_scale=None,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        bmm1_scale=None,
        bmm2_scale=None,
    )
    kwargs.update(overrides)
    return kwargs


def _generated_fa_backend_without_init(backend_cls):
    backend = object.__new__(backend_cls)
    backend._generated_fa_workspace = _fa_common._BatchMLAGeneratedFaWorkspace(
        torch.device("cpu")
    )
    return backend


def test_selected_backend_run_from_wrapper_receives_complete_superset_once():
    calls = []
    expected = object()

    class _SelectedBackend:
        def run(self, **kwargs):
            raise AssertionError(f"wrapper called narrow run directly with {kwargs}")

        def run_from_wrapper(self, **kwargs):
            calls.append(kwargs)
            return expected

    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa3")
    backend = _SelectedBackend()
    wrapper._backend_impl = backend
    wrapper._selected_backend = "fa3"
    expected_kwargs = _run_from_wrapper_kwargs()
    run_kwargs = dict(expected_kwargs)

    result = wrapper.run(
        run_kwargs.pop("q_nope"),
        run_kwargs.pop("q_pe"),
        run_kwargs.pop("ckv_cache"),
        run_kwargs.pop("kpe_cache"),
        **run_kwargs,
    )

    assert result is expected
    assert calls == [expected_kwargs]
    assert wrapper._backend_impl is backend


@pytest.mark.parametrize(
    "backend_type", tuple(batch_mla_core._WRAPPER_BACKEND_TYPES.values())
)
def test_wrapper_backend_run_from_wrapper_signatures_are_uniform_and_non_variadic(
    backend_type,
):
    signature = inspect.signature(backend_type.run_from_wrapper)

    assert tuple(signature.parameters) == _RUN_FROM_WRAPPER_PARAMETERS
    assert signature.parameters["self"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(signature.parameters.values())[1:]
    )
    assert all(
        parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == _RUN_FROM_WRAPPER_RETURN


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"sinks": object()}, "sinks are not supported"),
        (
            {"skip_softmax_threshold_scale_factor": 1.0},
            "skip_softmax_threshold_scale_factor is not supported",
        ),
        ({"bmm1_scale": 1.0}, "bmm1_scale is not supported"),
        ({"bmm2_scale": 1.0}, "bmm2_scale is not supported"),
        ({"kv_len": object()}, "kv_len is only supported with cutlass"),
        ({"page_table": object()}, "page_table is only supported with cutlass"),
        ({"o_scale": 1.0}, "o_scale is only supported with the cutlass"),
        ({"ckv_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
        ({"kpe_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
    ),
)
def test_fa2_run_from_wrapper_rejects_unsupported_options_before_run(
    monkeypatch, overrides, match
):
    backend = _generated_fa_backend_without_init(
        fa2_backend._BatchMLAPagedAttentionFa2Backend
    )
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(**_run_from_wrapper_kwargs(**overrides))

    assert calls == []


def test_fa3_run_from_wrapper_forwards_per_tensor_scales_to_run(monkeypatch):
    backend = _generated_fa_backend_without_init(
        fa3_backend._BatchMLAPagedAttentionFa3Backend
    )
    calls = []
    expected = object()
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: calls.append(kwargs) or expected,
    )
    kwargs = _run_from_wrapper_kwargs(ckv_scale=0.25, kpe_scale=0.5)

    result = backend.run_from_wrapper(**kwargs)

    assert result is expected
    assert calls == [
        {
            key: kwargs[key]
            for key in (
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "out",
                "lse",
                "return_lse",
                "profiler_buffer",
                "return_lse_base_on_e",
                "ckv_scale",
                "kpe_scale",
            )
        }
    ]


@pytest.mark.parametrize("scale_name", ("bmm1_scale", "bmm2_scale"))
def test_fa3_run_from_wrapper_rejects_fused_bmm_scales_before_run(
    monkeypatch, scale_name
):
    backend = _generated_fa_backend_without_init(
        fa3_backend._BatchMLAPagedAttentionFa3Backend
    )
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=f"{scale_name} is not supported"):
        backend.run_from_wrapper(**_run_from_wrapper_kwargs(**{scale_name: 1.0}))

    assert calls == []


def _valid_cutlass_run_from_wrapper_kwargs(**overrides):
    kwargs = _run_from_wrapper_kwargs(
        lse=None,
        return_lse=False,
        profiler_buffer=None,
        return_lse_base_on_e=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_cutlass_run_unplanned_from_wrapper_has_explicit_signature():
    backend_type = cutlass_backend._BatchMLAPagedAttentionCutlassBackend

    unplanned_signature = inspect.signature(backend_type.run_unplanned_from_wrapper)
    assert tuple(unplanned_signature.parameters) == (
        "float_workspace_buffer",
        *_RUN_FROM_WRAPPER_PARAMETERS[1:],
    )
    assert (
        unplanned_signature.parameters["float_workspace_buffer"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(unplanned_signature.parameters.values())[1:]
    )


def test_cutlass_run_from_wrapper_forwards_only_narrow_run_inputs(monkeypatch):
    backend = object.__new__(cutlass_backend._BatchMLAPagedAttentionCutlassBackend)
    calls = []
    expected = object()
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: calls.append(kwargs) or expected,
    )
    kwargs = _valid_cutlass_run_from_wrapper_kwargs()

    result = backend.run_from_wrapper(**kwargs)

    assert result is expected
    assert calls == [
        {
            key: kwargs[key]
            for key in (
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "out",
                "kv_len",
                "page_table",
                "o_scale",
            )
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"sinks": object()}, "sinks are not supported"),
        (
            {"skip_softmax_threshold_scale_factor": 1.0},
            "skip_softmax_threshold_scale_factor is not supported",
        ),
        ({"bmm1_scale": 1.0}, "bmm1_scale is not supported"),
        ({"bmm2_scale": 1.0}, "bmm2_scale is not supported"),
        ({"return_lse": True}, "return_lse is not supported"),
        ({"lse": object()}, "lse is not supported"),
        ({"profiler_buffer": object()}, "profiler_buffer is not supported"),
        (
            {"return_lse_base_on_e": True},
            "return_lse_base_on_e is not supported",
        ),
        ({"ckv_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
        ({"kpe_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
    ),
)
def test_cutlass_run_from_wrapper_rejects_inapplicable_options_before_run(
    monkeypatch, overrides, match
):
    backend = object.__new__(cutlass_backend._BatchMLAPagedAttentionCutlassBackend)
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(**_valid_cutlass_run_from_wrapper_kwargs(**overrides))

    assert calls == []


def test_cutlass_run_from_wrapper_preserves_option_validation_order(monkeypatch):
    backend = object.__new__(cutlass_backend._BatchMLAPagedAttentionCutlassBackend)
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("ran backend")),
    )

    with pytest.raises(ValueError, match="sinks are not supported"):
        backend.run_from_wrapper(**_run_from_wrapper_kwargs(sinks=object()))


def test_cutlass_run_without_plan_plans_and_runs_narrowly_once():
    workspace = object()
    calls = []
    q_nope = torch.empty((2, 128, 512), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 128, 64), dtype=torch.bfloat16)
    ckv_cache = torch.empty((8, 32, 512), dtype=torch.bfloat16)
    kpe_cache = torch.empty((8, 32, 64), dtype=torch.bfloat16)
    expected = object()

    class _InspectableBackend(cutlass_backend._BatchMLAPagedAttentionCutlassBackend):
        @staticmethod
        def _validate_wrapper_run_options(**kwargs):
            calls.append(("validate", kwargs))
            return cutlass_backend._BatchMLAPagedAttentionCutlassBackend._validate_wrapper_run_options(
                **kwargs
            )

        def __init__(self, float_workspace_buffer):
            calls.append(("construct", float_workspace_buffer))

        def plan(self, **kwargs):
            calls.append(("plan", kwargs))

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            return expected

    kwargs = _valid_cutlass_run_from_wrapper_kwargs(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
    )

    result = _InspectableBackend.run_unplanned_from_wrapper(workspace, **kwargs)

    assert result is expected
    assert [name for name, _ in calls] == ["validate", "construct", "plan", "run"]
    assert calls[2][1] == {
        "num_heads": 128,
        "head_dim_ckv": 512,
        "head_dim_kpe": 64,
        "page_size": 32,
        "causal": False,
        "sm_scale": 1.0 / (128 + 64) ** 0.5,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "use_profiler": False,
        "batch_size": 2,
        "kv_len": None,
        "page_table": None,
    }
    assert calls[3][1] == {
        key: kwargs[key]
        for key in (
            "q_nope",
            "q_pe",
            "ckv_cache",
            "kpe_cache",
            "out",
            "kv_len",
            "page_table",
            "o_scale",
        )
    }


def test_cutlass_run_without_plan_rejects_before_construction():
    constructions = []

    class _InspectableBackend(cutlass_backend._BatchMLAPagedAttentionCutlassBackend):
        def __init__(self, float_workspace_buffer):
            constructions.append(float_workspace_buffer)

    with pytest.raises(ValueError, match="sinks are not supported"):
        _InspectableBackend.run_unplanned_from_wrapper(
            object(),
            **_valid_cutlass_run_from_wrapper_kwargs(sinks=object()),
        )

    assert constructions == []


def test_cutlass_wrapper_run_without_plan_delegates_without_persisting(monkeypatch):
    calls = []
    expected = object()

    def run_unplanned(cls, float_workspace_buffer, **kwargs):
        calls.append((cls, float_workspace_buffer, kwargs))
        return expected

    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "run_unplanned_from_wrapper",
        classmethod(run_unplanned),
        raising=False,
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")
    kwargs = _valid_cutlass_run_from_wrapper_kwargs()
    expected_kwargs = dict(kwargs)

    result = wrapper.run(
        kwargs.pop("q_nope"),
        kwargs.pop("q_pe"),
        kwargs.pop("ckv_cache"),
        kwargs.pop("kpe_cache"),
        **kwargs,
    )

    assert result is expected
    assert calls == [
        (
            cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
            wrapper._float_workspace_buffer,
            expected_kwargs,
        )
    ]
    assert wrapper._backend_impl is None
    assert wrapper._selected_backend is None


def _valid_trtllm_gen_run_from_wrapper_kwargs(**overrides):
    kwargs = _run_from_wrapper_kwargs(
        profiler_buffer=None,
        return_lse_base_on_e=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_trtllm_gen_run_from_wrapper_forwards_only_narrow_run_inputs(monkeypatch):
    backend = object.__new__(trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend)
    calls = []
    expected = object()
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: calls.append(kwargs) or expected,
    )
    kwargs = _valid_trtllm_gen_run_from_wrapper_kwargs(
        sinks=object(),
        skip_softmax_threshold_scale_factor=0.75,
        bmm1_scale=0.25,
        bmm2_scale=0.5,
    )

    result = backend.run_from_wrapper(**kwargs)

    assert result is expected
    assert calls == [
        {
            key: kwargs[key]
            for key in (
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "out",
                "lse",
                "return_lse",
                "sinks",
                "skip_softmax_threshold_scale_factor",
                "bmm1_scale",
                "bmm2_scale",
            )
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"profiler_buffer": object()}, "profiler_buffer is not supported"),
        ({"kv_len": object()}, "kv_len is not supported"),
        ({"page_table": object()}, "page_table is not supported"),
        ({"return_lse_base_on_e": True}, "return_lse_base_on_e is not supported"),
        ({"o_scale": 1.0}, "o_scale is not supported"),
        ({"ckv_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
        ({"kpe_scale": 1.0}, "ckv_scale / kpe_scale are only supported"),
    ),
)
def test_trtllm_gen_run_from_wrapper_rejects_inapplicable_options_before_run(
    monkeypatch, overrides, match
):
    backend = object.__new__(trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend)
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(
            **_valid_trtllm_gen_run_from_wrapper_kwargs(**overrides)
        )

    assert calls == []


def _valid_cute_dsl_run_from_wrapper_kwargs(**overrides):
    kwargs = _run_from_wrapper_kwargs(
        profiler_buffer=None,
        return_lse_base_on_e=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_cute_dsl_run_from_wrapper_forwards_only_narrow_run_inputs(monkeypatch):
    backend = object.__new__(cute_dsl_backend._BatchMLAPagedAttentionCuteDslBackend)
    calls = []
    expected = object()
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: calls.append(kwargs) or expected,
    )
    kwargs = _valid_cute_dsl_run_from_wrapper_kwargs(
        sinks=object(),
        bmm1_scale=0.25,
        bmm2_scale=0.5,
    )

    result = backend.run_from_wrapper(**kwargs)

    assert result is expected
    assert calls == [
        {
            key: kwargs[key]
            for key in (
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "out",
                "lse",
                "return_lse",
                "sinks",
                "bmm1_scale",
                "bmm2_scale",
            )
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"profiler_buffer": object()}, "profiler_buffer is not supported"),
        ({"kv_len": object()}, "kv_len and page_table are not supported"),
        ({"page_table": object()}, "kv_len and page_table are not supported"),
        ({"return_lse_base_on_e": True}, "return_lse_base_on_e is not supported"),
        ({"o_scale": 1.0}, "o_scale is not supported"),
        ({"ckv_scale": 1.0}, "ckv_scale / kpe_scale are not supported"),
        ({"kpe_scale": 1.0}, "ckv_scale / kpe_scale are not supported"),
        (
            {"skip_softmax_threshold_scale_factor": 1.0},
            "skip_softmax_threshold_scale_factor is not supported",
        ),
    ),
)
def test_cute_dsl_run_from_wrapper_rejects_inapplicable_options_before_run(
    monkeypatch, overrides, match
):
    backend = object.__new__(cute_dsl_backend._BatchMLAPagedAttentionCuteDslBackend)
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(**_valid_cute_dsl_run_from_wrapper_kwargs(**overrides))

    assert calls == []


def _valid_xqa_run_from_wrapper_kwargs(**overrides):
    kwargs = _run_from_wrapper_kwargs(
        lse=None,
        return_lse=False,
        profiler_buffer=None,
        return_lse_base_on_e=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_xqa_run_from_wrapper_forwards_only_narrow_run_inputs(monkeypatch):
    backend = object.__new__(xqa_backend._BatchMLAPagedAttentionXqaBackend)
    calls = []
    expected = object()
    monkeypatch.setattr(
        backend,
        "run",
        lambda **kwargs: calls.append(kwargs) or expected,
    )
    kwargs = _valid_xqa_run_from_wrapper_kwargs(
        bmm1_scale=0.25,
        bmm2_scale=0.5,
    )

    result = backend.run_from_wrapper(**kwargs)

    assert result is expected
    assert calls == [
        {
            key: kwargs[key]
            for key in (
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "out",
            )
        }
        | {
            "lse": None,
            "return_lse": False,
            "bmm1_scale": kwargs["bmm1_scale"],
            "bmm2_scale": kwargs["bmm2_scale"],
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"return_lse": True}, "does not support LSE output"),
        ({"lse": object()}, "does not support LSE output"),
        ({"profiler_buffer": object()}, "profiler_buffer is not supported"),
        ({"kv_len": object()}, "kv_len and page_table are not supported"),
        ({"page_table": object()}, "kv_len and page_table are not supported"),
        ({"return_lse_base_on_e": True}, "return_lse_base_on_e is not supported"),
        ({"o_scale": 1.0}, "o_scale is not supported"),
        ({"ckv_scale": 1.0}, "ckv_scale / kpe_scale are not supported"),
        ({"kpe_scale": 1.0}, "ckv_scale / kpe_scale are not supported"),
        ({"sinks": object()}, "sinks are not supported"),
        (
            {"skip_softmax_threshold_scale_factor": 1.0},
            "skip_softmax_threshold_scale_factor is not supported",
        ),
        ({"bmm1_scale": torch.tensor(1.0)}, "tensor scales are not supported"),
        ({"bmm2_scale": torch.tensor(1.0)}, "tensor scales are not supported"),
    ),
)
def test_xqa_run_from_wrapper_rejects_inapplicable_options_before_run(
    monkeypatch, overrides, match
):
    backend = object.__new__(xqa_backend._BatchMLAPagedAttentionXqaBackend)
    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(**_valid_xqa_run_from_wrapper_kwargs(**overrides))

    assert calls == []


@pytest.mark.parametrize(
    ("backend_type", "kwargs", "match"),
    (
        (
            trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend,
            _valid_trtllm_gen_run_from_wrapper_kwargs(
                profiler_buffer=object(), kv_len=object()
            ),
            "profiler_buffer is not supported",
        ),
        (
            cute_dsl_backend._BatchMLAPagedAttentionCuteDslBackend,
            _valid_cute_dsl_run_from_wrapper_kwargs(
                profiler_buffer=object(), kv_len=object()
            ),
            "profiler_buffer is not supported",
        ),
        (
            xqa_backend._BatchMLAPagedAttentionXqaBackend,
            _valid_xqa_run_from_wrapper_kwargs(
                return_lse=True, profiler_buffer=object()
            ),
            "does not support LSE output",
        ),
    ),
)
def test_trtllm_gen_run_from_wrapper_and_cute_dsl_run_from_wrapper_and_xqa_run_from_wrapper_preserve_validation_order(
    monkeypatch, backend_type, kwargs, match
):
    backend = object.__new__(backend_type)
    monkeypatch.setattr(
        backend,
        "run",
        lambda **run_kwargs: (_ for _ in ()).throw(
            AssertionError(f"ran backend with {run_kwargs}")
        ),
    )

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(**kwargs)


def test_functional_sparse_ignores_legacy_compatibility_knobs(monkeypatch) -> None:
    expected = object()
    forwarded = []

    def recording_impl(**kwargs):
        forwarded.append(kwargs)
        return expected

    monkeypatch.setattr(mla_core, "_run_mla_decode_sparse_sm120", recording_impl)

    result = mla_core._run_mla_decode_sparse(
        query=object(),
        kv_cache=object(),
        workspace_buffer=object(),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=object(),
        seq_lens=object(),
        max_seq_len=1,
        backend="sparse",
        cute_dsl_impl="monolithic",
        enable_pdl=True,
        is_var_seq=False,
        cum_seq_lens_q=object(),
        max_q_len=7,
    )

    assert result is expected
    assert (
        not {
            "cute_dsl_impl",
            "enable_pdl",
            "is_var_seq",
            "cum_seq_lens_q",
            "max_q_len",
        }
        & forwarded[0].keys()
    )


def _common_plan_args(**overrides):
    args = dict(
        num_heads=1,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=32,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    args.update(overrides)
    return args


def _csr_plan_args(**overrides):
    args = dict(
        qo_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indices=torch.tensor([0, 7, 5], dtype=torch.int32),
        kv_len_arr=torch.tensor([33, 1], dtype=torch.int32),
        **_common_plan_args(),
    )
    args.update(overrides)
    return args


def _dense_plan_args(**overrides):
    args = dict(
        cum_seq_lens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        block_tables=torch.tensor([[0, 7, 91, 92], [5, 93, 94, 95]], dtype=torch.int32),
        seq_lens=torch.tensor([33, 1], dtype=torch.int32),
        max_q_len=2,
        **_common_plan_args(),
    )
    args.update(overrides)
    return args


def _cutlass_plan_args():
    common = _common_plan_args(
        num_heads=128,
        sm_scale=1.0 / (128 + 64) ** 0.5,
    )
    return {
        **_dense_plan_args(
            cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            max_q_len=1,
        ),
        **common,
    }


def _xqa_plan_args(metadata_form):
    common = _common_plan_args(
        num_heads=128,
        sm_scale=1.0 / (128 + 64) ** 0.5,
    )
    if metadata_form == "dense":
        return {
            **_dense_plan_args(
                cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                max_q_len=1,
            ),
            **common,
        }
    return _csr_plan_args(
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        **common,
    )


def _patch_backend(monkeypatch, backend_name):
    backend = _RecordingBackend(backend_name)
    if backend_name == "fa2":

        class _Fa2Backend(fa2_backend._BatchMLAPagedAttentionFa2Backend):
            def __new__(cls, *args):
                return backend

        backend_factory = _Fa2Backend
    else:

        class _TrtllmGenBackend(
            trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend
        ):
            def __new__(cls, *args):
                return backend

        backend_factory = _TrtllmGenBackend
    _patch_plan_from_wrapper_owner(monkeypatch, backend_name, backend_factory)
    return backend


def _patch_plan_from_wrapper_owner(monkeypatch, backend_name, owner):
    backend_type = batch_mla_core._WRAPPER_BACKEND_TYPES[backend_name]
    plan_from_wrapper = backend_type.__dict__["plan_from_wrapper"].__func__

    def dispatch(cls, args):
        assert cls is backend_type
        return plan_from_wrapper(owner, args)

    monkeypatch.setattr(
        backend_type,
        "plan_from_wrapper",
        classmethod(dispatch),
    )


def _capture_plan_request(monkeypatch, *, requested_backend="fa3", plan_kwargs=None):
    captured = {}
    backend = _RecordingBackend(requested_backend)

    backend_type = batch_mla_core._WRAPPER_BACKEND_TYPES[requested_backend]

    def capture(cls, args):
        assert cls is backend_type
        captured["args"] = args
        return batch_mla_planning._MLAWrapperPlanResult(backend)

    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), backend=requested_backend
    )
    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(
            backend_type,
            "plan_from_wrapper",
            classmethod(capture),
        )
        wrapper.plan(**(_csr_plan_args() if plan_kwargs is None else plan_kwargs))
    return captured["args"]


def _assert_generated_fa_plan_from_wrapper(monkeypatch, *, backend_name, backend_cls):
    request = _capture_plan_request(
        monkeypatch,
        requested_backend=backend_name,
    )
    backends = []

    class _InspectableBackend(backend_cls):
        def __init__(self, *args):
            super().__init__(*args)
            self.constructor_args = args
            self.plan_kwargs = None
            backends.append(self)

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

    result = _InspectableBackend.plan_from_wrapper(request)

    assert len(backends) == 1
    backend = backends[0]
    assert result.backend_impl is backend
    expected_resources = (
        request._float_workspace_buffer,
        request._generated_fa_workspace,
        request._use_cuda_graph,
        request._qo_indptr_buf,
        request._kv_indptr_buf,
        request._kv_indices_buf,
        request._kv_len_arr_buf,
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            backend.constructor_args, expected_resources, strict=True
        )
    )
    assert backend.plan_kwargs == {
        "qo_indptr": request.csr.qo_indptr,
        "kv_indptr": request.csr.kv_indptr,
        "kv_indices": request.csr.kv_indices,
        "kv_len_arr": request.csr.kv_len_arr,
        "num_heads": request.num_heads,
        "head_dim_ckv": request.head_dim_ckv,
        "head_dim_kpe": request.head_dim_kpe,
        "page_size": request.page_size,
        "causal": request.causal,
        "sm_scale": request.sm_scale,
        "q_data_type": request.q_data_type,
        "kv_data_type": request.kv_data_type,
        "use_profiler": request.use_profiler,
    }


def test_fa2_plan_from_wrapper_owns_generated_fa_adapter(monkeypatch):
    _assert_generated_fa_plan_from_wrapper(
        monkeypatch,
        backend_name="fa2",
        backend_cls=fa2_backend._BatchMLAPagedAttentionFa2Backend,
    )


def test_fa3_plan_from_wrapper_owns_generated_fa_adapter(monkeypatch):
    _assert_generated_fa_plan_from_wrapper(
        monkeypatch,
        backend_name="fa3",
        backend_cls=fa3_backend._BatchMLAPagedAttentionFa3Backend,
    )


def _record_cutlass_plan_from_wrapper(args):
    backends = []

    class _InspectableBackend(cutlass_backend._BatchMLAPagedAttentionCutlassBackend):
        def __init__(self, float_workspace_buffer):
            super().__init__(float_workspace_buffer)
            self.plan_kwargs = None
            backends.append(self)

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

    result = _InspectableBackend.plan_from_wrapper(args)
    assert len(backends) == 1
    return result, backends[0]


def test_cutlass_plan_from_wrapper_adapts_canonical_dense(monkeypatch):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="cutlass",
        plan_kwargs=_cutlass_plan_args(),
    )
    dense_calls = []
    original_dense = batch_mla_planning._MLAPlanArguments.dense

    def recording_dense(self, *, table_width_alignment):
        dense = original_dense(self, table_width_alignment=table_width_alignment)
        dense_calls.append((table_width_alignment, dense))
        return dense

    monkeypatch.setattr(batch_mla_planning._MLAPlanArguments, "dense", recording_dense)

    result, backend = _record_cutlass_plan_from_wrapper(args)
    dense = dense_calls[0][1]

    assert [alignment for alignment, _ in dense_calls] == [128 // args.page_size]
    assert backend._float_workspace_buffer is args._float_workspace_buffer
    assert backend.plan_kwargs == {
        "num_heads": args.num_heads,
        "head_dim_ckv": args.head_dim_ckv,
        "head_dim_kpe": args.head_dim_kpe,
        "page_size": args.page_size,
        "causal": args.causal,
        "sm_scale": args.sm_scale,
        "q_data_type": args.q_data_type,
        "kv_data_type": args.kv_data_type,
        "use_profiler": args.use_profiler,
        "batch_size": 2,
        "kv_len": dense.seq_lens,
        "page_table": dense.block_tables,
    }
    assert result.backend_impl is backend


def test_cutlass_plan_from_wrapper_materializes_csr_as_dense_once(monkeypatch):
    args = _capture_plan_request(monkeypatch, requested_backend="cutlass")
    calls = []
    original_derive_dense = batch_mla_planning._derive_dense_from_csr

    def recording_derive_dense(*args, **kwargs):
        calls.append(kwargs["table_width_alignment"])
        return original_derive_dense(*args, **kwargs)

    monkeypatch.setattr(
        batch_mla_planning,
        "_derive_dense_from_csr",
        recording_derive_dense,
    )

    result, backend = _record_cutlass_plan_from_wrapper(args)
    dense = args.dense(table_width_alignment=128 // args.page_size)

    assert calls == [128 // args.page_size]
    assert result.backend_impl is backend
    assert backend.plan_kwargs["kv_len"] is dense.seq_lens
    assert backend.plan_kwargs["page_table"] is dense.block_tables


def test_cutlass_plan_from_wrapper_accepts_csr_only_without_selection_history(
    monkeypatch,
):
    csr_request = _capture_plan_request(monkeypatch, requested_backend="cutlass")
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("constructed CUTLASS backend")
        ),
    )
    monkeypatch.setattr(
        cutlass_backend,
        "get_mla_module",
        lambda: (_ for _ in ()).throw(AssertionError("loaded CUTLASS module")),
    )

    with pytest.raises(AssertionError, match="constructed CUTLASS backend"):
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend.plan_from_wrapper(
            csr_request
        )


def _record_explicit_dense_plan_from_wrapper(args, backend_cls):
    backends = []

    class _InspectableBackend(backend_cls):
        def __init__(self, float_workspace_buffer):
            super().__init__(float_workspace_buffer)
            self.plan_kwargs = None
            backends.append(self)

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

    result = _InspectableBackend.plan_from_wrapper(args)

    assert len(backends) == 1
    backend = backends[0]
    assert backend._float_workspace_buffer is args._float_workspace_buffer
    assert result.backend_impl is backend
    assert all(value is not args for value in vars(backend).values())
    assert all(value is not args for value in vars(result).values())
    return result, backend


def _assert_explicit_dense_plan_rejects_before_construction(
    args, backend_cls, *, match
):
    constructions = []

    class _InspectableBackend(backend_cls):
        def __init__(self, *constructor_args):
            constructions.append(constructor_args)
            super().__init__(*constructor_args)

    with pytest.raises(ValueError, match=match) as exc_info:
        _InspectableBackend.plan_from_wrapper(args)

    assert type(exc_info.value) is ValueError
    assert constructions == []


def test_trtllm_gen_plan_from_wrapper_owns_native_dense_adapter(monkeypatch):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="trtllm-gen",
        plan_kwargs=_csr_plan_args(
            qk_nope_head_dim=128,
            enable_pdl=True,
            is_var_seq=True,
            use_sinks=True,
        ),
    )

    result, backend = _record_explicit_dense_plan_from_wrapper(
        args,
        trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend,
    )
    dense = args.native_dense

    native_width = int((args.csr.kv_indptr[1:] - args.csr.kv_indptr[:-1]).max().item())
    assert dense.block_tables.shape[1] == native_width
    assert backend.plan_kwargs == {
        "cum_seq_lens_q": dense.cum_seq_lens_q,
        "block_tables": dense.block_tables,
        "seq_lens": dense.seq_lens,
        "max_q_len": dense.max_q_len,
        "num_heads": args.num_heads,
        "head_dim_ckv": args.head_dim_ckv,
        "head_dim_kpe": args.head_dim_kpe,
        "page_size": args.page_size,
        "causal": args.causal,
        "sm_scale": args.sm_scale,
        "q_data_type": args.q_data_type,
        "kv_data_type": args.kv_data_type,
        "use_profiler": args.use_profiler,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "enable_pdl": args.enable_pdl,
        "is_var_seq": args.is_var_seq,
        "use_sinks": args.use_sinks,
    }


def test_trtllm_gen_plan_from_wrapper_rejects_inapplicable_options_first(
    monkeypatch,
):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="trtllm-gen",
        plan_kwargs=_csr_plan_args(cute_dsl_impl="monolithic"),
    )

    _assert_explicit_dense_plan_rejects_before_construction(
        args,
        trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend,
        match="cute_dsl_impl is not supported",
    )


def test_cute_dsl_plan_from_wrapper_owns_aligned_dense_adapter(monkeypatch):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="cute-dsl",
        plan_kwargs=_csr_plan_args(
            is_var_seq=True,
            cute_dsl_impl="modular",
            use_sinks=True,
        ),
    )

    result, backend = _record_explicit_dense_plan_from_wrapper(
        args,
        cute_dsl_backend._BatchMLAPagedAttentionCuteDslBackend,
    )
    dense = args.dense(table_width_alignment=128 // args.page_size)

    assert dense.block_tables.shape[1] % (128 // args.page_size) == 0
    assert backend.plan_kwargs == {
        "cum_seq_lens_q": dense.cum_seq_lens_q,
        "block_tables": dense.block_tables,
        "seq_lens": dense.seq_lens,
        "max_q_len": dense.max_q_len,
        "num_heads": args.num_heads,
        "head_dim_ckv": args.head_dim_ckv,
        "head_dim_kpe": args.head_dim_kpe,
        "page_size": args.page_size,
        "causal": args.causal,
        "sm_scale": args.sm_scale,
        "q_data_type": args.q_data_type,
        "kv_data_type": args.kv_data_type,
        "use_profiler": args.use_profiler,
        "is_var_seq": args.is_var_seq,
        "cute_dsl_impl": args.cute_dsl_impl,
        "use_sinks": args.use_sinks,
    }


def test_cute_dsl_plan_from_wrapper_rejects_inapplicable_options_first(
    monkeypatch,
):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="cute-dsl",
        plan_kwargs=_csr_plan_args(enable_pdl=True),
    )

    _assert_explicit_dense_plan_rejects_before_construction(
        args,
        cute_dsl_backend._BatchMLAPagedAttentionCuteDslBackend,
        match="enable_pdl is not supported",
    )


def test_xqa_plan_from_wrapper_owns_aligned_dense_adapter(monkeypatch):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="xqa",
        plan_kwargs=_xqa_plan_args("csr") | {"enable_pdl": True},
    )

    result, backend = _record_explicit_dense_plan_from_wrapper(
        args,
        xqa_backend._BatchMLAPagedAttentionXqaBackend,
    )
    dense = args.dense(table_width_alignment=128 // args.page_size)

    assert dense.block_tables.shape[1] % (128 // args.page_size) == 0
    assert backend.plan_kwargs == {
        "cum_seq_lens_q": dense.cum_seq_lens_q,
        "block_tables": dense.block_tables,
        "seq_lens": dense.seq_lens,
        "max_q_len": dense.max_q_len,
        "num_heads": args.num_heads,
        "head_dim_ckv": args.head_dim_ckv,
        "head_dim_kpe": args.head_dim_kpe,
        "page_size": args.page_size,
        "causal": args.causal,
        "sm_scale": args.sm_scale,
        "q_data_type": args.q_data_type,
        "kv_data_type": args.kv_data_type,
        "use_profiler": args.use_profiler,
        "enable_pdl": args.enable_pdl,
    }


def test_xqa_plan_from_wrapper_rejects_inapplicable_options_first(monkeypatch):
    args = _capture_plan_request(
        monkeypatch,
        requested_backend="xqa",
        plan_kwargs=_xqa_plan_args("csr") | {"use_profiler": True},
    )

    _assert_explicit_dense_plan_rejects_before_construction(
        args,
        xqa_backend._BatchMLAPagedAttentionXqaBackend,
        match="use_profiler is not supported",
    )


def test_plan_backend_commits_complete_result_only_after_success(monkeypatch):
    request = _capture_plan_request(monkeypatch, requested_backend="fa2")
    backend_type = fa2_backend._BatchMLAPagedAttentionFa2Backend
    successful_backend = _RecordingBackend("fa2")

    def accept(cls, args):
        assert cls is backend_type
        assert args is request
        return batch_mla_planning._MLAWrapperPlanResult(successful_backend)

    monkeypatch.setattr(backend_type, "plan_from_wrapper", classmethod(accept))
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper._plan_backend("fa2", request)

    successful_state = (successful_backend, "fa2")
    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
    ) == successful_state

    def reject(cls, args):
        assert cls is backend_type
        assert args is request
        raise backend_errors._BackendPlanUnsupportedError("fa2 unavailable")

    monkeypatch.setattr(backend_type, "plan_from_wrapper", classmethod(reject))
    with pytest.raises(
        backend_errors._BackendPlanUnsupportedError, match="fa2 unavailable"
    ):
        wrapper._plan_backend("fa2", request)

    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
    ) == successful_state


def test_wrapper_does_not_retain_duplicate_plan_metadata():
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")

    assert not hasattr(wrapper, "_csr_plan_metadata")
    assert not hasattr(wrapper, "_dense_plan_metadata")


def test_plan_arguments_do_not_expose_result_only_metadata_observers():
    assert not hasattr(batch_mla_planning._MLAPlanArguments, "resolved_csr")
    assert not hasattr(batch_mla_planning._MLAPlanArguments, "resolved_dense")


def test_explicit_and_auto_plan_route_through_plan_backend(monkeypatch):
    calls = []

    def plan_backend(self, backend, args):
        calls.append((self._backend, backend, args))
        if self._backend == "auto" and backend == "fa3":
            raise backend_errors._BackendPlanUnsupportedError("fa3 unavailable")
        self._selected_backend = backend

    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_backend",
        plan_backend,
        raising=False,
    )
    direct_result = batch_mla_planning._MLAWrapperPlanResult(
        _RecordingBackend("direct")
    )
    for backend_type in (
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        fa3_backend._BatchMLAPagedAttentionFa3Backend,
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
    ):
        monkeypatch.setattr(
            backend_type,
            "plan_from_wrapper",
            classmethod(lambda cls, args: direct_result),
        )
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda _: "fa3")

    explicit = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    explicit.plan(**_csr_plan_args())
    assert [(requested, candidate) for requested, candidate, _ in calls] == [
        ("fa2", "fa2")
    ]

    automatic = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    automatic.plan(**_csr_plan_args())
    assert [(requested, candidate) for requested, candidate, _ in calls] == [
        ("fa2", "fa2"),
        ("auto", "fa3"),
        ("auto", "cutlass"),
    ]
    assert calls[1][2] is calls[2][2]


def test_backend_class_plan_dispatch_reuses_request_and_commits_success_only(
    monkeypatch,
):
    calls = []
    successful_backend = _RecordingBackend("cutlass")
    pending_request = None
    request_id = None

    def reject_fa3(cls, args):
        nonlocal pending_request, request_id
        assert cls is fa3_backend._BatchMLAPagedAttentionFa3Backend
        calls.append("fa3")
        pending_request = args
        request_id = id(args)
        raise backend_errors._BackendPlanUnsupportedError("fa3 unavailable")

    def accept_cutlass(cls, args):
        nonlocal pending_request
        assert cls is cutlass_backend._BatchMLAPagedAttentionCutlassBackend
        assert args is pending_request
        assert (
            wrapper._backend_impl,
            wrapper._selected_backend,
        ) == initial_state
        calls.append("cutlass")
        pending_request = None
        return batch_mla_planning._MLAWrapperPlanResult(successful_backend)

    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda _: "fa3")
    monkeypatch.setattr(
        fa3_backend._BatchMLAPagedAttentionFa3Backend,
        "plan_from_wrapper",
        classmethod(reject_fa3),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(accept_cutlass),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    initial_state = (object(), "fa2")
    wrapper._backend_impl, wrapper._selected_backend = initial_state

    wrapper.plan(**_csr_plan_args())

    assert calls == ["fa3", "cutlass"]
    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
    ) == (successful_backend, "cutlass")
    gc.collect()
    assert request_id is not None
    assert all(id(obj) != request_id for obj in gc.get_objects())


def test_plan_request_private_resources_are_not_public_or_repr_fields():
    fields = dataclasses.fields(batch_mla_planning._MLAPlanArguments)
    public_names = {field.name for field in fields if not field.name.startswith("_")}
    assert public_names == {
        "qo_indptr",
        "kv_indptr",
        "kv_indices",
        "kv_len_arr",
        "num_heads",
        "head_dim_ckv",
        "head_dim_kpe",
        "page_size",
        "causal",
        "sm_scale",
        "q_data_type",
        "kv_data_type",
        "use_profiler",
        "cum_seq_lens_q",
        "block_tables",
        "seq_lens",
        "max_q_len",
        "qk_nope_head_dim",
        "enable_pdl",
        "is_var_seq",
        "cute_dsl_impl",
        "use_sinks",
    }
    private_fields = [field for field in fields if field.name.startswith("_")]
    assert {field.name for field in private_fields} >= {
        "_float_workspace_buffer",
        "_generated_fa_workspace",
        "_use_cuda_graph",
        "_qo_indptr_buf",
        "_kv_indptr_buf",
        "_kv_indices_buf",
        "_kv_len_arr_buf",
    }
    assert all(field.repr is False for field in private_fields)
    assert "__dict__" not in batch_mla_planning._MLAPlanArguments.__slots__


def test_plan_arguments_audit_reports_unconsumed_public_argument(monkeypatch):
    request = _capture_plan_request(monkeypatch, requested_backend="fa2")
    public_fields = [
        field.name
        for field in dataclasses.fields(batch_mla_planning._MLAPlanArguments)
        if not field.name.startswith("_") and field.name != "use_sinks"
    ]

    with (
        pytest.raises(AssertionError, match="use_sinks"),
        request.audit_public_argument_access("test"),
    ):
        for field_name in public_fields:
            getattr(request, field_name)


def test_plan_arguments_audit_counts_csr_resolution_as_metadata_access(
    monkeypatch,
):
    request = _capture_plan_request(monkeypatch, requested_backend="fa2")
    metadata_fields = {
        "qo_indptr",
        "kv_indptr",
        "kv_indices",
        "kv_len_arr",
        "cum_seq_lens_q",
        "block_tables",
        "seq_lens",
        "max_q_len",
    }

    with request.audit_public_argument_access("test"):
        _ = request.csr
        for field in dataclasses.fields(batch_mla_planning._MLAPlanArguments):
            if not field.name.startswith("_") and field.name not in metadata_fields:
                getattr(request, field.name)


def test_plan_from_wrapper_audit_reports_unconsumed_public_argument(monkeypatch):
    request = _capture_plan_request(monkeypatch, requested_backend="fa2")

    class _Backend:
        pass

    def plan_from_wrapper(cls, args):
        assert cls is _Backend
        _ = args.num_heads

    _Backend.plan_from_wrapper = classmethod(
        batch_mla_planning._audit_plan_from_wrapper_arguments(plan_from_wrapper)
    )

    with pytest.raises(AssertionError, match="use_sinks"):
        _Backend.plan_from_wrapper(request)


@pytest.mark.parametrize(
    ("plan_kwargs", "derived_helper"),
    (
        (_csr_plan_args(), "_derive_dense_from_csr"),
        (_dense_plan_args(), "_derive_csr_from_dense"),
    ),
)
def test_selection_descriptor_does_not_materialize_alternate_metadata(
    monkeypatch, plan_kwargs, derived_helper
):
    request = _capture_plan_request(
        monkeypatch,
        requested_backend="fa2",
        plan_kwargs=plan_kwargs,
    )
    monkeypatch.setattr(
        batch_mla_planning,
        derived_helper,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selection materialized alternate metadata")
        ),
    )

    descriptor = request.selection_descriptor()

    assert descriptor.has_csr is (plan_kwargs.get("qo_indptr") is not None)
    assert descriptor.has_dense is (plan_kwargs.get("block_tables") is not None)


def test_plan_arguments_validate_metadata_forms_at_construction():
    plan_kwargs = _csr_plan_args()
    plan_kwargs["kv_indices"] = None

    with pytest.raises(ValueError, match="CSR metadata form is partial"):
        batch_mla_planning._MLAPlanArguments(
            **plan_kwargs,
            _float_workspace_buffer=torch.empty(1),
            _generated_fa_workspace=_fa_common._BatchMLAGeneratedFaWorkspace(
                torch.device("cpu")
            ),
            _use_cuda_graph=False,
            _qo_indptr_buf=None,
            _kv_indptr_buf=None,
            _kv_indices_buf=None,
            _kv_len_arr_buf=None,
        )


def test_plan_accepts_legacy_positional_csr(monkeypatch):
    backend = _RecordingBackend("fa2")
    received = []

    def plan_fa2(cls, request):
        received.append(request)
        return batch_mla_planning._MLAWrapperPlanResult(backend)

    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(plan_fa2),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    args = _csr_plan_args()

    wrapper.plan(*args.values())

    assert wrapper._selected_backend == "fa2"
    assert received[0].qo_indptr is args["qo_indptr"]
    assert received[0].kv_indices is args["kv_indices"]


def test_plan_accepts_canonical_dense_metadata(monkeypatch):
    backend = _patch_backend(monkeypatch, "trtllm-gen")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), backend="trtllm-gen"
    )
    native_width_table = torch.arange(66, dtype=torch.int32).reshape(2, 33)
    args = _dense_plan_args(
        block_tables=native_width_table,
        qk_nope_head_dim=128,
    )

    wrapper.plan(**args)

    assert wrapper._selected_backend == "trtllm-gen"
    assert backend.plan_kwargs["block_tables"] is args["block_tables"]
    assert backend.plan_kwargs["block_tables"].shape[1] == 33
    assert backend.plan_kwargs["seq_lens"] is args["seq_lens"]


def test_plan_accepts_equivalent_csr_and_dense_metadata(monkeypatch):
    backend = _patch_backend(monkeypatch, "fa2")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")

    wrapper.plan(
        **_csr_plan_args(),
        **{
            key: value
            for key, value in _dense_plan_args().items()
            if key not in _common_plan_args()
        },
    )

    assert wrapper._backend_impl is backend
    assert backend.plan_kwargs["qo_indptr"] is not None


@pytest.mark.parametrize("invalid_form", ("partial", "mixed", "mismatched"))
def test_plan_rejects_invalid_metadata_forms(monkeypatch, invalid_form):
    constructions = []

    class _Fa2Backend(fa2_backend._BatchMLAPagedAttentionFa2Backend):
        def __new__(cls, *args):
            constructions.append(args)
            return _RecordingBackend("fa2")

    _patch_plan_from_wrapper_owner(monkeypatch, "fa2", _Fa2Backend)
    if invalid_form == "partial":
        args = {**_common_plan_args(), "qo_indptr": _csr_plan_args()["qo_indptr"]}
    elif invalid_form == "mixed":
        args = {
            **_common_plan_args(),
            "qo_indptr": _csr_plan_args()["qo_indptr"],
            "block_tables": _dense_plan_args()["block_tables"],
        }
    else:
        args = {
            **_csr_plan_args(),
            **{
                key: value
                for key, value in _dense_plan_args(
                    block_tables=torch.tensor(
                        [[8, 7, 91, 92], [5, 93, 94, 95]], dtype=torch.int32
                    )
                ).items()
                if key not in _common_plan_args()
            },
        }
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")

    with pytest.raises(ValueError, match="partial|missing|required|equivalent"):
        wrapper.plan(**args)

    assert constructions == []


def test_dense_metadata_is_converted_to_live_csr_pages(monkeypatch):
    backend = _patch_backend(monkeypatch, "fa2")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")

    wrapper.plan(**_dense_plan_args())

    assert torch.equal(
        backend.plan_kwargs["kv_indptr"], torch.tensor([0, 2, 3], dtype=torch.int32)
    )
    assert torch.equal(
        backend.plan_kwargs["kv_indices"], torch.tensor([0, 7, 5], dtype=torch.int32)
    )


def test_trtllm_csr_metadata_is_converted_to_native_width_dense_table(monkeypatch):
    backend = _patch_backend(monkeypatch, "trtllm-gen")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), backend="trtllm-gen"
    )

    wrapper.plan(**_csr_plan_args(qk_nope_head_dim=128))

    assert backend.plan_kwargs["block_tables"].shape == (2, 2)
    assert torch.equal(
        backend.plan_kwargs["block_tables"],
        torch.tensor([[0, 7], [5, 0]], dtype=torch.int32),
    )


def test_failed_replan_preserves_last_successful_plan(monkeypatch):
    backend = _patch_backend(monkeypatch, "fa2")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper.plan(**_dense_plan_args())
    original_state = (
        wrapper._backend_impl,
        wrapper._selected_backend,
    )
    monkeypatch.setattr(
        batch_mla_planning,
        "_derive_csr_from_dense",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("conversion failed")),
    )

    with pytest.raises(ValueError, match="conversion failed"):
        wrapper.plan(**_dense_plan_args())

    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
    ) == original_state
    assert wrapper._backend_impl is backend


def test_auto_all_candidates_rejected_preserves_existing_plan(monkeypatch):
    backend = _RecordingBackend("fa2")
    fa2_attempts = 0

    def plan_fa2(cls, request):
        nonlocal fa2_attempts
        fa2_attempts += 1
        if fa2_attempts == 1:
            return batch_mla_planning._MLAWrapperPlanResult(backend)
        raise backend_errors._BackendPlanUnsupportedError("fa2 later unavailable")

    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(plan_fa2),
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_maybe_warn_blackwell_auto_fallback",
        classmethod(lambda cls, device, selected_backend: None),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, request: (_ for _ in ()).throw(
                backend_errors._BackendPlanUnsupportedError("cutlass later unavailable")
            )
        ),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    wrapper.plan(**_csr_plan_args())
    original_state = (
        wrapper._backend_impl,
        wrapper._selected_backend,
    )

    with pytest.raises(backend_errors._BackendPlanUnsupportedError) as exc_info:
        wrapper.plan(**_csr_plan_args())

    message = str(exc_info.value)
    assert "rejected all candidates [fa2, cutlass]" in message
    assert "fa2 later unavailable" in message
    assert "cutlass later unavailable" in message
    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
    ) == original_state


def test_run_does_not_repeat_plan_time_metadata_conversion(monkeypatch):
    backend = _patch_backend(monkeypatch, "fa2")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    wrapper.plan(**_dense_plan_args())
    for name in ("_derive_csr_from_dense", "_derive_dense_from_csr"):
        monkeypatch.setattr(
            batch_mla_planning,
            name,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("run converted metadata")
            ),
        )

    result = wrapper.run(
        torch.empty((3, 1, 512), dtype=torch.bfloat16),
        torch.empty((3, 1, 64), dtype=torch.bfloat16),
        torch.empty((3, 32, 512), dtype=torch.bfloat16),
        torch.empty((3, 32, 64), dtype=torch.bfloat16),
    )

    assert result is backend.result


def test_first_post_plan_run_is_hot(monkeypatch):
    backend = _patch_backend(monkeypatch, "fa2")
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_maybe_warn_blackwell_auto_fallback",
        classmethod(lambda cls, device, selected_backend: None),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    wrapper.plan(**_dense_plan_args())

    def fail(message):
        return lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(message))

    monkeypatch.setattr(
        batch_mla_core, "determine_mla_backend", fail("run selected a backend")
    )
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(fail("run planned a backend")),
    )
    monkeypatch.setattr(
        fa2_backend,
        "_get_batch_mla_fa2_module",
        fail("run loaded an FA2 module"),
    )
    for name in ("_derive_csr_from_dense", "_derive_dense_from_csr"):
        monkeypatch.setattr(batch_mla_planning, name, fail("run converted metadata"))

    result = wrapper.run(
        torch.empty((3, 1, 512), dtype=torch.bfloat16),
        torch.empty((3, 1, 64), dtype=torch.bfloat16),
        torch.empty((3, 32, 512), dtype=torch.bfloat16),
        torch.empty((3, 32, 64), dtype=torch.bfloat16),
    )

    assert result is backend.result


class _FakeCutlassModule:
    def __init__(self):
        self.run_args = None

    def cutlass_mla_paged_attention(self, *args):
        self.run_args = args


def test_cutlass_planned_metadata_allows_runtime_omission(monkeypatch):
    module = _FakeCutlassModule()
    monkeypatch.setattr(
        cutlass_backend, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(cutlass_backend, "get_mla_module", lambda: module)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(4), backend="cutlass")
    plan_args = _cutlass_plan_args()
    wrapper.plan(**plan_args)
    monkeypatch.setattr(
        cutlass_backend,
        "get_mla_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("run reloaded the CUTLASS module")
        ),
    )

    result = wrapper.run(
        torch.empty((2, 128, 512), dtype=torch.bfloat16),
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((8, 32, 512), dtype=torch.bfloat16),
        torch.empty((8, 32, 64), dtype=torch.bfloat16),
    )

    assert result.shape == (2, 128, 512)
    assert module.run_args[5] is plan_args["seq_lens"]
    assert module.run_args[6] is plan_args["block_tables"]


@pytest.mark.parametrize("mismatched_metadata", ("kv_len", "page_table"))
def test_cutlass_planned_metadata_rejects_non_aliasing_runtime_values(
    monkeypatch, mismatched_metadata
):
    monkeypatch.setattr(
        cutlass_backend, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(cutlass_backend, "get_mla_module", _FakeCutlassModule)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(4), backend="cutlass")
    plan_args = _cutlass_plan_args()
    wrapper.plan(**plan_args)
    runtime_metadata = {
        "kv_len": plan_args["seq_lens"],
        "page_table": plan_args["block_tables"],
    }
    runtime_metadata[mismatched_metadata] = runtime_metadata[
        mismatched_metadata
    ].clone()

    with pytest.raises(
        ValueError,
        match=f"same tensor view as planned {mismatched_metadata}",
    ):
        wrapper.run(
            torch.empty((2, 128, 512), dtype=torch.bfloat16),
            torch.empty((2, 128, 64), dtype=torch.bfloat16),
            torch.empty((8, 32, 512), dtype=torch.bfloat16),
            torch.empty((8, 32, 64), dtype=torch.bfloat16),
            **runtime_metadata,
        )


@pytest.mark.parametrize("metadata_form", ("dense", "csr"))
def test_xqa_wrapper_plan_run_adapts_dense_and_csr_metadata(monkeypatch, metadata_form):
    backend = _RecordingBackend("xqa")

    class _XqaBackend(xqa_backend._BatchMLAPagedAttentionXqaBackend):
        def __new__(cls, *args):
            return backend

    _patch_plan_from_wrapper_owner(monkeypatch, "xqa", _XqaBackend)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="xqa")

    wrapper.plan(**_xqa_plan_args(metadata_form))
    result = wrapper.run(
        torch.empty((2, 128, 512), dtype=torch.bfloat16),
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((8, 32, 512), dtype=torch.bfloat16),
        torch.empty((8, 32, 64), dtype=torch.bfloat16),
    )

    assert wrapper._selected_backend == "xqa"
    assert isinstance(backend.plan_kwargs["block_tables"], torch.Tensor)
    assert set(backend.run_kwargs) == set(_RUN_FROM_WRAPPER_PARAMETERS[1:])
    assert result is backend.result


def test_auto_falls_back_in_order_for_supported_rejections(monkeypatch, caplog):
    backend = _RecordingBackend("cutlass")
    calls = []
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa3")
    monkeypatch.setattr(
        fa3_backend._BatchMLAPagedAttentionFa3Backend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, request: calls.append("fa3")
            or (_ for _ in ()).throw(
                backend_errors._BackendPlanUnsupportedError("fa3 unavailable")
            )
        ),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, request: calls.append("cutlass")
            or batch_mla_planning._MLAWrapperPlanResult(backend)
        ),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")

    with caplog.at_level(logging.DEBUG, logger=batch_mla_core.__name__):
        wrapper.plan(**_csr_plan_args())

    assert calls == ["fa3", "cutlass"]
    assert wrapper._selected_backend == "cutlass"
    assert "automatically rejected backend 'fa3': fa3 unavailable" in caplog.text


def test_auto_warns_before_planning_generated_fa_backend(monkeypatch):
    events = []
    backend = _RecordingBackend("fa2")
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, request: events.append("plan")
            or batch_mla_planning._MLAWrapperPlanResult(backend)
        ),
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_maybe_warn_blackwell_auto_fallback",
        classmethod(lambda cls, device, selected_backend: events.append("warning")),
    )

    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    wrapper.plan(**_csr_plan_args())

    assert events == ["warning", "plan"]


def test_plan_backend_does_not_issue_auto_fallback_warning(monkeypatch):
    request = _capture_plan_request(monkeypatch, requested_backend="fa2")
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_maybe_warn_blackwell_auto_fallback",
        classmethod(
            lambda cls, device, selected_backend: (_ for _ in ()).throw(
                AssertionError("_plan_backend issued automatic fallback warning")
            )
        ),
    )
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, args: batch_mla_planning._MLAWrapperPlanResult(
                _RecordingBackend("fa2")
            )
        ),
    )

    wrapper._plan_backend("fa2", request)

    assert wrapper._selected_backend == "fa2"


def test_auto_does_not_fallback_from_unexpected_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda cls, request: calls.append("fa2")
            or (_ for _ in ()).throw(RuntimeError("unexpected failure"))
        ),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda cls, request: calls.append("cutlass")),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")

    with pytest.raises(RuntimeError, match="unexpected failure"):
        wrapper.plan(**_csr_plan_args())

    assert calls == ["fa2"]


def test_fa3_support_rejection_uses_fallback_exception(monkeypatch):
    backend = object.__new__(fa3_backend._BatchMLAPagedAttentionFa3Backend)
    backend.device = torch.device("cpu")
    monkeypatch.setattr(fa3_backend, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fa3_backend,
        "_get_batch_mla_fa3_module",
        lambda *args: (_ for _ in ()).throw(AssertionError("loaded FA3 module")),
    )

    with pytest.raises(
        backend_errors._BackendPlanUnsupportedError, match="requires an SM90"
    ):
        backend.plan(
            **_csr_plan_args(
                kv_data_type=torch.float8_e4m3fn,
                use_profiler=False,
            )
        )
