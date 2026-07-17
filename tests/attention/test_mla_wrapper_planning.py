"""Planning and dispatch behavior for the public batch MLA wrapper."""

import logging

import pytest
import torch

import flashinfer._backend as backend_errors
from flashinfer.mla import _core as mla_core
from flashinfer.mla._batch_mla import _core as batch_mla_core
from flashinfer.mla._batch_mla import _planning as batch_mla_planning
from flashinfer.mla._batch_mla._backends import cutlass_backend
from flashinfer.mla._batch_mla._backends import fa2_backend
from flashinfer.mla._batch_mla._backends import fa3_backend


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
    assert not {
        "cute_dsl_impl",
        "enable_pdl",
        "is_var_seq",
        "cum_seq_lens_q",
        "max_q_len",
    } & forwarded[0].keys()


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
        block_tables=torch.tensor(
            [[0, 7, 91, 92], [5, 93, 94, 95]], dtype=torch.int32
        ),
        seq_lens=torch.tensor([33, 1], dtype=torch.int32),
        max_q_len=2,
        **_common_plan_args(),
    )
    args.update(overrides)
    return args


def _cutlass_plan_args(metadata_form):
    common = _common_plan_args(
        num_heads=128,
        sm_scale=1.0 / (128 + 64) ** 0.5,
    )
    if metadata_form == "canonical":
        return {
            **_dense_plan_args(
                cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                max_q_len=1,
            ),
            **common,
        }
    csr = _csr_plan_args(
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        **common,
    )
    return {
        **csr,
        "kv_len": csr["kv_len_arr"],
        "page_table": torch.tensor(
            [[0, 7, 91, 92], [5, 93, 94, 95]], dtype=torch.int32
        ),
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
    class_name = {
        "fa2": "_BatchMLAPagedAttentionFa2Backend",
        "trtllm-gen": "_BatchMLAPagedAttentionTrtllmGenBackend",
    }[backend_name]
    monkeypatch.setattr(batch_mla_core, class_name, lambda *args: backend)
    return backend


def test_plan_accepts_legacy_positional_csr(monkeypatch):
    backend = _RecordingBackend("fa2")
    received = []
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_fa2",
        lambda self, request: received.append(request)
        or batch_mla_planning._MLAWrapperPlanResult(backend, request.csr, None),
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

    wrapper.plan(**_csr_plan_args(), **{
        key: value
        for key, value in _dense_plan_args().items()
        if key not in _common_plan_args()
    })

    assert wrapper._backend_impl is backend
    assert wrapper._csr_plan_metadata.qo_indptr is backend.plan_kwargs["qo_indptr"]


@pytest.mark.parametrize("invalid_form", ("partial", "mixed", "mismatched"))
def test_plan_rejects_invalid_metadata_forms(monkeypatch, invalid_form):
    constructions = []
    monkeypatch.setattr(
        batch_mla_core,
        "_BatchMLAPagedAttentionFa2Backend",
        lambda *args: constructions.append(args),
    )
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
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
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
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
    ) == original_state
    assert wrapper._backend_impl is backend


def test_auto_all_candidates_rejected_preserves_existing_plan(monkeypatch):
    backend = _RecordingBackend("fa2")
    fa2_attempts = 0

    def plan_fa2(self, request):
        nonlocal fa2_attempts
        fa2_attempts += 1
        if fa2_attempts == 1:
            return batch_mla_planning._MLAWrapperPlanResult(
                backend, request.csr, None
            )
        raise backend_errors._BackendPlanUnsupportedError("fa2 later unavailable")

    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper, "_plan_fa2", plan_fa2
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_maybe_warn_blackwell_auto_fallback",
        classmethod(lambda cls, device, selected_backend: None),
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_cutlass",
        lambda self, request: (_ for _ in ()).throw(
            backend_errors._BackendPlanUnsupportedError(
                "cutlass later unavailable"
            )
        ),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    wrapper.plan(**_csr_plan_args())
    original_state = (
        wrapper._backend_impl,
        wrapper._selected_backend,
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
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
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
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
        batch_mla_core,
        "_BatchMLAPagedAttentionFa2Backend",
        fail("run constructed a backend"),
    )
    monkeypatch.setattr(
        fa2_backend,
        "_get_batch_mla_fa2_module",
        fail("run loaded an FA2 module"),
    )
    for name in ("_derive_csr_from_dense", "_derive_dense_from_csr"):
        monkeypatch.setattr(
            batch_mla_planning, name, fail("run converted metadata")
        )

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


@pytest.mark.parametrize("metadata_form", ("canonical", "legacy"))
def test_cutlass_planned_metadata_allows_runtime_omission(
    monkeypatch, metadata_form
):
    module = _FakeCutlassModule()
    monkeypatch.setattr(
        cutlass_backend, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(cutlass_backend, "get_mla_module", lambda: module)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(4), backend="cutlass"
    )
    plan_args = _cutlass_plan_args(metadata_form)
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

    expected_kv_len = (
        plan_args["seq_lens"]
        if metadata_form == "canonical"
        else plan_args["kv_len"]
    )
    expected_page_table = (
        plan_args["block_tables"]
        if metadata_form == "canonical"
        else plan_args["page_table"]
    )
    assert result.shape == (2, 128, 512)
    assert module.run_args[5] is expected_kv_len
    assert module.run_args[6] is expected_page_table


@pytest.mark.parametrize("mismatched_metadata", ("kv_len", "page_table"))
def test_cutlass_planned_metadata_rejects_non_aliasing_runtime_values(
    monkeypatch, mismatched_metadata
):
    monkeypatch.setattr(
        cutlass_backend, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(cutlass_backend, "get_mla_module", _FakeCutlassModule)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(4), backend="cutlass"
    )
    plan_args = _cutlass_plan_args("canonical")
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
def test_xqa_wrapper_plan_run_adapts_dense_and_csr_metadata(
    monkeypatch, metadata_form
):
    backend = _RecordingBackend("xqa")
    monkeypatch.setattr(
        batch_mla_core,
        "_BatchMLAPagedAttentionXqaBackend",
        lambda workspace: backend,
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="xqa")

    wrapper.plan(**_xqa_plan_args(metadata_form))
    result = wrapper.run(
        torch.empty((2, 128, 512), dtype=torch.bfloat16),
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((8, 32, 512), dtype=torch.bfloat16),
        torch.empty((8, 32, 64), dtype=torch.bfloat16),
    )

    assert wrapper._selected_backend == "xqa"
    assert wrapper._dense_plan_metadata is not None
    assert (
        backend.plan_kwargs["block_tables"]
        is wrapper._dense_plan_metadata.block_tables
    )
    assert set(backend.run_kwargs) == {
        "q_nope",
        "q_pe",
        "ckv_cache",
        "kpe_cache",
        "out",
        "lse",
        "return_lse",
        "bmm1_scale",
        "bmm2_scale",
    }
    assert result is backend.result


def test_auto_falls_back_in_order_for_supported_rejections(monkeypatch, caplog):
    backend = _RecordingBackend("cutlass")
    calls = []
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa3")
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_fa3",
        lambda self, request: calls.append("fa3")
        or (_ for _ in ()).throw(
            backend_errors._BackendPlanUnsupportedError("fa3 unavailable")
        ),
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_cutlass",
        lambda self, request: calls.append("cutlass")
        or batch_mla_planning._MLAWrapperPlanResult(backend, None, None),
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")

    with caplog.at_level(logging.DEBUG, logger=batch_mla_core.__name__):
        wrapper.plan(**_csr_plan_args())

    assert calls == ["fa3", "cutlass"]
    assert wrapper._selected_backend == "cutlass"
    assert "automatically rejected backend 'fa3': fa3 unavailable" in caplog.text


def test_auto_does_not_fallback_from_unexpected_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_fa2",
        lambda self, request: calls.append("fa2")
        or (_ for _ in ()).throw(RuntimeError("unexpected failure")),
    )
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_cutlass",
        lambda self, request: calls.append("cutlass"),
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
