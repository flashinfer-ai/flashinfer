"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import warnings

import torch
import pytest


COMMON_PLAN_KWARGS = dict(
    num_heads=16,
    head_dim_ckv=4,
    head_dim_kpe=2,
    page_size=1,
    causal=False,
    sm_scale=0.125,
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16,
)


POSITIONAL_MLA_ARGUMENTS_WARNING = (
    "Positional MLA arguments are deprecated; pass plan() and run() arguments "
    "by keyword instead. Positional calling will be removed in a future release."
)

LEGACY_MLA_TENSOR_ARGUMENTS_WARNING = (
    "Legacy MLA tensor arguments q_nope/q_pe and ckv_cache/kpe_cache are "
    "deprecated; pass query= and kv_cache= structural values instead. This "
    "compatibility path will be removed in a future release."
)


class _FakeBatchMLAModule:
    def __init__(self):
        self.plan_calls = []
        self.cutlass_calls = []

    def plan(self, *args):
        self.plan_calls.append(args)
        return {"planned": True}

    def run(self, *args):
        self.run_args = args

    def cutlass_mla_paged_attention(self, *args):
        self.cutlass_calls.append(args)


def _minimal_uninitialized_wrapper(wrapper_cls, *, use_cuda_graph=False):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper._float_workspace_buffer = torch.empty(16, dtype=torch.uint8)
    wrapper._int_workspace_buffer = torch.empty(16, dtype=torch.uint8)
    wrapper._pin_memory_int_workspace_buffer = torch.empty(16, dtype=torch.uint8)
    wrapper._use_cuda_graph = use_cuda_graph
    wrapper._backend = "fa2"
    wrapper.device = torch.device("cpu")
    wrapper._qo_indptr_buf = None
    wrapper._kv_indptr_buf = None
    wrapper._kv_indices_buf = None
    wrapper._kv_len_arr_buf = None
    wrapper._requested_backend = "fa2"
    wrapper._planned_backend = None
    wrapper._input_contract = None
    wrapper._planned_query_layout = None
    wrapper._planned_kv_cache_layout = None
    wrapper._legacy_flat_csr_plan = False
    wrapper._warned_positional_arguments = False
    wrapper._warned_legacy_tensor_arguments = False
    wrapper._warned_legacy_dynamic_lse = False
    wrapper._retired_cuda_graph_backends = []
    return wrapper


def _patch_fake_fa_module(monkeypatch, fake_module):
    import flashinfer.mla._batch_mla._backends.fa2_backend as fa2_backend
    import flashinfer.mla._batch_mla._backends.fa3_backend as fa3_backend

    monkeypatch.setattr(fa2_backend, "get_batch_mla_module", lambda *args: fake_module)
    monkeypatch.setattr(fa3_backend, "get_batch_mla_module", lambda *args: fake_module)


def _patch_fake_cutlass_module(monkeypatch, fake_module):
    import flashinfer.mla._batch_mla._backends.cutlass_backend as cutlass_backend

    monkeypatch.setattr(cutlass_backend, "get_mla_module", lambda: fake_module)
    monkeypatch.setattr(
        cutlass_backend, "_get_compute_capability", lambda device: (10, 0)
    )


def _dense_metadata():
    import flashinfer.mla as mla

    return mla.MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_tables=torch.tensor([[7], [8]], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
    )


def test_mla_plan_metadata_accepts_cpu_dense_and_derives_csr():
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = mla.MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
    )
    resolver = _MLAPlanMetadataResolver(
        metadata=metadata, page_size=1, device=torch.device("cpu")
    )

    csr = resolver.resolve_csr()

    assert csr.qo_indptr.device.type == "cpu"
    assert csr.kv_indices.tolist() == [0, 1]


def test_mla_plan_metadata_rejects_partial_dense_form():
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = mla.MLAPlanMetadata(cum_seq_lens_q=torch.tensor([0], dtype=torch.int32))
    resolver = _MLAPlanMetadataResolver(
        metadata=metadata, page_size=1, device=torch.device("cpu")
    )

    with pytest.raises(ValueError, match="dense metadata form is partial"):
        resolver.resolve_csr()


def test_mla_plan_metadata_rejects_unequal_dual_forms():
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = mla.MLAPlanMetadata.dual(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32),
        block_tables=torch.tensor([[4]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    resolver = _MLAPlanMetadataResolver(
        metadata=metadata, page_size=1, device=torch.device("cpu")
    )

    with pytest.raises(ValueError, match="logically equivalent"):
        resolver.resolve_csr()


def test_keyword_metadata_plan_is_transactional_after_failed_replan(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)

    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)
    old_backend = wrapper._planned_backend
    old_contract = wrapper._input_contract
    old_kv_indices = wrapper._kv_indices_buf

    with pytest.raises(ValueError, match="dense metadata form is partial"):
        wrapper.plan(
            metadata=mla.MLAPlanMetadata(
                cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32)
            ),
            **COMMON_PLAN_KWARGS,
        )

    assert wrapper._planned_backend is old_backend
    assert wrapper._input_contract == old_contract
    assert wrapper._kv_indices_buf is old_kv_indices


@pytest.mark.parametrize("backend_name", ["fa2", "fa3"])
def test_successful_fa_plan_publishes_sglang_fast_replay_mirrors(
    monkeypatch, backend_name
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._backend = backend_name
    wrapper._qo_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indices_buf = torch.empty(2, dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.empty(2, dtype=torch.int32)

    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)

    backend = wrapper._planned_backend
    assert wrapper._cached_module is backend._cached_module
    assert wrapper._int_workspace_buffer is backend._int_workspace_buffer
    assert (
        wrapper._pin_memory_int_workspace_buffer
        is backend._pin_memory_int_workspace_buffer
    )


def test_failed_fa_replan_keeps_sglang_fast_replay_mirrors_transactional(
    monkeypatch,
):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends.fa2_backend as fa2_backend

    first = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, first)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)
    old_mirrors = (
        wrapper._cached_module,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
    )

    class _FailingPlanModule(_FakeBatchMLAModule):
        def plan(self, *args):
            raise RuntimeError("backend plan failed")

    monkeypatch.setattr(
        fa2_backend, "get_batch_mla_module", lambda *args: _FailingPlanModule()
    )
    with pytest.raises(RuntimeError, match="backend plan failed"):
        wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)

    assert (
        wrapper._cached_module,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
    ) == old_mirrors


def test_legacy_flat_plan_keeps_split_defaults(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    monkeypatch.setattr(mla.BatchMLAPagedAttentionWrapper, "_legacy_plan_warned", False)

    with pytest.warns(DeprecationWarning):
        wrapper.plan(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([7, 8], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            **COMMON_PLAN_KWARGS,
        )

    assert wrapper._input_contract.query_layout == "split"
    assert wrapper._input_contract.kv_cache_layout == "split"


def test_legacy_flat_plan_temporarily_keeps_dynamic_lse(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    csr = (
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([7, 8], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    with pytest.warns(DeprecationWarning):
        wrapper.plan(*csr, **COMMON_PLAN_KWARGS)

    q_nope = torch.empty(2, 16, 4, dtype=torch.bfloat16)
    q_pe = torch.empty(2, 16, 2, dtype=torch.bfloat16)
    ckv_cache = torch.empty(2, 1, 4, dtype=torch.bfloat16)
    kpe_cache = torch.empty(2, 1, 2, dtype=torch.bfloat16)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _, lse_base2 = wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=True)
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)
        _, lse_basee = wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            return_lse=True,
            return_lse_base_on_e=True,
        )

    lse_fallback_warnings = [
        warning
        for warning in recorded
        if "Legacy flat CSR MLA plans temporarily allow dynamic LSE"
        in str(warning.message)
    ]
    assert len(lse_fallback_warnings) == 1
    assert lse_base2.shape == (2, 16)
    assert lse_basee.shape == (2, 16)


def test_positional_plan_arguments_warn_once_per_wrapper(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    monkeypatch.setattr(mla.BatchMLAPagedAttentionWrapper, "_legacy_plan_warned", True)
    csr = (
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([7, 8], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.plan(*csr, **COMMON_PLAN_KWARGS)
        wrapper.plan(*csr, **COMMON_PLAN_KWARGS)

    positional_warnings = [
        warning
        for warning in recorded
        if str(warning.message) == POSITIONAL_MLA_ARGUMENTS_WARNING
    ]
    assert len(positional_warnings) == 1
    assert issubclass(positional_warnings[0].category, DeprecationWarning)


def test_legacy_flat_plan_warning_points_to_the_caller(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    monkeypatch.setattr(mla.BatchMLAPagedAttentionWrapper, "_legacy_plan_warned", False)
    csr = (
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([7, 8], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.plan(*csr, **COMMON_PLAN_KWARGS)

    flat_metadata_warning = next(
        warning
        for warning in recorded
        if "Passing flat BatchMLAPagedAttentionWrapper.plan metadata"
        in str(warning.message)
    )
    assert flat_metadata_warning.filename.endswith("test_mla_wrapper.py")


def test_positional_run_arguments_warn_once_per_wrapper(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)
    q_nope, q_pe = query.split((4, 2), dim=-1)
    ckv_cache, kpe_cache = kv_cache.split((4, 2), dim=-1)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)

    positional_warnings = [
        warning
        for warning in recorded
        if str(warning.message) == POSITIONAL_MLA_ARGUMENTS_WARNING
    ]
    assert len(positional_warnings) == 1
    assert issubclass(positional_warnings[0].category, DeprecationWarning)
    legacy_warnings = [
        warning
        for warning in recorded
        if str(warning.message) == LEGACY_MLA_TENSOR_ARGUMENTS_WARNING
    ]
    assert len(legacy_warnings) == 1
    assert issubclass(legacy_warnings[0].category, DeprecationWarning)


def test_legacy_run_tensor_keywords_warn_once_at_caller(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)
    q_nope, q_pe = query.split((4, 2), dim=-1)
    ckv_cache, kpe_cache = kv_cache.split((4, 2), dim=-1)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
        )
        wrapper.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
        )

    legacy_warnings = [
        warning
        for warning in recorded
        if str(warning.message) == LEGACY_MLA_TENSOR_ARGUMENTS_WARNING
    ]
    assert len(legacy_warnings) == 1
    assert issubclass(legacy_warnings[0].category, DeprecationWarning)
    assert legacy_warnings[0].filename.endswith("test_mla_wrapper.py")
    assert not any(
        str(warning.message) == POSITIONAL_MLA_ARGUMENTS_WARNING for warning in recorded
    )


@pytest.mark.parametrize("legacy_group", ["query", "kv_cache"])
def test_each_legacy_run_tensor_group_warns(monkeypatch, legacy_group):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="split",
        kv_cache_layout="split",
        **COMMON_PLAN_KWARGS,
    )
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)
    q_nope, q_pe = query.split((4, 2), dim=-1)
    ckv_cache, kpe_cache = kv_cache.split((4, 2), dim=-1)
    run_kwargs = {"query": query, "kv_cache": kv_cache}
    if legacy_group == "query":
        run_kwargs.pop("query")
        run_kwargs.update(q_nope=q_nope, q_pe=q_pe)
    else:
        run_kwargs.pop("kv_cache")
        run_kwargs.update(ckv_cache=ckv_cache, kpe_cache=kpe_cache)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.run(**run_kwargs)

    assert (
        sum(
            str(warning.message) == LEGACY_MLA_TENSOR_ARGUMENTS_WARNING
            for warning in recorded
        )
        == 1
    )


def test_legacy_run_tensor_groups_share_one_warning_per_wrapper(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="split",
        kv_cache_layout="split",
        **COMMON_PLAN_KWARGS,
    )
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)
    q_nope, q_pe = query.split((4, 2), dim=-1)
    ckv_cache, kpe_cache = kv_cache.split((4, 2), dim=-1)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.run(q_nope=q_nope, q_pe=q_pe, kv_cache=kv_cache)
        wrapper.run(query=query, ckv_cache=ckv_cache, kpe_cache=kpe_cache)

    assert (
        sum(
            str(warning.message) == LEGACY_MLA_TENSOR_ARGUMENTS_WARNING
            for warning in recorded
        )
        == 1
    )


def test_structural_split_run_does_not_warn_for_legacy_tensor_arguments(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="split",
        kv_cache_layout="split",
        **COMMON_PLAN_KWARGS,
    )
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        wrapper.run(
            query=query.split((4, 2), dim=-1),
            kv_cache=kv_cache.split((4, 2), dim=-1),
        )

    assert not any(
        str(warning.message) == LEGACY_MLA_TENSOR_ARGUMENTS_WARNING
        for warning in recorded
    )


def test_legacy_csr_keyword_plan_remains_supported(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    monkeypatch.setattr(mla.BatchMLAPagedAttentionWrapper, "_legacy_plan_warned", False)

    with pytest.warns(DeprecationWarning):
        wrapper.plan(
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            kv_indices=torch.tensor([0], dtype=torch.int32),
            kv_len_arr=torch.tensor([1], dtype=torch.int32),
            **COMMON_PLAN_KWARGS,
        )

    assert wrapper._input_contract.query_layout == "split"


def test_plan_rejects_surplus_and_duplicate_legacy_arguments(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    csr = (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    tail = (16, 4, 2, 1, False, 0.125, torch.bfloat16, torch.bfloat16, False)

    with pytest.raises(TypeError):
        wrapper.plan(*csr, *tail, "surplus")
    with pytest.raises(TypeError, match="multiple values"):
        wrapper.plan(*csr, *tail, use_profiler=True)


def test_plan_rejects_mixed_metadata_object_and_flat_fields():
    import flashinfer.mla as mla

    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)

    with pytest.raises(ValueError, match="metadata object and flat metadata"):
        wrapper.plan(
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            metadata=_dense_metadata(),
            **COMMON_PLAN_KWARGS,
        )


def test_canonical_csr_rejects_mismatched_batch_dimensions():
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = mla.MLAPlanMetadata.csr(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="batch dimensions must agree"):
        _MLAPlanMetadataResolver(
            metadata=metadata, page_size=1, device=torch.device("cpu")
        ).resolve_csr()


def test_canonical_csr_accepts_equal_dual_and_rejects_other_device():
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = mla.MLAPlanMetadata.dual(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32),
        block_tables=torch.tensor([[3]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    resolver = _MLAPlanMetadataResolver(
        metadata=metadata, page_size=1, device=torch.device("cpu")
    )
    assert resolver.resolve_csr().kv_indices.tolist() == [3]

    other_device = mla.MLAPlanMetadata.csr(
        torch.empty(2, dtype=torch.int32, device="meta"),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="CPU or wrapper device"):
        _MLAPlanMetadataResolver(
            metadata=other_device, page_size=1, device=torch.device("cpu")
        ).resolve_csr()


def test_dense_metadata_alignment_is_enforced():
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    with pytest.raises(ValueError, match="positive multiple of 4"):
        _MLAPlanMetadataResolver(
            metadata=_dense_metadata(), page_size=1, device=torch.device("cpu")
        ).resolve_dense(table_width_alignment=4)


def test_failed_backend_replan_keeps_previous_runnable_backend(monkeypatch):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends.fa2_backend as fa2_backend

    first = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, first)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)
    old_backend = wrapper._planned_backend
    old_indices = wrapper._kv_indices_buf.clone()

    class _FailingPlanModule(_FakeBatchMLAModule):
        def plan(self, *args):
            raise RuntimeError("backend plan failed")

    monkeypatch.setattr(
        fa2_backend, "get_batch_mla_module", lambda *args: _FailingPlanModule()
    )
    with pytest.raises(RuntimeError, match="backend plan failed"):
        wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)

    assert wrapper._planned_backend is old_backend
    assert torch.equal(wrapper._kv_indices_buf, old_indices)


@pytest.mark.parametrize("first_scaled", [False, True])
def test_planless_cutlass_legacy_bridge_validates_each_call_without_plan_state(
    monkeypatch, first_scaled
):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends.cutlass_backend as cutlass_backend

    fake_module = _FakeBatchMLAModule()
    monkeypatch.setattr(cutlass_backend, "get_mla_module", lambda: fake_module)
    monkeypatch.setattr(
        cutlass_backend, "_get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch, "arange", pytest.fail)

    wrapper = mla.BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutlass"
    )
    monkeypatch.setattr(wrapper, "plan", pytest.fail)
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    kv_len = torch.tensor([64, 64], dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    bf16_out = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    fp8_out = torch.empty(2, 128, 512, dtype=torch.float8_e4m3fn)

    scaled_kwargs = {"out": fp8_out, "o_scale": 0.5}
    unscaled_kwargs = {"out": bf16_out}
    ordered = (
        (scaled_kwargs, unscaled_kwargs)
        if first_scaled
        else (unscaled_kwargs, scaled_kwargs)
    )

    with pytest.warns(DeprecationWarning, match="without first calling plan"):
        first_out = wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
            **ordered[0],
        )
    second_out = wrapper.run(
        query=query,
        kv_cache=kv_cache,
        kv_len=kv_len,
        page_table=page_table,
        **ordered[1],
    )

    assert first_out is ordered[0]["out"]
    assert second_out is ordered[1]["out"]
    assert len(fake_module.cutlass_calls) == 2
    assert getattr(wrapper, "_planned_backend", None) is None
    assert wrapper._input_contract is None


def test_planless_cutlass_rejects_return_lse_before_plan_state_mutation(monkeypatch):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends.cutlass_backend as cutlass_backend

    monkeypatch.setattr(
        cutlass_backend, "_get_compute_capability", lambda device: (10, 0)
    )
    wrapper = mla.BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutlass"
    )
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    kv_len = torch.tensor([64, 64], dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="return_lse is not supported"):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
            return_lse=True,
        )

    assert getattr(wrapper, "_planned_backend", None) is None
    assert wrapper._input_contract is None


@pytest.mark.parametrize(
    "plan_kwargs,run_kwargs,match",
    [
        ({}, {"return_lse": True}, "LSE mode"),
        ({"lse_mode": "base2"}, {"return_lse_base_on_e": True}, "LSE mode"),
        ({}, {"o_scale": 0.5}, "o_scale"),
    ],
)
def test_run_rejects_planned_value_contract_mismatch(
    monkeypatch, plan_kwargs, run_kwargs, match
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS, **plan_kwargs)
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=match):
        wrapper.run(query=query, kv_cache=kv_cache, **run_kwargs)


def test_run_returns_caller_owned_output_and_lse_by_identity(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(metadata=_dense_metadata(), lse_mode="base2", **COMMON_PLAN_KWARGS)
    query = torch.empty(2, 16, 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 6, dtype=torch.bfloat16)
    out = torch.empty(2, 16, 4, dtype=torch.bfloat16)
    lse = torch.empty(2, 16, dtype=torch.float32)

    actual_out, actual_lse = wrapper.run(
        query=query, kv_cache=kv_cache, out=out, lse=lse, return_lse=True
    )

    assert actual_out is out
    assert actual_lse is lse


def _cutlass_plan_kwargs(page_size=64):
    import flashinfer.mla as mla

    return {
        "metadata": mla.MLAPlanMetadata.dense(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.zeros((2, 128 // page_size), dtype=torch.int32),
            torch.full((2,), page_size, dtype=torch.int32),
        ),
        "num_heads": 128,
        "head_dim_ckv": 512,
        "head_dim_kpe": 64,
        "page_size": page_size,
        "causal": False,
        "sm_scale": 1.0 / (128 + 64) ** 0.5,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
    }


def test_planned_cutlass_reuses_plan_owned_empty_lse(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper.plan(**_cutlass_plan_kwargs())
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)

    def fail_empty(*args, **kwargs):
        raise AssertionError("planned CUTLASS run must not allocate torch.empty")

    monkeypatch.setattr(torch, "empty", fail_empty)

    assert wrapper.run(query=query, kv_cache=kv_cache, out=out) is out
    assert wrapper.run(query=query, kv_cache=kv_cache, out=out) is out

    first_empty_lse = fake_module.cutlass_calls[0][2]
    second_empty_lse = fake_module.cutlass_calls[1][2]
    assert first_empty_lse.numel() == 0
    assert first_empty_lse is second_empty_lse


@pytest.mark.parametrize("page_size", [0, 127, 256])
def test_planned_cutlass_rejects_invalid_page_size_before_launch(
    monkeypatch, page_size
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    block_tables_width = 1 if page_size <= 0 else max(1, 128 // page_size)
    seq_len = max(page_size, 1)
    metadata = mla.MLAPlanMetadata.dense(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.zeros((2, block_tables_width), dtype=torch.int32),
        torch.full((2,), seq_len, dtype=torch.int32),
    )

    plan_kwargs = _cutlass_plan_kwargs(1)
    plan_kwargs["metadata"] = metadata
    plan_kwargs["page_size"] = page_size

    with pytest.raises(ValueError, match="page_size"):
        wrapper.plan(**plan_kwargs)

    assert fake_module.cutlass_calls == []


@pytest.mark.parametrize("page_size", [0, 127, 256])
def test_planless_cutlass_rejects_invalid_page_size_before_launch(
    monkeypatch, page_size
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, max(page_size, 0), 576, dtype=torch.bfloat16)
    kv_len = torch.ones(2, dtype=torch.int32)
    page_table = torch.zeros((2, 1), dtype=torch.int32)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)

    with (
        pytest.warns(DeprecationWarning, match="CUTLASS"),
        pytest.raises(ValueError, match="page_size"),
    ):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
            out=out,
        )

    assert fake_module.cutlass_calls == []


@pytest.mark.parametrize("backend_name", ["fa2", "fa3"])
def test_fa_backends_share_fp8_plan_validation(monkeypatch, backend_name):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends._fa_common as fa_common

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    monkeypatch.setattr(
        fa_common, "get_compute_capability", lambda device: (9, 0), raising=False
    )
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = backend_name

    with pytest.raises(ValueError, match=r"q_data_type=torch\.bfloat16"):
        wrapper.plan(
            metadata=_dense_metadata(),
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            **{
                key: value
                for key, value in COMMON_PLAN_KWARGS.items()
                if key not in ("q_data_type", "kv_data_type")
            },
        )


@pytest.mark.parametrize("backend_name", ["fa2", "fa3"])
def test_fa_backends_reject_non_int32_kv_indices(monkeypatch, backend_name):
    import flashinfer.mla as mla

    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = backend_name
    metadata = mla.MLAPlanMetadata.csr(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match=r"kv_indices.*torch\.int32"):
        wrapper.plan(metadata=metadata, **COMMON_PLAN_KWARGS)


# Public API compatibility


def test_batch_mla_wrapper_public_imports_remain_compatible():
    import flashinfer
    import flashinfer.mla as mla
    from flashinfer.mla import _core

    assert mla.BatchMLAPagedAttentionWrapper is _core.BatchMLAPagedAttentionWrapper
    assert flashinfer.BatchMLAPagedAttentionWrapper is mla.BatchMLAPagedAttentionWrapper
    assert _core.MLAPlanMetadata is mla.MLAPlanMetadata
    assert hasattr(mla, "MLAPlanMetadata")


# Structural wrapper inputs and plan/run contracts


def _packed_inputs():
    query = torch.empty(2, COMMON_PLAN_KWARGS["num_heads"], 6, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, COMMON_PLAN_KWARGS["page_size"], 6, dtype=torch.bfloat16)
    return query, kv_cache


def _independent_split_inputs():
    q_nope = torch.empty(2, COMMON_PLAN_KWARGS["num_heads"], 4, dtype=torch.bfloat16)
    q_pe = torch.empty(2, COMMON_PLAN_KWARGS["num_heads"], 2, dtype=torch.bfloat16)
    ckv_cache = torch.empty(2, COMMON_PLAN_KWARGS["page_size"], 4, dtype=torch.bfloat16)
    kpe_cache = torch.empty(2, COMMON_PLAN_KWARGS["page_size"], 2, dtype=torch.bfloat16)
    return (q_nope, q_pe), (ckv_cache, kpe_cache)


def test_packed_plan_rejects_independent_split_without_copy(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()

    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="packed",
        kv_cache_layout="packed",
        **COMMON_PLAN_KWARGS,
    )
    query, kv_cache = _independent_split_inputs()

    with pytest.raises(ValueError, match="zero-copy"):
        wrapper.run(query=query, kv_cache=kv_cache)


def test_packed_plan_accepts_adjacent_split_views(monkeypatch):
    import flashinfer.mla as mla

    class _RunnableFakeBatchMLAModule(_FakeBatchMLAModule):
        def run(self, *args):
            self.run_args = args

    fake_module = _RunnableFakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="packed",
        kv_cache_layout="packed",
        **COMMON_PLAN_KWARGS,
    )
    query, kv_cache = _packed_inputs()

    out = wrapper.run(
        query=(query[..., :4], query[..., 4:]),
        kv_cache=(kv_cache[..., :4], kv_cache[..., 4:]),
    )

    assert out.shape == query[..., :4].shape
    assert fake_module.run_args[3].data_ptr() == query[..., :4].data_ptr()
    assert fake_module.run_args[4].data_ptr() == query[..., 4:].data_ptr()


def test_packed_plan_accepts_independent_zero_width_split_without_copy(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="packed",
        kv_cache_layout="packed",
        **{**COMMON_PLAN_KWARGS, "head_dim_kpe": 0},
    )
    q_nope = torch.empty(2, 16, 4, dtype=torch.bfloat16)
    q_pe = torch.empty(2, 16, 0, dtype=torch.bfloat16)
    ckv_cache = torch.empty(2, 1, 4, dtype=torch.bfloat16)
    kpe_cache = torch.empty(2, 1, 0, dtype=torch.bfloat16)

    out = wrapper.run(
        query=(q_nope, q_pe),
        kv_cache=(ckv_cache, kpe_cache),
    )

    assert out.shape == q_nope.shape
    assert fake_module.run_args[3] is q_nope
    assert fake_module.run_args[4] is q_pe
    assert fake_module.run_args[5] is ckv_cache
    assert fake_module.run_args[6] is kpe_cache


def test_structural_parser_accepts_trusted_redundant_forms():
    from flashinfer.mla._batch_mla._contracts import (
        _resolve_structural_mla_input,
        _structural_mla_input_facts,
    )

    packed = torch.empty(2, 3, 6, dtype=torch.bfloat16)
    left = torch.empty(2, 3, 4, dtype=torch.bfloat16)
    right = torch.empty(2, 3, 2, dtype=torch.bfloat16)

    assert (
        _structural_mla_input_facts(
            (packed, (left, right)), widths=(4, 2), name="query"
        )[0]
        == "dual"
    )
    assert (
        _resolve_structural_mla_input(
            (packed, (left, right)),
            desired="packed",
            widths=(4, 2),
            name="query",
        )
        is packed
    )
    assert _resolve_structural_mla_input(
        ((left, right), packed),
        desired="split",
        widths=(4, 2),
        name="query",
    ) == (left, right)


@pytest.mark.parametrize(
    "value,match",
    [
        ([torch.empty(1), torch.empty(1)], "exact 2-tuple"),
        ((torch.empty(1),), "length 2"),
        (
            ((torch.empty(1), torch.empty(1)), (torch.empty(1), torch.empty(1))),
            "malformed nesting",
        ),
        ((torch.empty(1), object()), "leaves must be torch.Tensor"),
    ],
)
def test_structural_parser_rejects_malformed_tuple_grammar(value, match):
    from flashinfer.mla._batch_mla._contracts import _structural_mla_input_facts

    with pytest.raises((TypeError, ValueError), match=match):
        _structural_mla_input_facts(value, widths=(4, 2), name="query")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("rank", "rank 3"),
        ("leading_shape", "shapes must match"),
        ("dtype", "dtypes must match"),
        ("width", "last dimensions"),
    ],
)
def test_split_structural_inputs_are_validated(mutation, match):
    from flashinfer.mla._batch_mla._contracts import _resolve_structural_mla_input

    left = torch.empty(2, 3, 4, dtype=torch.bfloat16)
    right = torch.empty(2, 3, 2, dtype=torch.bfloat16)
    if mutation == "rank":
        right = torch.empty(6, 2, dtype=torch.bfloat16)
    elif mutation == "leading_shape":
        right = torch.empty(1, 3, 2, dtype=torch.bfloat16)
    elif mutation == "dtype":
        right = right.float()
    elif mutation == "width":
        right = torch.empty(2, 3, 3, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=match):
        _resolve_structural_mla_input(
            (left, right), desired="split", widths=(4, 2), name="query"
        )


def test_zero_width_split_can_resolve_packed_without_empty_data_ptr():
    from flashinfer.mla._batch_mla._contracts import _resolve_structural_mla_input

    left = torch.empty(2, 3, 4, dtype=torch.bfloat16)
    unrelated_empty = torch.empty(2, 3, 0, dtype=torch.bfloat16)

    packed = _resolve_structural_mla_input(
        (left, unrelated_empty),
        desired="packed",
        widths=(4, 0),
        name="query",
        accepted="packed",
    )

    assert packed is left


def test_adjacent_split_storage_identity_covers_alias_overlap_and_zero_width():
    from flashinfer.mla._batch_mla._contracts import _are_adjacent_last_dim_views

    packed = torch.empty(2, 3, 6, dtype=torch.bfloat16)

    assert _are_adjacent_last_dim_views(packed[..., :4], packed[..., 4:])
    assert not _are_adjacent_last_dim_views(
        torch.empty(2, 3, 4, dtype=torch.bfloat16),
        torch.empty(2, 3, 2, dtype=torch.bfloat16),
    )
    assert not _are_adjacent_last_dim_views(packed[..., :4], packed[..., 2:4])
    assert not _are_adjacent_last_dim_views(packed[..., :4], packed[..., 4:4])
    if torch.cuda.is_available():
        assert not _are_adjacent_last_dim_views(
            packed[..., :4],
            torch.empty(2, 3, 2, dtype=torch.bfloat16, device="cuda"),
        )


def test_input_contract_rejects_run_output_dtype_mismatch():
    from flashinfer.mla._batch_mla._contracts import MLAInputContract

    contract = MLAInputContract(
        lse_mode="none",
        output_dtype=torch.bfloat16,
        output_scale="none",
        scale_mode="default",
    )

    with pytest.raises(ValueError, match="output dtype"):
        contract.validate_run_options(
            out=torch.empty(1, dtype=torch.float16),
            lse=None,
            return_lse=False,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            ckv_scale_arr=None,
            kpe_scale=None,
        )


def test_input_contract_rejects_incomplete_run_kv_scale_mode():
    from flashinfer.mla._batch_mla._contracts import MLAInputContract

    contract = MLAInputContract(
        lse_mode="none",
        output_dtype=torch.bfloat16,
        output_scale="none",
        scale_mode="kv-per-tensor",
    )

    with pytest.raises(ValueError, match="scale mode"):
        contract.validate_run_options(
            out=None,
            lse=None,
            return_lse=False,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=1.0,
            ckv_scale_arr=None,
            kpe_scale=None,
        )


def test_unplanned_cutlass_compatibility_runs_dynamic_shape_without_plan(monkeypatch):
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._backends import cutlass_backend

    fake_module = _FakeBatchMLAModule()
    monkeypatch.setattr(cutlass_backend, "get_mla_module", lambda: fake_module)
    monkeypatch.setattr(
        cutlass_backend, "_get_compute_capability", lambda device: (10, 0)
    )
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    monkeypatch.setattr(wrapper, "plan", pytest.fail)
    query = (
        torch.empty(2, 128, 512, dtype=torch.bfloat16),
        torch.empty(2, 128, 64, dtype=torch.bfloat16),
    )
    kv_cache = (
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
    )
    kv_len = torch.tensor([1, 1], dtype=torch.int32)
    page_table = torch.zeros((2, 128), dtype=torch.int32)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)

    with pytest.warns(DeprecationWarning, match="CUTLASS") as recorded:
        assert (
            wrapper.run(
                query=query,
                kv_cache=kv_cache,
                kv_len=kv_len,
                page_table=page_table,
                out=out,
            )
            is out
        )

    first_call = fake_module.cutlass_calls[0]
    assert recorded[0].filename.endswith("test_mla_wrapper.py")
    assert first_call[3].shape == (2, 128, 576)
    assert first_call[4].shape == (2, 1, 576)
    assert first_call[5] is kv_len
    assert first_call[6] is page_table
    assert getattr(wrapper, "_planned_backend", None) is None

    larger_query = (
        torch.empty(3, 128, 512, dtype=torch.bfloat16),
        torch.empty(3, 128, 64, dtype=torch.bfloat16),
    )
    larger_kv_cache = (
        torch.empty(3, 2, 512, dtype=torch.bfloat16),
        torch.empty(3, 2, 64, dtype=torch.bfloat16),
    )
    larger_out = torch.empty(3, 128, 512, dtype=torch.bfloat16)
    assert (
        wrapper.run(
            query=larger_query,
            kv_cache=larger_kv_cache,
            kv_len=torch.tensor([1, 1, 1], dtype=torch.int32),
            page_table=torch.zeros((3, 64), dtype=torch.int32),
            out=larger_out,
        )
        is larger_out
    )
    assert len(fake_module.cutlass_calls) == 2
    assert getattr(wrapper, "_planned_backend", None) is None


# CUDA graph planning


def _csr_metadata(mla, qo_indptr, kv_indptr, kv_indices, kv_len_arr):
    return mla.MLAPlanMetadata.csr(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
    )


def _small_plan_kwargs():
    return {
        "num_heads": 16,
        "head_dim_ckv": 4,
        "head_dim_kpe": 2,
        "page_size": 1,
        "causal": False,
        "sm_scale": 0.125,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
    }


def test_batch_mla_plan_reuses_cuda_graph_buffers(monkeypatch):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper,
        use_cuda_graph=True,
    )

    qo_buf = torch.empty(3, dtype=torch.int32)
    kv_indptr_buf = torch.empty(3, dtype=torch.int32)
    kv_indices_buf = torch.full((8,), -1, dtype=torch.int32)
    kv_len_buf = torch.empty(2, dtype=torch.int32)
    wrapper._qo_indptr_buf = qo_buf
    wrapper._kv_indptr_buf = kv_indptr_buf
    wrapper._kv_indices_buf = kv_indices_buf
    wrapper._kv_len_arr_buf = kv_len_buf

    qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 3, 5], dtype=torch.int32)
    kv_indices = torch.tensor([7, 8, 9, 10, 11], dtype=torch.int32)
    kv_len_arr = torch.tensor([3, 2], dtype=torch.int32)

    wrapper.plan(
        metadata=_csr_metadata(mla, qo_indptr, kv_indptr, kv_indices, kv_len_arr),
        num_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=False,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float16,
    )

    assert wrapper._qo_indptr_buf is qo_buf
    assert wrapper._kv_indptr_buf is kv_indptr_buf
    assert wrapper._kv_indices_buf is kv_indices_buf
    assert wrapper._kv_len_arr_buf is kv_len_buf
    assert torch.equal(qo_buf, qo_indptr)
    assert torch.equal(kv_indptr_buf, kv_indptr)
    assert torch.equal(kv_indices_buf[: len(kv_indices)], kv_indices)
    assert torch.equal(kv_indices_buf[len(kv_indices) :], torch.full((3,), -1))
    assert torch.equal(kv_len_buf, kv_len_arr)

    wrapper.plan(
        metadata=_csr_metadata(
            mla,
            qo_buf,
            kv_indptr_buf,
            kv_indices_buf[: len(kv_indices)],
            kv_len_buf,
        ),
        num_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=False,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float16,
    )


def test_cuda_graph_replan_failure_rolls_back_reserved_metadata(monkeypatch):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends.fa2_backend as fa2_backend

    fake_module = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._qo_indptr_buf = torch.full((3,), -1, dtype=torch.int32)
    wrapper._kv_indptr_buf = torch.full((3,), -1, dtype=torch.int32)
    wrapper._kv_indices_buf = torch.full((8,), -1, dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.full((2,), -1, dtype=torch.int32)
    common = _small_plan_kwargs()
    wrapper.plan(
        metadata=_csr_metadata(
            mla,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
        ),
        **common,
    )
    previous_backend = wrapper._planned_backend
    previous_plan_info = wrapper._plan_info
    int_workspace = wrapper._int_workspace_buffer
    pin_workspace = wrapper._pin_memory_int_workspace_buffer
    int_workspace.fill_(17)
    pin_workspace.fill_(23)
    int_workspace_snapshot = int_workspace.clone()
    pin_workspace_snapshot = pin_workspace.clone()
    snapshots = tuple(
        tensor.clone()
        for tensor in (
            wrapper._qo_indptr_buf,
            wrapper._kv_indptr_buf,
            wrapper._kv_indices_buf,
            wrapper._kv_len_arr_buf,
        )
    )

    failing_module = None

    class _FailsAfterPlanningStarts(_FakeBatchMLAModule):
        def plan(self, *args):
            self.int_workspace_arg = args[1]
            self.pin_workspace_arg = args[2]
            args[1].fill_(99)
            args[2].fill_(88)
            raise RuntimeError("candidate plan failure")

    def failing_module_loader(*args):
        nonlocal failing_module
        failing_module = _FailsAfterPlanningStarts()
        return failing_module

    monkeypatch.setattr(
        fa2_backend,
        "get_batch_mla_module",
        failing_module_loader,
    )
    with pytest.raises(RuntimeError, match="candidate plan failure"):
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([6, 7], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **common,
        )

    assert failing_module is not None
    assert failing_module.int_workspace_arg is int_workspace
    assert failing_module.pin_workspace_arg is pin_workspace
    assert wrapper._planned_backend is previous_backend
    assert wrapper._plan_info is previous_plan_info
    assert torch.equal(int_workspace, int_workspace_snapshot)
    assert torch.equal(pin_workspace, pin_workspace_snapshot)
    assert wrapper._int_workspace_buffer is int_workspace
    assert wrapper._pin_memory_int_workspace_buffer is pin_workspace
    for actual, expected in zip(
        (
            wrapper._qo_indptr_buf,
            wrapper._kv_indptr_buf,
            wrapper._kv_indices_buf,
            wrapper._kv_len_arr_buf,
        ),
        snapshots,
        strict=True,
    ):
        assert torch.equal(actual, expected)


def test_cuda_graph_fa_replan_retains_captured_backend_storage(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._qo_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indices_buf = torch.empty(8, dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.empty(2, dtype=torch.int32)
    common = _small_plan_kwargs()
    wrapper.plan(
        metadata=_csr_metadata(
            mla,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
        ),
        **common,
    )
    captured_backend = wrapper._planned_backend
    captured_int_workspace = wrapper._int_workspace_buffer
    captured_pin_workspace = wrapper._pin_memory_int_workspace_buffer

    wrapper.plan(
        metadata=_csr_metadata(
            mla,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([3, 4], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
        ),
        **common,
    )

    replanned_backend = wrapper._planned_backend
    assert replanned_backend is not captured_backend
    assert replanned_backend._int_workspace_buffer is captured_int_workspace
    assert replanned_backend._pin_memory_int_workspace_buffer is captured_pin_workspace
    assert wrapper._int_workspace_buffer is captured_int_workspace
    assert wrapper._pin_memory_int_workspace_buffer is captured_pin_workspace
    assert captured_backend in wrapper._retired_cuda_graph_backends


@pytest.mark.parametrize(
    "field", ["dtype", "device", "capacity", "contiguity", "overlap"]
)
def test_cuda_graph_reserved_buffer_preflight_rejects_unsafe_buffers(
    monkeypatch, field
):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._qo_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indices_buf = torch.empty(8, dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.empty(2, dtype=torch.int32)
    if field == "dtype":
        wrapper._kv_indices_buf = torch.empty(8, dtype=torch.int64)
    elif field == "device":
        wrapper._kv_indices_buf = torch.empty(8, dtype=torch.int32, device="meta")
    elif field == "capacity":
        wrapper._kv_indices_buf = torch.empty(1, dtype=torch.int32)
    elif field == "contiguity":
        wrapper._kv_indices_buf = torch.empty(16, dtype=torch.int32)[::2]
    else:
        shared = torch.empty(6, dtype=torch.int32)
        wrapper._qo_indptr_buf = shared[:3]
        wrapper._kv_indptr_buf = shared[1:4]

    with pytest.raises(ValueError, match="CUDA graph"):
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **_small_plan_kwargs(),
        )


def test_cuda_graph_staging_rejects_cross_source_target_alias(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    qo_reserved = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_indptr_reserved = torch.tensor([0, 5, 10], dtype=torch.int32)
    wrapper._qo_indptr_buf = qo_reserved
    wrapper._kv_indptr_buf = kv_indptr_reserved
    wrapper._kv_indices_buf = torch.empty(2, dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.empty(2, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"CUDA graph.*source.*reserved"):
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                kv_indptr_reserved,
                qo_reserved,
                torch.tensor([3, 4], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **_small_plan_kwargs(),
        )


def test_cuda_graph_staging_rejects_rank2_kv_indices_reserve(monkeypatch):
    import flashinfer.mla as mla

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._qo_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indptr_buf = torch.empty(3, dtype=torch.int32)
    wrapper._kv_indices_buf = torch.empty((2, 2), dtype=torch.int32)
    wrapper._kv_len_arr_buf = torch.empty(2, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"CUDA graph.*kv_indices.*rank 1"):
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([3, 4], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **_small_plan_kwargs(),
        )


def test_cuda_graph_cutlass_replan_is_rejected(monkeypatch):
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._backends import cutlass_backend

    class _Backend:
        _backend = "cutlass"

    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda cls, args: _Backend()),
    )
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._backend = "cutlass"
    metadata = mla.MLAPlanMetadata.dense(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros((1, 128), dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    kwargs = dict(
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=False,
        sm_scale=1.0 / (192**0.5),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    wrapper.plan(metadata=metadata, **kwargs)

    with pytest.raises(RuntimeError, match=r"CUDA graph.*replan"):
        wrapper.plan(metadata=metadata, **kwargs)
