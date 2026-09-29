"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import gc
import math
import warnings
import weakref

import pytest
import torch

from flashinfer.mla._batch_mla import _wrapper
from flashinfer.mla._batch_mla._backends._capabilities import (
    MLAPlanCapabilities,
    _BackendPlanUnsupportedError,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_mla_module_proxy_copies_only_planned_workspace_prefix():
    from flashinfer.jit.attention.modules import _BatchMLAModuleProxy

    class _RawModule:
        def plan(self, _float_workspace, _int_workspace, scratch, *_args):
            scratch.view(torch.uint8).fill_(0xAB)
            return [0] * 18, 37

        def run(self, *args):
            return args

    raw = _RawModule()
    proxy = _BatchMLAModuleProxy(raw)
    int_workspace = torch.full((64,), 0xCD, dtype=torch.uint8, device="cuda")
    scratch = torch.empty(64, dtype=torch.uint8, pin_memory=True)

    plan_info, staged_int_workspace_bytes = proxy.plan_with_staged_workspace_bytes(
        torch.empty(1, dtype=torch.uint8, device="cuda"),
        int_workspace,
        scratch,
        *([None] * 6),
    )
    torch.cuda.synchronize()

    assert plan_info == [0] * 18
    assert staged_int_workspace_bytes == 37
    assert (
        proxy.plan(
            torch.empty(1, dtype=torch.uint8, device="cuda"),
            int_workspace,
            scratch,
            *([None] * 6),
        )
        == [0] * 18
    )
    torch.cuda.synchronize()
    assert torch.equal(
        int_workspace.cpu(),
        torch.tensor([0xAB] * 37 + [0xCD] * 27, dtype=torch.uint8),
    )
    assert proxy.run("forwarded") == ("forwarded",)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("staged_int_workspace_bytes", [-1, 65])
def test_batch_mla_module_proxy_rejects_invalid_workspace_prefix(
    staged_int_workspace_bytes,
):
    from flashinfer.jit.attention.modules import _BatchMLAModuleProxy

    class _RawModule:
        def plan(self, *_args):
            return [0] * 18, staged_int_workspace_bytes

    proxy = _BatchMLAModuleProxy(_RawModule())
    with pytest.raises(ValueError, match="staged_int_workspace_bytes"):
        proxy.plan_with_staged_workspace_bytes(
            torch.empty(1, dtype=torch.uint8, device="cuda"),
            torch.empty(64, dtype=torch.uint8, device="cuda"),
            torch.empty(64, dtype=torch.uint8, pin_memory=True),
            *([None] * 6),
        )


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
    wrapper._backend_type = _wrapper._BACKEND_TYPES["fa2"]
    wrapper._planned_backend = None
    wrapper._input_contract = None
    wrapper._planned_query_layout = None
    wrapper._planned_kv_cache_layout = None
    wrapper._legacy_flat_csr_plan = False
    wrapper._warned_positional_arguments = False
    wrapper._warned_legacy_tensor_arguments = False
    wrapper._warned_legacy_dynamic_lse = False
    return wrapper


def _patch_fake_fa_module(monkeypatch, fake_module):
    import flashinfer.mla._batch_mla._backends.fa2_backend as fa2_backend
    import flashinfer.mla._batch_mla._backends.fa3_backend as fa3_backend

    import flashinfer.mla._batch_mla._backends._fa_common as fa_common

    # These tests model the native planner with tiny 4/2-dimensional CPU tensors.
    # Keep metadata/transaction behavior real; mock the native support boundary
    # alongside the native implementation. Native support has separate tests.
    for module in (fa_common, fa2_backend, fa3_backend):
        monkeypatch.setattr(
            module, "_validate_generated_fa_plan", lambda **kwargs: None
        )
    monkeypatch.setattr(fa_common, "_validate_fa_plan_workload", lambda *args: None)
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
    wrapper._backend_type = _wrapper._BACKEND_TYPES[backend_name]
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


def test_dense_metadata_alignment_is_padded_without_mutating_input():
    from flashinfer.mla._batch_mla._planning import _MLAPlanMetadataResolver

    metadata = _dense_metadata()
    original = metadata.block_tables.clone()
    resolver = _MLAPlanMetadataResolver(
        metadata=metadata, page_size=1, device=torch.device("cpu")
    )
    dense = resolver.resolve_dense(table_width_alignment=4)
    assert dense.block_tables.shape == (2, 4)
    assert torch.equal(dense.block_tables[:, :1], original)
    assert torch.equal(metadata.block_tables, original)
    assert resolver.resolve_dense(table_width_alignment=4) is dense
    assert resolver.resolve_csr().kv_indices.tolist() == [7, 8]


@pytest.mark.parametrize("backend_name", ["fa2", "cutile"])
def test_failed_backend_replan_keeps_previous_runnable_backend(
    monkeypatch, backend_name
):
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla._backends import fa2_backend, cutile_backend

    def fail_plan(*args, **kwargs):
        raise RuntimeError("backend plan failed")

    if backend_name == "cutile":
        wrapper, kernel = _planned_cutile_wrapper(monkeypatch)
        plan_kwargs = _cutile_contract_plan_kwargs()
        query, cache = _cutile_contract_inputs()
        monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", lambda: fail_plan)
        old_indices = None
    else:
        kernel = _FakeBatchMLAModule()
        _patch_fake_fa_module(monkeypatch, kernel)
        wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
        plan_kwargs = dict(
            metadata=_dense_metadata(),
            query_layout="split",
            kv_cache_layout="split",
            **COMMON_PLAN_KWARGS,
        )
        wrapper.plan(**plan_kwargs)
        query = (
            torch.zeros(2, 16, 4, dtype=torch.bfloat16),
            torch.zeros(2, 16, 2, dtype=torch.bfloat16),
        )
        cache = (
            torch.zeros(9, 1, 4, dtype=torch.bfloat16),
            torch.zeros(9, 1, 2, dtype=torch.bfloat16),
        )
        old_indices = wrapper._kv_indices_buf.clone()
        failing = _FakeBatchMLAModule()
        failing.plan = fail_plan
        monkeypatch.setattr(fa2_backend, "get_batch_mla_module", lambda *args: failing)

    old_backend = wrapper._planned_backend
    old_contract = wrapper._input_contract
    old_workspace = wrapper._float_workspace_buffer.clone()
    with pytest.raises(RuntimeError, match="backend plan failed"):
        wrapper.plan(**plan_kwargs)
    assert wrapper._planned_backend is old_backend
    assert wrapper._input_contract is old_contract
    torch.testing.assert_close(wrapper._float_workspace_buffer, old_workspace)
    if old_indices is not None:
        assert torch.equal(wrapper._kv_indices_buf, old_indices)
    out = torch.empty_like(query[0])
    assert wrapper.run(query=query, kv_cache=cache, out=out) is out
    if backend_name == "cutile":
        assert len(kernel.calls) == 1
    else:
        assert hasattr(kernel, "run_args")


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
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
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


@pytest.mark.parametrize("tensor_name", ["query", "kv_cache", "out"])
def test_planned_cutlass_rejects_non_contiguous_launch_tensors_before_dispatch(
    monkeypatch, tensor_name
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
    wrapper.plan(**_cutlass_plan_kwargs())
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    if tensor_name == "query":
        query = torch.empty(2, 128, 1152, dtype=torch.bfloat16)[..., :576]
    elif tensor_name == "kv_cache":
        kv_cache = torch.empty(2, 64, 1152, dtype=torch.bfloat16)[..., :576]
    else:
        out = torch.empty(2, 128, 1024, dtype=torch.bfloat16)[..., :512]

    assert not {
        "query": query,
        "kv_cache": kv_cache,
        "out": out,
    }[tensor_name].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(query=query, kv_cache=kv_cache, out=out)

    assert fake_module.cutlass_calls == []


@pytest.mark.parametrize("tensor_name", ["query", "kv_cache", "out"])
def test_planned_cutlass_rejects_workspace_device_mismatch_before_dispatch(
    monkeypatch, tensor_name
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
    wrapper.plan(**_cutlass_plan_kwargs())
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    if tensor_name == "query":
        query = torch.empty(2, 128, 576, dtype=torch.bfloat16, device="meta")
        out = None
    elif tensor_name == "kv_cache":
        kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16, device="meta")
        out = None
    else:
        out = torch.empty(2, 128, 512, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="workspace device"):
        wrapper.run(query=query, kv_cache=kv_cache, out=out)

    assert fake_module.cutlass_calls == []


@pytest.mark.parametrize("page_size", [0, 127, 256])
def test_planned_cutlass_rejects_invalid_page_size_before_launch(
    monkeypatch, page_size
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
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

    with pytest.raises(
        ValueError if page_size == 0 else _BackendPlanUnsupportedError,
        match="page_size",
    ):
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
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
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

    monkeypatch.setattr(
        fa_common, "get_compute_capability", lambda device: (9, 0), raising=False
    )
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = backend_name
    wrapper._backend_type = _wrapper._BACKEND_TYPES[backend_name]

    with pytest.raises(
        _BackendPlanUnsupportedError, match=r"q_data_type=torch\.bfloat16"
    ):
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
    wrapper._backend_type = _wrapper._BACKEND_TYPES[backend_name]
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


@pytest.mark.parametrize(
    ("scale_kwargs", "message"),
    [
        (
            {},
            "Exactly one of ckv_scale or ckv_scale_arr is required when "
            "kv_data_type is FP8.",
        ),
        (
            {"ckv_scale": 1.0},
            "kpe_scale is required when kv_data_type is FP8.",
        ),
        (
            {"ckv_scale": 1.0, "ckv_scale_arr": torch.ones(1), "kpe_scale": 1.0},
            "Exactly one of ckv_scale or ckv_scale_arr is required when "
            "kv_data_type is FP8.",
        ),
    ],
)
def test_input_contract_preserves_fp8_scale_diagnostics(scale_kwargs, message):
    from flashinfer.mla._batch_mla._contracts import MLAInputContract

    contract = MLAInputContract(
        lse_mode="none",
        output_dtype=torch.bfloat16,
        output_scale="none",
        scale_mode="kv-per-tensor",
    )

    with pytest.raises(ValueError) as exc_info:
        contract.validate_run_options(
            out=None,
            lse=None,
            return_lse=False,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=scale_kwargs.get("ckv_scale"),
            ckv_scale_arr=scale_kwargs.get("ckv_scale_arr"),
            kpe_scale=scale_kwargs.get("kpe_scale"),
        )

    assert str(exc_info.value) == message


def test_input_contract_rejects_scales_for_default_mode_with_established_message():
    from flashinfer.mla._batch_mla._contracts import MLAInputContract

    contract = MLAInputContract(
        lse_mode="none",
        output_dtype=torch.bfloat16,
        output_scale="none",
        scale_mode="default",
    )

    with pytest.raises(ValueError) as exc_info:
        contract.validate_run_options(
            out=None,
            lse=None,
            return_lse=False,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            ckv_scale_arr=torch.ones(1),
            kpe_scale=None,
        )

    assert str(exc_info.value) == (
        "ckv_scale / ckv_scale_arr / kpe_scale are only valid when kv_data_type is FP8."
    )


def _planned_split_wrapper(monkeypatch):
    import flashinfer.mla as mla
    import flashinfer.mla._batch_mla._backends._fa_common as fa_common

    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    monkeypatch.setattr(
        fa_common, "get_compute_capability", lambda device: (9, 0), raising=False
    )
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper.plan(
        metadata=_dense_metadata(),
        query_layout="split",
        kv_cache_layout="split",
        **(
            COMMON_PLAN_KWARGS
            | {
                "head_dim_ckv": 512,
                "head_dim_kpe": 64,
                "kv_data_type": torch.float8_e4m3fn,
            }
        ),
    )
    return wrapper


@pytest.mark.parametrize(
    ("query_dtypes", "kv_dtypes", "message"),
    [
        (
            (torch.bfloat16, torch.float16),
            (torch.float8_e4m3fn, torch.float8_e4m3fn),
            "q_pe.dtype=torch.float16 does not match the planned "
            "q_data_type=torch.bfloat16.",
        ),
        (
            (torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16),
            "kpe_cache.dtype=torch.bfloat16 does not match the planned "
            "kv_data_type=torch.float8_e4m3fn.",
        ),
    ],
)
def test_split_run_preserves_planned_leaf_dtype_diagnostics(
    monkeypatch, query_dtypes, kv_dtypes, message
):
    wrapper = _planned_split_wrapper(monkeypatch)
    q_nope = torch.empty(2, 16, 512, dtype=query_dtypes[0])
    q_pe = torch.empty(2, 16, 64, dtype=query_dtypes[1])
    ckv_cache = torch.empty(2, 1, 512, dtype=kv_dtypes[0])
    kpe_cache = torch.empty(2, 1, 64, dtype=kv_dtypes[1])

    with pytest.raises(ValueError) as exc_info:
        wrapper.run(
            query=(q_nope, q_pe),
            kv_cache=(ckv_cache, kpe_cache),
            ckv_scale=1.0,
            kpe_scale=1.0,
        )

    assert str(exc_info.value) == message


def test_validation_precedes_structural_split_lowering_for_fp8_scales(monkeypatch):
    wrapper = _planned_split_wrapper(monkeypatch)
    q_nope = torch.empty(2, 16, 512, dtype=torch.float16)
    q_pe = torch.empty(2, 16, 64, dtype=torch.bfloat16)
    ckv_cache = torch.empty(2, 1, 512, dtype=torch.float8_e4m3fn)
    kpe_cache = torch.empty(2, 1, 64, dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError) as exc_info:
        wrapper.run(
            query=(q_nope, q_pe),
            kv_cache=(ckv_cache, kpe_cache),
        )

    assert str(exc_info.value) == (
        "Exactly one of ckv_scale or ckv_scale_arr is required when "
        "kv_data_type is FP8."
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
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
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


@pytest.mark.parametrize("tensor_name", ["query", "kv_cache", "out"])
def test_planless_cutlass_rejects_non_contiguous_launch_tensors_before_dispatch(
    monkeypatch, tensor_name
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    if tensor_name == "query":
        query = torch.empty(2, 128, 1152, dtype=torch.bfloat16)[..., :576]
    elif tensor_name == "kv_cache":
        kv_cache = torch.empty(2, 64, 1152, dtype=torch.bfloat16)[..., :576]
    else:
        out = torch.empty(2, 128, 1024, dtype=torch.bfloat16)[..., :512]
    kv_len = torch.full((2,), 64, dtype=torch.int32)
    page_table = torch.zeros((2, 2), dtype=torch.int32)

    assert not {
        "query": query,
        "kv_cache": kv_cache,
        "out": out,
    }[tensor_name].is_contiguous()
    with (
        pytest.warns(DeprecationWarning, match="CUTLASS"),
        pytest.raises(ValueError, match="contiguous"),
    ):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
            out=out,
        )

    assert fake_module.cutlass_calls == []


@pytest.mark.parametrize("tensor_name", ["query", "kv_cache", "out"])
def test_planless_cutlass_rejects_workspace_device_mismatch_before_dispatch(
    monkeypatch, tensor_name
):
    import flashinfer.mla as mla

    fake_module = _FakeBatchMLAModule()
    _patch_fake_cutlass_module(monkeypatch, fake_module)
    wrapper = _minimal_uninitialized_wrapper(mla.BatchMLAPagedAttentionWrapper)
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
    query = torch.empty(2, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    if tensor_name == "query":
        query = torch.empty(2, 128, 576, dtype=torch.bfloat16, device="meta")
        out = None
    elif tensor_name == "kv_cache":
        kv_cache = torch.empty(2, 64, 576, dtype=torch.bfloat16, device="meta")
        out = None
    else:
        out = torch.empty(2, 128, 512, dtype=torch.bfloat16, device="meta")
    kv_len = torch.full((2,), 64, dtype=torch.int32)
    page_table = torch.zeros((2, 2), dtype=torch.int32)

    with (
        pytest.warns(DeprecationWarning, match="CUTLASS"),
        pytest.raises(ValueError, match="workspace device"),
    ):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
            out=out,
        )

    assert fake_module.cutlass_calls == []


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
    previous_backend._staged_int_workspace_bytes = 8
    int_workspace.fill_(17)
    pin_workspace.fill_(23)
    int_workspace_snapshot = int_workspace.clone()
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
            args[1][:8].fill_(99)
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
    assert failing_module.pin_workspace_arg is not pin_workspace
    assert wrapper._planned_backend is previous_backend
    assert wrapper._plan_info is previous_plan_info
    assert torch.equal(int_workspace, int_workspace_snapshot)
    assert torch.equal(pin_workspace, torch.full_like(pin_workspace, 23))
    assert wrapper._int_workspace_buffer is int_workspace
    assert wrapper._pin_memory_int_workspace_buffer is pin_workspace
    previous_backend._cached_module.run("old-plan")
    assert previous_backend._cached_module.run_args == ("old-plan",)
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


def test_cuda_graph_replan_keeps_only_current_backend_and_stable_graph_storage(
    monkeypatch,
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
    stable_tensors = (
        wrapper._int_workspace_buffer,
        wrapper._qo_indptr_buf,
        wrapper._kv_indptr_buf,
        wrapper._kv_indices_buf,
        wrapper._kv_len_arr_buf,
    )
    old_backend_refs = []
    for index in range(32):
        old_backend_refs.append(weakref.ref(wrapper._planned_backend))
        previous_pin_workspace = (
            wrapper._planned_backend._pin_memory_int_workspace_buffer
        )
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([index, index + 1], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **common,
        )
        assert (
            wrapper._planned_backend._pin_memory_int_workspace_buffer
            is not previous_pin_workspace
        )

    gc.collect()

    assert all(
        actual is expected
        for actual, expected in zip(
            (
                wrapper._int_workspace_buffer,
                wrapper._qo_indptr_buf,
                wrapper._kv_indptr_buf,
                wrapper._kv_indices_buf,
                wrapper._kv_len_arr_buf,
            ),
            stable_tensors,
            strict=True,
        )
    )
    assert all(old_backend_ref() is None for old_backend_ref in old_backend_refs)
    assert not hasattr(wrapper, "_retired_cuda_graph_backends")


@pytest.mark.parametrize(
    "field", ["dtype", "device", "capacity", "contiguity", "overlap"]
)
@pytest.mark.parametrize("missing_buffer", [False, True])
def test_cuda_graph_reserved_buffer_preflight_rejects_unsafe_buffers(
    monkeypatch, field, missing_buffer
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
    if missing_buffer:
        wrapper._kv_len_arr_buf = None

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


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize(
    "missing", ["all", "qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr"]
)
def test_fa_graph_missing_reserved_buffers_are_unsupported(
    monkeypatch, backend, missing
):
    import flashinfer.mla as mla

    native = _FakeBatchMLAModule()
    _patch_fake_fa_module(monkeypatch, native)
    buffers = {
        "qo_indptr": torch.empty(3, dtype=torch.int32),
        "kv_indptr": torch.empty(3, dtype=torch.int32),
        "kv_indices": torch.empty(2, dtype=torch.int32),
        "kv_len_arr": torch.empty(2, dtype=torch.int32),
    }
    if missing == "all":
        buffers.clear()
    else:
        buffers.pop(missing)
    wrapper = mla.BatchMLAPagedAttentionWrapper(
        torch.empty(16, dtype=torch.uint8),
        backend=backend,
        use_cuda_graph=True,
        **buffers,
    )
    with pytest.raises(_BackendPlanUnsupportedError, match="requires reserved"):
        wrapper.plan(
            metadata=_csr_metadata(
                mla,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([1, 1], dtype=torch.int32),
            ),
            **_small_plan_kwargs(),
        )
    assert not native.plan_calls
    assert wrapper._planned_backend is None


@pytest.mark.parametrize("missing_buffer", [False, True])
def test_cuda_graph_staging_rejects_cross_source_target_alias(
    monkeypatch, missing_buffer
):
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
    if missing_buffer:
        wrapper._kv_len_arr_buf = None

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
        _plan_capabilities = (
            cutlass_backend._BatchMLAPagedAttentionCutlassBackend._plan_capabilities
        )

    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda cls, args: _Backend()),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "preflight_plan_from_wrapper",
        classmethod(lambda cls, args: None),
    )
    wrapper = _minimal_uninitialized_wrapper(
        mla.BatchMLAPagedAttentionWrapper, use_cuda_graph=True
    )
    wrapper._backend = "cutlass"
    wrapper._backend_type = _wrapper._BACKEND_TYPES["cutlass"]
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


# Planned backend public wrapper integration coverage

_PLANNED_MLA_WORKSPACE_BYTES = 256 * 1024 * 1024


def _skip_if_planned_backend_runtime_is_unavailable(backend):
    if not torch.cuda.is_available():
        pytest.skip(f"{backend} planned MLA runtime test requires CUDA")

    capability = torch.cuda.get_device_capability(0)
    if backend == "xqa":
        from flashinfer.utils import is_sm12x_supported

        if not is_sm12x_supported(torch.device("cuda:0")):
            pytest.skip("xqa planned MLA requires a supported SM12x/CUDA configuration")
        return

    if capability not in ((10, 0), (10, 3)):
        pytest.skip(f"{backend} planned MLA requires SM100/SM103, got {capability}")
    if backend == "cutile":
        pytest.importorskip("cuda.tile.compilation")
        from flashinfer.cutile.cutile_common import is_cuda_tile_available

        if not is_cuda_tile_available():
            pytest.skip("cuTile compiler toolchain is unavailable")
    if backend.startswith("cute-dsl"):
        from flashinfer.cute_dsl import is_cute_dsl_available
        from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported

        if not is_cute_dsl_available():
            pytest.skip("CuTe DSL is unavailable")
        if not is_cute_dsl_arch_supported(*capability):
            pytest.skip("installed CuTe DSL does not support this GPU architecture")


def _make_planned_backend_runtime_case(
    backend,
    dtype,
    *,
    use_sinks=False,
    use_cuda_graph=False,
    plan=True,
):
    import flashinfer.mla as mla

    _skip_if_planned_backend_runtime_is_unavailable(backend)
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    batch_size = 1
    q_len = 1
    page_size = 64
    table_width = 2
    num_heads = 128
    head_dim_ckv = 512
    head_dim_kpe = 64
    head_dim = head_dim_ckv + head_dim_kpe
    max_seq_len = table_width * page_size

    query = torch.randn(
        (batch_size * q_len, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        (batch_size * table_width, page_size, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    if dtype == torch.float8_e4m3fn:
        query = (query * 0.1).to(dtype)
        kv_cache = (kv_cache * 0.1).to(dtype)
    else:
        query = query.to(dtype)
        kv_cache = kv_cache.to(dtype)

    block_tables = torch.arange(
        batch_size * table_width, dtype=torch.int32, device=device
    ).reshape(batch_size, table_width)
    seq_lens = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
    metadata = mla.MLAPlanMetadata.dense(
        torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        block_tables,
        seq_lens,
        max_q_len=q_len,
    )
    workspace = torch.empty(
        _PLANNED_MLA_WORKSPACE_BYTES, dtype=torch.uint8, device=device
    )
    wrapper = mla.BatchMLAPagedAttentionWrapper(
        workspace,
        backend=backend,
        use_cuda_graph=use_cuda_graph,
    )
    is_cute_dsl = backend.startswith("cute-dsl")
    bmm1_scale = 1.0 / math.sqrt(512 if is_cute_dsl else 192)
    bmm2_scale = 1.0
    plan_kwargs = dict(
        metadata=metadata,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=bmm1_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_dtype=torch.bfloat16,
        enable_pdl=backend in ("trtllm-gen", "xqa"),
        use_sinks=use_sinks,
        scale_mode="default" if backend == "cutile" else "bmm-scalar",
    )
    if plan:
        wrapper.plan(**plan_kwargs)

    return {
        "backend": backend,
        "wrapper": wrapper,
        "workspace": workspace,
        "query": query,
        "kv_cache": kv_cache,
        "out": torch.empty(
            (batch_size * q_len, num_heads, head_dim_ckv),
            dtype=torch.bfloat16,
            device=device,
        ),
        "sinks": (
            torch.randn((num_heads,), dtype=torch.float32, device=device)
            if use_sinks
            else None
        ),
        "metadata": metadata,
        "plan_kwargs": plan_kwargs,
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "head_dim_ckv": head_dim_ckv,
        "head_dim_kpe": head_dim_kpe,
        "use_cuda_graph": use_cuda_graph,
    }


def _run_planned_backend_runtime_case(case):
    scales = (
        {}
        if case["backend"] == "cutile"
        else dict(bmm1_scale=case["bmm1_scale"], bmm2_scale=case["bmm2_scale"])
    )
    result = case["wrapper"].run(
        query=case["query"],
        kv_cache=case["kv_cache"],
        out=case["out"],
        sinks=case["sinks"],
        **scales,
    )
    assert result is case["out"]
    return result


def _direct_planned_backend_runtime_oracle(case):
    import flashinfer

    backend = case["backend"]
    if backend == "cutile":
        dim = case["head_dim_ckv"]
        return _cutile_reference(
            (case["query"][..., :dim], case["query"][..., dim:]),
            (case["kv_cache"][..., :dim], case["kv_cache"][..., dim:]),
            case["metadata"].seq_lens,
            case["metadata"].block_tables,
            scale=case["bmm1_scale"],
        )
    query_4d = case["query"].reshape(
        1,
        1,
        case["num_heads"],
        case["head_dim_ckv"] + case["head_dim_kpe"],
    )
    direct_workspace = torch.empty_like(case["workspace"])
    direct_out = torch.empty(
        (1, 1, case["num_heads"], case["head_dim_ckv"]),
        dtype=torch.bfloat16,
        device=case["out"].device,
    )

    if backend == "trtllm-gen":
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_4d,
            kv_cache=case["kv_cache"].unsqueeze(1),
            workspace_buffer=direct_workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=case["head_dim_ckv"],
            qk_rope_head_dim=case["head_dim_kpe"],
            block_tables=case["metadata"].block_tables,
            seq_lens=case["metadata"].seq_lens,
            max_seq_len=case["max_seq_len"],
            out=direct_out,
            bmm1_scale=case["bmm1_scale"],
            bmm2_scale=case["bmm2_scale"],
            sinks=case["sinks"],
            enable_pdl=True,
            backend="trtllm-gen",
            is_var_seq=False,
        ).reshape_as(case["out"])

    if backend == "xqa":
        return flashinfer.decode.xqa_batch_decode_with_kv_cache_mla(
            query=query_4d,
            kv_cache=case["kv_cache"].unsqueeze(1),
            workspace_buffer=direct_workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=case["head_dim_ckv"],
            qk_rope_head_dim=case["head_dim_kpe"],
            block_tables=case["metadata"].block_tables,
            seq_lens=case["metadata"].seq_lens,
            max_seq_len=case["max_seq_len"],
            out=case["out"].new_empty(case["out"].shape),
            bmm1_scale=case["bmm1_scale"],
            bmm2_scale=case["bmm2_scale"],
            enable_pdl=True,
        ).reshape_as(case["out"])

    cute_dsl_impl = "monolithic" if backend == "cute-dsl-monolithic" else "modular"
    return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query_4d,
        case["kv_cache"],
        direct_workspace,
        512,
        case["head_dim_ckv"],
        case["head_dim_kpe"],
        block_tables=case["metadata"].block_tables,
        seq_lens=case["metadata"].seq_lens,
        max_seq_len=case["max_seq_len"],
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        out=direct_out,
        sinks=None if case["sinks"] is None else [case["sinks"]],
        enable_pdl=False,
        is_var_seq=case["use_cuda_graph"],
        backend="cute-dsl",
        cute_dsl_impl=cute_dsl_impl,
    ).reshape_as(case["out"])


@pytest.mark.parametrize(
    ("backend", "dtype", "use_sinks"),
    [
        pytest.param("cutile", torch.bfloat16, False, id="cutile-bf16"),
        pytest.param("trtllm-gen", torch.bfloat16, False, id="trtllm-gen-bf16"),
        pytest.param("trtllm-gen", torch.float8_e4m3fn, False, id="trtllm-gen-fp8"),
        pytest.param("xqa", torch.bfloat16, False, id="xqa-bf16"),
        pytest.param("xqa", torch.float8_e4m3fn, False, id="xqa-fp8"),
        pytest.param(
            "cute-dsl-monolithic",
            torch.bfloat16,
            False,
            id="cute-dsl-monolithic-bf16",
        ),
        pytest.param(
            "cute-dsl-monolithic",
            torch.float8_e4m3fn,
            False,
            id="cute-dsl-monolithic-fp8",
        ),
        pytest.param(
            "cute-dsl-modular",
            torch.bfloat16,
            False,
            id="cute-dsl-modular-bf16",
        ),
        pytest.param(
            "cute-dsl",
            torch.bfloat16,
            False,
            id="cute-dsl-alias-bf16",
        ),
    ],
)
def test_planned_backend_wrapper_matches_direct(backend, dtype, use_sinks):
    from dataclasses import replace

    case = _make_planned_backend_runtime_case(
        backend,
        dtype,
        use_sinks=use_sinks,
        plan=False,
    )
    # Cover CPU staging with BF16 and native device metadata with FP8.
    if dtype == torch.bfloat16:
        metadata = case["metadata"]
        case["plan_kwargs"]["metadata"] = replace(
            metadata,
            cum_seq_lens_q=metadata.cum_seq_lens_q.cpu(),
            block_tables=metadata.block_tables.cpu(),
            seq_lens=metadata.seq_lens.cpu(),
        )
    case["wrapper"].plan(**case["plan_kwargs"])

    actual = _run_planned_backend_runtime_case(case)
    expected = _direct_planned_backend_runtime_oracle(case)

    tolerance = 1e-1 if dtype == torch.float8_e4m3fn else 2e-2
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    ("backend", "source"),
    [
        ("trtllm-gen", "cum_seq_lens_q"),
        ("trtllm-gen", "csr"),
        ("xqa", "seq_lens"),
        ("xqa", "csr"),
        ("cute-dsl-monolithic", "block_tables"),
        ("cute-dsl", "csr"),
    ],
)
def test_planned_backend_graph_requires_device_dense_metadata(backend, source):
    from dataclasses import replace

    import flashinfer.mla as mla

    case = _make_planned_backend_runtime_case(
        backend, torch.bfloat16, use_cuda_graph=True, plan=False
    )
    metadata = case["metadata"]
    if source == "csr":
        metadata = mla.MLAPlanMetadata.csr(
            metadata.cum_seq_lens_q,
            torch.tensor([0, 2], dtype=torch.int32, device="cuda:0"),
            metadata.block_tables.flatten(),
            torch.full_like(metadata.seq_lens, 128),
        )
    else:
        metadata = replace(metadata, **{source: getattr(metadata, source).cpu()})
    case["plan_kwargs"]["metadata"] = metadata
    with pytest.raises(
        _BackendPlanUnsupportedError, match="CUDA graph.*dense metadata.*device"
    ):
        case["wrapper"].plan(**case["plan_kwargs"])
    assert case["wrapper"]._planned_backend is None


@pytest.mark.parametrize("backend", ["cute-dsl-monolithic", "cute-dsl-modular"])
def test_cute_dsl_planned_pads_unaligned_dense_table(backend):
    from dataclasses import replace

    case = _make_planned_backend_runtime_case(backend, torch.bfloat16, plan=False)
    case["plan_kwargs"]["metadata"] = replace(
        case["metadata"], block_tables=case["metadata"].block_tables[:, :1]
    )
    case["wrapper"].plan(**case["plan_kwargs"])
    state = case["wrapper"]._planned_backend._execution_state
    assert state.block_tables.shape[1] == 128 // case["plan_kwargs"]["page_size"]
    actual = _run_planned_backend_runtime_case(case)
    expected = _direct_planned_backend_runtime_oracle(case)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("backend", ["xqa", "cute-dsl-monolithic", "cute-dsl-modular"])
def test_planned_backend_pads_csr_table(backend):
    import flashinfer.mla as mla

    case = _make_planned_backend_runtime_case(backend, torch.bfloat16, plan=False)
    metadata = case["metadata"]
    case["plan_kwargs"]["metadata"] = mla.MLAPlanMetadata.csr(
        metadata.cum_seq_lens_q,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"),
        metadata.block_tables[:, :1].flatten(),
        metadata.seq_lens,
    )
    case["wrapper"].plan(**case["plan_kwargs"])
    torch.testing.assert_close(
        _run_planned_backend_runtime_case(case).float(),
        _direct_planned_backend_runtime_oracle(case).float(),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("q_lengths", [(1,), (1, 2)])
def test_trtllm_gen_planned_query_length_upper_bound(q_lengths):
    from dataclasses import replace

    import flashinfer

    case = _make_planned_backend_runtime_case("trtllm-gen", torch.bfloat16, plan=False)
    case["plan_kwargs"]["causal"] = True
    total_q = sum(q_lengths)
    max_q_len = max(q_lengths) + 1
    batch_size = len(q_lengths)
    offsets = torch.tensor((0, *q_lengths), dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    case["plan_kwargs"]["metadata"] = replace(
        case["metadata"],
        cum_seq_lens_q=offsets,
        block_tables=case["metadata"].block_tables.repeat(batch_size, 1),
        seq_lens=case["metadata"].seq_lens.repeat(batch_size),
        max_q_len=max_q_len,
    )
    # Extra backing rows expose an incorrect launch length without relying on
    # an out-of-allocation GPU access.
    storage_rows = batch_size * max_q_len + 1
    query_storage = torch.randn(
        (storage_rows, 128, 576), dtype=torch.bfloat16, device="cuda"
    )
    output_storage = torch.full(
        (storage_rows, 128, 512), 777.0, dtype=torch.bfloat16, device="cuda"
    )
    case["query"] = query_storage[:total_q]
    case["out"] = output_storage[:total_q]
    case["wrapper"].plan(**case["plan_kwargs"])
    actual = _run_planned_backend_runtime_case(case)
    torch.cuda.synchronize()
    assert torch.all(output_storage[total_q:] == 777), "output guard rows overwritten"

    metadata = case["plan_kwargs"]["metadata"]
    expected = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=case["query"],
        kv_cache=case["kv_cache"].unsqueeze(1),
        workspace_buffer=torch.empty_like(case["workspace"]),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=metadata.block_tables,
        seq_lens=metadata.seq_lens,
        max_seq_len=case["max_seq_len"],
        cum_seq_lens_q=offsets,
        max_q_len=max(q_lengths),
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        enable_pdl=True,
        backend="trtllm-gen",
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_trtllm_gen_planned_rejects_no_rope():
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    case = _make_planned_backend_runtime_case("trtllm-gen", torch.bfloat16, plan=False)
    case["plan_kwargs"]["head_dim_kpe"] = 0
    with pytest.raises(_BackendPlanUnsupportedError, match="MLA dimensions"):
        case["wrapper"].plan(**case["plan_kwargs"])


@pytest.mark.parametrize("actual_q_len", [1, 2])
def test_xqa_planned_query_length_upper_bound(actual_q_len):
    from dataclasses import replace

    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    case = _make_planned_backend_runtime_case("xqa", torch.bfloat16, plan=False)
    metadata = replace(
        case["metadata"],
        cum_seq_lens_q=torch.tensor(
            [0, actual_q_len], dtype=torch.int32, device="cuda"
        ),
        max_q_len=2,
    )
    kwargs = {**case["plan_kwargs"], "metadata": metadata}
    if actual_q_len != 1:
        with pytest.raises(_BackendPlanUnsupportedError, match="one query token"):
            case["wrapper"].plan(**kwargs)
        return

    case["wrapper"].plan(**kwargs)
    torch.testing.assert_close(
        _run_planned_backend_runtime_case(case).float(),
        _direct_planned_backend_runtime_oracle(case).float(),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("scale_name", ["bmm1_scale", "bmm2_scale"])
def test_xqa_planned_integer_scale(scale_name):
    case = _make_planned_backend_runtime_case("xqa", torch.bfloat16)
    case[scale_name] = 1
    actual = _run_planned_backend_runtime_case(case)
    case[scale_name] = 1.0
    expected = _direct_planned_backend_runtime_oracle(case)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "backend",
    ["trtllm-gen", "xqa", "cute-dsl-monolithic", "cutile"],
)
def test_planned_backend_wrapper_cuda_graph_replay_and_replan_rejection(backend):
    case = _make_planned_backend_runtime_case(
        backend,
        torch.bfloat16,
        use_cuda_graph=True,
    )
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            _run_planned_backend_runtime_case(case)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_planned_backend_runtime_case(case)

    with pytest.raises(RuntimeError, match="cannot replan"):
        case["wrapper"].plan(**case["plan_kwargs"])

    case["out"].fill_(-9)
    case["query"].add_(0.125)
    case["kv_cache"].mul_(0.5)
    case["metadata"].seq_lens.fill_(32)
    case["metadata"].block_tables[:, 0].fill_(1)
    graph.replay()
    torch.cuda.synchronize()

    expected = _direct_planned_backend_runtime_oracle(case)
    torch.testing.assert_close(
        case["out"].float(),
        expected.float(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_cute_dsl_modular_wrapper_rejects_cuda_graph_planning():
    case = _make_planned_backend_runtime_case(
        "cute-dsl-modular",
        torch.bfloat16,
        use_cuda_graph=True,
        plan=False,
    )

    with pytest.raises(ValueError, match="does not support CUDA graph"):
        case["wrapper"].plan(**case["plan_kwargs"])


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        pytest.param(ValueError, "bad input", id="invalid-input"),
        pytest.param(RuntimeError, "compile or launch failed", id="compile-launch"),
    ],
)
def test_cute_dsl_alias_does_not_hide_planning_errors(monkeypatch, error_type, message):
    import flashinfer.mla as mla
    from flashinfer.mla._batch_mla import _wrapper

    calls = []

    def capabilities(backend_name):
        return MLAPlanCapabilities(
            backend_name=backend_name,
            lse_modes=frozenset({"none"}),
            kv_layouts=frozenset({"combined"}),
            output_scales=frozenset({"none"}),
            scale_modes=frozenset({"default"}),
        )

    class _ExplodingBackend:
        _plan_capabilities = capabilities("cute-dsl-monolithic")

        @classmethod
        def plan_from_wrapper(cls, args):
            calls.append("monolithic")
            raise error_type(message)

    class _UnexpectedFallbackBackend:
        _plan_capabilities = capabilities("cute-dsl-modular")

        @classmethod
        def plan_from_wrapper(cls, args):
            calls.append("modular")
            return object()

    monkeypatch.setattr(
        _wrapper._BatchMLAPagedAttentionCuteDslBackend,
        "_candidate_types",
        (_ExplodingBackend, _UnexpectedFallbackBackend),
    )
    wrapper = mla.BatchMLAPagedAttentionWrapper(
        torch.empty(1, dtype=torch.uint8),
        backend="cute-dsl",
    )

    with pytest.raises(error_type, match=message):
        wrapper.plan(metadata=_dense_metadata(), **COMMON_PLAN_KWARGS)

    assert calls == ["monolithic"]


# cuTile split-layout, dynamic metadata, and graph numerical coverage.


@pytest.fixture
def cutile_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("prepared cuTile acceptance requires SM100")
    pytest.importorskip("cuda.tile.compilation")
    from flashinfer.cutile.cutile_common import is_cuda_tile_available

    if not is_cuda_tile_available():
        pytest.skip("cuTile compiler toolchain is unavailable")
    prior = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior


def _make_cutile_plan(batch, heads, page, length, dtype, dim=512):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata

    width = max(1, math.ceil(length / page))
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    if batch > 1:
        lengths[1] = min(13, length)
    table = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(
        batch, width
    )
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        block_tables=table,
        seq_lens=lengths,
        max_q_len=8,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, device="cuda", dtype=torch.uint8), backend="cutile"
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=heads,
        head_dim_ckv=dim,
        head_dim_kpe=64,
        page_size=page,
        causal=True,
        sm_scale=1 / math.sqrt(dim + 64),
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )
    return wrapper, lengths, table


def _cutile_inputs(batch, heads, page, pools, dtype, adjacent, offset, dim=512):
    if adjacent:
        q = torch.randn(batch, heads, dim + 64, device="cuda", dtype=dtype)
        kv = torch.randn(pools, page, dim + 64, device="cuda", dtype=dtype)
        query, cache = (q[..., :dim], q[..., dim:]), (kv[..., :dim], kv[..., dim:])
    else:
        query = tuple(
            torch.randn(batch, heads, d, device="cuda", dtype=dtype) for d in (dim, 64)
        )
        cache = tuple(
            torch.randn(pools, page, d, device="cuda", dtype=dtype) for d in (dim, 64)
        )
    storage = torch.empty(batch * heads * dim + offset, device="cuda", dtype=dtype)
    out = storage[offset:].view(batch, heads, dim)
    return query, cache, out


def _cutile_reference(query, cache, lengths, table, scale=None):
    dim = query[0].shape[-1]
    scale = 1 / math.sqrt(dim + 64) if scale is None else scale
    result = []
    for batch, length in enumerate(lengths.tolist()):
        if length == 0:
            result.append(torch.zeros_like(query[0][batch], dtype=torch.float32))
            continue
        kv = cache[0][table[batch].long()].reshape(-1, dim)[:length].float()
        kr = cache[1][table[batch].long()].reshape(-1, 64)[:length].float()
        scores = (
            query[0][batch].float() @ kv.T + query[1][batch].float() @ kr.T
        ) * scale
        result.append(scores.softmax(-1) @ kv)
    return torch.stack(result)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [1, 2])
def test_cutile_small_page_large_batch_numerics(cutile_sm100, dtype, heads, page):
    """Small key tiles must remain correct with the large-batch launch policy."""
    torch.manual_seed(4031)
    batch, length = 32, 525
    wrapper, lengths, table = _make_cutile_plan(batch, heads, page, length, dtype)
    lengths[1::3] = 131
    lengths[2::3] = 262
    query, cache, out = _cutile_inputs(
        batch, heads, page, batch * table.shape[1], dtype, True, 0
    )
    expected = _cutile_reference(query, cache, lengths, table)

    def run():
        return wrapper.run(query=query, kv_cache=cache, out=out)

    assert run() is out
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "batch,heads,page,length,dtype,dim",
    [
        (1, 16, 128, 96, torch.bfloat16, 512),
        (1, 16, 128, 384, torch.bfloat16, 512),
        (16, 64, 64, 129, torch.bfloat16, 512),
        (16, 128, 128, 384, torch.bfloat16, 512),
        (4, 24, 16, 65, torch.float16, 512),
        (2, 8, 2, 17, torch.float16, 512),
        (17, 64, 64, 129, torch.bfloat16, 512),
        (1, 1, 1, 17, torch.bfloat16, 256),
        (4, 32, 1, 2048, torch.bfloat16, 512),
        (1, 16, 1, 129, torch.bfloat16, 512),
        (3, 7, 16, 65, torch.float16, 256),
        (2, 129, 64, 65, torch.bfloat16, 512),
    ],
)
def test_cutile_split_layout_numerics_and_graph(
    cutile_sm100, batch, heads, page, length, dtype, dim
):
    wrapper, lengths, table = _make_cutile_plan(batch, heads, page, length, dtype, dim)
    minimum_pools = batch * table.shape[1]
    for adjacent in (False, True):
        query, cache, out = _cutile_inputs(
            batch, heads, page, minimum_pools, dtype, adjacent, 1, dim
        )

        def run():
            return wrapper.run(query=query, kv_cache=cache, out=out)

        for current_length in (length, 0, min(13, length), length):
            lengths.fill_(current_length)
            expected = _cutile_reference(query, cache, lengths, table)
            out.fill_(math.nan)
            assert run() is out
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        out.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)
        query[0].add_(0.125)
        expected = _cutile_reference(query, cache, lengths, table)
        out.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)


# cuTile adapter contracts at the public wrapper boundary.


class _FakeCutileKernel:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        seq_lens,
        block_tables,
        k_scale,
        v_scale,
        max_seq_len=-1,
        outputs=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "q_nope": q_nope,
                "q_pe": q_pe,
                "ckv_cache": ckv_cache,
                "kpe_cache": kpe_cache,
                "seq_lens": seq_lens,
                "block_tables": block_tables,
                "k_scale": k_scale,
                "v_scale": v_scale,
                "max_seq_len": max_seq_len,
                "outputs": outputs,
                "kwargs": kwargs,
            }
        )
        if outputs is None:
            outputs = torch.empty_like(q_nope)
        outputs.copy_(q_nope)
        return outputs


def _patch_fake_cutile_kernel(monkeypatch, kernel):
    from flashinfer.mla._batch_mla._backends import cutile_backend

    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        cutile_backend, "get_cutile_mla_decode", lambda: lambda **kwargs: kernel
    )


def _cutile_contract_metadata():
    from flashinfer.mla import MLAPlanMetadata

    return MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
    )


def _cutile_contract_plan_kwargs(metadata=None, **overrides):
    kwargs = {
        "metadata": _cutile_contract_metadata() if metadata is None else metadata,
        "num_heads": 16,
        "head_dim_ckv": 512,
        "head_dim_kpe": 64,
        "page_size": 2,
        "causal": False,
        "sm_scale": 1.0 / math.sqrt(576),
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "query_layout": "split",
        "kv_cache_layout": "split",
    }
    kwargs.update(overrides)
    return kwargs


def _cutile_contract_inputs(*, num_heads=16, page_size=2, dtype=torch.bfloat16):
    return (
        (
            torch.empty(2, num_heads, 512, dtype=dtype),
            torch.empty(2, num_heads, 64, dtype=dtype),
        ),
        (
            torch.empty(2, page_size, 512, dtype=dtype),
            torch.empty(2, page_size, 64, dtype=dtype),
        ),
    )


def _planned_cutile_wrapper(monkeypatch, *, use_cuda_graph=False, metadata=None):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_fake_cutile_kernel(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8),
        use_cuda_graph=use_cuda_graph,
        backend="cutile",
    )
    wrapper.plan(**_cutile_contract_plan_kwargs(metadata))
    return wrapper, kernel


def test_cutile_lazy_kernel_lookup_and_retained_dense_metadata(monkeypatch):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    metadata = _cutile_contract_metadata()
    kernel = _FakeCutileKernel()
    getter_calls = []
    from flashinfer.mla._batch_mla._backends import cutile_backend

    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: (10, 0)
    )

    def get_kernel():
        getter_calls.append(None)
        return lambda **kwargs: kernel

    monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", get_kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    with pytest.raises(ValueError, match="head_dim"):
        wrapper.plan(**_cutile_contract_plan_kwargs(metadata, head_dim_ckv=128))
    assert getter_calls == []

    wrapper.plan(**_cutile_contract_plan_kwargs(metadata))
    assert getter_calls == [None]

    query, kv_cache = _cutile_contract_inputs()
    out = torch.empty_like(query[0])
    actual = wrapper.run(query=query, kv_cache=kv_cache, out=out)

    assert actual is out
    assert getter_calls == [None]
    assert len(kernel.calls) == 1
    call = kernel.calls[0]
    assert call["q_nope"] is query[0]
    assert call["q_pe"] is query[1]
    assert call["ckv_cache"] is kv_cache[0]
    assert call["kpe_cache"] is kv_cache[1]
    assert call["seq_lens"] is metadata.seq_lens
    assert call["block_tables"] is metadata.block_tables
    assert call["outputs"] is out


def test_cutile_packed_contract_is_lowered_to_zero_copy_split_views(monkeypatch):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_fake_cutile_kernel(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    wrapper.plan(
        **_cutile_contract_plan_kwargs(query_layout="packed", kv_cache_layout="packed")
    )
    query = torch.empty(2, 16, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 2, 576, dtype=torch.bfloat16)
    wrapper.run(query=query, kv_cache=kv_cache)

    call = kernel.calls[0]
    assert (
        call["q_nope"].untyped_storage().data_ptr()
        == query.untyped_storage().data_ptr()
    )
    assert (
        call["q_pe"].untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
    )
    assert (
        call["ckv_cache"].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )
    assert (
        call["kpe_cache"].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )


def test_cutile_runtime_metadata_override_is_paired_and_zero_copy(monkeypatch):
    wrapper, kernel = _planned_cutile_wrapper(monkeypatch)
    query, kv_cache = _cutile_contract_inputs()
    seq_lens = torch.tensor([2, 3], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="both be omitted or both be provided"):
        wrapper.run(query=query, kv_cache=kv_cache, kv_len=seq_lens)
    with pytest.raises(ValueError, match="both be omitted or both be provided"):
        wrapper.run(query=query, kv_cache=kv_cache, page_table=block_tables)

    wrapper.run(
        query=query,
        kv_cache=kv_cache,
        kv_len=seq_lens,
        page_table=block_tables,
    )
    call = kernel.calls[-1]
    assert call["seq_lens"] is seq_lens
    assert call["block_tables"] is block_tables


@pytest.mark.parametrize(
    ("kv_len", "page_table", "match"),
    [
        (
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "torch.int32",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            "torch.int32",
        ),
        (
            torch.empty(4, dtype=torch.int32)[::2],
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "contiguous",
        ),
        (
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "shape",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([[0, 1]], dtype=torch.int32),
            "shape",
        ),
        (
            torch.empty(2, dtype=torch.int32, device="meta"),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "workspace device",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.empty((2, 2), dtype=torch.int32, device="meta"),
            "workspace device",
        ),
    ],
)
def test_cutile_runtime_metadata_override_rejects_unsafe_tensors(
    monkeypatch, kv_len, page_table, match
):
    wrapper, kernel = _planned_cutile_wrapper(monkeypatch)
    query, kv_cache = _cutile_contract_inputs()

    with pytest.raises(ValueError, match=match):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
        )
    assert kernel.calls == []


@pytest.mark.parametrize(
    "plan_overrides",
    [
        {"lse_mode": "base2"},
        {"output_dtype": torch.float8_e4m3fn, "output_scale": "per-tensor"},
        {"scale_mode": "kv-per-tensor"},
        {"skip_softmax": True},
        {"head_dim_ckv": 128},
        {"q_data_type": torch.float32, "kv_data_type": torch.float32},
    ],
)
def test_cutile_plan_rejects_unsupported_contracts(monkeypatch, plan_overrides):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    monkeypatch.setattr(
        "flashinfer.mla._batch_mla._backends.cutile_backend._get_compute_capability",
        lambda device: (10, 0),
    )

    def unexpected_preparation():
        pytest.fail("unsupported cuTile plan attempted native preparation")

    monkeypatch.setattr(
        "flashinfer.mla._batch_mla._backends.cutile_backend.get_cutile_mla_decode",
        unexpected_preparation,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    with pytest.raises(_BackendPlanUnsupportedError):
        wrapper.plan(**_cutile_contract_plan_kwargs(**plan_overrides))


@pytest.mark.parametrize("page_size", [True, 0, 3, 129])
def test_cutile_rejects_unsupported_page_sizes(page_size):
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _validate_cutile_page_size,
    )

    with pytest.raises(ValueError, match=r"\[1, 128\]"):
        _validate_cutile_page_size(page_size)


@pytest.mark.parametrize("num_heads", [True, 0, -1, 1.5])
def test_cutile_rejects_unsupported_head_counts(num_heads):
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _validate_cutile_num_heads,
    )

    with pytest.raises(ValueError, match=r"positive integer"):
        _validate_cutile_num_heads(num_heads)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_cutile_plan_accepts_supported_blackwell_architectures(monkeypatch, capability):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.mla._batch_mla._backends import cutile_backend

    kernel = _FakeCutileKernel()
    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: capability
    )
    monkeypatch.setattr(
        cutile_backend, "get_cutile_mla_decode", lambda: lambda **kwargs: kernel
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    wrapper.plan(**_cutile_contract_plan_kwargs())
    query, kv_cache = _cutile_contract_inputs()
    wrapper.run(query=query, kv_cache=kv_cache)
    assert len(kernel.calls) == 1


@pytest.mark.parametrize("capability", [(9, 0), (12, 2)])
def test_cutile_plan_rejects_undemonstrated_architectures(monkeypatch, capability):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.mla._batch_mla._backends import cutile_backend

    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: capability
    )

    def unexpected_preparation():
        pytest.fail("unsupported cuTile plan attempted native preparation")

    monkeypatch.setattr(
        "flashinfer.mla._batch_mla._backends.cutile_backend.get_cutile_mla_decode",
        unexpected_preparation,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    with pytest.raises(_BackendPlanUnsupportedError, match="validated Blackwell"):
        wrapper.plan(**_cutile_contract_plan_kwargs())


@pytest.mark.parametrize(
    ("run_overrides", "match"),
    [
        ({"return_lse": True}, "LSE mode"),
        ({"lse": torch.empty(2, 16)}, "LSE mode"),
        ({"o_scale": 0.5}, "o_scale"),
        ({"ckv_scale": 1.0, "kpe_scale": 1.0}, "only valid"),
        ({"profiler_buffer": torch.empty(1)}, "profiler"),
    ],
)
def test_cutile_run_rejects_unsupported_options_before_dispatch(
    monkeypatch, run_overrides, match
):
    wrapper, kernel = _planned_cutile_wrapper(monkeypatch)
    query, kv_cache = _cutile_contract_inputs()

    with pytest.raises(ValueError, match=match):
        wrapper.run(query=query, kv_cache=kv_cache, **run_overrides)
    assert kernel.calls == []


@pytest.mark.parametrize("case", ["shape", "overlap"])
def test_cutile_rejects_unsafe_output_before_dispatch(monkeypatch, case):
    wrapper, kernel = _planned_cutile_wrapper(monkeypatch)
    query, kv_cache = _cutile_contract_inputs()
    if case == "shape":
        out = torch.empty(2, 16, 511, dtype=torch.bfloat16)
    else:
        out = query[0]

    with pytest.raises(ValueError):
        wrapper.run(query=query, kv_cache=kv_cache, out=out)
    assert kernel.calls == []


def test_cutile_empty_zero_width_table_is_typed_unsupported(monkeypatch):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    _patch_fake_cutile_kernel(monkeypatch, _FakeCutileKernel())
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32),
        block_tables=torch.empty((1, 0), dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    with pytest.raises(_BackendPlanUnsupportedError, match="positive width"):
        wrapper.plan(**_cutile_contract_plan_kwargs(metadata))


@pytest.mark.parametrize("scale", [0, -1, 0.0, -0.125])
def test_cutile_nonpositive_finite_scale_is_typed_unsupported(monkeypatch, scale):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    _patch_fake_cutile_kernel(monkeypatch, _FakeCutileKernel())
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    with pytest.raises(_BackendPlanUnsupportedError, match="positive sm_scale"):
        wrapper.plan(**_cutile_contract_plan_kwargs(sm_scale=scale))


@pytest.mark.parametrize(
    "scale,error",
    [
        (True, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-float("inf"), ValueError),
    ],
)
def test_cutile_malformed_scale_remains_caller_error(monkeypatch, scale, error):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    _patch_fake_cutile_kernel(monkeypatch, _FakeCutileKernel())
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    with pytest.raises(error, match="sm_scale"):
        wrapper.plan(**_cutile_contract_plan_kwargs(sm_scale=scale))


def _rollback_plan_args(*, graph=False, **overrides):
    from flashinfer.mla._batch_mla._planning import _MLAPlanArguments

    return _MLAPlanArguments(
        **{
            **COMMON_PLAN_KWARGS,
            "metadata": _dense_metadata(),
            "output_dtype": torch.bfloat16,
            "kv_layout": "combined",
            "_float_workspace_buffer": torch.full((128,), 17, dtype=torch.uint8),
            "_use_cuda_graph": graph,
            "_qo_indptr_buf": torch.full((3,), -1, dtype=torch.int32),
            "_kv_indptr_buf": torch.full((3,), -1, dtype=torch.int32),
            "_kv_indices_buf": torch.full((8,), -1, dtype=torch.int32),
            "_kv_len_arr_buf": torch.full((2,), -1, dtype=torch.int32),
            "_graph_plan_int_workspace_buffer": (
                torch.full((64,), 23, dtype=torch.uint8) if graph else None
            ),
            **overrides,
        }
    )


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("failure", ["native", "unsupported", "audit"])
def test_fa_backend_planning_failure_boundaries(monkeypatch, backend, failure):
    from contextlib import contextmanager
    from flashinfer.mla._batch_mla._planning import _MLAPlanArguments

    class Native(_FakeBatchMLAModule):
        def plan(self, *args):
            args[1].fill_(99)
            if failure == "native":
                raise RuntimeError("injected native failure")
            if failure == "unsupported":
                raise _BackendPlanUnsupportedError("injected unsupported failure")
            return super().plan(*args)

    _patch_fake_fa_module(monkeypatch, Native())
    args = _rollback_plan_args(graph=True)
    buffers = (
        args._graph_plan_int_workspace_buffer,
        args._qo_indptr_buf,
        args._kv_indptr_buf,
        args._kv_indices_buf,
        args._kv_len_arr_buf,
    )
    before = [buffer.clone() for buffer in buffers]
    if failure == "audit":
        original_audit = _MLAPlanArguments.audit_public_argument_access

        @contextmanager
        def fail_audit(self, name):
            with original_audit(self, name):
                yield
            raise AssertionError("injected audit failure")

        monkeypatch.setattr(
            _MLAPlanArguments, "audit_public_argument_access", fail_audit
        )

    backend_type = _wrapper._BACKEND_TYPES[backend]
    error_type = {
        "unsupported": _BackendPlanUnsupportedError,
        "audit": AssertionError,
    }.get(failure, RuntimeError)
    with pytest.raises(error_type, match=f"injected {failure} failure"):
        # Direct calls must also roll back without wrapper intervention.
        backend_type.plan_from_wrapper(args)
    if failure == "audit":
        # Developer audits run after successful planning, outside rollback.
        assert torch.all(args._graph_plan_int_workspace_buffer == 99)
    else:
        for actual, expected in zip(buffers, before, strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("graph", [False, True])
def test_fa_plan_does_not_clone_float_workspace(monkeypatch, backend, graph):
    _patch_fake_fa_module(monkeypatch, _FakeBatchMLAModule())
    args = _rollback_plan_args(graph=graph)
    wrapper = _wrapper.BatchMLAPagedAttentionWrapper(
        args._float_workspace_buffer,
        backend=backend,
        use_cuda_graph=graph,
        qo_indptr=args._qo_indptr_buf,
        kv_indptr=args._kv_indptr_buf,
        kv_indices=args._kv_indices_buf,
        kv_len_arr=args._kv_len_arr_buf,
    )
    original_clone = torch.Tensor.clone

    def clone(tensor, *pos, **kw):
        assert (
            tensor.untyped_storage().data_ptr()
            != args._float_workspace_buffer.untyped_storage().data_ptr()
        )
        return original_clone(tensor, *pos, **kw)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    wrapper.plan(metadata=args.metadata, **COMMON_PLAN_KWARGS)


@pytest.mark.parametrize("failure", [None, "native", "audit"])
def test_xqa_backend_local_rollback_and_snapshot_prefix(monkeypatch, failure):
    from contextlib import contextmanager
    from flashinfer.mla._batch_mla._planning import _MLAPlanArguments
    from flashinfer.mla._batch_mla._backends import xqa_backend
    from flashinfer.mla import MLAPlanMetadata

    monkeypatch.setattr(
        xqa_backend, "_validate_xqa_device_capability", lambda device: None
    )
    monkeypatch.setattr(xqa_backend, "device_support_pdl", lambda device: False)
    monkeypatch.setattr(xqa_backend, "get_xqa_module_mla", lambda *args: object())

    def sm_count(device):
        if failure == "native":
            raise RuntimeError("injected native failure")
        return 1

    monkeypatch.setattr(xqa_backend, "get_device_sm_count", sm_count)
    workspace = torch.full((128 * 1024 * 1024,), 17, dtype=torch.uint8)
    args = _rollback_plan_args(
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=32,
        _float_workspace_buffer=workspace,
        metadata=MLAPlanMetadata(
            cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            block_tables=torch.zeros((2, 4), dtype=torch.int32),
            seq_lens=torch.tensor([32, 32], dtype=torch.int32),
            max_q_len=1,
        ),
    )
    original_clone = torch.Tensor.clone
    copied_bytes = []

    def clone(tensor, *pos, **kw):
        if (
            tensor.untyped_storage().data_ptr()
            == workspace.untyped_storage().data_ptr()
        ):
            copied_bytes.append(tensor.numel() * tensor.element_size())
        return original_clone(tensor, *pos, **kw)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    backend_type = xqa_backend._BatchMLAPagedAttentionXqaBackend
    if failure == "audit":
        original_audit = _MLAPlanArguments.audit_public_argument_access

        @contextmanager
        def fail_audit(self, name):
            with original_audit(self, name):
                yield
            raise AssertionError("injected audit failure")

        monkeypatch.setattr(
            _MLAPlanArguments, "audit_public_argument_access", fail_audit
        )
    if failure is not None:
        error_type = AssertionError if failure == "audit" else RuntimeError
        with pytest.raises(error_type, match=f"injected {failure} failure"):
            backend_type.plan_from_wrapper(args)
        if failure == "audit":
            assert torch.all(workspace[: 8 * 1024 * 1024] == 0)
            assert torch.all(workspace[8 * 1024 * 1024 :] == 17)
        else:
            assert torch.all(workspace == 17)
    else:
        backend_type.plan_from_wrapper(args)
        assert torch.all(workspace[: 8 * 1024 * 1024] == 0)
        assert torch.all(workspace[8 * 1024 * 1024 :] == 17)
    assert copied_bytes == [8 * 1024 * 1024]


@pytest.mark.parametrize("backend", ["fa2", "auto", "cute-dsl"])
@pytest.mark.parametrize("failure", [False, True])
def test_wrapper_warns_once_after_successful_backend_plan(
    monkeypatch, backend, failure
):
    from dataclasses import replace
    from flashinfer.mla._batch_mla import _auto_policy

    metadata = _dense_metadata()
    events = []

    class Backend:
        _backend = "fa2"
        _plan_capabilities = replace(
            _wrapper._BACKEND_TYPES["fa2"]._plan_capabilities, is_experimental=True
        )

        @staticmethod
        def plan_from_wrapper(plan_args):
            assert plan_args.metadata is metadata
            events.append("plan")
            if failure:
                raise RuntimeError("injected planner failure")
            return planned

    planned = Backend()
    monkeypatch.setitem(_wrapper._BACKEND_TYPES, "fa2", Backend)
    monkeypatch.setattr(
        _wrapper._BatchMLAPagedAttentionCuteDslBackend, "_candidate_types", (Backend,)
    )
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", lambda args: ("fa2",))
    monkeypatch.setenv(
        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1" if backend == "auto" else "0"
    )
    wrapper = _wrapper.BatchMLAPagedAttentionWrapper(
        torch.empty(64, dtype=torch.uint8), backend=backend
    )

    expected_published = None

    def observe_warning(api_name, selected_backend, **kwargs):
        # Warnings belong after selection but before successful state publication.
        assert wrapper._planned_backend is expected_published
        events.append((api_name, selected_backend, kwargs))

    monkeypatch.setattr(_wrapper, "warn_experimental_backend_once", observe_warning)
    if failure:
        with pytest.raises(RuntimeError, match="injected planner failure"):
            wrapper.plan(metadata=metadata, **COMMON_PLAN_KWARGS)
        assert events == ["plan"]
        assert wrapper._planned_backend is None
    else:
        # Replanning must retain the requested auto policy after _backend changes.
        for _ in range(2):
            events.clear()
            expected_published = wrapper._planned_backend
            wrapper.plan(metadata=metadata, **COMMON_PLAN_KWARGS)
            assert wrapper._planned_backend is planned
            assert events == [
                "plan",
                (
                    "BatchMLAPagedAttentionWrapper",
                    "fa2",
                    {"automatic": backend == "auto"},
                ),
            ]
