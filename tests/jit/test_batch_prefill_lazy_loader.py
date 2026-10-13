from types import SimpleNamespace
from contextlib import contextmanager, nullcontext

import pytest
import torch

import flashinfer.prefill as prefill
from flashinfer.cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


_MODULE_ARGS = (
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.int32,
    128,
    128,
    0,
    False,
    False,
    False,
)


class _FakeSpec:
    def __init__(self, module):
        self.module = module
        self.build_count = 0

    def build_and_load(self):
        self.build_count += 1
        return self.module


def _raw_module(label, calls):
    def paged_run(*args):
        calls.append(label)

    return SimpleNamespace(
        plan=lambda *args: [1],
        workspace_size=lambda *args: 0,
        ragged_run=lambda *args: None,
        paged_run=paged_run,
    )


def _run_paged(module, paged_k_cache, paged_v_cache, **kwargs):
    tensor = torch.empty(1)
    return module.paged_run(
        tensor,
        tensor,
        [1],
        tensor,
        paged_k_cache,
        paged_v_cache,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        None,
        0,
        0,
        -1,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        0.0,
        1.0,
        None,
        None,
        None,
        1.0,
        1.0,
        0,
        0,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clear_batch_prefill_module_cache():
    prefill.get_batch_prefill_module.cache_clear()
    yield
    prefill.get_batch_prefill_module.cache_clear()


def test_fa2_selects_stride_variant_once(monkeypatch):
    calls = []
    primary_spec = _FakeSpec(_raw_module("primary", calls))
    independent_spec = _FakeSpec(_raw_module("independent", calls))
    monkeypatch.setattr(
        prefill, "_gen_batch_prefill_primary_module", lambda *args: primary_spec
    )
    monkeypatch.setattr(
        prefill,
        "_gen_batch_prefill_independent_paged_module",
        lambda *args: independent_spec,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    module = prefill.get_batch_prefill_module("fa2", *_MODULE_ARGS)
    assert primary_spec.build_count == 1
    assert independent_spec.build_count == 0

    shape = (2, 3, 4, 5)
    equal_k = torch.empty_strided(shape, (60, 20, 5, 1))
    equal_v = torch.empty_strided(shape, (60, 20, 5, 1))
    unequal_v = torch.empty_strided(shape, (80, 20, 5, 1))
    # Scale-factor layouts must not trigger the independent data-stride module.
    scales = dict(
        key_block_scales=torch.empty_strided(shape, (60, 20, 5, 1), dtype=torch.uint8),
        value_block_scales=torch.empty_strided(
            shape, (80, 20, 5, 1), dtype=torch.uint8
        ),
    )
    _run_paged(module, equal_k, equal_v, **scales)
    assert calls == ["primary"]
    assert independent_spec.build_count == 0
    _run_paged(module, equal_k, unequal_v)
    _run_paged(module, equal_k, equal_v)

    assert calls == ["primary", "independent", "primary"]
    assert independent_spec.build_count == 1
    module.prewarm_paged_kv_stride_variant("independent")
    assert independent_spec.build_count == 1
    with pytest.raises(ValueError, match="variant must be 'independent'"):
        module.prewarm_paged_kv_stride_variant("unknown")


def test_non_fa2_keeps_public_full_module_and_rejects_prewarm(monkeypatch):
    calls = []
    public_spec = _FakeSpec(_raw_module("public", calls))
    monkeypatch.setattr(prefill, "gen_batch_prefill_module", lambda *args: public_spec)

    def unexpected_specialized_generator(*args):
        raise AssertionError("non-FA2 must not construct a stride-specialized module")

    monkeypatch.setattr(
        prefill, "_gen_batch_prefill_primary_module", unexpected_specialized_generator
    )
    monkeypatch.setattr(
        prefill,
        "_gen_batch_prefill_independent_paged_module",
        unexpected_specialized_generator,
    )

    module = prefill.get_batch_prefill_module("fa3", *_MODULE_ARGS)
    shape = (2, 3, 4, 5)
    _run_paged(
        module,
        torch.empty_strided(shape, (60, 20, 5, 1)),
        torch.empty_strided(shape, (80, 20, 5, 1)),
    )
    assert calls == ["public"]
    assert public_spec.build_count == 1
    with pytest.raises(RuntimeError, match="only for standard FA2"):
        module.prewarm_paged_kv_stride_variant()


class _PrewarmRecorder:
    def __init__(self):
        self.variants = []

    def prewarm_paged_kv_stride_variant(self, variant):
        self.variants.append(variant)


@pytest.mark.parametrize(
    "wrapper_cls",
    [prefill.BatchPrefillWithPagedKVCacheWrapper, BatchDecodeWithPagedKVCacheWrapper],
)
def test_standard_wrappers_expose_post_plan_prewarm(wrapper_cls, monkeypatch):
    devices = []

    @contextmanager
    def device_context(device):
        devices.append(device)
        yield

    monkeypatch.setattr(torch.cuda, "device", device_context)
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.device = torch.device("cuda:1")
    wrapper._plan_info = [1]
    wrapper._cached_module = _PrewarmRecorder()
    wrapper._jit_module = None
    wrapper._backend = "fa2"
    wrapper._use_tensor_cores = True

    wrapper.prewarm_paged_kv_stride_variant("independent")
    assert wrapper._cached_module.variants == ["independent"]
    assert devices == [torch.device("cuda:1")]

    wrapper._plan_info = None
    with pytest.raises(RuntimeError, match=r"plan\(\) must complete"):
        wrapper.prewarm_paged_kv_stride_variant()


def test_cascade_wrappers_delegate_prewarm():
    level_one = _PrewarmRecorder()
    level_two = _PrewarmRecorder()
    multi_level = MultiLevelCascadeAttentionWrapper.__new__(
        MultiLevelCascadeAttentionWrapper
    )
    multi_level._batch_prefill_wrappers = [level_one, level_two]
    multi_level.prewarm_paged_kv_stride_variant("independent")
    assert level_one.variants == ["independent"]
    assert level_two.variants == ["independent"]

    prefill_child = _PrewarmRecorder()
    shared_prefill = BatchPrefillWithSharedPrefixPagedKVCacheWrapper.__new__(
        BatchPrefillWithSharedPrefixPagedKVCacheWrapper
    )
    shared_prefill._batch_prefill_wrapper = prefill_child
    shared_prefill.prewarm_paged_kv_stride_variant("independent")
    assert prefill_child.variants == ["independent"]
