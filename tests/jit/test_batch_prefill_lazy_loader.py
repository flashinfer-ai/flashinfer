from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest
import torch

import flashinfer.prefill as prefill
from flashinfer.cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.jit import MissingJITCacheError


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
    def __init__(self, module, *, error=None, delay_seconds=0.0):
        self.module = module
        self.error = error
        self.delay_seconds = delay_seconds
        self.build_count = 0
        self._counter_lock = threading.Lock()

    def build_and_load(self):
        with self._counter_lock:
            self.build_count += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
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


def _run_paged(module, paged_k_cache, paged_v_cache):
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
    )


@pytest.fixture(autouse=True)
def _clear_batch_prefill_module_cache():
    prefill.get_batch_prefill_module.cache_clear()
    yield
    prefill.get_batch_prefill_module.cache_clear()


def test_lazy_holder_capture_order_warm_fast_path_and_no_jit(monkeypatch):
    loaded_module = object()
    spec = _FakeSpec(loaded_module)
    holder = prefill._LazyBatchPrefillIndependentModule(spec)
    capture_states = iter((False, True))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: next(capture_states)
    )

    with pytest.raises(RuntimeError, match="before capture"):
        holder.get()
    assert spec.build_count == 0
    assert not holder.is_loaded

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    holder.prewarm()
    assert holder.is_loaded
    assert spec.build_count == 1

    def unexpected_capture_check():
        raise AssertionError("warm fast path must not query capture state")

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", unexpected_capture_check
    )
    assert holder.get() is loaded_module
    assert spec.build_count == 1

    missing_spec = object()
    no_jit_spec = _FakeSpec(
        object(),
        error=MissingJITCacheError("generic cache miss", spec=missing_spec),
    )
    no_jit_holder = prefill._LazyBatchPrefillIndependentModule(no_jit_spec)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(MissingJITCacheError) as exc_info:
        no_jit_holder.get()
    assert exc_info.value.spec is missing_spec
    assert isinstance(exc_info.value.__cause__, MissingJITCacheError)
    message = str(exc_info.value)
    assert "Unequal K/V data strides" in message
    assert "not included in the default JIT cache" in message
    assert "prewarm_paged_kv_stride_variant('independent')" in message


def test_concurrent_first_use_loads_once(monkeypatch):
    loaded_module = object()
    spec = _FakeSpec(loaded_module, delay_seconds=0.05)
    holder = prefill._LazyBatchPrefillIndependentModule(spec)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        modules = list(pool.map(lambda _: holder.get(), range(8)))

    assert all(module is loaded_module for module in modules)
    assert spec.build_count == 1


def test_fa2_routes_complete_data_strides_under_one_logical_operation(monkeypatch):
    calls = []
    primary_spec = _FakeSpec(_raw_module("primary", calls))
    independent_spec = _FakeSpec(_raw_module("independent", calls))
    registered_ops = []

    def record_custom_op(name, *args, **kwargs):
        def decorator(func):
            registered_ops.append(name)
            return func

        return decorator

    monkeypatch.setattr(prefill, "register_custom_op", record_custom_op)
    monkeypatch.setattr(
        prefill, "register_fake_op", lambda *args, **kwargs: lambda f: f
    )
    monkeypatch.setattr(
        prefill, "_gen_batch_prefill_primary_module", lambda *args: primary_spec
    )
    monkeypatch.setattr(
        prefill,
        "_gen_batch_prefill_independent_paged_module",
        lambda *args: independent_spec,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    module = prefill.get_batch_prefill_module("fa2", *_MODULE_ARGS)
    logical_uri = prefill.get_batch_prefill_uri("fa2", *_MODULE_ARGS)
    assert registered_ops == [
        f"flashinfer::{logical_uri}_ragged_run",
        f"flashinfer::{logical_uri}_paged_run",
    ]
    assert primary_spec.build_count == 1
    assert independent_spec.build_count == 0

    shape = (2, 3, 4, 5)
    equal_k = torch.empty_strided(shape, (60, 20, 5, 1))
    equal_v = torch.empty_strided(shape, (60, 20, 5, 1))
    unequal_v = torch.empty_strided(shape, (80, 20, 5, 1))
    _run_paged(module, equal_k, equal_v)
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


def test_standard_wrappers_expose_post_plan_prewarm():
    cached_modules = []

    prefill_wrapper = prefill.BatchPrefillWithPagedKVCacheWrapper.__new__(
        prefill.BatchPrefillWithPagedKVCacheWrapper
    )
    prefill_wrapper._plan_info = [1]
    prefill_wrapper._cached_module = _PrewarmRecorder()
    prefill_wrapper._jit_module = None
    prefill_wrapper._backend = "fa2"
    cached_modules.append(prefill_wrapper._cached_module)

    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper.__new__(
        BatchDecodeWithPagedKVCacheWrapper
    )
    decode_wrapper._plan_info = [1]
    decode_wrapper._cached_module = _PrewarmRecorder()
    decode_wrapper._jit_module = None
    decode_wrapper._backend = "fa2"
    decode_wrapper._use_tensor_cores = True
    cached_modules.append(decode_wrapper._cached_module)

    for wrapper in (prefill_wrapper, decode_wrapper):
        wrapper.prewarm_paged_kv_stride_variant("independent")

    assert all(module.variants == ["independent"] for module in cached_modules)

    unplanned = prefill.BatchPrefillWithPagedKVCacheWrapper.__new__(
        prefill.BatchPrefillWithPagedKVCacheWrapper
    )
    unplanned._plan_info = None
    unplanned._cached_module = None
    unplanned._jit_module = None
    unplanned._backend = "fa2"
    with pytest.raises(RuntimeError, match=r"plan\(\) must complete"):
        unplanned.prewarm_paged_kv_stride_variant()


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
