"""Shared lazy paged-KV loader contracts for persistent attention and prefill."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest
import torch

from flashinfer.attention import _core as attention
import flashinfer.prefill as prefill
from flashinfer.jit import MissingJITCacheError
from flashinfer.jit.core import JitSpec


class _FakeSpec:
    def __init__(self, module, *, error=None, delay=0.0, path=None):
        self.module = module
        self.error = error
        self.delay = delay
        self.build_count = 0
        self.load_count = 0
        self._lock = threading.Lock()
        self.name = "test_batch_attention_independent"
        self.jit_library_path = path
        self.is_aot = False

    def build_and_load(self):
        with self._lock:
            self.build_count += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.module

    def load(self, *args):
        self.load_count += 1
        if self.error is not None:
            raise self.error
        return self.module


@pytest.fixture(params=["persistent", "prefill"])
def holder_factory(request):
    if request.param == "persistent":
        return attention._LazyBatchAttentionIndependentModule
    return prefill._LazyBatchPrefillIndependentModule


@pytest.fixture(autouse=True)
def _isolate_cuda_and_jit(monkeypatch):
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_holder_cold_capture_and_locked_capture_recheck(holder_factory, monkeypatch):
    spec = _FakeSpec(object())
    holder = holder_factory(spec)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarm_paged_kv_stride_variant"):
        holder.get()
    assert not holder.is_loaded
    assert spec.build_count == spec.load_count == 0

    # Capture may start while the caller waits for the initialization lock.
    states = iter((False, True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: next(states))
    with pytest.raises(RuntimeError, match="before capture"):
        holder.get()
    assert not holder.is_loaded
    assert spec.build_count == spec.load_count == 0


def test_holder_prewarm_once_and_warm_capture_fast_path(holder_factory, monkeypatch):
    module = object()
    spec = _FakeSpec(module)
    holder = holder_factory(spec)
    holder.prewarm()
    assert holder.is_loaded and spec.build_count == 1

    def unexpected_capture_check():
        raise AssertionError("Loaded fast path must not inspect capture state")

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", unexpected_capture_check
    )
    assert holder.get() is module
    holder.prewarm()
    assert spec.build_count == 1


def test_concurrent_first_use_loads_exactly_once(holder_factory):
    module = object()
    spec = _FakeSpec(module, delay=0.03)
    holder = holder_factory(spec)
    start = threading.Barrier(8)

    def load(_):
        start.wait(timeout=5)
        return holder.get()

    with ThreadPoolExecutor(max_workers=8) as pool:
        modules = list(pool.map(load, range(8)))
    assert all(value is module for value in modules)
    assert spec.build_count == 1


def test_failed_load_does_not_poison_holder(holder_factory):
    module = object()
    spec = _FakeSpec(module, error=RuntimeError("compiler failed"))
    holder = holder_factory(spec)
    with pytest.raises(RuntimeError, match="compiler failed"):
        holder.get()
    assert not holder.is_loaded
    spec.error = None
    assert holder.get() is module
    assert holder.is_loaded and spec.build_count == 2


def test_missing_cache_error_preserves_cause_and_actionable_guidance(holder_factory):
    missing_spec = object()
    missing = MissingJITCacheError("missing independent binary", spec=missing_spec)
    spec = _FakeSpec(object(), error=missing)
    holder = holder_factory(spec)
    with pytest.raises(MissingJITCacheError) as exc:
        holder.get()
    assert exc.value.spec is missing_spec
    assert exc.value.__cause__ is missing
    message = str(exc.value)
    assert "Unequal K/V data strides" in message
    assert "prewarm_paged_kv_stride_variant('independent')" in message
    assert "not included in the default JIT cache" in message
    assert not holder.is_loaded


class _CacheProviderSpec(JitSpec):
    """Exercise the real JitSpec no-JIT gate with a recognized provider fake."""

    def __init__(self, directory, *, cached):
        self.name = "batch_attention_independent_provider_test"
        self.directory = directory
        self.cached = cached
        self.module = object()
        self.build_count = 0
        self.load_count = 0

    @property
    def lock_path(self):
        return self.directory / "module.lock"

    @property
    def is_compiled(self):
        return self.cached

    def get_library_path(self):
        return self.directory / "module.so"

    def try_load(self):
        return self.load() if self.cached else None

    def build(self):
        self.build_count += 1

    def load(self):
        self.load_count += 1
        return self.module


def test_no_jit_recognized_cache_provider_loads_without_build(
    holder_factory, monkeypatch, tmp_path
):
    spec = _CacheProviderSpec(tmp_path, cached=True)
    holder = holder_factory(spec)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert holder.get() is spec.module
    assert holder.is_loaded
    assert spec.build_count == 0 and spec.load_count == 1


def test_no_jit_cache_miss_errors_and_can_retry(holder_factory, monkeypatch, tmp_path):
    spec = _CacheProviderSpec(tmp_path, cached=False)
    holder = holder_factory(spec)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    with pytest.raises(
        MissingJITCacheError, match="prewarm_paged_kv_stride_variant"
    ) as exc:
        holder.get()
    assert exc.value.spec is spec
    assert not holder.is_loaded and spec.build_count == spec.load_count == 0
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
    assert holder.get() is spec.module
    assert holder.is_loaded and spec.build_count == spec.load_count == 1
