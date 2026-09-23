"""Shared lazy paged-KV loader contracts for persistent attention and prefill."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import threading
import time

import pytest
import torch

from flashinfer.attention import _core as attention
import flashinfer.prefill as prefill
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.jit import MissingJITCacheError
from flashinfer.jit.core import JitSpec
from flashinfer.jit import core as jit_core, env as jit_env


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
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())


@pytest.fixture
def heterogeneous_provider(monkeypatch, tmp_path):
    """Use real provider selection and fallback with two simulated CUDA devices."""
    state = SimpleNamespace(device=0, capturing=False, builds=0)
    architectures = {0: "sm90a", 1: "sm100a"}

    @contextmanager
    def device_context(device):
        previous = state.device
        index = (
            device
            if device is None or isinstance(device, int)
            else torch.device(device).index
        )
        if index is not None:
            state.device = index
        try:
            yield
        finally:
            state.device = previous

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state.device)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(
        jit_env, "_cuda_architecture_for_device", lambda index: architectures[index]
    )
    monkeypatch.setattr(
        jit_env, "_target_cuda_architectures", lambda: frozenset(architectures.values())
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    artifact = jit_env.AOTArtifact(
        "hopper", tmp_path / "hopper.so", frozenset({"sm90a"})
    )
    monkeypatch.setattr(jit_env, "get_aot_artifacts", lambda name: (artifact,))
    provider = SimpleNamespace(run=lambda output: output.fill_(9))
    fallback = SimpleNamespace(run=lambda output: output.fill_(10))
    spec = jit_core.JitSpecNvcc(
        name="test_independent_heterogeneous",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )
    monkeypatch.setattr(spec, "load", lambda path=None: provider if path else fallback)
    original_build = spec.build

    def build(*args, **kwargs):
        import os

        if os.environ.get("FLASHINFER_DISABLE_JIT"):
            return original_build(*args, **kwargs)
        assert not state.capturing, "fallback compilation reached CUDA capture"
        state.builds += 1

    monkeypatch.setattr(spec, "build", build)
    return state, spec


@pytest.mark.parametrize("disable_jit", [False, True])
def test_heterogeneous_prewarm_resolves_device(
    holder_factory, heterogeneous_provider, monkeypatch, disable_jit
):
    state, spec = heterogeneous_provider
    holder = holder_factory(spec)
    if disable_jit:
        monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    with torch.cuda.device(1):
        if disable_jit:
            with pytest.raises(
                MissingJITCacheError, match="prewarm_paged_kv_stride_variant"
            ):
                holder.prewarm()
            assert not holder.is_loaded
            monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
        holder.prewarm()
        assert state.builds == 1
        state.capturing = True
        output = torch.empty(1)
        holder.get().run(output)
        assert output.item() == 10
        assert state.builds == 1


def test_heterogeneous_capture_requires_each_device_prewarmed(
    holder_factory, heterogeneous_provider, monkeypatch
):
    state, spec = heterogeneous_provider
    holder = holder_factory(spec)
    holder.prewarm(torch.device("cuda:0"))
    assert state.builds == 0
    state.capturing = True
    output = torch.empty(1)
    holder.get(torch.device("cuda:0")).run(output)
    assert output.item() == 9
    with pytest.raises(RuntimeError, match="prewarm_paged_kv_stride_variant"):
        holder.get(torch.device("cuda:1")).run(output)
    assert state.builds == 0
    state.capturing = False
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    with pytest.raises(MissingJITCacheError, match="prewarm_paged_kv_stride_variant"):
        holder.prewarm(torch.device("cuda:1"))
    # A failed second device must not poison the already prepared provider.
    holder.get(torch.device("cuda:0")).run(output)
    assert output.item() == 9
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
    holder.prewarm(torch.device("cuda:1"))
    assert state.device == 0 and state.builds == 1
    state.capturing = True
    holder.get(torch.device("cuda:1")).run(output)
    assert output.item() == 10
    holder.get(torch.device("cuda:0")).run(output)
    assert output.item() == 9 and state.builds == 1


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        attention.BatchAttention,
        prefill.BatchPrefillWithPagedKVCacheWrapper,
        BatchDecodeWithPagedKVCacheWrapper,
    ],
)
def test_wrapper_prewarm_resolves_its_device_after_another_device_is_warm(
    wrapper_cls, heterogeneous_provider
):
    state, spec = heterogeneous_provider
    holder = attention._LazyBatchAttentionIndependentModule(spec)
    holder.prewarm(torch.device("cuda:0"))
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper._plan_info = [1]
    wrapper.device = torch.device("cuda:1")
    if wrapper_cls is attention.BatchAttention:
        wrapper.float_workspace_buffer = SimpleNamespace(device=wrapper.device)
        wrapper._independent_module = holder
    else:
        wrapper._cached_module = SimpleNamespace(
            prewarm_paged_kv_stride_variant=lambda variant: holder.prewarm()
        )
        wrapper._jit_module = None
        wrapper._backend = "fa2"
        wrapper._use_tensor_cores = True
    wrapper.prewarm_paged_kv_stride_variant()
    assert state.builds == 1 and state.device == 0


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
