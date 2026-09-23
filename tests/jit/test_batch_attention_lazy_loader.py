"""Persistent attention factory, dispatch and prewarm contracts."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashinfer.attention import _core as attention

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
)


class _FakeSpec:
    def __init__(self, module):
        self.module = module
        self.build_count = 0

    def build_and_load(self):
        self.build_count += 1
        return self.module


@pytest.fixture(autouse=True)
def _isolate_caches_and_cuda(monkeypatch):
    attention.get_holistic_attention_module.cache_clear()
    attention.get_holistic_attention_independent_module.cache_clear()
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    yield
    attention.get_holistic_attention_module.cache_clear()
    attention.get_holistic_attention_independent_module.cache_clear()


def test_factories_load_primary_but_only_construct_independent(monkeypatch):
    primary = _FakeSpec(object())
    independent = _FakeSpec(object())
    calls = []
    monkeypatch.setattr(
        attention, "_gen_batch_attention_primary_module", lambda *a: primary
    )

    def independent_generator(*args):
        calls.append(args)
        return independent

    monkeypatch.setattr(
        attention, "_gen_batch_attention_independent_module", independent_generator
    )
    assert attention.get_holistic_attention_module(*_MODULE_ARGS) is primary.module
    holder = attention.get_holistic_attention_independent_module(*_MODULE_ARGS)
    assert holder is attention.get_holistic_attention_independent_module(*_MODULE_ARGS)
    assert calls == [_MODULE_ARGS]
    assert primary.build_count == 1 and independent.build_count == 0
    assert not holder.is_loaded
    assert holder.get() is independent.module
    assert independent.build_count == 1


def _wrapper(monkeypatch):
    calls, devices = [], []

    def native(label, fill):
        def run(*args):
            # Preserve effects visible to the wrapper's caller, not only a label.
            calls.append((label, args[4], args[5], args[-2], args[-1]))
            args[7].fill_(fill)
            args[8].fill_(fill + 10)

        return SimpleNamespace(run=run)

    @contextmanager
    def device_context(device):
        devices.append(device)
        yield

    monkeypatch.setattr(torch.cuda, "device", device_context)
    wrapper = attention.BatchAttention.__new__(attention.BatchAttention)
    wrapper._kv_layout = "NHD"
    wrapper._plan_info = [1]
    wrapper._use_profiler = False
    wrapper._logits_soft_cap = 0.0
    wrapper._sm_scale = None
    wrapper._mask_mode = 0
    wrapper._num_qo_heads = 8
    wrapper._num_kv_heads = 2
    wrapper._page_size = 3
    wrapper._kv_indices = torch.tensor([0, 1], dtype=torch.int32)
    wrapper.float_workspace_buffer = torch.empty(1)
    wrapper.int_workspace_buffer = torch.empty(1, dtype=torch.int32)
    wrapper.module = native("equal", 1)
    spec = _FakeSpec(native("independent", 2))
    wrapper._independent_module = attention._LazyBatchAttentionIndependentModule(spec)
    return wrapper, spec, calls, devices


@pytest.mark.parametrize("changed_axis", [0, 1, 2])
def test_wrapper_selects_original_data_strides_each_run(monkeypatch, changed_axis):
    wrapper, spec, calls, devices = _wrapper(monkeypatch)
    q = torch.empty(1, 8, 8)
    shape, equal_strides = (2, 3, 2, 8), (256, 64, 16, 1)
    unequal_strides = list(equal_strides)
    unequal_strides[changed_axis] += 8
    k = torch.empty_strided(shape, equal_strides)
    equal_v = torch.empty_strided(shape, equal_strides)
    unequal_v = torch.empty_strided(shape, tuple(unequal_strides))
    sfk = torch.empty_strided(shape, equal_strides, dtype=torch.uint8)
    sfv = torch.empty_strided(shape, (512, 64, 16, 1), dtype=torch.uint8)
    for v, expected in ((equal_v, 1), (unequal_v, 2), (equal_v, 1)):
        out, lse = wrapper.run(q, (k, v), kv_cache_sf=(sfk, sfv))
        assert torch.equal(out, torch.full_like(out, expected))
        assert torch.equal(lse, torch.full_like(lse, expected + 10))
    assert [call[0] for call in calls] == ["equal", "independent", "equal"]
    assert all(call[3] is sfk and call[4] is sfv for call in calls)
    assert spec.build_count == 1
    assert devices and all(device == q.device for device in devices)


def test_stride_predicate_does_not_narrow_large_original_values(monkeypatch):
    wrapper, spec, calls, _ = _wrapper(monkeypatch)
    q = torch.empty(1, 8, 8)
    # Singleton outer dimensions allow huge metadata strides without huge storage.
    k = torch.empty_strided((1, 1, 1, 8), (64, 32, 16, 1))
    v = torch.empty_strided((1, 1, 1, 8), (64 + 2**32, 32, 16, 1))
    wrapper.run(q, (k, v))
    assert calls[0][0] == "independent" and spec.build_count == 1


def test_prewarm_requires_plan_and_valid_variant(monkeypatch):
    unplanned = attention.BatchAttention.__new__(attention.BatchAttention)
    with pytest.raises(RuntimeError, match=r"plan\(\)"):
        unplanned.prewarm_paged_kv_stride_variant()
    wrapper, spec, _, devices = _wrapper(monkeypatch)
    for invalid in ("equal", "unsupported"):
        with pytest.raises(ValueError, match="variant"):
            wrapper.prewarm_paged_kv_stride_variant(invalid)
    assert not wrapper._independent_module.is_loaded and spec.build_count == 0
    wrapper.prewarm_paged_kv_stride_variant("independent")
    assert wrapper._independent_module.is_loaded and spec.build_count == 1
    assert devices and all(
        device == wrapper.float_workspace_buffer.device for device in devices
    )
