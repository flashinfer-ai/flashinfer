"""Behavioural tests for the Cake MegaMoE workspace TopK reducer.

Host-side tests run without a GPU: device selection follows the exact compute
capability, the capability is queried once per device, and the single
generated source carries exactly the kernel it launches.  The GPU tests
(``arch_blackwell``) check the reducer's results against an ordered FP32
reference, its handling of arbitrary workspace capacities and live token
counts, and the argument checks of the binding.
"""

from __future__ import annotations

import re

import pytest
import torch

from flashinfer.jit import cake_megamoe_topk_reduce as reducer

_ARCHS = ("sm_100a", "sm_103a")
_CAPABILITIES = {"sm_100a": (10, 0), "sm_103a": (10, 3)}
_HIDDEN = 4096
_TOP_K = 6
# Deployment shapes (num_tokens, capacity) plus capacities outside {256, 4096}.
_SHAPES = (
    (1, 256),
    (8, 256),
    (64, 256),
    (128, 256),
    (256, 256),
    (4096, 4096),
    (37, 64),
    (100, 300),
    (1000, 4096),
)


@pytest.fixture
def fresh_capability_cache():
    reducer._device_capability.cache_clear()
    yield
    reducer._device_capability.cache_clear()


def test_supported_capabilities_are_exact_blackwell_targets():
    assert reducer.supported_capabilities() == ((10, 0), (10, 3))
    assert tuple(reducer._ARCHS) == _ARCHS


@pytest.mark.parametrize("arch", _ARCHS)
def test_resolve_arch_maps_exact_capability(monkeypatch, fresh_capability_cache, arch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: _CAPABILITIES[arch]
    )
    assert reducer.resolve_arch() == arch
    assert reducer.resolve_arch(torch.device("cuda", 0)) == arch
    assert reducer.supports_device() is True
    assert reducer.is_cake_megamoe_topk_reduce_module_loaded() is (
        arch in reducer._LOADED_MODULES
    )


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_resolve_arch_rejects_other_capabilities(
    monkeypatch, fresh_capability_cache, capability
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    with pytest.raises(NotImplementedError, match="published for compute capabilities"):
        reducer.resolve_arch()
    assert reducer.supports_device() is False
    assert reducer.is_cake_megamoe_topk_reduce_module_loaded() is False


def test_capability_is_queried_once_per_device(monkeypatch, fresh_capability_cache):
    calls: list[int] = []

    def fake_capability(device=None):
        calls.append(device)
        return (10, 0)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_capability)
    for _ in range(3):
        assert reducer.resolve_arch(0) == "sm_100a"
        assert reducer.supports_device(torch.device("cuda", 0)) is True
        assert reducer.resolve_arch() == "sm_100a"
    assert calls == [0]
    assert reducer.resolve_arch(1) == "sm_100a"
    assert calls == [0, 1]


def test_device_index_normalisation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert reducer._device_index(None) == 2
    assert reducer._device_index(3) == 3
    assert reducer._device_index(torch.device("cuda")) == 2
    assert reducer._device_index(torch.device("cuda", 1)) == 1
    assert reducer._device_index("cuda:4") == 4


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="unknown MegaMoE TopK reducer arch"):
        reducer.get_cake_megamoe_topk_reduce_uri("sm_90a")
    with pytest.raises(ValueError, match="unknown MegaMoE TopK reducer arch"):
        reducer.gen_cake_megamoe_topk_reduce_module("sm_90a")


def test_module_uri_is_arch_specific():
    uris = {arch: reducer.get_cake_megamoe_topk_reduce_uri(arch) for arch in _ARCHS}
    assert uris == {
        "sm_100a": "cake_megamoe_topk_reduce_sm100a",
        "sm_103a": "cake_megamoe_topk_reduce_sm103a",
    }


def test_single_generated_source_serves_every_arch():
    csrc_dir = reducer._get_csrc_dir()
    entries = sorted(path.name for path in csrc_dir.iterdir())
    assert entries == [reducer._BINDING_HEADER, reducer._SOURCE_FILE]
    source = (csrc_dir / reducer._SOURCE_FILE).read_text(encoding="utf-8")
    assert source.count(reducer._KERNEL_SYMBOL) == 1
    assert "__launch_bounds__(256)" in source
    # Arch-neutral: no SM100-only or SM103-only instructions.
    assert "tcgen05" not in source
    assert "mbarrier" not in source
    # Every device helper in the source is called by the kernel.
    helpers = re.findall(r"^__device__ __forceinline__ \w+ (\w+)\(", source, re.M)
    assert helpers
    kernel_body = source[source.index(reducer._KERNEL_SYMBOL) :]
    unused = [name for name in helpers if f"{name}(" not in kernel_body]
    assert unused == []
    binding = (csrc_dir / reducer._BINDING_HEADER).read_text(encoding="utf-8")
    assert f'#include "{reducer._SOURCE_FILE}"' in binding
    assert f"{reducer._KERNEL_SYMBOL}<<<" in binding


def _ordered_reference(partials: torch.Tensor, num_tokens: int) -> torch.Tensor:
    acc = partials[:num_tokens, 0].float()
    for k in range(1, _TOP_K):
        acc = acc + partials[:num_tokens, k].float()
    return acc.to(torch.bfloat16)


def _require_reducer_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in reducer.supported_capabilities():
        pytest.skip("reducer requires exact SM100a or SM103a")
    return device


@pytest.mark.arch_blackwell
class TestReduceOnDevice:
    @pytest.mark.parametrize(("num_tokens", "capacity"), _SHAPES)
    def test_matches_ordered_reference_and_leaves_idle_rows(self, num_tokens, capacity):
        device = _require_reducer_device()
        generator = torch.Generator(device=device).manual_seed(
            num_tokens * 7919 + capacity
        )
        partials = torch.randn(
            capacity,
            _TOP_K,
            _HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        out = torch.full(
            (capacity, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        )
        reducer.run_cake_megamoe_topk_reduce(partials, out, num_tokens)
        torch.cuda.synchronize()
        expected = _ordered_reference(partials, num_tokens)
        # The kernel accumulates the six BF16 rows in FP32 in K order and
        # rounds once, exactly like the reference: results are bitwise equal.
        assert torch.equal(out[:num_tokens], expected)
        assert torch.isnan(out[num_tokens:]).all()
        assert reducer.is_cake_megamoe_topk_reduce_module_loaded(device)
        assert reducer.supports_device(device)

    def test_zero_tokens_is_a_no_op(self):
        device = _require_reducer_device()
        partials = torch.randn(16, _TOP_K, _HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.full((16, _HIDDEN), 3.0, dtype=torch.bfloat16, device=device)
        reducer.run_cake_megamoe_topk_reduce(partials, out, 0)
        torch.cuda.synchronize()
        assert torch.equal(out, torch.full_like(out, 3.0))

    def test_rejects_num_tokens_above_capacity(self):
        device = _require_reducer_device()
        partials = torch.zeros(8, _TOP_K, _HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.zeros(8, _HIDDEN, dtype=torch.bfloat16, device=device)
        with pytest.raises(Exception, match="num_tokens must be in"):
            reducer.run_cake_megamoe_topk_reduce(partials, out, 9)

    def test_rejects_mismatched_shapes(self):
        device = _require_reducer_device()
        out = torch.zeros(8, _HIDDEN, dtype=torch.bfloat16, device=device)
        narrow = torch.zeros(
            8, _TOP_K, _HIDDEN // 2, dtype=torch.bfloat16, device=device
        )
        with pytest.raises(Exception, match="partials must have shape"):
            reducer.run_cake_megamoe_topk_reduce(narrow, out, 8)
        partials = torch.zeros(8, _TOP_K, _HIDDEN, dtype=torch.bfloat16, device=device)
        short = torch.zeros(4, _HIDDEN, dtype=torch.bfloat16, device=device)
        with pytest.raises(Exception, match="out must have shape"):
            reducer.run_cake_megamoe_topk_reduce(partials, short, 4)

    def test_rejects_overlapping_buffers(self):
        device = _require_reducer_device()
        capacity = 8
        storage = torch.zeros(
            capacity * _TOP_K * _HIDDEN + capacity * _HIDDEN,
            dtype=torch.bfloat16,
            device=device,
        )
        partials = storage[: capacity * _TOP_K * _HIDDEN].view(
            capacity, _TOP_K, _HIDDEN
        )
        # 128-byte aligned window that starts inside ``partials``.
        out = storage[2 * _HIDDEN : 2 * _HIDDEN + capacity * _HIDDEN].view(
            capacity, _HIDDEN
        )
        with pytest.raises(Exception, match="must not overlap"):
            reducer.run_cake_megamoe_topk_reduce(partials, out, capacity)
