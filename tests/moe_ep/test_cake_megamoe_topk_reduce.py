"""Behavioural tests for the Cake MegaMoE workspace TopK reducer.

Host-side tests run without a GPU: device selection follows the exact compute
capability, the capability is queried once per device, and the one generated
source carries exactly the four CTA shapes the binding launches.  The GPU tests
(``arch_blackwell``) check the reducer's results bit for bit against an ordered
FP32 reference on every token tier and tier boundary, at arbitrary workspace
capacities and at 128-byte aligned inputs inside larger buffers; that each live
token count launches the CTA shape its architecture's tiers pick; and the
argument checks of the binding.
"""

from __future__ import annotations

import re

import pytest
import torch

from flashinfer.jit import cake_megamoe_topk_reduce as reducer
from tests.test_helpers.cake_profiler import cuda_kernel_names

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
_KERNEL_PREFIX = "kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_"
# CTA shape -> threads per CTA, in the order of reducer._KERNEL_SYMBOLS.
_SHAPE_THREADS = {
    "t512x2": 512,
    "t512x4": 512,
    "t512x4_pf256": 512,
    "t1024x4_pf256": 1024,
}
# Live-token tiers of each architecture: (largest num_tokens of the tier, CTA
# shape); the last tier has no bound.
_TIERS = {
    "sm_100a": (
        (48, "t512x2"),
        (96, "t512x4"),
        (2048, "t512x4_pf256"),
        (None, "t1024x4_pf256"),
    ),
    "sm_103a": (
        (48, "t512x2"),
        (96, "t512x4"),
        (192, "t512x4_pf256"),
        (256, "t1024x4_pf256"),
        (1024, "t512x4_pf256"),
        (None, "t1024x4_pf256"),
    ),
}
# Both architectures' tier boundaries: the last and first token count of each tier.
_TIER_EDGES = (48, 49, 96, 97, 192, 193, 256, 257, 1024, 1025, 2048, 2049)


def _expected_kernel(arch: str, num_tokens: int) -> str:
    for bound, shape in _TIERS[arch]:
        if bound is None or num_tokens <= bound:
            return _KERNEL_PREFIX + shape
    raise AssertionError("the last tier has no bound")


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


def test_each_module_compiles_its_own_token_tiers():
    for arch, define in (("sm_100a", "100"), ("sm_103a", "103")):
        flags = reducer._ARCHS[arch]["nvcc_flags"]
        assert f"-DCAKE_MEGAMOE_TOPK_REDUCE_ARCH={define}" in flags


def test_tier_tables_cover_every_token_count():
    assert tuple(_KERNEL_PREFIX + shape for shape in _SHAPE_THREADS) == (
        reducer._KERNEL_SYMBOLS
    )
    for tiers in _TIERS.values():
        bounds = [bound for bound, _ in tiers]
        assert bounds[-1] is None
        assert bounds[:-1] == sorted(bounds[:-1])
        assert {shape for _, shape in tiers} == set(_SHAPE_THREADS)


def test_one_generated_source_carries_every_cta_shape():
    csrc_dir = reducer._get_csrc_dir()
    entries = sorted(path.name for path in csrc_dir.iterdir())
    assert entries == [reducer._BINDING_HEADER, reducer._SOURCE_FILE]
    source = (csrc_dir / reducer._SOURCE_FILE).read_text(encoding="utf-8")
    binding = (csrc_dir / reducer._BINDING_HEADER).read_text(encoding="utf-8")
    assert f'#include "{reducer._SOURCE_FILE}"' in binding
    assert source.count("__global__") == len(reducer._KERNEL_SYMBOLS)
    for symbol, threads in zip(
        reducer._KERNEL_SYMBOLS, _SHAPE_THREADS.values(), strict=True
    ):
        # One definition per shape, with its CTA size, launched by the binding.
        assert len(re.findall(rf"\b{symbol}\(", source)) == 1
        assert f"__launch_bounds__({threads}) void\n{symbol}(" in source
        assert f"{symbol}<<<" in binding
    # Arch-neutral: no SM100-only or SM103-only instructions.
    assert "tcgen05" not in source
    assert "mbarrier" not in source
    # Every device helper in the source is called by a kernel.
    helpers = re.findall(r"^__device__ __forceinline__ \w+ (\w+)\(", source, re.M)
    assert helpers
    kernels = source[source.index(reducer._KERNEL_SYMBOLS[0]) :]
    unused = [name for name in helpers if f"{name}(" not in kernels]
    assert unused == []


def _ordered_reference(partials: torch.Tensor, num_tokens: int) -> torch.Tensor:
    acc = partials[:num_tokens, 0].float()
    for k in range(1, _TOP_K):
        acc = acc + partials[:num_tokens, k].float()
    return acc.to(torch.bfloat16)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16)


def _require_reducer_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in reducer.supported_capabilities():
        pytest.skip("reducer requires exact SM100a or SM103a")
    return device


def _random_partials(
    capacity: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(
        capacity,
        _TOP_K,
        _HIDDEN,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )


def _profiled_kernel_names(fn) -> list[str]:
    """CUDA kernel names of one call of ``fn`` (memcpy / memset records dropped), read
    from a sentinel-bracketed profiler session: a session that loses its device records
    (``CUPTI-LOST-RECORDS``) is retried, and the test skips rather than reporting a
    launch it cannot see."""
    return cuda_kernel_names(fn)


@pytest.mark.arch_blackwell
class TestReduceOnDevice:
    @pytest.mark.parametrize(
        ("num_tokens", "capacity"),
        _SHAPES + tuple((num_tokens, 4096) for num_tokens in _TIER_EDGES),
    )
    def test_matches_ordered_reference_and_leaves_idle_rows(self, num_tokens, capacity):
        device = _require_reducer_device()
        generator = torch.Generator(device=device).manual_seed(
            num_tokens * 7919 + capacity
        )
        partials = _random_partials(capacity, device, generator)
        out = torch.full(
            (capacity, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        )
        reducer.run_cake_megamoe_topk_reduce(partials, out, num_tokens)
        torch.cuda.synchronize()
        expected = _ordered_reference(partials, num_tokens)
        # The kernel accumulates the six BF16 rows in FP32 in K order and
        # rounds once, exactly like the reference: results are bitwise equal.
        assert torch.equal(_bits(out[:num_tokens]), _bits(expected))
        assert torch.isnan(out[num_tokens:]).all()
        assert reducer.is_cake_megamoe_topk_reduce_module_loaded(device)
        assert reducer.supports_device(device)

    @pytest.mark.parametrize("num_tokens", (1, 37, 97, 256, 1025))
    @pytest.mark.parametrize("offset_bytes", (128, 384))
    def test_matches_at_aligned_offsets_inside_larger_buffers(
        self, num_tokens, offset_bytes
    ):
        # 128-byte aligned (not 256-byte aligned) views; the bytes around them
        # must stay untouched.
        device = _require_reducer_device()
        capacity = num_tokens + 3
        offset = offset_bytes // 2
        generator = torch.Generator(device=device).manual_seed(num_tokens + offset)
        partials_numel = capacity * _TOP_K * _HIDDEN
        partials_storage = torch.randn(
            partials_numel + 2 * offset,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).bfloat16()
        out_storage = torch.full(
            (capacity * _HIDDEN + 2 * offset,),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        partials = partials_storage[offset : offset + partials_numel].view(
            capacity, _TOP_K, _HIDDEN
        )
        out = out_storage[offset : offset + capacity * _HIDDEN].view(capacity, _HIDDEN)
        assert partials.data_ptr() % 128 == 0 and partials.data_ptr() % 256 != 0
        assert out.data_ptr() % 128 == 0 and out.data_ptr() % 256 != 0
        partials_before = partials_storage.clone()
        reducer.run_cake_megamoe_topk_reduce(partials, out, num_tokens)
        torch.cuda.synchronize()
        expected = _ordered_reference(partials, num_tokens)
        assert torch.equal(_bits(out[:num_tokens]), _bits(expected))
        assert torch.isnan(out[num_tokens:]).all()
        assert torch.isnan(out_storage[:offset]).all()
        assert torch.isnan(out_storage[offset + capacity * _HIDDEN :]).all()
        assert torch.equal(_bits(partials_storage), _bits(partials_before))

    @pytest.mark.parametrize("num_tokens", (1, 8, 64, 128, 256, 4096) + _TIER_EDGES)
    def test_launches_the_cta_shape_of_its_token_tier(self, num_tokens):
        device = _require_reducer_device()
        arch = reducer.resolve_arch(device)
        partials = torch.zeros(
            num_tokens, _TOP_K, _HIDDEN, dtype=torch.bfloat16, device=device
        )
        out = torch.empty(num_tokens, _HIDDEN, dtype=torch.bfloat16, device=device)
        reducer.run_cake_megamoe_topk_reduce(partials, out, num_tokens)
        torch.cuda.synchronize()
        names = _profiled_kernel_names(
            lambda: reducer.run_cake_megamoe_topk_reduce(partials, out, num_tokens)
        )
        assert names == [_expected_kernel(arch, num_tokens)]

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

    def test_rejects_misaligned_buffers(self):
        device = _require_reducer_device()
        capacity = 8
        # 64-byte offsets: 32 BF16 elements past a 128-byte aligned allocation.
        partials_storage = torch.zeros(
            capacity * _TOP_K * _HIDDEN + 32, dtype=torch.bfloat16, device=device
        )
        out_storage = torch.zeros(
            capacity * _HIDDEN + 32, dtype=torch.bfloat16, device=device
        )
        partials = partials_storage[: capacity * _TOP_K * _HIDDEN].view(
            capacity, _TOP_K, _HIDDEN
        )
        out = out_storage[: capacity * _HIDDEN].view(capacity, _HIDDEN)
        shifted_partials = partials_storage[32:].view(capacity, _TOP_K, _HIDDEN)
        shifted_out = out_storage[32:].view(capacity, _HIDDEN)
        with pytest.raises(Exception, match="partials must be 128-byte aligned"):
            reducer.run_cake_megamoe_topk_reduce(shifted_partials, out, capacity)
        with pytest.raises(Exception, match="out must be 128-byte aligned"):
            reducer.run_cake_megamoe_topk_reduce(partials, shifted_out, capacity)

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
