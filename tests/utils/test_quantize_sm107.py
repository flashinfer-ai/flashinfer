"""Regression coverage for SM107 quantization dispatch (not quantization math)."""

import pytest
import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer import utils


@pytest.mark.parametrize("is_sm107", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("sf_layout", [0, 1, 2])
@pytest.mark.parametrize(
    "m,k,eligible_layouts",
    [
        (1024, 16384, ()),
        (1536, 4096, ()),
        (1536, 6144, ()),
        (1536, 8192, (0,)),
        (2048, 4096, (0,)),
        (2048, 16384, (0,)),
        (3072, 6144, (0, 1)),
        (4096, 4096, (0, 1, 2)),
        (4096, 8192, (0, 1, 2)),
        (4097, 8192, (0, 1, 2)),
        (8192, 4096, (0, 1, 2)),
        (8192, 4128, ()),
        (32768, 512, (0,)),
    ],
)
def test_nvfp4_sm107_tma_policy(
    monkeypatch, is_sm107, dtype, sf_layout, m, k, eligible_layouts
):
    pytest.importorskip("cutlass.cute")
    from flashinfer.quantization.kernels.nvfp4_quantize import _should_use_tma

    monkeypatch.delenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", raising=False)
    expected = (
        sf_layout in eligible_layouts
        and is_sm107
        and dtype in (torch.float16, torch.bfloat16)
    )
    assert (
        _should_use_tma(m, k, dtype, is_sm107=is_sm107, sf_layout=sf_layout) == expected
    )
    monkeypatch.setenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", "0")
    assert not _should_use_tma(m, k, dtype, is_sm107=is_sm107)


def test_nvfp4_tma_explicit_override(monkeypatch):
    pytest.importorskip("cutlass.cute")
    from flashinfer.quantization.kernels.nvfp4_quantize import _should_use_tma

    monkeypatch.setenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", "1")
    assert _should_use_tma(2048, 16384, torch.float16)
    assert not _should_use_tma(2048, 16384, torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "module", ["mxfp8_quantize", "mxfp4_quantize", "nvfp4_quantize"]
)
@pytest.mark.parametrize(
    "m,k,elt_bytes,expected",
    [
        # 16-bit inputs
        (512, 8192, 2, False),
        (1024, 4096, 2, False),
        (1024, 8192, 2, True),
        (2048, 4096, 2, True),
        (4096, 2048, 2, False),
        (8192, 2048, 2, False),
        (16384, 2048, 2, True),
        (16384, 1024, 2, False),
        (8192, 24576, 2, True),
        # FP32 (mxfp8): rows are 4x wider in bytes, but K must still be >= 2048
        (1024, 4096, 4, True),
        (2048, 2048, 4, True),
        (1024, 2048, 4, False),
        (16384, 1024, 4, False),
        # FP8 (nvfp4): 1-byte rows need K >= 8192 for 8 KiB rows
        (4096, 4096, 1, False),
        (2048, 8192, 1, True),
        (1024, 8192, 1, False),
        (16384, 2048, 1, False),
        (32768, 4096, 1, True),
    ],
)
def test_sm107_tile128x4_policy(module, m, k, elt_bytes, expected):
    pytest.importorskip("cutlass.cute")
    import importlib

    mod = importlib.import_module(f"flashinfer.quantization.kernels.{module}")
    assert mod._use_sm107_tile128x4(m, k, elt_bytes) == expected


@pytest.mark.parametrize("is_float32", [False, True])
@pytest.mark.parametrize(
    "m,k,expected_16bit,expected_fp32",
    [
        (1024, 12288, False, False),
        (1024, 32768, False, False),
        (1536, 6144, False, False),
        (1536, 8192, False, False),
        (1536, 10240, True, False),
        (1536, 12288, True, False),
        (1536, 14336, False, False),
        (1536, 16384, False, False),
        (1536, 20480, True, True),
        (1536, 24576, True, True),
        (1536, 28672, True, True),
        (1536, 32768, True, True),
    ],
)
def test_mxfp8_sm107_small_block_policy(
    m, k, expected_16bit, expected_fp32, is_float32
):
    pytest.importorskip("cutlass.cute")
    from flashinfer.quantization.kernels.mxfp8_quantize import _use_sm107_small_blocks

    expected = expected_fp32 if is_float32 else expected_16bit
    assert _use_sm107_small_blocks(m, k, is_float32) == expected


def _require_sm107():
    pytest.importorskip("cutlass.cute")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 dispatch regression")
    from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(10, 7):
        pytest.skip("Installed CuTe DSL does not support SM107")


def _quantize(variant, x, layout, scale):
    if variant == "mxfp8":
        return flashinfer.mxfp8_quantize(
            x, sf_swizzle_layout=layout, backend="cute-dsl"
        )
    return flashinfer.fp4_quantize(
        x,
        global_scale=scale if variant == "nvfp4" else None,
        sf_vec_size=16 if variant == "nvfp4" else 32,
        sf_use_ue8m0=variant == "mxfp4",
        is_sf_swizzled_layout=layout != SfLayout.layout_linear,
        is_sf_8x4_layout=layout == SfLayout.layout_8x4,
        backend="cute-dsl",
    )


def _assert_equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        # Compare packed bytes, including scale padding and FP8 signed zero.
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


_FORMAT_DTYPES = [
    (variant, dtype)
    for variant in ("nvfp4", "mxfp4", "mxfp8")
    for dtype in (torch.bfloat16, torch.float16)
] + [("mxfp8", torch.float32), ("nvfp4", torch.float8_e4m3fn)]


@pytest.mark.parametrize("variant,dtype", _FORMAT_DTYPES)
@pytest.mark.parametrize(
    "layout", [SfLayout.layout_128x4, SfLayout.layout_8x4, SfLayout.layout_linear]
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),
        (257, 4128),
        (768, 8192),
        (1024, 8192),
        (1537, 6144),
        (1537, 12288),
        (1537, 24576),
        (2048, 16384),
        (3073, 6144),
        (4095, 4096),
        (4097, 8192),
        (8193, 4096),
    ],
)
def test_sm107_quantize_matches_previous_dispatch(
    monkeypatch, variant, dtype, layout, shape
):
    _require_sm107()
    monkeypatch.delenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", raising=False)
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    torch.manual_seed(42)
    m, k = shape
    # Offset views preserve the required 16-byte (FP8/FP16/BF16) or 32-byte
    # (FP32) load alignment without relying on the allocator's 256-byte alignment.
    offset = 16 if dtype == torch.float8_e4m3fn else 8
    storage = torch.randn(m * k + offset, device="cuda").to(dtype)
    x = storage[offset:].view(m, k)
    scale = torch.tensor(1.0, device="cuda")
    actual = _quantize(variant, x, layout, scale)
    with monkeypatch.context() as previous:
        # Only the host dispatch changes; compilation still targets the real GPU.
        previous.setattr(utils, "get_compute_capability", lambda device: (10, 0))
        expected = _quantize(variant, x, layout, scale)
    _assert_equal(actual, expected)


@pytest.mark.parametrize("variant,dtype", _FORMAT_DTYPES)
@pytest.mark.parametrize(
    "layout", [SfLayout.layout_128x4, SfLayout.layout_8x4, SfLayout.layout_linear]
)
@pytest.mark.parametrize("shape", [(4097, 8192), (1537, 12288)])
def test_sm107_quantize_cuda_graph(monkeypatch, variant, dtype, layout, shape):
    _require_sm107()
    monkeypatch.delenv("FLASHINFER_NVFP4_QUANTIZE_USE_TMA", raising=False)
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    x = torch.randn(*shape, device="cuda").to(dtype)
    scale = torch.tensor(1.0, device="cuda")
    expected = _quantize(variant, x, layout, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _quantize(variant, x, layout, scale)
    graph.replay()
    torch.cuda.synchronize()
    _assert_equal(actual, expected)
