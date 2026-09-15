"""CUDA L2T packing preserves bits, capture mutability, and its FFI boundary."""

from typing import Literal

import pytest
import torch
from typing_extensions import assert_never

from flashinfer.gemm import svdquant_sm120_cutlass as backend


@pytest.fixture(autouse=True)
def sm120_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")


def _pack(source: torch.Tensor, output: torch.Tensor) -> None:
    backend.get_nvfp4_svdquant_sm120_module().nvfp4_svdquant_pack_l2t_sm120(
        source, output
    )


def _source(k: int, offset: int = 0) -> torch.Tensor:
    """Visit every BF16 bit pattern, including signed zeros and NaN payloads."""
    bits = torch.arange(k * 32 + offset, dtype=torch.int32, device="cuda")
    bits = (bits * 40503 + 17).to(torch.int16)
    return bits[offset:].view(torch.bfloat16).reshape(k, 32)


def _reference_bits(source: torch.Tensor) -> torch.Tensor:
    k = source.shape[0]
    return (
        source.view(torch.int16)
        .reshape(k // 16, 2, 4, 2, 2, 2, 8)
        .permute(0, 4, 5, 6, 2, 1, 3)
        .contiguous()
        .reshape(k, 32)
    )


def _guarded_output(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full((k * 32 + 32,), 0x1234, dtype=torch.int16, device="cuda")
    output = storage[16:-16].view(torch.bfloat16).reshape(k, 32)
    return output, storage


def _assert_packed(
    expected: torch.Tensor, output: torch.Tensor, storage: torch.Tensor
) -> None:
    assert torch.equal(expected, output.view(torch.int16)), "packed bits differ"
    assert bool((storage[:16] == 0x1234).all()), "leading pack output guard changed"
    assert bool((storage[-16:] == 0x1234).all()), "trailing pack output guard changed"


def test_sm120_pack_module_export() -> None:
    module = backend.get_nvfp4_svdquant_sm120_module()
    assert callable(getattr(module, "nvfp4_svdquant_pack_l2t_sm120", None)), (
        "SM120 module must export nvfp4_svdquant_pack_l2t_sm120"
    )


@pytest.mark.parametrize("k", (3072, 5120, 5376, 7168))
@pytest.mark.parametrize("source_offset", (0, 1))
def test_sm120_pack_preserves_arbitrary_bits(k: int, source_offset: int) -> None:
    source = _source(k, source_offset)
    expected = _reference_bits(source)
    output, storage = _guarded_output(k)
    for sentinel in (0x55AA, -21846):
        output.view(torch.int16).fill_(sentinel)
        _pack(source, output)
        torch.cuda.synchronize()
        _assert_packed(expected, output, storage)
    actual = backend._pack_sm120_m537_l2t(source)
    assert torch.equal(expected, actual.view(torch.int16))


@pytest.mark.parametrize("k", (3072, 5120, 5376, 7168))
@pytest.mark.parametrize("graph_calls", (1, 32))
def test_sm120_pack_inference_graph_observes_same_address_updates(
    k: int, graph_calls: int
) -> None:
    with torch.inference_mode():
        source = _source(k)
        output, storage = _guarded_output(k)
        for _ in range(3):
            _pack(source, output)
            cached_output = backend._cached_sm120_m537_l2t(source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_calls):
                _pack(source, output)
                cached_output = backend._cached_sm120_m537_l2t(source)
        addresses = source.data_ptr(), output.data_ptr(), cached_output.data_ptr()
        assert source.is_inference()
        for mask in (0x1234, -1, 0x4321):
            source.view(torch.int16).bitwise_xor_(mask)
            output.view(torch.int16).fill_(0)
            cached_output.view(torch.int16).fill_(0)
            graph.replay()
            torch.cuda.synchronize()
            expected = _reference_bits(source)
            _assert_packed(expected, output, storage)
            assert torch.equal(expected, cached_output.view(torch.int16))
            assert addresses == (
                source.data_ptr(),
                output.data_ptr(),
                cached_output.data_ptr(),
            )


InvalidInput = Literal[
    "dtype",
    "cpu",
    "dimensions",
    "noncontiguous",
    "output_alignment",
    "alias",
    "overlap",
    "rank",
    "k",
    "output_shape",
]


@pytest.mark.parametrize(
    ("index", "invalid"),
    (
        (0, "dtype"),
        (1, "dtype"),
        (0, "cpu"),
        (1, "cpu"),
        (0, "dimensions"),
        (1, "dimensions"),
        (0, "noncontiguous"),
        (1, "noncontiguous"),
        (1, "output_alignment"),
        (1, "alias"),
        (1, "overlap"),
        (0, "rank"),
        (0, "k"),
        (1, "output_shape"),
    ),
)
def test_sm120_pack_rejects_invalid_arguments(
    index: int, invalid: InvalidInput
) -> None:
    k = 3072
    source = _source(k)
    output, storage = _guarded_output(k)
    inputs = [source, output]
    match invalid:
        case "dtype":
            inputs[index] = inputs[index].to(torch.float32)
        case "cpu":
            inputs[index] = inputs[index].cpu()
        case "dimensions":
            inputs[index] = inputs[index].flatten()
        case "noncontiguous":
            inputs[index] = torch.empty(
                (32, k), dtype=torch.bfloat16, device="cuda"
            ).t()
        case "output_alignment":
            inputs[1] = torch.empty((k * 32 + 1,), dtype=torch.bfloat16, device="cuda")[
                1:
            ].view(k, 32)
        case "alias":
            inputs[1] = source
        case "overlap":
            shared = torch.empty((k * 32 + 4,), dtype=torch.bfloat16, device="cuda")
            inputs = [shared[:-4].view(k, 32), shared[4:].view(k, 32)]
        case "rank":
            inputs = [
                torch.empty((k, 16), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
        case "k":
            inputs = [
                torch.empty((4096, 32), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
        case "output_shape":
            inputs[1] = torch.empty((k + 16, 32), dtype=torch.bfloat16, device="cuda")
        case unexpected:
            assert_never(unexpected)
    with pytest.raises(RuntimeError):
        _pack(*inputs)
    _pack(source, output)
    torch.cuda.synchronize()
    _assert_packed(_reference_bits(source), output, storage)


def test_sm120_pack_accepts_disjoint_views_in_one_allocation() -> None:
    k = 3072
    storage = torch.empty((2 * k, 32), dtype=torch.bfloat16, device="cuda")
    source, output = storage[:k], storage[k:]
    source.copy_(_source(k))
    expected = _reference_bits(source)
    _pack(source, output)
    torch.cuda.synchronize()
    assert torch.equal(expected, output.view(torch.int16))


@pytest.mark.parametrize("dtype", (torch.int32, torch.bfloat16))
def test_sm120_pack_helper_keeps_cuda_fallbacks(dtype: torch.dtype) -> None:
    k = 3072
    source = _source(k)
    if dtype == torch.int32:
        source = source.view(torch.int16).to(dtype)
    else:
        source = source.t().contiguous().t()
        assert not source.is_contiguous()
    expected = (
        source.reshape(k // 16, 2, 4, 2, 2, 2, 8)
        .permute(0, 4, 5, 6, 2, 1, 3)
        .contiguous()
        .reshape(k, 32)
    )
    actual = backend._pack_sm120_m537_l2t(source)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


def test_sm120_pack_rejects_different_cuda_devices() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    source = _source(3072)
    other = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    output = torch.empty_like(source, device=f"cuda:{other}")
    with pytest.raises(RuntimeError):
        _pack(source, output)
