"""MXFP8 host dispatch must reject unsupported devices before JIT loading."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flashinfer import mxfp8_quantize
from flashinfer.quantization import fp8_quantization


def _fake_cuda_input():
    # A genuine Tensor with a CUDA device, without allocating CUDA storage.
    with FakeTensorMode():
        return torch.empty((1, 32), dtype=torch.bfloat16, device="cuda:0")


@pytest.mark.parametrize("capability", [(8, 0), (9, 0)])
def test_mxfp8_quantize_rejects_unsupported_cuda_before_jit(monkeypatch, capability):
    input = _fake_cuda_input()
    capability_query = Mock(return_value=capability)
    loader = Mock(
        side_effect=AssertionError("unsupported device reached the JIT loader")
    )
    pdl_query = Mock(return_value=True)
    monkeypatch.setattr(fp8_quantization, "get_compute_capability", capability_query)
    monkeypatch.setattr(fp8_quantization, "get_mxfp8_quantization_sm100_module", loader)
    monkeypatch.setattr(fp8_quantization, "device_support_pdl", pdl_query)

    with pytest.raises(ValueError, match="requires SM100 or newer"):
        mxfp8_quantize(input, backend="cuda")

    capability_query.assert_called_once_with(input.device)
    loader.assert_not_called()
    pdl_query.assert_not_called()


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_mxfp8_quantize_keeps_supported_cuda_dispatch(monkeypatch, capability):
    input = _fake_cuda_input()
    output = (object(), object())
    quantize = Mock(return_value=output)
    loader = Mock(return_value=SimpleNamespace(mxfp8_quantize_sm100=quantize))
    monkeypatch.setattr(
        fp8_quantization, "get_compute_capability", lambda device: capability
    )
    monkeypatch.setattr(fp8_quantization, "get_mxfp8_quantization_sm100_module", loader)

    assert mxfp8_quantize(input, backend="cuda", enable_pdl=False) == output
    loader.assert_called_once_with()
    quantize.assert_called_once_with(
        input, fp8_quantization.SfLayout.layout_128x4, 32, False
    )


def test_mxfp8_quantize_keeps_cpu_host_dispatch(monkeypatch):
    input = torch.ones((1, 32), dtype=torch.float32)
    output = (object(), object())
    capability_query = Mock(
        side_effect=AssertionError("CPU input queried CUDA capability")
    )
    quantize = Mock(return_value=output)
    monkeypatch.setattr(fp8_quantization, "get_compute_capability", capability_query)
    monkeypatch.setattr(
        fp8_quantization,
        "get_mxfp8_quantization_sm100_module",
        lambda: SimpleNamespace(mxfp8_quantize_sm100=quantize),
    )

    assert mxfp8_quantize(input, backend="cuda", enable_pdl=False) == output
    capability_query.assert_not_called()
    quantize.assert_called_once_with(
        input, fp8_quantization.SfLayout.layout_128x4, 32, False
    )
