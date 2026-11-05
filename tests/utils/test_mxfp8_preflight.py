"""CPU checks for the MXFP8 preflight API and its benchmark consumer."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flashinfer import mxfp8_quantize, utils
from flashinfer.quantization import fp8_quantization as fp8
from flashinfer.tllm_enums import SfLayout

pytestmark = pytest.mark.cpu_only


def _fake_cuda(shape=(2, 32), dtype=torch.bfloat16, step=1):
    with FakeTensorMode():
        if step == 1:
            return torch.empty(shape, dtype=dtype, device="cuda:1")
        return torch.empty_strided(
            (shape[0], shape[1] // step),
            (shape[1], step),
            dtype=dtype,
            device="cuda:1",
        )


def test_preflight_accepts_call_arguments_without_execution(monkeypatch):
    assert (
        inspect.signature(mxfp8_quantize).parameters
        == inspect.signature(mxfp8_quantize.check_inputs).parameters
    )
    x = torch.empty((2, 32), dtype=torch.float32)
    unexpected = Mock(side_effect=AssertionError("preflight must not execute"))
    monkeypatch.setattr(fp8, "get_mxfp8_quantization_sm100_module", unexpected)
    monkeypatch.setattr(fp8, "device_support_pdl", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "zeros", unexpected)

    # The host route ignores alignment and PDL, including an explicit layout
    # overriding the legacy layout flag.
    assert (
        mxfp8_quantize.check_inputs(x, True, 0, None, "auto", SfLayout.layout_linear)
        is None
    )
    unexpected.assert_not_called()


@pytest.mark.parametrize("kwargs", [{}, {"backend": "cuda"}])
def test_native_dtype_rejection_is_shared_with_execution(monkeypatch, kwargs):
    x = torch.empty((2, 32), dtype=torch.float16)
    loader = Mock(side_effect=AssertionError("invalid input reached JIT"))
    monkeypatch.setattr(fp8, "get_mxfp8_quantization_sm100_module", loader)
    for call in (mxfp8_quantize.check_inputs, mxfp8_quantize):
        with pytest.raises(ValueError, match="2-D float32"):
            call(x, **kwargs)
    loader.assert_not_called()


def test_explicit_host_layout_overrides_legacy_flag():
    x = torch.empty((2, 32), dtype=torch.float32)
    with pytest.raises(ValueError, match="not supported.*CPU"):
        mxfp8_quantize.check_inputs(x, False, sf_swizzle_layout=SfLayout.layout_8x4)


def test_native_cuda_checks_the_input_device(monkeypatch):
    x = _fake_cuda()
    cc = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", cc)
    with pytest.raises(RuntimeError, match="requires SM100"):
        mxfp8_quantize.check_inputs(x)
    cc.assert_called_once_with(x.device)
    cc.return_value = (10, 0)
    assert mxfp8_quantize.check_inputs(x) is None


def test_cute_preflight_preserves_copyable_inputs_and_architecture_policy(monkeypatch):
    from flashinfer import cute_dsl

    x = _fake_cuda((2, 64), torch.float32, step=2)
    assert not x.is_contiguous()
    unexpected = Mock(side_effect=AssertionError("unexpected runtime detection"))
    monkeypatch.setattr(utils, "get_compute_capability", unexpected)
    monkeypatch.setattr(fp8, "device_support_pdl", unexpected)
    # CuTe prepares this layout itself and supports fp32. Do not impose the
    # native CUDA backend's SM100 or contiguous-input restrictions here.
    assert mxfp8_quantize.check_inputs(x, backend="cute-dsl") is None
    dependency_probe = Mock(side_effect=RuntimeError("reached CuTe dependency probe"))
    monkeypatch.setattr(cute_dsl, "is_cute_dsl_available", dependency_probe)
    with pytest.raises(RuntimeError, match="reached CuTe dependency probe"):
        mxfp8_quantize(x, backend="cute-dsl")
    dependency_probe.assert_called_once_with()
    unexpected.assert_not_called()


@pytest.mark.parametrize(
    "shape,layout,message",
    [
        ((2, 32), torch.sparse_coo, "strided layout"),
        ((2, 0), torch.strided, "nonempty last dimension"),
        ((2, 3, 0), torch.strided, "nonempty last dimension"),
    ],
)
def test_cute_rejects_inputs_before_runtime_setup(monkeypatch, shape, layout, message):
    from flashinfer import cute_dsl

    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16, device="cuda:1", layout=layout)
    unexpected = Mock(side_effect=AssertionError("invalid input reached runtime setup"))
    monkeypatch.setattr(cute_dsl, "is_cute_dsl_available", unexpected)
    monkeypatch.setattr(utils, "device_support_pdl", unexpected)
    monkeypatch.setattr(fp8, "get_mxfp8_quantization_sm100_module", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "zeros", unexpected)
    for call in (mxfp8_quantize.check_inputs, mxfp8_quantize):
        with pytest.raises(ValueError, match=message):
            call(x, backend="cute-dsl")
    unexpected.assert_not_called()


def test_native_cuda_rejects_inputs_that_require_a_copy():
    x = _fake_cuda((2, 64), step=2)
    with pytest.raises(ValueError, match="contiguous"):
        mxfp8_quantize.check_inputs(x, backend="cuda")


def test_batch_invariant_reuses_its_runtime_validation(monkeypatch):
    from flashinfer.batch_invariant import support
    from flashinfer.batch_invariant.quantization import mx_api
    from flashinfer.cute_dsl import availability

    x = _fake_cuda((2, 65568))
    valid = _fake_cuda((32,), torch.float32)
    monkeypatch.setattr(utils, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(availability, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(availability, "is_cute_dsl_arch_supported", lambda *cc: True)
    unexpected = Mock(side_effect=AssertionError("input preparation must not run"))
    monkeypatch.setattr(mx_api, "_rows", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    assert mxfp8_quantize.check_inputs(valid, batch_invariant=True) is None
    for call in (mxfp8_quantize.check_inputs, mxfp8_quantize):
        with pytest.raises(
            support.BatchInvariantUnsupportedError, match="rows of at most"
        ):
            call(x, batch_invariant=True)
    unexpected.assert_not_called()


def test_benchmark_uses_public_checks_and_keeps_backend_order(monkeypatch, capsys):
    from benchmarks.routines.quantization import _filter_mxfp8_backends

    x = _fake_cuda(dtype=torch.float32)
    backends = ["cuda", "cute-dsl", "auto"]
    # Native CUDA rejects fp32 before any device query; CuTe accepts it.
    cc = Mock(side_effect=AssertionError("dtype rejection must precede device query"))
    monkeypatch.setattr(utils, "get_compute_capability", cc)
    assert _filter_mxfp8_backends(
        x,
        backends,
        is_sf_swizzled_layout=True,
        alignment=32,
        enable_pdl=None,
    ) == ["cute-dsl"]
    assert backends == ["cuda", "cute-dsl", "auto"]
    assert "requires float16 or bfloat16" in capsys.readouterr().out
    cc.assert_not_called()


def test_benchmark_rejects_dtype_before_allocating_input(monkeypatch):
    from benchmarks.routines import quantization as benchmark

    args = SimpleNamespace(
        verbose=0,
        generate_repro_command=False,
        backends=["cuda"],
        routine="mxfp8_quantize",
        m=10**9,
        k=4096,
        is_sf_swizzled_layout=True,
        alignment=32,
        enable_pdl=None,
        no_cuda_graph=False,
        refcheck=False,
        input_dtype="float32",
    )
    monkeypatch.setattr(benchmark, "get_device", lambda args: torch.device("cuda:1"))
    monkeypatch.setattr(
        benchmark,
        "filter_backends_by_compute_capability",
        lambda backends, *args: backends,
    )
    allocate = Mock(side_effect=AssertionError("rejected case must not allocate input"))
    monkeypatch.setattr(torch, "randn", allocate)
    assert benchmark.testMxfp8Quantize(args) == []
    allocate.assert_not_called()
    assert args.backends == ["cuda"]
