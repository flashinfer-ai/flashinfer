import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer import utils
from flashinfer.quantization import fp4_quantization as fp4


_CASES = [
    ("fp4_quantize", {}),
    ("fp4_quantize", {"sf_vec_size": 32, "sf_use_ue8m0": True}),
    ("nvfp4_quantize", {}),
    ("nvfp4_quantize", {"per_token_activation": True}),
    ("mxfp4_quantize", {}),
]


def _quantize(name, x, backend, kwargs):
    if name == "nvfp4_quantize":
        kwargs = {"a_global_sf": torch.ones(1), **kwargs}
    return getattr(fp4, name)(x, backend=backend, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_nvfp4_per_token_sf_vec_size_rejected_before_unsupported_arch(
    monkeypatch, backend
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)

    x = SimpleNamespace()
    with pytest.raises(
        ValueError,
        match="Per-token NVFP4 quantization only supports sf_vec_size=16",
    ):
        _quantize(
            "nvfp4_quantize",
            x,
            backend,
            {"per_token_activation": True, "sf_vec_size": 32},
        )

    capability_query.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_fp4_explicit_recipe_rejected_before_unsupported_arch(monkeypatch, backend):
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    scale_to = Mock(side_effect=AssertionError("global_scale.to must not run"))
    x = SimpleNamespace(device=torch.device("cuda:3"))

    with pytest.raises(ValueError, match="nvfp4_4over6 applies to NVFP4 only"):
        fp4.fp4_quantize(
            x,
            global_scale=SimpleNamespace(to=scale_to),
            sf_vec_size=32,
            sf_use_ue8m0=True,
            nvfp4_4over6=fp4.NVFP44Over6Config(),
            backend=backend,
        )

    capability_query.assert_not_called()
    scale_to.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [(8, 9), (9, 0)])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
@pytest.mark.parametrize("name,kwargs", _CASES)
def test_unsupported_quantizer_arch_rejected_before_backend(
    monkeypatch, capability, backend, name, kwargs
):
    device = torch.device("cuda:3")
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: 0 if name == "fp4_quantize" else 3
    )
    x = SimpleNamespace(device=device, shape=(2, 32), is_cuda=True)
    capability_query = Mock(
        side_effect=lambda queried: capability if queried == device else (10, 0)
    )
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    fp4_factory = Mock(side_effect=AssertionError("FP4 JIT must not run"))
    monkeypatch.setattr(fp4, "get_fp4_quantization_module", fp4_factory)
    cute_backend = Mock(side_effect=AssertionError("CuTe-DSL backend must not run"))
    monkeypatch.setattr(fp4, "_fp4_quantize_cute_dsl", cute_backend)
    scale_to = Mock(side_effect=AssertionError("global_scale.to must not run"))
    if name == "fp4_quantize":
        kwargs = {"global_scale": SimpleNamespace(to=scale_to), **kwargs}

    major, minor = capability
    with pytest.raises(
        RuntimeError, match=f"{name} requires SM100 or newer; got SM{major}{minor}"
    ):
        _quantize(name, x, backend, kwargs)

    capability_query.assert_called_once_with(device)
    fp4_factory.assert_not_called()
    cute_backend.assert_not_called()
    scale_to.assert_not_called()


class _BackendReached(Exception):
    pass


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_supported_quantizer_arch_reaches_existing_dispatch(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    x = SimpleNamespace(
        device=torch.device("cuda:3"),
        shape=(2, 32),
        dtype=torch.bfloat16,
        stride=lambda dim: 32 if dim == -2 else 1,
    )
    capability_query = Mock(
        side_effect=lambda device: capability if device == x.device else (9, 0)
    )
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    monkeypatch.setattr(fp4, "get_compute_capability", capability_query)
    factory = Mock(side_effect=_BackendReached)
    monkeypatch.setattr(fp4, "get_fp4_quantization_module", factory)
    with pytest.raises(_BackendReached):
        fp4.fp4_quantize(x, backend="cuda", enable_pdl=False)
    factory.assert_called_once()
    assert capability_query.call_count > 0
    assert all(call.args[0] == x.device for call in capability_query.call_args_list)


@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
@pytest.mark.parametrize("name,kwargs", _CASES)
def test_quantizer_rejects_real_unsupported_gpu(backend, name, kwargs):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU below SM100")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = utils.get_compute_capability(device)
    if major >= 10:
        pytest.skip("Requires a CUDA GPU below SM100")
    x = torch.ones(2, 32, dtype=torch.bfloat16, device=device)
    with pytest.raises(
        RuntimeError, match=f"requires SM100 or newer; got SM{major}{minor}"
    ):
        _quantize(name, x, backend, kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ["nvfp4_quantize", "mxfp4_quantize"])
def test_cpu_input_quantizer_checks_selected_cuda_device(monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    with pytest.raises(RuntimeError, match=f"{name} requires SM100 or newer; got SM90"):
        _quantize(name, torch.ones(2, 32), "cuda", {})
    capability_query.assert_called_once_with(torch.device("cuda:3"))


_WRAPPER_CASES = [
    ("nvfp4_quantize", "cuda", {}),
    ("nvfp4_quantize", "cuda", {"per_token_activation": True}),
    ("nvfp4_quantize", "cute-dsl", {}),
    ("nvfp4_quantize", "cute-dsl", {"per_token_activation": True}),
    ("mxfp4_quantize", "cuda", {}),
    ("mxfp4_quantize", "cute-dsl", {}),
]


def _wrapper_target(name, backend):
    return torch.device(
        "cuda:3" if name == "mxfp4_quantize" and backend == "cute-dsl" else "cuda:0"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,backend,kwargs", _WRAPPER_CASES)
def test_wrapper_rejects_unsupported_launch_device(monkeypatch, name, backend, kwargs):
    target = _wrapper_target(name, backend)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(
        side_effect=lambda device: (9, 0) if device == target else (10, 0)
    )
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    x = SimpleNamespace(device=torch.device("cuda:3"), is_cuda=True)
    with pytest.raises(RuntimeError, match=f"{name} requires SM100 or newer; got SM90"):
        _quantize(name, x, backend, kwargs)
    capability_query.assert_called_once_with(target)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,backend,kwargs", _WRAPPER_CASES)
def test_wrapper_preserves_supported_launch_device(monkeypatch, name, backend, kwargs):
    target = _wrapper_target(name, backend)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(
        side_effect=lambda device: (10, 0) if device == target else (9, 0)
    )
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    monkeypatch.setattr(fp4, "get_compute_capability", capability_query)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    x = SimpleNamespace(device=torch.device("cuda:3"), is_cuda=True)
    x.cuda = Mock(return_value=SimpleNamespace(device=torch.device("cuda:0")))
    backend_call = Mock(side_effect=_BackendReached)
    monkeypatch.setattr(fp4, "fp4_quantize", backend_call)
    monkeypatch.setattr(fp4, "get_fp4_quantization_module", backend_call)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.cute_dsl",
        SimpleNamespace(is_cute_dsl_available=lambda: True),
    )
    kernels = SimpleNamespace(
        SF_LAYOUT_128x4=0,
        SF_LAYOUT_8x4=1,
        SF_LAYOUT_LINEAR=2,
        nvfp4_quantize_cute_dsl=backend_call,
        nvfp4_quantize_per_token_cute_dsl=backend_call,
        mxfp4_quantize_cute_dsl=backend_call,
    )
    for module in ("nvfp4_quantize", "mxfp4_quantize"):
        monkeypatch.setitem(
            sys.modules, f"flashinfer.quantization.kernels.{module}", kernels
        )
    with pytest.raises(_BackendReached):
        _quantize(name, x, backend, kwargs)
    backend_call.assert_called_once()
    assert capability_query.call_count > 0
    assert all(call.args[0] == target for call in capability_query.call_args_list)
