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
@pytest.mark.parametrize(
    "name,backend,kwargs,error",
    [
        pytest.param(
            "fp4_quantize",
            "cute-dsl",
            {"sf_vec_size": 32},
            "CuTe-DSL backend does not support sf_vec_size=32",
            id="direct-cute-parameter-pair",
        ),
        pytest.param(
            "nvfp4_quantize",
            "cute-dsl",
            {"sf_vec_size": 32},
            "CuTe-DSL backend only supports sf_vec_size=16, got 32",
            id="nvfp4-cute-global-sf-vec-size",
        ),
        pytest.param(
            "nvfp4_quantize",
            "cute-dsl",
            {"per_token_activation": True, "expanded_idx_to_permuted_idx": object()},
            "CuTe-DSL per-token NVFP4 quantization does not support expanded_idx_to_permuted_idx",
            id="nvfp4-cute-per-token-expanded-idx",
        ),
        pytest.param(
            "nvfp4_quantize",
            "cuda",
            {"expanded_idx_to_permuted_idx": object()},
            "expanded_idx_to_permuted_idx is only supported with per_token_activation=True",
            id="nvfp4-cuda-global-expanded-idx",
        ),
        pytest.param(
            "nvfp4_quantize",
            "cuda",
            {"sf_vec_size": 32, "nvfp4_4over6": fp4.NVFP44Over6Config()},
            "nvfp4_4over6 applies to NVFP4 only",
            id="nvfp4-cuda-global-explicit-mxfp4-recipe",
        ),
    ],
)
def test_pure_quantizer_parameters_rejected_before_unsupported_arch(
    monkeypatch, name, backend, kwargs, error
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    copy = Mock(side_effect=AssertionError("CUDA copy must not run"))
    scale_to = Mock(side_effect=AssertionError("global_scale.to must not run"))
    x = SimpleNamespace(device=torch.device("cuda:3"), dtype=torch.bfloat16, cuda=copy)
    if name == "fp4_quantize":
        kwargs = {"global_scale": SimpleNamespace(to=scale_to), **kwargs}

    with pytest.raises(ValueError, match=error):
        _quantize(name, x, backend, kwargs)

    capability_query.assert_not_called()
    copy.assert_not_called()
    scale_to.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "name,backend,from_env",
    [
        pytest.param("fp4_quantize", "cute-dsl", True, id="direct-cute-environment"),
        pytest.param("nvfp4_quantize", "cute-dsl", True, id="nvfp4-cute-environment"),
        pytest.param("nvfp4_quantize", "cuda", False, id="nvfp4-cuda-explicit"),
        pytest.param(
            "nvfp4_quantize", "cuda", True, id="nvfp4-cuda-environment-control"
        ),
    ],
)
def test_fp8_recipe_validation_before_unsupported_arch(
    monkeypatch, name, backend, from_env
):
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    copy = Mock(side_effect=AssertionError("CUDA copy must not run"))
    x = SimpleNamespace(
        device=torch.device("cuda:3"), dtype=torch.float8_e4m3fn, cuda=copy
    )
    kwargs = {} if from_env else {"nvfp4_4over6": fp4.NVFP44Over6Config()}
    legacy_cuda_env = backend == "cuda" and from_env
    exception = RuntimeError if legacy_cuda_env else ValueError
    error = (
        "nvfp4_quantize requires SM100 or newer; got SM90"
        if legacy_cuda_env
        else "requires fp16 or bf16 input"
    )

    with pytest.raises(exception, match=error):
        _quantize(name, x, backend, kwargs)

    if legacy_cuda_env:
        capability_query.assert_called_once_with(torch.device("cuda:0"))
    else:
        capability_query.assert_not_called()
    copy.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "backend,recipe", [("cuda", None), ("cute-dsl", fp4.NVFP44Over6Config())]
)
def test_per_token_host_scale_rejected_before_unsupported_arch(
    monkeypatch, backend, recipe
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(return_value=(9, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)

    with pytest.raises(ValueError, match="does not match the requested NVFP4"):
        fp4.nvfp4_quantize(
            SimpleNamespace(),
            1.0,
            per_token_activation=True,
            backend=backend,
            nvfp4_4over6=recipe,
        )

    capability_query.assert_not_called()


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
    item_read = Mock(side_effect=AssertionError("Tensor.item must not run"))
    monkeypatch.setattr(torch.Tensor, "item", item_read)
    if name == "nvfp4_quantize" and kwargs.get("per_token_activation"):
        kwargs = {"nvfp4_4over6": None, **kwargs}
    x = SimpleNamespace(device=torch.device("cuda:3"), is_cuda=True)
    with pytest.raises(RuntimeError, match=f"{name} requires SM100 or newer; got SM90"):
        _quantize(name, x, backend, kwargs)
    capability_query.assert_called_once_with(target)
    item_read.assert_not_called()


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


@pytest.mark.cpu_only
def test_nvfp4_per_token_cuda_tensor_scale_avoids_host_item_read(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(return_value=(10, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    monkeypatch.setattr(fp4, "get_compute_capability", capability_query)
    item_read = Mock(side_effect=AssertionError("Tensor.item must not run"))
    monkeypatch.setattr(torch.Tensor, "item", item_read)
    cuda_copy = Mock(side_effect=AssertionError("Tensor.cuda must not run"))
    monkeypatch.setattr(torch.Tensor, "cuda", cuda_copy)

    # Use a non-CPU tensor (meta device works without a GPU) to verify device
    # tensors stay on device without calling .item() or .cuda().
    a_global_sf = torch.empty(1, dtype=torch.float32, device="meta")
    x_cuda = SimpleNamespace(device=a_global_sf.device)
    x = SimpleNamespace(
        device=a_global_sf.device, is_cuda=True, cuda=Mock(return_value=x_cuda)
    )
    sentinel = (object(), object(), object())
    quant_op = Mock(return_value=sentinel)
    monkeypatch.setattr(
        fp4,
        "get_fp4_quantization_module",
        Mock(
            return_value=SimpleNamespace(nvfp4_quant_and_per_token_scale_sm100=quant_op)
        ),
    )

    result = fp4.nvfp4_quantize(
        x,
        a_global_sf,
        per_token_activation=True,
        backend="cuda",
        nvfp4_4over6=None,
    )

    assert result == sentinel
    item_read.assert_not_called()
    cuda_copy.assert_not_called()
    quant_op.assert_called_once()
    assert quant_op.call_args.args[1] is a_global_sf


@pytest.mark.cpu_only
def test_nvfp4_per_token_cpu_tensor_scale_uses_host_scalar_and_validates_recipe(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    capability_query = Mock(return_value=(10, 0))
    monkeypatch.setattr(utils, "get_compute_capability", capability_query)
    monkeypatch.setattr(fp4, "get_compute_capability", capability_query)
    cuda_copy = Mock(side_effect=AssertionError("CPU scale must not be copied to CUDA"))
    monkeypatch.setattr(torch.Tensor, "cuda", cuda_copy)

    x_cuda = SimpleNamespace(device=torch.device("cuda:0"))
    x = SimpleNamespace(
        device=torch.device("cuda:0"), is_cuda=True, cuda=Mock(return_value=x_cuda)
    )
    scale_448 = torch.tensor([1.0 / (448.0 * 6.0)], dtype=torch.float32, device="cpu")
    sentinel = (object(), object(), object())
    quant_op = Mock(return_value=sentinel)
    monkeypatch.setattr(
        fp4,
        "get_fp4_quantization_module",
        Mock(
            return_value=SimpleNamespace(nvfp4_quant_and_per_token_scale_sm100=quant_op)
        ),
    )

    # Matching CPU tensor stays on host as a Python float (no H2D copy).
    result = fp4.nvfp4_quantize(
        x,
        scale_448,
        per_token_activation=True,
        backend="cuda",
        nvfp4_4over6=None,
    )
    assert result == sentinel
    cuda_copy.assert_not_called()
    quant_op.assert_called_once()
    assert isinstance(quant_op.call_args.args[1], float)
    assert quant_op.call_args.args[1] == pytest.approx(1.0 / (448.0 * 6.0))

    # Mismatched CPU tensor with explicit 256 recipe still raises ValueError.
    with pytest.raises(ValueError, match="does not match the requested NVFP4"):
        fp4.nvfp4_quantize(
            x,
            scale_448,
            per_token_activation=True,
            backend="cuda",
            nvfp4_4over6=fp4.NVFP44Over6Config(e4m3_max=256),
        )
    cuda_copy.assert_not_called()


@pytest.mark.cpu_only
def test_stale_fp4_aot_module_fallback_and_float_compatibility(monkeypatch):
    from flashinfer.jit.core import JitSpecNvcc

    stale_scalar_fn = Mock()
    stale_module = SimpleNamespace(nvfp4_quant_and_per_token_scale=stale_scalar_fn)
    fresh_scalar_fn = Mock()
    fresh_tensor_fn = Mock()
    fresh_module = SimpleNamespace(
        nvfp4_quant_and_per_token_scale=fresh_scalar_fn,
        nvfp4_quant_and_per_token_scale_tensor=fresh_tensor_fn,
    )

    spec = JitSpecNvcc(
        name="fp4_quantization_100",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )
    monkeypatch.setattr(spec, "build_and_load", Mock(return_value=stale_module))
    fallback_mock = Mock(return_value=fresh_module)
    monkeypatch.setattr(spec, "_build_and_load_jit_fallback", fallback_mock)
    monkeypatch.setattr(fp4, "gen_fp4_quantization_sm100_module", lambda: spec)

    x = torch.empty((2, 32), dtype=torch.bfloat16, device="meta")
    out = torch.empty((2, 16), dtype=torch.uint8, device="meta")
    sf = torch.empty((2, 2), dtype=torch.uint8, device="meta")
    pts = torch.empty((2,), dtype=torch.float32, device="meta")

    fp4.get_fp4_quantization_module.cache_clear()
    try:
        # Loading a stale module must not eagerly compile; scalar/CPU-tensor
        # calls keep using the 8-arg export, and JIT fallback runs lazily once
        # when a device tensor actually requires the new export.
        monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
        mod = fp4.get_fp4_quantization_module("100")
        fallback_mock.assert_not_called()

        mod.nvfp4_quant_and_per_token_scale_out_sm100(x, 1.0, out, sf, pts)
        mod.nvfp4_quant_and_per_token_scale_out_sm100(
            x, torch.tensor([1.0], dtype=torch.float32), out, sf, pts
        )
        fallback_mock.assert_not_called()
        assert stale_scalar_fn.call_count == 2
        assert all(len(c.args) == 8 for c in stale_scalar_fn.call_args_list)

        dev_scale = torch.empty((1,), dtype=torch.float32, device="meta")
        mod.nvfp4_quant_and_per_token_scale_out_sm100(x, dev_scale, out, sf, pts)
        mod.nvfp4_quant_and_per_token_scale_out_sm100(x, dev_scale, out, sf, pts)
        fallback_mock.assert_called_once()
        assert fresh_tensor_fn.call_count == 2
        assert len(fresh_tensor_fn.call_args.args) == 8

        # If lazy JIT fallback fails (e.g. no nvcc), loading and scalar calls
        # still work while device-tensor calls raise a clear error naming the
        # missing export.
        fp4.get_fp4_quantization_module.cache_clear()
        stale_scalar_fn.reset_mock()
        compile_err = RuntimeError("nvcc not found")
        failing_fallback = Mock(side_effect=compile_err)
        monkeypatch.setattr(spec, "_build_and_load_jit_fallback", failing_fallback)
        stale_mod_no_nvcc = fp4.get_fp4_quantization_module("100")
        failing_fallback.assert_not_called()

        stale_mod_no_nvcc.nvfp4_quant_and_per_token_scale_out_sm100(
            x, 1.0, out, sf, pts
        )
        stale_scalar_fn.assert_called_once()
        failing_fallback.assert_not_called()

        with pytest.raises(
            RuntimeError, match="nvfp4_quant_and_per_token_scale_tensor"
        ) as excinfo:
            stale_mod_no_nvcc.nvfp4_quant_and_per_token_scale_out_sm100(
                x, dev_scale, out, sf, pts
            )
        assert excinfo.value.__cause__ is compile_err

        # With JIT disabled, stale module stays usable for 8-arg float/CPU scale
        # calls and raises a clear error without invoking fallback on device tensors.
        fp4.get_fp4_quantization_module.cache_clear()
        stale_scalar_fn.reset_mock()
        fallback_mock.reset_mock()
        monkeypatch.setattr(spec, "_build_and_load_jit_fallback", fallback_mock)
        monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
        stale_mod = fp4.get_fp4_quantization_module("100")
        fallback_mock.assert_not_called()

        stale_mod.nvfp4_quant_and_per_token_scale_out_sm100(x, 1.0, out, sf, pts)
        stale_scalar_fn.assert_called_once()

        with pytest.raises(
            RuntimeError, match="nvfp4_quant_and_per_token_scale_tensor"
        ):
            stale_mod.nvfp4_quant_and_per_token_scale_out_sm100(
                x, dev_scale, out, sf, pts
            )
        fallback_mock.assert_not_called()
    finally:
        fp4.get_fp4_quantization_module.cache_clear()
