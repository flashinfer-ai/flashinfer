import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest
import torch


def _dispatcher():
    return importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )


def test_public_package_exports_exact_callable_api():
    import flashinfer.comm as comm
    from flashinfer.comm import all_gather_matmul

    assert callable(all_gather_matmul)
    assert comm.all_gather_matmul is all_gather_matmul
    assert (
        all_gather_matmul.__module__
        == "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )
    signature = inspect.signature(all_gather_matmul)
    assert str(signature.parameters["backend"].default) == "auto"
    source = inspect.getsourcefile(all_gather_matmul)
    assert source is not None and source.endswith(
        "flashinfer/comm/all_gather_matmul/all_gather_matmul.py"
    )


def test_public_package_exports_exact_prepare_api():
    import flashinfer.comm as comm

    subpackage = importlib.import_module("flashinfer.comm.all_gather_matmul")
    prepare = comm.prepare_all_gather_matmul

    assert callable(prepare)
    assert subpackage.prepare_all_gather_matmul is prepare
    assert prepare.__module__ == "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    signature = inspect.signature(prepare)
    assert tuple(signature.parameters) == (
        "inp",
        "w",
        "group",
        "backend",
        "verbose",
    )
    assert signature.parameters["inp"].annotation is torch.Tensor
    assert signature.parameters["w"].annotation is torch.Tensor
    assert signature.parameters["group"].annotation is torch.distributed.ProcessGroup
    assert signature.parameters["backend"].annotation is str
    assert signature.parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["backend"].default == "auto"
    assert signature.parameters["verbose"].annotation is bool
    assert signature.parameters["verbose"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["verbose"].default is False
    assert signature.return_annotation == Callable[[torch.Tensor], torch.Tensor]


@pytest.mark.parametrize("backend", ["auto", "cake"])
def test_prepare_backend_forwards_exact_binding(monkeypatch, backend):
    dispatcher = _dispatcher()
    backend_module = ModuleType(
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
    )
    inp = object()
    weight = object()
    subgroup = object()
    launcher = object()
    calls = []

    def fake_prepare(actual_inp, actual_weight, actual_group, *, verbose):
        calls.append((actual_inp, actual_weight, actual_group, verbose))
        return launcher

    backend_module._prepare_all_gather_matmul_cake_packed_qkv_sm103 = fake_prepare
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)

    assert (
        dispatcher.prepare_all_gather_matmul(
            inp, weight, subgroup, backend=backend, verbose=True
        )
        is launcher
    )
    assert calls == [(inp, weight, subgroup, True)]


def test_prepare_unsupported_input_failure_propagates(monkeypatch):
    dispatcher = _dispatcher()
    backend_module = ModuleType(
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
    )

    def reject(*args, **kwargs):
        raise ValueError("unsupported prepared configuration")

    backend_module._prepare_all_gather_matmul_cake_packed_qkv_sm103 = reject
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)

    with pytest.raises(ValueError, match="unsupported prepared configuration"):
        dispatcher.prepare_all_gather_matmul(object(), object(), object())


def test_prepare_unknown_backend_fails_before_dispatch(monkeypatch):
    dispatcher = _dispatcher()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul",
        None,
    )

    with pytest.raises(ValueError, match="exactly 'auto' or 'cake'"):
        dispatcher.prepare_all_gather_matmul(
            object(), object(), object(), backend="cutile"
        )


def test_auto_backend_preserves_blackwell_cutile_route(monkeypatch):
    dispatcher = _dispatcher()
    backend_module = ModuleType(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_cutile"
    )
    inp = SimpleNamespace(device="cuda:3")
    group = object()
    result = object()
    monkeypatch.setattr(
        dispatcher.torch.cuda, "get_device_capability", lambda device: (10, 3)
    )
    backend_module.all_gather_matmul_cutile = (
        lambda actual_inp, actual_w, actual_group, *, verbose: result
    )
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)

    assert dispatcher.all_gather_matmul(inp, object(), group) is result


def test_auto_backend_preserves_pre_blackwell_triton_route(monkeypatch):
    dispatcher = _dispatcher()
    backend_module = ModuleType(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_triton"
    )
    inp = SimpleNamespace(device="cuda:1")
    group = object()
    result = object()
    monkeypatch.setattr(
        dispatcher.torch.cuda, "get_device_capability", lambda device: (9, 0)
    )
    backend_module.all_gather_matmul_triton = (
        lambda actual_inp, actual_w, actual_group, *, verbose: result
    )
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)

    assert dispatcher.all_gather_matmul(inp, object(), group) is result


def test_explicit_cake_backend_forwards_exact_subgroup(monkeypatch):
    dispatcher = _dispatcher()
    backend_module = ModuleType(
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
    )
    subgroup = object()
    inp = object()
    weight = object()
    result = object()
    calls = []

    def fake_backend(actual_inp, actual_weight, actual_group, *, backend, verbose):
        calls.append((actual_inp, actual_weight, actual_group, backend, verbose))
        return result

    backend_module.all_gather_matmul_cake = fake_backend
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_cutile",
        None,
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_triton",
        None,
    )

    assert (
        dispatcher.all_gather_matmul(
            inp, weight, subgroup, backend="cake", verbose=True
        )
        is result
    )
    assert calls == [(inp, weight, subgroup, "cake", True)]


def test_unknown_backend_fails_before_dispatch():
    dispatcher = _dispatcher()

    with pytest.raises(ValueError, match="exactly 'auto' or 'cake'"):
        dispatcher.all_gather_matmul(object(), object(), object(), backend="cutile")
