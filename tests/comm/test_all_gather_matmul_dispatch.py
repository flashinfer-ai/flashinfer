import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _dispatcher():
    return importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
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
        "flashinfer.comm.all_gather_matmul.all_gather_matmul_cake"
    )
    subgroup = object()
    inp = object()
    weight = object()
    result = object()
    calls = []

    def fake_backend(actual_inp, actual_weight, actual_group, *, verbose):
        calls.append((actual_inp, actual_weight, actual_group, verbose))
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
    assert calls == [(inp, weight, subgroup, True)]


def test_unknown_backend_fails_before_dispatch():
    dispatcher = _dispatcher()

    with pytest.raises(ValueError, match="exactly 'auto' or 'cake'"):
        dispatcher.all_gather_matmul(object(), object(), object(), backend="cutile")
