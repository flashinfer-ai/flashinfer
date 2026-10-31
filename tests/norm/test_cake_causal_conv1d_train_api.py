"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests of the selectable-mode entry points of the Cake causal convolution + SiLU training kernels: the single-rank
facade ``flashinfer.experimental.cake.cake_causal_conv1d_train`` and the KCP facade
``flashinfer.cake_causal_conv1d_train_kcp``.

Dispatch (CPU, the packages replaced by recorders): ``deterministic=True`` (the default) calls the deterministic
package with the caller's arguments unchanged, ``deterministic=False`` the non-deterministic package; a non-bool
mode is a ``TypeError``; under ``torch.use_deterministic_algorithms(True)`` a ``deterministic=False`` call runs the
deterministic package and warns once per entry point, without handing it a non-deterministic workspace.  The
entry points are marked experimental, mirror the packages' signatures, and leave every deterministic and legacy
name unchanged; ``import flashinfer`` imports neither facade.  On a device (once the non-deterministic package is
rendered): the facades return the packages' bits.  The numerical contract itself is
``test_cake_causal_conv1d_train.py``'s.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import types
import warnings

import pytest
import torch

import flashinfer
import flashinfer.cake_causal_conv1d_det_train as det_facade
import flashinfer.cake_causal_conv1d_det_train_kcp as det_kcp_facade
import flashinfer.cake_causal_conv1d_train_kcp as kcp_facade
import flashinfer.experimental.cake.cake_causal_conv1d_train as facade
import flashinfer.experimental.cake.cake_causal_conv1d_train_backend as backend
from flashinfer.api_logging import ExperimentalWarning

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
SEED = 8066
WIDTH = 4
DW_ATOL, DW_RTOL = 5e-2, 1e-4
FALLBACK = "runs the deterministic implementation"

# entry point -> (deterministic package name, non-deterministic package name)
ENTRIES = {
    "cake_causal_conv1d_train_forward": (
        "cake_causal_conv1d_det_train_forward",
        "cake_causal_conv1d_train_forward",
    ),
    "cake_causal_conv1d_train_backward": (
        "cake_causal_conv1d_det_train_backward",
        "cake_causal_conv1d_train_backward",
    ),
    "cake_causal_conv1d": ("cake_causal_conv1d_det", "cake_causal_conv1d"),
}
HELPERS = {
    "cake_causal_conv1d_train_forward_workspace": "cake_causal_conv1d_det_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes": "cake_causal_conv1d_det_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace": "cake_causal_conv1d_det_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes": "cake_causal_conv1d_det_train_backward_workspace_bytes",
}
KCP_ENTRIES = {
    "cake_causal_conv1d_kcp": "cake_causal_conv1d_det_kcp",
    "cake_causal_conv1d_multi_kcp": "cake_causal_conv1d_det_multi_kcp",
}


class _Recorder:
    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log

    def __call__(self, *args, **kwargs):
        self.log.append((self.name, args, kwargs))
        return self.name


def _namespace(prefix: str, names, log: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{n: _Recorder(f"{prefix}.{n}", log) for n in names})


@pytest.fixture
def recorded(monkeypatch):
    """Both packages replaced by recorders behind the single-rank facade; the fallback warnings reset."""

    log: list = []
    det_names = [d for d, _ in ENTRIES.values()] + list(HELPERS.values()) + [
        "is_cake_causal_conv1d_det_train_supported"
    ]
    nondet_names = [n for _, n in ENTRIES.values()] + list(HELPERS) + [
        "is_cake_causal_conv1d_train_supported"
    ]
    monkeypatch.setattr(facade, "_det", lambda: _namespace("det", det_names, log))
    monkeypatch.setattr(facade, "_backend", lambda: _namespace("nondet", nondet_names, log))
    monkeypatch.setattr(facade, "_FALLBACK_WARNED", set())
    yield log


@pytest.fixture
def kcp_recorded(monkeypatch):
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import kcp as kcp_backend

    log: list = []
    for name, det_name in KCP_ENTRIES.items():
        monkeypatch.setattr(det_kcp_facade, det_name, _Recorder(f"det.{det_name}", log))
        monkeypatch.setattr(kcp_backend, name, _Recorder(f"nondet.{name}", log))
    monkeypatch.setattr(kcp_facade, "_FALLBACK_WARNED", set())
    yield log


@pytest.fixture
def torch_deterministic():
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def _quiet(call):
    """``call()`` with the one-time experimental notices silenced; returns ``(result, fallback warnings)``."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ExperimentalWarning)
        result = call()
    return result, [w for w in caught if FALLBACK in str(w.message)]


# ----------------------------------------------------------------------------------------------------------------
# Dispatch (CPU).


@pytest.mark.cpu_only
def test_entry_points_dispatch_by_keyword(recorded) -> None:
    x, w, cu, dy, b, ws = (object() for _ in range(6))
    calls = {
        "cake_causal_conv1d_train_forward": ((x, w, cu), {"bias": b, "out": None, "workspace": ws}),
        "cake_causal_conv1d_train_backward": (
            (x, w, cu, dy),
            {"bias": b, "silu_grad_preactivation": "fp32", "workspace": ws, "dx": None, "dweight": None, "dbias": None},
        ),
        "cake_causal_conv1d": ((x, w, cu, b), {"silu_grad_preactivation": "fp32"}),
    }  # fmt: skip
    for entry, (det_name, nondet_name) in ENTRIES.items():
        args, kwargs = calls[entry]
        for mode in ({}, {"deterministic": True}):
            recorded.clear()
            assert _quiet(lambda: getattr(facade, entry)(*args, **kwargs, **mode))[0] == f"det.{det_name}"
            ((name, got_args, got_kwargs),) = recorded
            assert got_args == args and got_kwargs == kwargs, entry
        recorded.clear()
        assert _quiet(lambda: getattr(facade, entry)(*args, **kwargs, deterministic=False))[0] == (
            f"nondet.{nondet_name}"
        )
        ((name, got_args, got_kwargs),) = recorded
        assert got_args == args and got_kwargs == {**kwargs, "deterministic": False}, entry


@pytest.mark.cpu_only
def test_helpers_dispatch_by_keyword(recorded) -> None:
    device = torch.device("cpu")
    for helper, det_name in HELPERS.items():
        recorded.clear()
        assert getattr(facade, helper)(64, 256, device) == f"det.{det_name}"
        assert recorded[-1][1:] == ((64, 256, device), {})
        assert getattr(facade, helper)(64, 256, device, deterministic=False) == f"nondet.{helper}"
        assert recorded[-1][1:] == ((64, 256, device), {"deterministic": False})
    assert facade.is_cake_causal_conv1d_train_supported(device) == "det.is_cake_causal_conv1d_det_train_supported"
    assert facade.is_cake_causal_conv1d_train_supported(device, deterministic=False) == (
        "nondet.is_cake_causal_conv1d_train_supported"
    )


@pytest.mark.cpu_only
def test_mode_must_be_a_bool(recorded) -> None:
    positional = {"cake_causal_conv1d_train_forward": 3, "cake_causal_conv1d_train_backward": 4, "cake_causal_conv1d": 3}
    for bad in (0, 1, None, "False", torch.tensor(False)):
        for entry in ENTRIES:
            args = (None,) * positional[entry]
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                _quiet(lambda: getattr(facade, entry)(*args, deterministic=bad))
        for helper in HELPERS:
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                getattr(facade, helper)(64, 256, torch.device("cpu"), deterministic=bad)
        for entry in KCP_ENTRIES:
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                _quiet(lambda: getattr(kcp_facade, entry)(None, None, None, deterministic=bad))
    assert recorded == []


@pytest.mark.cpu_only
def test_torch_deterministic_mode_falls_back_and_warns_once(recorded, torch_deterministic) -> None:
    x, w, cu, dy, ws = (object() for _ in range(5))
    for entry, (det_name, _) in ENTRIES.items():
        args = (x, w, cu, dy) if entry.endswith("backward") else (x, w, cu)
        for repeat in range(3):
            recorded.clear()
            result, fallback = _quiet(lambda: getattr(facade, entry)(*args, deterministic=False))
            assert result == f"det.{det_name}", entry
            assert len(fallback) == (1 if repeat == 0 else 0), (entry, repeat)
            assert fallback == [] or fallback[0].category is UserWarning
            assert fallback == [] or f"{entry}(deterministic=False)" in str(fallback[0].message)
    # A non-deterministic backward workspace is not handed to the deterministic package; a deterministic call's is.
    recorded.clear()
    _quiet(lambda: facade.cake_causal_conv1d_train_backward(x, w, cu, dy, workspace=ws, deterministic=False))
    assert recorded[-1][2]["workspace"] is None
    _quiet(lambda: facade.cake_causal_conv1d_train_backward(x, w, cu, dy, workspace=ws))
    assert recorded[-1][2]["workspace"] is ws
    # The forward workspace (the same int32 counters in both packages) is passed on.
    _quiet(lambda: facade.cake_causal_conv1d_train_forward(x, w, cu, workspace=ws, deterministic=False))
    assert recorded[-1][2]["workspace"] is ws
    # The helpers size the package their keyword names, whatever torch's mode.
    assert facade.cake_causal_conv1d_train_backward_workspace_bytes(1, 8, "cpu", deterministic=False) == (
        "nondet.cake_causal_conv1d_train_backward_workspace_bytes"
    )


@pytest.mark.cpu_only
def test_kcp_facade_dispatch(kcp_recorded) -> None:
    x, w, ctx, handle = (object() for _ in range(4))
    for entry, det_name in KCP_ENTRIES.items():
        for mode in ({}, {"deterministic": True}):
            kcp_recorded.clear()
            result, _ = _quiet(lambda: getattr(kcp_facade, entry)(x, w, ctx, recompute_handle=handle, **mode))
            assert result == f"det.{det_name}"
            assert kcp_recorded == [(f"det.{det_name}", (x, w, ctx), {"recompute_handle": handle})]
        kcp_recorded.clear()
        result, _ = _quiet(lambda: getattr(kcp_facade, entry)(x, w, ctx, recompute_handle=handle, deterministic=False))
        assert result == f"nondet.{entry}"
        assert kcp_recorded == [
            (f"nondet.{entry}", (x, w, ctx), {"recompute_handle": handle, "deterministic": False})
        ]


@pytest.mark.cpu_only
def test_kcp_facade_falls_back_under_torch_deterministic_mode(kcp_recorded, torch_deterministic) -> None:
    x, w, ctx = (object() for _ in range(3))
    for entry, det_name in KCP_ENTRIES.items():
        for repeat in range(2):
            result, fallback = _quiet(lambda: getattr(kcp_facade, entry)(x, w, ctx, deterministic=False))
            assert result == f"det.{det_name}"
            assert len(fallback) == (1 if repeat == 0 else 0), (entry, repeat)


# ----------------------------------------------------------------------------------------------------------------
# Surface (CPU).


def _parameters(fn) -> list[tuple]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.cpu_only
def test_facade_signatures_mirror_the_packages() -> None:
    """The facade's entry points take exactly the non-deterministic package's parameters (``deterministic=True``
    keyword-only last), which are the deterministic package's plus ``deterministic``."""

    for entry, (det_name, nondet_name) in ENTRIES.items():
        mine = _parameters(getattr(facade, entry))
        assert mine == _parameters(getattr(backend, nondet_name)), entry
        assert mine[-1] == ("deterministic", inspect.Parameter.KEYWORD_ONLY, True), entry
        assert mine[:-1] == _parameters(getattr(det_facade, det_name)), entry
    for helper in HELPERS:
        assert _parameters(getattr(facade, helper)) == _parameters(getattr(backend, helper)), helper
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import kcp as kcp_backend

    for entry, det_name in KCP_ENTRIES.items():
        mine = _parameters(getattr(kcp_facade, entry))
        assert mine[-1] == ("deterministic", inspect.Parameter.KEYWORD_ONLY, True), entry
        assert mine[:-1] == _parameters(getattr(det_kcp_facade, det_name)), entry
        theirs = {name for name, *_ in _parameters(getattr(kcp_backend, entry))}
        assert {name for name, *_ in mine} <= theirs, entry


@pytest.mark.cpu_only
def test_entry_points_are_experimental() -> None:
    for entry in ENTRIES:
        assert getattr(getattr(facade, entry), "is_experimental", False) is True, entry
    for entry in KCP_ENTRIES:
        assert getattr(getattr(kcp_facade, entry), "is_experimental", False) is True, entry
    assert set(facade.__all__) >= set(ENTRIES) | set(HELPERS) | {"is_cake_causal_conv1d_train_supported"}
    assert set(kcp_facade.__all__) == set(KCP_ENTRIES) | {"RecomputeHandleKcp"}
    assert facade.CakeCausalConv1dFunction is backend.CakeCausalConv1dFunction


@pytest.mark.cpu_only
def test_deterministic_and_legacy_names_are_unchanged() -> None:
    """Every deterministic name, and every legacy compatibility name, is still the deterministic object: the new
    same-spelled names live in other modules only."""

    assert flashinfer.cake_causal_conv1d is det_facade.cake_causal_conv1d_det
    assert flashinfer.cake_causal_conv1d_train_forward is det_facade.cake_causal_conv1d_det_train_forward
    assert flashinfer.cake_causal_conv1d_train_backward is det_facade.cake_causal_conv1d_det_train_backward
    assert flashinfer.CakeCausalConv1dFunction is det_facade.CakeCausalConv1dDetFunction
    assert flashinfer.cake_causal_conv1d_train is det_facade
    assert det_facade.cake_causal_conv1d_kcp is det_kcp_facade.cake_causal_conv1d_det_kcp
    for entry in ENTRIES:
        assert getattr(det_facade, entry) is not getattr(facade, entry), entry
    assert det_facade.CakeCausalConv1dFunction is not facade.CakeCausalConv1dFunction
    assert det_kcp_facade.cake_causal_conv1d_det_kcp is not kcp_facade.cake_causal_conv1d_kcp


@pytest.mark.cpu_only
def test_core_does_not_import_the_experimental_package() -> None:
    """``import flashinfer`` imports neither facade nor the non-deterministic package; the KCP facade imports no
    package until a call; the single-rank facade imports the non-deterministic package but no KCP host module."""

    code = (
        "import sys\n"
        "import flashinfer\n"
        "new = ('flashinfer.experimental.cake', 'flashinfer.cake_causal_conv1d_train_kcp')\n"
        "assert not [m for m in sys.modules if m.startswith(new)], sorted(m for m in sys.modules if m.startswith(new))\n"
        "import flashinfer.cake_causal_conv1d_train_kcp\n"
        "assert 'flashinfer.experimental.cake' not in sys.modules\n"
        "import flashinfer.experimental.cake.cake_causal_conv1d_train\n"
        "kcp = [m for m in sys.modules if m.endswith(('.kcp', '.kcp_jit')) and 'conv1d' in m]\n"
        "assert not kcp, kcp\n"
        "print('IMPORT_ISOLATION_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "IMPORT_ISOLATION_OK" in r.stdout, (r.stdout, r.stderr)


# ----------------------------------------------------------------------------------------------------------------
# On a device: the facades return the packages' bits.


def _device() -> torch.device:
    return torch.device("cuda:0")


def _gpu_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and backend.is_cake_causal_conv1d_train_supported(_device())
        and det_facade.is_cake_causal_conv1d_det_train_supported(_device())
    )


gpu = pytest.mark.skipif(
    not _gpu_supported(),
    reason="needs an SM100 / SM103 / SM107 device and the rendered non-deterministic package (TODO-EXPORT)",
)


def _problem(S: int = 1000, C: int = 512):
    generator = torch.Generator(device=_device()).manual_seed(SEED)
    x = torch.randn(S, C, device=_device(), generator=generator).to(torch.bfloat16)
    dy = torch.randn(S, C, device=_device(), generator=generator).to(torch.bfloat16)
    w = 0.5 * torch.randn(C, WIDTH, device=_device(), generator=generator)
    b = 0.5 * torch.randn(C, device=_device(), generator=generator)
    cu = torch.tensor([0, 369, 659, 868, S], dtype=torch.int32, device=_device())
    return x, w, cu, dy, b


@gpu
def test_facade_returns_the_packages_bits() -> None:
    x, w, cu, dy, b = _problem()
    S, C = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        y_det = det_facade.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)
        assert torch.equal(facade.cake_causal_conv1d_train_forward(x, w, cu, bias=b), y_det)
        y_nondet = facade.cake_causal_conv1d_train_forward(x, w, cu, bias=b, deterministic=False)
        assert torch.equal(y_nondet, y_det)
        det_grads = det_facade.cake_causal_conv1d_det_train_backward(x, w, cu, dy, bias=b)
        grads = facade.cake_causal_conv1d_train_backward(x, w, cu, dy, bias=b)
        assert all(torch.equal(a, e) for a, e in zip(grads, det_grads, strict=True))
        ws = facade.cake_causal_conv1d_train_backward_workspace(S, C, _device(), deterministic=False)
        assert ws.dtype == torch.uint8 and int(torch.count_nonzero(ws)) == 0
        assert ws.numel() == facade.cake_causal_conv1d_train_backward_workspace_bytes(
            S, C, _device(), deterministic=False
        )
        dx, dw, db = facade.cake_causal_conv1d_train_backward(
            x, w, cu, dy, bias=b, workspace=ws, deterministic=False
        )
        assert int(torch.count_nonzero(ws)) == 0
        assert torch.equal(dx, det_grads[0])
        torch.testing.assert_close(dw, det_grads[1], atol=DW_ATOL, rtol=DW_RTOL)
        torch.testing.assert_close(db, det_grads[2], atol=DW_ATOL, rtol=DW_RTOL)
        assert facade.cake_causal_conv1d_train_forward_workspace_bytes(
            S, C, _device(), deterministic=False
        ) == det_facade.cake_causal_conv1d_det_train_forward_workspace_bytes(S, C, _device())


@gpu
def test_autograd_in_both_modes() -> None:
    x0, w0, cu, dy, b0 = _problem()
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for mode in (True, False):
            x = x0.clone().requires_grad_(True)
            w = w0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            y = facade.cake_causal_conv1d(x, w, cu, b, deterministic=mode)
            y.backward(dy)
            results[mode] = (y.detach(), x.grad, w.grad, b.grad)
    assert torch.equal(results[True][0], results[False][0])
    assert torch.equal(results[True][1], results[False][1])
    torch.testing.assert_close(results[False][2], results[True][2], atol=DW_ATOL, rtol=DW_RTOL)
    torch.testing.assert_close(results[False][3], results[True][3], atol=DW_ATOL, rtol=DW_RTOL)


@gpu
def test_torch_deterministic_mode_runs_the_deterministic_package(torch_deterministic, monkeypatch) -> None:
    monkeypatch.setattr(facade, "_FALLBACK_WARNED", set())
    x0, w0, cu, dy, _ = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        expected = det_facade.cake_causal_conv1d_det_train_backward(x0, w0, cu, dy)
        with pytest.warns(UserWarning, match=FALLBACK):
            got = facade.cake_causal_conv1d_train_backward(x0, w0, cu, dy, deterministic=False)
        assert all(torch.equal(a, e) for a, e in zip(got, expected, strict=True))
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        facade.cake_causal_conv1d(x, w, cu, deterministic=False).backward(dy)
        assert torch.equal(x.grad, expected[0]) and torch.equal(w.grad, expected[1])
