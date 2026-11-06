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

Tests of the experimental entry points of the non-deterministic Cake causal convolution + SiLU training package: the
single-rank module ``flashinfer.experimental.cake.cake_causal_conv1d_train`` and the KCP module
``flashinfer.experimental.cake.cake_causal_conv1d_train_kcp``.

Mode rule (CPU, the packages replaced by recorders): every entry point, helper and support query runs the
non-deterministic package with the caller's arguments unchanged when ``deterministic=False``; any other value (the
default ``True``, ``None``, ints, strings, tensors) raises ``ValueError`` naming the deterministic entry point before
anything runs; nothing ever runs the deterministic package, also under ``torch.use_deterministic_algorithms(True)``
(strict and ``warn_only=True``), where the call reaches the non-deterministic package unchanged and without a
warning of its own.  The deterministic names take no ``deterministic`` keyword (``TypeError``); the deterministic package's names before the det rename are gone.  The entry points
are marked experimental and mirror the package's signatures; ``import flashinfer`` imports neither module.  The
opt-in ``bwd_grid_mult`` is refused with the deterministic package's errors.  On a device (once the package is
rendered): the entry points return the package's bits (``y`` and ``dx`` bitwise the deterministic package's, the
parameter gradients within budget), at every ``bwd_grid_mult``; under torch's deterministic mode the forward runs and
the backward raises (strict) or warns and runs (``warn_only``), also when the mode changes between a forward and its
backward and through ``torch.utils.checkpoint`` (reentrant and not).  The numerical contract itself is
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
import flashinfer.experimental.cake.cake_causal_conv1d_train_kcp as kcp_facade
import flashinfer.experimental.cake.cake_causal_conv1d_train as facade
import flashinfer.experimental.cake.cake_causal_conv1d_train_backend as backend
from flashinfer.api_logging import ExperimentalWarning

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
SEED = 8066
WIDTH = 4
DW_ATOL, DW_RTOL = 5e-2, 1e-4
# Every deterministic value but False: refused with a ValueError that names the deterministic entry point.
REFUSED_MODES = (True, None, 0, 1, "False", torch.tensor(False))

# entry point -> deterministic package name
ENTRIES = {
    "cake_causal_conv1d_train_forward": "cake_causal_conv1d_det_train_forward",
    "cake_causal_conv1d_train_backward": "cake_causal_conv1d_det_train_backward",
    "cake_causal_conv1d": "cake_causal_conv1d_det",
}
HELPERS = {
    "cake_causal_conv1d_train_forward_workspace": "cake_causal_conv1d_det_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes": "cake_causal_conv1d_det_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace": "cake_causal_conv1d_det_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes": "cake_causal_conv1d_det_train_backward_workspace_bytes",
}
SUPPORTED = (
    "is_cake_causal_conv1d_train_supported",
    "is_cake_causal_conv1d_det_train_supported",
)
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
    """The package behind the single-rank module replaced by recorders; the deterministic package's entry points
    replaced by recorders too (none may run)."""

    log: list = []
    nondet_names = [*ENTRIES, *HELPERS, SUPPORTED[0]]
    monkeypatch.setattr(
        facade, "_backend", lambda: _namespace("nondet", nondet_names, log)
    )
    for det_name in [*ENTRIES.values(), *HELPERS.values(), SUPPORTED[1]]:
        monkeypatch.setattr(det_facade, det_name, _Recorder(f"det.{det_name}", log))
    yield log


@pytest.fixture
def kcp_recorded(monkeypatch):
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as kcp_backend,
    )

    log: list = []
    for name, det_name in KCP_ENTRIES.items():
        monkeypatch.setattr(det_kcp_facade, det_name, _Recorder(f"det.{det_name}", log))
        monkeypatch.setattr(kcp_backend, name, _Recorder(f"nondet.{name}", log))
    yield log


@pytest.fixture(params=[False, True], ids=["strict", "warn_only"])
def torch_deterministic(request):
    """torch's deterministic mode, strict and with ``warn_only=True``."""

    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=request.param)
    try:
        yield request.param
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@pytest.fixture
def torch_mode():
    """Set torch's deterministic mode inside a test (``None`` = off); the caller's mode is restored afterwards."""

    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    def set_mode(warn_only):
        if warn_only is None:
            torch.use_deterministic_algorithms(False)
        else:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

    try:
        yield set_mode
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def _quiet(call):
    """``call()`` with the one-time experimental notices silenced; returns ``(result, other warnings)``."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ExperimentalWarning)
        result = call()
    return result, list(caught)


def _refused(entry: str, det_entry: str):
    """The ``ValueError`` of a refused mode: it names the entry point, the deterministic one, and ``False``."""

    return pytest.raises(
        ValueError,
        match=rf"^{entry} runs the non-deterministic package .* needs deterministic=False, got .*; the "
        rf"deterministic entry point is flashinfer\.cake_causal_conv1d_det_train(_kcp)?\.{det_entry} \(pass "
        r"deterministic=False here",
    )


# ----------------------------------------------------------------------------------------------------------------
# The mode rule (CPU).


@pytest.mark.cpu_only
def test_entry_points_run_the_package_with_false(recorded) -> None:
    x, w, cu, dy, b, ws = (object() for _ in range(6))
    calls = {
        "cake_causal_conv1d_train_forward": ((x, w, cu), {"bias": b, "out": None, "workspace": ws}),
        "cake_causal_conv1d_train_backward": (
            (x, w, cu, dy),
            {"bias": b, "silu_grad_preactivation": "fp32", "workspace": ws, "dx": None, "dweight": None, "dbias": None,
             "bwd_grid_mult": 3},
        ),
        "cake_causal_conv1d": ((x, w, cu, b), {"silu_grad_preactivation": "fp32", "bwd_grid_mult": 3}),
    }  # fmt: skip
    for entry in ENTRIES:
        args, kwargs = calls[entry]
        recorded.clear()
        result, caught = _quiet(
            lambda: getattr(facade, entry)(*args, **kwargs, deterministic=False)
        )
        assert result == f"nondet.{entry}" and caught == [], entry
        ((name, got_args, got_kwargs),) = recorded
        assert got_args == args and got_kwargs == {**kwargs, "deterministic": False}, (
            entry
        )


@pytest.mark.cpu_only
def test_helpers_run_the_package_with_false(recorded) -> None:
    device = torch.device("cpu")
    for helper in HELPERS:
        # The backward helpers size for a bwd_grid_mult (passed on by keyword); the forward ones don't.
        sized = {"bwd_grid_mult": 4} if "backward" in helper else {}
        recorded.clear()
        assert (
            getattr(facade, helper)(64, 256, device, **sized, deterministic=False)
            == f"nondet.{helper}"
        )
        assert recorded == [
            (f"nondet.{helper}", (64, 256, device), {**sized, "deterministic": False})
        ], helper
    recorded.clear()
    assert (
        facade.is_cake_causal_conv1d_train_supported(device, deterministic=False)
        == "nondet.is_cake_causal_conv1d_train_supported"
    )
    # The package's support query takes no mode keyword.
    assert recorded == [("nondet.is_cake_causal_conv1d_train_supported", (device,), {})]


@pytest.mark.cpu_only
def test_every_other_mode_is_refused_naming_the_deterministic_entry(
    recorded, kcp_recorded
) -> None:
    """The default and every value but ``False`` raise ``ValueError`` before anything runs, from every entry point,
    helper and support query of both modules (the KCP keyword path included); nothing reaches either package."""

    positional = {
        "cake_causal_conv1d_train_forward": 3,
        "cake_causal_conv1d_train_backward": 4,
        "cake_causal_conv1d": 3,
    }
    cpu = torch.device("cpu")
    for mode in ({}, *({"deterministic": bad} for bad in REFUSED_MODES)):
        for entry, det_name in ENTRIES.items():
            args = (None,) * positional[entry]
            with _refused(entry, det_name):
                _quiet(lambda: getattr(facade, entry)(*args, **mode))
        for helper, det_name in HELPERS.items():
            with _refused(helper, det_name):
                getattr(facade, helper)(64, 256, cpu, **mode)
        with _refused(*SUPPORTED):
            facade.is_cake_causal_conv1d_train_supported(cpu, **mode)
        for entry, det_name in KCP_ENTRIES.items():
            for keywords in ({}, {"cooperative": False}, {"bwd_reserve_sms": 1}):
                with _refused(entry, det_name):
                    _quiet(
                        lambda: getattr(kcp_facade, entry)(
                            None, None, None, **keywords, **mode
                        )
                    )
    assert recorded == [] and kcp_recorded == []


@pytest.mark.cpu_only
def test_bwd_grid_mult_is_validated_like_the_deterministic_package() -> None:
    """A ``bwd_grid_mult`` outside ``BWD_GRID_MULTS`` raises the deterministic package's ``TypeError`` (a non-int,
    ``bool`` included) or ``ValueError`` (another int) with the same message, from every entry point that takes it
    (and the package's autograd Function), before any tensor or device check: never a silent default."""

    cpu = torch.device("cpu")
    calls = {
        "backward": (
            lambda m: facade.cake_causal_conv1d_train_backward(None, None, None, None, bwd_grid_mult=m, deterministic=False),
            lambda m: det_facade.cake_causal_conv1d_det_train_backward(None, None, None, None, bwd_grid_mult=m),
        ),
        "autograd": (
            lambda m: facade.cake_causal_conv1d(None, None, None, bwd_grid_mult=m, deterministic=False),
            lambda m: det_facade.cake_causal_conv1d_det(None, None, None, bwd_grid_mult=m),
        ),
        "bwd_ws": (
            lambda m: facade.cake_causal_conv1d_train_backward_workspace(64, 256, cpu, bwd_grid_mult=m, deterministic=False),
            lambda m: det_facade.cake_causal_conv1d_det_train_backward_workspace(64, 256, cpu, bwd_grid_mult=m),
        ),
        "bwd_ws_bytes": (
            lambda m: facade.cake_causal_conv1d_train_backward_workspace_bytes(64, 256, cpu, bwd_grid_mult=m, deterministic=False),
            lambda m: det_facade.cake_causal_conv1d_det_train_backward_workspace_bytes(64, 256, cpu, bwd_grid_mult=m),
        ),
    }  # fmt: skip
    for bad in (0, 5, -1, True, False, 2.0, "2", None, torch.tensor(2)):
        expected = ValueError if type(bad) is int else TypeError
        for name, pair in calls.items():
            errors = []
            for call in pair:
                with pytest.raises((TypeError, ValueError)) as info:
                    _quiet(lambda: call(bad))
                errors.append((type(info.value), str(info.value)))
            assert errors[0] == errors[1], (name, bad, errors)
            assert errors[0][0] is expected, (name, bad, errors)
            assert errors[0][1].startswith("bwd_grid_mult must be"), (name, bad, errors)
        with pytest.raises(expected, match="bwd_grid_mult must be"):
            backend.CakeCausalConv1dFunction.apply(
                None, None, None, None, "bf16", bad, False
            )


@pytest.mark.cpu_only
def test_torch_deterministic_mode_changes_nothing_here(
    recorded, kcp_recorded, torch_deterministic
) -> None:
    """Strict and ``warn_only=True`` alike: ``deterministic=False`` reaches the non-deterministic package unchanged
    and without a warning from these modules (the package's backward raises PyTorch's alert itself); every other
    value is still refused; nothing runs the deterministic package."""

    x, w, cu, dy, b, ctx = (object() for _ in range(6))
    # The autograd entry takes the bias positionally (and passes it on so); give it one.
    entry_args = {
        "cake_causal_conv1d_train_forward": (x, w, cu),
        "cake_causal_conv1d_train_backward": (x, w, cu, dy),
        "cake_causal_conv1d": (x, w, cu, b),
    }
    for entry in ENTRIES:
        args = entry_args[entry]
        recorded.clear()
        result, caught = _quiet(
            lambda: getattr(facade, entry)(*args, deterministic=False)
        )
        assert result == f"nondet.{entry}" and caught == [], (entry, caught)
        ((name, got_args, got_kwargs),) = recorded
        assert name == f"nondet.{entry}" and got_args == args, entry
        assert got_kwargs["deterministic"] is False, entry
        with _refused(entry, ENTRIES[entry]):
            _quiet(lambda: getattr(facade, entry)(*args))
    for entry in KCP_ENTRIES:
        kcp_recorded.clear()
        result, caught = _quiet(
            lambda: getattr(kcp_facade, entry)(
                x, w, ctx, deterministic=False, bwd_reserve_sms=1
            )
        )
        assert result == f"nondet.{entry}" and caught == [], (entry, caught)
        assert kcp_recorded == [
            (
                f"nondet.{entry}",
                (x, w, ctx),
                {
                    "recompute_handle": None,
                    "deterministic": False,
                    "bwd_reserve_sms": 1,
                },
            )
        ]
    assert not any(name.startswith("det.") for name, *_ in recorded + kcp_recorded)


@pytest.mark.cpu_only
def test_kcp_entry_points_run_the_package_with_false(kcp_recorded) -> None:
    """``recompute_handle`` always reaches the package; ``cooperative`` and ``bwd_reserve_sms`` only when given
    (``None`` keeps the package's defaults); the package validates them."""

    x, w, ctx, handle = (object() for _ in range(4))
    for entry in KCP_ENTRIES:
        for keywords in (
            {},
            {"cooperative": False},
            {"cooperative": True},
            {"bwd_reserve_sms": 0},
            {"bwd_reserve_sms": 2, "cooperative": None},
        ):
            kcp_recorded.clear()
            result, caught = _quiet(
                lambda: getattr(kcp_facade, entry)(
                    x, w, ctx, recompute_handle=handle, deterministic=False, **keywords
                )
            )
            assert result == f"nondet.{entry}" and caught == []
            given = {k: v for k, v in keywords.items() if v is not None}
            assert kcp_recorded == [
                (
                    f"nondet.{entry}",
                    (x, w, ctx),
                    {"recompute_handle": handle, "deterministic": False, **given},
                )
            ], (entry, keywords)


@pytest.mark.cpu_only
def test_kcp_package_refuses_a_cooperative_backward() -> None:
    """The non-deterministic package's main backward is one ordinary launch: ``cooperative=True`` is a
    ``ValueError`` and a non-``bool`` a ``TypeError``, both before anything runs, also through the KCP module."""

    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as kcp_backend,
    )

    x, w, ctx = (object() for _ in range(3))
    for entry in KCP_ENTRIES:
        for fn in (getattr(kcp_backend, entry), getattr(kcp_facade, entry)):
            with pytest.raises(
                ValueError,
                match=rf"^{entry}: cooperative=True applies to the deterministic entry point "
                rf"flashinfer\.cake_causal_conv1d_det_train_kcp\.{KCP_ENTRIES[entry]} only; ",
            ):
                _quiet(lambda: fn(x, w, ctx, cooperative=True, deterministic=False))
            with pytest.raises(
                TypeError, match=f"{entry}: cooperative must be None or a bool"
            ):
                _quiet(lambda: fn(x, w, ctx, cooperative=1, deterministic=False))


@pytest.mark.cpu_only
def test_package_refusals_name_the_deterministic_entry() -> None:
    """Every entry point of the package (single-rank and ``kcp``) refuses a mode but ``False`` naming its own
    deterministic entry point, which exists; the map covers exactly the entry points that check the mode."""

    import importlib
    import re
    from pathlib import Path

    sources = [
        Path(backend.__file__),
        Path(backend.__file__).with_name("kcp.py"),
    ]
    checked = {
        name
        for path in sources
        for name in re.findall(
            r'_require_nondeterministic\(\s*deterministic,\s*"(\w+)"', path.read_text()
        )
    }
    assert checked == set(backend._DET_ENTRIES), checked ^ set(backend._DET_ENTRIES)
    for entry, det_path in backend._DET_ENTRIES.items():
        module, _, name = det_path.rpartition(".")
        assert "det" in name.lower() and hasattr(
            importlib.import_module(module), name
        ), det_path
        for bad in REFUSED_MODES:
            with pytest.raises(
                ValueError,
                match=rf"^{entry} is the non-deterministic implementation .* needs deterministic=False, got "
                rf".*; the deterministic entry point is {re.escape(det_path)} \(pass deterministic=False here",
            ):
                backend._require_nondeterministic(bad, entry)
        backend._require_nondeterministic(False, entry)
    cpu = torch.device("cpu")
    calls = {
        "cake_causal_conv1d_train_forward": lambda: backend.cake_causal_conv1d_train_forward(None, None, None),
        "cake_causal_conv1d_train_backward": lambda: backend.cake_causal_conv1d_train_backward(None, None, None, None),
        "cake_causal_conv1d_train_backward_workspace_bytes": lambda: backend.cake_causal_conv1d_train_backward_workspace_bytes(64, 256, cpu),
    }  # fmt: skip
    for entry, call in calls.items():
        with pytest.raises(ValueError, match=re.escape(backend._DET_ENTRIES[entry])):
            call()


# ----------------------------------------------------------------------------------------------------------------
# Surface (CPU).


def _parameters(fn) -> list[tuple]:
    return [
        (p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()
    ]


@pytest.mark.cpu_only
def test_entry_points_mirror_the_package() -> None:
    """The single-rank module's entry points and helpers take exactly the package's parameters (``deterministic=True``
    keyword-only last), which are the deterministic package's plus ``deterministic``; the KCP module's take a subset
    of the package's, ending in ``deterministic`` and ``bwd_reserve_sms``."""

    for entry, det_name in {**ENTRIES, **HELPERS}.items():
        mine = _parameters(getattr(facade, entry))
        assert mine == _parameters(getattr(backend, entry)), entry
        assert mine[-1] == ("deterministic", inspect.Parameter.KEYWORD_ONLY, True), (
            entry
        )
        assert mine[:-1] == _parameters(getattr(det_facade, det_name)), entry
    assert _parameters(facade.is_cake_causal_conv1d_train_supported) == [
        ("device", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("deterministic", inspect.Parameter.KEYWORD_ONLY, True),
    ]
    for name in (
        "cake_causal_conv1d_train_backward",
        "cake_causal_conv1d",
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_train_backward_workspace_bytes",
    ):
        for module in (facade, backend):
            parameter = inspect.signature(getattr(module, name)).parameters[
                "bwd_grid_mult"
            ]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (module, name)
            assert parameter.default == 1, (module, name)
    function = list(
        inspect.signature(backend.CakeCausalConv1dFunction.forward).parameters
    )
    assert function[-2:] == ["bwd_grid_mult", "deterministic"], function
    assert backend.BWD_GRID_MULTS == (1, 2, 3, 4)
    assert facade.BWD_GRID_MULTS is backend.BWD_GRID_MULTS
    assert "BWD_GRID_MULTS" in backend.__all__ and "BWD_GRID_MULTS" in facade.__all__
    import flashinfer.cake_causal_conv1d_det_train_backend as det_backend

    assert backend.BWD_GRID_MULTS == det_backend.BWD_GRID_MULTS
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as kcp_backend,
    )

    for entry in KCP_ENTRIES:
        mine = _parameters(getattr(kcp_facade, entry))
        assert mine[-2:] == [
            ("deterministic", inspect.Parameter.KEYWORD_ONLY, True),
            ("bwd_reserve_sms", inspect.Parameter.KEYWORD_ONLY, None),
        ], entry
        theirs = {p[0]: p for p in _parameters(getattr(kcp_backend, entry))}
        assert all(theirs.get(p[0]) == p for p in mine), entry


@pytest.mark.cpu_only
def test_entry_points_are_experimental() -> None:
    for entry in (*ENTRIES, *HELPERS, SUPPORTED[0]):
        assert getattr(getattr(facade, entry), "is_experimental", False) is True, entry
    for entry in KCP_ENTRIES:
        assert getattr(getattr(kcp_facade, entry), "is_experimental", False) is True, (
            entry
        )
    assert set(facade.__all__) >= set(ENTRIES) | set(HELPERS) | {SUPPORTED[0]}
    assert set(kcp_facade.__all__) == set(KCP_ENTRIES) | {"RecomputeHandleKcp"}
    assert facade.CakeCausalConv1dFunction is backend.CakeCausalConv1dFunction


@pytest.mark.cpu_only
def test_deterministic_names_take_no_mode_keyword() -> None:
    """The deterministic package's entry points carry ``det``, take no ``deterministic`` keyword (passing one is a
    ``TypeError`` from the call itself) and are other objects than these modules' entry points."""

    cpu = torch.device("cpu")
    det_calls = {
        "cake_causal_conv1d_det": lambda **k: det_facade.cake_causal_conv1d_det(None, None, None, **k),
        "cake_causal_conv1d_det_train_forward": lambda **k: det_facade.cake_causal_conv1d_det_train_forward(None, None, None, **k),
        "cake_causal_conv1d_det_train_backward": lambda **k: det_facade.cake_causal_conv1d_det_train_backward(None, None, None, None, **k),
        "cake_causal_conv1d_det_train_backward_workspace_bytes": lambda **k: det_facade.cake_causal_conv1d_det_train_backward_workspace_bytes(64, 256, cpu, **k),
        "is_cake_causal_conv1d_det_train_supported": lambda **k: det_facade.is_cake_causal_conv1d_det_train_supported(cpu, **k),
        "cake_causal_conv1d_det_kcp": lambda **k: det_kcp_facade.cake_causal_conv1d_det_kcp(None, None, None, **k),
        "cake_causal_conv1d_det_multi_kcp": lambda **k: det_kcp_facade.cake_causal_conv1d_det_multi_kcp(None, None, None, **k),
    }  # fmt: skip
    for name, call in det_calls.items():
        module = det_kcp_facade if name.endswith("_kcp") else det_facade
        assert "det" in name
        assert (
            "deterministic" not in inspect.signature(getattr(module, name)).parameters
        ), name
        for mode in (True, False):
            with pytest.raises(TypeError, match="deterministic"):
                call(deterministic=mode)
    for entry, det_name in {**ENTRIES, **HELPERS}.items():
        assert getattr(facade, entry) is not getattr(det_facade, det_name), entry
    for entry, det_name in KCP_ENTRIES.items():
        assert getattr(kcp_facade, entry) is not getattr(det_kcp_facade, det_name), (
            entry
        )


@pytest.mark.cpu_only
def test_deterministic_names_are_unchanged_and_carry_det() -> None:
    """Every deterministic name is the deterministic object and carries ``det``; the names the deterministic package
    had before the det rename are bound nowhere, and the same-spelled non-deterministic names live in these modules
    only."""

    for name in (
        "cake_causal_conv1d_det",
        "cake_causal_conv1d_det_train_forward",
        "cake_causal_conv1d_det_train_backward",
        "CakeCausalConv1dDetFunction",
    ):
        assert getattr(flashinfer, name) is getattr(det_facade, name), name
    for name in (
        "cake_causal_conv1d",
        "cake_causal_conv1d_train_forward",
        "cake_causal_conv1d_train_backward",
        "CakeCausalConv1dFunction",
        "cake_causal_conv1d_train",
        "cake_causal_conv1d_train_backend",
    ):
        assert not hasattr(flashinfer, name), name
    for name in (
        "cake_causal_conv1d_kcp",
        "cake_causal_conv1d_kcp_multi",
        "KcpRecomputeHandle",
    ):
        assert not hasattr(det_facade, name), name
        assert not hasattr(det_kcp_facade, name), name
    for entry in ENTRIES:
        assert not hasattr(det_facade, entry), entry
    assert not hasattr(det_facade, "CakeCausalConv1dFunction")
    assert (
        det_kcp_facade.cake_causal_conv1d_det_kcp
        is not kcp_facade.cake_causal_conv1d_kcp
    )


@pytest.mark.cpu_only
def test_core_does_not_import_the_experimental_package() -> None:
    """``import flashinfer`` imports neither module nor the non-deterministic package; the KCP module imports
    neither the package nor the single-rank module until a call; the single-rank module imports the
    non-deterministic package but no KCP host module."""

    code = (
        "import sys\n"
        "import flashinfer\n"
        "cake = 'flashinfer.experimental.cake'\n"
        "loaded = sorted(m for m in sys.modules if m == cake or m.startswith(cake + '.'))\n"
        "assert not loaded, loaded\n"
        "import flashinfer.experimental.cake.cake_causal_conv1d_train_kcp\n"
        "loaded = sorted(m for m in sys.modules if m.startswith(cake + '.') and m != cake + '.cake_causal_conv1d_train_kcp')\n"
        "assert not loaded, loaded\n"
        "import flashinfer.experimental.cake.cake_causal_conv1d_train\n"
        "kcp = [m for m in sys.modules if m.endswith(('.kcp', '.kcp_jit')) and 'conv1d' in m]\n"
        "assert not kcp, kcp\n"
        "print('IMPORT_ISOLATION_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0 and "IMPORT_ISOLATION_OK" in r.stdout, (r.stdout, r.stderr)


# ----------------------------------------------------------------------------------------------------------------
# On a device: the entry points return the package's bits; torch's deterministic mode.


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
ND = {"deterministic": False}
ALERT = "cake_causal_conv1d_train_backward does not have a deterministic implementation"


def _problem(S: int = 1000, C: int = 512):
    generator = torch.Generator(device=_device()).manual_seed(SEED)
    x = torch.randn(S, C, device=_device(), generator=generator).to(torch.bfloat16)
    dy = torch.randn(S, C, device=_device(), generator=generator).to(torch.bfloat16)
    w = 0.5 * torch.randn(C, WIDTH, device=_device(), generator=generator)
    b = 0.5 * torch.randn(C, device=_device(), generator=generator)
    cu = torch.tensor([0, 369, 659, 868, S], dtype=torch.int32, device=_device())
    return x, w, cu, dy, b


def _close(got, want) -> None:
    torch.testing.assert_close(got, want, atol=DW_ATOL, rtol=DW_RTOL)


@gpu
def test_entry_points_return_the_packages_bits() -> None:
    x, w, cu, dy, b = _problem()
    S, C = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        y_det = det_facade.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)
        y = facade.cake_causal_conv1d_train_forward(x, w, cu, bias=b, **ND)
        assert torch.equal(y, y_det)
        det_grads = det_facade.cake_causal_conv1d_det_train_backward(
            x, w, cu, dy, bias=b
        )
        ws = facade.cake_causal_conv1d_train_backward_workspace(S, C, _device(), **ND)
        assert ws.dtype == torch.uint8 and int(torch.count_nonzero(ws)) == 0
        assert ws.numel() == facade.cake_causal_conv1d_train_backward_workspace_bytes(
            S, C, _device(), **ND
        )
        dx, dw, db = facade.cake_causal_conv1d_train_backward(
            x, w, cu, dy, bias=b, workspace=ws, **ND
        )
        assert int(torch.count_nonzero(ws)) == 0
        assert torch.equal(dx, det_grads[0])
        _close(dw, det_grads[1])
        _close(db, det_grads[2])
        assert facade.cake_causal_conv1d_train_forward_workspace_bytes(
            S, C, _device(), **ND
        ) == det_facade.cake_causal_conv1d_det_train_forward_workspace_bytes(
            S, C, _device()
        )
        assert facade.is_cake_causal_conv1d_train_supported(_device(), **ND) is True


@gpu
def test_bwd_grid_mult() -> None:
    """Every ``bwd_grid_mult``: ``dx`` bitwise the deterministic default's and the gradients within its budget,
    with a workspace of the same size at every multiple, left zero (explicit and autograd entries)."""

    x, w, cu, dy, b = _problem()
    S, C = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        default = det_facade.cake_causal_conv1d_det_train_backward(x, w, cu, dy, bias=b)
        nbytes = facade.cake_causal_conv1d_train_backward_workspace_bytes(
            S, C, _device(), **ND
        )
        for mult in backend.BWD_GRID_MULTS:
            ws = facade.cake_causal_conv1d_train_backward_workspace(
                S, C, _device(), bwd_grid_mult=mult, **ND
            )
            assert ws.numel() == nbytes and int(torch.count_nonzero(ws)) == 0
            dx, dw, db = facade.cake_causal_conv1d_train_backward(
                x, w, cu, dy, bias=b, workspace=ws, bwd_grid_mult=mult, **ND
            )
            assert int(torch.count_nonzero(ws)) == 0
            assert torch.equal(dx, default[0]), mult
            _close(dw, default[1])
            _close(db, default[2])
            xg = x.clone().requires_grad_(True)
            wg = w.clone().requires_grad_(True)
            bg = b.clone().requires_grad_(True)
            facade.cake_causal_conv1d(
                xg, wg, cu, bg, bwd_grid_mult=mult, **ND
            ).backward(dy)
            assert torch.equal(xg.grad, default[0]), mult
            _close(wg.grad, default[1])
            _close(bg.grad, default[2])


def _autograd(x0, w0, cu, dy, b0=None, *, before_backward=None, checkpointed=None):
    """``(y, x.grad, w.grad[, b.grad])`` of one autograd pass through the single-rank module; ``checkpointed``
    (``True`` / ``False``) wraps the call in ``torch.utils.checkpoint`` with that ``use_reentrant``."""

    from torch.utils.checkpoint import checkpoint

    leaves = [t.clone().requires_grad_(True) for t in (x0, w0, b0) if t is not None]

    def layer(*args):
        return facade.cake_causal_conv1d(
            args[0], args[1], cu, args[2] if len(args) > 2 else None, **ND
        )

    if checkpointed is None:
        y = layer(*leaves)
    else:
        y = checkpoint(layer, *leaves, use_reentrant=checkpointed)
    if before_backward is not None:
        before_backward()
    y.backward(dy)
    return (y.detach(), *(leaf.grad for leaf in leaves))


@gpu
def test_autograd_returns_the_packages_bits() -> None:
    x, w, cu, dy, b = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        expected = det_facade.cake_causal_conv1d_det_train_backward(
            x, w, cu, dy, bias=b
        )
        y_det = det_facade.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)
        y, dx, dw, db = _autograd(x, w, cu, dy, b)
    assert torch.equal(y, y_det) and torch.equal(dx, expected[0])
    _close(dw, expected[1])
    _close(db, expected[2])


@gpu
def test_torch_deterministic_mode_alerts_on_the_backward(torch_mode) -> None:
    """Under torch's deterministic mode the forward runs (the deterministic forward's bits); the backward raises
    PyTorch's ``RuntimeError`` (strict) or warns and runs (``warn_only=True``) -- explicit and autograd entries,
    also when the mode is turned on between a forward and its backward.  It never runs the deterministic package
    instead.  A forward under the mode whose backward runs without it runs the package."""

    x, w, cu, dy, _ = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        expected = det_facade.cake_causal_conv1d_det_train_backward(x, w, cu, dy)
        y_det = det_facade.cake_causal_conv1d_det_train_forward(x, w, cu)

        torch_mode(False)
        y = facade.cake_causal_conv1d_train_forward(x, w, cu, **ND)
        assert torch.equal(y, y_det)
        with pytest.raises(RuntimeError, match=ALERT):
            facade.cake_causal_conv1d_train_backward(x, w, cu, dy, **ND)
        with pytest.raises(RuntimeError, match=ALERT):
            _autograd(x, w, cu, dy)
        torch_mode(None)
        with pytest.raises(RuntimeError, match=ALERT):
            _autograd(x, w, cu, dy, before_backward=lambda: torch_mode(False))

        torch_mode(True)
        with pytest.warns(UserWarning, match=ALERT):
            dx, dw = facade.cake_causal_conv1d_train_backward(x, w, cu, dy, **ND)
        assert torch.equal(dx, expected[0])
        _close(dw, expected[1])
        torch_mode(None)
        with pytest.warns(UserWarning, match=ALERT):
            y, dx, dw = _autograd(
                x, w, cu, dy, before_backward=lambda: torch_mode(True)
            )
        assert torch.equal(y, y_det) and torch.equal(dx, expected[0])
        _close(dw, expected[1])

        torch_mode(False)
        y, dx, dw = _autograd(x, w, cu, dy, before_backward=lambda: torch_mode(None))
        assert torch.equal(y, y_det) and torch.equal(dx, expected[0])
        _close(dw, expected[1])


@gpu
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_checkpoint(torch_mode, reentrant) -> None:
    """``torch.utils.checkpoint`` (2 x forward + backward): ``y`` / ``dx`` bitwise the uncheckpointed pass's, the
    gradients within budget; with torch's deterministic mode turned on before the backward, the backward raises:
    the recompute runs the same package."""

    x, w, cu, dy, b = _problem()
    torch_mode(None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        direct = _autograd(x, w, cu, dy, b)
        got = _autograd(x, w, cu, dy, b, checkpointed=reentrant)
        assert torch.equal(got[0], direct[0]) and torch.equal(got[1], direct[1])
        _close(got[2], direct[2])
        _close(got[3], direct[3])
        with pytest.raises(RuntimeError, match=ALERT):
            _autograd(
                x,
                w,
                cu,
                dy,
                b,
                checkpointed=reentrant,
                before_backward=lambda: torch_mode(False),
            )
