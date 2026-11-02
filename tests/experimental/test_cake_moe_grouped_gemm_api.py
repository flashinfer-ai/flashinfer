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

flashinfer public-API tests of the selectable-mode Cake MoE grouped GEMM.

The facade ``flashinfer.experimental.cake.grouped_mm`` (``grouped_gemm_fwd``,
``grouped_gemm_dgrad``, ``grouped_gemm_wgrad``, ``cake_grouped_mm``) dispatches
``deterministic=True`` (the default) to the deterministic package
``flashinfer.experimental.cake_moe_grouped_det_gemm`` and ``deterministic=False``
to ``flashinfer.experimental.cake.cake_moe_grouped_gemm``; torch's deterministic
mode turns ``False`` into ``True`` with one ``UserWarning`` per entry point.  This
suite covers on CPU the experimental-API marking and signatures of the facade,
its dispatch table (with the package functions replaced by recorders), the caller
frame its fallback warning names, its deferred imports, the package's refusal of
every mode but ``deterministic=False``,
the deterministic-mode alert of the weight gradient, and the legacy names that
stay aliases of the deterministic objects; on a GPU, the alert of the autograd
backward.
"""

import inspect
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

import flashinfer.experimental.cake.cake_moe_grouped_gemm as pkg
import flashinfer.experimental.cake.grouped_mm as facade
import flashinfer.experimental.cake_moe_grouped_det_gemm as det_pkg
import flashinfer.grouped_mm as grouped_mm
from flashinfer.experimental.cake.cake_moe_grouped_gemm import cake_autograd
from flashinfer.experimental.cake.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_det_gemm import (
    cake_autograd as det_autograd,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = (
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "cake_grouped_mm",
)
# facade entry -> (deterministic package module, function name)
DET_TARGETS = {
    "grouped_gemm_fwd": (det_pkg, "grouped_det_gemm_fwd"),
    "grouped_gemm_dgrad": (det_pkg, "grouped_det_gemm_dgrad"),
    "grouped_gemm_wgrad": (det_pkg, "grouped_det_gemm_wgrad"),
    "cake_grouped_mm": (det_autograd, "cake_grouped_det_mm"),
}
NEW_TARGETS = {
    "grouped_gemm_fwd": (pkg, "grouped_gemm_fwd"),
    "grouped_gemm_dgrad": (pkg, "grouped_gemm_dgrad"),
    "grouped_gemm_wgrad": (pkg, "grouped_gemm_wgrad"),
    "cake_grouped_mm": (cake_autograd, "cake_grouped_mm"),
}


def _package_function(name):
    module, attr = NEW_TARGETS[name]
    return getattr(module, attr)


class _TorchDeterministic:
    """``torch.use_deterministic_algorithms(True, warn_only=...)`` inside the block, the
    previous setting restored after it."""

    def __init__(self, warn_only=False):
        self.warn_only = warn_only

    def __enter__(self):
        self.saved = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(True, warn_only=self.warn_only)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.saved[0], warn_only=self.saved[1])
        return False


def _run_cpu_subprocess(code, **env):
    """Run ``code`` in a fresh interpreter with no visible GPU and the extra environment
    ``env``; its last stdout line."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()[-1]


# ---------------------------------------------------------------------------
# The facade (CPU)
# ---------------------------------------------------------------------------


def test_facade_entry_points_are_the_experimental_api():
    """Each facade function is marked experimental with the feature
    ``cake_moe_grouped_gemm.<name>``, takes the signature of the package function of
    the same name (``deterministic: bool = True`` included), documents the
    ``deterministic`` keyword behind the experimental banner, and is in ``__all__``;
    the package functions are unmarked."""
    assert sorted(facade.__all__) == sorted(ENTRY_POINTS)
    for name in ENTRY_POINTS:
        entry, impl = getattr(facade, name), _package_function(name)
        assert entry.is_experimental is True, name
        assert entry.experimental_feature == f"cake_moe_grouped_gemm.{name}", name
        assert inspect.signature(entry) == inspect.signature(impl), name
        parameter = inspect.signature(entry).parameters["deterministic"]
        assert parameter.default is True, name
        assert entry.__doc__.startswith(".. warning::"), name
        assert "deterministic" in entry.__doc__, name
        assert not hasattr(impl, "is_experimental"), name
    for name in ENTRY_POINTS[:3]:
        parameter = inspect.signature(getattr(facade, name)).parameters["deterministic"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name


def _recorders(monkeypatch):
    """Replace the package functions the facade hands off to by recorders; return the log."""
    log = []
    for side, targets in (("det", DET_TARGETS), ("new", NEW_TARGETS)):
        for entry, (module, attr) in targets.items():

            def record(*args, _side=side, _entry=entry, **kwargs):
                log.append((_entry, _side, kwargs.get("deterministic", "absent")))
                return "result"

            monkeypatch.setattr(module, attr, record)
    monkeypatch.setattr(facade, "_FALLBACK_WARNED", set())
    return log


_OMIT = object()


def _call(name, deterministic=_OMIT):
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    w = torch.zeros(2, 256, 64, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)
    args = (x, w, offs) if name != "grouped_gemm_wgrad" else (x, x, offs)
    if deterministic is _OMIT:
        return getattr(facade, name)(*args)
    return getattr(facade, name)(*args, deterministic=deterministic)


def _user_warnings(records):
    return [w for w in records if w.category is UserWarning]


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_facade_dispatch_table(name, monkeypatch):
    """``deterministic`` omitted or ``True``: the deterministic package; ``False``: the
    new package with ``deterministic=False``; ``False`` under torch's deterministic mode
    (also with ``warn_only``): the deterministic package and exactly one
    ``UserWarning`` for the entry point over repeated calls."""
    log = _recorders(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert _call(name) == "result"
        assert _call(name, True) == "result"
        assert _call(name, False) == "result"
    assert _user_warnings(rec) == []
    det_kw = True if name == "cake_grouped_mm" else "absent"
    assert log == [
        (name, "det", det_kw),
        (name, "det", det_kw),
        (name, "new", False),
    ]
    log.clear()
    for warn_only in (False, True):
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            for _ in range(3):
                assert _call(name, False) == "result"
            assert _call(name, True) == "result"
        user = _user_warnings(rec)
        # One warning per entry point and process (the second block is silent).
        assert len(user) == (1 if not warn_only else 0), [str(w.message) for w in user]
        if user:
            assert f"{name}(deterministic=False)" in str(user[0].message)
            # The warning names the caller's line (``_call`` in this file), not a frame
            # of the facade or of its decorators.
            assert os.path.samefile(user[0].filename, __file__), user[0].filename
    assert log == [(name, "det", det_kw)] * 8


_FALLBACK_CALLER = """
import warnings

import torch

import flashinfer.experimental.cake.grouped_mm as facade

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


offs = torch.tensor([2, 4], dtype=torch.int32)
operands = {
    "grouped_gemm_fwd": (bf16(4, 64), bf16(2, 256, 64), offs),
    "grouped_gemm_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    "grouped_gemm_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_grouped_mm": (bf16(4, 64), bf16(2, 256, 64), offs),
}
torch.use_deterministic_algorithms(True)
for name, args in operands.items():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            getattr(facade, name)(*args, deterministic=False)
        except ValueError:  # CPU operands: the deterministic package refuses them
            pass
        else:
            raise AssertionError("CPU operands must be rejected on the host")
    [w] = [w for w in rec if w.category is UserWarning]
    assert w.filename == "<string>", (name, w.filename, w.lineno)
print("OK")
"""


@pytest.mark.parametrize("log_level", ["0", "1", "3"])
def test_fallback_warning_names_the_caller(log_level):
    """The deterministic-fallback ``UserWarning`` points at the caller's frame (the
    ``-c`` program) at every ``FLASHINFER_LOGLEVEL``: API logging adds wrapper frames
    between the caller and the facade, so a fixed ``stacklevel`` would name a frame of
    flashinfer instead."""
    out = _run_cpu_subprocess(
        _FALLBACK_CALLER, FLASHINFER_LOGLEVEL=log_level, FLASHINFER_LOGDEST="stderr"
    )
    assert out == "OK"


@pytest.mark.parametrize("name", ENTRY_POINTS)
@pytest.mark.parametrize("value", [0, 1, None, "False"])
def test_facade_rejects_a_non_bool_mode(name, value, monkeypatch):
    _recorders(monkeypatch)
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        _call(name, value)


_WARN_ONCE = """
import warnings

import torch

import flashinfer.api_logging as api_logging
import flashinfer.experimental.cake.grouped_mm as facade

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


# CPU operands fail the device check of either package: every call raises the
# host-side ValueError and touches no kernel.
offs = torch.tensor([2, 4], dtype=torch.int32)
operands = {
    "grouped_gemm_fwd": (bf16(4, 64), bf16(2, 256, 64), offs),
    "grouped_gemm_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    "grouped_gemm_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_grouped_mm": (bf16(4, 64), bf16(2, 256, 64), offs),
}


def warnings_of(fn, args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn(*args, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("CPU operands must be rejected on the host")
    return rec


for name, args in operands.items():
    entry = getattr(facade, name)
    [w] = warnings_of(entry, args)  # first call: exactly one warning
    assert w.category is api_logging.ExperimentalWarning, (name, w.category)
    message = str(w.message)
    assert f"'cake_moe_grouped_gemm.{name}' is an experimental" in message, message
    assert warnings_of(entry, args) == [], name  # second call: silent
    assert warnings_of(entry, args, deterministic=False) == [], name
print("OK")
"""


def test_facade_warns_once_per_entry_point():
    """Each facade function emits ``ExperimentalWarning`` once, on its first call, in
    both modes together; runs in a subprocess so the once-only flags do not depend on
    test order."""
    assert _run_cpu_subprocess(_WARN_ONCE) == "OK"


def test_importing_the_facade_imports_neither_package():
    """The facade reaches both packages through deferred imports only."""
    code = (
        "import json, sys\n"
        "import flashinfer.experimental.cake.grouped_mm\n"
        "import flashinfer.grouped_mm\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('flashinfer.experimental.cake_moe_grouped_det_gemm')"
        " or m.startswith('flashinfer.experimental.cake.cake_moe_grouped_gemm'))))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


# ---------------------------------------------------------------------------
# The package's numerical-mode checks and the deterministic-mode alert (CPU)
# ---------------------------------------------------------------------------


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


OFFS = torch.tensor([2, 4], dtype=torch.int32)
PACKAGE_CALLS = {
    "grouped_gemm_fwd": lambda **kw: pkg.grouped_gemm_fwd(
        _bf16(4, 64), _bf16(2, 256, 64), OFFS, **kw
    ),
    "grouped_gemm_dgrad": lambda **kw: pkg.grouped_gemm_dgrad(
        _bf16(4, 64), _bf16(2, 64, 256), OFFS, **kw
    ),
    "grouped_gemm_wgrad": lambda **kw: pkg.grouped_gemm_wgrad(
        _bf16(4, 256), _bf16(4, 64), OFFS, **kw
    ),
    "prepare_grouped_gemm_fwd": lambda **kw: cb.prepare_grouped_gemm_fwd(
        _bf16(4, 64), _bf16(2, 256, 64), OFFS, **kw
    ),
    "prepare_grouped_gemm_dgrad": lambda **kw: cb.prepare_grouped_gemm_dgrad(
        _bf16(4, 64), _bf16(2, 64, 256), OFFS, **kw
    ),
    "prepare_grouped_gemm_wgrad": lambda **kw: cb.prepare_grouped_gemm_wgrad(
        _bf16(4, 256), _bf16(4, 64), OFFS, **kw
    ),
    "cake_grouped_mm": lambda **kw: pkg.cake_grouped_mm(
        _bf16(4, 64), _bf16(2, 256, 64), OFFS, **kw
    ),
}


@pytest.mark.parametrize("name", list(PACKAGE_CALLS))
def test_package_refuses_every_mode_but_false(name):
    """Every entry point of the package raises ``ValueError`` naming the deterministic
    package unless ``deterministic=False``; with ``False`` it reaches its host checks
    (CPU operands: the device check)."""
    call = PACKAGE_CALLS[name]
    for kwargs in (
        {},
        {"deterministic": True},
        {"deterministic": 1},
        {"deterministic": None},
    ):
        with pytest.raises(ValueError, match="cake_moe_grouped_det_gemm") as info:
            call(**kwargs)
        assert "deterministic=False" in str(info.value), (name, kwargs)
    with pytest.raises(ValueError) as info:
        call(deterministic=False)
    assert "cake_moe_grouped_det_gemm" not in str(info.value), name


@pytest.mark.parametrize("name", ["grouped_gemm_wgrad", "prepare_grouped_gemm_wgrad"])
def test_weight_gradient_alerts_under_torch_deterministic_mode(name):
    """Under ``torch.use_deterministic_algorithms(True)`` the weight-gradient entry points
    raise the non-determinism alert before anything else: a ``RuntimeError``
    naming the entry point and the deterministic package, or with ``warn_only=True``
    one ``UserWarning`` saying the same and naming the caller's line, after which the
    call proceeds (CPU operands: the device check).  No fallback to the deterministic
    package happens."""
    call = PACKAGE_CALLS[name]
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match=name) as info,
    ):
        call(deterministic=False)
    assert "cake_moe_grouped_det_gemm" in str(info.value)
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            call(deterministic=False)
    alerts = [w for w in rec if name in str(w.message)]
    assert len(alerts) == 1, [str(w.message) for w in rec]
    assert alerts[0].category is UserWarning
    assert "cake_moe_grouped_det_gemm" in str(alerts[0].message)
    # The caller's frame (the ``PACKAGE_CALLS`` entry in this file), not the package's.
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    # Outside torch's deterministic mode there is no alert.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            call(deterministic=False)
    assert rec == []


def test_wgrad_launch_checks_deterministic_mode_before_launching():
    """A weight-gradient ``GroupedGemmLaunch`` checks torch's deterministic mode at every
    ``launch()``, before it launches anything: a ``RuntimeError`` naming
    ``GroupedGemmLaunch.launch`` and the deterministic package."""
    launch = cb.GroupedGemmLaunch(
        op="wgrad",
        record_name="cake_moe_grouped_gemm_wgrad_bf16_k512",
        stages=(),
        plan={},
        out=torch.empty(0, dtype=torch.bfloat16),
        launches=(),
    )
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match="GroupedGemmLaunch.launch") as info,
    ):
        launch.launch()
    assert "cake_moe_grouped_det_gemm" in str(info.value)


@pytest.mark.parametrize(
    "name",
    [
        "grouped_gemm_fwd",
        "grouped_gemm_dgrad",
        "prepare_grouped_gemm_fwd",
        "prepare_grouped_gemm_dgrad",
    ],
)
def test_forward_and_data_gradient_do_not_alert(name):
    """The forward and data-gradient entry points give the deterministic bits and do not
    alert under torch's deterministic mode."""
    with (
        _TorchDeterministic(warn_only=False),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        with pytest.raises(ValueError):  # CPU operands: the device check
            PACKAGE_CALLS[name](deterministic=False)
    assert rec == []


def test_package_has_no_stable_api_handoff():
    """``grouped_mm_bf16(backend="cake")`` stays on the deterministic package: the new
    package has no handoff function for it."""
    for name in ("grouped_mm_bf16_cake", "grouped_det_mm_bf16_cake"):
        assert not hasattr(cb, name), name
    source = Path(grouped_mm.core.__file__).read_text()
    assert "experimental.cake_moe_grouped_det_gemm.cake_backend import" in source
    assert "experimental.cake.cake_moe_grouped_gemm" not in source


# ---------------------------------------------------------------------------
# Legacy names stay deterministic (CPU)
# ---------------------------------------------------------------------------


def test_legacy_names_remain_the_deterministic_objects():
    """The det-free legacy names of ``flashinfer.grouped_mm`` and of the legacy package
    path ``flashinfer.experimental.cake_moe_grouped_gemm`` resolve to the deterministic
    objects, never to the facade or the new package."""
    import flashinfer.experimental.cake_moe_grouped_gemm as legacy

    for old, det in (
        ("grouped_gemm_fwd", "grouped_det_gemm_fwd"),
        ("grouped_gemm_dgrad", "grouped_det_gemm_dgrad"),
        ("grouped_gemm_wgrad", "grouped_det_gemm_wgrad"),
        ("cake_grouped_mm", "cake_grouped_det_mm"),
    ):
        assert getattr(grouped_mm, old) is getattr(grouped_mm, det), old
        assert getattr(grouped_mm, old) is not getattr(facade, old), old
        assert getattr(legacy, old) is getattr(det_pkg, det), old
        assert getattr(legacy, old) is not getattr(pkg, old), old
    for old, det in (
        ("CakeGroupedMm", "CakeGroupedDetMm"),
        ("GroupedGemmLaunch", "GroupedDetGemmLaunch"),
        ("prepare_grouped_gemm_fwd", "prepare_grouped_det_gemm_fwd"),
        ("prepare_grouped_gemm_dgrad", "prepare_grouped_det_gemm_dgrad"),
        ("prepare_grouped_gemm_wgrad", "prepare_grouped_det_gemm_wgrad"),
    ):
        assert getattr(legacy, old) is getattr(det_pkg, det), old
        assert getattr(legacy, old) is not getattr(pkg, old), old
    from tests.test_helpers.cake_package import cake_package

    assert cake_package("cake_moe_grouped_gemm") is det_pkg
    assert cake_package("cake.cake_moe_grouped_gemm") is pkg
    assert pkg.__name__ == "flashinfer.experimental.cake.cake_moe_grouped_gemm"


# ---------------------------------------------------------------------------
# The autograd backward's alert (GPU)
# ---------------------------------------------------------------------------


def _require_device():
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    if cb.arch_for(device) is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    for op in ("fwd", "dgrad", "wgrad"):
        if not cb.generated_program_available(device, op):
            pytest.skip(f"no generated program for {op} on {cb.arch_for(device)}")
    return device


def test_prepared_wgrad_launch_alerts_when_deterministic_mode_is_switched_on():
    """A weight gradient prepared with torch's deterministic mode off alerts at its
    launch once the mode is on: a ``RuntimeError`` (nothing launched), or with
    ``warn_only=True`` one ``UserWarning`` naming the caller's line, after which it
    launches and gives the bits of a launch with the mode off (the 256-wide k tile of
    this shape is deterministic).  A forward launch does not alert."""
    device = _require_device()
    sizes = [300, 0, 1, 700]
    offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    g = torch.randn(sum(sizes), 512, device=device).to(torch.bfloat16)
    x = (torch.randn(sum(sizes), 256, device=device) / 16).to(torch.bfloat16)
    w = (torch.randn(4, 512, 256, device=device) / 16).to(torch.bfloat16)
    launch = cb.prepare_grouped_gemm_wgrad(g, x, offs, deterministic=False)
    fwd = cb.prepare_grouped_gemm_fwd(x, w, offs, deterministic=False)
    ref = launch.launch().clone()
    launch.out.fill_(float("nan"))
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match="GroupedGemmLaunch.launch"),
    ):
        launch.launch()
    torch.cuda.synchronize()
    assert torch.isnan(launch.out.float()).all()  # the alert came before the launch
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        out = launch.launch()
        fwd.launch()
    torch.cuda.synchronize()
    alerts = [w for w in rec if "GroupedGemmLaunch.launch" in str(w.message)]
    assert len(alerts) == 1 and alerts[0].category is UserWarning, [
        str(w.message) for w in rec
    ]
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    assert torch.equal(out, ref)


def test_autograd_backward_alerts_and_facade_falls_back():
    """Under torch's deterministic mode the package's ``CakeGroupedMm`` backward alerts
    when it computes ``dW`` (``RuntimeError``; a warning with ``warn_only``), and not when
    only ``dX`` is needed; the facade with ``deterministic=False`` runs the deterministic
    wrapper instead and gives its bits."""
    device = _require_device()
    sizes = [300, 0, 1, 700]
    offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    x0 = torch.randn(sum(sizes), 256, device=device).to(torch.bfloat16)
    w0 = (torch.randn(4, 512, 256, device=device) / 16).to(torch.bfloat16)
    g = torch.randn(sum(sizes), 512, device=device).to(torch.bfloat16)

    def run(fn, needs_w=True):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(needs_w)
        fn(x, w).backward(g)
        torch.cuda.synchronize()
        return x.grad, w.grad

    package = lambda x, w: pkg.cake_grouped_mm(x, w, offs, deterministic=False)  # noqa: E731
    with _TorchDeterministic(warn_only=False):
        with pytest.raises(RuntimeError, match="CakeGroupedMm.backward"):
            run(package)
        dx, dw = run(package, needs_w=False)
        assert dx is not None and dw is None
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        run(package)
    assert any("CakeGroupedMm.backward" in str(w.message) for w in rec)
    det = run(lambda x, w: det_pkg.cake_grouped_det_mm(x, w, offs))
    with _TorchDeterministic(warn_only=False), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback = run(lambda x, w: facade.cake_grouped_mm(x, w, offs, False))
    assert torch.equal(fallback[0], det[0]) and torch.equal(fallback[1], det[1])


@pytest.mark.parametrize(
    "deterministic", [False, True], ids=["nondeterministic", "deterministic"]
)
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_facade_checkpoint_early_stop_skips_the_forward_launch(
    early_stop, deterministic, monkeypatch
):
    """Through the facade, with ``deterministic=False`` (this package) or ``True`` (the
    deterministic package), the package's wrapper launches its forward after autograd
    saved the inputs: a non-reentrant checkpoint of one GEMM launches the forward once
    with early stop on (the recomputation stops before it) and twice with it off, with
    the bits of the plain call for ``Y`` and ``dX``."""
    device = _require_device()
    sizes = [300, 0, 1, 700]
    offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    x0 = torch.randn(sum(sizes), 256, device=device).to(torch.bfloat16)
    w0 = (torch.randn(4, 512, 256, device=device) / 16).to(torch.bfloat16)
    g = torch.randn(sum(sizes), 512, device=device).to(torch.bfloat16)
    launches = []
    module = det_autograd if deterministic else cake_autograd
    fwd_op = module._fwd_op

    def counted(*args):
        launches.append(args[3].shape)
        fwd_op(*args)

    monkeypatch.setattr(module, "_fwd_op", counted)

    def gemm(x, w):
        return facade.cake_grouped_mm(x, w, offs, deterministic)

    def run(checkpointed):
        launches.clear()
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            if checkpointed:
                y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
            else:
                y = gemm(x, w)
            y.backward(g)
        torch.cuda.synchronize()
        return y.detach(), x.grad, len(launches)

    y0, dx0, n0 = run(False)
    y1, dx1, n1 = run(True)
    assert n0 == 1
    assert n1 == (1 if early_stop else 2)
    assert torch.equal(y1, y0) and torch.equal(dx1, dx0)
