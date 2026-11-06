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

flashinfer public-API tests of the non-deterministic Cake MoE grouped GEMM.

The facade ``flashinfer.cake_moe_grouped_gemm_nondet_train`` (``grouped_gemm_fwd``,
``grouped_gemm_dgrad``, ``grouped_gemm_wgrad``, ``cake_grouped_mm``) and the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`` it hands off to are
non-deterministic only: the keyword-only ``deterministic`` must be the bool ``False``
(every other value, the default ``True`` included, raises ``ValueError`` naming the
deterministic entry of ``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``), nothing
routes to the deterministic package, and under ``torch.use_deterministic_algorithms(True)``
PyTorch's non-determinism alert is raised immediately before an order-varying launch
only (the weight gradient whose 512-wide k tile splits its tail).  This suite covers on
CPU the experimental-API marking and signatures of the facade, its handoff (with the
package functions replaced by recorders and the deterministic ones by tripwires), the
mode refusal of the facade and of the package, its deferred imports, the alert contract
of prepared launches (with recorded launches standing in for the programs: an
order-varying one alerts, a fixed-order one never does; the alert names the caller's
frame under ``warn_only``), and that the det-free names exist only as non-deterministic
objects; on a GPU, the alert of prepared launches and of the autograd backward on an
order-varying and on a fixed-order shape.
"""

import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

import flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train as pkg
import flashinfer.cake_moe_grouped_gemm_nondet_train as facade
import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as det_pkg
import flashinfer.grouped_mm as grouped_mm
from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
    cake_autograd,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
    cake_backend as cb,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
    cake_autograd as det_autograd,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = (
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "cake_grouped_mm",
)
# facade entry -> the deterministic entry its mode refusal names
DET_ENTRY = {
    "grouped_gemm_fwd": "cake_moe_grouped_gemm_det_train_forward",
    "grouped_gemm_dgrad": "cake_moe_grouped_gemm_det_train_dgrad",
    "grouped_gemm_wgrad": "cake_moe_grouped_gemm_det_train_wgrad",
    "cake_grouped_mm": "cake_moe_grouped_gemm_det",
}
# The deterministic objects nothing here may call (module, attribute).
DET_TARGETS = (
    (det_pkg, "cake_moe_grouped_gemm_det_train_forward"),
    (det_pkg, "cake_moe_grouped_gemm_det_train_dgrad"),
    (det_pkg, "cake_moe_grouped_gemm_det_train_wgrad"),
    (det_autograd, "cake_moe_grouped_gemm_det"),
)
NEW_TARGETS = {
    "grouped_gemm_fwd": (pkg, "grouped_gemm_fwd"),
    "grouped_gemm_dgrad": (pkg, "grouped_gemm_dgrad"),
    "grouped_gemm_wgrad": (pkg, "grouped_gemm_wgrad"),
    "cake_grouped_mm": (cake_autograd, "cake_grouped_mm"),
}
MODE_HINT = "pass deterministic=False for the non-deterministic path"
# Every value of ``deterministic`` but the bool False (``_OMIT``: the default True).
_OMIT = object()
NOT_FALSE = (_OMIT, True, None, 0, 1, "False")
NOT_FALSE_IDS = ("omitted", "True", "None", "0", "1", "str")


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


def _user_warnings(records):
    return [w for w in records if w.category is UserWarning]


def _tripwires(monkeypatch):
    """Replace every deterministic object the facade used to route to by a function that
    fails the test when called."""

    for module, attr in DET_TARGETS:

        def tripwire(*args, _attr=attr, **kwargs):
            raise AssertionError(
                f"{_attr} called: nothing may route to the det package"
            )

        monkeypatch.setattr(module, attr, tripwire)


# ---------------------------------------------------------------------------
# The facade (CPU)
# ---------------------------------------------------------------------------


def test_facade_entry_points_are_the_experimental_api():
    """Each facade function is marked experimental with the feature
    ``cake_moe_grouped_gemm.<name>``, takes the signature of the package function of
    the same name (the keyword-only ``deterministic: bool = True`` included), documents
    the ``deterministic`` keyword behind the experimental banner, and is in ``__all__``;
    the package functions are unmarked."""
    assert sorted(facade.__all__) == sorted(ENTRY_POINTS)
    for name in ENTRY_POINTS:
        entry, impl = getattr(facade, name), _package_function(name)
        assert entry.is_experimental is True, name
        assert entry.experimental_feature == f"cake_moe_grouped_gemm.{name}", name
        assert inspect.signature(entry) == inspect.signature(impl), name
        parameter = inspect.signature(entry).parameters["deterministic"]
        assert parameter.default is True, name
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert entry.__doc__.startswith(".. warning::"), name
        assert "deterministic" in entry.__doc__, name
        assert DET_ENTRY[name] in entry.__doc__, name
        assert not hasattr(impl, "is_experimental"), name


def _recorders(monkeypatch):
    """Replace the package functions the facade hands off to by recorders and the
    deterministic objects by tripwires; return the log of ``(entry, deterministic)``."""
    log = []
    for entry, (module, attr) in NEW_TARGETS.items():

        def record(*args, _entry=entry, **kwargs):
            log.append((_entry, kwargs.get("deterministic", "absent")))
            return "result"

        monkeypatch.setattr(module, attr, record)
    _tripwires(monkeypatch)
    return log


def _operands(name):
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    w = torch.zeros(2, 256, 64, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)
    return (x, w, offs) if name != "grouped_gemm_wgrad" else (x, x, offs)


def _call(name, deterministic=_OMIT):
    args = _operands(name)
    if deterministic is _OMIT:
        return getattr(facade, name)(*args)
    return getattr(facade, name)(*args, deterministic=deterministic)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_facade_hands_every_call_to_the_package(name, monkeypatch):
    """``deterministic=False`` reaches the package function of the same name with
    ``deterministic=False``; under torch's deterministic mode (strict and
    ``warn_only``) the facade neither reroutes to the deterministic package nor warns.
    The deterministic objects are never called."""
    log = _recorders(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert _call(name, False) == "result"
        for warn_only in (False, True):
            with _TorchDeterministic(warn_only):
                for _ in range(2):
                    assert _call(name, False) == "result"
    assert _user_warnings(rec) == []
    assert log == [(name, False)] * 5


@pytest.mark.parametrize("name", ENTRY_POINTS)
@pytest.mark.parametrize("value", NOT_FALSE, ids=NOT_FALSE_IDS)
def test_facade_rejects_every_mode_but_false(name, value, monkeypatch):
    """Every value of ``deterministic`` but the bool ``False`` (the default ``True``,
    ``None``, ``0``, ``1``, ``"False"``) raises ``ValueError`` naming the deterministic
    entry and saying to pass ``deterministic=False``, also under torch's deterministic
    mode (strict and ``warn_only``); nothing reaches the deterministic package."""
    _tripwires(monkeypatch)
    for mode in (
        contextlib.nullcontext(),
        _TorchDeterministic(warn_only=False),
        _TorchDeterministic(warn_only=True),
    ):
        with mode, pytest.raises(ValueError) as info:
            _call(name, value)
        message = str(info.value)
        assert DET_ENTRY[name] in message, message
        assert MODE_HINT in message, message
        assert "cake_moe_grouped_gemm_det_train" in message, message


def test_facade_has_no_deterministic_route():
    """The facade holds no routing state and imports nothing from the deterministic
    package (no fallback warning flags, no mode helper)."""
    for name in ("_FALLBACK_WARNED", "_deterministic", "_caller_stacklevel"):
        assert not hasattr(facade, name), name
    source = Path(facade.__file__).read_text()
    assert "experimental.cake.cake_moe_grouped_gemm_det_train import" not in source
    assert "NotImplementedError" not in source


def test_facade_positional_mode_is_refused(monkeypatch):
    """``deterministic`` is keyword-only on every facade entry, ``cake_grouped_mm``
    included: a fourth positional ``False`` of ``cake_grouped_mm`` raises
    ``TypeError``."""
    log = _recorders(monkeypatch)
    with pytest.raises(TypeError):
        facade.cake_grouped_mm(*_operands("cake_grouped_mm"), False)
    assert log == []


_WARN_ONCE = """
import warnings

import torch

import flashinfer.api_logging as api_logging
import flashinfer.cake_moe_grouped_gemm_nondet_train as facade

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


# CPU operands fail the device check of the package (deterministic=False) and the
# default mode fails the mode check: every call raises a host-side ValueError and
# touches no kernel.
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
    [w] = warnings_of(entry, args, deterministic=False)  # first call: one warning
    assert w.category is api_logging.ExperimentalWarning, (name, w.category)
    message = str(w.message)
    assert f"'cake_moe_grouped_gemm.{name}' is an experimental" in message, message
    assert warnings_of(entry, args, deterministic=False) == [], name  # then silent
    assert warnings_of(entry, args) == [], name
print("OK")
"""


def test_facade_warns_once_per_entry_point():
    """Each facade function emits ``ExperimentalWarning`` once, on its first call; runs
    in a subprocess so the once-only flags do not depend on test order."""
    assert _run_cpu_subprocess(_WARN_ONCE) == "OK"


def test_importing_the_facade_imports_neither_package():
    """The facade reaches the package through deferred imports only."""
    code = (
        "import json, sys\n"
        "import flashinfer.cake_moe_grouped_gemm_nondet_train\n"
        "import flashinfer.grouped_mm\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train')"
        " or m.startswith('flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train'))))\n"
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
    "CakeGroupedMm": lambda **kw: cake_autograd.CakeGroupedMm.apply(
        _bf16(4, 64), _bf16(2, 256, 64), OFFS, *kw.values()
    ),
}


@pytest.mark.parametrize("name", list(PACKAGE_CALLS))
def test_package_refuses_every_mode_but_false(name):
    """Every entry point of the package raises ``ValueError`` naming its deterministic
    entry (``grouped_det_gemm_*``, ``prepare_grouped_det_gemm_*``,
    ``cake_moe_grouped_gemm_det``, ``CakeMoEGroupedGemmDetFunction``) and saying to pass
    ``deterministic=False`` unless ``deterministic`` is the bool ``False``; with
    ``False`` it reaches its host checks (CPU operands: the device check)."""
    call = PACKAGE_CALLS[name]
    for value in NOT_FALSE:
        kwargs = {} if value is _OMIT else {"deterministic": value}
        with pytest.raises(ValueError) as info:
            call(**kwargs)
        message = str(info.value)
        assert f"cake_moe_grouped_gemm_det_train.{cb._DET_ENTRIES[name]}" in message, (
            name,
            value,
            message,
        )
        assert MODE_HINT in message, (name, value)
    with pytest.raises(ValueError) as info:
        call(deterministic=False)
    assert "cake_moe_grouped_gemm_det_train" not in str(info.value), name


def test_package_mode_keyword_only():
    """``deterministic`` is keyword-only on every package function, the autograd
    wrapper included (a positional fourth argument raises ``TypeError``); only the
    ``torch.autograd.Function`` takes it positionally."""
    for fn in (
        pkg.grouped_gemm_fwd,
        pkg.grouped_gemm_dgrad,
        pkg.grouped_gemm_wgrad,
        pkg.cake_grouped_mm,
        cb.prepare_grouped_gemm_fwd,
        cb.prepare_grouped_gemm_dgrad,
        cb.prepare_grouped_gemm_wgrad,
    ):
        parameter = inspect.signature(fn).parameters["deterministic"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, fn
        assert parameter.default is True, fn
    with pytest.raises(TypeError):
        pkg.cake_grouped_mm(_bf16(4, 64), _bf16(2, 256, 64), OFFS, False)
    assert set(cb._DET_ENTRIES) == set(PACKAGE_CALLS)


@pytest.mark.parametrize(
    "entry,det",
    [
        ("CakeGroupedMm.backward", "CakeMoEGroupedGemmDetFunction"),
        ("GroupedGemmLaunch.launch", "CakeMoEGroupedGemmDetLaunch.launch"),
        ("grouped_gemm_wgrad", "cake_moe_grouped_gemm_det_train_wgrad"),
    ],
)
def test_alert_names_the_deterministic_entry(entry, det):
    """The alert of every entry that can launch an order-varying weight gradient names the
    entry and its deterministic counterpart: the ``RuntimeError`` (strict) and the
    ``UserWarning`` (``warn_only``); nothing with the mode off."""
    want = f"flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train.{det}"
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError) as info,
    ):
        cb._alert_not_deterministic(entry)
    assert str(info.value).startswith(entry) and want in str(info.value)
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        cb._alert_not_deterministic(entry)
    alerts = _user_warnings(rec)
    assert len(alerts) == 1 and want in str(alerts[0].message), alerts
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cb._alert_not_deterministic(entry)
    assert rec == []


@pytest.mark.parametrize("value", [True, False, 0, 1, "junk"], ids=str)
def test_autograd_function_refuses_a_stray_internal_slot(value):
    """``CakeGroupedMm.apply`` takes the mode fourth and nothing but ``None`` fifth (an
    internal slot; the fuse flag is sixth): ``apply(x, w, offs, False, True)`` -- the
    deterministic package's ``apply(x, w, offs, None, True)`` with the mode put first --
    raises ``TypeError`` before anything runs; ``apply(x, w, offs, False, None, True)``
    reaches the host checks (CPU operands: the device check)."""
    x, w = _bf16(4, 64), _bf16(2, 256, 64)
    with pytest.raises(TypeError, match="takes no fifth argument"):
        cake_autograd.CakeGroupedMm.apply(x, w, OFFS, False, value)
    with pytest.raises(ValueError, match="device"):
        cake_autograd.CakeGroupedMm.apply(x, w, OFFS, False, None, True)


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capturing"])
def test_backward_never_queries_a_captured_stream(capturing, monkeypatch):
    """The backward binds the CUDA context of its thread with one runtime call: the
    capture-status query, then a stream query only outside a CUDA graph capture (querying
    a stream that is being captured ends the capture with an error), with and without the
    fuse flag (CPU stand-ins for the programs)."""
    ran = []
    for name in ("_fwd_op", "_dgrad_op", "_wgrad_op", "_wgrad_acc_op"):
        monkeypatch.setattr(
            cake_autograd, name, lambda *args, _name=name: ran.append(_name)
        )
    queries = []
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: types.SimpleNamespace(
            query=lambda: queries.append(device) or True
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    x = _bf16(4, 64).requires_grad_(True)
    w = _bf16(2, 256, 64).requires_grad_(True)
    w.main_grad = torch.zeros(2, 256, 64)
    for fuse in (False, True):
        y = cake_autograd.CakeGroupedMm.apply(x, w, OFFS, False, None, fuse)
        y.backward(torch.ones_like(y))
    assert len(queries) == (0 if capturing else 2), queries
    want = ["_fwd_op", "_dgrad_op", "_wgrad_op"]
    want += ["_fwd_op", "_dgrad_op", "_wgrad_acc_op"]
    assert ran == want, ran


@pytest.mark.parametrize("name", [n for n in PACKAGE_CALLS if n != "CakeGroupedMm"])
def test_entry_points_and_preparation_do_not_alert(name):
    """Under torch's deterministic mode (strict and ``warn_only``) no entry point alerts
    before it launches anything: the preparation and the host checks run as with the
    mode off (CPU operands: the device check's ``ValueError``), with no warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as info:
            PACKAGE_CALLS[name](deterministic=False)
    want = str(info.value)
    assert _user_warnings(rec) == []
    for warn_only in (False, True):
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as info:
                PACKAGE_CALLS[name](deterministic=False)
        assert str(info.value) == want, (warn_only, str(info.value))
        assert _user_warnings(rec) == [], [str(w.message) for w in rec]


def _recorded_launch(monkeypatch, *, op="wgrad", plan=None, programs=1):
    """A ``GroupedGemmLaunch`` whose programs are recorders (``programs`` of them; the
    log counts their calls), with ``tvm_ffi``'s stream scope replaced by a no-op so it
    runs on the CPU."""
    monkeypatch.setattr(
        cb, "tvm_ffi", types.SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )
    log = []
    launches = tuple(
        cb._Launch(entry=lambda *args: log.append("launch"), arguments=())
        for _ in range(programs)
    )
    launch = cb.GroupedGemmLaunch(
        op=op,
        record_name="cake_moe_grouped_gemm_wgrad_bf16_k512",
        stages=("main",) * programs,
        plan=dict(plan or {}),
        out=torch.zeros(1),
        launches=launches,
    )
    return launch, log


SPLIT_TAIL = {"unordered_tail": True, "tail_splits": 3}


def test_order_varying_launch_alerts_immediately_before_it_runs(monkeypatch):
    """An order-varying launch (the weight gradient with an unordered split tail) checks
    torch's deterministic mode at every ``launch()``: strict, a ``RuntimeError`` naming
    ``GroupedGemmLaunch.launch`` and the deterministic entry
    (``CakeMoEGroupedGemmDetLaunch.launch``), and nothing launches; ``warn_only``, one
    ``UserWarning`` naming both and the caller's line, then it launches; mode off, it
    launches silently."""
    launch, log = _recorded_launch(monkeypatch, plan=SPLIT_TAIL)
    assert launch.order_varying
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match="GroupedGemmLaunch.launch") as info,
    ):
        launch.launch()
    assert "cake_moe_grouped_gemm_det_train.CakeMoEGroupedGemmDetLaunch.launch" in str(
        info.value
    )
    assert log == []
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        launch.launch()
    alerts = _user_warnings(rec)
    assert len(alerts) == 1, [str(w.message) for w in rec]
    assert "GroupedGemmLaunch.launch" in str(alerts[0].message)
    assert "cake_moe_grouped_gemm_det_train.CakeMoEGroupedGemmDetLaunch.launch" in str(
        alerts[0].message
    )
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    assert log == ["launch"]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        launch()
    assert rec == [] and log == ["launch"] * 2


@pytest.mark.parametrize(
    "op,plan,programs",
    [
        ("wgrad", {"unordered_tail": True, "tail_splits": 1}, 1),  # unsplit tail
        ("wgrad", {"unordered_tail": False, "tail_splits": 4}, 2),  # ordered reduce
        ("wgrad", SPLIT_TAIL, 0),  # sum_m == 0: nothing to launch
        ("fwd", SPLIT_TAIL, 1),
        ("dgrad", {}, 1),
    ],
    ids=["unsplit", "ordered_tail", "no_rows", "fwd", "dgrad"],
)
def test_fixed_order_launches_never_alert(op, plan, programs, monkeypatch):
    """A launch with a fixed order (the forward, the data gradient, the 256-wide k
    tile's ordered tail reduce, an unsplit tail, no rows) is not ``order_varying`` and
    runs under torch's strict deterministic mode without an alert."""
    launch, log = _recorded_launch(monkeypatch, op=op, plan=plan, programs=programs)
    assert not launch.order_varying
    with (
        _TorchDeterministic(warn_only=False),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        launch.launch()
    assert _user_warnings(rec) == []
    assert log == ["launch"] * programs


@pytest.mark.parametrize("via", ["package", "facade"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_weight_gradient_alerts_with_the_entry_name(via, accumulate, monkeypatch):
    """``grouped_gemm_wgrad`` (the package's, and the facade's handoff to it) raises the
    alert with its own name immediately before an order-varying launch (overwrite or
    accumulate): strict, a ``RuntimeError`` and nothing launches; ``warn_only``, a
    ``UserWarning`` naming the caller's line, then it launches; the deterministic
    package is never called."""
    _tripwires(monkeypatch)
    launch, log = _recorded_launch(monkeypatch, plan=SPLIT_TAIL)
    monkeypatch.setattr(cb, "_prepare_wgrad", lambda *args, **kwargs: launch)
    fn = pkg.grouped_gemm_wgrad if via == "package" else facade.grouped_gemm_wgrad
    g, x = _bf16(4, 256), _bf16(4, 64)
    out = torch.zeros(2, 256, 64)

    def call():
        return fn(g, x, OFFS, out=out, deterministic=False, accumulate=accumulate)

    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match="grouped_gemm_wgrad"),
    ):
        call()
    assert log == []
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        call()
    alerts = _user_warnings(rec)
    assert len(alerts) == 1 and "grouped_gemm_wgrad" in str(alerts[0].message)
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    assert log == ["launch"]


def test_package_has_no_stable_api_handoff():
    """``grouped_mm_bf16(backend="cake")`` stays on the deterministic package: the new
    package has no handoff function for it."""
    for name in ("grouped_mm_bf16_cake", "grouped_det_mm_bf16_cake"):
        assert not hasattr(cb, name), name
    source = Path(grouped_mm.core.__file__).read_text()
    assert (
        "experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import"
        in source
    )
    assert "experimental.cake.cake_moe_grouped_gemm_nondet_train" not in source


# ---------------------------------------------------------------------------
# Det-free names are non-deterministic only (CPU)
# ---------------------------------------------------------------------------


def test_det_free_names_are_only_the_non_deterministic_objects():
    """After the removal of the det-free aliases of the deterministic objects, the
    spellings ``grouped_gemm_*``, ``prepare_grouped_gemm_*``, ``GroupedGemmLaunch``,
    ``CakeGroupedMm`` and ``cake_grouped_mm`` exist only in the non-deterministic
    package and its facade: the old package path does not import,
    ``flashinfer.grouped_mm`` and the deterministic package do not have them, and the
    test-helper key ``cake_moe_grouped_gemm`` is gone."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashinfer.experimental.cake_moe_grouped_gemm")
    for name in ("grouped_gemm_fwd", "grouped_gemm_dgrad", "grouped_gemm_wgrad"):
        assert not hasattr(grouped_mm, name), name
        assert not hasattr(det_pkg, name), name
        det_name = "cake_moe_grouped_gemm_det_train_" + name[13:].replace(
            "fwd", "forward"
        )
        assert getattr(facade, name) is not getattr(det_pkg, det_name)
    assert not hasattr(grouped_mm, "cake_grouped_mm")
    for name in (
        "CakeGroupedMm",
        "GroupedGemmLaunch",
        "cake_grouped_mm",
        "prepare_grouped_gemm_fwd",
        "prepare_grouped_gemm_dgrad",
        "prepare_grouped_gemm_wgrad",
    ):
        assert not hasattr(det_pkg, name), name
        assert getattr(pkg, name).__module__.startswith(pkg.__name__), name
    from tests.test_helpers.cake_package import PACKAGES, cake_package

    assert "cake_moe_grouped_gemm" not in PACKAGES
    assert cake_package("cake_moe_grouped_gemm_nondet_train") is pkg
    assert (
        pkg.__name__
        == "flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train"
    )


# ---------------------------------------------------------------------------
# The alert of prepared launches and of the autograd backward (GPU)
# ---------------------------------------------------------------------------


def _base():
    """The package's backend suite (its shapes and helpers), imported on first use."""
    from tests.experimental import test_cake_moe_grouped_gemm_nondet_train as base

    return base


def test_prepared_wgrad_launch_alerts_only_when_order_varying():
    """A weight gradient prepared with torch's deterministic mode off alerts at its
    launch once the mode is on when the launch is order-varying (the 512-wide k tile
    with a split tail): strict, a ``RuntimeError`` and nothing launched (``out`` keeps
    its NaN fill); ``warn_only``, one ``UserWarning`` naming the caller's line, then the
    launch, within the budget of the fp64 reference.  The 256-wide k tile (ordered tail
    reduce) and a forward launch never alert, and give the bits of a launch with the
    mode off."""
    base = _base()
    device = base._require_autograd_programs()
    sizes, (n, k) = base._tile_case(device, "k512")
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=60)
    launch = cb.prepare_grouped_gemm_wgrad(g, x, offs, deterministic=False)
    assert launch.order_varying, launch.plan
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
    torch.cuda.synchronize()
    alerts = [w for w in rec if "GroupedGemmLaunch.launch" in str(w.message)]
    assert len(alerts) == 1 and alerts[0].category is UserWarning, [
        str(w.message) for w in rec
    ]
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    base._assert_wgrad_close(out, base._reference("wgrad", g, x, sizes))

    sizes = base.RAGGED_E4
    x, w, g, offs = base._inputs(sizes, 512, 256, device, seed=61)
    ordered = cb.prepare_grouped_gemm_wgrad(g, x, offs, deterministic=False)
    assert not ordered.order_varying, ordered.plan
    fwd = cb.prepare_grouped_gemm_fwd(x, w, offs, deterministic=False)
    ref = ordered.launch().clone()
    y_ref = fwd.launch().clone()
    ordered.out.fill_(float("nan"))
    with (
        _TorchDeterministic(warn_only=False),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        out = ordered.launch()
        y = fwd.launch()
    torch.cuda.synchronize()
    assert _user_warnings(rec) == []
    assert torch.equal(out, ref) and torch.equal(y, y_ref)


def test_autograd_backward_alerts_only_before_an_order_varying_weight_gradient():
    """Under torch's deterministic mode the package's ``CakeGroupedMm`` backward (through
    the package and through the facade with ``deterministic=False``; no fallback)
    alerts immediately before an order-varying ``dW`` launch (``RuntimeError``; a
    ``UserWarning`` with ``warn_only``), not when only ``dX`` is needed, and never on a
    fixed-order shape, whose gradients have the bits of a run with the mode off."""
    base = _base()
    device = base._require_autograd_programs()

    def run(fn, x0, w0, g, needs_w=True):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(needs_w)
        fn(x, w).backward(g)
        torch.cuda.synchronize()
        return x.grad, w.grad

    sizes, (n, k) = base._tile_case(device, "k512")
    x0, w0, g, offs = base._inputs(sizes, n, k, device, seed=62)
    package = lambda x, w: pkg.cake_grouped_mm(x, w, offs, deterministic=False)  # noqa: E731
    through_facade = lambda x, w: facade.cake_grouped_mm(  # noqa: E731
        x, w, offs, deterministic=False
    )
    with _TorchDeterministic(warn_only=False):
        for fn in (package, through_facade):
            with pytest.raises(RuntimeError, match="CakeGroupedMm.backward"):
                run(fn, x0, w0, g)
        dx, dw = run(package, x0, w0, g, needs_w=False)
        assert dx is not None and dw is None
    with (
        _TorchDeterministic(warn_only=True),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        dx, dw = run(package, x0, w0, g)
    alerts = [w for w in rec if "CakeGroupedMm.backward" in str(w.message)]
    assert alerts and all(w.category is UserWarning for w in alerts), [
        str(w.message) for w in rec
    ]
    base._assert_wgrad_close(dw, base._reference("wgrad", g, x0, sizes))

    sizes = base.RAGGED_E4
    x0, w0, g, offs = base._inputs(sizes, 512, 256, device, seed=63)
    want = run(package, x0, w0, g)
    with (
        _TorchDeterministic(warn_only=False),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        got = run(package, x0, w0, g)
    assert _user_warnings(rec) == []
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_facade_checkpoint_early_stop_skips_the_forward_launch(early_stop, monkeypatch):
    """Through the facade (``deterministic=False``) the package's wrapper launches its
    forward after autograd saved the inputs: a non-reentrant checkpoint of one GEMM
    launches the forward once with early stop on (the recomputation stops before it)
    and twice with it off, with the bits of the plain call for ``Y`` and ``dX``."""
    base = _base()
    device = base._require_autograd_programs()
    sizes = base.RAGGED_E4
    x0, w0, g, offs = base._inputs(sizes, 512, 256, device, seed=64)
    launches = []
    fwd_op = cake_autograd._fwd_op

    def counted(*args):
        launches.append(args[3].shape)
        fwd_op(*args)

    monkeypatch.setattr(cake_autograd, "_fwd_op", counted)

    def gemm(x, w):
        return facade.cake_grouped_mm(x, w, offs, deterministic=False)

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
