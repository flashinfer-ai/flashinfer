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

The module ``flashinfer.cake_moe_grouped_gemm_nondet_train`` and the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`` it hands off to
are the non-deterministic mode: no entry takes a ``deterministic`` argument (one
raises ``TypeError`` before anything runs) and nothing routes to the deterministic
module.  Every output of the shipped registry and constants repeats bit for bit (the
512-wide k tile's weight gradient finishes a split tail with the ordered one-launch
tail), so no launch alerts under ``torch.use_deterministic_algorithms(True)``.  The
non-determinism alert is dormant: it is raised immediately before an order-varying
launch only (a weight gradient planned with an unordered split tail, which only an
architecture listed in ``unordered_tail_archs`` -- empty in the shipped constants --
can produce).  This suite covers on CPU the stable-API status (no experimental
marking, no warning) and signatures of the module, the support query, its handoff
(with the package functions replaced by recorders and the deterministic ones by
tripwires), the refusal of a stale mode argument, its deferred imports, and the alert
contract of prepared launches (with recorded launches standing in for the programs:
an order-varying one alerts, a fixed-order one -- the ordered one-launch split tail
included -- never does; the alert names the caller's frame under ``warn_only``); on a
GPU, that a prepared weight gradient with a split ordered tail, the weight-gradient
entry and the autograd backward on that shape (and on a 256-wide k tile shape) never
alert under the strict and the ``warn_only`` mode and give the bits of a run with the
mode off.  The names of the module, the removed ones included, are covered by
``test_cake_moe_grouped_gemm_nondet_train_names.py``.
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

import flashinfer.cake_moe_grouped_gemm_nondet_train as facade
import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as det_pkg
import flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train as pkg
import flashinfer.grouped_mm as grouped_mm
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
    cake_autograd as det_autograd,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
    cake_backend as det_cb,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
    cake_autograd,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
    cake_backend as cb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "cake_moe_grouped_gemm_nondet_train"
ENTRY_POINTS = (
    f"{PREFIX}_forward",
    f"{PREFIX}_dgrad",
    f"{PREFIX}_wgrad",
    f"{PREFIX}_prepare_forward",
    f"{PREFIX}_prepare_dgrad",
    f"{PREFIX}_prepare_wgrad",
    "cake_moe_grouped_gemm_nondet",
)
LAZY_TYPES = {
    "CakeMoEGroupedGemmNondetFunction": "cake_autograd",
    "CakeMoEGroupedGemmNondetLaunch": "cake_backend",
}
# The deterministic objects nothing here may call (module, attribute).
DET_TARGETS = (
    (det_pkg, "cake_moe_grouped_gemm_det_train_forward"),
    (det_pkg, "cake_moe_grouped_gemm_det_train_dgrad"),
    (det_pkg, "cake_moe_grouped_gemm_det_train_wgrad"),
    (det_cb, "cake_moe_grouped_gemm_det_train_prepare_forward"),
    (det_cb, "cake_moe_grouped_gemm_det_train_prepare_dgrad"),
    (det_cb, "cake_moe_grouped_gemm_det_train_prepare_wgrad"),
    (det_autograd, "cake_moe_grouped_gemm_det"),
)
# facade entry -> the package object it hands off to (module, attribute)
NEW_TARGETS = {
    f"{PREFIX}_forward": (pkg, f"{PREFIX}_forward"),
    f"{PREFIX}_dgrad": (pkg, f"{PREFIX}_dgrad"),
    f"{PREFIX}_wgrad": (pkg, f"{PREFIX}_wgrad"),
    f"{PREFIX}_prepare_forward": (cb, f"{PREFIX}_prepare_forward"),
    f"{PREFIX}_prepare_dgrad": (cb, f"{PREFIX}_prepare_dgrad"),
    f"{PREFIX}_prepare_wgrad": (cb, f"{PREFIX}_prepare_wgrad"),
    "cake_moe_grouped_gemm_nondet": (cake_autograd, "cake_moe_grouped_gemm_nondet"),
}
OFFS = torch.tensor([2, 4], dtype=torch.int32)


def _package_function(name):
    module, attr = NEW_TARGETS[name]
    return getattr(module, attr)


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def _operands(name):
    """bfloat16 CPU operands of the entry ``name``: they pass its dtype and shape checks
    and fail its device check, so a call raises the host-side ``ValueError`` and
    touches no kernel."""
    if name.endswith("wgrad"):
        return _bf16(4, 256), _bf16(4, 64), OFFS
    if name.endswith("dgrad"):
        return _bf16(4, 64), _bf16(2, 64, 256), OFFS
    return _bf16(4, 64), _bf16(2, 256, 64), OFFS


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
    """Replace every deterministic entry by a function that fails the test when
    called."""

    for module, attr in DET_TARGETS:

        def tripwire(*args, _attr=attr, **kwargs):
            raise AssertionError(
                f"{_attr} called: nothing may route to the det package"
            )

        monkeypatch.setattr(module, attr, tripwire)


# ---------------------------------------------------------------------------
# The module (CPU)
# ---------------------------------------------------------------------------


def test_facade_entry_points_are_stable_handoffs():
    """``flashinfer.cake_moe_grouped_gemm_nondet_train.<name>`` is a stable
    FlashInfer API (``@flashinfer_api``; no experimental marking) with the signature
    (no ``deterministic`` parameter) and the docstring of the package function of the
    same name, and is listed in the module's ``__all__``.  The two types of the module
    are the package's classes, resolved on first access."""
    assert "flashinfer_experimental_api" not in Path(facade.__file__).read_text()
    for name in ENTRY_POINTS:
        entry, impl = getattr(facade, name), _package_function(name)
        assert name in facade.__all__, name
        assert getattr(pkg, name) is impl and entry is not impl, name
        assert not getattr(entry, "is_experimental", False), name
        assert not hasattr(entry, "experimental_feature"), name
        signature = inspect.signature(entry)
        assert signature == inspect.signature(impl), name
        assert "deterministic" not in signature.parameters, name
        assert entry.__doc__ == impl.__doc__, name
        assert not hasattr(impl, "is_experimental"), name
    for name, submodule in LAZY_TYPES.items():
        assert name in facade.__all__ and name in dir(facade), name
        module = importlib.import_module(f"{pkg.__name__}.{submodule}")
        assert getattr(facade, name) is getattr(module, name) is getattr(pkg, name)


def _recorders(monkeypatch):
    """Replace the package functions the module hands off to by recorders and the
    deterministic entries by tripwires; return the log of ``(entry, keywords)``."""
    log = []
    for entry, (module, attr) in NEW_TARGETS.items():

        def record(*args, _entry=entry, **kwargs):
            log.append((_entry, sorted(kwargs)))
            return "result"

        monkeypatch.setattr(module, attr, record)
    _tripwires(monkeypatch)
    return log


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_facade_hands_every_call_to_the_package(name, monkeypatch):
    """Every call reaches the package function of the same name, with the three
    operands positional and every other argument by keyword; under torch's
    deterministic mode (strict and ``warn_only``) the module neither reroutes to the
    deterministic module nor warns.  The deterministic entries are never called."""
    keywords = sorted(list(inspect.signature(getattr(facade, name)).parameters)[3:])
    log = _recorders(monkeypatch)
    entry = getattr(facade, name)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert entry(*_operands(name)) == "result"
        for warn_only in (False, True):
            with _TorchDeterministic(warn_only):
                for _ in range(2):
                    assert entry(*_operands(name)) == "result"
    assert _user_warnings(rec) == []
    assert log == [(name, keywords)] * 5


def test_facade_has_no_deterministic_route():
    """The module holds no routing state and imports nothing from the deterministic
    package (no fallback warning flags, no mode helper)."""
    for name in ("_FALLBACK_WARNED", "_deterministic", "_caller_stacklevel"):
        assert not hasattr(facade, name), name
    source = Path(facade.__file__).read_text()
    assert "experimental.cake.cake_moe_grouped_gemm_det_train" not in source
    assert "raise NotImplementedError" not in source


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("via", ["package", "facade"])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entries_take_no_deterministic_keyword(name, via, value, monkeypatch):
    """No entry of the module or of the package takes ``deterministic``: either value
    raises ``TypeError`` naming the keyword before anything runs (the recorders of the
    module's handoff and the deterministic tripwires stay untouched), also under
    torch's deterministic mode."""
    entry = getattr(facade, name) if via == "facade" else _package_function(name)
    log = _recorders(monkeypatch)
    for mode in (contextlib.nullcontext(), _TorchDeterministic(warn_only=False)):
        with mode, pytest.raises(TypeError, match="deterministic"):
            entry(*_operands(name), deterministic=value)
    assert log == []


@pytest.mark.parametrize("via", ["package", "facade"])
@pytest.mark.parametrize("value", [True, False])
def test_stale_positional_mode_is_a_type_error(via, value, monkeypatch):
    """The fuse flag of ``cake_moe_grouped_gemm_nondet`` is keyword-only: a stale
    fourth positional mode argument raises ``TypeError`` before anything runs."""
    entry = (
        facade.cake_moe_grouped_gemm_nondet
        if via == "facade"
        else pkg.cake_moe_grouped_gemm_nondet
    )
    log = _recorders(monkeypatch)
    with pytest.raises(TypeError):
        entry(*_operands("cake_moe_grouped_gemm_nondet"), value)
    assert log == []


_NO_WARNING = """
import warnings

import torch

import flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train as pkg
import flashinfer.cake_moe_grouped_gemm_nondet_train as facade

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


# bfloat16 CPU operands pass the dtype and shape checks and fail the device
# check, so every call raises the host-side ValueError and touches no kernel.
offs = torch.tensor([2, 4], dtype=torch.int32)
prefix = "cake_moe_grouped_gemm_nondet_train"
operands = {
    f"{prefix}_forward": (bf16(4, 64), bf16(2, 256, 64), offs),
    f"{prefix}_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    f"{prefix}_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    f"{prefix}_prepare_forward": (bf16(4, 64), bf16(2, 256, 64), offs),
    f"{prefix}_prepare_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    f"{prefix}_prepare_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_moe_grouped_gemm_nondet": (bf16(4, 64), bf16(2, 256, 64), offs),
}


def warnings_of(fn, args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("CPU operands must be rejected on the host")
    return rec


for name in pkg.__all__:  # load the submodules outside the recorded calls
    getattr(pkg, name)

for name, args in operands.items():
    entry, impl = getattr(facade, name), getattr(pkg, name)
    for fn in (entry, entry, impl):  # the first call, a second call, the package
        rec = warnings_of(fn, args)
        assert rec == [], (name, [f"{w.category.__name__}: {w.message}" for w in rec])
print("OK")
"""


def test_facade_entry_points_do_not_warn():
    """Each entry point hands CPU operands to the package function, which raises the
    host-side ``ValueError``, with no warning: no ``ExperimentalWarning`` on the
    first call or later, and none from the package function.  Runs in a fresh
    interpreter, so each call there is the first of its entry point."""
    assert _run_cpu_subprocess(_NO_WARNING) == "OK"


def test_importing_the_facade_imports_neither_package():
    """The module reaches the package through deferred imports only."""
    code = (
        "import json, sys\n"
        "import flashinfer.cake_moe_grouped_gemm_nondet_train\n"
        "import flashinfer.grouped_mm\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train')"
        " or m.startswith("
        "'flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train'))))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


_SUPPORT_OFF_CUDA = """
import json
import sys

import torch

import flashinfer.cake_moe_grouped_gemm_nondet_train as facade

assert not torch.cuda.is_available()
query = facade.is_cake_moe_grouped_gemm_nondet_train_supported
devices = (None, "cuda", torch.device("cuda", 0), "cpu")
answers = [query()] + [query(device) for device in devices]
loaded = sorted(m for m in sys.modules if m.startswith(facade._PACKAGE))
print(json.dumps([answers, loaded]))
"""


def test_support_query_is_false_off_cuda():
    """``is_cake_moe_grouped_gemm_nondet_train_supported`` is a stable FlashInfer API
    (``@flashinfer_api``; not marked experimental) that is ``False`` without CUDA
    (the default device, an explicit CUDA device, a CPU device) and then imports
    none of the package; a device that is not CUDA is ``False`` on any host."""
    answers, loaded = json.loads(_run_cpu_subprocess(_SUPPORT_OFF_CUDA))
    assert answers == [False] * 5 and loaded == []
    query = facade.is_cake_moe_grouped_gemm_nondet_train_supported
    for device in ("cpu", torch.device("cpu"), "meta"):
        assert query(device) is False, device
    assert not getattr(query, "is_experimental", False)
    assert "is_cake_moe_grouped_gemm_nondet_train_supported" in facade.__all__
    assert list(inspect.signature(query).parameters) == ["device"]
    assert inspect.signature(query).parameters["device"].default is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_support_query_on_this_device():
    """On a CUDA device the support query is the backend's program query of every
    operation (``cake_jit.OPS``), for the default device and an explicit index; an
    index past the visible devices is ``False``."""
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import cake_jit

    query = facade.is_cake_moe_grouped_gemm_nondet_train_supported
    device = torch.device("cuda", torch.cuda.current_device())
    want = all(cb.generated_program_available(device, op) for op in cake_jit.OPS)
    assert query() is want and query(device) is want
    assert query(torch.device("cuda", torch.cuda.device_count())) is False


# ---------------------------------------------------------------------------
# The package's deterministic-mode alert (CPU)
# ---------------------------------------------------------------------------


def _package_entry(name):
    """The package callable ``name``: an entry point, or the autograd function's
    ``apply``."""
    if name == "CakeMoEGroupedGemmNondetFunction":
        return cake_autograd.CakeMoEGroupedGemmNondetFunction.apply
    return _package_function(name)


@pytest.mark.parametrize(
    "entry,det",
    [
        ("CakeMoEGroupedGemmNondetFunction.backward", "CakeMoEGroupedGemmDetFunction"),
        ("CakeMoEGroupedGemmNondetLaunch.launch", "CakeMoEGroupedGemmDetLaunch.launch"),
        (f"{PREFIX}_wgrad", "cake_moe_grouped_gemm_det_train_wgrad"),
    ],
)
def test_alert_names_the_deterministic_entry(entry, det):
    """The alert of every entry that can launch an order-varying weight gradient names
    the entry and its counterpart in ``flashinfer.cake_moe_grouped_gemm_det_train``:
    the ``RuntimeError`` (strict) and the ``UserWarning`` (``warn_only``); nothing
    with the mode off."""
    want = f"flashinfer.cake_moe_grouped_gemm_det_train.{det}"
    assert cb._ALERT_DET_ENTRIES[entry] == det
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
    """``CakeMoEGroupedGemmNondetFunction.apply`` takes nothing but ``None`` fourth (an
    internal slot; the fuse flag is fifth): ``apply(x, w, offs, value)`` -- a stale
    mode argument -- raises ``TypeError`` before anything runs;
    ``apply(x, w, offs, None, True)`` reaches the host checks (CPU operands: the
    device check)."""
    x, w = _bf16(4, 64), _bf16(2, 256, 64)
    with pytest.raises(TypeError, match="takes no fourth argument"):
        cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(x, w, OFFS, value)
    with pytest.raises(ValueError, match="device"):
        cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(x, w, OFFS, None, True)


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
        y = cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(x, w, OFFS, None, fuse)
        y.backward(torch.ones_like(y))
    assert len(queries) == (0 if capturing else 2), queries
    want = ["_fwd_op", "_dgrad_op", "_wgrad_op"]
    want += ["_fwd_op", "_dgrad_op", "_wgrad_acc_op"]
    assert ran == want, ran


@pytest.mark.parametrize("name", [*ENTRY_POINTS, "CakeMoEGroupedGemmNondetFunction"])
def test_entry_points_and_preparation_do_not_alert(name):
    """Under torch's deterministic mode (strict and ``warn_only``) no entry point alerts
    before it launches anything: the preparation and the host checks run as with the
    mode off (CPU operands: the device check's ``ValueError``), with no warning."""
    call = _package_entry(name)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as info:
            call(*_operands(name))
    want = str(info.value)
    assert _user_warnings(rec) == []
    for warn_only in (False, True):
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as info:
                call(*_operands(name))
        assert str(info.value) == want, (warn_only, str(info.value))
        assert _user_warnings(rec) == [], [str(w.message) for w in rec]


def _recorded_launch(monkeypatch, *, op="wgrad", plan=None, programs=1):
    """A ``CakeMoEGroupedGemmNondetLaunch`` whose programs are recorders (``programs`` of them; the
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
    launch = cb.CakeMoEGroupedGemmNondetLaunch(
        op=op,
        record_name="cake_moe_grouped_gemm_wgrad_bf16_k512",
        stages=("main",) * programs,
        plan=dict(plan or {}),
        out=torch.zeros(1),
        launches=launches,
    )
    return launch, log


# The plan of a weight gradient with an unordered split tail: order-varying.  Only an
# architecture listed in ``unordered_tail_archs`` (empty in the shipped constants)
# plans one.
SPLIT_TAIL = {"unordered_tail": True, "tail_splits": 3}
# The plan of a 512-wide k tile weight gradient whose split tail the ordered one-launch
# tail finishes (the shipped constants): a fixed order.
ORDERED_SPLIT_TAIL = {"unordered_tail": False, "ordered_tail": True, "tail_splits": 4}


def test_order_varying_launch_alerts_immediately_before_it_runs(monkeypatch):
    """An order-varying launch (a weight gradient with an unordered split tail, which no
    plan of the shipped constants is: the alert is dormant) checks torch's
    deterministic mode at every ``launch()``: strict, a ``RuntimeError`` naming
    ``CakeMoEGroupedGemmNondetLaunch.launch`` and the deterministic entry
    (``CakeMoEGroupedGemmDetLaunch.launch``), and nothing launches; ``warn_only``, one
    ``UserWarning`` naming both and the caller's line, then it launches; mode off, it
    launches silently."""
    launch, log = _recorded_launch(monkeypatch, plan=SPLIT_TAIL)
    assert launch.order_varying
    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(
            RuntimeError, match="CakeMoEGroupedGemmNondetLaunch.launch"
        ) as info,
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
    assert "CakeMoEGroupedGemmNondetLaunch.launch" in str(alerts[0].message)
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
        ("wgrad", {"unordered_tail": False, "tail_splits": 4}, 2),  # tail reduce
        ("wgrad", ORDERED_SPLIT_TAIL, 1),  # ordered one-launch split tail
        ("wgrad", SPLIT_TAIL, 0),  # sum_m == 0: nothing to launch
        ("fwd", SPLIT_TAIL, 1),
        ("dgrad", {}, 1),
    ],
    ids=["unsplit", "tail_reduce", "ordered_tail", "no_rows", "fwd", "dgrad"],
)
def test_fixed_order_launches_never_alert(op, plan, programs, monkeypatch):
    """A launch with a fixed order (the forward, the data gradient, the 256-wide k
    tile's two-launch ordered tail reduce, the 512-wide k tile's ordered one-launch
    split tail, an unsplit tail, no rows) is not ``order_varying`` and runs under
    torch's strict deterministic mode without an alert."""
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
    """``cake_moe_grouped_gemm_nondet_train_wgrad`` (the package's, and the module's
    handoff to it) raises the alert with its own name immediately before an
    order-varying launch (overwrite or accumulate): strict, a ``RuntimeError`` and
    nothing launches; ``warn_only``, a ``UserWarning`` naming the caller's line, then
    it launches; the deterministic module is never called.  A launch whose split tail
    the ordered one-launch tail finishes runs under the strict mode with no alert and
    no warning."""
    _tripwires(monkeypatch)
    launch, log = _recorded_launch(monkeypatch, plan=SPLIT_TAIL)
    monkeypatch.setattr(cb, "_prepare_wgrad", lambda *args, **kwargs: launch)
    fn = (
        pkg.cake_moe_grouped_gemm_nondet_train_wgrad
        if via == "package"
        else facade.cake_moe_grouped_gemm_nondet_train_wgrad
    )
    g, x = _bf16(4, 256), _bf16(4, 64)
    out = torch.zeros(2, 256, 64)

    def call():
        return fn(g, x, OFFS, out=out, accumulate=accumulate)

    with (
        _TorchDeterministic(warn_only=False),
        pytest.raises(RuntimeError, match="cake_moe_grouped_gemm_nondet_train_wgrad"),
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
    assert len(alerts) == 1 and "cake_moe_grouped_gemm_nondet_train_wgrad" in str(
        alerts[0].message
    )
    assert os.path.samefile(alerts[0].filename, __file__), alerts[0].filename
    assert log == ["launch"]
    ordered, log = _recorded_launch(monkeypatch, plan=ORDERED_SPLIT_TAIL)
    assert not ordered.order_varying
    monkeypatch.setattr(cb, "_prepare_wgrad", lambda *args, **kwargs: ordered)
    with (
        _TorchDeterministic(warn_only=False),
        warnings.catch_warnings(record=True) as rec,
    ):
        warnings.simplefilter("always")
        call()
    assert _user_warnings(rec) == [], [str(w.message) for w in rec]
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
# No alert from prepared launches, the weight-gradient entry and the autograd
# backward (GPU)
# ---------------------------------------------------------------------------


def _base():
    """The package's backend suite (its shapes and helpers), imported on first use."""
    from tests.experimental import test_cake_moe_grouped_gemm_nondet_train as base

    return base


def _assert_split_ordered_tail(launch):
    """``launch`` is a 512-wide k tile weight gradient whose tail is split and finished
    in the same launch by the ordered one-launch tail, so it is not
    ``order_varying``."""
    plan = launch.plan
    assert launch.op == "wgrad" and plan["tile_k"] == 512, plan
    assert plan["ordered_tail"] and not plan["unordered_tail"], plan
    assert plan["tail_splits"] > 1 and plan["launches"] == 1, plan
    assert not launch.order_varying, plan


def test_prepared_split_tail_wgrad_never_alerts():
    """A 512-wide k tile weight gradient whose tail is split (the ordered one-launch
    tail: ``plan["ordered_tail"]`` and ``plan["tail_splits"] > 1``, not
    ``order_varying``), prepared with torch's deterministic mode off, launches under the
    strict and the ``warn_only`` mode with no alert (no ``RuntimeError``, no
    ``UserWarning``) and gives the bits of its launch with the mode off (within the
    budget of the fp64 reference), ``out`` NaN-filled and the stream's tail slots
    garbage-filled before every launch; so does
    ``cake_moe_grouped_gemm_nondet_train_wgrad`` (the package's and the module's) on
    that shape.  The 256-wide k tile (two-launch ordered tail reduce) and a forward
    launch never alert either, and give the bits of a launch with the mode off."""
    base = _base()
    device = base._require_autograd_programs()
    sizes, (n, k) = base._tile_case(device, "k512")
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=60)
    launch = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs)
    _assert_split_ordered_tail(launch)
    ref = launch.launch().clone()
    torch.cuda.synchronize()
    base._assert_wgrad_close(ref, base._reference("wgrad", g, x, sizes))
    calls = {
        "launch": launch.launch,
        "package": lambda: pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs),
        "facade": lambda: facade.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs),
    }
    for warn_only in (False, True):
        for label, call in calls.items():
            launch.out.fill_(float("nan"))
            assert base._poison_tail_sums(device), "no kept tail slots on this stream"
            with (
                _TorchDeterministic(warn_only),
                warnings.catch_warnings(record=True) as rec,
            ):
                warnings.simplefilter("always")
                out = call()
            torch.cuda.synchronize()
            what = f"{label} warn_only={warn_only}"
            assert _user_warnings(rec) == [], (what, [str(r.message) for r in rec])
            base._assert_same_bits(out, ref, what)
    base._assert_tail_at_rest(device)

    sizes = base.RAGGED_E4
    x, w, g, offs = base._inputs(sizes, 512, 256, device, seed=61)
    k256 = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs)
    assert k256.plan["tile_k"] == 256 and not k256.order_varying, k256.plan
    fwd = cb.cake_moe_grouped_gemm_nondet_train_prepare_forward(x, w, offs)
    ref = k256.launch().clone()
    y_ref = fwd.launch().clone()
    for warn_only in (False, True):
        k256.out.fill_(float("nan"))
        fwd.out.fill_(float("nan"))
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            out = k256.launch()
            y = fwd.launch()
        torch.cuda.synchronize()
        assert _user_warnings(rec) == [], (warn_only, [str(r.message) for r in rec])
        base._assert_same_bits(out, ref, f"k256 dW warn_only={warn_only}")
        base._assert_same_bits(y, y_ref, f"Y warn_only={warn_only}")


def test_autograd_backward_never_alerts():
    """Under torch's deterministic mode (strict and ``warn_only``) the backward of the
    package's ``CakeMoEGroupedGemmNondetFunction`` (through the package and through the
    module; no fallback) never alerts or warns, on a 512-wide k tile shape whose weight
    gradient splits its ordered one-launch tail and on a 256-wide k tile shape, with
    ``dW`` needed or not, and its gradients have the bits of a run with the mode off
    (the stream's tail slots garbage-filled before every run; ``dW`` within the budget
    of the fp64 reference)."""
    base = _base()
    device = base._require_autograd_programs()
    entries = {
        "package": pkg.cake_moe_grouped_gemm_nondet,
        "facade": facade.cake_moe_grouped_gemm_nondet,
    }

    def run(entry, x0, w0, g, offs, needs_w=True):
        base._poison_tail_sums(device)
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(needs_w)
        entry(x, w, offs).backward(g)
        torch.cuda.synchronize()
        return x.grad, w.grad

    k512_sizes, k512_shape = base._tile_case(device, "k512")
    cases = (
        ("k512", k512_sizes, k512_shape, 62),
        ("k256", base.RAGGED_E4, (512, 256), 63),
    )
    for tile, sizes, (n, k), seed in cases:
        x0, w0, g, offs = base._inputs(sizes, n, k, device, seed=seed)
        # The backward prepares this weight gradient (same operands, same plan).
        planned = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x0, offs)
        if tile == "k512":
            _assert_split_ordered_tail(planned)
        else:
            assert planned.plan["tile_k"] == 256, planned.plan
            assert not planned.order_varying, planned.plan
        want = run(entries["package"], x0, w0, g, offs)
        base._assert_wgrad_close(want[1], base._reference("wgrad", g, x0, sizes))
        for warn_only in (False, True):
            for via, entry in entries.items():
                with (
                    _TorchDeterministic(warn_only),
                    warnings.catch_warnings(record=True) as rec,
                ):
                    warnings.simplefilter("always")
                    dx, dw = run(entry, x0, w0, g, offs)
                    dx_only, no_dw = run(entry, x0, w0, g, offs, needs_w=False)
                what = f"{tile} {via} warn_only={warn_only}"
                assert _user_warnings(rec) == [], (what, [str(r.message) for r in rec])
                base._assert_same_bits(dx, want[0], f"dX {what}")
                base._assert_same_bits(dw, want[1], f"dW {what}")
                assert no_dw is None, what
                base._assert_same_bits(dx_only, want[0], f"dX alone {what}")


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_facade_checkpoint_early_stop_skips_the_forward_launch(early_stop, monkeypatch):
    """Through the module the package's wrapper launches its forward after autograd
    saved the inputs: a non-reentrant checkpoint of one GEMM launches the forward once
    with early stop on (the recomputation stops before it) and twice with it off, with
    the bits of the plain call for ``Y`` and ``dX``."""
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
        return facade.cake_moe_grouped_gemm_nondet(x, w, offs)

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
