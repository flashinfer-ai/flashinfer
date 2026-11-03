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

flashinfer public-API tests of the Cake packed KDA chunked training with a selectable numerical mode: the facade
``flashinfer.cake_kda_train`` and the non-deterministic package ``flashinfer.experimental.cake.cake_kda_train``.

Every facade entry takes ``deterministic: bool = True``: ``True`` (the default) runs the deterministic package
``flashinfer.experimental.cake_kda_det_train``, ``False`` the non-deterministic one, and torch's deterministic mode
turns ``False`` into the deterministic package with one ``UserWarning`` per process.  On the CPU this suite covers the
facade's marking, signatures and dispatch table (the package functions replaced by recorders), its refusal of a
non-bool mode, the refusal of every mode but ``deterministic=False`` by the package entries, its autograd node
``ChunkKDAFunction`` and its workspace ``KdaWorkspace`` (also under torch's
deterministic mode, which they do not alert: the S1 schedule has no unordered reduction), the raw gate mode (#7816)
on both packages (the facade forwards its keywords to the package of the call's mode, this package takes them with
the deterministic package's checks, a workspace serves one gate mode, either backward reads either package's saved
set of either gate mode), the in-kernel Q/K L2 norm (#7817) on both packages (the facade forwards
``use_qk_l2norm_in_kernel`` to the package of the call's mode, this package takes it with the deterministic package's
checks and byte counts, either package's saved-set check reads either package's l2n set), the legacy names
(``flashinfer.kda_train``, ``flashinfer.experimental.cake_kda_train``) that stay the deterministic objects, the
deferred imports of the facade and the package's isolation from the deterministic modules, and a dispatch sweep
over architecture x path x heads 1..128 (one record per stage, the fused gradient and the GLM intra program agreeing
on S1 as the independent table says, the plan's records the registry's, every record of the pair stages launched
on every architecture it ships for; the in-kernel Q/K L2 norm modes select the deterministic pair on every cell)
with the plan's pairing guard.  On a GPU: the public entries against both packages, the workspaces of
either mode, torch's deterministic mode, the mode the autograd node records at its forward, the saved set of
either package and the raw gate mode of both packages (bitwise the same results).
"""

import contextlib
import dataclasses
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

import flashinfer.cake_kda_det_train as det_api
import flashinfer.cake_kda_train as facade
from flashinfer.api_logging import ExperimentalWarning
from tests.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_kda_train")
cake_backend = cake_package("cake.cake_kda_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_train", "cake_jit")
det_pkg = cake_package("cake_kda_det_train")
det_backend = cake_package("cake_kda_det_train", "cake_backend")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAD_ORDER = ("dq", "dk", "dv", "dbeta", "dg")  # the order the backward returns
# facade entry -> its deterministic facade name (the shared spellings of flashinfer.kda_train)
NAME_MAP = {
    "chunk_kda_fwd": "chunk_kda_det_fwd",
    "chunk_kda_bwd": "chunk_kda_det_bwd",
    "chunk_kda": "chunk_kda_det",
    "kda_train_arena": "kda_det_train_arena",
    "kda_train_workspace": "kda_det_train_workspace",
    "kda_train_workspace_bytes": "kda_det_train_workspace_bytes",
    "kda_train_saved_bytes": "kda_det_train_saved_bytes",
    "is_kda_train_supported": "is_kda_det_train_supported",
}
EXPERIMENTAL_ENTRIES = ("chunk_kda_fwd", "chunk_kda_bwd", "chunk_kda")
# The backend function each facade entry hands off to: (deterministic backend attribute, this package's).
TARGETS = {
    "chunk_kda_fwd": ("chunk_kda_det_fwd", "chunk_kda_fwd"),
    "chunk_kda_bwd": ("chunk_kda_det_bwd", "chunk_kda_bwd"),
    "chunk_kda": ("chunk_kda_det", "chunk_kda"),
    "kda_train_workspace_bytes": ("workspace_bytes", "workspace_bytes"),
    "kda_train_saved_bytes": ("saved_bytes", "saved_bytes"),
    "is_kda_train_supported": ("is_supported", "is_supported"),
}
_FALLBACK = "runs the deterministic package"
_ALERT = "does not have a deterministic implementation"

# The S1 cells (the backend suite's table, spelled again: the suites import nothing from each other).  The saved-set
# paths run the S1 pair from S1_SAVED_MIN_H heads, the replay up to S1_REPLAY_MAX_H heads (None: every count), the
# serial rows of 9..15 heads the deterministic pair everywhere; the sm_107a cells are provisional.
SAVED_PATHS = ("light", "light_states", "tape")
S1_SAVED_MIN_H = {"sm_100a": 16, "sm_103a": 1, "sm_107a": 1}
S1_REPLAY_MAX_H = {"sm_100a": None, "sm_103a": 8, "sm_107a": None}
SEG_PLAN_HEADS_BY_ARCH = {"sm_103a": (16,), "sm_107a": (16,)}
S1_GLM = {
    "sm_100a": "glm_intra_v21",
    "sm_103a": "glm_intra_v22",
    "sm_107a": "glm_intra_v23",
}
S1_RECORDS = frozenset(
    (
        "dqkg_davq_v24",
        "dqkg_int_davq_v24",
        "dqkg_davq_nohint_v24",
        "dqkg_int_davq_nohint_v24",
        *S1_GLM.values(),
    )
)


def _s1(arch: str, path: str, h: int) -> bool:
    if path not in (*SAVED_PATHS, "replay"):
        return False
    seg = h <= 8 or h in SEG_PLAN_HEADS_BY_ARCH.get(arch, ())
    if not seg and h < 16:
        return False
    if path == "replay":
        bound = S1_REPLAY_MAX_H[arch]
        return bound is None or h <= bound
    return h >= S1_SAVED_MIN_H[arch]


class _TorchDeterministic:
    """``torch.use_deterministic_algorithms(True, warn_only=...)`` inside the block, the previous setting restored
    after it."""

    def __init__(self, warn_only: bool = False):
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


def _run_cpu_subprocess(code: str) -> str:
    """Run ``code`` in a fresh interpreter with no visible GPU; its last stdout line."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()[-1]


def _fallback_warnings(records) -> list:
    return [
        w for w in records if w.category is UserWarning and _FALLBACK in str(w.message)
    ]


# --------------------------------------------------------------------------- the facade (CPU)


def test_facade_exports_and_marking() -> None:
    assert sorted(facade.__all__) == sorted(["DEFAULT_SCALE", *NAME_MAP])
    for name in EXPERIMENTAL_ENTRIES:
        fn = getattr(facade, name)
        assert getattr(fn, "is_experimental", False) is True, name
        assert fn.experimental_feature == facade._FEATURE
        assert ".. warning::" in fn.__doc__ and "deterministic" in fn.__doc__, name
    assert facade.DEFAULT_SCALE == det_api.DEFAULT_SCALE == 128**-0.5
    assert facade.DEFAULT_SCALE == cake_backend.SCALE == det_backend.SCALE
    # the package entries are not the experimental API themselves
    for name in EXPERIMENTAL_ENTRIES:
        assert not hasattr(getattr(cake_backend, name), "is_experimental"), name


def test_signatures_mirror_the_deterministic_facade() -> None:
    """Each entry takes the parameters of its deterministic facade entry, in order, kinds and defaults the same, and
    a keyword-only ``deterministic: bool = True`` after them; ``chunk_kda`` also takes the keyword-only
    ``checkpoint_handoff="auto"`` (the non-deterministic package's checkpoint hand-off) before ``deterministic``."""
    for name, det_name in NAME_MAP.items():
        params = list(inspect.signature(getattr(facade, name)).parameters.values())
        if name == "chunk_kda":
            handoff = params.pop(-2)
            assert handoff.name == "checkpoint_handoff", name
            assert handoff.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert handoff.default == "auto", name
        det_params = list(
            inspect.signature(getattr(det_api, det_name)).parameters.values()
        )
        last = params[-1]
        assert last.name == "deterministic", name
        assert last.kind is inspect.Parameter.KEYWORD_ONLY and last.default is True, (
            name
        )
        assert [(p.name, p.kind, p.default) for p in params[:-1]] == [
            (p.name, p.kind, p.default) for p in det_params
        ], name
    # the package's entries take the keyword too (its default refuses: see test_package_refuses_every_mode_but_false)
    for name in EXPERIMENTAL_ENTRIES:
        parameter = inspect.signature(getattr(cake_backend, name)).parameters[
            "deterministic"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is True, name


def _recorders(monkeypatch) -> list:
    """Replace the backend functions the facade hands off to by recorders; return the log of
    ``(entry, side, kwargs)``.  The fallback warning's once-per-process flag starts unset."""
    log: list = []
    for entry, (det_attr, nd_attr) in TARGETS.items():
        for side, module, attr in (
            ("det", det_backend, det_attr),
            ("nd", cake_backend, nd_attr),
        ):

            def record(*args, _side=side, _entry=entry, **kwargs):
                log.append((_entry, _side, kwargs))
                return _side

            monkeypatch.setattr(module, attr, record)
    for side, cls in (
        ("det", det_backend.KdaDetWorkspace),
        ("nd", cake_backend.KdaWorkspace),
    ):

        def for_problem(*args, _side=side, **kwargs):
            log.append(("kda_train_workspace", _side, kwargs))
            return _side

        monkeypatch.setattr(cls, "for_problem", for_problem)
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    return log


_OMIT = object()


def _call(name: str, deterministic=_OMIT):
    """One call of facade entry ``name`` on placeholder operands (the recorders never read them)."""
    operands = tuple(object() for _ in range(6))
    args = {
        "chunk_kda_fwd": operands,
        "chunk_kda_bwd": (*operands, object()),
        "chunk_kda": operands,
        "kda_train_arena": (),
        "kda_train_workspace": (1, 1, 1, "cpu"),
        "kda_train_workspace_bytes": (1, 1, 1),
        "kda_train_saved_bytes": (1, 1),
        "is_kda_train_supported": (),
    }[name]
    kwargs = {} if deterministic is _OMIT else {"deterministic": deterministic}
    if name == "kda_train_workspace_bytes":
        kwargs["sm_count"] = 148
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return getattr(facade, name)(*args, **kwargs)


def _side_of(name: str, result) -> str:
    """Which package served a call: the recorder's tag, or the arena option's type."""
    if name == "kda_train_arena":
        if isinstance(result, cake_backend.KdaArena):
            return "nd"
        assert isinstance(result, det_backend.KdaDetArena), type(result)
        return "det"
    return result


@pytest.mark.parametrize("name", list(NAME_MAP))
def test_facade_dispatch_table(name, monkeypatch) -> None:
    """``deterministic`` omitted or ``True``: the deterministic package; ``False``: this package (its entries with
    ``deterministic=False``); ``False`` under torch's deterministic mode (also with ``warn_only``): the
    deterministic package and one ``UserWarning`` per process, naming the caller's line."""
    log = _recorders(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sides = [_side_of(name, _call(name, mode)) for mode in (_OMIT, True, False)]
    assert sides == ["det", "det", "nd"]
    assert _fallback_warnings(rec) == []
    for entry, side, kwargs in log:
        assert entry == name
        if side == "det":
            assert "deterministic" not in kwargs, kwargs
        elif name in (*EXPERIMENTAL_ENTRIES, "kda_train_workspace"):
            assert kwargs["deterministic"] is False, kwargs
    log.clear()
    for warn_only in (False, True):
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            sides = [_side_of(name, _call(name, False)) for _ in range(3)]
            sides.append(_side_of(name, _call(name, True)))
        assert sides == ["det"] * 4
        fallback = _fallback_warnings(rec)
        # one warning per process: the first block warns once, the second is silent
        assert len(fallback) == (0 if warn_only else 1), [
            str(w.message) for w in fallback
        ]
        if fallback:
            assert f"{name}: deterministic=False" in str(fallback[0].message)
            assert os.path.samefile(fallback[0].filename, __file__), fallback[
                0
            ].filename
    assert all(side == "det" for _entry, side, _kw in log)


@pytest.mark.parametrize("name", list(NAME_MAP))
@pytest.mark.parametrize("value", [0, 1, None, "False", 1.0], ids=repr)
def test_facade_rejects_a_non_bool_mode(name, value, monkeypatch) -> None:
    log = _recorders(monkeypatch)
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        _call(name, value)
    with (
        _TorchDeterministic(),
        pytest.raises(TypeError, match="deterministic must be a bool"),
    ):
        _call(name, value)
    assert log == []


def test_facade_routes_the_arena_option_to_the_package(monkeypatch) -> None:
    """An arena option of either package serves both: the call hands the routed package its own type with the same
    fields; the option factory returns the routed package's type."""
    log = _recorders(monkeypatch)
    det_arena = facade.kda_train_arena(lean=True, tail_passes=2)
    nd_arena = facade.kda_train_arena(lean=True, tail_passes=2, deterministic=False)
    assert isinstance(det_arena, det_backend.KdaDetArena)
    assert isinstance(nd_arena, cake_backend.KdaArena)
    assert det_arena == det_backend.KdaDetArena(lean=True, tail_passes=2)
    assert nd_arena == cake_backend.KdaArena(lean=True, tail_passes=2)
    operands = tuple(object() for _ in range(6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for arena in (det_arena, nd_arena):
            facade.chunk_kda_fwd(*operands, arena=arena)
            facade.chunk_kda_fwd(*operands, arena=arena, deterministic=False)
    # the calls in order: each option to the deterministic package, then to this one
    sources = (det_arena, det_arena, nd_arena, nd_arena)
    assert [side for _entry, side, _kw in log] == ["det", "nd", "det", "nd"]
    for (_entry, side, kwargs), source in zip(log, sources, strict=True):
        got = kwargs["arena"]
        own = det_backend.KdaDetArena if side == "det" else cake_backend.KdaArena
        assert isinstance(got, own), (side, type(got))
        assert dataclasses.astuple(got) == dataclasses.astuple(source)
        assert (got is source) is isinstance(source, own)


def test_byte_queries_are_the_packages_and_agree() -> None:
    # An explicit SM count keeps the queries device-free.  The two packages' workspaces and saved sets have the same
    # size (the layouts are the same; the S1 programs bind the deterministic forms' buffers).
    for S, N, H in ((1, 1, 1), (129, 2, 4), (4096, 3, 16), (261393, 4, 64)):
        for kwargs in ({}, {"backward": False}, {"save_mode": "tape"}):
            nd = facade.kda_train_workspace_bytes(
                S, N, H, sm_count=148, deterministic=False, **kwargs
            )
            assert nd == cake_backend.workspace_bytes(S, N, H, sm_count=148, **kwargs)
            assert nd == facade.kda_train_workspace_bytes(
                S, N, H, sm_count=148, **kwargs
            )
            assert nd == det_backend.workspace_bytes(S, N, H, sm_count=148, **kwargs)
        for kwargs in ({}, {"save_mode": "tape"}):
            nd = facade.kda_train_saved_bytes(
                S, H, N, sm_count=148, deterministic=False, **kwargs
            )
            assert nd == cake_backend.saved_bytes(S, H, N, sm_count=148, **kwargs)
            assert nd == det_backend.saved_bytes(S, H, N, sm_count=148, **kwargs)
        assert facade.kda_train_saved_bytes(S, H, deterministic=False) == 256 * S * H


def test_support_query_follows_the_capability_table() -> None:
    for deterministic in (True, False):
        if not torch.cuda.is_available():
            assert facade.is_kda_train_supported(deterministic=deterministic) is False
            continue
        index = torch.cuda.current_device()
        expected = (
            cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
            is not None
        )
        assert facade.is_kda_train_supported(deterministic=deterministic) is expected
        assert (
            facade.is_kda_train_supported(index, deterministic=deterministic)
            is expected
        )


_WARN_ONCE = """
import warnings

import torch

import flashinfer.cake_kda_train as facade

assert not torch.cuda.is_available()
torch.use_deterministic_algorithms(True)
caught = []
for name, args in (
    ("kda_train_saved_bytes", (1, 1)),
    ("kda_train_workspace_bytes", (1, 1, 1)),
    ("kda_train_saved_bytes", (64, 2)),
):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        kwargs = {"sm_count": 148} if name == "kda_train_workspace_bytes" else {}
        getattr(facade, name)(*args, deterministic=False, **kwargs)
    caught += [w for w in rec if w.category is UserWarning]
assert len(caught) == 1, [str(w.message) for w in caught]
assert caught[0].filename == "<string>", (caught[0].filename, caught[0].lineno)
print("OK")
"""


def test_fallback_warns_once_per_process_and_names_the_caller() -> None:
    """In a fresh process with torch's deterministic mode on, the first ``deterministic=False`` call of any entry
    warns (at the caller's frame, the ``-c`` program), and no later call of any entry warns again."""
    assert _run_cpu_subprocess(_WARN_ONCE) == "OK"


def test_importing_the_facade_imports_neither_package() -> None:
    """The facade reaches both packages through deferred imports only."""
    code = (
        "import json, sys\n"
        "import flashinfer.cake_kda_train\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('flashinfer.experimental.cake_kda_det_train')"
        " or m.startswith('flashinfer.experimental.cake.cake_kda_train'))))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


def test_importing_the_package_imports_no_deterministic_module() -> None:
    """The package and every module of it import nothing of the deterministic package, of its facade, of the legacy
    compatibility package or of the legacy names (an import of either would let a launch reach a deterministic
    program)."""
    code = (
        "import json, sys\n"
        "import flashinfer\n"
        "before = set(sys.modules)\n"
        "import flashinfer.experimental.cake.cake_kda_train\n"
        "from flashinfer.experimental.cake.cake_kda_train import cake_backend, cake_build, cake_jit\n"
        "new = set(sys.modules) - before\n"
        "assert 'flashinfer.experimental.cake.cake_kda_train.cake_backend' in new\n"
        "bad = sorted(m for m in sys.modules if 'cake_kda_det_train' in m"
        " or m == 'flashinfer.experimental.cake_kda_train'"
        " or m.startswith('flashinfer.experimental.cake_kda_train.')"
        " or m in ('flashinfer.kda_train', 'flashinfer.cake_kda_train'))\n"
        "print(json.dumps(bad))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


# --------------------------------------------------------------------------- the package's mode keyword (CPU)


def _bad_operands():
    """CPU operands whose ``v`` has the wrong rank: an accepted mode reaches the host check and stops there."""
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 1, dtype=torch.bfloat16)
    g = torch.zeros(4, 1, 128)
    beta = torch.zeros(4, 1, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    return q, q, v, g, beta, cu


PACKAGE_CALLS = {
    "chunk_kda_fwd": lambda **kw: cake_backend.chunk_kda_fwd(*_bad_operands(), **kw),
    "chunk_kda_bwd": lambda **kw: cake_backend.chunk_kda_bwd(
        *_bad_operands(), torch.zeros(4, 1, 128, dtype=torch.bfloat16), **kw
    ),
    "chunk_kda": lambda **kw: cake_backend.chunk_kda(*_bad_operands(), **kw),
    # the autograd node: the mode is its trailing positional argument (omitted: the refused default), after the
    # checkpoint hand-off's (None: "auto")
    "ChunkKDAFunction": lambda **kw: cake_backend.ChunkKDAFunction.apply(
        *_bad_operands(), cake_backend.SCALE, None, None, None, None, None, *kw.values()
    ),
}


@pytest.mark.parametrize("name", list(PACKAGE_CALLS))
def test_package_refuses_every_mode_but_false(name) -> None:
    """Every entry of the package raises ``ValueError`` naming ``deterministic=False`` and the deterministic package
    unless ``deterministic=False`` (a call that swaps only the import path of the deterministic package fails); the
    refusal comes first, under torch's deterministic mode too; with ``False`` the call reaches its host checks."""
    call = PACKAGE_CALLS[name]
    for kwargs in (
        {},
        {"deterministic": True},
        {"deterministic": None},
        {"deterministic": 0},
        {"deterministic": "False"},
    ):
        for torch_det in (False, True):
            with (
                _TorchDeterministic() if torch_det else contextlib.nullcontext(),
                pytest.raises(ValueError, match="deterministic=False") as info,
            ):
                call(**kwargs)
            assert "cake_kda_det_train" in str(info.value), (name, kwargs)
            assert str(info.value).startswith(name), (name, str(info.value))
    with pytest.raises(ValueError, match="v: expected 3-D") as info:
        call(deterministic=False)
    assert "cake_kda_det_train" not in str(info.value), name


@pytest.mark.parametrize("name", list(PACKAGE_CALLS))
def test_package_entries_do_not_alert_under_torch_deterministic_mode(name) -> None:
    """The S1 schedule has a fixed order and no unordered reduction: under torch's deterministic mode (an alert as
    an error, a warning as an error) each entry with ``deterministic=False`` proceeds to its host checks, never to an
    alert and never to the deterministic package."""
    for warn_only in (False, True):
        with _TorchDeterministic(warn_only), warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="v: expected 3-D") as info:
                PACKAGE_CALLS[name](deterministic=False)
            assert _ALERT not in str(info.value)


def test_package_workspace_refuses_every_mode_but_false() -> None:
    """The workspace launches this package's programs: its constructor, ``for_problem`` and ``for_budget`` raise
    ``ValueError`` naming ``deterministic=False`` and the deterministic package unless ``deterministic=False`` (a
    workspace whose import path alone was swapped from the deterministic package's ``KdaWorkspace`` fails instead
    of handing a deterministic entry this package's pairs); the refusal comes before the device is read, under
    torch's deterministic mode too; with ``False`` the construction reaches its device check."""
    S, N, H = 129, 1, 4
    plan = cake_backend.plan(S, N, H, 148, "sm_100a")
    ws_type = cake_backend.KdaWorkspace
    calls = {
        "constructor": lambda **kw: ws_type(plan, "cpu", **kw),
        "for_problem": lambda **kw: ws_type.for_problem(S, N, H, "cpu", **kw),
        "for_budget": lambda **kw: ws_type.for_budget(
            S, N, H, "cpu", arena_budget_bytes=1 << 40, **kw
        ),
    }
    for name, call in calls.items():
        for kwargs in (
            {},
            {"deterministic": True},
            {"deterministic": None},
            {"deterministic": 0},
            {"deterministic": "False"},
        ):
            for torch_det in (False, True):
                with (
                    _TorchDeterministic() if torch_det else contextlib.nullcontext(),
                    pytest.raises(ValueError, match="deterministic=False") as info,
                ):
                    call(**kwargs)
                assert "cake_kda_det_train" in str(info.value), (name, kwargs)
                assert str(info.value).startswith("KdaWorkspace"), (
                    name,
                    str(info.value),
                )
        with pytest.raises(ValueError, match="CUDA devices") as info:
            call(deterministic=False)
        assert "cake_kda_det_train" not in str(info.value), name


# the raw gate mode (#7816): the keywords at their off values
GATE_OFF = {
    "use_gate_in_kernel": False,
    "A_log": None,
    "dt_bias": None,
    "lower_bound": None,
}


def _gate_on() -> list:
    return [
        {"use_gate_in_kernel": True},
        {"A_log": torch.zeros(4)},
        {"dt_bias": torch.zeros(4, 128)},
        {"lower_bound": -5.0},
        {
            "use_gate_in_kernel": True,
            "A_log": torch.zeros(1),
            "dt_bias": torch.zeros(128),
            "lower_bound": -5.0,
        },
    ]


def _raw_cpu_call():
    """CPU operands of the raw gate mode that pass every operand check (the BF16 raw gate, one head) and its
    keywords: a call with a stand-in workspace reaches the workspace's type check (``TypeError``), which reads no
    device."""
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    beta = torch.zeros(4, 1, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    raw = {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(1),
        "dt_bias": torch.zeros(128),
        "lower_bound": -5.0,
    }
    return (q, q, q, q.clone(), beta, cu), raw


@pytest.mark.parametrize("name", EXPERIMENTAL_ENTRIES)
def test_package_accepts_the_raw_gate(name) -> None:
    """The raw gate mode runs on this package too: with ``deterministic=False`` each entry takes its keywords and
    goes on to the host checks (the stand-in operands' check, as with the keywords off or omitted); the mode's own
    checks are the deterministic package's (a non-bool ``use_gate_in_kernel``, an FP32 raw gate, a missing lower
    bound or one outside ``[-5, 0)``, the parameters in g mode: ``ValueError``), and a well-formed raw call on CPU
    tensors passes them all and reaches the workspace's type check."""
    call = PACKAGE_CALLS[name]
    for kw in _gate_on():
        with pytest.raises(ValueError, match="v: expected 3-D") as info:
            call(deterministic=False, **kw)
        assert "deterministic=True" not in str(info.value), kw
    with pytest.raises(ValueError, match="v: expected 3-D"):
        call(deterministic=False, **GATE_OFF)
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        call(deterministic=False, use_gate_in_kernel=1)
    entry = getattr(cake_backend, name)
    ops, raw = _raw_cpu_call()
    dout = (
        (torch.zeros(4, 1, 128, dtype=torch.bfloat16),)
        if name == "chunk_kda_bwd"
        else ()
    )
    g32 = (*ops[:3], torch.zeros(4, 1, 128), *ops[4:])
    # the arena option given: the backward reads no option from the stand-in workspace
    stand_in = {"workspace": object(), "arena": cake_backend.KdaArena()}
    with pytest.raises(TypeError, match="KdaWorkspace") as info:
        entry(*ops, *dout, deterministic=False, **stand_in, **raw)
    assert "deterministic=False" in str(info.value), name
    with pytest.raises(ValueError, match="is torch.bfloat16"):
        entry(*g32, *dout, deterministic=False, **raw)
    with pytest.raises(ValueError, match="needs the gate's lower bound"):
        entry(*ops, *dout, deterministic=False, **dict(raw, lower_bound=None))
    with pytest.raises(ValueError, match="lower_bound must satisfy"):
        entry(*ops, *dout, deterministic=False, **dict(raw, lower_bound=-6.0))
    with pytest.raises(ValueError, match="taken with use_gate_in_kernel=True only"):
        entry(*g32, *dout, deterministic=False, A_log=raw["A_log"])


def test_autograd_node_takes_the_raw_gate_after_the_mode() -> None:
    """``ChunkKDAFunction.apply`` takes the raw gate mode's arguments after ``deterministic`` (``use_gate_in_kernel``,
    ``A_log``, ``dt_bias``, ``lower_bound``): a raw call on CPU tensors with a stand-in workspace passes the raw
    checks and reaches the workspace's type check, and an ``apply`` written for the deterministic Function's
    signature (the raw arguments right after ``doc_lengths``) passes a ``bool`` in the ``checkpoint_handoff`` slot
    and raises ``TypeError`` naming it."""
    ops, raw = _raw_cpu_call()
    scale = cake_backend.SCALE
    with pytest.raises(TypeError, match="KdaWorkspace"):
        cake_backend.ChunkKDAFunction.apply(
            *ops, scale, object(), None, None, None, None, False, *raw.values()
        )
    with pytest.raises(TypeError, match="checkpoint_handoff"):
        cake_backend.ChunkKDAFunction.apply(
            *ops, scale, None, None, None, None, *raw.values()
        )
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_backend.ChunkKDAFunction.apply(
            *ops, scale, None, None, None, None, None, False, 1, *list(raw.values())[1:]
        )


def test_either_backward_reads_either_packages_saved_set_by_gate_mode() -> None:
    """Both packages' saved sets carry ``gate`` / ``lower_bound`` and both backwards apply the same rule
    (``_check_saved_gate``): a set of the call's gate mode passes, of either package's type, and in raw mode the
    FP32 bits of ``lower_bound`` decide; a set of the other gate mode or of another lower bound is refused with
    ``ValueError`` by either package, in both directions.  A set of the in-kernel Q/K L2 norm (``qk_l2norm``) of
    either gate mode reaches the host checks of a call with ``use_qk_l2norm_in_kernel=True``."""
    t = torch.empty(0)
    scale = cake_backend.SCALE
    H = 2
    params = (torch.zeros(H), torch.zeros(H * 128), (H * 128,), -5.0)
    checks = (
        (cake_backend._check_saved_gate, cake_backend._GateParams(*params)),
        (det_backend._check_saved_gate, det_backend._GateParams(*params)),
    )
    for saved_type in (cake_backend.KdaChunkSaved, det_backend.KdaDetChunkSaved):
        raw = saved_type(
            4, t, t, None, 1, H, 1, scale, qg=t, gate="raw", lower_bound=-5.0
        )
        explicit = saved_type(4, t, t, None, 1, H, 1, scale, qg=t)
        assert (explicit.gate, explicit.lower_bound) == ("g", None)
        for check, gp in checks:
            check(raw, gp)
            check(explicit, None)
            # a lower bound with the same FP32 value
            check(dataclasses.replace(raw, lower_bound=-5.0 + 1e-12), gp)
            with pytest.raises(ValueError, match="gate mode 'raw'"):
                check(raw, None)
            with pytest.raises(ValueError, match="gate mode 'g'"):
                check(explicit, gp)
            for bound in (-4.0, None, -5.0 + 1e-6):
                with pytest.raises(ValueError, match="lower_bound"):
                    check(dataclasses.replace(raw, lower_bound=bound), gp)
    l2n = types.SimpleNamespace(gate="g", qk_l2norm=True)
    l2n_raw = types.SimpleNamespace(gate="raw", lower_bound=-5.0, qk_l2norm=True)
    for saved in (l2n, l2n_raw):
        with pytest.raises(ValueError, match="v: expected 3-D"):
            PACKAGE_CALLS["chunk_kda_bwd"](
                saved=saved, deterministic=False, use_qk_l2norm_in_kernel=True
            )


def test_package_workspace_takes_the_gate_mode() -> None:
    """``workspace_bytes`` and ``KdaWorkspace`` take the gate mode of the calls they serve (``"g"``, the default,
    or ``"raw"``): the raw mode's bytes are the deterministic package's raw bytes plus, on a backward workspace of
    the R1c rows (``sm_100a`` / ``sm_103a``), the FP64 accumulators ``param_acc`` (``8 * stripes * H * 2 * 128``
    bytes; none on ``sm_107a`` or a forward workspace), any other value is a ``ValueError`` raised before the device
    is read, a known one reaches the device check, and a constructor whose ``gate`` is not its plan's is refused."""
    S, N, H = 129, 1, 4
    assert cake_backend.workspace_bytes(
        S, N, H, sm_count=148, gate="g"
    ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for h in (H, 64):
            # one replica, but 8 on sm_100a from 32 heads
            stripes = (
                None
                if arch == "sm_107a"
                else (8 if arch == "sm_100a" and h >= 32 else 1)
            )
            p = cake_backend.plan(S, N, h, 148, arch, gate="raw")
            assert cake_backend._r1c_stripes(p) == stripes, (arch, h)
            for backward in (True, False):
                acc = 8 * stripes * h * 2 * 128 if stripes and backward else 0
                assert (
                    cake_backend.workspace_bytes(
                        S, N, h, sm_count=148, arch=arch, gate="raw", backward=backward
                    )
                    == det_backend.workspace_bytes(
                        S, N, h, sm_count=148, arch=arch, gate="raw", backward=backward
                    )
                    + acc
                ), (arch, h, backward)
    ws_type = cake_backend.KdaWorkspace
    for gate in (None, "G", "activated"):
        with pytest.raises(ValueError, match="gate must be one of"):
            cake_backend.workspace_bytes(S, N, H, sm_count=148, gate=gate)
        with pytest.raises(ValueError, match="gate must be one of"):
            ws_type.for_problem(S, N, H, "cpu", gate=gate, deterministic=False)
        with pytest.raises(ValueError, match="gate must be one of"):
            ws_type.for_budget(
                S,
                N,
                H,
                "cpu",
                arena_budget_bytes=1 << 40,
                gate=gate,
                deterministic=False,
            )
    for gate in ("g", "raw"):
        with pytest.raises(ValueError, match="CUDA devices"):
            ws_type.for_problem(S, N, H, "cpu", gate=gate, deterministic=False)
        p = cake_backend.plan(S, N, H, 148, "sm_100a", gate=gate)
        other = "g" if gate == "raw" else "raw"
        with pytest.raises(ValueError, match=f"gate={other!r}"):
            ws_type(p, "cpu", gate=other, deterministic=False)
        with pytest.raises(ValueError, match="CUDA devices"):
            ws_type(p, "cpu", gate=gate, deterministic=False)


def test_facade_forwards_the_raw_gate(monkeypatch) -> None:
    """Both modes hand the raw gate keywords (the same objects) to their package, and ``use_gate_in_kernel`` to its
    workspace and byte query as ``gate="raw"``; the off values reach the package as they are; under torch's
    deterministic mode the deterministic package receives them."""
    log = _recorders(monkeypatch)
    operands = tuple(object() for _ in range(6))
    gate = {
        "use_gate_in_kernel": True,
        "A_log": object(),
        "dt_bias": object(),
        "lower_bound": -5.0,
    }
    calls = {
        "chunk_kda_fwd": lambda **kw: facade.chunk_kda_fwd(*operands, **kw),
        "chunk_kda_bwd": lambda **kw: facade.chunk_kda_bwd(*operands, object(), **kw),
        "chunk_kda": lambda **kw: facade.chunk_kda(*operands, **kw),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, call in calls.items():
            for mode, side in (({}, "det"), ({"deterministic": False}, "nd")):
                assert call(**mode, **gate) == side, (name, mode)
                for key, value in gate.items():
                    assert log[-1][2][key] is value, (name, mode, key)
                assert call(**mode) == side, (name, mode)
                assert {key: log[-1][2][key] for key in GATE_OFF} == GATE_OFF, name
        with _TorchDeterministic():
            for name, call in calls.items():
                assert call(deterministic=False, **gate) == "det", name
                assert log[-1][2]["use_gate_in_kernel"] is True, name
        for mode, side in (({}, "det"), ({"deterministic": False}, "nd")):
            assert (
                facade.kda_train_workspace(
                    1, 1, 1, "cpu", use_gate_in_kernel=True, **mode
                )
                == side
            )
            assert log[-1][2]["gate"] == "raw"
            assert facade.kda_train_workspace(1, 1, 1, "cpu", **mode) == side
            assert log[-1][2]["gate"] == "g"
            assert (
                facade.kda_train_workspace_bytes(
                    1, 1, 1, sm_count=148, use_gate_in_kernel=True, **mode
                )
                == side
            )
            assert log[-1][2]["gate"] == "raw"
        for bad in (1, None, "True"):
            with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
                facade.kda_train_workspace_bytes(
                    1, 1, 1, sm_count=148, use_gate_in_kernel=bad
                )


def test_facade_runs_the_raw_gate_on_the_non_deterministic_package() -> None:
    """Without recorders: ``deterministic=False`` with the raw gate reaches this package, whose entries go on to
    their host checks, and whose byte query gives the raw mode's bytes (the deterministic package's, plus on a
    backward workspace of an R1c row, ``sm_100a`` / ``sm_103a``, the FP64 parameter-gradient accumulators)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for name, args in (
            ("chunk_kda_fwd", _bad_operands()),
            ("chunk_kda_bwd", (*_bad_operands(), torch.zeros(4, 1, 128))),
            ("chunk_kda", _bad_operands()),
        ):
            with pytest.raises(ValueError, match="v: expected 3-D") as info:
                getattr(facade, name)(
                    *args, use_gate_in_kernel=True, deterministic=False
                )
            assert "cake_kda_det_train" not in str(info.value), name
        for backward in (True, False):
            got = facade.kda_train_workspace_bytes(
                129,
                1,
                4,
                sm_count=148,
                backward=backward,
                use_gate_in_kernel=True,
                deterministic=False,
            )
            assert got == cake_backend.workspace_bytes(
                129, 1, 4, sm_count=148, backward=backward, gate="raw"
            )
            acc = cake_backend._param_acc_bytes(
                cake_backend.plan(129, 1, 4, 148, gate="raw")
            )
            assert got == facade.kda_train_workspace_bytes(
                129, 1, 4, sm_count=148, backward=backward, use_gate_in_kernel=True
            ) + (acc if backward else 0)


# the in-kernel Q/K L2 norm (#7817): the deterministic package's keyword, on both packages
_NOT_BOOL = "use_qk_l2norm_in_kernel must be a bool"


@pytest.mark.parametrize("name", EXPERIMENTAL_ENTRIES)
def test_package_takes_the_in_kernel_qk_l2norm(name) -> None:
    """The in-kernel Q/K L2 norm runs on this package too: with ``deterministic=False`` each entry takes
    ``use_qk_l2norm_in_kernel=True`` (and ``False``, as the omitted keyword) on to its host checks, and raises
    ``ValueError`` for a value that is not a bool before them."""
    call = PACKAGE_CALLS[name]
    for value in (True, False):
        with pytest.raises(ValueError, match="v: expected 3-D") as info:
            call(deterministic=False, use_qk_l2norm_in_kernel=value, **GATE_OFF)
        assert "L2 norm" not in str(info.value), (value, str(info.value))
    for value in (1, 0, "True", None):
        with pytest.raises(ValueError, match=_NOT_BOOL):
            call(deterministic=False, use_qk_l2norm_in_kernel=value)


def _l2n_saved(saved_type, S: int, H: int, **fields):
    t = torch.empty(0)
    return saved_type(4, t, t, None, S, H, 1, cake_backend.SCALE, qg=t, **fields)


def test_either_packages_check_reads_either_packages_l2n_saved_set() -> None:
    """Both packages' saved sets carry ``qk_rstd`` / ``qk_l2norm`` under the same names and both backwards apply the
    same rule (``_check_saved_l2n``): a set of the call's in-kernel Q/K L2 norm mode passes, of either package's
    type, in both directions; a set of the other mode, a missing or stray ``qk_rstd`` and a ``qk_rstd`` that is not
    FP32 ``[2, S, H]`` contiguous are refused with ``ValueError`` by either package."""
    S, H = 3, 2
    cpu = torch.device("cpu")
    rstd = torch.zeros(2, S, H, dtype=torch.float32)
    checks = (cake_backend._check_saved_l2n, det_backend._check_saved_l2n)
    passed = 0
    for saved_type in (cake_backend.KdaChunkSaved, det_backend.KdaDetChunkSaved):
        on = _l2n_saved(saved_type, S, H, qk_rstd=rstd, qk_l2norm=True)
        off = _l2n_saved(saved_type, S, H)
        assert (off.qk_rstd, off.qk_l2norm) == (None, False)
        for check in checks:
            for mode in ("fused", "fallback"):
                check(on, mode, S, H, cpu)
                passed += 1
            check(off, "off", S, H, cpu)
            passed += 1
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=True; this backward"
            ):
                check(on, "off", S, H, cpu)
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=False; this backward"
            ):
                check(off, "fused", S, H, cpu)
            with pytest.raises(ValueError, match="carries none|has none"):
                check(_l2n_saved(saved_type, S, H, qk_l2norm=True), "fused", S, H, cpu)
            with pytest.raises(ValueError, match="without the in-kernel"):
                check(_l2n_saved(saved_type, S, H, qk_rstd=rstd), "off", S, H, cpu)
            for bad in (
                torch.zeros(2, S + 1, H),
                torch.zeros(2, S, H, dtype=torch.float64),
                torch.zeros(2, H, S).transpose(1, 2),
            ):
                with pytest.raises(ValueError):
                    check(
                        _l2n_saved(saved_type, S, H, qk_rstd=bad, qk_l2norm=True),
                        "fused",
                        S,
                        H,
                        cpu,
                    )
    assert passed == 2 * 2 * 3
    # the entry: an l2n set reaches the host checks of a call with the flag
    with pytest.raises(ValueError, match="v: expected 3-D"):
        PACKAGE_CALLS["chunk_kda_bwd"](
            saved=_l2n_saved(
                cake_backend.KdaChunkSaved, 4, 1, qk_rstd=rstd, qk_l2norm=True
            ),
            deterministic=False,
            use_qk_l2norm_in_kernel=True,
        )


def test_package_queries_take_the_in_kernel_qk_l2norm_layout() -> None:
    """``workspace_bytes``, ``saved_bytes``, ``output_bytes`` and ``KdaWorkspace.for_problem`` take the
    deterministic package's ``l2n``: ``"off"`` gives the bytes of the omitted keyword, an l2n mode (``"fused"`` /
    ``"fallback"``, which runs the deterministic package's records in either gate mode) the deterministic package's
    bytes of that mode, and the saved set adds the reciprocal norms (``8 S H`` bytes); any other value is a
    ``ValueError`` raised before the device is read, a known one reaches the device check."""
    S, N, H = 129, 1, 4
    assert (
        tuple(cake_backend.L2N_MODES)
        == tuple(det_backend.L2N_MODES)
        == (
            "off",
            "fused",
            "fallback",
        )
    )
    assert cake_backend.workspace_bytes(
        S, N, H, sm_count=148, l2n="off"
    ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
    assert cake_backend.saved_bytes(
        S, H, N, sm_count=148, l2n="off"
    ) == cake_backend.saved_bytes(S, H, N, sm_count=148)
    compared = 0
    for l2n in cake_backend.L2N_MODES:
        for gate in ("g", "raw"):
            if (gate, l2n) == ("g", "off"):
                continue  # the S1 pairs' cells: test_cake_kda_train.py's byte tests
            for backward in (True, False):
                for save_mode in ("light", "tape"):
                    kw = dict(
                        backward=backward,
                        save_mode=save_mode,
                        sm_count=148,
                        arch="sm_100a",
                        gate=gate,
                        l2n=l2n,
                    )
                    assert cake_backend.workspace_bytes(
                        S, N, H, **kw
                    ) == det_backend.workspace_bytes(S, N, H, **kw), kw
                    compared += 1
            assert cake_backend.output_bytes(
                S, H, backward=True, gate=gate, l2n=l2n
            ) == det_backend.output_bytes(S, H, backward=True, gate=gate, l2n=l2n)
        for save_mode in ("light", "tape"):
            q = dict(save_mode=save_mode, sm_count=148, arch="sm_100a")
            nd = cake_backend.saved_bytes(S, H, N, **q, l2n=l2n)
            off = cake_backend.saved_bytes(S, H, N, **q)
            assert nd == off + (0 if l2n == "off" else 8 * S * H), (l2n, save_mode)
            if l2n != "off":
                assert nd == det_backend.saved_bytes(S, H, N, **q, l2n=l2n)
        with pytest.raises(ValueError, match="CUDA devices"):
            cake_backend.KdaWorkspace.for_problem(
                S, N, H, "cpu", l2n=l2n, deterministic=False
            )
    assert compared == (3 * 2 - 1) * 2 * 2
    for l2n in (None, "OFF", "on", True):
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.workspace_bytes(S, N, H, sm_count=148, l2n=l2n)
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.saved_bytes(S, H, N, sm_count=148, l2n=l2n)
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.KdaWorkspace.for_problem(
                S, N, H, "cpu", l2n=l2n, deterministic=False
            )


def test_facade_forwards_the_in_kernel_qk_l2norm(monkeypatch) -> None:
    """Both modes hand ``use_qk_l2norm_in_kernel`` (the same object) to their package, and to its workspace and
    byte queries as ``l2n="fused"`` (``"off"`` without it); under torch's deterministic mode the deterministic
    package receives it; the workspace and the byte queries refuse a non-bool."""
    log = _recorders(monkeypatch)
    operands = tuple(object() for _ in range(6))
    calls = {
        "chunk_kda_fwd": lambda **kw: facade.chunk_kda_fwd(*operands, **kw),
        "chunk_kda_bwd": lambda **kw: facade.chunk_kda_bwd(*operands, object(), **kw),
        "chunk_kda": lambda **kw: facade.chunk_kda(*operands, **kw),
    }
    queries = {
        "kda_train_workspace": lambda **kw: facade.kda_train_workspace(
            1, 1, 1, "cpu", **kw
        ),
        "kda_train_workspace_bytes": lambda **kw: facade.kda_train_workspace_bytes(
            1, 1, 1, sm_count=148, **kw
        ),
        "kda_train_saved_bytes": lambda **kw: facade.kda_train_saved_bytes(1, 1, **kw),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, call in calls.items():
            for mode, side in (({}, "det"), ({"deterministic": False}, "nd")):
                assert call(use_qk_l2norm_in_kernel=True, **mode) == side, name
                assert log[-1][2]["use_qk_l2norm_in_kernel"] is True, (name, side)
                assert call(**mode) == side, name
                assert log[-1][2]["use_qk_l2norm_in_kernel"] is False, (name, side)
        with _TorchDeterministic():
            for name, call in calls.items():
                assert call(deterministic=False, use_qk_l2norm_in_kernel=True) == "det"
                assert log[-1][2]["use_qk_l2norm_in_kernel"] is True, name
        for name, query in queries.items():
            for mode, side in (({}, "det"), ({"deterministic": False}, "nd")):
                assert query(use_qk_l2norm_in_kernel=True, **mode) == side, name
                assert log[-1][2]["l2n"] == "fused", (name, side)
                assert query(**mode) == side, name
                assert log[-1][2]["l2n"] == "off", (name, side)
            for bad in (1, None, "True"):
                with pytest.raises(
                    ValueError, match="use_qk_l2norm_in_kernel must be a bool"
                ):
                    query(use_qk_l2norm_in_kernel=bad)


def test_facade_runs_the_in_kernel_qk_l2norm_on_the_non_deterministic_package() -> None:
    """Without recorders: ``deterministic=False`` with ``use_qk_l2norm_in_kernel=True`` reaches this package, whose
    entries go on to their host checks, whose byte queries give the l2n mode's bytes (the deterministic package's)
    and whose workspace reaches its device check."""
    on = {"use_qk_l2norm_in_kernel": True, "deterministic": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for name, args in (
            ("chunk_kda_fwd", _bad_operands()),
            ("chunk_kda_bwd", (*_bad_operands(), torch.zeros(4, 1, 128))),
            ("chunk_kda", _bad_operands()),
        ):
            with pytest.raises(ValueError, match="v: expected 3-D") as info:
                getattr(facade, name)(*args, **on)
            assert "cake_kda_det_train" not in str(info.value), name
        for backward in (True, False):
            got = facade.kda_train_workspace_bytes(
                129, 1, 4, sm_count=148, backward=backward, **on
            )
            assert got == cake_backend.workspace_bytes(
                129, 1, 4, sm_count=148, backward=backward, l2n="fused"
            )
            assert got == facade.kda_train_workspace_bytes(
                129, 1, 4, sm_count=148, backward=backward, use_qk_l2norm_in_kernel=True
            )
        got = facade.kda_train_saved_bytes(129, 4, 1, sm_count=148, **on)
        assert got == cake_backend.saved_bytes(129, 4, 1, sm_count=148, l2n="fused")
        assert got == facade.kda_train_saved_bytes(
            129, 4, 1, sm_count=148, use_qk_l2norm_in_kernel=True
        )
        with pytest.raises(ValueError, match="CUDA devices"):
            facade.kda_train_workspace(129, 1, 4, "cpu", **on)


def test_package_exports_the_det_free_names_only() -> None:
    for name in (
        "chunk_kda",
        "chunk_kda_bwd",
        "chunk_kda_fwd",
        "ChunkKDAFunction",
        "KdaChunkSaved",
        "KdaWorkspace",
    ):
        assert name in pkg.__all__, name
        assert getattr(pkg, name) is getattr(cake_backend, name), name
    assert "S1_RECORDS" in pkg.__all__ and pkg.S1_RECORDS is cake_jit.S1_RECORDS
    # the in-kernel Q/K L2 norm modes are the backend's, as in the deterministic package
    assert "L2N_MODES" in cake_backend.__all__ and "L2N_MODES" not in pkg.__all__
    for name in (
        "chunk_kda_det",
        "chunk_kda_det_fwd",
        "chunk_kda_det_bwd",
        "ChunkKDADetFunction",
        "KdaDetChunkSaved",
        "KdaDetWorkspace",
        "KdaDetArena",
    ):
        assert not hasattr(pkg, name) and not hasattr(cake_backend, name), name
    assert pkg.__name__ == "flashinfer.experimental.cake.cake_kda_train"
    assert set(cake_jit.S1_RECORDS) == S1_RECORDS


# --------------------------------------------------------------------------- legacy names stay deterministic (CPU)


def test_legacy_names_stay_deterministic() -> None:
    """The det-free spellings of ``flashinfer.kda_train`` are the deterministic facade's objects, never this
    facade's; the legacy package path ``flashinfer.experimental.cake_kda_train`` stays the deterministic package's
    compatibility alias, never this package; the test-alias keys follow."""
    import flashinfer.experimental.cake_kda_train as legacy_pkg
    import flashinfer.kda_train as legacy

    for name, det_name in NAME_MAP.items():
        assert getattr(legacy, name) is getattr(det_api, det_name), name
        assert getattr(legacy, name) is not getattr(facade, name), name
        assert (
            "deterministic" not in inspect.signature(getattr(legacy, name)).parameters
        )
    assert legacy.DEFAULT_SCALE == facade.DEFAULT_SCALE
    for name, det_name in (
        ("chunk_kda", "chunk_kda_det"),
        ("chunk_kda_fwd", "chunk_kda_det_fwd"),
        ("chunk_kda_bwd", "chunk_kda_det_bwd"),
        ("ChunkKDAFunction", "ChunkKDADetFunction"),
        ("KdaChunkSaved", "KdaDetChunkSaved"),
        ("KdaWorkspace", "KdaDetWorkspace"),
    ):
        assert getattr(legacy_pkg, name) is getattr(det_pkg, det_name), name
        assert getattr(legacy_pkg, name) is not getattr(pkg, name), name
    assert cake_package("cake_kda_train") is legacy_pkg
    assert cake_package("cake_kda_det_train") is det_pkg
    assert cake_package("cake.cake_kda_train") is pkg
    assert legacy_pkg is not pkg and det_pkg is not pkg


# --------------------------------------------------------------------------- the dispatch sweep (CPU)

SWEEP_ROWS = ((4096, 1, 148), (65536, 300, 160))  # (S, num_docs, sm_count)
PAIR_STAGES = ("dqkg", "glm_intra", "gate_epilogue")


def _serves(name: str, l2n: str) -> bool:
    """Whether record ``name`` serves the in-kernel Q/K L2 norm mode ``l2n``: a record of an l2n stage serves the
    modes its ``l2n`` lists (``["off"]`` when absent), every other record every mode."""
    record = cake_jit.MODULES[name]
    return record["stage"] not in cake_jit.L2N_STAGES or l2n in record.get(
        "l2n", ["off"]
    )


def _admitted(stage: str, gate: str = "g", l2n: str = "off", **scalars) -> list[str]:
    """Every record of ``stage`` of the gate class ``gate`` serving the mode ``l2n`` whose rule admits the scalars,
    the ``otherwise`` fallback included."""
    num_segments = cake_jit.segment_plan(
        scalars["num_seqs"], scalars["num_heads"], scalars["sm_count"], scalars["arch"]
    )[2]
    return [
        name
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, l2n)
        and cake_jit._admits(
            cake_jit.MODULES[name]["select_rule"],
            ctas=scalars["num_chunks_pad"] * scalars["num_heads"],
            num_seqs=scalars["num_seqs"],
            num_heads=scalars["num_heads"],
            sm_count=scalars["sm_count"],
            num_segments=num_segments,
            arch=scalars["arch"],
            path=scalars["path"],
            l2n=l2n,
        )
    ]


def test_the_pairs_agree_on_every_head_count() -> None:
    """Over architecture x path x heads 1..128 (two packings each): the plan's record of every stage is the
    registry's, the only one whose rule admits the problem (or the stage's fallback); on every backward path the
    fused gradient and the GLM intra program are both S1 records or neither, exactly on the cells of the independent
    table; every g-class record of the pair stages runs on architectures it ships for, and every program of the pair
    stages (the S1 records, the deterministic copies and the raw class, which shares four fused gradient programs
    with the g class) runs on exactly the architectures it ships for, in g or raw mode."""
    used: dict[str, set] = {}
    raw_used: dict[str, set] = {}
    for arch in cake_jit.ARCHES:
        for H in range(1, 129):
            for S, N, sm in SWEEP_ROWS:
                p = cake_backend.plan(
                    S, N, H, sm, arch
                )  # the plan's pairing guard runs here
                for path, by_stage in p.records.items():
                    for stage, name in by_stage.items():
                        scalars = dict(
                            num_chunks_pad=p.NC_pad,
                            num_seqs=N,
                            num_heads=H,
                            sm_count=sm,
                            arch=arch,
                            path=path,
                        )
                        assert cake_jit.select_record(stage, **scalars) == name
                        admitted = _admitted(stage, **scalars)
                        rules = [
                            cake_jit.MODULES[n]["select_rule"]["kind"] for n in admitted
                        ]
                        positive = [
                            n
                            for n, k in zip(admitted, rules, strict=True)
                            if k != "otherwise"
                        ]
                        if positive:
                            assert positive == [name], (
                                stage,
                                arch,
                                path,
                                H,
                                N,
                                sm,
                                admitted,
                            )
                        else:
                            assert admitted == ([name] if name else []), (
                                stage,
                                arch,
                                path,
                                H,
                            )
                        if name is not None and stage in PAIR_STAGES:
                            assert arch in cake_jit.MODULES[name]["arches"], (
                                name,
                                arch,
                            )
                            used.setdefault(name, set()).add(arch)
                        raw = (
                            cake_jit.select_record(stage, **scalars, gate="raw")
                            if stage in PAIR_STAGES
                            else None
                        )
                        if raw is not None:
                            assert arch in cake_jit.MODULES[raw]["arches"], (raw, arch)
                            raw_used.setdefault(raw, set()).add(arch)
                    if "dqkg" not in by_stage:
                        continue
                    s1 = _s1(arch, path, H)
                    tag = (arch, path, H, N, sm)
                    assert (by_stage["dqkg"] in S1_RECORDS) is s1, tag
                    assert (by_stage["glm_intra"] in S1_RECORDS) is s1, tag
                    if s1:
                        assert by_stage["glm_intra"] == S1_GLM[arch], tag
                    # the gate epilogue follows either GLM intra program of the unfused architecture
                    assert (by_stage["gate_epilogue"] is not None) is (
                        arch == "sm_100a"
                    ), tag
    # the flag-off classes (the in-kernel Q/K L2 norm's are swept by test_the_l2n_modes_select_the_deterministic_pair)
    for stage in PAIR_STAGES:
        assert {n for n in cake_jit.records_of_stage(stage) if _serves(n, "off")} == {
            name for name in used if cake_jit.MODULES[name]["stage"] == stage
        }, stage
        assert {
            n for n in cake_jit.records_of_stage(stage, "raw") if _serves(n, "off")
        } == {name for name in raw_used if cake_jit.MODULES[name]["stage"] == stage}, (
            stage
        )
    # every program of the pair stages (the S1 records, the deterministic copies and the raw class) runs on every
    # architecture it ships for, in one mode or the other: no shipped build of these stages is unreachable
    reached = {
        (cake_jit.MODULES[name]["library"], arch)
        for runs in (used, raw_used)
        for name, arches in runs.items()
        for arch in arches
    }
    shipped = {
        (cake_jit.MODULES[name]["library"], arch)
        for stage in PAIR_STAGES
        for gate in ("g", "raw")
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, "off")
        for arch in cake_jit.MODULES[name]["arches"]
    }
    assert reached == shipped


def _det_name(name: str | None) -> str | None:
    """The deterministic record a record of this package copies: the fused gradient's raw and l2n classes are the
    deterministic ``dqkg`` records with a class suffix; every other copy keeps its name."""
    if name is not None and name.startswith("dqkg"):
        return name.removesuffix("_raw").removesuffix("_l2n")
    return name


def test_the_l2n_modes_select_the_deterministic_pair() -> None:
    """Over architecture x path x heads 1..128 (two packings each), both gate modes and both in-kernel Q/K L2 norm
    modes (``"fused"``, ``"fallback"``): the plan's records are the deterministic package's plan of the same mode,
    record for record (the fused gradient's raw and l2n classes by their deterministic names), never an S1 record,
    never the gate epilogue; on the l2n stages each is the only record of its mode class whose rule admits the
    problem; and every program of the pair stages serving an l2n mode runs on exactly the architectures it ships
    for."""
    reached: set = set()
    cells = 0
    for arch in cake_jit.ARCHES:
        for H in range(1, 129):
            for S, N, sm in SWEEP_ROWS:
                for gate in ("g", "raw"):
                    for l2n in ("fused", "fallback"):
                        p = cake_backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n)
                        dp = det_backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n)
                        assert (p.gate, p.l2n) == (dp.gate, dp.l2n) == (gate, l2n)
                        assert set(p.records) == set(dp.records)
                        for path, by_stage in p.records.items():
                            tag = (arch, H, N, sm, gate, l2n, path)
                            assert {
                                stage: _det_name(name)
                                for stage, name in by_stage.items()
                            } == dp.records[path], tag
                            assert not set(by_stage.values()) & S1_RECORDS, tag
                            assert by_stage.get("gate_epilogue") is None, tag
                            for stage, name in by_stage.items():
                                if stage not in cake_jit.L2N_STAGES:
                                    continue
                                scalars = dict(
                                    num_chunks_pad=p.NC_pad,
                                    num_seqs=N,
                                    num_heads=H,
                                    sm_count=sm,
                                    arch=arch,
                                    path=path,
                                )
                                assert (
                                    cake_jit.select_record(
                                        stage, **scalars, gate=gate, l2n=l2n
                                    )
                                    == name
                                ), (*tag, stage)
                                admitted = _admitted(stage, gate, l2n, **scalars)
                                positive = [
                                    n
                                    for n in admitted
                                    if cake_jit.MODULES[n]["select_rule"]["kind"]
                                    != "otherwise"
                                ]
                                assert (positive or admitted) == (
                                    [name] if name else []
                                ), (*tag, stage, admitted)
                                if name is not None and stage in PAIR_STAGES:
                                    assert arch in cake_jit.MODULES[name]["arches"]
                                    reached.add(
                                        (cake_jit.MODULES[name]["library"], arch)
                                    )
                            cells += 1
    assert cells > 0
    # the gate epilogue never runs in an l2n mode (its GLM intra programs write the gate gradients themselves)
    shipped = {
        (cake_jit.MODULES[name]["library"], arch)
        for stage in ("dqkg", "glm_intra")
        for gate in ("g", "raw")
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, "fused") or _serves(name, "fallback")
        for arch in cake_jit.MODULES[name]["arches"]
    }
    assert reached == shipped


def test_the_plan_refuses_a_mixed_pair(monkeypatch) -> None:
    """A registry whose fused gradient and GLM intra program disagree on S1 on some path is refused by the plan:
    swapping the rules of the sm_103a GLM intra programs pairs the S1 fused gradient with the deterministic GLM
    intra program on the saved-set paths of a 4-head row."""
    v14, v22 = cake_jit.MODULES["glm_intra_v14"], cake_jit.MODULES["glm_intra_v22"]
    cake_backend._plan.cache_clear()
    try:
        monkeypatch.setitem(v14, "select_rule", v22["select_rule"])
        monkeypatch.setitem(
            v22, "select_rule", cake_jit.MODULES["glm_intra_v18"]["select_rule"]
        )
        with pytest.raises(ValueError, match="S1_RECORDS"):
            cake_backend.plan(4096, 1, 4, 148, "sm_103a")
    finally:
        monkeypatch.undo()
        cake_backend._plan.cache_clear()
    assert (
        cake_backend.plan(4096, 1, 4, 148, "sm_103a").records["light"]["glm_intra"]
        == (S1_GLM["sm_103a"])
    )


# --------------------------------------------------------------------------- the public entries (GPU)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is None
    ):
        pytest.skip("not an exported compute capability")
    for module in (cake_jit, cake_package("cake_kda_det_train", "cake_jit")):
        if not (Path(module.__file__).resolve().parent / "build.json").exists():
            pytest.skip("a package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def _operands(inputs):
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _same(got, want) -> bool:
    return all(_bits_equal(a, b) for a, b in zip(got, want, strict=True))


# a row that runs the S1 pair from its saved set on every architecture (16 heads) and one that runs it on the replay
# of every architecture (2 heads, segmented)
GPU_ROWS = (((129, 63), 16), ((63, 1, 65), 2))


@pytest.mark.parametrize("shape", GPU_ROWS, ids=["h16", "h2"])
def test_public_entries_run_the_packages(shape) -> None:
    """``deterministic=True`` (and the default) gives the deterministic package's bits, ``False`` this package's;
    the two agree bit for bit on ``out`` and every gradient but ``dg``; the workspaces of either mode serve their
    own package and are refused by the other."""
    device = _device_or_skip()
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    q, k, v, g, beta, cu, dout = _operands(
        make_inputs(lengths, heads, device, batched=True)
    )
    det_out, det_saved = det_backend.chunk_kda_det_fwd(
        q, k, v, g, beta, cu, scale=facade.DEFAULT_SCALE
    )
    det_grads = det_backend.chunk_kda_det_bwd(
        q, k, v, g, beta, cu, dout, saved=det_saved, scale=facade.DEFAULT_SCALE
    )
    nd_out, nd_saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, scale=facade.DEFAULT_SCALE, deterministic=False
    )
    nd_grads = cake_backend.chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        cu,
        dout,
        saved=nd_saved,
        scale=facade.DEFAULT_SCALE,
        deterministic=False,
    )
    assert _bits_equal(nd_out, det_out) and _same(nd_grads[:4], det_grads[:4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for mode, out_want, grads_want in (
            ({}, det_out, det_grads),
            ({"deterministic": True}, det_out, det_grads),
            ({"deterministic": False}, nd_out, nd_grads),
        ):
            ws = facade.kda_train_workspace(S, N, heads, device, **mode)
            own = (
                cake_backend.KdaWorkspace
                if mode.get("deterministic") is False
                else det_backend.KdaDetWorkspace
            )
            assert type(ws) is own, mode
            assert ws.bytes == facade.kda_train_workspace_bytes(S, N, heads, **mode)
            for workspace in (None, ws):
                out, saved = facade.chunk_kda_fwd(
                    q, k, v, g, beta, cu, workspace=workspace, **mode
                )
                assert _bits_equal(out, out_want), mode
                grads = facade.chunk_kda_bwd(
                    q, k, v, g, beta, cu, dout, saved=saved, workspace=workspace, **mode
                )
                assert _same(grads, grads_want), mode
            leaves = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)
            ]
            o = facade.chunk_kda(*leaves, cu, **mode)
            assert _bits_equal(o, out_want), mode
            o.backward(dout)
            for leaf, want in zip(
                leaves,
                (
                    grads_want[0],
                    grads_want[1],
                    grads_want[2],
                    grads_want[4],
                    grads_want[3],
                ),
                strict=True,
            ):
                assert _bits_equal(leaf.grad, want), mode
        # a workspace serves its own package: a deterministic=True call refuses this package's, this package
        # refuses the deterministic one's
        nd_ws = facade.kda_train_workspace(S, N, heads, device, deterministic=False)
        det_ws = facade.kda_train_workspace(S, N, heads, device)
        with pytest.raises(TypeError, match="deterministic=False"):
            facade.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=nd_ws)
        with pytest.raises(TypeError, match="KdaWorkspace"):
            facade.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=det_ws, deterministic=False
            )


def test_torch_deterministic_mode_runs_the_deterministic_package(monkeypatch) -> None:
    """Under torch's deterministic mode (``warn_only`` either way) ``deterministic=False`` gives the deterministic
    package's bits, ``dg`` included, from the low-level entries, the autograd entry and a call that brings this
    package's workspace (run on a workspace of its own with the same option); the fallback warns once per process."""
    device = _device_or_skip()
    lengths, heads = (129, 63), 16
    q, k, v, g, beta, cu, dout = _operands(make_inputs(lengths, heads, device))
    out_want, saved = det_backend.chunk_kda_det_fwd(
        q, k, v, g, beta, cu, scale=facade.DEFAULT_SCALE
    )
    grads_want = det_backend.chunk_kda_det_bwd(
        q, k, v, g, beta, cu, dout, saved=saved, scale=facade.DEFAULT_SCALE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        nd_ws = facade.kda_train_workspace(
            sum(lengths), len(lengths), heads, device, deterministic=False
        )
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    for warn_only in (False, True):
        with (
            _TorchDeterministic(warn_only),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            for workspace in (None, nd_ws):
                out, saved = facade.chunk_kda_fwd(
                    q, k, v, g, beta, cu, workspace=workspace, deterministic=False
                )
                assert type(saved) is det_backend.KdaDetChunkSaved
                grads = facade.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=saved,
                    workspace=workspace,
                    deterministic=False,
                )
                assert _bits_equal(out, out_want) and _same(grads, grads_want), (
                    workspace
                )
            leaves = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)
            ]
            o = facade.chunk_kda(*leaves, cu, deterministic=False)
            assert o.grad_fn._forward_cls is det_backend.ChunkKDADetFunction
            o.backward(dout)
            for leaf, want in zip(
                leaves,
                (
                    grads_want[0],
                    grads_want[1],
                    grads_want[2],
                    grads_want[4],
                    grads_want[3],
                ),
                strict=True,
            ):
                assert _bits_equal(leaf.grad, want), warn_only
        fallback = _fallback_warnings(rec)
        assert len(fallback) == (0 if warn_only else 1), [
            str(w.message) for w in fallback
        ]


def test_autograd_records_the_forward_mode() -> None:
    """The autograd node is the forward's package: ``deterministic=False`` gives this package's node, whose backward
    runs this package even when torch's deterministic mode is switched on after the forward; ``True`` gives the
    deterministic package's node, whose backward runs that package."""
    device = _device_or_skip()
    lengths, heads = (129, 63), 16
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = _operands(inputs)
    want = {}
    for mode, fwd, bwd in (
        (False, cake_backend.chunk_kda_fwd, cake_backend.chunk_kda_bwd),
        (True, det_backend.chunk_kda_det_fwd, det_backend.chunk_kda_det_bwd),
    ):
        extra = {} if mode else {"deterministic": False}
        _, saved = fwd(q, k, v, g, beta, cu, scale=facade.DEFAULT_SCALE, **extra)
        want[mode] = bwd(
            q, k, v, g, beta, cu, dout, saved=saved, scale=facade.DEFAULT_SCALE, **extra
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for mode, node in (
            (False, cake_backend.ChunkKDAFunction),
            (True, det_backend.ChunkKDADetFunction),
        ):
            leaves = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)
            ]
            o = facade.chunk_kda(*leaves, cu, deterministic=mode)
            assert o.grad_fn._forward_cls is node, mode
            with (
                _TorchDeterministic(warn_only=True),
                warnings.catch_warnings(record=True) as rec,
            ):
                warnings.simplefilter("always")
                o.backward(dout)
            assert _fallback_warnings(rec) == [], mode
            grads = want[mode]
            for leaf, ref in zip(
                leaves, (grads[0], grads[1], grads[2], grads[4], grads[3]), strict=True
            ):
                assert _bits_equal(leaf.grad, ref), mode


def test_the_backward_reads_a_saved_set_of_either_package() -> None:
    """This package's forward writes a saved set recording ``deterministic=False`` in the deterministic package's
    format: each package's backward reads the other's set and gives the gradients of its own."""
    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    q, k, v, g, beta, cu, dout = _operands(make_inputs(lengths, heads, device))
    scale = facade.DEFAULT_SCALE
    for save_mode in (None, "tape"):
        nd_out, nd_saved = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, scale=scale, save_mode=save_mode, deterministic=False
        )
        det_out, det_saved = det_backend.chunk_kda_det_fwd(
            q, k, v, g, beta, cu, scale=scale, save_mode=save_mode
        )
        assert nd_saved.deterministic is False and _bits_equal(nd_out, det_out)
        assert nd_saved.version == det_saved.version
        assert _bits_equal(nd_saved.Aqk, det_saved.Aqk)
        assert _bits_equal(nd_saved.Akk, det_saved.Akk)
        for name in ("qg", "w", "u", "h", "v_new", "seg_m", "hseg", "seg_state"):
            a, b = getattr(nd_saved, name), getattr(det_saved, name)
            assert (a is None) == (b is None), (save_mode, name)
        own = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=nd_saved, scale=scale, deterministic=False
        )
        other = cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=det_saved,
            scale=scale,
            deterministic=False,
        )
        assert _same(other, own), save_mode
        det_own = det_backend.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=det_saved, scale=scale
        )
        det_other = det_backend.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=nd_saved, scale=scale
        )
        assert _same(det_other, det_own), save_mode


def test_public_entries_run_the_raw_gate_on_both_packages() -> None:
    """The raw gate mode through the facade: ``deterministic=True`` and ``False`` give the same ``out`` and seven
    results bit for bit (explicit calls with and without a workspace of the raw mode, the autograd node: this
    package runs the deterministic package's raw-mode programs), but for ``dA_log`` / ``ddt_bias`` on an R1c row
    (``sm_100a`` / ``sm_103a``: the FP64 parameter-gradient reduce), which hold the R1c bound of the deterministic
    per-chunk partials instead (``test_cake_kda_train._r1c_exact``); each mode's workspace of the raw mode serves
    its package, with the byte query's size; a workspace of the g mode is refused by a raw call (``ValueError``)."""
    from tests.experimental.test_cake_kda_train import (
        RAW_RETURN_NAMES,
        _r1c_exact,
        _r1c_param_check,
    )

    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_raw_inputs(lengths, heads, device)
    q, k, v, raw, beta, cu, dout = _operands(inputs)
    kw = {
        "use_gate_in_kernel": True,
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
    }
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for det in (True, False):
            ws = facade.kda_train_workspace(
                S, N, heads, device, use_gate_in_kernel=True, deterministic=det
            )
            assert ws.gate == "raw", det
            # the arena, and on a non-deterministic workspace of an R1c row its accumulators
            assert ws.bytes + (0 if det else ws.param_acc_bytes) == (
                facade.kda_train_workspace_bytes(
                    S, N, heads, use_gate_in_kernel=True, deterministic=det
                )
            )
            runs = []
            for workspace in (None, ws):
                out, saved = facade.chunk_kda_fwd(
                    q, k, v, raw, beta, cu, workspace=workspace, deterministic=det, **kw
                )
                assert saved.gate == "raw", det
                grads = facade.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    raw,
                    beta,
                    cu,
                    dout,
                    saved=saved,
                    workspace=workspace,
                    deterministic=det,
                    **kw,
                )
                assert len(grads) == 7, det
                runs.append((out, grads))
            leaves = [
                t.detach().clone().requires_grad_(True)
                for t in (q, k, v, raw, beta, inputs["A_log"], inputs["dt_bias"])
            ]
            o = facade.chunk_kda(
                *leaves[:5],
                cu,
                use_gate_in_kernel=True,
                A_log=leaves[5],
                dt_bias=leaves[6],
                lower_bound=LOWER_BOUND,
                deterministic=det,
            )
            o.backward(dout)
            # the leaves' order (q, k, v, raw, beta, A_log, dt_bias) -> the backward's
            gr = [leaf.grad for leaf in leaves]
            runs.append((o.detach(), (gr[0], gr[1], gr[2], gr[4], gr[3], gr[5], gr[6])))
            g_ws = facade.kda_train_workspace(S, N, heads, device, deterministic=det)
            with pytest.raises(ValueError, match="gate mode"):
                facade.chunk_kda_fwd(
                    q, k, v, raw, beta, cu, workspace=g_ws, deterministic=det, **kw
                )
            results[det] = runs
    want_out, want_grads = results[True][0]
    # the R1c bound of this row (None where the device runs the deterministic pair at this head count)
    r1c = _r1c_exact(
        inputs, dict(zip(RAW_RETURN_NAMES[5:], want_grads[5:], strict=True))
    )
    for det, runs in results.items():
        for i, (out, grads) in enumerate(runs):
            assert _bits_equal(out, want_out), (det, i)
            if det or r1c is None:
                assert _same(grads, want_grads), (det, i)
                continue
            assert _same(grads[:5], want_grads[:5]), (det, i)
            _r1c_param_check(
                dict(zip(RAW_RETURN_NAMES[5:], grads[5:], strict=True)),
                r1c,
                tag=(det, i),
            )
