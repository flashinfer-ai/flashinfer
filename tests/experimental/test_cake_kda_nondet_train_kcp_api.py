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

flashinfer public-API tests of the Cake non-deterministic context-parallel (KCP) packed KDA training package
(``flashinfer.experimental.cake.cake_kda_nondet_train_kcp``) and of its facade ``flashinfer.cake_kda_nondet_train_kcp``,
device-free.  The facade hands every call to this package with the caller's arguments, also under
``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only=True``) and without a warning of its own; nothing
runs the deterministic package (``flashinfer.cake_kda_det_train_kcp``).  The facade entries are stable (not
experimental) and defined in the facade, its types resolve lazily to the backend classes, and the three entries that
take ``**options`` state the options in their signatures (keyword-only, the option keys of
``CakeKDANondetKcpOptions``), on the package and on the facade alike.  The package refuses a workspace and a saved set
of the deterministic package with ``ValueError`` naming the mode, and its entries raise no alert under torch's
deterministic mode.  A context holds no mode, and each package keeps its own table sets on a shared context.  The
package reaches its single-rank sibling ``flashinfer.experimental.cake.cake_kda_nondet_train`` through
``_single_rank`` (the ``common.requires.packages`` of its manifests) and never imports a deterministic package.  The
raw gate mode and the in-kernel q / k L2 norm are keywords of its entries and byte queries, as on the deterministic
package; the local stages dispatch over architecture x head count.  The names (the mode word, no ``deterministic``
keyword, no alias of an earlier name) are ``test_cake_kda_nondet_train_kcp_names.py``'s.
"""

import ast
import dataclasses
import importlib
import inspect
import json
import pathlib
import subprocess
import sys
import types
import warnings

import pytest
import torch

import flashinfer.cake_kda_nondet_train_kcp as kcp_api
from tests.experimental.test_cake_kda_nondet_train_kcp import (
    SAVED_MODE_RE,
    WORKSPACE_MODE_RE,
)

pytestmark = pytest.mark.cpu_only

PACKAGE = "flashinfer.experimental.cake.cake_kda_nondet_train_kcp"
DET_PACKAGE = "flashinfer.experimental.cake.cake_kda_det_train_kcp"
LOCAL_PACKAGE = "flashinfer.experimental.cake.cake_kda_nondet_train"
FACADE = "flashinfer.cake_kda_nondet_train_kcp"
SUBMODULES = ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit", "cake_build")

# facade entry -> the package module that defines the entry of the same name it calls
ENTRY_BACKEND = {
    "cake_kda_nondet_train_kcp_partition": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_build_context": "cake_kcp_plan",
    "cake_kda_nondet_kcp": "cake_kcp",
    "cake_kda_nondet_train_kcp_forward": "cake_kcp",
    "cake_kda_nondet_train_kcp_backward": "cake_kcp",
    "cake_kda_nondet_train_kcp_workspace_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_saved_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_comm_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_reduce_gate_param_grads": "cake_kcp",
    "cake_kda_nondet_train_kcp_handoff_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_handoff_stats": "cake_kcp",
}
# the types the facade resolves on first access
TYPE_NAMES = (
    "CakeKDANondetKcpContext",
    "CakeKDANondetKcpOptions",
    "CakeKDANondetKcpPartition",
    "CakeKDANondetKcpRecomputeHandle",
    "CakeKDANondetKcpSaved",
    "CakeKDANondetKcpWorkspace",
)
# (module, name) of the package's names: the package exports each of them from that module
BACKEND_NAMES = (
    ("cake_kcp", "CakeKDANondetKcpCall"),
    ("cake_kcp", "CakeKDANondetKcpFunction"),
    ("cake_kcp", "CakeKDANondetKcpOptions"),
    ("cake_kcp", "CakeKDANondetKcpRecomputeHandle"),
    ("cake_kcp", "CakeKDANondetKcpSaved"),
    ("cake_kcp", "CakeKDANondetKcpWorkspace"),
    ("cake_kcp", "cake_kda_nondet_kcp"),
    ("cake_kcp", "cake_kda_nondet_train_kcp_backward"),
    ("cake_kcp", "cake_kda_nondet_train_kcp_forward"),
    ("cake_kcp", "cake_kda_nondet_train_kcp_handoff_stats"),
    ("cake_kcp", "cake_kda_nondet_train_kcp_reduce_gate_param_grads"),
    ("cake_kcp_plan", "CakeKDANondetKcpContext"),
    ("cake_kcp_plan", "CakeKDANondetKcpFragment"),
    ("cake_kcp_plan", "CakeKDANondetKcpPartition"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_build_context"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_comm_bytes"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_handoff_bytes"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_partition"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_saved_bytes"),
    ("cake_kcp_plan", "cake_kda_nondet_train_kcp_workspace_bytes"),
)
# the entries that take ``**options`` (validated by the package's options check); their signatures state the options
COMPUTE_ENTRIES = (
    "cake_kda_nondet_kcp",
    "cake_kda_nondet_train_kcp_backward",
    "cake_kda_nondet_train_kcp_forward",
)
# the deterministic package's entry of the same role (its objects must never run behind this facade)
DET_NAMES = {
    entry: entry.replace("_nondet_", "_det_")
    for entry in ENTRY_BACKEND
    if "handoff" not in entry  # the deterministic package has no checkpoint hand-off
}
MODE_CU = [0, 100, 740, 817]


def _mod(sub: str | None = None):
    return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


def _det(sub: str):
    return importlib.import_module(f"{DET_PACKAGE}.{sub}")


def _required_arguments(fn) -> tuple[list[object], dict[str, object]]:
    """Placeholder objects for the parameters of ``fn`` without a default (positional, then keyword-only)."""
    args, kwargs = [], {}
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty or p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[p.name] = object()
        else:
            args.append(object())
    return args, kwargs


@pytest.fixture
def torch_deterministic():
    """Restore torch's deterministic-algorithms setting (and its warn-only flag) after the test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _recorders(monkeypatch) -> list:
    """Replace every entry of this package that the facade calls by a recorder returning ``nd:<name>``, and every
    entry of the deterministic package of the same role by one returning ``det:<name>`` (none may run); the calls
    ``(tag, args, kwargs)`` land in the returned list."""
    seen: list = []

    def recorder(tag):
        def fn(*args, **kwargs):
            seen.append((tag, args, kwargs))
            return tag

        return fn

    for entry, sub in ENTRY_BACKEND.items():
        monkeypatch.setattr(_mod(sub), entry, recorder(f"nd:{entry}"))
        if entry in DET_NAMES:
            monkeypatch.setattr(_det(sub), DET_NAMES[entry], recorder(f"det:{entry}"))
    return seen


# ----------------------------------------------------------------------------------------------------------------
# the facade


@pytest.mark.parametrize("new", sorted(ENTRY_BACKEND))
def test_facade_entries_carry_the_nondet_names(new: str) -> None:
    fn = getattr(kcp_api, new)
    assert fn.__name__ == new and fn.__module__ == FACADE
    assert new in kcp_api.__all__


def test_facade_and_package_export_their_names() -> None:
    """The facade exports ``DEFAULT_SCALE`` (the single-rank facade's), its entries and its types; the package its
    types and entries, each served from the module that defines it."""
    assert sorted(kcp_api.__all__) == sorted(
        ["DEFAULT_SCALE", *ENTRY_BACKEND, *TYPE_NAMES]
    )
    single_rank = importlib.import_module("flashinfer.cake_kda_nondet_train")
    assert kcp_api.DEFAULT_SCALE == single_rank.DEFAULT_SCALE == 128**-0.5
    assert sorted(_mod().__all__) == sorted(name for _, name in BACKEND_NAMES)


@pytest.mark.parametrize("new", sorted(ENTRY_BACKEND))
def test_facade_entries_are_stable_api(
    new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #8928: the facade is a stable top-level API (flashinfer_api, not flashinfer_experimental_api): no
    # experimental marker, and a call emits no ExperimentalWarning.  Device-free: the backend entry is a recorder.
    fn = getattr(kcp_api, new)
    assert not getattr(fn, "is_experimental", False), new
    assert not hasattr(fn, "experimental_feature"), new
    assert not hasattr(kcp_api, "_FEATURE")
    backend = _mod(ENTRY_BACKEND[new])
    calls = []
    monkeypatch.setattr(backend, new, lambda *a, **k: calls.append(new))
    args, kwargs = _required_arguments(fn)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    assert calls == [new]
    # by class name: the test imports nothing from the experimental-API machinery
    assert not [w for w in seen if w.category.__name__ == "ExperimentalWarning"], new


@pytest.mark.parametrize("new", TYPE_NAMES)
def test_facade_types_resolve_lazily_to_the_backend_classes(new: str) -> None:
    module = _mod(kcp_api._LAZY_TYPES[new])
    cls = getattr(module, new)
    assert getattr(kcp_api, new) is cls
    assert cls.__name__ == new and cls.__qualname__ == new
    assert new in kcp_api.__all__
    with pytest.raises(AttributeError):
        _ = kcp_api.NoSuchName  # type: ignore[attr-defined]


@pytest.mark.parametrize("module,new", BACKEND_NAMES)
def test_backend_definitions_carry_the_nondet_names(module: str, new: str) -> None:
    mod = _mod(module)
    obj = getattr(mod, new)
    assert obj.__name__ == new
    assert new in mod.__all__
    package = _mod()
    assert new in package.__all__ and getattr(package, new) is obj


def test_package_exports_resolve_lazily_to_its_modules() -> None:
    """The package's names resolve on first access to the objects of its own modules."""
    package = _mod()
    for name in package.__all__:
        obj = getattr(package, name)
        assert obj is getattr(_mod(package._SUBMODULE_OF[name]), name), name
        assert "Det" not in name and "_det_" not in name, name


@pytest.mark.parametrize("name", COMPUTE_ENTRIES)
def test_option_entries_state_the_option_keys(name: str) -> None:
    # The backend entry takes ``**options``; its signature states them instead: the keyword-only parameters it adds
    # to its own are the option keys (``cake_kcp._OPTION_KEYS``) with the defaults of their
    # ``CakeKDANondetKcpOptions`` fields, then ``_kcp_test_knobs`` (``None``); ``deterministic`` is none of them.
    # The facade entry, which forwards ``**options``, states the same parameters (names, kinds, defaults).
    cake_kcp = _mod("cake_kcp")
    backend = getattr(cake_kcp, name)
    code = backend.__code__
    own = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    params = inspect.signature(backend).parameters
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())
    added = [n for n in params if n not in own]
    assert added == [*cake_kcp._OPTION_KEYS, "_kcp_test_knobs"]
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in added)
    defaults = {
        f.name: f.default for f in dataclasses.fields(cake_kcp.CakeKDANondetKcpOptions)
    }
    assert {n: params[n].default for n in cake_kcp._OPTION_KEYS} == {
        n: defaults[n] for n in cake_kcp._OPTION_KEYS
    }
    assert params["_kcp_test_knobs"].default is None
    assert "deterministic" not in params
    assert "deterministic" not in own

    def stated(fn):
        ps = inspect.signature(fn).parameters.values()
        keyword_only = inspect.Parameter.KEYWORD_ONLY
        return (
            [(p.name, p.kind, p.default) for p in ps if p.kind is not keyword_only],
            {p.name: p.default for p in ps if p.kind is keyword_only},
        )

    assert stated(getattr(kcp_api, name)) == stated(backend)


def _facade_calls() -> dict:
    """``(args, kwargs)`` of one call of every facade entry with stand-in arguments (the recorders read nothing)."""
    t = torch.zeros(1)
    ctx = object()
    return {
        "cake_kda_nondet_train_kcp_partition": (([0, 4], 1), {}),
        "cake_kda_nondet_train_kcp_build_context": ((object(), None, 0, "cpu"), {}),
        "cake_kda_nondet_kcp": ((t, t, t, t, t, ctx), {}),
        "cake_kda_nondet_train_kcp_forward": ((t, t, t, t, t, ctx), {}),
        "cake_kda_nondet_train_kcp_backward": ((t, t, t, t, t, ctx, t), {}),
        "cake_kda_nondet_train_kcp_workspace_bytes": ((ctx,), {"H": 4}),
        "cake_kda_nondet_train_kcp_saved_bytes": ((ctx,), {"H": 4}),
        "cake_kda_nondet_train_kcp_comm_bytes": ((ctx,), {"H": 4}),
        "cake_kda_nondet_train_kcp_reduce_gate_param_grads": ((ctx, t, t), {}),
        "cake_kda_nondet_train_kcp_handoff_bytes": ((ctx,), {"H": 4}),
        "cake_kda_nondet_train_kcp_handoff_stats": ((), {}),
    }


def _check_forwarded(seen: list) -> None:
    """Every facade entry, called once, ran this package's entry of the same name, and only that."""
    calls = _facade_calls()
    assert set(calls) == set(ENTRY_BACKEND)
    for entry, (args, kwargs) in calls.items():
        seen.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = getattr(kcp_api, entry)(*args, **kwargs)
        assert result == f"nd:{entry}", entry
        assert [tag for tag, _, _ in seen] == [f"nd:{entry}"], (entry, seen)
        assert "deterministic" not in seen[0][2], entry
        assert caught == [], (entry, [str(w.message) for w in caught])


def test_facade_forwards_every_call_to_this_package(monkeypatch) -> None:
    """Every facade entry runs this package's entry of the same name, never the deterministic package's, passes no
    mode and warns nothing."""
    assert not torch.are_deterministic_algorithms_enabled()
    _check_forwarded(_recorders(monkeypatch))


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_facade_runs_this_package_under_torch_deterministic_mode(
    monkeypatch, torch_deterministic, warn_only
) -> None:
    """Rule 8 of #8928: under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only=True``) every facade
    entry still runs this package's entry with the caller's arguments; nothing routes to the deterministic package
    and nothing warns (the facade and the package never read torch's setting)."""
    seen = _recorders(monkeypatch)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    _check_forwarded(seen)


def test_errors_name_the_nondet_entry_points() -> None:
    plan = _mod("cake_kcp_plan")
    with pytest.raises(ValueError, match=r"^cake_kda_nondet_train_kcp_partition: "):
        plan.cake_kda_nondet_train_kcp_partition([0, 64], 0)
    with pytest.raises(ValueError, match=r"^cake_kda_nondet_train_kcp_partition: "):
        plan.cake_kda_nondet_train_kcp_partition([0, 64], 1, "no-such-layout")


def test_package_names_resolve_on_first_access_only() -> None:
    """Importing the facade or the package imports neither the host module nor the plan (nor the single-rank
    package's modules)."""
    code = (
        "import sys, importlib\n"
        "for name in sys.argv[1:]: importlib.import_module(name)\n"
        f"print(sorted(m for m in sys.modules if m.startswith('{PACKAGE}.') or m.startswith('{LOCAL_PACKAGE}.')))\n"
    )
    for name in (FACADE, PACKAGE):
        out = subprocess.run(
            [sys.executable, "-c", code, name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert out == "[]", (name, out)


def test_workspace_query_resolves_save_transfer_like_the_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unset save_transfer resolves to the plan default, as the a2a forward does; an explicit value passes through.
    cake_kcp_plan = _mod("cake_kcp_plan")
    calls = []
    monkeypatch.setattr(
        cake_kcp_plan,
        "cake_kda_nondet_train_kcp_workspace_bytes",
        lambda ctx, **kw: calls.append(kw) or 0,
    )
    ctx = object()
    kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(ctx, backward=False)
    kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(
        ctx, backward=False, save_transfer=False
    )
    kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(ctx)
    assert [kw["save_transfer"] for kw in calls] == [
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
        False,
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
    ]
    assert cake_kcp_plan.SAVE_TRANSFER_DEFAULT is True
    assert (
        inspect.signature(kcp_api.cake_kda_nondet_train_kcp_workspace_bytes)
        .parameters["save_transfer"]
        .default
        is None
    )


def test_recompute_handle_keywords_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KCP checkpoint recompute handle on the public module (issue #7815): ``cake_kda_nondet_kcp`` takes
    ``recompute_handle`` / ``checkpoint_keep`` keyword-only and forwards both (``None`` by default) beside its
    ``checkpoint_handoff`` (``"auto"`` by default); ``cake_kda_nondet_train_kcp_saved_bytes`` /
    ``cake_kda_nondet_train_kcp_comm_bytes`` forward ``checkpoint_keep`` (and ``save_transfer=None`` under a tier,
    ``False`` without one), the saved-set query also its ``checkpoint_handoff`` (``False`` by default);
    ``CakeKDANondetKcpRecomputeHandle`` resolves lazily to the KCP module's class.  Device-free (recorders)."""
    cake_kcp, cake_kcp_plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_saved_bytes", recorder("saved")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_comm_bytes", recorder("comm")
    )
    operands = tuple(object() for _ in range(6))
    handle, ctx = object(), object()
    assert (
        kcp_api.cake_kda_nondet_kcp(
            *operands, recompute_handle=handle, checkpoint_keep="transfer"
        )
        == "autograd"
    )
    assert kcp_api.cake_kda_nondet_kcp(*operands, exchange="ring") == "autograd"
    assert (
        kcp_api.cake_kda_nondet_train_kcp_saved_bytes(ctx, checkpoint_keep="states")
        == "saved"
    )
    assert (
        kcp_api.cake_kda_nondet_train_kcp_comm_bytes(ctx, H=4, checkpoint_keep="states")
        == "comm"
    )
    assert kcp_api.cake_kda_nondet_train_kcp_comm_bytes(ctx, H=4) == "comm"
    base = {
        "scale": None,
        "save_mode": "light",
        "boundary_mode": "saved",
        "workspace": None,
        "checkpoint_handoff": "auto",
        "saved_rows": "auto",  # forwarded explicitly
        "use_qk_l2norm_in_kernel": False,  # issue #7817: forwarded explicitly
    }
    assert (
        calls
        == [
            (
                "autograd",
                operands,
                {**base, "recompute_handle": handle, "checkpoint_keep": "transfer"},
            ),
            (
                "autograd",
                operands,
                {
                    **base,
                    "recompute_handle": None,
                    "checkpoint_keep": None,
                    "exchange": "ring",
                },
            ),
            (
                "saved",
                (ctx,),
                {
                    "H": 64,
                    "save_mode": "light",
                    "save_transfer": None,
                    "with_conv": True,
                    "C": 8192,
                    "checkpoint_keep": "states",
                    "use_qk_l2norm_in_kernel": False,
                    "use_gate_in_kernel": False,
                    "checkpoint_handoff": False,
                    "saved_rows": "auto",  # forwarded explicitly
                },
            ),
            (
                "comm",
                (ctx,),
                {
                    "H": 4,
                    "C": 8192,
                    "save_transfer": None,  # the tier fixes it (an explicit contradicting value raises in the query)
                    "boundary_mode": "saved",
                    "checkpoint_keep": "states",
                },
            ),
            (
                "comm",
                (ctx,),
                {
                    "H": 4,
                    "C": 8192,
                    "save_transfer": False,  # no tier: the public default, unchanged
                    "boundary_mode": "saved",
                    "checkpoint_keep": None,
                },
            ),
        ]
    )
    params = inspect.signature(kcp_api.cake_kda_nondet_kcp).parameters
    for name in ("recompute_handle", "checkpoint_keep"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
    assert params["checkpoint_handoff"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["checkpoint_handoff"].default == "auto"
    for fn in (
        kcp_api.cake_kda_nondet_train_kcp_saved_bytes,
        kcp_api.cake_kda_nondet_train_kcp_comm_bytes,
    ):
        p = inspect.signature(fn).parameters["checkpoint_keep"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert (
        kcp_api.CakeKDANondetKcpRecomputeHandle
        is cake_kcp.CakeKDANondetKcpRecomputeHandle
    )
    assert "CakeKDANondetKcpRecomputeHandle" in kcp_api.__all__


def test_public_module_forwards_the_raw_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw gate mode on the public module (issue #7816): the KCP wrappers forward ``use_gate_in_kernel`` /
    ``A_log`` / ``dt_bias`` / ``lower_bound`` unchanged (the same objects),
    ``cake_kda_nondet_train_kcp_workspace_bytes`` and ``cake_kda_nondet_train_kcp_saved_bytes`` forward
    ``use_gate_in_kernel`` (a keyword-only ``bool = False``) and ``cake_kda_nondet_train_kcp_reduce_gate_param_grads``
    forwards to the KCP module's helper, which the package lists in ``__all__``.  Device-free (recorders)."""
    cake_kcp, cake_kcp_plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    package = _mod()
    assert "cake_kda_nondet_train_kcp_reduce_gate_param_grads" in package.__all__
    assert (
        package.cake_kda_nondet_train_kcp_reduce_gate_param_grads
        is cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads
    )
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_train_kcp_forward", recorder("fwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_train_kcp_backward", recorder("bwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp,
        "cake_kda_nondet_train_kcp_reduce_gate_param_grads",
        recorder("reduce"),
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_workspace_bytes", recorder("bytes")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_saved_bytes", recorder("saved")
    )
    p = inspect.signature(kcp_api.cake_kda_nondet_train_kcp_saved_bytes).parameters[
        "use_gate_in_kernel"
    ]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    operands = tuple(object() for _ in range(6))
    kw = {
        "use_gate_in_kernel": True,
        "A_log": object(),
        "dt_bias": object(),
        "lower_bound": -5.0,
    }
    reduce_args = (object(), object(), object())
    assert kcp_api.cake_kda_nondet_train_kcp_forward(*operands, **kw) == "fwd"
    assert (
        kcp_api.cake_kda_nondet_train_kcp_backward(*operands, object(), **kw) == "bwd"
    )
    assert kcp_api.cake_kda_nondet_kcp(*operands, **kw) == "autograd"
    assert (
        kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(
            object(), use_gate_in_kernel=True
        )
        == "bytes"
    )
    assert (
        kcp_api.cake_kda_nondet_train_kcp_reduce_gate_param_grads(*reduce_args)
        == "reduce"
    )
    for gate in (True, False):
        assert (
            kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
                object(), use_gate_in_kernel=gate
            )
            == "saved"
        )
    assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(object()) == "saved"
    for name, _, got in calls[:3]:
        assert all(got[k] is v for k, v in kw.items()), name
    assert calls[3][2]["use_gate_in_kernel"] is True
    assert calls[4] == ("reduce", reduce_args, {})
    assert [c[2]["use_gate_in_kernel"] for c in calls[5:]] == [True, False, False]
    assert "cake_kda_nondet_train_kcp_reduce_gate_param_grads" in kcp_api.__all__


def test_public_module_forwards_the_qk_l2norm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-kernel Q / K L2 norm on the public module (issue #7817): ``use_qk_l2norm_in_kernel`` is a keyword-only
    ``bool = False`` of the three KCP wrappers and the workspace / saved byte queries, and each forwards it to the
    KCP module unchanged.  Device-free (recorders)."""
    cake_kcp, cake_kcp_plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    for name in (
        "cake_kda_nondet_train_kcp_forward",
        "cake_kda_nondet_train_kcp_backward",
        "cake_kda_nondet_kcp",
        "cake_kda_nondet_train_kcp_workspace_bytes",
        "cake_kda_nondet_train_kcp_saved_bytes",
    ):
        p = inspect.signature(getattr(kcp_api, name)).parameters[
            "use_qk_l2norm_in_kernel"
        ]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False, name
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_train_kcp_forward", recorder("fwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_train_kcp_backward", recorder("bwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_workspace_bytes", recorder("ws")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_nondet_train_kcp_saved_bytes", recorder("saved")
    )
    operands = tuple(object() for _ in range(6))
    kw = {"use_qk_l2norm_in_kernel": True}
    assert kcp_api.cake_kda_nondet_train_kcp_forward(*operands, **kw) == "fwd"
    assert (
        kcp_api.cake_kda_nondet_train_kcp_backward(*operands, object(), **kw) == "bwd"
    )
    assert kcp_api.cake_kda_nondet_kcp(*operands, **kw) == "autograd"
    assert kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(object(), **kw) == "ws"
    assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(object(), **kw) == "saved"
    assert [c[0] for c in calls] == ["fwd", "bwd", "autograd", "ws", "saved"]
    for name, _, got in calls:
        assert got["use_qk_l2norm_in_kernel"] is True, name


# ----------------------------------------------------------------------------------------------------------------
# the package: torch's deterministic mode and the deterministic package's objects


def _package_call(entry: str, **kw):
    """One package compute entry with stand-in operands: the checks of the call's keywords run before anything
    reads them."""
    kcp = _mod("cake_kcp")
    t = torch.zeros(1)
    args = (t, t, t, t, t, object())
    if entry == "cake_kda_nondet_train_kcp_backward":
        return kcp.cake_kda_nondet_train_kcp_backward(*args, t, **kw)
    return getattr(kcp, entry)(*args, **kw)


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_raise_no_alert_under_torch_deterministic_mode(
    entry, warn_only, torch_deterministic
) -> None:
    """Rule 8 of #8928: every path of the package is fixed-order (no unordered reduction) and no entry reads torch's
    setting, so with ``torch.use_deterministic_algorithms(True)`` (and ``warn_only=True``) a call raises no alert and
    warns nothing: the stand-in operands fail later, on the operand checks, and the same error as with the mode
    off."""
    torch.use_deterministic_algorithms(False)
    with pytest.raises(Exception) as off:  # noqa: B017 - the stand-in operands' error
        _package_call(entry)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        # a warning (the warn-only alert) would surface as the raised exception
        warnings.simplefilter("error")
        with pytest.raises(Exception) as info:  # noqa: B017 - the stand-in operands' error, never the alert
            _package_call(entry)
    message = str(info.value)
    assert "use_deterministic_algorithms" not in message, message
    assert "does not have a deterministic implementation" not in message, message
    assert (type(info.value), message) == (type(off.value), str(off.value))


def _saved_kwargs(mod) -> dict:
    """The fields of a minimal saved set (nothing reads its tensors before the type check)."""
    return dict(
        version=mod.KCP_SAVED_VERSION,
        S_in=torch.zeros(4),
        c1_recv=None,
        M_seg=None,
        T_r=10,
        H=4,
        n_frag=1,
        P=1,
        Hg=4,
        scale=128**-0.5,
        save_mode="light",
        save_transfer=False,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        ctx=None,
    )


def test_package_refuses_the_deterministic_packages_workspace_and_saved_set() -> None:
    """R6 of #8928, host only: the backward of this package refuses a saved set that is not a
    ``CakeKDANondetKcpSaved`` -- the deterministic package's ``CakeKDADetKcpSaved`` (the same version, members and
    metadata) or any other object -- and a ``CakeKDADetKcpWorkspace`` with ``ValueError`` naming the mode, before it
    reads an option, a graph key, an operand or a tensor (the operands here are placeholders), through the facade
    alike; ``_call_workspace`` (the workspace check of every entry point) refuses that workspace alike and keeps
    ``TypeError`` for any other object."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    assert kcp.KCP_SAVED_VERSION == det_kcp.KCP_SAVED_VERSION
    det_saved = det_kcp.CakeKDADetKcpSaved(**_saved_kwargs(det_kcp))
    saved = kcp.CakeKDANondetKcpSaved(**_saved_kwargs(kcp))
    assert det_saved.meta() == saved.meta() and det_saved.flags() == saved.flags()
    det_ws = det_kcp.CakeKDADetKcpWorkspace()
    t = torch.zeros(1)
    ops = (t, t, t, t, t, object(), t)
    ws_re = WORKSPACE_MODE_RE.format(det_kcp.__name__)
    for backward in (
        kcp.cake_kda_nondet_train_kcp_backward,
        kcp_api.cake_kda_nondet_train_kcp_backward,
    ):
        for graph in ("off", "replay"):
            for bad, name in (
                (det_saved, f"{det_kcp.__name__}.CakeKDADetKcpSaved"),
                (object(), "builtins.object"),
            ):
                with pytest.raises(
                    ValueError, match=SAVED_MODE_RE.format(name)
                ) as info:
                    backward(*ops, saved=bad, graph=graph)
                assert "a saved set of the deterministic package" in str(info.value)
            for kw in ({"saved": saved}, {}):
                with pytest.raises(ValueError, match=ws_re):
                    backward(*ops, workspace=det_ws, graph=graph, **kw)
    backend = types.SimpleNamespace(dtypes={"state": torch.float32})
    with pytest.raises(ValueError, match=ws_re):
        kcp._call_workspace(det_ws, backend, "light", "g", False, "cpu")
    with pytest.raises(
        TypeError, match="workspace must be a CakeKDANondetKcpWorkspace"
    ):
        kcp._call_workspace(object(), backend, "light", "g", False, "cpu")


# ----------------------------------------------------------------------------------------------------------------
# the raw gate mode (#7816) and the in-kernel q / k L2 norm (#7817)

# each keyword at its off value
MODE_OFF = {
    "use_gate_in_kernel": False,
    "A_log": None,
    "dt_bias": None,
    "lower_bound": None,
    "use_qk_l2norm_in_kernel": False,
}
RAW_KEYWORDS = ("use_gate_in_kernel", "A_log", "dt_bias", "lower_bound")
# the package's refusals of the raw gate and of the in-kernel q / k L2 norm before it ran the modes (no message may
# carry them now)
RAW_GATE_REFUSAL = "runs on the deterministic package only"
QK_NORM = "in-kernel q / k L2 norm runs on the deterministic package only"
DET_COMPUTE_NAMES = {
    "cake_kda_nondet_kcp": "cake_kda_det_kcp",
    "cake_kda_nondet_train_kcp_forward": "cake_kda_det_train_kcp_forward",
    "cake_kda_nondet_train_kcp_backward": "cake_kda_det_train_kcp_backward",
}


def _raw_gate_keywords() -> dict:
    return {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(4),
        "dt_bias": torch.zeros(4, 128),
        "lower_bound": -5.0,
    }


def _cpu_ctx(plan, mode: str):
    """A rank-0 context of ``MODE_CU`` over two ranks, built by ``plan``, the plan module of the package of ``mode``
    (``"nondet"`` this one, ``"det"`` the deterministic one)."""
    part = getattr(plan, f"cake_kda_{mode}_train_kcp_partition")(MODE_CU, 2)
    build = getattr(plan, f"cake_kda_{mode}_train_kcp_build_context")
    return build(part, None, 0, "cpu", sm_count=148)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_take_the_raw_gate_keywords(entry) -> None:
    """The raw gate mode runs on this package: every compute entry takes ``use_gate_in_kernel`` / ``A_log`` /
    ``dt_bias`` / ``lower_bound`` as keyword-only parameters with the deterministic entry's defaults (``False``,
    ``None``), so they never reach the options (whose validator does not know them).  A raw call with stand-in
    operands fails on the operand checks, never with a refusal of the mode or an unknown keyword; a non-bool mode is
    the single-rank rule's ``ValueError``, raised before any work."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    params = inspect.signature(getattr(kcp, entry)).parameters
    det_params = inspect.signature(
        getattr(det_kcp, DET_COMPUTE_NAMES[entry])
    ).parameters
    for key in RAW_KEYWORDS:
        assert params[key].kind is inspect.Parameter.KEYWORD_ONLY, (entry, key)
        assert params[key].default is det_params[key].default is MODE_OFF[key], (
            entry,
            key,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        kcp._options({"use_gate_in_kernel": True})
    with pytest.raises(Exception) as info:  # noqa: B017 - the stand-in operands' error, never the mode's
        _package_call(entry, **_raw_gate_keywords())
    message = str(info.value)
    for refusal in ("unexpected keyword", RAW_GATE_REFUSAL, QK_NORM):
        assert refusal not in message, message
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        _package_call(entry, use_gate_in_kernel=1)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_take_the_in_kernel_qk_norm(entry) -> None:
    """The in-kernel q / k L2 norm runs on this package: every compute entry takes ``use_qk_l2norm_in_kernel`` as a
    keyword-only parameter with the deterministic entry's default (``False``), so it never reaches the options (whose
    validator refuses it, as the deterministic package's does).  A flag-on call with stand-in operands (also with the
    raw gate requested) passes the capability check (the records serve the mode on every architecture they list) and
    fails on the operand checks (``ValueError`` of the 1-D ``v``), never with a refusal of the mode or an unknown
    keyword; a non-bool value is a ``ValueError`` raised before any work."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    p = inspect.signature(getattr(kcp, entry)).parameters["use_qk_l2norm_in_kernel"]
    dp = inspect.signature(getattr(det_kcp, DET_COMPUTE_NAMES[entry])).parameters[
        "use_qk_l2norm_in_kernel"
    ]
    assert p.kind is dp.kind is inspect.Parameter.KEYWORD_ONLY, entry
    assert p.default is dp.default is False, entry
    for kw in ({}, _raw_gate_keywords()):
        with pytest.raises(ValueError, match=r"^v: expected 3-D") as info:
            _package_call(entry, use_qk_l2norm_in_kernel=True, **kw)
        message = str(info.value)
        for refusal in ("unexpected keyword", RAW_GATE_REFUSAL, QK_NORM):
            assert refusal not in message, message
    for value in (1, 0, None, "true"):
        with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel must be a bool"):
            _package_call(entry, use_qk_l2norm_in_kernel=value)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_accept_the_modes_off(entry) -> None:
    """An entry that spells out every off value fails on the stand-in operands, not on a keyword.
    ``use_qk_l2norm_in_kernel`` is an entry keyword, not an option: the options' validator refuses it at either value
    (``TypeError``), as the deterministic package's does."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    for value in (False, True):
        for mod in (kcp, det_kcp):
            with pytest.raises(TypeError, match="use_qk_l2norm_in_kernel"):
                mod._options({"use_qk_l2norm_in_kernel": value})
    with pytest.raises(Exception) as info:  # noqa: B017 - the stand-in operands' error, never a keyword's
        _package_call(entry, **MODE_OFF)
    message = str(info.value)
    for refusal in ("unexpected keyword", RAW_GATE_REFUSAL, QK_NORM):
        assert refusal not in message, message


def test_package_byte_queries_take_the_raw_gate() -> None:
    """The package's workspace and saved-set byte queries take ``use_gate_in_kernel`` as the deterministic queries
    do: ``False`` gives the bytes of the omitted keyword; ``True`` gives the raw plan's workspace bytes (the
    deterministic query's answer for the same partition, never more than the explicit mode's, the facade's answer)
    and the saved-set bytes of the explicit gate (the saved set does not depend on the gate mode); a non-bool is a
    ``ValueError`` and ``True`` with the lean local arena a ``NotImplementedError``."""
    plan, det_plan = _mod("cake_kcp_plan"), _det("cake_kcp_plan")
    ctx, det_ctx = _cpu_ctx(plan, "nondet"), _cpu_ctx(det_plan, "det")
    assert plan.cake_kda_nondet_train_kcp_workspace_bytes(
        ctx, H=4, use_gate_in_kernel=False
    ) == plan.cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=4)
    for backward in (True, False):
        raw = plan.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=4, backward=backward, use_gate_in_kernel=True
        )
        assert raw == det_plan.cake_kda_det_train_kcp_workspace_bytes(
            det_ctx, H=4, backward=backward, use_gate_in_kernel=True
        ), backward
        assert raw <= plan.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=4, backward=backward
        )
        assert (
            kcp_api.cake_kda_nondet_train_kcp_workspace_bytes(
                ctx, H=4, backward=backward, use_gate_in_kernel=True
            )
            == raw
        ), backward
    saved = plan.cake_kda_nondet_train_kcp_saved_bytes(ctx, H=4)
    for gate in (False, True):
        assert (
            plan.cake_kda_nondet_train_kcp_saved_bytes(
                ctx, H=4, use_gate_in_kernel=gate
            )
            == saved
        )
        assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
            ctx, H=4, use_gate_in_kernel=gate
        ) == kcp_api.cake_kda_nondet_train_kcp_saved_bytes(ctx, H=4), gate
    for query in (
        "cake_kda_nondet_train_kcp_workspace_bytes",
        "cake_kda_nondet_train_kcp_saved_bytes",
    ):
        for value in (1, None, "raw"):
            with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
                getattr(plan, query)(ctx, H=4, use_gate_in_kernel=value)
    with pytest.raises(NotImplementedError, match='local_arena="lean"'):
        plan.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=4, use_gate_in_kernel=True, local_arena="lean"
        )


def test_package_byte_queries_take_the_in_kernel_qk_norm(monkeypatch) -> None:
    """The package's workspace and saved-set byte queries take ``use_qk_l2norm_in_kernel`` as the deterministic queries
    do (keyword-only, ``False`` by default): ``False`` gives the bytes of the omitted keyword; a non-bool is a
    ``ValueError`` before the context is read; with ``True`` both queries run the rank-independent capability check
    first (``NotImplementedError`` with its reason while it fails, on this package's queries and on the facade's
    alike) and, once it passes, the saved set is ``8 T_r H`` bytes (the FP32 rstd) larger.  The communication query
    does not take the keyword."""
    plan, det_plan = _mod("cake_kcp_plan"), _det("cake_kcp_plan")
    ctx = _cpu_ctx(plan, "nondet")
    for query, det_query in (
        (
            "cake_kda_nondet_train_kcp_workspace_bytes",
            "cake_kda_det_train_kcp_workspace_bytes",
        ),
        ("cake_kda_nondet_train_kcp_saved_bytes", "cake_kda_det_train_kcp_saved_bytes"),
    ):
        fn = getattr(plan, query)
        p = inspect.signature(fn).parameters["use_qk_l2norm_in_kernel"]
        dp = inspect.signature(getattr(det_plan, det_query)).parameters[
            "use_qk_l2norm_in_kernel"
        ]
        assert p.kind is dp.kind is inspect.Parameter.KEYWORD_ONLY, query
        assert p.default is dp.default is False, query
        assert fn(ctx, H=4, use_qk_l2norm_in_kernel=False) == fn(ctx, H=4), query
        for value in (1, None, "true"):
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel must be a bool"
            ):
                fn(object(), use_qk_l2norm_in_kernel=value)
        with monkeypatch.context() as m:
            m.setattr(plan, "qk_l2norm_missing", lambda *a, **k: "forced")
            want = f"{plan.QK_L2NORM_PENDING} (forced)"
            with pytest.raises(NotImplementedError) as info:
                fn(ctx, H=4, use_qk_l2norm_in_kernel=True)
            assert str(info.value) == want, query
            with pytest.raises(NotImplementedError) as info:
                getattr(kcp_api, query)(ctx, H=4, use_qk_l2norm_in_kernel=True)
            assert str(info.value) == want, query
    assert (
        "use_qk_l2norm_in_kernel"
        not in inspect.signature(plan.cake_kda_nondet_train_kcp_comm_bytes).parameters
    )
    with monkeypatch.context() as m:
        m.setattr(plan, "qk_l2norm_missing", lambda *a, **k: None)
        for H in (4, 64):
            on = plan.cake_kda_nondet_train_kcp_saved_bytes(
                ctx, H=H, use_qk_l2norm_in_kernel=True
            )
            assert (
                on - plan.cake_kda_nondet_train_kcp_saved_bytes(ctx, H=H)
                == 8 * int(ctx.T_r) * H
                > 0
            )


def test_gate_parameter_grads_reduce_like_the_deterministic_helper() -> None:
    """``cake_kda_nondet_train_kcp_reduce_gate_param_grads`` (facade and package) takes no mode and returns, on one
    rank, the bits of the deterministic package's ``cake_kda_det_train_kcp_reduce_gate_param_grads``."""
    one = type("OneRank", (), {"world_size": 1, "group": None})()
    dA_log = torch.arange(4, dtype=torch.float32)
    ddt_bias = torch.linspace(-1.0, 1.0, 4 * 128).view(4, 128)
    det_res = _det("cake_kcp").cake_kda_det_train_kcp_reduce_gate_param_grads(
        one, dA_log, ddt_bias
    )
    for fn in (
        kcp_api.cake_kda_nondet_train_kcp_reduce_gate_param_grads,
        _mod("cake_kcp").cake_kda_nondet_train_kcp_reduce_gate_param_grads,
    ):
        assert list(inspect.signature(fn).parameters) == ["ctx", "dA_log", "ddt_bias"]
        res = fn(one, dA_log, ddt_bias)
        for a, b in zip(res, det_res, strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


# ----------------------------------------------------------------------------------------------------------------
# the context holds no mode; a context serves both packages


def test_the_context_holds_no_mode() -> None:
    """No field of ``CakeKDANondetKcpContext`` / ``CakeKDANondetKcpPartition`` / ``CakeKDANondetKcpFragment`` names a
    mode, nor any attribute of a built context; the two packages build the same partition (its fragment table) and
    their facades give the same byte answers."""
    plan, det_plan = _mod("cake_kcp_plan"), _det("cake_kcp_plan")
    det_api = importlib.import_module("flashinfer.cake_kda_det_train_kcp")
    part = plan.cake_kda_nondet_train_kcp_partition(MODE_CU, 2, "zigzag")
    ctx = plan.cake_kda_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    names = [n for n in dir(ctx) if not n.startswith("__")]
    names += [f.name for f in dataclasses.fields(plan.CakeKDANondetKcpFragment)]
    names += [f.name for f in dataclasses.fields(plan.CakeKDANondetKcpPartition)]
    names += list(part.to_dict())
    bad = sorted({n for n in names if "determin" in n.lower() or n.lower() == "mode"})
    assert not bad, bad
    det_part = det_plan.cake_kda_det_train_kcp_partition(MODE_CU, 2, "zigzag")
    assert part.to_dict() == det_part.to_dict()
    for rank in range(2):
        nd_ctx = kcp_api.cake_kda_nondet_train_kcp_build_context(
            part, None, rank, "cpu", sm_count=148
        )
        det_ctx = det_plan.cake_kda_det_train_kcp_build_context(
            det_part, None, rank, "cpu", sm_count=148
        )
        assert isinstance(nd_ctx, plan.CakeKDANondetKcpContext)
        for query in ("workspace_bytes", "saved_bytes", "comm_bytes"):
            ours = getattr(kcp_api, f"cake_kda_nondet_train_kcp_{query}")
            theirs = getattr(det_api, f"cake_kda_det_train_kcp_{query}")
            assert ours(nd_ctx, H=4) == theirs(det_ctx, H=4), (rank, query)


def test_each_package_keeps_its_own_table_sets_on_a_shared_context() -> None:
    """A context serves both packages: each package's host module keys its launch tables on the context with keys of
    its own (this package's carry :data:`cake_kcp.TABLE_SET_TAG`), so neither reads a table set of the other's
    module."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    plan = _mod("cake_kcp_plan")
    part = plan.cake_kda_nondet_train_kcp_partition(MODE_CU, 2, "zigzag")
    ctx = plan.cake_kda_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    nd_tb = kcp.CakeKDANondetKcpWorkspace(local=False).tables(ctx, 4)
    det_tb = det_kcp.CakeKDADetKcpWorkspace(local=False).tables(ctx, 4)
    assert type(nd_tb) is kcp._Tables and type(det_tb) is det_kcp._Tables
    assert nd_tb is not det_tb
    keys = list(ctx._table_sets)
    assert len(keys) == 2 and sum(k[-1] == kcp.TABLE_SET_TAG for k in keys) == 1, keys


# ----------------------------------------------------------------------------------------------------------------
# import isolation: the single-rank sibling by name, never a deterministic package


def _package_root() -> pathlib.Path:
    return pathlib.Path(_mod().__file__).resolve().parent


def _package_sources() -> list[pathlib.Path]:
    files = sorted(_package_root().glob("*.py"))
    assert {p.name for p in files} >= {
        "__init__.py",
        "_single_rank.py",
        *(f"{s}.py" for s in SUBMODULES),
    }
    return files


def _absolute(node: ast.ImportFrom) -> str:
    """The absolute module of an ``ImportFrom`` of a module of this package."""
    if not node.level:
        return node.module or ""
    base = PACKAGE.split(".")[: len(PACKAGE.split(".")) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imported_modules(path: pathlib.Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(node)
            out.append((node.lineno, module))
            out += [(node.lineno, f"{module}.{a.name}") for a in node.names]
    return out


def _forbidden(module: str) -> bool:
    """A deterministic package or one of its modules."""
    return any(p.startswith("cake_kda_det_train") for p in module.split("."))


def test_the_package_never_imports_a_deterministic_package() -> None:
    """Statically: no module of the package imports a deterministic package (``cake_kda_det_train`` /
    ``cake_kda_det_train_kcp`` or a module of them), and no import of the package reaches above its parent (no
    ``ImportFrom`` of level two or more).  The single-rank sibling is reached through ``_single_rank`` only: the host
    module, the registry and the plan import ``_single_rank``, whose ``NAME`` is the sibling ``cake_kda_nondet_train``
    under this package's parent -- the one package that both manifests declare (``common.requires.packages``).  The
    facade imports no deterministic package either."""
    users = set()
    for path in _package_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level < 2, (path.name, node.lineno, node.module)
                if node.level == 1 and node.module is None:
                    if any(a.name == "_single_rank" for a in node.names):
                        users.add(path.name)
        for lineno, module in _imported_modules(path):
            assert not _forbidden(module), (path.name, lineno, module)
    assert {"cake_kcp.py", "cake_kcp_jit.py", "cake_kcp_plan.py"} <= users, users
    single_rank = importlib.import_module(f"{PACKAGE}._single_rank")
    assert single_rank.NAME == LOCAL_PACKAGE
    leaf = single_rank.NAME.rpartition(".")[2]
    assert leaf == "cake_kda_nondet_train"
    for manifest in ("build.json", "build.test.json"):
        common = json.loads((_package_root() / manifest).read_text())["common"]
        assert common["requires"]["packages"] == [leaf], manifest
    facade = pathlib.Path(kcp_api.__file__).resolve()
    for node in ast.walk(ast.parse(facade.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            module = "flashinfer" + ("." + node.module if node.module else "")
            assert not _forbidden(module), (node.lineno, module)
            for a in node.names:
                assert not _forbidden(a.name), (node.lineno, module, a.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            names += [a.name for a in node.names]
            assert not any(_forbidden(n) for n in names), (node.lineno, names)


def test_importing_the_package_imports_no_deterministic_module() -> None:
    """At run time, in a fresh interpreter: importing every module of the package, the facade, and resolving the
    plan's function-local single-rank modules loads no module of a deterministic package (no ``cake_kda_det_train``
    in ``sys.modules``) and does load this package's single-rank sibling: ``_single_rank`` resolves to
    ``flashinfer.experimental.cake.cake_kda_nondet_train``, whose ``cake_jit`` / ``cake_backend`` the host module and
    the registry bind."""
    code = (
        "import sys, importlib\n"
        f"importlib.import_module('{FACADE}')\n"
        f"for sub in {SUBMODULES!r}: importlib.import_module('{PACKAGE}.' + sub)\n"
        f"kcp = sys.modules['{PACKAGE}.cake_kcp']\n"
        f"jit = sys.modules['{PACKAGE}.cake_kcp_jit']\n"
        f"plan = sys.modules['{PACKAGE}.cake_kcp_plan']\n"
        f"single_rank = importlib.import_module('{PACKAGE}._single_rank')\n"
        "plan._qk_l2norm_inputs()\n"
        f"assert single_rank.NAME == '{LOCAL_PACKAGE}', single_rank.NAME\n"
        f"assert kcp.cake_jit is jit.cake_jit is sys.modules['{LOCAL_PACKAGE}.cake_jit']\n"
        f"assert kcp.cake_backend is sys.modules['{LOCAL_PACKAGE}.cake_backend']\n"
        "print(sorted(m for m in sys.modules if 'cake_kda_det_train' in m))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert out == "[]", out


def test_the_package_reads_no_environment_variable() -> None:
    """No ``os.environ`` / ``os.getenv`` anywhere in the package or the facade."""
    for path in _package_sources() + [pathlib.Path(kcp_api.__file__).resolve()]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in (
                "environ",
                "getenv",
                "putenv",
            ):
                if path.name == "cake_build.py":
                    continue  # the shared loader (byte copy of every Cake package's) reads its build cache paths
                raise AssertionError(
                    f"{path.name}: reads the environment ({node.attr})"
                )


def test_the_registry_names_no_library_of_the_deterministic_package() -> None:
    """Every record of this package's KCP registry (shipped and test-only) names a ``cake_kda_train_`` library and a
    kernel symbol of its own, none of the deterministic package's."""
    jit, det_jit = _mod("cake_kcp_jit"), _det("cake_kcp_jit")
    ours = {**jit.MODULES, **jit.TEST_MODULES}
    theirs = {**det_jit.MODULES, **det_jit.TEST_MODULES}
    libraries = {r["library"] for r in ours.values()}
    assert libraries and all(n.startswith("cake_kda_train_") for n in libraries)
    assert not libraries & {r["library"] for r in theirs.values()}
    assert not {r["kernel_symbol"] for r in ours.values()} & {
        r["kernel_symbol"] for r in theirs.values()
    }


# ----------------------------------------------------------------------------------------------------------------
# dispatch of the local stages (the single-rank records the KCP launches)

SWEEP_SM = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 216}


def _nd_replay_glm(arch: str, H: int) -> str:
    """The GLM intra record of the single-rank package's replay path (the KCP backward): its ``S1`` form where the
    routing table sends the replay to the ``S1`` pair (``sm_100a`` every head count but 9 to 15, ``sm_103a`` up to 8
    heads, ``sm_107a`` -- provisional, while the registry ships ``glm_intra_v23`` -- every head count but 9 to 15),
    else the deterministic copy."""
    jit = importlib.import_module(f"{LOCAL_PACKAGE}.cake_jit")
    det = {
        "sm_100a": "glm_intra",
        "sm_103a": "glm_intra_v14",
        "sm_107a": "glm_intra_v18",
    }[arch]
    if 9 <= H <= 15:
        return det
    s1 = {
        "sm_100a": "glm_intra_v21",
        "sm_103a": "glm_intra_v22" if H <= 8 else None,
        "sm_107a": "glm_intra_v23" if "glm_intra_v23" in jit.MODULES else None,
    }[arch]
    return s1 or det


@pytest.mark.parametrize("arch", sorted(SWEEP_SM))
def test_local_stage_dispatch_sweep(arch) -> None:
    """For every head count 1..128 the plan of a rank's local problem (the single-rank package's, as the KCP builds
    it) selects on the KCP's paths exactly one record per stage; on the replay path (the KCP backward) the fused
    gradient and the GLM intra program are the ``S1`` pair or the deterministic pair together, the ``S1`` pair on the
    rows of the routing table; the forward path (``fwd_nosave``) launches no ``S1`` record."""
    kcp = _mod("cake_kcp")
    backend = importlib.import_module(f"{LOCAL_PACKAGE}.cake_backend")
    jit = importlib.import_module(f"{LOCAL_PACKAGE}.cake_jit")
    s1 = set(jit.S1_RECORDS)
    for H in range(1, 129):
        plan = backend.plan(3000, 3, H, SWEEP_SM[arch], arch)
        dqkg = plan.record("dqkg", kcp.LOCAL_PATH_BWD)
        glm = plan.record("glm_intra", kcp.LOCAL_PATH_BWD)
        assert dqkg and glm, (H, dqkg, glm)
        assert (dqkg in s1) == (glm in s1), (arch, H, dqkg, glm)
        assert glm == _nd_replay_glm(arch, H), (arch, H, glm)
        for stage in kcp.LOCAL_STAGES_FWD:
            if stage in kcp.KCP_LOCAL_RECORDS:
                continue
            if stage in kcp.WY_DIRECT_RECORDS and not plan.dispatched(
                stage, kcp.LOCAL_PATH_FWD
            ):
                continue
            assert plan.record(stage, kcp.LOCAL_PATH_FWD) not in s1, (arch, H, stage)
