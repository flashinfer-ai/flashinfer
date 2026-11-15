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

flashinfer public-API tests of the Cake deterministic context-parallel (KCP) packed KDA training package
(``flashinfer.experimental.cake.cake_kda_det_train_kcp``), device-free: the keywords the entry points of
``flashinfer.cake_kda_det_train_kcp`` forward, and the #8928 names.  The shared check of #8928
(``tests.cake.test_helpers.cake_mode_names.assert_cake_mode_module``) covers the facade: the mode word right after the
op, no ``deterministic`` keyword, each object bound under one name only (the facade, the ``flashinfer`` top level,
the package and its modules) and the earlier names and package path failing at import; the package states the
mode and takes no ``deterministic`` keyword either.  The three entries that take ``**options`` state the options
in their signatures (keyword-only, the option keys of ``CakeKDADetKcpOptions``), checked against the options.
This file adds what the shared check does not cover: the facade entries are stable (not experimental) and defined
in the facade, the types resolve lazily to the backend classes, the backend modules define the names, errors name
the entry points, and importing the facade or the package loads no host module.
"""

import dataclasses
import importlib
import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_kda_det_train_kcp as kcp_api
from tests.cake.test_helpers.cake_mode_names import (
    assert_cake_mode_module,
    assert_mode_names,
    assert_no_deterministic_keyword,
)


pytestmark = pytest.mark.cpu_only

PACKAGE = "flashinfer.experimental.cake.cake_kda_det_train_kcp"
# the package path before the move under flashinfer/experimental/cake/ (#8928 rule 1): no alias
PREVIOUS_PACKAGE = "flashinfer.experimental.cake_kda_det_train_kcp"

# the entry points of flashinfer.cake_kda_det_train_kcp
FACADE_NAMES = (
    "cake_kda_det_train_kcp_partition",
    "cake_kda_det_train_kcp_build_context",
    "cake_kda_det_kcp",
    "cake_kda_det_train_kcp_forward",
    "cake_kda_det_train_kcp_backward",
    "cake_kda_det_train_kcp_workspace_bytes",
    "cake_kda_det_train_kcp_saved_bytes",
    "cake_kda_det_train_kcp_comm_bytes",
    "cake_kda_det_train_kcp_reduce_gate_param_grads",
)
# the types the facade resolves on first access
TYPE_NAMES = (
    "CakeKDADetKcpContext",
    "CakeKDADetKcpOptions",
    "CakeKDADetKcpPartition",
    "CakeKDADetKcpRecomputeHandle",
    "CakeKDADetKcpSaved",
    "CakeKDADetKcpWorkspace",
)
# (module, det name) of the backend modules' definitions
BACKEND_NAMES = (
    ("cake_kcp", "CakeKDADetKcpFunction"),
    ("cake_kcp", "CakeKDADetKcpCall"),
    ("cake_kcp", "CakeKDADetKcpOptions"),
    ("cake_kcp", "CakeKDADetKcpRecomputeHandle"),
    ("cake_kcp", "CakeKDADetKcpSaved"),
    ("cake_kcp", "CakeKDADetKcpWorkspace"),
    ("cake_kcp", "cake_kda_det_kcp"),
    ("cake_kcp", "cake_kda_det_train_kcp_backward"),
    ("cake_kcp", "cake_kda_det_train_kcp_forward"),
    ("cake_kcp_plan", "CakeKDADetKcpContext"),
    ("cake_kcp_plan", "CakeKDADetKcpFragment"),
    ("cake_kcp_plan", "CakeKDADetKcpPartition"),
    ("cake_kcp_plan", "cake_kda_det_train_kcp_build_context"),
    ("cake_kcp_plan", "cake_kda_det_train_kcp_comm_bytes"),
    ("cake_kcp_plan", "cake_kda_det_train_kcp_partition"),
    ("cake_kcp_plan", "cake_kda_det_train_kcp_saved_bytes"),
    ("cake_kcp_plan", "cake_kda_det_train_kcp_workspace_bytes"),
)
# the names before the det rename: neither the facade nor the package nor its modules bind them
PREVIOUS_NAMES = (
    "ChunkKDAKcpFunction",
    "KcpCall",
    "KcpContext",
    "KcpFragment",
    "KcpOptions",
    "KcpPartition",
    "KcpRecomputeHandle",
    "KcpSaved",
    "KcpWorkspace",
    "build_kcp_context",
    "chunk_kda_kcp",
    "chunk_kda_kcp_bwd",
    "chunk_kda_kcp_fwd",
    "kcp_comm_bytes",
    "kcp_partition",
    "kcp_reduce_gate_param_grads",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
)
# the det names before the #8928 rename (the issue's table "KCP entries: KDA det as the example"): no alias, no shim
DET_NAMES_BEFORE_8928 = (
    "CallDetKcp",
    "ChunkKDADetFunctionKcp",
    "ContextDetKcp",
    "FragmentDetKcp",
    "OptionsDetKcp",
    "PartitionDetKcp",
    "RecomputeHandleDetKcp",
    "SavedDetKcp",
    "WorkspaceDetKcp",
    "build_context_det_kcp",
    "chunk_kda_det_bwd_kcp",
    "chunk_kda_det_fwd_kcp",
    "chunk_kda_det_kcp",
    "kda_det_comm_bytes_kcp",
    "kda_det_saved_bytes_kcp",
    "kda_det_workspace_bytes_kcp",
    "partition_det_kcp",
    "reduce_gate_param_grads_det_kcp",
)
# the backend module each facade entry forwards to (the same name)
ENTRY_BACKEND = {
    "cake_kda_det_train_kcp_partition": "cake_kcp_plan",
    "cake_kda_det_train_kcp_build_context": "cake_kcp_plan",
    "cake_kda_det_kcp": "cake_kcp",
    "cake_kda_det_train_kcp_forward": "cake_kcp",
    "cake_kda_det_train_kcp_backward": "cake_kcp",
    "cake_kda_det_train_kcp_workspace_bytes": "cake_kcp_plan",
    "cake_kda_det_train_kcp_saved_bytes": "cake_kcp_plan",
    "cake_kda_det_train_kcp_comm_bytes": "cake_kcp_plan",
    "cake_kda_det_train_kcp_reduce_gate_param_grads": "cake_kcp",
}
FACADE = "flashinfer.cake_kda_det_train_kcp"
REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULES = ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit", "cake_build")
# the entries that take ``**options`` (validated by the backend's options check); their signatures state the options
OPTIONS_ENTRIES = (
    "cake_kda_det_kcp",
    "cake_kda_det_train_kcp_backward",
    "cake_kda_det_train_kcp_forward",
)
# the package path before the move and the names before #8928 and before the det rename: no alias, no shim
OLD_MODULES = (PREVIOUS_PACKAGE, *(f"{PREVIOUS_PACKAGE}.{m}" for m in SUBMODULES))
OLD_NAMES = tuple(
    (module, old)
    for old in PREVIOUS_NAMES + DET_NAMES_BEFORE_8928
    for module in (
        FACADE,
        PACKAGE,
        *(f"{PACKAGE}.{m}" for m in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit")),
    )
)


def _backend(module: str):
    return importlib.import_module(f"{PACKAGE}.{module}")


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


@pytest.mark.parametrize("new", FACADE_NAMES)
def test_facade_entries_carry_the_det_names(new: str) -> None:
    fn = getattr(kcp_api, new)
    assert fn.__name__ == new and fn.__module__ == "flashinfer.cake_kda_det_train_kcp"
    assert new in kcp_api.__all__


@pytest.mark.parametrize("new", FACADE_NAMES)
def test_facade_entries_are_stable_api(
    new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #8928: the facade is a stable top-level API (flashinfer_api, not flashinfer_experimental_api): no
    # experimental marker, and a call emits no ExperimentalWarning.  Device-free: the backend entry is a recorder.
    fn = getattr(kcp_api, new)
    assert not getattr(fn, "is_experimental", False), new
    assert not hasattr(fn, "experimental_feature"), new
    backend = importlib.import_module(f"{PACKAGE}.{ENTRY_BACKEND[new]}")
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
    module = importlib.import_module(f"{PACKAGE}.{kcp_api._LAZY_TYPES[new]}")
    cls = getattr(module, new)
    assert getattr(kcp_api, new) is cls
    assert cls.__name__ == new and cls.__qualname__ == new
    assert new in kcp_api.__all__
    with pytest.raises(AttributeError):
        _ = kcp_api.NoSuchName  # type: ignore[attr-defined]


@pytest.mark.parametrize("module,new", BACKEND_NAMES)
def test_backend_definitions_carry_the_det_names(module: str, new: str) -> None:
    mod = importlib.import_module(f"{PACKAGE}.{module}")
    obj = getattr(mod, new)
    assert obj.__name__ == new
    assert new in mod.__all__
    package = importlib.import_module(PACKAGE)
    if new in package.__all__:
        assert getattr(package, new) is obj


def test_facade_follows_the_cake_mode_names() -> None:
    # the shared check: the mode word, no ``deterministic`` keyword, each object bound under its facade name only
    # (the facade, the flashinfer top level, the package and the modules it serves its names from, including the
    # backend object each entry calls) and the earlier names and package path failing at import
    assert_cake_mode_module(
        kcp_api,
        "det",
        namespaces=[
            flashinfer,
            importlib.import_module(PACKAGE),
            *(_backend(m) for m in ("cake_kcp", "cake_kcp_plan")),
        ],
        forwards=[
            (name, getattr(_backend(module), name))
            for name, module in ENTRY_BACKEND.items()
        ],
        old_modules=OLD_MODULES,
        old_names=OLD_NAMES,
        cwd=REPO_ROOT,
    )


def test_package_states_the_mode_and_takes_no_deterministic_keyword() -> None:
    package = importlib.import_module(PACKAGE)
    assert_mode_names(package, "det")
    assert_no_deterministic_keyword(package)


@pytest.mark.parametrize("name", OPTIONS_ENTRIES)
def test_option_entries_state_the_option_keys(name: str) -> None:
    # The backend entry takes ``**options``; its signature states them instead: the keyword-only parameters it adds
    # to its own are the option keys (``cake_kcp._OPTION_KEYS``) with the defaults of their ``CakeKDADetKcpOptions``
    # fields, then ``_kcp_test_knobs`` (``None``).  The facade entry, which forwards ``**options``, states the same
    # parameters (names, kinds, defaults).
    cake_kcp = _backend("cake_kcp")
    backend = getattr(cake_kcp, name)
    code = backend.__code__
    own = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    params = inspect.signature(backend).parameters
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())
    added = [n for n in params if n not in own]
    assert added == [*cake_kcp._OPTION_KEYS, "_kcp_test_knobs"]
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in added)
    defaults = {
        f.name: f.default for f in dataclasses.fields(cake_kcp.CakeKDADetKcpOptions)
    }
    assert {n: params[n].default for n in cake_kcp._OPTION_KEYS} == {
        n: defaults[n] for n in cake_kcp._OPTION_KEYS
    }
    assert params["_kcp_test_knobs"].default is None

    def stated(fn):
        ps = inspect.signature(fn).parameters.values()
        keyword_only = inspect.Parameter.KEYWORD_ONLY
        return (
            [(p.name, p.kind, p.default) for p in ps if p.kind is not keyword_only],
            {p.name: p.default for p in ps if p.kind is keyword_only},
        )

    assert stated(getattr(kcp_api, name)) == stated(backend)


def test_errors_name_the_det_entry_points() -> None:
    plan = importlib.import_module(f"{PACKAGE}.cake_kcp_plan")
    with pytest.raises(ValueError, match=r"^cake_kda_det_train_kcp_partition: "):
        plan.cake_kda_det_train_kcp_partition([0, 64], 0)
    with pytest.raises(ValueError, match=r"^cake_kda_det_train_kcp_partition: "):
        plan.cake_kda_det_train_kcp_partition([0, 64], 1, "no-such-layout")


def test_package_names_resolve_on_first_access_only() -> None:
    # Importing the facade or the package imports neither the host module nor the plan.
    code = (
        "import sys, importlib\n"
        "for name in sys.argv[1:]: importlib.import_module(name)\n"
        f"print(sorted(m for m in sys.modules if m.startswith('{PACKAGE}.')))\n"
    )
    for name in ("flashinfer.cake_kda_det_train_kcp", PACKAGE):
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
    from flashinfer.experimental.cake.cake_kda_det_train_kcp import cake_kcp_plan

    calls = []
    monkeypatch.setattr(
        cake_kcp_plan,
        "cake_kda_det_train_kcp_workspace_bytes",
        lambda ctx, **kw: calls.append(kw) or 0,
    )
    ctx = object()
    kcp_api.cake_kda_det_train_kcp_workspace_bytes(ctx, backward=False)
    kcp_api.cake_kda_det_train_kcp_workspace_bytes(
        ctx, backward=False, save_transfer=False
    )
    kcp_api.cake_kda_det_train_kcp_workspace_bytes(ctx)
    assert [kw["save_transfer"] for kw in calls] == [
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
        False,
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
    ]
    assert cake_kcp_plan.SAVE_TRANSFER_DEFAULT is True
    assert (
        inspect.signature(kcp_api.cake_kda_det_train_kcp_workspace_bytes)
        .parameters["save_transfer"]
        .default
        is None
    )


def test_recompute_handle_keywords_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KCP checkpoint recompute handle on the public module (issue #7815): ``cake_kda_det_kcp`` takes
    ``recompute_handle`` / ``checkpoint_keep`` keyword-only and forwards both (``None`` by default);
    ``cake_kda_det_train_kcp_saved_bytes`` / ``cake_kda_det_train_kcp_comm_bytes`` forward ``checkpoint_keep`` (and ``save_transfer=None`` under a tier,
    ``False`` without one); ``CakeKDADetKcpRecomputeHandle`` resolves lazily to
    the KCP module's class.  Device-free (recorders)."""
    from flashinfer.experimental.cake.cake_kda_det_train_kcp import (
        cake_kcp,
        cake_kcp_plan,
    )

    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "cake_kda_det_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_saved_bytes", recorder("saved")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_comm_bytes", recorder("comm")
    )
    operands = tuple(object() for _ in range(6))
    handle, ctx = object(), object()
    assert (
        kcp_api.cake_kda_det_kcp(
            *operands, recompute_handle=handle, checkpoint_keep="transfer"
        )
        == "autograd"
    )
    assert kcp_api.cake_kda_det_kcp(*operands, exchange="ring") == "autograd"
    assert (
        kcp_api.cake_kda_det_train_kcp_saved_bytes(ctx, checkpoint_keep="states")
        == "saved"
    )
    assert (
        kcp_api.cake_kda_det_train_kcp_comm_bytes(ctx, H=4, checkpoint_keep="states")
        == "comm"
    )
    assert kcp_api.cake_kda_det_train_kcp_comm_bytes(ctx, H=4) == "comm"
    base = {
        "scale": None,
        "save_mode": "light",
        "boundary_mode": "saved",
        "workspace": None,
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
    params = inspect.signature(kcp_api.cake_kda_det_kcp).parameters
    for name in ("recompute_handle", "checkpoint_keep"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
    for fn in (
        kcp_api.cake_kda_det_train_kcp_saved_bytes,
        kcp_api.cake_kda_det_train_kcp_comm_bytes,
    ):
        p = inspect.signature(fn).parameters["checkpoint_keep"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert kcp_api.CakeKDADetKcpRecomputeHandle is cake_kcp.CakeKDADetKcpRecomputeHandle
    assert "CakeKDADetKcpRecomputeHandle" in kcp_api.__all__
    with pytest.raises(AttributeError):
        _ = kcp_api.NoSuchName  # type: ignore[attr-defined]


def test_public_module_forwards_the_raw_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw gate mode on the public module (issue #7816): the KCP wrappers forward ``use_gate_in_kernel`` /
    ``A_log`` / ``dt_bias`` / ``lower_bound`` through ``**options`` unchanged, ``cake_kda_det_train_kcp_workspace_bytes`` and
    ``cake_kda_det_train_kcp_saved_bytes`` forward ``use_gate_in_kernel`` (a keyword-only ``bool = False``; the saved query
    reads it for the in-kernel Q / K L2 norm's check) and ``cake_kda_det_train_kcp_reduce_gate_param_grads`` forwards to the KCP
    module's helper,
    which the package lists in ``__all__``.  Device-free (recorders)."""
    from flashinfer.experimental.cake.cake_kda_det_train_kcp import (
        cake_kcp,
        cake_kcp_plan,
    )

    package = importlib.import_module(PACKAGE)
    assert "cake_kda_det_train_kcp_reduce_gate_param_grads" in package.__all__
    assert (
        package.cake_kda_det_train_kcp_reduce_gate_param_grads
        is cake_kcp.cake_kda_det_train_kcp_reduce_gate_param_grads
    )
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "cake_kda_det_train_kcp_forward", recorder("fwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_det_train_kcp_backward", recorder("bwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_det_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp, "cake_kda_det_train_kcp_reduce_gate_param_grads", recorder("reduce")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_workspace_bytes", recorder("bytes")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_saved_bytes", recorder("saved")
    )
    p = inspect.signature(kcp_api.cake_kda_det_train_kcp_saved_bytes).parameters[
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
    assert kcp_api.cake_kda_det_train_kcp_forward(*operands, **kw) == "fwd"
    assert kcp_api.cake_kda_det_train_kcp_backward(*operands, object(), **kw) == "bwd"
    assert kcp_api.cake_kda_det_kcp(*operands, **kw) == "autograd"
    assert (
        kcp_api.cake_kda_det_train_kcp_workspace_bytes(
            object(), use_gate_in_kernel=True
        )
        == "bytes"
    )
    assert (
        kcp_api.cake_kda_det_train_kcp_reduce_gate_param_grads(*reduce_args) == "reduce"
    )
    for gate in (True, False):
        assert (
            kcp_api.cake_kda_det_train_kcp_saved_bytes(
                object(), use_gate_in_kernel=gate
            )
            == "saved"
        )
    assert kcp_api.cake_kda_det_train_kcp_saved_bytes(object()) == "saved"
    for name, _, got in calls[:3]:
        assert all(got[k] is v for k, v in kw.items()), name
    assert calls[3][2]["use_gate_in_kernel"] is True
    assert calls[4] == ("reduce", reduce_args, {})
    assert [c[2]["use_gate_in_kernel"] for c in calls[5:]] == [True, False, False]
    assert "cake_kda_det_train_kcp_reduce_gate_param_grads" in kcp_api.__all__


def test_public_module_forwards_the_qk_l2norm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-kernel Q / K L2 norm on the public module (issue #7817): ``use_qk_l2norm_in_kernel`` is a keyword-only
    ``bool = False`` of the three KCP wrappers and the workspace / saved byte queries, and each forwards it to the
    KCP module unchanged.  Device-free (recorders)."""
    from flashinfer.experimental.cake.cake_kda_det_train_kcp import (
        cake_kcp,
        cake_kcp_plan,
    )

    for name in (
        "cake_kda_det_train_kcp_forward",
        "cake_kda_det_train_kcp_backward",
        "cake_kda_det_kcp",
        "cake_kda_det_train_kcp_workspace_bytes",
        "cake_kda_det_train_kcp_saved_bytes",
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

    monkeypatch.setattr(cake_kcp, "cake_kda_det_train_kcp_forward", recorder("fwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_det_train_kcp_backward", recorder("bwd"))
    monkeypatch.setattr(cake_kcp, "cake_kda_det_kcp", recorder("autograd"))
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_workspace_bytes", recorder("ws")
    )
    monkeypatch.setattr(
        cake_kcp_plan, "cake_kda_det_train_kcp_saved_bytes", recorder("saved")
    )
    operands = tuple(object() for _ in range(6))
    kw = {"use_qk_l2norm_in_kernel": True}
    assert kcp_api.cake_kda_det_train_kcp_forward(*operands, **kw) == "fwd"
    assert kcp_api.cake_kda_det_train_kcp_backward(*operands, object(), **kw) == "bwd"
    assert kcp_api.cake_kda_det_kcp(*operands, **kw) == "autograd"
    assert kcp_api.cake_kda_det_train_kcp_workspace_bytes(object(), **kw) == "ws"
    assert kcp_api.cake_kda_det_train_kcp_saved_bytes(object(), **kw) == "saved"
    assert [c[0] for c in calls] == ["fwd", "bwd", "autograd", "ws", "saved"]
    for name, _, got in calls:
        assert got["use_qk_l2norm_in_kernel"] is True, name
