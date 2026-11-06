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
``flashinfer.cake_kda_det_train_kcp`` forward, the #8928 names of the facade, the package and its modules (the mode
word, one prefix, no ``deterministic`` keyword), and the absence of the earlier names and package path.
"""

import importlib
import inspect
import re
import subprocess
import sys
import warnings

import pytest

import flashinfer.cake_kda_det_train as det_api
import flashinfer.cake_kda_det_train_kcp as kcp_api
from flashinfer.api_logging import ExperimentalWarning

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
# #8928 rule 7: every public name carries the mode; ALL_CAPS module constants keep their names
CONSTANT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# rules 2-6 and 10: the entries and types of the package share one prefix
MODE_NAME = re.compile(
    r"^(?:cake_kda_det_kcp|cake_kda_det_train_kcp_[a-z0-9_]+|is_cake_kda_det_train_kcp_supported"
    r"|CakeKDADetKcp[A-Za-z0-9]+)$"
)


@pytest.mark.parametrize("new", FACADE_NAMES)
def test_facade_entries_carry_the_det_names(new: str) -> None:
    fn = getattr(kcp_api, new)
    assert fn.__name__ == new and fn.__module__ == "flashinfer.cake_kda_det_train_kcp"
    assert fn.experimental_feature == det_api._FEATURE
    assert new in kcp_api.__all__


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


@pytest.mark.parametrize("old", PREVIOUS_NAMES + DET_NAMES_BEFORE_8928)
def test_previous_names_are_gone(old: str) -> None:
    package = importlib.import_module(PACKAGE)
    modules = [kcp_api, package] + [
        importlib.import_module(f"{PACKAGE}.{sub}")
        for sub in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit")
    ]
    for mod in modules:
        assert not hasattr(mod, old), (mod.__name__, old)
        assert old not in getattr(mod, "__all__", ()), (mod.__name__, old)
    assert old not in dir(kcp_api) and old not in dir(package)
    for name in ("flashinfer.cake_kda_det_train_kcp", PACKAGE):
        with pytest.raises(ImportError):
            exec(f"from {name} import {old}", {})


@pytest.mark.parametrize(
    "module", ("", ".cake_kcp", ".cake_kcp_plan", ".cake_kcp_jit", ".cake_build")
)
def test_previous_package_path_is_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{PREVIOUS_PACKAGE}{module}")


@pytest.mark.parametrize("where", ("facade", "package"))
def test_public_names_carry_the_mode(where: str) -> None:
    # #8928 rules 2-7 and 10: every public name of the facade and of the backend package carries the mode and the
    # package's prefix; ALL_CAPS constants keep their names
    mod = kcp_api if where == "facade" else importlib.import_module(PACKAGE)
    names = [name for name in mod.__all__ if not CONSTANT_NAME.match(name)]
    assert names
    for name in names:
        assert re.search(r"(?:^|_)det(?:_|$)|Det", name), name
        assert MODE_NAME.match(name), name


@pytest.mark.parametrize("where", ("facade", "package"))
def test_no_entry_takes_deterministic(where: str) -> None:
    # #8928 rule 7: the mode is part of the name only
    import torch

    mod = kcp_api if where == "facade" else importlib.import_module(PACKAGE)
    checked = 0
    for name in mod.__all__:
        obj = getattr(mod, name)
        if not callable(obj):
            continue
        if inspect.isclass(obj):
            obj = (
                obj.forward
                if issubclass(obj, torch.autograd.Function)
                else obj.__init__
            )
        assert "deterministic" not in inspect.signature(obj).parameters, name
        checked += 1
    assert checked
    # the entry points' **options refuse the keyword like any unknown one
    cake_kcp = importlib.import_module(f"{PACKAGE}.cake_kcp")
    with pytest.raises(TypeError, match="deterministic"):
        cake_kcp._options({"deterministic": True})


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
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
            kcp_api.cake_kda_det_train_kcp_comm_bytes(
                ctx, H=4, checkpoint_keep="states"
            )
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert kcp_api.cake_kda_det_train_kcp_forward(*operands, **kw) == "fwd"
        assert (
            kcp_api.cake_kda_det_train_kcp_backward(*operands, object(), **kw) == "bwd"
        )
        assert kcp_api.cake_kda_det_kcp(*operands, **kw) == "autograd"
        assert (
            kcp_api.cake_kda_det_train_kcp_workspace_bytes(
                object(), use_gate_in_kernel=True
            )
            == "bytes"
        )
        assert (
            kcp_api.cake_kda_det_train_kcp_reduce_gate_param_grads(*reduce_args)
            == "reduce"
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert kcp_api.cake_kda_det_train_kcp_forward(*operands, **kw) == "fwd"
        assert (
            kcp_api.cake_kda_det_train_kcp_backward(*operands, object(), **kw) == "bwd"
        )
        assert kcp_api.cake_kda_det_kcp(*operands, **kw) == "autograd"
        assert kcp_api.cake_kda_det_train_kcp_workspace_bytes(object(), **kw) == "ws"
        assert kcp_api.cake_kda_det_train_kcp_saved_bytes(object(), **kw) == "saved"
    assert [c[0] for c in calls] == ["fwd", "bwd", "autograd", "ws", "saved"]
    for name, _, got in calls:
        assert got["use_qk_l2norm_in_kernel"] is True, name
