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
(``flashinfer.experimental.cake_kda_det_train_kcp``), device-free: the keywords the entry points of
``flashinfer.cake_kda_det_train_kcp`` forward, and the names before the det rename -- in ``flashinfer.kda_train``,
in the modules ``cake_kcp`` / ``cake_kcp_plan`` and at the previous module paths -- bound to the same objects.
"""

import importlib
import inspect
import subprocess
import sys
import warnings

import pytest

import flashinfer.cake_kda_det_train as det_api
import flashinfer.cake_kda_det_train_kcp as kcp_api
import flashinfer.kda_train as kda_train
from flashinfer.api_logging import ExperimentalWarning

pytestmark = pytest.mark.cpu_only

PACKAGE = "flashinfer.experimental.cake_kda_det_train_kcp"
OLD_PACKAGE = "flashinfer.experimental.cake_kda_train"

# (previous name, det name) of the entry points: flashinfer.kda_train -> flashinfer.cake_kda_det_train_kcp
FACADE_PAIRS = (
    ("kcp_partition", "partition_det_kcp"),
    ("build_kcp_context", "build_context_det_kcp"),
    ("chunk_kda_kcp", "chunk_kda_det_kcp"),
    ("chunk_kda_kcp_fwd", "chunk_kda_det_fwd_kcp"),
    ("chunk_kda_kcp_bwd", "chunk_kda_det_bwd_kcp"),
    ("kcp_workspace_bytes", "kda_det_workspace_bytes_kcp"),
    ("kcp_saved_bytes", "kda_det_saved_bytes_kcp"),
    ("kcp_comm_bytes", "kda_det_comm_bytes_kcp"),
    ("kcp_reduce_gate_param_grads", "reduce_gate_param_grads_det_kcp"),
)
# (previous name, det name) of the types the facades resolve on first access
TYPE_PAIRS = (
    ("KcpContext", "ContextDetKcp"),
    ("KcpOptions", "OptionsDetKcp"),
    ("KcpPartition", "PartitionDetKcp"),
    ("KcpRecomputeHandle", "RecomputeHandleDetKcp"),
    ("KcpSaved", "SavedDetKcp"),
    ("KcpWorkspace", "WorkspaceDetKcp"),
)
# (module, previous name, det name) of the backend modules' renamed definitions
BACKEND_PAIRS = (
    ("cake_kcp", "ChunkKDAKcpFunction", "ChunkKDADetFunctionKcp"),
    ("cake_kcp", "KcpCall", "CallDetKcp"),
    ("cake_kcp", "KcpOptions", "OptionsDetKcp"),
    ("cake_kcp", "KcpRecomputeHandle", "RecomputeHandleDetKcp"),
    ("cake_kcp", "KcpSaved", "SavedDetKcp"),
    ("cake_kcp", "KcpWorkspace", "WorkspaceDetKcp"),
    ("cake_kcp", "chunk_kda_kcp", "chunk_kda_det_kcp"),
    ("cake_kcp", "chunk_kda_kcp_bwd", "chunk_kda_det_bwd_kcp"),
    ("cake_kcp", "chunk_kda_kcp_fwd", "chunk_kda_det_fwd_kcp"),
    ("cake_kcp_plan", "KcpContext", "ContextDetKcp"),
    ("cake_kcp_plan", "KcpFragment", "FragmentDetKcp"),
    ("cake_kcp_plan", "KcpPartition", "PartitionDetKcp"),
    ("cake_kcp_plan", "build_kcp_context", "build_context_det_kcp"),
    ("cake_kcp_plan", "kcp_comm_bytes", "kda_det_comm_bytes_kcp"),
    ("cake_kcp_plan", "kcp_partition", "partition_det_kcp"),
    ("cake_kcp_plan", "kcp_saved_bytes", "kda_det_saved_bytes_kcp"),
    ("cake_kcp_plan", "kcp_workspace_bytes", "kda_det_workspace_bytes_kcp"),
)


@pytest.mark.parametrize("old,new", FACADE_PAIRS)
def test_facade_previous_names_are_the_det_objects(old: str, new: str) -> None:
    fn = getattr(kcp_api, new)
    assert getattr(kda_train, old) is fn
    assert fn.__name__ == new and fn.__module__ == "flashinfer.cake_kda_det_train_kcp"
    assert fn.experimental_feature == det_api._FEATURE
    assert new in kcp_api.__all__ and old not in kcp_api.__all__
    assert old in kda_train.__all__


@pytest.mark.parametrize("old,new", TYPE_PAIRS)
def test_facade_types_resolve_lazily_to_the_backend_classes(old: str, new: str) -> None:
    module = importlib.import_module(f"{PACKAGE}.{kcp_api._LAZY_TYPES[new]}")
    cls = getattr(module, new)
    assert getattr(kcp_api, new) is cls
    assert getattr(kda_train, old) is cls
    assert getattr(module, old) is cls
    assert cls.__name__ == new and cls.__qualname__ == new
    assert new in kcp_api.__all__
    if old == "KcpRecomputeHandle":
        assert old in kda_train.__all__
    with pytest.raises(AttributeError):
        _ = kcp_api.NoSuchName  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = kda_train.NoSuchName  # type: ignore[attr-defined]


@pytest.mark.parametrize("module,old,new", BACKEND_PAIRS)
def test_backend_previous_names_are_the_det_objects(
    module: str, old: str, new: str
) -> None:
    mod = importlib.import_module(f"{PACKAGE}.{module}")
    obj = getattr(mod, new)
    assert getattr(mod, old) is obj
    assert obj.__name__ == new
    assert new in mod.__all__ and old not in mod.__all__
    package = importlib.import_module(PACKAGE)
    if new in package.__all__:
        assert getattr(package, new) is obj


@pytest.mark.parametrize("submodule", ("cake_kcp", "cake_kcp_jit", "cake_kcp_plan"))
def test_previous_module_paths_are_the_det_kcp_modules(submodule: str) -> None:
    new = importlib.import_module(f"{PACKAGE}.{submodule}")
    assert importlib.import_module(f"{OLD_PACKAGE}.{submodule}") is new
    assert new.__name__ == f"{PACKAGE}.{submodule}"


def test_errors_name_the_det_entry_points() -> None:
    plan = importlib.import_module(f"{PACKAGE}.cake_kcp_plan")
    with pytest.raises(ValueError, match=r"^partition_det_kcp: "):
        plan.partition_det_kcp([0, 64], 0)
    with pytest.raises(ValueError, match=r"^partition_det_kcp: "):
        plan.kcp_partition([0, 64], 1, "no-such-layout")


def test_package_names_resolve_on_first_access_only() -> None:
    # Importing the facade, the package or the previous facade imports neither the host module nor the plan.
    code = (
        "import sys, importlib\n"
        "for name in sys.argv[1:]: importlib.import_module(name)\n"
        f"print(sorted(m for m in sys.modules if m.startswith('{PACKAGE}.')))\n"
    )
    for name in ("flashinfer.cake_kda_det_train_kcp", PACKAGE, "flashinfer.kda_train"):
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
    from flashinfer.experimental.cake_kda_det_train_kcp import cake_kcp_plan

    calls = []
    monkeypatch.setattr(
        cake_kcp_plan,
        "kda_det_workspace_bytes_kcp",
        lambda ctx, **kw: calls.append(kw) or 0,
    )
    ctx = object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        kcp_api.kda_det_workspace_bytes_kcp(ctx, backward=False)
        kcp_api.kda_det_workspace_bytes_kcp(ctx, backward=False, save_transfer=False)
        kcp_api.kda_det_workspace_bytes_kcp(ctx)
    assert [kw["save_transfer"] for kw in calls] == [
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
        False,
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
    ]
    assert cake_kcp_plan.SAVE_TRANSFER_DEFAULT is True
    assert (
        inspect.signature(kcp_api.kda_det_workspace_bytes_kcp)
        .parameters["save_transfer"]
        .default
        is None
    )


def test_recompute_handle_keywords_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KCP checkpoint recompute handle on the public module (issue #7815): ``chunk_kda_det_kcp`` takes
    ``recompute_handle`` / ``checkpoint_keep`` keyword-only and forwards both (``None`` by default);
    ``kda_det_saved_bytes_kcp`` / ``kda_det_comm_bytes_kcp`` forward ``checkpoint_keep`` (and ``save_transfer=None`` under a tier,
    ``False`` without one); ``RecomputeHandleDetKcp`` resolves lazily to
    the KCP module's class.  Device-free (recorders)."""
    from flashinfer.experimental.cake_kda_det_train_kcp import cake_kcp, cake_kcp_plan

    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "chunk_kda_det_kcp", recorder("autograd"))
    monkeypatch.setattr(cake_kcp_plan, "kda_det_saved_bytes_kcp", recorder("saved"))
    monkeypatch.setattr(cake_kcp_plan, "kda_det_comm_bytes_kcp", recorder("comm"))
    operands = tuple(object() for _ in range(6))
    handle, ctx = object(), object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert (
            kcp_api.chunk_kda_det_kcp(
                *operands, recompute_handle=handle, checkpoint_keep="transfer"
            )
            == "autograd"
        )
        assert kcp_api.chunk_kda_det_kcp(*operands, exchange="ring") == "autograd"
        assert kcp_api.kda_det_saved_bytes_kcp(ctx, checkpoint_keep="states") == "saved"
        assert (
            kcp_api.kda_det_comm_bytes_kcp(ctx, H=4, checkpoint_keep="states") == "comm"
        )
        assert kcp_api.kda_det_comm_bytes_kcp(ctx, H=4) == "comm"
    base = {
        "scale": None,
        "save_mode": "light",
        "boundary_mode": "saved",
        "workspace": None,
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
    params = inspect.signature(kcp_api.chunk_kda_det_kcp).parameters
    for name in ("recompute_handle", "checkpoint_keep"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
    for fn in (kcp_api.kda_det_saved_bytes_kcp, kcp_api.kda_det_comm_bytes_kcp):
        p = inspect.signature(fn).parameters["checkpoint_keep"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert kcp_api.RecomputeHandleDetKcp is cake_kcp.RecomputeHandleDetKcp
    assert "RecomputeHandleDetKcp" in kcp_api.__all__
    with pytest.raises(AttributeError):
        _ = kcp_api.NoSuchName  # type: ignore[attr-defined]


def test_public_module_forwards_the_raw_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw gate mode on the public module (issue #7816): the KCP wrappers forward ``use_gate_in_kernel`` /
    ``A_log`` / ``dt_bias`` / ``lower_bound`` through ``**options`` unchanged, ``kda_det_workspace_bytes_kcp``
    forwards ``use_gate_in_kernel`` and ``reduce_gate_param_grads_det_kcp`` forwards to the KCP module's helper,
    which the package lists in ``__all__``; ``flashinfer.kda_train`` binds ``kcp_reduce_gate_param_grads`` to the
    same object.  Device-free (recorders)."""
    from flashinfer.experimental.cake_kda_det_train_kcp import cake_kcp, cake_kcp_plan

    package = importlib.import_module(PACKAGE)
    assert "reduce_gate_param_grads_det_kcp" in package.__all__
    assert (
        package.reduce_gate_param_grads_det_kcp
        is cake_kcp.reduce_gate_param_grads_det_kcp
    )
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "chunk_kda_det_fwd_kcp", recorder("fwd"))
    monkeypatch.setattr(cake_kcp, "chunk_kda_det_bwd_kcp", recorder("bwd"))
    monkeypatch.setattr(cake_kcp, "chunk_kda_det_kcp", recorder("autograd"))
    monkeypatch.setattr(cake_kcp, "reduce_gate_param_grads_det_kcp", recorder("reduce"))
    monkeypatch.setattr(cake_kcp_plan, "kda_det_workspace_bytes_kcp", recorder("bytes"))
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
        assert kcp_api.chunk_kda_det_fwd_kcp(*operands, **kw) == "fwd"
        assert kcp_api.chunk_kda_det_bwd_kcp(*operands, object(), **kw) == "bwd"
        assert kcp_api.chunk_kda_det_kcp(*operands, **kw) == "autograd"
        assert (
            kcp_api.kda_det_workspace_bytes_kcp(object(), use_gate_in_kernel=True)
            == "bytes"
        )
        assert kcp_api.reduce_gate_param_grads_det_kcp(*reduce_args) == "reduce"
    for name, _, got in calls[:3]:
        assert all(got[k] is v for k, v in kw.items()), name
    assert calls[3][2]["use_gate_in_kernel"] is True
    assert calls[4] == ("reduce", reduce_args, {})
    assert "reduce_gate_param_grads_det_kcp" in kcp_api.__all__
    assert (
        kda_train.kcp_reduce_gate_param_grads is kcp_api.reduce_gate_param_grads_det_kcp
    )
    assert "kcp_reduce_gate_param_grads" in kda_train.__all__
