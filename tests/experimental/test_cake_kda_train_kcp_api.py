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
(``flashinfer.experimental.cake_kda_train_kcp``) and of the facade ``flashinfer.cake_kda_train_kcp`` that routes
between it and the deterministic package, device-free: the numerical mode (``deterministic`` on every entry: the
package refuses everything but ``False`` and raises no alert under ``torch.use_deterministic_algorithms``; the
facade routes ``True`` and torch's deterministic mode to the deterministic package, warns once, refuses a non-bool,
a saved set and a workspace of the other package and converts its partition; the autograd node records its
forward's mode; no context holds a mode and a context of either package serves both), the names (the
deterministic package's previous names in ``flashinfer.kda_train`` and the compatibility package
``flashinfer.experimental.cake_kda_train`` stay the deterministic objects; this package copies none of them and no
public name carries a mode), the import isolation
(the package imports the non-deterministic single-rank package ``flashinfer.experimental.cake.cake_kda_train``, never
a deterministic package or the compatibility package) and the dispatch of the local stages over architecture x head
count (the ``S1`` pair of the replay path, the KCP backward), and the deterministic package's raw gate and in-kernel
q / k L2 norm (the package refuses them, their off values pass; the facade forwards them with ``deterministic=True``
and routes ``reduce_gate_param_grads_kcp`` there).
"""

import ast
import dataclasses
import importlib
import inspect
import pathlib
import subprocess
import sys
import warnings

import pytest
import torch

import flashinfer.cake_kda_det_train_kcp as det_api
import flashinfer.cake_kda_train_kcp as kcp_api
import flashinfer.kda_train as kda_train
from flashinfer.api_logging import ExperimentalWarning

pytestmark = pytest.mark.cpu_only

PACKAGE = "flashinfer.experimental.cake_kda_train_kcp"
DET_PACKAGE = "flashinfer.experimental.cake_kda_det_train_kcp"
COMPAT_PACKAGE = "flashinfer.experimental.cake_kda_train"
LOCAL_PACKAGE = "flashinfer.experimental.cake.cake_kda_train"
SUBMODULES = ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit", "cake_build")

# facade entry -> the deterministic facade's entry it routes to under deterministic=True
FACADE_DET_NAMES = {
    "partition_kcp": "partition_det_kcp",
    "build_context_kcp": "build_context_det_kcp",
    "chunk_kda_kcp": "chunk_kda_det_kcp",
    "chunk_kda_fwd_kcp": "chunk_kda_det_fwd_kcp",
    "chunk_kda_bwd_kcp": "chunk_kda_det_bwd_kcp",
    "kda_workspace_bytes_kcp": "kda_det_workspace_bytes_kcp",
    "kda_saved_bytes_kcp": "kda_det_saved_bytes_kcp",
    "kda_comm_bytes_kcp": "kda_det_comm_bytes_kcp",
}
# facade entry -> the package module that defines it (both packages)
ENTRY_MODULE = {
    "partition_kcp": "cake_kcp_plan",
    "build_context_kcp": "cake_kcp_plan",
    "chunk_kda_kcp": "cake_kcp",
    "chunk_kda_fwd_kcp": "cake_kcp",
    "chunk_kda_bwd_kcp": "cake_kcp",
    "kda_workspace_bytes_kcp": "cake_kcp_plan",
    "kda_saved_bytes_kcp": "cake_kcp_plan",
    "kda_comm_bytes_kcp": "cake_kcp_plan",
}
COMPUTE_ENTRIES = ("chunk_kda_kcp", "chunk_kda_fwd_kcp", "chunk_kda_bwd_kcp")
# The deterministic package's previous names (flashinfer.kda_train and the module-level aliases of its modules).
# This package defines none of them; ``chunk_kda_kcp`` is the one spelling both carry (a previous name there, this
# package's entry here).
LEGACY_FACADE_NAMES = {
    "kcp_partition": "partition_det_kcp",
    "build_kcp_context": "build_context_det_kcp",
    "chunk_kda_kcp": "chunk_kda_det_kcp",
    "chunk_kda_kcp_fwd": "chunk_kda_det_fwd_kcp",
    "chunk_kda_kcp_bwd": "chunk_kda_det_bwd_kcp",
    "kcp_workspace_bytes": "kda_det_workspace_bytes_kcp",
    "kcp_saved_bytes": "kda_det_saved_bytes_kcp",
    "kcp_comm_bytes": "kda_det_comm_bytes_kcp",
}
LEGACY_TYPE_NAMES = {
    "KcpContext": "ContextDetKcp",
    "KcpOptions": "OptionsDetKcp",
    "KcpPartition": "PartitionDetKcp",
    "KcpRecomputeHandle": "RecomputeHandleDetKcp",
    "KcpSaved": "SavedDetKcp",
    "KcpWorkspace": "WorkspaceDetKcp",
}
LEGACY_MODULE_ALIASES = {
    "cake_kcp": (
        "ChunkKDAKcpFunction",
        "KcpCall",
        "KcpOptions",
        "KcpRecomputeHandle",
        "KcpSaved",
        "KcpWorkspace",
        "chunk_kda_kcp_bwd",
        "chunk_kda_kcp_fwd",
        "kcp_comm_bytes",
        "kcp_saved_bytes",
        "kcp_workspace_bytes",
    ),
    "cake_kcp_plan": (
        "KcpContext",
        "KcpFragment",
        "KcpPartition",
        "build_kcp_context",
        "kcp_comm_bytes",
        "kcp_partition",
        "kcp_saved_bytes",
        "kcp_workspace_bytes",
    ),
}
SHARED_SPELLINGS = {"chunk_kda_kcp"}
MODE_CU = [0, 100, 740, 817]


def _mod(sub: str | None = None):
    return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


def _det(sub: str):
    return importlib.import_module(f"{DET_PACKAGE}.{sub}")


@pytest.fixture
def torch_deterministic():
    """Restore torch's deterministic-algorithms setting (and its warn-only flag) after the test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture(autouse=True)
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


# ----------------------------------------------------------------------------------------------------------------
# the mode keyword


def test_every_entry_takes_the_mode_keyword_only_defaulting_to_true() -> None:
    """``deterministic`` is a keyword-only parameter defaulting to ``True`` on every facade entry and on the package's
    compute entries (whose default is refused: a caller states the mode); the plan functions and byte queries of the
    package take none (they hold no mode)."""
    for name in FACADE_DET_NAMES:
        p = inspect.signature(getattr(kcp_api, name)).parameters["deterministic"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True, name
    kcp = _mod("cake_kcp")
    for name in COMPUTE_ENTRIES:
        p = inspect.signature(getattr(kcp, name)).parameters["deterministic"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True, name
    plan = _mod("cake_kcp_plan")
    for name in (
        "partition_kcp",
        "build_context_kcp",
        "kda_workspace_bytes_kcp",
        "kda_saved_bytes_kcp",
        "kda_comm_bytes_kcp",
    ):
        assert "deterministic" not in inspect.signature(getattr(plan, name)).parameters


def _package_call(entry: str, **kw):
    """One package compute entry with stand-in operands: the mode guard runs before anything reads them."""
    kcp = _mod("cake_kcp")
    t = torch.zeros(1)
    args = (t, t, t, t, t, object())
    if entry == "chunk_kda_bwd_kcp":
        return kcp.chunk_kda_bwd_kcp(*args, t, **kw)
    return getattr(kcp, entry)(*args, **kw)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
@pytest.mark.parametrize(
    "value", [True, "omitted", 1, None, 0, "False"], ids=lambda v: repr(v)
)
def test_package_entries_refuse_every_mode_but_false(entry, value) -> None:
    """The package ships no deterministic schedule and never falls back: every value but ``False`` (the default
    ``True``, an omitted keyword, ``1``, ``None``, ``0``, ``"False"``) is a ``ValueError`` naming the entry, raised
    before any work."""
    kw = {} if value == "omitted" else {"deterministic": value}
    with pytest.raises(
        ValueError, match=rf"^{entry} is the non-deterministic .*deterministic=False"
    ):
        _package_call(entry, **kw)


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_raise_no_alert_under_torch_deterministic_mode(
    entry, warn_only, torch_deterministic
) -> None:
    """Every path of the package is fixed-order (no unordered reduction), so with
    ``torch.use_deterministic_algorithms(True)`` (and ``warn_only=True``) a ``deterministic=False`` call raises no
    alert and warns nothing: the stand-in operands fail later, on the operand checks, not on the mode."""
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # a warning (the warn-only alert) would surface as the raised exception
        with pytest.raises(Exception) as info:  # noqa: B017 - the stand-in operands' error, never the alert
            _package_call(entry, deterministic=False)
    message = str(info.value)
    assert "use_deterministic_algorithms" not in message, message
    assert "is the non-deterministic" not in message, message


def test_autograd_function_records_the_forward_mode() -> None:
    """The node records its forward's mode (``fctx.deterministic = False``) and its backward runs that mode: a node
    carrying another mode is refused before any work (``ValueError`` of the backward entry), one carrying none fails."""
    kcp = _mod("cake_kcp")
    src = inspect.getsource(kcp.ChunkKDAFunctionKcp.forward)
    assert "fctx.deterministic = False" in src
    assert "deterministic=False" in src

    class StandIn:
        saved_tensors = tuple(torch.zeros(1) for _ in range(5))
        kcp = (object(), None, "light", None, {}, None)

    fctx = StandIn()
    fctx.deterministic = True
    with pytest.raises(ValueError, match="^chunk_kda_bwd_kcp is the non-deterministic"):
        kcp.ChunkKDAFunctionKcp.backward(fctx, torch.zeros(1))
    with pytest.raises(AttributeError):
        kcp.ChunkKDAFunctionKcp.backward(StandIn(), torch.zeros(1))


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


def test_a_saved_set_is_read_by_its_own_packages_backward_only() -> None:
    """The package backward refuses the deterministic package's ``SavedDetKcp`` (and any other object) with
    ``ValueError``; the facade's deterministic route refuses this package's ``SavedKcp`` before the deterministic
    backward sees it."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    det_saved = det_kcp.SavedDetKcp(**_saved_kwargs(det_kcp))
    nd_saved = kcp.SavedKcp(**_saved_kwargs(kcp))
    for bad in (det_saved, object()):
        with pytest.raises(ValueError, match="SavedKcp .*is required"):
            _package_call("chunk_kda_bwd_kcp", saved=bad, deterministic=False)
    t = torch.zeros(1)
    for kw in ({}, {"deterministic": True}):
        with pytest.raises(
            ValueError, match="comes from the non-deterministic package"
        ):
            kcp_api.chunk_kda_bwd_kcp(t, t, t, t, t, object(), t, saved=nd_saved, **kw)


def _route_recorders(monkeypatch) -> list:
    """Replace every routed entry of both packages by a recorder returning its tag (``det:<name>`` /
    ``nd:<name>``); the calls ``(tag, kwargs)`` land in the returned list."""
    seen: list = []

    def recorder(tag):
        def fn(*args, **kwargs):
            seen.append((tag, kwargs))
            return tag

        return fn

    for entry, det_name in FACADE_DET_NAMES.items():
        sub = ENTRY_MODULE[entry]
        monkeypatch.setattr(_det(sub), det_name, recorder(f"det:{det_name}"))
        monkeypatch.setattr(_mod(sub), entry, recorder(f"nd:{entry}"))
    return seen


def _facade_calls() -> dict:
    t = torch.zeros(1)
    ctx = object()  # the recorders read nothing
    return {
        "partition_kcp": lambda **m: kcp_api.partition_kcp([0, 4], 1, **m),
        "build_context_kcp": lambda **m: kcp_api.build_context_kcp(
            object(), None, 0, "cpu", **m
        ),
        "chunk_kda_kcp": lambda **m: kcp_api.chunk_kda_kcp(t, t, t, t, t, ctx, **m),
        "chunk_kda_fwd_kcp": lambda **m: kcp_api.chunk_kda_fwd_kcp(
            t, t, t, t, t, ctx, **m
        ),
        "chunk_kda_bwd_kcp": lambda **m: kcp_api.chunk_kda_bwd_kcp(
            t, t, t, t, t, ctx, t, **m
        ),
        "kda_workspace_bytes_kcp": lambda **m: kcp_api.kda_workspace_bytes_kcp(
            ctx, H=4, **m
        ),
        "kda_saved_bytes_kcp": lambda **m: kcp_api.kda_saved_bytes_kcp(ctx, H=4, **m),
        "kda_comm_bytes_kcp": lambda **m: kcp_api.kda_comm_bytes_kcp(ctx, H=4, **m),
    }


def test_facade_routes_by_mode(monkeypatch) -> None:
    """``deterministic=True`` (the default) runs the deterministic package's object, ``False`` this package's (the
    compute entries with ``deterministic=False``); a non-bool is a ``TypeError``; no deterministic module is imported
    by a ``False`` call that was not imported before."""
    assert not torch.are_deterministic_algorithms_enabled()
    seen = _route_recorders(monkeypatch)
    for entry, call in _facade_calls().items():
        det_tag = f"det:{FACADE_DET_NAMES[entry]}"
        for kw in ({}, {"deterministic": True}):
            seen.clear()
            assert call(**kw) == det_tag, (entry, kw)
            assert "deterministic" not in seen[-1][1], (
                entry
            )  # the deterministic package takes no mode
        seen.clear()
        assert call(deterministic=False) == f"nd:{entry}", entry
        if entry in COMPUTE_ENTRIES:
            assert seen[-1][1]["deterministic"] is False, entry
        else:
            assert "deterministic" not in seen[-1][1], entry  # the plan holds no mode
        for bad in (1, 0, None, "False", "True"):
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                call(deterministic=bad)


def test_facade_under_torch_deterministic_mode_takes_the_deterministic_package_and_warns_once(
    monkeypatch, torch_deterministic
) -> None:
    """With ``torch.use_deterministic_algorithms(True)`` every ``deterministic=False`` facade call runs the
    deterministic package, and the process warns once (``UserWarning``; the flag shared with
    ``flashinfer.cake_kda_train`` is reset here)."""
    seen = _route_recorders(monkeypatch)
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", False)
    torch.use_deterministic_algorithms(True)
    calls = _facade_calls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for entry, call in calls.items():
                assert call(deterministic=False) == f"det:{FACADE_DET_NAMES[entry]}", (
                    entry
                )
                assert "deterministic" not in seen[-1][1], entry
    fallback = [
        w
        for w in caught
        if "deterministic=False runs the deterministic" in str(w.message)
    ]
    assert len(fallback) == 1, [str(w.message) for w in fallback]
    assert issubclass(fallback[0].category, UserWarning)
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        calls["chunk_kda_kcp"](deterministic=None)


def test_facade_workspace_of_the_other_package(
    monkeypatch, torch_deterministic
) -> None:
    """A ``WorkspaceKcp`` of this package launches its local records: a ``deterministic=True`` call refuses it
    (``TypeError``), and under torch's deterministic algorithms a ``deterministic=False`` call that falls back to the
    deterministic package runs on a workspace of its own (``workspace=None``)."""
    seen = _route_recorders(monkeypatch)
    ws = _mod("cake_kcp").WorkspaceKcp()
    t = torch.zeros(1)
    with pytest.raises(TypeError, match="WorkspaceKcp of deterministic=False"):
        kcp_api.chunk_kda_fwd_kcp(t, t, t, t, t, object(), workspace=ws)
    assert (
        kcp_api.chunk_kda_fwd_kcp(
            t, t, t, t, t, object(), workspace=ws, deterministic=False
        )
        == "nd:chunk_kda_fwd_kcp"
    )
    assert seen[-1][1]["workspace"] is ws
    torch.use_deterministic_algorithms(True)
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", True)
    assert (
        kcp_api.chunk_kda_fwd_kcp(
            t, t, t, t, t, object(), workspace=ws, deterministic=False
        )
        == "det:chunk_kda_det_fwd_kcp"
    )
    assert seen[-1][1]["workspace"] is None


def test_facade_converts_a_partition_of_the_other_package(
    monkeypatch, torch_deterministic
) -> None:
    """A partition of either package builds the context of both: ``build_context_kcp`` converts one of the other
    package to the routed package's type with the same fields (``layout`` included).  A default (deterministic)
    partition with ``deterministic=False`` builds this package's context; a partition of this package followed by
    ``torch.use_deterministic_algorithms(True)`` and a ``deterministic=False`` context call builds the deterministic
    context and warns once."""
    plan, det_plan = _mod("cake_kcp_plan"), _det("cake_kcp_plan")
    nd_part = kcp_api.partition_kcp(MODE_CU, 2, "zigzag", deterministic=False)
    det_part = kcp_api.partition_kcp(MODE_CU, 2, "zigzag")
    for det, part, own, fragment in (
        (True, nd_part, det_plan.PartitionDetKcp, det_plan.FragmentDetKcp),
        (False, det_part, plan.PartitionKcp, plan.FragmentKcp),
    ):
        got = kcp_api._partition_for(det, part)
        assert type(got) is own, det
        assert all(type(f) is fragment for frs in got.fragments for f in frs), det
        assert got.layout == part.layout == "zigzag", det
        assert got.to_dict() == part.to_dict(), det
        # the routed package's own partition passes through unchanged
        assert kcp_api._partition_for(det, got) is got, det
    want = {
        query: getattr(kcp_api, query)(
            kcp_api.build_context_kcp(det_part, None, 0, "cpu", sm_count=148), H=4
        )
        for query in (
            "kda_workspace_bytes_kcp",
            "kda_saved_bytes_kcp",
            "kda_comm_bytes_kcp",
        )
    }
    nd_ctx = kcp_api.build_context_kcp(
        det_part, None, 0, "cpu", sm_count=148, deterministic=False
    )
    assert isinstance(nd_ctx, plan.ContextKcp)
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", False)
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det_ctx = kcp_api.build_context_kcp(
            nd_part, None, 0, "cpu", sm_count=148, deterministic=False
        )
    assert isinstance(det_ctx, det_plan.ContextDetKcp)
    fallback = [
        w
        for w in caught
        if "deterministic=False runs the deterministic" in str(w.message)
    ]
    assert len(fallback) == 1, [str(w.message) for w in caught]
    for query, value in want.items():
        assert getattr(kcp_api, query)(det_ctx, H=4) == value, query


def test_facade_byte_queries_resolve_save_transfer_like_the_deterministic_facade(
    monkeypatch,
) -> None:
    """The byte queries forward the same keywords on both routes (an unset ``save_transfer`` resolves as the
    deterministic facade resolves it), so the two routes answer the same query."""
    seen = _route_recorders(monkeypatch)
    ctx = object()
    for det in (True, False):
        kcp_api.kda_workspace_bytes_kcp(ctx, backward=False, deterministic=det)
        kcp_api.kda_saved_bytes_kcp(ctx, checkpoint_keep="states", deterministic=det)
        kcp_api.kda_comm_bytes_kcp(ctx, H=4, deterministic=det)
    by_route = [kw for _tag, kw in seen]
    assert by_route[:3] == by_route[3:]
    assert by_route[0]["save_transfer"] is _mod("cake_kcp_plan").SAVE_TRANSFER_DEFAULT
    assert (
        by_route[1]["save_transfer"] is None and by_route[2]["save_transfer"] is False
    )
    assert by_route[0]["use_gate_in_kernel"] is False
    assert by_route[0]["use_qk_l2norm_in_kernel"] is False
    assert by_route[1]["use_qk_l2norm_in_kernel"] is False


# ----------------------------------------------------------------------------------------------------------------
# the deterministic package's modes: the raw gate (#7816) and the in-kernel q / k L2 norm (#7817)

# each keyword at its off value
MODE_OFF = {
    "use_gate_in_kernel": False,
    "A_log": None,
    "dt_bias": None,
    "lower_bound": None,
    "use_qk_l2norm_in_kernel": False,
}
RAW_GATE_ON = [
    {"use_gate_in_kernel": True},
    {"use_gate_in_kernel": 1},
    {"A_log": torch.zeros(4)},
    {"dt_bias": torch.zeros(4, 128)},
    {"lower_bound": -5.0},
    {"use_gate_in_kernel": True, "use_qk_l2norm_in_kernel": True},
]
RAW_GATE = "runs on the deterministic package only; pass deterministic=True"
QK_NORM = "in-kernel q / k L2 norm runs on the deterministic package only"


def _raw_gate_keywords() -> dict:
    return {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(4),
        "dt_bias": torch.zeros(4, 128),
        "lower_bound": -5.0,
    }


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
@pytest.mark.parametrize(
    "kw",
    RAW_GATE_ON,
    ids=["gate_true", "gate_1", "A_log", "dt_bias", "lower_bound", "gate_and_qk_norm"],
)
def test_package_entries_refuse_the_raw_gate(entry, kw) -> None:
    """The raw gate runs on the deterministic package only: ``use_gate_in_kernel`` other than ``False`` or any of
    ``A_log`` / ``dt_bias`` / ``lower_bound`` other than ``None`` is a ``ValueError`` naming ``deterministic=True``,
    raised before any work (also with the in-kernel norm requested)."""
    with pytest.raises(ValueError, match=RAW_GATE):
        _package_call(entry, deterministic=False, **kw)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_refuse_the_in_kernel_qk_norm(entry) -> None:
    """``use_qk_l2norm_in_kernel=True`` is a ``ValueError`` before any work: the package has no in-kernel norm."""
    with pytest.raises(ValueError, match=QK_NORM):
        _package_call(entry, deterministic=False, use_qk_l2norm_in_kernel=True)


@pytest.mark.parametrize("entry", COMPUTE_ENTRIES)
def test_package_entries_accept_the_modes_off(entry) -> None:
    """The off values are this package's mode: the options validate as if they were omitted, and an entry that
    spells them out fails on the stand-in operands, not on a keyword."""
    kcp = _mod("cake_kcp")
    assert kcp._options(dict(MODE_OFF)) == kcp._options({})
    with pytest.raises(Exception) as info:  # noqa: B017 - the stand-in operands' error, never a keyword's
        _package_call(entry, deterministic=False, **MODE_OFF)
    message = str(info.value)
    for refusal in ("unexpected keyword", RAW_GATE, QK_NORM):
        assert refusal not in message, message


def test_package_byte_query_refuses_the_raw_gate() -> None:
    """The package's workspace byte query takes the deterministic query's ``use_gate_in_kernel``: ``False`` gives the
    bytes of the omitted keyword, any other value is a ``ValueError`` before the context is read."""
    plan = _mod("cake_kcp_plan")
    ctx = kcp_api.build_context_kcp(
        kcp_api.partition_kcp(MODE_CU, 2, deterministic=False),
        None,
        0,
        "cpu",
        sm_count=148,
        deterministic=False,
    )
    assert plan.kda_workspace_bytes_kcp(
        ctx, H=4, use_gate_in_kernel=False
    ) == plan.kda_workspace_bytes_kcp(ctx, H=4)
    for value in (True, 1, None):
        with pytest.raises(ValueError, match="raw gate mode runs on the deterministic"):
            plan.kda_workspace_bytes_kcp(object(), use_gate_in_kernel=value)
    with pytest.raises(ValueError, match="raw gate mode runs on the deterministic"):
        kcp_api.kda_workspace_bytes_kcp(
            object(), use_gate_in_kernel=True, deterministic=False
        )


def test_package_byte_queries_refuse_the_in_kernel_qk_norm() -> None:
    """The package's workspace and saved-set byte queries take the deterministic queries' ``use_qk_l2norm_in_kernel``:
    ``False`` gives the bytes of the omitted keyword, any other value is a ``ValueError`` naming
    ``deterministic=True``, before the context is read; the facade's ``deterministic=False`` route raises the same."""
    plan = _mod("cake_kcp_plan")
    ctx = kcp_api.build_context_kcp(
        kcp_api.partition_kcp(MODE_CU, 2, deterministic=False),
        None,
        0,
        "cpu",
        sm_count=148,
        deterministic=False,
    )
    for query in ("kda_workspace_bytes_kcp", "kda_saved_bytes_kcp"):
        fn = getattr(plan, query)
        assert fn(ctx, H=4, use_qk_l2norm_in_kernel=False) == fn(ctx, H=4), query
        for value in (True, 1, None):
            with pytest.raises(ValueError, match=QK_NORM) as info:
                fn(object(), use_qk_l2norm_in_kernel=value)
            assert "deterministic=True" in str(info.value), query
        with pytest.raises(ValueError, match=QK_NORM) as info:
            getattr(kcp_api, query)(
                object(), use_qk_l2norm_in_kernel=True, deterministic=False
            )
        assert "deterministic=True" in str(info.value), query


def test_facade_byte_queries_forward_the_in_kernel_qk_norm(
    monkeypatch, torch_deterministic
) -> None:
    """``deterministic=True`` hands ``use_qk_l2norm_in_kernel`` (and the raw gate's ``use_gate_in_kernel``) to the
    deterministic byte queries unchanged; ``deterministic=False`` hands the value to this package's queries (which
    refuse ``True``: :func:`test_package_byte_queries_refuse_the_in_kernel_qk_norm`); under torch's deterministic
    algorithms a ``deterministic=False`` query is the deterministic package's and receives it."""
    seen = _route_recorders(monkeypatch)
    ctx = object()
    for query in ("kda_workspace_bytes_kcp", "kda_saved_bytes_kcp"):
        det_tag = f"det:{FACADE_DET_NAMES[query]}"
        assert getattr(kcp_api, query)(ctx, use_qk_l2norm_in_kernel=True) == det_tag
        assert seen[-1][1]["use_qk_l2norm_in_kernel"] is True, query
        assert getattr(kcp_api, query)(ctx, deterministic=False) == f"nd:{query}", query
        assert seen[-1][1]["use_qk_l2norm_in_kernel"] is False, query
    kcp_api.kda_workspace_bytes_kcp(
        ctx, use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True
    )
    assert seen[-1][1]["use_gate_in_kernel"] is True
    assert seen[-1][1]["use_qk_l2norm_in_kernel"] is True
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", True)
    torch.use_deterministic_algorithms(True)
    for query in ("kda_workspace_bytes_kcp", "kda_saved_bytes_kcp"):
        assert (
            getattr(kcp_api, query)(
                ctx, use_qk_l2norm_in_kernel=True, deterministic=False
            )
            == f"det:{FACADE_DET_NAMES[query]}"
        ), query
        assert seen[-1][1]["use_qk_l2norm_in_kernel"] is True, query


def test_facade_forwards_the_raw_gate_to_the_deterministic_package(
    monkeypatch, torch_deterministic
) -> None:
    """``deterministic=True`` forwards the raw gate keywords (the same objects) to the deterministic entries and
    ``use_gate_in_kernel`` to its workspace byte query; ``deterministic=False`` refuses them, unless torch's
    deterministic algorithms route the call to the deterministic package, which then receives them."""
    seen = _route_recorders(monkeypatch)
    calls = _facade_calls()
    gate = _raw_gate_keywords()
    for entry in COMPUTE_ENTRIES:
        assert calls[entry](**gate) == f"det:{FACADE_DET_NAMES[entry]}", entry
        for key, value in gate.items():
            assert seen[-1][1][key] is value, (entry, key)
    kcp_api.kda_workspace_bytes_kcp(object(), use_gate_in_kernel=True)
    assert seen[-1][1]["use_gate_in_kernel"] is True
    monkeypatch.undo()  # the package's own entries: the refusal is theirs
    t = torch.zeros(1)
    with pytest.raises(ValueError, match=RAW_GATE):
        kcp_api.chunk_kda_fwd_kcp(t, t, t, t, t, object(), deterministic=False, **gate)
    with pytest.raises(ValueError, match=QK_NORM):
        kcp_api.chunk_kda_kcp(
            t, t, t, t, t, object(), deterministic=False, use_qk_l2norm_in_kernel=True
        )
    seen = _route_recorders(monkeypatch)
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", True)
    torch.use_deterministic_algorithms(True)
    for entry in COMPUTE_ENTRIES:
        assert calls[entry](deterministic=False, **gate) == (
            f"det:{FACADE_DET_NAMES[entry]}"
        ), entry
        assert seen[-1][1]["use_gate_in_kernel"] is True, entry


def test_facade_reduces_the_gate_parameter_grads_on_the_deterministic_package(
    monkeypatch, torch_deterministic
) -> None:
    """``reduce_gate_param_grads_kcp`` is the deterministic package's ``reduce_gate_param_grads_det_kcp`` behind the
    mode keyword (keyword-only, default ``True``): ``deterministic=False`` is a ``ValueError`` (the raw gate runs on
    the deterministic package only) unless torch's deterministic algorithms route it there; a non-bool is a
    ``TypeError``."""
    p = inspect.signature(kcp_api.reduce_gate_param_grads_kcp).parameters[
        "deterministic"
    ]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True
    seen = []

    def recorder(*args, **kwargs):
        seen.append((args, kwargs))
        return "det:reduce_gate_param_grads_det_kcp"

    monkeypatch.setattr(_det("cake_kcp"), "reduce_gate_param_grads_det_kcp", recorder)
    ctx, dA_log, ddt_bias = object(), torch.zeros(4), torch.zeros(4, 128)
    for kw in ({}, {"deterministic": True}):
        seen.clear()
        assert (
            kcp_api.reduce_gate_param_grads_kcp(ctx, dA_log, ddt_bias, **kw)
            == "det:reduce_gate_param_grads_det_kcp"
        )
        assert seen == [((ctx, dA_log, ddt_bias), {})]
    with pytest.raises(ValueError, match="pass deterministic=True"):
        kcp_api.reduce_gate_param_grads_kcp(ctx, dA_log, ddt_bias, deterministic=False)
    for bad in (1, 0, None, "False"):
        with pytest.raises(TypeError, match="deterministic must be a bool"):
            kcp_api.reduce_gate_param_grads_kcp(
                ctx, dA_log, ddt_bias, deterministic=bad
            )
    owner = sys.modules[kcp_api._deterministic.__module__]
    monkeypatch.setattr(owner, "_DET_FALLBACK_WARNED", True)
    torch.use_deterministic_algorithms(True)
    assert (
        kcp_api.reduce_gate_param_grads_kcp(ctx, dA_log, ddt_bias, deterministic=False)
        == "det:reduce_gate_param_grads_det_kcp"
    )


# ----------------------------------------------------------------------------------------------------------------
# the context holds no mode; one context serves both packages


def test_the_context_holds_no_mode() -> None:
    """No field of ``ContextKcp`` / ``PartitionKcp`` / ``FragmentKcp`` names a mode, nor any attribute of a built
    context; the facade's two routes build the same partition (its fragment table) and the same byte answers."""
    plan, det_plan = _mod("cake_kcp_plan"), _det("cake_kcp_plan")
    part = plan.partition_kcp(MODE_CU, 2, "zigzag")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    names = [n for n in dir(ctx) if not n.startswith("__")]
    names += [f.name for f in dataclasses.fields(plan.FragmentKcp)]
    names += [f.name for f in dataclasses.fields(plan.PartitionKcp)]
    names += list(part.to_dict())
    bad = sorted({n for n in names if "determin" in n.lower() or n.lower() == "mode"})
    assert not bad, bad
    nd_part = kcp_api.partition_kcp(MODE_CU, 2, "zigzag", deterministic=False)
    det_part = kcp_api.partition_kcp(MODE_CU, 2, "zigzag")
    assert isinstance(nd_part, plan.PartitionKcp)
    assert isinstance(det_part, det_plan.PartitionDetKcp)
    assert nd_part.to_dict() == det_part.to_dict()
    for rank in range(2):
        nd_ctx = kcp_api.build_context_kcp(
            nd_part, None, rank, "cpu", sm_count=148, deterministic=False
        )
        det_ctx = kcp_api.build_context_kcp(det_part, None, rank, "cpu", sm_count=148)
        assert isinstance(nd_ctx, plan.ContextKcp)
        assert isinstance(det_ctx, det_plan.ContextDetKcp)
        for query in (
            "kda_workspace_bytes_kcp",
            "kda_saved_bytes_kcp",
            "kda_comm_bytes_kcp",
        ):
            fn = getattr(kcp_api, query)
            assert fn(nd_ctx, H=4, deterministic=False) == fn(det_ctx, H=4), (
                rank,
                query,
            )


def test_each_package_keeps_its_own_table_sets_on_a_shared_context() -> None:
    """A context serves both packages: each package's host module keys its launch tables on the context with keys of
    its own (this package's carry :data:`cake_kcp.TABLE_SET_TAG`), so neither reads a table set of the other's
    module."""
    kcp, det_kcp = _mod("cake_kcp"), _det("cake_kcp")
    plan = _mod("cake_kcp_plan")
    part = plan.partition_kcp(MODE_CU, 2, "zigzag")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    nd_tb = kcp.WorkspaceKcp(local=False).tables(ctx, 4)
    det_tb = det_kcp.WorkspaceDetKcp(local=False).tables(ctx, 4)
    assert type(nd_tb) is kcp._Tables and type(det_tb) is det_kcp._Tables
    assert nd_tb is not det_tb
    keys = list(ctx._table_sets)
    assert len(keys) == 2 and sum(k[-1] == kcp.TABLE_SET_TAG for k in keys) == 1, keys


# ----------------------------------------------------------------------------------------------------------------
# names


def _nd_objects() -> list:
    """Every public object of this package (its modules' ``__all__`` and their own callables) and of the facade."""
    objs = [getattr(kcp_api, n) for n in kcp_api.__all__]
    for sub in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit"):
        mod = _mod(sub)
        objs += [getattr(mod, n) for n in getattr(mod, "__all__", ())]
        objs += [
            v
            for v in vars(mod).values()
            if callable(v) and str(getattr(v, "__module__", "")).startswith(PACKAGE)
        ]
    return objs


def test_legacy_names_stay_the_deterministic_objects() -> None:
    """``flashinfer.kda_train`` and the compatibility package ``flashinfer.experimental.cake_kda_train`` stay the
    deterministic package's paths: every previous name IS its deterministic object and none is an object of this
    package or of the facade (``chunk_kda_kcp``, spelled by both, included)."""
    nd = _nd_objects()
    assert len(nd) > 50

    def is_nd(obj) -> bool:
        return any(obj is o for o in nd)

    for old, new in LEGACY_FACADE_NAMES.items():
        obj = getattr(kda_train, old)
        assert obj is getattr(det_api, new), old
        assert not is_nd(obj), old
    for old, new in LEGACY_TYPE_NAMES.items():
        obj = getattr(kda_train, old)
        assert obj is getattr(det_api, new), old
        assert not is_nd(obj), old
    assert kda_train.chunk_kda_kcp is not kcp_api.chunk_kda_kcp
    assert kda_train.chunk_kda_kcp is not _mod("cake_kcp").chunk_kda_kcp
    for sub in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit"):
        compat = importlib.import_module(f"{COMPAT_PACKAGE}.{sub}")
        assert compat is _det(sub), sub
        assert compat is not _mod(sub), sub
    for name in (
        "ContextKcp",
        "PartitionKcp",
        "SavedKcp",
        "WorkspaceKcp",
        "OptionsKcp",
        "RecomputeHandleKcp",
    ):
        assert getattr(kcp_api, name) is getattr(_mod(), name)
        assert getattr(kcp_api, name) is not getattr(
            det_api, name.replace("Kcp", "DetKcp")
        )


def test_the_package_defines_no_previous_name() -> None:
    """No previous name of the deterministic package was copied: the package, its modules and the facade bind none
    of them (``chunk_kda_kcp`` is this package's own entry, defined under its own name)."""
    legacy = (
        set(LEGACY_FACADE_NAMES)
        | set(LEGACY_TYPE_NAMES)
        | {n for names in LEGACY_MODULE_ALIASES.values() for n in names}
    ) - SHARED_SPELLINGS
    package = _mod()
    for name in sorted(legacy):
        assert not hasattr(kcp_api, name), name
        assert name not in package.__all__, name
        for sub in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit"):
            assert not hasattr(_mod(sub), name), (sub, name)
    entry = _mod("cake_kcp").chunk_kda_kcp
    assert package.chunk_kda_kcp is entry
    assert (
        entry.__module__ == f"{PACKAGE}.cake_kcp" and entry.__name__ == "chunk_kda_kcp"
    )


def test_no_public_name_names_a_mode() -> None:
    """The mode lives in the ``deterministic`` keyword, never in a public name: no exported name of the package, its
    modules or the facade contains ``det``."""
    names = set(_mod().__all__) | set(kcp_api.__all__)
    names |= {n for n in dir(kcp_api) if not n.startswith("_")}
    for sub in ("cake_kcp", "cake_kcp_plan", "cake_kcp_jit"):
        names |= set(getattr(_mod(sub), "__all__", ()))
    bad = sorted(n for n in names if "det" in n.lower().replace("detach", ""))
    assert not bad, bad


def test_package_exports_resolve_lazily_to_its_modules() -> None:
    """The package's names resolve on first access to the objects of its own modules."""
    package = _mod()
    for name in package.__all__:
        obj = getattr(package, name)
        assert obj is getattr(_mod(package._SUBMODULE_OF[name]), name), name
        assert "Det" not in name and "_det_" not in name, name


def test_package_names_resolve_on_first_access_only() -> None:
    """Importing the facade or the package imports neither the host module nor the plan (nor the single-rank
    package's modules)."""
    code = (
        "import sys, importlib\n"
        "for name in sys.argv[1:]: importlib.import_module(name)\n"
        f"print(sorted(m for m in sys.modules if m.startswith('{PACKAGE}.') or m.startswith('{LOCAL_PACKAGE}.')))\n"
    )
    for name in ("flashinfer.cake_kda_train_kcp", PACKAGE):
        out = subprocess.run(
            [sys.executable, "-c", code, name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert out == "[]", (name, out)


# ----------------------------------------------------------------------------------------------------------------
# import isolation


def _package_sources() -> list[pathlib.Path]:
    root = pathlib.Path(_mod().__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    assert {p.name for p in files} >= {"__init__.py", *(f"{s}.py" for s in SUBMODULES)}
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
    """A deterministic package, the compatibility package or one of their modules."""
    parts = module.split(".")
    return any(p.startswith("cake_kda_det_train") for p in parts) or (
        module == COMPAT_PACKAGE or module.startswith(COMPAT_PACKAGE + ".")
    )


def test_the_package_never_imports_a_deterministic_package() -> None:
    """Statically: no module of the package imports ``flashinfer.experimental.cake_kda_det_train`` /
    ``cake_kda_det_train_kcp`` (or a module of them) or the compatibility package
    ``flashinfer.experimental.cake_kda_train`` -- a relative ``..cake_kda_train`` would bind the latter and launch
    the deterministic records; the single-rank package it imports is ``flashinfer.experimental.cake.cake_kda_train``.
    The facade imports the deterministic package only inside its deterministic route's functions."""
    local = set()
    for path in _package_sources():
        for lineno, module in _imported_modules(path):
            assert not _forbidden(module), (path.name, lineno, module)
            if module.startswith(LOCAL_PACKAGE):
                local.add(path.name)
    assert {"cake_kcp.py", "cake_kcp_jit.py", "cake_kcp_plan.py"} <= local, local
    facade = pathlib.Path(kcp_api.__file__).resolve()
    tree = ast.parse(facade.read_text())
    top = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            module = "flashinfer" + ("." + node.module if node.module else "")
            if _forbidden(module):
                assert id(node) not in top, (node.lineno, module)


def test_importing_the_package_imports_no_deterministic_module() -> None:
    """At run time: importing every module of the package (and resolving its single-rank records) leaves no module
    of a deterministic package or of the compatibility package in ``sys.modules``."""
    code = (
        "import sys, importlib\n"
        f"for sub in {SUBMODULES!r}: importlib.import_module('{PACKAGE}.' + sub)\n"
        f"jit = importlib.import_module('{PACKAGE}.cake_kcp_jit')\n"
        "bad = sorted(m for m in sys.modules if 'cake_kda_det_train' in m or m == "
        f"'{COMPAT_PACKAGE}' or m.startswith('{COMPAT_PACKAGE}.'))\n"
        f"assert '{LOCAL_PACKAGE}.cake_jit' in sys.modules\n"
        "print(bad)\n"
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
    """The GLM intra record of the single-rank package's replay path (the KCP backward in both modes): its ``S1`` form
    where the routing table sends the replay to the ``S1`` pair (``sm_100a`` every head count but 9 to 15, ``sm_103a``
    up to 8 heads, ``sm_107a`` -- provisional, while the registry ships ``glm_intra_v23`` -- every head count but 9 to
    15), else the deterministic copy."""
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
